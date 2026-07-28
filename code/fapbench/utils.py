from __future__ import annotations

import json
import logging
import os
import sys
import random
try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore[assignment]
import time
from datetime import date, datetime, time as datetime_time
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    """Convert common scientific-Python/config values to JSON-safe objects.

    PyYAML parses unquoted values such as ``2014-12-31`` as
    :class:`datetime.date`.  Supporting standard-library dates here prevents a
    completed benchmark from failing while writing ``resolved_config.json``.
    """
    if value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return np.datetime_as_string(value)
    if isinstance(value, (pd.Timestamp, pd.Period, pd.Timedelta)):
        return str(value)
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(data: Any, path: str | Path) -> None:
    """Write JSON atomically so failures never leave a truncated target file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                data,
                handle,
                indent=2,
                default=json_default,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def configure_logger(
    output_dir: str | Path,
    *,
    name: str = "fapbench",
    level: str | int = "INFO",
    filename: str = "run.log",
    console: bool = True,
    overwrite: bool = False,
) -> logging.Logger:
    """Create a per-run logger that writes to both terminal and file.

    Handlers are rebuilt on every call so repeated runs in the same Python
    process do not duplicate log lines or continue writing to an old file.
    """
    output_dir = ensure_dir(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    log_path = output_dir / filename
    file_handler = logging.FileHandler(
        log_path,
        mode="w" if overwrite else "a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Persistent log file: %s", log_path.resolve())
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and close every handler owned by a per-run logger.

    Grid execution creates a distinct logger name for each output directory so
    concurrent runs never write into one another's files.  Python's logging
    manager retains those logger objects for the life of the interpreter;
    without explicit cleanup, every completed run leaves its ``FileHandler``
    open.  Long publication grids can therefore hit macOS's soft file-
    descriptor limit while spawning the next client worker process.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()


@contextmanager
def log_stage(logger: logging.Logger, name: str) -> Iterator[None]:
    """Log stage start, completion time, and exception traceback."""
    started = time.perf_counter()
    logger.info("START | %s", name)
    try:
        yield
    except Exception:
        logger.exception(
            "FAILED | %s | elapsed=%s",
            name,
            format_seconds(time.perf_counter() - started),
        )
        raise
    else:
        logger.info(
            "DONE  | %s | elapsed=%s",
            name,
            format_seconds(time.perf_counter() - started),
        )


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minute = divmod(minutes, 60)
    return f"{int(hours)}h {int(minute)}m {sec:.0f}s"



def process_peak_rss_mb() -> float:
    """Return peak resident memory for the current process in MiB.

    ``ru_maxrss`` is reported in bytes on macOS and KiB on Linux. The value is
    process-level and cumulative, so method-level metrics are snapshots of the
    peak reached by the time a method finishes rather than isolated allocations.
    """
    if resource is None:
        return float("nan")
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024**2)
    return value / 1024.0


def frame_memory_mb(frame: pd.DataFrame) -> float:
    return float(frame.memory_usage(index=True, deep=True).sum() / (1024**2))


def log_frame(
    logger: logging.Logger,
    name: str,
    frame: pd.DataFrame,
    *,
    date_col: str | None = None,
    entity_col: str | None = None,
    client_col: str | None = None,
) -> None:
    details = [
        f"rows={len(frame):,}",
        f"cols={frame.shape[1]:,}",
        f"memory={frame_memory_mb(frame):.1f} MB",
    ]
    if date_col and date_col in frame.columns:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        if dates.notna().any():
            details.append(f"date_min={dates.min().date()}")
            details.append(f"date_max={dates.max().date()}")
            details.append(f"n_dates={dates.nunique():,}")
    if entity_col and entity_col in frame.columns:
        details.append(f"n_entities={frame[entity_col].nunique(dropna=True):,}")
    if client_col and client_col in frame.columns:
        details.append(f"n_clients={frame[client_col].nunique(dropna=True):,}")
    logger.info("%s | %s", name, " | ".join(details))

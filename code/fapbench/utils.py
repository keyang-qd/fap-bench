from __future__ import annotations

import json
import logging
import random
import time
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


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

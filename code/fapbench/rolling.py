"""Aggregation for the expanding-window rolling-refit grid.

`configs/icaif_rolling_refit.yaml` trains one model per test year: for test year
Y it fits on everything through Y-5, validates on Y-4..Y-1, and predicts Y. Each
window is an ordinary run that writes its own monthly outputs, so this module's
job is only to stitch them back together.

Stitching is valid because portfolios are formed *within* each month from that
month's cross-section: a monthly Rank IC or a monthly net return produced by the
window covering 1997 is the same quantity as one produced by the window covering
1998, and the windows tile the calendar exactly. `_concatenate` enforces that --
a month appearing in two windows would mean overlapping test sets and is an
error, not something to silently average away.

The point of all this is statistical power. The single-window protocol tests on
60 months, which is too few to separate any two methods; concatenating 34 windows
gives ~408 months of out-of-sample evaluation for the same reference protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fapbench.evaluation.statistics import (
    annualized_sharpe,
    moving_block_bootstrap_mean,
    newey_west_mean_test,
)

ROLLING_PHASE = "rolling_refit"
REFERENCE_METHOD = "centralized"


def _window_dirs(phase_dir: Path) -> list[Path]:
    return sorted(p for p in phase_dir.glob("refit_test_*") if p.is_dir())


def _concatenate(frames: list[pd.Series], what: str) -> pd.Series:
    """Join per-window monthly series, refusing to merge overlapping windows."""
    if not frames:
        return pd.Series(dtype=float)
    joined = pd.concat(frames).sort_index()
    duplicated = joined.index.duplicated()
    if duplicated.any():
        months = sorted({str(m) for m in joined.index[duplicated]})[:5]
        raise ValueError(
            f"Rolling windows overlap on {duplicated.sum()} {what} month(s) "
            f"(e.g. {months}); test windows must tile the calendar exactly."
        )
    return joined


def _series_across_windows(
    phase_dir: Path, method: str, seed_name: str, filename: str, column: str
) -> pd.Series:
    parts: list[pd.Series] = []
    for window in _window_dirs(phase_dir):
        path = window / seed_name / filename.format(method=method)
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["target_date"])
        if column not in frame.columns:
            continue
        parts.append(frame.set_index("target_date")[column])
    return _concatenate(parts, column)


def _seed_names(phase_dir: Path) -> list[str]:
    windows = _window_dirs(phase_dir)
    if not windows:
        return []
    return sorted({p.name for p in windows[0].glob("seed_*") if p.is_dir()})


def rolling_pooled_metrics(
    output_root: Path,
    *,
    methods: tuple[str, ...] = (
        "centralized", "local", "fedavg", "fedprox", "scaffold", "fedadam",
    ),
    block_length: int = 6,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Per-method pooled Rank IC and net Sharpe over the concatenated window.

    Each method's monthly series is seed-averaged first, matching the unit of
    inference used everywhere else in the paper: seeds share one market history,
    so they reduce estimation noise rather than adding independent samples.
    """
    phase_dir = Path(output_root) / ROLLING_PHASE
    if not phase_dir.exists():
        return pd.DataFrame()
    seeds = _seed_names(phase_dir)
    if not seeds:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for method in methods:
        ic_by_seed, ret_by_seed = [], []
        for seed_name in seeds:
            ic = _series_across_windows(
                phase_dir, method, seed_name,
                "monthly_predictive_metrics_{method}.csv", "rank_ic",
            )
            ret = _series_across_windows(
                phase_dir, method, seed_name,
                "portfolio_returns_client_neutral_{method}.csv", "net_return",
            )
            if not ic.empty:
                ic_by_seed.append(ic)
            if not ret.empty:
                ret_by_seed.append(ret)
        if not ic_by_seed:
            continue
        rank_ic = pd.concat(ic_by_seed, axis=1).mean(axis=1).dropna()
        row: dict[str, Any] = {
            "method": method,
            "n_seeds": len(ic_by_seed),
            "n_months": int(len(rank_ic)),
            "rank_ic": float(rank_ic.mean()),
        }
        test = newey_west_mean_test(rank_ic.values)
        row["rank_ic_t"] = test["t_statistic"]
        row["rank_ic_p"] = test["p_value"]
        if ret_by_seed:
            net = pd.concat(ret_by_seed, axis=1).mean(axis=1).dropna()
            row["net_sharpe"] = float(annualized_sharpe(net.values))
            row["net_annual_return"] = float(net.mean() * 12.0)
            row["n_return_months"] = int(len(net))
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_vs_centralized(
    output_root: Path,
    *,
    methods: tuple[str, ...] = ("local", "fedavg", "fedprox", "scaffold", "fedadam"),
    block_length: int = 6,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Test each method's net-return difference from Centralized, pooled.

    This is the single-window headline test run on ~408 months instead of 60.
    Whether the gaps that were unresolvable at n=60 separate here is the whole
    question; either answer is reportable, and a null at this length is a much
    stronger statement than a null at 60.
    """
    phase_dir = Path(output_root) / ROLLING_PHASE
    if not phase_dir.exists():
        return pd.DataFrame()
    seeds = _seed_names(phase_dir)
    if not seeds:
        return pd.DataFrame()

    def _pooled(method: str) -> pd.Series:
        per_seed = [
            s
            for s in (
                _series_across_windows(
                    phase_dir, method, seed_name,
                    "portfolio_returns_client_neutral_{method}.csv", "net_return",
                )
                for seed_name in seeds
            )
            if not s.empty
        ]
        if not per_seed:
            return pd.Series(dtype=float)
        return pd.concat(per_seed, axis=1).mean(axis=1)

    reference = _pooled(REFERENCE_METHOD)
    if reference.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for method in methods:
        candidate = _pooled(method)
        if candidate.empty:
            continue
        aligned = pd.concat(
            [candidate.rename("m"), reference.rename("ref")], axis=1, join="inner"
        ).dropna()
        if aligned.empty:
            continue
        difference = (aligned["m"] - aligned["ref"]).values
        test = newey_west_mean_test(difference)
        interval = moving_block_bootstrap_mean(
            difference, block_length=block_length, n_bootstrap=n_bootstrap
        )
        rows.append(
            {
                "method": method,
                "comparison": f"{method} minus {REFERENCE_METHOD}",
                "n_months": int(len(difference)),
                "mean_monthly_difference": test["mean_difference"],
                "hac_t_statistic": test["t_statistic"],
                "hac_p_value": test["p_value"],
                "ci_lower": interval["ci_lower"],
                "ci_upper": interval["ci_upper"],
                "excludes_zero": bool(
                    interval["ci_lower"] > 0 or interval["ci_upper"] < 0
                ),
                "net_sharpe": float(annualized_sharpe(aligned["m"].values)),
                "reference_net_sharpe": float(annualized_sharpe(aligned["ref"].values)),
            }
        )
    return pd.DataFrame(rows)


def build_rolling_outputs(output_root: Path) -> dict[str, pd.DataFrame]:
    """Emit the rolling-refit tables; empty until the grid has been run."""
    output_root = Path(output_root)
    tables = {
        "rolling_pooled_metrics": rolling_pooled_metrics(output_root),
        "rolling_vs_centralized": rolling_vs_centralized(output_root),
    }
    tables_dir = output_root / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    return tables

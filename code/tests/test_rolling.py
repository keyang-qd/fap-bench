"""Tests for expanding-window rolling-refit aggregation."""
from pathlib import Path

import pandas as pd
import pytest

from fapbench.rolling import (
    _concatenate,
    build_rolling_outputs,
    rolling_pooled_metrics,
    rolling_vs_centralized,
)

METHODS = ("centralized", "fedavg")


def _write_window(root: Path, year: int, seed: str, ic: float, ret: float) -> None:
    for method in METHODS:
        seed_dir = root / f"refit_test_{year}" / seed
        seed_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range(f"{year}-01-31", periods=12, freq="ME")
        bump = 0.0 if method == "centralized" else 0.004
        pd.DataFrame(
            {"target_date": dates, "rank_ic": ic + bump}
        ).to_csv(seed_dir / f"monthly_predictive_metrics_{method}.csv", index=False)
        pd.DataFrame(
            {"target_date": dates, "net_return": ret + bump}
        ).to_csv(
            seed_dir / f"portfolio_returns_client_neutral_{method}.csv", index=False
        )


def _grid(tmp_path: Path, years=(1990, 1991, 1992)) -> Path:
    root = tmp_path / "rolling_refit"
    for i, year in enumerate(years):
        for seed in ("seed_11", "seed_22"):
            _write_window(root, year, seed, ic=0.01 + 0.001 * i, ret=0.005)
    return tmp_path


def test_pooled_metrics_span_every_window(tmp_path: Path):
    """Pooling must use all windows, not just the first."""
    table = rolling_pooled_metrics(_grid(tmp_path), methods=METHODS)
    assert set(table["method"]) == set(METHODS)
    # 3 windows x 12 months, and both seeds contribute.
    assert table["n_months"].eq(36).all()
    assert table["n_seeds"].eq(2).all()


def test_overlapping_windows_are_rejected(tmp_path: Path):
    """Two windows covering the same month means overlapping test sets.

    Silently averaging them would understate uncertainty by reusing months, so
    this must fail loudly rather than produce a plausible number.
    """
    root = tmp_path / "rolling_refit"
    for seed in ("seed_11",):
        _write_window(root, 1990, seed, ic=0.01, ret=0.005)
        _write_window(root, 1990, seed, ic=0.02, ret=0.006)
    # A second directory covering the same calendar year.
    dup = root / "refit_test_1990_copy"
    (dup / "seed_11").mkdir(parents=True)
    for method in METHODS:
        dates = pd.date_range("1990-01-31", periods=12, freq="ME")
        pd.DataFrame({"target_date": dates, "rank_ic": 0.03}).to_csv(
            dup / "seed_11" / f"monthly_predictive_metrics_{method}.csv", index=False
        )
    with pytest.raises(ValueError, match="tile the calendar exactly"):
        rolling_pooled_metrics(tmp_path, methods=METHODS)


def test_vs_centralized_recovers_the_planted_gap(tmp_path: Path):
    table = rolling_vs_centralized(tmp_path := _grid(tmp_path), methods=("fedavg",))
    row = table.iloc[0]
    assert row["n_months"] == 36
    # fedavg was written 0.004/month above centralized.
    assert abs(row["mean_monthly_difference"] - 0.004) < 1e-9


def test_missing_grid_yields_empty_tables(tmp_path: Path):
    tables = build_rolling_outputs(tmp_path)
    assert tables["rolling_pooled_metrics"].empty
    assert tables["rolling_vs_centralized"].empty
    for name in tables:
        assert (tmp_path / "paper_tables" / f"{name}.csv").exists()


def test_concatenate_passes_through_disjoint_windows():
    a = pd.Series([1.0, 2.0], index=pd.to_datetime(["1990-01-31", "1990-02-28"]))
    b = pd.Series([3.0], index=pd.to_datetime(["1990-03-31"]))
    assert len(_concatenate([a, b], "rank_ic")) == 3

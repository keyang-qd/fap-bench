"""Tests for expanding-window rolling-refit aggregation."""
from pathlib import Path

import numpy as np
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


def test_rolling_phase_is_selectable_and_isolated():
    """The investable rerun must be a different phase, not a different estimator.

    Two reviewers' accept conditions rest on comparing the unrestricted and
    investable refits, which is only meaningful if both go through the same code.
    This pins that the phase is a parameter and that an absent phase yields an
    empty frame rather than silently falling back to the other one.
    """
    from fapbench.rolling import ROLLING_INVESTABLE_PHASE, ROLLING_PHASE

    assert ROLLING_PHASE != ROLLING_INVESTABLE_PHASE
    assert rolling_pooled_metrics(Path("/nonexistent"), phase=ROLLING_PHASE).empty
    assert rolling_pooled_metrics(
        Path("/nonexistent"), phase=ROLLING_INVESTABLE_PHASE
    ).empty
    assert rolling_vs_centralized(
        Path("/nonexistent"), phase=ROLLING_INVESTABLE_PHASE
    ).empty


def test_rolling_output_reports_window_length(tmp_path):
    """A partial grid must be visibly partial.

    `build_rolling_outputs` reads whatever windows exist, so running it mid-grid
    produces a real-looking table computed from a fraction of the sample -- the
    manuscript quotes 408 months, and an 84-month table would otherwise look the
    same. `n_months` is what makes the difference detectable, so it must be present.
    """
    table = rolling_pooled_metrics(_grid(tmp_path), methods=METHODS)
    assert "n_months" in table.columns
    assert (table["n_months"] > 0).all()


def test_investable_flag_does_not_overwrite_the_unrestricted_config(tmp_path):
    """Regression: --investable changes output_dir but not the output filename, so
    running it with the default --output silently overwrote the unrestricted grid
    and left two config files pointing at one output directory -- which would have
    mixed two non-comparable universes into the same results tree."""
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "make_rolling_config",
        Path(__file__).resolve().parent.parent / "scripts" / "make_rolling_config.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    default = tmp_path / "icaif_rolling_refit.yaml"
    argv = ["make_rolling_config.py", "--investable", "--output", str(default)]
    old = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old

    assert not default.exists(), "the unrestricted filename must not be written"
    assert (tmp_path / "icaif_rolling_refit_investable.yaml").exists()


def test_the_two_rolling_configs_never_share_an_output_directory():
    """Two grids writing to one directory would interleave universes silently."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent / "configs"
    unrestricted = yaml.safe_load((root / "icaif_rolling_refit.yaml").read_text())
    investable = yaml.safe_load(
        (root / "icaif_rolling_refit_investable.yaml").read_text()
    )
    assert unrestricted["output_dir"] != investable["output_dir"]
    assert "universe_filter" not in unrestricted.get("base_overrides", {})
    assert "universe_filter" in investable["base_overrides"]


def test_seed_sensitivity_recomputes_both_arms_from_the_same_runs(tmp_path):
    """The comparison must differ only in seed count, never in window coverage.

    The claim it supports -- that cutting replication leaves the predictive
    ordering intact and scrambles the economic one -- is only meaningful if the
    two arms see the identical months. A subset that silently dropped windows
    would manufacture the asymmetry.
    """
    from fapbench.rolling import ROLLING_INVESTABLE_PHASE, seed_sensitivity

    phase = tmp_path / ROLLING_INVESTABLE_PHASE
    # Windows must tile the calendar without overlap, so each gets its own year.
    window_months = {
        "refit_test_2000": pd.date_range("2000-01-31", periods=12, freq="ME"),
        "refit_test_2001": pd.date_range("2001-01-31", periods=12, freq="ME"),
    }
    for window, months in window_months.items():
        for seed in (11, 22, 33, 44):
            run = phase / window / f"seed_{seed}"
            run.mkdir(parents=True)
            for method in ("centralized", "fedavg"):
                offset = 0.001 * seed if method == "fedavg" else 0.002
                pd.DataFrame(
                    {"target_date": months, "rank_ic": np.linspace(0.01, 0.02, 12) + offset}
                ).to_csv(run / f"monthly_predictive_metrics_{method}.csv", index=False)
                pd.DataFrame(
                    {"target_date": months,
                     "net_return": np.linspace(0.001, 0.004, 12) + offset}
                ).to_csv(
                    run / f"portfolio_returns_client_neutral_{method}.csv", index=False
                )

    table = seed_sensitivity(
        tmp_path, subset=(11, 22), methods=("centralized", "fedavg")
    )
    assert not table.empty
    assert set(table["metric"]) == {"rank_ic", "net_sharpe"}
    # Same window coverage on both sides -- the property the claim rests on.
    assert table["n_months"].nunique() == 1
    assert (table["n_seeds_reduced"] == 2).all()
    assert (table["n_seeds_full"] == 4).all()


def test_rolling_pooled_metrics_rejects_unknown_seed(tmp_path):
    """A typo'd seed must fail loudly rather than silently averaging fewer runs."""
    from fapbench.rolling import ROLLING_INVESTABLE_PHASE, rolling_pooled_metrics

    run = tmp_path / ROLLING_INVESTABLE_PHASE / "refit_test_2000" / "seed_11"
    run.mkdir(parents=True)
    months = pd.date_range("2000-01-31", periods=6, freq="ME")
    pd.DataFrame({"target_date": months, "rank_ic": [0.01] * 6}).to_csv(
        run / "monthly_predictive_metrics_fedavg.csv", index=False
    )
    with pytest.raises(ValueError, match="absent"):
        rolling_pooled_metrics(
            tmp_path,
            methods=("fedavg",),
            phase=ROLLING_INVESTABLE_PHASE,
            seeds=["seed_11", "seed_999"],
        )


def test_every_manuscript_rolling_cell_is_registered(tmp_path: Path):
    """Each rolling cell the manuscript reads must come out of one command.

    The widened cell's three tables -- the ones behind the paper's only
    detectable economic result -- were produced by hand outside
    `build_rolling_outputs` for several rounds, so `run_publication_experiments`
    did not regenerate the headline table. Nothing failed; the tables simply sat
    on disk, and a grid re-run would have left them stale with no error. This
    pins the registry against the phase constants so adding a cell without
    registering it fails here rather than at submission.
    """
    from fapbench.rolling import (
        ROLLING_INVESTABLE_PHASE,
        ROLLING_LINEAR_PHASE,
        ROLLING_LINEAR_TUNED20_PHASE,
        ROLLING_PHASE,
        ROLLING_TUNED20_PHASE,
    )

    phases = {
        ROLLING_PHASE: "",
        ROLLING_INVESTABLE_PHASE: "_investable",
        ROLLING_LINEAR_PHASE: "_linear",
        ROLLING_TUNED20_PHASE: "_tuned20",
        ROLLING_LINEAR_TUNED20_PHASE: "_linear_tuned20",
    }
    # Distinct output directories: two cells sharing one would let the rolling
    # concatenation splice them into a single series with nothing raised.
    assert len(set(phases)) == len(phases)

    tables = build_rolling_outputs(tmp_path)
    for suffix in phases.values():
        for stem in ("rolling_pooled_metrics", "rolling_vs_centralized"):
            assert f"{stem}{suffix}" in tables, f"{stem}{suffix} is not registered"

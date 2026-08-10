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
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from fapbench.evaluation.statistics import (
    annualized_sharpe,
    moving_block_bootstrap_mean,
    newey_west_mean_test,
)

ROLLING_PHASE = "rolling_refit"
# The same exercise behind the reference protocol's investable screen. It exists
# because the unrestricted refits cannot speak to the reference protocol: two
# reviewers independently objected that the paper's defence of its all-negative
# reference table is demonstrated on the universe the paper rejects. Both phases go
# through the identical estimator so the two are directly comparable.
ROLLING_INVESTABLE_PHASE = "rolling_refit_investable"
# The unrestricted linear/ridge cell, refit over the same 34 windows. Section 4.2
# identified this as the one cell whose observed spread (0.374) sits inside the
# 408-month detectable band, i.e. the only configuration in the benchmark where
# the economic question is answerable rather than bounded -- and the manuscript
# previously had to say "we have not refit it over 408 months".
ROLLING_LINEAR_PHASE = "rolling_refit_linear"
# The widened cell -- 20 characteristics behind the investable screen at the
# architecture its own validation sweep selected. This carries the manuscript's
# only detectable economic result, and until now its three tables were produced
# by hand outside build_rolling_outputs: the paper's headline depended on a step
# that `run_publication_experiments.py` did not reproduce. Registering it here
# closes that, and the same estimators as every other cell now apply to it.
ROLLING_TUNED20_PHASE = "rolling_refit_investable_tuned20"
# The widened cell's panel and universe with the model class swapped to
# linear/ridge. A reviewer asked the sharpest available version of the
# model-class objection: the coordination cost is measured on an MLP that the
# paper's own ablation shows is dominated by a linear model, so does the
# shortfall survive the swap or is it a fact about non-convex optimization? The
# architecture selection does not transfer (a linear model has no width to
# select), which is a caveat on this cell in the same way FedAdam's unretuned
# server rate is a caveat on the MLP one.
ROLLING_LINEAR_TUNED20_PHASE = "rolling_refit_investable_linear_tuned20"
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


def _assert_complete_seeds(phase_dir: Path, seeds: Sequence[str]) -> None:
    """Refuse to pool seeds that do not cover every window.

    This exists because the silent version of this bug already happened. A grid
    was extended from 3 seeds to 5 and stopped partway; the added seeds covered
    27 of 34 windows. Nothing raised. ``rolling_pooled_metrics`` averaged 5 seeds
    over the early windows and 3 over the late ones, still reported
    ``n_months=408``, and moved a published net Sharpe gap from 0.298 to 0.327 --
    a number the manuscript quotes and the audit binds. The audit did catch it,
    but only after the tables were regenerated, which is far too late to be the
    first line of defence.

    Callers that genuinely want a subset pass ``seeds=`` explicitly, which skips
    this check by construction: the point is that an *accidental* partial state
    cannot be pooled, not that subsets are forbidden.
    """
    windows = _window_dirs(phase_dir)
    expected = len(windows)
    incomplete: dict[str, int] = {}
    for seed in seeds:
        covered = sum(1 for w in windows if (w / seed).is_dir())
        if covered != expected:
            incomplete[seed] = covered
    if incomplete:
        detail = ", ".join(
            f"{s} ({n}/{expected})" for s, n in sorted(incomplete.items())
        )
        raise ValueError(
            f"{phase_dir.name}: seeds do not cover every window: {detail}. "
            "Pooling them would average different seed counts across the "
            "calendar while still reporting the full month count. Finish the "
            "grid, or pass seeds=[...] explicitly to pool a chosen subset."
        )


def rolling_pooled_metrics(
    output_root: Path,
    *,
    methods: tuple[str, ...] = (
        "centralized", "local", "fedavg", "fedprox", "scaffold", "fedadam",
    ),
    block_length: int = 6,
    n_bootstrap: int = 5000,
    phase: str = ROLLING_PHASE,
    seeds: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-method pooled Rank IC and net Sharpe over the concatenated window.

    Each method's monthly series is seed-averaged first, matching the unit of
    inference used everywhere else in the paper: seeds share one market history,
    so they reduce estimation noise rather than adding independent samples.

    ``seeds`` restricts the average to a subset already on disk, which is what
    ``seed_sensitivity`` uses to recompute this cell at a smaller replication
    count without retraining anything.
    """
    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    available = _seed_names(phase_dir)
    if seeds is None:
        _assert_complete_seeds(phase_dir, available)
        seeds = available
    else:
        missing = sorted(set(seeds) - set(available))
        if missing:
            raise ValueError(f"requested seeds absent from {phase}: {missing}")
        seeds = list(seeds)
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
    phase: str = ROLLING_PHASE,
    seeds: Sequence[str] | None = None,
    reference_method: str = REFERENCE_METHOD,
) -> pd.DataFrame:
    """Test each method's net-return difference from a reference arm, pooled.

    This is the single-window headline test run on ~408 months instead of 60.
    Whether the gaps that were unresolvable at n=60 separate here is the whole
    question; either answer is reportable, and a null at this length is a much
    stronger statement than a null at 60.

    `reference_method` exists because a reviewer noticed something the
    Centralized-only version structurally cannot show: in the widened cell the
    three federated arms are 0.52 of net Sharpe apart from each other, which is
    wider than that cell's own detectable gap. Every test we ran compared each
    arm against Centralized, so a difference *between* aggregation rules had no
    row anywhere in the paper, and the manuscript concluded that the rule did
    not matter. Pointing the same estimator at FedAvg answers it directly.
    """
    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    # `seeds` mirrors rolling_pooled_metrics: the completeness guard below tells
    # callers to pass an explicit subset, so this function has to accept one.
    available = _seed_names(phase_dir)
    if seeds is None:
        _assert_complete_seeds(phase_dir, available)
        seeds = available
    else:
        missing = sorted(set(seeds) - set(available))
        if missing:
            raise ValueError(f"requested seeds absent from {phase}: {missing}")
        seeds = list(seeds)
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

    reference = _pooled(reference_method)
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
                "comparison": f"{method} minus {reference_method}",
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


def rolling_rank_ic_vs_reference(
    output_root: Path,
    *,
    methods: tuple[str, ...] = ("local", "fedavg", "scaffold", "fedadam"),
    block_length: int = 6,
    n_bootstrap: int = 5000,
    phase: str = ROLLING_PHASE,
    reference_method: str = REFERENCE_METHOD,
) -> pd.DataFrame:
    """Paired Rank IC difference against a reference arm, tested like the rest.

    Why this exists
    ---------------
    A reviewer objected to an asymmetry that was real: the manuscript gave
    net-return contrasts HAC standard errors, moving-block bootstrap intervals
    and a place in a counted test family, while the Rank IC comparison it
    narrated as a finding -- FedAdam out-predicting Centralized in the reference
    cell -- rested on a paired t with no serial-correlation treatment and no
    family. "Either bring Rank IC into the same inferential regime or stop
    narrating it as a finding" is the right demand, and this is the first half.

    The estimator is deliberately identical to ``rolling_vs_centralized``: form
    the monthly difference within each seed, average across seeds, then apply
    Newey-West and the same moving-block bootstrap. Only the input series
    changes, from client-neutral net return to monthly Rank IC, so a reader can
    compare the two axes without also comparing two test designs.
    """
    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    seeds = _seed_names(phase_dir)
    if not seeds:
        return pd.DataFrame()
    _assert_complete_seeds(phase_dir, seeds)

    def _pooled(method: str) -> pd.Series:
        per_seed = [
            s
            for s in (
                _series_across_windows(
                    phase_dir, method, seed,
                    "monthly_predictive_metrics_{method}.csv", "rank_ic",
                )
                for seed in seeds
            )
            if not s.empty
        ]
        if not per_seed:
            return pd.Series(dtype=float)
        return pd.concat(per_seed, axis=1).mean(axis=1)

    reference = _pooled(reference_method)
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
                "comparison": f"{method} minus {reference_method}",
                "n_months": int(len(difference)),
                "n_seeds": len(seeds),
                "rank_ic_difference": test["mean_difference"],
                "hac_t_statistic": test["t_statistic"],
                "rank_ic_p": test["p_value"],
                "ci_lower": interval["ci_lower"],
                "ci_upper": interval["ci_upper"],
                "excludes_zero": bool(
                    interval["ci_lower"] > 0 or interval["ci_upper"] < 0
                ),
            }
        )
    return pd.DataFrame(rows)


def rolling_by_decade(
    output_root: Path,
    *,
    phase: str = ROLLING_INVESTABLE_PHASE,
    methods: Sequence[str] = ("centralized", "local", "fedavg", "scaffold", "fedadam"),
) -> pd.DataFrame:
    """Per-decade Rank IC and net return across the expanding-window refits.

    Why this exists
    ---------------
    The manuscript called the reference cell's all-negative net Sharpe column a
    "window artifact, removed by refitting". A reviewer pointed out that the two
    cells are not independent evidence -- the 408-month window *contains* the
    60-month one -- so a pooled positive average is equally consistent with
    monotone signal decay, in which case ranking methods on the long cell means
    ranking them on a regime that no longer exists.

    The two readings make different predictions and this settles which holds. It
    turns out to be neither cleanly: Rank IC does not decay (FedAdam's is highest
    in the 2020s), but net return in the 2020s is genuinely negative across
    methods. So the short cell's negative economic column reflects a real
    property of that sub-period rather than sampling noise alone, while the
    predictive signal it measures is intact. The manuscript now says that instead
    of "artifact".
    """
    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for window in _window_dirs(phase_dir):
        try:
            year = int(window.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        for method in methods:
            ic_parts, return_parts = [], []
            for seed_dir in sorted(window.glob("seed_*")):
                ic_path = seed_dir / f"monthly_predictive_metrics_{method}.csv"
                if ic_path.exists():
                    frame = pd.read_csv(ic_path)
                    if "rank_ic" in frame and "target_date" in frame:
                        ic_parts.append(frame.set_index("target_date")["rank_ic"])
                return_path = (
                    seed_dir / f"portfolio_returns_client_neutral_{method}.csv"
                )
                if return_path.exists():
                    frame = pd.read_csv(return_path)
                    date_cols = [c for c in frame.columns if "date" in c.lower()]
                    if date_cols and "net_return" in frame:
                        return_parts.append(
                            frame.set_index(date_cols[0])["net_return"]
                        )
            if not ic_parts:
                continue
            rows.append(
                {
                    "test_year": year,
                    "decade": (year // 10) * 10,
                    "method": method,
                    "rank_ic": float(pd.concat(ic_parts, axis=1).mean(axis=1).mean()),
                    "net_return": (
                        float(pd.concat(return_parts, axis=1).mean(axis=1).mean())
                        if return_parts
                        else float("nan")
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["decade", "method"], as_index=False)[["rank_ic", "net_return"]]
        .mean()
        .sort_values(["method", "decade"])
        .reset_index(drop=True)
    )


def staleness_isolation(
    output_root: Path,
    *,
    phase: str = ROLLING_INVESTABLE_PHASE,
    test_years: Sequence[int] = (2019, 2020, 2021, 2022, 2023),
    methods: Sequence[str] = ("centralized", "local", "fedavg", "scaffold", "fedadam"),
) -> pd.DataFrame:
    """Annual refits restricted to the reference cell's own 60 test months.

    A reviewer proposed that the reference cell's all-negative net Sharpe is a
    staleness artifact: Table 3 freezes training at 2014 and never refits across
    2019--2023, while the 408-month cell refits annually and turns positive. The
    two differ in refit frequency *and* in sample, so the comparison as it stood
    could not separate them.

    Restricting the expanding-window refits to exactly the reference cell's test
    months holds the sample fixed and varies only refit frequency, which is the
    clean contrast. It refutes the hypothesis: refitting makes 2019--2023 worse,
    not better. The long cell's positive Sharpes therefore come from the earlier
    decades it adds, not from fresher models.
    """
    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method in methods:
        ic_parts, return_parts = [], []
        for year in test_years:
            window = phase_dir / f"refit_test_{year}"
            if not window.is_dir():
                continue
            per_seed_ic, per_seed_ret = [], []
            for seed_dir in sorted(window.glob("seed_*")):
                ic_path = seed_dir / f"monthly_predictive_metrics_{method}.csv"
                if ic_path.exists():
                    frame = pd.read_csv(ic_path)
                    if {"target_date", "rank_ic"} <= set(frame.columns):
                        per_seed_ic.append(frame.set_index("target_date")["rank_ic"])
                ret_path = seed_dir / f"portfolio_returns_client_neutral_{method}.csv"
                if ret_path.exists():
                    frame = pd.read_csv(ret_path)
                    date_cols = [c for c in frame.columns if "date" in c.lower()]
                    if date_cols and "net_return" in frame:
                        per_seed_ret.append(frame.set_index(date_cols[0])["net_return"])
            if per_seed_ic:
                ic_parts.append(pd.concat(per_seed_ic, axis=1).mean(axis=1))
            if per_seed_ret:
                return_parts.append(pd.concat(per_seed_ret, axis=1).mean(axis=1))
        if not ic_parts:
            continue
        rank_ic = pd.concat(ic_parts)
        row = {
            "method": method,
            "n_months": int(len(rank_ic)),
            "refit_rank_ic": float(rank_ic.mean()),
        }
        if return_parts:
            net = pd.concat(return_parts)
            row["refit_net_sharpe"] = float(annualized_sharpe(net.values))
        rows.append(row)
    return pd.DataFrame(rows)


def seed_sensitivity(
    output_root: Path,
    *,
    subset: Sequence[int] = (11, 22, 33),
    phase: str = ROLLING_INVESTABLE_PHASE,
    methods: tuple[str, ...] = (
        "centralized", "local", "fedavg", "scaffold", "fedadam",
    ),
) -> pd.DataFrame:
    """How far the cell's method ordering moves when replication is cut.

    Why this exists
    ---------------
    This cell was first reported at three seeds and later rerun at ten, over the
    identical 34 windows and 408 months. Comparing the two is a controlled test
    of how much the ranking owes to replication alone: nothing about the market,
    the protocol, or the models differs between them.

    The result is the paper's own thesis reproduced by a second route. The
    predictive ordering is untouched by the change while the economic ordering is
    nearly scrambled, which is a direct demonstration that economic comparisons
    here are directional -- not an inference from significance tests, which is how
    the rest of the paper argues it.

    Both arms are recomputed from the same runs, so this needs no training and
    cannot drift from what the grid actually contains.
    """
    subset_names = [f"seed_{value}" for value in subset]
    full = rolling_pooled_metrics(output_root, methods=methods, phase=phase)
    if full.empty:
        return pd.DataFrame()
    reduced = rolling_pooled_metrics(
        output_root, methods=methods, phase=phase, seeds=subset_names
    )
    if reduced.empty:
        return pd.DataFrame()

    full = full.set_index("method")
    reduced = reduced.set_index("method")
    shared = [m for m in methods if m in full.index and m in reduced.index]
    rows: list[dict[str, Any]] = []
    for metric in ("rank_ic", "net_sharpe"):
        if metric not in full.columns or metric not in reduced.columns:
            continue
        few = reduced.loc[shared, metric].astype(float)
        many = full.loc[shared, metric].astype(float)
        # Spearman on five points, computed directly rather than via scipy so the
        # package keeps its existing dependency set.
        rank_few = few.rank()
        rank_many = many.rank()
        correlation = float(rank_few.corr(rank_many, method="pearson"))
        rows.append(
            {
                "metric": metric,
                "n_seeds_reduced": int(reduced["n_seeds"].max()),
                "n_seeds_full": int(full["n_seeds"].max()),
                "n_months": int(full["n_months"].max()),
                "spearman_rank_correlation": correlation,
                "mean_absolute_relative_change": float(
                    ((many - few).abs() / few.abs()).mean()
                ),
                "order_reduced": ">".join(
                    few.sort_values(ascending=False).index.tolist()
                ),
                "order_full": ">".join(
                    many.sort_values(ascending=False).index.tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_rolling_outputs(output_root: Path) -> dict[str, pd.DataFrame]:
    """Emit the rolling-refit tables; empty until the grid has been run."""
    output_root = Path(output_root)
    tables = {
        "rolling_pooled_metrics": rolling_pooled_metrics(output_root),
        "rolling_vs_centralized": rolling_vs_centralized(output_root),
        # FedProx is absent from both rolling grids, so it is dropped from the
        # investable method lists rather than producing an all-NaN row.
        "rolling_pooled_metrics_investable": rolling_pooled_metrics(
            output_root,
            methods=("centralized", "local", "fedavg", "scaffold", "fedadam"),
            phase=ROLLING_INVESTABLE_PHASE,
        ),
        "rolling_by_decade": rolling_by_decade(output_root),
        "staleness_isolation": staleness_isolation(output_root),
        "rolling_seed_sensitivity": seed_sensitivity(output_root),
        "rolling_vs_centralized_investable": rolling_vs_centralized(
            output_root,
            methods=("local", "fedavg", "scaffold", "fedadam"),
            phase=ROLLING_INVESTABLE_PHASE,
        ),
        "rolling_pooled_metrics_linear": rolling_pooled_metrics(
            output_root,
            methods=("centralized", "local", "fedavg", "scaffold", "fedadam"),
            phase=ROLLING_LINEAR_PHASE,
        ),
        "rolling_vs_centralized_linear": rolling_vs_centralized(
            output_root,
            methods=("local", "fedavg", "scaffold", "fedadam"),
            phase=ROLLING_LINEAR_PHASE,
        ),
        "rolling_pooled_metrics_tuned20": rolling_pooled_metrics(
            output_root,
            methods=("centralized", "local", "fedavg", "scaffold", "fedadam"),
            phase=ROLLING_TUNED20_PHASE,
        ),
        "rolling_vs_centralized_tuned20": rolling_vs_centralized(
            output_root,
            methods=("local", "fedavg", "scaffold", "fedadam"),
            phase=ROLLING_TUNED20_PHASE,
        ),
        "rolling_pooled_metrics_linear_tuned20": rolling_pooled_metrics(
            output_root,
            methods=("centralized", "local", "fedavg", "scaffold", "fedadam"),
            phase=ROLLING_LINEAR_TUNED20_PHASE,
        ),
        "rolling_vs_centralized_linear_tuned20": rolling_vs_centralized(
            output_root,
            methods=("local", "fedavg", "scaffold", "fedadam"),
            phase=ROLLING_LINEAR_TUNED20_PHASE,
        ),
        # The predictive axis, tested the same way as the economic one, so the
        # manuscript stops narrating a Rank IC finding on weaker inference than
        # it demands of its net-Sharpe nulls.
        "rolling_rank_ic_vs_centralized_investable": rolling_rank_ic_vs_reference(
            output_root, phase=ROLLING_INVESTABLE_PHASE
        ),
        "rolling_rank_ic_vs_centralized_tuned20": rolling_rank_ic_vs_reference(
            output_root, phase=ROLLING_TUNED20_PHASE
        ),
    }
    tables_dir = output_root / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    return tables

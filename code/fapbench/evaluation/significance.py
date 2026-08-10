from __future__ import annotations

"""Cross-seed aggregation of the paired-return significance tests.

``fapbench.evaluation.statistics`` implements a Newey-West/HAC paired-return
test and a moving-block-bootstrap Sharpe interval, but both operate on a
single already-completed run directory (one scenario, one seed). The
functions here apply those tests independently to every seed directory of a
scenario (or scenario pair) and combine the per-seed results.

Seeds share the same realized market history and differ only in model
initialization and training-path noise, so per-seed test statistics are not
independent draws in the classical sense. Combined p-values here should be
read as a directional summary of consistency across seeds -- a stronger
statement than a single seed's test, but not a small-sample-corrected exact
test across truly independent samples.
"""

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from fapbench.evaluation.predictive import _safe_corr
from fapbench.evaluation.statistics import (
    analyze_portfolio_results,
    benjamini_hochberg,
    moving_block_bootstrap_mean,
    newey_west_mean_test,
)


def _seed_dirs(scenario_dir: Path) -> list[Path]:
    return sorted(
        (path for path in scenario_dir.glob("seed_*") if path.is_dir()),
        key=lambda path: path.name,
    )


def _seed_id(seed_dir: Path) -> int | str:
    name = seed_dir.name
    if name.startswith("seed_"):
        suffix = name[len("seed_") :]
        try:
            return int(suffix)
        except ValueError:
            return suffix
    return name


def _signed_z_from_two_sided_p(t_statistic: float, p_value: float) -> float:
    """Convert a two-sided HAC p-value back to a signed z, keeping the test's sign."""
    if not np.isfinite(t_statistic) or not np.isfinite(p_value):
        return float("nan")
    p_value = min(max(float(p_value), 1e-300), 1.0 - 1e-16)
    magnitude = float(norm.isf(p_value / 2.0))
    return float(np.sign(t_statistic)) * magnitude


def stouffer_combine(
    t_statistics: Iterable[float], p_values: Iterable[float]
) -> dict[str, float | int]:
    """Combine independent two-sided tests into one signed z and p-value.

    Each (t_statistic, p_value) pair is converted to a signed z-score so that
    consistently-signed effects reinforce each other and offsetting-sign
    effects cancel, then combined as sum(z_i) / sqrt(n).
    """
    z_scores = [
        _signed_z_from_two_sided_p(t, p)
        for t, p in zip(t_statistics, p_values)
    ]
    z_scores = [z for z in z_scores if np.isfinite(z)]
    if not z_scores:
        return {
            "n_combined": 0,
            "combined_z": float("nan"),
            "combined_p_value": float("nan"),
        }
    combined_z = float(np.sum(z_scores) / np.sqrt(len(z_scores)))
    combined_p_value = float(2.0 * norm.sf(abs(combined_z)))
    return {
        "n_combined": len(z_scores),
        "combined_z": combined_z,
        "combined_p_value": combined_p_value,
    }


def aggregate_seed_significance(
    scenario_dir: Path,
    *,
    variant: str = "client_neutral",
    return_col: str = "net_return",
    reference_method: str | None = None,
    all_pairs: bool = False,
    periods_per_year: int = 12,
    hac_lag: int | None = None,
    block_length: int = 6,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the existing per-run HAC test and bootstrap Sharpe CI in every seed_* directory.

    Returns (per_seed_tests, per_seed_sharpe_intervals); both are empty
    DataFrames if no completed seed directory is found under ``scenario_dir``.
    """
    scenario_dir = Path(scenario_dir)
    test_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    for seed_dir in _seed_dirs(scenario_dir):
        if not (seed_dir / "benchmark_results.csv").exists():
            continue
        try:
            tests, intervals = analyze_portfolio_results(
                seed_dir,
                variant=variant,
                return_col=return_col,
                reference_method=reference_method,
                all_pairs=all_pairs,
                periods_per_year=periods_per_year,
                hac_lag=hac_lag,
                block_length=block_length,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        except (FileNotFoundError, KeyError, ValueError):
            continue
        tests.insert(0, "seed", _seed_id(seed_dir))
        intervals.insert(0, "seed", _seed_id(seed_dir))
        test_frames.append(tests)
        interval_frames.append(intervals)
    per_seed_tests = (
        pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    )
    per_seed_intervals = (
        pd.concat(interval_frames, ignore_index=True)
        if interval_frames
        else pd.DataFrame()
    )
    return per_seed_tests, per_seed_intervals


def combine_seed_tests(per_seed_tests: pd.DataFrame) -> pd.DataFrame:
    """Combine per-seed HAC paired-return tests with Stouffer's method."""
    if per_seed_tests.empty:
        return pd.DataFrame()
    group_cols = [
        column
        for column in [
            "reference_method",
            "comparison_method",
            "difference_definition",
            "variant",
            "return_col",
        ]
        if column in per_seed_tests.columns
    ]
    rows: list[dict[str, Any]] = []
    for key, group in per_seed_tests.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        combined = stouffer_combine(group["t_statistic"], group["p_value"])
        row: dict[str, Any] = dict(zip(group_cols, key_values))
        row.update(
            {
                "n_seeds": int(group["seed"].nunique()),
                "mean_t_statistic": float(
                    pd.to_numeric(group["t_statistic"], errors="coerce").mean()
                ),
                "mean_mean_difference": float(
                    pd.to_numeric(group["mean_difference"], errors="coerce").mean()
                ),
                "fraction_seeds_significant_at_5pct": float(
                    (pd.to_numeric(group["p_value"], errors="coerce") < 0.05).mean()
                ),
                **combined,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def combine_seed_intervals(per_seed_intervals: pd.DataFrame) -> pd.DataFrame:
    """Average per-seed bootstrap Sharpe point estimates and interval endpoints.

    This is a pooled point summary across seeds' bootstrap distributions, not a
    single joint bootstrap; it is reported alongside, not instead of, the
    cross-seed mean/std already used elsewhere in the benchmark.
    """
    if per_seed_intervals.empty:
        return pd.DataFrame()
    group_cols = [
        column for column in ["method", "variant"] if column in per_seed_intervals.columns
    ]
    numeric_cols = ["sharpe", "ci_lower", "ci_upper"]
    summary = per_seed_intervals.groupby(group_cols, as_index=False)[numeric_cols].mean()
    counts = per_seed_intervals.groupby(group_cols, as_index=False)["seed"].nunique()
    counts = counts.rename(columns={"seed": "n_seeds"})
    return summary.merge(counts, on=group_cols)


def identification_hac_contrasts(
    phase_dir: Path,
    pairs: Sequence[tuple[str, str, str]],
    *,
    methods: Sequence[str] | None = None,
    variant: str = "client_neutral",
    return_col: str = "net_return",
    hac_lag: int | None = None,
) -> pd.DataFrame:
    """Per-seed HAC test on treatment-minus-control monthly net-return differences.

    Thin wrapper around ``return_series_hac_contrasts`` for scenario pairs that
    both live under one phase directory (the original identification-vs-
    matched-control use case). See that function for the test itself.
    """
    phase_dir = Path(phase_dir)
    return return_series_hac_contrasts(
        [
            (phase_dir / treatment, phase_dir / control, label)
            for treatment, control, label in pairs
        ],
        methods=methods,
        variant=variant,
        return_col=return_col,
        hac_lag=hac_lag,
    )


def return_series_hac_contrasts(
    pairs: Sequence[tuple[Path, Path, str]],
    *,
    methods: Sequence[str] | None = None,
    variant: str = "client_neutral",
    return_col: str = "net_return",
    hac_lag: int | None = None,
) -> pd.DataFrame:
    """Per-seed HAC test on treatment-minus-control monthly net-return differences.

    For each ``(treatment_dir, control_dir, label)`` triple and each method,
    aligns the treatment and control seed directories by seed id and by date,
    computes the monthly difference in the requested portfolio-return variant,
    and runs the Newey-West mean-zero test on that difference series within each
    seed. Per-seed results are then combined with Stouffer's method. This
    complements the existing seed-level mean/std contrast (computed across only
    a handful of seed-level point estimates) with a test that also uses the
    within-seed time-series variation of the ~60-month test period. Unlike
    ``identification_hac_contrasts``, the two directories may live under
    different phases (e.g. a heterogeneity-type scenario against the plain
    ``core`` reference protocol, not a same-phase matched control).
    """
    rows: list[dict[str, Any]] = []
    for treatment_dir, control_dir, label in pairs:
        treatment_dir = Path(treatment_dir)
        control_dir = Path(control_dir)
        if not treatment_dir.exists() or not control_dir.exists():
            continue
        treatment_seeds = {_seed_id(p): p for p in _seed_dirs(treatment_dir)}
        control_seeds = {_seed_id(p): p for p in _seed_dirs(control_dir)}
        common_seeds = sorted(
            set(treatment_seeds) & set(control_seeds), key=str
        )
        if not common_seeds:
            continue
        resolved_methods = list(methods) if methods is not None else None
        if resolved_methods is None:
            benchmark_path = treatment_seeds[common_seeds[0]] / "benchmark_results.csv"
            if benchmark_path.exists():
                resolved_methods = (
                    pd.read_csv(benchmark_path)["method"].astype(str).unique().tolist()
                )
            else:
                resolved_methods = []
        for method in resolved_methods:
            per_seed_rows: list[dict[str, Any]] = []
            filename = {
                "primary": f"portfolio_returns_{method}.csv",
                "global": f"portfolio_returns_global_{method}.csv",
                "client_neutral": f"portfolio_returns_client_neutral_{method}.csv",
            }.get(variant.lower())
            if filename is None:
                raise ValueError("variant must be primary, global, or client_neutral.")
            for seed_id in common_seeds:
                treatment_path = treatment_seeds[seed_id] / filename
                control_path = control_seeds[seed_id] / filename
                if not treatment_path.exists() or not control_path.exists():
                    continue
                treatment_frame = pd.read_csv(treatment_path)
                control_frame = pd.read_csv(control_path)
                if return_col not in treatment_frame or return_col not in control_frame:
                    continue
                date_candidates = [
                    column for column in treatment_frame.columns if "date" in column.lower()
                ]
                if not date_candidates:
                    continue
                date_col = date_candidates[0]
                treatment_series = pd.Series(
                    pd.to_numeric(treatment_frame[return_col], errors="coerce").to_numpy(),
                    index=pd.to_datetime(treatment_frame[date_col], errors="coerce"),
                )
                control_series = pd.Series(
                    pd.to_numeric(control_frame[return_col], errors="coerce").to_numpy(),
                    index=pd.to_datetime(control_frame[date_col], errors="coerce"),
                )
                aligned = pd.concat(
                    [treatment_series, control_series], axis=1, join="inner"
                ).dropna()
                if aligned.empty:
                    continue
                difference = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                test = newey_west_mean_test(difference, max_lag=hac_lag)
                per_seed_rows.append({"seed": seed_id, **test})
            if not per_seed_rows:
                continue
            per_seed_frame = pd.DataFrame(per_seed_rows)
            combined = stouffer_combine(
                per_seed_frame["t_statistic"], per_seed_frame["p_value"]
            )
            rows.append(
                {
                    "contrast": label,
                    "treatment": treatment_dir.name,
                    "control": control_dir.name,
                    "method": method,
                    "variant": variant,
                    "return_col": return_col,
                    "n_seeds": int(per_seed_frame["seed"].nunique()),
                    "mean_t_statistic": float(per_seed_frame["t_statistic"].mean()),
                    "mean_difference_of_means": float(
                        per_seed_frame["mean_difference"].mean()
                    ),
                    "fraction_seeds_significant_at_5pct": float(
                        (per_seed_frame["p_value"] < 0.05).mean()
                    ),
                    **combined,
                }
            )
    return pd.DataFrame(rows)


def _monthly_client_dispersion_series(
    seed_dir: Path, method: str
) -> tuple[pd.Series, pd.Series]:
    """Per-seed monthly (cross-client std, worst-client) Rank-IC series.

    Reads the seed's raw predictions, computes a Rank IC per (month, client)
    with the same estimator used elsewhere in the benchmark
    (``fapbench.evaluation.predictive._safe_corr``, ``rank=True``), then
    collapses across clients within each month into a dispersion statistic.
    This is a different quantity from the client-neutral net-return series
    used by ``identification_hac_contrasts``: that series is a cross-client
    *mean* per month, so a treatment whose per-client effect is randomly
    signed (e.g. ``build_covariate_shift_parameters``) can largely cancel out
    of it even when the treatment clearly changes the *spread* across
    clients. Testing the spread directly avoids that cancellation.
    """
    path = seed_dir / f"predictions_{method}.parquet"
    if not path.exists():
        return pd.Series(dtype=float), pd.Series(dtype=float)
    frame = pd.read_parquet(
        path, columns=["target_date", "client_id", "y_true", "y_pred"]
    )
    records: list[tuple[Any, Any, float]] = []
    for (date, client), group in frame.groupby(
        ["target_date", "client_id"], sort=True
    ):
        rank_ic = _safe_corr(group["y_true"], group["y_pred"], rank=True)
        records.append((date, client, rank_ic))
    long = pd.DataFrame(records, columns=["date", "client_id", "rank_ic"])
    wide = long.pivot(index="date", columns="client_id", values="rank_ic")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    std_series = wide.std(axis=1, ddof=1, skipna=True)
    worst_series = wide.min(axis=1, skipna=True)
    return std_series, worst_series


def matched_control_placebo(
    controls: Sequence[tuple[Path, str]],
    *,
    methods: Sequence[str] | None = None,
    statistics: Sequence[str] = ("std",),
    hac_lag: int | None = None,
) -> pd.DataFrame:
    """Control-versus-control placebo on the dispersion estimator.

    A reviewer raised the one alternative explanation the manuscript's existing
    checks do not rule out. Per-client monthly Rank IC is near zero with large
    sampling noise, so cross-client *dispersion* could move with effective
    per-client sample size and feature configuration without any economically
    meaningful degradation -- in which case a nonzero treatment-minus-control
    difference would be an artifact of the statistic rather than evidence of
    structure.

    The placebo settles it without new training. Instead of differencing a
    treatment against its matched control, this differences the control arm
    against *itself*: seeds are split into disjoint pairs and one member of each
    pair is differenced against the other, through the identical estimator,
    lag rule, and Stouffer combination the real contrast uses. Both sides are
    then draws from the same distribution, so the true effect is exactly zero
    and anything the estimator reports is its own noise floor.

    Seeds are paired in sorted order rather than at random: the pairing has to be
    reproducible from the artifact, and a random pairing would make the reported
    placebo depend on an unrecorded draw.
    """
    cache: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}

    def _series(seed_dir: Path, method: str) -> tuple[pd.Series, pd.Series]:
        key = (str(seed_dir), method)
        if key not in cache:
            cache[key] = _monthly_client_dispersion_series(seed_dir, method)
        return cache[key]

    rows: list[dict[str, Any]] = []
    for control_dir, label in controls:
        control_dir = Path(control_dir)
        if not control_dir.exists():
            continue
        seeds = {_seed_id(p): p for p in _seed_dirs(control_dir)}
        ordered = sorted(seeds, key=str)
        # Disjoint pairs: (0,1), (2,3), ... A seed is never differenced against
        # itself, which would give an identically zero series and a degenerate
        # test rather than a placebo.
        seed_pairs = [
            (ordered[i], ordered[i + 1]) for i in range(0, len(ordered) - 1, 2)
        ]
        if not seed_pairs:
            continue
        resolved_methods = list(methods) if methods is not None else None
        if resolved_methods is None:
            benchmark_path = seeds[ordered[0]] / "benchmark_results.csv"
            if benchmark_path.exists():
                resolved_methods = (
                    pd.read_csv(benchmark_path)["method"].astype(str).unique().tolist()
                )
            else:
                resolved_methods = []
        for method in resolved_methods:
            for statistic in statistics:
                stat_index = 0 if statistic == "std" else 1
                per_pair_rows: list[dict[str, Any]] = []
                for left, right in seed_pairs:
                    left_series = _series(seeds[left], method)[stat_index]
                    right_series = _series(seeds[right], method)[stat_index]
                    aligned = pd.concat(
                        [left_series, right_series], axis=1, join="inner"
                    ).dropna()
                    if aligned.empty:
                        continue
                    difference = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                    test = newey_west_mean_test(difference, max_lag=hac_lag)
                    per_pair_rows.append({"pair": f"{left}|{right}", **test})
                if not per_pair_rows:
                    continue
                frame = pd.DataFrame(per_pair_rows)
                combined = stouffer_combine(
                    frame["t_statistic"], frame["p_value"]
                )
                rows.append(
                    {
                        "contrast": label,
                        "method": method,
                        "statistic": statistic,
                        "n_pairs": int(len(frame)),
                        "mean_t_statistic": float(frame["t_statistic"].mean()),
                        "mean_difference_of_means": float(
                            frame["mean_difference"].mean()
                        ),
                        "max_abs_difference_of_means": float(
                            frame["mean_difference"].abs().max()
                        ),
                        "fraction_pairs_significant_at_5pct": float(
                            (frame["p_value"] < 0.05).mean()
                        ),
                        **combined,
                    }
                )
    return pd.DataFrame(rows)


def heterogeneity_dispersion_hac_contrasts(
    pairs: Sequence[tuple[Path, Path, str]],
    *,
    methods: Sequence[str] | None = None,
    statistics: Sequence[str] = ("std", "worst"),
    hac_lag: int | None = None,
) -> pd.DataFrame:
    """Per-seed HAC test on treatment-minus-control monthly cross-client dispersion.

    Complements ``identification_hac_contrasts`` (which tests the
    client-neutral *mean* net-return series) by testing the *dispersion*
    across clients instead. For each ``(treatment_dir, control_dir, label)``
    triple -- unlike ``identification_hac_contrasts``, the two directories may
    live under different phases, e.g. a ``heterogeneity`` scenario against the
    ``core`` reference protocol -- aligns seeds and computes the monthly
    difference in cross-client Rank-IC dispersion (std across clients, or the
    worst single client), runs the Newey-West mean-zero test on that
    difference series within each seed, and combines per-seed results with
    Stouffer's method.
    """
    cache: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}

    def _series(seed_dir: Path, method: str) -> tuple[pd.Series, pd.Series]:
        key = (str(seed_dir), method)
        if key not in cache:
            cache[key] = _monthly_client_dispersion_series(seed_dir, method)
        return cache[key]

    rows: list[dict[str, Any]] = []
    for treatment_dir, control_dir, label in pairs:
        treatment_dir = Path(treatment_dir)
        control_dir = Path(control_dir)
        if not treatment_dir.exists() or not control_dir.exists():
            continue
        treatment_seeds = {_seed_id(p): p for p in _seed_dirs(treatment_dir)}
        control_seeds = {_seed_id(p): p for p in _seed_dirs(control_dir)}
        common_seeds = sorted(set(treatment_seeds) & set(control_seeds), key=str)
        if not common_seeds:
            continue
        resolved_methods = list(methods) if methods is not None else None
        if resolved_methods is None:
            benchmark_path = treatment_seeds[common_seeds[0]] / "benchmark_results.csv"
            if benchmark_path.exists():
                resolved_methods = (
                    pd.read_csv(benchmark_path)["method"].astype(str).unique().tolist()
                )
            else:
                resolved_methods = []
        for method in resolved_methods:
            for statistic in statistics:
                stat_index = 0 if statistic == "std" else 1
                per_seed_rows: list[dict[str, Any]] = []
                for seed_id in common_seeds:
                    treatment_series = _series(treatment_seeds[seed_id], method)[
                        stat_index
                    ]
                    control_series = _series(control_seeds[seed_id], method)[
                        stat_index
                    ]
                    aligned = pd.concat(
                        [treatment_series, control_series], axis=1, join="inner"
                    ).dropna()
                    if aligned.empty:
                        continue
                    difference = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                    test = newey_west_mean_test(difference, max_lag=hac_lag)
                    per_seed_rows.append({"seed": seed_id, **test})
                if not per_seed_rows:
                    continue
                per_seed_frame = pd.DataFrame(per_seed_rows)
                combined = stouffer_combine(
                    per_seed_frame["t_statistic"], per_seed_frame["p_value"]
                )
                rows.append(
                    {
                        "contrast": label,
                        "method": method,
                        "statistic": statistic,
                        "n_seeds": int(per_seed_frame["seed"].nunique()),
                        "mean_t_statistic": float(per_seed_frame["t_statistic"].mean()),
                        "mean_difference_of_means": float(
                            per_seed_frame["mean_difference"].mean()
                        ),
                        "fraction_seeds_significant_at_5pct": float(
                            (per_seed_frame["p_value"] < 0.05).mean()
                        ),
                        **combined,
                    }
                )
    return pd.DataFrame(rows)


def _pair_seed_difference_matrix(
    treatment_dir: Path,
    control_dir: Path,
    method: str,
    stat_index: int,
    series_fn: Callable[[Path, str], tuple[pd.Series, pd.Series]],
) -> pd.DataFrame | None:
    """Treatment-minus-control monthly differences for one scenario pair.

    Returns a months x seeds frame, or None when the pair contributes nothing.
    Shared verbatim by the single-window and expanding-window estimators so the
    two can never silently diverge in how a difference series is built.
    """
    treatment_dir = Path(treatment_dir)
    control_dir = Path(control_dir)
    if not treatment_dir.exists() or not control_dir.exists():
        return None
    treatment_seeds = {_seed_id(p): p for p in _seed_dirs(treatment_dir)}
    control_seeds = {_seed_id(p): p for p in _seed_dirs(control_dir)}
    common_seeds = sorted(set(treatment_seeds) & set(control_seeds), key=str)
    if not common_seeds:
        return None
    per_seed: list[pd.Series] = []
    for seed_id in common_seeds:
        treatment = series_fn(treatment_seeds[seed_id], method)[stat_index]
        control = series_fn(control_seeds[seed_id], method)[stat_index]
        aligned = pd.concat([treatment, control], axis=1, join="inner").dropna()
        if aligned.empty:
            continue
        per_seed.append((aligned.iloc[:, 0] - aligned.iloc[:, 1]).rename(str(seed_id)))
    if not per_seed:
        return None
    return pd.concat(per_seed, axis=1)


def rolling_dispersion_contrasts(
    labelled_windows: Mapping[str, Sequence[tuple[Path, Path]]],
    *,
    methods: Sequence[str] | None = None,
    statistics: Sequence[str] = ("std",),
    hac_lag: int | None = None,
    block_length: int = 6,
    n_bootstrap: int = 5000,
    bootstrap_seed: int = 42,
    period_label: str = "full",
) -> pd.DataFrame:
    """Matched-control dispersion contrasts over expanding-window refits.

    ``seed_averaged_dispersion_contrasts`` tests one contrast on one training
    window, so its series is as long as that window's test block -- 60 months at
    the reference protocol. A reviewer's sharpest methodological point was that
    the paper already builds a 408-month expanding-window sample for its economic
    axis but leaves RQ3, whose claims are the sharpest in the paper, at n=60.

    This estimator closes that gap. Each refit window contributes its own
    treatment-minus-control monthly difference matrix, computed by exactly the
    same helper the single-window estimator uses; the matrices are then
    concatenated along the time axis into one long series before the
    Newey-West test and the moving-block bootstrap run.

    Concatenation is legitimate here for the same reason it is in
    ``fapbench.rolling``: dispersion is computed within a month across clients,
    and windows contribute disjoint test months, so adjacent windows join
    without overlap or double-counting. The seed-averaging step is unchanged --
    seeds remain repeated measurements, not independent draws.

    ``labelled_windows`` maps a contrast label to that contrast's per-window
    ``(treatment_dir, control_dir)`` pairs.
    """
    cache: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}

    def _series(seed_dir: Path, method: str) -> tuple[pd.Series, pd.Series]:
        key = (str(seed_dir), method)
        if key not in cache:
            cache[key] = _monthly_client_dispersion_series(seed_dir, method)
        return cache[key]

    rows: list[dict[str, Any]] = []
    for label, window_pairs in labelled_windows.items():
        resolved_methods = list(methods) if methods is not None else []
        for method in resolved_methods:
            for statistic in statistics:
                stat_index = 0 if statistic == "std" else 1
                matrices = [
                    m
                    for treatment_dir, control_dir in window_pairs
                    if (
                        m := _pair_seed_difference_matrix(
                            treatment_dir, control_dir, method, stat_index, _series
                        )
                    )
                    is not None
                ]
                if not matrices:
                    continue
                matrix = pd.concat(matrices, axis=0).sort_index()
                # Windows must not overlap: a repeated month would be counted
                # twice and would deflate the HAC standard error.
                if matrix.index.duplicated().any():
                    raise ValueError(
                        f"[{label}/{method}/{statistic}] expanding windows overlap: "
                        f"{int(matrix.index.duplicated().sum())} duplicated months. "
                        "Refit windows must contribute disjoint test months."
                    )
                averaged = matrix.mean(axis=1).dropna()
                if len(averaged) < 3:
                    continue
                test = newey_west_mean_test(averaged, max_lag=hac_lag)
                interval = moving_block_bootstrap_mean(
                    averaged,
                    block_length=block_length,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
                per_seed_means = matrix.mean(axis=0)
                pooled_sign = np.sign(float(averaged.mean()))
                rows.append(
                    {
                        "contrast": label,
                        "period": period_label,
                        "method": method,
                        "statistic": statistic,
                        "n_windows": int(len(matrices)),
                        "n_seeds": int(matrix.shape[1]),
                        "n_months": int(len(averaged)),
                        "effect": float(averaged.mean()),
                        "hac_t_statistic": float(test["t_statistic"]),
                        "hac_p_value": float(test["p_value"]),
                        "ci_lower": float(interval["ci_lower"]),
                        "ci_upper": float(interval["ci_upper"]),
                        "excludes_zero": bool(
                            interval["ci_lower"] > 0 or interval["ci_upper"] < 0
                        ),
                        "seed_sign_agreement": (
                            float((np.sign(per_seed_means) == pooled_sign).mean())
                            if pooled_sign != 0
                            else np.nan
                        ),
                        "block_length": int(interval["block_length"]),
                        "n_bootstrap": int(interval["n_bootstrap"]),
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["hac_q_value"] = benjamini_hochberg(frame["hac_p_value"])
    return frame


def seed_averaged_dispersion_contrasts(
    pairs: Sequence[tuple[Path, Path, str]],
    *,
    methods: Sequence[str] | None = None,
    statistics: Sequence[str] = ("std", "worst"),
    hac_lag: int | None = None,
    block_length: int = 6,
    n_bootstrap: int = 5000,
    bootstrap_seed: int = 42,
    months: Callable[[pd.DatetimeIndex], Any] | None = None,
    period_label: str = "full",
) -> pd.DataFrame:
    """Matched-control dispersion contrasts with calendar month as the unit of inference.

    Replaces the Stouffer-across-seeds combination used by
    ``heterogeneity_dispersion_hac_contrasts``. Training seeds share one realized
    market history and differ only in optimization-path noise, so they are not
    independent draws from the data-generating process and combining their
    per-seed p-values understates the true tail probability by a wide margin.

    Here the treatment-minus-control monthly difference is computed within each
    seed and then *averaged across seeds* into a single ~60-month series, so
    seeds act as repeated measurements that reduce estimation noise rather than
    as independent samples. Inference is then a Newey-West mean-zero test on that
    single series, plus a moving-block bootstrap interval that respects its serial
    dependence. Averaging preserves the mean, so reported effect sizes are
    unchanged relative to the per-seed estimator -- only the uncertainty
    statement changes.

    ``seed_sign_agreement`` is reported descriptively (the fraction of seeds whose
    own mean difference shares the pooled sign); it carries no independence claim.

    ``months`` optionally restricts the averaged series to a subperiod: it is
    called with the series' DatetimeIndex and must return a boolean mask. The
    2019--2023 window is short and includes the COVID shock, so a reader needs to
    know whether an effect is a regime artifact; passing e.g.
    ``months=lambda i: i.year != 2020`` re-runs the identical test off-COVID.
    ``period_label`` is recorded in the output so subperiods can be stacked into
    one table.
    """
    cache: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}

    def _series(seed_dir: Path, method: str) -> tuple[pd.Series, pd.Series]:
        key = (str(seed_dir), method)
        if key not in cache:
            cache[key] = _monthly_client_dispersion_series(seed_dir, method)
        return cache[key]

    rows: list[dict[str, Any]] = []
    for treatment_dir, control_dir, label in pairs:
        treatment_dir = Path(treatment_dir)
        control_dir = Path(control_dir)
        if not treatment_dir.exists() or not control_dir.exists():
            continue
        treatment_seeds = {_seed_id(p): p for p in _seed_dirs(treatment_dir)}
        control_seeds = {_seed_id(p): p for p in _seed_dirs(control_dir)}
        common_seeds = sorted(set(treatment_seeds) & set(control_seeds), key=str)
        if not common_seeds:
            continue
        resolved_methods = list(methods) if methods is not None else None
        if resolved_methods is None:
            benchmark_path = treatment_seeds[common_seeds[0]] / "benchmark_results.csv"
            resolved_methods = (
                pd.read_csv(benchmark_path)["method"].astype(str).unique().tolist()
                if benchmark_path.exists()
                else []
            )
        for method in resolved_methods:
            for statistic in statistics:
                stat_index = 0 if statistic == "std" else 1
                matrix = _pair_seed_difference_matrix(
                    treatment_dir, control_dir, method, stat_index, _series
                )
                if matrix is None:
                    continue
                if months is not None:
                    index = pd.DatetimeIndex(matrix.index)
                    matrix = matrix.loc[np.asarray(months(index), dtype=bool)]
                    if matrix.empty:
                        continue
                averaged = matrix.mean(axis=1).dropna()
                if len(averaged) < 3:
                    continue
                test = newey_west_mean_test(averaged, max_lag=hac_lag)
                interval = moving_block_bootstrap_mean(
                    averaged,
                    block_length=block_length,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
                per_seed_means = matrix.mean(axis=0)
                pooled_sign = np.sign(float(averaged.mean()))
                agreement = (
                    float((np.sign(per_seed_means) == pooled_sign).mean())
                    if pooled_sign != 0
                    else np.nan
                )
                rows.append(
                    {
                        "contrast": label,
                        "period": period_label,
                        "method": method,
                        "statistic": statistic,
                        "n_seeds": int(matrix.shape[1]),
                        "n_months": int(len(averaged)),
                        "effect": float(averaged.mean()),
                        "hac_t_statistic": float(test["t_statistic"]),
                        "hac_p_value": float(test["p_value"]),
                        "ci_lower": float(interval["ci_lower"]),
                        "ci_upper": float(interval["ci_upper"]),
                        "excludes_zero": bool(
                            interval["ci_lower"] > 0 or interval["ci_upper"] < 0
                        ),
                        "seed_sign_agreement": agreement,
                        "block_length": int(interval["block_length"]),
                        "n_bootstrap": int(interval["n_bootstrap"]),
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        # FDR control across the whole reported family, not per contrast.
        frame["hac_q_value"] = benjamini_hochberg(frame["hac_p_value"])
    return frame

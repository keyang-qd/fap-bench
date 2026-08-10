from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fapbench.evaluation.significance import (
    aggregate_seed_significance,
    combine_seed_intervals,
    combine_seed_tests,
    heterogeneity_dispersion_hac_contrasts,
    identification_hac_contrasts,
    seed_averaged_dispersion_contrasts,
)
from fapbench.rolling import (
    ROLLING_LINEAR_TUNED20_PHASE,
    ROLLING_TUNED20_PHASE,
)
from fapbench.utils import write_json


RAW_METHODS = ["centralized", "local", "fedavg", "fedprox", "scaffold", "fedadam"]
COMPARISON_METHODS = ["centralized", "fedavg", "scaffold", "fedadam"]

# The headline protocol. This is the investable universe (microcaps excluded),
# not the unrestricted one: a reviewer objected that the asset-pricing
# literature would reject a headline configuration whose predictability sits in
# the microcap segment, and the sensitivity run confirmed it does -- Centralized
# Rank IC falls 0.0452 -> 0.0111 when microcaps are dropped. The unrestricted
# universe is still reported, as a sensitivity, under UNRESTRICTED_SCENARIO.
#
# Both budgets are emitted because the reference allocation (50 rounds x 1 local
# epoch) leaves FedProx indistinguishable from FedAvg; the 10x5 allocation is
# where the proximal term has room to act.
REFERENCE_SCENARIO = "ff12_exclude_microcaps"
REFERENCE_SCENARIO_10X5 = "ff12_exclude_microcaps_more_local_steps"
UNRESTRICTED_SCENARIO = "ff12_full_participation"

# Shared by _identification_contrasts (seed-level mean/std) and the HAC
# significance pass in build_publication_outputs (within-seed time-series
# test), so both report contrasts for exactly the same treatment/control pairs.
IDENTIFICATION_PAIRS: list[tuple[str, str, str]] = [
    (
        "size_imbalance_ratio_20_identified",
        "size_uniform_matched_ratio_20",
        "size imbalance minus matched retention",
    ),
    (
        "temporal_persistent_strong_identified",
        "temporal_random_matched_persistent_strong",
        "persistent timing minus matched random",
    ),
    (
        "temporal_block_outages_identified",
        "temporal_random_matched_block_outages",
        "block timing minus matched random",
    ),
]

# Compares each heterogeneity scenario against the core reference protocol on
# cross-client Rank-IC dispersion (std across clients, and the worst single
# client) rather than the client-neutral mean net-return series tested by
# IDENTIFICATION_PAIRS above. covariate_shift/feature_heterogeneity apply a
# randomly-signed per-client perturbation (see
# fapbench.partition.statistical.build_covariate_shift_parameters), so a
# mean-based test can largely cancel out even when the treatment clearly
# widens the spread across clients; centralized is excluded from the feature
# scenarios because they report centralized_masked/centralized_oracle instead.
HETEROGENEITY_DISPERSION_SCENARIOS: list[tuple[str, str, list[str]]] = [
    (
        "covariate_shift_mild",
        "Cov. mild",
        ["centralized", "fedadam", "fedavg", "scaffold"],
    ),
    (
        "covariate_shift_strong",
        "Cov. strong",
        ["centralized", "fedadam", "fedavg", "scaffold"],
    ),
    ("feature_heterogeneity_mild", "Feature mild", ["fedadam", "fedavg", "scaffold"]),
    (
        "feature_heterogeneity_moderate",
        "Feature moderate",
        ["fedadam", "fedavg", "scaffold"],
    ),
    (
        "feature_heterogeneity_severe",
        "Feature severe",
        ["fedadam", "fedavg", "scaffold"],
    ),
]

# Compares each Family A heterogeneity treatment (covariate shift, feature
# heterogeneity) against its own matched control (covariate_shift_*_matched /
# feature_heterogeneity_*_matched in configs/icaif_heterogeneity.yaml),
# closing the identification-rigor gap with IDENTIFICATION_PAIRS above:
# Family B (size imbalance, temporal outages) already had a matched-control
# comparison; Family A previously only had a plain-reference comparison
# (HETEROGENEITY_DISPERSION_SCENARIOS). Unlike IDENTIFICATION_PAIRS, methods
# are auto-detected per pair below (return_series_hac_contrasts /
# heterogeneity_dispersion_hac_contrasts read them from each pair's own
# benchmark_results.csv) rather than hardcoded, since a treatment and its
# matched control are always the same scenario family (both covariate or both
# feature) and therefore report the same method set.
HETEROGENEITY_MATCHED_CONTROL_PAIRS: list[tuple[str, str, str]] = [
    (
        "covariate_shift_mild",
        "covariate_shift_mild_matched",
        "covariate shift mild minus matched control",
    ),
    (
        "covariate_shift_strong",
        "covariate_shift_strong_matched",
        "covariate shift strong minus matched control",
    ),
    (
        "feature_heterogeneity_mild",
        "feature_heterogeneity_mild_matched",
        "feature heterogeneity mild minus matched control",
    ),
    (
        "feature_heterogeneity_moderate",
        "feature_heterogeneity_moderate_matched",
        "feature heterogeneity moderate minus matched control",
    ),
    (
        "feature_heterogeneity_severe",
        "feature_heterogeneity_severe_matched",
        "feature heterogeneity severe minus matched control",
    ),
]


def _numeric_mean(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in {"seed"}
    ]
    if not numeric:
        return frame[group_cols].drop_duplicates().reset_index(drop=True)
    return frame.groupby(group_cols, as_index=False)[numeric].mean()


def select_fedadam_validation(
    results_path: Path,
    *,
    scenario_prefix: str,
    method: str,
    metric: str,
    mode: str,
    constraint_metric: str,
    constraint_min: float,
) -> dict[str, Any]:
    if not metric.startswith("validation_") or not constraint_metric.startswith("validation_"):
        raise ValueError("FedAdam selection must use validation-only metrics.")
    frame = pd.read_csv(results_path)
    required = {"scenario", "method", metric, constraint_metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing validation-selection columns: {missing}")
    candidates = frame.loc[
        frame["scenario"].astype(str).str.startswith(scenario_prefix)
        & (frame["method"].astype(str) == method)
    ].copy()
    candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")
    candidates[constraint_metric] = pd.to_numeric(
        candidates[constraint_metric], errors="coerce"
    )
    candidates = candidates.dropna(subset=[metric, constraint_metric])
    if candidates.empty:
        raise ValueError("No FedAdam validation candidates were found.")
    summary = (
        candidates.groupby(["scenario", "method"], as_index=False)[
            [metric, constraint_metric]
        ]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    eligible = summary.loc[
        summary[f"{constraint_metric}_mean"] >= float(constraint_min)
    ].copy()
    if eligible.empty:
        raise ValueError("No FedAdam candidate satisfies the validation R2 constraint.")
    eligible = eligible.sort_values(
        f"{metric}_mean", ascending=(mode == "min"), kind="stable"
    )
    chosen = eligible.iloc[0]
    chosen_scenario = str(chosen["scenario"])
    config_candidates = sorted(
        results_path.parent.glob(f"{chosen_scenario}/seed_*/resolved_config.json")
    )
    if not config_candidates:
        raise FileNotFoundError(
            f"No resolved_config.json was found for {chosen_scenario}."
        )
    resolved = json.loads(config_candidates[0].read_text(encoding="utf-8"))
    server_lr = float(resolved["federated"]["server_lr"])
    return {
        "selection_source": str(results_path.resolve()),
        "selection_scope": "validation_only",
        "scenario_prefix": scenario_prefix,
        "method_filter": method,
        "metric": metric,
        "mode": mode,
        "constraint_metric": constraint_metric,
        "constraint_min": float(constraint_min),
        "chosen_scenario": chosen_scenario,
        "chosen_method": str(chosen["method"]),
        "selected_server_lr": server_lr,
        "validation_mean": float(chosen[f"{metric}_mean"]),
        "validation_std": (
            float(chosen[f"{metric}_std"])
            if pd.notna(chosen[f"{metric}_std"])
            else None
        ),
        "validation_constraint_mean": float(
            chosen[f"{constraint_metric}_mean"]
        ),
        "n_seeds": int(chosen[f"{metric}_count"]),
        "candidate_summary": summary.to_dict(orient="records"),
        "eligible_candidates": eligible.to_dict(orient="records"),
    }


def combine_phase_results(output_root: Path, phases: Iterable[str]) -> pd.DataFrame:
    frames = []
    for phase in phases:
        path = output_root / phase / "grid_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "suite", phase)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [c for c in ["suite", "scenario", "seed", "method"] if c in combined]
    return combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def _aggregate_with_uncertainty(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != "seed"
    ]
    summary = frame.groupby(["suite", "scenario", "method"], as_index=False)[numeric].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def _available(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def reference_cell_mde(
    output_root: Path,
    *,
    phase: str = "core",
    scenario: str = REFERENCE_SCENARIO,
    reference_method: str = "centralized",
    methods: Iterable[str] = ("fedavg", "scaffold", "fedadam", "local"),
    power: float = 0.80,
    alpha: float = 0.05,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Minimum detectable client-neutral net Sharpe gap at the reference protocol.

    The paper calls the reference cell a null-calibration configuration and says
    it cannot rank methods. A reviewer's sharpest point was that this is asserted
    rather than quantified: without a minimum detectable effect, a future
    submitter cannot tell whether their reported improvement is meaningful, which
    is the one thing a benchmark must supply.

    This computes it directly. For each method the seed-averaged monthly
    client-neutral net-return difference against ``reference_method`` is formed,
    its Newey-West standard error taken, and the two-sided MDE at the requested
    power expressed first as a monthly mean return and then as an annualized
    Sharpe gap using the reference arm's monthly return volatility.
    """
    from scipy.stats import norm

    from fapbench.evaluation.statistics import newey_west_mean_test

    scenario_dir = Path(output_root) / phase / scenario
    if not scenario_dir.exists():
        return pd.DataFrame()
    seed_dirs = sorted(scenario_dir.glob("seed_*"))
    if not seed_dirs:
        return pd.DataFrame()

    def _series(seed_dir: Path, method: str) -> pd.Series | None:
        path = seed_dir / f"portfolio_returns_client_neutral_{method}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        date_cols = [c for c in frame.columns if "date" in c.lower()]
        if not date_cols or "net_return" not in frame:
            return None
        return frame.set_index(date_cols[0])["net_return"]

    reference_panel = [s for s in (_series(d, reference_method) for d in seed_dirs) if s is not None]
    if not reference_panel:
        return pd.DataFrame()
    reference_volatility = float(pd.concat(reference_panel, axis=1).mean(axis=1).std())
    multiplier = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))

    rows: list[dict[str, Any]] = []
    for method in methods:
        differences = []
        for seed_dir in seed_dirs:
            treatment, control = _series(seed_dir, method), _series(seed_dir, reference_method)
            if treatment is None or control is None:
                continue
            aligned = pd.concat([treatment, control], axis=1, join="inner").dropna()
            if aligned.empty:
                continue
            differences.append(aligned.iloc[:, 0] - aligned.iloc[:, 1])
        if not differences:
            continue
        averaged = pd.concat(differences, axis=1).mean(axis=1).dropna()
        if len(averaged) < 3:
            continue
        test = newey_west_mean_test(averaged)
        t_statistic = float(test["t_statistic"])
        mean_difference = float(averaged.mean())
        standard_error = (
            abs(mean_difference / t_statistic) if t_statistic != 0 else float("nan")
        )
        mde_monthly = multiplier * standard_error
        rows.append(
            {
                "method": method,
                "reference_method": reference_method,
                "n_months": int(len(averaged)),
                "n_seeds": len(differences),
                "mean_monthly_difference": mean_difference,
                "hac_standard_error": standard_error,
                "power": power,
                "alpha": alpha,
                "mde_monthly_net_return": mde_monthly,
                "reference_monthly_volatility": reference_volatility,
                "mde_annualized_sharpe_gap": (
                    mde_monthly / reference_volatility * np.sqrt(periods_per_year)
                    if reference_volatility > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def economic_power_prescription(
    output_root: Path,
    *,
    reference_method: str = "centralized",
    power: float = 0.80,
    crsp_history_months: int = 1176,
) -> pd.DataFrame:
    """How much data would make the economic axis discriminative?

    A benchmark that reports "no method separates" owes future submitters more
    than the diagnosis: it should say what would make the instrument sharp. A
    reviewer asked exactly this -- what sample size would resolve the economic
    axis, and whether it is reachable with CRSP at all.

    The minimum detectable effect scales as 1/sqrt(n), so for an observed gap g
    and a detectable gap of ``mde`` at ``n`` months, the sample that would put
    the gap at the detection threshold is ``n * (mde / g)^2``. Applying it to the
    408-month cell turns an open null into a bounded statement about what this
    task can support.

    ``crsp_history_months`` is the full CRSP monthly window (1926--2023) and is
    the ceiling any US equity study of this design could reach, so the
    ``exceeds_crsp_history`` column is the operative answer rather than the raw
    month count.
    """
    tables = Path(output_root) / "paper_tables"
    mde_path = tables / "rolling_cell_mde.csv"
    pooled_path = tables / "rolling_pooled_metrics_investable.csv"
    if not mde_path.exists() or not pooled_path.exists():
        return pd.DataFrame()
    mde = pd.read_csv(mde_path).set_index("method")
    pooled = pd.read_csv(pooled_path).set_index("method")
    if reference_method not in pooled.index:
        return pd.DataFrame()
    reference_sharpe = float(pooled.loc[reference_method, "net_sharpe"])

    rows: list[dict[str, Any]] = []
    for method in mde.index:
        if method not in pooled.index:
            continue
        gap = abs(float(pooled.loc[method, "net_sharpe"]) - reference_sharpe)
        detectable = float(mde.loc[method, "mde_annualized_sharpe_gap"])
        observed_months = int(mde.loc[method, "n_months"])
        if gap <= 0 or not np.isfinite(detectable):
            continue
        required = observed_months * (detectable / gap) ** 2
        rows.append(
            {
                "method": method,
                "reference_method": reference_method,
                "observed_months": observed_months,
                "observed_sharpe_gap": gap,
                "detectable_sharpe_gap": detectable,
                "power": power,
                "required_months": required,
                "required_years": required / 12.0,
                "crsp_history_months": crsp_history_months,
                "exceeds_crsp_history": bool(required > crsp_history_months),
            }
        )
    return pd.DataFrame(rows).sort_values("required_months").reset_index(drop=True)


def select_expanded_capacity(
    results_path: Path,
    *,
    scenario_prefix: str = "expanded_capacity_",
    method: str = "centralized",
    metric: str = "validation_selected_rank_ic",
    mode: str = "max",
) -> dict[str, Any]:
    """Pick the 20-characteristic architecture/learning rate on validation only.

    The manuscript attributes the 20-characteristic arm's Rank IC collapse to
    holding architecture and optimizer at seven-input values, and therefore
    declines to read that arm as a verdict on the wider panel. That reading is
    honest but untested unless a re-tuned arm is actually run.

    The guard here is the point of the function: like the FedAdam server-rate
    selection it refuses any metric not prefixed ``validation_``. Selecting a
    capacity on test Rank IC and then reporting that same test Rank IC would
    manufacture exactly the recovery the sweep is supposed to test for, and it
    would be invisible in the output.

    Returns the winning scenario together with the full ranking, so the margin
    over the reference architecture is inspectable rather than implicit.
    """
    if not metric.startswith("validation_"):
        raise ValueError(
            "Capacity selection must use a validation-only metric; got "
            f"{metric!r}. Selecting on test metrics would fabricate the recovery "
            "this sweep exists to test."
        )
    frame = pd.read_csv(results_path)
    required = {"scenario", "method", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing capacity-selection columns: {missing}")
    candidates = frame.loc[
        frame["scenario"].astype(str).str.startswith(scenario_prefix)
        & (frame["method"].astype(str) == method)
    ].copy()
    candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")
    candidates = candidates.dropna(subset=[metric])
    if candidates.empty:
        raise ValueError(f"No capacity candidates matched prefix {scenario_prefix!r}.")
    summary = (
        candidates.groupby("scenario", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=(mode != "max"))
        .reset_index(drop=True)
    )
    winner = summary.iloc[0]
    reference = summary.loc[summary["scenario"].str.contains("h64_32")]
    return {
        "selected_scenario": str(winner["scenario"]),
        "selected_value": float(winner[metric]),
        "metric": metric,
        "method": method,
        "n_candidates": int(len(summary)),
        "reference_best_value": (
            float(reference[metric].max()) if not reference.empty else float("nan")
        ),
        "ranking": summary.to_dict(orient="records"),
    }


def client_local_level_comparison(
    output_root: Path,
    *,
    global_phase: str = "core",
    local_phase: str = "client_local_levels",
    methods: Iterable[str] = (
        "centralized",
        "local",
        "fedavg",
        "fedprox",
        "scaffold",
        "fedadam",
    ),
    hac_lag: int | None = None,
) -> pd.DataFrame:
    """Levels under global vs client-local standardization, scenario by scenario.

    The benchmark's default preprocessing ranks and z-scores within month over the
    *full* cross-section, which no client in the stated deployment could compute --
    a within-month cross-sectional rank needs every client's holdings at once.
    Three reviewers flagged this as the paper being strict about temporal leakage
    and loose about the cross-silo leakage that is its actual subject.

    The existing sensitivity covers RQ3's dispersion contrast only. This covers the
    *levels*: Table 3's two budgets and the RQ2 universe lever, with the same seeds
    as the published grid so preprocessing is the only thing that differs.

    Reports the paired monthly net-return difference with a Newey-West test, using
    the same seed-averaged, calendar-month-as-unit convention as the rest of the
    paper. The two arms share a market history and a seed set, so the pairing is
    within-seed and the difference isolates preprocessing.
    """
    from fapbench.evaluation.statistics import newey_west_mean_test

    global_dir = Path(output_root) / global_phase
    local_dir = Path(output_root) / local_phase
    if not local_dir.exists():
        return pd.DataFrame()

    def _returns(seed_dir: Path, method: str) -> pd.Series | None:
        path = seed_dir / f"portfolio_returns_client_neutral_{method}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        date_cols = [c for c in frame.columns if "date" in c.lower()]
        if not date_cols or "net_return" not in frame:
            return None
        return frame.set_index(date_cols[0])["net_return"]

    def _metric(seed_dir: Path, method: str, column: str) -> float | None:
        path = seed_dir / "benchmark_results.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        row = frame.loc[frame["method"].astype(str) == method]
        if row.empty or column not in row:
            return None
        return float(row.iloc[0][column])

    def _monthly_rank_ic(seed_dir: Path, method: str) -> pd.Series | None:
        """Per-month Rank IC, so the predictive change can be tested rather than
        eyeballed. The Rank IC shift is the larger of the two effects here, so it
        needs the same inferential standard as the economic one."""
        path = seed_dir / f"monthly_predictive_metrics_{method}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        if "target_date" not in frame or "rank_ic" not in frame:
            return None
        return frame.set_index("target_date")["rank_ic"]

    rows: list[dict[str, Any]] = []
    for scenario_dir in sorted(p for p in local_dir.iterdir() if p.is_dir()):
        reference_dir = global_dir / scenario_dir.name
        if not reference_dir.exists():
            continue
        local_seeds = {p.name: p for p in sorted(scenario_dir.glob("seed_*"))}
        global_seeds = {p.name: p for p in sorted(reference_dir.glob("seed_*"))}
        shared = sorted(set(local_seeds) & set(global_seeds))
        if not shared:
            continue
        for method in methods:
            differences, local_ic, global_ic, local_sh, global_sh = [], [], [], [], []
            ic_differences: list[pd.Series] = []
            for seed in shared:
                a = _returns(local_seeds[seed], method)
                b = _returns(global_seeds[seed], method)
                if a is not None and b is not None:
                    aligned = pd.concat([a, b], axis=1, join="inner").dropna()
                    if not aligned.empty:
                        differences.append(aligned.iloc[:, 0] - aligned.iloc[:, 1])
                ic_a = _monthly_rank_ic(local_seeds[seed], method)
                ic_b = _monthly_rank_ic(global_seeds[seed], method)
                if ic_a is not None and ic_b is not None:
                    ic_aligned = pd.concat([ic_a, ic_b], axis=1, join="inner").dropna()
                    if not ic_aligned.empty:
                        ic_differences.append(
                            ic_aligned.iloc[:, 0] - ic_aligned.iloc[:, 1]
                        )
                for store, directory, column in (
                    (local_ic, local_seeds[seed], "rank_ic"),
                    (global_ic, global_seeds[seed], "rank_ic"),
                    (local_sh, local_seeds[seed], "client_neutral_portfolio_net_sharpe"),
                    (global_sh, global_seeds[seed], "client_neutral_portfolio_net_sharpe"),
                ):
                    value = _metric(directory, method, column)
                    if value is not None:
                        store.append(value)
            if not differences:
                continue
            averaged = pd.concat(differences, axis=1).mean(axis=1).dropna()
            if len(averaged) < 3:
                continue
            test = newey_west_mean_test(averaged, max_lag=hac_lag)
            ic_p, ic_t = np.nan, np.nan
            if ic_differences:
                ic_averaged = pd.concat(ic_differences, axis=1).mean(axis=1).dropna()
                if len(ic_averaged) >= 3:
                    ic_test = newey_west_mean_test(ic_averaged, max_lag=hac_lag)
                    ic_t = float(ic_test["t_statistic"])
                    ic_p = float(ic_test["p_value"])
            rows.append(
                {
                    "scenario": scenario_dir.name,
                    "method": method,
                    "n_seeds": len(shared),
                    "n_months": int(len(averaged)),
                    "rank_ic_hac_t_statistic": ic_t,
                    "rank_ic_hac_p_value": ic_p,
                    "rank_ic_global": float(np.mean(global_ic)) if global_ic else np.nan,
                    "rank_ic_client_local": float(np.mean(local_ic)) if local_ic else np.nan,
                    "net_sharpe_global": float(np.mean(global_sh)) if global_sh else np.nan,
                    "net_sharpe_client_local": float(np.mean(local_sh)) if local_sh else np.nan,
                    "mean_monthly_return_difference": float(averaged.mean()),
                    "hac_t_statistic": float(test["t_statistic"]),
                    "hac_p_value": float(test["p_value"]),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["rank_ic_delta"] = frame["rank_ic_client_local"] - frame["rank_ic_global"]
        frame["net_sharpe_delta"] = (
            frame["net_sharpe_client_local"] - frame["net_sharpe_global"]
        )
        from fapbench.evaluation.statistics import benjamini_hochberg

        frame["hac_q_value"] = benjamini_hochberg(frame["hac_p_value"])
    return frame


def rolling_cell_mde(
    output_root: Path,
    *,
    phase: str = "rolling_refit_investable",
    reference_method: str = "centralized",
    methods: Iterable[str] = ("fedavg", "scaffold", "fedadam", "local"),
    power: float = 0.80,
    alpha: float = 0.05,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Minimum detectable net Sharpe gap on the 408-month expanding-window sample.

    The manuscript quantifies the MDE at the 60-month reference cell and then
    asserts that the 408-month result is "a null, not a power limit". A reviewer
    correctly identified that as the one place where the paper asserts precisely
    what it elsewhere insists on computing: without the MDE at n=408 the claim is
    a hope about sample size, not a statement about it.

    This computes it with the same estimator used at n=60, differing only in that
    each window contributes its own disjoint block of test months, concatenated
    into one series before the Newey-West standard error is taken.
    """
    from scipy.stats import norm

    from fapbench.evaluation.statistics import newey_west_mean_test

    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    windows = sorted(p for p in phase_dir.glob("refit_test_*") if p.is_dir())
    if not windows:
        return pd.DataFrame()

    def _series(seed_dir: Path, method: str) -> pd.Series | None:
        path = seed_dir / f"portfolio_returns_client_neutral_{method}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        date_cols = [c for c in frame.columns if "date" in c.lower()]
        if not date_cols or "net_return" not in frame:
            return None
        return frame.set_index(date_cols[0])["net_return"]

    def _pooled(method: str) -> pd.Series | None:
        """Seed-averaged monthly series, concatenated across disjoint windows."""
        blocks = []
        for window in windows:
            per_seed = [
                s
                for s in (_series(d, method) for d in sorted(window.glob("seed_*")))
                if s is not None
            ]
            if per_seed:
                blocks.append(pd.concat(per_seed, axis=1).mean(axis=1))
        if not blocks:
            return None
        pooled = pd.concat(blocks).sort_index()
        return pooled[~pooled.index.duplicated()]

    reference_series = _pooled(reference_method)
    if reference_series is None:
        return pd.DataFrame()
    reference_volatility = float(reference_series.std())
    multiplier = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))

    rows: list[dict[str, Any]] = []
    for method in methods:
        series = _pooled(method)
        if series is None:
            continue
        aligned = pd.concat([series, reference_series], axis=1, join="inner").dropna()
        if len(aligned) < 3:
            continue
        difference = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        test = newey_west_mean_test(difference)
        t_statistic = float(test["t_statistic"])
        mean_difference = float(difference.mean())
        standard_error = (
            abs(mean_difference / t_statistic) if t_statistic != 0 else float("nan")
        )
        mde_monthly = multiplier * standard_error
        rows.append(
            {
                "method": method,
                "reference_method": reference_method,
                "n_windows": len(windows),
                "n_months": int(len(difference)),
                "mean_monthly_difference": mean_difference,
                "hac_standard_error": standard_error,
                "mde_monthly_net_return": mde_monthly,
                "reference_monthly_volatility": reference_volatility,
                "mde_annualized_sharpe_gap": (
                    mde_monthly / reference_volatility * np.sqrt(periods_per_year)
                    if reference_volatility > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def rolling_seed_variance(
    output_root: Path,
    *,
    phase: str = "rolling_refit_investable",
    reference_method: str = "centralized",
    methods: Iterable[str] = ("centralized", "local", "fedavg", "scaffold", "fedadam"),
    block_length: int = 6,
    n_bootstrap: int = 5000,
    power: float = 0.80,
    alpha: float = 0.05,
    periods_per_year: int = 12,
    random_state: int = 20260804,
) -> pd.DataFrame:
    """Seed uncertainty in the 408-month cell, which the headline numbers average out.

    Two distinct things get conflated when a benchmark averages over seeds, and a
    reviewer was right that this paper reported neither.

    First, the *point estimate*. Table 2's net Sharpe is the Sharpe of the
    seed-averaged monthly series, so it describes a 10-seed ensemble rather than
    a run anyone would deploy. Averaging predictions across seeds diversifies away
    seed-specific noise, and the effect is not small: the mean of the per-seed
    Sharpes is roughly half the Sharpe of the averaged series. Both belong in the
    paper, because they answer different questions -- "what does this method
    achieve here" against "what would one training run achieve".

    Second, the *interval*. rolling_cell_mde takes a Newey-West standard error on
    the seed-averaged difference series, which propagates month-to-month
    uncertainty and none of the across-seed uncertainty. This resamples both: seeds
    with replacement, and month blocks with replacement, in one two-stage
    bootstrap. To keep the comparison honest, the time-only interval is recomputed
    from the *same* bootstrap rather than compared against the Newey-West figure --
    otherwise the ratio would confound the estimator with the resampling design.
    """
    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    windows = sorted(p for p in phase_dir.glob("refit_test_*") if p.is_dir())
    if not windows:
        return pd.DataFrame()

    def _series(seed_dir: Path, method: str) -> pd.Series | None:
        path = seed_dir / f"portfolio_returns_client_neutral_{method}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        date_cols = [c for c in frame.columns if "date" in c.lower()]
        if not date_cols or "net_return" not in frame:
            return None
        return frame.set_index(date_cols[0])["net_return"]

    def _panel(method: str) -> pd.DataFrame:
        """Months x seeds, concatenated across the disjoint refit windows."""
        seed_names = sorted({d.name for w in windows for d in w.glob("seed_*")})
        columns: dict[str, pd.Series] = {}
        for seed in seed_names:
            blocks = [
                s
                for s in (
                    _series(w / seed, method) for w in windows if (w / seed).exists()
                )
                if s is not None
            ]
            if blocks:
                columns[seed] = pd.concat(blocks)
        return pd.DataFrame(columns)

    def _sharpe(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size < 3:
            return float("nan")
        deviation = values.std(ddof=1)
        if deviation == 0:
            return float("nan")
        return float(np.sqrt(periods_per_year) * values.mean() / deviation)

    rng = np.random.default_rng(random_state)

    def _block_index(n_months: int) -> np.ndarray:
        starts = rng.integers(0, n_months - block_length + 1,
                              size=int(np.ceil(n_months / block_length)))
        return np.concatenate(
            [np.arange(s, s + block_length) for s in starts]
        )[:n_months]

    from scipy.stats import norm

    multiplier = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    reference_panel = _panel(reference_method)
    if reference_panel.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for method in methods:
        panel = _panel(method)
        if panel.empty:
            continue
        averaged = panel.mean(axis=1)
        per_seed = np.array(
            [_sharpe(panel[c].dropna().to_numpy()) for c in panel.columns]
        )
        per_seed = per_seed[np.isfinite(per_seed)]
        row: dict[str, Any] = {
            "method": method,
            "n_seeds": int(len(per_seed)),
            "n_months": int(len(averaged)),
            "net_sharpe_seed_averaged": _sharpe(averaged.to_numpy()),
            "net_sharpe_per_seed_mean": float(per_seed.mean()) if len(per_seed) else np.nan,
            "net_sharpe_per_seed_sd": (
                float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan
            ),
        }
        if method != reference_method:
            shared = [c for c in panel.columns if c in reference_panel.columns]
            difference = (panel[shared] - reference_panel[shared]).dropna()
            matrix = difference.to_numpy()
            n_months, n_seeds = matrix.shape
            time_only, joint = [], []
            for _ in range(n_bootstrap):
                months = _block_index(n_months)
                time_only.append(matrix[months, :].mean(axis=1).mean())
                seeds = rng.integers(0, n_seeds, size=n_seeds)
                joint.append(matrix[np.ix_(months, seeds)].mean(axis=1).mean())
            t_lo, t_hi = np.percentile(time_only, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            j_lo, j_hi = np.percentile(joint, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            z = 2 * float(norm.ppf(1.0 - alpha / 2.0))
            se_time, se_joint = (t_hi - t_lo) / z, (j_hi - j_lo) / z
            volatility = float(difference.mean(axis=1).std(ddof=1))
            scale = (
                np.sqrt(periods_per_year) / volatility if volatility > 0 else float("nan")
            )
            row.update(
                {
                    "mean_monthly_difference": float(matrix.mean(axis=1).mean()),
                    "ci_lower_time_only": float(t_lo),
                    "ci_upper_time_only": float(t_hi),
                    "ci_lower_joint": float(j_lo),
                    "ci_upper_joint": float(j_hi),
                    "ci_width_ratio_joint_over_time": (
                        float((j_hi - j_lo) / (t_hi - t_lo)) if t_hi > t_lo else np.nan
                    ),
                    "mde_annualized_sharpe_time_only": float(
                        multiplier * se_time * scale
                    ),
                    "mde_annualized_sharpe_joint": float(multiplier * se_joint * scale),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


# Every table the manuscript quotes a *descriptive* p-value from, with the column
# that carries it. The matched-control family (74 tests) is deliberately absent:
# it already carries q values, and re-correcting it here would double-count.
# client_local_level_comparison is absent for the same reason -- the manuscript
# quotes it as "0/18 survive correction", so it is corrected already.
DESCRIPTIVE_TEST_TABLES: tuple[tuple[str, str], ...] = (
    ("rolling_vs_centralized_investable", "hac_p_value"),
    ("rolling_pooled_metrics_investable", "rank_ic_p"),
    # The paired Rank IC contrasts. They were narrated as a finding while
    # sitting outside every family the paper counts, which a reviewer correctly
    # called an inferential inconsistency; now they are tested like the
    # net-return contrasts and counted like them too.
    ("rolling_rank_ic_vs_centralized_investable", "rank_ic_p"),
    ("rolling_rank_ic_vs_centralized_tuned20", "rank_ic_p"),
    ("robustness_lever_significance", "p"),
    ("linear_separation", "hac_p_value"),
    ("ensemble_local_comparison", "combined_p_value"),
    ("robustness_fedprox_mu", "rank_ic_p"),
    ("robustness_fedprox_mu", "client_neutral_portfolio_net_sharpe_p"),
    ("finetune_per_client", "paired_p"),
    ("positive_control", "rank_ic_p"),
    ("positive_control", "net_return_p"),
)


def descriptive_test_census(tables_dir: Path) -> pd.DataFrame:
    """Count -- and re-correct -- every descriptive test the manuscript quotes.

    The manuscript applies Benjamini-Hochberg once, across one declared family of
    74 matched-control tests, and marks everything else descriptive. A reviewer
    objected that a paper which demotes an entire experimental cell for
    insufficient power should not then carry an uncounted tail of uncorrected
    comparisons, and asked for either a wider correction or a stated count.

    This gives both. It enumerates the descriptive tests from the tables that back
    them, so the count is derived rather than asserted, and applies BH to them as
    a second family so the manuscript can say whether correcting them would have
    changed anything. Treating them as one family is the conservative reading:
    they are heterogeneous in purpose, so pooling them dilutes more than splitting
    would.
    """
    from fapbench.evaluation.statistics import benjamini_hochberg

    rows: list[dict[str, Any]] = []
    for stem, column in DESCRIPTIVE_TEST_TABLES:
        path = Path(tables_dir) / f"{stem}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if column not in frame.columns:
            continue
        for value in pd.to_numeric(frame[column], errors="coerce").dropna():
            rows.append({"source": stem, "column": column, "p_value": float(value)})
    census = pd.DataFrame(rows)
    if census.empty:
        return census
    census["q_value"] = benjamini_hochberg(census["p_value"])
    census["significant_uncorrected"] = census["p_value"] < 0.05
    census["significant_corrected"] = census["q_value"] < 0.05
    return census


def calibration_diagnostics(
    output_root: Path,
    *,
    phase: str = "core",
    scenario: str = REFERENCE_SCENARIO,
    methods: Iterable[str] = ("fedavg", "fedprox", "scaffold", "fedadam"),
) -> pd.DataFrame:
    """Validation-fitted affine calibration, applied to *every* federated arm.

    Table 3 reports a calibrated OOS R^2 for FedAdam alone, because FedAdam is the
    only arm whose raw R^2 is a visible outlier. A reviewer objected that this is
    not internally comparable: the calibrated FedAdam figure then sits above
    Centralized's uncalibrated one, inviting a comparison the numbers do not
    support. This computes the same calibration for all four federated arms so the
    claim can be checked rather than taken on faith.

    It also reports the constant-predictor benchmark. The validation fit shrinks
    the slope nearly to zero, so the calibrated predictor is close to an intercept
    -- and the calibrated R^2 is correspondingly close to what predicting a
    constant achieves. That is the honest reading of the calibrated number: it
    measures the unconditional mean, not method skill, which is precisely why the
    arms land on top of each other.
    """
    from fapbench.evaluation.calibration import fit_linear_prediction_calibrator

    scenario_dir = Path(output_root) / phase / scenario
    if not scenario_dir.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method in methods:
        raw_r2, cal_r2, slopes, intercepts = [], [], [], []
        for seed_dir in sorted(scenario_dir.glob("seed_*")):
            validation = seed_dir / f"validation_predictions_{method}.parquet"
            test = seed_dir / f"predictions_{method}.parquet"
            if not (validation.exists() and test.exists()):
                continue
            calibrator = fit_linear_prediction_calibrator(pd.read_parquet(validation))
            frame = pd.read_parquet(test)
            y = frame["y_true"].to_numpy(dtype=float)
            p = frame["y_pred"].to_numpy(dtype=float)
            denominator = float(np.sum(y**2))
            if denominator <= 0:
                continue
            calibrated = calibrator.intercept + calibrator.slope * p
            raw_r2.append(1.0 - float(np.sum((y - p) ** 2)) / denominator)
            cal_r2.append(1.0 - float(np.sum((y - calibrated) ** 2)) / denominator)
            slopes.append(calibrator.slope)
            intercepts.append(calibrator.intercept)
        if not cal_r2:
            continue
        rows.append(
            {
                "method": method,
                "n_seeds": len(cal_r2),
                "raw_oos_r2_zero": float(np.mean(raw_r2)),
                "calibrated_oos_r2_zero": float(np.mean(cal_r2)),
                "calibrated_oos_r2_zero_std": float(np.std(cal_r2)),
                "calibration_slope": float(np.mean(slopes)),
                "calibration_intercept": float(np.mean(intercepts)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # The intercept-only reference: what the *best* constant predictor scores on
    # the same test panel. If the calibrated arms sit near this, the calibrated
    # R^2 is carrying the mean rather than the model.
    any_test = next(iter(sorted(scenario_dir.glob("seed_*/predictions_fedavg.parquet"))), None)
    if any_test is not None:
        y = pd.read_parquet(any_test)["y_true"].to_numpy(dtype=float)
        best_constant = float(np.mean(y))
        frame["best_constant_predictor"] = best_constant
        frame["best_constant_oos_r2_zero"] = 1.0 - float(
            np.sum((y - best_constant) ** 2)
        ) / float(np.sum(y**2))
    return frame


def _main_results_table(
    core: pd.DataFrame, scenario: str = REFERENCE_SCENARIO
) -> pd.DataFrame:
    raw = core.loc[
        (core["scenario"] == scenario) & core["method"].isin(RAW_METHODS)
    ].copy()
    columns = _available(
        raw,
        [
            "rank_ic",
            "oos_r2_zero",
            "client_worst_rank_ic",
            "client_neutral_portfolio_net_sharpe",
            "selected_checkpoint_communicated_bytes_fp32",
            "communicated_bytes_fp32",
        ],
    )
    table = raw.groupby("method", as_index=False)[columns].agg(["mean", "std"])
    table.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in table.columns
    ]
    calibrated = core.loc[
        (core["scenario"] == scenario) & (core["method"] == "fedadam_calibrated")
    ]
    if not calibrated.empty and "oos_r2_zero" in calibrated:
        calibrated_value = float(pd.to_numeric(calibrated["oos_r2_zero"]).mean())
        table["calibrated_oos_r2_zero_mean"] = np.nan
        table.loc[
            table["method"] == "fedadam", "calibrated_oos_r2_zero_mean"
        ] = calibrated_value
    communication = None
    for candidate in [
        "selected_checkpoint_communicated_bytes_fp32_mean",
        "communicated_bytes_fp32_mean",
    ]:
        if candidate in table:
            communication = candidate
            break
    if communication:
        table["selected_communication_mb"] = table[communication] / 1_000_000.0
    order = {method: i for i, method in enumerate(RAW_METHODS)}
    table["_order"] = table["method"].map(order)
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _protocol_table(core: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        "ff12_full_participation",
        "ff12_half_participation",
        "ff12_more_local_steps",
        "ff12_price_above_5",
        "ff12_exclude_microcaps",
        # The investable universe at the 10x5 budget, so RQ2 can say whether the
        # microcap collapse is a property of the universe or of the training
        # schedule. Without this cell the two are confounded.
        "ff12_exclude_microcaps_more_local_steps",
    ]
    frame = core.loc[
        core["scenario"].isin(scenarios)
        & core["method"].isin(COMPARISON_METHODS)
    ].copy()
    columns = _available(
        frame,
        [
            "rank_ic",
            "oos_r2_zero",
            "client_neutral_portfolio_net_sharpe",
            "selected_checkpoint_communicated_bytes_fp32",
            "communicated_bytes_fp32",
        ],
    )
    return frame.groupby(["scenario", "method"], as_index=False)[columns].mean()


def _partition_table(partitions: pd.DataFrame) -> pd.DataFrame:
    frame = partitions.loc[partitions["method"].isin(COMPARISON_METHODS)].copy()
    columns = _available(
        frame,
        [
            "rank_ic",
            "oos_r2_zero",
            "global_portfolio_net_sharpe",
            "client_neutral_portfolio_net_sharpe",
            "heterogeneity_n_clients",
        ],
    )
    return frame.groupby(["scenario", "method"], as_index=False)[columns].mean()


def _heterogeneity_table(core: pd.DataFrame, heterogeneity: pd.DataFrame) -> pd.DataFrame:
    # Deliberately UNRESTRICTED_SCENARIO, not REFERENCE_SCENARIO: the
    # heterogeneity grid was run on the unrestricted universe, so its reference
    # arm must be the unrestricted core cell or the comparison would differ in
    # two things at once. RQ3's contrasts are treatment-minus-control within one
    # universe, so the universe cancels inside each contrast -- but the levels
    # here are not comparable to Table 3's, and the manuscript says so.
    reference = core.loc[
        (core["scenario"] == UNRESTRICTED_SCENARIO)
        & core["method"].isin(COMPARISON_METHODS)
    ].copy()
    reference["scenario"] = "reference"
    frame = pd.concat([reference, heterogeneity], ignore_index=True, sort=False)
    frame = frame.loc[frame["method"].isin(COMPARISON_METHODS)].copy()
    columns = _available(
        frame,
        [
            "rank_ic",
            "oos_r2_zero",
            "client_neutral_portfolio_net_sharpe",
            "heterogeneity_feature_distribution_wasserstein",
            "heterogeneity_feature_jaccard_dissimilarity",
        ],
    )
    return frame.groupby(["scenario", "method"], as_index=False)[columns].mean()


def _identification_contrasts(identification: pd.DataFrame) -> pd.DataFrame:
    pairs = IDENTIFICATION_PAIRS
    metrics = _available(
        identification,
        ["rank_ic", "oos_r2_zero", "client_neutral_portfolio_net_sharpe"],
    )
    rows = []
    for treatment, control, label in pairs:
        left = identification.loc[
            identification["scenario"] == treatment,
            ["seed", "method", *metrics],
        ].copy()
        right = identification.loc[
            identification["scenario"] == control,
            ["seed", "method", *metrics],
        ].copy()
        merged = left.merge(right, on=["seed", "method"], suffixes=("_t", "_c"))
        for method, group in merged.groupby("method"):
            row: dict[str, Any] = {
                "contrast": label,
                "treatment": treatment,
                "control": control,
                "method": method,
                "n_seeds": int(group["seed"].nunique()),
            }
            for metric in metrics:
                delta = pd.to_numeric(group[f"{metric}_t"], errors="coerce") - pd.to_numeric(
                    group[f"{metric}_c"], errors="coerce"
                )
                row[f"delta_{metric}_mean"] = float(delta.mean())
                row[f"delta_{metric}_std"] = float(delta.std())
            rows.append(row)
    return pd.DataFrame(rows)


def _label_table(label: pd.DataFrame) -> pd.DataFrame:
    columns = _available(
        label,
        [
            "heterogeneity_label_distribution_wasserstein",
            "heterogeneity_controlled_label_skew_retained_fraction",
            "heterogeneity_controlled_label_skew_concentration",
            "heterogeneity_n_observations",
        ],
    )
    return label.groupby(["scenario", "method"], as_index=False)[columns].mean()


def _expanded_features_table(
    core: pd.DataFrame, core_expanded: pd.DataFrame
) -> pd.DataFrame:
    """Tier 2 robustness check: does RQ1 replicate with a richer feature set?

    Compares the unrestricted-universe scenario under the original
    7-characteristic panel (core) against the same scenario under the
    20-characteristic panel (core_expanded_features, see
    EXPANDED_CHARACTERISTIC_COMPONENTS in scripts/data_recipe2_revised.py),
    for the methods common to both phases.

    Like _heterogeneity_table this stays on UNRESTRICTED_SCENARIO rather than the
    headline REFERENCE_SCENARIO, because the expanded-panel grid was run there;
    holding the universe fixed is what makes the panel the only thing varying.
    """
    if core.empty or core_expanded.empty:
        return pd.DataFrame()
    scenario = UNRESTRICTED_SCENARIO
    columns = _available(
        core,
        ["rank_ic", "oos_r2_zero", "client_neutral_portfolio_net_sharpe"],
    )
    tagged = []
    for label, frame in [("7_characteristics", core), ("20_characteristics", core_expanded)]:
        subset = frame.loc[
            (frame["scenario"] == scenario) & frame["method"].isin(RAW_METHODS)
        ].copy()
        if subset.empty:
            continue
        subset["feature_set"] = label
        tagged.append(subset[["feature_set", "method", *columns]])
    if not tagged:
        return pd.DataFrame()
    combined = pd.concat(tagged, ignore_index=True)
    table = combined.groupby(["feature_set", "method"], as_index=False)[columns].agg(
        ["mean", "std"]
    )
    table.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in table.columns
    ]
    return table


def _plot_grouped_bars(
    table: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    output: Path,
    title: str,
    ylabel: str,
) -> None:
    if table.empty or y not in table:
        return
    pivot = table.pivot(index=x, columns=group, values=y)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_identification(table: pd.DataFrame, output: Path) -> None:
    metric = "delta_client_neutral_portfolio_net_sharpe_mean"
    if table.empty or metric not in table:
        return
    _plot_grouped_bars(
        table,
        x="contrast",
        y=metric,
        group="method",
        output=output,
        title="Matched-control effects on net Sharpe",
        ylabel="Treatment minus control",
    )


def build_publication_outputs(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    phase_names = [
        "fedadam_validation",
        "core",
        "partitions",
        "heterogeneity",
        "identification",
        "label_diagnostics",
        # Tier 2 robustness check (EXPERIMENT_ENHANCEMENT_PLAN.md); optional --
        # only present when core_expanded_features was included in the run.
        "core_expanded_features",
    ]
    combined = combine_phase_results(output_root, phase_names)
    if combined.empty:
        raise RuntimeError("No completed grid_results.csv files were found.")
    combined.to_csv(output_root / "all_grid_results.csv", index=False)
    _aggregate_with_uncertainty(combined).to_csv(
        output_root / "all_grid_summary.csv", index=False
    )

    def phase(name: str) -> pd.DataFrame:
        path = output_root / name / "grid_results.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    core = phase("core")
    partitions = phase("partitions")
    heterogeneity = phase("heterogeneity")
    identification = phase("identification")
    label = phase("label_diagnostics")
    core_expanded_features = phase("core_expanded_features")

    tables_dir = output_root / "paper_tables"
    figures_dir = output_root / "paper_figures"
    tables = {
        "main_results": _main_results_table(core, REFERENCE_SCENARIO),
        # The 10x5 block of Table 3. It used to be transcribed into the LaTeX by
        # hand from the raw grid, so its 40 numbers were the only ones in the
        # headline table that audit_manuscript_numbers.py could not check.
        "main_results_10x5": _main_results_table(core, REFERENCE_SCENARIO_10X5),
        "main_results_unrestricted": _main_results_table(core, UNRESTRICTED_SCENARIO),
        "protocol_sensitivity": _protocol_table(core),
        "partition_comparison": _partition_table(partitions),
        "heterogeneity": _heterogeneity_table(core, heterogeneity),
        "identification_contrasts": _identification_contrasts(identification),
        "label_diagnostics": _label_table(label),
        "expanded_features_comparison": _expanded_features_table(
            core, core_expanded_features
        ),
    }
    reference_core_dir = output_root / "core" / REFERENCE_SCENARIO
    per_seed_tests, per_seed_intervals = aggregate_seed_significance(
        reference_core_dir,
        variant="client_neutral",
        reference_method="centralized",
    )
    tables["main_results_significance"] = combine_seed_tests(per_seed_tests)
    tables["main_results_sharpe_intervals"] = combine_seed_intervals(per_seed_intervals)
    tables["identification_hac_tests"] = identification_hac_contrasts(
        output_root / "identification",
        IDENTIFICATION_PAIRS,
        variant="client_neutral",
    )
    heterogeneity_dir = output_root / "heterogeneity"
    dispersion_frames = [
        heterogeneity_dispersion_hac_contrasts(
            [(heterogeneity_dir / scenario, reference_core_dir, label)],
            methods=methods,
        )
        for scenario, label, methods in HETEROGENEITY_DISPERSION_SCENARIOS
    ]
    tables["heterogeneity_dispersion_hac_tests"] = pd.concat(
        dispersion_frames, ignore_index=True
    )
    tables["heterogeneity_matched_control_hac_tests"] = identification_hac_contrasts(
        heterogeneity_dir,
        HETEROGENEITY_MATCHED_CONTROL_PAIRS,
        variant="client_neutral",
    )
    matched_control_dispersion_frames = [
        heterogeneity_dispersion_hac_contrasts(
            [(heterogeneity_dir / treatment, heterogeneity_dir / control, label)]
        )
        for treatment, control, label in HETEROGENEITY_MATCHED_CONTROL_PAIRS
    ]
    tables["heterogeneity_matched_control_dispersion_hac_tests"] = pd.concat(
        matched_control_dispersion_frames, ignore_index=True
    )
    # The placebo. Runs the dispersion estimator with no effect to find: each
    # control arm differenced against itself across disjoint seed pairs. One
    # control per family is enough to establish the noise floor, and it must
    # include a system-level one, since that is the side whose reported effects
    # are small enough for the floor to matter.
    from fapbench.evaluation.significance import matched_control_placebo

    tables["matched_control_placebo"] = matched_control_placebo(
        [
            (
                heterogeneity_dir / "feature_heterogeneity_severe_matched",
                "feature severe control",
            ),
            (
                heterogeneity_dir / "covariate_shift_strong_matched",
                "covariate strong control",
            ),
            (
                output_root / "identification" / "temporal_random_matched_block_outages",
                "block outages control",
            ),
        ],
        methods=["fedavg", "scaffold", "fedadam"],
    )
    # Same dispersion test applied to the system-level matched-control pairs, so
    # that all eight heterogeneity treatments are measured on one metric against
    # one kind of control. identification_hac_tests above tests the *mean*
    # net-return series for the same pairs; this tests cross-client spread.
    identification_dir = output_root / "identification"
    identification_dispersion_frames = [
        heterogeneity_dispersion_hac_contrasts(
            [(identification_dir / treatment, identification_dir / control, label)]
        )
        for treatment, control, label in IDENTIFICATION_PAIRS
    ]
    tables["identification_dispersion_hac_tests"] = pd.concat(
        identification_dispersion_frames, ignore_index=True
    )

    # Primary inference for the paper's matched-control claims. Unlike the
    # Stouffer-combined tables above (retained for comparison), this treats the
    # calendar month as the unit of inference: the treatment-minus-control series
    # is averaged across seeds first, then tested with HAC + a moving-block
    # bootstrap, with Benjamini-Hochberg FDR control across the whole family.
    matched_control_pairs = [
        (heterogeneity_dir / treatment, heterogeneity_dir / control, label)
        for treatment, control, label in HETEROGENEITY_MATCHED_CONTROL_PAIRS
    ] + [
        (identification_dir / treatment, identification_dir / control, label)
        for treatment, control, label in IDENTIFICATION_PAIRS
    ]
    tables["matched_control_dispersion_monthly"] = seed_averaged_dispersion_contrasts(
        matched_control_pairs
    )
    tables["calibration_diagnostics"] = calibration_diagnostics(output_root)
    tables["reference_cell_mde"] = reference_cell_mde(output_root)
    tables["rolling_cell_mde"] = rolling_cell_mde(output_root)
    # The same MDE estimator on the unrestricted linear/ridge refits, so the new
    # cell is calibrated before it is read -- the discipline the whole paper
    # argues for would be hollow if its newest cell were exempt from it.
    tables["rolling_cell_mde_linear"] = rolling_cell_mde(
        output_root, phase="rolling_refit_linear"
    )
    tables["rolling_cell_mde_tuned20"] = rolling_cell_mde(
        output_root, phase=ROLLING_TUNED20_PHASE
    )
    tables["rolling_cell_mde_linear_tuned20"] = rolling_cell_mde(
        output_root, phase=ROLLING_LINEAR_TUNED20_PHASE
    )
    tables["rolling_seed_variance"] = rolling_seed_variance(output_root)
    # The same two-stage bootstrap on the widened cell. A reviewer noticed the
    # asymmetry: the reference cell reports both a seed-averaged and a single-run
    # detectable gap, the widened cell -- the one carrying the paper's only
    # detectable result -- reported only the first. It is the cell where the
    # answer matters most, and it does not flatter the headline: the single-run
    # figure lands ABOVE both observed shortfalls.
    tables["rolling_seed_variance_tuned20"] = rolling_seed_variance(
        output_root, phase=ROLLING_TUNED20_PHASE
    )
    tables["rolling_seed_variance_linear_tuned20"] = rolling_seed_variance(
        output_root, phase=ROLLING_LINEAR_TUNED20_PHASE
    )
    tables["economic_power_prescription"] = economic_power_prescription(output_root)
    tables["client_local_level_comparison"] = client_local_level_comparison(output_root)

    for name, table in tables.items():
        _write_table(table, tables_dir / f"{name}.csv")

    # Last, and after the write loop on purpose: the census reads the descriptive
    # tables back off disk, so every one of them has to exist first.
    tables["descriptive_test_census"] = descriptive_test_census(tables_dir)
    _write_table(tables["descriptive_test_census"], tables_dir / "descriptive_test_census.csv")

    _plot_grouped_bars(
        tables["protocol_sensitivity"],
        x="scenario",
        y="rank_ic",
        group="method",
        output=figures_dir / "protocol_rank_ic.png",
        title="Protocol and investability sensitivity",
        ylabel="Rank IC",
    )
    partition_metric = (
        "global_portfolio_net_sharpe"
        if "global_portfolio_net_sharpe" in tables["partition_comparison"]
        else "rank_ic"
    )
    _plot_grouped_bars(
        tables["partition_comparison"],
        x="scenario",
        y=partition_metric,
        group="method",
        output=figures_dir / "partition_comparison.png",
        title="Common-universe client partitions",
        ylabel=partition_metric.replace("_", " ").title(),
    )
    _plot_grouped_bars(
        tables["heterogeneity"],
        x="scenario",
        y="rank_ic",
        group="method",
        output=figures_dir / "heterogeneity_rank_ic.png",
        title="Controlled covariate and feature heterogeneity",
        ylabel="Rank IC",
    )
    _plot_identification(
        tables["identification_contrasts"],
        figures_dir / "identification_net_sharpe_delta.png",
    )
    _plot_grouped_bars(
        tables["expanded_features_comparison"],
        x="feature_set",
        y="rank_ic_mean",
        group="method",
        output=figures_dir / "expanded_features_rank_ic.png",
        title="Original (7) vs. expanded (20) characteristic panel",
        ylabel="Rank IC",
    )

    return {
        "combined_rows": int(len(combined)),
        "phases_found": sorted(combined["suite"].unique().tolist()),
        "tables": {name: str((tables_dir / f"{name}.csv").resolve()) for name in tables},
        "figures_dir": str(figures_dir.resolve()),
    }


def write_publication_report(
    output_root: Path,
    *,
    selection: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    build: dict[str, Any] | None,
    phase_status: list[dict[str, Any]],
) -> Path:
    lines = [
        "# ICAIF Publication Experiment Report",
        "",
        "This directory is the publication-facing source of truth. Grids use "
        "seeds [11, 22, ...] (5 seeds for fedadam_validation/partitions/"
        "label_diagnostics, 10 for core, 20 for identification and "
        "heterogeneity) "
        "and FedAdam server learning rate 0.003 after validation-only selection.",
        "",
        "## Phase status",
        "",
        "| Phase | Status | Output |",
        "|---|---|---|",
    ]
    for row in phase_status:
        lines.append(f"| {row['phase']} | {row['status']} | `{row['output_dir']}` |")
    if selection is not None:
        lines.extend([
            "",
            "## FedAdam selection",
            "",
            f"- Chosen scenario: `{selection['chosen_scenario']}`",
            f"- Selected server learning rate: `{selection['selected_server_lr']}`",
            f"- Validation Rank IC: `{selection['validation_mean']:.6f}`",
            f"- Validation OOS R2 constraint mean: `{selection['validation_constraint_mean']:.6f}`",
        ])
    if audit is not None:
        lines.extend([
            "",
            "## Identification audits",
            "",
            f"- Overall passed: `{audit.get('passed')}`",
            f"- Common-universe partition audit: `{audit.get('partition_common_universe', {}).get('passed')}`",
            f"- Matched-control audit: `{audit.get('matched_controls', {}).get('passed')}`",
        ])
    if build is not None:
        lines.extend([
            "",
            "## Paper-facing outputs",
            "",
            f"- Combined rows: `{build['combined_rows']}`",
            "- Tables: `paper_tables/`",
            "- Figures: `paper_figures/`",
            "- Full merged results: `all_grid_results.csv`",
            "- Full merged summary: `all_grid_summary.csv`",
            "- Reference-protocol HAC significance vs. Centralized: "
            "`paper_tables/main_results_significance.csv`",
            "- Reference-protocol bootstrap Sharpe intervals: "
            "`paper_tables/main_results_sharpe_intervals.csv`",
            "- Matched-control HAC significance: "
            "`paper_tables/identification_hac_tests.csv`",
        ])
    path = output_root / "PUBLICATION_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

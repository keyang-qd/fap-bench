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

    for name, table in tables.items():
        _write_table(table, tables_dir / f"{name}.csv")

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

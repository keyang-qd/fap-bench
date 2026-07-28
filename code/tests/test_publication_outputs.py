from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fapbench.publication import build_publication_outputs, select_fedadam_validation


def _write_phase(root: Path, phase: str, rows: list[dict]) -> None:
    path = root / phase
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "grid_results.csv", index=False)


def test_validation_selection_uses_constraint_and_resolved_lr(tmp_path: Path) -> None:
    root = tmp_path / "fedadam_validation"
    rows = []
    for scenario, rank_ic, r2, lr in [
        ("fedadam_validation_lr_0_003", 0.04, -0.03, 0.003),
        ("fedadam_validation_lr_0_01", 0.06, -0.20, 0.01),
    ]:
        for seed in [11, 22]:
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "method": "fedadam",
                "validation_selected_rank_ic": rank_ic,
                "validation_selected_oos_r2_zero": r2,
            })
            run = root / scenario / f"seed_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "resolved_config.json").write_text(
                json.dumps({"federated": {"server_lr": lr}}), encoding="utf-8"
            )
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "grid_results.csv", index=False)
    report = select_fedadam_validation(
        root / "grid_results.csv",
        scenario_prefix="fedadam_validation_lr_",
        method="fedadam",
        metric="validation_selected_rank_ic",
        mode="max",
        constraint_metric="validation_selected_oos_r2_zero",
        constraint_min=-0.05,
    )
    assert report["selected_server_lr"] == 0.003


_COMMON_METRICS = {
    "rank_ic": 0.03,
    "oos_r2_zero": 0.001,
    "client_worst_rank_ic": -0.01,
    "client_neutral_portfolio_net_sharpe": 0.2,
    "global_portfolio_net_sharpe": 0.25,
    "selected_checkpoint_communicated_bytes_fp32": 1_000_000.0,
    "heterogeneity_n_clients": 12.0,
}


def _write_core_publication_phases(tmp_path: Path) -> None:
    """Every phase except core_expanded_features, which is genuinely optional."""
    common = _COMMON_METRICS
    core_rows = []
    for scenario in [
        "ff12_full_participation", "ff12_half_participation",
        "ff12_more_local_steps", "ff12_price_above_5", "ff12_exclude_microcaps",
        # The reference protocol's second budget. Table 3's 10x5 block comes from
        # here, so leaving it out makes main_results_10x5 silently empty.
        "ff12_exclude_microcaps_more_local_steps",
    ]:
        for seed in [11, 22]:
            for method in ["centralized", "local", "fedavg", "fedprox", "scaffold", "fedadam"]:
                core_rows.append({"scenario": scenario, "seed": seed, "method": method, **common})
            core_rows.append({
                "scenario": scenario, "seed": seed, "method": "fedadam_calibrated",
                **{**common, "oos_r2_zero": 0.002},
            })
    _write_phase(tmp_path, "core", core_rows)

    partition_rows = [
        {"scenario": s, "seed": seed, "method": method, **common}
        for s in ["common_partition_ff12", "common_partition_iid_12"]
        for seed in [11, 22]
        for method in ["centralized", "fedavg", "scaffold", "fedadam"]
    ]
    _write_phase(tmp_path, "partitions", partition_rows)

    hetero_rows = [
        {"scenario": s, "seed": seed, "method": method, **common,
         "heterogeneity_feature_distribution_wasserstein": 0.1,
         "heterogeneity_feature_jaccard_dissimilarity": 0.2}
        for s in [
            "covariate_shift_mild", "feature_heterogeneity_severe",
            # Matched-control scenario for HETEROGENEITY_MATCHED_CONTROL_PAIRS;
            # identification_hac_contrasts/heterogeneity_dispersion_hac_contrasts
            # read raw seed-level run directories directly rather than this
            # grid_results.csv row, so this only exercises the point-estimate
            # side of the pipeline (see test_significance.py for HAC coverage).
            "covariate_shift_mild_matched",
        ]
        for seed in [11, 22]
        for method in ["centralized", "fedavg", "scaffold", "fedadam"]
    ]
    _write_phase(tmp_path, "heterogeneity", hetero_rows)

    identification_rows = [
        {"scenario": s, "seed": seed, "method": method, **common}
        for s in [
            "size_uniform_matched_ratio_20", "size_imbalance_ratio_20_identified",
            "temporal_persistent_strong_identified", "temporal_random_matched_persistent_strong",
            "temporal_block_outages_identified", "temporal_random_matched_block_outages",
        ]
        for seed in [11, 22]
        for method in ["fedavg", "scaffold", "fedadam"]
    ]
    _write_phase(tmp_path, "identification", identification_rows)

    label_rows = [
        {"scenario": s, "seed": seed, "method": "diagnostics_only",
         "heterogeneity_label_distribution_wasserstein": 0.05,
         "heterogeneity_controlled_label_skew_retained_fraction": 0.6,
         "heterogeneity_controlled_label_skew_concentration": 2.0,
         "heterogeneity_n_observations": 1000.0}
        for s in ["label_skew_mild_diagnostics", "label_skew_strong_diagnostics"]
        for seed in [11, 22]
    ]
    _write_phase(tmp_path, "label_diagnostics", label_rows)

    _write_phase(tmp_path, "fedadam_validation", [
        {"scenario": "fedadam_validation_lr_0_003", "seed": 11, "method": "fedadam", **common}
    ])


def test_publication_builder_surfaces_tables_and_figures(tmp_path: Path) -> None:
    _write_core_publication_phases(tmp_path)

    expanded_rows = [
        {"scenario": "ff12_full_participation", "seed": seed, "method": method,
         **{**_COMMON_METRICS, "rank_ic": 0.05}}
        for seed in [11, 22]
        for method in ["centralized", "local", "fedavg", "fedprox", "scaffold", "fedadam"]
    ]
    _write_phase(tmp_path, "core_expanded_features", expanded_rows)

    report = build_publication_outputs(tmp_path)
    assert report["combined_rows"] > 0
    # Both blocks of Table 3 must actually carry rows. Asserting only that the
    # file exists would pass on an empty frame, which is precisely what happens
    # if REFERENCE_SCENARIO is re-pointed at a scenario the grid never ran.
    for name in ["main_results", "main_results_10x5", "main_results_unrestricted"]:
        path = tmp_path / "paper_tables" / f"{name}.csv"
        assert path.exists(), name
        assert not pd.read_csv(path).empty, name
    assert (tmp_path / "paper_tables" / "identification_contrasts.csv").exists()
    assert (tmp_path / "paper_figures" / "protocol_rank_ic.png").exists()
    assert (tmp_path / "paper_tables" / "heterogeneity_matched_control_hac_tests.csv").exists()
    assert (
        tmp_path / "paper_tables" / "heterogeneity_matched_control_dispersion_hac_tests.csv"
    ).exists()
    # Same dispersion test over IDENTIFICATION_PAIRS, so all eight heterogeneity
    # treatments share one metric and one control type (drives Figure 2c).
    assert (
        tmp_path / "paper_tables" / "identification_dispersion_hac_tests.csv"
    ).exists()

    expanded_table = pd.read_csv(tmp_path / "paper_tables" / "expanded_features_comparison.csv")
    assert set(expanded_table["feature_set"]) == {"7_characteristics", "20_characteristics"}
    assert (tmp_path / "paper_figures" / "expanded_features_rank_ic.png").exists()


def test_publication_builder_tolerates_missing_expanded_features_phase(
    tmp_path: Path,
) -> None:
    """A run without core_expanded_features (e.g. --only core identification) still builds."""
    _write_core_publication_phases(tmp_path)

    report = build_publication_outputs(tmp_path)
    assert report["combined_rows"] > 0
    assert (tmp_path / "paper_tables" / "main_results.csv").exists()
    expanded_path = tmp_path / "paper_tables" / "expanded_features_comparison.csv"
    assert expanded_path.exists()
    assert expanded_path.read_text(encoding="utf-8").strip() == ""
    assert not (tmp_path / "paper_figures" / "expanded_features_rank_ic.png").exists()

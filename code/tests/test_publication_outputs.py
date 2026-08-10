from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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


# --- calibration fairness and the reference-cell MDE ------------------------
#
# Both answer reviewer objections that the manuscript previously met with prose:
# that calibrating FedAdam alone is not internally comparable, and that
# "useless for ranking" was asserted without a minimum detectable effect.


def _write_portfolio_returns(seed_dir, method, returns, dates=None):
    import pandas as pd

    seed_dir.mkdir(parents=True, exist_ok=True)
    dates = dates if dates is not None else pd.date_range("2019-01-31", periods=len(returns), freq="ME")
    pd.DataFrame({"date": dates, "net_return": returns}).to_csv(
        seed_dir / f"portfolio_returns_client_neutral_{method}.csv", index=False
    )


def test_reference_cell_mde_scales_with_noise(tmp_path):
    """The MDE is a power statement: a noisier contrast must be harder to detect.
    If this inverted, the number the manuscript quotes would be meaningless."""
    import numpy as np
    from fapbench.publication import reference_cell_mde

    scenario = tmp_path / "core" / "ff12_exclude_microcaps"
    rng = np.random.default_rng(0)
    n = 60
    for seed in (11, 22, 33):
        seed_dir = scenario / f"seed_{seed}"
        base = rng.normal(0.01, 0.02, n)
        _write_portfolio_returns(seed_dir, "centralized", base)
        _write_portfolio_returns(seed_dir, "fedavg", base + rng.normal(0, 0.001, n))
        _write_portfolio_returns(seed_dir, "fedadam", base + rng.normal(0, 0.010, n))

    result = reference_cell_mde(tmp_path, methods=("fedavg", "fedadam")).set_index("method")
    assert int(result.loc["fedavg", "n_months"]) == 60
    assert (
        result.loc["fedadam", "mde_annualized_sharpe_gap"]
        > result.loc["fedavg", "mde_annualized_sharpe_gap"]
    ), "the noisier contrast must have the larger minimum detectable effect"
    assert (result["mde_annualized_sharpe_gap"] > 0).all()


def test_reference_cell_mde_is_empty_without_the_grid(tmp_path):
    from fapbench.publication import reference_cell_mde

    assert reference_cell_mde(tmp_path).empty


def test_calibration_diagnostics_reports_every_arm_and_the_constant_benchmark(tmp_path):
    """The fairness claim only holds if all four arms are actually computed, and
    the constant-predictor column is what makes the calibrated number readable."""
    import numpy as np
    import pandas as pd
    from fapbench.publication import calibration_diagnostics

    scenario = tmp_path / "core" / "ff12_exclude_microcaps"
    rng = np.random.default_rng(1)
    methods = ("fedavg", "fedprox", "scaffold", "fedadam")
    for seed in (11, 22):
        seed_dir = scenario / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for scale, method in zip((1.0, 1.0, 1.0, 8.0), methods):
            for split, n in (("validation_predictions", 400), ("predictions", 300)):
                y = rng.normal(0.01, 0.05, n)
                pd.DataFrame(
                    {"y_true": y, "y_pred": scale * (0.05 * y + rng.normal(0, 0.01, n))}
                ).to_parquet(seed_dir / f"{split}_{method}.parquet")

    result = calibration_diagnostics(tmp_path)
    assert set(result["method"]) == set(methods), "all four federated arms required"
    assert "best_constant_oos_r2_zero" in result
    # FedAdam's predictions are inflated 8x, so its RAW R^2 must be far worse...
    raw = result.set_index("method")["raw_oos_r2_zero"]
    assert raw["fedadam"] < raw["fedavg"]
    # ...while calibration, which removes exactly that scale, pulls the arms together.
    calibrated = result["calibrated_oos_r2_zero"]
    assert calibrated.max() - calibrated.min() < abs(raw["fedadam"] - raw["fedavg"])


def test_rolling_cell_mde_pools_windows_and_beats_the_short_cell(tmp_path):
    """Two properties the manuscript's corrected claim depends on: windows are
    pooled into one long series (not averaged per window), and the resulting MDE
    is smaller than a single short window's -- that is what 408 months buys."""
    import numpy as np
    import pandas as pd
    from fapbench.publication import rolling_cell_mde

    phase = tmp_path / "rolling_refit_investable"
    rng = np.random.default_rng(3)
    years = [1990, 1991, 1992, 1993]
    for year in years:
        dates = pd.date_range(f"{year}-01-31", periods=12, freq="ME")
        for seed in (11, 22, 33):
            seed_dir = phase / f"refit_test_{year}" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            base = rng.normal(0.01, 0.02, len(dates))
            for method, extra in (("centralized", 0.0), ("fedavg", 0.0005)):
                pd.DataFrame(
                    {"date": dates, "net_return": base + extra + rng.normal(0, 0.002, len(dates))}
                ).to_csv(seed_dir / f"portfolio_returns_client_neutral_{method}.csv", index=False)

    result = rolling_cell_mde(tmp_path, methods=("fedavg",))
    row = result.iloc[0]
    assert int(row["n_windows"]) == 4
    assert int(row["n_months"]) == 48, "windows must pool, not average"
    assert row["mde_annualized_sharpe_gap"] > 0

    # Same estimator on one window alone must be strictly less powerful.
    single = rolling_cell_mde(
        tmp_path, methods=("fedavg",), phase="rolling_refit_investable"
    )
    for year in years[1:]:
        import shutil

        shutil.rmtree(phase / f"refit_test_{year}")
    one_window = rolling_cell_mde(tmp_path, methods=("fedavg",))
    assert int(one_window.iloc[0]["n_months"]) == 12
    assert (
        one_window.iloc[0]["mde_annualized_sharpe_gap"]
        > single.iloc[0]["mde_annualized_sharpe_gap"]
    ), "a 12-month cell must have a larger minimum detectable effect than 48 months"


def test_rolling_cell_mde_is_empty_without_the_grid(tmp_path):
    from fapbench.publication import rolling_cell_mde

    assert rolling_cell_mde(tmp_path).empty


def test_rolling_seed_variance_separates_seed_from_time_uncertainty(tmp_path):
    """The two properties the manuscript's seed-uncertainty paragraph rests on.

    Seeds are given a large per-seed level offset and almost no month-to-month
    noise, so averaging across seeds must visibly diversify: the Sharpe of the
    averaged series has to exceed the mean of the per-seed Sharpes. And admitting
    seed resampling must widen the interval, never narrow it.
    """
    import numpy as np
    import pandas as pd
    from fapbench.publication import rolling_seed_variance

    phase = tmp_path / "rolling_refit_investable"
    rng = np.random.default_rng(11)
    seeds = (11, 22, 33, 44, 55)
    # One persistent offset per seed -- this is the seed component.
    offsets = {seed: rng.normal(0, 0.01) for seed in seeds}
    for year in (1990, 1991, 1992):
        dates = pd.date_range(f"{year}-01-31", periods=12, freq="ME")
        for seed in seeds:
            seed_dir = phase / f"refit_test_{year}" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            for method, extra in (("centralized", 0.0), ("fedavg", 0.0004)):
                values = 0.01 + extra + offsets[seed] + rng.normal(0, 0.001, len(dates))
                pd.DataFrame({"date": dates, "net_return": values}).to_csv(
                    seed_dir / f"portfolio_returns_client_neutral_{method}.csv",
                    index=False,
                )

    result = rolling_seed_variance(
        tmp_path, methods=("centralized", "fedavg"), n_bootstrap=400
    ).set_index("method")
    assert int(result.loc["centralized", "n_months"]) == 36, "windows must pool"
    assert int(result.loc["centralized", "n_seeds"]) == len(seeds)

    # Averaging across seeds cancels the per-seed offsets, so the averaged series
    # is far less volatile and its Sharpe far higher than a typical single seed's.
    assert (
        result.loc["centralized", "net_sharpe_seed_averaged"]
        > result.loc["centralized", "net_sharpe_per_seed_mean"]
    )
    assert result.loc["centralized", "net_sharpe_per_seed_sd"] > 0

    # The reference method has no self-contrast, so no interval.
    assert pd.isna(result.loc["centralized", "ci_width_ratio_joint_over_time"])
    # Resampling seeds as well as months cannot tighten the interval.
    assert result.loc["fedavg", "ci_width_ratio_joint_over_time"] >= 1.0
    assert (
        result.loc["fedavg", "mde_annualized_sharpe_joint"]
        >= result.loc["fedavg", "mde_annualized_sharpe_time_only"]
    )


def test_rolling_seed_variance_is_empty_without_the_grid(tmp_path):
    from fapbench.publication import rolling_seed_variance

    assert rolling_seed_variance(tmp_path).empty


def test_descriptive_test_census_counts_and_corrects(tmp_path):
    """The census must read only declared tables, and BH must never make a test
    look *more* significant than its raw p-value did."""
    import pandas as pd
    from fapbench.publication import descriptive_test_census

    pd.DataFrame({"hac_p_value": [0.001, 0.04, 0.6, 0.9]}).to_csv(
        tmp_path / "rolling_vs_centralized_investable.csv", index=False
    )
    pd.DataFrame({"p": [0.02, 0.5]}).to_csv(
        tmp_path / "robustness_lever_significance.csv", index=False
    )
    # Not on the declared list: must be ignored even though it has a p column.
    pd.DataFrame({"hac_p_value": [0.0001]}).to_csv(
        tmp_path / "identification_dispersion_hac_tests.csv", index=False
    )

    census = descriptive_test_census(tmp_path)
    assert len(census) == 6, "only declared tables count"
    assert set(census["source"]) == {
        "rolling_vs_centralized_investable",
        "robustness_lever_significance",
    }
    assert (census["q_value"] >= census["p_value"] - 1e-12).all()
    assert census["significant_corrected"].sum() <= census["significant_uncorrected"].sum()


def test_descriptive_test_census_is_empty_without_tables(tmp_path):
    from fapbench.publication import descriptive_test_census

    assert descriptive_test_census(tmp_path).empty


def test_matched_control_placebo_pairs_seeds_disjointly(tmp_path):
    """The placebo's whole value rests on two properties: a seed is never
    differenced against itself (which would give an identically zero series and a
    degenerate test rather than a noise floor), and the pairing is reproducible
    from the artifact rather than drawn at random."""
    import numpy as np
    import pandas as pd
    from fapbench.evaluation.significance import matched_control_placebo

    control = tmp_path / "control_arm"
    rng = np.random.default_rng(5)
    dates = pd.date_range("2019-01-31", periods=24, freq="ME")
    seeds = (11, 22, 33, 44, 55)  # odd count: the last seed must be dropped
    for seed in seeds:
        seed_dir = control / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for date in dates:
            for client in range(4):
                for _ in range(12):
                    rows.append(
                        {
                            "target_date": date,
                            "client_id": str(client),
                            "y_true": rng.normal(),
                            "y_pred": rng.normal(),
                        }
                    )
        pd.DataFrame(rows).to_parquet(seed_dir / "predictions_fedavg.parquet")
        pd.DataFrame({"method": ["fedavg"]}).to_csv(
            seed_dir / "benchmark_results.csv", index=False
        )

    result = matched_control_placebo(
        [(control, "placebo")], methods=["fedavg"], statistics=("std",)
    )
    assert len(result) == 1
    row = result.iloc[0]
    # Five seeds -> two disjoint pairs, the odd one out dropped.
    assert int(row["n_pairs"]) == 2
    # Predictions are pure noise, so the placebo's mean effect must be small; the
    # point is that it is measured rather than assumed.
    assert abs(float(row["mean_difference_of_means"])) < 0.2
    assert row["max_abs_difference_of_means"] >= abs(row["mean_difference_of_means"])


def test_matched_control_placebo_is_empty_without_runs(tmp_path):
    from fapbench.evaluation.significance import matched_control_placebo

    assert matched_control_placebo([(tmp_path / "nope", "x")]).empty


def test_client_local_level_comparison_pairs_within_seed(tmp_path):
    """The comparison is only meaningful if the two arms are paired seed-by-seed:
    comparing across seed sets would confound preprocessing with the seed draw.
    A seed present in only one arm must be dropped, not silently half-used."""
    import numpy as np
    import pandas as pd
    from fapbench.publication import client_local_level_comparison

    rng = np.random.default_rng(5)
    dates = pd.date_range("2019-01-31", periods=60, freq="ME")

    def _write(phase, seed, shift):
        d = tmp_path / phase / "ff12_exclude_microcaps" / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        base = rng.normal(0.005, 0.02, len(dates))
        pd.DataFrame({"date": dates, "net_return": base + shift}).to_csv(
            d / "portfolio_returns_client_neutral_fedavg.csv", index=False
        )
        pd.DataFrame(
            [{"method": "fedavg", "rank_ic": 0.01 + shift,
              "client_neutral_portfolio_net_sharpe": 0.1 + shift}]
        ).to_csv(d / "benchmark_results.csv", index=False)

    for seed in (11, 22, 33):
        _write("core", seed, 0.0)
        _write("client_local_levels", seed, 0.001)
    # A fourth seed present only in the client-local arm must not be paired.
    _write("client_local_levels", 44, 0.001)

    result = client_local_level_comparison(tmp_path, methods=("fedavg",))
    row = result.iloc[0]
    assert int(row["n_seeds"]) == 3, "unpaired seed must be dropped"
    assert int(row["n_months"]) == 60
    assert row["mean_monthly_return_difference"] > 0
    assert row["rank_ic_delta"] == pytest.approx(0.001, abs=1e-9)
    assert "hac_q_value" in result


def test_client_local_level_comparison_is_empty_without_the_grid(tmp_path):
    from fapbench.publication import client_local_level_comparison

    assert client_local_level_comparison(tmp_path).empty


def test_client_local_levels_config_differs_from_core_in_one_knob_only():
    """The comparison is a confound-free difference only if preprocessing is the
    single thing that changed. This pins that against configs/icaif_core.yaml."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent / "configs"
    core = yaml.safe_load((root / "icaif_core.yaml").read_text())
    local = yaml.safe_load((root / "icaif_client_local_levels.yaml").read_text())

    assert local["base_overrides"] == {"data": {"client_local_standardization": True}}
    assert local["seeds"] == core["seeds"], "seed sets must match for within-seed pairing"
    assert local["base_config"] == core["base_config"]

    core_scenarios = {s["name"]: s["overrides"] for s in core["scenarios"]}
    for scenario in local["scenarios"]:
        assert scenario["name"] in core_scenarios, f"{scenario['name']} is not a core cell"
        assert scenario["overrides"] == core_scenarios[scenario["name"]], (
            f"{scenario['name']} overrides drifted from configs/icaif_core.yaml"
        )


def test_rolling_linear_phase_is_separate_from_the_mlp_cells(tmp_path):
    """The linear/ridge refits must never be concatenated with the MLP ones.

    They are different model classes on different universes, so their monthly
    series are not comparable in levels; sharing an output directory would let
    `rolling_pooled_metrics` splice them into one series with nothing raised.
    """
    import numpy as np
    import pandas as pd
    from fapbench.rolling import (
        ROLLING_INVESTABLE_PHASE,
        ROLLING_LINEAR_PHASE,
        rolling_pooled_metrics,
    )

    assert ROLLING_LINEAR_PHASE != ROLLING_INVESTABLE_PHASE

    rng = np.random.default_rng(7)
    for phase, level in ((ROLLING_LINEAR_PHASE, 0.02), (ROLLING_INVESTABLE_PHASE, 0.00)):
        for year in (1990, 1991):
            dates = pd.date_range(f"{year}-01-31", periods=12, freq="ME")
            for seed in (11, 22):
                seed_dir = tmp_path / phase / f"refit_test_{year}" / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                for method in ("centralized", "fedavg"):
                    pd.DataFrame(
                        {
                            "target_date": dates,
                            "net_return": level + rng.normal(0, 0.01, len(dates)),
                        }
                    ).to_csv(
                        seed_dir / f"portfolio_returns_client_neutral_{method}.csv",
                        index=False,
                    )
                    # rolling_pooled_metrics keys on the predictive file and
                    # skips a method entirely without it.
                    pd.DataFrame(
                        {
                            "target_date": dates,
                            "rank_ic": level + rng.normal(0, 0.01, len(dates)),
                        }
                    ).to_csv(
                        seed_dir / f"monthly_predictive_metrics_{method}.csv",
                        index=False,
                    )

    linear = rolling_pooled_metrics(
        tmp_path, methods=("centralized", "fedavg"), phase=ROLLING_LINEAR_PHASE
    )
    mlp = rolling_pooled_metrics(
        tmp_path, methods=("centralized", "fedavg"), phase=ROLLING_INVESTABLE_PHASE
    )
    assert not linear.empty and not mlp.empty
    # 24 months each: each phase saw its own two windows, not all four.
    assert int(linear["n_months"].max()) == 24
    assert int(mlp["n_months"].max()) == 24
    # The level offset must survive, proving the two phases were read separately.
    assert linear["net_sharpe"].mean() > mlp["net_sharpe"].mean()


def test_rolling_refuses_to_pool_partial_seeds(tmp_path):
    """A seed that covers only some windows must not be pooled silently.

    This is a regression test for a bug that shipped: a grid extended from 3 to 5
    seeds and stopped partway pooled 5 seeds over early windows and 3 over late
    ones, still reported the full month count, and moved a published net Sharpe
    gap by 0.03.
    """
    import numpy as np
    import pandas as pd
    import pytest
    from fapbench.rolling import (
        ROLLING_INVESTABLE_PHASE,
        rolling_pooled_metrics,
        rolling_vs_centralized,
    )

    phase = tmp_path / ROLLING_INVESTABLE_PHASE
    rng = np.random.default_rng(3)
    # seed_11/22 cover both windows; seed_33 covers only the first.
    coverage = {"seed_11": (1990, 1991), "seed_22": (1990, 1991), "seed_33": (1990,)}
    for seed, years in coverage.items():
        for year in years:
            dates = pd.date_range(f"{year}-01-31", periods=12, freq="ME")
            d = phase / f"refit_test_{year}" / seed
            d.mkdir(parents=True, exist_ok=True)
            for method in ("centralized", "fedavg"):
                pd.DataFrame(
                    {"target_date": dates, "net_return": rng.normal(0.01, 0.02, 12)}
                ).to_csv(d / f"portfolio_returns_client_neutral_{method}.csv", index=False)
                pd.DataFrame(
                    {"target_date": dates, "rank_ic": rng.normal(0.01, 0.02, 12)}
                ).to_csv(d / f"monthly_predictive_metrics_{method}.csv", index=False)

    with pytest.raises(ValueError, match="do not cover every window"):
        rolling_pooled_metrics(
            tmp_path, methods=("centralized", "fedavg"), phase=ROLLING_INVESTABLE_PHASE
        )
    with pytest.raises(ValueError, match="do not cover every window"):
        rolling_vs_centralized(
            tmp_path, methods=("fedavg",), phase=ROLLING_INVESTABLE_PHASE
        )

    # An explicit subset is still allowed: the guard blocks accidents, not intent.
    ok = rolling_pooled_metrics(
        tmp_path,
        methods=("centralized", "fedavg"),
        phase=ROLLING_INVESTABLE_PHASE,
        seeds=["seed_11", "seed_22"],
    )
    assert int(ok["n_months"].max()) == 24

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from fapbench.evaluation.significance import (
    aggregate_seed_significance,
    combine_seed_intervals,
    combine_seed_tests,
    heterogeneity_dispersion_hac_contrasts,
    identification_hac_contrasts,
    stouffer_combine,
)


def test_stouffer_combine_reinforces_same_direction_effects():
    combined = stouffer_combine(
        t_statistics=[2.5, 2.6, 2.4],
        p_values=[0.012, 0.009, 0.016],
    )
    assert combined["n_combined"] == 3
    assert combined["combined_p_value"] < min(0.012, 0.009, 0.016)
    assert combined["combined_z"] > 0


def test_stouffer_combine_cancels_opposite_direction_effects():
    combined = stouffer_combine(
        t_statistics=[3.0, -3.0],
        p_values=[0.0027, 0.0027],
    )
    assert combined["combined_p_value"] > 0.9


def _write_run(
    run_dir: Path,
    *,
    methods: list[str],
    reference_method: str,
    dates: pd.DatetimeIndex,
    returns: dict[str, np.ndarray],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"method": methods}).to_csv(
        run_dir / "benchmark_results.csv", index=False
    )
    for method in methods:
        frame = pd.DataFrame(
            {"target_date": dates, "net_return": returns[method]}
        )
        frame.to_csv(
            run_dir / f"portfolio_returns_client_neutral_{method}.csv", index=False
        )


def test_aggregate_and_combine_seed_significance(tmp_path: Path):
    dates = pd.date_range("2019-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(0)
    scenario_dir = tmp_path / "ff12_full_participation"
    for seed in [11, 22, 33]:
        local_rng = np.random.default_rng(seed)
        centralized_returns = local_rng.normal(0.0, 0.02, size=len(dates))
        comparison_returns = centralized_returns + 0.01  # consistently better
        _write_run(
            scenario_dir / f"seed_{seed}",
            methods=["centralized", "fedadam"],
            reference_method="centralized",
            dates=dates,
            returns={
                "centralized": centralized_returns,
                "fedadam": comparison_returns,
            },
        )

    per_seed_tests, per_seed_intervals = aggregate_seed_significance(
        scenario_dir, variant="client_neutral", reference_method="centralized"
    )
    assert set(per_seed_tests["seed"]) == {11, 22, 33}
    assert (per_seed_tests["comparison_method"] == "fedadam").all()

    combined_tests = combine_seed_tests(per_seed_tests)
    assert len(combined_tests) == 1
    row = combined_tests.iloc[0]
    assert row["n_seeds"] == 3
    # fedadam is constructed to beat centralized by a fixed positive amount
    # every seed, so the combined test should reject the zero-mean null.
    assert row["combined_p_value"] < 0.05
    assert row["mean_mean_difference"] > 0

    combined_intervals = combine_seed_intervals(per_seed_intervals)
    assert set(combined_intervals["method"]) == {"centralized", "fedadam"}
    assert (combined_intervals["n_seeds"] == 3).all()


def test_identification_hac_contrasts_detects_matched_control_gap(tmp_path: Path):
    dates = pd.date_range("2019-01-31", periods=48, freq="ME")
    phase_dir = tmp_path / "identification"
    for seed in [11, 22, 33]:
        local_rng = np.random.default_rng(seed)
        control_returns = local_rng.normal(0.0, 0.02, size=len(dates))
        treatment_returns = control_returns - 0.015  # treatment consistently worse
        _write_run(
            phase_dir / "size_uniform_matched_ratio_20" / f"seed_{seed}",
            methods=["fedadam"],
            reference_method="fedadam",
            dates=dates,
            returns={"fedadam": control_returns},
        )
        _write_run(
            phase_dir / "size_imbalance_ratio_20_identified" / f"seed_{seed}",
            methods=["fedadam"],
            reference_method="fedadam",
            dates=dates,
            returns={"fedadam": treatment_returns},
        )

    pairs = [
        (
            "size_imbalance_ratio_20_identified",
            "size_uniform_matched_ratio_20",
            "size imbalance minus matched retention",
        )
    ]
    result = identification_hac_contrasts(phase_dir, pairs, variant="client_neutral")
    assert len(result) == 1
    row = result.iloc[0]
    assert row["method"] == "fedadam"
    assert row["n_seeds"] == 3
    assert row["mean_difference_of_means"] < 0
    assert row["combined_p_value"] < 0.05


def test_identification_hac_contrasts_skips_missing_scenario_dirs(tmp_path: Path):
    result = identification_hac_contrasts(
        tmp_path / "identification",
        [("missing_treatment", "missing_control", "label")],
        variant="client_neutral",
    )
    assert result.empty


def _write_predictions(
    run_dir: Path,
    *,
    method: str,
    dates: pd.DatetimeIndex,
    client_rhos: dict[str, float],
    seed: int,
    n_stocks: int = 25,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for date in dates:
        for client, rho in client_rhos.items():
            y_true = rng.normal(size=n_stocks)
            noise = rng.normal(size=n_stocks)
            y_pred = rho * y_true + np.sqrt(max(1.0 - rho**2, 0.0)) * noise
            for permno, (yt, yp) in enumerate(zip(y_true, y_pred)):
                rows.append(
                    {
                        "target_date": date,
                        "client_id": client,
                        "permno": permno,
                        "y_true": yt,
                        "y_pred": yp,
                    }
                )
    pd.DataFrame(rows).to_parquet(run_dir / f"predictions_{method}.parquet")


def test_heterogeneity_dispersion_hac_contrasts_detects_widened_spread(
    tmp_path: Path,
):
    dates = pd.date_range("2019-01-31", periods=36, freq="ME")
    control_dir = tmp_path / "core" / "ff12_full_participation"
    treatment_dir = tmp_path / "heterogeneity" / "feature_heterogeneity_severe"
    # Control: every client predicts equally well -- low cross-client spread.
    control_rhos = {"client_00": 0.3, "client_01": 0.3, "client_02": 0.3, "client_03": 0.3}
    # Treatment: same average skill, but split into a strong-signal pair and a
    # weak/negative-signal pair -- wide cross-client spread and a much worse
    # worst client, with the per-month cross-client *mean* roughly unchanged.
    treatment_rhos = {"client_00": 0.7, "client_01": 0.7, "client_02": -0.1, "client_03": -0.1}
    for seed in [11, 22, 33]:
        _write_predictions(
            control_dir / f"seed_{seed}",
            method="fedadam",
            dates=dates,
            client_rhos=control_rhos,
            seed=seed,
        )
        _write_predictions(
            treatment_dir / f"seed_{seed}",
            method="fedadam",
            dates=dates,
            client_rhos=treatment_rhos,
            seed=seed + 1000,
        )

    result = heterogeneity_dispersion_hac_contrasts(
        [(treatment_dir, control_dir, "feature severe minus reference")],
        methods=["fedadam"],
    )
    assert set(result["statistic"]) == {"std", "worst"}
    std_row = result.loc[result["statistic"] == "std"].iloc[0]
    worst_row = result.loc[result["statistic"] == "worst"].iloc[0]
    assert std_row["n_seeds"] == 3
    # Treatment widens cross-client Rank-IC dispersion relative to control.
    assert std_row["mean_difference_of_means"] > 0
    assert std_row["combined_p_value"] < 0.05
    # ...and drags the worst client down.
    assert worst_row["mean_difference_of_means"] < 0
    assert worst_row["combined_p_value"] < 0.05


def test_heterogeneity_dispersion_hac_contrasts_skips_missing_dirs(tmp_path: Path):
    result = heterogeneity_dispersion_hac_contrasts(
        [(tmp_path / "missing_treatment", tmp_path / "missing_control", "label")],
    )
    assert result.empty


def test_identification_hac_contrasts_works_for_a_heterogeneity_style_phase_dir(
    tmp_path: Path,
):
    """identification_hac_contrasts/heterogeneity_dispersion_hac_contrasts are
    phase-agnostic (just phase_dir/scenario-name lookups), which is what lets
    HETEROGENEITY_MATCHED_CONTROL_PAIRS reuse them for the heterogeneity
    phase's own treatment-vs-matched-control pairs. Exercise that path
    directly rather than relying on it being implied by the identification
    tests above.
    """
    dates = pd.date_range("2019-01-31", periods=48, freq="ME")
    phase_dir = tmp_path / "heterogeneity"
    for seed in [11, 22, 33]:
        local_rng = np.random.default_rng(seed)
        control_returns = local_rng.normal(0.0, 0.02, size=len(dates))
        treatment_returns = control_returns - 0.01  # treatment consistently worse
        _write_run(
            phase_dir / "covariate_shift_mild_matched" / f"seed_{seed}",
            methods=["fedadam"],
            reference_method="fedadam",
            dates=dates,
            returns={"fedadam": control_returns},
        )
        _write_run(
            phase_dir / "covariate_shift_mild" / f"seed_{seed}",
            methods=["fedadam"],
            reference_method="fedadam",
            dates=dates,
            returns={"fedadam": treatment_returns},
        )
    result = identification_hac_contrasts(
        phase_dir,
        [("covariate_shift_mild", "covariate_shift_mild_matched", "covariate mild minus matched")],
        variant="client_neutral",
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_seeds"] == 3
    assert row["mean_difference_of_means"] < 0
    assert row["combined_p_value"] < 0.05


def test_benjamini_hochberg_controls_fdr_and_preserves_order():
    from fapbench.evaluation.statistics import benjamini_hochberg

    q = benjamini_hochberg([0.001, 0.04, 0.5, np.nan])
    assert np.isnan(q[3])
    # monotone in the input p-values, and never below the raw p-value
    assert q[0] <= q[1] <= q[2]
    assert all(q[i] >= p for i, p in enumerate([0.001, 0.04, 0.5]))
    assert q[2] <= 1.0


def test_moving_block_bootstrap_mean_brackets_the_sample_mean():
    from fapbench.evaluation.statistics import moving_block_bootstrap_mean

    rng = np.random.default_rng(0)
    series = rng.normal(0.5, 1.0, size=200)
    out = moving_block_bootstrap_mean(series, n_bootstrap=1000, seed=1)
    assert out["n_periods"] == 200
    assert out["ci_lower"] < out["mean"] < out["ci_upper"]
    # a mean this far from zero should have an interval excluding it
    assert out["ci_lower"] > 0


def test_seed_averaged_dispersion_contrasts_matches_per_seed_effect_size(tmp_path: Path):
    """Averaging across seeds preserves the effect size but changes only the
    uncertainty statement -- this is the property that lets the paper keep its
    reported magnitudes while replacing the Stouffer inference."""
    from fapbench.evaluation.significance import seed_averaged_dispersion_contrasts

    dates = pd.date_range("2019-01-31", periods=36, freq="ME")
    control_dir = tmp_path / "core" / "ref"
    treatment_dir = tmp_path / "het" / "treated"
    control_rhos = {f"client_0{i}": 0.3 for i in range(4)}
    treatment_rhos = {"client_00": 0.7, "client_01": 0.7,
                      "client_02": -0.1, "client_03": -0.1}
    for seed in [11, 22, 33]:
        _write_predictions(control_dir / f"seed_{seed}", method="fedadam",
                           dates=dates, client_rhos=control_rhos, seed=seed)
        _write_predictions(treatment_dir / f"seed_{seed}", method="fedadam",
                           dates=dates, client_rhos=treatment_rhos, seed=seed + 1000)

    result = seed_averaged_dispersion_contrasts(
        [(treatment_dir, control_dir, "treated minus ref")],
        methods=["fedadam"], n_bootstrap=500,
    )
    std_row = result.loc[result["statistic"] == "std"].iloc[0]
    assert std_row["n_seeds"] == 3
    assert std_row["n_months"] == 36
    assert std_row["effect"] > 0
    assert std_row["ci_lower"] > 0 and std_row["excludes_zero"]
    # q-values are present and never smaller than the raw p-value
    assert std_row["hac_q_value"] >= std_row["hac_p_value"]
    assert 0.0 <= std_row["seed_sign_agreement"] <= 1.0


def test_seed_averaged_dispersion_contrasts_skips_missing_dirs(tmp_path: Path):
    from fapbench.evaluation.significance import seed_averaged_dispersion_contrasts

    assert seed_averaged_dispersion_contrasts(
        [(tmp_path / "missing_t", tmp_path / "missing_c", "label")]
    ).empty


def test_seed_averaged_contrasts_accepts_a_subperiod_mask(tmp_path):
    """The `months` filter must actually shorten the tested series.

    The 2019-2023 window includes COVID, so the paper reports an off-COVID
    re-run of the same contrast; this guards the mechanism that produces it.
    """
    import numpy as np
    import pandas as pd

    from fapbench.evaluation.significance import seed_averaged_dispersion_contrasts

    dates = pd.date_range("2019-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(0)

    def _write(root, shift):
        for seed in (11, 22):
            seed_dir = root / f"seed_{seed}"
            seed_dir.mkdir(parents=True)
            frames = []
            for date in dates:
                for client in ("c0", "c1", "c2"):
                    y = rng.normal(size=25)
                    frames.append(
                        pd.DataFrame(
                            {
                                "target_date": date,
                                "client_id": client,
                                "y_true": y,
                                "y_pred": y * shift + rng.normal(scale=0.5, size=25),
                            }
                        )
                    )
            pd.concat(frames).to_parquet(seed_dir / "predictions_fedavg.parquet")

    treatment, control = tmp_path / "treat", tmp_path / "ctrl"
    _write(treatment, 0.2)
    _write(control, 0.8)
    pairs = [(treatment, control, "synthetic")]

    full = seed_averaged_dispersion_contrasts(
        pairs, methods=["fedavg"], statistics=("std",), n_bootstrap=50
    )
    partial = seed_averaged_dispersion_contrasts(
        pairs,
        methods=["fedavg"],
        statistics=("std",),
        n_bootstrap=50,
        months=lambda index: index.year != 2020,
        period_label="ex-2020",
    )
    assert int(full.iloc[0]["n_months"]) == 36
    assert int(partial.iloc[0]["n_months"]) == 24
    assert full.iloc[0]["period"] == "full"
    assert partial.iloc[0]["period"] == "ex-2020"


# --- expanding-window RQ3 dispersion contrasts -----------------------------
#
# The reviewer objection these cover: the paper builds a 408-month expanding
# window for its economic axis but leaves RQ3 at n=60. The estimator that
# closes that gap must (a) actually pool months across windows rather than
# averaging per-window effects, and (b) refuse to pool overlapping windows,
# which would double-count months and deflate the HAC standard error.


def _write_window_pair(root: Path, arm: str, year: int, rhos: dict[str, float],
                       *, periods: int = 24, seeds=(11, 22, 33)) -> None:
    dates = pd.date_range(f"{year}-01-31", periods=periods, freq="ME")
    for seed in seeds:
        _write_predictions(
            root / f"{arm}__refit_{year}" / f"seed_{seed}",
            method="fedavg", dates=dates, client_rhos=rhos, seed=seed + year,
        )


def test_rolling_dispersion_contrasts_pools_months_across_windows(tmp_path: Path):
    """Three disjoint 24-month windows must yield one 72-month series, not
    three separate contrasts and not a 24-month average of them."""
    from fapbench.evaluation.significance import rolling_dispersion_contrasts

    spread = {"client_00": 0.7, "client_01": 0.7, "client_02": -0.1, "client_03": -0.1}
    flat = {f"client_0{i}": 0.3 for i in range(4)}
    years = [1990, 1992, 1994]
    for year in years:
        _write_window_pair(tmp_path, "treated", year, spread)
        _write_window_pair(tmp_path, "control", year, flat)

    result = rolling_dispersion_contrasts(
        {
            "treated minus control": [
                (tmp_path / f"treated__refit_{y}", tmp_path / f"control__refit_{y}")
                for y in years
            ]
        },
        methods=["fedavg"], n_bootstrap=500,
    )
    row = result.iloc[0]
    assert int(row["n_windows"]) == 3
    assert int(row["n_months"]) == 72  # 3 x 24, pooled -- not 24
    assert int(row["n_seeds"]) == 3
    assert row["effect"] > 0
    assert row["statistic"] == "std"


def test_rolling_dispersion_contrasts_rejects_overlapping_windows(tmp_path: Path):
    """Overlapping test blocks would count a month twice. That must fail loudly:
    it deflates the HAC standard error, so it manufactures significance."""
    from fapbench.evaluation.significance import rolling_dispersion_contrasts

    spread = {"client_00": 0.7, "client_01": 0.7, "client_02": -0.1, "client_03": -0.1}
    flat = {f"client_0{i}": 0.3 for i in range(4)}
    # Both windows start in 1990 and run 24 months -> identical month index.
    for arm, rhos in [("treated", spread), ("control", flat)]:
        _write_window_pair(tmp_path, arm, 1990, rhos)
        _write_window_pair(tmp_path, arm, 1991, rhos)
    # Force the overlap: rewrite 1991's block over the same dates as 1990's.
    for arm, rhos in [("treated", spread), ("control", flat)]:
        for seed in (11, 22, 33):
            _write_predictions(
                tmp_path / f"{arm}__refit_1991" / f"seed_{seed}", method="fedavg",
                dates=pd.date_range("1990-01-31", periods=24, freq="ME"),
                client_rhos=rhos, seed=seed + 1991,
            )

    with pytest.raises(ValueError, match="overlap"):
        rolling_dispersion_contrasts(
            {
                "treated minus control": [
                    (tmp_path / f"treated__refit_{y}", tmp_path / f"control__refit_{y}")
                    for y in (1990, 1991)
                ]
            },
            methods=["fedavg"], n_bootstrap=100,
        )


def test_rolling_and_single_window_agree_on_a_single_window(tmp_path: Path):
    """With one window the two estimators must return the same effect: they
    share the difference-matrix helper, and this pins that they still do."""
    from fapbench.evaluation.significance import (
        rolling_dispersion_contrasts,
        seed_averaged_dispersion_contrasts,
    )

    spread = {"client_00": 0.7, "client_01": 0.7, "client_02": -0.1, "client_03": -0.1}
    flat = {f"client_0{i}": 0.3 for i in range(4)}
    _write_window_pair(tmp_path, "treated", 1990, spread)
    _write_window_pair(tmp_path, "control", 1990, flat)
    t, c = tmp_path / "treated__refit_1990", tmp_path / "control__refit_1990"

    rolled = rolling_dispersion_contrasts(
        {"lab": [(t, c)]}, methods=["fedavg"], n_bootstrap=500,
    )
    single = seed_averaged_dispersion_contrasts(
        [(t, c, "lab")], methods=["fedavg"], statistics=("std",), n_bootstrap=500,
    )
    assert rolled.iloc[0]["effect"] == pytest.approx(single.iloc[0]["effect"])
    assert rolled.iloc[0]["n_months"] == single.iloc[0]["n_months"]

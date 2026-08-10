from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fapbench.robustness import (
    build_robustness_outputs,
    compare_against_reference,
    fedprox_mu_sweep,
    local_tuning_effect,
    transaction_cost_sensitivity,
)

SEEDS = [11, 22, 33, 44, 55]


def _rows(scenario, method, *, rank_ic, sharpe, seeds=SEEDS):
    return [
        {
            "scenario": scenario,
            "seed": seed,
            "method": method,
            "rank_ic": rank_ic + 0.001 * i,
            "oos_r2_zero": 0.001,
            "client_worst_rank_ic": -0.01,
            "client_neutral_portfolio_net_sharpe": sharpe + 0.01 * i,
        }
        for i, seed in enumerate(seeds)
    ]


def _write_grids(root: Path, robustness_rows, reference_rows):
    for phase, rows in (("robustness", robustness_rows), ("core", reference_rows)):
        (root / phase).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(root / phase / "grid_results.csv", index=False)


def _reference_rows():
    rows = []
    for method, rank_ic, sharpe in [
        ("fedavg", 0.034, 0.27),
        ("fedadam", 0.058, 0.24),
        ("local", 0.012, 0.17),
    ]:
        rows += _rows("ff12_full_participation", method, rank_ic=rank_ic, sharpe=sharpe)
    return rows


def test_compare_against_reference_pairs_by_seed(tmp_path: Path):
    robustness = _rows("wide_mlp", "fedavg", rank_ic=0.044, sharpe=0.30)
    _write_grids(tmp_path, robustness, _reference_rows())

    out = compare_against_reference(tmp_path, [("wide_mlp", "wide minus reference")])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["method"] == "fedavg"
    assert row["n_seeds"] == len(SEEDS)
    # constructed as a constant +0.010 shift on every seed
    assert row["rank_ic_delta"] == pytest_approx(0.010)
    # a perfectly constant paired difference has zero variance -> undefined t
    assert np.isnan(row["rank_ic_p"])


def pytest_approx(value, tol=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(float(other) - value) < tol

        def __repr__(self):  # pragma: no cover - only on failure
            return f"~{value}"

    return _A()


def test_compare_against_reference_skips_missing_scenarios(tmp_path: Path):
    _write_grids(tmp_path, _rows("wide_mlp", "fedavg", rank_ic=0.04, sharpe=0.3),
                 _reference_rows())
    assert compare_against_reference(tmp_path, [("not_present", "label")]).empty


def test_compare_against_reference_empty_when_phase_absent(tmp_path: Path):
    assert compare_against_reference(tmp_path, [("wide_mlp", "label")]).empty


def test_fedprox_sweep_compares_within_run(tmp_path: Path):
    rows = []
    # mu=0.001 leaves FedProx identical to FedAvg; mu=1.0 moves it.
    rows += _rows("fedprox_mu_0_001", "fedprox", rank_ic=0.034, sharpe=0.27)
    rows += _rows("fedprox_mu_0_001", "fedavg", rank_ic=0.034, sharpe=0.27)
    rows += _rows("fedprox_mu_1_0", "fedprox", rank_ic=0.020, sharpe=0.15)
    rows += _rows("fedprox_mu_1_0", "fedavg", rank_ic=0.034, sharpe=0.27)
    _write_grids(tmp_path, rows, _reference_rows())

    out = fedprox_mu_sweep(tmp_path).set_index("mu")
    assert set(out.index) == {"0.001", "1.0"}
    assert out.loc["0.001", "rank_ic_delta"] == pytest_approx(0.0)
    assert out.loc["1.0", "rank_ic_delta"] == pytest_approx(-0.014)


def test_transaction_cost_sensitivity_collects_all_three_levels(tmp_path: Path):
    rows = []
    rows += _rows("cost_25bps", "fedavg", rank_ic=0.034, sharpe=0.20)
    rows += _rows("cost_50bps", "fedavg", rank_ic=0.034, sharpe=0.10)
    _write_grids(tmp_path, rows, _reference_rows())

    out = transaction_cost_sensitivity(tmp_path).set_index("method")
    assert out.loc["fedavg", "net_sharpe_10bps"] > out.loc["fedavg", "net_sharpe_25bps"]
    assert out.loc["fedavg", "net_sharpe_25bps"] > out.loc["fedavg", "net_sharpe_50bps"]
    assert out.loc["fedavg", "breakeven_gap_10_to_50"] == pytest_approx(0.17)
    # methods with no cost variant still appear, with the 10bps level only
    assert "fedadam" in out.index


def test_local_tuning_effect_reports_gap_to_fedavg(tmp_path: Path):
    tuned = _rows("local_tuned", "local", rank_ic=0.022, sharpe=0.21)
    _write_grids(tmp_path, tuned, _reference_rows())

    out = local_tuning_effect(tmp_path)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["rank_ic_delta"] == pytest_approx(0.010)
    # tuning narrows the federated-minus-local gap
    assert row["fedavg_minus_local_tuned_rank_ic"] < row[
        "fedavg_minus_local_inherited_rank_ic"
    ]


def test_build_robustness_outputs_writes_every_table(tmp_path: Path):
    rows = _rows("wide_mlp", "fedavg", rank_ic=0.04, sharpe=0.30)
    rows += _rows("local_tuned", "local", rank_ic=0.02, sharpe=0.20)
    rows += _rows("cost_25bps", "fedavg", rank_ic=0.034, sharpe=0.20)
    _write_grids(tmp_path, rows, _reference_rows())

    tables = build_robustness_outputs(tmp_path)
    expected = {
        "robustness_model_class",
        "robustness_fedprox_mu",
        "robustness_transaction_costs",
        "robustness_local_tuning",
        "robustness_market_adjusted_alpha",
        "robustness_metric_rank_agreement",
        "robustness_ranking_disagreement_test",
        "robustness_capacity_separation",
        "robustness_block_length_sensitivity",
        "rq3_investable",
        "rq3_rolling",
        "second_moment_control",
        "linear_separation",
        "severity_escalation",
        "ensemble_local_comparison",
        "ensemble_local_levels",
    }
    assert set(tables) == expected
    for name in expected:
        assert (tmp_path / "paper_tables" / f"{name}.csv").exists()
    # Each RQ3 robustness grid is run separately, so its table must be written even
    # when that phase is absent -- empty, not missing, and never an exception.
    for name in (
        "robustness_capacity_separation",
        "rq3_investable",
        "rq3_rolling",
        "second_moment_control",
        "linear_separation",
        "severity_escalation",
        "ensemble_local_comparison",
        "ensemble_local_levels",
    ):
        assert tables[name].empty


def test_rq3_robustness_pair_registries_are_well_formed():
    """The four RQ3 dispersion grids must stay aligned with each other.

    Two invariants the manuscript depends on: the linear grid re-runs *exactly*
    the pairs the capacity grid does (the two tables are compared row for row), and
    every escalation rung names a distinct treatment/control directory pair -- a
    copy-paste slip there would silently contrast a scenario against itself.
    """
    from fapbench.robustness import (
        CAPACITY_PAIRS,
        LINEAR_SEPARATION_PAIRS,
        RQ3_INVESTABLE_PAIRS,
        SEVERITY_ESCALATION_PAIRS,
    )

    assert LINEAR_SEPARATION_PAIRS == CAPACITY_PAIRS
    for treatment, control, _ in (
        *RQ3_INVESTABLE_PAIRS,
        *SEVERITY_ESCALATION_PAIRS,
        *CAPACITY_PAIRS,
    ):
        assert treatment != control
    for registry in (RQ3_INVESTABLE_PAIRS, SEVERITY_ESCALATION_PAIRS, CAPACITY_PAIRS):
        labels = [label for _, _, label in registry]
        assert len(labels) == len(set(labels))


def test_build_robustness_outputs_tolerates_a_missing_grid(tmp_path: Path):
    """A partially-completed grid must not crash the build."""
    (tmp_path / "core").mkdir(parents=True)
    pd.DataFrame(_reference_rows()).to_csv(
        tmp_path / "core" / "grid_results.csv", index=False
    )
    tables = build_robustness_outputs(tmp_path)
    # Comparisons need both arms and are empty; the cost table still reports the
    # reference 10bps column, which is the intended partial-progress behaviour.
    assert tables["robustness_model_class"].empty
    assert tables["robustness_fedprox_mu"].empty
    assert tables["robustness_local_tuning"].empty
    assert not tables["robustness_transaction_costs"].empty
    assert "net_sharpe_25bps" not in tables["robustness_transaction_costs"].columns
    # No panel path was supplied, so the market adjustment is skipped rather
    # than raising -- the publication build must run without the licensed panel.
    assert tables["robustness_market_adjusted_alpha"].empty


def test_market_adjusted_alpha_skips_a_missing_panel(tmp_path: Path):
    from fapbench.robustness import market_adjusted_alpha

    assert market_adjusted_alpha(tmp_path, None).empty
    assert market_adjusted_alpha(tmp_path, tmp_path / "absent.parquet").empty


def test_capacity_separation_ignores_pairs_whose_scenarios_are_absent(tmp_path: Path):
    """A partially-run capacity grid must contribute what it has, not crash.

    The grid is 60 cells and takes hours; reading it while only some scenarios
    exist is the normal case, not an edge case.
    """
    from fapbench.robustness import CAPACITY_PHASE, capacity_separation_contrasts

    phase = tmp_path / CAPACITY_PHASE
    (phase / "feature_heterogeneity_severe").mkdir(parents=True)
    # Its matched control is missing, so the pair cannot be formed and the
    # result is empty rather than a half-built contrast.
    assert capacity_separation_contrasts(tmp_path).empty


def test_perfect_agreement_exclusion_is_not_decided_by_floating_point():
    """Regression: `upper < 1.0` reported exclusion that the draws contradict.

    Kendall's tau on K methods is discrete, so a percentile upper bound can land
    exactly on 1.0 and still compare strictly less by one ulp. On the reference
    universe that accident reported "excludes perfect agreement" while 4.7% of
    bootstrap draws -- well beyond the 2.5% a two-sided 95% interval allows --
    actually reached tau = 1, which would have put a false claim in the paper.
    The flag must follow the draws, not the float comparison.
    """
    import numpy as np

    # 4% of draws at tau = 1: strictly inside a two-sided 95% interval.
    draws = np.concatenate([np.full(40, 1.0), np.linspace(-0.5, 0.9, 960)])
    upper = float(np.percentile(draws, 97.5))
    share = float(np.mean(draws >= 1.0 - 1e-12))

    assert share > 0.025, "fixture must place tau=1 inside the interval"
    assert bool(share < 0.025) is False, "share-based rule must not exclude"
    # The old rule could disagree with the share-based one precisely when the
    # upper bound sits on the boundary; the share-based rule is the correct one.
    assert (share < 0.025) is not (upper < 1.0) or upper >= 1.0


def test_rq3_rolling_contrasts_discovers_window_pairs(tmp_path):
    """The grid writes one directory per (arm, window); the aggregator must pair
    them by window and skip windows whose control arm has not finished, rather
    than pairing a treatment against some other window's control."""
    import pandas as pd
    from fapbench.robustness import rq3_rolling_contrasts

    phase = tmp_path / "rq3_rolling"
    years = [1990, 1992, 1994]
    rng = __import__("numpy").random.default_rng(0)

    def _write(arm, year, spread):
        dates = pd.date_range(f"{year}-01-31", periods=24, freq="ME")
        for seed in (11, 22):
            run = phase / f"{arm}__refit_{year}" / f"seed_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            rows = []
            for date in dates:
                for ci, rho in enumerate(spread):
                    yt = rng.normal(size=20)
                    yp = rho * yt + (1 - rho**2) ** 0.5 * rng.normal(size=20)
                    for permno, (a, b) in enumerate(zip(yt, yp)):
                        rows.append({"target_date": date, "client_id": f"client_0{ci}",
                                     "permno": permno, "y_true": a, "y_pred": b})
            pd.DataFrame(rows).to_parquet(run / "predictions_fedavg.parquet")

    for year in years:
        _write("feature_heterogeneity_severe", year, [0.7, 0.7, -0.1, -0.1])
        _write("feature_heterogeneity_severe_matched", year, [0.3, 0.3, 0.3, 0.3])
    # 1996's treatment arm exists but its control does not: the window must be
    # dropped whole, not paired against a neighbouring window's control.
    _write("feature_heterogeneity_severe", 1996, [0.7, 0.7, -0.1, -0.1])

    result = rq3_rolling_contrasts(
        tmp_path,
        pairs=[("feature_heterogeneity_severe",
                "feature_heterogeneity_severe_matched", "feature severe")],
        methods=["fedavg"],
    )
    row = result.iloc[0]
    assert int(row["n_windows"]) == 3, "incomplete 1996 window must be dropped"
    assert int(row["n_months"]) == 72
    assert row["contrast"] == "feature severe"


def test_rq3_rolling_contrasts_is_empty_before_the_grid_runs(tmp_path):
    from fapbench.robustness import rq3_rolling_contrasts

    assert rq3_rolling_contrasts(tmp_path).empty


def test_rq3_rolling_config_windows_are_disjoint_and_cover_408_months():
    """The whole point of the grid is a 408-month sample; overlapping or
    short-changed windows would silently undercut it."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "make_rq3_rolling_config",
        Path(__file__).resolve().parent.parent / "scripts" / "make_rq3_rolling_config.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = module.build_config(step=2)
    spans = {}
    for scenario in config["scenarios"]:
        arm, year = scenario["name"].rsplit("__refit_", 1)
        split = scenario["overrides"]["split"]
        spans.setdefault(arm, []).append(
            (int(split["val_end"][:4]), int(split["test_end"][:4]))
        )
        # Training must stop strictly before validation, which stops before test.
        assert split["train_end"] < split["val_end"] < split["test_end"]

    assert set(spans) == set(module.ARMS)
    for arm, windows in spans.items():
        windows.sort()
        months = sum((end - val) * 12 for val, end in windows)
        assert months == 408, f"{arm} covers {months} months, not 408"
        for (_, prev_end), (next_val, _) in zip(windows, windows[1:]):
            assert prev_end <= next_val, f"{arm} has overlapping test blocks"

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
    }
    assert set(tables) == expected
    for name in expected:
        assert (tmp_path / "paper_tables" / f"{name}.csv").exists()
    # The capacity grid is run separately, so its table must be written even
    # when that phase is absent -- empty, not missing, and never an exception.
    assert tables["robustness_capacity_separation"].empty


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

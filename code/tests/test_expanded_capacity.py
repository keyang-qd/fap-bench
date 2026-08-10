"""Capacity re-selection on the 20-characteristic panel (reviewer W2).

The paper declines to read the 20-characteristic arm as a verdict on the wider
panel because that arm holds architecture and optimizer at seven-input values.
These tests guard the one way that argument could be made dishonestly: selecting
a capacity on test performance and then reporting the same test performance as a
recovery.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fapbench.publication import select_expanded_capacity


def _results(tmp_path, rows):
    path = tmp_path / "grid_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(scenario, method, val, test=0.0):
    return {
        "scenario": scenario,
        "method": method,
        "validation_selected_rank_ic": val,
        "rank_ic": test,
    }


def test_selection_refuses_a_test_metric(tmp_path):
    """The load-bearing guard. Selecting capacity on test Rank IC and reporting
    that same figure would fabricate the recovery the sweep tests for."""
    path = _results(tmp_path, [_row("expanded_capacity_h64_32_lr0_01", "centralized", 0.01)])
    with pytest.raises(ValueError, match="validation-only"):
        select_expanded_capacity(path, metric="rank_ic")


def test_selection_picks_the_best_validation_candidate_and_reports_the_margin(tmp_path):
    """The reference architecture must be reported alongside the winner, so a
    marginal 'recovery' cannot be presented as a decisive one."""
    rows = []
    for seed_val in (0.010, 0.012):
        rows.append(_row("expanded_capacity_h64_32_lr0_01", "centralized", seed_val))
    for seed_val in (0.020, 0.022):
        rows.append(_row("expanded_capacity_h256_128_lr0_003", "centralized", seed_val))
    result = select_expanded_capacity(_results(tmp_path, rows))

    assert result["selected_scenario"] == "expanded_capacity_h256_128_lr0_003"
    assert result["selected_value"] == pytest.approx(0.021)
    assert result["reference_best_value"] == pytest.approx(0.011)
    assert result["n_candidates"] == 2
    assert [r["scenario"] for r in result["ranking"]][0] == result["selected_scenario"]


def test_selection_can_return_the_reference_architecture(tmp_path):
    """Including [64,32] in the sweep is what makes it a test rather than a
    search: if nothing wider helps, selection must be able to recover the
    published configuration."""
    rows = [
        _row("expanded_capacity_h64_32_lr0_01", "centralized", 0.030),
        _row("expanded_capacity_h256_128_lr0_01", "centralized", 0.010),
    ]
    result = select_expanded_capacity(_results(tmp_path, rows))
    assert "h64_32" in result["selected_scenario"]
    assert result["selected_value"] == pytest.approx(result["reference_best_value"])


def test_selection_is_method_scoped(tmp_path):
    """Selecting on Centralized must not silently average in FedAvg's rows."""
    rows = [
        _row("expanded_capacity_h64_32_lr0_01", "centralized", 0.030),
        _row("expanded_capacity_h64_32_lr0_01", "fedavg", 0.001),
        _row("expanded_capacity_h256_128_lr0_01", "centralized", 0.010),
        _row("expanded_capacity_h256_128_lr0_01", "fedavg", 0.099),
    ]
    on_centralized = select_expanded_capacity(_results(tmp_path, rows))
    on_fedavg = select_expanded_capacity(_results(tmp_path, rows), method="fedavg")
    assert "h64_32" in on_centralized["selected_scenario"]
    assert "h256_128" in on_fedavg["selected_scenario"]


def test_empty_and_missing_columns_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="No capacity candidates"):
        select_expanded_capacity(_results(tmp_path, [_row("other_scenario", "centralized", 0.1)]))
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"scenario": "expanded_capacity_x"}]).to_csv(bad, index=False)
    with pytest.raises(KeyError, match="Missing capacity-selection columns"):
        select_expanded_capacity(bad)


def test_sweep_config_includes_the_reference_architecture_and_is_investable():
    """Two properties the comparison depends on: the sweep must be able to select
    the published architecture, and it must select for the universe the headline
    table reports."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent / "configs"
    sweep = yaml.safe_load((root / "icaif_expanded_capacity_sweep.yaml").read_text())
    core = yaml.safe_load((root / "icaif_core.yaml").read_text())

    names = [s["name"] for s in sweep["scenarios"]]
    assert all(n.startswith("expanded_capacity_") for n in names)
    assert any("h64_32" in n for n in names), "reference architecture must be a candidate"

    reference_filter = next(
        s for s in core["scenarios"] if s["name"] == "ff12_exclude_microcaps"
    )["overrides"]["universe_filter"]
    assert sweep["base_overrides"]["universe_filter"] == reference_filter
    assert sweep["base_config"] == "crsp_compustat_profitability_expanded.yaml"

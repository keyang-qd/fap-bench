from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"


def _load(name: str):
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_fedadam_publication_value_and_calibration_are_frozen() -> None:
    base = _load("crsp_compustat_profitability.yaml")
    assert base["federated"]["server_lr"] == 0.003
    assert base["federated"]["server_lr_selection"]["scope"] == "validation_only"
    assert base["evaluation"]["validation_prediction_calibration"]["enabled"] is True


def test_publication_plan_contains_only_reported_phases() -> None:
    plan = _load("icaif_publication.yaml")
    assert [phase["name"] for phase in plan["phases"]] == [
        "fedadam_validation",
        "core",
        "partitions",
        "heterogeneity",
        "identification",
        "label_diagnostics",
        "core_expanded_features",
    ]
    assert plan["expected_fedadam_server_lr"] == 0.003


def test_core_grid_preserves_all_headline_methods_and_protocols() -> None:
    base = _load("crsp_compustat_profitability.yaml")
    core = _load("icaif_core.yaml")
    assert base["baselines"]["centralized"] is True
    assert base["baselines"]["local"] is True
    assert base["federated"]["algorithms"] == [
        "fedavg", "fedprox", "scaffold", "fedadam"
    ]
    assert [s["name"] for s in core["scenarios"]] == [
        "ff12_full_participation",
        "ff12_half_participation",
        "ff12_more_local_steps",
        "ff12_price_above_5",
        "ff12_exclude_microcaps",
        "ff12_exclude_microcaps_more_local_steps",
    ]
    local_steps = next(s for s in core["scenarios"] if s["name"] == "ff12_more_local_steps")
    fed = local_steps["overrides"]["federated"]
    assert fed["rounds"] * fed["local_epochs"] == 50


def test_noncore_grids_drop_unreported_repeated_methods() -> None:
    for name in ["icaif_partitions.yaml", "icaif_heterogeneity.yaml"]:
        cfg = _load(name)
        assert cfg["base_overrides"]["baselines"]["local"] is False
        assert cfg["base_overrides"]["federated"]["algorithms"] == [
            "fedavg", "scaffold", "fedadam"
        ]
        assert cfg["base_overrides"]["federated"]["server_lr"] == 0.003
    identification = _load("icaif_identification.yaml")
    assert identification["base_overrides"]["baselines"]["centralized"] is False
    assert identification["base_overrides"]["federated"]["algorithms"] == [
        "fedavg", "scaffold", "fedadam"
    ]


def test_identification_grid_keeps_only_strong_matched_controls() -> None:
    cfg = _load("icaif_identification.yaml")
    assert [s["name"] for s in cfg["scenarios"]] == [
        "size_uniform_matched_ratio_20",
        "size_imbalance_ratio_20_identified",
        "temporal_persistent_strong_identified",
        "temporal_random_matched_persistent_strong",
        "temporal_block_outages_identified",
        "temporal_random_matched_block_outages",
    ]


def test_expanded_features_config_adds_value_investment_momentum() -> None:
    base = _load("crsp_compustat_profitability_expanded.yaml")
    assert base["data"]["feature_columns"] == [
        "gpoa_zscore", "roa_zscore", "roe_zscore", "roic_zscore",
        "opoa_zscore", "gmar_zscore", "fcfoa_zscore",
        "bm_zscore", "ep_zscore", "cfp_zscore", "sp_zscore", "dy_zscore",
        "agr_zscore", "capx_at_zscore", "lev_zscore", "acc_at_zscore",
        "rdoa_zscore", "logme_zscore", "mom_2_12_zscore", "mom_1_1_zscore",
    ]
    assert len(base["data"]["feature_columns"]) == 20
    # Points at the Tier 2 robustness-check panel, not the primary one.
    assert base["data"]["path"] != _load("crsp_compustat_profitability.yaml")["data"]["path"]


def test_expanded_core_grid_reuses_core_scenarios_and_seeds() -> None:
    core = _load("icaif_core.yaml")
    expanded = _load("icaif_core_expanded_features.yaml")
    assert expanded["base_config"] == "crsp_compustat_profitability_expanded.yaml"
    # Every expanded scenario must exist in core with byte-identical overrides,
    # so the 7- vs 20-characteristic comparison is like-for-like. Core may carry
    # extra scenarios the expanded panel does not need -- notably the
    # investable-universe/low-communication cell, which exists only to make the
    # headline table coherent and supports no 20-characteristic claim.
    core_by_name = {s["name"]: s["overrides"] for s in core["scenarios"]}
    expanded_names = [s["name"] for s in expanded["scenarios"]]
    assert set(expanded_names) <= set(core_by_name)
    for scenario in expanded["scenarios"]:
        assert scenario["overrides"] == core_by_name[scenario["name"]], scenario["name"]
    # Matches core's 10 seeds rather than a separate 5-seed list: per-seed
    # Sharpe on the expanded panel is noisy enough that 5 seeds could not
    # support a confident robustness claim.
    assert expanded["seeds"] == core["seeds"]
    assert expanded["output_dir"] != core["output_dir"]


def test_obsolete_real_data_grids_were_removed() -> None:
    obsolete = [
        "crsp_compustat_profitability_partition_grid.yaml",
        "crsp_compustat_profitability_hyperparameter_grid.yaml",
        "crsp_compustat_profitability_calibration_grid.yaml",
        "crsp_compustat_profitability_grid_quick.yaml",
        "crsp_compustat_profitability_grid_draft.yaml",
    ]
    assert all(not (CONFIGS / name).exists() for name in obsolete)


def test_local_tuning_grid_defaults_to_inherited_configuration():
    """Disabled tuning must reproduce the inherited federated hyperparameters
    exactly, so enabling the feature is the only thing that changes results."""
    from fapbench.runner import _local_tuning_grid

    assert _local_tuning_grid({}, 0.01, 0.0) == [(0.01, 0.0)]
    assert _local_tuning_grid({"local_tuning": {"enabled": False}}, 0.02, 0.1) == [
        (0.02, 0.1)
    ]


def test_local_tuning_grid_expands_to_full_cross_product():
    from fapbench.runner import _local_tuning_grid

    grid = _local_tuning_grid(
        {
            "local_tuning": {
                "enabled": True,
                "learning_rates": [0.001, 0.01],
                "weight_decays": [0.0, 0.01],
            }
        },
        0.01,
        0.0,
    )
    assert grid == [(0.001, 0.0), (0.001, 0.01), (0.01, 0.0), (0.01, 0.01)]


def test_local_tuning_grid_falls_back_when_lists_are_empty():
    from fapbench.runner import _local_tuning_grid

    grid = _local_tuning_grid(
        {"local_tuning": {"enabled": True, "learning_rates": [], "weight_decays": []}},
        0.03,
        0.02,
    )
    assert grid == [(0.03, 0.02)]


def test_local_tuning_requires_validation_selection():
    """Without per-client validation there is no signal to select on, so the
    sweep would silently keep the first candidate. Fail loudly instead."""
    import pytest

    from fapbench.runner import _local_tuning_grid

    with pytest.raises(ValueError, match="model_selection.enabled"):
        _local_tuning_grid(
            {"local_tuning": {"enabled": True, "learning_rates": [0.001, 0.01]}},
            0.01,
            0.0,
            selection_enabled=False,
        )
    # disabled tuning is unaffected by the selection setting
    assert _local_tuning_grid({}, 0.01, 0.0, selection_enabled=False) == [(0.01, 0.0)]


def test_no_stale_research_question_labels_in_typeset_sources():
    """No visible "RQ<n>" may name a section the manuscript does not have.

    The bug this originally guarded: cutting a research question renumbered the
    rest, and a generated caption kept saying "(RQ2)" for what had become RQ1.
    The audit checks numbers, not labels, so nothing caught it.

    The manuscript has since replaced RQ headers with descriptive section
    titles, which voids the old invariant (headers consecutive from RQ1, intro
    announcing the count) but not the failure mode -- it inverts it. With no RQ
    sections left, any surviving "RQ<n>" a reader can see points at nothing. So
    the check is repointed rather than dropped, and it still covers the original
    bug: restore RQ headers and the consecutive check applies again.
    """
    import re
    from pathlib import Path

    paper = Path("/Users/keyang/paper-fap-bench")
    if not paper.exists():  # code repo may be checked out alone
        return

    experiment = (paper / "inputs" / "experiment.tex").read_text()
    headers = re.findall(r"\\(?:subsection|paragraph)\{(RQ\d): ([^?]+)\?\}", experiment)

    if headers:
        numbers = [h[0] for h in headers]
        assert numbers == sorted(set(numbers), key=numbers.index), "duplicate RQ labels"
        assert numbers == [f"RQ{i}" for i in range(1, len(numbers) + 1)], (
            f"RQ headers must be consecutive from RQ1; got {numbers}"
        )
        return

    # \label/\ref keys are internal identifiers and never render, so a key
    # named sec:rq2 is fine; anything a reader sees is not.
    sources = sorted((paper / "inputs").glob("*.tex")) + sorted((paper / "tables").glob("*.tex"))
    offenders = []
    for path in sources:
        for line in path.read_text().splitlines():
            visible = re.sub(r"\\(?:label|ref|eqref)\{[^}]*\}", "", line)
            if re.search(r"\bRQ\d", visible):
                offenders.append(f"{path.name}: {line.strip()[:90]}")
    assert not offenders, (
        "manuscript has no RQ sections, but these still name one:\n  "
        + "\n  ".join(offenders)
    )

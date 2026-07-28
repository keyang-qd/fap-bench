from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fapbench.audit import (
    validate_heterogeneity_matched_controls,
    validate_identification_grid,
    validate_partition_grid,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_partition_grid_passes_on_matching_fingerprints(tmp_path: Path) -> None:
    for scenario in ["common_partition_ff12", "common_partition_exchange"]:
        for seed in [11, 22]:
            _write_json(
                tmp_path / scenario / f"seed_{seed}" / "partition_common_universe.json",
                {"n_rows_after": 1000, "row_fingerprint_sha256": "abc123"},
            )
    for seed in [11, 22]:
        _write_json(
            tmp_path / "common_partition_size_quintile" / f"seed_{seed}" / "heterogeneity.json",
            {"n_clients": 5},
        )
    result = validate_partition_grid(tmp_path)
    assert result["passed"] is True
    assert result["unique_fingerprints"] == ["abc123"]
    assert result["size_partition_client_counts"] == [5, 5]


def test_validate_partition_grid_fails_on_mismatched_fingerprints(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "common_partition_ff12" / "seed_11" / "partition_common_universe.json",
        {"n_rows_after": 1000, "row_fingerprint_sha256": "abc123"},
    )
    _write_json(
        tmp_path / "common_partition_exchange" / "seed_11" / "partition_common_universe.json",
        {"n_rows_after": 1000, "row_fingerprint_sha256": "different"},
    )
    result = validate_partition_grid(tmp_path)
    assert result["passed"] is False
    assert len(result["unique_fingerprints"]) == 2


def _write_schedule(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["round", "client_id"]).to_csv(path, index=False)


def test_validate_identification_grid_passes_on_matched_schedules_and_rows(
    tmp_path: Path,
) -> None:
    schedule = [(0, "a"), (0, "b"), (1, "a"), (1, "c")]
    for scenario in [
        "temporal_persistent_strong_identified",
        "temporal_random_matched_persistent_strong",
        "temporal_block_outages_identified",
        "temporal_random_matched_block_outages",
    ]:
        _write_schedule(
            tmp_path / scenario / "seed_11" / "participation_schedule.csv", schedule
        )
    _write_json(
        tmp_path / "size_imbalance_ratio_20_identified" / "seed_11" / "run_metadata.json",
        {"train_rows": 10000},
    )
    _write_json(
        tmp_path / "size_uniform_matched_ratio_20" / "seed_11" / "run_metadata.json",
        {"train_rows": 10050},
    )
    result = validate_identification_grid(tmp_path)
    assert result["passed"] is True
    assert result["temporal_matched_counts_passed"] is True
    assert result["size_matched_rows_passed_at_2pct"] is True


def test_validate_identification_grid_fails_when_row_counts_diverge(tmp_path: Path) -> None:
    schedule = [(0, "a"), (0, "b")]
    for scenario in [
        "temporal_persistent_strong_identified",
        "temporal_random_matched_persistent_strong",
        "temporal_block_outages_identified",
        "temporal_random_matched_block_outages",
    ]:
        _write_schedule(
            tmp_path / scenario / "seed_11" / "participation_schedule.csv", schedule
        )
    _write_json(
        tmp_path / "size_imbalance_ratio_20_identified" / "seed_11" / "run_metadata.json",
        {"train_rows": 10000},
    )
    _write_json(
        tmp_path / "size_uniform_matched_ratio_20" / "seed_11" / "run_metadata.json",
        {"train_rows": 5000},
    )
    result = validate_identification_grid(tmp_path)
    assert result["passed"] is False
    assert result["size_matched_rows_passed_at_2pct"] is False


def _write_covariate_params(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_validate_heterogeneity_matched_controls_passes_on_zero_dispersion(
    tmp_path: Path,
) -> None:
    for scenario in ["covariate_shift_mild_matched", "covariate_shift_strong_matched"]:
        _write_covariate_params(
            tmp_path / scenario / "seed_11" / "covariate_shift_parameters.csv",
            [
                {"client_id": "c0", "feature": "f0", "location_shift": 0.1, "scale_multiplier": 1.2},
                {"client_id": "c1", "feature": "f0", "location_shift": 0.1, "scale_multiplier": 1.2},
            ],
        )
    for scenario in [
        "feature_heterogeneity_mild_matched",
        "feature_heterogeneity_moderate_matched",
        "feature_heterogeneity_severe_matched",
    ]:
        _write_json(
            tmp_path / scenario / "seed_11" / "heterogeneity.json",
            {"feature_jaccard_dissimilarity": 0.0},
        )
    result = validate_heterogeneity_matched_controls(tmp_path)
    assert result["passed"] is True
    assert result["covariate_matched_control_passed"] is True
    assert result["feature_matched_control_passed"] is True


def test_validate_heterogeneity_matched_controls_fails_on_nonzero_dispersion(
    tmp_path: Path,
) -> None:
    _write_covariate_params(
        tmp_path / "covariate_shift_mild_matched" / "seed_11" / "covariate_shift_parameters.csv",
        [
            {"client_id": "c0", "feature": "f0", "location_shift": 0.1, "scale_multiplier": 1.2},
            {"client_id": "c1", "feature": "f0", "location_shift": 0.2, "scale_multiplier": 1.2},
        ],
    )
    _write_json(
        tmp_path / "feature_heterogeneity_mild_matched" / "seed_11" / "heterogeneity.json",
        {"feature_jaccard_dissimilarity": 0.05},
    )
    result = validate_heterogeneity_matched_controls(tmp_path)
    assert result["passed"] is False
    assert result["covariate_matched_control_passed"] is False
    assert result["feature_matched_control_passed"] is False


def test_validate_heterogeneity_matched_controls_fails_when_nothing_found(
    tmp_path: Path,
) -> None:
    result = validate_heterogeneity_matched_controls(tmp_path)
    assert result["passed"] is False

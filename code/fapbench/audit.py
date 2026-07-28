from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_partition_grid(root: Path) -> dict[str, Any]:
    reports = []
    for path in sorted(root.glob("*/seed_*/partition_common_universe.json")):
        payload = _read_json(path)
        reports.append({
            "run": str(path.parent.relative_to(root)),
            "rows": payload.get("n_rows_after"),
            "fingerprint": payload.get("row_fingerprint_sha256"),
        })
    fingerprints = {row["fingerprint"] for row in reports}
    row_counts = {row["rows"] for row in reports}
    size_counts = [
        int(_read_json(path).get("n_clients", -1))
        for path in sorted(root.glob("common_partition_size_quintile/seed_*/heterogeneity.json"))
    ]
    passed = bool(reports) and len(fingerprints) == 1 and len(row_counts) == 1
    if size_counts:
        passed = passed and set(size_counts) == {5}
    return {
        "passed": passed,
        "n_runs_checked": len(reports),
        "unique_fingerprints": sorted(x for x in fingerprints if x),
        "unique_row_counts": sorted(x for x in row_counts if x is not None),
        "size_partition_client_counts": size_counts,
        "runs": reports,
    }


def _round_counts(path: Path) -> list[int]:
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return frame.groupby("round")["client_id"].nunique().sort_index().astype(int).tolist()


def _client_counts(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.groupby("client_id")["round"].nunique().astype(int).sort_index().to_dict()


def validate_identification_grid(root: Path) -> dict[str, Any]:
    temporal_pairs = [
        ("temporal_persistent_strong_identified", "temporal_random_matched_persistent_strong"),
        ("temporal_block_outages_identified", "temporal_random_matched_block_outages"),
    ]
    seed_dirs = sorted({p.name for p in root.glob("*/seed_*")})
    temporal_checks = []
    for structured, randomized in temporal_pairs:
        for seed_dir in seed_dirs:
            left = root / structured / seed_dir / "participation_schedule.csv"
            right = root / randomized / seed_dir / "participation_schedule.csv"
            if not left.exists() or not right.exists():
                continue
            temporal_checks.append({
                "structured": structured,
                "randomized": randomized,
                "seed": seed_dir,
                "per_round_counts_equal": _round_counts(left) == _round_counts(right),
                "per_client_totals_equal": _client_counts(left) == _client_counts(right),
            })

    size_checks = []
    for seed_dir in seed_dirs:
        left = root / "size_imbalance_ratio_20_identified" / seed_dir / "run_metadata.json"
        right = root / "size_uniform_matched_ratio_20" / seed_dir / "run_metadata.json"
        if not left.exists() or not right.exists():
            continue
        left_rows = int(_read_json(left).get("train_rows", -1))
        right_rows = int(_read_json(right).get("train_rows", -1))
        rel = abs(left_rows - right_rows) / left_rows if left_rows > 0 else None
        size_checks.append({
            "ratio": 20,
            "seed": seed_dir,
            "imbalanced_rows": left_rows,
            "uniform_control_rows": right_rows,
            "relative_row_difference": rel,
        })

    temporal_passed = bool(temporal_checks) and all(
        x["per_round_counts_equal"] and x["per_client_totals_equal"]
        for x in temporal_checks
    )
    size_passed = bool(size_checks) and all(
        x["relative_row_difference"] is not None and x["relative_row_difference"] <= 0.02
        for x in size_checks
    )
    return {
        "passed": temporal_passed and size_passed,
        "temporal_matched_counts_passed": temporal_passed,
        "size_matched_rows_passed_at_2pct": size_passed,
        "temporal_checks": temporal_checks,
        "size_checks": size_checks,
    }


def validate_heterogeneity_matched_controls(root: Path) -> dict[str, Any]:
    """Verify the covariate-shift and feature-heterogeneity matched controls
    actually zero cross-client dispersion, mirroring validate_identification_grid's
    role for the size/temporal matched controls.

    Unlike those (a tolerance-based row-count match, since retention sampling
    only hits its target fraction approximately), both checks here are exact:
    the matched-control builders collapse the treatment's realized draws to a
    single shared value applied identically to every client, so there should
    be zero cross-client variation at all.
    """
    covariate_pairs = [
        "covariate_shift_mild_matched",
        "covariate_shift_strong_matched",
    ]
    feature_pairs = [
        "feature_heterogeneity_mild_matched",
        "feature_heterogeneity_moderate_matched",
        "feature_heterogeneity_severe_matched",
    ]

    covariate_checks = []
    for scenario in covariate_pairs:
        for path in sorted(root.glob(f"{scenario}/seed_*/covariate_shift_parameters.csv")):
            frame = pd.read_csv(path)
            unique_counts = frame.groupby("feature")[
                ["location_shift", "scale_multiplier"]
            ].nunique()
            covariate_checks.append({
                "scenario": scenario,
                "seed": path.parent.name,
                "exactly_one_value_per_feature": bool((unique_counts == 1).all().all()),
            })

    feature_checks = []
    for scenario in feature_pairs:
        for path in sorted(root.glob(f"{scenario}/seed_*/heterogeneity.json")):
            dissimilarity = _read_json(path).get("feature_jaccard_dissimilarity")
            feature_checks.append({
                "scenario": scenario,
                "seed": path.parent.name,
                "feature_jaccard_dissimilarity": dissimilarity,
                "zero_dissimilarity": dissimilarity == 0.0,
            })

    covariate_passed = bool(covariate_checks) and all(
        x["exactly_one_value_per_feature"] for x in covariate_checks
    )
    feature_passed = bool(feature_checks) and all(
        x["zero_dissimilarity"] for x in feature_checks
    )
    return {
        "passed": covariate_passed and feature_passed,
        "covariate_matched_control_passed": covariate_passed,
        "feature_matched_control_passed": feature_passed,
        "covariate_checks": covariate_checks,
        "feature_checks": feature_checks,
    }

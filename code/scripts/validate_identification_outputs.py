#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        reports.append(
            {
                "run": str(path.parent.relative_to(root)),
                "rows": payload.get("n_rows_after"),
                "fingerprint": payload.get("row_fingerprint_sha256"),
            }
        )
    fingerprints = {row["fingerprint"] for row in reports}
    row_counts = {row["rows"] for row in reports}
    size_client_counts = []
    for path in sorted(root.glob("common_partition_size_quintile/seed_*/heterogeneity.json")):
        size_client_counts.append(int(_read_json(path).get("n_clients", -1)))
    passed = bool(reports) and len(fingerprints) == 1 and len(row_counts) == 1
    if size_client_counts:
        passed = passed and set(size_client_counts) == {5}
    return {
        "passed": passed,
        "n_runs_checked": len(reports),
        "unique_fingerprints": sorted(value for value in fingerprints if value),
        "unique_row_counts": sorted(value for value in row_counts if value is not None),
        "size_partition_client_counts": size_client_counts,
        "runs": reports,
    }


def _round_counts(path: Path) -> list[int]:
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return (
        frame.groupby("round", observed=True)["client_id"]
        .nunique()
        .sort_index()
        .astype(int)
        .tolist()
    )




def _client_counts(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return (
        frame.groupby("client_id", observed=True)["round"]
        .nunique()
        .astype(int)
        .sort_index()
        .to_dict()
    )

def validate_identification_grid(root: Path) -> dict[str, Any]:
    temporal_checks = []
    pairs = [
        (
            "temporal_persistent_mild_identified",
            "temporal_random_matched_persistent_mild",
        ),
        (
            "temporal_persistent_strong_identified",
            "temporal_random_matched_persistent_strong",
        ),
        (
            "temporal_block_outages_identified",
            "temporal_random_matched_block_outages",
        ),
    ]
    seeds = sorted(
        path.name for path in (root / "identification_reference").glob("seed_*")
    )
    for structured, randomized in pairs:
        for seed_dir in seeds:
            left = root / structured / seed_dir / "participation_schedule.csv"
            right = root / randomized / seed_dir / "participation_schedule.csv"
            if not left.exists() or not right.exists():
                continue
            left_counts = _round_counts(left)
            right_counts = _round_counts(right)
            left_client_counts = _client_counts(left)
            right_client_counts = _client_counts(right)
            temporal_checks.append(
                {
                    "structured": structured,
                    "randomized": randomized,
                    "seed": seed_dir,
                    "per_round_counts_equal": left_counts == right_counts,
                    "per_client_totals_equal": (
                        left_client_counts == right_client_counts
                    ),
                    "active_slots_structured": int(sum(left_counts)),
                    "active_slots_randomized": int(sum(right_counts)),
                }
            )

    size_checks = []
    for ratio in [5, 20]:
        imbalanced = f"size_imbalance_ratio_{ratio}_identified"
        controlled = f"size_uniform_matched_ratio_{ratio}"
        for seed_dir in seeds:
            left = root / imbalanced / seed_dir / "run_metadata.json"
            right = root / controlled / seed_dir / "run_metadata.json"
            if not left.exists() or not right.exists():
                continue
            left_rows = int(_read_json(left).get("train_rows", -1))
            right_rows = int(_read_json(right).get("train_rows", -1))
            size_checks.append(
                {
                    "ratio": ratio,
                    "seed": seed_dir,
                    "imbalanced_rows": left_rows,
                    "uniform_control_rows": right_rows,
                    "absolute_row_difference": abs(left_rows - right_rows),
                    "relative_row_difference": (
                        abs(left_rows - right_rows) / left_rows if left_rows > 0 else None
                    ),
                }
            )

    temporal_passed = bool(temporal_checks) and all(
        row["per_round_counts_equal"] and row["per_client_totals_equal"]
        for row in temporal_checks
    )
    size_passed = bool(size_checks) and all(
        row["relative_row_difference"] is not None
        and row["relative_row_difference"] <= 0.02
        for row in size_checks
    )
    return {
        "passed": temporal_passed and size_passed,
        "temporal_matched_counts_passed": temporal_passed,
        "size_matched_rows_passed_at_2pct": size_passed,
        "temporal_checks": temporal_checks,
        "size_checks": size_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-grid", type=Path)
    parser.add_argument("--identification-grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {}
    if args.partition_grid is not None:
        report["partition_common_universe"] = validate_partition_grid(
            args.partition_grid
        )
    if args.identification_grid is not None:
        report["matched_controls"] = validate_identification_grid(
            args.identification_grid
        )
    report["passed"] = bool(report) and all(
        section.get("passed", False) for section in report.values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

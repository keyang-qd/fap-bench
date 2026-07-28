#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fapbench.audit import (
    validate_heterogeneity_matched_controls,
    validate_identification_grid,
    validate_partition_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-grid", type=Path)
    parser.add_argument("--identification-grid", type=Path)
    parser.add_argument("--heterogeneity-grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    if args.partition_grid is not None:
        report["partition_common_universe"] = validate_partition_grid(args.partition_grid)
    if args.identification_grid is not None:
        report["matched_controls"] = validate_identification_grid(args.identification_grid)
    if args.heterogeneity_grid is not None:
        report["heterogeneity_matched_controls"] = validate_heterogeneity_matched_controls(
            args.heterogeneity_grid
        )
    report["passed"] = bool(report) and all(
        section.get("passed", False)
        for name, section in report.items()
        if name != "passed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fapbench.sanity import run_sanity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify IID assignment and one-step FedAvg aggregation."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entities", type=int, default=120)
    parser.add_argument("--clients", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, default=Path("outputs/iid_sanity.json"))
    args = parser.parse_args()
    report = run_sanity(args.seed, args.entities, args.clients, args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

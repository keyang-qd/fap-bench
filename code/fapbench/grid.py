from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fapbench.config import deep_update, load_config
from fapbench.runner import run
from fapbench.utils import ensure_dir, write_json


def run_grid(grid_config: dict[str, Any]) -> Path:
    grid_path = Path(grid_config["_config_path"])
    base_path = Path(grid_config["base_config"])
    if not base_path.is_absolute():
        base_path = (grid_path.parent / base_path).resolve()
    base = load_config(base_path)

    grid_output = ensure_dir(grid_config.get("output_dir", "outputs/grid"))
    seeds = [int(seed) for seed in grid_config.get("seeds", [base.get("seed", 42)])]
    scenarios = grid_config.get("scenarios", [])
    if not scenarios:
        raise ValueError("The grid config must contain at least one scenario.")

    combined_rows: list[pd.DataFrame] = []
    run_manifest: list[dict[str, Any]] = []
    for scenario in scenarios:
        name = str(scenario["name"])
        overrides = scenario.get("overrides", {})
        for seed in seeds:
            config = deep_update(base, overrides)
            config["seed"] = seed
            run_dir = grid_output / name / f"seed_{seed}"
            config["output_dir"] = str(run_dir)
            output = run(config)

            results = pd.read_csv(output / "benchmark_results.csv")
            results.insert(0, "seed", seed)
            results.insert(0, "scenario", name)
            heterogeneity = pd.read_json(
                output / "heterogeneity.json", typ="series"
            ).to_dict()
            for key, value in heterogeneity.items():
                if isinstance(value, (int, float)):
                    results[f"heterogeneity_{key}"] = value
            combined_rows.append(results)
            run_manifest.append(
                {"scenario": name, "seed": seed, "output_dir": str(output.resolve())}
            )

    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(grid_output / "grid_results.csv", index=False)

    numeric = combined.select_dtypes(include="number").columns.tolist()
    summary = (
        combined.groupby(["scenario", "method"], as_index=False)[numeric]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(grid_output / "grid_summary.csv", index=False)
    write_json(run_manifest, grid_output / "run_manifest.json")
    return grid_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple FAP-Bench scenarios and seeds.")
    parser.add_argument("--grid", required=True, help="Path to grid YAML.")
    args = parser.parse_args()
    output = run_grid(load_config(args.grid))
    print(f"Grid results written to {output.resolve()}")


if __name__ == "__main__":
    main()

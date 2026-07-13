from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fapbench.config import load_config
from fapbench.utils import configure_logger, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repair a completed FAP-Bench run that failed only while saving "
            "resolved_config.json. No model is retrained."
        )
    )
    parser.add_argument("--config", required=True, help="Original YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run output directory. Defaults to output_dir in the YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(
        output_dir,
        name=f"fapbench.repair.{abs(hash(str(output_dir.resolve())))}",
        filename="repair.log",
        console=True,
        overwrite=True,
    )

    results_path = output_dir / "benchmark_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} does not exist; the run did not reach summary output."
        )

    results = pd.read_csv(results_path)
    if results.empty or "method" not in results.columns:
        raise ValueError(f"{results_path} is empty or missing the method column.")

    missing_files: list[str] = []
    for method in results["method"].astype(str):
        for filename in [
            f"metrics_{method}.json",
            f"predictions_{method}.parquet",
        ]:
            if not (output_dir / filename).exists():
                missing_files.append(filename)

    write_json(config, output_dir / "resolved_config.json")
    recovery = {
        "status": "recovered_without_retraining",
        "methods": results["method"].astype(str).tolist(),
        "benchmark_rows": int(len(results)),
        "missing_expected_files": missing_files,
    }
    write_json(recovery, output_dir / "recovery_status.json")

    logger.info(
        "Recovered completed run | output=%s | methods=%s | missing_files=%s",
        output_dir.resolve(),
        ", ".join(recovery["methods"]),
        len(missing_files),
    )
    if missing_files:
        logger.warning("Missing expected files | %s", ", ".join(missing_files))
    else:
        logger.info("All expected metrics and prediction files are present.")


if __name__ == "__main__":
    main()

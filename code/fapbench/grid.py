from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fapbench.config import deep_update, load_config
from fapbench.data import load_panel
from fapbench.runner import resolve_data_path, run
from fapbench.utils import (
    configure_logger,
    ensure_dir,
    format_seconds,
    frame_memory_mb,
    log_stage,
    write_json,
)


def _completed_run_files(output_dir: Path) -> bool:
    """Return True only when the files needed for grid aggregation exist."""
    results_path = output_dir / "benchmark_results.csv"
    heterogeneity_path = output_dir / "heterogeneity.json"
    if not results_path.exists() or not heterogeneity_path.exists():
        return False
    try:
        results = pd.read_csv(results_path, nrows=1)
        if results.empty:
            return False
        with heterogeneity_path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except Exception:
        return False
    return True


def _collect_run_results(
    output: Path,
    *,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    results = pd.read_csv(output / "benchmark_results.csv")
    results.insert(0, "seed", seed)
    results.insert(0, "scenario", scenario)
    with (output / "heterogeneity.json").open("r", encoding="utf-8") as handle:
        heterogeneity = json.load(handle)
    for key, value in heterogeneity.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            results[f"heterogeneity_{key}"] = value
    return results


def run_grid(grid_config: dict[str, Any]) -> Path:
    grid_path = Path(grid_config["_config_path"])
    base_path = Path(grid_config["base_config"])
    if not base_path.is_absolute():
        base_path = (grid_path.parent / base_path).resolve()
    base = load_config(base_path)

    grid_output = ensure_dir(grid_config.get("output_dir", "outputs/grid"))
    logging_cfg = grid_config.get("logging", {})
    logger = configure_logger(
        grid_output,
        name=f"fapbench.grid.{abs(hash(str(grid_output.resolve())))}",
        level=logging_cfg.get("level", "INFO"),
        filename=logging_cfg.get("file", "grid.log"),
        console=bool(logging_cfg.get("console", True)),
        overwrite=bool(logging_cfg.get("overwrite", False)),
    )
    seeds = [int(seed) for seed in grid_config.get("seeds", [base.get("seed", 42)])]
    scenarios = grid_config.get("scenarios", [])
    if not scenarios:
        raise ValueError("The grid config must contain at least one scenario.")

    total_runs = len(scenarios) * len(seeds)
    continue_on_error = bool(grid_config.get("continue_on_error", True))
    resume_completed = bool(grid_config.get("resume_completed", True))
    preload_panel_once = bool(grid_config.get("preload_panel_once", True))

    logger.info(
        "Grid start | scenarios=%s | seeds=%s | total_runs=%s | output=%s",
        len(scenarios),
        seeds,
        total_runs,
        grid_output.resolve(),
    )
    logger.info(
        "Grid execution | resume_completed=%s | preload_panel_once=%s | "
        "runs_are_sequential=true | client_parallelism_inherited_from_base=%s",
        resume_completed,
        preload_panel_once,
        base.get("performance", {}).get("parallel_clients", 1),
    )

    # Grid runs are intentionally sequential. Each run already uses a reusable
    # multi-process client pool; parallelizing whole runs as well would create
    # nested process pools and severe CPU/RAM oversubscription on a laptop.
    raw_panel: pd.DataFrame | None = None
    if preload_panel_once:
        data_path = resolve_data_path(base)
        with log_stage(logger, "Preload source panel once for all grid runs"):
            raw_panel = load_panel(data_path)
            logger.info(
                "Grid panel cache | path=%s | rows=%s | cols=%s | memory=%.1f MB",
                data_path,
                f"{len(raw_panel):,}",
                raw_panel.shape[1],
                frame_memory_mb(raw_panel),
            )

    started = time.perf_counter()
    combined_rows: list[pd.DataFrame] = []
    run_manifest: list[dict[str, Any]] = []
    run_index = 0
    completed_count = 0
    reused_count = 0

    try:
        for scenario in scenarios:
            name = str(scenario["name"])
            overrides = scenario.get("overrides", {})
            logger.info("Scenario start | name=%s | seeds=%s", name, seeds)
            for seed in seeds:
                run_index += 1
                config = deep_update(base, overrides)
                config["seed"] = seed
                run_dir = grid_output / name / f"seed_{seed}"
                config["output_dir"] = str(run_dir)
                config["logging"] = deep_update(
                    config.get("logging", {}),
                    grid_config.get("child_logging", {}),
                )
                config["logging"].setdefault("file", "run.log")
                config["logging"].setdefault("console", True)

                run_started = time.perf_counter()
                logger.info(
                    "Grid run=%s/%s | scenario=%s | seed=%s | start",
                    run_index,
                    total_runs,
                    name,
                    seed,
                )
                try:
                    if resume_completed and _completed_run_files(run_dir):
                        output = run_dir
                        status = "reused_completed"
                        reused_count += 1
                        logger.info(
                            "Grid run=%s/%s | scenario=%s | seed=%s | "
                            "resume hit; training skipped",
                            run_index,
                            total_runs,
                            name,
                            seed,
                        )
                    else:
                        output = run(
                            config,
                            preloaded_raw_panel=raw_panel,
                        )
                        status = "completed"
                        completed_count += 1

                    results = _collect_run_results(
                        output,
                        scenario=name,
                        seed=seed,
                    )
                    combined_rows.append(results)
                    run_manifest.append(
                        {
                            "scenario": name,
                            "seed": seed,
                            "status": status,
                            "output_dir": str(output.resolve()),
                            "elapsed_seconds": float(
                                time.perf_counter() - run_started
                            ),
                        }
                    )
                    write_json(run_manifest, grid_output / "run_manifest.json")
                    logger.info(
                        "Grid run=%s/%s | scenario=%s | seed=%s | done | "
                        "status=%s | methods=%s | elapsed=%s",
                        run_index,
                        total_runs,
                        name,
                        seed,
                        status,
                        len(results),
                        format_seconds(time.perf_counter() - run_started),
                    )
                except Exception as exc:
                    run_manifest.append(
                        {
                            "scenario": name,
                            "seed": seed,
                            "status": "failed",
                            "output_dir": str(run_dir.resolve()),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "elapsed_seconds": float(
                                time.perf_counter() - run_started
                            ),
                        }
                    )
                    write_json(run_manifest, grid_output / "run_manifest.json")
                    logger.exception(
                        "Grid run=%s/%s | scenario=%s | seed=%s | failed",
                        run_index,
                        total_runs,
                        name,
                        seed,
                    )
                    if not continue_on_error:
                        raise
                finally:
                    # Encourage prompt release of model/data copies between long runs.
                    gc.collect()

        with log_stage(logger, "Aggregate grid outputs"):
            write_json(run_manifest, grid_output / "run_manifest.json")
            if not combined_rows:
                raise RuntimeError("Every grid run failed; no results can be aggregated.")
            combined = pd.concat(combined_rows, ignore_index=True)
            combined.to_csv(grid_output / "grid_results.csv", index=False)

            numeric = [
                column
                for column in combined.select_dtypes(include="number").columns
                if column != "seed"
            ]
            summary = (
                combined.groupby(["scenario", "method"], as_index=False)[numeric]
                .agg(["mean", "std", "count"])
            )
            summary.columns = [
                "_".join(str(part) for part in column if str(part))
                if isinstance(column, tuple)
                else str(column)
                for column in summary.columns
            ]
            summary.to_csv(grid_output / "grid_summary.csv", index=False)

        logger.info(
            "Grid complete | total_runs=%s | newly_completed=%s | reused=%s | "
            "rows=%s | elapsed=%s",
            total_runs,
            completed_count,
            reused_count,
            len(combined),
            format_seconds(time.perf_counter() - started),
        )
        return grid_output
    except Exception:
        logger.exception(
            "Grid failed | processed_runs=%s/%s | elapsed=%s",
            run_index,
            total_runs,
            format_seconds(time.perf_counter() - started),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple FAP-Bench scenarios and seeds.")
    parser.add_argument("--grid", required=True, help="Path to grid YAML.")
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()
    config = load_config(args.grid)
    if args.log_level or args.log_file:
        config.setdefault("logging", {})
        if args.log_level:
            config["logging"]["level"] = args.log_level
        if args.log_file:
            config["logging"]["file"] = args.log_file
    output = run_grid(config)
    print(f"Grid results written to {output.resolve()}")


if __name__ == "__main__":
    main()

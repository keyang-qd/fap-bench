from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
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
    """Collect one run without repeatedly inserting columns.

    Building all metadata columns in one DataFrame and concatenating once avoids
    pandas block fragmentation, which previously produced dozens of
    PerformanceWarning messages during every grid run.
    """
    results = pd.read_csv(output / "benchmark_results.csv").reset_index(drop=True)
    with (output / "heterogeneity.json").open("r", encoding="utf-8") as handle:
        heterogeneity = json.load(handle)

    metadata: dict[str, Any] = {
        "scenario": scenario,
        "seed": int(seed),
    }
    metadata.update(
        {
            f"heterogeneity_{key}": value
            for key, value in heterogeneity.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    metadata_frame = pd.DataFrame(
        {key: [value] * len(results) for key, value in metadata.items()}
    )
    return pd.concat([metadata_frame, results], axis=1, copy=False).copy()


def _scenario_workload(base: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Return a transparent, approximate compute budget for one seed.

    A full-client-pass equivalent is one local epoch over all training clients.
    This is not a wall-clock prediction, but it makes smoke-test and full-grid
    configurations directly comparable before expensive training starts.
    """
    config = deep_update(base, scenario.get("overrides", {}))
    fed = config.get("federated", {})
    baselines = config.get("baselines", {})
    algorithms = list(fed.get("algorithms", []))
    rounds = int(fed.get("rounds", 0))
    local_epochs = int(fed.get("local_epochs", 1))
    participation = float(fed.get("participation_rate", 1.0))
    fl_passes = len(algorithms) * rounds * local_epochs * participation

    if bool(baselines.get("match_federated_local_passes", False)):
        baseline_epochs = max(1, int(np.ceil(rounds * local_epochs * participation)))
    else:
        configured = baselines.get("epochs", 5)
        baseline_epochs = (
            max(1, int(np.ceil(rounds * local_epochs * participation)))
            if configured is None
            else int(configured)
        )
    baseline_methods = int(bool(baselines.get("centralized", True)))
    baseline_methods += int(bool(baselines.get("local", True)))
    # centralized_masked is relevant only in feature-heterogeneity scenarios.
    feature_enabled = bool(config.get("feature_heterogeneity", {}).get("enabled", False))
    if feature_enabled:
        baseline_methods += int(bool(baselines.get("centralized_masked", True)))
    baseline_passes = baseline_methods * baseline_epochs
    return {
        "name": str(scenario.get("name", "unnamed")),
        "algorithms": algorithms,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "participation_rate": participation,
        "fl_full_client_passes": float(fl_passes),
        "baseline_methods": baseline_methods,
        "baseline_epochs": baseline_epochs,
        "baseline_full_data_passes": float(baseline_passes),
        "total_pass_equivalents": float(fl_passes + baseline_passes),
    }


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

    workload_rows = [_scenario_workload(base, scenario) for scenario in scenarios]
    total_pass_equivalents = sum(
        row["total_pass_equivalents"] for row in workload_rows
    ) * len(seeds)
    logger.info(
        "Grid workload estimate | full_client_pass_equivalents=%.1f | "
        "note=relative_compute_budget_not_wall_clock",
        total_pass_equivalents,
    )
    for row in workload_rows:
        logger.info(
            "Scenario budget | name=%s | algorithms=%s | rounds=%s | "
            "local_epochs=%s | participation=%.3f | fl_passes=%.1f | "
            "baseline_methods=%s | baseline_epochs=%s | total_passes=%.1f",
            row["name"],
            ",".join(row["algorithms"]) or "none",
            row["rounds"],
            row["local_epochs"],
            row["participation_rate"],
            row["fl_full_client_passes"],
            row["baseline_methods"],
            row["baseline_epochs"],
            row["total_pass_equivalents"],
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
    completed_durations: list[float] = []

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
                    run_elapsed = time.perf_counter() - run_started
                    if status == "completed":
                        completed_durations.append(run_elapsed)
                    remaining_runs = total_runs - run_index
                    average_seconds = (
                        float(np.mean(completed_durations))
                        if completed_durations
                        else run_elapsed
                    )
                    logger.info(
                        "Grid run=%s/%s | scenario=%s | seed=%s | done | "
                        "status=%s | methods=%s | elapsed=%s | "
                        "rough_remaining=%s",
                        run_index,
                        total_runs,
                        name,
                        seed,
                        status,
                        len(results),
                        format_seconds(run_elapsed),
                        format_seconds(average_seconds * remaining_runs),
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
            combined = pd.concat(combined_rows, ignore_index=True).copy()
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

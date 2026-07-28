from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
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
    close_logger,
    configure_logger,
    ensure_dir,
    format_seconds,
    frame_memory_mb,
    json_default,
    log_stage,
    write_json,
)


def _resume_relevant_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical experiment config for resumability checks.

    ``yaml.safe_load`` represents unquoted dates as :class:`datetime.date`, while
    ``resolved_config.json`` necessarily stores those values as ISO strings. A
    direct dictionary comparison therefore rejected every completed run even when
    its configuration was unchanged. Canonicalizing through JSON applies the same
    conversions used by :func:`write_json` before comparing the configurations.

    Output locations and logging verbosity do not change an experiment. Everything
    else, including data path, partition, methods, seeds, and FedAdam server LR,
    must match before a completed directory may be reused.
    """
    ignored = {"_config_path", "output_dir", "logging"}
    relevant = {key: value for key, value in config.items() if key not in ignored}
    return json.loads(
        json.dumps(
            relevant,
            default=json_default,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _completed_run_files(
    output_dir: Path, expected_config: dict[str, Any] | None = None
) -> bool:
    """Return True only when complete outputs match the requested configuration."""
    results_path = output_dir / "benchmark_results.csv"
    heterogeneity_path = output_dir / "heterogeneity.json"
    resolved_config_path = output_dir / "resolved_config.json"
    if not results_path.exists() or not heterogeneity_path.exists():
        return False
    try:
        results = pd.read_csv(results_path, nrows=1)
        if results.empty:
            return False
        with heterogeneity_path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        if expected_config is not None:
            if not resolved_config_path.exists():
                return False
            with resolved_config_path.open("r", encoding="utf-8") as handle:
                completed_config = json.load(handle)
            if _resume_relevant_config(completed_config) != _resume_relevant_config(
                expected_config
            ):
                return False
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
    if bool(config.get("diagnostics_only", False)):
        return {
            "name": str(scenario.get("name", "unnamed")),
            "algorithms": [],
            "rounds": 0,
            "local_epochs": 0,
            "participation_rate": 0.0,
            "fl_full_client_passes": 0.0,
            "baseline_methods": 0,
            "baseline_epochs": 0,
            "baseline_full_data_passes": 0.0,
            "total_pass_equivalents": 0.0,
        }
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


def _run_grid_worker(config: dict[str, Any]) -> str:
    """Top-level worker entry point for spawn-based parallel grid execution."""
    return str(run(config, preloaded_raw_panel=None).resolve())


def run_grid(grid_config: dict[str, Any]) -> Path:
    grid_path = Path(grid_config["_config_path"])
    base_path = Path(grid_config["base_config"])
    if not base_path.is_absolute():
        base_path = (grid_path.parent / base_path).resolve()
    base = load_config(base_path)
    base = deep_update(base, grid_config.get("base_overrides", {}))

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
    execution_cfg = grid_config.get("grid_execution", {})
    parallel_runs = max(1, int(execution_cfg.get("parallel_runs", 1)))
    process_start_method = str(execution_cfg.get("start_method", "spawn"))
    base_client_workers = int(base.get("performance", {}).get("parallel_clients", 1))
    max_total_client_workers = max(
        1,
        int(execution_cfg.get("max_total_client_workers", base_client_workers)),
    )
    per_run_client_workers = max(
        1,
        int(
            execution_cfg.get(
                "parallel_clients_per_run",
                max_total_client_workers // parallel_runs,
            )
        ),
    )
    max_total_torch_threads = max(
        1,
        int(
            execution_cfg.get(
                "max_total_torch_threads",
                max(1, int(base.get("torch_num_threads", 1))) * parallel_runs,
            )
        ),
    )
    per_run_torch_threads = max(
        1,
        int(
            execution_cfg.get(
                "torch_threads_per_run",
                max_total_torch_threads // parallel_runs,
            )
        ),
    )

    if parallel_runs > 1 and preload_panel_once:
        logger.warning(
            "grid_execution.parallel_runs=%s disables preload_panel_once because "
            "the full panel is not copied between spawned run processes.",
            parallel_runs,
        )
        preload_panel_once = False

    logger.info(
        "Grid start | scenarios=%s | seeds=%s | total_runs=%s | output=%s",
        len(scenarios),
        seeds,
        total_runs,
        grid_output.resolve(),
    )
    logger.info(
        "Grid execution | resume_completed=%s | preload_panel_once=%s | "
        "parallel_runs=%s | parallel_clients_per_run=%s | torch_threads_per_run=%s | "
        "max_total_client_workers=%s",
        resume_completed,
        preload_panel_once,
        parallel_runs,
        per_run_client_workers,
        per_run_torch_threads,
        max_total_client_workers,
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

    cells: list[dict[str, Any]] = []
    cell_index = 0
    for scenario in scenarios:
        name = str(scenario["name"])
        overrides = scenario.get("overrides", {})
        for seed in seeds:
            cell_index += 1
            config = deep_update(base, overrides)
            config["seed"] = int(seed)
            run_dir = grid_output / name / f"seed_{seed}"
            config["output_dir"] = str(run_dir)
            config["logging"] = deep_update(
                config.get("logging", {}), grid_config.get("child_logging", {})
            )
            config["logging"].setdefault("file", "run.log")
            config["logging"].setdefault("console", True)
            config.setdefault("performance", {})
            config["performance"]["parallel_clients"] = per_run_client_workers
            config["torch_num_threads"] = per_run_torch_threads
            if parallel_runs > 1 and not bool(
                execution_cfg.get("parallel_child_console", False)
            ):
                config["logging"]["console"] = False
            cells.append(
                {
                    "index": cell_index,
                    "scenario": name,
                    "seed": int(seed),
                    "run_dir": run_dir,
                    "config": config,
                }
            )

    started = time.perf_counter()
    combined_rows: list[pd.DataFrame] = []
    run_manifest: list[dict[str, Any]] = []
    completed_count = 0
    reused_count = 0
    completed_durations: list[float] = []

    def record_success(
        cell: dict[str, Any], output: Path, status: str, elapsed: float
    ) -> None:
        nonlocal completed_count, reused_count
        if status == "completed":
            completed_count += 1
            completed_durations.append(float(elapsed))
        else:
            reused_count += 1
        results = _collect_run_results(
            output, scenario=cell["scenario"], seed=cell["seed"]
        )
        combined_rows.append(results)
        run_manifest.append(
            {
                "scenario": cell["scenario"],
                "seed": cell["seed"],
                "status": status,
                "output_dir": str(output.resolve()),
                "elapsed_seconds": float(elapsed),
                "parallel_runs": int(parallel_runs),
                "parallel_clients_per_run": int(per_run_client_workers),
            }
        )
        write_json(run_manifest, grid_output / "run_manifest.json")
        logger.info(
            "Grid run=%s/%s | scenario=%s | seed=%s | done | status=%s | "
            "methods=%s | elapsed=%s",
            cell["index"],
            total_runs,
            cell["scenario"],
            cell["seed"],
            status,
            len(results),
            format_seconds(elapsed),
        )

    def record_failure(cell: dict[str, Any], exc: BaseException, elapsed: float) -> None:
        run_manifest.append(
            {
                "scenario": cell["scenario"],
                "seed": cell["seed"],
                "status": "failed",
                "output_dir": str(cell["run_dir"].resolve()),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "elapsed_seconds": float(elapsed),
                "parallel_runs": int(parallel_runs),
                "parallel_clients_per_run": int(per_run_client_workers),
            }
        )
        write_json(run_manifest, grid_output / "run_manifest.json")
        logger.error(
            "Grid run=%s/%s | scenario=%s | seed=%s | failed | %s: %s",
            cell["index"],
            total_runs,
            cell["scenario"],
            cell["seed"],
            type(exc).__name__,
            exc,
        )

    try:
        pending: list[dict[str, Any]] = []
        for cell in cells:
            if resume_completed and _completed_run_files(
                cell["run_dir"], expected_config=cell["config"]
            ):
                record_success(cell, cell["run_dir"], "reused_completed", 0.0)
            else:
                pending.append(cell)

        if parallel_runs == 1:
            for cell in pending:
                run_started = time.perf_counter()
                logger.info(
                    "Grid run=%s/%s | scenario=%s | seed=%s | start",
                    cell["index"], total_runs, cell["scenario"], cell["seed"]
                )
                try:
                    output = run(
                        cell["config"], preloaded_raw_panel=raw_panel
                    )
                    record_success(
                        cell,
                        output,
                        "completed",
                        time.perf_counter() - run_started,
                    )
                except Exception as exc:
                    record_failure(cell, exc, time.perf_counter() - run_started)
                    if not continue_on_error:
                        raise
                finally:
                    gc.collect()
        else:
            logger.info(
                "Parallel grid launch | pending_runs=%s | workers=%s | start_method=%s",
                len(pending), parallel_runs, process_start_method
            )
            context = get_context(process_start_method)
            with ProcessPoolExecutor(
                max_workers=parallel_runs, mp_context=context
            ) as pool:
                futures = {}
                start_times: dict[Any, float] = {}
                for cell in pending:
                    logger.info(
                        "Grid run=%s/%s | scenario=%s | seed=%s | submitted",
                        cell["index"], total_runs, cell["scenario"], cell["seed"]
                    )
                    future = pool.submit(_run_grid_worker, cell["config"])
                    futures[future] = cell
                    start_times[future] = time.perf_counter()
                for future in as_completed(futures):
                    cell = futures[future]
                    elapsed = time.perf_counter() - start_times[future]
                    try:
                        output = Path(future.result())
                        record_success(cell, output, "completed", elapsed)
                    except Exception as exc:
                        record_failure(cell, exc, elapsed)
                        if not continue_on_error:
                            for pending_future in futures:
                                pending_future.cancel()
                            raise

        with log_stage(logger, "Aggregate grid outputs"):
            run_manifest.sort(
                key=lambda row: (str(row.get("scenario", "")), int(row.get("seed", 0)))
            )
            write_json(run_manifest, grid_output / "run_manifest.json")
            if not combined_rows:
                raise RuntimeError("Every grid run failed; no results can be aggregated.")
            combined = pd.concat(combined_rows, ignore_index=True).copy()
            combined = combined.sort_values(
                ["scenario", "seed", "method"], kind="stable"
            ).reset_index(drop=True)
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
            "Grid failed | manifest_rows=%s/%s | elapsed=%s",
            len(run_manifest),
            total_runs,
            format_seconds(time.perf_counter() - started),
        )
        raise
    finally:
        close_logger(logger)


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

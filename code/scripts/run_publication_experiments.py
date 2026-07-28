#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fapbench.audit import (
    validate_heterogeneity_matched_controls,
    validate_identification_grid,
    validate_partition_grid,
)
from fapbench.config import deep_update, load_config
from fapbench.grid import run_grid
from fapbench.publication import (
    build_publication_outputs,
    select_fedadam_validation,
    write_publication_report,
)
from fapbench.robustness import REFERENCE_PHASE, build_robustness_outputs
from fapbench.rolling import build_rolling_outputs
from fapbench.utils import write_json
from fapbench.sanity import run_sanity

# data_recipe2_revised.py is a sibling script, not part of the fapbench
# package. `python scripts/run_publication_experiments.py` puts this file's
# own directory on sys.path[0] automatically, but importing this module as
# scripts.run_publication_experiments (e.g. from tests) does not -- so add it
# explicitly to support both invocation styles.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from data_recipe2_revised import data_recipe2  # noqa: E402


# core_expanded_features trains on a different panel (20 characteristics,
# built with include_expanded_characteristics=True) than every other phase
# (7 characteristics). --data-path must not be applied to it, or it would
# point that phase at a panel missing the 13 expanded zscore columns (see
# EXPANDED_CHARACTERISTIC_COMPONENTS in data_recipe2_revised.py) and crash.
# It gets its own --expanded-data-path instead.
EXPANDED_FEATURES_PHASE = "core_expanded_features"


def _resolve_from_plan(plan_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (plan_path.parent / path).resolve()


def _effective_grid(
    plan_path: Path,
    phase: dict[str, Any],
    *,
    output_root: Path,
    data_path: Path | None,
) -> dict[str, Any]:
    config_path = _resolve_from_plan(plan_path, str(phase["grid"]))
    config = load_config(config_path)
    config["output_dir"] = str(output_root / str(phase["name"]))
    if data_path is not None:
        config["base_overrides"] = deep_update(
            config.get("base_overrides", {}),
            {"data": {"path": str(data_path.resolve())}},
        )
    return config


def _ensure_expanded_panel(
    expanded_data_path: Path,
    *,
    raw_data_dir: Path,
    rebuild: bool,
    log_level: str,
) -> None:
    if expanded_data_path.exists() and not rebuild:
        print(f"Expanded panel already exists, reusing: {expanded_data_path}")
        return
    print(
        f"Building expanded (20-characteristic) panel from {raw_data_dir} "
        f"-> {expanded_data_path} (this takes roughly as long as the original "
        "panel build, ~30 minutes on the reference machine) ..."
    )
    data_recipe2(
        input_path=raw_data_dir,
        output_path=expanded_data_path,
        log_level=log_level,
        include_expanded_characteristics=True,
    )


def _save_effective_config(config: dict[str, Any], path: Path) -> None:
    serializable = {key: value for key, value in config.items() if key != "_config_path"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")


def _resolve_effective_data_path(grid_config: dict[str, Any]) -> Path:
    """Mirror fapbench.grid.run_grid's own base_config + base_overrides merge,
    just far enough to read out the final data.path before training starts."""
    grid_path = Path(grid_config["_config_path"])
    base_path = Path(grid_config["base_config"])
    if not base_path.is_absolute():
        base_path = (grid_path.parent / base_path).resolve()
    base = load_config(base_path)
    base = deep_update(base, grid_config.get("base_overrides", {}))
    return Path(base["data"]["path"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete ICAIF publication suite once: validation-only FedAdam "
            "selection, headline benchmarks, partitions, heterogeneity, matched "
            "controls, diagnostics, audits, tables, and figures."
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/icaif_publication.yaml"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help=(
            "Override data.path in every phase except core_expanded_features "
            "(which uses --expanded-data-path instead, since it trains on a "
            "different panel) without editing YAML files."
        ),
    )
    parser.add_argument(
        "--expanded-data-path",
        type=Path,
        help=(
            "Override data.path for the core_expanded_features phase only. "
            "Defaults to whatever configs/crsp_compustat_profitability_expanded.yaml "
            "specifies (data0712_expanded.parquet next to --raw-data-dir)."
        ),
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("/Users/keyang/qdata"),
        help=(
            "Directory containing compustat.csv/ccm.csv/crsp.csv/delisting.csv, "
            "used only to auto-build the expanded panel for core_expanded_features "
            "if it does not already exist at --expanded-data-path."
        ),
    )
    parser.add_argument(
        "--rebuild-expanded-panel",
        action="store_true",
        help="Rebuild the expanded panel even if it already exists at --expanded-data-path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the publication output directory.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only named phases. Final cross-phase report is skipped.",
    )
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else Path(plan["output_root"]).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    requested = set(args.only or [])
    phases = [
        phase
        for phase in plan["phases"]
        if not requested or str(phase["name"]) in requested
    ]
    if requested:
        found = {str(phase["name"]) for phase in phases}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Unknown publication phases: {missing}")
    if args.skip_validation:
        phases = [p for p in phases if str(p["name"]) != "fedadam_validation"]

    effective_configs: dict[str, dict[str, Any]] = {}
    for phase in phases:
        name = str(phase["name"])
        phase_data_path = (
            args.expanded_data_path if name == EXPANDED_FEATURES_PHASE else args.data_path
        )
        config = _effective_grid(
            plan_path,
            phase,
            output_root=output_root,
            data_path=phase_data_path,
        )
        effective_configs[name] = config
        _save_effective_config(
            config, output_root / "effective_configs" / f"{name}.yaml"
        )

    manifest = {
        "plan": str(plan_path),
        "output_root": str(output_root),
        "data_path_override": str(args.data_path.resolve()) if args.data_path else None,
        "expanded_data_path_override": (
            str(args.expanded_data_path.resolve()) if args.expanded_data_path else None
        ),
        "phases": [
            {
                "name": name,
                "grid": config["_config_path"],
                "output_dir": config["output_dir"],
                "n_scenarios": len(config.get("scenarios", [])),
                "n_seeds": len(config.get("seeds", [])),
            }
            for name, config in effective_configs.items()
        ],
        "expected_fedadam_server_lr": float(plan["expected_fedadam_server_lr"]),
        "dry_run": bool(args.dry_run),
    }
    write_json(manifest, output_root / "publication_manifest.json")
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        print("Dry run complete; no model training was started.")
        return

    phase_status: list[dict[str, Any]] = []
    if bool(plan.get("run_iid_sanity", True)) and not args.skip_sanity and not requested:
        sanity = run_sanity(seed=42, n_entities=120, n_clients=12, tolerance=1.0e-6)
        write_json(sanity, output_root / "iid_sanity.json")
        if sanity["status"] != "passed":
            raise RuntimeError("IID sanity check failed.")

    selection_report: dict[str, Any] | None = None
    for phase in phases:
        name = str(phase["name"])
        print(f"\n=== Running publication phase: {name} ===")
        if name == EXPANDED_FEATURES_PHASE:
            _ensure_expanded_panel(
                _resolve_effective_data_path(effective_configs[name]),
                raw_data_dir=args.raw_data_dir.resolve(),
                rebuild=args.rebuild_expanded_panel,
                log_level="INFO",
            )
        output = run_grid(effective_configs[name])
        phase_status.append(
            {"phase": name, "status": "completed", "output_dir": str(output.resolve())}
        )
        write_json(phase_status, output_root / "phase_status.json")

        if name == str(plan["selection"]["phase"]):
            selection_cfg = plan["selection"]
            selection_report = select_fedadam_validation(
                output / "grid_results.csv",
                scenario_prefix=str(selection_cfg["scenario_prefix"]),
                method=str(selection_cfg["method"]),
                metric=str(selection_cfg["metric"]),
                mode=str(selection_cfg["mode"]),
                constraint_metric=str(selection_cfg["constraint_metric"]),
                constraint_min=float(selection_cfg["constraint_min"]),
            )
            write_json(selection_report, output_root / "fedadam_selection.json")
            expected = float(plan["expected_fedadam_server_lr"])
            selected = float(selection_report["selected_server_lr"])
            if not np.isclose(selected, expected, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    "Validation-only FedAdam selection no longer matches the frozen "
                    f"publication value: selected={selected}, expected={expected}."
                )

    if requested:
        print("Selected phases completed. Cross-phase paper outputs were not rebuilt.")
        return

    if selection_report is None:
        selection_path = output_root / "fedadam_selection.json"
        if selection_path.exists():
            selection_report = json.loads(selection_path.read_text(encoding="utf-8"))

    audit = {
        "partition_common_universe": validate_partition_grid(output_root / "partitions"),
        "matched_controls": validate_identification_grid(output_root / "identification"),
        "heterogeneity_matched_controls": validate_heterogeneity_matched_controls(
            output_root / "heterogeneity"
        ),
    }
    audit["passed"] = all(section.get("passed", False) for section in audit.values())
    write_json(audit, output_root / "identification_audit.json")
    if not audit["passed"]:
        raise RuntimeError("Publication identification audit failed.")

    build = build_publication_outputs(output_root)
    # Reviewer-requested robustness tables (configs/icaif_robustness.yaml). This
    # phase is optional: it is run as a separate grid, so the tables are written
    # only once its outputs exist and are silently empty until then.
    # The market-adjusted alpha needs the panel itself (to build the market
    # proxy), so pass the core phase's resolved data path when it is available.
    core_config = effective_configs.get(REFERENCE_PHASE)
    panel_path = (
        _resolve_effective_data_path(core_config) if core_config is not None else None
    )
    robustness = build_robustness_outputs(output_root, panel_path)
    build["robustness_tables"] = {
        name: int(len(table)) for name, table in robustness.items()
    }
    # Expanding-window refits (configs/icaif_rolling_refit.yaml) are a separate
    # grid like the robustness one, so these stay empty until it has been run.
    rolling = build_rolling_outputs(output_root)
    build["rolling_tables"] = {name: int(len(t)) for name, t in rolling.items()}
    write_json(build, output_root / "publication_build.json")
    report = write_publication_report(
        output_root,
        selection=selection_report,
        audit=audit,
        build=build,
        phase_status=phase_status,
    )
    print(f"\nPublication suite complete: {report.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a hyperparameter scenario using validation-only columns from "
            "a completed grid. Test metrics are intentionally inaccessible to the "
            "selection rule."
        )
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scenario-prefix", required=True)
    parser.add_argument("--method", help="Optional exact method name to retain.")
    parser.add_argument("--metric", default="validation_best_score")
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument(
        "--constraint-metric",
        help="Optional validation_* metric used as an eligibility constraint.",
    )
    parser.add_argument("--constraint-min", type=float)
    parser.add_argument("--constraint-max", type=float)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-yaml", type=Path)
    args = parser.parse_args()

    if not args.metric.startswith("validation_"):
        raise ValueError(
            "The selection metric must start with 'validation_' to prevent test-set tuning."
        )
    if args.constraint_metric is not None and not args.constraint_metric.startswith(
        "validation_"
    ):
        raise ValueError(
            "The constraint metric must start with 'validation_' to prevent test-set tuning."
        )
    frame = pd.read_csv(args.results)
    required = {"scenario", "seed", "method", args.metric}
    if args.constraint_metric is not None:
        required.add(args.constraint_metric)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    candidates = frame.loc[
        frame["scenario"].astype(str).str.startswith(args.scenario_prefix)
    ].copy()
    if args.method is not None:
        candidates = candidates.loc[
            candidates["method"].astype(str) == str(args.method)
        ].copy()
    candidates[args.metric] = pd.to_numeric(candidates[args.metric], errors="coerce")
    candidates = candidates.dropna(subset=[args.metric])
    if candidates.empty:
        raise ValueError("No validation-only candidates matched the requested prefix.")

    aggregations = {args.metric: ["mean", "std", "count"]}
    if args.constraint_metric is not None:
        candidates[args.constraint_metric] = pd.to_numeric(
            candidates[args.constraint_metric], errors="coerce"
        )
        aggregations[args.constraint_metric] = ["mean", "std", "count"]
    grouped = candidates.groupby(["scenario", "method"]).agg(aggregations)
    grouped.columns = [
        f"{metric}_{statistic}" for metric, statistic in grouped.columns
    ]
    summary = grouped.reset_index()
    primary_mean = f"{args.metric}_mean"
    primary_std = f"{args.metric}_std"
    primary_count = f"{args.metric}_count"
    eligible = summary.copy()
    if args.constraint_metric is not None:
        constraint_mean = f"{args.constraint_metric}_mean"
        if args.constraint_min is not None:
            eligible = eligible.loc[eligible[constraint_mean] >= args.constraint_min]
        if args.constraint_max is not None:
            eligible = eligible.loc[eligible[constraint_mean] <= args.constraint_max]
    if eligible.empty:
        raise ValueError("No candidate satisfies the validation-only constraint.")
    eligible = eligible.sort_values(
        primary_mean, ascending=args.mode == "min"
    )
    chosen = eligible.iloc[0]
    chosen_scenario = str(chosen["scenario"])
    chosen_method = str(chosen["method"])

    grid_root = args.results.resolve().parent
    config_candidates = sorted((grid_root / chosen_scenario).glob("seed_*/resolved_config.json"))
    resolved_config = None
    if config_candidates:
        resolved_config = json.loads(config_candidates[0].read_text(encoding="utf-8"))

    report = {
        "selection_source": str(args.results.resolve()),
        "selection_scope": "validation_only",
        "scenario_prefix": args.scenario_prefix,
        "method_filter": args.method,
        "metric": args.metric,
        "mode": args.mode,
        "chosen_scenario": chosen_scenario,
        "chosen_method": chosen_method,
        "validation_mean": float(chosen[primary_mean]),
        "validation_std": (
            float(chosen[primary_std]) if pd.notna(chosen[primary_std]) else None
        ),
        "n_seeds": int(chosen[primary_count]),
        "constraint_metric": args.constraint_metric,
        "constraint_min": args.constraint_min,
        "constraint_max": args.constraint_max,
        "candidate_summary": summary.to_dict(orient="records"),
        "eligible_candidates": eligible.to_dict(orient="records"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.output_yaml is not None:
        if resolved_config is None:
            raise FileNotFoundError(
                "Could not locate a resolved_config.json for the chosen scenario."
            )
        frozen = {
            "selection_provenance": {
                "scope": "validation_only",
                "metric": args.metric,
                "mode": args.mode,
                "chosen_scenario": chosen_scenario,
            },
            "federated": resolved_config.get("federated", {}),
            "model_selection": resolved_config.get("model_selection", {}),
        }
        args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
        args.output_yaml.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

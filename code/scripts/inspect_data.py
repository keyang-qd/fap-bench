from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fapbench.data import (
    PanelSpec,
    infer_feature_columns,
    load_panel,
    validate_crsp_compustat_profitability_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect whether a panel is benchmark-ready.")
    parser.add_argument("path")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--entity-col", default="permno")
    parser.add_argument("--target-col", default="retadj_1mn")
    parser.add_argument("--domain-col", default=None)
    parser.add_argument(
        "--source-profile",
        choices=["generic", "crsp_compustat_profitability_v1", "crsp_compustat_profitability_v2"],
        default="generic",
    )
    args = parser.parse_args()

    frame = load_panel(args.path)
    spec = PanelSpec(
        date_col=args.date_col,
        entity_col=args.entity_col,
        target_col=args.target_col,
    )
    report: dict[str, object] = {
        "path": str(Path(args.path).resolve()),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "date_min": None,
        "date_max": None,
        "n_dates": None,
        "n_entities": None,
        "n_domains": None,
        "target_missing_fraction": None,
        "inferred_features": [],
        "warnings": [],
    }
    warnings: list[str] = []
    required = [args.date_col, args.entity_col, args.target_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        warnings.append(f"Missing required columns: {missing}")
    else:
        dates = pd.to_datetime(frame[args.date_col], errors="coerce")
        report["date_min"] = str(dates.min())
        report["date_max"] = str(dates.max())
        report["n_dates"] = int(dates.nunique())
        report["n_entities"] = int(frame[args.entity_col].nunique())
        report["target_missing_fraction"] = float(frame[args.target_col].isna().mean())
        try:
            report["inferred_features"] = infer_feature_columns(frame, spec)
        except Exception as exc:
            warnings.append(str(exc))

        if report["n_entities"] < 2:
            warnings.append("At least two entities are required for federated clients.")
        if report["n_dates"] < 12:
            warnings.append("Fewer than 12 dates is insufficient for a meaningful monthly OOS study.")

    if args.domain_col:
        if args.domain_col not in frame.columns:
            warnings.append(f"Configured domain column {args.domain_col!r} is absent.")
        else:
            report["n_domains"] = int(frame[args.domain_col].nunique(dropna=True))
            if report["n_domains"] < 2:
                warnings.append("The domain column produces fewer than two clients.")

    if args.source_profile in {"crsp_compustat_profitability_v1", "crsp_compustat_profitability_v2"}:
        report["source_validation"] = validate_crsp_compustat_profitability_panel(frame)
        source = report["source_validation"]
        status = source.get("target_alignment_status")
        if status == "contemporaneous_target":
            warnings.append(
                "retadj_1mn is contemporaneous; configure target_shift_months: 1."
            )
        elif status == "pre_aligned_exact_t_plus_1":
            warnings.append(
                "Target is already aligned to t+1; configure target_shift_months: 0 and split on target_date."
            )
        else:
            warnings.append("Target alignment could not be verified automatically.")
        if not source.get("has_sic"):
            warnings.append("Retain SIC before using the recommended FF12 domain partition.")
        if source.get("date_year_month_mismatch_rows", 0) > 0:
            warnings.append(
                "Formation date and year_month disagree for some rows."
            )
        if source.get("target_date_period_mismatch_rows", 0) > 0:
            warnings.append(
                "target_date and target_year_month disagree for some rows."
            )

    report["warnings"] = warnings
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Build the expanded 20-characteristic panel (see
EXPANDED_CHARACTERISTIC_COMPONENTS in data_recipe2_revised.py for the full
value/investment/financing/accrual/intangible/size/momentum list).

This is a separate entry point from data_recipe2_revised.py's own __main__
block so the original camera-ready panel build (7 profitability
characteristics, include_expanded_characteristics=False) stays exactly as it
was. Run this only for the Tier 2 robustness check described in
EXPERIMENT_ENHANCEMENT_PLAN.md; it is not part of the primary publication
pipeline.
"""

import argparse
from pathlib import Path

from data_recipe2_revised import data_recipe2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the expanded CRSP-Compustat characteristic panel."
    )
    parser.add_argument("--input-path", default="/Users/keyang/qdata")
    parser.add_argument(
        "--output-path", default="/Users/keyang/qdata/data0712_expanded.parquet"
    )
    parser.add_argument("--log-file", default="/Users/keyang/qdata/data0712_expanded_build.log")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    panel = data_recipe2(
        input_path=args.input_path,
        output_path=args.output_path,
        log_file=args.log_file,
        log_level=args.log_level,
        include_expanded_characteristics=True,
    )
    print(panel.shape)
    print(
        panel[
            [
                "permno", "year_month", "target_year_month", "retadj_1mn",
                "bm", "agr", "mom_2_12", "rdoa", "logme",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from fapbench.evaluation.statistics import analyze_portfolio_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired HAC return tests and block-bootstrap Sharpe intervals."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=["primary", "global", "client_neutral"],
        default="primary",
    )
    parser.add_argument("--return-col", default="net_return")
    parser.add_argument("--reference-method", default=None)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--periods-per-year", type=int, default=12)
    parser.add_argument("--hac-lag", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tests, intervals = analyze_portfolio_results(
        output_dir,
        variant=args.variant,
        return_col=args.return_col,
        reference_method=args.reference_method,
        all_pairs=args.all_pairs,
        periods_per_year=args.periods_per_year,
        hac_lag=args.hac_lag,
        block_length=args.block_length,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    tests_path = output_dir / f"pairwise_portfolio_tests_{args.variant}.csv"
    intervals_path = output_dir / f"sharpe_intervals_{args.variant}.csv"
    tests.to_csv(tests_path, index=False)
    intervals.to_csv(intervals_path, index=False)
    print(f"Wrote {tests_path}")
    print(f"Wrote {intervals_path}")


if __name__ == "__main__":
    main()

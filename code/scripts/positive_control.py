#!/usr/bin/env python3
"""Positive control: does the evaluation machinery recover a known-real effect?

Why this exists
---------------
Every headline result in the paper is a null -- no method separates from
Centralized, system-level heterogeneity has no resolvable footprint. A reviewer
made the decisive objection: under standard errors this wide, "no method
separates" and "the estimator has no signal to lose" are observationally
equivalent, and nothing in the paper demonstrates the pipeline can detect a
difference that is known to exist.

This script supplies that anchor. It feeds two textbook cross-sectional
anomalies -- book-to-market (be/me) and size (-me) -- through the *same*
universe screen and the *same* portfolio/predictive evaluation code the
benchmark uses for model predictions, with no training anywhere. If the
machinery recovers the documented sign and magnitude of these premia, then the
nulls elsewhere are statements about the models, not about the measurement.

Both windows are reported, because the two answer different questions:
  * 2019--2023 (the 60-month reference cell) -- the cell the paper calls a
    null-calibration configuration.
  * 1990--2023 (the 408-month cell) -- where the paper says ranking belongs.
Value and size are both well documented to have been weak-to-negative in the
recent window, so a null there is informative rather than disqualifying; the
long window is the real test.

Usage:
    python scripts/positive_control.py --panel /path/to/data0712.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fapbench.data import apply_universe_filters
from fapbench.evaluation.portfolio import long_short_portfolio_metrics
from fapbench.evaluation.predictive import (
    monthly_predictive_metrics,
    predictive_metrics,
)
from fapbench.evaluation.statistics import newey_west_mean_test

# The reference protocol's screen, copied from a resolved run config rather than
# retyped: NYSE 20th-percentile market cap and a $5 price floor.
UNIVERSE = dict(
    enabled=True,
    price_col="prc",
    min_abs_price=5.0,
    market_cap_col="me",
    exclude_bottom_market_cap_fraction=0.2,
    breakpoint_exchange_col="exchcd",
    breakpoint_exchange_values=(1,),
    min_stocks_per_month=0,
)
COLUMNS = [
    "permno", "formation_date", "target_date",
    "be", "me", "prc", "exchcd", "retadj_1mn",
]
WINDOWS = {
    "reference_60m": ("2019-01-01", "2023-12-31"),
    "long_408m": ("1990-01-01", "2023-12-31"),
}


def signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Two anomalies with documented long-run premia, as raw characteristics.

    Book equity is non-positive for a minority of firm-months, where
    book-to-market is not defined; those rows drop rather than being clipped to
    a boundary value that would be ranked as extreme value.
    """
    book_to_market = frame["be"] / frame["me"]
    book_to_market = book_to_market.where(frame["be"] > 0)
    return {
        # Long high book-to-market (value), short low (growth).
        "book_to_market": book_to_market,
        # Long small, short large: the sign convention that makes the documented
        # premium positive, so a negative number here is a real failure and not
        # a flipped axis.
        "size_smb": -frame["me"],
    }


def evaluate(frame: pd.DataFrame, score: pd.Series, label: str) -> dict[str, object]:
    panel = pd.DataFrame(
        {
            "permno": frame["permno"].to_numpy(),
            "date": frame["target_date"].to_numpy(),
            "y_true": frame["retadj_1mn"].to_numpy(),
            "y_pred": score.to_numpy(),
        }
    ).dropna()
    if panel.empty:
        return {"signal": label, "n_rows": 0}

    pred = predictive_metrics(panel, date_col="date", target_col="y_true",
                              prediction_col="y_pred")
    returns, port = long_short_portfolio_metrics(
        panel, date_col="date", entity_col="permno", target_col="y_true",
        prediction_col="y_pred", client_col=None, quantile=0.1,
        transaction_cost_bps=10.0,
    )
    # Same inference convention as Table 5: Newey--West on the monthly series,
    # so the control's t-statistic is comparable to the benchmark's own.
    monthly = monthly_predictive_metrics(panel, date_col="date",
                                         target_col="y_true",
                                         prediction_col="y_pred")
    ic_col = "rank_ic" if "rank_ic" in monthly.columns else monthly.columns[-1]
    ic_test = newey_west_mean_test(monthly[ic_col].dropna())
    net_col = "net_return" if "net_return" in returns.columns else "return"
    ret_test = newey_west_mean_test(returns[net_col].dropna())
    return {
        "signal": label,
        "n_rows": int(len(panel)),
        "n_months": int(panel["date"].nunique()),
        "rank_ic": float(pred.get("rank_ic", float("nan"))),
        "rank_ic_t": float(ic_test["t_statistic"]),
        "rank_ic_p": float(ic_test["p_value"]),
        "net_sharpe": float(port.get("net_sharpe", float("nan"))),
        "gross_sharpe": float(port.get("sharpe", float("nan"))),
        "net_annual_return": float(port.get("net_annual_return", float("nan"))),
        "net_return_t": float(ret_test["t_statistic"]),
        "net_return_p": float(ret_test["p_value"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel, columns=COLUMNS)
    frame["formation_date"] = pd.to_datetime(frame["formation_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])

    # Both universes, because the reference screen removes microcaps -- which is
    # exactly where the size and value premia are documented to live. Running
    # only the screened universe would confound "the machinery cannot detect a
    # known effect" with "this universe does not contain that effect."
    universes = {
        "investable": UNIVERSE,
        "unrestricted": {**UNIVERSE, "enabled": False},
    }
    records: list[dict[str, object]] = []
    for universe_name, spec in universes.items():
        screened, report = apply_universe_filters(
            frame, date_col="formation_date", **spec
        )
        print(f"\n[{universe_name}] {report['n_rows_before']:,} -> "
              f"{report['n_rows_after']:,} rows")
        for window, (start, end) in WINDOWS.items():
            mask = screened["target_date"].between(start, end)
            sub = screened.loc[mask].copy()
            for label, score in signals(sub).items():
                row = evaluate(sub, score, label)
                row["window"] = window
                row["universe"] = universe_name
                records.append(row)
                print(
                    f"  {window:14s} {label:16s} "
                    f"n={row.get('n_months', 0):3d}m "
                    f"rank_ic={row.get('rank_ic', float('nan')):+.4f} "
                    f"(t={row.get('rank_ic_t', float('nan')):+.1f}) "
                    f"net_sharpe={row.get('net_sharpe', float('nan')):+.3f} "
                    f"(ret p={row.get('net_return_p', float('nan')):.3f})"
                )

    table = pd.DataFrame(records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")
    print(json.dumps(records, indent=1, default=float))


if __name__ == "__main__":
    main()

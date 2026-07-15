from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_panel(
    n_stocks: int = 120,
    n_months: int = 60,
    n_features: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    sectors = np.array([10, 20, 30, 40, 50, 60])
    stock_sector = rng.choice(sectors, size=n_stocks)
    stock_effect = rng.normal(0.0, 0.01, size=n_stocks)
    beta = rng.normal(0.0, 0.025, size=n_features)

    for month_index, date in enumerate(dates):
        regime = 1.0 + 0.4 * np.sin(month_index / 8.0)
        common_shock = rng.normal(0.0, 0.015)
        for stock in range(n_stocks):
            sector = stock_sector[stock]
            x = rng.normal(loc=(sector / 100.0), scale=regime, size=n_features)
            sector_alpha = (sector - sectors.mean()) / 10000.0
            target = float(x @ beta + sector_alpha + stock_effect[stock] + common_shock + rng.normal(0, 0.04))
            row = {
                "permno": 10000 + stock,
                "date": date,
                "sic2": int(sector),
                "exchcd": int(1 + stock % 3),
                "me": float(np.exp(rng.normal(7.0, 1.0))),
                "retadj_1mn": target,
            }
            row.update({f"feature_{idx:02d}": float(value) for idx, value in enumerate(x)})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic.parquet")
    parser.add_argument("--stocks", type=int, default=120)
    parser.add_argument("--months", type=int, default=60)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = make_panel(args.stocks, args.months, args.features, args.seed)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(output, index=False)
    elif suffix in {".pkl", ".pickle"}:
        frame.to_pickle(output)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(output, index=False)
    else:
        raise ValueError(
            "Synthetic output must use .parquet, .csv, .pkl, or .pickle."
        )
    print(f"Wrote {len(frame):,} rows to {output}")


if __name__ == "__main__":
    main()

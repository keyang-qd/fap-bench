#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _aggregate(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    data = frame[["scenario", "method", x, y]].copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    return (
        data.dropna()
        .groupby(["scenario", "method"], as_index=False)[[x, y]]
        .mean()
    )


def _scatter(table: pd.DataFrame, x: str, y: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for method, group in table.groupby("method", sort=True):
        ax.scatter(group[x], group[y], label=str(method), s=48)
        for row in group.itertuples(index=False):
            ax.annotate(
                str(row.scenario),
                (getattr(row, x), getattr(row, y)),
                fontsize=7,
                alpha=0.75,
            )
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot communication/time versus predictive/economic performance."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.results)

    communication_x = "communicated_bytes_fp32"
    rank_y = "rank_ic"
    if communication_x in frame.columns and rank_y in frame.columns:
        table = _aggregate(frame, communication_x, rank_y)
        table["communicated_megabytes_fp32"] = table[communication_x] / 1_000_000.0
        _scatter(
            table,
            "communicated_megabytes_fp32",
            rank_y,
            args.output_dir / "communication_vs_rank_ic.png",
        )
        table.to_csv(
            args.output_dir / "communication_vs_rank_ic.csv", index=False
        )

    time_x = "training_elapsed_seconds"
    sharpe_y = "client_neutral_portfolio_net_sharpe"
    if time_x in frame.columns and sharpe_y in frame.columns:
        table = _aggregate(frame, time_x, sharpe_y)
        table["training_minutes"] = table[time_x] / 60.0
        _scatter(
            table,
            "training_minutes",
            sharpe_y,
            args.output_dir / "training_time_vs_client_neutral_sharpe.png",
        )
        table.to_csv(
            args.output_dir / "training_time_vs_client_neutral_sharpe.csv",
            index=False,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_metric(frame: pd.DataFrame, metric: str, output: Path) -> None:
    pivot = frame.pivot_table(index="scenario", columns="method", values=metric, aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Scenario")
    ax.set_title(f"{metric.replace('_', ' ').title()} across benchmark scenarios")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def _plot_heterogeneity_curve(
    frame: pd.DataFrame,
    heterogeneity_col: str,
    metric: str,
    output: Path,
) -> None:
    summary = (
        frame.groupby([heterogeneity_col, "method"], as_index=False)[metric]
        .mean()
        .sort_values(heterogeneity_col)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in summary.groupby("method"):
        ax.plot(group[heterogeneity_col], group[metric], marker="o", label=method)
    ax.set_xlabel(heterogeneity_col.replace("heterogeneity_", "").replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Heterogeneity–performance relationship")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready benchmark figures.")
    parser.add_argument("--results", required=True, help="grid_results.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results_path = Path(args.results)
    frame = pd.read_csv(results_path)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["rank_ic", "sharpe", "oos_r2_zero", "client_std_rank_ic"]:
        if metric in frame.columns:
            _plot_metric(frame, metric, output_dir / f"{metric}_by_scenario.png")

    candidates = [
        "heterogeneity_feature_jaccard_dissimilarity",
        "heterogeneity_temporal_unavailability",
        "heterogeneity_feature_distribution_wasserstein",
    ]
    for column in candidates:
        if column in frame.columns and frame[column].nunique() > 1:
            _plot_heterogeneity_curve(
                frame,
                heterogeneity_col=column,
                metric="rank_ic",
                output=output_dir / f"rank_ic_vs_{column}.png",
            )
    print(f"Figures written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Emit configs/icaif_rolling_refit.yaml: one scenario per expanding-window refit.

Why this exists
---------------
The single-window protocol trains through 2014 and tests 2019-2023, so inference
runs on 60 monthly observations. That caps the benchmark's discriminative power:
no method's net Sharpe separates from Centralized, every interval in the headline
table contains zero, and RQ1's Kendall tau carries an interval spanning most of
its range. More seeds cannot fix this -- seeds share one market history and add
design randomness, not statistical power. Only more test months help.

Expanding-window refits give them. For test year Y the model trains on everything
through Y-5, validates on the four years Y-4..Y-1, and predicts Y. Rolling Y from
1990 to 2023 yields 34 windows and ~408 out-of-sample months, roughly a 6x
increase, while also removing the staleness of scoring 2023 with a model frozen in
2014 and diluting a test window in which COVID is 20% of the sample.

Each window keeps the reference protocol's validation *shape* (a four-year block
immediately before the test year), so window results are comparable to each other
and to the single-window table.

Design note: rolling is expressed as a grid of scenarios rather than a new
training loop. Every window is an ordinary run, so partitioning, budget matching,
checkpoint selection, resumability, and per-month output writing are all reused
unchanged. `fapbench/rolling.py` then concatenates the per-window monthly series
into one continuous series -- valid because portfolios are formed within each
month, so monthly returns from adjacent windows join without overlap.

Usage:
    python scripts/make_rolling_config.py
    caffeinate -dims python -u scripts/run_grid.py --grid configs/icaif_rolling_refit.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Panel runs 1964-02..2023-12. The first test year is chosen so its training
# block still spans two decades: Y=1990 trains on 1964-1985.
FIRST_TEST_YEAR = 1990
LAST_TEST_YEAR = 2023
VALIDATION_YEARS = 4  # matches the single-window protocol's 2015-2018 block


def build_config(
    first_year: int = FIRST_TEST_YEAR,
    last_year: int = LAST_TEST_YEAR,
    validation_years: int = VALIDATION_YEARS,
    step: int = 1,
) -> dict:
    scenarios = []
    for test_year in range(first_year, last_year + 1, step):
        train_end = f"{test_year - validation_years - 1}-12-31"
        val_end = f"{test_year - 1}-12-31"
        test_end = f"{test_year + step - 1}-12-31"
        scenarios.append(
            {
                "name": f"refit_test_{test_year}",
                "overrides": {
                    "split": {
                        "train_end": train_end,
                        "val_end": val_end,
                        "test_end": test_end,
                    },
                    "federated": {
                        "rounds": 50,
                        "participation_rate": 1.0,
                        "local_epochs": 1,
                        "server_lr": 0.003,
                    },
                },
            }
        )
    return {
        "base_config": "crsp_compustat_profitability.yaml",
        "output_dir": "outputs/icaif_publication/rolling_refit",
        "continue_on_error": False,
        "preload_panel_once": True,
        "resume_completed": True,
        "base_overrides": {
            # FedProx is dropped here alone among the grids. It is ~10% of each
            # cell and tracks FedAvg to four decimals at both budgets and across
            # the whole mu sweep, so it buys no separation at any sample length;
            # the headline table still reports it. Local is deliberately KEPT
            # despite being the single most expensive method (24% of a cell):
            # Local-vs-Centralized is the widest gap in the headline table
            # (0.170 vs 0.504 net Sharpe), so it is the pair most likely to
            # resolve at 408 months, which is the entire question this grid asks.
            "baselines": {"centralized_masked": False},
            "federated": {"algorithms": ["fedavg", "scaffold", "fedadam"]},
        },
        "grid_execution": {
            # 3 concurrent runs on a 10-core machine; raise only when this grid
            # has the machine to itself.
            "parallel_runs": 3,
            "max_total_client_workers": 9,
            "max_total_torch_threads": 9,
            "start_method": "spawn",
            "parallel_child_console": False,
        },
        "child_logging": {"console": True, "round_interval": 10},
        # Three seeds, not the ten used elsewhere. Seeds vary initialization and
        # partition draws only; with ~408 test months the binding constraint is
        # the market history, not optimization noise, and the unit of inference
        # is the calendar month either way. Trading seeds for windows buys
        # statistical power; trading windows for seeds does not.
        "seeds": [11, 22, 33],
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "configs"
        / "icaif_rolling_refit.yaml",
    )
    parser.add_argument("--first-year", type=int, default=FIRST_TEST_YEAR)
    parser.add_argument("--last-year", type=int, default=LAST_TEST_YEAR)
    parser.add_argument(
        "--step", type=int, default=1, help="Years per refit; 2 halves the cost."
    )
    args = parser.parse_args()

    config = build_config(args.first_year, args.last_year, step=args.step)
    header = (
        "# GENERATED by scripts/make_rolling_config.py -- do not edit by hand.\n"
        f"# {len(config['scenarios'])} expanding-window refits, "
        f"{len(config['scenarios']) * len(config['seeds'])} cells, "
        f"~{len(config['scenarios']) * 12 * args.step} out-of-sample months.\n"
    )
    args.output.write_text(
        header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(
        f"Wrote {args.output} "
        f"({len(config['scenarios'])} windows x {len(config['seeds'])} seeds)"
    )


if __name__ == "__main__":
    main()

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


# The investable universe, copied verbatim from `ff12_exclude_microcaps` in
# configs/icaif_core.yaml. A reviewer's objection made this necessary: the
# expanding-window exercise is the paper's defence of the all-negative reference
# table ("a power limit rather than a demonstrated equivalence"), but it was only
# ever run on the *unrestricted* universe -- the one the paper explicitly rejects
# as a reference. Demonstrating a power limit on a universe the paper does not use
# does not defend the universe it does use.
INVESTABLE_UNIVERSE_FILTER = {
    "enabled": True,
    "price_col": "prc",
    "min_abs_price": 5.0,
    "market_cap_col": "me",
    "exclude_bottom_market_cap_fraction": 0.2,
    "breakpoint_exchange_col": "exchcd",
    "breakpoint_exchange_values": [1],
}


# The linear/ridge arm, copied verbatim from `linear_ridge_weak` in
# configs/icaif_robustness.yaml so the 408-month cell and the 60-month lever row
# are the same model. `hidden_dims: []` reduces AssetPricingMLP to a single output
# layer (linear regression) and `weight_decay` supplies the L2 penalty, making it
# ridge. Copying rather than importing is deliberate: if the robustness grid's
# penalty is ever retuned, the refit must not silently follow it and become
# incomparable to the row the manuscript quotes against.
LINEAR_MODEL = {"hidden_dims": [], "dropout": 0.0, "batch_norm": False}
LINEAR_WEIGHT_DECAY = 0.0001

# The wide panel at the architecture its own validation sweep selected. This is
# the cell that has BOTH signal and power: the reference MLP predicts too weakly
# for a null to be informative, and linear/ridge has no coordination cost to
# measure, so neither answers "is the null about federation or about a model with
# no alpha?". This one does. Architecture copied from
# expanded_capacity_selection.json (selected_scenario
# expanded_capacity_h128_64_lr0_01), chosen on VALIDATION Rank IC alone.
EXPANDED_TUNED_MODEL = {"hidden_dims": [128, 64]}
EXPANDED_TUNED_CLIENT_LR = 0.01
EXPANDED_BASE_CONFIG = "crsp_compustat_profitability_expanded.yaml"

# Client-level DP at the aggregation step (fapbench/privacy.py). The clip norm is
# NOT a free parameter: it is calibrated to the measured update-norm distribution
# of the cell it is applied to. Probes with a deliberately huge clip give a median
# update norm of 0.0046 on the reference cell and 0.0042 on the tuned panel, so
# 0.005 clips roughly half the clients in both. A clip chosen without measuring
# (the shipped default of 1.0) is ~200x too large, clips nothing, and turns the
# experiment into a report about noise.
DP_CLIP_NORM = 0.005


def _output_dir(
    *,
    investable: bool,
    linear: bool,
    expanded_tuned: bool = False,
    dp_noise_multiplier: float | None = None,
) -> str:
    """One directory per (universe, model class). These cells are not comparable
    in levels, so sharing a directory would let the rolling concatenation mix
    them into one series without any error being raised."""
    stem = "outputs/icaif_publication/rolling_refit"
    if investable:
        stem += "_investable"
    if linear:
        stem += "_linear"
    if expanded_tuned:
        stem += "_tuned20"
    if dp_noise_multiplier is not None:
        # Encoded in the directory name: two DP levels must never share an output
        # directory, and neither may share one with the non-private cell.
        tag = f"{dp_noise_multiplier:g}".replace(".", "p")
        stem += f"_dp{tag}"
    return stem


def build_config(
    first_year: int = FIRST_TEST_YEAR,
    last_year: int = LAST_TEST_YEAR,
    validation_years: int = VALIDATION_YEARS,
    step: int = 1,
    *,
    investable: bool = False,
    linear: bool = False,
    expanded_tuned: bool = False,
    dp_noise_multiplier: float | None = None,
    parallel_runs: int = 3,
    seeds: list[int] | None = None,
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
                        **(
                            {"weight_decay": LINEAR_WEIGHT_DECAY}
                            if linear
                            else {}
                        ),
                        **(
                            {"client_lr": EXPANDED_TUNED_CLIENT_LR}
                            if expanded_tuned
                            else {}
                        ),
                    },
                },
            }
        )
    return {
        "base_config": (
            EXPANDED_BASE_CONFIG if expanded_tuned else "crsp_compustat_profitability.yaml"
        ),
        "output_dir": _output_dir(
            investable=investable,
            linear=linear,
            expanded_tuned=expanded_tuned,
            dp_noise_multiplier=dp_noise_multiplier,
        ),
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
            **(
                {"universe_filter": INVESTABLE_UNIVERSE_FILTER}
                if investable
                else {}
            ),
            # Model class is resolved once, not twice. Written as two separate
            # dict-splats these keys collided silently: `--linear
            # --expanded-tuned` emitted EXPANDED_TUNED_MODEL, so a config
            # requesting linear/ridge trained the MLP and nothing complained.
            # --linear names the model class, --expanded-tuned names the panel,
            # universe and client rate; when both are given the model class wins
            # and the widened cell's *architecture* selection simply does not
            # apply (a linear model has no hidden width to select).
            **(
                {"model": dict(LINEAR_MODEL)}
                if linear
                else {"model": dict(EXPANDED_TUNED_MODEL)}
                if expanded_tuned
                else {}
            ),
            **(
                {
                    "federated": {
                        "algorithms": ["fedavg", "scaffold", "fedadam"],
                        "dp": {
                            "enabled": True,
                            "clip_norm": DP_CLIP_NORM,
                            "noise_multiplier": float(dp_noise_multiplier),
                            "delta": 1e-5,
                        },
                    }
                }
                if dp_noise_multiplier is not None
                else {}
            ),
        },
        "grid_execution": {
            # 3 concurrent runs on a 10-core machine; raise only when this grid
            # has the machine to itself.
            "parallel_runs": parallel_runs,
            "max_total_client_workers": 3 * parallel_runs,
            "max_total_torch_threads": 3 * parallel_runs,
            "start_method": "spawn",
            "parallel_child_console": False,
        },
        "child_logging": {"console": True, "round_interval": 10},
        # Three seeds by default, not the ten used elsewhere: seeds vary
        # initialization and partition draws only, and with ~408 test months the
        # binding constraint is the market history rather than optimization
        # noise. A reviewer objected that the paper's designated ranking cell
        # should not have LESS replication than the cell it defers to, which is
        # fair -- seed averaging still reduces noise in the monthly series the
        # HAC test runs on. Pass --seeds to raise it; the existing runs are
        # reused, so only the added seeds train.
        "seeds": seeds or [11, 22, 33],
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
    parser.add_argument(
        "--investable",
        action="store_true",
        help=(
            "Apply the reference protocol's investable-universe screen. Writes to "
            "a separate output_dir so the unrestricted refits stay intact -- the "
            "two are not comparable in levels and must not share a directory."
        ),
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help=(
            "Use the linear/ridge model class instead of the reference MLP, "
            "matching `linear_ridge_weak` in the robustness grid. Writes to a "
            "separate output_dir: the two model classes are not comparable in "
            "levels and must not share one."
        ),
    )
    parser.add_argument(
        "--expanded-tuned",
        action="store_true",
        help=(
            "20-characteristic panel at the [128,64]/lr0.01 architecture its "
            "validation sweep selected. Combine with --investable for the "
            "reference universe."
        ),
    )
    parser.add_argument(
        "--dp-noise-multiplier",
        type=float,
        default=None,
        help=(
            "Enable client-level DP at this noise multiplier, with the calibrated "
            "clip norm. 0 gives the clipping-only control, which isolates the cost "
            "of clipping and uniform averaging from the cost of the noise."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seed list; defaults to 11 22 33. Raising it reuses completed runs.",
    )
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=3,
        help="Concurrent runs; lower it when another grid shares the machine.",
    )
    args = parser.parse_args()

    # --investable changes output_dir but not the output FILENAME, so invoking it
    # without --output silently overwrites the unrestricted config and leaves two
    # grid files pointing at one output directory. Default the filename off the
    # flag instead of trusting the caller to remember.
    if args.output.name == "icaif_rolling_refit.yaml":
        suffix = ""
        if args.investable:
            suffix += "_investable"
        if args.linear:
            suffix += "_linear"
        if args.expanded_tuned:
            suffix += "_tuned20"
        if args.dp_noise_multiplier is not None:
            suffix += "_dp" + f"{args.dp_noise_multiplier:g}".replace(".", "p")
        if suffix:
            args.output = args.output.with_name(
                f"icaif_rolling_refit{suffix}.yaml"
            )

    config = build_config(
        args.first_year,
        args.last_year,
        step=args.step,
        investable=args.investable,
        linear=args.linear,
        expanded_tuned=args.expanded_tuned,
        dp_noise_multiplier=args.dp_noise_multiplier,
        parallel_runs=args.parallel_runs,
        seeds=args.seeds,
    )
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

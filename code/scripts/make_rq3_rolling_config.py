#!/usr/bin/env python3
"""Emit configs/icaif_rq3_rolling.yaml: RQ3's matched-control pairs, refit over
expanding windows, so the dispersion contrasts are tested on ~408 months
instead of the reference protocol's 60.

Why this exists
---------------
A reviewer's sharpest methodological point: the paper already builds a
408-month expanding-window sample to defend its *economic* axis, but leaves
RQ3 -- whose claims are the sharpest in the paper, and the ones a reader is
most likely to act on -- at n=60, where the bootstrap intervals are disclosed
as anti-conservative (89-92% coverage at nominal 95%). Running the same
contrasts at 6.8x the sample either retires that caveat or shows it mattered.
Either outcome is worth knowing, and neither is knowable by argument.

Design
------
Every arm is an ordinary run, exactly as in `make_rolling_config.py`: rolling
is expressed as a grid, not a new training loop, so partitioning, budget
matching, checkpoint selection, and resumability are reused unchanged. The
scenario name encodes both the heterogeneity arm and the refit window
(`{arm}__refit_{year}`), and `fapbench.robustness.rq3_rolling_contrasts`
concatenates each arm's per-window monthly dispersion series before testing.

Three pairs, the same set as `icaif_capacity_separation.yaml` and deliberately
so -- the two tables are meant to be read against each other row for row:
  - feature_heterogeneity_severe : largest data-level effect (0.0199, FedAvg)
  - covariate_shift_strong       : the other data-level family (0.0131, FedAvg)
  - temporal_block_outages       : largest system-level effect (0.0020, SCAFFOLD)

Universe: unrestricted, matching the main heterogeneity grid rather than the
investable reference. This is deliberate. The quoted effects the 408-month run
is meant to re-test (0.0199 against 0.0020) are unrestricted numbers, so an
investable rerun would change two things at once and answer neither question.
The investable arm already exists separately as `rq3_investable`.

Step size
---------
`--step 2` gives 17 windows whose test blocks are two years each: still ~408
out-of-sample months, at half the cell count of annual refits. The refit is
biennial rather than annual, so a model is on average one year staler at
prediction time than in the economic-axis grid -- worth stating, and if
anything conservative. RQ3 measures a *contrast* between two arms that share
the window, so staleness differences cancel in the difference.

Usage:
    python scripts/make_rq3_rolling_config.py --step 2
    caffeinate -dims python -u scripts/run_grid.py --grid configs/icaif_rq3_rolling.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

FIRST_TEST_YEAR = 1990
LAST_TEST_YEAR = 2023
VALIDATION_YEARS = 4

# Rounds/participation/server_lr are pinned to the reference protocol's values,
# identical across all six arms -- the contrast is only meaningful if the two
# sides of each pair differ in the heterogeneity knob and nothing else.
_BUDGET = {"rounds": 50, "participation_rate": 1.0, "server_lr": 0.003}

# Copied verbatim from configs/icaif_capacity_separation.yaml, minus that grid's
# wide-model override: this run holds capacity at the reference MLP so the
# effects are comparable to the headline RQ3 numbers rather than to the
# capacity check.
ARMS: dict[str, dict] = {
    "feature_heterogeneity_severe": {
        "feature_heterogeneity": {
            "enabled": True,
            "common_fraction": 0.15,
            "private_fraction": 0.35,
            "shared_private_selection": False,
        },
        "federated": dict(_BUDGET),
    },
    "feature_heterogeneity_severe_matched": {
        "feature_heterogeneity": {
            "enabled": True,
            "common_fraction": 0.15,
            "private_fraction": 0.35,
            "shared_private_selection": True,
        },
        "federated": dict(_BUDGET),
    },
    "covariate_shift_strong": {
        "statistical_heterogeneity": {
            "enabled": True,
            "client_size_imbalance": {"enabled": False},
            "client_size_matched_control": {"enabled": False},
            "label_skew": {"enabled": False},
            "covariate_shift": {
                "enabled": True,
                "location_std": 0.5,
                "log_scale_std": 0.3,
            },
            "covariate_shift_matched_control": {"enabled": False},
        },
        "federated": dict(_BUDGET),
    },
    "covariate_shift_strong_matched": {
        "statistical_heterogeneity": {
            "enabled": True,
            "client_size_imbalance": {"enabled": False},
            "client_size_matched_control": {"enabled": False},
            "label_skew": {"enabled": False},
            "covariate_shift": {"enabled": False},
            "covariate_shift_matched_control": {
                "enabled": True,
                "location_std": 0.5,
                "log_scale_std": 0.3,
            },
        },
        "federated": dict(_BUDGET),
    },
    "temporal_block_outages_identified": {
        "temporal_heterogeneity": {
            "enabled": True,
            "mode": "block",
            "block_fraction": 0.40,
        },
        "federated": dict(_BUDGET),
    },
    "temporal_random_matched_block_outages": {
        "temporal_heterogeneity": {
            "enabled": True,
            "mode": "matched_random",
            "matched_reference": {"mode": "block", "block_fraction": 0.40},
        },
        "federated": dict(_BUDGET),
    },
}


def scenario_name(arm: str, test_year: int) -> str:
    """`{arm}__refit_{year}`. Parsed back by robustness.rq3_rolling_contrasts."""
    return f"{arm}__refit_{test_year}"


def window_years(
    first_year: int = FIRST_TEST_YEAR,
    last_year: int = LAST_TEST_YEAR,
    step: int = 2,
) -> list[int]:
    return list(range(first_year, last_year + 1, step))


def build_config(
    first_year: int = FIRST_TEST_YEAR,
    last_year: int = LAST_TEST_YEAR,
    validation_years: int = VALIDATION_YEARS,
    step: int = 2,
    *,
    parallel_runs: int = 2,
    seeds: list[int] | None = None,
) -> dict:
    scenarios = []
    for test_year in window_years(first_year, last_year, step):
        split = {
            "train_end": f"{test_year - validation_years - 1}-12-31",
            "val_end": f"{test_year - 1}-12-31",
            "test_end": f"{test_year + step - 1}-12-31",
        }
        for arm, overrides in ARMS.items():
            scenarios.append(
                {
                    "name": scenario_name(arm, test_year),
                    "overrides": {**overrides, "split": split},
                }
            )
    return {
        "base_config": "crsp_compustat_profitability.yaml",
        "output_dir": "outputs/icaif_publication/rq3_rolling",
        "continue_on_error": False,
        "preload_panel_once": True,
        "resume_completed": True,
        "base_overrides": {
            # Only the three federated methods carry the dispersion contrast.
            # Dropping the centralized and local baselines is what makes this
            # grid affordable at all -- local alone is ~24% of a cell.
            "baselines": {
                "centralized": False,
                "centralized_masked": False,
                "local": False,
            },
            "federated": {"algorithms": ["fedavg", "scaffold", "fedadam"]},
        },
        "grid_execution": {
            "parallel_runs": parallel_runs,
            "max_total_client_workers": 3 * parallel_runs,
            "max_total_torch_threads": 3 * parallel_runs,
            "start_method": "spawn",
            "parallel_child_console": False,
        },
        "child_logging": {"console": True, "round_interval": 10},
        # Three seeds, matching the rolling economic grid and for the same
        # reason: with ~408 test months the binding constraint is the market
        # history, not optimization noise, and the unit of inference is the
        # calendar month either way.
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
        / "icaif_rq3_rolling.yaml",
    )
    parser.add_argument("--first-year", type=int, default=FIRST_TEST_YEAR)
    parser.add_argument("--last-year", type=int, default=LAST_TEST_YEAR)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--parallel-runs", type=int, default=2)
    args = parser.parse_args()

    config = build_config(
        args.first_year,
        args.last_year,
        step=args.step,
        parallel_runs=args.parallel_runs,
    )
    n_windows = len(window_years(args.first_year, args.last_year, args.step))
    header = (
        "# GENERATED by scripts/make_rq3_rolling_config.py -- do not edit by hand.\n"
        f"# {n_windows} expanding-window refits x {len(ARMS)} arms, "
        f"{len(config['scenarios']) * len(config['seeds'])} cells, "
        f"~{n_windows * 12 * args.step} out-of-sample months.\n"
    )
    args.output.write_text(
        header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(
        f"Wrote {args.output} ({n_windows} windows x {len(ARMS)} arms "
        f"x {len(config['seeds'])} seeds = "
        f"{len(config['scenarios']) * len(config['seeds'])} cells)"
    )


if __name__ == "__main__":
    main()

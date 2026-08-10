"""Aggregation for the reviewer-requested robustness grid.

`configs/icaif_robustness.yaml` answers four questions raised in review, each by
comparing a variant scenario against the reference protocol
(`core/ff12_full_participation`) that the main results use:

1. Is the 2,625-parameter MLP the right base learner, or does a linear/ridge
   model do as well?                                  (R1 W1, R2 3.5, Q11)
2. Does the paper's story depend on model capacity?   (R1 W1, Q1)
3. Is FedProx distinguishable from FedAvg at any mu?  (R1 W5, Q4)
4. Do the economic results survive realistic costs?   (R1 W10, R2 3.6, Q10)
5. Is the local-only baseline undertuned?             (R1 W6, R2 Q12)
6. Is the reported economic performance a beta artifact rather than genuine
   alpha?                                             (R1 W10)

Each table is emitted as a paired comparison rather than a bare level, because
the reviewer questions are all of the form "does X change the conclusion?".
Differences are paired by seed so cross-seed noise -- which is large here, with
Sharpe standard deviations of 0.2-0.5 -- cancels between arms.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from fapbench import publication
from fapbench.evaluation.significance import (
    aggregate_seed_significance,
    combine_seed_tests,
    rolling_dispersion_contrasts,
    seed_averaged_dispersion_contrasts,
)
from fapbench.publication import (
    HETEROGENEITY_MATCHED_CONTROL_PAIRS,
    IDENTIFICATION_PAIRS,
)
from fapbench.evaluation.statistics import newey_west_ols

REFERENCE_PHASE = "core"
REFERENCE_SCENARIO = "ff12_full_participation"
ROBUSTNESS_PHASE = "robustness"

# Metrics carried through every robustness table.
METRICS = [
    "rank_ic",
    "oos_r2_zero",
    "client_worst_rank_ic",
    "client_neutral_portfolio_net_sharpe",
]


def _phase_frame(output_root: Path, phase: str) -> pd.DataFrame:
    path = Path(output_root) / phase / "grid_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _paired_difference(
    variant: pd.DataFrame, reference: pd.DataFrame, metric: str
) -> dict[str, Any]:
    """Seed-paired difference (variant minus reference) for one metric.

    Pairing by seed removes the shared market history and the shared
    initialization draw, so the resulting t-test is over seeds only. With ten
    seeds this is a weak test by construction; it is reported alongside the
    effect size, never instead of it.
    """
    if metric not in variant.columns or metric not in reference.columns:
        return {}
    left = variant.set_index("seed")[metric]
    right = reference.set_index("seed")[metric]
    common = sorted(set(left.index) & set(right.index))
    if len(common) < 2:
        return {}
    diff = (left.loc[common] - right.loc[common]).astype(float).dropna()
    if diff.empty:
        return {}
    result = {
        f"{metric}_variant": float(left.loc[common].mean()),
        f"{metric}_reference": float(right.loc[common].mean()),
        f"{metric}_delta": float(diff.mean()),
    }
    # A difference that is constant to within floating-point noise carries no
    # information about sampling variability, but ttest_1samp will happily
    # return something like 1e-63 for it. Guard on relative spread, not on
    # exact zero variance, so degenerate comparisons report NaN instead of an
    # absurd p-value.
    spread = float(diff.std(ddof=1))
    scale = max(abs(float(diff.mean())), 1e-12)
    if len(diff) >= 2 and spread > 1e-9 * scale:
        result[f"{metric}_p"] = float(sps.ttest_1samp(diff, 0.0).pvalue)
    else:
        result[f"{metric}_p"] = float("nan")
    return result


def compare_against_reference(
    output_root: Path,
    scenarios: list[tuple[str, str]],
    *,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Seed-paired comparison of robustness scenarios against the core reference.

    ``scenarios`` is a list of (scenario_name, label) pairs.
    """
    output_root = Path(output_root)
    robustness = _phase_frame(output_root, ROBUSTNESS_PHASE)
    reference = _phase_frame(output_root, REFERENCE_PHASE)
    if robustness.empty or reference.empty:
        return pd.DataFrame()
    reference = reference[reference["scenario"] == REFERENCE_SCENARIO]
    rows: list[dict[str, Any]] = []
    for scenario, label in scenarios:
        variant_all = robustness[robustness["scenario"] == scenario]
        if variant_all.empty:
            continue
        available = (
            methods
            if methods is not None
            else sorted(set(variant_all["method"]) & set(reference["method"]))
        )
        for method in available:
            variant = variant_all[variant_all["method"] == method]
            ref = reference[reference["method"] == method]
            if variant.empty or ref.empty:
                continue
            row: dict[str, Any] = {
                "comparison": label,
                "scenario": scenario,
                "method": method,
                "n_seeds": int(
                    len(set(variant["seed"]) & set(ref["seed"]))
                ),
            }
            for metric in METRICS:
                row.update(_paired_difference(variant, ref, metric))
            rows.append(row)
    return pd.DataFrame(rows)


def fedprox_mu_sweep(output_root: Path) -> pd.DataFrame:
    """Is FedProx distinguishable from FedAvg at any mu?

    Compares FedProx under each swept mu against FedAvg *from the same run*, so
    the two arms share data, initialization, and participation schedule. If
    every mu gives a difference indistinguishable from zero, FedProx is a no-op
    on this task and should be reported as such rather than printed as a
    separate method.
    """
    frame = _phase_frame(Path(output_root), ROBUSTNESS_PHASE)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for scenario in sorted(
        s for s in frame["scenario"].unique() if str(s).startswith("fedprox_mu_")
    ):
        subset = frame[frame["scenario"] == scenario]
        prox = subset[subset["method"] == "fedprox"]
        avg = subset[subset["method"] == "fedavg"]
        if prox.empty or avg.empty:
            continue
        mu = str(scenario).replace("fedprox_mu_", "").replace("_", ".")
        row: dict[str, Any] = {
            "scenario": scenario,
            "mu": mu,
            "n_seeds": int(len(set(prox["seed"]) & set(avg["seed"]))),
        }
        for metric in METRICS:
            row.update(_paired_difference(prox, avg, metric))
        rows.append(row)
    return pd.DataFrame(rows)


def transaction_cost_sensitivity(output_root: Path) -> pd.DataFrame:
    """Client-neutral net Sharpe at 10, 25 and 50 bps.

    The 10 bps column is the reference protocol; the others are the robustness
    scenarios, which retrain identically and differ only in the evaluation-side
    cost.
    """
    output_root = Path(output_root)
    robustness = _phase_frame(output_root, ROBUSTNESS_PHASE)
    reference = _phase_frame(output_root, REFERENCE_PHASE)
    if reference.empty:
        return pd.DataFrame()
    metric = "client_neutral_portfolio_net_sharpe"
    base = reference[reference["scenario"] == REFERENCE_SCENARIO]
    rows: list[dict[str, Any]] = []
    for method in sorted(base["method"].unique()):
        row: dict[str, Any] = {"method": method}
        ref_method = base[base["method"] == method]
        if metric in ref_method.columns:
            row["net_sharpe_10bps"] = float(ref_method[metric].mean())
        for scenario, label in (("cost_25bps", "25"), ("cost_50bps", "50")):
            if robustness.empty:
                continue
            variant = robustness[
                (robustness["scenario"] == scenario)
                & (robustness["method"] == method)
            ]
            if variant.empty or metric not in variant.columns:
                continue
            row[f"net_sharpe_{label}bps"] = float(variant[metric].mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and {"net_sharpe_10bps", "net_sharpe_50bps"} <= set(frame.columns):
        frame["breakeven_gap_10_to_50"] = (
            frame["net_sharpe_10bps"] - frame["net_sharpe_50bps"]
        )
    return frame


def local_tuning_effect(output_root: Path) -> pd.DataFrame:
    """Does tuning the local-only baseline change the 'federation helps' gap?

    Local models otherwise inherit the federated client hyperparameters, so a
    tuned local arm is the honest comparison. Reports tuned-vs-inherited local
    performance and the resulting gap to FedAvg.
    """
    output_root = Path(output_root)
    robustness = _phase_frame(output_root, ROBUSTNESS_PHASE)
    reference = _phase_frame(output_root, REFERENCE_PHASE)
    if robustness.empty or reference.empty:
        return pd.DataFrame()
    tuned = robustness[
        (robustness["scenario"] == "local_tuned") & (robustness["method"] == "local")
    ]
    inherited = reference[
        (reference["scenario"] == REFERENCE_SCENARIO) & (reference["method"] == "local")
    ]
    if tuned.empty or inherited.empty:
        return pd.DataFrame()
    row: dict[str, Any] = {
        "comparison": "local tuned minus local inherited",
        "n_seeds": int(len(set(tuned["seed"]) & set(inherited["seed"]))),
    }
    for metric in METRICS:
        row.update(_paired_difference(tuned, inherited, metric))
    # Gap that the RQ1 "federation helps" reading depends on.
    fedavg = reference[
        (reference["scenario"] == REFERENCE_SCENARIO) & (reference["method"] == "fedavg")
    ]
    for label, local_arm in (("inherited", inherited), ("tuned", tuned)):
        if not fedavg.empty and "rank_ic" in fedavg.columns:
            row[f"fedavg_minus_local_{label}_rank_ic"] = float(
                fedavg["rank_ic"].mean() - local_arm["rank_ic"].mean()
            )
    return pd.DataFrame([row])


def ranking_disagreement_test(
    output_root: Path,
    *,
    phase: str = REFERENCE_PHASE,
    scenario: str = REFERENCE_SCENARIO,
    methods: tuple[str, ...] = (
        "centralized", "local", "fedavg", "fedprox", "scaffold", "fedadam",
    ),
    block_length: int = 6,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Is the cross-metric ranking disagreement larger than estimation noise?

    RQ1 claims predictive and economic metrics rank methods differently. A
    reviewer's objection is that a disagreement between two noisily-estimated
    orderings can be produced by noise alone, so the claim needs a test rather
    than a description of one realization.

    Test: if both metrics induced a single common ordering, Kendall's tau
    between them would be 1, and sampling noise could only pull an estimate
    below it. We resample the 60 test months with a moving-block bootstrap --
    the *same* month indices for every method, so cross-method correlation is
    preserved -- recompute each method's full-sample mean Rank IC and net
    Sharpe, and recompute tau. If the bootstrap interval for tau excludes 1,
    the two metrics do not induce the same ordering, beyond noise.

    Monthly series are averaged across seeds first, matching the unit of
    inference used for the matched-control contrasts.
    """
    output_root = Path(output_root)
    scenario_dir = output_root / phase / scenario
    if not scenario_dir.exists():
        return pd.DataFrame()

    rank_ic: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}
    for method in methods:
        ic_series, ret_series = [], []
        for seed_dir in sorted(scenario_dir.glob("seed_*")):
            ic_path = seed_dir / f"monthly_predictive_metrics_{method}.csv"
            ret_path = seed_dir / f"portfolio_returns_client_neutral_{method}.csv"
            if not ic_path.exists() or not ret_path.exists():
                continue
            ic_series.append(
                pd.read_csv(ic_path, parse_dates=["target_date"])
                .set_index("target_date")["rank_ic"]
            )
            ret_series.append(
                pd.read_csv(ret_path, parse_dates=["target_date"])
                .set_index("target_date")["net_return"]
            )
        if not ic_series or not ret_series:
            continue
        rank_ic[method] = pd.concat(ic_series, axis=1).mean(axis=1)
        returns[method] = pd.concat(ret_series, axis=1).mean(axis=1)
    if len(rank_ic) < 3:
        return pd.DataFrame()

    ic_frame = pd.DataFrame(rank_ic).dropna()
    ret_frame = pd.DataFrame(returns).dropna()
    months = ic_frame.index.intersection(ret_frame.index)
    ic_frame, ret_frame = ic_frame.loc[months], ret_frame.loc[months]
    n = len(months)
    if n < block_length + 1:
        return pd.DataFrame()

    def _tau(ic_block: pd.DataFrame, ret_block: pd.DataFrame) -> float:
        mean_ic = ic_block.mean()
        sharpe = ret_block.mean() / ret_block.std(ddof=1) * np.sqrt(12.0)
        if not np.isfinite(sharpe).all():
            return float("nan")
        return float(sps.kendalltau(mean_ic, sharpe).statistic)

    observed = _tau(ic_frame, ret_frame)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    draws = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        index = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        draws[b] = _tau(ic_frame.iloc[index], ret_frame.iloc[index])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return pd.DataFrame()
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return pd.DataFrame([{
        "scenario": scenario,
        "n_methods": int(ic_frame.shape[1]),
        "n_months": int(n),
        "kendall_tau": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        # The null of one common ordering implies tau = 1. Decide exclusion from
        # the share of draws at tau = 1, not from `upper < 1.0`: Kendall's tau on
        # K methods is discrete (multiples of 2/(K(K-1))), so a percentile upper
        # bound can land on 1.0 and still be reported as strictly below it by
        # one ulp. On the investable universe that floating-point accident
        # reported "excludes perfect agreement" while 4.7% of draws -- well above
        # the 2.5% a two-sided 95% interval allows -- actually reached tau = 1.
        "excludes_perfect_agreement": bool(
            float(np.mean(draws >= 1.0 - 1e-12)) < 0.025
        ),
        "share_of_draws_at_perfect_agreement": float(np.mean(draws >= 1.0 - 1e-12)),
        "block_length": int(block_length),
        "n_bootstrap": int(draws.size),
    }])


def metric_rank_agreement(
    output_root: Path,
    *,
    scenarios: tuple[tuple[str, str, str], ...] = (
        (REFERENCE_PHASE, REFERENCE_SCENARIO, "reference MLP [64,32]"),
        (ROBUSTNESS_PHASE, "linear_ridge_weak", "linear/ridge"),
        (ROBUSTNESS_PHASE, "wide_mlp", "wide MLP [256,128]"),
    ),
) -> pd.DataFrame:
    """Does RQ1's metric disagreement survive a change of model class?

    RQ1 claims that predictive and economic metrics rank methods differently.
    Because the linear model beats the reference MLP outright, a reader can
    reasonably ask whether RQ1 is an artifact of a badly-chosen architecture.
    This table answers that directly: for each model class it reports the
    Spearman correlation between mean Rank IC and mean client-neutral net Sharpe
    across methods, and whether one method tops both. Disagreement that holds at
    every model class is a property of the evaluation, not of the architecture.
    """
    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for phase, scenario, label in scenarios:
        frame = _phase_frame(output_root, phase)
        if frame.empty:
            continue
        subset = frame[frame["scenario"] == scenario]
        needed = {"method", "rank_ic", "client_neutral_portfolio_net_sharpe"}
        if subset.empty or not needed.issubset(subset.columns):
            continue
        table = subset.groupby("method")[
            ["rank_ic", "client_neutral_portfolio_net_sharpe"]
        ].mean()
        if len(table) < 3:
            continue
        correlation = sps.spearmanr(
            table["rank_ic"], table["client_neutral_portfolio_net_sharpe"]
        ).statistic
        best_rank_ic = str(table["rank_ic"].idxmax())
        best_sharpe = str(table["client_neutral_portfolio_net_sharpe"].idxmax())
        rows.append(
            {
                "model_class": label,
                "scenario": scenario,
                "n_methods": int(len(table)),
                "spearman_rank_ic_vs_net_sharpe": float(correlation),
                "best_rank_ic_method": best_rank_ic,
                "best_net_sharpe_method": best_sharpe,
                "same_method_wins_both": bool(best_rank_ic == best_sharpe),
            }
        )
    return pd.DataFrame(rows)


def market_adjusted_alpha(
    output_root: Path,
    panel_path: Path | None,
    *,
    methods: tuple[str, ...] = ("centralized", "fedavg", "fedadam", "scaffold"),
    return_column: str = "retadj_1mn",
    date_column: str = "target_date",
) -> pd.DataFrame:
    """Split reference-protocol net returns into market exposure and alpha.

    The long--short decile construction is often assumed to be market neutral,
    which would make a raw net Sharpe automatically an alpha. That assumption is
    testable: regress each method's seed-averaged monthly client-neutral net
    return on an equal-weighted proxy for the panel's own market return and read
    off ``beta``. This is a first-order check, not a full factor model -- the
    proxy is the equal-weighted own-universe return, not the value-weighted
    market or an FF5+momentum set -- so it bounds the beta artifact rather than
    pricing the strategy.

    Returns an empty frame when ``panel_path`` is None or missing, so the
    publication build stays runnable without the licensed panel.
    """
    output_root = Path(output_root)
    if panel_path is None or not Path(panel_path).exists():
        return pd.DataFrame()
    panel = pd.read_parquet(Path(panel_path), columns=[date_column, return_column])
    market = (
        panel.groupby(date_column)[return_column].mean().rename("market_return")
    )
    market.index = pd.to_datetime(market.index)

    scenario_dir = output_root / REFERENCE_PHASE / REFERENCE_SCENARIO
    rows: list[dict[str, Any]] = []
    for method in methods:
        series: list[pd.Series] = []
        for seed_dir in sorted(scenario_dir.glob("seed_*")):
            path = seed_dir / f"portfolio_returns_client_neutral_{method}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, parse_dates=["target_date"])
            series.append(frame.set_index("target_date")["net_return"])
        if not series:
            continue
        # Seeds share one market history, so average them into a single monthly
        # series before testing -- the same unit of inference the matched-control
        # contrasts use.
        averaged = pd.concat(series, axis=1).mean(axis=1)
        aligned = pd.concat(
            [averaged.rename("strategy"), market], axis=1, join="inner"
        ).dropna()
        if aligned.empty:
            continue
        fit = newey_west_ols(aligned["strategy"], aligned["market_return"])
        rows.append(
            {
                "method": method,
                "n_seeds": len(series),
                "n_months": fit["n_periods"],
                "mean_net_return_annualized": float(aligned["strategy"].mean() * 12.0),
                "market_return_annualized": float(aligned["market_return"].mean() * 12.0),
                "beta": fit["beta"],
                "beta_p": fit["beta_p"],
                "alpha_monthly": fit["alpha"],
                "alpha_annualized": float(fit["alpha"] * 12.0)
                if np.isfinite(fit["alpha"])
                else np.nan,
                "alpha_p": fit["alpha_p"],
                "r_squared": fit["r_squared"],
            }
        )
    return pd.DataFrame(rows)


# Matched-control pairs re-run at [256, 128] by
# configs/icaif_capacity_separation.yaml. Two data-level and one system-level,
# chosen to span RQ3's separation claim rather than to be exhaustive: if the
# separation is a property of the heterogeneity families rather than of the
# 2,625-parameter model, the first two must stay an order of magnitude above the
# third at 13x the capacity.
CAPACITY_PAIRS: list[tuple[str, str, str]] = [
    (
        "feature_heterogeneity_severe",
        "feature_heterogeneity_severe_matched",
        "feature severe",
    ),
    (
        "covariate_shift_strong",
        "covariate_shift_strong_matched",
        "covariate strong",
    ),
    (
        "temporal_block_outages_identified",
        "temporal_random_matched_block_outages",
        "block outages",
    ),
]
CAPACITY_PHASE = "capacity_separation"


def capacity_separation_contrasts(
    output_root: Path,
    *,
    methods: Sequence[str] = ("fedavg", "scaffold", "fedadam"),
) -> pd.DataFrame:
    """Re-measure RQ3's matched-control dispersion contrasts at higher capacity.

    R1's W1 calls this the load-bearing robustness check, and the `wide_mlp`
    scenario does not answer it: that widens the model under the *reference*
    protocol, showing whether method rankings move with capacity, not whether
    the matched-control separation does. The concern is that a small model may
    converge to nearly the same point regardless of update schedule, which would
    make "system-level heterogeneity has a small footprint" a statement about
    capacity rather than about federated learning.

    Uses the identical estimator as the reference-capacity contrasts
    (seed-averaged monthly differences, HAC + moving-block bootstrap), so the two
    are directly comparable. Empty until the grid has been run.
    """
    return _phase_dispersion_contrasts(
        output_root, CAPACITY_PHASE, CAPACITY_PAIRS, methods=methods
    )


def _phase_dispersion_contrasts(
    output_root: Path,
    phase: str,
    pairs: Sequence[tuple[str, str, str]],
    *,
    methods: Sequence[str] = ("fedavg", "scaffold", "fedadam"),
) -> pd.DataFrame:
    """Re-measure matched-control dispersion contrasts inside one robustness phase.

    Every RQ3 robustness grid answers the same question at a different setting --
    higher capacity, the investable universe, a linear model, escalated severity --
    so they must all be measured with the *identical* estimator as the
    reference-capacity contrasts (seed-averaged monthly differences, HAC plus
    moving-block bootstrap) or the comparison of magnitudes is meaningless. Sharing
    one function is what guarantees that. Empty until the grid has been run.
    """
    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    resolved = [
        (phase_dir / treatment, phase_dir / control, label)
        for treatment, control, label in pairs
        if (phase_dir / treatment).is_dir() and (phase_dir / control).is_dir()
    ]
    if not resolved:
        return pd.DataFrame()
    return seed_averaged_dispersion_contrasts(
        resolved, methods=list(methods), statistics=("std",)
    )


# RQ3 at 408 months. Same three pairs as CAPACITY_PAIRS -- the set that spans the
# separation claim -- but refit over expanding windows so the dispersion contrast
# is tested on ~408 monthly observations instead of the reference protocol's 60.
# The tighter covariate-shift control. Cross-phase: the treatment lives in the
# heterogeneity grid and only the control arm is new, so this cannot use
# _phase_dispersion_contrasts (which resolves both sides inside one phase).
SECOND_MOMENT_TREATMENT = ("heterogeneity", "covariate_shift_strong")
SECOND_MOMENT_CONTROL = (
    "second_moment_control",
    "covariate_shift_strong_second_moment",
)


def second_moment_control_contrasts(
    output_root: Path,
    *,
    methods: Sequence[str] = ("fedavg", "scaffold", "fedadam"),
) -> pd.DataFrame:
    """Covariate-shift dispersion contrast against a variance-matched control.

    The published contrast controls the pooled first moment only, which leaves a
    22% pooled-variance gap at the strong setting and makes the reported effect an
    upper bound of unstated width. This re-measures it against a control matching
    both moments, using the identical estimator so the two numbers are directly
    comparable: whatever survives here is not attributable to the looser match.
    """
    root = Path(output_root)
    treatment = root / SECOND_MOMENT_TREATMENT[0] / SECOND_MOMENT_TREATMENT[1]
    control = root / SECOND_MOMENT_CONTROL[0] / SECOND_MOMENT_CONTROL[1]
    if not treatment.is_dir() or not control.is_dir():
        return pd.DataFrame()
    return seed_averaged_dispersion_contrasts(
        [(treatment, control, "covariate strong (second-moment control)")],
        methods=list(methods),
        statistics=("std",),
    )


RQ3_ROLLING_PHASE = "rq3_rolling"
RQ3_ROLLING_PAIRS = CAPACITY_PAIRS


def rq3_rolling_contrasts(
    output_root: Path,
    *,
    pairs: Sequence[tuple[str, str, str]] = RQ3_ROLLING_PAIRS,
    methods: Sequence[str] = ("fedavg", "scaffold", "fedadam"),
    phase: str = RQ3_ROLLING_PHASE,
) -> pd.DataFrame:
    """Matched-control dispersion contrasts pooled across expanding-window refits.

    The grid writes one scenario directory per (arm, window) as
    ``{arm}__refit_{year}``; this discovers the windows present for each pair and
    hands them to ``rolling_dispersion_contrasts``, which concatenates each
    contrast's per-window monthly difference series before testing. A window is
    used only when *both* of its arms are present, so a partially-complete grid
    yields a shorter series rather than a mismatched one.
    """
    phase_dir = Path(output_root) / phase
    if not phase_dir.exists():
        return pd.DataFrame()

    labelled: dict[str, list[tuple[Path, Path]]] = {}
    for treatment, control, label in pairs:
        windows: list[tuple[Path, Path]] = []
        for treatment_dir in sorted(phase_dir.glob(f"{treatment}__refit_*")):
            year = treatment_dir.name.rsplit("__refit_", 1)[-1]
            control_dir = phase_dir / f"{control}__refit_{year}"
            if treatment_dir.is_dir() and control_dir.is_dir():
                windows.append((treatment_dir, control_dir))
        if windows:
            labelled[label] = windows
    if not labelled:
        return pd.DataFrame()
    return rolling_dispersion_contrasts(
        labelled, methods=list(methods), statistics=("std",)
    )


# RQ3 survives the reference screen: the severe feature contrast re-run on the
# investable universe rather than the unrestricted one the heterogeneity grid used.
RQ3_INVESTABLE_PAIRS: list[tuple[str, str, str]] = [
    (
        "feature_heterogeneity_severe",
        "feature_heterogeneity_severe_matched",
        "feature severe (investable)",
    ),
]

# The same three pairs that span the separation claim, refit under linear/ridge.
# Reusing CAPACITY_PAIRS' labels is deliberate: the two tables are meant to be
# read against each other row for row.
LINEAR_SEPARATION_PAIRS = CAPACITY_PAIRS

# Escalation ladders. The objection is that the two families are dialled in
# incommensurable units, so nothing establishes that "severe" masking and a 40%
# outage block sit at comparable severities. The answer is constructive: push the
# system-level dials up (outages 40% -> 60% -> 80%, imbalance r=20 -> 50 -> 100)
# and report where they land. Each rung carries its own matched control.
SEVERITY_ESCALATION_PAIRS: list[tuple[str, str, str]] = [
    ("temporal_block_outages_060", "temporal_random_matched_block_060",
     "block outages 60%"),
    ("temporal_block_outages_080", "temporal_random_matched_block_080",
     "block outages 80%"),
    ("size_imbalance_ratio_50", "size_uniform_matched_ratio_50",
     "size imbalance r=50"),
    ("size_imbalance_ratio_100", "size_uniform_matched_ratio_100",
     "size imbalance r=100"),
]


def block_length_sensitivity(
    output_root: Path,
    *,
    block_lengths: Sequence[int] = (3, 6, 12),
    methods: Sequence[str] = ("fedavg", "scaffold", "fedadam"),
) -> pd.DataFrame:
    """Re-run every matched-control dispersion contrast at several block lengths.

    A reviewer's objection: L=6 on n=60 leaves only ~10 effective blocks, which is
    thin for a 95% interval -- and RQ3's central claim is a comparison of interval
    *widths* across two regimes, so width is not incidental here. Reporting one L
    and asserting robustness is not enough; the comparison has to be shown to
    survive the choice.

    Emits one row per (contrast, method, statistic, L) with the interval width, so
    both the per-contrast sensitivity and the regime gap can be read off.
    """
    output_root = Path(output_root)
    pairs: list[tuple[Path, Path, str]] = []
    for treatment, control, label in HETEROGENEITY_MATCHED_CONTROL_PAIRS:
        pairs.append((
            output_root / "heterogeneity" / treatment,
            output_root / "heterogeneity" / control,
            label,
        ))
    for treatment, control, label in IDENTIFICATION_PAIRS:
        pairs.append((
            output_root / "identification" / treatment,
            output_root / "identification" / control,
            label,
        ))
    pairs = [p for p in pairs if p[0].is_dir() and p[1].is_dir()]
    if not pairs:
        return pd.DataFrame()

    frames = []
    for length in block_lengths:
        frame = seed_averaged_dispersion_contrasts(
            pairs, methods=list(methods), statistics=("std",), block_length=length
        )
        if frame.empty:
            continue
        frame = frame.copy()
        frame["ci_width"] = frame["ci_upper"] - frame["ci_lower"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


ENSEMBLE_PHASE = "ensemble_local"
# Both universes, because RQ2's own finding is that universe construction is the
# largest lever in the benchmark: an ensemble-versus-federation verdict that held
# on only one universe would be exactly the protocol-contingent claim the paper
# warns against.
ENSEMBLE_SCENARIOS = ("ff12_exclude_microcaps", "ff12_full_participation")
# Every federated method plus the two things the ensemble has to be read against:
# `local`, which shares its training and differs only in aggregation, and
# `centralized`, the pooled upper bound.
# `centralized` is here because it is the paper's stated economic target ("a net
# Sharpe distinguishable from Centralized's on the investable universe"), so the
# ensemble has to be tested against it and not only against the federated rows.
ENSEMBLE_REFERENCE_METHODS = ("fedavg", "local", "centralized")
# The metrics that are well defined for a *ranking*. Squared-error metrics are
# deliberately absent: the ensemble's prediction is a cross-sectional percentile,
# not a return, so the runner writes them as NaN for that row.
ENSEMBLE_LEVEL_METRICS = (
    "rank_ic",
    "client_neutral_portfolio_net_sharpe",
    "global_portfolio_net_sharpe",
    "client_worst_rank_ic",
    "selected_checkpoint_communicated_bytes_fp32",
)


def ensemble_local_comparison(output_root: Path) -> pd.DataFrame:
    """Is federation better than averaging the K locally-trained models?

    The paper compares federated training against centralized and local-only
    training but not against the cheapest alternative: train locally, ship the K
    finished models once, average their opinions, coordinate nothing. If that
    matched federated training the benchmark's central comparison would be far
    less interesting, so it is measured.

    Both reference methods matter and answer different questions. Against
    `fedavg`, the question is whether coordination earns its rounds. Against
    `local`, it is how much of the local baseline's deficit is a *training*
    problem rather than an *aggregation* problem -- the ensemble shares the local
    baseline's models exactly, so the difference between those two rows is
    aggregation alone.

    Every method is trained inside the same run as the ensemble, so these are
    exactly paired: same splits, same client assignment, same market history, no
    cross-grid assumption. Empty until the grid has been run.
    """
    phase_dir = Path(output_root) / ENSEMBLE_PHASE
    if not phase_dir.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for scenario in ENSEMBLE_SCENARIOS:
        scenario_dir = phase_dir / scenario
        if not scenario_dir.is_dir():
            continue
        for reference_method in ENSEMBLE_REFERENCE_METHODS:
            per_seed_tests, _ = aggregate_seed_significance(
                scenario_dir,
                variant="client_neutral",
                reference_method=reference_method,
            )
            if per_seed_tests.empty:
                continue
            combined = combine_seed_tests(per_seed_tests)
            if combined.empty:
                continue
            combined = combined.copy()
            combined.insert(0, "scenario", scenario)
            frames.append(combined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ensemble_local_levels(output_root: Path) -> pd.DataFrame:
    """Seed-mean levels for the ensemble grid, so the test above has magnitudes.

    A combined p-value with no level beside it is unreadable: "the ensemble does
    not separate from FedAvg" means something different at Rank IC 0.01 than at
    0.05. Reported per scenario and method for the metrics that are well defined
    for a *ranking* -- squared-error metrics are blank for the ensemble row by
    construction, since its prediction is a percentile rather than a return.
    """
    # Read the per-seed files rather than the grid's aggregate `grid_results.csv`:
    # the aggregate is only written when the whole grid finishes, so depending on it
    # makes this table unavailable while a long grid is still running, and unusable
    # if the grid is resumed in pieces.
    phase_dir = Path(output_root) / ENSEMBLE_PHASE
    if not phase_dir.exists():
        return pd.DataFrame()
    per_seed: list[pd.DataFrame] = []
    for scenario in ENSEMBLE_SCENARIOS:
        for seed_dir in sorted((phase_dir / scenario).glob("seed_*")):
            results_path = seed_dir / "benchmark_results.csv"
            if not results_path.exists():
                continue
            # Subset before adding columns. benchmark_results.csv carries a few
            # hundred columns and only five are aggregated here, so narrowing
            # first is both faster and avoids the fragmentation warning that any
            # column addition on a frame that wide provokes.
            seed_frame = pd.read_csv(results_path)
            keep = ["method"] + [
                column for column in ENSEMBLE_LEVEL_METRICS
                if column in seed_frame.columns
            ]
            per_seed.append(
                seed_frame[keep].assign(scenario=scenario, seed_dir=seed_dir.name)
            )
    if not per_seed:
        return pd.DataFrame()
    frame = pd.concat(per_seed, ignore_index=True)
    metrics = [c for c in ENSEMBLE_LEVEL_METRICS if c in frame.columns]
    if not metrics:
        return pd.DataFrame()
    grouped = (
        frame.groupby(["scenario", "method"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def build_robustness_outputs(
    output_root: Path, panel_path: Path | None = None
) -> dict[str, pd.DataFrame]:
    """Emit every robustness table. Missing scenarios yield empty frames."""
    output_root = Path(output_root)
    tables = {
        "robustness_capacity_separation": capacity_separation_contrasts(output_root),
        # Named without the `robustness_` prefix because the manuscript cites them
        # as RQ3 results, not as robustness appendix material, and the manuscript
        # audit binds these exact filenames.
        "rq3_investable": _phase_dispersion_contrasts(
            output_root, "rq3_investable", RQ3_INVESTABLE_PAIRS
        ),
        "linear_separation": _phase_dispersion_contrasts(
            output_root, "linear_separation", LINEAR_SEPARATION_PAIRS
        ),
        "severity_escalation": _phase_dispersion_contrasts(
            output_root, "severity_escalation", SEVERITY_ESCALATION_PAIRS
        ),
        "rq3_rolling": rq3_rolling_contrasts(output_root),
        "second_moment_control": second_moment_control_contrasts(output_root),
        "ensemble_local_comparison": ensemble_local_comparison(output_root),
        "ensemble_local_levels": ensemble_local_levels(output_root),
        "robustness_block_length_sensitivity": block_length_sensitivity(output_root),
        "robustness_market_adjusted_alpha": market_adjusted_alpha(
            output_root, panel_path
        ),
        "robustness_metric_rank_agreement": metric_rank_agreement(output_root),
        # Run on both universes. The first row is the headline protocol (the
        # investable universe); the second shows the disagreement is not an
        # artifact of that screen. REFERENCE_SCENARIO in this module stays the
        # unrestricted universe because every *other* robustness variant was run
        # there and must be differenced against a matching arm -- only this test,
        # which is a property of the reference protocol itself, follows the
        # manuscript's headline.
        "robustness_ranking_disagreement_test": pd.concat(
            [
                ranking_disagreement_test(
                    output_root, scenario=publication.REFERENCE_SCENARIO
                ),
                ranking_disagreement_test(
                    output_root, scenario=publication.UNRESTRICTED_SCENARIO
                ),
            ],
            ignore_index=True,
        ),
        "robustness_model_class": compare_against_reference(
            output_root,
            [
                ("linear_ridge_weak", "linear/ridge (weak penalty) minus MLP"),
                ("linear_ridge_strong", "linear/ridge (strong penalty) minus MLP"),
                ("wide_mlp", "wide MLP minus reference MLP"),
            ],
        ),
        "robustness_fedprox_mu": fedprox_mu_sweep(output_root),
        "robustness_transaction_costs": transaction_cost_sensitivity(output_root),
        "robustness_local_tuning": local_tuning_effect(output_root),
    }
    tables_dir = output_root / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    return tables

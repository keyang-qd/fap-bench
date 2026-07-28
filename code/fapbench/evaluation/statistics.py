from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


def _bartlett_max_lag(n: int, max_lag: int | None) -> int:
    """Newey--West automatic bandwidth, shared by the mean test and the OLS fit."""
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    return max(0, min(int(max_lag), n - 1))


def newey_west_ols(
    y: Iterable[float],
    x: Iterable[float],
    max_lag: int | None = None,
) -> dict[str, float | int]:
    """OLS of ``y`` on a constant and ``x`` with Newey--West HAC standard errors.

    Used for the market-adjusted alpha: regressing a strategy's monthly net
    return on a market proxy separates the part of measured performance that is
    market exposure (``beta``) from the part that is not (``alpha``). Bartlett
    kernel and automatic bandwidth match :func:`newey_west_mean_test`, and
    p-values likewise use the normal approximation.
    """
    y_values = np.asarray(list(y), dtype=float)
    x_values = np.asarray(list(x), dtype=float)
    if len(y_values) != len(x_values):
        raise ValueError("y and x must have the same length.")
    finite = np.isfinite(y_values) & np.isfinite(x_values)
    y_values, x_values = y_values[finite], x_values[finite]
    n = len(y_values)
    empty = {
        "n_periods": int(n), "alpha": np.nan, "alpha_t": np.nan,
        "alpha_p": np.nan, "beta": np.nan, "beta_t": np.nan,
        "beta_p": np.nan, "r_squared": np.nan, "max_lag": 0,
    }
    if n < 3 or np.allclose(x_values, x_values[0]):
        return empty
    design = np.column_stack([np.ones(n), x_values])
    xtx = design.T @ design
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return empty
    coefficients = xtx_inv @ design.T @ y_values
    residuals = y_values - design @ coefficients

    max_lag = _bartlett_max_lag(n, max_lag)
    weighted = design * residuals[:, None]
    meat = weighted.T @ weighted
    for lag in range(1, max_lag + 1):
        cross = weighted[lag:].T @ weighted[:-lag]
        weight = 1.0 - lag / (max_lag + 1.0)
        meat += weight * (cross + cross.T)
    covariance = xtx_inv @ meat @ xtx_inv
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    total = float(np.sum((y_values - y_values.mean()) ** 2))
    stats: dict[str, float | int] = {
        "n_periods": int(n),
        "r_squared": float(1.0 - np.sum(residuals**2) / total) if total > 0 else np.nan,
        "max_lag": int(max_lag),
    }
    for name, index in (("alpha", 0), ("beta", 1)):
        estimate = float(coefficients[index])
        error = float(errors[index])
        t_statistic = estimate / error if error > 0 else float("nan")
        stats[name] = estimate
        stats[f"{name}_t"] = float(t_statistic)
        stats[f"{name}_p"] = (
            float(2.0 * norm.sf(abs(t_statistic))) if np.isfinite(t_statistic) else np.nan
        )
    return stats


def newey_west_mean_test(
    differences: Iterable[float],
    max_lag: int | None = None,
) -> dict[str, float | int]:
    """Two-sided HAC test that the mean paired return difference equals zero."""
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return {
            "n_periods": int(n),
            "mean_difference": np.nan,
            "hac_standard_error": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "max_lag": 0,
        }
    max_lag = _bartlett_max_lag(n, max_lag)
    centered = values - values.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for lag in range(1, max_lag + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n
        )
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = float(np.sqrt(long_run_variance / n))
    mean_difference = float(values.mean())
    t_statistic = (
        mean_difference / standard_error
        if standard_error > 0
        else float("nan")
    )
    p_value = (
        float(2.0 * norm.sf(abs(t_statistic)))
        if np.isfinite(t_statistic)
        else float("nan")
    )
    return {
        "n_periods": int(n),
        "mean_difference": mean_difference,
        "hac_standard_error": standard_error,
        "t_statistic": float(t_statistic),
        "p_value": p_value,
        "max_lag": int(max_lag),
    }


def annualized_sharpe(
    returns: Iterable[float], periods_per_year: int = 12
) -> float:
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    volatility = float(values.std(ddof=1))
    if volatility <= 0:
        return float("nan")
    return float(
        values.mean() / volatility * np.sqrt(float(periods_per_year))
    )


def moving_block_bootstrap_sharpe(
    returns: Iterable[float],
    *,
    periods_per_year: int = 12,
    block_length: int = 6,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Moving-block bootstrap confidence interval for annualized Sharpe."""
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return {
            "n_periods": int(n),
            "sharpe": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "block_length": int(block_length),
            "n_bootstrap": int(n_bootstrap),
        }
    block_length = max(1, min(int(block_length), n))
    n_bootstrap = max(1, int(n_bootstrap))
    rng = np.random.default_rng(seed)
    start_points = np.arange(n)
    bootstrap_values: list[float] = []
    n_blocks = int(np.ceil(n / block_length))
    for _ in range(n_bootstrap):
        sampled: list[float] = []
        starts = rng.choice(start_points, size=n_blocks, replace=True)
        for start in starts:
            indices = (start + np.arange(block_length)) % n
            sampled.extend(values[indices].tolist())
        bootstrap_values.append(
            annualized_sharpe(sampled[:n], periods_per_year)
        )
    distribution = np.asarray(bootstrap_values, dtype=float)
    distribution = distribution[np.isfinite(distribution)]
    alpha = 1.0 - float(confidence_level)
    return {
        "n_periods": int(n),
        "sharpe": annualized_sharpe(values, periods_per_year),
        "ci_lower": (
            float(np.quantile(distribution, alpha / 2.0))
            if len(distribution)
            else np.nan
        ),
        "ci_upper": (
            float(np.quantile(distribution, 1.0 - alpha / 2.0))
            if len(distribution)
            else np.nan
        ),
        "block_length": int(block_length),
        "n_bootstrap": int(n_bootstrap),
    }


def moving_block_bootstrap_mean(
    values: Iterable[float],
    *,
    block_length: int = 6,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Moving-block bootstrap confidence interval for the mean of a time series.

    Companion to ``moving_block_bootstrap_sharpe`` for series whose statistic of
    interest is the mean rather than the Sharpe ratio -- in particular the
    treatment-minus-control monthly difference series used by the matched-control
    contrasts. Blocks preserve the serial dependence that an i.i.d. bootstrap
    would destroy, which matters because monthly cross-client dispersion is
    persistent.
    """
    series = np.asarray(list(values), dtype=float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 3:
        return {
            "n_periods": int(n),
            "mean": float(series.mean()) if n else np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "block_length": int(block_length),
            "n_bootstrap": int(n_bootstrap),
        }
    block_length = max(1, min(int(block_length), n))
    n_bootstrap = max(1, int(n_bootstrap))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    draws = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        draws[i] = series[indices].mean()
    alpha = 1.0 - float(confidence_level)
    return {
        "n_periods": int(n),
        "mean": float(series.mean()),
        "ci_lower": float(np.quantile(draws, alpha / 2.0)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "block_length": int(block_length),
        "n_bootstrap": int(n_bootstrap),
    }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR adjustment.

    Returns q-values aligned with the input order. NaN inputs pass through as
    NaN and are excluded from the ranking, so a family containing an
    undefined test is still adjusted over the tests that are defined.
    """
    raw = np.asarray(list(p_values), dtype=float)
    q = np.full(raw.shape, np.nan, dtype=float)
    finite = np.isfinite(raw)
    if not finite.any():
        return q
    values = raw[finite]
    m = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    # Step-up: enforce monotonicity from the largest p-value downward.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    q[finite] = out
    return q


def _portfolio_path(output_dir: Path, method: str, variant: str) -> Path:
    variant = variant.lower()
    if variant == "primary":
        filename = f"portfolio_returns_{method}.csv"
    elif variant == "global":
        filename = f"portfolio_returns_global_{method}.csv"
    elif variant in {"client_neutral", "client-neutral"}:
        filename = f"portfolio_returns_client_neutral_{method}.csv"
    else:
        raise ValueError("variant must be primary, global, or client_neutral.")
    return output_dir / filename


def analyze_portfolio_results(
    output_dir: str | Path,
    *,
    variant: str = "primary",
    return_col: str = "net_return",
    date_col: str | None = None,
    reference_method: str | None = None,
    all_pairs: bool = False,
    periods_per_year: int = 12,
    hac_lag: int | None = None,
    block_length: int = 6,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    benchmark_path = output_dir / "benchmark_results.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {benchmark_path}")
    benchmark = pd.read_csv(benchmark_path)
    methods = benchmark["method"].astype(str).tolist()
    if not methods:
        raise ValueError("benchmark_results.csv contains no methods.")
    if reference_method is None:
        for candidate in ["centralized", "centralized_oracle", "fedavg"]:
            if candidate in methods:
                reference_method = candidate
                break
    if reference_method is None or reference_method not in methods:
        raise ValueError("A valid reference_method is required.")

    series: dict[str, pd.Series] = {}
    sharpe_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        path = _portfolio_path(output_dir, method, variant)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        resolved_date_col = date_col
        if resolved_date_col is None:
            candidates = [
                column
                for column in frame.columns
                if "date" in column.lower()
            ]
            if not candidates:
                raise KeyError(f"No date column found in {path.name}.")
            resolved_date_col = candidates[0]
        if return_col not in frame:
            raise KeyError(f"{return_col!r} is absent from {path.name}.")
        dates = pd.to_datetime(frame[resolved_date_col], errors="coerce")
        values = pd.to_numeric(frame[return_col], errors="coerce")
        valid = dates.notna() & values.notna()
        method_series = pd.Series(
            values.loc[valid].to_numpy(dtype=float),
            index=dates.loc[valid],
            name=method,
        ).sort_index()
        series[method] = method_series
        interval = moving_block_bootstrap_sharpe(
            method_series,
            periods_per_year=periods_per_year,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            seed=seed + method_index,
        )
        sharpe_rows.append({"method": method, "variant": variant, **interval})

    if reference_method not in series:
        raise FileNotFoundError(
            f"Portfolio returns for reference method {reference_method!r} are missing."
        )
    pairs = (
        list(combinations(series, 2))
        if all_pairs
        else [
            (reference_method, method)
            for method in series
            if method != reference_method
        ]
    )
    test_rows: list[dict[str, object]] = []
    for left_method, right_method in pairs:
        aligned = pd.concat(
            [series[left_method], series[right_method]], axis=1, join="inner"
        ).dropna()
        difference = aligned[right_method] - aligned[left_method]
        test = newey_west_mean_test(difference, max_lag=hac_lag)
        test_rows.append(
            {
                "reference_method": left_method,
                "comparison_method": right_method,
                "difference_definition": "comparison_minus_reference",
                "variant": variant,
                "return_col": return_col,
                **test,
            }
        )
    return pd.DataFrame(test_rows), pd.DataFrame(sharpe_rows)

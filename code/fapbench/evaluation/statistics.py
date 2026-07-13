from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


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
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(int(max_lag), n - 1))
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

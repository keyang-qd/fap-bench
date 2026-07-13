from __future__ import annotations

import numpy as np
import pandas as pd


def _maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else np.nan


def _summarize_returns(
    returns: pd.Series, periods_per_year: int
) -> tuple[float, float, float]:
    annual_return = float(returns.mean() * periods_per_year)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    return annual_return, annual_vol, float(sharpe)


def _empty_portfolio_metrics() -> dict[str, float]:
    return {
        "annual_return": np.nan,
        "annual_volatility": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan,
        "mean_turnover": np.nan,
        "net_annual_return": np.nan,
        "net_annual_volatility": np.nan,
        "net_sharpe": np.nan,
        "net_max_drawdown": np.nan,
        "n_months": 0,
        "mean_long_client_hhi": np.nan,
        "mean_short_client_hhi": np.nan,
        "mean_long_max_client_share": np.nan,
        "mean_short_max_client_share": np.nan,
    }


def calibrate_predictions_within_client(
    frame: pd.DataFrame,
    date_col: str = "date",
    client_col: str = "client_id",
    prediction_col: str = "y_pred",
    output_col: str = "portfolio_score",
    method: str = "percentile",
) -> pd.DataFrame:
    """Calibrate separately trained client scores before pooled ranking.

    Raw outputs from local models generally have incomparable locations and scales.
    Percentile ranks are therefore the default pooled local-score representation.
    """
    required = {date_col, client_col, prediction_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns for score calibration: {missing}")
    output = frame.copy()
    groups = output.groupby([date_col, client_col], sort=False)[prediction_col]
    method = method.lower()
    if method == "percentile":
        output[output_col] = groups.rank(method="average", pct=True)
    elif method == "zscore":
        mean = groups.transform("mean")
        std = groups.transform(lambda values: values.std(ddof=0))
        output[output_col] = (output[prediction_col] - mean) / std.replace(0.0, np.nan)
        output[output_col] = output[output_col].fillna(0.0)
    elif method == "none":
        output[output_col] = output[prediction_col]
    else:
        raise ValueError("Score calibration method must be percentile, zscore, or none.")
    return output


def _client_concentration(
    leg: pd.DataFrame,
    client_col: str | None,
) -> tuple[float, float]:
    if client_col is None or client_col not in leg.columns or leg.empty:
        return float("nan"), float("nan")
    shares = leg[client_col].astype(str).value_counts(normalize=True)
    return float((shares**2).sum()), float(shares.max())


def long_short_portfolio_metrics(
    frame: pd.DataFrame,
    date_col: str = "date",
    entity_col: str = "permno",
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    client_col: str | None = "client_id",
    quantile: float = 0.1,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5).")
    weights_by_date: dict[pd.Timestamp, dict[object, float]] = {}
    rows: list[dict[str, float | pd.Timestamp]] = []
    requested_columns = [entity_col, target_col, prediction_col]
    if client_col is not None and client_col in frame.columns:
        requested_columns.append(client_col)

    for date, group in frame.groupby(date_col, sort=True):
        group = group[requested_columns].dropna(
            subset=[entity_col, target_col, prediction_col]
        )
        n = len(group)
        n_leg = max(1, int(np.floor(n * quantile)))
        if n < 2 * n_leg:
            continue
        ordered = group.sort_values(prediction_col)
        short = ordered.head(n_leg)
        long = ordered.tail(n_leg)
        weights: dict[object, float] = {
            entity: 1.0 / n_leg for entity in long[entity_col]
        }
        for entity in short[entity_col]:
            weights[entity] = weights.get(entity, 0.0) - 1.0 / n_leg
        realized = dict(zip(group[entity_col], group[target_col], strict=False))
        gross_return = sum(
            weights[entity] * realized.get(entity, 0.0) for entity in weights
        )
        long_hhi, long_max_share = _client_concentration(long, client_col)
        short_hhi, short_max_share = _client_concentration(short, client_col)
        weights_by_date[pd.Timestamp(date)] = weights
        rows.append(
            {
                date_col: pd.Timestamp(date),
                "gross_return": float(gross_return),
                "n_assets": int(n),
                "n_per_leg": int(n_leg),
                "long_client_hhi": long_hhi,
                "short_client_hhi": short_hhi,
                "long_max_client_share": long_max_share,
                "short_max_client_share": short_max_share,
            }
        )

    portfolio = pd.DataFrame(rows)
    if portfolio.empty:
        return portfolio, _empty_portfolio_metrics()
    portfolio = portfolio.sort_values(date_col).reset_index(drop=True)

    turnovers: list[float] = []
    previous: dict[object, float] = {}
    for date in portfolio[date_col]:
        current = weights_by_date[pd.Timestamp(date)]
        universe = set(previous) | set(current)
        turnover = 0.5 * sum(
            abs(current.get(asset, 0.0) - previous.get(asset, 0.0))
            for asset in universe
        )
        turnovers.append(float(turnover))
        previous = current
    portfolio["turnover"] = turnovers
    cost_rate = transaction_cost_bps / 10_000.0
    portfolio["net_return"] = (
        portfolio["gross_return"] - cost_rate * portfolio["turnover"]
    )

    annual_return, annual_vol, sharpe = _summarize_returns(
        portfolio["gross_return"], periods_per_year
    )
    net_annual_return, net_annual_vol, net_sharpe = _summarize_returns(
        portfolio["net_return"], periods_per_year
    )
    metrics = {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": _maximum_drawdown(portfolio["gross_return"]),
        "mean_turnover": float(portfolio["turnover"].mean()),
        "net_annual_return": net_annual_return,
        "net_annual_volatility": net_annual_vol,
        "net_sharpe": net_sharpe,
        "net_max_drawdown": _maximum_drawdown(portfolio["net_return"]),
        "n_months": int(len(portfolio)),
        "mean_long_client_hhi": float(portfolio["long_client_hhi"].mean()),
        "mean_short_client_hhi": float(portfolio["short_client_hhi"].mean()),
        "mean_long_max_client_share": float(
            portfolio["long_max_client_share"].mean()
        ),
        "mean_short_max_client_share": float(
            portfolio["short_max_client_share"].mean()
        ),
    }
    return portfolio, metrics


def long_short_client_exposures(
    frame: pd.DataFrame,
    date_col: str = "date",
    client_col: str = "client_id",
    prediction_col: str = "y_pred",
    quantile: float = 0.1,
) -> pd.DataFrame:
    """Return monthly long/short client shares for concentration diagnostics."""
    rows: list[dict[str, object]] = []
    for date, group in frame.groupby(date_col, sort=True):
        valid = group[[client_col, prediction_col]].dropna()
        n_leg = max(1, int(np.floor(len(valid) * quantile)))
        if len(valid) < 2 * n_leg:
            continue
        ordered = valid.sort_values(prediction_col)
        legs = {"short": ordered.head(n_leg), "long": ordered.tail(n_leg)}
        for leg_name, leg in legs.items():
            shares = leg[client_col].astype(str).value_counts(normalize=True)
            for client_id, share in shares.items():
                rows.append(
                    {
                        date_col: pd.Timestamp(date),
                        client_col: str(client_id),
                        "leg": leg_name,
                        "share": float(share),
                        "n_leg": int(n_leg),
                    }
                )
    return pd.DataFrame(rows)


def client_portfolio_metrics(
    frame: pd.DataFrame,
    client_col: str = "client_id",
    date_col: str = "date",
    entity_col: str = "permno",
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    quantile: float = 0.1,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate an independent long-short portfolio for every client."""
    rows: list[dict[str, object]] = []
    for client_id, group in frame.groupby(client_col, sort=True):
        _, metrics = long_short_portfolio_metrics(
            group,
            date_col=date_col,
            entity_col=entity_col,
            target_col=target_col,
            prediction_col=prediction_col,
            client_col=None,
            quantile=quantile,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
        )
        rows.append({client_col: str(client_id), **metrics, "n_obs": int(len(group))})
    metrics_frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    if metrics_frame.empty:
        return metrics_frame, summary
    metric_names = [
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "net_annual_return",
        "net_annual_volatility",
        "net_sharpe",
        "net_max_drawdown",
    ]
    weights = pd.to_numeric(metrics_frame["n_obs"], errors="coerce").fillna(0.0)
    lower_is_better = {
        "annual_volatility",
        "mean_turnover",
    }
    # Drawdowns are negative; larger values are better (closer to zero).
    for metric in metric_names:
        values = pd.to_numeric(metrics_frame[metric], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        finite = values.notna()
        cleaned = values.loc[finite]
        if cleaned.empty:
            for statistic in [
                "mean",
                "median",
                "std",
                "weighted_mean",
                "best",
                "worst",
                "range",
            ]:
                summary[f"{statistic}_{metric}"] = float("nan")
            continue
        summary[f"mean_{metric}"] = float(cleaned.mean())
        summary[f"median_{metric}"] = float(cleaned.median())
        summary[f"std_{metric}"] = float(cleaned.std(ddof=0))
        if weights.loc[finite].sum() > 0:
            summary[f"weighted_mean_{metric}"] = float(
                np.average(cleaned, weights=weights.loc[finite])
            )
        else:
            summary[f"weighted_mean_{metric}"] = float("nan")
        if metric in lower_is_better:
            summary[f"best_{metric}"] = float(cleaned.min())
            summary[f"worst_{metric}"] = float(cleaned.max())
        else:
            summary[f"best_{metric}"] = float(cleaned.max())
            summary[f"worst_{metric}"] = float(cleaned.min())
        summary[f"range_{metric}"] = float(cleaned.max() - cleaned.min())
    summary["client_count"] = int(len(metrics_frame))
    return metrics_frame, summary


def client_averaged_long_short_metrics(
    frame: pd.DataFrame,
    client_col: str = "client_id",
    date_col: str = "date",
    entity_col: str = "permno",
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    quantile: float = 0.1,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build long-short portfolios inside each client, then equal-weight clients.

    This client-neutral portfolio is the primary domain-heterogeneity economic
    evaluation because it removes pooled score-scale and client-size effects.
    """
    client_returns: list[pd.DataFrame] = []
    for client_id, group in frame.groupby(client_col, sort=True):
        portfolio, _ = long_short_portfolio_metrics(
            group,
            date_col=date_col,
            entity_col=entity_col,
            target_col=target_col,
            prediction_col=prediction_col,
            client_col=None,
            quantile=quantile,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
        )
        if portfolio.empty:
            continue
        portfolio = portfolio[
            [date_col, "gross_return", "net_return", "turnover"]
        ].copy()
        portfolio[client_col] = str(client_id)
        client_returns.append(portfolio)

    if not client_returns:
        empty = pd.DataFrame(
            columns=[date_col, "gross_return", "net_return", "turnover"]
        )
        return empty, _empty_portfolio_metrics()

    stacked = pd.concat(client_returns, ignore_index=True)
    aggregate = (
        stacked.groupby(date_col, as_index=False)[
            ["gross_return", "net_return", "turnover"]
        ]
        .mean()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    annual_return, annual_vol, sharpe = _summarize_returns(
        aggregate["gross_return"], periods_per_year
    )
    net_annual_return, net_annual_vol, net_sharpe = _summarize_returns(
        aggregate["net_return"], periods_per_year
    )
    metrics = {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": _maximum_drawdown(aggregate["gross_return"]),
        "mean_turnover": float(aggregate["turnover"].mean()),
        "net_annual_return": net_annual_return,
        "net_annual_volatility": net_annual_vol,
        "net_sharpe": net_sharpe,
        "net_max_drawdown": _maximum_drawdown(aggregate["net_return"]),
        "n_months": int(len(aggregate)),
        "mean_long_client_hhi": np.nan,
        "mean_short_client_hhi": np.nan,
        "mean_long_max_client_share": np.nan,
        "mean_short_max_client_share": np.nan,
    }
    return aggregate, metrics

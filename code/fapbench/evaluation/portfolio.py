from __future__ import annotations

import numpy as np
import pandas as pd


def _maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else np.nan


def long_short_portfolio_metrics(
    frame: pd.DataFrame,
    date_col: str = "date",
    entity_col: str = "permno",
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    quantile: float = 0.1,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5).")
    weights_by_date: dict[pd.Timestamp, dict[object, float]] = {}
    rows: list[dict[str, float | pd.Timestamp]] = []

    for date, group in frame.groupby(date_col, sort=True):
        group = group[[entity_col, target_col, prediction_col]].dropna()
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
        gross_return = sum(weights[entity] * realized.get(entity, 0.0) for entity in weights)
        weights_by_date[pd.Timestamp(date)] = weights
        rows.append({date_col: pd.Timestamp(date), "gross_return": float(gross_return)})

    portfolio = pd.DataFrame(rows).sort_values(date_col).reset_index(drop=True)
    if portfolio.empty:
        metrics = {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "mean_turnover": np.nan,
            "net_annual_return": np.nan,
            "net_sharpe": np.nan,
        }
        return portfolio, metrics

    turnovers: list[float] = []
    previous: dict[object, float] = {}
    for date in portfolio[date_col]:
        current = weights_by_date[pd.Timestamp(date)]
        universe = set(previous) | set(current)
        turnover = 0.5 * sum(abs(current.get(asset, 0.0) - previous.get(asset, 0.0)) for asset in universe)
        turnovers.append(float(turnover))
        previous = current
    portfolio["turnover"] = turnovers
    cost_rate = transaction_cost_bps / 10_000.0
    portfolio["net_return"] = portfolio["gross_return"] - cost_rate * portfolio["turnover"]

    def summarize(returns: pd.Series) -> tuple[float, float, float]:
        annual_return = float(returns.mean() * periods_per_year)
        annual_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
        sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
        return annual_return, annual_vol, sharpe

    annual_return, annual_vol, sharpe = summarize(portfolio["gross_return"])
    net_annual_return, net_annual_vol, net_sharpe = summarize(portfolio["net_return"])
    metrics = {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": float(sharpe),
        "max_drawdown": _maximum_drawdown(portfolio["gross_return"]),
        "mean_turnover": float(portfolio["turnover"].mean()),
        "net_annual_return": net_annual_return,
        "net_annual_volatility": net_annual_vol,
        "net_sharpe": float(net_sharpe),
        "net_max_drawdown": _maximum_drawdown(portfolio["net_return"]),
        "n_months": int(len(portfolio)),
    }
    return portfolio, metrics



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

    This is the economically coherent local-only reference because predictions from
    separately trained client models need not be comparable on a pooled score scale.
    """
    client_returns: list[pd.DataFrame] = []
    for client_id, group in frame.groupby(client_col, sort=True):
        portfolio, _ = long_short_portfolio_metrics(
            group,
            date_col=date_col,
            entity_col=entity_col,
            target_col=target_col,
            prediction_col=prediction_col,
            quantile=quantile,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
        )
        if portfolio.empty:
            continue
        portfolio = portfolio[[date_col, "gross_return", "net_return", "turnover"]].copy()
        portfolio[client_col] = str(client_id)
        client_returns.append(portfolio)

    if not client_returns:
        empty = pd.DataFrame(columns=[date_col, "gross_return", "net_return", "turnover"])
        return empty, {
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
        }

    stacked = pd.concat(client_returns, ignore_index=True)
    aggregate = (
        stacked.groupby(date_col, as_index=False)[["gross_return", "net_return", "turnover"]]
        .mean()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    def summarize(returns: pd.Series) -> tuple[float, float, float]:
        annual_return = float(returns.mean() * periods_per_year)
        annual_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
        sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
        return annual_return, annual_vol, sharpe

    annual_return, annual_vol, sharpe = summarize(aggregate["gross_return"])
    net_annual_return, net_annual_vol, net_sharpe = summarize(aggregate["net_return"])
    metrics = {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": float(sharpe),
        "max_drawdown": _maximum_drawdown(aggregate["gross_return"]),
        "mean_turnover": float(aggregate["turnover"].mean()),
        "net_annual_return": net_annual_return,
        "net_annual_volatility": net_annual_vol,
        "net_sharpe": float(net_sharpe),
        "net_max_drawdown": _maximum_drawdown(aggregate["net_return"]),
        "n_months": int(len(aggregate)),
    }
    return aggregate, metrics

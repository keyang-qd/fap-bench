from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _safe_corr(left: pd.Series, right: pd.Series, rank: bool = False) -> float:
    valid = np.isfinite(left.to_numpy()) & np.isfinite(right.to_numpy())
    if valid.sum() < 3:
        return float("nan")
    a = left.to_numpy()[valid]
    b = right.to_numpy()[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if rank:
        return float(spearmanr(a, b).statistic)
    return float(np.corrcoef(a, b)[0, 1])


def predictive_metrics(
    frame: pd.DataFrame,
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    date_col: str = "date",
) -> dict[str, float]:
    valid = frame[[target_col, prediction_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"mse": np.nan, "mae": np.nan, "oos_r2_zero": np.nan, "ic": np.nan, "rank_ic": np.nan}
    y = valid[target_col].to_numpy(dtype=float)
    prediction = valid[prediction_col].to_numpy(dtype=float)
    residual_ss = np.sum((y - prediction) ** 2)
    zero_ss = np.sum(y ** 2)

    joined = frame[[date_col, target_col, prediction_col]].replace([np.inf, -np.inf], np.nan).dropna()
    monthly_ic = joined.groupby(date_col).apply(
        lambda group: _safe_corr(group[target_col], group[prediction_col]),
        include_groups=False,
    )
    monthly_rank_ic = joined.groupby(date_col).apply(
        lambda group: _safe_corr(group[target_col], group[prediction_col], rank=True),
        include_groups=False,
    )
    return {
        "mse": float(np.mean((y - prediction) ** 2)),
        "mae": float(np.mean(np.abs(y - prediction))),
        "oos_r2_zero": float(1.0 - residual_ss / zero_ss) if zero_ss > 0 else np.nan,
        "ic": float(monthly_ic.mean(skipna=True)),
        "rank_ic": float(monthly_rank_ic.mean(skipna=True)),
    }


def client_predictive_metrics(
    frame: pd.DataFrame,
    client_col: str = "client_id",
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    date_col: str = "date",
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    for client_id, group in frame.groupby(client_col, sort=True):
        metrics = predictive_metrics(group, target_col, prediction_col, date_col)
        rows.append({client_col: str(client_id), **metrics, "n_obs": len(group)})
    client_frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for metric in ["mse", "mae", "oos_r2_zero", "ic", "rank_ic"]:
        values = client_frame[metric].replace([np.inf, -np.inf], np.nan)
        summary[f"client_mean_{metric}"] = float(values.mean(skipna=True))
        summary[f"client_std_{metric}"] = float(values.std(skipna=True, ddof=0))
        if metric in {"mse", "mae"}:
            summary[f"client_worst_{metric}"] = float(values.max(skipna=True))
        else:
            summary[f"client_worst_{metric}"] = float(values.min(skipna=True))
    return client_frame, summary

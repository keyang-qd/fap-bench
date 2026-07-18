from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _safe_corr(left: pd.Series, right: pd.Series, rank: bool = False) -> float:
    valid = np.isfinite(left.to_numpy()) & np.isfinite(right.to_numpy())
    if valid.sum() < 3:
        return float("nan")
    a = left.to_numpy(dtype=float)[valid]
    b = right.to_numpy(dtype=float)[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if rank:
        return float(spearmanr(a, b).statistic)
    return float(np.corrcoef(a, b)[0, 1])


def _monthly_correlations(
    frame: pd.DataFrame,
    date_col: str,
    target_col: str,
    prediction_col: str,
    rank: bool,
) -> pd.Series:
    values: dict[object, float] = {}
    for date, group in frame.groupby(date_col, sort=True):
        values[date] = _safe_corr(
            group[target_col], group[prediction_col], rank=rank
        )
    return pd.Series(values, dtype=float)


def _safe_mean(values: pd.Series) -> float:
    cleaned = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(cleaned.mean()) if not cleaned.empty else float("nan")


def _safe_std(values: pd.Series, ddof: int = 1) -> float:
    cleaned = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(cleaned) <= ddof:
        return float("nan")
    return float(cleaned.std(ddof=ddof))


def _ratio(mean: float, std: float) -> float:
    return float(mean / std) if np.isfinite(std) and std > 0 else float("nan")


def monthly_predictive_metrics(
    frame: pd.DataFrame,
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    date_col: str = "date",
) -> pd.DataFrame:
    """Return monthly cross-sectional predictive diagnostics.

    The benchmark saves this table so IC dispersion, ICIR, positive-month ratios,
    and statistical tests can be recomputed without re-running model inference.
    """
    if date_col not in frame.columns:
        raise KeyError(f"Evaluation date column {date_col!r} is absent.")

    rows: list[dict[str, float | int | object]] = []
    columns = [date_col, target_col, prediction_col]
    for date, group in frame[columns].groupby(date_col, sort=True):
        valid = group[[target_col, prediction_col]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if valid.empty:
            continue
        y = valid[target_col].to_numpy(dtype=float)
        prediction = valid[prediction_col].to_numpy(dtype=float)
        residual_ss = float(np.sum((y - prediction) ** 2))
        zero_ss = float(np.sum(y**2))
        rows.append(
            {
                date_col: date,
                "n_obs": int(len(valid)),
                "mse": float(np.mean((y - prediction) ** 2)),
                "mae": float(np.mean(np.abs(y - prediction))),
                "oos_r2_zero": (
                    float(1.0 - residual_ss / zero_ss)
                    if zero_ss > 0
                    else float("nan")
                ),
                "ic": _safe_corr(
                    valid[target_col], valid[prediction_col], rank=False
                ),
                "rank_ic": _safe_corr(
                    valid[target_col], valid[prediction_col], rank=True
                ),
                "prediction_mean": float(np.mean(prediction)),
                "prediction_std": float(np.std(prediction, ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def predictive_metrics(
    frame: pd.DataFrame,
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    date_col: str = "date",
) -> dict[str, float]:
    if date_col not in frame.columns:
        raise KeyError(f"Evaluation date column {date_col!r} is absent.")
    valid = (
        frame[[target_col, prediction_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if valid.empty:
        return {
            "mse": np.nan,
            "mae": np.nan,
            "oos_r2_zero": np.nan,
            "ic": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_positive_ratio": np.nan,
            "rank_ic": np.nan,
            "rank_ic_std": np.nan,
            "rank_icir": np.nan,
            "rank_ic_positive_ratio": np.nan,
            "n_ic_months": 0,
            "prediction_mean": np.nan,
            "prediction_std": np.nan,
            "target_mean": np.nan,
            "target_std": np.nan,
            "calibration_intercept_ols": np.nan,
            "calibration_slope_ols": np.nan,
        }
    y = valid[target_col].to_numpy(dtype=float)
    prediction = valid[prediction_col].to_numpy(dtype=float)
    residual_ss = np.sum((y - prediction) ** 2)
    zero_ss = np.sum(y**2)

    joined = (
        frame[[date_col, target_col, prediction_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    monthly_ic = _monthly_correlations(
        joined, date_col, target_col, prediction_col, rank=False
    )
    monthly_rank_ic = _monthly_correlations(
        joined, date_col, target_col, prediction_col, rank=True
    )
    ic_mean = _safe_mean(monthly_ic)
    ic_std = _safe_std(monthly_ic)
    rank_ic_mean = _safe_mean(monthly_rank_ic)
    rank_ic_std = _safe_std(monthly_rank_ic)
    valid_ic = monthly_ic.replace([np.inf, -np.inf], np.nan).dropna()
    valid_rank_ic = monthly_rank_ic.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "mse": float(np.mean((y - prediction) ** 2)),
        "mae": float(np.mean(np.abs(y - prediction))),
        "oos_r2_zero": (
            float(1.0 - residual_ss / zero_ss) if zero_ss > 0 else np.nan
        ),
        "ic": ic_mean,
        "ic_std": ic_std,
        "icir": _ratio(ic_mean, ic_std),
        "ic_positive_ratio": (
            float((valid_ic > 0).mean()) if not valid_ic.empty else np.nan
        ),
        "rank_ic": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_icir": _ratio(rank_ic_mean, rank_ic_std),
        "rank_ic_positive_ratio": (
            float((valid_rank_ic > 0).mean())
            if not valid_rank_ic.empty
            else np.nan
        ),
        "n_ic_months": int(len(valid_ic)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction, ddof=0)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y, ddof=0)),
        "calibration_intercept_ols": float(
            np.mean(y)
            - (
                np.mean((prediction - np.mean(prediction)) * (y - np.mean(y)))
                / (np.mean((prediction - np.mean(prediction)) ** 2) + 1.0e-12)
            )
            * np.mean(prediction)
        ),
        "calibration_slope_ols": float(
            np.mean((prediction - np.mean(prediction)) * (y - np.mean(y)))
            / (np.mean((prediction - np.mean(prediction)) ** 2) + 1.0e-12)
        ),
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
        metrics = predictive_metrics(
            group, target_col, prediction_col, date_col
        )
        rows.append({client_col: str(client_id), **metrics, "n_obs": len(group)})
    client_frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    if client_frame.empty:
        return client_frame, summary

    metric_names = [
        "mse",
        "mae",
        "oos_r2_zero",
        "ic",
        "ic_std",
        "icir",
        "ic_positive_ratio",
        "rank_ic",
        "rank_ic_std",
        "rank_icir",
        "rank_ic_positive_ratio",
        "prediction_mean",
        "prediction_std",
    ]
    weights = pd.to_numeric(client_frame["n_obs"], errors="coerce").fillna(0.0)
    for metric in metric_names:
        if metric not in client_frame:
            continue
        values = pd.to_numeric(client_frame[metric], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        finite = values.notna()
        summary[f"client_mean_{metric}"] = float(values.mean(skipna=True))
        summary[f"client_median_{metric}"] = float(values.median(skipna=True))
        summary[f"client_std_{metric}"] = float(values.std(skipna=True, ddof=0))
        if finite.any() and weights.loc[finite].sum() > 0:
            summary[f"client_weighted_mean_{metric}"] = float(
                np.average(values.loc[finite], weights=weights.loc[finite])
            )
        else:
            summary[f"client_weighted_mean_{metric}"] = float("nan")

        if metric in {"mse", "mae", "ic_std", "rank_ic_std"}:
            best = values.min(skipna=True)
            worst = values.max(skipna=True)
        else:
            best = values.max(skipna=True)
            worst = values.min(skipna=True)
        summary[f"client_best_{metric}"] = float(best)
        summary[f"client_worst_{metric}"] = float(worst)
        summary[f"client_range_{metric}"] = float(
            values.max(skipna=True) - values.min(skipna=True)
        )
    summary["client_count"] = int(len(client_frame))
    return client_frame, summary

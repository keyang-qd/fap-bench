from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearPredictionCalibrator:
    intercept: float
    slope: float
    raw_slope: float
    n_obs: int
    nonnegative_slope: bool
    direction_valid: bool

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return {"method": "validation_linear", **asdict(self)}


def fit_linear_prediction_calibrator(
    validation_predictions: pd.DataFrame,
    *,
    target_col: str = "y_true",
    prediction_col: str = "y_pred",
    nonnegative_slope: bool = True,
    ridge: float = 1.0e-12,
) -> LinearPredictionCalibrator:
    """Fit y = intercept + slope * prediction using validation data only."""
    valid = validation_predictions[[target_col, prediction_col]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(valid) < 3:
        raise ValueError("At least three validation predictions are required for calibration.")
    y = valid[target_col].to_numpy(dtype=float)
    x = valid[prediction_col].to_numpy(dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    variance = float(np.mean((x - x_mean) ** 2))
    covariance = float(np.mean((x - x_mean) * (y - y_mean)))
    raw_slope = covariance / (variance + float(ridge))
    slope = float(raw_slope)
    if nonnegative_slope:
        slope = max(0.0, slope)
    intercept = y_mean - slope * x_mean
    return LinearPredictionCalibrator(
        intercept=float(intercept),
        slope=float(slope),
        raw_slope=float(raw_slope),
        n_obs=int(len(valid)),
        nonnegative_slope=bool(nonnegative_slope),
        direction_valid=bool(slope > 0.0),
    )


def apply_linear_prediction_calibrator(
    predictions: pd.DataFrame,
    calibrator: LinearPredictionCalibrator,
    *,
    prediction_col: str = "y_pred",
    output_col: str = "y_pred",
    preserve_raw_col: str = "y_pred_uncalibrated",
) -> pd.DataFrame:
    data = predictions.copy()
    raw = pd.to_numeric(data[prediction_col], errors="coerce")
    if preserve_raw_col:
        data[preserve_raw_col] = raw
    data[output_col] = calibrator.intercept + calibrator.slope * raw
    return data

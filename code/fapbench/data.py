from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE = {
    "permno", "gvkey", "cusip", "ticker", "date", "datadate", "year_month",
    "year", "fyear", "client_id", "split", "ret", "retx", "retadj",
    "retadj_1mn", "target", "label", "port_weight", "shares", "shares_last",
    "shares_chg", "prc", "shrout", "me", "me_lag", "shrcd", "exchcd",
    "cfacpr", "cfacshr",
}


@dataclass(frozen=True)
class PanelSpec:
    date_col: str = "date"
    entity_col: str = "permno"
    target_col: str = "retadj_1mn"
    target_shift_months: int = 0
    feature_columns: tuple[str, ...] | None = None
    excluded_columns: tuple[str, ...] = ()
    min_feature_coverage: float = 0.5


def load_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported data format: {suffix}")
    return frame


def infer_feature_columns(frame: pd.DataFrame, spec: PanelSpec) -> list[str]:
    if spec.feature_columns:
        missing = sorted(set(spec.feature_columns) - set(frame.columns))
        if missing:
            raise KeyError(f"Configured feature columns are missing: {missing}")
        return list(spec.feature_columns)

    excluded = DEFAULT_EXCLUDE | set(spec.excluded_columns) | {
        spec.date_col, spec.entity_col, spec.target_col
    }
    numeric = [
        col for col in frame.select_dtypes(include=[np.number, "bool"]).columns
        if col not in excluded
    ]
    if not numeric:
        raise ValueError("No numeric feature columns remained after exclusions.")
    coverage = frame[numeric].notna().mean()
    kept = coverage[coverage >= spec.min_feature_coverage].index.tolist()
    if not kept:
        raise ValueError(
            "No features satisfy min_feature_coverage="
            f"{spec.min_feature_coverage:.2f}."
        )
    return kept


def prepare_panel(frame: pd.DataFrame, spec: PanelSpec) -> tuple[pd.DataFrame, list[str]]:
    required = {spec.date_col, spec.entity_col, spec.target_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")

    data = frame.copy()
    data[spec.date_col] = pd.to_datetime(data[spec.date_col], errors="coerce")
    data = data.dropna(subset=[spec.date_col, spec.entity_col])
    data = data.sort_values([spec.entity_col, spec.date_col]).reset_index(drop=True)

    if spec.target_shift_months:
        # A positive value predicts a future target. For example, 1 maps features
        # at t to the raw target observed at t+1 within each security.
        data[spec.target_col] = data.groupby(spec.entity_col, sort=False)[
            spec.target_col
        ].shift(-spec.target_shift_months)

    features = infer_feature_columns(data, spec)
    for col in features + [spec.target_col]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[spec.target_col]).reset_index(drop=True)
    return data, features


def split_by_time(
    frame: pd.DataFrame,
    date_col: str,
    train_end: str | None = None,
    val_end: str | None = None,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
) -> pd.DataFrame:
    data = frame.copy()
    dates = np.array(sorted(pd.to_datetime(data[date_col].dropna().unique())))
    if len(dates) < 3:
        raise ValueError("At least three distinct dates are required for train/val/test splits.")

    if train_end is not None and val_end is not None:
        train_boundary = pd.Timestamp(train_end)
        val_boundary = pd.Timestamp(val_end)
        if val_boundary <= train_boundary:
            raise ValueError("val_end must be later than train_end.")
        split = np.where(
            data[date_col] <= train_boundary,
            "train",
            np.where(data[date_col] <= val_boundary, "val", "test"),
        )
    else:
        ratios = np.asarray(ratios, dtype=float)
        if len(ratios) != 3 or np.any(ratios <= 0):
            raise ValueError("ratios must contain three positive values.")
        ratios = ratios / ratios.sum()
        n_dates = len(dates)
        train_idx = max(1, min(n_dates - 2, int(np.floor(n_dates * ratios[0]))))
        val_idx = max(train_idx + 1, min(n_dates - 1, int(np.floor(n_dates * ratios[:2].sum()))))
        train_boundary = pd.Timestamp(dates[train_idx - 1])
        val_boundary = pd.Timestamp(dates[val_idx - 1])
        split = np.where(
            data[date_col] <= train_boundary,
            "train",
            np.where(data[date_col] <= val_boundary, "val", "test"),
        )

    data["split"] = split
    if set(data["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("The selected boundaries did not produce all three splits.")
    return data


class PanelPreprocessor:
    """Train-only imputation and optional scaling for panel features."""

    def __init__(self, feature_columns: Iterable[str], scaling: str = "global"):
        self.feature_columns = list(feature_columns)
        if scaling not in {"global", "none", "cross_sectional"}:
            raise ValueError("scaling must be global, none, or cross_sectional.")
        self.scaling = scaling
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "PanelPreprocessor":
        x = train[self.feature_columns]
        self.medians_ = x.median(axis=0).fillna(0.0)
        filled = x.fillna(self.medians_)
        self.means_ = filled.mean(axis=0)
        self.stds_ = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        if self.medians_ is None or self.means_ is None or self.stds_ is None:
            raise RuntimeError("PanelPreprocessor must be fitted before transform.")
        data = frame.copy()
        x = data[self.feature_columns].fillna(self.medians_)
        if self.scaling == "global":
            x = (x - self.means_) / self.stds_
        elif self.scaling == "cross_sectional":
            # Cross-sectional normalization uses only same-month information.
            grouped = x.groupby(data[date_col])
            mean = grouped.transform("mean")
            std = grouped.transform("std").replace(0.0, 1.0).fillna(1.0)
            x = (x - mean) / std
        data.loc[:, self.feature_columns] = x.clip(-10.0, 10.0)
        return data

    def fit_transform_splits(self, frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
        train = frame.loc[frame["split"] == "train"]
        self.fit(train)
        return self.transform(frame, date_col=date_col)

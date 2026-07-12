from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE = {
    "permno", "gvkey", "cusip", "ticker", "date", "formation_date",
    "target_date", "datadate", "year_month", "target_year_month", "year",
    "fyear", "sic", "sic2", "naics", "sector", "ff12", "client_id",
    "client_id_domain", "split", "ret", "retx", "retadj", "retadj_1mn",
    "target", "label", "port_weight", "shares", "shares_last", "shares_chg",
    "prc", "shrout", "me", "me_lag", "shrcd", "exchcd", "cfacpr", "cfacshr",
}


@dataclass(frozen=True)
class PanelSpec:
    date_col: str = "date"
    entity_col: str = "permno"
    target_col: str = "retadj_1mn"
    target_shift_months: int = 0
    target_date_col: str = "target_date"
    target_period_col: str = "target_year_month"
    feature_columns: tuple[str, ...] | None = None
    excluded_columns: tuple[str, ...] = ()
    min_feature_coverage: float = 0.5
    require_unique_entity_month: bool = True
    validate_target_alignment: bool = True


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
        spec.date_col,
        spec.entity_col,
        spec.target_col,
        spec.target_date_col,
        spec.target_period_col,
    }
    numeric = [
        col
        for col in frame.select_dtypes(include=[np.number, "bool"]).columns
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


def _coerce_month_period(series: pd.Series) -> pd.Series:
    """Coerce Period/string/datetime monthly labels to Period[M]."""
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.asfreq("M")
    return pd.to_datetime(series.astype(str), errors="coerce").dt.to_period("M")


def _align_target_by_calendar_month(data: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    """Align a raw monthly target to an exact future calendar month.

    When ``target_shift_months == 0``, the target is assumed already aligned by the
    data generator. Existing ``target_date``/``target_year_month`` fields are
    preserved. When no target-date field exists, the target is treated as
    contemporaneous with the formation date.

    A positive horizon h maps features dated in month t to the target observed in
    month t+h for the same security. This uses an entity-month join rather than
    ``groupby().shift()`` so a missing calendar month is not silently interpreted as
    a one-month-ahead observation.
    """
    output = data.copy()
    feature_period = output[spec.date_col].dt.to_period("M")

    if spec.target_shift_months < 0:
        raise ValueError("target_shift_months must be non-negative.")

    if spec.target_shift_months == 0:
        if spec.target_period_col in output.columns:
            output[spec.target_period_col] = _coerce_month_period(output[spec.target_period_col])
        if spec.target_date_col in output.columns:
            output[spec.target_date_col] = pd.to_datetime(
                output[spec.target_date_col], errors="coerce"
            )
        elif spec.target_period_col in output.columns:
            output[spec.target_date_col] = (
                output[spec.target_period_col]
                .dt.to_timestamp(how="end")
                .dt.normalize()
            )
        else:
            output[spec.target_date_col] = output[spec.date_col]
            output[spec.target_period_col] = feature_period
        return output

    lookup = output[[spec.entity_col, spec.date_col, spec.target_col]].copy()
    lookup["_target_period"] = lookup[spec.date_col].dt.to_period("M")
    lookup = lookup.rename(
        columns={
            spec.target_col: "_aligned_target",
            spec.date_col: spec.target_date_col,
        }
    )

    duplicate_lookup = lookup.duplicated(
        [spec.entity_col, "_target_period"], keep=False
    )
    if duplicate_lookup.any():
        examples = lookup.loc[
            duplicate_lookup, [spec.entity_col, "_target_period"]
        ].head(5)
        raise ValueError(
            "Cannot align future targets because entity-month keys are duplicated. "
            f"Examples: {examples.to_dict(orient='records')}"
        )

    output["_target_period"] = feature_period + spec.target_shift_months
    output = output.drop(
        columns=[spec.target_date_col, spec.target_period_col], errors="ignore"
    )
    output = output.merge(
        lookup[
            [
                spec.entity_col,
                "_target_period",
                "_aligned_target",
                spec.target_date_col,
            ]
        ],
        on=[spec.entity_col, "_target_period"],
        how="left",
        validate="many_to_one",
    )
    output[spec.target_col] = output.pop("_aligned_target")
    output[spec.target_period_col] = output["_target_period"]
    output = output.drop(columns=["_target_period"])
    return output


def _validate_target_months(data: pd.DataFrame, spec: PanelSpec) -> None:
    if not spec.validate_target_alignment:
        return
    formation_period = data[spec.date_col].dt.to_period("M")
    target_dates = pd.to_datetime(data[spec.target_date_col], errors="coerce")
    target_period = target_dates.dt.to_period("M")

    if spec.target_period_col in data.columns:
        declared_period = _coerce_month_period(data[spec.target_period_col])
        valid = target_period.notna() & declared_period.notna()
        mismatch = valid & (target_period != declared_period)
        if mismatch.any():
            raise ValueError(
                f"{spec.target_date_col} and {spec.target_period_col} disagree for "
                f"{int(mismatch.sum())} rows."
            )

    if spec.target_shift_months > 0:
        expected = formation_period + spec.target_shift_months
        valid = expected.notna() & target_period.notna()
        mismatch = valid & (expected != target_period)
        if mismatch.any():
            examples = data.loc[
                mismatch, [spec.entity_col, spec.date_col, spec.target_date_col]
            ].head(5)
            raise ValueError(
                "Future target alignment does not match target_shift_months. "
                f"Examples: {examples.to_dict(orient='records')}"
            )


def prepare_panel(
    frame: pd.DataFrame, spec: PanelSpec
) -> tuple[pd.DataFrame, list[str]]:
    required = {spec.date_col, spec.entity_col, spec.target_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")

    data = frame.copy()
    data[spec.date_col] = pd.to_datetime(data[spec.date_col], errors="coerce")
    if spec.target_date_col in data.columns:
        data[spec.target_date_col] = pd.to_datetime(
            data[spec.target_date_col], errors="coerce"
        )
    data = data.dropna(subset=[spec.date_col, spec.entity_col])
    data = data.sort_values([spec.entity_col, spec.date_col]).reset_index(drop=True)

    entity_month = pd.DataFrame(
        {
            spec.entity_col: data[spec.entity_col],
            "_month": data[spec.date_col].dt.to_period("M"),
        }
    )
    duplicates = entity_month.duplicated(
        [spec.entity_col, "_month"], keep=False
    )
    if spec.require_unique_entity_month and duplicates.any():
        examples = entity_month.loc[duplicates].head(5).to_dict(orient="records")
        raise ValueError(
            "The panel must contain at most one row per entity-month. "
            f"Examples of duplicates: {examples}"
        )

    data = _align_target_by_calendar_month(data, spec)
    _validate_target_months(data, spec)

    features = infer_feature_columns(data, spec)
    for col in features + [spec.target_col]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=[spec.target_col, spec.target_date_col]
    ).reset_index(drop=True)
    return data, features


def validate_crsp_compustat_profitability_panel(
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Generate diagnostics for either the original or revised data recipe."""
    report: dict[str, object] = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
    }
    if "permno" in frame:
        report["n_entities"] = int(frame["permno"].nunique(dropna=True))

    preferred_date_col = "formation_date" if "formation_date" in frame else "date"
    if preferred_date_col in frame:
        dates = pd.to_datetime(frame[preferred_date_col], errors="coerce")
        report["formation_date_column"] = preferred_date_col
        report["date_min"] = (
            None if dates.dropna().empty else str(dates.min().date())
        )
        report["date_max"] = (
            None if dates.dropna().empty else str(dates.max().date())
        )
    else:
        dates = pd.Series(pd.NaT, index=frame.index)

    if {"permno", preferred_date_col}.issubset(frame.columns):
        months = dates.dt.to_period("M")
        report["duplicate_entity_month_rows"] = int(
            pd.DataFrame({"permno": frame["permno"], "month": months})
            .duplicated(["permno", "month"], keep=False)
            .sum()
        )
        report["duplicate_entity_date_rows"] = int(
            frame.assign(_date=dates)
            .duplicated(["permno", "_date"], keep=False)
            .sum()
        )

    if "year_month" in frame and preferred_date_col in frame:
        recipe_month = _coerce_month_period(frame["year_month"])
        date_month = dates.dt.to_period("M")
        valid = date_month.notna() & recipe_month.notna()
        report["date_year_month_mismatch_rows"] = int(
            (recipe_month[valid] != date_month[valid]).sum()
        )

    if {"retadj_1mn", "retadj"}.issubset(frame.columns):
        left = pd.to_numeric(frame["retadj_1mn"], errors="coerce")
        right = pd.to_numeric(frame["retadj"], errors="coerce")
        valid = left.notna() & right.notna()
        report["retadj_1mn_equals_same_month_retadj_share"] = (
            float(np.isclose(left[valid], right[valid], equal_nan=False).mean())
            if valid.any()
            else None
        )

    exact_pre_aligned = False
    if {"year_month", "target_year_month"}.issubset(frame.columns):
        formation_period = _coerce_month_period(frame["year_month"])
        target_period = _coerce_month_period(frame["target_year_month"])
        valid = formation_period.notna() & target_period.notna()
        alignment_share = (
            float(((formation_period[valid] + 1) == target_period[valid]).mean())
            if valid.any()
            else None
        )
        report["target_year_month_is_exact_t_plus_1_share"] = alignment_share
        exact_pre_aligned = alignment_share is not None and alignment_share > 0.999

    if {"target_date", "target_year_month"}.issubset(frame.columns):
        target_date_period = pd.to_datetime(
            frame["target_date"], errors="coerce"
        ).dt.to_period("M")
        target_period = _coerce_month_period(frame["target_year_month"])
        valid = target_date_period.notna() & target_period.notna()
        report["target_date_period_mismatch_rows"] = int(
            (target_date_period[valid] != target_period[valid]).sum()
        )

    if "shrcd" in frame:
        shrcd = pd.to_numeric(frame["shrcd"], errors="coerce")
        report["non_common_share_rows"] = int(
            (shrcd.notna() & ~shrcd.isin([10, 11])).sum()
        )
    if "exchcd" in frame:
        exchcd = pd.to_numeric(frame["exchcd"], errors="coerce")
        report["non_major_exchange_rows"] = int(
            (exchcd.notna() & ~exchcd.isin([1, 2, 3])).sum()
        )

    report["has_sic"] = bool("sic" in frame.columns)
    report["has_sic2"] = bool("sic2" in frame.columns)
    report["has_port_weight"] = bool("port_weight" in frame.columns)
    report["has_target_date"] = bool("target_date" in frame.columns)
    report["has_target_year_month"] = bool("target_year_month" in frame.columns)

    zscore_cols = [
        c
        for c in [
            "gpoa_zscore",
            "roa_zscore",
            "roe_zscore",
            "roic_zscore",
            "opoa_zscore",
            "gmar_zscore",
            "fcfoa_zscore",
        ]
        if c in frame.columns
    ]
    report["available_profitability_features"] = zscore_cols + (
        ["profitability_simple"]
        if "profitability_simple" in frame.columns
        else []
    )

    if exact_pre_aligned and "target_date" in frame.columns:
        report["target_alignment_status"] = "pre_aligned_exact_t_plus_1"
        report["recommended_target_shift_months"] = 0
    else:
        contemporaneous_share = report.get(
            "retadj_1mn_equals_same_month_retadj_share"
        )
        if isinstance(contemporaneous_share, float) and contemporaneous_share > 0.99:
            report["target_alignment_status"] = "contemporaneous_target"
            report["recommended_target_shift_months"] = 1
        else:
            report["target_alignment_status"] = "alignment_not_verified"
            report["recommended_target_shift_months"] = None

    report["recommended_date_column"] = preferred_date_col
    report["recommended_split_date_column"] = (
        "target_date" if "target_date" in frame.columns else preferred_date_col
    )
    report["recommended_scaling"] = "none"
    report["recommended_domain_column"] = (
        "sic (map to FF12)" if "sic" in frame.columns else "retain SIC in the data recipe"
    )
    return report


def split_by_time(
    frame: pd.DataFrame,
    date_col: str,
    train_end: str | None = None,
    val_end: str | None = None,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
) -> pd.DataFrame:
    if date_col not in frame.columns:
        raise KeyError(f"Split date column {date_col!r} is absent from the panel.")
    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    if data[date_col].isna().any():
        raise ValueError(f"Split date column {date_col!r} contains missing/invalid values.")
    dates = np.array(sorted(data[date_col].unique()))
    if len(dates) < 3:
        raise ValueError(
            "At least three distinct dates are required for train/val/test splits."
        )

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
        train_idx = max(
            1, min(n_dates - 2, int(np.floor(n_dates * ratios[0])))
        )
        val_idx = max(
            train_idx + 1,
            min(n_dates - 1, int(np.floor(n_dates * ratios[:2].sum()))),
        )
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
        self.stds_ = (
            filled.std(axis=0, ddof=0)
            .replace(0.0, 1.0)
            .fillna(1.0)
        )
        return self

    def transform(
        self, frame: pd.DataFrame, date_col: str = "date"
    ) -> pd.DataFrame:
        if self.medians_ is None or self.means_ is None or self.stds_ is None:
            raise RuntimeError(
                "PanelPreprocessor must be fitted before transform."
            )
        data = frame.copy()
        x = data[self.feature_columns].fillna(self.medians_)
        if self.scaling == "global":
            x = (x - self.means_) / self.stds_
        elif self.scaling == "cross_sectional":
            grouped = x.groupby(data[date_col])
            mean = grouped.transform("mean")
            std = grouped.transform("std").replace(0.0, 1.0).fillna(1.0)
            x = (x - mean) / std
        data.loc[:, self.feature_columns] = x.clip(-10.0, 10.0)
        return data

    def fit_transform_splits(
        self, frame: pd.DataFrame, date_col: str
    ) -> pd.DataFrame:
        train = frame.loc[frame["split"] == "train"]
        if train.empty:
            raise ValueError("Training split is empty.")
        self.fit(train)
        return self.transform(frame, date_col=date_col)

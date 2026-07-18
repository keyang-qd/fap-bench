from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import hashlib

import numpy as np
import pandas as pd

from fapbench.partition.industry import map_sic_to_ff12


def apply_partition_common_universe(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str],
    finite_numeric_columns: Iterable[str] = (),
    positive_columns: Iterable[str] = (),
    ff12_sic_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restrict all partition experiments to one row-identical universe.

    The filter is intentionally applied before client assignment.  Every
    partition therefore receives the same firm-month observations; only the
    client labels differ.  This prevents partition comparisons from being
    confounded by missing SIC, exchange, or market-cap fields.
    """
    required = list(dict.fromkeys(str(value) for value in required_columns))
    finite_numeric = list(dict.fromkeys(str(value) for value in finite_numeric_columns))
    positive = list(dict.fromkeys(str(value) for value in positive_columns))
    needed = set(required) | set(finite_numeric) | set(positive)
    if ff12_sic_col is not None:
        needed.add(str(ff12_sic_col))
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise KeyError(f"Common-universe columns are missing: {missing}")

    mask = pd.Series(True, index=frame.index, dtype=bool)
    reasons: dict[str, int] = {}

    for column in required:
        valid = frame[column].notna()
        reasons[f"missing_{column}"] = int((~valid).sum())
        mask &= valid

    for column in finite_numeric:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
        valid_series = pd.Series(valid, index=frame.index)
        reasons[f"nonfinite_{column}"] = int((~valid_series).sum())
        mask &= valid_series

    for column in positive:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric) & (numeric > 0)
        reasons[f"nonpositive_{column}"] = int((~valid).sum())
        mask &= valid

    if ff12_sic_col is not None:
        ff12 = map_sic_to_ff12(frame[str(ff12_sic_col)])
        valid = ff12.notna()
        reasons[f"invalid_ff12_{ff12_sic_col}"] = int((~valid).sum())
        mask &= valid

    filtered = frame.loc[mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Partition common-universe filter removed every row.")

    fingerprint_columns = [
        column
        for column in ["permno", "formation_date", "target_date", "split"]
        if column in filtered.columns
    ]
    hashed = pd.util.hash_pandas_object(
        filtered[fingerprint_columns], index=False
    ).to_numpy(dtype=np.uint64, copy=False)
    row_fingerprint = hashlib.sha256(hashed.tobytes()).hexdigest()

    report: dict[str, Any] = {
        "enabled": True,
        "row_fingerprint_sha256": row_fingerprint,
        "fingerprint_columns": fingerprint_columns,
        "required_columns": required,
        "finite_numeric_columns": finite_numeric,
        "positive_columns": positive,
        "ff12_sic_col": ff12_sic_col,
        "n_rows_before": int(len(frame)),
        "n_rows_after": int(len(filtered)),
        "n_rows_dropped": int(len(frame) - len(filtered)),
        "retained_fraction": float(len(filtered) / len(frame)),
        **reasons,
    }
    if "split" in frame.columns:
        before = frame["split"].astype(str).value_counts().to_dict()
        after = filtered["split"].astype(str).value_counts().to_dict()
        report["split_rows_before"] = {str(k): int(v) for k, v in before.items()}
        report["split_rows_after"] = {str(k): int(v) for k, v in after.items()}
    return filtered, report

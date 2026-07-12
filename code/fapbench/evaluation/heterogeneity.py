from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from fapbench.partition.feature import feature_jaccard_dissimilarity


def _average_pairwise_wasserstein(
    frame: pd.DataFrame,
    columns: list[str],
    client_col: str,
    max_rows_per_client: int = 5000,
    seed: int = 42,
) -> float:
    clients = sorted(frame[client_col].astype(str).unique())
    if len(clients) < 2 or not columns:
        return 0.0
    rng = np.random.default_rng(seed)
    client_arrays: dict[str, np.ndarray] = {}
    for client in clients:
        values = frame.loc[frame[client_col].astype(str) == client, columns].to_numpy(dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values) > max_rows_per_client:
            values = values[rng.choice(len(values), max_rows_per_client, replace=False)]
        client_arrays[client] = values

    distances: list[float] = []
    for left, right in combinations(clients, 2):
        a, b = client_arrays[left], client_arrays[right]
        if len(a) == 0 or len(b) == 0:
            continue
        per_feature = [wasserstein_distance(a[:, idx], b[:, idx]) for idx in range(len(columns))]
        distances.append(float(np.mean(per_feature)))
    return float(np.mean(distances)) if distances else np.nan


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        values = values - values.min()
    if values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * sorted_values)) / (n * sorted_values.sum()))


def partition_heterogeneity_report(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    client_col: str = "client_id",
    feature_masks: Mapping[str, np.ndarray] | None = None,
    temporal_unavailability_value: float = 0.0,
    seed: int = 42,
) -> dict[str, float | int]:
    client_sizes = frame.groupby(client_col).size().to_numpy(dtype=float)
    report: dict[str, float | int] = {
        "n_clients": int(frame[client_col].nunique()),
        "n_observations": int(len(frame)),
        "client_size_mean": float(client_sizes.mean()),
        "client_size_cv": float(client_sizes.std(ddof=0) / client_sizes.mean()) if client_sizes.mean() else np.nan,
        "client_size_gini": _gini(client_sizes),
        "feature_distribution_wasserstein": _average_pairwise_wasserstein(
            frame, feature_columns, client_col, seed=seed
        ),
        "label_distribution_wasserstein": _average_pairwise_wasserstein(
            frame, [target_col], client_col, seed=seed
        ),
        "temporal_unavailability": float(temporal_unavailability_value),
    }
    report["feature_jaccard_dissimilarity"] = (
        feature_jaccard_dissimilarity(feature_masks) if feature_masks else 0.0
    )
    return report

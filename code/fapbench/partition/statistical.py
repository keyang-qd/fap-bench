from __future__ import annotations

import zlib
from collections.abc import Iterable

import numpy as np
import pandas as pd


def _stable_seed(seed: int, value: str, salt: int = 0) -> int:
    digest = zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF
    return int((int(seed) + int(salt) + digest) % (2**32 - 1))


def apply_client_size_imbalance(
    train_frame: pd.DataFrame,
    *,
    entity_col: str,
    client_col: str = "client_id",
    imbalance_ratio: float = 1.0,
    seed: int = 42,
    min_entities_per_client: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subsample complete entity histories to create controlled client-size skew.

    The largest client retains all of its entities and the smallest retains
    approximately ``1 / imbalance_ratio``. Fractions are log-spaced and randomly
    assigned to clients using a deterministic seed. Complete entity histories are
    kept together, so the perturbation does not create partial security histories.
    """
    ratio = float(imbalance_ratio)
    if ratio < 1.0:
        raise ValueError("imbalance_ratio must be at least 1.0.")
    if min_entities_per_client < 1:
        raise ValueError("min_entities_per_client must be at least 1.")
    if train_frame.empty or ratio == 1.0:
        clients = sorted(train_frame[client_col].astype(str).unique())
        report = pd.DataFrame(
            {
                "client_id": clients,
                "requested_fraction": 1.0,
                "entities_before": [
                    train_frame.loc[
                        train_frame[client_col].astype(str) == client, entity_col
                    ].nunique()
                    for client in clients
                ],
            }
        )
        report["entities_after"] = report["entities_before"]
        report["rows_before"] = [
            int((train_frame[client_col].astype(str) == client).sum())
            for client in clients
        ]
        report["rows_after"] = report["rows_before"]
        report["realized_fraction"] = 1.0
        return train_frame.copy(), report

    clients = sorted(train_frame[client_col].astype(str).unique())
    rng = np.random.default_rng(seed)
    fractions = np.geomspace(1.0, 1.0 / ratio, num=len(clients))
    fractions = fractions[rng.permutation(len(fractions))]

    kept_parts: list[pd.DataFrame] = []
    report_rows: list[dict[str, float | int | str]] = []
    client_series = train_frame[client_col].astype(str)
    for client, fraction in zip(clients, fractions, strict=True):
        group = train_frame.loc[client_series == client]
        entities = group[entity_col].dropna().drop_duplicates().to_numpy()
        if len(entities) == 0:
            raise ValueError(f"Client {client!r} has no valid entities.")
        n_keep = max(
            min_entities_per_client,
            int(np.ceil(float(fraction) * len(entities))),
        )
        n_keep = min(len(entities), n_keep)
        client_rng = np.random.default_rng(_stable_seed(seed, client, salt=17))
        selected_entities = client_rng.choice(entities, size=n_keep, replace=False)
        selected = group.loc[group[entity_col].isin(selected_entities)].copy()
        kept_parts.append(selected)
        report_rows.append(
            {
                "client_id": client,
                "requested_fraction": float(fraction),
                "entities_before": int(len(entities)),
                "entities_after": int(n_keep),
                "rows_before": int(len(group)),
                "rows_after": int(len(selected)),
                "realized_fraction": float(len(selected) / len(group)),
            }
        )

    result = pd.concat(kept_parts, ignore_index=True)
    return result, pd.DataFrame(report_rows).sort_values("client_id").reset_index(drop=True)


def apply_regression_label_skew(
    train_frame: pd.DataFrame,
    *,
    target_col: str,
    client_col: str = "client_id",
    n_bins: int = 5,
    concentration: float = 1.0,
    retain_fraction: float = 0.7,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create continuous-label skew by weighted within-client subsampling.

    Global target quantile bins are fitted on the training split only. Each client
    receives deterministic Dirichlet preferences over these bins, then retains a
    fixed fraction of its rows without replacement. Lower concentration creates
    stronger label skew. Keeping the retention fraction fixed avoids conflating
    different skew strengths with different total training budgets.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if concentration <= 0:
        raise ValueError("concentration must be positive.")
    if not 0.0 < retain_fraction <= 1.0:
        raise ValueError("retain_fraction must be in (0, 1].")
    if train_frame.empty or retain_fraction == 1.0:
        clients = sorted(train_frame[client_col].astype(str).unique())
        return train_frame.copy(), pd.DataFrame(
            {
                "client_id": clients,
                "rows_before": [
                    int((train_frame[client_col].astype(str) == client).sum())
                    for client in clients
                ],
                "rows_after": [
                    int((train_frame[client_col].astype(str) == client).sum())
                    for client in clients
                ],
                "realized_fraction": 1.0,
                "dirichlet_concentration": float(concentration),
            }
        )

    target = pd.to_numeric(train_frame[target_col], errors="coerce")
    if target.notna().sum() < n_bins:
        raise ValueError("Not enough finite labels to build label-skew bins.")
    _, edges = pd.qcut(target, q=n_bins, retbins=True, duplicates="drop")
    edges = np.unique(np.asarray(edges, dtype=float))
    if len(edges) < 3:
        raise ValueError("Training labels do not produce at least two quantile bins.")
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins = pd.cut(target, bins=edges, labels=False, include_lowest=True)
    n_realized_bins = len(edges) - 1

    kept_parts: list[pd.DataFrame] = []
    report_rows: list[dict[str, float | int | str]] = []
    client_series = train_frame[client_col].astype(str)
    for client in sorted(client_series.unique()):
        positions = np.flatnonzero((client_series == client).to_numpy())
        group_bins = bins.iloc[positions].to_numpy()
        valid = np.isfinite(group_bins)
        positions = positions[valid]
        group_bins = group_bins[valid].astype(int)
        if len(positions) == 0:
            raise ValueError(f"Client {client!r} has no finite labels.")
        n_keep = max(1, int(np.ceil(retain_fraction * len(positions))))
        n_keep = min(n_keep, len(positions))
        client_rng = np.random.default_rng(_stable_seed(seed, client, salt=31))
        preferences = client_rng.dirichlet(
            np.full(n_realized_bins, float(concentration), dtype=float)
        )
        row_weights = preferences[group_bins]
        if not np.isfinite(row_weights).all() or row_weights.sum() <= 0:
            row_weights = np.ones(len(positions), dtype=float)
        probabilities = row_weights / row_weights.sum()
        selected_positions = client_rng.choice(
            positions, size=n_keep, replace=False, p=probabilities
        )
        selected = train_frame.iloc[np.sort(selected_positions)].copy()
        kept_parts.append(selected)
        selected_bins = bins.iloc[selected_positions].value_counts(normalize=True)
        report: dict[str, float | int | str] = {
            "client_id": client,
            "rows_before": int(len(positions)),
            "rows_after": int(n_keep),
            "realized_fraction": float(n_keep / len(positions)),
            "dirichlet_concentration": float(concentration),
        }
        for bin_index in range(n_realized_bins):
            report[f"target_bin_share_{bin_index}"] = float(
                selected_bins.get(bin_index, 0.0)
            )
            report[f"target_bin_preference_{bin_index}"] = float(
                preferences[bin_index]
            )
        report_rows.append(report)

    result = pd.concat(kept_parts, ignore_index=True)
    return result, pd.DataFrame(report_rows).sort_values("client_id").reset_index(drop=True)


def build_covariate_shift_parameters(
    client_ids: Iterable[str],
    feature_names: Iterable[str],
    *,
    location_std: float = 0.0,
    log_scale_std: float = 0.0,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build deterministic client-specific affine feature transformations."""
    if location_std < 0 or log_scale_std < 0:
        raise ValueError("Covariate-shift standard deviations must be non-negative.")
    features = list(feature_names)
    locations: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for client in sorted(str(value) for value in client_ids):
        rng = np.random.default_rng(_stable_seed(seed, client, salt=47))
        locations[client] = rng.normal(
            loc=0.0, scale=float(location_std), size=len(features)
        ).astype(np.float32)
        scales[client] = np.exp(
            rng.normal(loc=0.0, scale=float(log_scale_std), size=len(features))
        ).astype(np.float32)
    return locations, scales


def apply_covariate_shift(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
    locations: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    *,
    client_col: str = "client_id",
) -> pd.DataFrame:
    """Apply the same client-specific feature pipeline to all time splits."""
    data = frame.copy()
    features = list(feature_names)
    for client, index in data.groupby(client_col, sort=False).groups.items():
        key = str(client)
        if key not in locations or key not in scales:
            raise KeyError(f"No covariate-shift parameters for client {key!r}.")
        values = data.loc[index, features].to_numpy(dtype=np.float32, copy=True)
        data.loc[index, features] = (
            values * scales[key].reshape(1, -1)
            + locations[key].reshape(1, -1)
        )
    return data


def covariate_shift_frame(
    feature_names: Iterable[str],
    locations: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
) -> pd.DataFrame:
    features = list(feature_names)
    rows: list[dict[str, float | str]] = []
    for client in sorted(locations):
        for index, feature in enumerate(features):
            rows.append(
                {
                    "client_id": client,
                    "feature": feature,
                    "location_shift": float(locations[client][index]),
                    "scale_multiplier": float(scales[client][index]),
                }
            )
    return pd.DataFrame(rows)

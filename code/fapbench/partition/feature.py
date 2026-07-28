from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def _common_and_private_pool(
    rng: np.random.Generator, n_features: int, common_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n_features)
    n_common = int(round(common_fraction * n_features))
    if common_fraction > 0 and n_common == 0:
        n_common = 1
    return perm[:n_common], perm[n_common:]


def build_feature_masks(
    feature_names: Iterable[str],
    client_ids: Iterable[str],
    common_fraction: float,
    private_fraction: float = 0.5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Construct common-core plus client-specific binary feature masks.

    ``common_fraction`` controls the fraction available to every client.
    ``private_fraction`` controls how much of the remaining pool each client sees.
    The realized pairwise overlap is reported separately rather than assumed.
    """
    features = list(feature_names)
    clients = list(client_ids)
    if not features:
        raise ValueError("feature_names cannot be empty.")
    if not clients:
        raise ValueError("client_ids cannot be empty.")
    if not 0.0 <= common_fraction <= 1.0:
        raise ValueError("common_fraction must be in [0, 1].")
    if not 0.0 <= private_fraction <= 1.0:
        raise ValueError("private_fraction must be in [0, 1].")

    rng = np.random.default_rng(seed)
    n_features = len(features)
    common, private_pool = _common_and_private_pool(rng, n_features, common_fraction)
    n_private = int(round(private_fraction * len(private_pool)))

    masks: dict[str, np.ndarray] = {}
    for client in clients:
        mask = np.zeros(n_features, dtype=np.float32)
        mask[common] = 1.0
        if n_private > 0:
            selected = rng.choice(private_pool, size=n_private, replace=False)
            mask[selected] = 1.0
        # Every client must observe at least one feature.
        if mask.sum() == 0:
            mask[int(rng.integers(0, n_features))] = 1.0
        masks[str(client)] = mask

    # Avoid features unavailable to every client unless the severe setting forces it.
    union = np.maximum.reduce(list(masks.values()))
    unavailable = np.flatnonzero(union == 0)
    for idx, feature_idx in enumerate(unavailable):
        masks[str(clients[idx % len(clients)])][feature_idx] = 1.0
    return masks


def build_feature_masks_matched_control(
    feature_names: Iterable[str],
    client_ids: Iterable[str],
    common_fraction: float,
    private_fraction: float = 0.5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Matched control for build_feature_masks: identical common+private
    selection for every client.

    build_feature_masks already gives every client the same feature *count*
    (n_common + n_private is identical by construction) -- only *identity*
    (which private features) varies per client. This control draws that
    private selection once instead of once per client, so every client sees
    the exact same feature set, isolating whether cross-client divergence in
    feature identity matters beyond having fewer features at all.

    Uses the same rng seed/prefix as build_feature_masks (via
    _common_and_private_pool), so the shared selection is a genuine collapse
    of the treatment's construction, not an independent draw. Deliberately
    skips the "every feature visible to someone" coverage-completion pass in
    build_feature_masks: that pass exists to give the federation full
    coverage under *independent* per-client draws, and would reintroduce
    per-client asymmetry if applied on top of a single shared draw (verified
    empirically: for the severe config on the 7-feature panel, only 3 of 7
    features are covered by one shared draw, so that pass would silently hand
    4 extra, unevenly-distributed features back to specific clients).
    """
    features = list(feature_names)
    clients = list(client_ids)
    if not features:
        raise ValueError("feature_names cannot be empty.")
    if not clients:
        raise ValueError("client_ids cannot be empty.")
    if not 0.0 <= common_fraction <= 1.0:
        raise ValueError("common_fraction must be in [0, 1].")
    if not 0.0 <= private_fraction <= 1.0:
        raise ValueError("private_fraction must be in [0, 1].")

    rng = np.random.default_rng(seed)
    n_features = len(features)
    common, private_pool = _common_and_private_pool(rng, n_features, common_fraction)
    n_private = int(round(private_fraction * len(private_pool)))

    mask = np.zeros(n_features, dtype=np.float32)
    mask[common] = 1.0
    if n_private > 0:
        selected = rng.choice(private_pool, size=n_private, replace=False)
        mask[selected] = 1.0
    if mask.sum() == 0:
        mask[int(rng.integers(0, n_features))] = 1.0
    return {str(client): mask.copy() for client in clients}


def apply_feature_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if x.shape[-1] != mask.shape[0]:
        raise ValueError("Feature matrix and mask have incompatible dimensions.")
    return x * mask.reshape(1, -1)


def feature_jaccard_dissimilarity(masks: Mapping[str, np.ndarray]) -> float:
    clients = list(masks)
    if len(clients) < 2:
        return 0.0
    distances: list[float] = []
    for i, left in enumerate(clients[:-1]):
        a = np.asarray(masks[left], dtype=bool)
        for right in clients[i + 1 :]:
            b = np.asarray(masks[right], dtype=bool)
            union = np.logical_or(a, b).sum()
            similarity = 1.0 if union == 0 else np.logical_and(a, b).sum() / union
            distances.append(1.0 - float(similarity))
    return float(np.mean(distances))


def apply_feature_masks_to_frame(
    frame,
    feature_names: Iterable[str],
    masks: Mapping[str, np.ndarray],
    client_col: str = "client_id",
):
    """Return a copy with client-specific unavailable features set to zero."""
    import pandas as pd

    data = frame.copy()
    features = list(feature_names)
    for client_id, index in data.groupby(client_col, sort=False).groups.items():
        key = str(client_id)
        if key not in masks:
            raise KeyError(f"No feature mask is defined for client {key!r}.")
        mask = np.asarray(masks[key], dtype=np.float32)
        if len(mask) != len(features):
            raise ValueError("Feature mask length does not match feature_names.")
        values = data.loc[index, features].to_numpy(dtype=float, copy=True)
        data.loc[index, features] = values * mask.reshape(1, -1)
    return data

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


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
    perm = rng.permutation(n_features)
    n_common = int(round(common_fraction * n_features))
    if common_fraction > 0 and n_common == 0:
        n_common = 1
    common = perm[:n_common]
    private_pool = perm[n_common:]
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

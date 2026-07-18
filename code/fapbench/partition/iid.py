from __future__ import annotations

import numpy as np
import pandas as pd


def assign_iid_clients(
    frame: pd.DataFrame,
    entity_col: str,
    num_clients: int,
    seed: int = 42,
    client_col: str = "client_id",
) -> pd.DataFrame:
    """Assign complete entity histories to approximately balanced random clients."""
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")
    # Arrow-backed pandas columns can expose a read-only NumPy view.  Do not
    # shuffle that view in place: doing so raises ``ValueError: array is
    # read-only`` for Parquet-loaded panels.  Materialize an owned array and
    # use ``permutation`` so the source values are never mutated.
    entities = (
        frame[entity_col]
        .dropna()
        .drop_duplicates()
        .to_numpy(copy=True)
    )
    if len(entities) < num_clients:
        raise ValueError(
            f"Only {len(entities)} unique entities are available for {num_clients} clients."
        )
    rng = np.random.default_rng(seed)
    entities = rng.permutation(entities)
    mapping = {
        entity: f"client_{idx % num_clients:02d}"
        for idx, entity in enumerate(entities)
    }
    data = frame.copy()
    data[client_col] = data[entity_col].map(mapping)
    return data

from __future__ import annotations

import numpy as np
import pandas as pd


def _collapse_categories(values: pd.Series, max_clients: int | None) -> pd.Series:
    values = values.astype("string").fillna("MISSING")
    if max_clients is None or values.nunique() <= max_clients:
        return values
    counts = values.value_counts()
    keep = set(counts.head(max_clients - 1).index)
    return values.where(values.isin(keep), "OTHER")


def assign_domain_clients(
    frame: pd.DataFrame,
    domain_col: str,
    max_clients: int | None = None,
    continuous_bins: int | None = None,
    client_col: str = "client_id",
    entity_col: str | None = None,
    persistent_entity: bool = True,
) -> pd.DataFrame:
    """Create financially meaningful clients from a sector/exchange/size domain.

    Categorical columns are mapped directly. A high-cardinality numeric column can
    be quantile-binned by setting ``continuous_bins``.
    """
    if domain_col not in frame.columns:
        raise KeyError(
            f"domain_col={domain_col!r} is absent. Add SIC/NAICS/sector, or use "
            "an available proxy such as exchcd or me."
        )
    data = frame.copy()
    series = data[domain_col]
    if continuous_bins is not None:
        if continuous_bins < 2:
            raise ValueError("continuous_bins must be at least 2.")
        numeric = pd.to_numeric(series, errors="coerce")
        valid_unique = numeric.nunique(dropna=True)
        if valid_unique < 2:
            raise ValueError(f"{domain_col} has fewer than two valid values.")
        bins = min(continuous_bins, valid_unique)
        ranked = numeric.rank(method="first")
        categories = pd.qcut(ranked, q=bins, duplicates="drop").astype("string")
        categories = categories.fillna("MISSING")
    else:
        categories = _collapse_categories(series, max_clients=max_clients)

    if persistent_entity and entity_col is not None:
        if entity_col not in data.columns:
            raise KeyError(f"entity_col={entity_col!r} is absent.")
        category_frame = pd.DataFrame({entity_col: data[entity_col], "_domain": categories})
        entity_domain = category_frame.groupby(entity_col, sort=False)["_domain"].agg(
            lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]
        )
        categories = data[entity_col].map(entity_domain).astype("string")

    labels = sorted(categories.unique().tolist())
    mapping = {label: f"client_{idx:02d}" for idx, label in enumerate(labels)}
    data[client_col] = categories.map(mapping)
    data[f"{client_col}_domain"] = categories
    if data[client_col].nunique() < 2:
        raise ValueError(
            f"Domain partition using {domain_col!r} produced fewer than two clients."
        )
    return data

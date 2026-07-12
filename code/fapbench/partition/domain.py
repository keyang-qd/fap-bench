from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_mask(
    frame: pd.DataFrame,
    fit_split_col: str | None,
    fit_split_value: str,
) -> pd.Series:
    if fit_split_col is None:
        return pd.Series(True, index=frame.index)
    if fit_split_col not in frame.columns:
        return pd.Series(True, index=frame.index)
    mask = frame[fit_split_col].astype(str) == str(fit_split_value)
    if not mask.any():
        raise ValueError(
            f"No rows satisfy {fit_split_col} == {fit_split_value!r}."
        )
    return mask


def _collapse_categories(
    values: pd.Series,
    fit_values: pd.Series,
    max_clients: int | None,
) -> pd.Series:
    values = values.astype("string").fillna("MISSING")
    fit_values = fit_values.astype("string").fillna("MISSING")
    if max_clients is None or fit_values.nunique() <= max_clients:
        return values
    if max_clients < 2:
        raise ValueError("max_clients must be at least 2 when categories are collapsed.")
    counts = fit_values.value_counts()
    keep = set(counts.head(max_clients - 1).index)
    return values.where(values.isin(keep), "OTHER")


def _quantile_categories(
    values: pd.Series,
    fit_values: pd.Series,
    continuous_bins: int,
) -> pd.Series:
    if continuous_bins < 2:
        raise ValueError("continuous_bins must be at least 2.")
    numeric = pd.to_numeric(values, errors="coerce")
    fit_numeric = pd.to_numeric(fit_values, errors="coerce").dropna()
    if fit_numeric.nunique() < 2:
        raise ValueError("The training-period domain variable has fewer than two valid values.")
    quantiles = np.linspace(0.0, 1.0, continuous_bins + 1)
    edges = np.unique(fit_numeric.quantile(quantiles).to_numpy(dtype=float))
    if len(edges) < 3:
        raise ValueError("Training-period quantiles do not produce at least two bins.")
    edges[0] = -np.inf
    edges[-1] = np.inf
    categories = pd.cut(numeric, bins=edges, include_lowest=True).astype("string")
    return categories.fillna("MISSING")


def assign_domain_clients(
    frame: pd.DataFrame,
    domain_col: str,
    max_clients: int | None = None,
    continuous_bins: int | None = None,
    client_col: str = "client_id",
    entity_col: str | None = None,
    persistent_entity: bool = True,
    fit_split_col: str | None = "split",
    fit_split_value: str = "train",
) -> pd.DataFrame:
    """Create domain clients without using validation/test data to fit the partition.

    Category collapsing, quantile-bin edges, and optional persistent entity labels are
    all estimated on the designated fit split (training by default). This prevents
    future periods from changing the client definition.
    """
    if domain_col not in frame.columns:
        raise KeyError(
            f"domain_col={domain_col!r} is absent. Add SIC/FF12/NAICS/sector, or "
            "use an available proxy such as exchcd or me."
        )
    data = frame.copy()
    mask = _fit_mask(data, fit_split_col, fit_split_value)
    series = data[domain_col]

    if continuous_bins is not None:
        categories = _quantile_categories(
            series, series.loc[mask], continuous_bins
        )
    else:
        categories = _collapse_categories(
            series, series.loc[mask], max_clients=max_clients
        )

    if persistent_entity and entity_col is not None:
        if entity_col not in data.columns:
            raise KeyError(f"entity_col={entity_col!r} is absent.")
        fit_categories = pd.DataFrame(
            {
                entity_col: data.loc[mask, entity_col],
                "_domain": categories.loc[mask],
            }
        )
        entity_domain = fit_categories.groupby(entity_col, sort=False)["_domain"].agg(
            lambda values: values.mode().iloc[0]
            if not values.mode().empty
            else values.iloc[0]
        )
        mapped = data[entity_col].map(entity_domain).astype("string")
        # New securities entering after the training period use their contemporaneous
        # domain; they do not affect the training-fitted mapping of existing entities.
        categories = mapped.fillna(categories)

    training_categories = set(categories.loc[mask].dropna().astype(str))
    unseen_categories = set(categories.loc[~mask].dropna().astype(str)) - training_categories
    if unseen_categories:
        raise ValueError(
            "Evaluation periods contain domain clients absent from training: "
            f"{sorted(unseen_categories)}. Merge rare categories, extend the training "
            "window, or use a fixed industry mapping such as FF12."
        )

    labels = sorted(training_categories)
    mapping = {label: f"client_{idx:02d}" for idx, label in enumerate(labels)}
    data[client_col] = categories.astype(str).map(mapping)
    data[f"{client_col}_domain"] = categories.astype("string")
    if data[client_col].isna().any():
        raise ValueError("Some rows could not be assigned to a training-defined client.")
    if data.loc[mask, client_col].nunique() < 2:
        raise ValueError(
            f"Domain partition using {domain_col!r} produced fewer than two training clients."
        )
    return data

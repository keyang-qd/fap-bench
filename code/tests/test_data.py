"""Tests for panel-level data transforms in fapbench.data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

def _client_panel(seed: int = 0, n: int = 480) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "client_id": rng.choice(list("abcd"), n),
        "date": rng.choice(pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]), n),
        "f1": rng.lognormal(size=n),
        "f2": rng.normal(size=n),
    })


def test_client_local_rank_zscore_standardizes_within_client_month():
    from fapbench.data import client_local_rank_zscore

    out = client_local_rank_zscore(_client_panel(), ["f1", "f2"])
    grouped = out.groupby(["client_id", "date"])[["f1", "f2"]]
    assert np.allclose(grouped.mean().to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(grouped.std(ddof=0).to_numpy(), 1.0, atol=1e-12)
    # Client sample sizes must be untouched: the size-imbalance treatments are
    # compared across arms, so dropping thin client-months would confound them.
    assert len(out) == 480


def test_client_local_rank_zscore_is_invariant_to_monotone_upstream_steps():
    """The claim that makes this a genuine client-local statistic.

    The panel is already globally rank-z-scored and then affinely rescaled by
    PanelPreprocessor. Both are monotone within a month, so re-ranking inside a
    client must give exactly what ranking the raw characteristic inside that
    client would have given -- otherwise this would be a re-scaling of a pooled
    statistic rather than a client-local one.
    """
    from fapbench.data import client_local_rank_zscore

    frame = _client_panel(seed=7)
    from_raw = client_local_rank_zscore(frame, ["f1"])["f1"]

    ranks = frame.groupby("date")["f1"].rank()
    by_date = ranks.groupby(frame["date"])
    globally = (ranks - by_date.transform("mean")) / by_date.transform("std").replace(0, np.nan)
    rescaled = frame.assign(f1=globally * 2.7 - 0.4)   # affine, monotone
    from_pipeline = client_local_rank_zscore(rescaled, ["f1"])["f1"]

    assert np.allclose(from_raw.to_numpy(), from_pipeline.to_numpy(), atol=1e-12)


def test_client_local_rank_zscore_handles_singleton_client_months():
    from fapbench.data import client_local_rank_zscore

    frame = pd.DataFrame({
        "client_id": ["a", "a", "b"],
        "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-01-31"]),
        "f1": [1.0, 2.0, 5.0],
    })
    out = client_local_rank_zscore(frame, ["f1"])
    # Client b has a single observation, so there is no cross-section to rank;
    # it becomes 0.0 (the standardized mean) rather than NaN or a dropped row.
    assert out.loc[2, "f1"] == 0.0
    assert not out["f1"].isna().any()


def test_client_local_rank_zscore_requires_the_client_column():
    from fapbench.data import client_local_rank_zscore

    frame = _client_panel().drop(columns=["client_id"])
    with pytest.raises(KeyError, match="client"):
        client_local_rank_zscore(frame, ["f1"])

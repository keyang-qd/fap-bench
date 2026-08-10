"""Exact federated ridge: the coordination cost is zero by construction.

These tests exist to make a claim in the manuscript checkable rather than
asserted -- that for the benchmark's strongest model class, federation is
lossless and costs 176 bytes per client, once.
"""
from __future__ import annotations

import numpy as np
import pytest

from fapbench.exact_ridge import (
    centralized_ridge,
    client_sufficient_statistics,
    federated_ridge,
    solve_ridge,
)


def _panel(n=4000, p=7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(scale=2.0, size=n)
    return X, y


@pytest.mark.parametrize("n_clients", [2, 5, 12, 30])
def test_federated_ridge_is_bit_identical_to_pooled(n_clients):
    """The load-bearing claim: splitting the panel across clients changes nothing.

    Not 'close' -- identical to floating-point round-off, at every client count,
    including a very uneven split.
    """
    X, y = _panel()
    rng = np.random.default_rng(1)
    assignment = rng.integers(0, n_clients, len(X))
    parts_X = [X[assignment == k] for k in range(n_clients)]
    parts_y = [y[assignment == k] for k in range(n_clients)]
    # Drop any empty client rather than feeding a degenerate Gram.
    keep = [i for i in range(n_clients) if len(parts_X[i]) > 0]
    parts_X = [parts_X[i] for i in keep]
    parts_y = [parts_y[i] for i in keep]

    pooled = centralized_ridge(X, y, l2_penalty=1.0)
    federated, _ = federated_ridge(parts_X, parts_y, l2_penalty=1.0)
    assert np.allclose(pooled, federated, rtol=0, atol=1e-10)


def test_extreme_size_imbalance_does_not_degrade_the_solution():
    """Size imbalance is a treatment elsewhere in the benchmark and costs accuracy
    for gradient averaging. Here it must cost exactly nothing."""
    X, y = _panel()
    # One client holds 99% of the rows, nine split the remainder.
    cut = int(0.99 * len(X))
    parts_X = [X[:cut]] + np.array_split(X[cut:], 9)
    parts_y = [y[:cut]] + np.array_split(y[cut:], 9)
    pooled = centralized_ridge(X, y, l2_penalty=1.0)
    federated, _ = federated_ridge(list(parts_X), list(parts_y), l2_penalty=1.0)
    assert np.allclose(pooled, federated, rtol=0, atol=1e-10)


def test_payload_is_44_floats_at_the_reference_panel_width():
    """The manuscript quotes 176 bytes per client for the 7-characteristic panel."""
    X, y = _panel(p=7)
    stats = client_sufficient_statistics(X, y)
    assert stats.gram.shape == (8, 8)  # 7 features + intercept
    assert stats.payload_floats == 8 * 9 // 2 + 8 == 44
    assert stats.payload_bytes_fp32 == 176

    _, total_bytes = federated_ridge([X[:2000], X[2000:]], [y[:2000], y[2000:]])
    assert total_bytes == 2 * 176


def test_intercept_penalty_convention_is_shared_by_both_paths():
    """If the two paths disagreed on penalizing the intercept, 'exact' would be
    quietly false. Pin that they agree under both conventions."""
    X, y = _panel()
    parts_X, parts_y = [X[:1500], X[1500:]], [y[:1500], y[1500:]]
    for penalize in (True, False):
        pooled = centralized_ridge(X, y, l2_penalty=3.0, penalize_intercept=penalize)
        fed, _ = federated_ridge(
            parts_X, parts_y, l2_penalty=3.0, penalize_intercept=penalize
        )
        assert np.allclose(pooled, fed, rtol=0, atol=1e-10)
    # And that the convention actually matters, so the test above is not vacuous.
    assert not np.allclose(
        centralized_ridge(X, y, l2_penalty=3.0, penalize_intercept=True),
        centralized_ridge(X, y, l2_penalty=3.0, penalize_intercept=False),
    )


def test_mismatched_feature_dimensions_are_rejected():
    X, y = _panel(p=7)
    other, other_y = _panel(p=5, seed=2)
    with pytest.raises(ValueError, match="same feature dimension"):
        solve_ridge(
            [
                client_sufficient_statistics(X, y),
                client_sufficient_statistics(other, other_y),
            ]
        )


def test_empty_and_ragged_inputs_are_rejected():
    X, y = _panel()
    with pytest.raises(ValueError, match="At least one"):
        solve_ridge([])
    with pytest.raises(ValueError, match="same length"):
        client_sufficient_statistics(X, y[:-1])

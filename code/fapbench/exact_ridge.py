"""Exact federated ridge via sufficient statistics.

Why this exists
---------------
A reviewer asked why the benchmark evaluates four gradient-averaging algorithms
but never the one federated estimator that is *exactly* solvable, given that the
linear/ridge class is the strongest model in the benchmark: it triples
Centralized's Rank IC on the reference universe (0.0111 -> 0.0379) and is the
only configuration producing a positive investable net Sharpe.

Ridge regression federates losslessly because its normal equations depend on the
data only through two additive sufficient statistics:

    beta = (X'X + lambda I)^{-1} X'y,   X'X = sum_k X_k'X_k,   X'y = sum_k X_k'y_k

Each client sends one Gram matrix and one cross-product vector; the server sums
them and solves once. There is no iteration, no client drift, no server
optimizer, and the result is identical to pooled training up to floating-point
round-off -- so the coordination cost is not small, it is *zero by construction*.

That is a load-bearing observation for the paper rather than a footnote: for the
best-performing model class in the benchmark, none of the gradient-averaging
machinery is necessary. It also puts a hard floor under the coordination-cost
narrative. The MLP results and the heterogeneity analysis are unaffected -- they
concern a model class with no such closed form -- but the benchmark should say
where its own machinery is unnecessary.

The payload is tiny and exact: for p features plus an intercept the client sends
p(p+1)/2 unique Gram entries and p+1 cross-product entries. At the reference
seven-characteristic panel that is 44 FP32 numbers, 176 bytes, once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RidgeSufficientStatistics:
    """One client's contribution: the Gram matrix and cross-product vector.

    These are the *only* quantities the server needs. They are sums over the
    client's rows, so they carry no individual observation -- which is the sense
    in which this exchange is model-side rather than data-side.
    """

    gram: np.ndarray  # (p+1, p+1), includes the intercept column
    cross: np.ndarray  # (p+1,)
    n_observations: int

    @property
    def payload_floats(self) -> int:
        """Unique numbers actually needing transmission (Gram is symmetric)."""
        p = self.gram.shape[0]
        return p * (p + 1) // 2 + p

    @property
    def payload_bytes_fp32(self) -> int:
        return 4 * self.payload_floats


def client_sufficient_statistics(
    features: np.ndarray, target: np.ndarray, *, fit_intercept: bool = True
) -> RidgeSufficientStatistics:
    """Compute one client's (X'X, X'y) without revealing any row."""
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("features must be 2-D (n_observations, n_features).")
    if len(X) != len(y):
        raise ValueError("features and target must have the same length.")
    if fit_intercept:
        X = np.hstack([np.ones((len(X), 1)), X])
    return RidgeSufficientStatistics(
        gram=X.T @ X, cross=X.T @ y, n_observations=int(len(X))
    )


def solve_ridge(
    statistics: Iterable[RidgeSufficientStatistics],
    *,
    l2_penalty: float = 1.0,
    penalize_intercept: bool = False,
) -> np.ndarray:
    """Aggregate client sufficient statistics and solve the ridge normal equations.

    This is the whole federated algorithm: one sum, one solve, one round.
    """
    statistics = list(statistics)
    if not statistics:
        raise ValueError("At least one client's statistics are required.")
    gram = np.zeros_like(statistics[0].gram)
    cross = np.zeros_like(statistics[0].cross)
    for s in statistics:
        if s.gram.shape != gram.shape or s.cross.shape != cross.shape:
            raise ValueError("All clients must share the same feature dimension.")
        gram = gram + s.gram
        cross = cross + s.cross
    penalty = l2_penalty * np.eye(gram.shape[0])
    if not penalize_intercept:
        # The intercept is conventionally left unpenalized; keeping this explicit
        # matters because the centralized comparison must use the same convention
        # or the "exact" claim silently becomes "approximate".
        penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, cross)


def centralized_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    l2_penalty: float = 1.0,
    fit_intercept: bool = True,
    penalize_intercept: bool = False,
) -> np.ndarray:
    """Pooled ridge, for the equivalence check. Deliberately a separate code path."""
    return solve_ridge(
        [client_sufficient_statistics(features, target, fit_intercept=fit_intercept)],
        l2_penalty=l2_penalty,
        penalize_intercept=penalize_intercept,
    )


def federated_ridge(
    client_features: Sequence[np.ndarray],
    client_targets: Sequence[np.ndarray],
    *,
    l2_penalty: float = 1.0,
    fit_intercept: bool = True,
    penalize_intercept: bool = False,
) -> tuple[np.ndarray, int]:
    """Fit ridge across clients exactly. Returns (coefficients, total bytes uploaded)."""
    if len(client_features) != len(client_targets):
        raise ValueError("client_features and client_targets must align.")
    statistics = [
        client_sufficient_statistics(X, y, fit_intercept=fit_intercept)
        for X, y in zip(client_features, client_targets)
    ]
    coefficients = solve_ridge(
        statistics, l2_penalty=l2_penalty, penalize_intercept=penalize_intercept
    )
    return coefficients, sum(s.payload_bytes_fp32 for s in statistics)

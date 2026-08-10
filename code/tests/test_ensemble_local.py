"""Tests for the ensemble-of-locals baseline.

The baseline answers "why federate at all, when you could ship the K finished
local models once and average their opinions?" -- so what has to be pinned down is
the aggregation rule, not the training, which is shared with the local baseline.

The load-bearing design choice is that *ranks* are averaged, not scores.
Independently trained local models are not calibrated against each other, so a
score average would let one member's scale decide the ensemble. Rank averaging
makes the result depend only on each member's ordering. That property is what the
first test asserts, because it is the thing a future refactor could quietly break
while every other test still passed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fapbench.runner import _ensemble_local_prediction_frame

FEATURES = ["f0", "f1"]


class _Affine(torch.nn.Module):
    """A fixed, non-trained model: w . x, then an optional monotone rescaling.

    Using an exactly-known model rather than a trained one keeps these tests
    about the aggregation rule. `scale`/`shift` stand in for a member being
    miscalibrated relative to the others while ordering stocks identically.
    """

    def __init__(self, weights, *, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))
        self.scale = float(scale)
        self.shift = float(shift)

    def forward(self, x):  # noqa: D102
        return (x @ self.weights) * self.scale + self.shift


def _frame(n_dates: int = 3, n_per_date: int = 6, n_clients: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for date_index in range(n_dates):
        for row_index in range(n_per_date):
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(date_index),
                    "permno": 1000 + row_index,
                    "client_id": str(row_index % n_clients),
                    "f0": float(rng.normal()),
                    "f1": float(rng.normal()),
                    "ret_next": float(rng.normal()),
                }
            )
    frame = pd.DataFrame(rows)
    frame["target_date"] = frame["date"] + pd.offsets.MonthEnd(1)
    return frame


def _predict(models, frame, feature_masks=None) -> pd.DataFrame:
    return _ensemble_local_prediction_frame(
        models,
        frame,
        FEATURES,
        "ret_next",
        "date",
        "permno",
        "client_id",
        "cpu",
        feature_masks,
        target_date_col="target_date",
    )


def test_ensemble_is_invariant_to_per_member_monotone_rescaling():
    """The property that motivates rank-averaging over score-averaging.

    Two ensembles whose members order stocks identically but whose output scales
    differ by orders of magnitude must produce the same ensemble prediction. A
    score-averaging implementation fails this: the member scaled by 1000 would
    dominate.
    """
    frame = _frame()
    plain = {
        "0": _Affine([1.0, 0.0]),
        "1": _Affine([0.0, 1.0]),
    }
    rescaled = {
        "0": _Affine([1.0, 0.0], scale=1000.0, shift=-5.0),
        "1": _Affine([0.0, 1.0], scale=0.001, shift=42.0),
    }
    left = _predict(plain, frame).sort_values(["date", "permno"]).reset_index(drop=True)
    right = (
        _predict(rescaled, frame).sort_values(["date", "permno"]).reset_index(drop=True)
    )
    pd.testing.assert_series_equal(left["y_pred"], right["y_pred"])
    # Guard against the assertion passing trivially on a constant column.
    assert left["y_pred"].nunique() > 1


def test_single_member_ensemble_reproduces_that_member_ranking():
    """K=1 must reduce to the one model's cross-sectional percentile rank."""
    frame = _frame()
    model = _Affine([1.0, -0.5])
    result = _predict({"0": model}, frame)
    with torch.no_grad():
        raw = (
            model(torch.as_tensor(frame[FEATURES].to_numpy(), dtype=torch.float32))
            .cpu()
            .numpy()
        )
    expected = (
        frame.assign(raw=raw)
        .groupby("date")["raw"]
        .rank(method="average", pct=True)
    )
    merged = frame.assign(expected=expected).merge(
        result[["date", "permno", "y_pred"]], on=["date", "permno"]
    )
    np.testing.assert_allclose(
        merged["y_pred"].to_numpy(), merged["expected"].to_numpy(), atol=1e-6
    )


def test_each_member_scores_under_its_own_feature_mask():
    """A client that never saw a characteristic must not get it back at inference.

    Both members here weight only `f1`, but member "0"'s mask zeroes `f1`. Its
    scores therefore collapse to a constant, its ranks to a within-month constant,
    and the ensemble must reflect that -- if masks were ignored, the two members
    would be identical and the ensemble would equal a single unmasked member.
    """
    frame = _frame()
    models = {"0": _Affine([0.0, 1.0]), "1": _Affine([0.0, 1.0])}
    masks = {
        "0": np.array([1.0, 0.0], dtype=np.float32),
        "1": np.array([1.0, 1.0], dtype=np.float32),
    }
    masked = _predict(models, frame, masks)
    unmasked_single = _predict({"1": models["1"]}, frame)
    merged = masked.merge(
        unmasked_single[["date", "permno", "y_pred"]],
        on=["date", "permno"],
        suffixes=("_masked", "_unmasked"),
    )
    assert not np.allclose(
        merged["y_pred_masked"].to_numpy(), merged["y_pred_unmasked"].to_numpy()
    )


def test_row_count_is_preserved_and_empty_models_are_rejected():
    frame = _frame()
    result = _predict({"0": _Affine([1.0, 1.0]), "1": _Affine([1.0, -1.0])}, frame)
    assert len(result) == len(frame)
    assert result["y_pred"].between(0.0, 1.0).all()
    with pytest.raises(ValueError, match="at least one local model"):
        _predict({}, frame)


def test_ensemble_baseline_is_disabled_by_default():
    """Every published config predates this baseline and must be unaffected."""
    from fapbench.config import load_config

    config = load_config("configs/synthetic_smoke.yaml")
    assert not bool(config.get("baselines", {}).get("ensemble_local", False))


def test_level_dependent_metrics_are_blanked_end_to_end(tmp_path):
    """A rank prediction must never be scored against realized returns.

    Without this the smoke run reports an OOS \(R^2\) of about -20, which is not a
    poor result but a category error: the prediction is a percentile in [0, 1] and
    the target is a monthly return. This test runs the real pipeline rather than the
    helper, because the blanking lives in the runner, not the aggregation.
    """
    import yaml

    from fapbench.config import load_config
    from fapbench.runner import run

    config = load_config("configs/synthetic_smoke.yaml")
    config["baselines"]["ensemble_local"] = True
    config["output_dir"] = str(tmp_path / "out")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run(load_config(config_path))
    results = pd.read_csv(tmp_path / "out" / "benchmark_results.csv")
    row = results.loc[results.method == "ensemble_local"]
    assert len(row) == 1
    for column in ("mse", "mae", "oos_r2_zero"):
        if column in row:
            assert row[column].isna().all(), f"{column} must be blank for a ranking"
    # The metrics that *are* meaningful for a ranking must still be present.
    assert row["rank_ic"].notna().all()
    assert row["client_neutral_portfolio_net_sharpe"].notna().all()


def test_ensemble_without_local_baseline_is_rejected(tmp_path):
    """Skipping silently would leave a gap that looks like a crashed run."""
    import yaml

    from fapbench.config import load_config
    from fapbench.runner import run

    config = load_config("configs/synthetic_smoke.yaml")
    config["baselines"]["local"] = False
    config["baselines"]["ensemble_local"] = True
    config["output_dir"] = str(tmp_path / "out")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="requires baselines.local"):
        run(load_config(config_path))

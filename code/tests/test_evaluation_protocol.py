import numpy as np
import pandas as pd
import pytest
import torch

from fapbench.evaluation import (
    calibrate_predictions_within_client,
    client_portfolio_metrics,
    long_short_portfolio_metrics,
    predictive_metrics,
)
from fapbench.selection import ValidationCheckpoint


def _panel() -> pd.DataFrame:
    rows = []
    for month in pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]):
        for client, offset, scale in [("a", 100.0, 10.0), ("b", -100.0, 0.1)]:
            for rank in range(10):
                rows.append(
                    {
                        "date": month,
                        "client_id": client,
                        "permno": f"{client}-{rank}",
                        "y_true": float(rank - 4.5),
                        "y_pred": offset + scale * rank,
                    }
                )
    return pd.DataFrame(rows)


def test_local_score_calibration_removes_client_location_and_scale() -> None:
    frame = _panel()
    calibrated = calibrate_predictions_within_client(frame)
    grouped = calibrated.groupby(["date", "client_id"])["portfolio_score"]
    assert grouped.min().min() == pytest.approx(0.1)
    assert grouped.max().max() == pytest.approx(1.0)
    assert grouped.mean().std() == pytest.approx(0.0)


def test_portfolio_reports_client_concentration() -> None:
    frame = _panel()
    _, metrics = long_short_portfolio_metrics(
        frame, date_col="date", quantile=0.2, client_col="client_id"
    )
    assert 0.0 < metrics["mean_long_client_hhi"] <= 1.0
    assert 0.0 < metrics["mean_short_client_hhi"] <= 1.0
    assert 0.0 < metrics["mean_long_max_client_share"] <= 1.0


def test_client_portfolio_summary_contains_worst_client() -> None:
    frame = _panel()
    per_client, summary = client_portfolio_metrics(
        frame, date_col="date", quantile=0.2
    )
    assert len(per_client) == 2
    assert "worst_net_sharpe" in summary
    assert "weighted_mean_net_sharpe" in summary


def test_predictive_metrics_include_ic_dispersion() -> None:
    metrics = predictive_metrics(_panel().query("client_id == 'a'"), date_col="date")
    assert metrics["rank_ic"] == pytest.approx(1.0)
    assert metrics["rank_ic_positive_ratio"] == pytest.approx(1.0)
    assert metrics["n_ic_months"] == 3
    assert "rank_icir" in metrics


def test_validation_checkpoint_restores_best_state() -> None:
    model = torch.nn.Linear(1, 1, bias=False)

    def score_fn(current: torch.nn.Module) -> float:
        return -abs(float(current.weight.detach().cpu().item()) - 2.0)

    selector = ValidationCheckpoint(
        score_fn=score_fn,
        metric="synthetic",
        mode="max",
        evaluate_every=1,
        total_steps=3,
    )
    for step, weight in enumerate([0.0, 2.0, 4.0], start=1):
        with torch.no_grad():
            model.weight.fill_(weight)
        selector(model, step, train_loss=float(step))
    selector.restore(model)
    assert float(model.weight.detach().cpu().item()) == pytest.approx(2.0)
    assert selector.best_step == 2

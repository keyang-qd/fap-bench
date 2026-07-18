from __future__ import annotations

import numpy as np
import pandas as pd

from fapbench.evaluation import (
    apply_linear_prediction_calibrator,
    fit_linear_prediction_calibrator,
    predictive_metrics,
)
from fapbench.partition import (
    apply_partition_common_universe,
    apply_size_matched_uniform_control,
    assign_domain_clients,
    randomize_schedule_client_identity,
)


def test_common_universe_produces_row_identical_partition_inputs() -> None:
    frame = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "sic": [2000, np.nan, 6000, 5000],
            "exchcd": [1, 2, np.nan, 3],
            "me": [10.0, 20.0, 30.0, -1.0],
            "split": ["train", "train", "test", "test"],
        }
    )
    filtered, report = apply_partition_common_universe(
        frame,
        required_columns=["sic", "exchcd", "me"],
        finite_numeric_columns=["sic", "exchcd", "me"],
        positive_columns=["me"],
        ff12_sic_col="sic",
    )
    assert filtered["permno"].tolist() == [1]
    assert report["n_rows_before"] == 4
    assert report["n_rows_after"] == 1


def test_continuous_domain_drop_missing_has_exact_requested_bins() -> None:
    frame = pd.DataFrame(
        {
            "permno": np.arange(11),
            "me": [np.nan] + list(np.arange(1.0, 11.0)),
            "split": ["train"] * 11,
        }
    )
    assigned = assign_domain_clients(
        frame,
        domain_col="me",
        continuous_bins=5,
        fit_split_col="split",
        missing_policy="drop",
    )
    assert len(assigned) == 10
    assert assigned["client_id"].nunique() == 5
    assert "MISSING" not in set(assigned["client_id_domain"].astype(str))


def test_size_matched_control_targets_same_total_coverage() -> None:
    rows = []
    for client_index, client in enumerate(["A", "B", "C", "D"]):
        for entity in range(30 + 5 * client_index):
            for month in range(2 + entity % 2):
                rows.append(
                    {
                        "client_id": client,
                        "permno": client_index * 1000 + entity,
                        "month": month,
                    }
                )
    frame = pd.DataFrame(rows)
    controlled, report, metadata = apply_size_matched_uniform_control(
        frame,
        entity_col="permno",
        matched_imbalance_ratio=20.0,
        seed=11,
    )
    assert controlled["client_id"].nunique() == 4
    assert report["requested_fraction"].nunique() == 1
    assert abs(metadata["row_match_error"]) < 0.1 * len(frame)


def test_randomized_schedule_preserves_round_counts_but_breaks_identity() -> None:
    reference = {
        0: ["A", "B"],
        1: ["A", "B", "C"],
        2: ["A"],
        3: ["A", "D"],
    }
    randomized = randomize_schedule_client_identity(
        reference, ["A", "B", "C", "D"], seed=17
    )
    assert [len(randomized[i]) for i in reference] == [len(reference[i]) for i in reference]
    reference_client_counts = {
        client: sum(client in active for active in reference.values())
        for client in ["A", "B", "C", "D"]
    }
    randomized_client_counts = {
        client: sum(client in active for active in randomized.values())
        for client in ["A", "B", "C", "D"]
    }
    assert randomized_client_counts == reference_client_counts
    assert randomized == randomize_schedule_client_identity(
        reference, ["A", "B", "C", "D"], seed=17
    )
    assert randomized != reference


def test_validation_linear_calibration_improves_scale_without_changing_rank() -> None:
    validation = pd.DataFrame(
        {
            "y_true": np.linspace(-1.0, 1.0, 100),
            "y_pred": 10.0 * np.linspace(-1.0, 1.0, 100) + 3.0,
        }
    )
    test = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2020-01-31", periods=5, freq="ME"), 20),
            "y_true": np.tile(np.linspace(-1.0, 1.0, 20), 5),
            "y_pred": np.tile(10.0 * np.linspace(-1.0, 1.0, 20) + 3.0, 5),
        }
    )
    calibrator = fit_linear_prediction_calibrator(validation)
    calibrated = apply_linear_prediction_calibrator(test, calibrator)
    raw_metrics = predictive_metrics(test, date_col="date")
    calibrated_metrics = predictive_metrics(calibrated, date_col="date")
    assert calibrated_metrics["mse"] < raw_metrics["mse"]
    assert np.isclose(calibrated_metrics["rank_ic"], raw_metrics["rank_ic"])
    assert np.isclose(calibrator.slope, 0.1, atol=1e-6)


def test_validation_calibration_flags_nonpositive_direction() -> None:
    validation = pd.DataFrame(
        {
            "y_true": np.linspace(1.0, -1.0, 50),
            "y_pred": np.linspace(-1.0, 1.0, 50),
        }
    )
    calibrator = fit_linear_prediction_calibrator(
        validation, nonnegative_slope=True
    )
    assert calibrator.raw_slope < 0.0
    assert calibrator.slope == 0.0
    assert calibrator.direction_valid is False

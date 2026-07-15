from __future__ import annotations

import numpy as np
import pandas as pd

from fapbench.partition import (
    ClientSampler,
    apply_client_size_imbalance,
    apply_covariate_shift,
    apply_regression_label_skew,
    build_covariate_shift_parameters,
    temporal_availability_report,
)


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for client_index, client in enumerate(["A", "B", "C", "D"]):
        for entity in range(20):
            for month in range(3):
                rows.append(
                    {
                        "client_id": client,
                        "permno": client_index * 100 + entity,
                        "month": month,
                        "x1": rng.normal(),
                        "x2": rng.normal(),
                        "target": rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_client_size_imbalance_keeps_complete_entity_histories() -> None:
    frame = _panel()
    result, report = apply_client_size_imbalance(
        frame,
        entity_col="permno",
        imbalance_ratio=10.0,
        seed=11,
    )
    original_months = frame.groupby("permno")["month"].nunique()
    retained_months = result.groupby("permno")["month"].nunique()
    assert (retained_months == original_months.loc[retained_months.index]).all()
    assert result["client_id"].nunique() == frame["client_id"].nunique()
    assert report["realized_fraction"].max() > report["realized_fraction"].min()


def test_regression_label_skew_is_reproducible_and_budget_matched() -> None:
    frame = _panel()
    left, left_report = apply_regression_label_skew(
        frame,
        target_col="target",
        concentration=0.2,
        retain_fraction=0.5,
        seed=13,
    )
    right, right_report = apply_regression_label_skew(
        frame,
        target_col="target",
        concentration=0.2,
        retain_fraction=0.5,
        seed=13,
    )
    pd.testing.assert_frame_equal(left, right)
    pd.testing.assert_frame_equal(left_report, right_report)
    counts = left.groupby("client_id").size()
    assert counts.nunique() == 1
    assert len(left) == 120


def test_covariate_shift_parameters_apply_consistently() -> None:
    frame = _panel()
    locations, scales = build_covariate_shift_parameters(
        ["A", "B", "C", "D"],
        ["x1", "x2"],
        location_std=0.2,
        log_scale_std=0.1,
        seed=17,
    )
    shifted = apply_covariate_shift(frame, ["x1", "x2"], locations, scales)
    row = frame.loc[frame["client_id"] == "A"].iloc[0]
    shifted_row = shifted.loc[shifted["client_id"] == "A"].iloc[0]
    np.testing.assert_allclose(
        shifted_row[["x1", "x2"]].to_numpy(dtype=float),
        row[["x1", "x2"]].to_numpy(dtype=float) * scales["A"] + locations["A"],
    )


def test_persistent_temporal_schedule_is_reproducible_and_heterogeneous() -> None:
    sampler = ClientSampler(
        ["A", "B", "C", "D", "E", "F"],
        participation_rate=1.0,
        seed=19,
        mode="persistent",
        total_rounds=30,
        min_availability=0.1,
        max_availability=1.0,
    )
    schedule = sampler.schedule(30)
    assert schedule == sampler.schedule(30)
    report, summary = temporal_availability_report(schedule, sampler.client_ids)
    assert report["availability_rate"].nunique() > 1
    assert summary["temporal_availability_cv"] > 0
    assert all(len(active) >= 1 for active in schedule.values())


def test_block_temporal_schedule_has_contiguous_outages() -> None:
    sampler = ClientSampler(
        ["A", "B", "C", "D"],
        participation_rate=1.0,
        seed=23,
        mode="block",
        total_rounds=20,
        block_fraction=0.25,
    )
    schedule = sampler.schedule(20)
    report, _ = temporal_availability_report(schedule, sampler.client_ids)
    assert (report["active_rounds"] == 15).all()

import numpy as np
import pandas as pd

from fapbench.partition import (
    ClientSampler,
    assign_domain_clients,
    assign_iid_clients,
    build_feature_masks,
    build_feature_masks_matched_control,
    feature_jaccard_dissimilarity,
)


def test_iid_keeps_entity_history_together():
    frame = pd.DataFrame({"permno": np.repeat(np.arange(12), 3), "value": 1.0})
    assigned = assign_iid_clients(frame, "permno", num_clients=3, seed=1)
    assert assigned.groupby("permno")["client_id"].nunique().max() == 1
    assert assigned["client_id"].nunique() == 3


def test_iid_accepts_read_only_entity_array(monkeypatch):
    """Parquet/Arrow-backed columns may expose read-only NumPy views."""
    frame = pd.DataFrame({"permno": np.repeat(np.arange(12), 2), "value": 1.0})
    original_to_numpy = pd.Series.to_numpy

    def readonly_to_numpy(series, *args, **kwargs):
        values = original_to_numpy(series, *args, **kwargs)
        values = np.asarray(values).copy()
        values.setflags(write=False)
        return values

    monkeypatch.setattr(pd.Series, "to_numpy", readonly_to_numpy)
    assigned = assign_iid_clients(frame, "permno", num_clients=3, seed=7)

    assert assigned.groupby("permno")["client_id"].nunique().max() == 1
    assert assigned["client_id"].nunique() == 3


def test_iid_assignment_is_deterministic_without_mutating_input():
    frame = pd.DataFrame({"permno": np.repeat(np.arange(12), 2), "value": 1.0})
    original = frame.copy(deep=True)

    first = assign_iid_clients(frame, "permno", num_clients=3, seed=9)
    second = assign_iid_clients(frame, "permno", num_clients=3, seed=9)

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_series_equal(first["client_id"], second["client_id"])


def test_domain_and_masks():
    frame = pd.DataFrame({"sector": [10, 10, 20, 20], "x": [1, 2, 3, 4]})
    assigned = assign_domain_clients(frame, "sector")
    clients = assigned["client_id"].unique().tolist()
    masks = build_feature_masks(["a", "b", "c", "d"], clients, 0.5, 0.5, seed=2)
    assert all(mask.shape == (4,) for mask in masks.values())
    assert 0 <= feature_jaccard_dissimilarity(masks) <= 1


def test_sampler_reproducible():
    sampler = ClientSampler(["a", "b", "c", "d"], 0.5, seed=3)
    assert sampler.sample(2) == sampler.sample(2)
    assert len(sampler.sample(2)) == 2


def test_feature_masks_matched_control_is_identical_across_clients():
    # Real 7-characteristic panel and severity configs from
    # configs/icaif_heterogeneity.yaml. Severe (common_fraction=0.15,
    # private_fraction=0.35) is a regression test: for this config only 3 of
    # 7 features are covered by a single shared draw, so a naive
    # "draw the private subset once" fix that still ran build_feature_masks's
    # coverage-completion pass would silently hand 4 extra, unevenly
    # distributed features back to specific clients -- exactly the asymmetry
    # this matched control is supposed to remove.
    features = [f"f{i}" for i in range(7)]
    clients = [f"client_{i:02d}" for i in range(12)]
    for common_fraction, private_fraction in [(0.70, 0.70), (0.50, 0.50), (0.15, 0.35)]:
        treatment = build_feature_masks(
            features, clients, common_fraction, private_fraction, seed=11
        )
        matched = build_feature_masks_matched_control(
            features, clients, common_fraction, private_fraction, seed=11
        )
        treatment_counts = {int(mask.sum()) for mask in treatment.values()}
        matched_counts = {int(mask.sum()) for mask in matched.values()}
        assert matched_counts == treatment_counts
        assert feature_jaccard_dissimilarity(matched) == 0.0
        first = next(iter(matched.values()))
        assert all(np.array_equal(mask, first) for mask in matched.values())


def test_domain_can_keep_entity_persistent():
    frame = pd.DataFrame({
        "permno": [1, 1, 1, 2, 2],
        "sector": [10, 10, 20, 20, 20],
    })
    assigned = assign_domain_clients(frame, "sector", entity_col="permno")
    assert assigned.groupby("permno")["client_id"].nunique().max() == 1

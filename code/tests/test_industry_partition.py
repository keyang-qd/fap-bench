import pandas as pd
import pytest

from fapbench.partition import add_ff12_column, assign_domain_clients, map_sic_to_ff12


def test_ff12_mapping_known_industries():
    sic = pd.Series([2080, 3714, 3571, 6020, 9999])
    assert map_sic_to_ff12(sic).tolist() == [1, 2, 6, 11, 12]


def test_domain_partition_rejects_test_only_client():
    frame = pd.DataFrame({
        "permno": [1, 2, 3],
        "split": ["train", "train", "test"],
        "sector": ["A", "B", "C"],
    })
    with pytest.raises(ValueError, match="absent from training"):
        assign_domain_clients(
            frame,
            domain_col="sector",
            persistent_entity=False,
        )


def test_add_ff12_column_uses_sic():
    frame = pd.DataFrame({"sic": [2834, 4911]})
    mapped = add_ff12_column(frame)
    assert mapped["ff12"].tolist() == ["Hlth", "Utils"]

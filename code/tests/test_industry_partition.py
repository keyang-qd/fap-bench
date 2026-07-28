import pandas as pd
import pytest

from fapbench.partition import (
    add_ff12_column,
    add_sic_major_group_column,
    assign_domain_clients,
    map_sic_to_ff12,
    sic_major_group,
)


def test_ff12_mapping_known_industries():
    sic = pd.Series([2080, 3714, 3571, 6020, 9999])
    assert map_sic_to_ff12(sic).tolist() == [1, 2, 6, 11, 12]


def test_sic_major_group_is_leading_two_digits():
    sic = pd.Series([2834, 6020, 100, 9999, None])
    result = sic_major_group(sic)
    assert result.iloc[:4].astype(int).tolist() == [28, 60, 1, 99]
    assert pd.isna(result.iloc[4])


def test_sic_major_group_has_higher_cardinality_than_ff12():
    sic = pd.Series([100, 2834, 3714, 3571, 4911, 6020, 7372, 9999])
    assert sic_major_group(sic).nunique() > map_sic_to_ff12(sic).nunique()


def test_add_sic_major_group_column_uses_sic():
    frame = pd.DataFrame({"sic": [2834, 4911]})
    mapped = add_sic_major_group_column(frame)
    assert mapped["sic_major_group"].tolist() == ["28", "49"]


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

import pandas as pd
import pytest

from fapbench.data import PanelSpec, prepare_panel, validate_crsp_compustat_profitability_panel


def test_exact_calendar_target_alignment_does_not_jump_over_missing_month():
    frame = pd.DataFrame({
        "permno": [1, 1, 1, 2, 2, 2],
        "date": pd.to_datetime([
            "2020-01-31", "2020-03-31", "2020-04-30",
            "2020-01-31", "2020-02-29", "2020-03-31",
        ]),
        "retadj_1mn": [0.01, 0.03, 0.04, 0.10, 0.20, 0.30],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    panel, _ = prepare_panel(
        frame,
        PanelSpec(feature_columns=("x",), target_shift_months=1),
    )
    one = panel.loc[panel["permno"] == 1]
    # January has no exact February observation and must be dropped rather than
    # using March as a fake one-month-ahead return.
    assert pd.Timestamp("2020-01-31") not in set(one["date"])
    march = one.loc[one["date"] == pd.Timestamp("2020-03-31")].iloc[0]
    assert march["retadj_1mn"] == pytest.approx(0.04)
    assert march["target_date"] == pd.Timestamp("2020-04-30")


def test_recipe_validator_recognizes_contemporaneous_target():
    frame = pd.DataFrame({
        "permno": [1, 1],
        "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
        "year_month": [pd.Period("2020-01", "M"), pd.Period("2020-02", "M")],
        "retadj": [0.1, 0.2],
        "retadj_1mn": [0.1, 0.2],
        "shrcd": [10, 11],
        "exchcd": [1, 3],
    })
    report = validate_crsp_compustat_profitability_panel(frame)
    assert report["retadj_1mn_equals_same_month_retadj_share"] == pytest.approx(1.0)
    assert report["recommended_target_shift_months"] == 1


def test_duplicate_entity_month_rejected():
    frame = pd.DataFrame({
        "permno": [1, 1],
        "date": pd.to_datetime(["2020-01-15", "2020-01-31"]),
        "retadj_1mn": [0.1, 0.2],
        "x": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="one row per entity-month"):
        prepare_panel(frame, PanelSpec(feature_columns=("x",)))

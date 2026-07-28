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


def test_pre_aligned_target_is_preserved_when_shift_is_zero():
    frame = pd.DataFrame({
        "permno": [1, 1],
        "formation_date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
        "year_month": [pd.Period("2020-01", "M"), pd.Period("2020-02", "M")],
        "target_date": pd.to_datetime(["2020-02-29", "2020-03-31"]),
        "target_year_month": [pd.Period("2020-02", "M"), pd.Period("2020-03", "M")],
        "retadj": [0.01, 0.02],
        "retadj_1mn": [0.02, 0.03],
        "x": [1.0, 2.0],
    })
    panel, _ = prepare_panel(
        frame,
        PanelSpec(
            date_col="formation_date",
            feature_columns=("x",),
            target_shift_months=0,
        ),
    )
    assert panel["target_date"].tolist() == list(frame["target_date"])
    report = validate_crsp_compustat_profitability_panel(frame)
    assert report["target_alignment_status"] == "pre_aligned_exact_t_plus_1"
    assert report["recommended_target_shift_months"] == 0


def test_universe_filter_supports_price_and_nyse_microcap_breakpoints():
    from fapbench.data import apply_universe_filters

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31"] * 5 + ["2020-02-29"] * 5
            ),
            "permno": list(range(10)),
            "prc": [2, 6, 7, 8, 9, 3, 6, 7, 8, 9],
            "me": [1, 2, 3, 4, 100, 1, 2, 3, 4, 100],
            "exchcd": [1, 1, 1, 2, 3] * 2,
        }
    )
    filtered, report = apply_universe_filters(
        frame,
        "date",
        enabled=True,
        min_abs_price=5.0,
        exclude_bottom_market_cap_fraction=0.5,
        breakpoint_exchange_col="exchcd",
        breakpoint_exchange_values=[1],
    )
    assert filtered["prc"].abs().min() >= 5.0
    assert filtered.groupby("date")["me"].min().min() >= 2.5
    assert report["n_rows_dropped"] > 0


def test_split_by_time_test_end_truncates_the_panel():
    """test_end must drop later rows, not reassign them.

    Rolling refits rely on this: each window tests exactly one step past its
    validation block, so windows concatenate into one continuous out-of-sample
    series with no month counted twice.
    """
    import pandas as pd
    import pytest

    from fapbench.data import split_by_time

    frame = pd.DataFrame(
        {"target_date": pd.date_range("2015-01-31", periods=72, freq="ME"), "x": 1.0}
    )
    full = split_by_time(
        frame, "target_date", train_end="2016-12-31", val_end="2017-12-31"
    )
    capped = split_by_time(
        frame,
        "target_date",
        train_end="2016-12-31",
        val_end="2017-12-31",
        test_end="2018-12-31",
    )
    assert full["split"].eq("test").sum() == 36        # 2018-01 .. 2020-12
    assert capped["split"].eq("test").sum() == 12      # 2018 only
    assert capped["target_date"].max() == pd.Timestamp("2018-12-31")
    # train/val blocks are untouched by the cap
    for name in ("train", "val"):
        assert full["split"].eq(name).sum() == capped["split"].eq(name).sum()
    with pytest.raises(ValueError, match="later than val_end"):
        split_by_time(
            frame,
            "target_date",
            train_end="2016-12-31",
            val_end="2017-12-31",
            test_end="2017-06-30",
        )

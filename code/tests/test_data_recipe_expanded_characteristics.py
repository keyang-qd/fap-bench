from __future__ import annotations

"""Regression coverage for scripts/data_recipe2_revised.py.

This script previously had no automated tests. Because
include_expanded_characteristics=False must reproduce the original
7-characteristic panel byte-for-byte (every published benchmark result was
built with the default path), these tests check both that the default path
is unaffected by the new optional code, and that the 13 new expanded
characteristics (see EXPANDED_CHARACTERISTIC_COMPONENTS) are computed
correctly when the flag is enabled.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.data_recipe2_revised import EXPANDED_CHARACTERISTIC_COMPONENTS, data_recipe2


COMPUSTAT_COMMON = {
    "indfmt": "INDL",
    "datafmt": "STD",
    "popsrc": "D",
    "consol": "C",
}


def _compustat_row(*, gvkey, datadate, fyear, sic, fields):
    return {**COMPUSTAT_COMMON, "gvkey": gvkey, "datadate": datadate, "fyear": fyear, "sic": sic, **fields}


# Hand-picked (not proportionally scaled) so that every profitability ratio
# differs meaningfully between the two companies in every shared calendar
# month -- proportional scaling previously produced accidental cross-sectional
# ties (identical ratios -> tied ranks -> NaN rank-zscore for that
# characteristic), which is a test-fixture pitfall, not a pipeline bug.
# dvt (total dividends) and xrd (R&D expense) are only read on the
# include_expanded_characteristics=True path. gvkey 1002 leaves xrd as NaN
# in both years specifically to exercise rdoa's "missing R&D means zero R&D"
# fillna(0) convention, which is different from every other ratio's
# genuine-missingness industry/global-median imputation.
GVKEY_1001_FY1962 = dict(
    seq=100.0, ceq=np.nan, pstk=0.0, at=100.0, lt=40.0, txditc=0.0, txdb=np.nan,
    itcb=np.nan, pstkrv=np.nan, pstkl=np.nan, act=50.0, lct=20.0, che=5.0, dlc=2.0,
    txp=1.0, revt=200.0, cogs=120.0, xint=5.0, xsga=20.0, oiadp=40.0, oibdp=45.0,
    txpd=4.0, ib=8.0, dp=5.0, capx=3.0, dltt=25.0, sale=180.0, dvt=2.0, xrd=4.0,
)
GVKEY_1001_FY1963 = dict(
    seq=110.0, ceq=np.nan, pstk=0.0, at=110.0, lt=44.0, txditc=0.0, txdb=np.nan,
    itcb=np.nan, pstkrv=np.nan, pstkl=np.nan, act=55.0, lct=22.0, che=5.5, dlc=2.2,
    txp=1.1, revt=220.0, cogs=132.0, xint=5.5, xsga=22.0, oiadp=44.0, oibdp=49.5,
    txpd=4.4, ib=8.8, dp=5.5, capx=3.3, dltt=27.5, sale=198.0, dvt=2.2, xrd=4.4,
)
GVKEY_1002_FY1962 = dict(
    seq=300.0, ceq=np.nan, pstk=0.0, at=400.0, lt=150.0, txditc=0.0, txdb=np.nan,
    itcb=np.nan, pstkrv=np.nan, pstkl=np.nan, act=120.0, lct=60.0, che=15.0, dlc=8.0,
    txp=3.0, revt=500.0, cogs=400.0, xint=15.0, xsga=50.0, oiadp=50.0, oibdp=60.0,
    txpd=5.0, ib=15.0, dp=12.0, capx=10.0, dltt=80.0, sale=450.0, dvt=4.0, xrd=np.nan,
)
GVKEY_1002_FY1963 = dict(
    seq=280.0, ceq=np.nan, pstk=0.0, at=380.0, lt=140.0, txditc=0.0, txdb=np.nan,
    itcb=np.nan, pstkrv=np.nan, pstkl=np.nan, act=110.0, lct=55.0, che=14.0, dlc=7.0,
    txp=2.5, revt=470.0, cogs=380.0, xint=14.0, xsga=48.0, oiadp=42.0, oibdp=52.0,
    txpd=4.0, ib=10.0, dp=11.0, capx=8.0, dltt=76.0, sale=420.0, dvt=3.6, xrd=np.nan,
)


def _write_synthetic_inputs(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)

    compustat = pd.DataFrame(
        [
            _compustat_row(gvkey=1001, datadate="1962-12-31", fyear=1962, sic=2834, fields=GVKEY_1001_FY1962),
            _compustat_row(gvkey=1001, datadate="1963-12-31", fyear=1963, sic=2834, fields=GVKEY_1001_FY1963),
            _compustat_row(gvkey=1002, datadate="1962-12-31", fyear=1962, sic=6020, fields=GVKEY_1002_FY1962),
            _compustat_row(gvkey=1002, datadate="1963-12-31", fyear=1963, sic=6020, fields=GVKEY_1002_FY1963),
        ]
    )
    compustat.to_csv(input_dir / "compustat.csv", index=False)

    ccm = pd.DataFrame(
        [
            {
                "gvkey": 1001, "permno": 10001, "permco": 10001,
                "linkdt": "1960-01-01", "linkenddt": "", "linktype": "LC", "linkprim": "P",
            },
            {
                "gvkey": 1002, "permno": 10002, "permco": 10002,
                "linkdt": "1960-01-01", "linkenddt": "", "linktype": "LC", "linkprim": "P",
            },
        ]
    )
    ccm.to_csv(input_dir / "ccm.csv", index=False)

    dates = pd.date_range("1963-01-31", "1965-05-31", freq="ME")
    crsp_rows = []
    for date in dates:
        crsp_rows.append(
            {
                "permco": 10001, "permno": 10001, "date": date.strftime("%Y-%m-%d"),
                "shrcd": 11, "exchcd": 1, "shrout": 1000.0, "prc": 50.0,
                "cfacpr": 1.0, "cfacshr": 1.0, "ret": 0.01, "retx": 0.01,
            }
        )
        crsp_rows.append(
            {
                "permco": 10002, "permno": 10002, "date": date.strftime("%Y-%m-%d"),
                "shrcd": 11, "exchcd": 1, "shrout": 2000.0, "prc": 30.0,
                "cfacpr": 1.0, "cfacshr": 1.0, "ret": -0.005, "retx": -0.005,
            }
        )
    pd.DataFrame(crsp_rows).to_csv(input_dir / "crsp.csv", index=False)

    pd.DataFrame(columns=["permno", "dlstdt", "dlret"]).to_csv(
        input_dir / "delisting.csv", index=False
    )


@pytest.fixture()
def synthetic_input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "qdata"
    _write_synthetic_inputs(input_dir)
    return input_dir


def _build(input_dir: Path, **kwargs) -> pd.DataFrame:
    return data_recipe2(
        input_path=input_dir,
        formation_start="1964-06",
        formation_end="1965-05",
        accounting_start="1963-01",
        accounting_end="2024-12",
        log_level="ERROR",
        **kwargs,
    )


def test_default_panel_has_no_expanded_columns(synthetic_input_dir: Path) -> None:
    panel = _build(synthetic_input_dir)
    assert not panel.empty
    for feature in EXPANDED_CHARACTERISTIC_COMPONENTS:
        assert feature not in panel.columns
        assert f"{feature}_zscore" not in panel.columns


def test_expanded_panel_adds_bm_agr_momentum_without_changing_original_columns(
    synthetic_input_dir: Path,
) -> None:
    baseline = _build(synthetic_input_dir, include_expanded_characteristics=False)
    expanded = _build(synthetic_input_dir, include_expanded_characteristics=True)

    assert not expanded.empty
    for feature in EXPANDED_CHARACTERISTIC_COMPONENTS:
        assert feature in expanded.columns
        assert f"{feature}_zscore" in expanded.columns

    # Additivity: every column already present in the default panel must be
    # byte-identical in the expanded panel (same rows survive because agr and
    # mom_2_12 are both non-missing for every retained row in this fixture).
    shared_columns = [c for c in baseline.columns if c in expanded.columns]
    assert shared_columns == list(baseline.columns)
    sort_keys = ["permno", "target_year_month"]
    common = (
        expanded[shared_columns]
        .sort_values(sort_keys)
        .reset_index(drop=True)
    )
    reference = (
        baseline[shared_columns]
        .sort_values(sort_keys)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(common, reference)


def test_agr_matches_asset_growth_direction(synthetic_input_dir: Path) -> None:
    expanded = _build(synthetic_input_dir, include_expanded_characteristics=True)
    by_permno = expanded.set_index("permno")
    # gvkey 1001 (permno 10001): at grows 100 -> 110, agr should be +0.10.
    assert by_permno.loc[10001, "agr"].iloc[0] == pytest.approx(0.10, abs=1e-9)
    # gvkey 1002 (permno 10002): at shrinks 200 -> 190, agr should be -0.05.
    assert by_permno.loc[10002, "agr"].iloc[0] == pytest.approx(-0.05, abs=1e-9)


def test_momentum_matches_constant_monthly_return_compounding(
    synthetic_input_dir: Path,
) -> None:
    expanded = _build(synthetic_input_dir, include_expanded_characteristics=True)
    by_permno = expanded.set_index("permno")
    expected_10001 = (1.01**11) - 1.0
    expected_10002 = (0.995**11) - 1.0
    assert by_permno.loc[10001, "mom_2_12"].iloc[0] == pytest.approx(
        expected_10001, abs=1e-9
    )
    assert by_permno.loc[10002, "mom_2_12"].iloc[0] == pytest.approx(
        expected_10002, abs=1e-9
    )


def test_bm_uses_book_equity_over_market_equity(synthetic_input_dir: Path) -> None:
    expanded = _build(synthetic_input_dir, include_expanded_characteristics=True)
    by_permno = expanded.set_index("permno")
    # be = seq + txditc - pstk = 110 + 0 - 0 = 110; me = |prc|*shrout/1000 = 50.
    assert by_permno.loc[10001, "bm"].iloc[0] == pytest.approx(110.0 / 50.0, rel=1e-6)
    # be = seq = 280; me = |30|*2000/1000 = 60.
    assert by_permno.loc[10002, "bm"].iloc[0] == pytest.approx(280.0 / 60.0, rel=1e-6)


def test_price_scaled_value_characteristics(synthetic_input_dir: Path) -> None:
    expanded = _build(synthetic_input_dir, include_expanded_characteristics=True)
    by_permno = expanded.set_index("permno")
    # me = 50 (permno 10001), me = 60 (permno 10002), from FY1963 fundamentals.
    assert by_permno.loc[10001, "ep"].iloc[0] == pytest.approx(8.8 / 50.0, rel=1e-6)
    assert by_permno.loc[10002, "ep"].iloc[0] == pytest.approx(10.0 / 60.0, rel=1e-6)

    assert by_permno.loc[10001, "cfp"].iloc[0] == pytest.approx((8.8 + 5.5) / 50.0, rel=1e-6)
    assert by_permno.loc[10002, "cfp"].iloc[0] == pytest.approx((10.0 + 11.0) / 60.0, rel=1e-6)

    assert by_permno.loc[10001, "sp"].iloc[0] == pytest.approx(198.0 / 50.0, rel=1e-6)
    assert by_permno.loc[10002, "sp"].iloc[0] == pytest.approx(420.0 / 60.0, rel=1e-6)

    assert by_permno.loc[10001, "dy"].iloc[0] == pytest.approx(2.2 / 50.0, rel=1e-6)
    assert by_permno.loc[10002, "dy"].iloc[0] == pytest.approx(3.6 / 60.0, rel=1e-6)


def test_asset_scaled_investment_and_financing_characteristics(
    synthetic_input_dir: Path,
) -> None:
    expanded = _build(synthetic_input_dir, include_expanded_characteristics=True)
    by_permno = expanded.set_index("permno")
    # capx_at = capx / at, FY1963: 3.3/110 and 8.0/380.
    assert by_permno.loc[10001, "capx_at"].iloc[0] == pytest.approx(3.3 / 110.0, rel=1e-6)
    assert by_permno.loc[10002, "capx_at"].iloc[0] == pytest.approx(8.0 / 380.0, rel=1e-6)

    # lev = lt / at, FY1963: 44/110 and 140/380.
    assert by_permno.loc[10001, "lev"].iloc[0] == pytest.approx(44.0 / 110.0, rel=1e-6)
    assert by_permno.loc[10002, "lev"].iloc[0] == pytest.approx(140.0 / 380.0, rel=1e-6)

    # acc = -(wc_chg - dp); wc(1962)=28, wc(1963)=30.8 -> wc_chg=2.8, dp=5.5
    # -> acc=2.7, acc_at=2.7/110 for permno 10001.
    assert by_permno.loc[10001, "acc_at"].iloc[0] == pytest.approx(2.7 / 110.0, rel=1e-6)
    # wc(1962)=56, wc(1963)=50.5 -> wc_chg=-5.5, dp=11 -> acc=16.5,
    # acc_at=16.5/380 for permno 10002.
    assert by_permno.loc[10002, "acc_at"].iloc[0] == pytest.approx(16.5 / 380.0, rel=1e-6)


def test_rdoa_treats_missing_rd_expense_as_zero_not_missing(
    synthetic_input_dir: Path,
) -> None:
    expanded = _build(synthetic_input_dir, include_expanded_characteristics=True)
    by_permno = expanded.set_index("permno")
    # permno 10001 reports xrd=4.4 in FY1963: rdoa = 4.4/110.
    assert by_permno.loc[10001, "rdoa"].iloc[0] == pytest.approx(4.4 / 110.0, rel=1e-6)
    # permno 10002 has xrd=NaN in both fiscal years, which the recipe fills
    # with 0 (no R&D reported) rather than industry/global-median imputing
    # it like a genuinely missing ratio -- so rdoa must be exactly 0, not NaN
    # and not some other company's R&D intensity.
    assert by_permno.loc[10002, "rdoa"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_logme_and_momentum_1_1(synthetic_input_dir: Path) -> None:
    expanded = _build(synthetic_input_dir, include_expanded_characteristics=True)
    by_permno = expanded.set_index("permno")
    assert by_permno.loc[10001, "logme"].iloc[0] == pytest.approx(np.log1p(50.0), rel=1e-6)
    assert by_permno.loc[10002, "logme"].iloc[0] == pytest.approx(np.log1p(60.0), rel=1e-6)

    # mom_1_1 (prior 1-month return / short-term reversal) is exactly the
    # formation month's own realized return, which is constant in this
    # fixture: 0.01 for permno 10001, -0.005 for permno 10002.
    assert by_permno.loc[10001, "mom_1_1"].iloc[0] == pytest.approx(0.01, abs=1e-9)
    assert by_permno.loc[10002, "mom_1_1"].iloc[0] == pytest.approx(-0.005, abs=1e-9)

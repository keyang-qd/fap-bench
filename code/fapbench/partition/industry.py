from __future__ import annotations

import pandas as pd


FF12_LABELS = {
    1: "NoDur",
    2: "Durbl",
    3: "Manuf",
    4: "Enrgy",
    5: "Chems",
    6: "BusEq",
    7: "Telcm",
    8: "Utils",
    9: "Shops",
    10: "Hlth",
    11: "Money",
    12: "Other",
}


def _between(sic: pd.Series, low: int, high: int) -> pd.Series:
    return sic.between(low, high, inclusive="both")


def map_sic_to_ff12(series: pd.Series) -> pd.Series:
    """Map four-digit SIC codes to the standard Fama--French 12 industries."""
    sic = pd.to_numeric(series, errors="coerce")
    result = pd.Series(12, index=series.index, dtype="Int64")
    result.loc[sic.isna()] = pd.NA

    nodur = (
        _between(sic, 100, 999)
        | _between(sic, 2000, 2399)
        | _between(sic, 2700, 2749)
        | _between(sic, 2770, 2799)
        | _between(sic, 3100, 3199)
        | _between(sic, 3940, 3989)
    )
    durbl = (
        _between(sic, 2500, 2519)
        | _between(sic, 2590, 2599)
        | _between(sic, 3630, 3659)
        | _between(sic, 3710, 3711)
        | (sic == 3714)
        | (sic == 3716)
        | _between(sic, 3750, 3751)
        | (sic == 3792)
        | _between(sic, 3900, 3939)
        | _between(sic, 3990, 3999)
    )
    manuf = (
        _between(sic, 2520, 2589)
        | _between(sic, 2600, 2699)
        | _between(sic, 2750, 2769)
        | _between(sic, 3000, 3099)
        | _between(sic, 3200, 3569)
        | _between(sic, 3580, 3629)
        | _between(sic, 3700, 3709)
        | _between(sic, 3712, 3713)
        | (sic == 3715)
        | _between(sic, 3717, 3749)
        | _between(sic, 3752, 3791)
        | _between(sic, 3793, 3799)
        | _between(sic, 3830, 3839)
        | _between(sic, 3860, 3899)
    )
    enrgy = _between(sic, 1200, 1399) | _between(sic, 2900, 2999)
    chems = _between(sic, 2800, 2829) | _between(sic, 2840, 2899)
    buseq = (
        _between(sic, 3570, 3579)
        | _between(sic, 3660, 3692)
        | _between(sic, 3694, 3699)
        | _between(sic, 3810, 3829)
        | _between(sic, 7370, 7379)
    )
    telcm = _between(sic, 4800, 4899)
    utils = _between(sic, 4900, 4949)
    shops = (
        _between(sic, 5000, 5999)
        | _between(sic, 7200, 7299)
        | _between(sic, 7600, 7699)
    )
    hlth = (
        _between(sic, 2830, 2839)
        | (sic == 3693)
        | _between(sic, 3840, 3859)
        | _between(sic, 8000, 8099)
    )
    money = _between(sic, 6000, 6999)

    for code, mask in [
        (1, nodur),
        (2, durbl),
        (3, manuf),
        (4, enrgy),
        (5, chems),
        (6, buseq),
        (7, telcm),
        (8, utils),
        (9, shops),
        (10, hlth),
        (11, money),
    ]:
        result.loc[mask] = code
    return result


def add_ff12_column(
    frame: pd.DataFrame,
    sic_col: str = "sic",
    output_col: str = "ff12",
    label_names: bool = True,
) -> pd.DataFrame:
    if sic_col not in frame.columns:
        raise KeyError(f"SIC column {sic_col!r} is absent.")
    data = frame.copy()
    codes = map_sic_to_ff12(data[sic_col])
    if label_names:
        data[output_col] = codes.map(FF12_LABELS).astype("string")
    else:
        data[output_col] = codes
    return data


def sic_major_group(series: pd.Series) -> pd.Series:
    """Two-digit SIC major group (e.g. 2834 -> 28).

    Unlike ``map_sic_to_ff12``, this is a deterministic numeric truncation,
    not a curated economic crosswalk: it does not require a hand-encoded
    table of SIC ranges. It exists to provide a higher-cardinality,
    unambiguous domain partition (dozens of clients rather than FF12's 12)
    for testing whether benchmark conclusions depend on client granularity,
    without the correctness risk of reproducing a large named-industry
    crosswalk (e.g. Fama-French 48) from memory rather than the canonical
    source file.
    """
    sic = pd.to_numeric(series, errors="coerce")
    valid = sic.notna() & (sic >= 100) & (sic < 10000)
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result.loc[valid] = (sic.loc[valid] // 100).astype("Int64")
    return result


def add_sic_major_group_column(
    frame: pd.DataFrame,
    sic_col: str = "sic",
    output_col: str = "sic_major_group",
) -> pd.DataFrame:
    if sic_col not in frame.columns:
        raise KeyError(f"SIC column {sic_col!r} is absent.")
    data = frame.copy()
    data[output_col] = sic_major_group(data[sic_col]).astype("string")
    return data

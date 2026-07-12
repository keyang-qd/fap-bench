from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


PROFITABILITY_COMPONENTS = [
    "gpoa",
    "roa",
    "roe",
    "roic",
    "opoa",
    "gmar",
    "fcfoa",
]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _exact_month_lag(
    df: pd.DataFrame,
    value_columns: Iterable[str],
    lag_months: int = 1,
    suffix: str = "_lag",
) -> pd.DataFrame:
    """Attach exact calendar-month lags rather than previous-observation lags."""
    value_columns = list(value_columns)
    lagged = df[["permno", "year_month", *value_columns]].copy()
    lagged["year_month"] = lagged["year_month"] + lag_months
    lagged = lagged.rename(columns={column: f"{column}{suffix}" for column in value_columns})
    return df.merge(lagged, on=["permno", "year_month"], how="left", validate="one_to_one")


def _cross_sectional_rank_zscore(series: pd.Series) -> pd.Series:
    """Replicate rank-then-z-score while handling tiny cross sections safely."""
    ranks = series.rank(method="average")
    valid = ranks.notna()
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    if valid.sum() < 2:
        return result
    values = ranks.loc[valid].to_numpy(dtype="float64")
    scale = values.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        return result
    result.loc[valid] = stats.zscore(values, ddof=0)
    return result


def _merge_next_month_target(
    formation: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Map characteristics at month t to the realized return in calendar month t+1."""
    target = monthly_returns[["permno", "year_month", "retadj"]].copy()
    target = target.rename(
        columns={
            "year_month": "target_year_month",
            "retadj": "retadj_1mn",
        }
    )

    result = formation.copy()
    result["target_year_month"] = result["year_month"] + 1
    result = result.merge(
        target,
        on=["permno", "target_year_month"],
        how="left",
        validate="many_to_one",
    )
    result["target_date"] = result["target_year_month"].dt.to_timestamp(how="end").dt.normalize()
    return result


def data_recipe2(
    input_path: str | os.PathLike[str] = "/Users/keyang/qdata",
    output_path: str | os.PathLike[str] | None = None,
    formation_start: str = "1964-01",
    formation_end: str = "2023-12",
    accounting_start: str = "1963-01",
    accounting_end: str = "2024-12",
) -> pd.DataFrame:
    """Build a monthly CRSP--Compustat profitability panel.

    The returned data use month-t characteristics and an exactly aligned month-(t+1)
    delisting-adjusted return in ``retadj_1mn``. Market variables and returns are
    never forward-filled.
    """
    input_path = Path(input_path)

    comp = pd.read_csv(input_path / "compustat.csv", low_memory=False)
    ccm = pd.read_csv(input_path / "ccm.csv", low_memory=False)
    crsp_m = pd.read_csv(input_path / "crsp.csv", low_memory=False)
    dlret = pd.read_csv(input_path / "delisting.csv", low_memory=False)

    # ------------------------------------------------------------------
    # Compustat annual fundamentals
    # ------------------------------------------------------------------
    _require_columns(
        comp,
        [
            "gvkey", "datadate", "indfmt", "datafmt", "popsrc", "consol",
            "seq", "ceq", "pstk", "at", "lt", "txditc", "txdb", "itcb",
            "pstkrv", "pstkl", "act", "lct", "che", "dlc", "txp", "revt",
            "cogs", "xint", "xsga", "oiadp", "txpd", "ib", "dp", "capx",
            "dltt", "sale", "sic",
        ],
        "compustat.csv",
    )
    _require_columns(
        ccm,
        ["gvkey", "permno", "permco", "linkdt", "linkenddt", "linktype", "linkprim"],
        "ccm.csv",
    )

    comp = comp.loc[
        (comp["indfmt"] == "INDL")
        & (comp["datafmt"] == "STD")
        & (comp["popsrc"] == "D")
        & (comp["consol"] == "C")
    ].copy()

    comp["gvkey"] = pd.to_numeric(comp["gvkey"], errors="coerce").astype("Int64")
    ccm["gvkey"] = pd.to_numeric(ccm["gvkey"], errors="coerce").astype("Int64")
    ccm["permno"] = pd.to_numeric(ccm["permno"], errors="coerce").astype("Int64")
    ccm["permco"] = pd.to_numeric(ccm["permco"], errors="coerce").astype("Int64")

    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"], errors="coerce")
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], errors="coerce")

    comp = comp.dropna(subset=["gvkey", "datadate"])
    ccm = ccm.dropna(subset=["gvkey", "permno", "linkdt"])

    # Conservative CCM filters commonly used for CRSP--Compustat linking.
    ccm = ccm.loc[
        ccm["linktype"].isin(["LC", "LU"])
        & ccm["linkprim"].isin(["P", "C"])
    ].copy()

    # Annual statements are assumed public from the following June onward.
    comp["availability_month"] = (
        comp["datadate"].dt.to_period("Y").dt.end_time
        + pd.offsets.MonthEnd(6)
    ).dt.to_period("M")

    comp_m = comp.merge(ccm, on="gvkey", how="inner", validate="many_to_many")
    comp_m = comp_m.loc[
        (comp_m["datadate"] >= comp_m["linkdt"])
        & (
            comp_m["linkenddt"].isna()
            | (comp_m["datadate"] <= comp_m["linkenddt"])
        )
    ].copy()

    # Prefer primary links deterministically before removing duplicate permno-dates.
    comp_m["_linkprim_order"] = comp_m["linkprim"].map({"P": 0, "C": 1}).fillna(9)
    comp_m["_linktype_order"] = comp_m["linktype"].map({"LC": 0, "LU": 1}).fillna(9)
    comp_m = (
        comp_m.sort_values(
            ["permno", "datadate", "_linkprim_order", "_linktype_order"]
        )
        .drop_duplicates(["permno", "datadate"], keep="first")
        .drop(columns=["_linkprim_order", "_linktype_order"])
    )

    numeric_comp_columns = [
        "seq", "ceq", "pstk", "at", "lt", "txditc", "txdb", "itcb",
        "pstkrv", "pstkl", "act", "lct", "che", "dlc", "txp", "revt",
        "cogs", "xint", "xsga", "oiadp", "txpd", "ib", "dp", "capx",
        "dltt", "sale", "sic", "fyear", "oibdp",
    ]
    _coerce_numeric(comp_m, numeric_comp_columns)

    common_equity_base = pd.concat(
        [comp_m["seq"], comp_m["ceq"], comp_m["pstk"], comp_m["at"] - comp_m["lt"]],
        axis=1,
    ).bfill(axis=1).iloc[:, 0]
    deferred_taxes = comp_m[["txditc", "txdb", "itcb"]].bfill(axis=1).iloc[:, 0].fillna(0)
    preferred_stock = comp_m[["pstkrv", "pstkl", "pstk"]].bfill(axis=1).iloc[:, 0].fillna(0)
    comp_m["be"] = common_equity_base + deferred_taxes - preferred_stock

    working_capital_inputs = ["act", "lct", "che", "dlc", "txp"]
    comp_m["wc"] = (
        comp_m["act"].fillna(0)
        - comp_m["lct"].fillna(0)
        - comp_m["che"].fillna(0)
        + comp_m["dlc"].fillna(0)
        + comp_m["txp"].fillna(0)
    )
    comp_m.loc[comp_m[working_capital_inputs].isna().all(axis=1), "wc"] = np.nan

    comp_m = comp_m.sort_values(["gvkey", "datadate"])
    comp_m["wc_last"] = comp_m.groupby("gvkey", observed=True)["wc"].shift(1)
    comp_m["wc_chg"] = comp_m["wc"] - comp_m["wc_last"]

    comp_m["op"] = (
        comp_m["revt"]
        - comp_m["cogs"].fillna(0)
        - comp_m["xint"].fillna(0)
        - comp_m["xsga"].fillna(0)
    )
    comp_m.loc[comp_m[["revt", "cogs", "xint", "xsga"]].isna().all(axis=1), "op"] = np.nan
    comp_m["gp"] = comp_m["revt"] - comp_m["cogs"].fillna(0)
    comp_m["nop"] = comp_m["oiadp"] - comp_m["txpd"].fillna(0)
    comp_m["fcf"] = (
        comp_m["ib"]
        + comp_m["dp"].fillna(0)
        - comp_m["wc_chg"].fillna(0)
        - comp_m["capx"].fillna(0)
    )
    comp_m["acc"] = -(comp_m["wc_chg"] - comp_m["dp"])
    comp_m["ic"] = comp_m["be"] + comp_m["dltt"]

    comp_m["roa"] = comp_m["ib"] / comp_m["at"]
    comp_m["gpoa"] = comp_m["gp"] / comp_m["at"]
    comp_m["opoa"] = comp_m["op"] / comp_m["at"]
    comp_m["roe"] = comp_m["ib"] / comp_m["be"]
    comp_m["fcfoa"] = comp_m["fcf"] / comp_m["at"]
    comp_m["roic"] = comp_m["oiadp"] / comp_m["ic"]
    comp_m["nopoa"] = comp_m["nop"] / comp_m["at"]
    comp_m["gmar"] = comp_m["gp"] / comp_m["sale"]

    comp_m["year_month"] = comp_m["availability_month"]
    comp_m["sic"] = pd.to_numeric(comp_m["sic"], errors="coerce").astype("Int64")
    comp_m["sic2"] = (comp_m["sic"] // 100).astype("Int64")

    compustat_columns = [
        "gvkey", "permno", "permco", "datadate", "year_month", "fyear",
        "indfmt", "consol", "popsrc", "datafmt", "sic", "sic2", "pstkrv",
        "che", "dlc", "dltt", "oibdp", "oiadp", "dp", "ib", "revt",
        "cogs", "xint", "xsga", "txpd", "capx", "be", "at", "sale",
        "wc", "wc_chg", "acc", "ic", "op", "gp", "nop", "fcf", "roa",
        "gpoa", "roe", "fcfoa", "roic", "nopoa", "opoa", "gmar",
    ]
    comp_m_final = comp_m[compustat_columns].replace([np.inf, -np.inf], np.nan)
    comp_m_final = (
        comp_m_final.sort_values(["permno", "year_month", "datadate"])
        .drop_duplicates(["permno", "year_month"], keep="last")
    )
    comp_m_final = comp_m_final.loc[
        comp_m_final["year_month"].between(accounting_start, accounting_end)
    ].copy()

    # Impute annual profitability ratios only; identifiers and dates are untouched.
    comp_m_final[PROFITABILITY_COMPONENTS] = (
        comp_m_final.groupby(["year_month", "sic2"], observed=True)[PROFITABILITY_COMPONENTS]
        .transform(lambda values: values.fillna(values.median()))
    )
    comp_m_final[PROFITABILITY_COMPONENTS] = (
        comp_m_final.groupby("year_month", observed=True)[PROFITABILITY_COMPONENTS]
        .transform(lambda values: values.fillna(values.median()))
    )

    # ------------------------------------------------------------------
    # CRSP monthly returns and market data
    # ------------------------------------------------------------------
    _require_columns(
        crsp_m,
        [
            "permco", "permno", "date", "shrcd", "exchcd", "shrout", "prc",
            "cfacpr", "cfacshr", "ret", "retx",
        ],
        "crsp.csv",
    )
    _require_columns(dlret, ["permno", "dlstdt", "dlret"], "delisting.csv")

    _coerce_numeric(
        crsp_m,
        ["permco", "permno", "shrcd", "exchcd", "shrout", "prc", "cfacpr", "cfacshr", "ret", "retx"],
    )
    _coerce_numeric(dlret, ["permno", "dlret"])

    crsp_m["date"] = pd.to_datetime(crsp_m["date"], errors="coerce")
    crsp_m["year_month"] = crsp_m["date"].dt.to_period("M")
    dlret["dlstdt"] = pd.to_datetime(dlret["dlstdt"], errors="coerce")
    dlret["year_month"] = dlret["dlstdt"].dt.to_period("M")

    crsp_m = crsp_m.dropna(subset=["permno", "year_month"]).copy()
    dlret = dlret.dropna(subset=["permno", "year_month"]).copy()

    # Collapse accidental duplicate delisting records within a permno-month.
    dlret = (
        dlret.sort_values(["permno", "year_month", "dlstdt"])
        .drop_duplicates(["permno", "year_month"], keep="last")
    )

    crsp = crsp_m.merge(
        dlret[["permno", "year_month", "dlret", "dlstdt"]],
        on=["permno", "year_month"],
        how="left",
        validate="many_to_one",
    )

    # Combine ordinary and delisting returns when both are observed.
    both_returns = crsp["ret"].notna() & crsp["dlret"].notna()
    crsp["retadj"] = crsp["ret"].combine_first(crsp["dlret"])
    crsp.loc[both_returns, "retadj"] = (
        (1.0 + crsp.loc[both_returns, "ret"])
        * (1.0 + crsp.loc[both_returns, "dlret"])
        - 1.0
    )

    # Keep only ordinary common shares on NYSE/AMEX/Nasdaq. No market variable is ffilled.
    crsp = crsp.loc[
        crsp["shrcd"].isin([10, 11])
        & crsp["exchcd"].isin([1, 2, 3])
    ].copy()
    crsp = (
        crsp.sort_values(["permno", "year_month", "date"])
        .drop_duplicates(["permno", "year_month"], keep="last")
    )

    # Preserve security-level returns for exact t+1 target construction even if
    # the representative share class for a company changes in the next month.
    monthly_target_returns = crsp[["permno", "year_month", "retadj"]].copy()

    crsp["me_security"] = crsp["prc"].abs() * crsp["shrout"]
    crsp["me_security"] = crsp["me_security"] / 1000.0

    # Assign company-level market equity to the largest security for each permco-month.
    company_me = (
        crsp.groupby(["year_month", "permco"], observed=True)["me_security"]
        .sum(min_count=1)
        .rename("me")
        .reset_index()
    )
    representative_rows = (
        crsp.sort_values(
            ["year_month", "permco", "me_security", "permno"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["year_month", "permco"], keep="first")
    )
    crsp2 = representative_rows.merge(
        company_me,
        on=["year_month", "permco"],
        how="left",
        validate="one_to_one",
    )

    crsp2 = crsp2.sort_values(["permno", "year_month"])
    crsp2 = crsp2.drop_duplicates(["permno", "year_month"], keep="last")

    # Exact calendar-month lag; a two-month gap is not treated as a one-month lag.
    crsp2 = _exact_month_lag(crsp2, ["me"], lag_months=1, suffix="_lag")
    crsp2["port_weight"] = crsp2["me_lag"]
    fallback_weight = crsp2["me"] / (1.0 + crsp2["retx"])
    crsp2["port_weight"] = crsp2["port_weight"].combine_first(fallback_weight)
    crsp2.loc[crsp2["retadj"].isna() & crsp2["prc"].isna(), "port_weight"] = np.nan

    crsp_columns = [
        "permno", "permco", "date", "year_month", "shrcd", "shrout", "prc",
        "cfacpr", "cfacshr", "exchcd", "ret", "retx", "dlret", "retadj",
        "me", "me_lag", "port_weight",
    ]
    crsp2 = crsp2[crsp_columns].replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Monthly panel: forward-fill fundamentals only
    # ------------------------------------------------------------------
    df = crsp2.merge(
        comp_m_final,
        on=["year_month", "permno"],
        how="left",
        suffixes=("", "_comp"),
        validate="one_to_one",
    )
    df = df.sort_values(["permno", "year_month"]).copy()

    fundamental_columns = [
        column
        for column in compustat_columns
        if column not in {"permno", "permco", "year_month"}
    ]
    df[fundamental_columns] = (
        df.groupby("permno", observed=True)[fundamental_columns].ffill()
    )

    # Prefer CRSP permco; use linked Compustat permco only when CRSP is unavailable.
    if "permco_comp" in df.columns:
        df["permco"] = df["permco"].fillna(df["permco_comp"])
        df = df.drop(columns=["permco_comp"])

    df["sic"] = pd.to_numeric(df["sic"], errors="coerce").astype("Int64")
    df["sic2"] = (df["sic"] // 100).astype("Int64")
    df["year"] = df["year_month"].dt.year.astype("Int64")

    raw_profitability_and_scale = [
        *PROFITABILITY_COMPONENTS,
        "ib", "at", "gp", "be", "fcf", "oiadp", "ic", "sale", "op",
    ]
    df[raw_profitability_and_scale] = (
        df.groupby(["year_month", "sic2"], observed=True)[raw_profitability_and_scale]
        .transform(lambda values: values.fillna(values.median()))
    )
    df[raw_profitability_and_scale] = (
        df.groupby("year_month", observed=True)[raw_profitability_and_scale]
        .transform(lambda values: values.fillna(values.median()))
    )

    # Construct monthly cross-sectional rank z-scores after monthly imputation.
    for feature in PROFITABILITY_COMPONENTS:
        df[f"{feature}_rank"] = df.groupby("year_month", observed=True)[feature].rank(method="average")
        df[f"{feature}_zscore"] = (
            df.groupby("year_month", observed=True)[feature]
            .transform(_cross_sectional_rank_zscore)
        )

    zscore_columns = [f"{feature}_zscore" for feature in PROFITABILITY_COMPONENTS]
    df["profitability_simple"] = df[zscore_columns].mean(axis=1, skipna=False)

    df["shares"] = df["shrout"] * df["cfacshr"]
    df = _exact_month_lag(df, ["shares"], lag_months=1, suffix="_last")
    df["shares_chg"] = df["shares"] / df["shares_last"]

    # Formation sample uses month-t information; target is realized in month t+1.
    df = df.loc[df["year_month"].between(formation_start, formation_end)].copy()
    df = _merge_next_month_target(df, monthly_target_returns)

    # Keep formation return separately and do not overwrite/fill it.
    df["formation_date"] = df["year_month"].dt.to_timestamp(how="end").dt.normalize()

    final_columns = [
        "permno", "permco", "gvkey", "datadate", "formation_date", "year_month",
        "target_date", "target_year_month", "fyear", "year", "sic", "sic2",
        "indfmt", "consol", "popsrc", "datafmt", "pstkrv", "che", "dlc",
        "dltt", "oibdp", "oiadp", "dp", "ib", "revt", "cogs", "xint",
        "xsga", "txpd", "capx", "be", "at", "sale", "wc", "wc_chg",
        "acc", "ic", "op", "gp", "nop", "fcf", "roa", "gpoa", "roe",
        "fcfoa", "roic", "nopoa", "gmar", "opoa", "date", "shrcd",
        "shrout", "prc", "cfacpr", "cfacshr", "exchcd", "ret", "retx",
        "dlret", "retadj", "me", "me_lag", "port_weight", "retadj_1mn",
        "shares", "shares_last", "shares_chg", "gpoa_zscore", "roa_zscore",
        "roe_zscore", "roic_zscore", "opoa_zscore", "gmar_zscore",
        "fcfoa_zscore", "profitability_simple",
    ]

    # Some source extracts may omit nonessential accounting fields.
    final_columns = [column for column in final_columns if column in df.columns]
    df = df[final_columns].replace([np.inf, -np.inf], np.nan)

    required_model_columns = [*zscore_columns, "profitability_simple", "retadj_1mn"]
    df = df.dropna(subset=required_model_columns)
    df = df.sort_values(["target_year_month", "permno"]).reset_index(drop=True)

    duplicate_count = int(df.duplicated(["permno", "year_month"]).sum())
    if duplicate_count:
        raise ValueError(f"Final panel contains {duplicate_count} duplicate permno-month rows")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".parquet":
            df.to_parquet(output_path, index=False)
        elif output_path.suffix.lower() == ".csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError("output_path must end in .parquet or .csv")

    return df


if __name__ == "__main__":
    panel = data_recipe2(
        input_path="/Users/keyang/qdata",
        output_path="/Users/keyang/qdata/profitability_panel_revised.parquet",
    )
    print(panel.shape)
    print(panel[["permno", "year_month", "target_year_month", "retadj_1mn"]].head())

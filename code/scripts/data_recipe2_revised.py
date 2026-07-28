from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

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

# Optional value/investment/leverage/intangible/momentum/size characteristics,
# added on top of the profitability set only when
# data_recipe2(..., include_expanded_characteristics=True) is requested
# (default False, so the original 7-characteristic panel and its existing
# tests/outputs are unaffected). Together with PROFITABILITY_COMPONENTS this
# brings the panel to 20 total characteristics, closing the gap between the
# original 7-characteristic panel and the ~20-25 characteristic target set in
# EXPERIMENT_ENHANCEMENT_PLAN.md Tier 2 needed to justify "high-dimensional"
# language in the paper's motivation:
#   value:      bm (book-to-market), ep (earnings yield), cfp (cash-flow
#               yield), sp (sales-to-price), dy (dividend yield)
#   investment: agr (asset growth), capx_at (capital investment intensity)
#   financing:  lev (book leverage, total liabilities / assets)
#   accruals:   acc_at (Sloan-1996-style accruals-to-assets)
#   intangible: rdoa (R&D intensity, R&D expense / assets)
#   size:       logme (log market equity)
#   momentum:   mom_2_12 (12-2 skip-month momentum), mom_1_1 (prior 1-month
#               return / short-term reversal)
EXPANDED_CHARACTERISTIC_COMPONENTS = [
    "bm", "ep", "cfp", "sp", "dy",
    "agr", "capx_at",
    "lev",
    "acc_at",
    "rdoa",
    "logme",
    "mom_2_12", "mom_1_1",
]

# Skip-month momentum window: cumulative return over months [t-12, t-2],
# excluding the most recent month t-1 (the standard Jegadeesh-Titman/UMD
# construction). Expressed as (start_lag, end_lag) exact calendar-month lags
# relative to the formation month.
MOMENTUM_START_LAG = 2
MOMENTUM_END_LAG = 12


LOGGER_NAME = "data_recipe"


def _configure_logger(
    log_level: str | int = "INFO",
    log_file: str | os.PathLike[str] | None = None,
) -> logging.Logger:
    """Create a terminal logger and, optionally, a persistent file logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rebuild handlers on each top-level run so repeated notebook calls do not
    # duplicate messages or keep writing to a stale log file.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Persistent log file: %s", log_path)

    return logger


def _frame_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(index=True, deep=True).sum() / (1024 ** 2))


def _log_frame(logger: logging.Logger, name: str, df: pd.DataFrame) -> None:
    logger.info(
        "%s | rows=%s | cols=%s | memory=%.1f MB",
        name,
        f"{len(df):,}",
        f"{df.shape[1]:,}",
        _frame_memory_mb(df),
    )


@contextmanager
def _log_stage(logger: logging.Logger, name: str) -> Iterator[None]:
    """Log stage start, completion time, and failures without changing logic."""
    started = time.perf_counter()
    logger.info("START | %s", name)
    try:
        yield
    except Exception:
        logger.exception("FAILED | %s | elapsed=%.2fs", name, time.perf_counter() - started)
        raise
    else:
        logger.info("DONE  | %s | elapsed=%.2fs", name, time.perf_counter() - started)


def _read_csv_with_log(path: Path, logger: logging.Logger) -> pd.DataFrame:
    size_mb = path.stat().st_size / (1024 ** 2) if path.exists() else float("nan")
    started = time.perf_counter()
    logger.info("Reading %s | file_size=%.1f MB", path.name, size_mb)
    frame = pd.read_csv(path, low_memory=False)
    logger.info(
        "Loaded %s | rows=%s | cols=%s | memory=%.1f MB | elapsed=%.2fs",
        path.name,
        f"{len(frame):,}",
        f"{frame.shape[1]:,}",
        _frame_memory_mb(frame),
        time.perf_counter() - started,
    )
    return frame


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


def _trailing_contiguous_return(
    df: pd.DataFrame,
    start_lag: int,
    end_lag: int,
    output_col: str,
) -> pd.DataFrame:
    """Cumulative return over months [t-end_lag, t-start_lag], requiring every
    intervening month to be present at an exact one-month calendar spacing.

    Reuses ``_exact_month_lag`` once per lag so each lagged return column goes
    through the same permno/year_month exact-match merge already used
    elsewhere in this recipe (e.g. for ``me_lag``), rather than a new rolling-
    window implementation. A missing intervening month yields NaN rather than
    a stale value, consistent with this recipe's convention of never
    forward-filling returns.
    """
    if start_lag < 1 or end_lag < start_lag:
        raise ValueError("Momentum window requires 1 <= start_lag <= end_lag.")
    result = df
    lag_columns = []
    for lag in range(start_lag, end_lag + 1):
        suffix = f"_lag{lag}"
        result = _exact_month_lag(result, ["retadj"], lag_months=lag, suffix=suffix)
        lag_columns.append(f"retadj{suffix}")
    complete = result[lag_columns].notna().all(axis=1)
    cumulative_return = (1.0 + result[lag_columns].fillna(0.0)).prod(axis=1) - 1.0
    result[output_col] = cumulative_return.where(complete)
    return result.drop(columns=lag_columns)


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
    log_level: str | int = "INFO",
    log_file: str | os.PathLike[str] | None = None,
    include_expanded_characteristics: bool = False,
) -> pd.DataFrame:
    """Build a monthly CRSP--Compustat profitability panel.

    The returned data use month-t characteristics and an exactly aligned month-(t+1)
    delisting-adjusted return in ``retadj_1mn``. Market variables and returns are
    never forward-filled.

    ``include_expanded_characteristics`` defaults to False and reproduces the
    original 7-characteristic profitability panel exactly. When True, it adds
    13 more characteristics across value (``bm``, ``ep``, ``cfp``, ``sp``,
    ``dy``), investment (``agr``, ``capx_at``), financing (``lev``),
    accruals (``acc_at``), intangible investment (``rdoa``), size
    (``logme``), and momentum (``mom_2_12``, ``mom_1_1``) -- see
    EXPANDED_CHARACTERISTIC_COMPONENTS -- each with its own cross-sectional
    rank z-score, bringing the panel to 20 characteristics total for the
    robustness check described in EXPERIMENT_ENHANCEMENT_PLAN.md.
    """
    total_started = time.perf_counter()
    logger = _configure_logger(log_level=log_level, log_file=log_file)
    input_path = Path(input_path)

    logger.info(
        "Building CRSP--Compustat panel | input=%s | output=%s | "
        "formation=%s..%s | accounting=%s..%s",
        input_path,
        output_path,
        formation_start,
        formation_end,
        accounting_start,
        accounting_end,
    )

    with _log_stage(logger, "Load source CSV files"):
        comp = _read_csv_with_log(input_path / "compustat.csv", logger)
        ccm = _read_csv_with_log(input_path / "ccm.csv", logger)
        crsp_m = _read_csv_with_log(input_path / "crsp.csv", logger)
        dlret = _read_csv_with_log(input_path / "delisting.csv", logger)

    # ------------------------------------------------------------------
    # Compustat annual fundamentals
    # ------------------------------------------------------------------
    required_compustat_columns = [
        "gvkey", "datadate", "indfmt", "datafmt", "popsrc", "consol",
        "seq", "ceq", "pstk", "at", "lt", "txditc", "txdb", "itcb",
        "pstkrv", "pstkl", "act", "lct", "che", "dlc", "txp", "revt",
        "cogs", "xint", "xsga", "oiadp", "txpd", "ib", "dp", "capx",
        "dltt", "sale", "sic",
    ]
    if include_expanded_characteristics:
        # dvt (total dividends, for dy) and xrd (R&D expense, for rdoa) are
        # only needed on the expanded-characteristics path -- required
        # unconditionally would break the default 7-characteristic path
        # against any Compustat extract/fixture that omits them.
        required_compustat_columns = [*required_compustat_columns, "dvt", "xrd"]
    _require_columns(comp, required_compustat_columns, "compustat.csv")
    _require_columns(
        ccm,
        ["gvkey", "permno", "permco", "linkdt", "linkenddt", "linktype", "linkprim"],
        "ccm.csv",
    )

    comp_before_filter = len(comp)
    comp = comp.loc[
        (comp["indfmt"] == "INDL")
        & (comp["datafmt"] == "STD")
        & (comp["popsrc"] == "D")
        & (comp["consol"] == "C")
    ].copy()
    logger.info(
        "Compustat format filter | kept=%s/%s (%.1f%%)",
        f"{len(comp):,}",
        f"{comp_before_filter:,}",
        100.0 * len(comp) / max(comp_before_filter, 1),
    )

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

    with _log_stage(logger, "Merge Compustat with CCM links"):
        comp_m = comp.merge(ccm, on="gvkey", how="inner", validate="many_to_many")
    _log_frame(logger, "Compustat--CCM raw merge", comp_m)
    linked_before = len(comp_m)
    comp_m = comp_m.loc[
        (comp_m["datadate"] >= comp_m["linkdt"])
        & (
            comp_m["linkenddt"].isna()
            | (comp_m["datadate"] <= comp_m["linkenddt"])
        )
    ].copy()
    logger.info(
        "CCM validity-window filter | kept=%s/%s",
        f"{len(comp_m):,}",
        f"{linked_before:,}",
    )

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
    if include_expanded_characteristics:
        numeric_comp_columns = [*numeric_comp_columns, "dvt", "xrd"]
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

    # Prior-fiscal-year total assets, computed the same way as wc_last above:
    # a strict same-entity shift over datadate-sorted annual filings, so no
    # look-ahead is introduced. Only used when include_expanded_characteristics
    # requests the "agr" (asset growth) characteristic below.
    comp_m["at_last"] = comp_m.groupby("gvkey", observed=True)["at"].shift(1)
    comp_m["agr"] = (comp_m["at"] - comp_m["at_last"]) / comp_m["at_last"]

    if include_expanded_characteristics:
        # Additional annual-level ratios, each an existing Compustat
        # fundamental scaled by total assets, following exactly the same
        # pattern as agr above (computed here, industry/global-median
        # imputed alongside the other annual ratios below, then
        # forward-filled monthly like every other fundamental).
        comp_m["capx_at"] = comp_m["capx"] / comp_m["at"]
        comp_m["lev"] = comp_m["lt"] / comp_m["at"]
        comp_m["acc_at"] = comp_m["acc"] / comp_m["at"]
        # A missing R&D expense in Compustat conventionally means the firm
        # reported no R&D that year, not that the value is unknown -- fill
        # with 0 before scaling, unlike the ratios above (and dvt below),
        # which are genuinely-missing values imputed via industry/global
        # medians instead.
        comp_m["rdoa"] = comp_m["xrd"].fillna(0.0) / comp_m["at"]

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
    if include_expanded_characteristics:
        compustat_columns = [
            *compustat_columns,
            "agr", "dvt", "capx_at", "lev", "acc_at", "rdoa",
        ]
    comp_m_final = comp_m[compustat_columns].replace([np.inf, -np.inf], np.nan)
    comp_m_final = (
        comp_m_final.sort_values(["permno", "year_month", "datadate"])
        .drop_duplicates(["permno", "year_month"], keep="last")
    )
    comp_m_final = comp_m_final.loc[
        comp_m_final["year_month"].between(accounting_start, accounting_end)
    ].copy()

    # Impute annual profitability ratios (and, when requested, agr) only;
    # identifiers and dates are untouched.
    annual_impute_columns = list(PROFITABILITY_COMPONENTS)
    if include_expanded_characteristics:
        annual_impute_columns.extend(
            ["agr", "dvt", "capx_at", "lev", "acc_at", "rdoa"]
        )
    with _log_stage(logger, "Impute annual profitability ratios"):
        missing_before = comp_m_final[annual_impute_columns].isna().mean().mean()
        comp_m_final[annual_impute_columns] = (
            comp_m_final.groupby(["year_month", "sic2"], observed=True)[annual_impute_columns]
            .transform(lambda values: values.fillna(values.median()))
        )
        comp_m_final[annual_impute_columns] = (
            comp_m_final.groupby("year_month", observed=True)[annual_impute_columns]
            .transform(lambda values: values.fillna(values.median()))
        )
        missing_after = comp_m_final[annual_impute_columns].isna().mean().mean()
    logger.info(
        "Annual profitability missing rate | before=%.2f%% | after=%.2f%%",
        100.0 * missing_before,
        100.0 * missing_after,
    )
    _log_frame(logger, "Annual Compustat panel", comp_m_final)

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

    with _log_stage(logger, "Merge CRSP with delisting returns"):
        crsp = crsp_m.merge(
            dlret[["permno", "year_month", "dlret", "dlstdt"]],
            on=["permno", "year_month"],
            how="left",
            validate="many_to_one",
        )
    logger.info(
        "Delisting coverage | rows_with_dlret=%s | both_ret_and_dlret=%s",
        f"{crsp['dlret'].notna().sum():,}",
        f"{(crsp['ret'].notna() & crsp['dlret'].notna()).sum():,}",
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
    crsp_before_filter = len(crsp)
    crsp = crsp.loc[
        crsp["shrcd"].isin([10, 11])
        & crsp["exchcd"].isin([1, 2, 3])
    ].copy()
    logger.info(
        "CRSP common-share/exchange filter | kept=%s/%s (%.1f%%)",
        f"{len(crsp):,}",
        f"{crsp_before_filter:,}",
        100.0 * len(crsp) / max(crsp_before_filter, 1),
    )
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
    with _log_stage(logger, "Aggregate company market equity and choose representative securities"):
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
    _log_frame(logger, "Representative-security CRSP panel", crsp2)

    crsp2 = crsp2.sort_values(["permno", "year_month"])
    crsp2 = crsp2.drop_duplicates(["permno", "year_month"], keep="last")

    # Exact calendar-month lag; a two-month gap is not treated as a one-month lag.
    with _log_stage(logger, "Attach exact one-month market-equity lag"):
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
    with _log_stage(logger, "Merge monthly CRSP panel with Compustat fundamentals"):
        df = crsp2.merge(
            comp_m_final,
            on=["year_month", "permno"],
            how="left",
            suffixes=("", "_comp"),
            validate="one_to_one",
        )
        df = df.sort_values(["permno", "year_month"]).copy()
    _log_frame(logger, "CRSP--Compustat monthly merge", df)

    fundamental_columns = [
        column
        for column in compustat_columns
        if column not in {"permno", "permco", "year_month"}
    ]
    with _log_stage(logger, "Forward-fill fundamentals within PERMNO"):
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
    if include_expanded_characteristics:
        raw_profitability_and_scale = [
            *raw_profitability_and_scale,
            "agr", "dvt", "dp", "capx_at", "lev", "acc_at", "rdoa",
        ]
    with _log_stage(logger, "Impute monthly profitability and scale variables"):
        monthly_missing_before = df[raw_profitability_and_scale].isna().mean().mean()
        df[raw_profitability_and_scale] = (
            df.groupby(["year_month", "sic2"], observed=True)[raw_profitability_and_scale]
            .transform(lambda values: values.fillna(values.median()))
        )
        df[raw_profitability_and_scale] = (
            df.groupby("year_month", observed=True)[raw_profitability_and_scale]
            .transform(lambda values: values.fillna(values.median()))
        )
        monthly_missing_after = df[raw_profitability_and_scale].isna().mean().mean()
    logger.info(
        "Monthly model-input missing rate | before=%.2f%% | after=%.2f%%",
        100.0 * monthly_missing_before,
        100.0 * monthly_missing_after,
    )

    if include_expanded_characteristics:
        # bm/ep/cfp/sp/dy all divide an already industry/global-median-imputed
        # annual fundamental (be/ib/dp/sale/dvt) by me, which is a raw CRSP
        # market variable that is never imputed and always present for the
        # formation universe.
        df["bm"] = df["be"] / df["me"]
        df["ep"] = df["ib"] / df["me"]
        df["cfp"] = (df["ib"] + df["dp"]) / df["me"]
        df["sp"] = df["sale"] / df["me"]
        df["dy"] = df["dvt"] / df["me"]
        df["logme"] = np.log1p(df["me"])
        # Prior 1-month return (short-term reversal). df["retadj"] at
        # formation month t is already the realized month-t return merged
        # from CRSP -- no additional lag merge is needed, since it is
        # exactly the "prior 1-month return" used to predict month t+1.
        df["mom_1_1"] = df["retadj"]
        with _log_stage(logger, "Construct skip-month momentum (mom_2_12)"):
            df = _trailing_contiguous_return(
                df, MOMENTUM_START_LAG, MOMENTUM_END_LAG, output_col="mom_2_12"
            )
        logger.info(
            "Expanded characteristic coverage | bm_nonmissing=%s | "
            "dy_nonmissing=%s | rdoa_nonmissing=%s | mom_2_12_nonmissing=%s",
            f"{int(df['bm'].notna().sum()):,}",
            f"{int(df['dy'].notna().sum()):,}",
            f"{int(df['rdoa'].notna().sum()):,}",
            f"{int(df['mom_2_12'].notna().sum()):,}",
        )

    # Construct monthly cross-sectional rank z-scores after monthly imputation.
    characteristics_to_score = list(PROFITABILITY_COMPONENTS)
    if include_expanded_characteristics:
        characteristics_to_score += EXPANDED_CHARACTERISTIC_COMPONENTS
    with _log_stage(logger, "Construct monthly cross-sectional rank z-scores"):
        for index, feature in enumerate(characteristics_to_score, start=1):
            feature_started = time.perf_counter()
            logger.info(
                "Z-score progress | %s/%s | feature=%s",
                index,
                len(characteristics_to_score),
                feature,
            )
            df[f"{feature}_rank"] = df.groupby("year_month", observed=True)[feature].rank(method="average")
            df[f"{feature}_zscore"] = (
                df.groupby("year_month", observed=True)[feature]
                .transform(_cross_sectional_rank_zscore)
            )
            nonmissing_zscore = int(df[f"{feature}_zscore"].notna().sum())
            log_method = logger.info if nonmissing_zscore > 0 else logger.warning
            log_method(
                "Z-score complete | feature=%s | nonmissing=%s | elapsed=%.2fs",
                feature,
                f"{nonmissing_zscore:,}",
                time.perf_counter() - feature_started,
            )

    zscore_columns = [f"{feature}_zscore" for feature in PROFITABILITY_COMPONENTS]
    df["profitability_simple"] = df[zscore_columns].mean(axis=1, skipna=False)
    expanded_zscore_columns = [
        f"{feature}_zscore" for feature in EXPANDED_CHARACTERISTIC_COMPONENTS
    ]

    df["shares"] = df["shrout"] * df["cfacshr"]
    with _log_stage(logger, "Attach exact one-month share-count lag"):
        df = _exact_month_lag(df, ["shares"], lag_months=1, suffix="_last")
    df["shares_chg"] = df["shares"] / df["shares_last"]

    # Formation sample uses month-t information; target is realized in month t+1.
    before_formation_filter = len(df)
    df = df.loc[df["year_month"].between(formation_start, formation_end)].copy()
    logger.info(
        "Formation-period filter | kept=%s/%s",
        f"{len(df):,}",
        f"{before_formation_filter:,}",
    )
    with _log_stage(logger, "Attach exact next-month return targets"):
        df = _merge_next_month_target(df, monthly_target_returns)
    target_coverage = float(df["retadj_1mn"].notna().mean()) if len(df) else 0.0
    target_log_method = logger.info if target_coverage >= 0.5 else logger.warning
    target_log_method(
        "Next-month target coverage | nonmissing=%s/%s (%.1f%%)",
        f"{df['retadj_1mn'].notna().sum():,}",
        f"{len(df):,}",
        100.0 * target_coverage,
    )

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
    if include_expanded_characteristics:
        final_columns = [
            *final_columns,
            *EXPANDED_CHARACTERISTIC_COMPONENTS,
            *expanded_zscore_columns,
        ]

    # Some source extracts may omit nonessential accounting fields.
    final_columns = [column for column in final_columns if column in df.columns]
    with _log_stage(logger, "Select and clean final output columns"):
        df = df[final_columns].replace([np.inf, -np.inf], np.nan)

    required_model_columns = [*zscore_columns, "profitability_simple", "retadj_1mn"]
    if include_expanded_characteristics:
        required_model_columns = [*required_model_columns, *expanded_zscore_columns]
    before_required_drop = len(df)
    df = df.dropna(subset=required_model_columns)
    df = df.sort_values(["target_year_month", "permno"]).reset_index(drop=True)
    logger.info(
        "Required-model-field filter | kept=%s/%s (%.1f%%)",
        f"{len(df):,}",
        f"{before_required_drop:,}",
        100.0 * len(df) / max(before_required_drop, 1),
    )
    _log_frame(logger, "Final modeling panel", df)
    if df.empty:
        logger.warning(
            "Final modeling panel is empty. Check source coverage, formation dates, "
            "and cross-sectional feature variation."
        )

    duplicate_count = int(df.duplicated(["permno", "year_month"]).sum())
    if duplicate_count:
        raise ValueError(f"Final panel contains {duplicate_count} duplicate permno-month rows")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with _log_stage(logger, f"Write output file: {output_path.name}"):
            if output_path.suffix.lower() == ".parquet":
                df.to_parquet(output_path, index=False)
            elif output_path.suffix.lower() == ".csv":
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("output_path must end in .parquet or .csv")
        logger.info(
            "Output written | path=%s | file_size=%.1f MB",
            output_path,
            output_path.stat().st_size / (1024 ** 2),
        )

    logger.info(
        "COMPLETE | rows=%s | columns=%s | total_elapsed=%.2fs",
        f"{len(df):,}",
        f"{df.shape[1]:,}",
        time.perf_counter() - total_started,
    )
    return df


if __name__ == "__main__":
    panel = data_recipe2(
        input_path="/Users/keyang/qdata",
        output_path="/Users/keyang/qdata/data0712.parquet",
        log_file="/Users/keyang/qdata/data0712_build.log",
        log_level="INFO",
    )
    print(panel.shape)
    print(panel[["permno", "year_month", "target_year_month", "retadj_1mn"]].head())

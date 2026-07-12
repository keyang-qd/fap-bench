# Alignment with `data_recipe2`

The uploaded sample is the final monthly CRSP/Compustat profitability panel produced by `data_recipe2`. The benchmark now treats it according to the following semantics.

## Correct interpretation

- `date` / `year_month` is the **feature-formation month**.
- Annual Compustat variables are conservatively made available in June following the fiscal year-end and then carried forward monthly.
- The seven profitability inputs are ranked and z-scored cross-sectionally within each month.
- `profitability_simple` is the equal-weight mean of the seven monthly z-scores.
- The stock universe is intended to be CRSP common shares (`SHRCD` 10/11) on NYSE, AMEX, or NASDAQ (`EXCHCD` 1/2/3).
- `retadj_1mn` is not forward aligned in the recipe: it is assigned directly from contemporaneous `retadj`.

Therefore the benchmark uses an exact entity-month join from formation month `t` to realized target month `t+1`, stores both `date` and `target_date`, and splits train/validation/test on `target_date` by default.

## Required edits to the data-generation script

### 1. Retain industry information

The recipe uses `sic` for imputation but removes it from the final output. Add `sic` to the final selected columns and derive:

```python
sic_numeric = pd.to_numeric(df["sic"], errors="coerce")
df["sic2"] = (sic_numeric // 100).astype("Int64")
```

For the first ICAIF version, `sic2` can be used as the domain/client label. A Fama–French 12-industry mapping is preferable later because raw two-digit SIC creates many small clients.

### 2. Do not forward-fill returns, prices, dates, or market equity

These statements are unsafe:

```python
df = df.groupby("idx").ffill()
df.update(df.groupby("permno")[var].ffill())
```

`var` contains `ret`, `retadj`, `prc`, `date`, and other market variables. A missing month can consequently inherit the previous month's return. Forward-fill only accounting variables and stable identifiers, for example:

```python
fundamental_cols = [
    "datadate", "fyear", "sic", "pstkrv", "che", "dlc", "dltt",
    "oibdp", "oiadp", "dp", "ib", "revt", "cogs", "xint", "xsga",
    "txpd", "capx", "be", "at", "sale", "wc", "wc_chg", "acc",
    "ic", "op", "gp", "nop", "fcf", "roa", "gpoa", "roe",
    "fcfoa", "roic", "nopoa", "gmar", "opoa",
]
df[fundamental_cols] = df.groupby("permno")[fundamental_cols].ffill()
```

Do not fill `ret`, `retx`, `retadj`, `prc`, `me`, or `date` across missing CRSP months.

### 3. Combine ordinary and delisting returns

The current expression ignores `dlret` whenever `ret` is non-missing. Use the combined return when both exist:

```python
ret = pd.to_numeric(crsp["ret"], errors="coerce")
dlret_value = pd.to_numeric(crsp["dlret"], errors="coerce")
crsp["retadj"] = np.where(
    ret.notna() & dlret_value.notna(),
    (1.0 + ret) * (1.0 + dlret_value) - 1.0,
    ret.combine_first(dlret_value),
)
```

Any additional imputation for missing delisting returns should be declared as a separate empirical convention rather than silently embedded.

### 4. Keep the raw return name honest

Prefer exporting the contemporaneous value as `retadj` and let FAP-Bench construct the future target. If `retadj_1mn` is retained for compatibility, document that it is still same-month in the source file and set:

```yaml
data:
  target_col: retadj_1mn
  target_shift_months: 1
```

### 5. Avoid re-standardizing existing z-scores

The selected eight inputs are already monthly cross-sectional rank z-scores or their average. Use:

```yaml
data:
  scaling: none
```

## Recommended benchmark configuration

Use `configs/crsp_compustat_profitability.yaml`. The runner also writes `source_validation.json`, including target-alignment, universe, duplicate-key, date consistency, and industry-column diagnostics.

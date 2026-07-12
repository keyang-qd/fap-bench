# Adaptation notes for the supplied sample

## What was found

The supplied `sample(1).parquet` has:

- 10 rows and 67 columns;
- one security (`permno = 10001`);
- 10 monthly dates from June 1987 to March 1988;
- one exchange code and no SIC, NAICS, or sector column;
- `retadj_1mn` as the configured target;
- eight immediately usable profitability features:
  `gpoa_zscore`, `roa_zscore`, `roe_zscore`, `roic_zscore`,
  `opoa_zscore`, `gmar_zscore`, `fcfoa_zscore`, and
  `profitability_simple`.

The sample is a schema example, not a trainable federated panel. It cannot form multiple clients, cross-sectional IC, or long-short portfolios.

## Full-data requirements

For a meaningful first ICAIF experiment, the full file should contain:

1. at least several hundred securities and preferably 10+ years of monthly dates;
2. a stable security ID (`permno`);
3. a correctly forward-aligned return target;
4. an industry/domain field, preferably `sic2`, `naics2`, or a sector label;
5. enough securities in every test month to form long and short legs;
6. delisting-adjusted returns and the sample filters used in the manuscript.

## Target alignment

The code does not assume that the name `retadj_1mn` guarantees forward alignment. Verify how this column was constructed:

- If row t already contains the t+1 return to be predicted, keep `target_shift_months: 0`.
- If row t contains the contemporaneous return, use `target_shift_months: 1`.

This must be documented in the paper because an accidental contemporaneous target would create leakage.

## Domain choices supported now

- Sector or industry: `domain_col: sic2` or a supplied sector column.
- Exchange: `domain_col: exchcd`.
- Size silo: `domain_col: me` with `continuous_bins: 5`.
- IID reference: `partition.type: iid`.

For the main paper, sector should be the primary domain split. Exchange and size splits are robustness settings rather than substitutes for sector heterogeneity.

## Recommended first production run

Start with one configuration only:

- sector clients;
- no feature masking;
- 100% participation;
- centralized, local, FedAvg, FedProx, SCAFFOLD, and FedAdam;
- three seeds;
- a fixed train/validation/test time split.

Only after this works should feature and temporal stress scenarios be added.

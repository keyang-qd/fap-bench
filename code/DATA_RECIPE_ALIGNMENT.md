# Alignment between the revised data recipe and FAP-Bench

## Revised output semantics

`data_recipe2_revised.py` now performs the target construction itself:

```text
formation month t characteristics
        -> exact calendar month t+1 retadj
```

The exported fields are:

- `formation_date` and `year_month` for model information at `t`;
- `target_date` and `target_year_month` for return realization at `t+1`;
- `retadj_1mn` for the pre-aligned target.

Accordingly, FAP-Bench must use `target_shift_months: 0`. Existing target dates are preserved rather than overwritten. Train/validation/test boundaries, monthly IC, RankIC, portfolio returns, and turnover dates use `target_date`.

## Checks performed by the benchmark

The source validator reports:

- duplicate `permno`--formation-month rows;
- consistency between `formation_date` and `year_month`;
- exact `year_month + 1 == target_year_month` alignment;
- consistency between `target_date` and `target_year_month`;
- common-share and exchange filters;
- availability of SIC/SIC2 and profitability features;
- whether the file looks pre-aligned or contemporaneous.

If the revised file is configured with `target_shift_months: 1`, the runner raises an error instead of silently shifting the target twice.

## Industry clients

The main configuration uses four-digit `sic` and maps it to Fama--French 12 industries inside the benchmark. This is preferable to taking the largest eleven raw SIC2 categories and pooling the rest, which is not a standard economic taxonomy.

All partition choices are fitted on the training split. A validation/test-only domain category raises an error instead of creating a client that never participated in training.

## Feature inputs

Use the seven profitability component z-scores. Do not include `profitability_simple` in feature-heterogeneity experiments because it is the exact average of those components and can transmit information about masked inputs.

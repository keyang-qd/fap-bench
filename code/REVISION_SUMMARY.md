# FAP-Bench Revision Summary

This revision implements the priority fixes identified from the first FF12
profitability experiment.

## 1. Fairer training and validation

- Centralized and local baselines can match the FL expected full-dataset-equivalent
  data passes, accounting for partial participation.
- The validation split is now used for checkpoint selection.
- Validation histories and selected steps are saved for every method.
- Communication at the selected FL checkpoint is reported separately from the
  total communication spent under the fixed training budget.
- Selected model checkpoints are saved under `checkpoints/`.

## 2. Correct local-only economic evaluation

- Raw outputs from separately trained local models are no longer treated as
  directly comparable across clients.
- The pooled local portfolio uses within-month, within-client percentile ranks by
  default.
- The uncalibrated pooled local portfolio remains available only as a clearly
  labelled diagnostic.

## 3. Separate portfolio interpretations

Each method now reports:

- unconstrained pooled global portfolio;
- client-neutral portfolio formed within each client and equal-weighted;
- standalone client portfolio metrics and their mean, median, weighted mean,
  dispersion, best client, and worst client.

The domain benchmark uses the client-neutral portfolio as the primary result.

## 4. Stronger diagnostics

- Monthly IC, RankIC, MSE, MAE, and prediction dispersion are saved.
- IC standard deviation, ICIR, positive-IC-month ratio, RankICIR, and corresponding
  client summaries are included.
- Long/short client shares, HHI concentration, and maximum client share are saved.
- Return, turnover, and transaction-cost definitions are written into result
  metadata.

## 5. Robustness and reproducibility

- Optional minimum-price and monthly market-cap filters were added, including
  NYSE-reference breakpoints.
- Missing SIC observations can be dropped or assigned to a dedicated client.
- `run_metadata.json` records status, software versions, dimensions, elapsed time,
  and failure details.
- Grid runs can continue after individual failures and preserve completed results.
- A five-seed CRSP--Compustat grid includes full/partial participation, additional
  local steps, a $5 screen, and microcap exclusion.

## 6. Statistical analysis

`scripts/analyze_results.py` computes:

- paired Newey--West tests on monthly portfolio-return differences;
- moving-block-bootstrap confidence intervals for annualized Sharpe ratios.

## Validation performed

- 21 unit tests pass.
- A complete synthetic end-to-end run passed for centralized, local, FedAvg,
  FedProx, SCAFFOLD, and FedAdam.
- Validation checkpoint restoration, checkpoint files, all revised result tables,
  statistical analysis, and grid aggregation were exercised successfully.

The full CRSP--Compustat experiment was not rerun in this environment because the
configured financial dataset path is local to the user's machine.

# Project notes

## Old schema sample versus revised production panel

The 10-row `sample(1).parquet` was produced by the original recipe and contains one stock. Its `retadj_1mn` equals same-month `retadj`, so it is useful only for schema inspection and requires `target_shift_months: 1` if loaded.

The production panel should instead be regenerated with `scripts/data_recipe2_revised.py`. That file contains a pre-aligned `retadj_1mn`, explicit formation/target dates, SIC/SIC2, corrected delisting returns, and no forward-filled market returns. It requires `target_shift_months: 0`.

## Recommended first empirical run

1. Fama--French 12 industry clients;
2. seven profitability component z-scores;
3. no feature masking;
4. 100% client participation;
5. centralized, local, FedAvg, FedProx, SCAFFOLD, and FedAdam;
6. at least three seeds;
7. split and evaluate by `target_date`.

After the main domain comparison is stable, add feature overlap and partial-participation stress tests.

## Evaluation protocol revision (2026-07-13)

The benchmark now distinguishes three economic evaluations:

1. `global_portfolio_*`: an unconstrained pooled long-short portfolio;
2. `client_neutral_portfolio_*`: long-short portfolios formed within clients and
   equal-weighted across clients (the default primary result for domain partitions);
3. `client_portfolio_*`: the distribution of independent client-level portfolio
   metrics, including mean, median, weighted mean, dispersion, and worst client.

For `local`, pooled scores are percentile-ranked within month and client before a
pooled portfolio is formed. The uncalibrated pooled result is retained only as an
explicitly labelled diagnostic. Monthly IC/RankIC tables, portfolio client
exposures, concentration measures, and validation checkpoint histories are saved.
Baseline epochs can be matched to `federated.rounds × federated.local_epochs`, and
all methods can select checkpoints on the validation period without changing the
fixed maximum training budget.

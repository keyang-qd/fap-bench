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

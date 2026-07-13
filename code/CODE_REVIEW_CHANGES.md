# Code review after revising the data generator

The second audit identified and fixed the following inconsistencies:

1. `target_shift_months: 0` previously overwrote an existing `target_date` with the formation date.
2. The financial YAML placed `split_date_col` under `data`, while the runner read only `split.date_col`.
3. IC, RankIC, and portfolio returns were grouped by formation date rather than target realization date.
4. The source validator still assumed that every `retadj_1mn` was contemporaneous.
5. Domain categories and persistent entity labels could be fitted using validation/test observations.
6. Feature masks were built from all-period clients rather than training clients.
7. Local evaluation silently skipped clients lacking a training model.
8. Raw SIC2 top-category collapsing was used instead of a standard industry mapping.
9. `profitability_simple` duplicated the seven component inputs and undermined feature masking.
10. The centralized reference used full features while FL methods used masked features, without a matched centralized baseline.
11. SCAFFOLD used Adam client updates even though its implemented control-variate update assumes SGD.
12. Monthly IC code required a pandas API added after the stated minimum dependency.

All items above are corrected in this package.
13. Local-only models previously started from different random initializations across clients; they now share the same initial state.
14. SCAFFOLD communication previously counted only model upload/download; control-variate communication is now included.

## 2026-07-13: JSON date serialization and late-failure recovery

- `json_default()` now supports Python `date`, `datetime`, and `time`, NumPy date/bool/scalar types, Pandas time types, `Path`, enums, sets, and `torch.device`.
- JSON files are written atomically, so a serialization failure cannot leave a truncated destination file.
- `resolved_config.json` is validated and saved before loading the panel or training models; unsupported configuration values now fail immediately.
- Added `scripts/repair_completed_run.py` to recover a run that completed all models but failed while saving the resolved config. It does not retrain models.

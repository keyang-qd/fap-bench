# Grid runtime: quick versus full

`crsp_compustat_profitability_grid_quick.yaml` is only a pipeline smoke test. It is not a reduced version whose wall time can be extrapolated directly to the final grid without accounting for methods, rounds, local epochs, participation, baselines, scenarios, and seeds.

## Relative compute budgets

A **full-client-pass equivalent** is one local epoch over all training clients. It is a transparent relative budget, not an exact wall-clock prediction.

| Configuration | Scenarios | Seeds | FL methods | FL pass equivalents | Baseline pass equivalents | Total |
|---|---:|---:|---:|---:|---:|---:|
| Grid quick | 2 | 1 | 1 | 10 | 0 | 10 |
| Grid draft | 5 | 3 | 4 | 1,080 | 540 | 1,620 |
| Grid final | 5 | 5 | 4 | 4,500 | 2,250 | 6,750 |

The final grid therefore has roughly **675 times** the nominal training budget of the smoke test.

For the first final-grid run alone:

- Centralized baseline: 50 epochs.
- Local-only baseline: 50 epochs over every client.
- Four federated algorithms: 50 rounds each at full participation.

This is approximately 300 full-data/client-pass equivalents, compared with 5 in the first quick run.

## Commands

Smoke test:

```bash
caffeinate -i python -u scripts/run_benchmark.py \
  --config configs/crsp_compustat_profitability_grid_quick.yaml
```

Full-method draft grid:

```bash
caffeinate -i python -u scripts/run_benchmark.py \
  --config configs/crsp_compustat_profitability_grid_draft.yaml
```

Final grid:

```bash
caffeinate -i python -u scripts/run_benchmark.py \
  --config configs/crsp_compustat_profitability_grid.yaml
```

The draft grid runs all four FL algorithms and both non-federated references, but uses three seeds and 20 pass equivalents instead of five seeds and 50. It is suitable for checking tables and conclusions before committing to the final run.

## Progress visibility

The final grid now sets child console logging to true and prints every fifth round. Detailed logs remain available at:

```text
outputs/crsp_compustat_profitability_grid/<scenario>/seed_<seed>/run.log
```

The grid log reports:

- Scenario compute budgets before training.
- Total relative compute budget.
- Run elapsed time.
- A rough remaining-time estimate after each completed run.

## Pandas warnings

The repeated `DataFrame is highly fragmented` warnings came from inserting scenario, seed, and heterogeneity columns one at a time. Grid result collection now constructs metadata once and concatenates it in one operation. This removes the warnings. It was not the main 24-hour bottleneck, but avoiding fragmentation also makes aggregation cleaner and slightly faster.

# Grid execution and performance

## Correct entry point

Both commands now work:

```bash
caffeinate -i python -u scripts/run_grid.py \
  --grid configs/crsp_compustat_profitability_grid.yaml
```

or the unified form:

```bash
caffeinate -i python -u scripts/run_benchmark.py \
  --config configs/crsp_compustat_profitability_grid.yaml
```

`run_benchmark.py` detects `base_config` plus `scenarios` and dispatches to the grid runner.

## Parallelism model

The grid runs one scenario/seed at a time. Inside each run, the FF12 clients are trained in parallel using the `performance.parallel_clients` value inherited from `crsp_compustat_profitability.yaml`.

This is intentional. Running multiple whole experiments concurrently while each experiment also creates a client process pool would oversubscribe CPU cores and duplicate the large panel in memory. On a 16 GB Apple-silicon laptop, that often makes execution slower or causes swap pressure.

## Grid-level acceleration

The grid runner now:

1. Loads the source Parquet only once and reuses the raw DataFrame across all runs.
2. Reuses completed scenario/seed outputs after interruption when `resume_completed: true`.
3. Runs garbage collection after each experiment.
4. Writes child-run logs to files without duplicating every training line in the terminal.
5. Inherits cached client arrays, shared-memory client data, and reusable multiprocessing pools from the base config.

The grid log is:

```text
outputs/crsp_compustat_profitability_grid/grid.log
```

Each run has its own log:

```text
outputs/crsp_compustat_profitability_grid/<scenario>/seed_<seed>/run.log
```

## Important experiment-budget correction

The previous local-step scenario used:

```yaml
rounds: 50
local_epochs: 5
```

which performs 250 local epochs per selected client and therefore uses five times the optimization budget of the 50-by-1 reference.

It is now:

```yaml
rounds: 10
local_epochs: 5
```

Both settings use 50 total local epochs, so the comparison focuses on communication frequency rather than mixing communication frequency with five times more computation.

## Resume behavior

After interruption, rerun the same command. A run is reused when both files exist and are readable:

```text
benchmark_results.csv
heterogeneity.json
```

The grid log will contain:

```text
resume hit; training skipped
```

Delete a specific scenario/seed directory to force that run to execute again.

## Quick performance check

Before launching the 25-run final grid, run:

```bash
caffeinate -i python -u scripts/run_benchmark.py \
  --config configs/crsp_compustat_profitability_grid_quick.yaml
```

This runs one seed, FedAvg only, and two short scenarios. It verifies process spawning, shared-memory caching, data reuse, output writing, and approximate per-round timing. It is not intended for paper results.

## Interpreting total workload

The full configuration contains five scenarios and five seeds, or 25 complete experiments. Every experiment still trains the selected algorithms independently; no legitimate code optimization removes this statistical replication. Runtime should therefore be estimated from one completed scenario/seed after the parallel warm-up, then multiplied by the remaining number of non-completed runs.

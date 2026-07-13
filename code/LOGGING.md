# Runtime Logging

The benchmark writes progress to both the terminal and a persistent log file.
Every normal benchmark run creates:

```text
<output_dir>/run.log
```

A grid experiment additionally creates:

```text
<grid_output>/grid.log
<grid_output>/<scenario>/seed_<seed>/run.log
```

## Configuration

Add the following block to a YAML configuration:

```yaml
logging:
  level: INFO
  file: run.log
  console: true
  overwrite: false

  # Centralized baselines: print every N epochs. Use 0 to disable.
  epoch_interval: 1

  # Local-only baseline: print every N completed clients.
  local_client_interval: 1
  local_epoch_interval: 0

  # Prediction stage: print every N clients.
  prediction_client_interval: 1

  # Federated algorithms: print every N communication rounds.
  round_interval: 1

  # Per-client logging inside every FL round is very verbose.
  federated_client_details: false
  federated_client_interval: 1
  federated_local_epoch_interval: 0
```

For a 50-round production run, `round_interval: 1` gives one start and one
completion message per round. Set it to `5` to report rounds 1, 5, 10, ..., 50.

## Logged stages

The log includes:

1. Input file path and size.
2. Panel rows, columns, memory, date range, and entity count.
3. Target alignment validation.
4. Train/validation/test sample counts.
5. Preprocessing and client partition statistics.
6. Feature-mask availability by client.
7. Heterogeneity metrics.
8. Centralized epoch loss.
9. Local-only client progress.
10. Every federated round's active clients, mean loss, communication, and time.
11. Prediction progress by client.
12. Predictive and portfolio metric highlights.
13. Output saving and total elapsed time.
14. Full exception traceback if a stage fails.

## Watching a running experiment

```bash
tail -f outputs/crsp_compustat_profitability/run.log
```

To show the last 100 lines:

```bash
tail -n 100 outputs/crsp_compustat_profitability/run.log
```

## Command-line overrides

```bash
python -u scripts/run_benchmark.py \
  --config configs/crsp_compustat_profitability.yaml \
  --log-level INFO \
  --log-file benchmark.log
```

Grid run:

```bash
python -u scripts/run_grid.py \
  --grid configs/synthetic_grid.yaml \
  --log-level INFO \
  --log-file grid.log
```

Using `python -u` is recommended for immediate terminal output, although the
persistent Python logger already flushes each record.

## Data-generation logging

`scripts/data_recipe2_revised.py` also logs its long-running data construction
stages. Call it with `log_file` to preserve the log:

```python
panel = data_recipe2(
    input_path="/Users/keyang/qdata",
    output_path="/Users/keyang/qdata/data0712.parquet",
    log_file="/Users/keyang/qdata/data0712_build.log",
    log_level="INFO",
)
```

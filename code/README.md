# FAP-Bench

A compact benchmark suite for federated cross-sectional asset pricing. The code supports financially meaningful client partitions, controlled feature availability, partial client participation, centralized/local references, FedAvg, FedProx, SCAFFOLD, and FedAdam, together with predictive, portfolio, client-level, communication, and heterogeneity metrics.

## Data contract for the revised CRSP--Compustat recipe

The included `scripts/data_recipe2_revised.py` produces one row for a formation month `t` with:

- `formation_date` / `year_month`: feature-formation month;
- `target_date` / `target_year_month`: exact calendar month `t+1`;
- `retadj`: contemporaneous formation-month return;
- `retadj_1mn`: delisting-adjusted return realized in `t+1`;
- `sic` / `sic2`: industry identifiers;
- seven monthly cross-sectional profitability rank z-scores;
- `profitability_simple`: their deterministic average.

For this revised file, use:

```yaml
data:
  date_col: formation_date
  target_col: retadj_1mn
  target_shift_months: 0
  target_date_col: target_date

split:
  date_col: target_date

evaluation:
  date_col: target_date
```

Do **not** set `target_shift_months: 1` for data produced by the revised recipe, because the target is already aligned. The older 10-row sample used during schema inspection is different: its `retadj_1mn` is contemporaneous and therefore requires a one-month shift.

## Installation

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Generate the financial panel

Edit the input/output paths at the bottom of `scripts/data_recipe2_revised.py`, or call it directly:

```python
from scripts.data_recipe2_revised import data_recipe2

panel = data_recipe2(
    input_path="/Users/keyang/qdata",
    output_path="/Users/keyang/qdata/profitability_panel_revised.parquet",
)
```

The recipe:

- combines ordinary and delisting returns as `(1 + RET)(1 + DLRET) - 1`;
- never forward-fills returns, prices, dates, or market equity;
- carries forward Compustat fundamentals only;
- constructs exact calendar-month lags and targets;
- retains SIC and derives SIC2.

## Run the financial benchmark

Start from `configs/crsp_compustat_profitability.yaml`, replace `data.path`, and run:

```bash
python scripts/run_benchmark.py \
  --config configs/crsp_compustat_profitability.yaml
```

The production configuration maps four-digit SIC codes to the standard Fama--French 12 industries. Partition definitions, category collapsing, feature masks, and participating-client schedules use training clients only; validation/test periods cannot redefine the client population.

The seven component z-scores are used as model inputs. `profitability_simple` is intentionally omitted because retaining a deterministic composite while masking its components would weaken the feature-heterogeneity experiment.

## Feature heterogeneity

Set:

```yaml
feature_heterogeneity:
  enabled: true
  common_fraction: 0.5
  private_fraction: 0.5
```

Unavailable features are set to zero after monthly rank standardization. In this setting the output includes:

- `centralized_oracle`: centralized access to all seven features;
- `centralized_masked`: pooled training under the same client-specific feature masks;
- `local`, FedAvg, FedProx, SCAFFOLD, and FedAdam under those masks.

The matched centralized reference is the appropriate comparison for isolating the cost of federation; the oracle is only an information upper reference.

## Client optimization

The paper configuration uses SGD for all client updates:

```yaml
federated:
  client_optimizer: sgd
  client_lr: 0.01
  client_momentum: 0.0
```

SCAFFOLD is restricted to SGD in this implementation because applying its control-variate correction inside Adam is not the standard algorithm.

## Smoke tests

```bash
python scripts/make_synthetic.py \
  --output data/synthetic.parquet \
  --stocks 40 --months 24 --features 8

python scripts/run_benchmark.py \
  --config configs/synthetic_smoke.yaml

pytest -q
```

## Evaluation protocol

The revised protocol separates predictive performance from three portfolio views:

- `global_portfolio_*`: unconstrained pooled ranking across the full universe;
- `client_neutral_portfolio_*`: portfolios formed within each client and then equal-weighted;
- `client_portfolio_*`: the distribution of standalone client portfolio outcomes.

For domain heterogeneity, `client_neutral` is the default primary portfolio. The
unconstrained global portfolio is still saved together with long/short client shares,
HHI concentration, and maximum client exposure. Local-only models are calibrated by
monthly within-client percentile ranks before any pooled ranking because their raw
output locations and scales are not comparable. The uncalibrated local portfolio is
retained only as an explicitly labelled diagnostic.

The validation split can select the best checkpoint by `rank_ic`, `ic`, `mse`,
`mae`, or another available predictive metric while preserving a fixed maximum
training budget. With `match_federated_local_passes: true`, centralized and local
baselines use the same expected full-dataset-equivalent number of passes as FL,
including partial participation.

## Output files

- `run_metadata.json`: run status, environment, dimensions, elapsed time, and errors;
- `resolved_config.json`: fully resolved benchmark configuration;
- `source_validation.json`: target alignment, date consistency, universe, and industry checks;
- `universe_filter.json`: price/microcap filter settings, cutoffs, and dropped rows;
- `heterogeneity.json`: measured client-size, distribution, feature, and participation heterogeneity;
- `partition_assignments.parquet`: formation/target dates and training-defined clients;
- `feature_masks.csv`: client-specific feature availability;
- `checkpoints/<method>.pt`: selected validation checkpoint for each method;
- `benchmark_results.csv`: consolidated method comparison;
- `training_<algorithm>.csv`: per-round loss and communication;
- `validation_history_<method>.csv`: validation metric and selected checkpoint;
- `predictions_<method>.parquet`: stock-level out-of-sample predictions and calibrated local portfolio score;
- `monthly_predictive_metrics_<method>.csv`: monthly MSE, IC, RankIC, and prediction dispersion;
- `portfolio_returns_<method>.csv`: explicitly selected primary portfolio;
- `portfolio_returns_global_<method>.csv`: unconstrained pooled portfolio;
- `portfolio_returns_client_neutral_<method>.csv`: equal-weighted within-client portfolio;
- `portfolio_client_exposures_<method>.csv`: monthly long/short client composition;
- `client_metrics_<method>.csv`: per-client predictive metrics;
- `client_portfolio_metrics_<method>.csv`: standalone client economic metrics.

## Multi-seed and robustness runs

The production grid contains five seeds, partial participation, additional local
steps, a $5 price screen, and a NYSE-breakpoint microcap exclusion:

```bash
python scripts/run_grid.py \
  --grid configs/crsp_compustat_profitability_grid.yaml
```

Failed grid cells are recorded in `run_manifest.json` and do not discard completed
runs when `continue_on_error: true`.

After a run, compute paired Newey--West return tests and moving-block-bootstrap
Sharpe confidence intervals with:

```bash
python scripts/analyze_results.py \
  --output-dir outputs/crsp_compustat_profitability \
  --variant primary \
  --bootstrap 2000
```

## Remaining publication work

The code is an experimental benchmark implementation, not the finished empirical
study. Before submission, run the full multi-seed grid and rolling windows, verify
SCAFFOLD and FedAdam against an established FL library on at least one setting, add
liquidity and value-weighted portfolio robustness checks, and document the exact
security-universe construction used in every table.

## Runtime logging

Long-running benchmark stages now log to both the terminal and a persistent file in each output directory. The default file is `run.log`; grid experiments additionally create `grid.log` at the grid root and one `run.log` per scenario/seed.

Recommended production settings:

```yaml
logging:
  level: INFO
  file: run.log
  console: true
  epoch_interval: 1
  local_client_interval: 1
  prediction_client_interval: 1
  round_interval: 1
  federated_client_details: false
  federated_local_epoch_interval: 0
```

The log reports data-loading time and memory, split sizes, client sizes, feature masks, heterogeneity calculation, each baseline, every communication round, prediction progress, evaluation metrics, file saving, and total elapsed time. To inspect progress while a run is active:

```bash
tail -f outputs/crsp_compustat_profitability/run.log
```

For maximum detail inside each federated round, set `federated_client_details: true`. This can produce thousands of lines for long runs, so it is disabled by default.

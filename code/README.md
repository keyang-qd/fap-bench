# FAP-Bench

A compact, reproducible implementation of the minimum viable benchmark proposed for a federated asset-pricing paper. It supports:

- persistent entity-level IID clients;
- financially meaningful domain clients (sector, exchange, region, or size bins);
- controlled feature-space heterogeneity through common-core/client-specific masks;
- temporal heterogeneity through partial client participation;
- centralized and local-only references;
- FedAvg, FedProx, SCAFFOLD, and FedAdam;
- predictive, economic, client-level, communication, and heterogeneity evaluation.

## Important data note

The supplied `sample(1).parquet` contains only 10 rows for a single `permno`. It is sufficient to verify the schema, but **not** sufficient for federated training or a cross-sectional portfolio. The full panel should contain many securities, many dates, and at least two domain groups.

The supplied sample contains no SIC/NAICS/sector field. The benchmark accepts any configured domain column. For the full study, add `sic2`, `naics2`, or a sector label. `exchcd` and `me`-quantile clients are supported as fallbacks, but they should not be presented as a substitute for sector heterogeneity without justification.

## Installation

```bash
cd fapbench
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Smoke test

```bash
python scripts/make_synthetic.py --output data/synthetic.parquet --stocks 40 --months 24 --features 8
python scripts/run_benchmark.py --config configs/synthetic_smoke.yaml
```

Main results appear in `outputs/synthetic_smoke/benchmark_results.csv`.

## Run on the full financial panel

1. Edit `configs/sample_parquet.yaml`.
2. Replace `data.path` with the full Parquet file.
3. Set `partition.domain_col` to `sic2`, `naics2`, or your sector column.
4. Confirm whether `retadj_1mn` is already forward aligned. Keep `target_shift_months: 0` when it is; use `1` only when the raw return must be shifted to t+1.
5. Run:

```bash
python scripts/run_benchmark.py --config configs/sample_parquet.yaml
```

## Experimental configurations for the paper

Create one YAML per scenario rather than using a full factorial design:

- **IID:** `partition.type: iid`, feature masks disabled, participation 1.0.
- **Domain:** `partition.type: domain`, `domain_col: sic2`.
- **Feature levels:** common fractions 1.0, 0.8, 0.5, 0.2 with a fixed domain split.
- **Temporal levels:** participation rates 1.0, 0.8, 0.5, 0.3 with masks disabled.
- **Combined stress:** one easy, moderate, and severe YAML.

## Output files

- `benchmark_results.csv`: consolidated model comparison.
- `heterogeneity.json`: measured client-size, feature, label, and temporal heterogeneity.
- `partition_assignments.parquet`: reproducible client assignment.
- `feature_masks.csv`: client-specific feature availability.
- `training_<algorithm>.csv`: per-round loss, active clients, and communicated parameters.
- `predictions_<method>.parquet`: stock-month predictions.
- `portfolio_returns_<method>.csv`: monthly long-short returns and turnover.
- `client_metrics_<method>.csv`: per-client predictive metrics.

## Methodological cautions

- Fit preprocessing on the training period only; this implementation does so.
- Preserve a stock's full history within one IID client to avoid a client definition that changes every month.
- Report realized Jaccard feature dissimilarity rather than calling the configured common fraction “feature overlap.”
- Treat centralized performance as a reference, not a guaranteed upper bound.
- Use multiple seeds and rolling periods for paper results.
- The SCAFFOLD and FedAdam implementations are research baselines. Before publication, validate hyperparameters and compare against an established library implementation on at least one configuration.

## Batch scenario grid

```bash
python scripts/run_grid.py --grid configs/synthetic_grid.yaml
python scripts/plot_results.py --results outputs/synthetic_grid/grid_results.csv
```

For the paper, use at least three seeds. The supplied synthetic grid uses one seed only to keep the smoke test fast.

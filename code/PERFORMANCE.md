# Runtime acceleration

The CRSP--Compustat benchmark is CPU intensive because the default experiment trains
centralized, local-only, FedAvg, FedProx, SCAFFOLD, and FedAdam models under a 50-pass
budget. The accelerated implementation preserves the configured rounds, local epochs,
batch size, learning rate, client sampling, and evaluation protocol. It changes only
how independent client computations and repeated data preparation are executed.

## Main optimizations

1. **Client arrays are built once.**
   The original trainer filtered the full training DataFrame and copied features into
   NumPy arrays for every client in every communication round. The revised runner
   materializes one contiguous float32 array per client once and reuses it across all
   rounds and all federated algorithms.

2. **Shared-memory process parallelism.**
   Independent client updates within a communication round run in spawned CPU worker
   processes. Feature and target arrays are stored once in shared memory, so workers do
   not receive a full copy of each client dataset on every round.

3. **One persistent worker pool for all FL algorithms.**
   The same worker pool and shared-memory client cache are reused by FedAvg, FedProx,
   SCAFFOLD, and FedAdam. The first FL round includes process startup cost; later rounds
   and later algorithms do not restart Python/PyTorch workers.

4. **Deterministic per-client seeds.**
   Each client-round task gets a stable seed derived from the global seed, round, and
   client ID. Serial and parallel execution therefore use the same shuffle/dropout
   streams for a given task.

5. **Cached validation arrays.**
   Repeated checkpoint selection no longer groups and copies the validation DataFrame
   on every evaluation. Masked float32 validation arrays are prepared once per method.

6. **Separate main-process and worker thread settings.**
   Centralized/local training can use several PyTorch CPU threads, while each client
   worker uses one thread to avoid nested oversubscription.

## Recommended Apple-silicon configuration

The supplied `configs/crsp_compustat_profitability.yaml` uses:

```yaml
# Centralized and local-only training in the main process.
torch_num_threads: 4
torch_num_interop_threads: 1

performance:
  cache_client_arrays: true
  parallel_clients: 4
  multiprocessing_start_method: spawn
  worker_torch_threads: 1
  prediction_batch_size: 65536
```

For an M1 Pro, start with four client workers. If Activity Monitor shows yellow/red
memory pressure or substantial swap usage, reduce `parallel_clients` to 2. If CPU usage
stays low and memory pressure is green, test 6 workers.

Do not set `worker_torch_threads` above 1 while using multiple client processes unless
you have profiled it. Four workers with four PyTorch threads each can create severe CPU
oversubscription.

## Running

```bash
cd /Users/keyang/fap-bench/code
pip install -e .

caffeinate -i python -u scripts/run_benchmark.py \
  --config configs/crsp_compustat_profitability.yaml
```

Useful log lines:

```text
Federated client cache | clients=12 | rows=... | memory=... MB
Reusable parallel runtime ready | workers=4 | reused_across_algorithms=true
fedavg | parallel clients enabled | workers=4 | ... | reused_runtime=True
fedavg | round=1/50 | done | ... | workers=4 | elapsed=...
fedprox | round=1/50 | done | ...
```

The first federated round can be noticeably slower because spawned workers import
Python, NumPy, pandas, and PyTorch. Judge steady-state speed from rounds 2 onward.

## Continuing after a partial old run

A process that is already running cannot adopt the new implementation. Stop it and
restart with the new code. To avoid rerunning completed methods, change the config
before restarting. For example, if centralized and local-only outputs already exist:

```yaml
baselines:
  centralized: false
  centralized_masked: false
  local: false

federated:
  algorithms: [fedavg, fedprox, scaffold, fedadam]
```

If FedAvg and FedProx are also complete:

```yaml
federated:
  algorithms: [scaffold, fedadam]
```

Use a new `output_dir` when comparing old and accelerated runs, or back up the existing
output directory first.

## Scientific comparability

The default accelerated config keeps the original `batch_size: 2048`, 50 rounds,
one local epoch, optimizer, and model-selection frequency. Therefore the computational
budget and benchmark definition are unchanged. Small floating-point differences can
still occur across hardware or PyTorch versions.

Increasing batch size, reducing rounds, reducing baseline epochs, or increasing
`model_selection.evaluate_every` can reduce runtime further, but those changes alter
the optimization or selection protocol and should be treated as a separate experiment.

## Validation performed

- Unit tests: serial and process-parallel FedAvg match under deterministic seeds,
  including dropout.
- Reuse test: one worker pool successfully runs FedAvg, FedProx, SCAFFOLD, and FedAdam
  sequentially.
- End-to-end synthetic benchmark: all baselines, all four FL algorithms, predictions,
  portfolio evaluation, JSON output, and summary generation complete successfully.

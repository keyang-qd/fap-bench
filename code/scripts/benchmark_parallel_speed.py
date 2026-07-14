from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np
import pandas as pd
import torch

from fapbench.algorithms import FederatedTrainer
from fapbench.models import AssetPricingMLP


def make_frame(clients: int, rows_per_client: int, features: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=features).astype(np.float32)
    pieces = []
    columns = [f"x{i}" for i in range(features)]
    for client_index in range(clients):
        x = rng.normal(size=(rows_per_client, features)).astype(np.float32)
        y = (x @ beta + rng.normal(scale=0.5, size=rows_per_client)).astype(np.float32)
        frame = pd.DataFrame(x, columns=columns)
        frame["target"] = y
        frame["client_id"] = f"C{client_index:02d}"
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def run(frame: pd.DataFrame, workers: int, rounds: int, batch_size: int) -> float:
    feature_columns = [column for column in frame.columns if column.startswith("x")]
    model_fn = partial(
        AssetPricingMLP,
        input_dim=len(feature_columns),
        hidden_dims=(64, 32),
        dropout=0.1,
        batch_norm=False,
    )
    torch.manual_seed(42)
    trainer = FederatedTrainer(
        algorithm="fedavg",
        model_fn=model_fn,
        feature_columns=feature_columns,
        target_col="target",
        rounds=rounds,
        local_epochs=1,
        batch_size=batch_size,
        client_lr=0.01,
        seed=42,
        device="cpu",
        parallel_clients=workers,
        worker_torch_threads=1,
        cache_client_arrays=True,
    )
    started = time.perf_counter()
    trainer.fit(frame)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=12)
    parser.add_argument("--rows-per-client", type=int, default=20000)
    parser.add_argument("--features", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    frame = make_frame(args.clients, args.rows_per_client, args.features, 42)
    serial = run(frame, 1, args.rounds, args.batch_size)
    parallel = run(frame, args.workers, args.rounds, args.batch_size)
    print(f"rows={len(frame):,}")
    print(f"serial_seconds={serial:.3f}")
    print(f"parallel_seconds={parallel:.3f}")
    print(f"speedup={serial / parallel:.2f}x")


if __name__ == "__main__":
    main()

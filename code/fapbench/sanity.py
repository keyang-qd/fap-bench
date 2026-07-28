from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from fapbench.algorithms.common import clone_state_dict, make_loader, train_one_client
from fapbench.algorithms.federated import FederatedTrainer
from fapbench.partition import assign_iid_clients
from fapbench.utils import set_seed


def _model_fn(n_features: int) -> Callable[[], nn.Module]:
    # The benchmark expects one prediction per row. Avoid Flatten collapsing the
    # batch for a single observation by using a tiny custom module instead.
    class LinearRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x).squeeze(-1)

    return LinearRegressor


def _max_state_difference(
    left: OrderedDict[str, torch.Tensor], right: OrderedDict[str, torch.Tensor]
) -> float:
    values = []
    for key in left:
        if torch.is_floating_point(left[key]):
            values.append(float(torch.max(torch.abs(left[key] - right[key]))))
    return max(values, default=0.0)


def run_sanity(seed: int, n_entities: int, n_clients: int, tolerance: float) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n_features = 4
    rows = []
    beta = np.asarray([0.4, -0.2, 0.1, 0.3])
    for entity in range(n_entities):
        for month in range(2):
            x = rng.normal(size=n_features)
            y = float(x @ beta + rng.normal(scale=0.01))
            rows.append(
                {
                    "permno": entity,
                    "month": month,
                    **{f"x{i}": float(x[i]) for i in range(n_features)},
                    "target": y,
                }
            )
    frame = assign_iid_clients(
        pd.DataFrame(rows), "permno", num_clients=n_clients, seed=seed
    )
    features = [f"x{i}" for i in range(n_features)]
    model_fn = _model_fn(n_features)

    set_seed(seed + 1)
    trainer = FederatedTrainer(
        algorithm="fedavg",
        model_fn=model_fn,
        feature_columns=features,
        target_col="target",
        rounds=1,
        local_epochs=1,
        batch_size=len(frame) + 1,
        client_lr=0.01,
        weight_decay=0.0,
        participation_rate=1.0,
        seed=seed + 2,
        device="cpu",
        parallel_clients=1,
    )
    initial_state = clone_state_dict(trainer.global_model)

    global_loader = make_loader(
        frame,
        features,
        "target",
        batch_size=len(frame) + 1,
        shuffle=False,
    )
    centralized, _ = train_one_client(
        model_fn,
        initial_state,
        global_loader,
        torch.device("cpu"),
        local_epochs=1,
        learning_rate=0.01,
        weight_decay=0.0,
        optimizer_name="sgd",
    )
    fedavg = trainer.fit(frame)

    difference = _max_state_difference(
        clone_state_dict(centralized), clone_state_dict(fedavg)
    )
    counts = frame.groupby("client_id")["permno"].nunique()
    report = {
        "status": "passed" if difference <= tolerance else "failed",
        "seed": seed,
        "n_entities": n_entities,
        "n_clients": n_clients,
        "min_entities_per_client": int(counts.min()),
        "max_entities_per_client": int(counts.max()),
        "max_parameter_abs_difference": difference,
        "tolerance": tolerance,
        "interpretation": (
            "One full-batch FedAvg round matches one centralized full-batch SGD step."
        ),
    }
    return report


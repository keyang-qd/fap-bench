from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import torch

from fapbench.algorithms import FederatedTrainer
from fapbench.models import AssetPricingMLP


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for client_id in ["A", "B", "C", "D"]:
        x = rng.normal(size=(64, 3)).astype(np.float32)
        y = (0.3 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2]).astype(np.float32)
        for index in range(len(y)):
            rows.append(
                {
                    "client_id": client_id,
                    "x1": x[index, 0],
                    "x2": x[index, 1],
                    "x3": x[index, 2],
                    "target": y[index],
                }
            )
    return pd.DataFrame(rows)


def _trainer(workers: int) -> FederatedTrainer:
    model_fn = partial(
        AssetPricingMLP,
        input_dim=3,
        hidden_dims=(8,),
        dropout=0.1,
        batch_norm=False,
    )
    torch.manual_seed(42)
    return FederatedTrainer(
        algorithm="fedavg",
        model_fn=model_fn,
        feature_columns=["x1", "x2", "x3"],
        target_col="target",
        rounds=2,
        local_epochs=1,
        batch_size=32,
        client_lr=0.01,
        seed=42,
        device="cpu",
        parallel_clients=workers,
        worker_torch_threads=1,
        cache_client_arrays=True,
    )


def test_parallel_client_training_matches_serial_with_dropout() -> None:
    frame = _frame()
    serial = _trainer(1)
    parallel = _trainer(2)
    serial_model = serial.fit(frame)
    parallel_model = parallel.fit(frame)
    for key, serial_value in serial_model.state_dict().items():
        torch.testing.assert_close(serial_value, parallel_model.state_dict()[key])


def test_parallel_runtime_can_be_reused_across_algorithms() -> None:
    from fapbench.algorithms.common import prepare_client_arrays
    from fapbench.algorithms.parallel import ParallelClientRuntime

    frame = _frame()
    feature_columns = ["x1", "x2", "x3"]
    arrays = prepare_client_arrays(frame, feature_columns, "target")
    model_fn = partial(
        AssetPricingMLP,
        input_dim=3,
        hidden_dims=(8,),
        dropout=0.1,
        batch_norm=False,
    )
    with ParallelClientRuntime(arrays, max_workers=2, worker_torch_threads=1) as runtime:
        for algorithm in ["fedavg", "fedprox", "scaffold", "fedadam"]:
            torch.manual_seed(42)
            trainer = FederatedTrainer(
                algorithm=algorithm,
                model_fn=model_fn,
                feature_columns=feature_columns,
                target_col="target",
                rounds=1,
                local_epochs=1,
                batch_size=32,
                client_lr=0.01,
                seed=42,
                device="cpu",
                parallel_clients=2,
                worker_torch_threads=1,
                cache_client_arrays=True,
            )
            model = trainer.fit(
                frame,
                client_arrays=arrays,
                parallel_runtime=runtime,
            )
            assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_federated_trainer_uses_supplied_client_schedule() -> None:
    frame = _frame()
    trainer = _trainer(1)
    schedule = {0: ["A"], 1: ["B", "C"]}
    trainer.fit(frame, client_schedule=schedule)
    history = trainer.history.to_frame()
    assert history["active_clients"].tolist() == [1, 2]


def test_fedprox_proximal_term_changes_the_trained_client():
    """A reviewer asked whether FedProx's proximal term does anything at all.

    FedProx tracks FedAvg to five decimals in the reported table, which is
    consistent with either a correct implementation in a regime where client drift
    is small (one local epoch, so there is little drift to penalize) or a silently
    inert penalty. This distinguishes them by running the real client-training path
    twice on identical data and seeds, changing only mu: a live penalty must move
    the trained parameters, and a large mu must hold them closer to the global point
    than a small one.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from fapbench.algorithms.common import train_one_client

    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = (x @ torch.tensor([1.0, -0.5, 0.25, 0.0])).unsqueeze(1)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    model_fn = partial(AssetPricingMLP, input_dim=4, hidden_dims=(8,))
    global_state = {k: v.clone() for k, v in model_fn().state_dict().items()}

    def train(mu: float) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model, _ = train_one_client(
            model_fn,
            global_state,
            loader,
            torch.device("cpu"),
            local_epochs=5,
            learning_rate=0.1,
            proximal_mu=mu,
        )
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def distance_from_global(state) -> float:
        return float(
            sum(torch.sum((state[k] - global_state[k]) ** 2) for k in global_state)
        )

    plain, penalized = train(0.0), train(1.0)
    # The penalty must actually change the result...
    assert any(
        not torch.allclose(plain[k], penalized[k]) for k in plain
    ), "mu had no effect: the proximal term is inert"
    # ...and in the direction the penalty defines.
    assert distance_from_global(penalized) < distance_from_global(plain)

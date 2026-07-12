from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import torch
from torch import nn

from fapbench.algorithms.common import make_loader, train_one_client


def train_centralized(
    train_frame: pd.DataFrame,
    model_fn: Callable[[], nn.Module],
    feature_columns: list[str],
    target_col: str,
    epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> nn.Module:
    device_obj = torch.device(device)
    initial = model_fn().to(device_obj)
    loader = make_loader(
        train_frame, feature_columns, target_col, batch_size=batch_size, shuffle=True
    )
    model, _ = train_one_client(
        model_fn,
        initial.state_dict(),
        loader,
        device_obj,
        local_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return model


def train_local_models(
    train_frame: pd.DataFrame,
    model_fn: Callable[[], nn.Module],
    feature_columns: list[str],
    target_col: str,
    client_col: str = "client_id",
    feature_masks: Mapping[str, Any] | None = None,
    epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> dict[str, nn.Module]:
    device_obj = torch.device(device)
    result: dict[str, nn.Module] = {}
    for client_id, client_frame in train_frame.groupby(client_col, sort=True):
        client_id = str(client_id)
        initial = model_fn().to(device_obj)
        mask = None if feature_masks is None else feature_masks.get(client_id)
        loader = make_loader(
            client_frame,
            feature_columns,
            target_col,
            batch_size=batch_size,
            shuffle=True,
            mask=mask,
        )
        model, _ = train_one_client(
            model_fn,
            initial.state_dict(),
            loader,
            device_obj,
            local_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        result[client_id] = model
    return result

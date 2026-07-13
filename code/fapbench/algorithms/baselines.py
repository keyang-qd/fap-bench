from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import torch
from torch import nn

from fapbench.algorithms.common import make_loader, train_one_client
from fapbench.utils import format_seconds


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
    optimizer_name: str = "sgd",
    momentum: float = 0.0,
    logger: logging.Logger | None = None,
    epoch_log_interval: int = 1,
    log_name: str = "centralized",
    epoch_end_callback: Callable[[nn.Module, int, float], None] | None = None,
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
        optimizer_name=optimizer_name,
        momentum=momentum,
        logger=logger,
        log_prefix=log_name,
        epoch_log_interval=epoch_log_interval,
        epoch_end_callback=epoch_end_callback,
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
    optimizer_name: str = "sgd",
    momentum: float = 0.0,
    logger: logging.Logger | None = None,
    client_log_interval: int = 1,
    epoch_log_interval: int = 0,
    epoch_end_callbacks: Mapping[str, Callable[[nn.Module, int, float], None]] | None = None,
) -> dict[str, nn.Module]:
    device_obj = torch.device(device)
    result: dict[str, nn.Module] = {}
    shared_initial = model_fn().to(device_obj).state_dict()
    groups = list(train_frame.groupby(client_col, sort=True))
    total_clients = len(groups)
    started = time.perf_counter()
    if logger is not None:
        logger.info(
            "local | start | clients=%s | rows=%s | epochs_per_client=%s | device=%s",
            total_clients,
            f"{len(train_frame):,}",
            epochs,
            device_obj,
        )

    for client_index, (client_id, client_frame) in enumerate(groups, start=1):
        client_id = str(client_id)
        mask = None if feature_masks is None else feature_masks.get(client_id)
        should_log_client = (
            logger is not None
            and client_log_interval > 0
            and (
                client_index == 1
                or client_index == total_clients
                or client_index % client_log_interval == 0
            )
        )
        if should_log_client:
            logger.info(
                "local | client=%s/%s | client_id=%s | rows=%s | start",
                client_index,
                total_clients,
                client_id,
                f"{len(client_frame):,}",
            )
        loader = make_loader(
            client_frame,
            feature_columns,
            target_col,
            batch_size=batch_size,
            shuffle=True,
            mask=mask,
        )
        model, loss = train_one_client(
            model_fn,
            shared_initial,
            loader,
            device_obj,
            local_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            momentum=momentum,
            logger=logger if epoch_log_interval > 0 else None,
            log_prefix=f"local client={client_id}",
            epoch_log_interval=epoch_log_interval,
            epoch_end_callback=(
                None
                if epoch_end_callbacks is None
                else epoch_end_callbacks.get(client_id)
            ),
        )
        result[client_id] = model
        if should_log_client:
            logger.info(
                "local | client=%s/%s | client_id=%s | done | mean_loss=%.8f",
                client_index,
                total_clients,
                client_id,
                loss,
            )

    if logger is not None:
        logger.info(
            "local | done | trained_clients=%s | elapsed=%s",
            len(result),
            format_seconds(time.perf_counter() - started),
        )
    return result

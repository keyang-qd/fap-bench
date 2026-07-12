from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

import pandas as pd
import torch
from torch import nn

from fapbench.algorithms.common import (
    clone_state_dict,
    make_loader,
    train_one_client,
    weighted_average_states,
)
from fapbench.partition.temporal import ClientSampler


@dataclass
class FederatedHistory:
    records: list[dict[str, float | int | str]] = field(default_factory=list)

    def add(self, **record: float | int | str) -> None:
        self.records.append(record)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


class FederatedTrainer:
    """Unified trainer for FedAvg, FedProx, SCAFFOLD, and FedAdam."""

    SUPPORTED = {"fedavg", "fedprox", "scaffold", "fedadam"}

    def __init__(
        self,
        algorithm: str,
        model_fn: Callable[[], nn.Module],
        feature_columns: list[str],
        target_col: str,
        client_col: str = "client_id",
        rounds: int = 20,
        local_epochs: int = 1,
        batch_size: int = 512,
        client_lr: float = 1e-3,
        weight_decay: float = 0.0,
        proximal_mu: float = 0.01,
        participation_rate: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        server_lr: float = 1e-2,
        server_beta1: float = 0.9,
        server_beta2: float = 0.99,
        server_tau: float = 1e-8,
    ) -> None:
        algorithm = algorithm.lower()
        if algorithm not in self.SUPPORTED:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        self.algorithm = algorithm
        self.model_fn = model_fn
        self.feature_columns = feature_columns
        self.target_col = target_col
        self.client_col = client_col
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.client_lr = client_lr
        self.weight_decay = weight_decay
        self.proximal_mu = proximal_mu
        self.participation_rate = participation_rate
        self.seed = seed
        self.device = torch.device(device)
        self.server_lr = server_lr
        self.server_beta1 = server_beta1
        self.server_beta2 = server_beta2
        self.server_tau = server_tau
        self.history = FederatedHistory()
        self.global_model = model_fn().to(self.device)
        self.client_controls: dict[str, OrderedDict[str, torch.Tensor]] = {}
        self.server_control: OrderedDict[str, torch.Tensor] | None = None
        self.server_m: OrderedDict[str, torch.Tensor] | None = None
        self.server_v: OrderedDict[str, torch.Tensor] | None = None

    @staticmethod
    def _zero_parameter_state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, torch.zeros_like(parameter.detach(), device="cpu"))
            for name, parameter in model.named_parameters()
        )

    def _initialize_scaffold(self, client_ids: list[str]) -> None:
        self.server_control = self._zero_parameter_state(self.global_model)
        self.client_controls = {
            client: self._zero_parameter_state(self.global_model) for client in client_ids
        }

    def _update_scaffold_controls(
        self,
        client_id: str,
        global_state: Mapping[str, torch.Tensor],
        local_model: nn.Module,
        steps: int,
    ) -> OrderedDict[str, torch.Tensor]:
        assert self.server_control is not None
        old_control = self.client_controls[client_id]
        new_control: OrderedDict[str, torch.Tensor] = OrderedDict()
        local_parameters = dict(local_model.named_parameters())
        denominator = max(1, steps) * self.client_lr
        for name, global_parameter in self.global_model.named_parameters():
            global_tensor = global_state[name].detach().cpu()
            local_tensor = local_parameters[name].detach().cpu()
            new_control[name] = (
                old_control[name]
                - self.server_control[name]
                + (global_tensor - local_tensor) / denominator
            )
        delta = OrderedDict(
            (name, new_control[name] - old_control[name]) for name in new_control
        )
        self.client_controls[client_id] = new_control
        return delta

    def _fedadam_update(
        self,
        global_state: Mapping[str, torch.Tensor],
        averaged_local: Mapping[str, torch.Tensor],
        round_index: int,
    ) -> OrderedDict[str, torch.Tensor]:
        if self.server_m is None or self.server_v is None:
            self.server_m = OrderedDict()
            self.server_v = OrderedDict()
            for key, value in global_state.items():
                if torch.is_floating_point(value):
                    self.server_m[key] = torch.zeros_like(value)
                    self.server_v[key] = torch.zeros_like(value)

        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, global_tensor in global_state.items():
            if not torch.is_floating_point(global_tensor):
                result[key] = global_tensor.detach().clone()
                continue
            # Pseudo-gradient points from averaged client model toward server model.
            gradient = global_tensor - averaged_local[key]
            self.server_m[key] = (
                self.server_beta1 * self.server_m[key]
                + (1.0 - self.server_beta1) * gradient
            )
            self.server_v[key] = (
                self.server_beta2 * self.server_v[key]
                + (1.0 - self.server_beta2) * gradient.square()
            )
            m_hat = self.server_m[key] / (1.0 - self.server_beta1 ** (round_index + 1))
            v_hat = self.server_v[key] / (1.0 - self.server_beta2 ** (round_index + 1))
            result[key] = global_tensor - self.server_lr * m_hat / (
                torch.sqrt(v_hat) + self.server_tau
            )
        return result

    def fit(
        self,
        train_frame: pd.DataFrame,
        feature_masks: Mapping[str, object] | None = None,
    ) -> nn.Module:
        client_ids = sorted(train_frame[self.client_col].astype(str).unique().tolist())
        if len(client_ids) < 2:
            raise ValueError("Federated training requires at least two clients.")
        sampler = ClientSampler(client_ids, self.participation_rate, self.seed)
        if self.algorithm == "scaffold":
            self._initialize_scaffold(client_ids)

        for round_index in range(self.rounds):
            selected = sampler.sample(round_index)
            global_state = clone_state_dict(self.global_model)
            local_states: list[Mapping[str, torch.Tensor]] = []
            local_weights: list[float] = []
            local_losses: list[float] = []
            scaffold_deltas: list[OrderedDict[str, torch.Tensor]] = []

            for client_id in selected:
                client_frame = train_frame.loc[
                    train_frame[self.client_col].astype(str) == client_id
                ]
                mask = None if feature_masks is None else feature_masks.get(client_id)
                loader = make_loader(
                    client_frame,
                    self.feature_columns,
                    self.target_col,
                    self.batch_size,
                    shuffle=True,
                    mask=mask,
                )
                local_model, loss = train_one_client(
                    self.model_fn,
                    global_state,
                    loader,
                    self.device,
                    self.local_epochs,
                    self.client_lr,
                    weight_decay=self.weight_decay,
                    proximal_mu=self.proximal_mu if self.algorithm == "fedprox" else 0.0,
                    server_control=self.server_control if self.algorithm == "scaffold" else None,
                    client_control=(
                        self.client_controls[client_id]
                        if self.algorithm == "scaffold"
                        else None
                    ),
                )
                local_states.append(clone_state_dict(local_model))
                local_weights.append(float(len(client_frame)))
                local_losses.append(loss)

                if self.algorithm == "scaffold":
                    steps = self.local_epochs * len(loader)
                    scaffold_deltas.append(
                        self._update_scaffold_controls(
                            client_id, global_state, local_model, steps=steps
                        )
                    )

            averaged_local = weighted_average_states(local_states, local_weights)
            if self.algorithm == "fedadam":
                new_state = self._fedadam_update(global_state, averaged_local, round_index)
            else:
                new_state = averaged_local
            self.global_model.load_state_dict(new_state, strict=True)

            if self.algorithm == "scaffold":
                assert self.server_control is not None
                scale = len(selected) / len(client_ids)
                for name in self.server_control:
                    mean_delta = sum(delta[name] for delta in scaffold_deltas) / len(scaffold_deltas)
                    self.server_control[name] = self.server_control[name] + scale * mean_delta

            parameter_count = sum(parameter.numel() for parameter in self.global_model.parameters())
            communicated = 2 * parameter_count * len(selected)  # download + upload
            self.history.add(
                round=round_index + 1,
                algorithm=self.algorithm,
                active_clients=len(selected),
                mean_train_loss=float(sum(local_losses) / len(local_losses)),
                communicated_parameters=int(communicated),
            )

        return self.global_model

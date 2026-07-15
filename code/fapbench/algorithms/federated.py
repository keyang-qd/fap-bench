from __future__ import annotations

import logging
import time
import zlib
from collections import OrderedDict
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from fapbench.algorithms.common import (
    ClientArrays,
    clone_state_dict,
    make_loader,
    make_loader_from_arrays,
    prepare_client_arrays,
    train_one_client,
    weighted_average_states,
)
from fapbench.algorithms.parallel import (
    ClientTrainResult,
    ClientTrainTask,
    ParallelClientRuntime,
    create_process_pool,
    numpy_state_to_torch,
    torch_state_to_numpy,
    train_shared_client,
)
from fapbench.partition.temporal import ClientSampler
from fapbench.utils import format_seconds


@dataclass
class FederatedHistory:
    records: list[dict[str, float | int | str]] = field(default_factory=list)

    def add(self, **record: float | int | str) -> None:
        self.records.append(record)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


class FederatedTrainer:
    """Unified trainer for FedAvg, FedProx, SCAFFOLD, and FedAdam.

    CPU runs can parallelize independent client updates with persistent spawned
    processes. Client arrays are materialized once and placed in shared memory,
    avoiding repeated full-DataFrame scans and per-round array copies.
    """

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
        client_optimizer: str = "sgd",
        client_momentum: float = 0.0,
        logger: logging.Logger | None = None,
        round_log_interval: int = 1,
        log_client_details: bool = False,
        client_log_interval: int = 1,
        local_epoch_log_interval: int = 0,
        parallel_clients: int = 1,
        multiprocessing_start_method: str = "spawn",
        worker_torch_threads: int = 1,
        cache_client_arrays: bool = True,
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
        self.client_optimizer = client_optimizer.lower()
        self.client_momentum = client_momentum
        self.logger = logger
        self.round_log_interval = max(0, int(round_log_interval))
        self.log_client_details = bool(log_client_details)
        self.client_log_interval = max(1, int(client_log_interval))
        self.local_epoch_log_interval = max(0, int(local_epoch_log_interval))
        self.parallel_clients = max(1, int(parallel_clients))
        self.multiprocessing_start_method = str(multiprocessing_start_method)
        self.worker_torch_threads = max(1, int(worker_torch_threads))
        self.cache_client_arrays = bool(cache_client_arrays)
        if self.client_optimizer not in {"sgd", "adam"}:
            raise ValueError("client_optimizer must be sgd or adam.")
        if self.algorithm == "scaffold" and self.client_optimizer != "sgd":
            raise ValueError("SCAFFOLD requires client_optimizer=sgd in this implementation.")
        if self.parallel_clients > 1 and self.device.type != "cpu":
            if self.logger is not None:
                self.logger.warning(
                    "%s | parallel client training is CPU-only; falling back to serial on device=%s",
                    self.algorithm,
                    self.device,
                )
            self.parallel_clients = 1
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

    def _update_scaffold_controls_from_state(
        self,
        client_id: str,
        global_state: Mapping[str, torch.Tensor],
        local_state: Mapping[str, torch.Tensor],
        steps: int,
    ) -> OrderedDict[str, torch.Tensor]:
        assert self.server_control is not None
        old_control = self.client_controls[client_id]
        new_control: OrderedDict[str, torch.Tensor] = OrderedDict()
        denominator = max(1, steps) * self.client_lr
        parameter_names = {name for name, _ in self.global_model.named_parameters()}
        for name in parameter_names:
            global_tensor = global_state[name].detach().cpu()
            local_tensor = local_state[name].detach().cpu()
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

    def _should_log_round(self, round_number: int) -> bool:
        return (
            self.logger is not None
            and self.round_log_interval > 0
            and (
                round_number == 1
                or round_number == self.rounds
                or round_number % self.round_log_interval == 0
            )
        )

    def _task_seed(self, round_index: int, client_id: str) -> int:
        client_hash = zlib.crc32(client_id.encode("utf-8")) & 0xFFFFFFFF
        return int((self.seed + 1_000_003 * round_index + client_hash) % (2**31 - 1))

    def _serial_client_result(
        self,
        client_id: str,
        round_index: int,
        global_state: Mapping[str, torch.Tensor],
        arrays: ClientArrays | None,
        client_frame: pd.DataFrame | None,
        mask: object | None,
    ) -> ClientTrainResult:
        task_seed = self._task_seed(round_index, client_id)
        torch.manual_seed(task_seed)
        np.random.seed(task_seed % (2**32 - 1))
        if arrays is not None:
            loader = make_loader_from_arrays(
                arrays.x,
                arrays.y,
                batch_size=self.batch_size,
                shuffle=True,
                seed=task_seed,
            )
            n_rows = arrays.n_rows
        else:
            assert client_frame is not None
            loader = make_loader(
                client_frame,
                self.feature_columns,
                self.target_col,
                self.batch_size,
                shuffle=True,
                mask=mask,
                seed=task_seed,
            )
            n_rows = len(loader.dataset)
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
            optimizer_name=self.client_optimizer,
            momentum=self.client_momentum,
            logger=self.logger if self.local_epoch_log_interval > 0 else None,
            log_prefix=f"{self.algorithm} round={round_index + 1} client={client_id}",
            epoch_log_interval=self.local_epoch_log_interval,
        )
        return ClientTrainResult(
            client_id=client_id,
            state=torch_state_to_numpy(clone_state_dict(local_model)),
            mean_loss=float(loss),
            n_rows=int(n_rows),
            steps=int(self.local_epochs * len(loader)),
        )

    def fit(
        self,
        train_frame: pd.DataFrame,
        feature_masks: Mapping[str, object] | None = None,
        round_end_callback: Callable[[nn.Module, int, float], None] | None = None,
        client_arrays: Mapping[str, ClientArrays] | None = None,
        parallel_runtime: ParallelClientRuntime | None = None,
        client_schedule: Mapping[int, list[str]] | None = None,
    ) -> nn.Module:
        client_ids = sorted(train_frame[self.client_col].astype(str).unique().tolist())
        if len(client_ids) < 2:
            raise ValueError("Federated training requires at least two clients.")
        sampler = ClientSampler(client_ids, self.participation_rate, self.seed)
        resolved_schedule: dict[int, list[str]] | None = None
        if client_schedule is not None:
            resolved_schedule = {}
            known_clients = set(client_ids)
            for round_index in range(self.rounds):
                if round_index not in client_schedule:
                    raise KeyError(
                        f"client_schedule is missing round index {round_index}."
                    )
                selected = sorted(
                    set(str(value) for value in client_schedule[round_index])
                )
                unknown = set(selected) - known_clients
                if unknown:
                    raise ValueError(
                        "client_schedule contains unknown clients: "
                        f"{sorted(unknown)}"
                    )
                if not selected:
                    raise ValueError(
                        f"client_schedule round {round_index} has no active clients."
                    )
                resolved_schedule[round_index] = selected
        if self.algorithm == "scaffold":
            self._initialize_scaffold(client_ids)

        cache_started = time.perf_counter()
        prepared_client_arrays: dict[str, ClientArrays] | None = (
            None if client_arrays is None else dict(client_arrays)
        )
        if prepared_client_arrays is None and (self.cache_client_arrays or self.parallel_clients > 1):
            prepared_client_arrays = prepare_client_arrays(
                train_frame,
                self.feature_columns,
                self.target_col,
                client_col=self.client_col,
                feature_masks=feature_masks,  # type: ignore[arg-type]
            )
        if prepared_client_arrays is not None and self.logger is not None:
            total_bytes = sum(arrays.nbytes for arrays in prepared_client_arrays.values())
            self.logger.info(
                "%s | client cache ready | clients=%s | rows=%s | memory=%.1f MB | elapsed=%s | reused=%s",
                self.algorithm,
                len(prepared_client_arrays),
                f"{sum(item.n_rows for item in prepared_client_arrays.values()):,}",
                total_bytes / 1024**2,
                format_seconds(time.perf_counter() - cache_started),
                client_arrays is not None,
            )

        workers = min(self.parallel_clients, len(client_ids))
        own_runtime: ParallelClientRuntime | None = None
        runtime = parallel_runtime
        if runtime is not None:
            missing_specs = set(client_ids) - set(runtime.specs)
            if missing_specs:
                raise KeyError(
                    f"Parallel runtime is missing clients: {sorted(missing_specs)}"
                )
            workers = min(runtime.max_workers, len(client_ids))
        elif workers > 1:
            if prepared_client_arrays is None:
                raise RuntimeError("Parallel training requires prepared client arrays.")
            own_runtime = ParallelClientRuntime(
                prepared_client_arrays,
                max_workers=workers,
                start_method=self.multiprocessing_start_method,
                worker_torch_threads=self.worker_torch_threads,
            )
            runtime = own_runtime
        use_parallel = runtime is not None and workers > 1
        if use_parallel and self.logger is not None:
            self.logger.info(
                "%s | parallel clients enabled | workers=%s | start_method=%s | worker_torch_threads=%s | reused_runtime=%s",
                self.algorithm,
                workers,
                self.multiprocessing_start_method,
                self.worker_torch_threads,
                parallel_runtime is not None,
            )

        parameter_count = sum(
            parameter.numel() for parameter in self.global_model.parameters()
        )
        total_started = time.perf_counter()
        cumulative_communicated = 0
        if self.logger is not None:
            client_sizes = train_frame.groupby(self.client_col, observed=True).size()
            self.logger.info(
                "%s | start | rounds=%s | clients=%s | rows=%s | participation=%.3f | "
                "local_epochs=%s | batch_size=%s | optimizer=%s | client_lr=%g | "
                "parameters=%s | device=%s | parallel_workers=%s | "
                "client_rows_min=%s | client_rows_median=%s | client_rows_max=%s",
                self.algorithm,
                self.rounds,
                len(client_ids),
                f"{len(train_frame):,}",
                self.participation_rate,
                self.local_epochs,
                self.batch_size,
                self.client_optimizer,
                self.client_lr,
                f"{parameter_count:,}",
                self.device,
                workers if use_parallel else 1,
                f"{int(client_sizes.min()):,}",
                f"{int(client_sizes.median()):,}",
                f"{int(client_sizes.max()):,}",
            )

        try:
            for round_index in range(self.rounds):
                round_number = round_index + 1
                round_started = time.perf_counter()
                selected = (
                    resolved_schedule[round_index]
                    if resolved_schedule is not None
                    else sampler.sample(round_index)
                )
                if self._should_log_round(round_number):
                    self.logger.info(
                        "%s | round=%s/%s | start | active_clients=%s | selected=%s",
                        self.algorithm,
                        round_number,
                        self.rounds,
                        len(selected),
                        ",".join(selected),
                    )

                global_state = clone_state_dict(self.global_model)
                results_by_client: dict[str, ClientTrainResult] = {}
                client_updates_started = time.perf_counter()

                if use_parallel:
                    assert runtime is not None
                    futures = {}
                    global_state_numpy = torch_state_to_numpy(global_state)
                    server_control_numpy = (
                        torch_state_to_numpy(self.server_control)
                        if self.algorithm == "scaffold" and self.server_control is not None
                        else None
                    )
                    client_controls_numpy = (
                        {
                            client_id: torch_state_to_numpy(self.client_controls[client_id])
                            for client_id in selected
                        }
                        if self.algorithm == "scaffold"
                        else {}
                    )
                    for client_id in selected:
                        task = ClientTrainTask(
                            client_id=client_id,
                            client_data=runtime.specs[client_id],
                            model_fn=self.model_fn,
                            global_state=global_state_numpy,
                            local_epochs=self.local_epochs,
                            batch_size=self.batch_size,
                            learning_rate=self.client_lr,
                            weight_decay=self.weight_decay,
                            proximal_mu=(
                                self.proximal_mu if self.algorithm == "fedprox" else 0.0
                            ),
                            optimizer_name=self.client_optimizer,
                            momentum=self.client_momentum,
                            seed=self._task_seed(round_index, client_id),
                            server_control=server_control_numpy,
                            client_control=(
                                client_controls_numpy[client_id]
                                if self.algorithm == "scaffold"
                                else None
                            ),
                        )
                        futures[runtime.pool.submit(train_shared_client, task)] = client_id
                    for completed_index, future in enumerate(as_completed(futures), start=1):
                        client_id = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f"Parallel client update failed for client {client_id!r} "
                                f"in {self.algorithm} round {round_number}."
                            ) from exc
                        results_by_client[result.client_id] = result
                        if (
                            self.logger is not None
                            and self.log_client_details
                            and (
                                completed_index == 1
                                or completed_index == len(selected)
                                or completed_index % self.client_log_interval == 0
                            )
                        ):
                            self.logger.info(
                                "%s | round=%s/%s | completed_clients=%s/%s | client_id=%s | mean_loss=%.8f",
                                self.algorithm,
                                round_number,
                                self.rounds,
                                completed_index,
                                len(selected),
                                result.client_id,
                                result.mean_loss,
                            )
                else:
                    for client_index, client_id in enumerate(selected, start=1):
                        client_started = time.perf_counter()
                        should_log_client = (
                            self.logger is not None
                            and self.log_client_details
                            and (
                                client_index == 1
                                or client_index == len(selected)
                                or client_index % self.client_log_interval == 0
                            )
                        )
                        if should_log_client:
                            self.logger.info(
                                "%s | round=%s/%s | client=%s/%s | client_id=%s | start",
                                self.algorithm,
                                round_number,
                                self.rounds,
                                client_index,
                                len(selected),
                                client_id,
                            )
                        arrays = None if prepared_client_arrays is None else prepared_client_arrays[client_id]
                        client_frame = None
                        mask = None
                        if arrays is None:
                            client_frame = train_frame.loc[
                                train_frame[self.client_col].astype(str) == client_id
                            ]
                            mask = None if feature_masks is None else feature_masks.get(client_id)
                        result = self._serial_client_result(
                            client_id,
                            round_index,
                            global_state,
                            arrays,
                            client_frame,
                            mask,
                        )
                        results_by_client[client_id] = result
                        if should_log_client:
                            self.logger.info(
                                "%s | round=%s/%s | client=%s/%s | client_id=%s | done | mean_loss=%.8f | elapsed=%s",
                                self.algorithm,
                                round_number,
                                self.rounds,
                                client_index,
                                len(selected),
                                client_id,
                                result.mean_loss,
                                format_seconds(time.perf_counter() - client_started),
                            )

                client_update_seconds = time.perf_counter() - client_updates_started
                ordered_results = [results_by_client[client_id] for client_id in selected]
                aggregation_started = time.perf_counter()
                local_states = [numpy_state_to_torch(result.state) for result in ordered_results]
                local_weights = [float(result.n_rows) for result in ordered_results]
                local_losses = [result.mean_loss for result in ordered_results]

                scaffold_deltas: list[OrderedDict[str, torch.Tensor]] = []
                if self.algorithm == "scaffold":
                    for result in ordered_results:
                        scaffold_deltas.append(
                            self._update_scaffold_controls_from_state(
                                result.client_id,
                                global_state,
                                numpy_state_to_torch(result.state),
                                steps=result.steps,
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

                aggregation_seconds = time.perf_counter() - aggregation_started
                vectors_per_client = 4 if self.algorithm == "scaffold" else 2
                communicated = vectors_per_client * parameter_count * len(selected)
                cumulative_communicated += communicated
                mean_loss = float(sum(local_losses) / len(local_losses))
                round_elapsed = time.perf_counter() - round_started
                self.history.add(
                    round=round_number,
                    algorithm=self.algorithm,
                    active_clients=len(selected),
                    mean_train_loss=mean_loss,
                    communicated_parameters=int(communicated),
                    cumulative_communicated_parameters=int(cumulative_communicated),
                    elapsed_seconds=float(round_elapsed),
                    client_update_seconds=float(client_update_seconds),
                    server_aggregation_seconds=float(aggregation_seconds),
                    parallel_workers=int(workers if use_parallel else 1),
                )
                if round_end_callback is not None:
                    round_end_callback(self.global_model, round_number, mean_loss)
                if self._should_log_round(round_number):
                    self.logger.info(
                        "%s | round=%s/%s | done | mean_train_loss=%.8f | "
                        "communicated_parameters=%s | cumulative=%s | workers=%s | elapsed=%s",
                        self.algorithm,
                        round_number,
                        self.rounds,
                        mean_loss,
                        f"{communicated:,}",
                        f"{cumulative_communicated:,}",
                        workers if use_parallel else 1,
                        format_seconds(round_elapsed),
                    )
        finally:
            if own_runtime is not None:
                own_runtime.close()

        if self.logger is not None:
            self.logger.info(
                "%s | done | rounds=%s | cumulative_communicated_parameters=%s | elapsed=%s",
                self.algorithm,
                self.rounds,
                f"{cumulative_communicated:,}",
                format_seconds(time.perf_counter() - total_started),
            )
        return self.global_model

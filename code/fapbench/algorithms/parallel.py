from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn

from fapbench.algorithms.common import (
    ClientArrays,
    clone_state_dict,
    make_loader_from_arrays,
    train_one_client,
)


NumpyState = OrderedDict[str, np.ndarray]


@dataclass(frozen=True)
class SharedArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SharedClientSpec:
    x: SharedArraySpec
    y: SharedArraySpec
    n_rows: int


@dataclass
class ClientTrainTask:
    client_id: str
    client_data: SharedClientSpec
    model_fn: Callable[[], nn.Module]
    global_state: Mapping[str, np.ndarray]
    local_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    proximal_mu: float
    optimizer_name: str
    momentum: float
    seed: int
    server_control: Mapping[str, np.ndarray] | None = None
    client_control: Mapping[str, np.ndarray] | None = None


@dataclass
class ClientTrainResult:
    client_id: str
    state: NumpyState
    mean_loss: float
    n_rows: int
    steps: int


def torch_state_to_numpy(state: Mapping[str, torch.Tensor]) -> NumpyState:
    return OrderedDict(
        (name, tensor.detach().cpu().numpy().copy()) for name, tensor in state.items()
    )


def numpy_state_to_torch(state: Mapping[str, np.ndarray]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, torch.from_numpy(np.asarray(array).copy())) for name, array in state.items()
    )


class SharedClientArrayStore:
    """Own shared-memory copies of client arrays for persistent worker processes."""

    def __init__(self, client_arrays: Mapping[str, ClientArrays]) -> None:
        self._segments: list[SharedMemory] = []
        self.specs: dict[str, SharedClientSpec] = {}
        try:
            for client_id, arrays in client_arrays.items():
                x_spec = self._put(arrays.x)
                y_spec = self._put(arrays.y)
                self.specs[str(client_id)] = SharedClientSpec(
                    x=x_spec,
                    y=y_spec,
                    n_rows=arrays.n_rows,
                )
        except Exception:
            self.close()
            raise

    def _put(self, array: np.ndarray) -> SharedArraySpec:
        contiguous = np.ascontiguousarray(array)
        shm = SharedMemory(create=True, size=max(1, contiguous.nbytes))
        target = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
        target[...] = contiguous
        self._segments.append(shm)
        return SharedArraySpec(
            name=shm.name,
            shape=tuple(int(value) for value in contiguous.shape),
            dtype=contiguous.dtype.str,
        )

    def close(self) -> None:
        for segment in self._segments:
            try:
                segment.close()
            except FileNotFoundError:
                pass
        for segment in self._segments:
            try:
                segment.unlink()
            except FileNotFoundError:
                pass
        self._segments.clear()

    def __enter__(self) -> SharedClientArrayStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def initialize_training_worker(torch_threads: int) -> None:
    threads = max(1, int(torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _attach(spec: SharedArraySpec) -> tuple[SharedMemory, np.ndarray]:
    shm = SharedMemory(name=spec.name, create=False)
    array = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
    return shm, array


def train_shared_client(task: ClientTrainTask) -> ClientTrainResult:
    x_shm, x = _attach(task.client_data.x)
    y_shm, y = _attach(task.client_data.y)
    try:
        torch.manual_seed(int(task.seed))
        np.random.seed(int(task.seed) % (2**32 - 1))
        global_state = numpy_state_to_torch(task.global_state)
        server_control = (
            None if task.server_control is None else numpy_state_to_torch(task.server_control)
        )
        client_control = (
            None if task.client_control is None else numpy_state_to_torch(task.client_control)
        )
        loader = make_loader_from_arrays(
            x,
            y,
            batch_size=task.batch_size,
            shuffle=True,
            seed=task.seed,
        )
        model, loss = train_one_client(
            task.model_fn,
            global_state,
            loader,
            torch.device("cpu"),
            local_epochs=task.local_epochs,
            learning_rate=task.learning_rate,
            weight_decay=task.weight_decay,
            proximal_mu=task.proximal_mu,
            server_control=server_control,
            client_control=client_control,
            optimizer_name=task.optimizer_name,
            momentum=task.momentum,
            logger=None,
            epoch_log_interval=0,
        )
        return ClientTrainResult(
            client_id=task.client_id,
            state=torch_state_to_numpy(clone_state_dict(model)),
            mean_loss=float(loss),
            n_rows=int(task.client_data.n_rows),
            steps=int(task.local_epochs * len(loader)),
        )
    finally:
        x_shm.close()
        y_shm.close()


def create_process_pool(
    max_workers: int,
    start_method: str = "spawn",
    worker_torch_threads: int = 1,
) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max(1, int(max_workers)),
        mp_context=get_context(start_method),
        initializer=initialize_training_worker,
        initargs=(max(1, int(worker_torch_threads)),),
    )


class ParallelClientRuntime:
    """Reusable shared-memory store and worker pool across FL algorithms."""

    def __init__(
        self,
        client_arrays: Mapping[str, ClientArrays],
        max_workers: int,
        start_method: str = "spawn",
        worker_torch_threads: int = 1,
    ) -> None:
        self.client_arrays = dict(client_arrays)
        self.store = SharedClientArrayStore(self.client_arrays)
        self.max_workers = max(1, int(max_workers))
        self.pool = create_process_pool(
            self.max_workers,
            start_method=start_method,
            worker_torch_threads=worker_torch_threads,
        )
        self._closed = False

    @property
    def specs(self) -> Mapping[str, SharedClientSpec]:
        return self.store.specs

    def close(self) -> None:
        if self._closed:
            return
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.store.close()
        self._closed = True

    def __enter__(self) -> ParallelClientRuntime:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

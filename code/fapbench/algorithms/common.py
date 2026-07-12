from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fapbench.partition.feature import apply_feature_mask


class PanelDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def frame_to_arrays(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    if mask is not None:
        x = apply_feature_mask(x, mask)
    y = frame[target_col].to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[valid], y[valid]


def make_loader(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    batch_size: int,
    shuffle: bool,
    mask: np.ndarray | None = None,
) -> DataLoader:
    x, y = frame_to_arrays(frame, feature_columns, target_col, mask=mask)
    if len(y) == 0:
        raise ValueError("A client split contains no valid rows.")
    dataset = PanelDataset(x, y)
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle)


def clone_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.detach().clone()) for name, tensor in model.state_dict().items())


def weighted_average_states(
    states: list[Mapping[str, torch.Tensor]],
    weights: list[float],
) -> OrderedDict[str, torch.Tensor]:
    if not states:
        raise ValueError("No model states were provided for aggregation.")
    weights_array = np.asarray(weights, dtype=float)
    if np.any(weights_array < 0) or weights_array.sum() <= 0:
        raise ValueError("Aggregation weights must be nonnegative and sum to a positive value.")
    weights_array = weights_array / weights_array.sum()
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in states[0]:
        first = states[0][key]
        if not torch.is_floating_point(first):
            result[key] = first.detach().clone()
            continue
        result[key] = sum(
            float(weight) * state[key].detach().to(dtype=first.dtype)
            for state, weight in zip(states, weights_array, strict=True)
        )
    return result


def predict_frame(
    model: nn.Module,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    device: torch.device,
    batch_size: int = 4096,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    x, _ = frame_to_arrays(frame, feature_columns, target_col, mask=mask)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(xb).detach().cpu().numpy())
    return np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)


def train_one_client(
    model_fn: Callable[[], nn.Module],
    global_state: Mapping[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    proximal_mu: float = 0.0,
    server_control: Mapping[str, torch.Tensor] | None = None,
    client_control: Mapping[str, torch.Tensor] | None = None,
) -> tuple[nn.Module, float]:
    model = model_fn().to(device)
    model.load_state_dict(global_state, strict=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    global_parameters = {
        name: tensor.detach().to(device)
        for name, tensor in global_state.items()
        if torch.is_floating_point(tensor)
    }
    total_loss = 0.0
    total_count = 0

    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = criterion(prediction, yb)
            if proximal_mu > 0:
                proximal = torch.zeros((), device=device)
                for name, parameter in model.named_parameters():
                    proximal = proximal + torch.sum((parameter - global_parameters[name]) ** 2)
                loss = loss + 0.5 * proximal_mu * proximal
            loss.backward()

            if server_control is not None and client_control is not None:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        correction = (
                            server_control[name].to(device) - client_control[name].to(device)
                        )
                        if parameter.grad is not None:
                            parameter.grad.add_(correction)

            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(yb)
            total_count += len(yb)

    return model, total_loss / max(total_count, 1)

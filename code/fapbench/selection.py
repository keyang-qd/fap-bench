from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn


class ValidationCheckpoint:
    """Track and restore the best validation checkpoint during a fixed budget.

    The training budget remains identical across methods; validation is used only
    to select the checkpoint evaluated on the test set.
    """

    def __init__(
        self,
        score_fn: Callable[[nn.Module], float],
        metric: str,
        mode: str = "max",
        evaluate_every: int = 1,
        total_steps: int | None = None,
        logger: logging.Logger | None = None,
        name: str = "model",
    ) -> None:
        mode = mode.lower()
        if mode not in {"max", "min"}:
            raise ValueError("Validation checkpoint mode must be max or min.")
        self.score_fn = score_fn
        self.metric = str(metric)
        self.mode = mode
        self.evaluate_every = max(1, int(evaluate_every))
        self.total_steps = None if total_steps is None else int(total_steps)
        self.logger = logger
        self.name = name
        self.best_score = -np.inf if mode == "max" else np.inf
        self.best_step: int | None = None
        self.best_state: OrderedDict[str, torch.Tensor] | None = None
        self.records: list[dict[str, float | int | str]] = []

    def _is_better(self, score: float) -> bool:
        if not np.isfinite(score):
            return False
        if self.best_state is None:
            return True
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def should_evaluate(self, step: int) -> bool:
        return (
            step == 1
            or step % self.evaluate_every == 0
            or (self.total_steps is not None and step == self.total_steps)
        )

    def __call__(self, model: nn.Module, step: int, train_loss: float) -> None:
        if not self.should_evaluate(step):
            return
        score = float(self.score_fn(model))
        selected = self._is_better(score)
        if selected:
            self.best_score = score
            self.best_step = int(step)
            self.best_state = OrderedDict(
                (name, tensor.detach().cpu().clone())
                for name, tensor in model.state_dict().items()
            )
        self.records.append(
            {
                "name": self.name,
                "step": int(step),
                "metric": self.metric,
                "validation_score": score,
                "train_loss": float(train_loss),
                "selected": bool(selected),
            }
        )
        if self.logger is not None:
            self.logger.info(
                "%s | validation | step=%s | %s=%.8f | selected=%s",
                self.name,
                step,
                self.metric,
                score,
                selected,
            )

    def restore(self, model: nn.Module) -> nn.Module:
        if self.best_state is not None:
            model.load_state_dict(self.best_state, strict=True)
        return model

    def metadata(self) -> dict[str, float | int | str | None]:
        score = (
            float(self.best_score)
            if self.best_state is not None and np.isfinite(self.best_score)
            else None
        )
        return {
            "selection_metric": self.metric,
            "selection_mode": self.mode,
            "selected_step": self.best_step,
            "validation_best_score": score,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def combine_selection_histories(
    selectors: Mapping[str, ValidationCheckpoint],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for client_id, selector in selectors.items():
        history = selector.to_frame()
        if history.empty:
            continue
        history.insert(0, "client_id", str(client_id))
        frames.append(history)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

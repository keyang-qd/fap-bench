from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ClientSampler:
    client_ids: list[str]
    participation_rate: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.client_ids:
            raise ValueError("client_ids cannot be empty.")
        if not 0.0 < self.participation_rate <= 1.0:
            raise ValueError("participation_rate must be in (0, 1].")

    def sample(self, round_index: int) -> list[str]:
        # Round-specific RNG makes selections reproducible even after restart.
        rng = np.random.default_rng(self.seed + 104729 * int(round_index))
        n = max(1, int(np.ceil(len(self.client_ids) * self.participation_rate)))
        selected = rng.choice(self.client_ids, size=n, replace=False)
        return sorted(str(value) for value in selected)

    def schedule(self, rounds: int) -> dict[int, list[str]]:
        return {round_idx: self.sample(round_idx) for round_idx in range(rounds)}


def temporal_unavailability(schedule: dict[int, Iterable[str]], all_clients: Iterable[str]) -> float:
    clients = set(all_clients)
    if not schedule or not clients:
        return 0.0
    participation = [len(set(active) & clients) / len(clients) for active in schedule.values()]
    return float(1.0 - np.mean(participation))

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _stable_hash(value: str) -> int:
    return zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF


@dataclass
class ClientSampler:
    """Deterministic client sampler with optional temporal heterogeneity.

    Modes
    -----
    random:
        Fixed-size random participation, matching the original implementation.
    persistent:
        Clients have heterogeneous, persistent availability probabilities. The
        configured ``participation_rate`` remains a per-round upper cap.
    block:
        Each client is unavailable for one deterministic contiguous block of
        rounds, then the participation cap is applied to the remaining clients.
    """

    client_ids: list[str]
    participation_rate: float = 1.0
    seed: int = 42
    mode: str = "random"
    total_rounds: int | None = None
    min_availability: float = 0.25
    max_availability: float = 1.0
    block_fraction: float = 0.35

    def __post_init__(self) -> None:
        self.client_ids = sorted(str(value) for value in self.client_ids)
        if not self.client_ids:
            raise ValueError("client_ids cannot be empty.")
        if not 0.0 < self.participation_rate <= 1.0:
            raise ValueError("participation_rate must be in (0, 1].")
        self.mode = str(self.mode).lower()
        if self.mode not in {"random", "persistent", "block"}:
            raise ValueError("ClientSampler mode must be random, persistent, or block.")
        if not 0.0 <= self.min_availability <= 1.0:
            raise ValueError("min_availability must be in [0, 1].")
        if not 0.0 <= self.max_availability <= 1.0:
            raise ValueError("max_availability must be in [0, 1].")
        if self.min_availability > self.max_availability:
            raise ValueError("min_availability cannot exceed max_availability.")
        if not 0.0 <= self.block_fraction < 1.0:
            raise ValueError("block_fraction must be in [0, 1).")
        if self.mode == "block" and (self.total_rounds is None or self.total_rounds < 1):
            raise ValueError("block mode requires total_rounds >= 1.")

    def availability_probabilities(self) -> dict[str, float]:
        if self.mode == "random":
            return {client: float(self.participation_rate) for client in self.client_ids}
        if self.mode == "block":
            base = 1.0 - float(self.block_fraction)
            return {
                client: float(min(base, self.participation_rate))
                for client in self.client_ids
            }
        rng = np.random.default_rng(self.seed + 7001)
        rates = np.linspace(
            float(self.min_availability),
            float(self.max_availability),
            num=len(self.client_ids),
        )
        rates = rates[rng.permutation(len(rates))]
        return {
            client: float(rate)
            for client, rate in zip(self.client_ids, rates, strict=True)
        }

    def _cap_candidates(
        self,
        candidates: list[str],
        *,
        round_index: int,
        rng: np.random.Generator,
        fallback_weights: np.ndarray | None = None,
    ) -> list[str]:
        target = max(1, int(np.ceil(len(self.client_ids) * self.participation_rate)))
        candidates = sorted(set(candidates))
        if not candidates:
            if fallback_weights is None:
                fallback_weights = np.ones(len(self.client_ids), dtype=float)
            fallback_weights = np.asarray(fallback_weights, dtype=float)
            fallback_weights = np.clip(fallback_weights, 0.0, None)
            if fallback_weights.sum() <= 0:
                fallback_weights = np.ones(len(self.client_ids), dtype=float)
            selected = rng.choice(
                self.client_ids,
                size=1,
                replace=False,
                p=fallback_weights / fallback_weights.sum(),
            )
            return [str(selected[0])]
        if len(candidates) <= target:
            return candidates
        selected = rng.choice(candidates, size=target, replace=False)
        return sorted(str(value) for value in selected)

    def sample(self, round_index: int) -> list[str]:
        rng = np.random.default_rng(self.seed + 104729 * int(round_index))
        if self.mode == "random":
            n = max(1, int(np.ceil(len(self.client_ids) * self.participation_rate)))
            selected = rng.choice(self.client_ids, size=n, replace=False)
            return sorted(str(value) for value in selected)

        if self.mode == "persistent":
            rates_map = self.availability_probabilities()
            rates = np.asarray([rates_map[client] for client in self.client_ids], dtype=float)
            active = [
                client
                for client, available in zip(
                    self.client_ids, rng.random(len(self.client_ids)) < rates, strict=True
                )
                if available
            ]
            return self._cap_candidates(
                active,
                round_index=round_index,
                rng=rng,
                fallback_weights=rates,
            )

        assert self.total_rounds is not None
        block_length = int(round(self.block_fraction * self.total_rounds))
        if self.block_fraction > 0 and block_length == 0:
            block_length = 1
        active: list[str] = []
        for client in self.client_ids:
            if block_length <= 0:
                active.append(client)
                continue
            start = int(
                (self.seed + _stable_hash(client))
                % max(1, self.total_rounds)
            )
            distance = (int(round_index) - start) % self.total_rounds
            if distance >= block_length:
                active.append(client)
        return self._cap_candidates(active, round_index=round_index, rng=rng)

    def schedule(self, rounds: int) -> dict[int, list[str]]:
        return {round_idx: self.sample(round_idx) for round_idx in range(rounds)}


def temporal_unavailability(
    schedule: dict[int, Iterable[str]], all_clients: Iterable[str]
) -> float:
    clients = set(str(value) for value in all_clients)
    if not schedule or not clients:
        return 0.0
    participation = [
        len(set(str(value) for value in active) & clients) / len(clients)
        for active in schedule.values()
    ]
    return float(1.0 - np.mean(participation))


def temporal_availability_report(
    schedule: dict[int, Iterable[str]], all_clients: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    clients = sorted(str(value) for value in all_clients)
    rounds = sorted(schedule)
    records: list[dict[str, float | int | str]] = []
    for client in clients:
        active_rounds = sum(client in set(map(str, schedule[index])) for index in rounds)
        rate = float(active_rounds / len(rounds)) if rounds else 0.0
        records.append(
            {
                "client_id": client,
                "active_rounds": int(active_rounds),
                "total_rounds": int(len(rounds)),
                "availability_rate": rate,
            }
        )
    frame = pd.DataFrame(records)
    rates = frame["availability_rate"].to_numpy(dtype=float) if not frame.empty else np.array([])
    summary: dict[str, float | int] = {
        "temporal_availability_mean": float(rates.mean()) if len(rates) else 0.0,
        "temporal_availability_std": float(rates.std(ddof=0)) if len(rates) else 0.0,
        "temporal_availability_cv": float(rates.std(ddof=0) / rates.mean())
        if len(rates) and rates.mean() > 0
        else 0.0,
        "temporal_availability_min": float(rates.min()) if len(rates) else 0.0,
        "temporal_availability_max": float(rates.max()) if len(rates) else 0.0,
        "never_participated_clients": int((rates == 0).sum()) if len(rates) else 0,
    }
    return frame, summary


def schedule_to_frame(schedule: dict[int, Iterable[str]]) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for round_index in sorted(schedule):
        for client in sorted(str(value) for value in schedule[round_index]):
            rows.append({"round": int(round_index + 1), "client_id": client})
    return pd.DataFrame(rows)


def randomize_schedule_client_identity(
    reference_schedule: dict[int, Iterable[str]],
    all_clients: Iterable[str],
    *,
    seed: int,
) -> dict[int, list[str]]:
    """Randomize timing while preserving both sides of the schedule budget.

    Degree-preserving 2x2 edge swaps keep (i) the number of active clients in
    every round and (ii) every client's total number of selected rounds exactly
    equal to the structured reference.  Consequently total communication and
    aggregate client-update work are matched; only temporal identity structure
    such as persistence or contiguous outage blocks is disrupted.
    """
    clients = sorted(str(value) for value in all_clients)
    rounds = sorted(int(value) for value in reference_schedule)
    if not clients:
        raise ValueError("all_clients cannot be empty.")
    if not rounds:
        return {}

    client_index = {client: index for index, client in enumerate(clients)}
    matrix = np.zeros((len(rounds), len(clients)), dtype=np.bool_)
    for row_index, round_index in enumerate(rounds):
        for value in reference_schedule[round_index]:
            client = str(value)
            if client not in client_index:
                raise ValueError(
                    f"Reference schedule contains unknown client {client!r}."
                )
            matrix[row_index, client_index[client]] = True
        if not matrix[row_index].any():
            raise ValueError("Every reference round must contain at least one client.")

    original_matrix = matrix.copy()
    rng = np.random.default_rng(int(seed))
    n_edges = int(matrix.sum())
    target_successful_swaps = max(100, 20 * n_edges)
    max_attempts = max(1000, 100 * target_successful_swaps)
    successful_swaps = 0
    attempts = 0
    while attempts < max_attempts and successful_swaps < target_successful_swaps:
        attempts += 1
        r1, r2 = rng.choice(len(rounds), size=2, replace=False)
        c1, c2 = rng.choice(len(clients), size=2, replace=False)
        a = bool(matrix[r1, c1])
        b = bool(matrix[r1, c2])
        c = bool(matrix[r2, c1])
        d = bool(matrix[r2, c2])
        if a and not b and not c and d:
            matrix[r1, c1] = False
            matrix[r1, c2] = True
            matrix[r2, c1] = True
            matrix[r2, c2] = False
            successful_swaps += 1
        elif not a and b and c and not d:
            matrix[r1, c1] = True
            matrix[r1, c2] = False
            matrix[r2, c1] = False
            matrix[r2, c2] = True
            successful_swaps += 1

    if np.array_equal(matrix, original_matrix):
        # A finite swap chain can return to its starting state.  When a valid
        # degree-preserving swap exists, apply one deterministic final swap so
        # the matched control actually removes temporal identity structure.
        changed = False
        for r1 in range(len(rounds)):
            if changed:
                break
            for r2 in range(r1 + 1, len(rounds)):
                if changed:
                    break
                for c1 in range(len(clients)):
                    if changed:
                        break
                    for c2 in range(c1 + 1, len(clients)):
                        a = bool(matrix[r1, c1])
                        b = bool(matrix[r1, c2])
                        c = bool(matrix[r2, c1])
                        d = bool(matrix[r2, c2])
                        if a and not b and not c and d:
                            matrix[r1, c1] = False
                            matrix[r1, c2] = True
                            matrix[r2, c1] = True
                            matrix[r2, c2] = False
                            changed = True
                            break
                        if not a and b and c and not d:
                            matrix[r1, c1] = True
                            matrix[r1, c2] = False
                            matrix[r2, c1] = False
                            matrix[r2, c2] = True
                            changed = True
                            break

    randomized: dict[int, list[str]] = {}
    for row_index, round_index in enumerate(rounds):
        randomized[round_index] = [
            clients[column_index]
            for column_index in np.flatnonzero(matrix[row_index])
        ]
    return randomized


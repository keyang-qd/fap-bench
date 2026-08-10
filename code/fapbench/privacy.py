"""Client-level differential privacy for the federated aggregation step.

Why client-level, and why at the server
---------------------------------------
A reviewer's standing objection is that no privacy mechanism is priced anywhere
in this benchmark, while privacy is the reason one federates at all -- and that
"aggregation choice buys little" is precisely the claim most likely to invert
under noise, because the noise a server must add does not interact identically
with plain averaging, control variates, and an adaptive server optimizer.

The mechanism here is DP-FedAvg [McMahan et al., 2018]: each client's *update*
is clipped to an L2 norm C, the clipped updates are averaged uniformly, and
Gaussian noise calibrated to that average's sensitivity is added at the server.
It protects the participation of a whole client, which is the unit a
cross-silo federation of institutions actually cares about, and it is the
mechanism secure aggregation is built to support.

Two consequences are deliberate and must be reported rather than hidden:

1. **Uniform averaging replaces sample-weighted averaging when DP is on.** With
   weights proportional to client rows, one client's influence on the mean --
   and therefore the sensitivity -- depends on the size distribution, so the
   noise scale would silently vary with the partition. Uniform averaging fixes
   the sensitivity at C/K. This is a protocol change, not a free addition.

2. **SCAFFOLD's control variates are not privatised.** Only the model update is
   clipped and noised. The control variates are a second channel carrying
   client information, so at a matched noise multiplier SCAFFOLD's actual
   guarantee is weaker than FedAvg's or FedAdam's. Reporting SCAFFOLD at the
   same sigma therefore *flatters* it, which is the safe direction for a claim
   that federated arms lose under privacy, and the wrong direction for any
   claim that SCAFFOLD is the better private choice.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DPConfig:
    """Client-level DP settings. `enabled=False` must leave training bit-identical."""

    enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    delta: float = 1e-5

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "DPConfig":
        if not mapping:
            return cls()
        enabled = bool(mapping.get("enabled", False))
        clip = float(mapping.get("clip_norm", 1.0))
        sigma = float(mapping.get("noise_multiplier", 0.0))
        delta = float(mapping.get("delta", 1e-5))
        if enabled:
            if clip <= 0:
                raise ValueError("dp.clip_norm must be positive when dp is enabled.")
            if sigma < 0:
                raise ValueError("dp.noise_multiplier must be non-negative.")
            if not 0 < delta < 1:
                raise ValueError("dp.delta must lie in (0, 1).")
        return cls(enabled=enabled, clip_norm=clip, noise_multiplier=sigma, delta=delta)


def update_norm(delta: OrderedDict[str, torch.Tensor]) -> float:
    """L2 norm of a client update, taken over all parameters jointly.

    Jointly, not per-tensor: the sensitivity bound is on the whole update
    vector, so clipping each tensor to C separately would admit a total norm of
    C*sqrt(n_tensors) and silently understate the noise required.
    """
    total = 0.0
    for tensor in delta.values():
        total += float(torch.sum(tensor.double() ** 2).item())
    return math.sqrt(total)


def clip_update(
    delta: OrderedDict[str, torch.Tensor], clip_norm: float
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    """Scale an update down to `clip_norm` if it exceeds it. Returns the norm."""
    norm = update_norm(delta)
    if norm <= clip_norm or norm == 0.0:
        return delta, norm
    scale = clip_norm / norm
    clipped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in delta.items():
        clipped[key] = tensor * scale
    return clipped, norm


def privatise_mean_update(
    deltas: list[OrderedDict[str, torch.Tensor]],
    *,
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float]]:
    """Clip, uniformly average, and add Gaussian noise to a set of client updates.

    Sensitivity of the uniform mean of K clipped updates is ``clip_norm / K``
    under add/remove of one client, so the noise standard deviation is
    ``noise_multiplier * clip_norm / K``. Noise is drawn per coordinate.
    """
    if not deltas:
        raise ValueError("privatise_mean_update requires at least one client update.")
    n_clients = len(deltas)
    clipped, norms = [], []
    for delta in deltas:
        c, norm = clip_update(delta, clip_norm)
        clipped.append(c)
        norms.append(norm)

    mean: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in clipped[0]:
        stacked = torch.stack([c[key] for c in clipped], dim=0)
        mean[key] = stacked.mean(dim=0)

    sigma = noise_multiplier * clip_norm / n_clients
    if sigma > 0:
        for key, tensor in mean.items():
            noise = torch.normal(
                mean=0.0,
                std=sigma,
                size=tensor.shape,
                generator=generator,
                dtype=tensor.dtype,
            )
            mean[key] = tensor + noise
    diagnostics = {
        "dp_clip_norm": float(clip_norm),
        "dp_noise_multiplier": float(noise_multiplier),
        "dp_noise_std": float(sigma),
        "dp_n_clients": float(n_clients),
        "dp_median_update_norm": float(np.median(norms)) if norms else 0.0,
        "dp_fraction_clipped": float(np.mean([n > clip_norm for n in norms])),
    }
    return mean, diagnostics


def rdp_epsilon(noise_multiplier: float, rounds: int, delta: float) -> float:
    """(epsilon, delta)-DP for `rounds` compositions of the Gaussian mechanism.

    Full participation means there is no subsampling amplification to get wrong,
    which is the main reason a hand-written accountant is defensible here: the
    Renyi divergence of the Gaussian mechanism at order alpha is exactly
    ``alpha / (2 sigma^2)``, composition over T rounds is ``T alpha / (2
    sigma^2)``, and the conversion to (eps, delta) is the standard one
    [Mironov 2017; Balle et al. 2020, Prop. 12]:

        eps = eps_rdp(alpha) + log((alpha - 1)/alpha) - (log delta + log alpha)/(alpha - 1)

    minimised over alpha > 1. If sampling is ever added to this benchmark, this
    function must be replaced by a real accountant rather than adjusted.
    """
    if noise_multiplier <= 0:
        return float("inf")
    if rounds <= 0:
        return 0.0
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1).")
    orders = np.concatenate(
        [np.linspace(1.01, 10.0, 900), np.linspace(10.0, 512.0, 900)]
    )
    best = float("inf")
    for alpha in orders:
        rdp = rounds * alpha / (2.0 * noise_multiplier**2)
        eps = (
            rdp
            + math.log((alpha - 1.0) / alpha)
            - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
        )
        best = min(best, eps)
    return float(max(best, 0.0))

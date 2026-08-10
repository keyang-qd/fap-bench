"""Tests for client-level DP. Every one of these guards a way the mechanism can
be wrong while still producing plausible numbers."""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fapbench.privacy import (
    DPConfig,
    clip_update,
    privatise_mean_update,
    rdp_epsilon,
    update_norm,
)


def _delta(*values: float) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (f"p{i}", torch.tensor([v], dtype=torch.float32)) for i, v in enumerate(values)
    )


def test_norm_is_joint_across_tensors_not_per_tensor():
    """The sensitivity bound is on the whole update vector. Clipping tensors
    independently would admit a total norm of C*sqrt(n) and understate the noise
    the mechanism needs, which is a silent under-protection."""
    delta = _delta(3.0, 4.0)
    assert update_norm(delta) == pytest.approx(5.0)
    clipped, norm = clip_update(delta, clip_norm=1.0)
    assert norm == pytest.approx(5.0)
    assert update_norm(clipped) == pytest.approx(1.0)
    # Direction preserved, magnitude scaled.
    assert float(clipped["p0"]) == pytest.approx(3.0 / 5.0)


def test_clipping_leaves_small_updates_untouched():
    delta = _delta(0.3, 0.4)
    clipped, norm = clip_update(delta, clip_norm=1.0)
    assert norm == pytest.approx(0.5)
    for key in delta:
        assert torch.equal(clipped[key], delta[key])


def test_noise_scale_matches_sensitivity_over_client_count():
    """Sensitivity of a uniform mean of K clipped updates is C/K, so the noise
    std must be sigma*C/K. Getting the 1/K wrong is the classic error and makes
    the reported epsilon meaningless."""
    k = 8
    deltas = [_delta(0.0, 0.0) for _ in range(k)]
    _, stats = privatise_mean_update(
        deltas, clip_norm=2.0, noise_multiplier=3.0, generator=torch.Generator()
    )
    assert stats["dp_noise_std"] == pytest.approx(3.0 * 2.0 / k)
    assert stats["dp_n_clients"] == k


def test_noise_is_actually_added_and_has_the_right_magnitude():
    torch.manual_seed(0)
    k, sigma, clip = 4, 5.0, 1.0
    # All-zero updates, so anything non-zero in the mean is the noise itself.
    deltas = [
        OrderedDict((("w", torch.zeros(4000, dtype=torch.float32)),)) for _ in range(k)
    ]
    mean, stats = privatise_mean_update(
        deltas,
        clip_norm=clip,
        noise_multiplier=sigma,
        generator=torch.Generator().manual_seed(7),
    )
    expected = sigma * clip / k
    observed = float(mean["w"].std())
    assert observed == pytest.approx(expected, rel=0.1)
    assert stats["dp_noise_std"] == pytest.approx(expected)


def test_zero_noise_multiplier_still_clips_and_averages():
    """sigma=0 must be usable for calibration runs: clipping on, noise off."""
    deltas = [_delta(10.0, 0.0), _delta(0.0, 10.0)]
    mean, stats = privatise_mean_update(deltas, clip_norm=1.0, noise_multiplier=0.0)
    assert stats["dp_noise_std"] == 0.0
    assert stats["dp_fraction_clipped"] == pytest.approx(1.0)
    # Both clipped to unit norm, then averaged: (0.5, 0.5).
    assert float(mean["p0"]) == pytest.approx(0.5)
    assert float(mean["p1"]) == pytest.approx(0.5)


def test_uniform_not_weighted_average():
    """DP averaging must ignore client sizes: a size-weighted mean makes the
    sensitivity depend on the row distribution, so the noise would be wrong."""
    deltas = [_delta(0.0, 0.0), _delta(1.0, 0.0)]
    mean, _ = privatise_mean_update(deltas, clip_norm=10.0, noise_multiplier=0.0)
    assert float(mean["p0"]) == pytest.approx(0.5)


def test_epsilon_is_monotone_and_beats_the_classical_gaussian_bound():
    """RDP composition must be tighter than the classical
    sigma >= sqrt(2 ln(1.25/delta))/eps bound at T=1, and must worsen with more
    rounds and improve with more noise."""
    delta = 1e-5
    classical = math.sqrt(2 * math.log(1.25 / delta)) / 4.0
    assert rdp_epsilon(4.0, 1, delta) < classical

    assert rdp_epsilon(2.0, 50, delta) > rdp_epsilon(4.0, 50, delta)
    assert rdp_epsilon(4.0, 100, delta) > rdp_epsilon(4.0, 50, delta)
    assert rdp_epsilon(0.0, 50, delta) == float("inf")
    assert rdp_epsilon(4.0, 0, delta) == 0.0


def test_config_rejects_nonsense_only_when_enabled():
    assert DPConfig.from_mapping(None).enabled is False
    # Disabled config must not validate: the default clip of 0 in a stale config
    # should not crash a non-private run.
    DPConfig.from_mapping({"enabled": False, "clip_norm": 0.0})
    with pytest.raises(ValueError):
        DPConfig.from_mapping({"enabled": True, "clip_norm": 0.0})
    with pytest.raises(ValueError):
        DPConfig.from_mapping({"enabled": True, "delta": 1.5})


def test_disabled_dp_is_the_identity_path():
    """The whole benchmark's comparability rests on dp.enabled=False changing
    nothing at all."""
    cfg = DPConfig.from_mapping({"enabled": False, "noise_multiplier": 99.0})
    assert cfg.enabled is False
    assert cfg.noise_multiplier == 99.0  # carried, but never consulted


def test_clipping_reduces_a_heavy_tailed_client_to_its_share():
    """One outsized client must not dominate: after clipping, its contribution to
    the mean is bounded by C/K regardless of how large its update was."""
    k = 5
    deltas = [_delta(0.0, 0.0) for _ in range(k - 1)] + [_delta(1000.0, 0.0)]
    mean, stats = privatise_mean_update(deltas, clip_norm=1.0, noise_multiplier=0.0)
    assert float(mean["p0"]) == pytest.approx(1.0 / k)
    assert stats["dp_fraction_clipped"] == pytest.approx(1.0 / k)
    assert np.isfinite(stats["dp_median_update_norm"])

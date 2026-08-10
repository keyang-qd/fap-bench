"""The tighter covariate-shift control: matches pooled mean AND variance.

These tests exist because the manuscript's headline RQ3 comparison is only an
estimate rather than a bound if this control is genuinely tighter than the
first-moment one. "Tighter" is a checkable property, not a claim to assert.
"""
from __future__ import annotations

import numpy as np
import pytest

from fapbench.partition import (
    build_covariate_shift_matched_control_parameters,
    build_covariate_shift_parameters,
    build_covariate_shift_second_moment_matched_control_parameters,
)

CLIENTS = [f"client_{i:02d}" for i in range(12)]
FEATURES = [f"f{i}" for i in range(7)]
STRONG = dict(location_std=0.5, log_scale_std=0.3, seed=42)


def _pooled_moments(locations, scales, sample):
    """Equal-weighted pooled mean/variance of the transformed feature."""
    values = np.concatenate(
        [locations[c][0] + scales[c][0] * sample for c in CLIENTS]
    )
    return float(values.mean()), float(values.var())


def test_second_moment_control_matches_pooled_variance_where_first_moment_does_not():
    """The load-bearing property. The first-moment control leaves a large pooled
    variance gap; the second-moment control closes it."""
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 1.0, 400_000)

    treatment = build_covariate_shift_parameters(CLIENTS, FEATURES, **STRONG)
    first = build_covariate_shift_matched_control_parameters(CLIENTS, FEATURES, **STRONG)
    second = build_covariate_shift_second_moment_matched_control_parameters(
        CLIENTS, FEATURES, **STRONG
    )

    mean_t, var_t = _pooled_moments(*treatment, sample)
    mean_1, var_1 = _pooled_moments(*first, sample)
    mean_2, var_2 = _pooled_moments(*second, sample)

    # Both controls match the pooled mean.
    assert mean_1 == pytest.approx(mean_t, abs=5e-3)
    assert mean_2 == pytest.approx(mean_t, abs=5e-3)
    # Only the second matches the pooled variance, and the improvement is large.
    assert var_2 == pytest.approx(var_t, rel=2e-3)
    assert abs(var_1 - var_t) > 50 * abs(var_2 - var_t), (
        "the second-moment control must be dramatically tighter on variance; "
        f"first-moment err {abs(var_1 - var_t):.4f}, second {abs(var_2 - var_t):.4f}"
    )


def test_second_moment_control_still_zeroes_cross_client_dispersion():
    """Eq. (1) requires Disp_k = 0. A tighter control that broke this would not
    be a matched control at all."""
    locations, scales = build_covariate_shift_second_moment_matched_control_parameters(
        CLIENTS, FEATURES, **STRONG
    )
    location_matrix = np.stack([locations[c] for c in CLIENTS])
    scale_matrix = np.stack([scales[c] for c in CLIENTS])
    assert location_matrix.std(axis=0).max() < 1e-6
    assert scale_matrix.std(axis=0).max() < 1e-6


def test_shared_scale_exceeds_the_first_moment_control():
    """Mechanism check: the shared scale absorbs the cross-client dispersion of
    a_k and b_k, so it must sit above the geometric mean the first-moment control
    uses. If it did not, the variance match could not be closing from below."""
    _, first_scales = build_covariate_shift_matched_control_parameters(
        CLIENTS, FEATURES, **STRONG
    )
    _, second_scales = build_covariate_shift_second_moment_matched_control_parameters(
        CLIENTS, FEATURES, **STRONG
    )
    assert (second_scales[CLIENTS[0]] > first_scales[CLIENTS[0]]).all()


def test_zero_dispersion_treatment_collapses_both_controls_to_the_same_map():
    """With no cross-client variation there is nothing extra to absorb, so the
    two controls must coincide. This pins that the tighter control is a strict
    generalization rather than a different transform."""
    degenerate = dict(location_std=0.0, log_scale_std=0.0, seed=42)
    _, first = build_covariate_shift_matched_control_parameters(
        CLIENTS, FEATURES, **degenerate
    )
    _, second = build_covariate_shift_second_moment_matched_control_parameters(
        CLIENTS, FEATURES, **degenerate
    )
    np.testing.assert_allclose(first[CLIENTS[0]], second[CLIENTS[0]], rtol=1e-5)


def test_base_std_must_be_positive():
    with pytest.raises(ValueError, match="base_std"):
        build_covariate_shift_second_moment_matched_control_parameters(
            CLIENTS, FEATURES, base_std=0.0, **STRONG
        )


def test_runner_rejects_two_covariate_arms_at_once():
    """Enabling two arms would silently apply whichever builder is wired last,
    producing a scenario directory whose name no longer describes the run."""
    from fapbench.runner import COVARIATE_SHIFT_ARMS, resolve_covariate_shift_arms

    assert resolve_covariate_shift_arms({}) == []
    for arm in COVARIATE_SHIFT_ARMS:
        assert resolve_covariate_shift_arms({arm: {"enabled": True}}) == [arm]
    with pytest.raises(ValueError, match="at most one"):
        resolve_covariate_shift_arms(
            {
                "covariate_shift": {"enabled": True},
                "covariate_shift_second_moment_control": {"enabled": True},
            }
        )
    # A disabled sibling must not trip the guard.
    assert resolve_covariate_shift_arms(
        {
            "covariate_shift": {"enabled": True},
            "covariate_shift_matched_control": {"enabled": False},
        }
    ) == ["covariate_shift"]


def test_second_moment_config_pairs_against_the_published_treatment():
    """The dispersion estimator uses only seeds common to both arms, so a seed
    mismatch would silently shrink the contrast instead of failing."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent / "configs"
    het = yaml.safe_load((root / "icaif_heterogeneity.yaml").read_text())
    new = yaml.safe_load((root / "icaif_second_moment_control.yaml").read_text())
    assert new["seeds"] == het["seeds"], "seed sets must match the treatment grid"

    treatment = next(
        s for s in het["scenarios"] if s["name"] == "covariate_shift_strong"
    )
    control = new["scenarios"][0]
    t_cfg = treatment["overrides"]["statistical_heterogeneity"]["covariate_shift"]
    c_cfg = control["overrides"]["statistical_heterogeneity"][
        "covariate_shift_second_moment_control"
    ]
    for key in ("location_std", "log_scale_std"):
        assert c_cfg[key] == t_cfg[key], (
            f"{key} must match the treatment or the control re-draws different "
            "per-client parameters and the pairing breaks"
        )

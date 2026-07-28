import numpy as np

from fapbench.evaluation.statistics import (
    annualized_sharpe,
    moving_block_bootstrap_sharpe,
    newey_west_mean_test,
    newey_west_ols,
)


def test_newey_west_detects_positive_constant_difference():
    result = newey_west_mean_test(np.linspace(0.01, 0.02, 60), max_lag=3)
    assert result["mean_difference"] > 0
    assert result["p_value"] < 0.05


def test_newey_west_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    market = rng.normal(0.01, 0.04, 240)
    strategy = 0.003 - 0.25 * market + rng.normal(0.0, 0.005, 240)
    result = newey_west_ols(strategy, market)
    assert result["n_periods"] == 240
    assert abs(result["beta"] - (-0.25)) < 0.05
    assert abs(result["alpha"] - 0.003) < 0.002
    # A beta that far from zero must be detectable at this sample length.
    assert result["beta_p"] < 0.01


def test_newey_west_ols_handles_degenerate_input():
    # A constant regressor carries no information, so the fit is undefined
    # rather than silently reporting a spurious beta.
    result = newey_west_ols([0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0])
    assert np.isnan(result["beta"])
    assert result["n_periods"] == 4


def test_block_bootstrap_sharpe_returns_ordered_interval():
    returns = np.sin(np.arange(60)) * 0.01 + 0.005
    result = moving_block_bootstrap_sharpe(
        returns, n_bootstrap=100, block_length=4, seed=1
    )
    assert np.isfinite(result["sharpe"])
    assert result["ci_lower"] <= result["ci_upper"]
    assert annualized_sharpe(returns) == result["sharpe"]

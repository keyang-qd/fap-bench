import numpy as np

from fapbench.evaluation.statistics import (
    annualized_sharpe,
    moving_block_bootstrap_sharpe,
    newey_west_mean_test,
)


def test_newey_west_detects_positive_constant_difference():
    result = newey_west_mean_test(np.linspace(0.01, 0.02, 60), max_lag=3)
    assert result["mean_difference"] > 0
    assert result["p_value"] < 0.05


def test_block_bootstrap_sharpe_returns_ordered_interval():
    returns = np.sin(np.arange(60)) * 0.01 + 0.005
    result = moving_block_bootstrap_sharpe(
        returns, n_bootstrap=100, block_length=4, seed=1
    )
    assert np.isfinite(result["sharpe"])
    assert result["ci_lower"] <= result["ci_upper"]
    assert annualized_sharpe(returns) == result["sharpe"]

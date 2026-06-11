from __future__ import annotations

import numpy as np
import pandas as pd

from futures_fund import cointegration as co


def _cointegrated_pair(n: int = 400, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """x is a random walk; y = 2*x + stationary noise -> y and x are cointegrated."""
    rng = np.random.default_rng(seed)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0)
    noise = pd.Series(rng.normal(0, 0.5, n))
    y = 2.0 * x + noise
    return y, x


def test_engle_granger_recovers_hedge_ratio_and_rejects_unit_root():
    y, x = _cointegrated_pair()
    hedge_ratio, pvalue, stat = co.engle_granger(y, x)
    assert abs(hedge_ratio - 2.0) < 0.1          # OLS slope ~ 2.0
    assert pvalue < 0.05                          # residual is stationary -> reject unit root
    assert stat < 0.0                             # ADF stat is negative for a stationary series


def test_engle_granger_non_cointegrated_high_pvalue():
    rng = np.random.default_rng(11)
    y = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)   # two independent random walks
    _, pvalue, _ = co.engle_granger(y, x)
    assert pvalue > 0.05

from __future__ import annotations

import math

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


def test_johansen_detects_cointegration_rank():
    y, x = _cointegrated_pair()
    frame = pd.DataFrame({"y": y, "x": x})
    out = co.johansen(frame)
    assert out["rank"] >= 1                        # at least one cointegrating relationship
    assert out["trace_stat"] > out["crit_95"]      # trace stat exceeds the 95% critical value
    assert math.isfinite(out["hedge_ratio"])


def test_johansen_independent_walks_rank_zero():
    rng = np.random.default_rng(3)
    a = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    b = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    out = co.johansen(pd.DataFrame({"a": a, "b": b}))
    assert out["rank"] == 0


def _ou_path(theta: float, mu: float, sigma: float, n: int = 2000, seed: int = 5) -> pd.Series:
    """Simulate a discrete OU process: s_{t+1} = s_t + theta*(mu - s_t) + sigma*eps."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    s[0] = mu
    for t in range(1, n):
        s[t] = s[t - 1] + theta * (mu - s[t - 1]) + sigma * rng.normal()
    return pd.Series(s)


def test_ou_fit_recovers_theta_and_mu():
    spread = _ou_path(theta=0.2, mu=5.0, sigma=0.3)
    theta, mu, sigma_eq = co.ou_fit(spread)
    assert abs(theta - 0.2) < 0.05
    assert abs(mu - 5.0) < 0.3
    assert sigma_eq > 0.0


def test_half_life_formula():
    assert abs(co.half_life(math.log(2)) - 1.0) < 1e-9     # theta = ln2 -> half-life 1 cycle
    assert abs(co.half_life(0.2) - (math.log(2) / 0.2)) < 1e-9


def test_half_life_non_mean_reverting_is_inf():
    assert co.half_life(0.0) == float("inf")
    assert co.half_life(-0.1) == float("inf")

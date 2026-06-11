"""Cointegration math for the pairs sleeve: Engle-Granger ADF + Johansen, OU half-life,
z-score machinery, and FDR/Bonferroni multiple-testing correction across candidate pairs.

Adapted for the market-neutral desk; statsmodels-backed. Pure functions, fail-soft.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger(y: pd.Series, x: pd.Series) -> tuple[float, float, float]:
    """OLS y~x then ADF on the residual spread. Returns (hedge_ratio, adf_pvalue, adf_stat).

    hedge_ratio is the OLS slope (the cointegrating beta: spread = y - hedge_ratio*x).
    A low adf_pvalue (< 0.05) means the residual is stationary -> the pair is cointegrated.
    """
    yv = pd.Series(y).reset_index(drop=True).astype(float)
    xv = pd.Series(x).reset_index(drop=True).astype(float)
    n = min(len(yv), len(xv))
    yv, xv = yv.iloc[:n], xv.iloc[:n]
    design = sm.add_constant(xv.to_numpy())
    model = sm.OLS(yv.to_numpy(), design).fit()
    hedge_ratio = float(model.params[1])
    resid = yv.to_numpy() - hedge_ratio * xv.to_numpy() - float(model.params[0])
    stat, pvalue, *_ = adfuller(resid, autolag="AIC")
    return hedge_ratio, float(pvalue), float(stat)


def johansen(frame: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Johansen trace test on a (T x n) price frame.

    Returns {trace_stat, crit_95, hedge_ratio, rank}: trace_stat/crit_95 for the r=0 hypothesis,
    rank = number of cointegrating vectors at 95%, hedge_ratio normalized from the first
    eigenvector so the first column has coefficient 1 (spread = col0 - hedge_ratio*col1).
    """
    arr = frame.dropna().to_numpy(dtype=float)
    res = coint_johansen(arr, det_order, k_ar_diff)
    trace = res.lr1                                  # trace statistics, descending r
    crit_95 = res.cvt[:, 1]                          # 95% critical values column
    rank = int(sum(1 for i in range(len(trace)) if trace[i] > crit_95[i]))
    vec = res.evec[:, 0]                             # first cointegrating eigenvector
    base = vec[0] if vec[0] != 0 else 1.0
    hedge_ratio = float(-vec[1] / base) if len(vec) > 1 else 0.0
    return {
        "trace_stat": float(trace[0]),
        "crit_95": float(crit_95[0]),
        "hedge_ratio": hedge_ratio,
        "rank": rank,
    }


def ou_fit(spread: pd.Series) -> tuple[float, float, float]:
    """Fit an OU process via AR(1) on the spread. Returns (theta, mu, sigma_eq).

    Discrete AR(1): s_{t+1} = a + b*s_t + eps. Then theta = 1 - b (mean-reversion speed),
    mu = a / (1 - b) (long-run mean), and sigma_eq = std(eps) / sqrt(1 - b^2) (equilibrium sd).
    """
    s = pd.Series(spread).dropna().reset_index(drop=True).astype(float).to_numpy()
    if len(s) < 3:
        return 0.0, float(s.mean()) if len(s) else 0.0, 0.0
    lagged = s[:-1]
    nxt = s[1:]
    if np.ptp(lagged) == 0:
        # Flat/constant spread (stale or pinned prices): the lagged regressor has no
        # variation, so OLS would fit a single parameter. Fail soft like the len<3 branch.
        return 0.0, float(s.mean()), 0.0
    design = sm.add_constant(lagged)
    model = sm.OLS(nxt, design).fit()
    a = float(model.params[0])
    b = float(model.params[1])
    theta = 1.0 - b
    mu = a / (1.0 - b) if abs(1.0 - b) > 1e-12 else float(s.mean())
    resid_sd = float(np.std(model.resid, ddof=2)) if len(model.resid) > 2 else 0.0
    denom = 1.0 - b * b
    sigma_eq = resid_sd / math.sqrt(denom) if denom > 0 else resid_sd
    return theta, mu, sigma_eq


def half_life(theta: float) -> float:
    """OU half-life in cycles = ln(2)/theta. inf if theta <= 0 (non-mean-reverting)."""
    if theta <= 0:
        return float("inf")
    return math.log(2) / theta

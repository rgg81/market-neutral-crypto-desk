"""Cointegration math for the pairs sleeve: Engle-Granger ADF + Johansen, OU half-life,
z-score machinery, and FDR/Bonferroni multiple-testing correction across candidate pairs.

Adapted for the market-neutral desk; statsmodels-backed. Pure functions, fail-soft.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from futures_fund.models import SpreadState


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


def spread_value(y: float, x: float, hedge_ratio: float) -> float:
    """The traded unit: y - hedge_ratio * x."""
    return float(y) - float(hedge_ratio) * float(x)


def zscore(spread_value: float, mu: float, sigma_eq: float) -> float:
    """(spread_value - mu) / sigma_eq; 0.0 if sigma_eq <= 0."""
    if sigma_eq <= 0:
        return 0.0
    return (float(spread_value) - float(mu)) / float(sigma_eq)


def spread_state(z: float, *, entry_z: float = 2.0, exit_z: float = 0.0, stop_z: float = 3.0,
                 prev_state: SpreadState = "flat") -> SpreadState:
    """OU state machine driving the traded spread.

    |z| >= stop_z  -> "stop" (hard exit).
    z >= entry_z   -> "short_spread" (spread is rich; short it for reversion).
    z <= -entry_z  -> "long_spread"  (spread is cheap; long it for reversion).
    |z| <= exit_z  -> "flat" (mean reached; close).
    Otherwise hold prev_state (no-trade hysteresis band between exit and entry).
    """
    az = abs(z)
    if az >= stop_z:
        return "stop"
    if z >= entry_z:
        return "short_spread"
    if z <= -entry_z:
        return "long_spread"
    if az <= exit_z:
        return "flat"
    return prev_state


def fdr_adjust(pvalues: list[float], *, alpha: float = 0.05,
               method: Literal["bh", "bonferroni"] = "bh") -> list[float]:
    """Benjamini-Hochberg (default) or Bonferroni correction across candidate pairs.

    Returns adjusted p-values in the ORIGINAL input order, each clamped to [0, 1]. BH adjustment:
    p_adj(i) = min over k>=rank(i) of (m/k * p_sorted(k)), enforced monotone non-decreasing.
    """
    m = len(pvalues)
    if m == 0:
        return []
    if method == "bonferroni":
        return [min(1.0, p * m) for p in pvalues]
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj_sorted = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):                    # rank = m..1 (largest p first)
        idx = order[rank - 1]
        val = min(1.0, pvalues[idx] * m / rank)
        prev = min(prev, val)
        adj_sorted[rank - 1] = prev
    out = [0.0] * m
    for rank, idx in enumerate(order):
        out[idx] = adj_sorted[rank]
    return out

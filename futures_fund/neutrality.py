from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


class NeutralityConfig(BaseModel):
    capital_usdt: float = 20000.0
    target_gross_usdt: float = 20000.0
    side_budget_usdt: float = 10000.0
    deployment_floor: float = 0.90
    dry_powder_frac: float = 0.10
    per_name_cap: float = 0.25
    cluster_cap: float = 0.40
    dollar_band: float = 0.03
    beta_band: float = 0.05
    drift_band: float = 0.20
    turnover_penalty: float = 0.001
    corr_threshold: float = 0.7
    stress_band_mult: float = 0.5

    @property
    def deploy_target_frac(self) -> float:
        """Per-side deployment target the optimizer scales each side up to: the midpoint of
        the [deployment_floor, 1 - dry_powder_frac] band. With defaults: (0.90 + 0.90)/2 =
        0.90 — i.e. deploy at the floor while still holding the full dry-powder reserve.
        Always lands in [floor, 1 - dry_powder] so both spec-§4 constraints hold by
        construction."""
        lo = self.deployment_floor
        hi = 1.0 - self.dry_powder_frac
        return (lo + hi) / 2.0


def dollar_residual(weights: dict[str, float], notionals: dict[str, float]) -> float:
    """Sum(long$) - Sum(short$) in USDT, using signed per-symbol notionals."""
    longs = sum(n for n in notionals.values() if n > 0.0)
    shorts = sum(-n for n in notionals.values() if n < 0.0)
    return longs - shorts


def beta_residual(weights: dict[str, float], betas: dict[str, float]) -> float:
    """Sum_i w_i * beta_i (equity-normalized beta-dollar exposure)."""
    return sum(w * betas.get(sym, 1.0) for sym, w in weights.items())


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance — stable, avoids unstable inversion. Drops rows with
    any NaN so the estimator sees a complete block."""
    clean = returns.dropna()
    if clean.shape[0] < 2 or clean.shape[1] == 0:
        n = returns.shape[1]
        return np.eye(n)
    return LedoitWolf().fit(clean.to_numpy()).covariance_


def _ivp(cov: np.ndarray, idx: list[int]) -> np.ndarray:
    """Inverse-variance portfolio weights for a sub-cluster (no matrix inversion)."""
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    return ivp / ivp.sum()


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    w = _ivp(cov, idx)
    sub = cov[np.ix_(idx, idx)]
    return float(w @ sub @ w)


def _quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    n = link[-1, 3]
    order = [link[-1, 0], link[-1, 1]]
    while max(order) >= n:
        new: list[int] = []
        for item in order:
            if item < n:
                new.append(item)
            else:
                row = link[item - n]
                new.append(row[0])
                new.append(row[1])
        order = new
    return order


def hrp_weights(cov: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Hierarchical Risk Parity: cluster -> quasi-diagonalize -> recursive bisection.
    No matrix inversion (only diagonal inverse-variance). Weights sum to 1.0."""
    n = len(labels)
    if n == 1:
        return {labels[0]: 1.0}
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    outer[outer == 0.0] = 1e-12
    corr = np.clip(cov / outer, -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = _quasi_diag(link)
    weights = np.ones(n)
    clusters = [sort_ix]
    while clusters:
        clusters = [
            c[j:k]
            for c in clusters
            for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            left = clusters[i]
            right = clusters[i + 1]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= 1.0 - alpha
    weights /= weights.sum()
    return {labels[i]: float(weights[i]) for i in range(n)}

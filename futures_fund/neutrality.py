from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from futures_fund.contracts import (
    CoinGeometry,
    SleeveSignal,
    SleeveTilt,
    TargetWeights,
    WeightLeg,
)
from futures_fund.models import RegimeState, SleeveName

# conviction_tilt/apply_conviction_tilts live canonically in sleeves/sentiment.py (§2.9); they are
# re-exported here because the optimizer below calls apply_conviction_tilts and downstream code +
# tests import these from neutrality. The redundant `as` aliases mark them as explicit re-exports
# so ruff preserves the public-API re-export.
from futures_fund.sleeves.sentiment import apply_conviction_tilts as apply_conviction_tilts
from futures_fund.sleeves.sentiment import conviction_tilt as conviction_tilt


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


def size_btc_hedge(
    weights: dict[str, float],
    betas: dict[str, float],
    *,
    equity: float,
    side_budget: float,
) -> float:
    """Signed BTC-perp hedge notional that absorbs the ALPHA legs' residual portfolio beta.
    BTC has beta 1.0, so the hedge weight equals the NEGATIVE of the residual beta of the
    legs passed in; converted to USDT via equity and clamped to fit INSIDE one per-side
    budget (never added on top). Call this on the alpha legs BEFORE project_neutral so the
    hedge is a real degree of freedom (the reviewer re-derives it the same way)."""
    resid_beta = beta_residual(weights, betas)
    hedge_weight = -resid_beta  # BTC beta == 1.0
    hedge_notional = hedge_weight * equity
    if hedge_notional > side_budget:
        hedge_notional = side_budget
    elif hedge_notional < -side_budget:
        hedge_notional = -side_budget
    return hedge_notional


def project_neutral(
    weights: dict[str, float],
    betas: dict[str, float],
    *,
    dollar_band: float,
    beta_band: float,
) -> dict[str, float]:
    """Least-norm projection of a signed weight vector onto the dollar+beta-neutral
    constraint set: removes the components of the vector in the span of the dollar direction
    (all-ones) and the beta direction so Sum(w_i) ~ 0 and Sum(w_i*beta_i) ~ 0. The result
    lives in the (n - 2)-dimensional null space of the two constraints, so a NON-TRIVIAL
    neutral book requires >= 3 distinct active names (with n <= 2 the only neutral point is
    0 — see the Task 11 degenerate-case note). Sentiment tilts are applied BEFORE this call,
    so sentiment cannot break neutrality (residuals are recomputed after). `dollar_band` /
    `beta_band` are accepted for signature stability with the reviewer's re-derivation and to
    document the bands this projection targets; the exact projection drives residuals to ~0,
    well inside the bands."""
    syms = list(weights.keys())
    if not syms:
        return {}
    w = np.array([weights[s] for s in syms], dtype=float)
    b = np.array([betas.get(s, 1.0) for s in syms], dtype=float)
    ones = np.ones(len(syms))

    # Constraint matrix C (2 x n): row0 = dollar (ones), row1 = beta.
    c = np.vstack([ones, b])
    residual = c @ w  # [dollar_resid, beta_resid]
    gram = c @ c.T  # 2 x 2
    try:
        correction = c.T @ np.linalg.solve(gram, residual)
    except np.linalg.LinAlgError:
        correction = c.T @ (np.linalg.pinv(gram) @ residual)
    w_proj = w - correction
    return {syms[i]: float(w_proj[i]) for i in range(len(syms))}


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance — stable, avoids unstable inversion. Drops rows with
    any NaN so the estimator sees a complete block."""
    clean = returns.dropna()
    if clean.shape[0] < 2 or clean.shape[1] == 0:
        n = returns.shape[1]
        return np.eye(n)
    return LedoitWolf().fit(clean.to_numpy()).covariance_


def cov_to_corr(cov: np.ndarray, labels: list[str]) -> dict[tuple[str, str], float]:
    """Pairwise correlation map {(a, b): rho} from a covariance matrix, for the cluster cap's
    union-find. Derived from the SAME Ledoit-Wolf covariance HRP uses, so the 'correlated-as-
    one' heat cap (spec §3/§9) runs on a real cross-correlation snapshot inside optimize_book
    rather than the empty map that made it inert."""
    std = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    outer = np.outer(std, std)
    outer[outer == 0.0] = 1e-18
    corr = np.clip(cov / outer, -1.0, 1.0)
    out: dict[tuple[str, str], float] = {}
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            out[(labels[i], labels[j])] = float(corr[i, j])
    return out


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


def risk_parity_budgets(
    sleeves: list[SleeveSignal], *, cov: np.ndarray | None = None
) -> dict[SleeveName, float]:
    """Risk-parity (or inverse-vol) budget across the sleeves; writes the result back onto
    each SleeveSignal.risk_budget_frac and returns the {sleeve: frac} map. Sums to 1.0.
    With no covariance supplied, falls back to an equal (inverse-unit-vol) split."""
    if not sleeves:
        return {}
    if cov is None or cov.shape[0] != len(sleeves):
        raw = np.ones(len(sleeves))
    else:
        var = np.diag(cov)
        # A zero/degenerate-variance sleeve would get inv-vol ~ 1/sqrt(1e-12) ~ 1e6 and absorb
        # ~100% of the budget — the OPPOSITE of risk balance (a dead sleeve cannot dominate).
        # Treat it as degenerate and route to a sensible fallback: if ANY sleeve has real
        # variance, drop the degenerate ones to 0 budget; if ALL are degenerate, equal-share.
        var_floor = 1e-10
        real = var > var_floor
        if real.any():
            raw = np.where(real, 1.0 / np.sqrt(np.clip(var, var_floor, None)), 0.0)
        else:
            raw = np.ones(len(sleeves))
    fracs = raw / raw.sum()
    out: dict[SleeveName, float] = {}
    for s, f in zip(sleeves, fracs, strict=True):
        s.risk_budget_frac = float(f)
        out[s.sleeve] = float(f)
    return out


def merge_sleeves(
    sleeves: list[SleeveSignal], geometries: list[CoinGeometry]
) -> dict[str, float]:
    """Combine already-risk-budgeted sleeve tilts into one signed pre-projection weight
    vector. Each tilt's signed target_weight is scaled by its sleeve risk_budget_frac and
    summed per symbol."""
    known = {g.symbol for g in geometries}
    merged: dict[str, float] = {}
    for s in sleeves:
        for tilt in s.tilts:
            if tilt.symbol not in known:
                continue
            merged[tilt.symbol] = merged.get(tilt.symbol, 0.0) + (
                tilt.target_weight * s.risk_budget_frac
            )
    return merged


def apply_hrp_weights(
    weights: dict[str, float], hrp: dict[str, float]
) -> dict[str, float]:
    """Reshape a signed weight vector so each side's per-name split follows the HRP weights,
    WITHOUT changing any sign or either side's total gross. This is how Ledoit-Wolf -> HRP
    (Task 8) actually shapes the book (spec §8): for each side, redistribute that side's gross
    across its names in proportion to the names' HRP weights (re-normalized within the side).
    Returns `weights` unchanged if `hrp` is empty (HRP unavailable / single name)."""
    if not hrp:
        return dict(weights)
    longs = {s: w for s, w in weights.items() if w > 0.0}
    shorts = {s: w for s, w in weights.items() if w < 0.0}
    out: dict[str, float] = {s: w for s, w in weights.items() if w == 0.0}
    for side in (longs, shorts):
        if not side:
            continue
        side_gross = sum(abs(w) for w in side.values())
        sign = 1.0 if next(iter(side.values())) > 0.0 else -1.0
        hrp_side = {s: hrp.get(s, 0.0) for s in side}
        hrp_sum = sum(hrp_side.values())
        if hrp_sum <= 0.0:
            # HRP has no info for this side's names: keep the original split.
            out.update(side)
            continue
        for s in side:
            out[s] = sign * side_gross * (hrp_side[s] / hrp_sum)
    return out


def apply_per_name_cap(
    weights: dict[str, float], *, per_name_cap: float
) -> dict[str, float]:
    """Clamp each symbol's weight magnitude to `per_name_cap`, preserving sign."""
    out: dict[str, float] = {}
    for sym, w in weights.items():
        if abs(w) > per_name_cap:
            out[sym] = per_name_cap if w > 0 else -per_name_cap
        else:
            out[sym] = w
    return out


def _corr_lookup(corr: Mapping[tuple[str, str], float], a: str, b: str) -> float:
    if (a, b) in corr:
        return corr[(a, b)]
    if (b, a) in corr:
        return corr[(b, a)]
    return 0.0


def cluster_roots(
    weights: dict[str, float],
    *,
    corr: Mapping[tuple[str, str], float],
    threshold: float = 0.7,
) -> dict[str, str]:
    """Map each symbol to its cluster root (a representative symbol). Union-find groups
    SAME-SIDE symbols whose pairwise correlation >= threshold (a long and short in correlated
    names are a natural hedge and are NOT clustered). Shared by `apply_cluster_cap` and the
    final post-projection cluster-cap audit so both see identical clustering."""
    syms = list(weights.keys())
    n = len(syms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    def side(w: float) -> int:
        return 1 if w > 0 else (-1 if w < 0 else 0)

    for i in range(n):
        for j in range(i + 1, n):
            if side(weights[syms[i]]) != 0 and side(weights[syms[i]]) == side(weights[syms[j]]):
                if _corr_lookup(corr, syms[i], syms[j]) >= threshold:
                    union(i, j)
    return {syms[idx]: syms[find(idx)] for idx in range(n)}


def apply_cluster_cap(
    weights: dict[str, float],
    *,
    corr: Mapping[tuple[str, str], float],
    cluster_cap: float,
    threshold: float = 0.7,
) -> dict[str, float]:
    """'Correlated-as-one' heat cap. Union-find groups SAME-SIDE symbols whose pairwise
    correlation >= threshold (a long and short in correlated names are a natural hedge and
    are NOT clustered). Scales down each cluster so its combined |weight| <= cluster_cap.
    Adapted from crypto-trade-claude-code-weekly portfolio_risk.cluster_heat."""
    roots = cluster_roots(weights, corr=corr, threshold=threshold)
    cluster_mag: dict[str, float] = {}
    for sym, root in roots.items():
        cluster_mag[root] = cluster_mag.get(root, 0.0) + abs(weights[sym])

    out: dict[str, float] = {}
    for sym, root in roots.items():
        mag = cluster_mag[root]
        if mag > cluster_cap and mag > 0.0:
            out[sym] = weights[sym] * (cluster_cap / mag)
        else:
            out[sym] = weights[sym]
    return out


def _apply_turnover_band(
    weights: dict[str, float],
    prior_weights: dict[str, float],
    *,
    drift_band: float,
    turnover_penalty: float,
) -> tuple[dict[str, float], float]:
    """No-trade drift band + L1 turnover penalty, applied BEFORE the final projection so the
    projection has the last say on neutrality. A symbol PRESENT in the prior whose target is
    within `drift_band` of its prior weight keeps the prior weight (no churn); otherwise it
    moves to target, shrunk toward prior by `turnover_penalty` (L1 damping). A symbol ABSENT
    from the prior (prior == 0) is ALWAYS-TRADE: it is never snapped to 0 by the band, so a
    fresh sub-drift-band leg survives the rebalance. Returns (adjusted_weights, l1_turnover)."""
    out: dict[str, float] = {}
    for sym, target in weights.items():
        if sym not in prior_weights:
            # fresh name: always trade (do not let the no-trade band delete it)
            out[sym] = target
            continue
        prior = prior_weights[sym]
        denom = abs(prior) if abs(prior) > 1e-12 else 1.0
        if abs(target - prior) / denom <= drift_band:
            out[sym] = prior
        else:
            out[sym] = target - turnover_penalty * (target - prior)
    l1 = sum(abs(out[s] - prior_weights.get(s, 0.0)) for s in out)
    return out, l1


def _scale_to_deploy_target(
    weights: dict[str, float], hedge_notional: float, *, equity: float,
    side_budget: float, deploy_target_frac: float,
) -> tuple[dict[str, float], float]:
    """Scale the projected (dollar+beta-neutral) book by a SINGLE positive scalar so each side's
    gross equals `deploy_target_frac * side_budget`. A positive scalar preserves BOTH
    neutralities exactly (Sum(k*w)=0, Sum(k*w*beta)=0) and scales each side's gross equally, so
    this restores the deployment floor WITHOUT re-breaking neutrality. The hedge notional is
    scaled by the same factor (it is part of the neutral vector). Returns
    (scaled_weights, scaled_hedge_notional).

    A dollar-neutral book has long_gross == short_gross in exact arithmetic; the only
    difference between the two side sums is floating-point summation noise. We scale on the
    SMALLER positive side so BOTH sides land at-or-above `deploy_target_frac` (never a hair
    below the floor from FP rounding); the larger side lands above target by at most that FP
    noise, so it stays inside the [floor, 1-dry_powder] band. The `max` fallback covers a
    degenerate one-sided book (a side is empty) so we still emit a book (flagged infeasible
    upstream)."""
    long_gross = sum(w for w in weights.values() if w > 0.0) * equity
    short_gross = -sum(w for w in weights.values() if w < 0.0) * equity
    # include the hedge in the side it sits on
    if hedge_notional > 0.0:
        long_gross += hedge_notional
    elif hedge_notional < 0.0:
        short_gross += -hedge_notional
    # bind on the smaller non-empty side so neither side rounds below the deployment floor;
    # fall back to the larger side if one side is empty (degenerate one-sided book).
    positive_sides = [g for g in (long_gross, short_gross) if g > 0.0]
    if not positive_sides:
        return dict(weights), hedge_notional
    binding = min(positive_sides) if len(positive_sides) == 2 else max(positive_sides)
    target_side_usd = deploy_target_frac * side_budget
    k = target_side_usd / binding
    return {s: w * k for s, w in weights.items()}, hedge_notional * k


def _cap_violations(
    weights: dict[str, float],
    *,
    corr: Mapping[tuple[str, str], float],
    per_name_cap: float,
    cluster_cap: float,
    corr_threshold: float,
    tol: float = 1e-6,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """Find caps breached by the FINAL book, in signed equity-weight units — the SAME units the
    pre-projection `apply_per_name_cap` / `apply_cluster_cap` clamp: the per-name cap is enforced
    as a fraction of EQUITY (notional = w * equity, so an at-cap leg is per_name_cap * equity).
    This does NOT claim design-doc compliance: with equity == 2x a side's budget, an at-cap leg
    (0.25 * equity) is ~50% of a side's budget, i.e. ~2x LOOSER than design-doc §4's 'fraction of a
    side' phrasing. That divergence is a known calibration item, intentionally left to the
    contract-pinned `per_name_cap = 0.25`: the equity-fraction convention predates this module and
    the canonical interface contract (and the small synthetic test fixtures) pin 0.25 to it, so
    this checker enforces exactly that convention rather than silently re-deriving a stricter one.
    Returns (per_name_overages, cluster_over): a per-name
    overage is (symbol, signed_weight_at_cap); cluster_over maps each over-cap >=2-member cluster
    root -> its combined |weight|. The dollar+beta-neutral, deploy-scaled book re-concentrates
    weight onto the high-beta absorbers, so this is re-checked AFTER projection+scale, not just
    before (a pre-projection-only cap is silently breached by the emitted book)."""
    per_name: list[tuple[str, float]] = []
    for sym, w in weights.items():
        if abs(w) > per_name_cap + tol:
            per_name.append((sym, per_name_cap if w > 0 else -per_name_cap))
    roots = cluster_roots(weights, corr=corr, threshold=corr_threshold)
    cluster_mag: dict[str, float] = {}
    members_of: dict[str, int] = {}
    for sym, root in roots.items():
        cluster_mag[root] = cluster_mag.get(root, 0.0) + abs(weights[sym])
        members_of[root] = members_of.get(root, 0) + 1
    cluster_over: dict[str, float] = {}
    for root, mag in cluster_mag.items():
        if members_of[root] >= 2 and mag > cluster_cap + tol:
            cluster_over[root] = mag
    return per_name, cluster_over


def _enforce_caps_neutral(
    alpha_weights: dict[str, float],
    betas: dict[str, float],
    hedge_notional: float,
    *,
    corr: Mapping[tuple[str, str], float],
    equity: float,
    side_budget: float,
    deploy_target_frac: float,
    per_name_cap: float,
    cluster_cap: float,
    corr_threshold: float,
) -> tuple[dict[str, float], float, bool]:
    """Enforce the per-name and cluster caps on the FINAL (post-projection, post-scale) book
    WITHOUT re-breaking neutrality OR the deployment target. Projection + the positive-scalar
    deploy scale re-concentrate weight onto the high-residual-beta absorbers, so caps applied
    only pre-projection (step 4) can be silently breached by the emitted book.

    A single positive re-scale after a clamp just re-breaks the cap (cap-then-scale oscillates),
    so we solve the capped book directly as one bounded least-distance projection over BOTH the
    alpha weights AND the BTC hedge magnitude: find the book closest to the deploy-scaled seed
    subject to (a) dollar+beta neutrality (the hedge is a beta-1.0 carrier and a real DOF, so it
    re-sizes to whatever the capped alpha legs leave residual), (b) each side's gross equal to its
    deploy target (floor + dry-powder band still hold), (c) sign-preserving per-name box
    |w_i| <= per_name_cap, (d) each correlated same-side cluster's combined |w| <= cluster_cap, and
    (e) the hedge magnitude inside one side's budget (spec §5). Returns
    (alpha_weights, hedge_notional, caps_ok); caps_ok=False (=> feasible=False upstream) when no
    cap-respecting neutral fully-deployed book exists for this universe (e.g. too few names per
    side: a side needs >= ceil(deploy_target*side_budget/equity / per_name_cap) names) — surfaced
    honestly rather than silently breaching a spec invariant. The dedicated BTC hedge is the
    benchmark hedge (spec §5), so it is NOT subject to the alpha per-name/cluster caps.

    Returns the input unchanged with caps_ok=True when the seed already respects the caps."""
    per_name0, cluster0 = _cap_violations(
        alpha_weights, corr=corr, per_name_cap=per_name_cap, cluster_cap=cluster_cap,
        corr_threshold=corr_threshold,
    )
    if not per_name0 and not cluster0:
        return dict(alpha_weights), hedge_notional, True
    if equity <= 0.0 or side_budget <= 0.0:
        return dict(alpha_weights), hedge_notional, False

    syms = list(alpha_weights.keys())
    seed = np.array([alpha_weights[s] for s in syms], dtype=float)
    signs = np.sign(seed)
    b = np.array([betas.get(s, 1.0) for s in syms], dtype=float)
    seed_hedge_w = hedge_notional / equity
    # The hedge keeps the sign size_btc_hedge picked (which side absorbs the residual beta) but its
    # MAGNITUDE re-sizes inside the solve. A ZERO seed hedge means the alpha legs already net
    # beta-neutral (e.g. all betas == 1.0): there is no residual for a hedge to carry, so we must
    # NOT inject a phantom hedge DOF. With a free hedge column whose dollar and beta coefficients
    # are both `hs`, the dollar and beta equality rows would be identical when every beta == 1.0 —
    # a rank deficiency that makes SLSQP fail with 'Singular matrix C in LSQ subproblem' and falsely
    # report a plainly-feasible book infeasible. When seed_hedge_w == 0 we pin the hedge magnitude
    # to 0 (bounds (0, 0) below) and zero its constraint coefficients; the equality-row dedup below
    # then drops any remaining redundancy so the solve runs full-rank on the real alpha DOFs.
    has_hedge = seed_hedge_w != 0.0
    hs = 1.0 if seed_hedge_w > 0 else -1.0  # only meaningful when has_hedge
    side_gross = deploy_target_frac * side_budget / equity  # equity-weight gross per side
    hedge_max = side_budget / equity                         # hedge inside one side's budget (§5)

    n = len(syms)
    long_idx = [i for i in range(n) if signs[i] > 0]
    short_idx = [i for i in range(n) if signs[i] < 0]

    # decision vector x = [w_0..w_{n-1}, hm]; hm >= 0 is the hedge magnitude (signed by hs).
    seed_x = np.append(seed, abs(seed_hedge_w))
    bounds: list[tuple[float, float]] = []
    for i in range(n):
        if signs[i] > 0:
            bounds.append((0.0, per_name_cap))
        elif signs[i] < 0:
            bounds.append((-per_name_cap, 0.0))
        else:
            bounds.append((0.0, 0.0))
    # hedge magnitude bound: 0-pinned when there is no seed hedge (no phantom DOF)
    bounds.append((0.0, hedge_max) if has_hedge else (0.0, 0.0))

    roots = cluster_roots(alpha_weights, corr=corr, threshold=corr_threshold)
    clusters: dict[str, list[int]] = {}
    for i, s in enumerate(syms):
        clusters.setdefault(roots[s], []).append(i)

    # Every equality below is LINEAR in x (x = [alpha weights..., hedge magnitude]). We assemble
    # each as a row (A·x == rhs), then drop linearly-DEPENDENT rows before handing them to SLSQP.
    # The hedge column (index n) only carries `hs` when there actually is a seed hedge; with no
    # hedge the column is 0 so it never couples the rows. Redundant rows arise routinely here —
    # e.g. all betas == 1.0 makes the beta row identical to the dollar row, and the two per-side
    # deployment rows already imply dollar-neutrality — and feeding SLSQP a rank-deficient equality
    # set yields a 'Singular matrix C in LSQ subproblem' failure that falsely reports a feasible
    # book infeasible. Deduplicating to a maximal independent set keeps the SAME feasible region
    # while giving SLSQP a full-rank Jacobian.
    hc = hs if has_hedge else 0.0
    eq_rows: list[tuple[np.ndarray, float]] = []
    dollar_row = np.zeros(n + 1)
    dollar_row[:n] = 1.0
    dollar_row[n] = hc
    eq_rows.append((dollar_row, 0.0))                                    # dollar neutrality
    beta_row = np.zeros(n + 1)
    beta_row[:n] = b
    beta_row[n] = hc
    eq_rows.append((beta_row, 0.0))                                   # beta neutrality (hedge β=1)
    hedge_long = hs if (has_hedge and hs > 0) else 0.0
    hedge_short = -hs if (has_hedge and hs < 0) else 0.0
    if long_idx:
        row = np.zeros(n + 1)
        row[long_idx] = 1.0
        row[n] = hedge_long
        eq_rows.append((row, side_gross))                               # long-side deployment
    if short_idx:
        row = np.zeros(n + 1)
        row[short_idx] = -1.0
        row[n] = hedge_short
        eq_rows.append((row, side_gross))                              # short-side deployment

    cons: list[dict] = []
    basis: list[np.ndarray] = []  # independent equality-coefficient rows kept so far
    for coeff, rhs in eq_rows:
        if basis:
            stacked = np.vstack([*basis, coeff])
            # a row is redundant iff appending it does not raise the rank of the basis it joins.
            if np.linalg.matrix_rank(stacked, tol=1e-9) <= len(basis):
                continue
        basis.append(coeff)
        cons.append({"type": "eq", "fun": lambda x, a=coeff, r=rhs: float(np.dot(a, x) - r)})
    for members in clusters.values():
        if len(members) >= 2:
            cons.append({"type": "ineq", "fun": lambda x, ix=members:
                         float(cluster_cap - np.sum(np.abs(x[ix]))) })

    def objective(x: np.ndarray) -> float:
        d = x - seed_x
        return float(np.dot(d, d))

    res = minimize(objective, seed_x, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-12})
    x_sol = res.x
    out = {syms[i]: float(x_sol[i]) for i in range(n)}
    hedge_out = hs * float(x_sol[n]) * equity
    hedge_w_out = hs * float(x_sol[n])
    per_name1, cluster1 = _cap_violations(
        out, corr=corr, per_name_cap=per_name_cap, cluster_cap=cluster_cap,
        corr_threshold=corr_threshold,
    )
    # verify the solver actually hit neutrality + deployment (SLSQP can return on a non-feasible
    # point when the constraint set is empty for this universe).
    dollar_ok = abs(sum(out.values()) + hedge_w_out) <= 1e-5
    beta_ok = abs(sum(out[s] * betas.get(s, 1.0) for s in out) + hedge_w_out) <= 1e-5
    long_g = sum(v for v in out.values() if v > 0) + (hedge_w_out if hedge_w_out > 0 else 0.0)
    short_g = -sum(v for v in out.values() if v < 0) + (-hedge_w_out if hedge_w_out < 0 else 0.0)
    deploy_ok = (
        (not long_idx or abs(long_g - side_gross) <= 1e-4)
        and (not short_idx or abs(short_g - side_gross) <= 1e-4)
    )
    caps_ok = (
        bool(res.success) and not per_name1 and not cluster1
        and dollar_ok and beta_ok and deploy_ok
    )
    if not caps_ok:
        # No cap-respecting, fully-deployed, neutral book exists for this (tiny / adverse-beta)
        # universe. NEVER sacrifice neutrality or the deployment floor to chase the cap: return
        # the ORIGINAL neutral, deployed seed UNCHANGED and flag caps_ok=False so optimize_book
        # sets feasible=False (the spec invariant is surfaced, not silently breached).
        return dict(alpha_weights), hedge_notional, False
    return out, hedge_out, True


def optimize_book(
    sleeves: list[SleeveSignal],
    geometries: list[CoinGeometry],
    *,
    equity: float,
    prior_legs: list[WeightLeg] | None,
    cfg: NeutralityConfig,
    regime: RegimeState | None = None,
    returns: pd.DataFrame | None = None,
) -> TargetWeights:
    """THE solver. Merge sleeves -> sentiment tilts -> HRP-shape (Ledoit-Wolf -> HRP) ->
    per-name & cluster caps -> turnover/no-trade band (vs prior) -> size BTC hedge on the
    alpha legs' residual beta (real DOF) and append it -> project alpha+hedge onto the
    dollar+beta-neutral set -> scale the neutral book to the per-side deployment target
    (single positive scalar, preserves neutrality) -> assemble TargetWeights with residuals
    + per-side deployment. Stress-tightens bands under a correlation-spike regime. Sets
    feasible=False (never silently un-neutral / under-deployed) if the bands or the floor
    cannot be met. `returns` (optional) is the per-symbol return frame used to build the
    Ledoit-Wolf covariance for HRP shaping; without it the merged split is used."""
    btc = "BTC/USDT:USDT"
    # {symbol: pair_id} from the pairs sleeve, for pair-level PnL attribution (§1.5). Each
    # pairs-sleeve SleeveTilt carries the pair_id it belongs to; we stamp it onto the emitted
    # alpha-leg WeightLeg so the spread is attributable end-to-end. If a symbol is later
    # absorbed/sized only as a hedge it keeps pair_id=None (the dedicated BTC hedge is never a
    # pair leg). Backward-compatible: with no pairs sleeve this map is empty and pair_id stays None.
    pair_id_by_symbol = _pair_ids_from_sleeves(sleeves)
    betas = {g.symbol: g.beta_btc for g in geometries}
    # BTC's beta TO ITSELF is 1.0 by construction (spec §5: BTC is the benchmark / hedge
    # instrument). `size_btc_hedge` sizes the hedge with beta_BTC == 1.0 and the hedge is
    # carried as its own leg with beta == 1.0; the projection and the final residual must use
    # the SAME BTC self-beta, or the alpha-BTC/hedge split leaves a residual w_hedge*(1 - beta)
    # that breaks beta-neutrality. Normalize it once here so every downstream step agrees.
    betas[btc] = 1.0

    # Stress-tighten bands under a correlation-spike regime.
    band_mult = 1.0
    if regime is not None and regime.quadrant in (
        "high_vol_trend", "high_vol_range", "transition"
    ):
        band_mult = cfg.stress_band_mult
    dollar_band = cfg.dollar_band * band_mult
    beta_band = cfg.beta_band * band_mult

    # 1. assign risk budgets + merge sleeve tilts into one signed vector
    risk_parity_budgets(sleeves)
    merged = merge_sleeves(sleeves, geometries)

    # 2. apply sentiment conviction tilts BEFORE projection (sign-preserving, capped)
    tilted_tilts = apply_conviction_tilts(
        [SleeveTilt(symbol=s, direction="long" if w >= 0 else "short", target_weight=w)
         for s, w in merged.items()],
        geometries,
    )
    weights = {t.symbol: t.target_weight for t in tilted_tilts}

    # 3. HRP shaping: Ledoit-Wolf shrunk covariance -> HRP -> reshape per-name split per side.
    #    The SAME covariance also yields the cross-correlation snapshot the cluster cap needs
    #    (step 4 + the final cap audit), so 'correlated-as-one' is functional in the real solver
    #    path rather than degenerating into a redundant per-name clamp on an empty corr map.
    corr: dict[tuple[str, str], float] = {}
    if returns is not None and not returns.empty:
        labels = [s for s in weights if s in returns.columns]
        if len(labels) >= 2:
            cov = ledoit_wolf_cov(returns[labels])
            hrp = hrp_weights(cov, labels)
            weights = apply_hrp_weights(weights, hrp)
            corr = cov_to_corr(cov, labels)

    # 4. per-name + cluster caps (first pass, pre-projection)
    weights = apply_per_name_cap(weights, per_name_cap=cfg.per_name_cap)
    weights = apply_cluster_cap(
        weights, corr=corr, cluster_cap=cfg.cluster_cap, threshold=cfg.corr_threshold
    )

    # 5. turnover / no-trade band vs the prior book — BEFORE projection (fresh names always-trade)
    prior_weights = {leg.symbol: leg.weight for leg in (prior_legs or [])
                     if leg.sleeve != "hedge"}
    turnover_l1 = 0.0
    if prior_weights:
        weights, turnover_l1 = _apply_turnover_band(
            weights, prior_weights,
            drift_band=cfg.drift_band, turnover_penalty=cfg.turnover_penalty,
        )

    # 6. size the BTC hedge on the ALPHA legs' residual beta (real DOF) and append it, then
    #    project the alpha+hedge vector onto the dollar+beta-neutral set. With the hedge
    #    appended there are >= 3 names, so projection yields a non-trivial neutral book.
    hedge_notional = size_btc_hedge(
        weights, betas, equity=equity, side_budget=cfg.side_budget_usdt
    )
    proj_in = dict(weights)
    proj_betas = dict(betas)
    if abs(hedge_notional) > 1e-9:
        proj_in[btc] = proj_in.get(btc, 0.0) + hedge_notional / equity
    projected = project_neutral(proj_in, proj_betas, dollar_band=dollar_band, beta_band=beta_band)
    # split the projected BTC weight back into (alpha BTC leg, hedge): the hedge keeps the
    # residual-beta share, the rest stays an alpha BTC leg. We carry the hedge as its own leg.
    hedge_weight = hedge_notional / equity if equity > 0 else 0.0
    alpha_weights = dict(projected)
    if abs(hedge_notional) > 1e-9:
        alpha_weights[btc] = projected.get(btc, 0.0) - hedge_weight
        if abs(alpha_weights[btc]) < 1e-12:
            alpha_weights.pop(btc, None)

    # 7. scale the neutral book up to the per-side deployment target (preserves neutrality)
    alpha_weights, hedge_notional = _scale_to_deploy_target(
        alpha_weights, hedge_notional, equity=equity,
        side_budget=cfg.side_budget_usdt, deploy_target_frac=cfg.deploy_target_frac,
    )

    # 7b. enforce the per-name & cluster caps on the FINAL (post-projection, post-scale) book.
    #     Projection + the deploy scale re-concentrate weight onto the high-beta absorbers, so the
    #     pre-projection caps (step 4) can be breached by the emitted book; this loop clamps the
    #     overage and re-projects/re-scales the free legs to a neutral fixed point. caps_ok=False
    #     => feasible=False (the caps and the deployment floor cannot both hold here).
    alpha_weights, hedge_notional, caps_ok = _enforce_caps_neutral(
        alpha_weights, betas, hedge_notional,
        corr=corr, equity=equity, side_budget=cfg.side_budget_usdt,
        deploy_target_frac=cfg.deploy_target_frac, per_name_cap=cfg.per_name_cap,
        cluster_cap=cfg.cluster_cap, corr_threshold=cfg.corr_threshold,
    )

    # 8. assemble legs (alpha legs) + the hedge leg
    legs: list[WeightLeg] = []
    notionals: dict[str, float] = {}
    full_weights: dict[str, float] = {}
    full_betas: dict[str, float] = {}
    for sym, w in alpha_weights.items():
        if abs(w) < 1e-9:
            continue
        notional = w * equity
        notionals[sym] = notional
        full_weights[sym] = w
        full_betas[sym] = betas.get(sym, 1.0)
        legs.append(WeightLeg(
            symbol=sym,
            direction="long" if w > 0 else "short",
            weight=w,
            target_notional=notional,
            beta_btc=betas.get(sym, 1.0),
            sleeve=_dominant_sleeve(sym, sleeves),
            pair_id=pair_id_by_symbol.get(sym),
        ))
    if abs(hedge_notional) > 1.0:
        hedge_w = hedge_notional / equity
        notionals["__hedge__"] = hedge_notional
        full_weights["__hedge__"] = hedge_w
        full_betas["__hedge__"] = 1.0
        legs.append(WeightLeg(
            symbol=btc,
            direction="long" if hedge_notional > 0 else "short",
            weight=hedge_w,
            target_notional=hedge_notional,
            beta_btc=1.0,
            sleeve="hedge",
        ))

    # residuals + per-side deployment (include hedge leg in dollar/beta sums)
    d_resid = dollar_residual(full_weights, notionals)
    d_resid_frac = abs(d_resid) / cfg.side_budget_usdt if cfg.side_budget_usdt > 0 else 0.0
    b_resid = beta_residual(full_weights, full_betas)
    gross_long = sum(n for n in notionals.values() if n > 0)
    gross_short = sum(-n for n in notionals.values() if n < 0)
    deploy_long = gross_long / cfg.side_budget_usdt if cfg.side_budget_usdt > 0 else 0.0
    deploy_short = gross_short / cfg.side_budget_usdt if cfg.side_budget_usdt > 0 else 0.0

    # Verify the per-name & cluster caps on the EMITTED book (alpha legs only; the dedicated
    # BTC hedge is the benchmark hedge, capped inside one side's budget by size_btc_hedge).
    # The per-name cap is enforced as a fraction of EQUITY (|w| <= per_name_cap, notional =
    # w * equity) — the same convention `apply_per_name_cap` and `_cap_violations` use. This is
    # NOT design-doc compliance: with equity == 2x a side's budget it is ~2x LOOSER than design-doc
    # §4's 'fraction of a side' phrasing, a known calibration item intentionally left to the
    # contract-pinned `per_name_cap = 0.25` (see `_cap_violations` for the full rationale). Never
    # silently breach the invariant that IS enforced here.
    alpha_final = {s: w for s, w in full_weights.items() if s != "__hedge__"}
    per_name_over, cluster_over = _cap_violations(
        alpha_final, corr=corr, per_name_cap=cfg.per_name_cap, cluster_cap=cfg.cluster_cap,
        corr_threshold=cfg.corr_threshold,
    )
    caps_respected = caps_ok and not per_name_over and not cluster_over

    feasible = (
        d_resid_frac <= dollar_band + 1e-6
        and abs(b_resid) <= beta_band + 1e-6
        and deploy_long >= cfg.deployment_floor - 1e-6
        and deploy_short >= cfg.deployment_floor - 1e-6
        and deploy_long <= (1.0 - cfg.dry_powder_frac) + 1e-6
        and deploy_short <= (1.0 - cfg.dry_powder_frac) + 1e-6
        and caps_respected
    )
    notes: list[str] = []
    if not feasible:
        if not caps_respected:
            notes.append(
                "constraint set infeasible: per-name or cluster cap breached on final book"
            )
        else:
            notes.append("constraint set infeasible: residual or deployment-floor breach")

    return TargetWeights(
        legs=legs,
        btc_hedge_notional=hedge_notional,
        dollar_residual=d_resid,
        dollar_residual_frac=d_resid_frac,
        beta_residual=b_resid,
        gross_long=gross_long,
        gross_short=gross_short,
        deploy_long_frac=deploy_long,
        deploy_short_frac=deploy_short,
        gross_notional=gross_long + gross_short,
        turnover_l1=turnover_l1,
        feasible=feasible,
        notes=notes,
        as_of_ts=datetime.now(UTC),
    )


def _pair_ids_from_sleeves(sleeves: list[SleeveSignal]) -> dict[str, str]:
    """{symbol: pair_id} for every symbol that originated from the pairs sleeve (§1.5).

    Derived from the pairs-sleeve SleeveTilt.pair_id so pair identity survives the
    merge_sleeves collapse to dict[str, float]. Used to stamp WeightLeg.pair_id for pair-level
    PnL attribution. Empty (so pair_id stays None) when there is no pairs sleeve."""
    out: dict[str, str] = {}
    for s in sleeves:
        if s.sleeve != "pairs":
            continue
        for t in s.tilts:
            if t.pair_id is not None:
                out[t.symbol] = t.pair_id
    return out


def _dominant_sleeve(symbol: str, sleeves: list[SleeveSignal]) -> SleeveName:
    """The sleeve contributing the largest |budgeted tilt| to this symbol (source attribution)."""
    best: tuple[float, SleeveName] = (-1.0, "factor")
    for s in sleeves:
        for t in s.tilts:
            if t.symbol == symbol:
                contrib = abs(t.target_weight) * s.risk_budget_frac
                if contrib > best[0]:
                    best = (contrib, s.sleeve)
    return best[1]

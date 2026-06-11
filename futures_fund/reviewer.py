from __future__ import annotations

from collections.abc import Mapping

from futures_fund.contracts import (
    CoinGeometry,
    ReviewerCheck,
    TargetWeights,
)
from futures_fund.neutrality import (
    NeutralityConfig,
    cluster_roots,
    size_btc_hedge,
)

# The every-cycle Adversarial Code & Calc Reviewer (§10 Guardian, §12). These checks (canonical
# names 1-6) re-derive the §5/§8 neutrality/hedge/deployment/cap numbers from GROUND TRUTH — the
# emitted legs, their directions, and the geometry betas — and compare them to the artifact's
# stated value. A check NEVER trusts `TargetWeights`' own residual/deployment fields (those are
# exactly what it is auditing): every `expected` is recomputed here, the artifact's stated number
# is `actual`, and `ok` is the in-band / matched comparison.


def _signed_notional(direction: str, target_notional: float) -> float:
    """Ground-truth signed leg notional in USDT: long => +|notional|, short => -|notional|.

    The sign is taken from `direction` (the load-bearing field) so the re-derivation is correct
    whether the artifact stored `target_notional` signed (the optimizer) or as a bare magnitude
    (a hand-built / tampered book). The reviewer must not depend on the artifact already having the
    right sign baked into `target_notional`."""
    mag = abs(target_notional)
    return mag if direction == "long" else -mag


def _alpha_legs(target: TargetWeights) -> list:
    """The ALPHA legs (everything except the dedicated BTC hedge leg). The hedge is sleeve
    "hedge"; `size_btc_hedge` (and thus the re-derivation) runs on the alpha legs only."""
    return [leg for leg in target.legs if leg.sleeve != "hedge"]


def check_dollar_neutral(target: TargetWeights, cfg: NeutralityConfig) -> ReviewerCheck:
    """Re-derive the dollar residual from the legs (Sum(long$) - Sum(short$)) and the residual
    FRACTION (|residual| / side_budget), then compare that recomputed fraction to `cfg.dollar_band`.

    The artifact's stated `dollar_residual_frac` is the `actual`; it is NOT trusted — an imbalanced
    book that claims `dollar_residual_frac == 0` is caught because the recomputed fraction is what
    decides `ok`."""
    longs = sum(
        abs(leg.target_notional) for leg in target.legs if leg.direction == "long"
    )
    shorts = sum(
        abs(leg.target_notional) for leg in target.legs if leg.direction == "short"
    )
    residual = longs - shorts
    side_budget = cfg.side_budget_usdt
    frac = abs(residual) / side_budget if side_budget > 0 else 0.0
    ok = frac <= cfg.dollar_band + 1e-9
    return ReviewerCheck(
        name="dollar_residual_in_band",
        ok=ok,
        expected=frac,
        actual=target.dollar_residual_frac,
        tolerance=cfg.dollar_band,
        detail=f"recomputed |Σlong$ - Σshort$|/side_budget = {frac:.6f} vs band {cfg.dollar_band}",
    )


def check_beta_neutral(
    target: TargetWeights, geometries: list[CoinGeometry], cfg: NeutralityConfig
) -> ReviewerCheck:
    """Re-derive the portfolio beta residual Σ w_i·β_i from the legs and the GEOMETRY betas (never
    the per-leg `beta_btc`, which the artifact could mis-state), in equity-weight units (the same
    convention `optimize_book` uses), and compare |residual| to `cfg.beta_band`.

    Weights are recomputed from the signed notionals / equity so a tampered `beta_residual` field
    cannot hide a one-sided beta exposure."""
    betas = {g.symbol: g.beta_btc for g in geometries}
    betas["BTC/USDT:USDT"] = 1.0  # BTC's self-beta is 1.0 (the hedge benchmark, spec §5)
    equity = cfg.capital_usdt
    resid = 0.0
    for leg in target.legs:
        w = _signed_notional(leg.direction, leg.target_notional) / equity if equity > 0 else 0.0
        resid += w * betas.get(leg.symbol, 1.0)
    ok = abs(resid) <= cfg.beta_band + 1e-9
    return ReviewerCheck(
        name="beta_residual_in_band",
        ok=ok,
        expected=resid,
        actual=target.beta_residual,
        tolerance=cfg.beta_band,
        detail=f"recomputed Σ w·β = {resid:.6f} vs band {cfg.beta_band}",
    )


def check_btc_hedge(
    target: TargetWeights, geometries: list[CoinGeometry], cfg: NeutralityConfig
) -> ReviewerCheck:
    """Re-derive the BTC-perp hedge notional the ALPHA legs' residual beta demands (via
    `neutrality.size_btc_hedge`) and compare it to the artifact's `btc_hedge_notional`.

    The hedge absorbs the alpha legs' residual portfolio beta; the reviewer recomputes it exactly
    as the optimizer does (alpha weights = signed notional / equity, geometry betas, one per-side
    budget cap) so a tampered hedge size is caught."""
    betas = {g.symbol: g.beta_btc for g in geometries}
    betas["BTC/USDT:USDT"] = 1.0
    equity = cfg.capital_usdt
    alpha_weights: dict[str, float] = {}
    for leg in _alpha_legs(target):
        w = _signed_notional(leg.direction, leg.target_notional) / equity if equity > 0 else 0.0
        alpha_weights[leg.symbol] = alpha_weights.get(leg.symbol, 0.0) + w
    expected = size_btc_hedge(
        alpha_weights, betas, equity=equity, side_budget=cfg.side_budget_usdt
    )
    actual = target.btc_hedge_notional
    ok = abs(expected - actual) <= 1e-6 * max(1.0, abs(expected), abs(actual))
    return ReviewerCheck(
        name="btc_hedge_sizing",
        ok=ok,
        expected=expected,
        actual=actual,
        tolerance=1e-6,
        detail=f"re-derived BTC hedge notional {expected:.4f} vs stated {actual:.4f}",
    )


def check_deployment_floor(target: TargetWeights, cfg: NeutralityConfig) -> ReviewerCheck:
    """Re-derive each side's deployed fraction (gross side$ / side_budget) from the legs and
    require BOTH sides to be at or above `cfg.deployment_floor`.

    The artifact's stated `deploy_long_frac` / `deploy_short_frac` are NOT trusted; a side that is
    actually under-deployed is caught because the recomputed fractions decide `ok`."""
    longs = sum(
        abs(leg.target_notional) for leg in target.legs if leg.direction == "long"
    )
    shorts = sum(
        abs(leg.target_notional) for leg in target.legs if leg.direction == "short"
    )
    side_budget = cfg.side_budget_usdt
    deploy_long = longs / side_budget if side_budget > 0 else 0.0
    deploy_short = shorts / side_budget if side_budget > 0 else 0.0
    floor = cfg.deployment_floor
    ok = deploy_long >= floor - 1e-9 and deploy_short >= floor - 1e-9
    worst = min(deploy_long, deploy_short)
    return ReviewerCheck(
        name="deployment_floor_both_sides",
        ok=ok,
        expected=floor,
        actual=worst,
        tolerance=1e-9,
        detail=(
            f"recomputed deploy long={deploy_long:.4f} short={deploy_short:.4f} "
            f"vs floor {floor}"
        ),
    )


def check_caps(
    target: TargetWeights,
    cfg: NeutralityConfig,
    *,
    corr: Mapping[tuple[str, str], float] | None = None,
) -> list[ReviewerCheck]:
    """Re-derive the per-name and cluster caps from the legs and emit BOTH the `per_name_cap` and
    `cluster_cap` checks (canonical names 5 + 6).

    - per-name: each ALPHA leg's |notional| / equity must be <= `cfg.per_name_cap` (the same
      equity-fraction convention `neutrality._cap_violations` enforces; the dedicated BTC hedge is
      the benchmark hedge and is exempt).
    - cluster: same-side legs whose pairwise correlation >= `cfg.corr_threshold` are unioned into a
      cluster (via `neutrality.cluster_roots`); each >=2-member cluster's combined |w| must be
      <= `cfg.cluster_cap`. With no correlation map supplied no legs cluster, so a single leg never
      breaches the cluster cap on its own."""
    corr = corr or {}
    equity = cfg.capital_usdt
    alpha = _alpha_legs(target)
    weights: dict[str, float] = {}
    for leg in alpha:
        w = _signed_notional(leg.direction, leg.target_notional) / equity if equity > 0 else 0.0
        weights[leg.symbol] = weights.get(leg.symbol, 0.0) + w

    # per-name cap
    worst_name = ""
    worst_mag = 0.0
    for sym, w in weights.items():
        if abs(w) > worst_mag:
            worst_mag = abs(w)
            worst_name = sym
    per_name_ok = worst_mag <= cfg.per_name_cap + 1e-9
    per_name_check = ReviewerCheck(
        name="per_name_cap",
        ok=per_name_ok,
        expected=cfg.per_name_cap,
        actual=worst_mag,
        tolerance=1e-9,
        detail=f"max |w| = {worst_mag:.4f} ({worst_name}) vs per_name_cap {cfg.per_name_cap}",
    )

    # cluster cap: union-find correlated same-side legs, sum |w| per cluster
    roots = cluster_roots(weights, corr=corr, threshold=cfg.corr_threshold)
    cluster_mag: dict[str, float] = {}
    members: dict[str, int] = {}
    for sym, root in roots.items():
        cluster_mag[root] = cluster_mag.get(root, 0.0) + abs(weights[sym])
        members[root] = members.get(root, 0) + 1
    worst_cluster_mag = 0.0
    worst_root = ""
    for root, mag in cluster_mag.items():
        if members[root] >= 2 and mag > worst_cluster_mag:
            worst_cluster_mag = mag
            worst_root = root
    cluster_ok = worst_cluster_mag <= cfg.cluster_cap + 1e-9
    cluster_check = ReviewerCheck(
        name="cluster_cap",
        ok=cluster_ok,
        expected=cfg.cluster_cap,
        actual=worst_cluster_mag,
        tolerance=1e-9,
        detail=(
            f"heaviest correlated cluster |w| = {worst_cluster_mag:.4f} "
            f"(root {worst_root}) vs cluster_cap {cfg.cluster_cap}"
        ),
    )
    return [per_name_check, cluster_check]

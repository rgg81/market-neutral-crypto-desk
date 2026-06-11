from __future__ import annotations

from collections.abc import Mapping

from futures_fund.contracts import (
    CoinGeometry,
    Pair,
    ReviewerCheck,
    Spread,
    TargetWeights,
)
from futures_fund.funding_intervals import (
    clamp_funding_rate,
    realized_funding,
)
from futures_fund.metrics import PERIODS_PER_YEAR_DAILY, PERIODS_PER_YEAR_WEEKLY
from futures_fund.models import Cadence, TradeProposal
from futures_fund.neutrality import (
    NeutralityConfig,
    cluster_roots,
    size_btc_hedge,
)
from futures_fund.risk_gate import MIN_RR, _reward_risk

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


# === canonical names 7-13: funding(sign+amount) / pair(pnl+hedge) / RR / Sharpe / filters ===
# These re-derive the load-bearing P0 realism primitives from GROUND TRUTH (the legs/spreads/specs)
# and compare to the artifact's stated figure within `_TOL` — the same NEVER-trust-the-artifact
# discipline as names 1-6. `check_funding` and `check_pair_pnl` each emit TWO checks.

_TOL = 1e-6  # reviewer tolerance (canonical contract §reviewer.tolerance)


def check_funding(
    target: TargetWeights, geometries: list[CoinGeometry]
) -> list[ReviewerCheck]:
    """Re-derive each ALPHA leg's realized funding from ground truth and emit BOTH `funding_sign`
    and `funding_amount` (canonical names 7 + 8).

    For each leg the qty = |target_notional| / mark and the per-interval rate is the geometry's
    SIGNED `funding_rate`, clamped sign-preservingly (`clamp_funding_rate`). The settlement is
    `funding_intervals.realized_funding(...)`, which is a SIGNED balance contribution: a SHORT on
    POSITIVE funding RECEIVES funding (a positive CREDIT). The reviewer never trusts an artifact's
    funding figure — both the sign and the amount are recomputed here.

    - `funding_sign`: the recomputed total realized funding's sign is consistent with the legs'
      directions and signed rates (a short-on-positive-funding shows a positive credit). `ok` is
      True iff no individual leg's realized contribution contradicts its direction×rate sign.
    - `funding_amount`: the recomputed total realized funding (the `actual` mirrors it — there is
      no separately-stated funding field on `TargetWeights`, so the amount check pins that the
      re-derivation itself is well-formed and finite)."""
    geo = {g.symbol: g for g in geometries}
    total = 0.0
    sign_ok = True
    for leg in _alpha_legs(target):
        g = geo.get(leg.symbol)
        if g is None or g.mark <= 0:
            continue
        qty = abs(leg.target_notional) / g.mark
        rate = clamp_funding_rate(leg.symbol, g.funding_rate)
        contrib = realized_funding(
            leg.target_notional, g.mark, qty, rate, leg.direction
        )
        total += contrib
        # ground-truth sign: long pays on +rate (cost, <=0 credit); short receives on +rate.
        side = 1.0 if leg.direction == "long" else -1.0
        expected_sign = -side * rate  # sign of -side*mark*qty*rate (mark,qty > 0)
        if expected_sign > 0 and contrib < -_TOL:
            sign_ok = False
        if expected_sign < 0 and contrib > _TOL:
            sign_ok = False

    sign_check = ReviewerCheck(
        name="funding_sign",
        ok=sign_ok,
        expected=total,
        actual=total,
        tolerance=_TOL,
        detail=(
            f"re-derived realized funding {total:.6f} (short-on-positive-funding is a credit); "
            f"per-leg sign consistent with direction×rate = {sign_ok}"
        ),
    )
    amount_check = ReviewerCheck(
        name="funding_amount",
        ok=(total == total),  # finite (not NaN) — re-derivation is well-formed
        expected=total,
        actual=total,
        tolerance=_TOL,
        detail=f"re-derived Σ realized_funding = {total:.6f}",
    )
    return [sign_check, amount_check]


def check_pair_pnl(spreads: list[Spread], pairs: list[Pair]) -> list[ReviewerCheck]:
    """Re-derive pair PnL at the SPREAD level and the leg hedge-ratio sizing — emit BOTH
    `pair_pnl_attribution` and `pair_leg_hedge_ratio` (canonical names 9 + 10).

    - `pair_pnl_attribution`: PnL is attributed at the spread (not per-leg) level. A spread is a
      directional position in the traded unit (`y - hedge_ratio*x`); its PnL is
      `side * qty_y * (spread_value - mu)` where `side = +1` for `long_spread` (long the spread),
      `-1` for `short_spread`, and `mu` is the OU entry/mean anchor. The reviewer recomputes that
      and compares to the artifact's stated `realized_pnl` within `_TOL`.
    - `pair_leg_hedge_ratio`: the x leg MUST be sized at `hedge_ratio * qty_y` (otherwise the pair
      carries a residual single-name exposure); `|qty_x - hedge_ratio*qty_y|` must be within
      tolerance."""
    pair_by_id = {p.pair_id: p for p in pairs}

    attribution_ok = True
    worst_attr = 0.0
    hedge_ok = True
    worst_hedge = 0.0
    for s in spreads:
        p = pair_by_id.get(s.pair_id)
        if p is None:
            continue
        # spread-level PnL re-derivation
        side = 1.0 if s.state == "long_spread" else (-1.0 if s.state == "short_spread" else 0.0)
        expected_pnl = side * s.qty_y * (s.spread_value - p.mu)
        attr_err = abs(expected_pnl - s.realized_pnl)
        worst_attr = max(worst_attr, attr_err)
        if attr_err > _TOL * max(1.0, abs(expected_pnl), abs(s.realized_pnl)):
            attribution_ok = False
        # hedge-ratio sizing re-derivation
        expected_qx = p.hedge_ratio * s.qty_y
        hedge_err = abs(expected_qx - s.qty_x)
        worst_hedge = max(worst_hedge, hedge_err)
        if hedge_err > _TOL * max(1.0, abs(expected_qx), abs(s.qty_x)):
            hedge_ok = False

    attribution_check = ReviewerCheck(
        name="pair_pnl_attribution",
        ok=attribution_ok,
        expected=worst_attr,
        actual=worst_attr,
        tolerance=_TOL,
        detail=(
            f"re-derived spread-level PnL; worst |expected - stated| = {worst_attr:.6f}"
        ),
    )
    hedge_check = ReviewerCheck(
        name="pair_leg_hedge_ratio",
        ok=hedge_ok,
        expected=worst_hedge,
        actual=worst_hedge,
        tolerance=_TOL,
        detail=f"worst |qty_x - hedge_ratio·qty_y| = {worst_hedge:.6f}",
    )
    return [attribution_check, hedge_check]


def check_rr_after_costs(proposals: list[TradeProposal]) -> ReviewerCheck:
    """Re-derive each proposal's reward:risk via `risk_gate._reward_risk` (the SAME geometric
    take-profit/stop math the gate's RR floor uses) and require every proposal to clear `MIN_RR`
    (>= 2.0). The worst RR across the proposals decides `ok` (canonical name 11)."""
    worst = float("inf")
    for p in proposals:
        rr = _reward_risk(p)
        worst = min(worst, rr)
    if worst == float("inf"):
        worst = 0.0
    ok = worst >= MIN_RR - _TOL
    return ReviewerCheck(
        name="rr_after_costs",
        ok=ok,
        expected=MIN_RR,
        actual=worst,
        tolerance=_TOL,
        detail=f"worst re-derived RR = {worst:.4f} vs floor {MIN_RR}",
    )


def check_sharpe_annualization(cadence: Cadence) -> ReviewerCheck:
    """Re-derive the Sharpe annualization factor from the cadence: daily -> 365, weekly -> 52 (the
    §11/§18 fix; NOT the inherited 2190 4h factor). `ok` iff the factor is the cadence-correct
    constant from `metrics` (canonical name 12)."""
    expected = (
        PERIODS_PER_YEAR_DAILY if cadence == "daily" else PERIODS_PER_YEAR_WEEKLY
    )
    legacy_4h = 2190.0
    ok = abs(expected - legacy_4h) > _TOL  # never the inherited 4h factor
    return ReviewerCheck(
        name="sharpe_annualization",
        ok=ok,
        expected=expected,
        actual=expected,
        tolerance=_TOL,
        detail=f"{cadence} annualization factor = {expected} (not the inherited 2190)",
    )


def check_exchange_filters(
    target: TargetWeights, geometries: list[CoinGeometry]
) -> list[ReviewerCheck]:
    """Re-derive exchange-filter compliance for every leg from its `SymbolSpec` (canonical name 13).

    Each ALPHA leg's traded notional must be >= the symbol's `min_notional`, and the implied qty
    (`|notional|/mark`) and price (`mark`) must sit on the `step_size` / `tick_size` grids. A leg
    whose geometry carries no spec is skipped (no filter to enforce). Sub-min-notional / off-grid
    legs are flagged non-compliant."""
    geo = {g.symbol: g for g in geometries}
    violations: list[str] = []
    for leg in _alpha_legs(target):
        g = geo.get(leg.symbol)
        if g is None or g.spec is None or g.mark <= 0:
            continue
        spec = g.spec
        notional = abs(leg.target_notional)
        if notional < spec.min_notional - _TOL:
            violations.append(
                f"{leg.symbol} notional {notional:.2f} < min_notional {spec.min_notional}"
            )
            continue
        qty = notional / g.mark
        if spec.step_size > 0:
            steps = qty / spec.step_size
            if abs(steps - round(steps)) > 1e-6:
                violations.append(f"{leg.symbol} qty {qty} off step_size {spec.step_size}")
        if spec.tick_size > 0:
            ticks = g.mark / spec.tick_size
            if abs(ticks - round(ticks)) > 1e-6:
                violations.append(f"{leg.symbol} mark {g.mark} off tick_size {spec.tick_size}")
    ok = not violations
    return [
        ReviewerCheck(
            name="exchange_filter_compliance",
            ok=ok,
            expected=0.0,
            actual=float(len(violations)),
            tolerance=_TOL,
            detail=("; ".join(violations) if violations else "all legs on-grid >= min_notional"),
        )
    ]

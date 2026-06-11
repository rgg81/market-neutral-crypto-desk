from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime

from futures_fund.contracts import (
    CoinGeometry,
    Pair,
    ReviewerCheck,
    ReviewerVerdict,
    SentimentReport,
    Spread,
    TargetWeights,
)
from futures_fund.cycle_io import cycle_dir
from futures_fund.funding_intervals import (
    clamp_funding_rate,
    realized_funding,
)
from futures_fund.market_data import is_crypto_perp
from futures_fund.metrics import PERIODS_PER_YEAR_DAILY, PERIODS_PER_YEAR_WEEKLY
from futures_fund.models import Cadence, TradeProposal
from futures_fund.neutrality import (
    NeutralityConfig,
    cluster_roots,
    size_btc_hedge,
)
from futures_fund.risk_gate import MIN_RR, _reward_risk
from futures_fund.sentiment_ingest import level_to_s, s_to_level, validate_point_in_time

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


def _physical_funding_sign(direction: str, rate: float) -> int:
    """GROUND-TRUTH funding sign from market physics, derived WITHOUT calling realized_funding.

    The settlement law (§11): on a POSITIVE funding rate longs PAY (debit, balance contribution
    < 0) and shorts RECEIVE (credit, > 0); on a NEGATIVE rate the sides swap; a zero rate settles
    nothing. So the expected balance-contribution sign is `-side(direction) * sign(rate)`. This is
    an INDEPENDENT re-derivation (it does not reuse `realized_funding`'s `-side*mark*qty*rate`
    expression), so comparing it against the primitive's actual output is a real falsification: if
    `realized_funding` ever flipped its sign convention the `funding_sign` check would fail."""
    if rate > 0:
        rate_sign = 1
    elif rate < 0:
        rate_sign = -1
    else:
        return 0
    side = 1 if direction == "long" else -1
    return -side * rate_sign


def check_funding(
    target: TargetWeights, geometries: list[CoinGeometry]
) -> list[ReviewerCheck]:
    """Re-derive each ALPHA leg's realized funding from ground truth and emit BOTH `funding_sign`
    and `funding_amount` (canonical names 7 + 8). Both checks are ADVERSARIAL cross-derivations: an
    independent re-derivation is the `expected` and the value produced by the audited primitive
    (`funding_intervals.realized_funding`) is the `actual`, so a sign/scale regression in the
    primitive falsifies the check (it is NOT a self-equal pin).

    For each leg the qty = |target_notional| / mark and the per-interval rate is the geometry's
    SIGNED `funding_rate`, clamped sign-preservingly (`clamp_funding_rate`). `realized_funding(...)`
    is a SIGNED balance contribution: a SHORT on POSITIVE funding RECEIVES funding (a positive
    CREDIT). The reviewer never trusts an artifact's funding figure.

    - `funding_sign`: per leg the EXPECTED contribution sign is re-derived from market physics
      (`_physical_funding_sign`, `-side*sign(rate)`, computed without the primitive) and compared
      to the SIGN of the primitive's output. `ok` is False if any leg's `realized_funding` sign
      contradicts the physical sign (the primitive flipped a credit to a debit or vice versa).
    - `funding_amount`: the EXPECTED total is the closed-form settlement `Σ -side*notional*rate`
      (independent of `realized_funding`'s mark*qty form; identical only when the primitive is
      correct, since mark*qty == |notional| for a leg sized at notional/mark) and the ACTUAL total
      is `Σ realized_funding(...)`. `ok` is False if they disagree beyond tolerance — so a primitive
      that, e.g., dropped the sign or used mark² would be caught."""
    geo = {g.symbol: g for g in geometries}
    expected_total = 0.0  # independent closed-form Σ -side*notional*rate
    actual_total = 0.0  # Σ realized_funding(...) (the audited primitive)
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
        actual_total += contrib
        # INDEPENDENT closed form: mark*qty == |notional| for a leg sized at notional/mark.
        side = 1.0 if leg.direction == "long" else -1.0
        expected_total += -side * abs(leg.target_notional) * rate
        # per-leg sign cross-check: physical expected sign vs the primitive's actual sign.
        exp_sign = _physical_funding_sign(leg.direction, rate)
        if exp_sign > 0 and contrib < -_TOL:
            sign_ok = False
        if exp_sign < 0 and contrib > _TOL:
            sign_ok = False
        if exp_sign == 0 and abs(contrib) > _TOL:
            sign_ok = False

    amount_ok = abs(expected_total - actual_total) <= _TOL * max(
        1.0, abs(expected_total), abs(actual_total)
    )

    sign_check = ReviewerCheck(
        name="funding_sign",
        ok=sign_ok,
        expected=expected_total,
        actual=actual_total,
        tolerance=_TOL,
        detail=(
            f"per-leg physical sign (-side·sign(rate)) vs realized_funding sign; "
            f"consistent = {sign_ok}; Σ realized = {actual_total:.6f} "
            f"(short-on-positive-funding is a credit)"
        ),
    )
    amount_check = ReviewerCheck(
        name="funding_amount",
        ok=amount_ok,
        expected=expected_total,
        actual=actual_total,
        tolerance=_TOL,
        detail=(
            f"closed-form Σ -side·notional·rate = {expected_total:.6f} vs "
            f"Σ realized_funding = {actual_total:.6f}"
        ),
    )
    return [sign_check, amount_check]


def check_pair_pnl(spreads: list[Spread], pairs: list[Pair]) -> list[ReviewerCheck]:
    """Re-derive pair PnL at the SPREAD level and the leg hedge-ratio sizing — emit BOTH
    `pair_pnl_attribution` and `pair_leg_hedge_ratio` (canonical names 9 + 10).

    - `pair_pnl_attribution`: PnL is attributed at the spread (not per-leg) level and measured
      SINCE ENTRY (standard realized PnL), not mark-to-mean. A spread is a directional position in
      the traded unit (`y - hedge_ratio*x`); its PnL is
      `side * qty_y * (spread_value - entry_spread)` where `side = +1` for `long_spread` (long the
      spread), `-1` for `short_spread`. A mean-reversion pair NEVER enters at the OU mean `mu`: it
      opens at `|z| >= entry_z`, so the entry spread is `mu - side*entry_z*sigma_eq` (a long_spread
      enters cheap at `z = -entry_z`, a short_spread enters rich at `z = +entry_z`). The reviewer
      reconstructs the entry spread from the OU params (`mu`, `sigma_eq` from the Pair) and the
      spread's own `entry_z`, then compares the re-derived PnL-since-entry to the artifact's stated
      `realized_pnl` within `_TOL`. (Anchoring on `mu` would only coincide with PnL-since-entry for
      a position entered exactly at the mean — which a mean-reversion entry never is — so this
      re-derivation matches production attribution.)
    - `pair_leg_hedge_ratio`: the x leg MUST be sized at `hedge_ratio * qty_y` (otherwise the pair
      carries a residual single-name exposure); `|qty_x - hedge_ratio*qty_y|` must be within
      tolerance."""
    pair_by_id = {p.pair_id: p for p in pairs}

    attribution_ok = True
    worst_attr = 0.0
    worst_expected_pnl = 0.0  # re-derived PnL of the worst-divergence spread (adversarial expected)
    worst_stated_pnl = 0.0    # the artifact's stated realized_pnl for that spread (actual)
    hedge_ok = True
    worst_hedge = 0.0
    for s in spreads:
        p = pair_by_id.get(s.pair_id)
        if p is None:
            continue
        # spread-level PnL re-derivation, measured SINCE ENTRY (not mark-to-mean). A
        # mean-reversion pair opens at |z| >= entry_z, never at the mean: a long_spread enters
        # cheap (z = -entry_z), a short_spread enters rich (z = +entry_z), so the entry spread is
        # mu - side*entry_z*sigma_eq. PnL = side * qty_y * (spread_now - entry_spread).
        side = 1.0 if s.state == "long_spread" else (-1.0 if s.state == "short_spread" else 0.0)
        entry_spread = p.mu - side * s.entry_z * p.sigma_eq
        expected_pnl = side * s.qty_y * (s.spread_value - entry_spread)
        attr_err = abs(expected_pnl - s.realized_pnl)
        if attr_err >= worst_attr:
            worst_attr = attr_err
            worst_expected_pnl = expected_pnl
            worst_stated_pnl = s.realized_pnl
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
        expected=worst_expected_pnl,
        actual=worst_stated_pnl,
        tolerance=_TOL,
        detail=(
            f"re-derived spread-level PnL-since-entry = {worst_expected_pnl:.6f} vs stated "
            f"{worst_stated_pnl:.6f} (worst |expected - stated| = {worst_attr:.6f})"
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
    (>= 2.0). The worst RR across the proposals decides `ok` (canonical name 11). With NO proposals
    the floor is vacuously satisfied (a cycle that opens nothing cannot violate the RR floor)."""
    if not proposals:
        return ReviewerCheck(
            name="rr_after_costs",
            ok=True,
            expected=MIN_RR,
            actual=MIN_RR,
            tolerance=_TOL,
            detail="no proposals to gate => RR floor vacuously satisfied",
        )
    worst = float("inf")
    for p in proposals:
        rr = _reward_risk(p)
        worst = min(worst, rr)
    ok = worst >= MIN_RR - _TOL
    return ReviewerCheck(
        name="rr_after_costs",
        ok=ok,
        expected=MIN_RR,
        actual=worst,
        tolerance=_TOL,
        detail=f"worst re-derived RR = {worst:.4f} vs floor {MIN_RR}",
    )


_SHARPE_FACTOR_SPEC: dict[Cadence, float] = {"daily": 365.0, "weekly": 52.0}
_LEGACY_4H_FACTOR = 2190.0  # the inherited (wrong-for-this-desk) 4h periods/yr (§11/§18)


def check_sharpe_annualization(cadence: Cadence) -> ReviewerCheck:
    """Cross-check the Sharpe annualization factor the PRODUCTION metrics module would apply
    against the spec ground truth (canonical name 12).

    `expected` is the spec-mandated periods/yr for this cadence (daily -> 365, weekly -> 52, the
    §11/§18 fix). `actual` is the constant the production `metrics` module actually exposes for that
    cadence (`PERIODS_PER_YEAR_DAILY` / `PERIODS_PER_YEAR_WEEKLY`) — the value `metrics.sharpe`
    annualizes with. `ok` is True iff they MATCH and the production constant is NOT the inherited
    2190 4h factor. So a regression that flips the metrics module back to the 4h factor (or to any
    other value) is FALSIFIED here — this is an independent expected-vs-actual comparison, not a
    self-equal pin."""
    expected = _SHARPE_FACTOR_SPEC[cadence]
    actual = (
        PERIODS_PER_YEAR_DAILY if cadence == "daily" else PERIODS_PER_YEAR_WEEKLY
    )
    matches = abs(expected - actual) <= _TOL
    not_legacy = abs(actual - _LEGACY_4H_FACTOR) > _TOL
    ok = matches and not_legacy
    return ReviewerCheck(
        name="sharpe_annualization",
        ok=ok,
        expected=expected,
        actual=actual,
        tolerance=_TOL,
        detail=(
            f"{cadence} spec periods/yr = {expected} vs metrics module {actual} "
            f"(must match and not be the inherited 2190)"
        ),
    )


def _round_qty_to_step(qty: float, step: float) -> float:
    """Floor a quantity onto the lot-step grid (floor, never round-up, so the book never
    over-fills). `step <= 0` => no constraint (qty passes through).

    This DEFINES the qty grid-rounding contract any future order-submission path must satisfy: the
    market-neutral repo has no executor / order-rounding primitive yet, so there is nothing to
    cross-derive against. The contract is the floor-to-`step_size` semantics of the weekly
    reference `orders.round_qty` (weekly repo `futures_fund/orders.py`); when an executor is ported
    here it MUST round identically or this check will (correctly) diverge from the submitted
    order."""
    if step <= 0:
        return qty
    return round(math.floor(qty / step) * step, 10)


def _round_price_to_tick(price: float, tick: float) -> float:
    """Round a price onto the tick grid (nearest tick). `tick <= 0` => no constraint.

    Like `_round_qty_to_step`, this DEFINES the price grid-rounding contract (nearest-`tick_size`)
    the future executor must satisfy; there is no executor price-rounding primitive in this repo to
    cross-derive against yet (it matches the weekly reference `orders.round_price` semantics)."""
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


def check_exchange_filters(
    target: TargetWeights, geometries: list[CoinGeometry]
) -> list[ReviewerCheck]:
    """Re-derive exchange-filter compliance for the ORDER THAT WOULD ACTUALLY BE SUBMITTED for each
    leg from its `SymbolSpec` (canonical name 13).

    Real Binance-futures filter compliance is about the qty/price an order carries AFTER
    grid-rounding (qty floored to `step_size`, price rounded to `tick_size`) — NOT whether the raw
    `|notional|/mark` ratio happens to land on the step grid (for an arbitrary notional ÷ arbitrary
    mark it almost never does, which would false-positive on every realistic book). This repo has no
    order-submission path yet, so the reviewer cannot cross-derive against a real executor; instead
    `_round_qty_to_step` / `_round_price_to_tick` DEFINE the grid-rounding contract the future
    executor must satisfy (floor-to-step qty, nearest-tick price — the weekly reference
    `orders.round_qty` / `orders.round_price` semantics). The reviewer applies that contract and
    then checks the REAL exchange constraints:

      - the floored qty must be > 0 (a leg that rounds to a zero lot can't be submitted), and
      - the executable notional (rounded_qty × rounded_price) must be >= `min_notional`.

    A leg whose geometry carries no spec is skipped (no filter to enforce). Sub-min-notional /
    rounds-to-dust legs are flagged non-compliant."""
    geo = {g.symbol: g for g in geometries}
    violations: list[str] = []
    for leg in _alpha_legs(target):
        g = geo.get(leg.symbol)
        if g is None or g.spec is None or g.mark <= 0:
            continue
        spec = g.spec
        notional = abs(leg.target_notional)
        # the order the executor would actually submit (grid-rounded qty @ grid-rounded price)
        raw_qty = notional / g.mark
        order_qty = _round_qty_to_step(raw_qty, spec.step_size)
        order_price = _round_price_to_tick(g.mark, spec.tick_size)
        if order_qty <= 0:
            violations.append(
                f"{leg.symbol} qty {raw_qty:.8f} rounds to 0 on step_size {spec.step_size}"
            )
            continue
        exec_notional = order_qty * order_price
        if exec_notional < spec.min_notional - _TOL:
            violations.append(
                f"{leg.symbol} executable notional {exec_notional:.2f} "
                f"< min_notional {spec.min_notional}"
            )
    ok = not violations
    return [
        ReviewerCheck(
            name="exchange_filter_compliance",
            ok=ok,
            expected=0.0,
            actual=float(len(violations)),
            tolerance=_TOL,
            detail=(
                "; ".join(violations)
                if violations
                else "all legs submit a non-zero lot >= min_notional after grid-rounding"
            ),
        )
    ]


# === canonical names 14-17: sentiment range/cap/PIT + crypto-only universe ==================
# These re-derive the §7 sentiment discipline and the §3 crypto-only mandate from GROUND TRUTH
# (the level<->s mapping, the before/after tilt magnitude, the source timestamps, the exchange
# market metadata) — the same NEVER-trust-the-artifact discipline as names 1-13. `check_sentiment`
# emits THREE checks (range + cap + point-in-time).

SENTIMENT_CAP = 0.25  # §7.2 conviction-tilt cap: |Δw| <= cap*|w| between target_before/after


def _leg_weights(target: TargetWeights) -> dict[str, float]:
    """Per-symbol signed equity-weight from the legs (sign from `direction`), for the cap delta."""
    out: dict[str, float] = {}
    for leg in target.legs:
        out[leg.symbol] = out.get(leg.symbol, 0.0) + _signed_notional(
            leg.direction, leg.target_notional
        )
    return out


def check_sentiment(
    sentiment: list[SentimentReport],
    target_before: TargetWeights,
    target_after: TargetWeights,
    *,
    cap: float = SENTIMENT_CAP,
) -> list[ReviewerCheck]:
    """Re-derive the §7 sentiment discipline from ground truth and emit THREE checks (canonical
    names 14 + 15 + 16). The reviewer NEVER trusts a report's stated figure — it round-trips the
    `level<->s` mapping itself, recomputes the tilt magnitude from the before/after books, and
    re-reads every source timestamp.

    - `sentiment_range`: every report's stated `s` must round-trip its `level` via the §7.1 ordinal
      mapping (`s_to_level(s) == level` AND `s == level_to_s(level)` within tolerance). A report
      whose numeric score contradicts its ordinal level (e.g. level "positive" but s -1.0) is
      caught.
    - `sentiment_cap_respected`: the per-symbol tilt the optimizer applied is `|w_after - w_before|`
      (signed equity-notional weights re-derived from the legs); it must be `<= cap*|w_before|` for
      every symbol. A tilt that grows a leg beyond the §7.2 cap (default 25%) is caught — this is
      the hard-veto the gate keys off when a sentiment tilt over-sizes a leg.
    - `sentiment_point_in_time`: every `SentimentSource.published_ts` must be strictly BEFORE its
      report's `as_of_ts` (`sentiment_ingest.validate_point_in_time`). A source published at/after
      the decision anchor (post-decision leakage) is caught."""
    # --- name 14: range (level <-> s round-trip) ---
    range_ok = True
    worst_sym = ""
    for r in sentiment:
        mapped_s = level_to_s(r.level)
        if abs(mapped_s - r.s) > 1e-9 or s_to_level(r.s) != r.level:
            range_ok = False
            worst_sym = r.symbol
    range_check = ReviewerCheck(
        name="sentiment_range",
        ok=range_ok,
        expected=0.0,
        actual=0.0 if range_ok else 1.0,
        tolerance=1e-9,
        detail=(
            f"{worst_sym} stated s does not round-trip its level (§7.1 mapping)"
            if not range_ok
            else "every report's s round-trips its ordinal level"
        ),
    )

    # --- name 15: cap (|Δw| <= cap*|w_before|) ---
    before_w = _leg_weights(target_before)
    after_w = _leg_weights(target_after)
    cap_ok = True
    worst_delta = 0.0
    worst_cap_sym = ""
    for sym in set(before_w) | set(after_w):
        wb = before_w.get(sym, 0.0)
        wa = after_w.get(sym, 0.0)
        delta = abs(wa - wb)
        allowed = cap * abs(wb)
        if delta > allowed + 1e-9:
            cap_ok = False
        if delta > worst_delta:
            worst_delta = delta
            worst_cap_sym = sym
    cap_check = ReviewerCheck(
        name="sentiment_cap_respected",
        ok=cap_ok,
        expected=cap,
        actual=worst_delta,
        tolerance=1e-9,
        detail=(
            f"worst |w_after - w_before| = {worst_delta:.2f} ({worst_cap_sym}) "
            f"vs cap {cap}·|w_before|"
        ),
    )

    # --- name 16: point-in-time (every source published before as_of) ---
    pit_ok = all(validate_point_in_time(r) for r in sentiment)
    pit_check = ReviewerCheck(
        name="sentiment_point_in_time",
        ok=pit_ok,
        expected=0.0,
        actual=0.0 if pit_ok else 1.0,
        tolerance=0.0,
        detail=(
            "every source.published_ts < report.as_of_ts"
            if pit_ok
            else "a source was published at/after the decision anchor (PIT leakage)"
        ),
    )
    return [range_check, cap_check, pit_check]


def check_crypto_only(geometries: list[CoinGeometry]) -> ReviewerCheck:
    """Re-verify the §3 CRYPTO-ONLY mandate: every traded geometry must be a cryptocurrency COIN
    perp (canonical name 17). Reuses `market_data.is_crypto_perp` on each geometry's `market_info`
    (the exchange `market["info"]`, carrying `underlyingType` / `contractType`) so a TradFi-wrapper
    perp — a tokenized stock (EQUITY), commodity (gold/silver/oil), index basket — is flagged. A
    geometry with no metadata is treated as crypto (fail-open on a metadata gap is the same posture
    `is_crypto_perp`'s `contractType` fallback takes)."""
    violations: list[str] = []
    for g in geometries:
        market = {"info": g.market_info or {}}
        if not is_crypto_perp(market):
            utype = (g.market_info or {}).get("underlyingType")
            violations.append(f"{g.symbol} ({utype})")
    ok = not violations
    return ReviewerCheck(
        name="crypto_only_universe",
        ok=ok,
        expected=0.0,
        actual=float(len(violations)),
        tolerance=0.0,
        detail=(
            "all geometries are cryptocurrency COIN perps"
            if ok
            else "non-crypto (TradFi-wrapper) perps in universe: " + ", ".join(violations)
        ),
    )


# === review_cycle (AND of all 17) + the deterministic gate flag ============================


def review_cycle(
    state_dir,
    memory_dir,
    cycle: int,
    cadence: Cadence,
    *,
    target: TargetWeights,
    geometries: list[CoinGeometry],
    spreads: list[Spread],
    sentiment: list[SentimentReport],
    cfg: NeutralityConfig,
    returns: list[float] | None = None,  # noqa: ARG001 — reserved for a future realized-Sharpe check
    pairs: list[Pair] | None = None,
    proposals: list[TradeProposal] | None = None,
    corr: Mapping[tuple[str, str], float] | None = None,
    target_before: TargetWeights | None = None,
    target_after: TargetWeights | None = None,
) -> ReviewerVerdict:
    """Run every canonical reviewer check (the full 17) against the cycle's artifacts, re-derived
    from ground truth, and AND them into a single deterministic `ReviewerVerdict` (§10 Guardian,
    §12). `passed` is the AND of all 17 `ReviewerCheck.ok`; `mismatches` is exactly the names of
    the failed checks (`[c.name for c in checks if not c.ok]`). This is the hard-veto verdict the
    execute step keys off via `reviewer_gate_ok`.

    The sentiment cap compares `target_before` vs `target_after` (the conviction-tilt before/after
    books, §7.3 ordering invariant); if not supplied they default to `target` (no tilt => cap
    trivially respected)."""
    before = target_before if target_before is not None else target
    after = target_after if target_after is not None else target
    checks: list[ReviewerCheck] = [
        check_dollar_neutral(target, cfg),                       # 1
        check_beta_neutral(target, geometries, cfg),             # 2
        check_btc_hedge(target, geometries, cfg),                # 3
        check_deployment_floor(target, cfg),                     # 4
        *check_caps(target, cfg, corr=corr),                     # 5 + 6
        *check_funding(target, geometries),                      # 7 + 8
        *check_pair_pnl(spreads, pairs or []),                   # 9 + 10
        check_rr_after_costs(proposals or []),                   # 11
        check_sharpe_annualization(cadence),                     # 12
        *check_exchange_filters(target, geometries),             # 13
        *check_sentiment(sentiment, before, after),              # 14 + 15 + 16
        check_crypto_only(geometries),                           # 17
    ]
    mismatches = [c.name for c in checks if not c.ok]
    return ReviewerVerdict(
        passed=not mismatches,
        checks=checks,
        mismatches=mismatches,
        cycle=cycle,
        cadence=cadence,
        reviewed_at=datetime.now(UTC),
    )


def reviewer_gate_ok(state_dir, cycle: int, cadence: Cadence) -> bool:
    """Read the persisted `reviewer.json` for this cadence cycle and return its `passed` flag — the
    DETERMINISTIC HALT flag the execute step checks before ANY fill (§10/§12, mandatory
    non-skippable stage). A MISSING verdict (the reviewer never ran) is treated as NOT ok, exactly
    like a failed one: absence must HALT just as hard as an explicit veto, so a skipped reviewer can
    never let a book through."""
    path = cycle_dir(state_dir, cycle, cadence=cadence) / "reviewer.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("passed", False))

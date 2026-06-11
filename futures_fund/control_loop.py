"""Two-cadence control loop (§9): weekly Selection + daily Rebalance.

This module owns the cadence -> (tf_minutes, loop, cycle-root) mapping. `cadence_due` wraps the P0
`scheduling.cycle_due` primitive with the right candle width and per-cadence cycle root.

CADENCE-ROOT INVARIANT (binding, §14 + canonical contract): every cadence's cycle artifacts live
under `state/<cadence>/cycle/<N>/` — the SAME root the due-gate reads (NOT `state/cycle/<cadence>`).
`cadence_cycle_root` below is the SINGLE source of truth for that path: the gate derives the root it
scans from it, and any future artifact writer MUST derive its write path from it too, so the gate
and the writer can never drift onto different roots.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from futures_fund.contracts import (
    CoinGeometry,
    SleeveSignal,
    SleeveTilt,
    Spread,
    TargetWeights,
    WeightLeg,
)
from futures_fund.cycle_io import save_output
from futures_fund.models import Cadence
from futures_fund.neutrality import NeutralityConfig, optimize_book, risk_parity_budgets
from futures_fund.scheduling import cycle_due

# Candle width per cadence: weekly = 7 days, daily = 1 day.
_CADENCE_TF = {"weekly": 7 * 1440, "daily": 1440}  # 10080 / 1440 minutes


def cadence_cycle_root(state_dir, cadence: Cadence) -> Path:
    """Canonical cycle-artifact root for a cadence: `state/<cadence>/cycle`.

    SINGLE source of truth for the CADENCE-ROOT INVARIANT. `cadence_due` scans exactly this root
    (via `scheduling.cycle_due(loop=cadence)`, which builds `state/<loop>/cycle`), and the artifact
    writer that persists cycle <N> MUST write under `cadence_cycle_root(state_dir, cadence)/str(N)`
    so the gate-read root and the writer root are provably identical."""
    return Path(state_dir) / cadence / "cycle"


def cadence_due(state_dir, now_utc: datetime, cadence: Cadence) -> tuple[str, int, str]:
    """Decide whether the current candle of `cadence` still needs a cycle (mode, n, reason).

    weekly -> tf_minutes=10080, loop="weekly"; daily -> tf_minutes=1440, loop="daily". `cycle_due`
    scans `state/<cadence>/cycle/*`, which equals `cadence_cycle_root(state_dir, cadence)` (the root
    the artifact writer must use — CADENCE-ROOT INVARIANT)."""
    tf = _CADENCE_TF[cadence]
    return cycle_due(state_dir, now_utc, tf_minutes=tf, loop=cadence)


def weekly_selection(
    state_dir,
    geometries: list[CoinGeometry],
    sleeves: list[SleeveSignal],
    *,
    equity: float,
    prior: TargetWeights | None,
    cfg: NeutralityConfig,
    cycle: int,
) -> TargetWeights:
    """Weekly Selection Meeting (§9): full re-selection of the symbol set + target weights.

    Risk-budgets the sleeves, runs `neutrality.optimize_book` to produce a dollar+beta-neutral,
    deployment-floor-respecting `TargetWeights`, then persists it under the cadence-segmented root
    `state/weekly/cycle/<cycle>/target_weights.json` (the SAME root the weekly due-gate reads —
    CADENCE-ROOT INVARIANT). When a `prior` book is supplied its legs seed the optimizer's
    turnover/no-trade band so only the deltas are traded (carry-over, §9)."""
    risk_parity_budgets(sleeves)
    tw = optimize_book(
        sleeves,
        geometries,
        equity=equity,
        prior_legs=prior.legs if prior else None,
        cfg=cfg,
    )
    save_output(state_dir, cycle, "target_weights", tw, cadence="weekly")
    return tw


def rebalance_deltas(prior: TargetWeights, target: TargetWeights) -> list[WeightLeg]:
    """Carry-over delta book (§9: "trade only the deltas").

    Keying on `(symbol, direction)`, emit a delta leg only when the leg is NEW or its
    `target_notional` moved beyond a $1 epsilon — so an unchanged overlapping leg (same symbol,
    same direction, same notional within $1) is EXCLUDED and the book is not churned. Legs present
    in `prior` but absent from `target` become zero-notional unwind deltas so the old exposure is
    flattened rather than silently carried."""
    prior_by = {(leg.symbol, leg.direction): leg for leg in prior.legs}
    out: list[WeightLeg] = []
    for leg in target.legs:
        p = prior_by.get((leg.symbol, leg.direction))
        if p is None or abs(leg.target_notional - p.target_notional) > 1.0:
            out.append(leg)  # carry-over: unchanged overlap excluded
    # removed legs (in prior, absent from target) become zero-notional unwinds
    tgt_keys = {(leg.symbol, leg.direction) for leg in target.legs}
    for (sym, d), p in prior_by.items():
        if (sym, d) not in tgt_keys:
            out.append(p.model_copy(update={"target_notional": 0.0, "weight": 0.0}))
    return out


def drift_exceeded(
    current_weight: float, target_weight: float, *, drift_band: float = 0.20
) -> bool:
    """Daily no-trade-band gate (§9): is a leg's weight far enough from target to warrant a trade?

    Returns True when the relative drift `|current - target| / |target|` exceeds `drift_band`. When
    the target weight is exactly 0.0 (the leg should be flat) any nonzero current weight is itself a
    breach, so the residual exposure gets traded out."""
    if target_weight == 0.0:
        return current_weight != 0.0
    return abs(current_weight - target_weight) / abs(target_weight) > drift_band


def neutrality_breached(target: TargetWeights, cfg: NeutralityConfig) -> bool:
    """Neutrality-breach trigger (§9): does the book violate either neutrality band?

    True when the dollar residual fraction exceeds `cfg.dollar_band` OR the absolute beta residual
    exceeds `cfg.beta_band` — either condition forces a daily rebalance trade even if every leg is
    individually inside its drift band, so the book never drifts off dollar/beta neutral."""
    return (
        target.dollar_residual_frac > cfg.dollar_band
        or abs(target.beta_residual) > cfg.beta_band
    )


def _sleeves_from_legs(legs: list[WeightLeg], as_of_ts: datetime) -> list[SleeveSignal]:
    """Reconstitute the per-sleeve tilt signals from a prior book's alpha legs (§9 fixed set).

    The Daily Rebalance Meeting keeps the SAME symbol set as the weekly target, so we re-derive the
    sleeve tilts from that book's legs rather than re-selecting. The BTC hedge leg is excluded — it
    is sized by the optimizer from the alpha legs' residual beta, never a tilt input. Each alpha
    leg's `weight` (signed by direction) becomes its sleeve's `SleeveTilt.target_weight`, grouped by
    the leg's originating `sleeve` so `optimize_book` re-runs against an identical universe."""
    by_sleeve: dict[str, list[SleeveTilt]] = {}
    for leg in legs:
        if leg.sleeve == "hedge":
            continue
        signed = abs(leg.weight) if leg.direction == "long" else -abs(leg.weight)
        by_sleeve.setdefault(leg.sleeve, []).append(
            SleeveTilt(
                symbol=leg.symbol,
                direction=leg.direction,
                target_weight=signed,
                pair_id=leg.pair_id,
            )
        )
    return [
        SleeveSignal(sleeve=sleeve, tilts=tilts, risk_budget_frac=1.0, as_of_ts=as_of_ts)
        for sleeve, tilts in by_sleeve.items()
    ]


def daily_rebalance(
    state_dir,
    target: TargetWeights,
    geometries: list[CoinGeometry],
    spreads: list[Spread],
    *,
    equity: float,
    cfg: NeutralityConfig,
    cycle: int,
) -> TargetWeights:
    """Daily Rebalance Meeting (§9): nudge the SAME symbol set back toward target within a band.

    Keeps the weekly `target`'s symbol set fixed (no re-selection), recomputes residuals/z/funding/
    sentiment by re-running `optimize_book` against that set (`prior_legs=target.legs`, so the
    turnover/no-trade band excludes unchanged legs), then trades ONLY the deltas via
    `rebalance_deltas(prior=target, target=recomputed)`. Two overrides force a trade on an
    otherwise in-band leg: any `Spread.state == "stop"` FLATTENS that pair's legs (zero-notional
    unwind deltas — the hard z-stop is the cointegration-break EXIT, §6.2, NOT a re-mark at target
    notional), and a `neutrality_breached` recomputed book forces the full recomputed leg set
    (dollar/beta drift off-neutral). The returned `TargetWeights` carries the recomputed book's
    residual/deployment metadata but its `legs` are the delta book the Trader must execute — an
    in-band, no-stop, neutral book therefore yields ZERO delta legs (no churn). Persisted under the
    cadence-segmented daily root `state/daily/cycle/<cycle>/target_weights.json` (the SAME root the
    daily due-gate reads — CADENCE-ROOT INVARIANT)."""
    sleeves = _sleeves_from_legs(target.legs, target.as_of_ts)
    recomputed = optimize_book(
        sleeves,
        geometries,
        equity=equity,
        prior_legs=target.legs,
        cfg=cfg,
    )

    # base delta book: only names whose notional moved (carry-over excludes unchanged overlap)
    delta_by_key: dict[tuple[str, str], WeightLeg] = {
        (leg.symbol, leg.direction): leg
        for leg in rebalance_deltas(target, recomputed)
    }

    # z-stop override: a hard z-stop (|z| >= stop_z) is the cointegration-break EXIT (§6.2: "hard
    # stop |z| >= 3"; the pairs sleeve treats state=="stop" as "emit no legs" == close). So a
    # stopped spread FLATTENS its pair's legs — NOT re-mark them at target notional. The pair_id is
    # carried on the PRIOR target's legs (the fixed set), so we resolve stopped pairs -> the prior
    # legs there and force a ZERO-notional unwind delta (mirroring rebalance_deltas' removed-leg
    # unwind) keyed by the prior leg's (symbol, direction) — the position that actually exists.
    stopped_pairs = {sp.pair_id for sp in spreads if sp.state == "stop"}
    if stopped_pairs:
        for leg in target.legs:
            if leg.pair_id in stopped_pairs:
                delta_by_key[(leg.symbol, leg.direction)] = leg.model_copy(
                    update={"target_notional": 0.0, "weight": 0.0}
                )

    # neutrality-breach override: an off-neutral recomputed book forces the FULL recomputed set.
    if neutrality_breached(recomputed, cfg):
        for leg in recomputed.legs:
            delta_by_key[(leg.symbol, leg.direction)] = leg

    delta_legs = list(delta_by_key.values())
    rebalanced = recomputed.model_copy(update={"legs": delta_legs})
    save_output(state_dir, cycle, "target_weights", rebalanced, cadence="daily")
    return rebalanced

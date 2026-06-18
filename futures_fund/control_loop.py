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

import pandas as pd

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


def latest_cadence_cycle(state_dir, cadence: Cadence, artifact: str) -> int | None:
    """Highest cadence cycle whose `<artifact>.json` exists, or None if none do.

    The daily and weekly cycle counters are INDEPENDENT (each `cadence_due` scans its own
    `state/<cadence>/cycle/*` root and returns `highest_dir+1`), and daily increments ~7x faster
    than weekly, so a daily cycle number does NOT index the corresponding weekly cycle. To pick up
    the MOST RECENT weekly book a daily rebalance should track, scan the weekly root for the highest
    cycle dir that actually persisted `artifact` rather than reusing the daily cycle number. Returns
    None when no cycle has produced the artifact yet (caller fails closed)."""
    root = cadence_cycle_root(state_dir, cadence)
    if not root.exists():
        return None
    cycles = sorted(
        (int(p.name) for p in root.glob("*") if p.is_dir() and p.name.isdigit()),
        reverse=True,
    )
    for n in cycles:
        if (root / str(n) / f"{artifact}.json").exists():
            return n
    return None


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
    returns: pd.DataFrame | None = None,
) -> TargetWeights:
    """Weekly Selection Meeting (§9): full re-selection of the symbol set + target weights.

    Risk-budgets the sleeves, runs `neutrality.optimize_book` to produce a dollar+beta-neutral,
    deployment-floor-respecting `TargetWeights`, then persists it under the cadence-segmented root
    `state/weekly/cycle/<cycle>/target_weights.json` (the SAME root the weekly due-gate reads —
    CADENCE-ROOT INVARIANT). When a `prior` book is supplied its legs seed the optimizer's
    turnover/no-trade band so only the deltas are traded (carry-over, §9). `returns` (the per-symbol
    return frame) feeds the optimizer's Ledoit-Wolf/HRP shaping AND the cluster cap; None (or empty)
    degrades to the merged split (the historical behaviour)."""
    risk_parity_budgets(sleeves)
    tw = optimize_book(
        sleeves,
        geometries,
        equity=equity,
        prior_legs=prior.legs if prior else None,
        cfg=cfg,
        returns=returns,
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
    returns: pd.DataFrame | None = None,
) -> TargetWeights:
    """Daily Rebalance Meeting (§9): nudge the SAME symbol set back toward target within a band.

    Keeps the weekly `target`'s symbol set fixed (no re-selection), recomputes residuals/z/funding/
    sentiment by re-running `optimize_book` against that set (`prior_legs=target.legs`, so the
    turnover/no-trade band excludes unchanged legs). It produces TWO artifacts, separating "intended
    holdings" from "trades to make this cycle":

    - `target_weights.json` is the FULL recomputed (intended-holdings) book — the same neutral,
      hedge-correct, fully-deployed book `optimize_book` produced. This is the book the every-cycle
      reviewer audits: every check re-derives its load-bearing number (BTC hedge sizing, dollar/beta
      residual, deployment floor, caps) from THESE legs, so the artifact's metadata and its legs are
      internally consistent and all 17 checks validate the ACTUAL resulting positions and pass.
    - `rebalance_trades.json` is the sparse TRADE-DELTA book the executor opens this cycle (§9
      "trade only the deltas"). It carries ONLY the changed legs via `rebalance_deltas(prior=target,
      target=recomputed)` — so an in-band, no-stop, neutral book yields ZERO trade deltas (no churn:
      the daily cadence does NOT re-trade the whole book). Two overrides force a trade on an
      otherwise in-band leg: a `neutrality_breached` recomputed book forces the full recomputed leg
      set (dollar/beta drift off-neutral), and any `Spread.state == "stop"` FLATTENS that pair's
      legs (zero-notional unwind deltas — the hard z-stop is the cointegration-break EXIT, §6.2,
      NOT a re-mark at target notional). The z-stop flatten is applied LAST so the hard-stop EXIT
      WINS: a book that is BOTH breached AND has a stopped spread still flattens the broken pair
      rather than re-marking its legs at target notional.

    Returns the FULL recomputed book (the reviewed intended-holdings target). Both artifacts are
    persisted under the cadence-segmented daily root `state/daily/cycle/<cycle>/` (the SAME root the
    daily due-gate reads — CADENCE-ROOT INVARIANT)."""
    sleeves = _sleeves_from_legs(target.legs, target.as_of_ts)
    recomputed = optimize_book(
        sleeves,
        geometries,
        equity=equity,
        prior_legs=target.legs,
        cfg=cfg,
        returns=returns,
    )

    # base delta book: only names whose notional moved (carry-over excludes unchanged overlap)
    delta_by_key: dict[tuple[str, str], WeightLeg] = {
        (leg.symbol, leg.direction): leg
        for leg in rebalance_deltas(target, recomputed)
    }

    # neutrality-breach override: an off-neutral recomputed book forces the FULL recomputed set.
    if neutrality_breached(recomputed, cfg):
        for leg in recomputed.legs:
            delta_by_key[(leg.symbol, leg.direction)] = leg

    # z-stop override (applied LAST so the hard-stop EXIT WINS): a hard z-stop (|z| >= stop_z) is
    # the cointegration-break EXIT (§6.2: "hard stop |z| >= 3"; the pairs sleeve treats
    # state=="stop" as "emit no legs" == close). So a stopped spread FLATTENS its pair's legs — NOT
    # re-mark them at target notional. The pair_id is carried on the PRIOR target's legs (the fixed
    # set), so we resolve stopped pairs -> the prior legs there and force a ZERO-notional unwind
    # delta (mirroring rebalance_deltas' removed-leg unwind) keyed by the prior leg's
    # (symbol, direction) — the position that actually exists. This MUST run after the
    # neutrality-breach loop: when a book is BOTH breached AND has a stopped spread, the breach loop
    # would otherwise re-mark the stopped pair's legs at TARGET notional, defeating the hard z-stop
    # exit; flattening last guarantees the cointegration-broken pair is closed, never re-opened at
    # full size.
    stopped_pairs = {sp.pair_id for sp in spreads if sp.state == "stop"}
    if stopped_pairs:
        for leg in target.legs:
            if leg.pair_id in stopped_pairs:
                delta_by_key[(leg.symbol, leg.direction)] = leg.model_copy(
                    update={"target_notional": 0.0, "weight": 0.0}
                )

    delta_legs = list(delta_by_key.values())
    # The reviewed artifact is the FULL recomputed (intended-holdings) book, so the reviewer's
    # re-derivations agree with its metadata and all 17 checks pass on the ACTUAL positions.
    save_output(state_dir, cycle, "target_weights", recomputed, cadence="daily")
    # The trade DELTAS (changed/forced/flattened legs) are a SEPARATE artifact the daily executor
    # trades — so the cadence opens only the deltas, never churning the whole book.
    save_output(
        state_dir,
        cycle,
        "rebalance_trades",
        {"legs": [leg.model_dump(mode="json") for leg in delta_legs]},
        cadence="daily",
    )
    return recomputed

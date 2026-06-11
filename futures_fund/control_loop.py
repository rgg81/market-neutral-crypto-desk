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

from futures_fund.contracts import CoinGeometry, SleeveSignal, TargetWeights
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

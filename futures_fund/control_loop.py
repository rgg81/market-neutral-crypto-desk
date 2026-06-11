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

from futures_fund.models import Cadence
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

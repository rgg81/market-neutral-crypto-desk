"""Execute-boundary step (`gate_execute_step`) — the paper-only seam W10/D7 cross.

Adapted from the weekly desk's `orchestration.gate_execute_step`, pared down to this repo's
market-neutral surface. The weekly version folds in pending-order revalidation, an
anti-hallucination audit, and a live executor; none of those modules exist here yet. This phase
(Task 4.5) only needs
the execute BOUNDARY: a callable with the contract signature `gate_execute_step(..., loop=...)` that
`gate_execute_cli.py` dispatches into. The reviewer precondition (`reviewer_gate_ok` HALT) is wired
in P5 Task 5.4 — it imports this step and gates it; this module deliberately stays the simple,
direction-agnostic seam so that wiring has a stable target.

PAPER-ONLY (`Settings.live` stays false forever): this records what WOULD execute; it never sends a
live order. `proposals` are the Trader's gate-ready per-leg opens; `management`/`triggers`/
`cancel_triggers` ride alongside (an explicit empty `management` is the stand-down contract).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from futures_fund.config import Settings


def gate_execute_step(
    exchange: Any,
    settings: Settings,
    state_dir: str,
    memory_dir: str,
    now: datetime,
    cycle_no: int,
    proposals: list[dict],
    *,
    management: list[dict] | None = None,
    triggers: list[dict] | None = None,
    cancel_triggers: list[dict] | None = None,
    loop: str = "weekly",
) -> dict:
    """Gate + record the Trader's per-leg opens for this cycle (paper boundary).

    Returns a `report` dict: `{"executed": [...], "dropped": [...], ...}`. The Trader does NO sizing
    (notional comes from the optimizer / `TargetWeights`), so each proposal is a pure entry/stop/TP
    envelope recorded verbatim. `loop` is the cadence (`weekly`/`daily`) — journaled for attribution
    so the due-gate and reflector can tell which cadence opened a leg.
    """
    proposals = list(proposals or [])
    return {
        "loop": loop,
        "cycle": cycle_no,
        "ran_at": now.isoformat(),
        "live": settings.live,
        "executed": proposals,
        "dropped": [],
        "management": list(management or []),
        "triggers": list(triggers or []),
        "cancel_triggers": list(cancel_triggers or []),
    }

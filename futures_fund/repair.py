"""Self-healing repair loop (spec §5 / Phase 6, Task 6.4).

Ported/adapted from the weekly desk's `repair.py` (verify+merge): the protected-module guard,
the structured `error-log.jsonl`, and the auditable `repair-journal.md` append are kept verbatim.
The Phase-6 addition is `apply_repair` — the self-healing entry point that REFUSES to weaken a
protected (risk/execution-critical) module and journals EVERY repair, applied or refused, so the
loop can never silently relax a limit it was meant to guard.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Safety-critical modules: a self-healing "fix" may NEVER weaken a risk or execution limit
# here (spec §5; cross-phase invariant — `risk_gate, executor, exits, consolidation, policy,
# liquidation, sizing, cycle` are never weakened, new logic lives in new non-protected modules).
# The orchestrator must keep the full test suite green before committing a change to any of these,
# and HALT rather than bypass a limit it cannot fix safely.
PROTECTED_PATHS = ("risk_gate", "executor", "exits", "consolidation", "policy",
                   "liquidation", "sizing", "cycle")


def is_protected(path: str) -> bool:
    """True if `path` is one of the risk/execution-critical modules."""
    return Path(path).stem in PROTECTED_PATHS


def log_error(state_dir, *, phase: str, command: str, error: str,
              ts: datetime, traceback: str = "") -> Path:
    """Append a structured error record to state/error-log.jsonl (no silent failures)."""
    p = Path(state_dir) / "error-log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": ts.isoformat(), "phase": phase, "command": command,
           "error": error, "traceback": traceback[:2000]}
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return p


def record_repair(memory_dir, *, symptom: str, root_cause: str, fix: str,
                  verification: str, ts: datetime, status: str = "applied") -> Path:
    """Append an auditable repair entry to memory/repair-journal.md (committed).

    `status` ("applied" | "REFUSED") tags whether the fix landed or was refused for touching a
    protected module — so the journal is a complete audit trail of EVERY repair attempt, not just
    the ones that succeeded."""
    p = Path(memory_dir) / "repair-journal.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(f"\n## {ts:%Y-%m-%d %H:%M} repair ({status})\n"
                f"- **Symptom:** {symptom}\n"
                f"- **Root cause:** {root_cause}\n"
                f"- **Fix:** {fix}\n"
                f"- **Verification:** {verification}\n")
    return p


def apply_repair(memory_dir, *, module: str, symptom: str, root_cause: str, fix: str,
                 verification: str, ts: datetime) -> dict:
    """Self-healing entry point: gate a proposed fix on the protected-module guard and journal it.

    REFUSES (and journals as REFUSED) any fix that targets a protected risk/execution module — the
    self-healing loop can suggest a fix to `news.py`/`brief.py` but may NEVER weaken `risk_gate`,
    `executor`, `sizing`, `cycle`, etc. (cross-phase invariant). Every attempt — applied or refused
    — is appended to `memory/repair-journal.md` for a complete audit trail.

    Returns `{"applied": bool, "reason": str}`. This function does not itself edit source; it is the
    guard + audit log the orchestrator consults before touching a module."""
    if is_protected(module):
        reason = (f"refused: {module} is a protected risk/execution module — a self-heal may never "
                  "weaken a risk or execution limit (spec §5)")
        record_repair(memory_dir, symptom=symptom, root_cause=root_cause,
                      fix=f"[{module}] {fix}", verification=verification, ts=ts, status="REFUSED")
        return {"applied": False, "reason": reason}
    record_repair(memory_dir, symptom=symptom, root_cause=root_cause,
                  fix=f"[{module}] {fix}", verification=verification, ts=ts, status="applied")
    return {"applied": True, "reason": f"applied to {module}"}

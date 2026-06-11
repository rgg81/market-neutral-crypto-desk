"""Execute-boundary CLI (Task 4.5) — gate + execute the Trader's proposals; persist + report.

    uv run python scripts/gate_execute_cli.py --cadence weekly --cycle N
    uv run python scripts/gate_execute_cli.py --cadence daily  --cycle N

The W10 / D7 stage of the SKILL.md ladders. Loads the cycle's `proposals.json` (the Trader's
gate-ready per-leg opens, plus management/triggers/cancel_triggers) from the SAME cadence-segmented
cycle root the due-gate scans (`state/<cadence>/cycle/<N>/`, CADENCE-ROOT INVARIANT), dispatches to
`gate_execute_step(..., loop=cadence)`, persists the resulting `report.json` under that same cadence
root, and prints it. `--cadence` is REQUIRED and selects both the artifact root and the journaled
loop attribution (`loop=cadence`).

PAPER-ONLY: `Settings.live` stays false forever. The reviewer precondition (`reviewer_gate_ok`
HALT) is wired in P5 Task 5.4, which gates this boundary BEFORE any execute; this CLI is the stable
target that wiring imports.

A missing/null `management` key must NEVER reach the step as `None` — that would close the whole
book by absence on a stand-down/HALT. We coerce to an empty review (keep holdings) and surface the
anomaly, mirroring the weekly desk's fail-safe.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund.config import load_settings
from futures_fund.cycle_io import load_output, save_output
from futures_fund.exchange import FuturesExchange
from futures_fund.models import Cadence
from futures_fund.orchestration import gate_execute_step
from futures_fund.reviewer import reviewer_gate_ok

_STATE_DIR = "state"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Gate + execute the Trader's proposals for one cadence cycle (paper boundary)."
    )
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--state-dir", default=_STATE_DIR)
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    # MANDATORY non-skippable reviewer stage (§10/§12): the every-cycle Adversarial Code & Calc
    # Reviewer must have written a PASSING `reviewer.json` (ReviewerVerdict.passed) under this
    # cadence cycle root BEFORE any fill. A missing/false verdict HALTs the execute boundary with
    # SystemExit(2) — absence HALTs just as hard as an explicit veto, so a skipped reviewer can
    # never let a book through. This precondition runs FIRST, before the exchange is even built.
    if not reviewer_gate_ok(args.state_dir, args.cycle, cadence):
        print(
            "HALT: reviewer gate not satisfied — no passing reviewer.json for "
            f"{cadence} cycle {args.cycle} (mandatory non-skippable stage, §10/§12). "
            "Run scripts/reviewer_cli.py first.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    settings = load_settings()
    ex = FuturesExchange.from_settings(settings)

    payload = load_output(args.state_dir, args.cycle, "proposals", cadence=cadence)
    # The agent path ALWAYS carries a holdings review (possibly empty). A missing/null `management`
    # key must NEVER reach the step as None — that would close the whole book by absence on a
    # stand-down/HALT. Coerce to an empty review (holdings KEPT) and surface the anomaly.
    if payload.get("management") is None:
        print(
            "WARNING: proposals.json has no 'management' key — treating as an empty holdings "
            "review (holdings KEPT, not closed by absence).",
            file=sys.stderr,
        )
    management = payload.get("management") or []
    triggers = payload.get("triggers") or []
    cancel_triggers = payload.get("cancel_triggers") or []  # retire decayed armed triggers

    now = datetime.now(UTC)  # gate-START instant
    report = gate_execute_step(
        ex,
        settings,
        args.state_dir,
        "memory",
        now,
        args.cycle,
        payload.get("proposals", []),
        management=management,
        triggers=triggers,
        cancel_triggers=cancel_triggers,
        loop=cadence,
    )
    # Persist under the SAME cadence root the due-gate scans, so the W10/D7 report.json the next
    # due_check reads lives exactly where it looks.
    save_output(args.state_dir, args.cycle, "report", report, cadence=cadence)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())

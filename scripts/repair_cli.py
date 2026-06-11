"""Self-healing repair CLI (SKILL.md Self-healing): gate a proposed fix on the protected-module
guard and journal it (applied or REFUSED) via `repair.apply_repair`.

    uv run python scripts/repair_cli.py --module cycle_prep --symptom ... --root-cause ... \\
        --fix ... --verification "uv run pytest -q"

Closes the MINOR repair-unwired gap. A fix to a protected risk/execution module is REFUSED and
journaled as REFUSED — the self-healing loop can never silently weaken a limit. Exit 0 always (the
guard decision is in the printed JSON `applied`); 2 only on an internal error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund.repair import apply_repair


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gate + journal a self-healing repair.")
    ap.add_argument("--module", required=True)
    ap.add_argument("--symptom", required=True)
    ap.add_argument("--root-cause", required=True)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--verification", required=True)
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args(argv)
    try:
        result = apply_repair(
            args.memory_dir, module=args.module, symptom=args.symptom,
            root_cause=args.root_cause, fix=args.fix, verification=args.verification,
            ts=datetime.now(UTC),
        )
    except Exception as e:  # noqa: BLE001 — surface, never crash the orchestrator silently
        print(f"ERROR: repair failed: {e!r}")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

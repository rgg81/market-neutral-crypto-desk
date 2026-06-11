"""Multi-cadence due-gate CLI (SKILL.md W2/D2). Run as the FIRST action each poll fire; prints ONE:

    DUE FRESH <N>   -> run a brand-new cycle end-to-end; create state/<cadence>/cycle/<N>/
    DUE RETRY <N>   -> a prior dir crashed before the gate; re-run/OVERWRITE that dir
    SKIP: <reason>  -> this candle is already served; stand down (liveness ping)
    ERROR: <reason> -> internal failure (exit 2); do NOT trade

    uv run python scripts/due_check.py state --loop weekly
    uv run python scripts/due_check.py state --loop daily

The first positional argument is the state root (SKILL.md passes the literal `state`); `--state-dir`
is also accepted for consistency with the rest of the CLI family. Routes through
`control_loop.cadence_due` so the candle width is cadence-correct (weekly=10080, daily=1440) and the
root scanned is `state/<cadence>/cycle/*` (CADENCE-ROOT INVARIANT). Exit 0 for DUE*/SKIP, 2 for
ERROR. Makes ZERO exchange/network calls and ZERO writes. Closes I1."""
from __future__ import annotations

import sys
from datetime import UTC, datetime


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    loop = None
    now_raw = None
    state_dir = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--loop" and i + 1 < len(argv):
            loop = argv[i + 1]
            i += 2
            continue
        if a == "--now" and i + 1 < len(argv):
            now_raw = argv[i + 1]
            i += 2
            continue
        if a == "--state-dir" and i + 1 < len(argv):
            state_dir = argv[i + 1]
            i += 2
            continue
        rest.append(a)
        i += 1
    if state_dir is None:
        state_dir = rest[0] if rest else "state"

    try:
        from futures_fund.control_loop import cadence_due
        if loop not in ("weekly", "daily"):
            print(f"ERROR: unknown loop {loop!r}; expected weekly|daily")
            return 2
        now = (datetime.fromisoformat(now_raw.replace("Z", "+00:00"))
               if now_raw else datetime.now(UTC))
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        mode, n, reason = cadence_due(state_dir, now, loop)
    except Exception as e:  # noqa: BLE001 — fail SAFE but visible
        print(f"ERROR: due_check failed before decision: {e!r}")
        return 2

    if mode in ("FRESH", "RETRY"):
        print(f"DUE {mode} {n}")
        print(reason)
        return 0
    print(f"SKIP: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

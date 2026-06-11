"""Apply a Reflector-decided lesson state change (Phase 6, Task 6.4). `confirm` is DSR-GATED: a
lesson only promotes CANDIDATE->VALIDATED when the desk's ALPHA edge is statistically proven.

    uv run python scripts/promote_lesson_cli.py --id <lesson_id> --action confirm|demote|retire

Adapted from the weekly desk's `promote_lesson_cli.py` (verify+merge). The DSR p-value is pulled
from `build_scorecard` (computed over the ALPHA series, return net of BTC-beta) and handed to
`statistically_promote`, which keeps the lesson CANDIDATE unless the edge clears the 0.95 DSR gate —
the statistical layer over the count-based promotion rule (spec §6). `demote`/`retire` are
unconditional state transitions.
"""
from __future__ import annotations

import argparse

from futures_fund.lessons import demote_lesson, retire_lesson, statistically_promote
from futures_fund.scorecard import build_scorecard


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Apply a lesson state change; `confirm` is DSR-gated on the alpha edge."
    )
    ap.add_argument("--id", required=True)
    ap.add_argument("--action", choices=["confirm", "demote", "retire"], required=True)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args(argv)
    if args.action == "confirm":
        dsr = build_scorecard(args.state_dir, args.memory_dir).get("dsr_pvalue", 0.0)
        ok = statistically_promote(args.memory_dir, args.id, dsr_pvalue=dsr)
    else:
        ok = {"demote": demote_lesson, "retire": retire_lesson}[args.action](
            args.memory_dir, args.id
        )
    print(f"{args.action} {args.id}: {'ok' if ok else 'not found'}")


if __name__ == "__main__":
    main()

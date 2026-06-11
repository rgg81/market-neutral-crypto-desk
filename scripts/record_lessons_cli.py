"""Deterministically persist the Reflector's lessons to the corpus (SKILL.md W11). The reflect
phase must ALWAYS append — never rely on the LLM Reflector to remember.

    uv run python scripts/record_lessons_cli.py --cycle N --cadence weekly

Reads this cycle's `lessons.json` (the Reflector agent's output) and appends each lesson via
`lessons.append_lesson` (validated against the `Lesson` contract). Closes I1 (record_lessons_cli.py
named in SKILL.md but absent). A missing artifact appends nothing (a cycle that minted no lessons is
fine). Each lesson dict supplies at least `text`; `ts` is stamped here.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund.cycle_io import load_output
from futures_fund.lessons import append_lesson
from futures_fund.models import Cadence


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Append the Reflector's lessons to the corpus (W11).")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    try:
        payload = load_output(args.state_dir, args.cycle, "lessons", cadence=cadence)
    except FileNotFoundError:
        payload = {}
    raw = payload.get("lessons", []) if isinstance(payload, dict) else (payload or [])

    now = datetime.now(UTC)
    ids: list[str] = []
    for fields in raw:
        if isinstance(fields, dict) and fields.get("text"):
            ids.append(append_lesson(args.memory_dir, fields, ts=now))
    print(json.dumps({"cycle": args.cycle, "cadence": cadence, "appended": len(ids),
                      "lesson_ids": ids}, default=str))


if __name__ == "__main__":
    sys.exit(main())

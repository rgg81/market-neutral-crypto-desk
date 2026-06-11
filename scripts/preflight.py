"""Preflight CLI (SKILL.md W3): fold every HELD symbol into the scout universe and emit per-symbol
briefs as the analysts' context -> `context.json`.

    uv run python scripts/preflight.py --cycle N --cadence weekly

Closes I1 (preflight.py named in SKILL.md but absent). Held symbols are resolved from the most
recent cadence cycle's executed `report.json` legs (a held leg must stay in the universe so the desk
can audit/close it even if it dropped out of the top-by-volume scan). Pure read + light assembly;
the analysts (LLM) reason over the briefs downstream.
"""
from __future__ import annotations

import argparse
import json
import sys

from futures_fund.control_loop import latest_cadence_cycle
from futures_fund.cycle_io import cycle_dir, load_output, save_output
from futures_fund.models import Cadence


def _held_symbols(state_dir, cadence: Cadence) -> list[str]:
    """Symbols from the most recent executed report's legs (the book currently held)."""
    n = latest_cadence_cycle(state_dir, cadence, "report")
    if n is None:
        return []
    path = cycle_dir(state_dir, n, cadence=cadence) / "report.json"
    try:
        executed = json.loads(path.read_text()).get("executed", [])
    except (OSError, json.JSONDecodeError):
        return []
    return [e["symbol"] for e in executed if isinstance(e, dict) and e.get("symbol")]


def build_briefs(universe: list[dict], held: list[str]) -> list[dict]:
    """One brief per symbol in (universe ∪ held). `held=True` flags positions the book carries so
    the analysts always audit them (even if they fell out of the volume-ranked scan)."""
    held_set = set(held)
    syms: list[str] = []
    for row in universe:
        s = row.get("symbol")
        if s and s not in syms:
            syms.append(s)
    for s in held:
        if s not in syms:
            syms.append(s)
    return [{"symbol": s, "held": s in held_set} for s in syms]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fold held symbols + build per-symbol briefs (W3).")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default="state")
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    try:
        universe = load_output(args.state_dir, args.cycle, "universe", cadence=cadence)["universe"]
    except FileNotFoundError:
        universe = []
    held = _held_symbols(args.state_dir, cadence)
    ctx = {"briefs": build_briefs(universe, held), "held": held}
    save_output(args.state_dir, args.cycle, "context", ctx, cadence=cadence)
    print(json.dumps(ctx, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())

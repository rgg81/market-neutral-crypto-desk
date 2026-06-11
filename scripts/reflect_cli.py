"""Reflection CLI (Phase 6, Task 6.4): emit the winners/losers payload for the Reflector subagent.

    uv run python scripts/reflect_cli.py --cadence weekly --cycle N
    uv run python scripts/reflect_cli.py --cadence daily  --cycle N   # light reflect

Adapted from the weekly desk's `reflect_cli.py` (verify+merge), re-keyed on ALPHA vs BTC-beta: the
neutral desk grades the spread, not raw P&L (spec §10). This builds the cycle's
`reflection_input.json` — closed decisions split into `winners`/`losers` by realized **alpha**
(`AlphaOutcome.alpha_return`), each carrying its journaled thesis + alpha/beta attribution — and
persists it under the cadence-segmented cycle root (`state/<cadence>/cycle/<N>/`, CADENCE INVARIANT)
for the Reflector agent (`agents/reflector.md`) to reason over, then prints it. The Reflector writes
`lessons.json`, which `record_lessons_cli.py` (in the SKILL.md ladder) deterministically appends to
the corpus via `lessons.append_lesson` — the reflect phase must ALWAYS persist, not rely on the LLM
to remember.
"""
from __future__ import annotations

import argparse
import json

from futures_fund.cycle_io import save_output
from futures_fund.journal import alpha_outcome, read_all_decisions
from futures_fund.models import Cadence


def build_reflection_input(memory_dir) -> dict:
    """Split closed decisions into winners/losers by realized ALPHA (return net of BTC-beta).

    A decision is 'closed' once its six alpha-vs-beta outcome fields have been patched (so
    `alpha_outcome` validates); decisions still open (or only partially patched) are skipped. Each
    entry carries the leg's identity + thesis context + the typed alpha/beta attribution so the
    Reflector can contrast what worked against what didn't, keyed on the spread."""
    winners: list[dict] = []
    losers: list[dict] = []
    for d in read_all_decisions(memory_dir):
        try:
            ao = alpha_outcome(d)
        except KeyError:
            continue  # not yet closed (outcome not fully patched) — skip
        entry = {
            "decision_id": d.get("id"),
            "cycle": d.get("cycle"),
            "symbol": d.get("symbol"),
            "direction": d.get("direction"),
            "regime": d.get("regime"),
            "setup": d.get("setup"),
            "rationale": d.get("rationale"),
            "r_multiple": d.get("r_multiple"),
            "alpha_return": ao.alpha_return,
            "beta_contribution": ao.beta_contribution,
            "pair_cointegrated_at_exit": ao.pair_cointegrated_at_exit,
            "funding_thesis_matched": ao.funding_thesis_matched,
            "neutrality_in_band": ao.neutrality_in_band,
            "sentiment_helped": ao.sentiment_helped,
        }
        (winners if ao.alpha_return > 0 else losers).append(entry)
    return {"winners": winners, "losers": losers,
            "n_closed": len(winners) + len(losers)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Emit the reflection_input.json (winners/losers by alpha) for the Reflector."
    )
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence
    payload = build_reflection_input(args.memory_dir)
    save_output(args.state_dir, args.cycle, "reflection_input", payload, cadence=cadence)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

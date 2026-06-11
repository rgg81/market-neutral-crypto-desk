"""Every-cycle Adversarial Code & Calc Reviewer CLI (§10 Guardian, §12) — Task 5.4.

    uv run python scripts/reviewer_cli.py --cadence weekly --cycle N
    uv run python scripts/reviewer_cli.py --cadence daily  --cycle N

The W9 / D6 MANDATORY non-skippable stage of the SKILL.md ladders, run BEFORE the W10/D7 execute
boundary. Loads the cycle's artifacts (`target_weights.json`, `geometries.json`, and — when present
— `spreads.json` / `sentiment.json`) from the SAME cadence-segmented cycle root the due-gate scans
(`state/<cadence>/cycle/<N>/`, CADENCE-ROOT INVARIANT), re-derives every load-bearing number from
ground truth via `reviewer.review_cycle` (the AND of all 17 canonical checks), persists the
resulting `reviewer.json` under that same cadence root, prints it, and — HARD VETO — exits 2 when
`ReviewerVerdict.passed` is false. `gate_execute_cli.py` then reads that persisted flag via
`reviewer_gate_ok` and refuses to fill unless this stage wrote a passing verdict.

Fail-closed: exits 2 if the artifacts the reviewer ACTUALLY audits (`target_weights`, `geometries`)
are missing — the reviewer never green-lights a cycle whose inputs it could not read.
"""
from __future__ import annotations

import argparse
import json
import sys

from futures_fund.config import Settings, load_settings
from futures_fund.contracts import (
    GeometryBundle,
    SentimentBatch,
    Spread,
    TargetWeights,
)
from futures_fund.cycle_io import load_output, save_output
from futures_fund.models import Cadence
from futures_fund.neutrality import NeutralityConfig
from futures_fund.reviewer import review_cycle

_STATE_DIR = "state"


def _neutrality_config(settings: Settings) -> NeutralityConfig:
    """Hydrate the P1 `NeutralityConfig` from `settings.neutrality` (defaults when empty)."""
    return NeutralityConfig(**(settings.neutrality or {}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Run the every-cycle adversarial reviewer for one cadence cycle (W9/D6)."
    )
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--state-dir", default=_STATE_DIR)
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    settings = load_settings()
    cfg = _neutrality_config(settings)

    # Required artifacts the reviewer audits — fail closed if absent (never bless an empty cycle).
    try:
        target = TargetWeights.model_validate(
            load_output(args.state_dir, args.cycle, "target_weights", cadence=cadence)
        )
        geometries = GeometryBundle.model_validate(
            load_output(args.state_dir, args.cycle, "geometries", cadence=cadence)
        ).geometries
    except FileNotFoundError as exc:
        print(f"HALT: missing reviewer input artifact: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    # Optional artifacts: a cycle with no pair book / no sentiment read still reviews cleanly.
    try:
        spreads = [
            Spread.model_validate(s)
            for s in load_output(args.state_dir, args.cycle, "spreads", cadence=cadence)["spreads"]
        ]
    except FileNotFoundError:
        spreads = []
    try:
        sentiment = SentimentBatch.model_validate(
            load_output(args.state_dir, args.cycle, "sentiment", cadence=cadence)
        ).reports
    except FileNotFoundError:
        sentiment = []

    verdict = review_cycle(
        args.state_dir,
        args.memory_dir,
        cycle=args.cycle,
        cadence=cadence,
        target=target,
        geometries=geometries,
        spreads=spreads,
        sentiment=sentiment,
        cfg=cfg,
        returns=None,
    )
    # Persist under the SAME cadence root the execute boundary's `reviewer_gate_ok` reads.
    save_output(args.state_dir, args.cycle, "reviewer", verdict, cadence=cadence)
    print(json.dumps(verdict.model_dump(), indent=2, default=str))

    # HARD VETO: a failed verdict HALTs the cadence here (the execute boundary also refuses fills).
    if not verdict.passed:
        print(
            f"HALT: reviewer verdict FAILED ({', '.join(verdict.mismatches)})",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())

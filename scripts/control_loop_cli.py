"""Two-cadence control-loop entrypoint (§9): run one weekly Selection or daily Rebalance meeting.

    uv run python scripts/control_loop_cli.py --cadence weekly --cycle 1
    uv run python scripts/control_loop_cli.py --cadence daily  --cycle 1

Loads the cycle's upstream geometry + sleeve artifacts from the SAME cadence-segmented cycle root
the due-gate reads (`state/<cadence>/cycle/<N>/`, CADENCE-ROOT INVARIANT), dispatches to
`control_loop.weekly_selection` / `daily_rebalance`, and prints the resulting `TargetWeights` as
JSON (the Trader's hand-off). Fail-closed: exits 2 if the upstream sleeve/geometry artifacts the
meeting needs are missing — the loop never runs on absent inputs.
"""
from __future__ import annotations

import argparse
import json
import sys

from futures_fund.config import Settings, load_settings
from futures_fund.contracts import GeometryBundle, SleeveSignal, Spread, TargetWeights
from futures_fund.control_loop import (
    daily_rebalance,
    latest_cadence_cycle,
    weekly_selection,
)
from futures_fund.cycle_io import load_output
from futures_fund.models import Cadence
from futures_fund.neutrality import NeutralityConfig

_STATE_DIR = "state"


def _neutrality_config(settings: Settings) -> NeutralityConfig:
    """Hydrate the P1 `NeutralityConfig` from `settings.neutrality`.

    Uses `NeutralityConfig` defaults when the config block is empty."""
    return NeutralityConfig(**(settings.neutrality or {}))


def _load_inputs(
    state_dir: str, cycle: int, cadence: Cadence
) -> tuple[GeometryBundle, list[SleeveSignal]]:
    """Load the cycle's geometry + sleeve artifacts, or fail closed (`SystemExit(2)`).

    Both live under `state/<cadence>/cycle/<N>/` — the SAME root the due-gate scans — so the loop
    reads exactly what an upstream geometry/sleeve build wrote there. A missing artifact means the
    upstream stage has not produced this cycle's inputs yet; the meeting MUST NOT run on partial or
    absent inputs, so we exit 2 rather than silently optimizing an empty book."""
    try:
        bundle = GeometryBundle.model_validate(
            load_output(state_dir, cycle, "geometries", cadence=cadence)
        )
        raw_sleeves = load_output(state_dir, cycle, "sleeves", cadence=cadence)
    except FileNotFoundError as exc:
        raise SystemExit(2) from exc
    sleeves = [SleeveSignal.model_validate(s) for s in raw_sleeves["sleeves"]]
    return bundle, sleeves


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run one weekly Selection / daily Rebalance meeting of the control loop."
    )
    parser.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--state-dir", default=_STATE_DIR)
    args = parser.parse_args(argv)
    cadence: Cadence = args.cadence

    settings = load_settings()
    cfg = _neutrality_config(settings)
    equity = settings.account_size_usdt
    bundle, sleeves = _load_inputs(args.state_dir, args.cycle, cadence)

    if cadence == "weekly":
        # Carry-over (§9): seed the optimizer's no-trade band with the prior weekly book when one
        # exists (so only the deltas are traded), else a clean re-selection.
        prior: TargetWeights | None = None
        if args.cycle > 1:
            try:
                prior = TargetWeights.model_validate(
                    load_output(args.state_dir, args.cycle - 1, "target_weights", cadence="weekly")
                )
            except FileNotFoundError:
                prior = None
        result = weekly_selection(
            args.state_dir,
            bundle.geometries,
            sleeves,
            equity=equity,
            prior=prior,
            cfg=cfg,
            cycle=args.cycle,
        )
    else:
        # Daily Rebalance keeps the SAME symbol set as the MOST RECENT weekly target. Weekly and
        # daily cycle counters are INDEPENDENT (each cadence's due-gate scans its own root and
        # daily increments ~7x faster), so the daily `args.cycle` does NOT index the matching weekly
        # cycle — resolve the highest weekly cycle that actually persisted a target_weights book
        # instead. Fail closed if no weekly target exists yet (no fixed set to rebalance toward).
        weekly_cycle = latest_cadence_cycle(args.state_dir, "weekly", "target_weights")
        if weekly_cycle is None:
            raise SystemExit(2)
        target = TargetWeights.model_validate(
            load_output(args.state_dir, weekly_cycle, "target_weights", cadence="weekly")
        )
        try:
            raw_spreads = load_output(args.state_dir, args.cycle, "spreads", cadence="daily")
            spreads = [Spread.model_validate(s) for s in raw_spreads["spreads"]]
        except FileNotFoundError:
            spreads = []
        result = daily_rebalance(
            args.state_dir,
            target,
            bundle.geometries,
            spreads=spreads,
            equity=equity,
            cfg=cfg,
            cycle=args.cycle,
        )

    print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())

"""Task 7.3 — end-to-end paper run driver (W/D SKILL.md ladders, serialized weekly-first).

    uv run python scripts/run_paper_cli.py
    uv run python scripts/run_paper_cli.py --now 2026-06-11T00:00:00+00:00   # pinned (offline)

Runs a FULL weekly->daily loop on the paper desk, end-to-end, under EXACTLY ONE run lock
(`runlock.single_flight(owner="paper")`). The two cadences are serialized WEEKLY-FIRST so the daily
Rebalance always tracks the freshest weekly book. Per cadence the driver walks the ladder seams,
each naming its exact reused CLI/function — the driver is deterministic glue, NOT new logic:

  * Step 3a  lock + due   `runlock.single_flight` then `control_loop.cadence_due` (SKIP -> continue,
                          FRESH/RETRY -> proceed with cycle n)
  * Step 4a  cadence step `control_loop_cli.main(["--cadence", cadence, "--cycle", n])`
                          -> persists `target_weights.json` under `state/<cadence>/cycle/<n>/`
  * (hand-off) derive the Trader's `proposals.json` from the persisted book's legs (no sizing — the
                          notional already comes from the optimizer / `TargetWeights`)
  * Step 5a  reviewer     `reviewer_cli.main([...])` -> `reviewer.json`; a failed verdict HALTs the
                          cadence with `SystemExit(2)` (hard veto, mandatory non-skippable stage)
  * Step 6a  execute      `gate_execute_cli.main([...])` -> `report.json` (the execute boundary also
                          re-checks `reviewer_gate_ok` and refuses fills without a passing verdict)
  * Step 7a  equity       `equity_log.record_equity(state_dir, now, equity, n)` -> equity point
  * Step 7a  reflect      `reflect_cli.main([...])` (light on daily) -> `reflection_input.json`

PAPER-ONLY: `Settings.live` stays false forever; the execute boundary records what WOULD fill and
never sends a live order. The exchange is built via `FuturesExchange.from_settings` inside
`gate_execute_cli` (injected/faked in the e2e test), so this driver runs fully offline.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund import equity_log, runlock
from futures_fund.config import load_settings
from futures_fund.contracts import TargetWeights, WeightLeg
from futures_fund.control_loop import cadence_due, latest_cadence_cycle
from futures_fund.cycle_io import cycle_dir, load_output, save_output
from futures_fund.models import Cadence
from scripts.cycle_prep_cli import main as cycle_prep_main
from scripts.scout_cli import main as scout_main

_STATE_DIR = "state"
_CADENCES: tuple[Cadence, ...] = ("weekly", "daily")  # serialized WEEKLY-FIRST


def _parse_now(raw: str | None) -> datetime:
    """Resolve the run instant (tz-aware UTC). `--now` pins it for deterministic/offline runs;
    absent, use wall-clock UTC. A naive timestamp is interpreted as UTC."""
    if raw is None:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _run_control_loop_step(state_dir, cadence: Cadence, cycle: int) -> None:
    """Step 4a — dispatch the cadence meeting via the control-loop CLI, persisting
    `target_weights.json` under `state/<cadence>/cycle/<cycle>/`. A seam (monkeypatched in tests) so
    the driver's ladder can be exercised without re-running the optimizer."""
    from scripts.control_loop_cli import main as control_loop_main

    control_loop_main(["--cadence", cadence, "--cycle", str(cycle), "--state-dir", str(state_dir)])


def _apply_deltas(base: TargetWeights, deltas: list[WeightLeg]) -> TargetWeights:
    """Apply a daily delta book onto the carry-over weekly target to get the HELD book.

    Each delta keyed on `(symbol, direction)` REPLACES the base leg (a zero-notional delta flattens
    that leg -> it leaves the held book). The result carries the recomputed neutrality/deployment
    metadata of `base` (the maintained, fully-deployed weekly target) so the reviewer audits the
    actual positions held, not the sparse delta the executor trades."""
    by_key = {(leg.symbol, leg.direction): leg for leg in base.legs}
    for d in deltas:
        if abs(d.target_notional) <= 0.0:
            by_key.pop((d.symbol, d.direction), None)  # flatten -> leg leaves the held book
        else:
            by_key[(d.symbol, d.direction)] = d
    return base.model_copy(update={"legs": list(by_key.values())})


def _resolve_held_book(state_dir, cadence: Cadence, cycle: int) -> TargetWeights:
    """The book the reviewer audits + the executor opens for this cadence cycle.

    WEEKLY: the produced `target_weights.json` IS the full held book. DAILY: the persisted
    `target_weights.json` is the sparse DELTA book (`daily_rebalance`'s "trade only the deltas"
    contract); the actual HELD book is the carry-over weekly target the daily cadence tracks with
    those deltas applied. We resolve the held book here (driver-level orchestration — no protected
    module is touched) and re-persist it as this cycle's `target_weights.json` so the reviewer's
    deployment-floor / neutrality re-derivation sees the real positions, not the empty no-churn
    delta. Falls back to the delta book if no weekly target exists (the reviewer then fails closed,
    as it should)."""
    book = TargetWeights.model_validate(
        load_output(state_dir, cycle, "target_weights", cadence=cadence)
    )
    if cadence == "weekly":
        return book
    weekly_cycle = latest_cadence_cycle(state_dir, "weekly", "target_weights")
    if weekly_cycle is None:
        return book
    weekly_target = TargetWeights.model_validate(
        load_output(state_dir, weekly_cycle, "target_weights", cadence="weekly")
    )
    held = _apply_deltas(weekly_target, book.legs)
    save_output(state_dir, cycle, "target_weights", held, cadence=cadence)
    return held


def _proposals_from_book(book: TargetWeights) -> list[dict]:
    """Derive the Trader's gate-ready per-leg opens from the persisted book's legs (the hand-off).

    The Trader does NO sizing — the notional already comes from the optimizer / `TargetWeights` — so
    each non-flat alpha/hedge leg becomes a market-entry proposal carrying its symbol/direction and
    target notional. Zero-notional legs (carry-over unwinds / flattens) are excluded: there is
    nothing to OPEN there."""
    proposals: list[dict] = []
    for leg in book.legs:
        if abs(leg.target_notional) <= 0.0:
            continue
        proposals.append({
            "symbol": leg.symbol,
            "direction": leg.direction,
            "target_notional": abs(leg.target_notional),
            "trigger_type": "market",
            "rationale": f"{leg.sleeve} leg (optimizer-sized)",
        })
    return proposals


def _write_proposals(state_dir, cadence: Cadence, cycle: int, book: TargetWeights) -> int:
    """Derive + persist `proposals.json` (from the resolved HELD book) under the same cadence cycle
    root the execute boundary loads from, and return the leg count."""
    proposals = _proposals_from_book(book)
    save_output(
        state_dir, cycle, "proposals",
        {"proposals": proposals, "management": [], "triggers": [], "cancel_triggers": []},
        cadence=cadence,
    )
    return len(proposals)


def _run_reviewer(state_dir, cadence: Cadence, cycle: int, memory_dir) -> None:
    """Step 5a — the every-cycle adversarial reviewer (W9/D6). HARD VETO: a failed verdict HALTs the
    cadence here with `SystemExit(2)` (the CLI raises it); the execute boundary independently
    refuses to fill without a passing verdict via `reviewer_gate_ok`."""
    from scripts.reviewer_cli import main as reviewer_main

    reviewer_main([
        "--cadence", cadence, "--cycle", str(cycle),
        "--state-dir", str(state_dir), "--memory-dir", str(memory_dir),
    ])


def _run_execute(state_dir, cadence: Cadence, cycle: int) -> None:
    """Step 6a — the gate+execute boundary (W10/D7) -> `report.json`. Re-checks `reviewer_gate_ok`
    and HALTs (`SystemExit(2)`) if no passing verdict exists for this cadence cycle."""
    from scripts.gate_execute_cli import main as gate_execute_main

    gate_execute_main(["--cadence", cadence, "--cycle", str(cycle), "--state-dir", str(state_dir)])


def _run_reflect(state_dir, cadence: Cadence, cycle: int, memory_dir) -> None:
    """Step 7a — emit the reflection input (winners/losers by alpha) for the Reflector (light on
    daily). Best-effort: reflection is a learning artifact, never a fill precondition, so a
    reflect-time error must not unwind a successfully executed cycle."""
    from scripts.reflect_cli import main as reflect_main

    try:
        reflect_main([
            "--cadence", cadence, "--cycle", str(cycle),
            "--state-dir", str(state_dir), "--memory-dir", str(memory_dir),
        ])
    except Exception as exc:  # noqa: BLE001 — reflection is non-fatal post-execute bookkeeping
        print(f"WARNING: reflect step failed for {cadence} cycle {cycle}: {exc!r}", file=sys.stderr)


def _run_producers(state_dir, cadence: Cadence, cycle: int, now: datetime) -> None:
    """Step 3b — scout the universe then build the cycle's upstream artifacts (geometries / sleeves
    / pairs / spreads) BEFORE the control-loop step consumes them. Closes C1: the loop no longer
    depends on a hand-seeded `_seed_upstream`. Both CLIs are seams (monkeypatched in tests) so the
    driver's ladder runs offline against a faked exchange. Idempotent on RETRY (overwrites the
    cycle's artifacts in place)."""
    scout_main(["--cycle", str(cycle), "--cadence", cadence, "--state-dir", str(state_dir)])
    cycle_prep_main([
        "--cycle", str(cycle), "--cadence", cadence, "--state-dir", str(state_dir),
        "--now", now.isoformat(),
    ])


def _run_cadence(
    cadence: Cadence, state_dir, memory_dir, now: datetime, equity: float
) -> bool:
    """Run ONE cadence end-to-end (Steps 4a-7a). Returns True if the cadence executed a cycle, False
    if its current candle was already served (SKIP — Step 3a). HALTs (`SystemExit(2)`) on a reviewer
    veto or a missing execute precondition. Called under the single run lock by `main`."""
    mode, cycle, _reason = cadence_due(state_dir, now, cadence)
    if mode == "SKIP":
        return False  # candle already served -> stand down (no re-run)

    # Step 3b — producers: scout + cycle-prep write geometries/sleeves/pairs/spreads.
    _run_producers(state_dir, cadence, cycle, now)
    # Step 4a — cadence step: persist target_weights.json under state/<cadence>/cycle/<cycle>/.
    _run_control_loop_step(state_dir, cadence, cycle)
    # Resolve the HELD book the reviewer audits + the executor opens (daily applies its sparse delta
    # onto the carry-over weekly target it tracks; weekly is already the full book).
    held = _resolve_held_book(state_dir, cadence, cycle)
    # Hand-off — derive proposals.json from the held book's legs.
    _write_proposals(state_dir, cadence, cycle, held)
    # Step 5a — reviewer gate (HARD VETO -> SystemExit(2) on a failed verdict).
    _run_reviewer(state_dir, cadence, cycle, memory_dir)
    # Step 6a — execute boundary (re-checks reviewer_gate_ok) -> report.json.
    _run_execute(state_dir, cadence, cycle)
    # Step 7a — equity point (the dashboard's return-series source) + reflect.
    equity_log.record_equity(state_dir, now, equity, cycle)
    _run_reflect(state_dir, cadence, cycle, memory_dir)
    return True


def _read_executed(state_dir, cadence: Cadence, cycle: int) -> list:
    """Best-effort read of a cadence cycle's executed report legs (for the run summary)."""
    try:
        path = cycle_dir(state_dir, cycle, cadence=cadence) / "report.json"
        return json.loads(path.read_text()).get("executed", [])
    except (OSError, json.JSONDecodeError):
        return []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end paper run: weekly+daily cadences, weekly-first, under one lock."
    )
    ap.add_argument("--now", default=None, help="ISO-8601 run instant (UTC); default wall-clock.")
    ap.add_argument("--state-dir", default=_STATE_DIR)
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args(argv)

    now = _parse_now(args.now)
    settings = load_settings()
    equity = settings.account_size_usdt  # PAPER-ONLY: live stays false forever.

    # Step 3a + Step 8 — the WHOLE weekly+daily run is serialized under ONE run lock. A concurrent
    # fire sees the lock held and stands down; a crash auto-heals after the stale window.
    with runlock.single_flight(args.state_dir, now, owner="paper") as ok:
        if not ok:
            print("STAND DOWN: another paper run holds the lock; skipping this fire.")
            return
        summary: dict[str, object] = {"ran_at": now.isoformat(), "live": settings.live,
                                      "cadences": {}}
        for cadence in _CADENCES:  # WEEKLY-FIRST, serialized
            ran = _run_cadence(cadence, args.state_dir, args.memory_dir, now, equity)
            mode, cycle, _reason = cadence_due(args.state_dir, now, cadence)
            summary["cadences"][cadence] = {  # type: ignore[index]
                "ran": ran,
                "cycle": cycle if ran else None,
            }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())

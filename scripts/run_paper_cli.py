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
from futures_fund.account import CostInputs, load_account, save_account
from futures_fund.config import load_settings
from futures_fund.contracts import TargetWeights, WeightLeg
from futures_fund.control_loop import cadence_due
from futures_fund.cycle_io import load_output, save_output
from futures_fund.journal import patch_outcome
from futures_fund.learning import close_alpha_outcomes, journal_open_legs
from futures_fund.models import Cadence
from futures_fund.pnl_attribution import append_ledger, build_cycle_pnl
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


def _reviewed_book(state_dir, cadence: Cadence, cycle: int) -> TargetWeights:
    """The full intended-holdings book the reviewer audits for this cadence cycle.

    For BOTH cadences the persisted `target_weights.json` IS the full, neutral, hedge-correct,
    fully-deployed book the optimizer produced — weekly from `weekly_selection`, daily from
    `daily_rebalance` (which now persists the FULL recomputed book, not the sparse delta). So the
    reviewer's hedge/dollar/beta/deployment/cap re-derivations agree with the artifact's metadata
    and all 17 checks validate the ACTUAL resulting positions. No re-persist / delta-application is
    needed."""
    return TargetWeights.model_validate(
        load_output(state_dir, cycle, "target_weights", cadence=cadence)
    )


def _trade_legs(state_dir, cadence: Cadence, cycle: int, book: TargetWeights) -> list[WeightLeg]:
    """The legs the executor actually OPENS this cadence cycle (separate from what the reviewer
    audits).

    WEEKLY: the full book is opened — every leg of `target_weights.json`. DAILY: only the TRADE
    DELTAS (`rebalance_trades.json` — the changed / breach-forced / z-stop-flattened legs
    `daily_rebalance` persisted) are traded, so the daily cadence nudges the book toward target
    without churning the whole (mostly-unchanged) book. If the daily deltas artifact is missing
    (legacy / fail-soft) the cadence opens nothing rather than re-trading the full book."""
    if cadence != "daily":
        return list(book.legs)
    try:
        raw = load_output(state_dir, cycle, "rebalance_trades", cadence=cadence)
    except FileNotFoundError:
        return []
    return [WeightLeg.model_validate(leg) for leg in raw["legs"]]


def _proposals_from_legs(legs: list[WeightLeg]) -> list[dict]:
    """Derive the Trader's gate-ready per-leg opens from the legs to TRADE this cycle (the
    hand-off).

    The Trader does NO sizing — the notional already comes from the optimizer / `TargetWeights` — so
    each non-flat alpha/hedge leg becomes a market-entry proposal carrying its symbol/direction and
    target notional. Zero-notional legs (carry-over unwinds / z-stop flattens) are excluded: there
    is nothing to OPEN there."""
    proposals: list[dict] = []
    for leg in legs:
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


def _write_proposals(state_dir, cadence: Cadence, cycle: int, legs: list[WeightLeg]) -> int:
    """Derive + persist `proposals.json` (from the legs to TRADE this cycle) under the same cadence
    cycle root the execute boundary loads from, and return the leg count."""
    proposals = _proposals_from_legs(legs)
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


def _run_producers(
    state_dir, cadence: Cadence, cycle: int, now: datetime, memory_dir="memory"
) -> None:
    """Step 3b — scout the universe then build the cycle's upstream artifacts (geometries / sleeves
    / pairs / spreads) BEFORE the control-loop step consumes them. Closes C1: the loop no longer
    depends on a hand-seeded `_seed_upstream`. Both CLIs are seams (monkeypatched in tests) so the
    driver's ladder runs offline against a faked exchange. Idempotent on RETRY (overwrites the
    cycle's artifacts in place). `memory_dir` lets cycle-prep read the lessons corpus for the
    READ-BACK overlay (link 4) — a no-op until the desk has learned a lesson."""
    scout_main(["--cycle", str(cycle), "--cadence", cadence, "--state-dir", str(state_dir)])
    cycle_prep_main([
        "--cycle", str(cycle), "--cadence", cadence, "--state-dir", str(state_dir),
        "--memory-dir", str(memory_dir), "--now", now.isoformat(),
    ])


def _run_cadence(
    cadence: Cadence, state_dir, memory_dir, now: datetime, equity: float,
    btc_symbol: str = "BTC/USDT:USDT",
) -> bool:
    """Run ONE cadence end-to-end (Steps 4a-7a). Returns True if the cadence executed a cycle, False
    if its current candle was already served (SKIP — Step 3a). HALTs (`SystemExit(2)`) on a reviewer
    veto or a missing execute precondition. Called under the single run lock by `main`."""
    mode, cycle, _reason = cadence_due(state_dir, now, cadence)
    if mode == "SKIP":
        return False  # candle already served -> stand down (no re-run)

    # Step 3b — producers: scout + cycle-prep write geometries/sleeves/pairs/spreads.
    _run_producers(state_dir, cadence, cycle, now, memory_dir)
    # Step 4a — cadence step: persist target_weights.json under state/<cadence>/cycle/<cycle>/. For
    # daily this is the FULL recomputed (intended-holdings) book PLUS a separate
    # rebalance_trades.json carrying the sparse trade deltas.
    _run_control_loop_step(state_dir, cadence, cycle)
    # The reviewer audits the FULL intended-holdings book (target_weights.json) for both cadences.
    reviewed = _reviewed_book(state_dir, cadence, cycle)
    # Hand-off — the executor opens only the legs to TRADE this cycle: the full book weekly, the
    # sparse rebalance_trades deltas daily (daily must NOT churn the whole, mostly-unchanged book).
    _write_proposals(state_dir, cadence, cycle, _trade_legs(state_dir, cadence, cycle, reviewed))
    # Step 5a — reviewer gate (HARD VETO -> SystemExit(2) on a failed verdict).
    _run_reviewer(state_dir, cadence, cycle, memory_dir)
    # Step 6a — execute boundary (re-checks reviewer_gate_ok) -> report.json.
    _run_execute(state_dir, cadence, cycle)
    # Step 7a — REALISTIC P&L: load the account, settle funding since the account's OWN funding
    # clock (NOT the cycle-collided equity series), reconcile the ACCOUNT to the FULL intended book
    # (`reviewed.legs` — the consolidated, neutral, hedge-correct book the reviewer validated), NOT
    # the sparse execution deltas. apply_fills computes delta = target - current internally and
    # charges frictions only on the delta, so feeding the full book each cycle trades ONLY what
    # changed (unchanged symbols -> delta 0 -> no-op; no churn) while keeping the HELD book correct
    # and market-neutral. A symbol DROPPED from the new book is FLATTENED (apply_fills closes any
    # held symbol absent from the fed legs). The gate/report.json/rebalance_trades.json execution
    # record is unchanged — that stays the "what we traded this cycle" view; only the ACCOUNT feed
    # changed. Then mark-to-market, record the REAL equity (replaces the old flat
    # settings.account_size_usdt), write pnl.json + ledger, patch each CLOSED leg's realized costs
    # onto the journal Decision that opened it, then save the account. settle_funding runs BEFORE
    # apply_fills so a position opened this cycle earns no funding for a pre-existence window
    # (Task 8a pins this).
    account = load_account(state_dir, equity)  # `equity` is the default cash on a cold dir
    bundle = _load_geometries(state_dir, cadence, cycle)
    marks, funding_by_symbol, intervals, costs = _geometry_cost_maps(bundle)
    prev_ts = account.last_funding_ts or now             # the per-account funding clock
    opening_equity = account.equity(marks)
    account.settle_funding(prev_ts, now, funding_by_symbol, intervals, marks)
    # LOUD GUARD: every non-flat leg of the FULL intended book must be priced before we reconcile —
    # a non-flat leg with no mark would be silently skipped by apply_fills, leaving the HELD book
    # non-neutral (the live BTC-hedge bug). Fail loudly here instead of producing a broken book.
    _assert_legs_priced(reviewed, marks)
    intended = _intended_fills(reviewed)
    account.apply_fills(
        intended, marks, costs,
        opened_ts=now, opened_cycle=cycle, opened_cadence=cadence,
    )
    # LEARNING link 1 — JOURNAL AT ENTRY. Record a Phase-1 Decision for every currently-held leg
    # (idempotent per leg, keyed on its OWN open cycle/cadence), carrying the entry context the
    # alpha attribution needs. Without this the journal stays empty and every reflection is empty.
    # Best-effort: capture is learning bookkeeping, never a fill precondition.
    try:
        journal_open_legs(
            memory_dir, account, cycle=cycle, cadence=cadence,
            leg_meta=_leg_meta(reviewed, bundle), marks=marks,
            funding_by_symbol=funding_by_symbol, btc_symbol=btc_symbol,
        )
    except Exception as exc:  # noqa: BLE001 — journaling must not unwind an executed cycle
        print(f"WARNING: journal-at-entry failed for {cadence} cycle {cycle}: {exc!r}",
              file=sys.stderr)
    turnover = sum(abs(float(t.get("target_notional", 0.0))) for t in intended)
    equity_now = account.equity(marks)
    equity_log.record_equity(state_dir, now, equity_now, cycle)
    rec = build_cycle_pnl(
        account, opening_equity=opening_equity, marks=marks, turnover_usd=turnover,
        cycle=cycle, cadence=cadence, now=now)
    save_output(state_dir, cycle, "pnl", rec, cadence=cadence)
    append_ledger(state_dir, rec)
    # LEARNING link 2 — ALPHA OUTCOME AT CLOSE. Drain the legs fully closed this cycle and patch
    # each onto the Decision that OPENED it (keyed on its OWN open cycle+cadence, NOT the current
    # cycle — a held-over leg keys on its earlier cycle, and weekly/daily cycle-N never collide).
    # When that Decision is found, the patch carries the six market-neutral ALPHA outcome fields
    # (return NET of BTC-beta and NET of costs) so the Decision becomes "closed" for the Reflector
    # AND the realized cost fields `improvement.carry_capture_rate` reads; a leg with no opening
    # Decision (opened before capture was live) degrades to the historical cost-only patch
    # (patch_outcome is fail-soft for an un-journaled leg). The book was reconciled to the neutral,
    # reviewer-passed target this cycle, so neutrality_in_band reads the dollar residual / band.
    neutrality_in_band = abs(getattr(reviewed, "dollar_residual_frac", 0.0)) <= 0.05
    for cyc, cad, sym, direction, outcome in close_alpha_outcomes(
        memory_dir, account.drain_closed_legs(), marks=marks, btc_symbol=btc_symbol,
        neutrality_in_band=neutrality_in_band,
    ):
        try:
            patch_outcome(memory_dir, cycle=cyc, symbol=sym, direction=direction,
                          outcome=outcome, cadence=cad)
        except Exception as exc:  # noqa: BLE001 — cost bookkeeping must not unwind an executed cycle
            print(f"WARNING: journal cost-patch failed for {sym} {direction}: {exc!r}",
                  file=sys.stderr)
    # Persist AFTER draining so an already-patched closed leg is not re-patched on a later run.
    save_account(state_dir, account)
    _run_reflect(state_dir, cadence, cycle, memory_dir)
    # LEARNING link 3 — mine candidate lessons from the alpha-keyed closed decisions and DSR-gate
    # their promotion (so a proven edge becomes a standing rule the next book reads back).
    _run_mine_lessons(state_dir, cadence, cycle, memory_dir, now)
    return True


def _run_mine_lessons(state_dir, cadence: Cadence, cycle: int, memory_dir, now: datetime) -> None:
    """Step 7b — distil candidate lessons from closed (alpha-keyed) decisions and gate
    candidate->validated promotion on the desk's measured DSR (`build_scorecard`). Best-effort:
    learning bookkeeping, never a fill precondition, so a mine-time error never unwinds a cycle."""
    try:
        from futures_fund.lesson_miner import mine_lessons
        from futures_fund.scorecard import build_scorecard

        dsr = float(build_scorecard(state_dir, memory_dir).get("dsr_pvalue", 0.0) or 0.0)
        mine_lessons(memory_dir, now=now, dsr_pvalue=dsr)
    except Exception as exc:  # noqa: BLE001 — lesson mining must not unwind an executed cycle
        print(f"WARNING: lesson-mine step failed for {cadence} cycle {cycle}: {exc!r}",
              file=sys.stderr)


def _assert_legs_priced(book: TargetWeights, marks: dict[str, float]) -> None:
    """LOUD GUARD against the market-neutrality bug class: every NON-FLAT leg of the intended book
    MUST have a (positive) mark before `apply_fills`.

    `apply_fills` silently SKIPS a leg whose `marks.get(sym)` is None/<=0 (it cannot size qty), so a
    non-flat leg with no mark would never open — most insidiously the BTC hedge leg when BTC is not
    in the priced universe — and the HELD book would silently go non-neutral while the leg-level
    book the reviewer audits stays correct. Rather than let that recur silently, fail LOUDLY here
    naming the unpriced symbol(s). Flat (zero-notional) legs are nothing to open, so a missing mark
    there is harmless and ignored."""
    missing = sorted({
        leg.symbol
        for leg in book.legs
        if abs(leg.target_notional) > 0.0
        and not (marks.get(leg.symbol) or 0.0) > 0.0
    })
    if missing:
        raise RuntimeError(
            "market-neutrality guard: non-flat intended legs have NO mark and would be silently "
            f"skipped by apply_fills (held book would be non-neutral): {missing}. The configured "
            "hedge/beta symbol must always be priced (see cycle_prep_cli) — refusing to fill."
        )


def _intended_fills(book: TargetWeights) -> list[dict]:
    """The FULL intended-holdings book as apply_fills leg dicts (symbol/direction/target_notional).

    This is the ACCOUNT feed: the consolidated, dollar+beta-neutral, hedge-correct book the reviewer
    validated — NOT the sparse execution deltas. Reconciling the account to this FULL book every
    cycle keeps the held positions market-neutral (apply_fills charges frictions only on the per-
    symbol delta, so unchanged legs are no-ops and a dropped symbol is flattened). Zero-notional
    legs are dropped (nothing to hold); same-symbol legs are consolidated inside apply_fills."""
    return [
        {"symbol": leg.symbol, "direction": leg.direction,
         "target_notional": abs(leg.target_notional)}
        for leg in book.legs
        if abs(leg.target_notional) > 0.0
    ]


def _leg_meta(book: TargetWeights, bundle: dict) -> dict[str, dict]:
    """Per-symbol entry context for the journal-at-entry capture (`learning.journal_open_legs`).

    Seeds from the cycle's geometries (so a HELD-OVER symbol not in this cycle's book still gets a
    `beta_btc`/sentiment fallback), then overlays the reviewed book's authoritative per-leg
    `beta_btc`/`sleeve`/`pair_id`. The held Position is the NET of any same-symbol legs, so a symbol
    on both sides resolves to its book-leg metadata (last leg wins) — fine for attribution tags."""
    meta: dict[str, dict] = {}
    for g in bundle.get("geometries", []):
        sym = g.get("symbol")
        if not sym:
            continue
        meta[sym] = {
            "beta_btc": float(g.get("beta_btc", 0.0) or 0.0),
            "sleeve": None,
            "pair_id": g.get("pair_id"),
            "sentiment_score": float(g.get("sentiment_score", 0.0) or 0.0),
            "regime": None,
        }
    for leg in book.legs:
        m = meta.setdefault(leg.symbol, {"sentiment_score": 0.0, "regime": None})
        m["beta_btc"] = leg.beta_btc
        m["sleeve"] = str(leg.sleeve)
        m["pair_id"] = leg.pair_id
    return meta


def _half_spread_bps(bids: list, asks: list, default: float) -> float:
    """Observed top-of-book half-spread in bps; `default` when a side is missing/degenerate."""
    if not bids or not asks:
        return default
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0 or best_ask < best_bid:
        return default
    return (best_ask - best_bid) / 2.0 / mid * 1e4


def _geometry_cost_maps(bundle: dict) -> tuple[dict, dict, dict, dict]:
    """From a geometries.json bundle build (marks, funding_by_symbol, intervals, costs).

    marks/funding/interval come straight off each CoinGeometry; costs is a CostInputs carrier (ADV
    + a 1bps half-spread default) so the paper fill uses the slippage fallback (never flat)."""
    marks: dict[str, float] = {}
    funding: dict[str, float] = {}
    intervals: dict[str, int] = {}
    costs: dict[str, CostInputs] = {}
    for g in bundle.get("geometries", []):
        sym = g.get("symbol")
        mark = g.get("mark")
        if not sym or mark is None:
            continue
        marks[sym] = float(mark)
        funding[sym] = float(g.get("funding_rate", 0.0))
        intervals[sym] = int(g.get("funding_interval_hours", 8) or 8)
        bids = [(float(p), float(q)) for p, q in (g.get("depth_bids") or [])]
        asks = [(float(p), float(q)) for p, q in (g.get("depth_asks") or [])]
        costs[sym] = CostInputs(
            adv_usd=float(g.get("adv_usd", 0.0)),
            half_spread_bps=_half_spread_bps(bids, asks, 1.0),
            depth_bids=bids, depth_asks=asks,
        )
    return marks, funding, intervals, costs


def _load_geometries(state_dir, cadence: Cadence, cycle: int) -> dict:
    """Best-effort read of this cycle's geometries.json (marks + funding + ADV)."""
    try:
        return load_output(state_dir, cycle, "geometries", cadence=cadence)
    except FileNotFoundError:
        return {"geometries": []}


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
            ran = _run_cadence(cadence, args.state_dir, args.memory_dir, now, equity,
                               btc_symbol=settings.beta.btc_symbol)
            mode, cycle, _reason = cadence_due(args.state_dir, now, cadence)
            summary["cadences"][cadence] = {  # type: ignore[index]
                "ran": ran,
                "cycle": cycle if ran else None,
            }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())

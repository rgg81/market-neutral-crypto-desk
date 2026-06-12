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

from futures_fund.account import load_account
from futures_fund.config import load_settings
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


def _prior_marks(state_dir, cadence: Cadence) -> dict[str, float]:
    """Mark prices from the most recently completed cycle's `geometries.json`.

    preflight.py (W3 step 2) runs BEFORE `cycle_prep_cli.py` (W3 step 3) writes THIS cycle's
    geometries, so the current cycle has no geometries yet. Resolve the latest cadence cycle that
    actually persisted `geometries` (same prior-state lookup `_held_symbols` uses for `report`) so
    open positions mark-to-market instead of collapsing equity to cash-only. Empty on a miss."""
    n = latest_cadence_cycle(state_dir, cadence, "geometries")
    if n is None:
        return {}
    try:
        bundle = load_output(state_dir, n, "geometries", cadence=cadence)
    except FileNotFoundError:
        return {}
    return {g["symbol"]: float(g["mark"]) for g in bundle.get("geometries", [])
            if g.get("symbol") and g.get("mark") is not None}


def _prior_pnl(state_dir, cadence: Cadence) -> dict:
    """The latest persisted `pnl.json` (last rebalance's realized cost/turnover).

    `pnl.json` is written at execute/P&L (W10/D7), long after preflight runs, so the current cycle
    has none yet. Resolve the most recently completed cycle's `pnl` so `last_rebalance_cost` /
    `last_rebalance_turnover_usd` reflect the last round-trip the trader paid. Empty on a miss."""
    n = latest_cadence_cycle(state_dir, cadence, "pnl")
    if n is None:
        return {}
    try:
        return load_output(state_dir, n, "pnl", cadence=cadence)
    except FileNotFoundError:
        return {}


def build_pnl_block(state_dir, *, marks: dict[str, float], last_pnl: dict,
                    default_cash: float) -> dict:
    """The realized cost/carry/PnL block folded into context.json (an ARTIFACT the external SKILL.md
    orchestrator reads when assembling prompts — there is no Python prompt-injection path here).

    Per-symbol realized funding carry (signed accrued_funding, + = received), current unrealized
    PnL, and accrued fees; plus the last rebalance's cost (fees+slippage) vs its turnover so the
    trader can weigh round-trip cost against the spread edge before churning a pair."""
    acct = load_account(state_dir, default_cash)
    upnl = acct.mark_to_market(marks)
    by_symbol = {
        sym: {
            "unrealized": upnl.get(sym, 0.0),
            "realized_funding": pos.accrued_funding,
            "accrued_fees": pos.accrued_fees,
        }
        for sym, pos in acct.positions.items()
    }
    return {
        "equity": acct.equity(marks),
        "total_fees": acct.fees_paid,
        "total_slippage": acct.slippage_paid,
        "total_funding_received": acct.funding_received,
        "total_funding_paid": acct.funding_paid,
        "last_rebalance_cost": float(last_pnl.get("fees_paid", 0.0))
        + float(last_pnl.get("slippage_paid", 0.0)),
        "last_rebalance_turnover_usd": float(last_pnl.get("turnover_usd", 0.0)),
        "by_symbol": by_symbol,
    }


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
    # Fold the realized cost/carry/PnL block (Phase 9) so the artifact context.json carries cost
    # data for the external orchestrator. load_settings() is safe with no config.yaml in cwd
    # (account_size_usdt defaults to 20000); default_cash is passed explicitly regardless.
    # marks/last_pnl come from the most recently COMPLETED cycle: preflight (W3 step 2) runs before
    # this cycle's geometries (W3 step 3) and pnl (W10/D7) exist, so sourcing from args.cycle would
    # always miss and zero out mark-to-market equity + last-rebalance cost.
    marks = _prior_marks(args.state_dir, cadence)
    last_pnl = _prior_pnl(args.state_dir, cadence)
    settings = load_settings()
    pnl_block = build_pnl_block(
        args.state_dir, marks=marks, last_pnl=last_pnl,
        default_cash=settings.account_size_usdt)
    briefs = build_briefs(universe, held)
    for b in briefs:
        per = pnl_block["by_symbol"].get(b["symbol"])
        if per is not None:
            b["pnl"] = per                          # per-symbol realized cost/carry on the brief
    ctx = {"briefs": briefs, "held": held, "pnl": pnl_block}
    save_output(args.state_dir, args.cycle, "context", ctx, cadence=cadence)
    print(json.dumps(ctx, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())

# tests/test_preflight_cli.py
from __future__ import annotations

import json
from datetime import UTC, datetime

from futures_fund.account import PaperAccount, Position, save_account
from futures_fund.cycle_io import cycle_dir, save_output
from scripts.preflight import build_briefs, main


def test_build_briefs_folds_held_symbols():
    universe = [{"symbol": "BTC/USDT:USDT"}, {"symbol": "ETH/USDT:USDT"}]
    held = ["SOL/USDT:USDT", "BTC/USDT:USDT"]  # SOL held but not in universe -> folded in
    briefs = build_briefs(universe, held)
    syms = {b["symbol"] for b in briefs}
    assert syms == {"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"}
    sol = next(b for b in briefs if b["symbol"] == "SOL/USDT:USDT")
    assert sol["held"] is True
    btc = next(b for b in briefs if b["symbol"] == "BTC/USDT:USDT")
    assert btc["held"] is True  # held AND in-universe


def test_preflight_writes_context(tmp_path, capsys):
    state = tmp_path / "state"
    save_output(state, 1, "universe",
                {"universe": [{"symbol": "BTC/USDT:USDT"}, {"symbol": "ETH/USDT:USDT"}]},
                cadence="weekly")
    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state)])
    ctx = json.loads((cycle_dir(state, 1, cadence="weekly") / "context.json").read_text())
    assert {b["symbol"] for b in ctx["briefs"]} == {"BTC/USDT:USDT", "ETH/USDT:USDT"}


def test_preflight_marks_from_prior_completed_cycle(tmp_path):
    """On a FRESH cycle N, preflight runs before THIS cycle's geometries/pnl exist; marks and
    last-rebalance cost must come from the most recently COMPLETED cycle, not args.cycle (which is
    always empty at preflight time). Otherwise equity collapses to cash-only and cost rows zero."""
    state = tmp_path / "state"
    # A held short carried in the account (opened in a prior cycle).
    acct = PaperAccount(cash=20_010.0)
    acct.positions["ETH/USDT:USDT"] = Position(
        symbol="ETH/USDT:USDT", direction="short", qty=2.0, entry_price=2000.0,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC), accrued_funding=6.0)
    save_account(state, acct)
    # Prior completed weekly cycle 1 persisted geometries (marks) + pnl (last rebalance cost).
    save_output(state, 1, "geometries",
                {"geometries": [{"symbol": "ETH/USDT:USDT", "mark": 1950.0}],
                 "as_of_ts": "2026-06-10T00:00:00+00:00"}, cadence="weekly")
    save_output(state, 1, "pnl",
                {"fees_paid": 4.0, "slippage_paid": 2.0, "turnover_usd": 4000.0}, cadence="weekly")
    # Fresh cycle 2: only universe exists yet (geometries/pnl for cycle 2 NOT written).
    save_output(state, 2, "universe", {"universe": [{"symbol": "BTC/USDT:USDT"}]}, cadence="weekly")

    main(["--cycle", "2", "--cadence", "weekly", "--state-dir", str(state)])
    ctx = json.loads((cycle_dir(state, 2, cadence="weekly") / "context.json").read_text())

    pnl = ctx["pnl"]
    # Short upnl at mark 1950 = (2000-1950)*2 = 100 -> equity marks-to-market the open position.
    assert pnl["equity"] == 20_010.0 + 100.0
    assert pnl["by_symbol"]["ETH/USDT:USDT"]["unrealized"] == 100.0
    # Last rebalance cost/turnover come from the prior cycle's pnl.json (not zeroed).
    assert pnl["last_rebalance_cost"] == 6.0
    assert pnl["last_rebalance_turnover_usd"] == 4000.0

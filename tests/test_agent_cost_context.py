from __future__ import annotations

import json
from datetime import UTC, datetime

from futures_fund.account import PaperAccount, Position, save_account
from futures_fund.cycle_io import save_output
from futures_fund.scorecard import _latest_pnl, build_scorecard
from scripts.preflight import build_pnl_block


def _seed_account(state):
    acct = PaperAccount(cash=20_010.0)
    acct.positions["ETH/USDT:USDT"] = Position(
        symbol="ETH/USDT:USDT", direction="short", qty=2.0, entry_price=2000.0,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC),
        accrued_funding=6.0, accrued_fees=2.0, accrued_slippage=1.0)
    acct.fees_paid = 4.0
    acct.slippage_paid = 2.0
    acct.funding_received = 6.0
    save_account(state, acct)
    return acct


def test_build_pnl_block_is_populated_from_the_account(tmp_path):
    state = tmp_path / "state"
    _seed_account(state)
    marks = {"ETH/USDT:USDT": 1950.0}              # short upnl = 100
    last_pnl = {"fees_paid": 4.0, "slippage_paid": 2.0, "turnover_usd": 4000.0}

    block = build_pnl_block(state, marks=marks, last_pnl=last_pnl, default_cash=20_000.0)

    assert block["equity"] == 20_010.0 + 100.0
    assert block["total_fees"] == 4.0
    assert block["total_slippage"] == 2.0
    assert block["total_funding_received"] == 6.0
    per = block["by_symbol"]["ETH/USDT:USDT"]
    assert per["unrealized"] == 100.0
    assert per["realized_funding"] == 6.0          # signed accrued_funding (+ = received)
    assert per["accrued_fees"] == 2.0
    assert block["last_rebalance_cost"] == 6.0     # fees 4 + slippage 2
    assert block["last_rebalance_turnover_usd"] == 4000.0


def test_pnl_block_fixture_contract_matches():
    fixture = json.loads(open("tests/fixtures/pnl_block.json").read())
    for key in ("equity", "total_fees", "total_slippage", "total_funding_received",
                "total_funding_paid", "last_rebalance_cost", "last_rebalance_turnover_usd",
                "by_symbol"):
        assert key in fixture
    for key in ("unrealized", "realized_funding", "accrued_fees"):
        assert key in next(iter(fixture["by_symbol"].values()))


def test_latest_pnl_reads_the_newest_cycle(tmp_path):
    state = tmp_path / "state"
    save_output(state, 1, "pnl", {"net_pnl": 1.0, "cycle": 1,
                                  "ts": "2026-06-10T00:00:00+00:00"}, cadence="weekly")
    save_output(state, 2, "pnl", {"net_pnl": 5.0, "cycle": 2,
                                  "ts": "2026-06-11T00:00:00+00:00"}, cadence="daily")
    rec = _latest_pnl(state)
    assert rec["net_pnl"] == 5.0                    # newest ts wins


def test_scorecard_carries_cost_transparency_keys(tmp_path):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    save_output(state, 1, "pnl", {
        "net_pnl": 8.0, "gross_pnl": 14.0, "fees_paid": 4.0, "slippage_paid": 2.0,
        "funding_net": 6.0, "cycle": 1, "ts": "2026-06-10T00:00:00+00:00"}, cadence="weekly")
    sc = build_scorecard(state, memory)
    assert sc["net_pnl"] == 8.0
    assert sc["gross_pnl"] == 14.0
    assert sc["total_fees"] == 4.0
    assert sc["total_slippage"] == 2.0
    assert sc["funding_net"] == 6.0
    assert abs(sc["cost_drag_bps"] - (6.0 / 14.0 * 1e4)) < 1e-6

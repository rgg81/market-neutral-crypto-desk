# tests/test_account_integration.py — create (or append if it exists from Task 12 ordering)
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_fund.account import CostInputs, PaperAccount
from futures_fund.pnl_attribution import build_cycle_pnl


def test_position_opened_this_cycle_earns_zero_funding_this_cycle():
    """Settle-BEFORE-fill: a leg opened in the same cycle as settle_funding earns no funding for
    that cycle's window (it did not exist for the pre-existence span)."""
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 11, 0, 0, tzinfo=UTC)  # 24h -> 3 settlements IF the leg existed
    # ORDER MATTERS: settle first (no positions yet -> 0), THEN open.
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, {})
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "short", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs, opened_ts=now)
    assert acct.funding_received == 0.0            # opened AFTER settle -> no funding this cycle
    assert acct.positions["ETH/USDT:USDT"].accrued_funding == 0.0


def test_weekly_cycle2_resend_does_not_double_qty():
    """The multi-week double-count guard at the book level: re-applying the IDENTICAL full weekly
    book on cycle 2 reconciles to delta 0 -> qty unchanged, frictions unchanged."""
    acct = PaperAccount(cash=20_000.0)
    costs = {
        "ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0),
        "BTC/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0),
    }
    marks = {"ETH/USDT:USDT": 2000.0, "BTC/USDT:USDT": 60_000.0}
    book = [
        {"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0},
        {"symbol": "BTC/USDT:USDT", "direction": "short", "target_notional": 6000.0},
    ]
    acct.apply_fills(book, marks, costs, opened_ts=datetime(2026, 6, 10, tzinfo=UTC))
    qty1 = {s: p.qty for s, p in acct.positions.items()}
    fees1, slip1 = acct.fees_paid, acct.slippage_paid
    # weekly cycle 2: the SAME full book again
    acct.apply_fills(book, marks, costs, opened_ts=datetime(2026, 6, 17, tzinfo=UTC))
    qty2 = {s: p.qty for s, p in acct.positions.items()}
    assert qty2 == qty1                            # NOT doubled to ~2x notional
    assert acct.fees_paid == fees1                 # 0 extra fee on the re-send
    assert acct.slippage_paid == slip1             # 0 extra slippage on the re-send


def test_two_cycle_equity_moves_off_constant_with_funding_and_fees():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0)}
    t0 = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)

    # cycle 1: settle (no positions -> 0), then open a short carry leg
    marks1 = {"ETH/USDT:USDT": 2000.0}
    opening1 = acct.equity(marks1)
    acct.settle_funding(t0, t0, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, marks1)
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "short", "target_notional": 4000.0}],
        marks1, costs, opened_ts=t0)
    rec1 = build_cycle_pnl(acct, opening_equity=opening1, marks=marks1,
                           turnover_usd=4000.0, cycle=1, cadence="weekly", now=t0)
    assert rec1["closing_equity"] != 20_000.0      # frictions moved equity off the constant
    assert rec1["fees_paid"] > 0.0
    assert rec1["slippage_paid"] > 0.0
    assert rec1["funding_received"] == 0.0         # leg opened AFTER cycle-1 settle

    # cycle 2 (one sim-day later): settle funding (3 events at 8h) from the account clock, re-mark
    t1 = t0 + timedelta(days=1)
    marks2 = {"ETH/USDT:USDT": 2000.0}
    opening2 = acct.equity(marks2)
    acct.settle_funding(acct.last_funding_ts, t1, {"ETH/USDT:USDT": 0.0005},
                        {"ETH/USDT:USDT": 8}, marks2)
    rec2 = build_cycle_pnl(acct, opening_equity=opening2, marks=marks2,
                           turnover_usd=0.0, cycle=2, cadence="daily", now=t1)

    assert rec2["funding_received"] > 0.0          # short + positive rate = received over the day
    assert rec2["funding_net"] > 0.0
    equities = [rec1["closing_equity"], rec2["closing_equity"]]
    assert len(set(equities)) == 2                 # not a flat 20000
    assert all(e != 20_000.0 for e in equities)
    assert rec2["net_pnl"] == rec2["gross_pnl"] - rec2["fees_paid"] - rec2["slippage_paid"]

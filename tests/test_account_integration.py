# tests/test_account_integration.py — create (or append if it exists from Task 12 ordering)
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import CostInputs, PaperAccount


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

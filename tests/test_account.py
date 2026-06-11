from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import CostInputs, PaperAccount, Position
from futures_fund.costs import count_funding_events


def _pos(symbol="ETH/USDT:USDT", direction="long", qty=2.0, entry=2000.0):
    return Position(
        symbol=symbol, direction=direction, qty=qty, entry_price=entry,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC),
    )


def test_account_persistence_round_trip():
    acct = PaperAccount(cash=20_000.0)
    acct.positions[_pos().symbol] = _pos()
    acct.realized_pnl = 12.5
    acct.fees_paid = 3.0
    acct.slippage_paid = 1.0
    acct.funding_received = 4.0
    acct.funding_paid = 2.0
    acct.last_funding_ts = datetime(2026, 6, 10, 8, tzinfo=UTC)

    restored = PaperAccount.from_dict(acct.to_dict())

    assert restored.cash == 20_000.0
    assert restored.realized_pnl == 12.5
    assert restored.fees_paid == 3.0
    assert restored.slippage_paid == 1.0
    assert restored.funding_received == 4.0
    assert restored.funding_paid == 2.0
    assert restored.last_funding_ts == datetime(2026, 6, 10, 8, tzinfo=UTC)
    pos = restored.positions["ETH/USDT:USDT"]
    assert pos.qty == 2.0
    assert pos.entry_price == 2000.0
    assert pos.direction == "long"


def test_fresh_account_has_no_funding_clock():
    assert PaperAccount(cash=20_000.0).last_funding_ts is None


def test_mark_to_market_and_equity_long_and_short():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="long", qty=2.0, entry=2000.0)
    acct.positions["BTC/USDT:USDT"] = _pos(
        symbol="BTC/USDT:USDT", direction="short", qty=0.1, entry=60_000.0)

    marks = {"ETH/USDT:USDT": 2100.0, "BTC/USDT:USDT": 59_000.0}
    upnl = acct.mark_to_market(marks)

    # long: 2*(2100-2000)=200 ; short: 0.1*(60000-59000)=100
    assert upnl["ETH/USDT:USDT"] == 200.0
    assert upnl["BTC/USDT:USDT"] == 100.0
    assert acct.equity(marks) == 20_000.0 + 300.0


def test_equity_skips_symbols_missing_a_mark():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="long", qty=2.0, entry=2000.0)
    # no mark for ETH -> contributes 0, equity == cash
    assert acct.equity({}) == 20_000.0


def test_apply_fills_opens_position_charges_fee_and_slippage():
    acct = PaperAccount(cash=20_000.0)
    executed = [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}]
    marks = {"ETH/USDT:USDT": 2000.0}
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0)}

    acct.apply_fills(executed, marks, costs)

    pos = acct.positions["ETH/USDT:USDT"]
    assert pos.qty == 2.0                          # 4000 / 2000
    assert pos.direction == "long"
    # taker fee on 4000 notional = 4000 * 0.0005 = 2.0 USDT, charged to cash + accrued
    assert pos.accrued_fees == 2.0
    assert acct.fees_paid == 2.0
    assert pos.accrued_slippage > 0.0
    assert acct.slippage_paid == pos.accrued_slippage
    # cash deducted by fee + slippage only (paper-margin: the notional itself consumes no cash)
    assert acct.cash == 20_000.0 - 2.0 - pos.accrued_slippage


def test_apply_fills_resend_same_target_is_a_noop():
    """The multi-week double-count guard: re-sending the IDENTICAL book trades 0 -> 0 frictions,
    same qty (NOT doubled)."""
    acct = PaperAccount(cash=20_000.0)
    book = [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}]
    marks = {"ETH/USDT:USDT": 2000.0}
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0)}
    acct.apply_fills(book, marks, costs)
    fees_after_open, slip_after_open = acct.fees_paid, acct.slippage_paid
    acct.apply_fills(book, marks, costs)            # same target again -> reconcile to delta 0
    assert acct.positions["ETH/USDT:USDT"].qty == 2.0        # NOT 4.0 — no double-count
    assert acct.fees_paid == fees_after_open                 # 0 extra fee
    assert acct.slippage_paid == slip_after_open             # 0 extra slippage


def test_apply_fills_increase_to_a_larger_target_blends_entry_vwap():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 2000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)            # target qty 1.0 @ 2000
    # raise the target to 4400 @ mark 2200 -> target qty 2.0, delta +1.0 filled @ 2200
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4400.0}],
        {"ETH/USDT:USDT": 2200.0}, costs)
    pos = acct.positions["ETH/USDT:USDT"]
    assert abs(pos.qty - 2.0) < 1e-9               # 4400/2200
    # blended VWAP: (1.0 @ 2000 + 1.0 @ 2200) / 2.0 = 2100
    assert abs(pos.entry_price - 2100.0) < 1e-6


def test_apply_fills_reduce_to_smaller_target_realizes_partial_pnl():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)            # long qty 2 @ 2000
    cash_after_open = acct.cash
    # lower target to 2200 @ mark 2200 -> target qty 1.0, reduce 1.0 @ 2200, realize 1*(2200-2000)
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 2200.0}],
        {"ETH/USDT:USDT": 2200.0}, costs)
    pos = acct.positions["ETH/USDT:USDT"]
    assert pos.direction == "long"
    assert abs(pos.qty - 1.0) < 1e-9               # 2200/2200
    assert abs(acct.realized_pnl - 1.0 * (2200.0 - 2000.0)) < 1e-6
    assert abs(pos.realized_pnl - 200.0) < 1e-6
    assert acct.cash > cash_after_open             # got the realized credit (fee>0, slip=0)


def test_apply_fills_zero_target_closes_and_pops():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)
    # target 0 at the SAME mark closes the whole 2.0 qty flat
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 0.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)
    assert "ETH/USDT:USDT" not in acct.positions
    assert abs(acct.realized_pnl) < 1e-6           # closed flat -> ~0 price pnl


def test_apply_fills_opposite_target_flips_side():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 2000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)            # long qty 1 @ 2000
    # target a SHORT 4000 @ 2000 -> target signed qty -2.0; close the +1 (flat pnl), open 2 short
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "short", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)
    pos = acct.positions["ETH/USDT:USDT"]
    assert pos.direction == "short"
    assert abs(pos.qty - 2.0) < 1e-9               # |−2.0| target
    assert pos.entry_price == 2000.0


def test_settle_funding_short_positive_rate_is_a_credit_and_advances_clock():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="short", qty=2.0, entry=2000.0)
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 11, 0, 0, tzinfo=UTC)       # 24h -> 3 settlements at 8h
    assert count_funding_events(prev, now, 8) == 3
    marks = {"ETH/USDT:USDT": 2000.0}
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, marks)

    # short + positive rate RECEIVES: realized_funding = -(-1)*2000*2*0.0005 = +2.0 per event
    expected = 3 * 2.0
    assert abs(acct.positions["ETH/USDT:USDT"].accrued_funding - expected) < 1e-9
    assert abs(acct.cash - (20_000.0 + expected)) < 1e-9
    assert abs(acct.funding_received - expected) < 1e-9
    assert acct.funding_paid == 0.0
    assert acct.last_funding_ts == now             # the funding clock advanced


def test_settle_funding_long_positive_rate_is_a_debit():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="long", qty=2.0, entry=2000.0)
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 10, 8, 1, tzinfo=UTC)       # 1 settlement at hour 8
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8},
                        {"ETH/USDT:USDT": 2000.0})
    assert abs(acct.positions["ETH/USDT:USDT"].accrued_funding - (-2.0)) < 1e-9
    assert abs(acct.cash - (20_000.0 - 2.0)) < 1e-9
    assert acct.funding_received == 0.0
    assert abs(acct.funding_paid - 2.0) < 1e-9


def test_settle_funding_no_events_still_advances_clock():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="short", qty=2.0, entry=2000.0)
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 10, 1, 0, tzinfo=UTC)       # < 8h -> 0 settlements
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8},
                        {"ETH/USDT:USDT": 2000.0})
    assert acct.cash == 20_000.0
    assert acct.positions["ETH/USDT:USDT"].accrued_funding == 0.0
    assert acct.last_funding_ts == now             # clock advances even with 0 events

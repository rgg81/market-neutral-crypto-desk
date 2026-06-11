from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import PaperAccount, Position


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

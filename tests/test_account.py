from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import (
    ClosedLeg,
    CostInputs,
    PaperAccount,
    Position,
    load_account,
    save_account,
)
from futures_fund.costs import count_funding_events
from futures_fund.journal import append_decision, patch_outcome, read_all_decisions
from scripts.run_paper_cli import _geometry_cost_maps, _leg_cost_patches


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


def test_load_account_fresh_inits_at_default_cash(tmp_path):
    acct = load_account(tmp_path / "state", default_cash=20_000.0)
    assert acct.cash == 20_000.0
    assert acct.positions == {}
    assert acct.realized_pnl == 0.0
    assert acct.last_funding_ts is None


def test_save_then_load_round_trips_at_state_root(tmp_path):
    state = tmp_path / "state"
    acct = PaperAccount(cash=15_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos()
    acct.fees_paid = 7.0
    acct.last_funding_ts = datetime(2026, 6, 10, 8, tzinfo=UTC)
    save_account(state, acct)
    assert (state / "account.json").exists()
    restored = load_account(state, default_cash=99_999.0)
    assert restored.cash == 15_000.0               # NOT the default — the file wins
    assert restored.fees_paid == 7.0
    assert restored.last_funding_ts == datetime(2026, 6, 10, 8, tzinfo=UTC)
    assert restored.positions["ETH/USDT:USDT"].qty == 2.0


def test_geometry_cost_maps_from_bundle():
    bundle = {"geometries": [
        {"symbol": "ETH/USDT:USDT", "mark": 2000.0, "funding_rate": 0.0005,
         "funding_interval_hours": 8.0, "adv_usd": 5_000_000.0, "beta_btc": 1.0,
         "momentum_20": 0.0, "realized_vol": 0.0, "sentiment_score": 0.0,
         "sentiment_conf": 0.0},
    ]}
    marks, funding, intervals, costs = _geometry_cost_maps(bundle)
    assert marks["ETH/USDT:USDT"] == 2000.0
    assert funding["ETH/USDT:USDT"] == 0.0005
    assert intervals["ETH/USDT:USDT"] == 8
    assert costs["ETH/USDT:USDT"].adv_usd == 5_000_000.0


def test_leg_cost_patches_drains_closed_legs_keyed_on_open_cycle():
    """`_leg_cost_patches` emits ONE tuple per CLOSED leg (not per open position), keyed on the
    cycle+cadence the leg was OPENED in — and drains the buffer so it is patched exactly once."""
    acct = PaperAccount(cash=20_000.0)
    # an OPEN position must NOT be patched (its P&L is still unrealized — "at close" only).
    acct.positions["BTC/USDT:USDT"] = _pos(symbol="BTC/USDT:USDT", direction="long")
    acct.closed_legs.append(ClosedLeg(
        symbol="ETH/USDT:USDT", direction="short", opened_cycle=3, opened_cadence="weekly",
        fees=4.0, slippage=2.0, realized_funding=6.0, realized_pnl=12.0))
    patches = _leg_cost_patches(acct)
    assert patches == [
        (3, "weekly", "ETH/USDT:USDT", "short",
         {"fees": 4.0, "slippage": 2.0, "realized_funding": 6.0, "realized_pnl": 12.0}),
    ]
    # drained -> a second call yields nothing (patched exactly once).
    assert _leg_cost_patches(acct) == []
    assert acct.closed_legs == []


def test_apply_fills_full_close_records_a_closed_leg_with_open_cycle():
    """A fully-closed leg is snapshotted into `closed_legs` carrying its OPEN cycle/cadence and its
    realized fees/slippage/funding/price-pnl — the realized outcome the 'at close' patch needs and
    that would otherwise be lost when the Position is popped (finding 2)."""
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    # open a long at cycle 3 weekly, accrue some funding on it.
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC), opened_cycle=3, opened_cadence="weekly")
    acct.positions["ETH/USDT:USDT"].accrued_funding = 1.5
    open_fees = acct.positions["ETH/USDT:USDT"].accrued_fees
    # close it flat at cycle 5 daily (target 0); profit on the 2000->2200 move.
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 0.0}],
        {"ETH/USDT:USDT": 2200.0}, costs,
        opened_ts=datetime(2026, 6, 12, tzinfo=UTC), opened_cycle=5, opened_cadence="daily")
    assert "ETH/USDT:USDT" not in acct.positions
    assert len(acct.closed_legs) == 1
    cl = acct.closed_legs[0]
    assert cl.symbol == "ETH/USDT:USDT" and cl.direction == "long"
    assert cl.opened_cycle == 3 and cl.opened_cadence == "weekly"   # the OPEN cycle, not the close
    assert abs(cl.realized_pnl - 2.0 * (2200.0 - 2000.0)) < 1e-6     # 2 qty * 200
    assert abs(cl.realized_funding - 1.5) < 1e-9
    assert cl.fees > open_fees                                       # open + close fees accrued


def test_closed_leg_survives_account_round_trip():
    acct = PaperAccount(cash=20_000.0)
    acct.closed_legs.append(ClosedLeg(
        symbol="ETH/USDT:USDT", direction="short", opened_cycle=2, opened_cadence="daily",
        fees=1.0, slippage=0.5, realized_funding=-0.25, realized_pnl=-3.0))
    restored = PaperAccount.from_dict(acct.to_dict())
    assert len(restored.closed_legs) == 1
    assert restored.closed_legs[0].opened_cycle == 2
    assert restored.closed_legs[0].opened_cadence == "daily"


# --------------------------------------------------------------------------------------------------
# Integration: a close-time cost patch ACTUALLY lands on the journaled Decision (finding 4).
# --------------------------------------------------------------------------------------------------
def _patch_closed_legs(memory_dir, account):
    """Mirror the run_paper_cli close-time loop: drain + patch each closed leg on its OPEN key."""
    for cyc, cad, sym, direction, outcome in _leg_cost_patches(account):
        patch_outcome(memory_dir, cycle=cyc, symbol=sym, direction=direction,
                      outcome=outcome, cadence=cad)


def test_held_over_leg_patches_onto_its_OPEN_cycle_decision(tmp_path):
    """The leg is OPENED at cycle 3 (journaled there) and CLOSED at cycle 7. The patch must land on
    the cycle-3 Decision — keying on the current (7) cycle would be a silent no-op (finding 1)."""
    memory = tmp_path / "memory"
    append_decision(memory, cycle=3, symbol="ETH/USDT:USDT", direction="short",
                    payload={"rationale": "carry"}, cadence="weekly")
    acct = PaperAccount(cash=20_000.0)
    acct.closed_legs.append(ClosedLeg(
        symbol="ETH/USDT:USDT", direction="short", opened_cycle=3, opened_cadence="weekly",
        fees=4.0, slippage=2.0, realized_funding=6.0, realized_pnl=12.0))

    _patch_closed_legs(memory, acct)

    rows = [d for d in read_all_decisions(memory)
            if d["cycle"] == 3 and d["symbol"] == "ETH/USDT:USDT"]
    assert len(rows) == 1
    assert rows[0]["realized_funding"] == 6.0       # landed on the OPEN-cycle decision
    assert rows[0]["fees"] == 4.0 and rows[0]["slippage"] == 2.0
    assert rows[0]["realized_pnl"] == 12.0


def test_daily_close_does_not_mis_key_onto_weekly_decision_at_same_cycle(tmp_path):
    """Weekly cycle-1 and daily cycle-1 share a cycle number (account.py:15). A DAILY close patch
    must land on the DAILY decision, never bleed onto the same-numbered WEEKLY one (finding 1)."""
    memory = tmp_path / "memory"
    append_decision(memory, cycle=1, symbol="ETH/USDT:USDT", direction="long",
                    payload={"rationale": "weekly book"}, cadence="weekly")
    append_decision(memory, cycle=1, symbol="ETH/USDT:USDT", direction="long",
                    payload={"rationale": "daily book"}, cadence="daily")
    acct = PaperAccount(cash=20_000.0)
    acct.closed_legs.append(ClosedLeg(
        symbol="ETH/USDT:USDT", direction="long", opened_cycle=1, opened_cadence="daily",
        fees=1.0, slippage=0.5, realized_funding=-0.3, realized_pnl=9.0))

    _patch_closed_legs(memory, acct)

    by_cad = {d["cadence"]: d for d in read_all_decisions(memory)}
    assert by_cad["daily"].get("realized_funding") == -0.3      # patched the DAILY decision
    assert by_cad["daily"].get("realized_pnl") == 9.0
    assert by_cad["weekly"].get("realized_funding") is None     # weekly untouched (no mis-key)
    assert by_cad["weekly"].get("realized_pnl") is None


def test_patch_outcome_no_journaled_leg_is_a_fail_soft_noop(tmp_path):
    """A closed leg whose open was never journaled patches nothing (returns False) and does not
    raise — cost bookkeeping must never unwind an executed cycle."""
    memory = tmp_path / "memory"
    landed = patch_outcome(memory, cycle=9, symbol="DOGE/USDT:USDT", direction="long",
                           outcome={"realized_funding": 1.0}, cadence="daily")
    assert landed is False
    assert read_all_decisions(memory) == []

from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import CostInputs, PaperAccount
from scripts.run_paper_cli import _geometry_cost_maps

_TS = datetime(2026, 6, 12, tzinfo=UTC)


def _btc_book():
    # deep: 1000 BTC at ~60000 each side -> a small clip is nearly free
    return ([(59999.0, 1000.0)], [(60001.0, 1000.0)])


def _thin_book():
    # thin alt: only 50 units near 100, then a steep step
    return ([(99.5, 50.0), (95.0, 50.0)], [(100.5, 50.0), (105.0, 50.0)])


def _account():
    return PaperAccount(cash=1_000_000.0)


def test_thin_book_slippage_materially_exceeds_btc_and_one_bp():
    # buy the SAME ~$50k notional on a deep BTC book vs a thin alt book
    deep_bids, deep_asks = _btc_book()
    thin_bids, thin_asks = _thin_book()

    btc_acct = _account()
    btc_acct.apply_fills(
        [{"symbol": "BTC/USDT:USDT", "direction": "long", "target_notional": 50_000.0}],
        marks={"BTC/USDT:USDT": 60000.0},
        costs={"BTC/USDT:USDT": CostInputs(depth_bids=deep_bids, depth_asks=deep_asks)},
        opened_ts=_TS, opened_cycle=1, opened_cadence="weekly",
    )
    alt_acct = _account()
    alt_acct.apply_fills(
        [{"symbol": "ALT/USDT:USDT", "direction": "long", "target_notional": 50_000.0}],
        marks={"ALT/USDT:USDT": 100.0},
        costs={"ALT/USDT:USDT": CostInputs(depth_bids=thin_bids, depth_asks=thin_asks)},
        opened_ts=_TS, opened_cycle=1, opened_cadence="weekly",
    )
    btc_slip = btc_acct.slippage_paid
    alt_slip = alt_acct.slippage_paid
    assert alt_slip > btc_slip
    # thin alt slippage is materially > 1bp of the 50k notional (1bp = 5.0 USDT)
    assert alt_slip > 5.0
    assert alt_slip > 5.0 * btc_slip  # at least an order of magnitude worse than BTC


def test_apply_fills_selects_ask_side_for_a_buy():
    # only the ASK side is thin; the BID side is deep. A BUY must cost from the thin ASK side.
    acct = _account()
    acct.apply_fills(
        [{"symbol": "ALT/USDT:USDT", "direction": "long", "target_notional": 4_000.0}],
        marks={"ALT/USDT:USDT": 100.0},
        costs={"ALT/USDT:USDT": CostInputs(
            depth_bids=[(100.0, 1_000_000.0)],            # deep bids (irrelevant for a buy)
            depth_asks=[(101.0, 10.0), (130.0, 10.0)])},  # thin asks -> real slippage
        opened_ts=_TS, opened_cycle=1, opened_cadence="weekly",
    )
    assert acct.slippage_paid > 5.0  # priced off the thin ASK side, not the deep bids


def test_over_depth_clip_is_under_costed_documented():
    # DOCUMENTED CAVEAT: vwap_fill prices slippage on the PARTIAL fill, but the position opens at
    # FULL target qty. Here the book holds only 10 units (~$1100) but we buy ~$50k. The position
    # qty is the full target (500 units) yet slippage is priced on the 10 filled units -> the
    # number is an UNDER-estimate for over-depth clips. We pin the exact under-cost so a future
    # reader treats it as a floor, not an exact cost.
    acct = _account()
    acct.apply_fills(
        [{"symbol": "ALT/USDT:USDT", "direction": "long", "target_notional": 50_000.0}],
        marks={"ALT/USDT:USDT": 100.0},
        costs={"ALT/USDT:USDT": CostInputs(
            depth_bids=[(100.0, 10.0)], depth_asks=[(110.0, 10.0)])},  # only 10 units @ +10
        opened_ts=_TS, opened_cycle=1, opened_cadence="weekly",
    )
    pos = acct.positions["ALT/USDT:USDT"]
    assert pos.qty == 500.0                       # FULL target qty opened (50_000 / 100)
    # slippage priced on the 10 filled units * (110 - 100) = 100 USDT (the partial), NOT on 500.
    assert acct.slippage_paid == 100.0            # under-costed vs the true 500-unit impact


def test_geometry_cost_maps_threads_both_book_sides():
    bundle = {"geometries": [{
        "symbol": "ALT/USDT:USDT", "mark": 100.0, "funding_rate": 0.0,
        "funding_interval_hours": 8, "adv_usd": 1e6,
        "depth_bids": [[99.0, 5.0]], "depth_asks": [[101.0, 4.0]],
    }]}
    _marks, _funding, _intervals, costs = _geometry_cost_maps(bundle)
    ci = costs["ALT/USDT:USDT"]
    assert ci.depth_asks == [(101.0, 4.0)]
    assert ci.depth_bids == [(99.0, 5.0)]
    assert ci.adv_usd == 1e6

from __future__ import annotations

from futures_fund.contracts import CoinGeometry, TargetWeights, WeightLeg
from futures_fund.models import TradeProposal
from futures_fund.risk_gate import _reward_risk
from futures_fund.trader_io import proposals_from_book

NOW = "2026-06-11T00:00:00+00:00"


def _book() -> TargetWeights:
    return TargetWeights(
        legs=[
            WeightLeg(symbol="BTC/USDT:USDT", direction="long", weight=0.45,
                      target_notional=9000.0, beta_btc=1.0, sleeve="factor"),
            WeightLeg(symbol="ETH/USDT:USDT", direction="short", weight=-0.45,
                      target_notional=9000.0, beta_btc=1.0, sleeve="factor"),
            # a flat (zero-notional) carry-over leg: must NOT become a proposal
            WeightLeg(symbol="SOL/USDT:USDT", direction="long", weight=0.0,
                      target_notional=0.0, beta_btc=1.0, sleeve="factor"),
        ],
        dollar_residual=0.0, dollar_residual_frac=0.0, beta_residual=0.0,
        gross_long=9000.0, gross_short=9000.0,
        deploy_long_frac=0.9, deploy_short_frac=0.9, gross_notional=18000.0, as_of_ts=NOW,
    )


def _geos() -> list[CoinGeometry]:
    return [
        CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0, funding_rate=0.0001,
                     funding_interval_hours=8.0),
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, funding_rate=0.0001,
                     funding_interval_hours=8.0),
    ]


def test_proposals_skip_flat_legs_and_validate_as_tradeproposal():
    props = proposals_from_book(_book(), _geos(), rr=2.0, stop_frac=0.02)
    assert [p.symbol for p in props] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert all(isinstance(p, TradeProposal) for p in props)
    # entry == mark, funding wired from geometry
    btc = next(p for p in props if p.symbol == "BTC/USDT:USDT")
    assert btc.entry == 60000.0
    assert btc.funding_rate == 0.0001
    assert btc.funding_interval_hours == 8.0


def test_proposals_clear_the_min_rr_floor():
    # stop 2% away, TP at rr*stop -> RR == rr exactly, so the gate's MIN_RR (2.0) is met.
    props = proposals_from_book(_book(), _geos(), rr=2.0, stop_frac=0.02)
    for p in props:
        assert _reward_risk(p) >= 2.0 - 1e-9


def test_short_leg_has_stop_above_and_tp_below_entry():
    props = proposals_from_book(_book(), _geos(), rr=2.0, stop_frac=0.02)
    eth = next(p for p in props if p.symbol == "ETH/USDT:USDT")
    assert eth.direction == "short"
    assert eth.stop > eth.entry           # short stop above entry
    assert eth.take_profits[0] < eth.entry  # short TP below entry

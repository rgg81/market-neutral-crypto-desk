from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import TargetWeights
from futures_fund.neutrality import NeutralityConfig, optimize_book

NOW = datetime(2026, 6, 11, tzinfo=UTC)


def test_optimize_book_returns_target_weights(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(
        sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg
    )
    assert isinstance(tw, TargetWeights)
    assert tw.feasible is True
    assert tw.as_of_ts is not None


def test_optimize_book_sets_per_side_deployment_and_gross(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    assert tw.gross_long > 0.0
    assert tw.gross_short > 0.0
    assert tw.gross_notional == tw.gross_long + tw.gross_short


def test_optimize_book_each_leg_has_target_notional(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    assert len(tw.legs) > 0
    for leg in tw.legs:
        assert leg.target_notional != 0.0
        assert leg.beta_btc != 0.0


def test_optimize_book_includes_hedge_leg_when_residual_beta(geometries):
    from futures_fund.contracts import SleeveSignal, SleeveTilt

    # A beta-imbalanced book: long the HIGH-beta name (SOL 1.5), short the LOW-beta name
    # (XRP 0.8). The alpha legs carry a NET LONG beta, so the BTC hedge MUST be a non-zero
    # SHORT BTC leg that absorbs it (the hedge is a real DOF, sized before projection).
    s = SleeveSignal(
        sleeve="factor",
        risk_budget_frac=1.0,
        as_of_ts=NOW,
        tilts=[
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )
    cfg = NeutralityConfig()
    tw = optimize_book([s], geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    hedge_legs = [leg for leg in tw.legs if leg.sleeve == "hedge"]
    # NON-vacuous: this beta-imbalanced book REQUIRES a materialized BTC hedge leg.
    assert tw.btc_hedge_notional < 0.0  # net long beta => short BTC hedge
    assert hedge_legs
    assert hedge_legs[0].symbol == "BTC/USDT:USDT"
    assert hedge_legs[0].direction == "short"

from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import CoinGeometry, SleeveTilt
from futures_fund.sleeves.sentiment import apply_conviction_tilts, conviction_tilt

_NOW = datetime(2026, 6, 11, tzinfo=UTC)


def test_conviction_tilt_positive_sentiment_boosts_long():
    # sign-aligned magnitude tilt |w|*(1 + kappa*sign(w)*s*conf) (canonical contract §7.2):
    # unclamped delta = 0.2 * 0.5*(+1)*0.8*1.0 = +0.08 (+40%), but |delta| <= cap*|w| = 0.25*0.2
    # => clamped to +0.05, so tilted = 0.2 + 0.05 = 0.25.
    assert abs(conviction_tilt(0.2, 0.8, 1.0, kappa=0.5) - 0.25) < 1e-9


def test_conviction_tilt_negative_sentiment_shrinks_long():
    # sign-aligned magnitude tilt |w|*(1 + kappa*sign(w)*s*conf) (canonical contract §7.2):
    # unclamped delta = 0.2 * 0.5*(+1)*(-0.8)*1.0 = -0.08 (-40%), but |delta| <= cap*|w| = 0.25*0.2
    # => clamped to -0.05, so tilted = 0.2 - 0.05 = 0.15.
    assert abs(conviction_tilt(0.2, -0.8, 1.0, kappa=0.5) - 0.15) < 1e-9


def test_conviction_tilt_never_flips_sign():
    # huge negative sentiment cannot push a long weight negative
    out = conviction_tilt(0.2, -1.0, 1.0, kappa=5.0)
    assert out >= 0.0


def test_conviction_tilt_cap_limits_delta_to_25pct():
    # cap=0.25 -> |delta w| <= 25% of |w|, so max tilted long = 0.2 * 1.25 = 0.25
    out = conviction_tilt(0.2, 1.0, 1.0, kappa=5.0, cap=0.25)
    assert abs(out - 0.25) < 1e-9


def test_conviction_tilt_zero_weight_stays_zero():
    # sentiment never OPENS a position on its own
    assert conviction_tilt(0.0, 1.0, 1.0, kappa=0.5) == 0.0


def test_conviction_tilt_short_leg_negative_weight():
    # short leg w=-0.2, positive sentiment should SHRINK the short magnitude (toward 0)
    out = conviction_tilt(-0.2, 0.8, 1.0, kappa=0.5)
    assert -0.2 < out <= 0.0


def test_apply_conviction_tilts_maps_per_symbol_geometry():
    # Per-symbol geometry mapping over a long and a short leg. Values follow the CANONICAL
    # conviction_tilt (interface contract §2.9): the sign-aligned MAGNITUDE tilt
    # |w| <- |w|*(1 + kappa*sign(w)*s*conf) clamped so |Delta w| <= cap*|w|. (The plan's Task 20
    # note arithmetic used the superseded scalar `w*(1 + kappa*s*conf)` form -- contract §2.9
    # marks it "wrong-for-shorts and superseded" -- so the canonical capped values are asserted
    # here. apply_conviction_tilts passes the RAW score; conviction_tilt signs by direction.)
    legs = [
        SleeveTilt(symbol="A/USDT:USDT", direction="long", target_weight=0.2),
        SleeveTilt(symbol="B/USDT:USDT", direction="short", target_weight=-0.2),
    ]
    geos = [
        CoinGeometry(symbol="A/USDT:USDT", mark=100.0, sentiment_score=0.8, sentiment_conf=1.0),
        CoinGeometry(symbol="B/USDT:USDT", mark=100.0, sentiment_score=0.8, sentiment_conf=1.0),
    ]
    out = apply_conviction_tilts(legs, geos, kappa=0.5, cap=0.25)
    by_sym = {t.symbol: t for t in out}
    # A: long, favorable sentiment -> magnitude grows, capped to +25% of |w| => 0.2 + 0.05 = 0.25.
    assert abs(by_sym["A/USDT:USDT"].target_weight - 0.25) < 1e-9
    # B: short leg, positive sentiment is UNFAVORABLE to a short, so its magnitude shrinks toward 0,
    #    capped to 25% of |w| => -0.2 + 0.05 = -0.15. Sign is preserved (still a short).
    assert abs(by_sym["B/USDT:USDT"].target_weight - (-0.15)) < 1e-9
    assert by_sym["B/USDT:USDT"].target_weight < 0.0


def test_apply_conviction_tilts_missing_geometry_is_unchanged():
    legs = [SleeveTilt(symbol="Z/USDT:USDT", direction="long", target_weight=0.2)]
    out = apply_conviction_tilts(legs, [], kappa=0.5, cap=0.25)
    assert out[0].target_weight == 0.2              # no geometry -> no tilt (fail-soft neutral)

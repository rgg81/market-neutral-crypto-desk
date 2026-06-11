from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.sleeves.sentiment import conviction_tilt

_NOW = datetime(2026, 6, 11, tzinfo=UTC)


def test_conviction_tilt_positive_sentiment_boosts_long():
    # w*(1 + kappa*s*conf) = 0.2 * (1 + 0.5*0.8*1.0) = 0.2 * 1.4 = 0.28
    assert abs(conviction_tilt(0.2, 0.8, 1.0, kappa=0.5) - 0.28) < 1e-9


def test_conviction_tilt_negative_sentiment_shrinks_long():
    # 0.2 * (1 + 0.5*(-0.8)*1.0) = 0.2 * 0.6 = 0.12
    assert abs(conviction_tilt(0.2, -0.8, 1.0, kappa=0.5) - 0.12) < 1e-9


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

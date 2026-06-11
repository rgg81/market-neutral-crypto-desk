from datetime import UTC, datetime

from futures_fund.contracts import CoinGeometry
from futures_fund.sleeves.carry import carry_signal

_NOW = datetime(2026, 6, 11, tzinfo=UTC)


def _geo(symbol: str, apr: float) -> CoinGeometry:
    return CoinGeometry(symbol=symbol, mark=100.0, funding_apr=apr)


def test_carry_signal_shorts_high_funding_longs_negative_funding():
    geos = [
        _geo("A/USDT:USDT", 0.30),     # high positive carry -> crowded long -> SHORT
        _geo("B/USDT:USDT", 0.10),
        _geo("C/USDT:USDT", -0.05),
        _geo("D/USDT:USDT", -0.25),    # negative carry -> SHORTS pay us -> LONG
    ]
    sig = carry_signal(geos, risk_budget_frac=0.25, now=_NOW)
    assert sig.sleeve == "carry"
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["A/USDT:USDT"].direction == "short"
    assert by_sym["D/USDT:USDT"].direction == "long"
    # signed, un-clamped raw_score == funding_apr (carry credit visible, never zeroed)
    assert by_sym["A/USDT:USDT"].raw_score == 0.30
    assert by_sym["D/USDT:USDT"].raw_score == -0.25
    # long weights positive, short weights negative
    assert by_sym["D/USDT:USDT"].target_weight > 0
    assert by_sym["A/USDT:USDT"].target_weight < 0


def test_carry_signal_top_frac_limits_legs():
    geos = [_geo(f"{c}/USDT:USDT", apr) for c, apr in
            zip("ABCDEF", [0.3, 0.2, 0.1, -0.1, -0.2, -0.3], strict=True)]
    # top_frac=1/3 matches the tercile convention used by the factor/sentiment sleeves:
    # floor(6 * 1/3) = 2 -> 2 longs + 2 shorts.
    sig = carry_signal(geos, risk_budget_frac=0.25, now=_NOW, top_frac=1 / 3)
    longs = [t for t in sig.tilts if t.direction == "long"]
    shorts = [t for t in sig.tilts if t.direction == "short"]
    assert len(longs) == 2
    assert len(shorts) == 2


def test_carry_signal_default_top_frac_is_exact_tercile():
    # The default top_frac must be the exact tercile (1/3), matching the factor/sentiment
    # sleeves' `tercile=1/3` convention -- NOT 0.33. For n=6 the two diverge:
    # floor(6 * 1/3) = floor(2.0) = 2, but floor(6 * 0.33) = floor(1.98) = 1.
    # Pin the agreed behavior on the default (no top_frac passed): 2 legs per side.
    geos = [_geo(f"{c}/USDT:USDT", apr) for c, apr in
            zip("ABCDEF", [0.3, 0.2, 0.1, -0.1, -0.2, -0.3], strict=True)]
    sig = carry_signal(geos, risk_budget_frac=0.25, now=_NOW)
    longs = [t for t in sig.tilts if t.direction == "long"]
    shorts = [t for t in sig.tilts if t.direction == "short"]
    assert len(longs) == 2
    assert len(shorts) == 2
    assert sig.diagnostics["k_per_side"] == 2


def test_carry_signal_empty_geometries():
    sig = carry_signal([], risk_budget_frac=0.25, now=_NOW)
    assert sig.tilts == []


def test_package_reexports_all_four_builders():
    from futures_fund.sleeves import (
        carry_signal,
        factor_signal,
        pairs_signal,
        sentiment_factor_signal,
    )
    assert callable(carry_signal)
    assert callable(pairs_signal)
    assert callable(factor_signal)
    assert callable(sentiment_factor_signal)

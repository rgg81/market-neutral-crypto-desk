from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import CoinGeometry
from futures_fund.sleeves.factor import factor_signal, rank_factor

_NOW = datetime(2026, 6, 11, tzinfo=UTC)


def _geo(symbol, *, mom=0.0, apr=0.0, vol=0.1):
    return CoinGeometry(symbol=symbol, mark=100.0, momentum_20=mom,
                        funding_apr=apr, realized_vol=vol)


def test_rank_factor_momentum_high_first():
    geos = [_geo("A/USDT:USDT", mom=0.1), _geo("B/USDT:USDT", mom=0.3),
            _geo("C/USDT:USDT", mom=-0.2)]
    ranked = rank_factor(geos, factor="momentum")
    assert [s for s, _ in ranked] == ["B/USDT:USDT", "A/USDT:USDT", "C/USDT:USDT"]


def test_rank_factor_carry_uses_negative_funding_apr():
    # carry factor: LOW funding_apr is attractive (we get paid), so score = -funding_apr
    geos = [_geo("A/USDT:USDT", apr=0.3), _geo("B/USDT:USDT", apr=-0.3)]
    ranked = rank_factor(geos, factor="carry")
    assert ranked[0][0] == "B/USDT:USDT"          # negative funding ranks best
    assert ranked[0][1] == 0.3                     # score = -(-0.3)


def test_rank_factor_low_vol_prefers_low_realized_vol():
    geos = [_geo("A/USDT:USDT", vol=0.5), _geo("B/USDT:USDT", vol=0.1)]
    ranked = rank_factor(geos, factor="low_vol")
    assert ranked[0][0] == "B/USDT:USDT"          # lower vol ranks best


def test_factor_signal_tercile_long_short_combined():
    # 6 names with monotone momentum; tercile (1/3 of 6 = 2) -> 2 longs (top) + 2 shorts (bottom)
    geos = [_geo(f"{c}/USDT:USDT", mom=m, vol=0.1)
            for c, m in zip("ABCDEF", [0.5, 0.4, 0.1, -0.1, -0.4, -0.5], strict=True)]
    sig = factor_signal(geos, risk_budget_frac=0.25, now=_NOW,
                        factors=["momentum"], tercile=1 / 3, weighting="equal")
    assert sig.sleeve == "factor"
    longs = {t.symbol for t in sig.tilts if t.direction == "long"}
    shorts = {t.symbol for t in sig.tilts if t.direction == "short"}
    assert longs == {"A/USDT:USDT", "B/USDT:USDT"}
    assert shorts == {"E/USDT:USDT", "F/USDT:USDT"}


def test_factor_signal_inverse_vol_weights_lower_vol_heavier():
    geos = [_geo("A/USDT:USDT", mom=0.5, vol=0.1),   # low vol -> heavier
            _geo("B/USDT:USDT", mom=0.4, vol=0.4),   # high vol -> lighter
            _geo("C/USDT:USDT", mom=-0.4, vol=0.2),
            _geo("D/USDT:USDT", mom=-0.5, vol=0.2)]
    sig = factor_signal(geos, risk_budget_frac=0.25, now=_NOW,
                        factors=["momentum"], tercile=0.5, weighting="inverse_vol")
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["A/USDT:USDT"].target_weight > by_sym["B/USDT:USDT"].target_weight
    # weights within the long side sum to ~1.0
    long_sum = sum(t.target_weight for t in sig.tilts if t.direction == "long")
    assert abs(long_sum - 1.0) < 1e-9


def test_factor_signal_combines_multiple_factors_by_rank():
    geos = [_geo("A/USDT:USDT", mom=0.9, apr=-0.3),   # best on both momentum & carry
            _geo("B/USDT:USDT", mom=0.1, apr=0.0),
            _geo("C/USDT:USDT", mom=-0.9, apr=0.3)]   # worst on both
    sig = factor_signal(geos, risk_budget_frac=0.25, now=_NOW,
                        factors=["momentum", "carry"], tercile=1 / 3, weighting="equal")
    longs = {t.symbol for t in sig.tilts if t.direction == "long"}
    shorts = {t.symbol for t in sig.tilts if t.direction == "short"}
    assert "A/USDT:USDT" in longs
    assert "C/USDT:USDT" in shorts


def test_factor_signal_n3_default_tercile_one_long_one_short_no_overlap():
    # Small-N no-overlap guard: n=3 with the DEFAULT tercile (1/3) gives k=floor(3/3)=1, so the
    # sleeve emits EXACTLY one long (top combined rank) and one short (bottom) with the long and
    # short sides DISJOINT -- the middle name is held out, never double-counted on both legs.
    geos = [_geo("A/USDT:USDT", mom=0.5),    # best momentum -> the single LONG
            _geo("B/USDT:USDT", mom=0.0),    # middle -> held out
            _geo("C/USDT:USDT", mom=-0.5)]   # worst momentum -> the single SHORT
    sig = factor_signal(geos, risk_budget_frac=0.0, now=_NOW,
                        factors=["momentum"], weighting="equal")  # default tercile=1/3
    longs = [t.symbol for t in sig.tilts if t.direction == "long"]
    shorts = [t.symbol for t in sig.tilts if t.direction == "short"]
    assert longs == ["A/USDT:USDT"]
    assert shorts == ["C/USDT:USDT"]
    assert len(longs) == 1 and len(shorts) == 1
    assert set(longs) & set(shorts) == set()           # disjoint sides on a tiny universe

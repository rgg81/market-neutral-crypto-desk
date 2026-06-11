from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import CoinGeometry
from futures_fund.sleeves.factor import rank_factor

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

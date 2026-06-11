# tests/test_cycle_prep.py
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from futures_fund.contracts import CoinGeometry as _CG
from futures_fund.contracts import GeometryBundle
from futures_fund.cycle_prep import build_geometries, build_sleeves

NOW = datetime(2026, 6, 11, tzinfo=UTC)


class _FakeExchange:
    """Duck-typed FuturesExchange: returns deterministic OHLCV + funding per symbol."""

    def __init__(self, marks: dict[str, float], funding: dict[str, float]):
        self._marks = marks
        self._funding = funding

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        # 120 candles; each symbol a random walk anchored at its mark.
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        n = 120
        base = self._marks[symbol]
        rets = rng.normal(0, 0.01, n)
        closes = base * np.exp(np.cumsum(rets))
        ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        return pd.DataFrame({
            "timestamp": ts, "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": 1000.0,
        })

    def funding(self, symbol):
        from futures_fund.market_data import FundingInfo
        return FundingInfo(
            symbol=symbol, current_rate=self._funding[symbol],
            next_funding_ts=NOW, interval_hours=8.0,
            mark_price=self._marks[symbol], index_price=self._marks[symbol],
        )

    def mark_price(self, symbol):
        return self._marks[symbol]


def _ex():
    return _FakeExchange(
        marks={"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0, "SOL/USDT:USDT": 150.0},
        funding={"BTC/USDT:USDT": 0.0001, "ETH/USDT:USDT": 0.0005, "SOL/USDT:USDT": -0.0003},
    )


def test_build_geometries_returns_one_geometry_per_symbol():
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    bundle = build_geometries(_ex(), symbols, now=NOW, btc_symbol="BTC/USDT:USDT",
                              beta_lookback=45)
    assert isinstance(bundle, GeometryBundle)
    assert {g.symbol for g in bundle.geometries} == set(symbols)


def test_btc_beta_is_one_and_funding_apr_is_signed():
    bundle = build_geometries(_ex(), ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
                              now=NOW, btc_symbol="BTC/USDT:USDT", beta_lookback=45)
    by = {g.symbol: g for g in bundle.geometries}
    assert by["BTC/USDT:USDT"].beta_btc == 1.0
    # SOL funding is negative -> funding_apr signed negative (carry credit visible)
    assert by["SOL/USDT:USDT"].funding_apr < 0.0
    assert by["ETH/USDT:USDT"].funding_apr > 0.0
    # mark carried through
    assert by["ETH/USDT:USDT"].mark == 3000.0


def test_funding_rate_is_clamped_sign_preserving():
    ex = _FakeExchange(
        marks={"BTC/USDT:USDT": 60000.0, "DOGE/USDT:USDT": 0.15},
        funding={"BTC/USDT:USDT": 0.0001, "DOGE/USDT:USDT": 0.5},  # 50% -> over alt cap 2%
    )
    bundle = build_geometries(ex, ["BTC/USDT:USDT", "DOGE/USDT:USDT"], now=NOW,
                              btc_symbol="BTC/USDT:USDT", beta_lookback=45)
    doge = next(g for g in bundle.geometries if g.symbol == "DOGE/USDT:USDT")
    assert doge.funding_rate == 0.02  # clamped to alt cap (PER_SYMBOL_CAP_DEFAULT), sign preserved


def _six_geos() -> list[_CG]:
    syms = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
            "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT"]
    out = []
    for i, s in enumerate(syms):
        out.append(_CG(symbol=s, mark=100.0 + i, beta_btc=1.0,
                       funding_rate=0.0001 * (i - 2), funding_interval_hours=8.0,
                       funding_apr=0.001 * (i - 2), momentum_20=0.1 * (i - 2),
                       realized_vol=0.5, sentiment_score=0.2 * (i - 2),
                       sentiment_conf=0.8))
    return out


def test_build_sleeves_emits_the_four_named_sleeves():
    sleeves = build_sleeves(_six_geos(), pairs=[], spreads=[], now=NOW)
    names = {s.sleeve for s in sleeves}
    assert names == {"carry", "pairs", "factor", "sentiment"}


def test_risk_budgets_assigned_and_sum_to_one():
    sleeves = build_sleeves(_six_geos(), pairs=[], spreads=[], now=NOW)
    total = sum(s.risk_budget_frac for s in sleeves)
    assert abs(total - 1.0) < 1e-9


def test_sleeves_round_trip_through_the_control_loop_cli_shape():
    # control_loop_cli loads {"sleeves": [SleeveSignal-dict, ...]}; assert that shape validates.
    from futures_fund.contracts import SleeveSignal
    sleeves = build_sleeves(_six_geos(), pairs=[], spreads=[], now=NOW)
    payload = {"sleeves": [s.model_dump(mode="json") for s in sleeves]}
    reloaded = [SleeveSignal.model_validate(s) for s in payload["sleeves"]]
    assert len(reloaded) == 4

# tests/test_cycle_prep.py
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from futures_fund.contracts import CoinGeometry as _CG
from futures_fund.contracts import GeometryBundle, Pair, Spread
from futures_fund.cycle_prep import (
    build_geometries,
    build_pairs_and_spreads,
    build_sleeves,
)
from futures_fund.market_data import FundingInfo

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


class _CointExchange:
    """Two cointegrated legs (y = 2x + noise) + an independent leg."""

    def __init__(self):
        rng = np.random.default_rng(7)
        n = 200
        x = np.cumsum(rng.normal(0, 1, n)) + 100.0
        noise = rng.normal(0, 0.5, n)
        self._series = {
            "AAA/USDT:USDT": pd.Series(x),
            "BBB/USDT:USDT": pd.Series(2.0 * x + noise),       # cointegrated with AAA
            "CCC/USDT:USDT": pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50.0),  # independent
        }
        self._marks = {"AAA/USDT:USDT": float(x[-1]),
                       "BBB/USDT:USDT": float(2.0 * x[-1] + noise[-1]),
                       "CCC/USDT:USDT": 50.0}

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        s = self._series[symbol]
        ts = pd.date_range("2026-01-01", periods=len(s), freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": s, "high": s, "low": s,
                             "close": s, "volume": 1.0})

    def mark_price(self, symbol):
        return self._marks[symbol]


def test_build_pairs_finds_the_cointegrated_pair():
    ex = _CointExchange()
    syms = ["AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT"]
    pairs, spreads = build_pairs_and_spreads(ex, syms, cycle=1, now=NOW,
                                             adf_pvalue_max=0.05, fdr_method="bh")
    assert all(isinstance(p, Pair) for p in pairs)
    assert all(isinstance(s, Spread) for s in spreads)
    # the AAA/BBB pair (cointegrated) survives FDR + select_pairs; one spread per kept pair
    kept_ids = {p.pair_id for p in pairs}
    assert any("AAA" in pid and "BBB" in pid for pid in kept_ids)
    assert {s.pair_id for s in spreads} == kept_ids


def test_build_pairs_round_trips_through_artifact_shape():
    ex = _CointExchange()
    pairs, spreads = build_pairs_and_spreads(
        ex, ["AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT"],
        cycle=1, now=NOW)
    pairs_payload = {"pairs": [p.model_dump(mode="json") for p in pairs]}
    spreads_payload = {"spreads": [s.model_dump(mode="json") for s in spreads]}
    assert [Pair.model_validate(p) for p in pairs_payload["pairs"]] == pairs
    assert [Spread.model_validate(s) for s in spreads_payload["spreads"]] == spreads


def test_coin_geometry_has_depth_and_quality_fields():
    g = _CG(
        symbol="BTC/USDT:USDT", mark=60000.0, adv_usd=2e9,
        depth_bids=[(60000.0, 5.0)], depth_asks=[(60001.0, 4.0)],
        onboard_date=1567965300000, chg_24h_pct=1.0,
    )
    assert g.depth_bids == [(60000.0, 5.0)]
    assert g.depth_asks == [(60001.0, 4.0)]
    assert g.onboard_date == 1567965300000
    assert g.chg_24h_pct == 1.0
    # defaults: a geometry built with no depth has empty books, not None crashes
    assert _CG(symbol="X/USDT:USDT", mark=1.0).depth_bids == []
    assert _CG(symbol="X/USDT:USDT", mark=1.0).onboard_date is None


_NOW_T7 = datetime(2026, 6, 12, tzinfo=UTC)


class _GeoExchange:
    def ohlcv(self, symbol, timeframe="4h", limit=500):
        ts = pd.date_range("2025-01-01", periods=60, freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 100.0,
                             "low": 100.0, "close": 100.0, "volume": 1.0})

    def funding(self, symbol):
        return FundingInfo(symbol=symbol, current_rate=0.0001,
                           next_funding_ts=_NOW_T7, interval_hours=8.0,
                           mark_price=100.0, index_price=100.0)

    def mark_price(self, symbol):
        return 100.0

    def depth(self, symbol, limit=20):
        return {"bids": [(99.0, 10.0)], "asks": [(101.0, 8.0)]}


def test_build_geometries_stamps_adv_depth_and_quality_metadata():
    rows = {"BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "vol_24h_usd": 2e9,
                              "chg_24h_pct": 1.5, "onboard_date": 1567965300000}}
    bundle = build_geometries(_GeoExchange(), ["BTC/USDT:USDT"], now=_NOW_T7,
                              universe_rows=rows)
    g = bundle.geometries[0]
    assert g.adv_usd == 2e9
    assert g.depth_asks == [(101.0, 8.0)]
    assert g.depth_bids == [(99.0, 10.0)]
    assert g.chg_24h_pct == 1.5
    assert g.onboard_date == 1567965300000


def test_build_geometries_fail_soft_without_depth_method():
    class _NoDepth(_GeoExchange):
        depth = None  # attribute present but not callable -> guarded
    bundle = build_geometries(_NoDepth(), ["BTC/USDT:USDT"], now=_NOW_T7)
    g = bundle.geometries[0]
    assert g.depth_bids == [] and g.depth_asks == []
    assert g.adv_usd == 0.0  # no universe_rows -> default


def test_build_sleeves_threads_carry_cap_into_carry_signal():
    geos = [_CG(symbol=f"{c}/USDT:USDT", mark=100.0, funding_apr=apr)
            for c, apr in zip("ABCDEF", [20.0, 1.0, 0.5, -0.5, -1.0, -20.0], strict=True)]
    sleeves = build_sleeves(geos, pairs=[], spreads=[], now=NOW, max_abs_apr=2.0)
    carry = next(s for s in sleeves if s.sleeve == "carry")
    scores = {t.symbol: t.raw_score for t in carry.tilts}
    # the extreme +20 APR name is shorted with a CAPPED +2.0 raw_score (bounded by the cap)
    assert scores["A/USDT:USDT"] == 2.0
    assert scores["F/USDT:USDT"] == -2.0
    assert carry.diagnostics["max_abs_apr"] == 2.0

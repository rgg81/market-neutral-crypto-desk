from __future__ import annotations

import json

import pandas as pd

from futures_fund.cycle_io import cycle_dir, load_output
from futures_fund.market_data import FundingInfo

_NOW_ISO = "2026-06-12T00:00:00+00:00"
_OLD_ONBOARD = "1567965300000"   # 2019 -> well past min_age_days
_ESTABLISHED = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
_MARKS = {"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0,
          "SOL/USDT:USDT": 150.0, "VELVET/USDT:USDT": 1.0}


class _FakeClient:
    markets = {
        **{s: {"info": {"underlyingType": "COIN", "onboardDate": _OLD_ONBOARD}}
           for s in _ESTABLISHED},
        # VELVET: future onboardDate (too young) -> dropped by the age gate
        "VELVET/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "9999999999999"}},
    }

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {
            **{s: {"last": _MARKS[s], "quoteVolume": 1e9, "percentage": 1.0}
               for s in _ESTABLISHED},
            "VELVET/USDT:USDT": {"last": 1.0, "quoteVolume": 9e8, "percentage": 130.0},
        }


class _FakeExchange:
    def depth(self, symbol, limit=20):
        mark = _MARKS[symbol]
        qty = 5_000_000.0 / mark                 # ~$5M/level -> full notional >> min_depth_usd
        return {"bids": [(mark * 0.999, qty)], "asks": [(mark * 1.001, qty)]}

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        ts = pd.date_range("2025-01-01", periods=60, freq="4h", tz="UTC")
        c = _MARKS[symbol]
        return pd.DataFrame({"timestamp": ts, "open": c, "high": c,
                             "low": c, "close": c, "volume": 1.0})

    def funding(self, symbol):
        return FundingInfo(symbol=symbol, current_rate=0.0001,
                           next_funding_ts=pd.Timestamp(_NOW_ISO).to_pydatetime(),
                           interval_hours=8.0, mark_price=_MARKS[symbol],
                           index_price=_MARKS[symbol])

    def mark_price(self, symbol):
        return _MARKS[symbol]

    @staticmethod
    def from_settings(settings):
        return _FakeExchange()


def test_scout_to_cycle_prep_excludes_young_and_pumped(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setattr("scripts.scout_cli.build_ccxt", lambda settings: _FakeClient())
    monkeypatch.setattr("scripts.scout_cli.FuturesExchange", _FakeExchange)
    monkeypatch.setattr("futures_fund.exchange.FuturesExchange.from_settings",
                        staticmethod(lambda settings: _FakeExchange()))

    from scripts.cycle_prep_cli import main as cycle_prep_main
    from scripts.scout_cli import main as scout_main

    scout_main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state), "--top", "30"])
    universe = json.loads(
        (cycle_dir(state, 1, cadence="weekly") / "universe.json").read_text())["universe"]
    syms = [r["symbol"] for r in universe]
    assert set(syms) == set(_ESTABLISHED)              # VELVET dropped (young + pumped)

    cycle_prep_main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state),
                     "--now", _NOW_ISO])
    geos = load_output(state, 1, "geometries", cadence="weekly")["geometries"]
    geo_syms = [g["symbol"] for g in geos]
    assert "VELVET/USDT:USDT" not in geo_syms
    assert set(geo_syms) == set(_ESTABLISHED)
    # honest-cost prerequisites are stamped: real ADV (not 0.0) and a non-empty crossing book
    btc = next(g for g in geos if g["symbol"] == "BTC/USDT:USDT")
    assert btc["adv_usd"] == 1e9
    assert btc["depth_asks"] and btc["depth_bids"]
    assert btc["onboard_date"] == int(_OLD_ONBOARD)

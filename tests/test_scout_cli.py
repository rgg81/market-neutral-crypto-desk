from __future__ import annotations

import json

import pandas as pd

from futures_fund.cycle_io import cycle_dir


class _FakeClient:
    markets = {
        "BTC/USDT:USDT": {"info": {"underlyingType": "COIN"}},
        "ETH/USDT:USDT": {"info": {"underlyingType": "COIN"}},
        "GOLD/USDT:USDT": {"info": {"underlyingType": "COMMODITY"}},  # excluded (not crypto)
    }

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {
            "BTC/USDT:USDT": {"last": 60000.0, "quoteVolume": 2e9, "percentage": 1.0},
            "ETH/USDT:USDT": {"last": 3000.0, "quoteVolume": 1e9, "percentage": 0.5},
            "GOLD/USDT:USDT": {"last": 2000.0, "quoteVolume": 5e9, "percentage": 0.1},
        }


class _FakeQualityClient:
    markets = {
        "BTC/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "1567965300000"}},
        "ETH/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "1574840700000"}},
        "VELVET/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "9999999999999"}},
    }

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {
            "BTC/USDT:USDT": {"last": 60000.0, "quoteVolume": 2e9, "percentage": 1.0},
            "ETH/USDT:USDT": {"last": 3000.0, "quoteVolume": 1e9, "percentage": 0.5},
            # new (future onboardDate, 9999999999999 ms ~= year 2286) AND +130% pump
            "VELVET/USDT:USDT": {"last": 1.0, "quoteVolume": 9e8, "percentage": 130.0},
        }


class _FakeQualityExchange:
    def depth(self, symbol, limit=20):
        return {"bids": [(1.0, 1_000_000.0)], "asks": [(1.0, 1_000_000.0)]}  # ~$1M/side, deep

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        ts = pd.date_range("2020-01-01", periods=200, freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0,
                             "low": 1.0, "close": 1.0, "volume": 1.0})


def test_scout_writes_crypto_only_universe(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.scout_cli.build_ccxt", lambda settings: _FakeClient())
    monkeypatch.setattr(
        "scripts.scout_cli.FuturesExchange",
        type("X", (), {"from_settings": staticmethod(lambda settings: _FakeQualityExchange())}))
    from scripts.scout_cli import main

    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(tmp_path / "state"),
          "--top", "30"])
    out = json.loads((cycle_dir(tmp_path / "state", 1, cadence="weekly") / "universe.json")
                     .read_text())
    syms = [r["symbol"] for r in out["universe"]]
    assert "BTC/USDT:USDT" in syms and "ETH/USDT:USDT" in syms
    assert "GOLD/USDT:USDT" not in syms  # TradFi-wrapper excluded by is_crypto_perp


def test_scout_excludes_new_and_pumped_names(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.scout_cli.build_ccxt", lambda settings: _FakeQualityClient())
    monkeypatch.setattr(
        "scripts.scout_cli.FuturesExchange",
        type("X", (), {"from_settings": staticmethod(lambda settings: _FakeQualityExchange())}))
    from scripts.scout_cli import main

    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(tmp_path / "state"),
          "--top", "30"])
    out = json.loads((cycle_dir(tmp_path / "state", 1, cadence="weekly") / "universe.json")
                     .read_text())
    syms = [r["symbol"] for r in out["universe"]]
    assert "BTC/USDT:USDT" in syms and "ETH/USDT:USDT" in syms
    assert "VELVET/USDT:USDT" not in syms  # young + pumped -> dropped by quality_filter
    # kept rows carry the metadata cycle_prep needs (proves quality_filter ran on the fake exchange,
    # not a silently-unpatched real-network from_settings)
    btc = next(r for r in out["universe"] if r["symbol"] == "BTC/USDT:USDT")
    assert btc["onboard_date"] == 1567965300000
    assert "chg_24h_pct" in btc and "vol_24h_usd" in btc

# tests/test_cycle_prep_cli.py
from __future__ import annotations

import numpy as np
import pandas as pd

from futures_fund.contracts import GeometryBundle, Pair, SleeveSignal, Spread
from futures_fund.cycle_io import cycle_dir, load_output
from futures_fund.market_data import FundingInfo


class _FakeExchange:
    def __init__(self, symbols):
        self._symbols = symbols
        self._marks = {s: 100.0 + i for i, s in enumerate(symbols)}

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        closes = self._marks[symbol] * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
        ts = pd.date_range("2026-01-01", periods=120, freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": closes, "high": closes,
                             "low": closes, "close": closes, "volume": 1.0})

    def funding(self, symbol):
        return FundingInfo(symbol=symbol, current_rate=0.0001, next_funding_ts=pd.Timestamp(
            "2026-06-11", tz="UTC").to_pydatetime(), interval_hours=8.0,
            mark_price=self._marks[symbol], index_price=self._marks[symbol])

    def mark_price(self, symbol):
        return self._marks[symbol]


def test_cycle_prep_cli_writes_all_four_artifacts(tmp_path, monkeypatch):
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT"]
    monkeypatch.setattr(
        "scripts.cycle_prep_cli.FuturesExchange.from_settings",
        lambda settings: _FakeExchange(symbols),
    )
    # universe.json the CLI reads its symbol set from
    from futures_fund.cycle_io import save_output
    save_output(tmp_path / "state", 1, "universe",
                {"universe": [{"symbol": s} for s in symbols]}, cadence="weekly")
    from scripts.cycle_prep_cli import main

    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(tmp_path / "state"),
          "--now", "2026-06-11T00:00:00+00:00"])

    root = cycle_dir(tmp_path / "state", 1, cadence="weekly")
    assert (root / "geometries.json").exists()
    assert (root / "sleeves.json").exists()
    assert (root / "pairs.json").exists()
    assert (root / "spreads.json").exists()
    # shapes the loop/reviewer load
    GeometryBundle.model_validate(load_output(tmp_path / "state", 1, "geometries",
                                              cadence="weekly"))
    sleeves = [SleeveSignal.model_validate(s)
               for s in load_output(tmp_path / "state", 1, "sleeves", cadence="weekly")["sleeves"]]
    assert {s.sleeve for s in sleeves} == {"carry", "pairs", "factor", "sentiment"}
    [Pair.model_validate(p)
     for p in load_output(tmp_path / "state", 1, "pairs", cadence="weekly")["pairs"]]
    [Spread.model_validate(s)
     for s in load_output(tmp_path / "state", 1, "spreads", cadence="weekly")["spreads"]]

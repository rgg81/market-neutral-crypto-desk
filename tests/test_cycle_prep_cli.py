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


def test_cycle_prep_cli_prices_btc_hedge_even_when_not_in_universe(tmp_path, monkeypatch):
    """The configured BTC hedge/beta symbol must ALWAYS get a priced CoinGeometry, even when the
    scout universe is all-alts and BTC is NOT selected. `optimize_book` unconditionally appends a
    BTC hedge leg; without a BTC geometry (hence no BTC mark) `apply_fills` silently SKIPS the hedge
    and the held book goes net-short. BINDS: pre-fix the CLI builds geometries only over the
    universe, so there is NO BTC geometry and this fails."""
    from futures_fund.config import load_settings

    btc = load_settings().beta.btc_symbol  # the hedge/beta reference (default BTC/USDT:USDT)
    # an all-ALTS universe that deliberately EXCLUDES the hedge/beta symbol
    universe = ["ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
                "ADA/USDT:USDT", "DOGE/USDT:USDT"]
    assert btc not in universe
    # the fake exchange CAN price BTC (it is infrastructure the desk always reads) — it just is not
    # in the tradable universe the scout selected.
    ex_symbols = [*universe, btc]
    monkeypatch.setattr(
        "scripts.cycle_prep_cli.FuturesExchange.from_settings",
        lambda settings: _FakeExchange(ex_symbols),
    )
    from futures_fund.cycle_io import save_output
    save_output(tmp_path / "state", 1, "universe",
                {"universe": [{"symbol": s} for s in universe]}, cadence="weekly")
    from scripts.cycle_prep_cli import main

    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(tmp_path / "state"),
          "--now", "2026-06-11T00:00:00+00:00"])

    bundle = GeometryBundle.model_validate(
        load_output(tmp_path / "state", 1, "geometries", cadence="weekly"))
    by_symbol = {g.symbol: g for g in bundle.geometries}
    # the BTC hedge/beta symbol is priced (real mark) even though it is NOT in the universe.
    assert btc in by_symbol, "the hedge/beta symbol must always get a CoinGeometry"
    assert by_symbol[btc].mark > 0.0
    assert by_symbol[btc].beta_btc == 1.0  # BTC self-beta
    # but BTC stays OUT of the tradable-alpha selection: it is not a pair leg and not a sleeve tilt.
    pairs = load_output(tmp_path / "state", 1, "pairs", cadence="weekly")["pairs"]
    pair_syms = {p["symbol_y"] for p in pairs} | {p["symbol_x"] for p in pairs}
    assert btc not in pair_syms, "the hedge symbol must not be forced into the alpha pair universe"
    sleeves = load_output(tmp_path / "state", 1, "sleeves", cadence="weekly")["sleeves"]
    sleeve_tilt_syms = {
        t["symbol"]
        for s in sleeves
        for t in s.get("tilts", [])
    }
    assert btc not in sleeve_tilt_syms, "the hedge symbol must not be forced as an alpha tilt"

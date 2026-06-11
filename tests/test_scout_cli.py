from __future__ import annotations

import json

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


def test_scout_writes_crypto_only_universe(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.scout_cli.build_ccxt", lambda settings: _FakeClient())
    from scripts.scout_cli import main

    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(tmp_path / "state"),
          "--top", "30"])
    out = json.loads((cycle_dir(tmp_path / "state", 1, cadence="weekly") / "universe.json")
                     .read_text())
    syms = [r["symbol"] for r in out["universe"]]
    assert "BTC/USDT:USDT" in syms and "ETH/USDT:USDT" in syms
    assert "GOLD/USDT:USDT" not in syms  # TradFi-wrapper excluded by is_crypto_perp

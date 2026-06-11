# tests/test_monitor_book.py
from __future__ import annotations

import json

from futures_fund.contracts import TargetWeights, WeightLeg
from futures_fund.monitor_book import write_monitor_book

NOW = "2026-06-11T00:00:00+00:00"


def _book() -> TargetWeights:
    return TargetWeights(
        legs=[
            WeightLeg(symbol="BTC/USDT:USDT", direction="long", weight=0.45,
                      target_notional=9000.0, beta_btc=1.0, sleeve="factor"),
            WeightLeg(symbol="ETH/USDT:USDT", direction="short", weight=-0.45,
                      target_notional=9000.0, beta_btc=1.0, sleeve="factor"),
        ],
        dollar_residual=0.0, dollar_residual_frac=0.0, beta_residual=0.0,
        gross_long=9000.0, gross_short=9000.0,
        deploy_long_frac=0.9, deploy_short_frac=0.9, gross_notional=18000.0, as_of_ts=NOW,
    )


def test_write_monitor_book_shapes_the_legs_the_monitor_reads(tmp_path):
    marks = {"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0}
    liqs = {"BTC/USDT:USDT": 30000.0, "ETH/USDT:USDT": 6000.0}
    write_monitor_book(tmp_path / "state", _book(), marks=marks, liq_prices=liqs,
                       balance=20000.0, peak_equity=20000.0)
    book = json.loads((tmp_path / "state" / "monitor_book.json").read_text())
    assert book["balance"] == 20000.0
    syms = {leg["symbol"] for leg in book["legs"]}
    assert syms == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    btc = next(leg for leg in book["legs"] if leg["symbol"] == "BTC/USDT:USDT")
    assert btc["mark"] == 60000.0 and btc["liq_price"] == 30000.0
    assert btc["notional"] == 9000.0 and btc["beta_btc"] == 1.0


def test_monitor_cli_evaluates_the_written_book(tmp_path):
    # the book the writer produces is consumed by the monitor without a HALT (neutral, no breach)
    from scripts.monitor_cli import main
    marks = {"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0}
    liqs = {"BTC/USDT:USDT": 30000.0, "ETH/USDT:USDT": 6000.0}
    write_monitor_book(tmp_path / "state", _book(), marks=marks, liq_prices=liqs,
                       balance=20000.0, peak_equity=20000.0)
    assert main(["--state-dir", str(tmp_path / "state")]) == 0  # no HALT

# tests/test_preflight_cli.py
from __future__ import annotations

import json

from futures_fund.cycle_io import cycle_dir, save_output
from scripts.preflight import build_briefs, main


def test_build_briefs_folds_held_symbols():
    universe = [{"symbol": "BTC/USDT:USDT"}, {"symbol": "ETH/USDT:USDT"}]
    held = ["SOL/USDT:USDT", "BTC/USDT:USDT"]  # SOL held but not in universe -> folded in
    briefs = build_briefs(universe, held)
    syms = {b["symbol"] for b in briefs}
    assert syms == {"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"}
    sol = next(b for b in briefs if b["symbol"] == "SOL/USDT:USDT")
    assert sol["held"] is True
    btc = next(b for b in briefs if b["symbol"] == "BTC/USDT:USDT")
    assert btc["held"] is True  # held AND in-universe


def test_preflight_writes_context(tmp_path, capsys):
    state = tmp_path / "state"
    save_output(state, 1, "universe",
                {"universe": [{"symbol": "BTC/USDT:USDT"}, {"symbol": "ETH/USDT:USDT"}]},
                cadence="weekly")
    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state)])
    ctx = json.loads((cycle_dir(state, 1, cadence="weekly") / "context.json").read_text())
    assert {b["symbol"] for b in ctx["briefs"]} == {"BTC/USDT:USDT", "ETH/USDT:USDT"}

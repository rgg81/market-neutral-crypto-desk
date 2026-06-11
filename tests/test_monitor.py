"""Task 3.8 — between-cycle light risk monitor (`scripts/monitor_cli.py`).

The monitor runs on a faster cron than the cadence cycles and trips HALT on a drawdown,
liq-distance, or NEUTRALITY-RESIDUAL breach (spec §9 / §19). It is a no-op when all three
guards are in band. Adapts the weekly desk's `monitor_cli.py` + `monitor.py` template and
extends it with a neutrality-residual trip computed from `neutrality.dollar_residual` /
`neutrality.beta_residual` against the `NeutralityConfig` bands.

These tests seed a self-contained live `monitor_book.json` under `state/` (the same cwd-relative
root the other CLIs use) and assert `set_halt` IS called on an imbalanced (out-of-band) book and
is NOT called on a neutral (in-band) book. `set_halt` is monkeypatched at the module boundary so
no real state is mutated.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from futures_fund.config import Settings


def _write_book(state_dir: Path, *, legs: list[dict], balance: float, peak_equity: float) -> None:
    """Seed the live book the monitor reads: account balance/peak + per-leg marks/liq/notional/beta.

    `dollar_residual` is Sum(long$) - Sum(short$) over signed `notional`; `beta_residual` is
    Sum_i weight_i * beta_i, weight_i = notional_i / equity. The monitor recomputes both from this
    artifact and compares against the `NeutralityConfig` bands — it never trusts a stored residual.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "monitor_book.json").write_text(
        json.dumps({"balance": balance, "peak_equity": peak_equity, "legs": legs})
    )


@pytest.fixture
def neutral_book(tmp_path, monkeypatch) -> Path:
    """A live book that is dollar- AND beta-neutral, well inside drawdown + liq-distance: every
    guard in band, so the monitor must be a pure no-op (no `set_halt`)."""
    state = tmp_path / "state"
    # +5000 long beta 1.0 / -5000 short beta 1.0: dollar residual 0, beta residual 0.
    legs = [
        {"symbol": "BTC/USDT:USDT", "mark": 60000.0, "liq_price": 30000.0,
         "notional": 5000.0, "beta_btc": 1.0},
        {"symbol": "ETH/USDT:USDT", "mark": 3000.0, "liq_price": 6000.0,
         "notional": -5000.0, "beta_btc": 1.0},
    ]
    _write_book(state, legs=legs, balance=20000.0, peak_equity=20000.0)
    monkeypatch.chdir(tmp_path)
    return state


@pytest.fixture
def imbalanced_book(tmp_path, monkeypatch) -> Path:
    """A live book whose dollar residual is WAY outside the 3% band: +8000 long vs -2000 short =
    +6000 net long, dollar_residual_frac = 6000 / 10000 = 0.6 >> dollar_band (0.03). Drawdown and
    liq-distance stay in band, so ONLY the neutrality trip fires (exercising the new guard)."""
    state = tmp_path / "state"
    legs = [
        {"symbol": "BTC/USDT:USDT", "mark": 60000.0, "liq_price": 30000.0,
         "notional": 8000.0, "beta_btc": 1.0},
        {"symbol": "ETH/USDT:USDT", "mark": 3000.0, "liq_price": 6000.0,
         "notional": -2000.0, "beta_btc": 1.0},
    ]
    _write_book(state, legs=legs, balance=20000.0, peak_equity=20000.0)
    monkeypatch.chdir(tmp_path)
    return state


def test_monitor_trips_halt_on_neutrality_breach(tmp_path, monkeypatch, imbalanced_book):
    halted = {}
    monkeypatch.setattr(
        "scripts.monitor_cli.load_settings", lambda *_a, **_k: Settings()
    )
    monkeypatch.setattr(
        "scripts.monitor_cli.set_halt", lambda *_a, **_k: halted.setdefault("h", True)
    )
    # imbalanced_book has dollar_residual_frac well above dollar_band
    from scripts.monitor_cli import main

    main([])
    assert halted.get("h") is True


def test_monitor_noop_when_in_band(tmp_path, monkeypatch, neutral_book):
    called = {"halt": False}
    monkeypatch.setattr(
        "scripts.monitor_cli.load_settings", lambda *_a, **_k: Settings()
    )
    monkeypatch.setattr(
        "scripts.monitor_cli.set_halt", lambda *_a, **_k: called.__setitem__("halt", True)
    )
    from scripts.monitor_cli import main

    main([])
    assert called["halt"] is False

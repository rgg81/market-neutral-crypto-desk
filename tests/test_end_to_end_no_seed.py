# tests/test_end_to_end_no_seed.py
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from futures_fund.contracts import Spread, TargetWeights
from futures_fund.cycle_io import load_output, save_output
from futures_fund.market_data import FundingInfo

NOW_ISO = "2026-06-11T00:00:00+00:00"

# A 6-name balanced universe: all beta~1, so a fully-deployed dollar+beta-neutral book respecting
# the per-name cap is feasible from BUILT (not seeded) inputs.
_UNIVERSE = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
             "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT"]
_MARKS = {"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0, "SOL/USDT:USDT": 150.0,
          "XRP/USDT:USDT": 0.6, "ADA/USDT:USDT": 0.5, "DOGE/USDT:USDT": 0.15}
# alternating funding signs so the carry sleeve has a two-sided cross-section
_FUNDING = {"BTC/USDT:USDT": 0.0001, "ETH/USDT:USDT": 0.0006, "SOL/USDT:USDT": -0.0004,
            "XRP/USDT:USDT": 0.0005, "ADA/USDT:USDT": -0.0003, "DOGE/USDT:USDT": 0.0007}


class _FakeCyclePrepExchange:
    """Duck-typed FuturesExchange: deterministic beta~1 OHLCV + funding for the universe."""

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        # Seed the per-symbol idio RNG off the symbol's POSITION (not `hash(symbol)`, which is
        # salted by PYTHONHASHSEED and would make the built betas — and thus optimizer feasibility —
        # flaky across pytest invocations). Deterministic across processes.
        rng = np.random.default_rng(_UNIVERSE.index(symbol) + 1)
        # all names track a common BTC factor (beta~1) + a small idiosyncratic component. The
        # factor vol dominates the idio noise so the OLS beta-to-BTC of every name clusters tightly
        # near 1.0 over the 45-point lookback — the precondition for a feasible dollar+beta-neutral
        # book respecting the per-name cap. (Per the plan's debug note: raise factor / lower idio
        # until betas cluster near 1.0; a test-data concern, the cointegration math is untouched.)
        factor = np.cumsum(np.random.default_rng(0).normal(0, 0.02, 120))
        idio = rng.normal(0, 0.0003, 120)
        closes = _MARKS[symbol] * np.exp(factor + idio)
        ts = pd.date_range("2026-01-01", periods=120, freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": closes, "high": closes,
                             "low": closes, "close": closes, "volume": 1.0})

    def funding(self, symbol):
        return FundingInfo(symbol=symbol, current_rate=_FUNDING[symbol],
                           next_funding_ts=pd.Timestamp(NOW_ISO).to_pydatetime(),
                           interval_hours=8.0, mark_price=_MARKS[symbol],
                           index_price=_MARKS[symbol])

    def mark_price(self, symbol):
        return _MARKS[symbol]


class _FakeScoutClient:
    markets = {s: {"info": {"underlyingType": "COIN"}} for s in _UNIVERSE}

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {s: {"last": _MARKS[s], "quoteVolume": 1e9, "percentage": 0.0} for s in _UNIVERSE}


@pytest.fixture
def no_seed_env(tmp_path, monkeypatch):
    """No `_seed_upstream`: the producers BUILD every upstream artifact from the fake exchange."""
    monkeypatch.setattr("scripts.scout_cli.build_ccxt", lambda settings: _FakeScoutClient())
    # `scripts.cycle_prep_cli.FuturesExchange` and `scripts.gate_execute_cli.FuturesExchange` are
    # the SAME class object (both `from futures_fund.exchange import FuturesExchange`), so patching
    # `.from_settings` on one patches it on the other. We therefore set it ONCE to the fake
    # cycle-prep exchange: cycle-prep reads its OHLCV/funding to BUILD the upstream artifacts, while
    # the paper-only gate+execute boundary records WOULD-fills and never dereferences the exchange
    # (the seeded E2E proves this — it injects a bare `object()` there and still executes).
    monkeypatch.setattr("futures_fund.exchange.FuturesExchange.from_settings",
                        staticmethod(lambda settings: _FakeCyclePrepExchange()))
    # min_adv_usd defaults to 50M; the fake tickers report 1e9 vol so they survive the floor.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_full_run_builds_a_neutral_deployed_book_without_seeding(no_seed_env):
    from scripts.run_paper_cli import main

    main(["--now", NOW_ISO])

    state = no_seed_env / "state"
    # geometries/sleeves/pairs were BUILT (not seeded) under the weekly root
    wk = state / "weekly" / "cycle" / "1"
    assert (wk / "geometries.json").exists()
    assert (wk / "sleeves.json").exists()
    assert (wk / "pairs.json").exists()

    tw = TargetWeights.model_validate(json.loads((wk / "target_weights.json").read_text()))
    assert tw.feasible is True
    assert tw.dollar_residual_frac <= 0.03 + 1e-6
    assert abs(tw.beta_residual) <= 0.05 + 1e-6
    # ~90% deployed each side (deployment floor honored)
    assert tw.deploy_long_frac >= 0.90 - 1e-6
    assert tw.deploy_short_frac >= 0.90 - 1e-6

    # reviewer passed (all 17 checks live, including the now-fed RR + pair-PnL)
    verdict = json.loads((wk / "reviewer.json").read_text())
    assert verdict["passed"] is True
    rr = next(c for c in verdict["checks"] if c["name"] == "rr_after_costs")
    assert "vacuously" not in rr["detail"]  # RR check ran on real reconstructed proposals

    # non-empty execution report + recorded equity
    report = json.loads((wk / "report.json").read_text())
    assert report["live"] is False
    assert report["executed"]
    # equity_log.record_equity writes state/equity-history.jsonl (same path the seeded E2E asserts).
    eq = state / "equity-history.jsonl"
    assert eq.exists() and eq.read_text().strip()

    # the daily cadence also ran weekly-first-then-daily under the same lock
    assert (state / "daily" / "cycle" / "1" / "report.json").exists()
    # lock released
    assert not (state / ".run.lock").exists()


def test_fabricated_pair_pnl_halts_the_wired_loop(no_seed_env):
    # LOOP-LEVEL C2 (not just unit-level): run the producers once to get an HONEST built book, then
    # TAMPER a produced spread's realized_pnl to a large value and re-run the reviewer in the fully
    # wired path. The reviewer must HALT (SystemExit(2)) with pair_pnl_attribution in mismatches.
    from scripts.reviewer_cli import main as reviewer_main
    from scripts.run_paper_cli import main

    main(["--now", NOW_ISO])  # honest end-to-end run builds geometries/pairs/spreads/target_weights

    state = no_seed_env / "state"
    # tamper ONE produced weekly spread's realized_pnl (a lied-about pair PnL)
    spreads_payload = load_output(state, 1, "spreads", cadence="weekly")
    assert spreads_payload["spreads"], "cycle-prep must have produced at least one spread to tamper"
    tampered = list(spreads_payload["spreads"])
    sp = Spread.model_validate(tampered[0])
    tampered[0] = sp.model_copy(update={"realized_pnl": 1_000_000.0}).model_dump(mode="json")
    save_output(state, 1, "spreads", {"spreads": tampered}, cadence="weekly")

    # re-run the SAME reviewer stage the wired loop runs; the fabricated PnL must veto.
    with pytest.raises(SystemExit) as exc:
        reviewer_main(["--cadence", "weekly", "--cycle", "1", "--state-dir", str(state)])
    assert exc.value.code == 2
    verdict = json.loads((state / "weekly" / "cycle" / "1" / "reviewer.json").read_text())
    assert verdict["passed"] is False
    assert "pair_pnl_attribution" in verdict["mismatches"]

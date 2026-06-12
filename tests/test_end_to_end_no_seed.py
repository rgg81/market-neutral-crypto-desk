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

    def depth(self, symbol, limit=20):
        # deep, symmetric book around the mark so every name clears the min_depth_usd floor
        mark = _MARKS[symbol]
        qty = 5_000_000.0 / mark  # ~$5M per level -> full top-of-book notional >> min_depth_usd
        return {"bids": [(mark * 0.999, qty)], "asks": [(mark * 1.001, qty)]}


class _FakeScoutClient:
    # old listing (2019 epoch) so the min_age_days gate keeps every name via onboard_date (NOT the
    # kline fallback) — exercises the Task 2 row -> Task 3 gate path end to end.
    markets = {s: {"info": {"underlyingType": "COIN", "onboardDate": "1567965300000"}}
               for s in _UNIVERSE}

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

    # Phase 9 — REALISTIC P&L: per-cycle pnl.json exists and equity MOVES (not flat 20000)
    import json as _json

    from futures_fund.equity_log import equity_series
    wk_pnl = state / "weekly" / "cycle" / "1" / "pnl.json"
    assert wk_pnl.exists()
    pnl_rec = _json.loads(wk_pnl.read_text())
    assert "closing_equity" in pnl_rec and "fees_paid" in pnl_rec and "funding_net" in pnl_rec
    # the account ledger + cumulative jsonl were persisted at the state root
    assert (state / "account.json").exists()
    assert (state / "ledger.jsonl").exists()
    # equity is no longer the flat constant: at least one recorded point differs from 20000
    equities = [v for _, v in equity_series(state)]
    assert any(abs(e - 20_000.0) > 1e-9 for e in equities), "equity must move off the flat constant"


def test_wired_loop_writes_ledger_account_and_real_equity(no_seed_env):
    # CENTRAL-CHANGE GUARD (run_paper_cli._run_cadence Step 7a): the wired loop must load the
    # account, settle funding, reconcile-apply this cycle's executed book, mark-to-market, and
    # write pnl.json + ledger.jsonl + account.json with the REAL account equity — not the old flat
    # settings.account_size_usdt. The two pre-existing E2E tests stop at "equity-history.jsonl
    # exists", so nothing else in the suite asserts these artifacts' CONTENTS through main()->
    # _run_cadence. This test pins them with INDEPENDENT expecteds derived from the account itself.
    from futures_fund.account import load_account
    from scripts.run_paper_cli import _geometry_cost_maps, main

    main(["--now", NOW_ISO])
    state = no_seed_env / "state"

    # 1) account.json was written and round-trips a POST-FILL book (the reconcile actually ran).
    assert (state / "account.json").exists()
    account = load_account(state, default_cash=-1.0)   # file wins; -1.0 proves it's not the default
    assert account.cash != -1.0
    assert account.positions, "the reconcile must have opened positions from the executed book"
    # frictions were charged on the fills (fees+slippage are never flat/zero here, §11).
    assert account.fees_paid > 0.0
    assert account.slippage_paid > 0.0

    # 2) The recorded equity is the REAL account equity at THIS cycle's marks (account.equity),
    #    NOT the hardcoded 20_000 settings.account_size_usdt recorded before. Fills are at
    #    the mark (0 unrealized at entry) and cost fees+slippage, so the real equity is STRICTLY
    #    below the flat account size; the regression (recording 20_000) would fail this.
    bundle = load_output(state, 1, "geometries", cadence="daily")
    marks, _funding, _intervals, _costs = _geometry_cost_maps(bundle)
    expected_equity = account.equity(marks)
    assert expected_equity < 20_000.0                  # NOT the flat settings.account_size_usdt
    points = [json.loads(line) for line in (state / "equity-history.jsonl").read_text().splitlines()
              if line.strip()]
    assert points
    # the LAST recorded point (daily ran after weekly under the same lock) is the real equity.
    assert abs(points[-1]["equity"] - expected_equity) < 1e-6

    # 3) pnl.json was written per cadence; the ACCOUNT is carried across cadences (weekly
    #    cold-starts at 20_000, daily opens at the weekly close — NOT reset), and the cumulative
    #    cost totals + closing equity are the real account values.
    weekly_pnl = load_output(state, 1, "pnl", cadence="weekly")
    daily_pnl = load_output(state, 1, "pnl", cadence="daily")
    assert weekly_pnl["cadence"] == "weekly" and daily_pnl["cadence"] == "daily"
    assert weekly_pnl["opening_equity"] == 20_000.0    # cold-start cash this run
    assert weekly_pnl["fees_paid"] > 0.0 and weekly_pnl["slippage_paid"] > 0.0
    # carried, not reset: daily opens where weekly closed (account persisted across cadences).
    assert abs(daily_pnl["opening_equity"] - weekly_pnl["closing_equity"]) < 1e-6
    assert daily_pnl["opening_equity"] < 20_000.0
    assert abs(daily_pnl["closing_equity"] - expected_equity) < 1e-6

    # 4) ledger.jsonl gained exactly ONE line per cadence that ran (weekly + daily = 2 — no
    #    double-append, no skipped append).
    ledger_lines = [line for line in (state / "ledger.jsonl").read_text().splitlines()
                    if line.strip()]
    assert len(ledger_lines) == 2
    last_ledger = json.loads(ledger_lines[-1])
    assert last_ledger["cadence"] == "daily"
    assert abs(last_ledger["closing_equity"] - expected_equity) < 1e-6


def test_wired_loop_held_account_book_is_dollar_neutral(no_seed_env):
    """REGRESSION GUARD for the market-neutrality bug: after a full wired run the HELD account book
    (the REAL positions on account.json) must be dollar-neutral within tolerance.

    This is the assertion that would have caught the live bug: the loop used to feed the SPARSE
    daily rebalance DELTAS into the account, so the held positions DRIFTED off the intended neutral
    book (net short, BTC hedge missing). The fix reconciles the account to the FULL intended book
    (reviewed.legs) every cycle. The intended books are dollar+beta-neutral, so the HELD book — the
    consolidated net of those legs — must be dollar-neutral too."""
    from futures_fund.account import load_account
    from scripts.run_paper_cli import _geometry_cost_maps, main

    main(["--now", NOW_ISO])
    state = no_seed_env / "state"

    account = load_account(state, default_cash=-1.0)
    assert account.positions, "the wired loop must have opened a held book"
    # marks for every held symbol (the universe is identical across cadences here).
    marks_w, *_ = _geometry_cost_maps(load_output(state, 1, "geometries", cadence="weekly"))
    marks_d, *_ = _geometry_cost_maps(load_output(state, 1, "geometries", cadence="daily"))
    marks = {**marks_w, **marks_d}
    held_long = sum(p.qty * marks[s] for s, p in account.positions.items() if p.direction == "long")
    held_short = sum(
        p.qty * marks[s] for s, p in account.positions.items() if p.direction == "short")
    gross = held_long + held_short
    assert gross > 0.0
    # |Sum long$ - Sum short$| is a small fraction of gross — the held book is market-neutral.
    # (Pre-fix the held book went net short ~$1.7k on a ~$18k gross; this guard would have failed.)
    assert abs(held_long - held_short) <= 0.03 * gross, (
        f"held book not dollar-neutral: long={held_long:.2f} short={held_short:.2f}")


# A 7th name the week-2 reselection swaps DOGE for — extends the balanced beta~1 cross-section.
_AVAX = "AVAX/USDT:USDT"
_ALL_MARKS = {**_MARKS, _AVAX: 30.0}
_ALL_FUNDING = {**_FUNDING, _AVAX: -0.0002}
_ALL_ORDER = list(_ALL_MARKS)  # stable RNG-seed index across both universes


def _drifting_universe_fakes(universe_cell):
    """Build (scout, cycle_prep) fakes that read the CURRENT universe from `universe_cell['now']`,
    so a run can SWAP the selected universe between cycles (week-2 drops DOGE, adds AVAX)."""

    class _Scout:
        def load_markets(self):
            return {s: {"info": {"underlyingType": "COIN", "onboardDate": "1567965300000"}}
                    for s in universe_cell["now"]}

        def fetch_tickers(self):
            return {s: {"last": _ALL_MARKS[s], "quoteVolume": 1e9, "percentage": 0.0}
                    for s in universe_cell["now"]}

    class _CyclePrep:
        def ohlcv(self, symbol, timeframe="4h", limit=500):
            rng = np.random.default_rng(_ALL_ORDER.index(symbol) + 1)
            factor = np.cumsum(np.random.default_rng(0).normal(0, 0.02, 120))
            idio = rng.normal(0, 0.0003, 120)
            closes = _ALL_MARKS[symbol] * np.exp(factor + idio)
            ts = pd.date_range("2026-01-01", periods=120, freq="4h", tz="UTC")
            return pd.DataFrame({"timestamp": ts, "open": closes, "high": closes,
                                 "low": closes, "close": closes, "volume": 1.0})

        def funding(self, symbol):
            return FundingInfo(symbol=symbol, current_rate=_ALL_FUNDING[symbol],
                               next_funding_ts=pd.Timestamp(NOW_ISO).to_pydatetime(),
                               interval_hours=8.0, mark_price=_ALL_MARKS[symbol],
                               index_price=_ALL_MARKS[symbol])

        def mark_price(self, symbol):
            return _ALL_MARKS[symbol]

        def depth(self, symbol, limit=20):
            mark = _ALL_MARKS[symbol]
            qty = 5_000_000.0 / mark
            return {"bids": [(mark * 0.999, qty)], "asks": [(mark * 1.001, qty)]}

    return _Scout(), _CyclePrep()


def test_two_weekly_runs_drop_a_symbol_and_keep_the_held_book_neutral(tmp_path, monkeypatch):
    """BINDING loop-level regression for the live market-neutrality bug. Two weekly runs a week
    apart: run 2 RESELECTS a different universe (DOGE dropped, AVAX added), so the week-2 intended
    book is dollar+beta-neutral but its symbol set DIFFERS. Through the REAL `_run_cadence` the HELD
    account book must, after run 2: (a) NOT retain the dropped DOGE (flattened — even though its
    mark vanished with its universe slot), and (b) stay dollar-neutral, equal to the new book.

    FAILS pre-fix: the loop fed the SPARSE daily/weekly delta into the account (and never closed a
    dropped symbol whose mark is gone), so the held book drifts net-imbalanced by the dropped leg's
    notional and KEEPS the dropped DOGE — the desk is silently NOT market-neutral. The fix
    reconciles the account to the FULL intended book each cycle and flattens dropped names."""
    universe_cell = {"now": list(_UNIVERSE)}                 # week 1: the 6-name base universe
    scout, cycle_prep = _drifting_universe_fakes(universe_cell)
    monkeypatch.setattr("scripts.scout_cli.build_ccxt", lambda settings: scout)
    monkeypatch.setattr("futures_fund.exchange.FuturesExchange.from_settings",
                        staticmethod(lambda settings: cycle_prep))
    monkeypatch.chdir(tmp_path)

    from datetime import UTC, datetime, timedelta

    from futures_fund.account import load_account
    from scripts.run_paper_cli import _geometry_cost_maps, main

    base = datetime(2026, 6, 11, tzinfo=UTC)
    main(["--now", base.isoformat()])                       # run 1: open the 6-name neutral book
    state = tmp_path / "state"
    acct1 = load_account(state, default_cash=-1.0)
    assert "DOGE/USDT:USDT" in acct1.positions              # DOGE held after run 1

    # run 2 (a week later): RESELECT — drop DOGE, add AVAX. Weekly fires cycle 2 (fresh candle).
    universe_cell["now"] = [s for s in _UNIVERSE if s != "DOGE/USDT:USDT"] + [_AVAX]
    main(["--now", (base + timedelta(days=8)).isoformat()])

    account = load_account(state, default_cash=-1.0)
    assert "DOGE/USDT:USDT" not in account.positions        # DROPPED symbol was flattened
    assert _AVAX in account.positions                       # the new name was opened
    # held book is dollar-neutral at the week-2 marks (DOGE's mark is GONE — it must be closed at
    # entry, not skipped). Use the week-2 weekly marks (the universe actually held now).
    marks, *_ = _geometry_cost_maps(load_output(state, 2, "geometries", cadence="weekly"))
    held_long = sum(p.qty * marks[s] for s, p in account.positions.items() if p.direction == "long")
    held_short = sum(
        p.qty * marks[s] for s, p in account.positions.items() if p.direction == "short")
    gross = held_long + held_short
    assert gross > 0.0
    assert abs(held_long - held_short) <= 0.03 * gross, (
        f"held book not dollar-neutral after a drop: long={held_long:.2f} short={held_short:.2f}")


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


def test_two_runs_one_day_apart_prove_nonzero_funding_in_pnl(no_seed_env):
    """The user's headline requirement: pnl.json carries NON-ZERO funding through the WIRED loop.

    Run main twice on the SAME state-dir, one sim-day apart. Run 1 opens the book (its funding
    clock starts at the first `now`; 0 events settled). Run 2 (+1 sim-day) settles funding over the
    elapsed day on the still-held book -> pnl.json funding fields are non-zero. Funding is NON-ZERO
    at the account level (funding_received + funding_paid > 0), independent of net sign on the book.

    The two `--now` instants are anchored to wall-clock-FUTURE midnights (not the fixed past
    NOW_ISO) so the daily due-gate is deterministic regardless of the real date: the daily
    `cadence_due` floors the report's wall-clock `ran_at` to find the served candle, so run 2's
    daily boundary must sit on a strictly later calendar day than run 1's wall-clock `ran_at`.
    Anchoring both runs to future days guarantees run 2's DAILY cadence fires a FRESH cycle-2 that
    settles the elapsed-day funding (weekly stays in the same week and SKIPs run 2, as expected)."""
    import json as _json
    from datetime import UTC, datetime, timedelta

    from scripts.run_paper_cli import main

    base = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)
    run1_now = base.isoformat()
    run2_now = (base + timedelta(days=1)).isoformat()

    main(["--now", run1_now])                         # run 1: open the book, clock starts here
    state = no_seed_env / "state"
    acct1 = _json.loads((state / "account.json").read_text())
    assert acct1["last_funding_ts"] is not None      # clock advanced on run 1

    main(["--now", run2_now])                         # run 2: a sim-day later -> funding settles

    # across ALL cycle-2 pnl.json files written on run 2, at least one must carry non-zero funding
    # over the elapsed day (robust to which cadence cadence_due fires one day later).
    pnls = list(state.glob("*/cycle/2/pnl.json"))
    activity = [
        _json.loads(p.read_text())["funding_received"] + _json.loads(p.read_text())["funding_paid"]
        for p in pnls
    ]
    assert any(a > 0.0 for a in activity), "wired loop must settle non-zero funding over a sim-day"
    # and the account clock advanced to the second run instant
    acct2 = _json.loads((state / "account.json").read_text())
    assert acct2["last_funding_ts"] is not None
    assert acct2["last_funding_ts"] != acct1["last_funding_ts"]

# Phase 8 — Integration Glue & Runnable Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the market-neutral desk runnable end-to-end from a fake (no-network) exchange WITHOUT hand-seeding — scout → cycle-prep (geometries + sleeves + spreads + pairs) → control loop → reviewer (proposals + pairs fed so all 17 checks are LIVE) → gate-execute → reflect → record-lessons → dashboard — and reconcile SKILL.md so the runbook is executable.

**Architecture:** "LLM proposes, code disposes." The deterministic Python spine (`futures_fund/`) already owns all math (beta, cointegration, sleeves, neutrality optimizer, reviewer). Phase 8 adds the missing *producer* glue that turns live (or faked) exchange reads into the EXACT artifact JSON shapes the control loop and reviewer already consume (`GeometryBundle`, `SleeveSignal[]`, `Spread[]`, `Pair[]`), fixes the reviewer CLI to feed `proposals=`/`pairs=` so `check_rr_after_costs`/`check_pair_pnl` stop passing vacuously, adds the thin orchestration CLIs the SKILL.md ladders name (`runlock_cli.py`/`due_check.py`/`scout_cli.py`/`preflight.py`/`record_lessons_cli.py`), and wires all producers into `run_paper_cli.py` so a full weekly+daily cycle runs without `_seed_upstream`.

**Tech Stack:** Python 3.11 · `uv` · `pydantic` v2 · `pandas`/`numpy` · `statsmodels` (cointegration/OU) · `scikit-learn` (Ledoit-Wolf) · `cvxpy` (optimizer) · `pytest` · `ruff` (selects E,F,I,UP,B, line-length 100).

---

## File Structure

All paths are absolute under project root `/home/roberto/crypto-trade-claude-code-market-neutral` (branch `phase1-neutrality`).

**New modules (deterministic spine):**
- `futures_fund/cycle_prep.py` — the single cycle-prep producer. Pure functions that turn exchange reads (or faked reads) into the four upstream artifacts the loop/reviewer consume: `build_geometries` (via `beta` + `funding_intervals` + `market_data`), `build_sleeves` (via the four sleeve builders + `risk_parity_budgets`), and `build_pairs_and_spreads` (via `cointegration.build_pair`/`build_spread`/`fdr_adjust` + `sleeves.pairs.select_pairs`). Returns validated contract objects; never persists (the CLI persists).
- `futures_fund/trader_io.py` — `proposals_from_book` reconstructs RR-capable `TradeProposal` objects from a `TargetWeights` book + geometries, so the reviewer's `check_rr_after_costs` re-derives a real RR (not the vacuous empty-list pass).

**New CLIs (`scripts/`):**
- `scripts/scout_cli.py` — `market_data.scan_universe` + `liquidity_floor` → `universe.json`.
- `scripts/cycle_prep_cli.py` — drives `cycle_prep` against a (faked) `FuturesExchange` → persists `geometries.json` + `sleeves.json` + `spreads.json` + `pairs.json` under the cadence cycle root.
- `scripts/runlock_cli.py` — `acquire`/`release`/`status --owner` over `futures_fund.runlock`.
- `scripts/due_check.py` — `state --loop weekly|daily` → emits `DUE FRESH/RETRY <N>` / `SKIP: <reason>` via `control_loop.cadence_due`.
- `scripts/preflight.py` — folds held symbols into the scout universe, builds per-symbol briefs → `context.json`.
- `scripts/record_lessons_cli.py` — appends `reflect`'s `lessons.json` into the corpus via `lessons.append_lesson`.

**Modified:**
- `scripts/reviewer_cli.py` — load `pairs.json` (`Pair[]`) and reconstruct RR-capable `TradeProposal[]` from the audited book + geometries via `trader_io.proposals_from_book` (the persisted `proposals.json` is `target_notional`-only and is intentionally NOT consumed for RR — it carries no entry/stop/TP geometry), pass `proposals=`/`pairs=` to `review_cycle`.
- `scripts/run_paper_cli.py` — insert a scout + cycle-prep producer step BEFORE the control-loop step; cycle-prep writes `pairs.json`/`spreads.json` the reviewer consumes.
- `tests/test_skill_md.py` — assert each named CLI file EXISTS on disk (reconciled together with SKILL.md in ONE task).
- `SKILL.md` — name the real producers (`cycle_prep_cli.py`) + driver (`run_paper_cli.py`) + dashboard (`dashboard_cli.py`).
- `README.md` (new) + `CLAUDE.md` (new) — operating rules.
- `futures_fund/monitor_book.py` (new, small) — `write_monitor_book` so `monitor_cli.py` has a writer. (minor task)
- `scripts/repair_cli.py` (new, small) — thin CLI over `repair.apply_repair`. (minor task)

**Test files (new):**
- `tests/test_cycle_prep.py`, `tests/test_trader_io.py`, `tests/test_scout_cli.py`, `tests/test_cycle_prep_cli.py`, `tests/test_runlock_cli.py`, `tests/test_due_check_cli.py`, `tests/test_preflight_cli.py`, `tests/test_record_lessons_cli.py`, `tests/test_reviewer_cli_live_checks.py`, `tests/test_end_to_end_no_seed.py`, `tests/test_docs_exist.py`, `tests/test_monitor_book.py`, `tests/test_repair_cli.py`.

**Sequencing rationale (dependency order):** producers (cycle_prep, trader_io) → CLIs that wrap them (scout, cycle_prep_cli, the thin orchestration CLIs) → reviewer-wiring (reviewer_cli loads pairs + reconstructs proposals) → run_paper_cli integration + the no-seed E2E test (ONE atomic commit) → SKILL.md reconcile + test_skill_md tightening (ONE atomic commit) → docs → minors.

**CLI flag convention (cross-CLI consistency, issue #9):** every Phase 8 CLI uses `--state-dir` for the state root (matching the pre-existing `scout`/`cycle_prep`/`control_loop`/`reviewer`/`gate_execute`/`record_lessons` family). `runlock_cli.py` and `due_check.py` therefore also take `--state-dir` (NOT `--state`). The SKILL.md ladder invocations (W1/W12/D1/D8) call `runlock_cli.py acquire --owner weekly` with no state flag (defaulting to `state`), so the runbook is unaffected by this standardization.

---

### Task 1: `trader_io.proposals_from_book` — reconstruct RR-capable `TradeProposal`s from a book

The reviewer's `check_rr_after_costs` (`futures_fund/reviewer.py:424`) requires `list[TradeProposal]`, where each `TradeProposal` (`futures_fund/models.py:47`) needs `entry/stop/take_profits/atr/confidence/horizon_hours/funding_rate`. `run_paper_cli._proposals_from_book` currently writes a `target_notional`-only dict that cannot compute RR. This producer builds geometric stop/TP from each book leg's mark + a fixed RR so the floor is actually enforced.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/trader_io.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_trader_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trader_io.py
from __future__ import annotations

from futures_fund.contracts import CoinGeometry, TargetWeights, WeightLeg
from futures_fund.models import TradeProposal
from futures_fund.risk_gate import _reward_risk
from futures_fund.trader_io import proposals_from_book

NOW = "2026-06-11T00:00:00+00:00"


def _book() -> TargetWeights:
    return TargetWeights(
        legs=[
            WeightLeg(symbol="BTC/USDT:USDT", direction="long", weight=0.45,
                      target_notional=9000.0, beta_btc=1.0, sleeve="factor"),
            WeightLeg(symbol="ETH/USDT:USDT", direction="short", weight=-0.45,
                      target_notional=9000.0, beta_btc=1.0, sleeve="factor"),
            # a flat (zero-notional) carry-over leg: must NOT become a proposal
            WeightLeg(symbol="SOL/USDT:USDT", direction="long", weight=0.0,
                      target_notional=0.0, beta_btc=1.0, sleeve="factor"),
        ],
        dollar_residual=0.0, dollar_residual_frac=0.0, beta_residual=0.0,
        gross_long=9000.0, gross_short=9000.0,
        deploy_long_frac=0.9, deploy_short_frac=0.9, gross_notional=18000.0, as_of_ts=NOW,
    )


def _geos() -> list[CoinGeometry]:
    return [
        CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0, funding_rate=0.0001,
                     funding_interval_hours=8.0),
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, funding_rate=0.0001,
                     funding_interval_hours=8.0),
    ]


def test_proposals_skip_flat_legs_and_validate_as_tradeproposal():
    props = proposals_from_book(_book(), _geos(), rr=2.0, stop_frac=0.02)
    assert [p.symbol for p in props] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert all(isinstance(p, TradeProposal) for p in props)
    # entry == mark, funding wired from geometry
    btc = next(p for p in props if p.symbol == "BTC/USDT:USDT")
    assert btc.entry == 60000.0
    assert btc.funding_rate == 0.0001
    assert btc.funding_interval_hours == 8.0


def test_proposals_clear_the_min_rr_floor():
    # stop 2% away, TP at rr*stop -> RR == rr exactly, so the gate's MIN_RR (2.0) is met.
    props = proposals_from_book(_book(), _geos(), rr=2.0, stop_frac=0.02)
    for p in props:
        assert _reward_risk(p) >= 2.0 - 1e-9


def test_short_leg_has_stop_above_and_tp_below_entry():
    props = proposals_from_book(_book(), _geos(), rr=2.0, stop_frac=0.02)
    eth = next(p for p in props if p.symbol == "ETH/USDT:USDT")
    assert eth.direction == "short"
    assert eth.stop > eth.entry           # short stop above entry
    assert eth.take_profits[0] < eth.entry  # short TP below entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trader_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'futures_fund.trader_io'`

- [ ] **Step 3: Write minimal implementation**

```python
# futures_fund/trader_io.py
"""Reconstruct gate-ready `TradeProposal`s from an optimizer `TargetWeights` book.

The Trader does NO sizing — notional comes from the optimizer (`WeightLeg.target_notional`).
But the every-cycle reviewer's `check_rr_after_costs` (and the live risk gate) require a full
`TradeProposal` with entry/stop/take_profit geometry to re-derive reward:risk. This module
maps each non-flat book leg + its `CoinGeometry` mark into a `TradeProposal` whose stop sits
`stop_frac` away on the loss side and whose nearest take-profit sits `rr*stop_frac` away on the
gain side — so the re-derived RR equals `rr` and the MIN_RR>=2 floor is actually enforced on the
real book (the prior `target_notional`-only proposal shape made the RR check pass vacuously).
"""
from __future__ import annotations

from futures_fund.contracts import CoinGeometry, TargetWeights
from futures_fund.models import TradeProposal


def proposals_from_book(
    book: TargetWeights,
    geometries: list[CoinGeometry],
    *,
    rr: float = 2.0,
    stop_frac: float = 0.02,
    horizon_hours: float = 168.0,
) -> list[TradeProposal]:
    """One `TradeProposal` per non-flat alpha/hedge leg, sized off the leg's mark.

    Zero-notional legs (carry-over unwinds/flattens) are excluded — there is nothing to OPEN.
    A leg with no matching geometry mark is skipped (cannot place geometric stops without a mark).
    stop = entry*(1 -/+ stop_frac); nearest take_profit = entry*(1 +/- rr*stop_frac), so
    `risk_gate._reward_risk` re-derives exactly `rr`.
    """
    geo = {g.symbol: g for g in geometries}
    out: list[TradeProposal] = []
    for leg in book.legs:
        if abs(leg.target_notional) <= 0.0:
            continue
        g = geo.get(leg.symbol)
        if g is None or g.mark <= 0.0:
            continue
        entry = g.mark
        if leg.direction == "long":
            stop = entry * (1.0 - stop_frac)
            tp = entry * (1.0 + rr * stop_frac)
        else:
            stop = entry * (1.0 + stop_frac)
            tp = entry * (1.0 - rr * stop_frac)
        out.append(TradeProposal(
            symbol=leg.symbol,
            direction=leg.direction,
            entry=entry,
            stop=stop,
            take_profits=[tp],
            atr=entry * stop_frac,
            confidence=0.6,
            horizon_hours=horizon_hours,
            funding_rate=g.funding_rate,
            funding_interval_hours=g.funding_interval_hours,
        ))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trader_io.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/trader_io.py tests/test_trader_io.py
git commit -m "feat(trader_io): reconstruct RR-capable TradeProposals from the optimizer book

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `cycle_prep.build_geometries` — geometries.json from beta + funding + market_data

`control_loop_cli._load_geometries` loads a `GeometryBundle` (`futures_fund/contracts.py:128`). No producer builds one from exchange reads. This step builds `CoinGeometry` rows (`contracts.py:100`) per symbol using `beta.beta_for_symbols`, `funding_intervals.funding_apr`/`clamp_funding_rate`/`funding_interval_hours`, and `market_data` parsers — against a minimal duck-typed exchange (the e2e fakes it).

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/cycle_prep.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_cycle_prep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cycle_prep.py
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from futures_fund.contracts import GeometryBundle
from futures_fund.cycle_prep import build_geometries

NOW = datetime(2026, 6, 11, tzinfo=UTC)


class _FakeExchange:
    """Duck-typed FuturesExchange: returns deterministic OHLCV + funding per symbol."""

    def __init__(self, marks: dict[str, float], funding: dict[str, float]):
        self._marks = marks
        self._funding = funding

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        # 120 candles; each symbol a random walk anchored at its mark.
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        n = 120
        base = self._marks[symbol]
        rets = rng.normal(0, 0.01, n)
        closes = base * np.exp(np.cumsum(rets))
        ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        return pd.DataFrame({
            "timestamp": ts, "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": 1000.0,
        })

    def funding(self, symbol):
        from futures_fund.market_data import FundingInfo
        return FundingInfo(
            symbol=symbol, current_rate=self._funding[symbol],
            next_funding_ts=NOW, interval_hours=8.0,
            mark_price=self._marks[symbol], index_price=self._marks[symbol],
        )

    def mark_price(self, symbol):
        return self._marks[symbol]


def _ex():
    return _FakeExchange(
        marks={"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0, "SOL/USDT:USDT": 150.0},
        funding={"BTC/USDT:USDT": 0.0001, "ETH/USDT:USDT": 0.0005, "SOL/USDT:USDT": -0.0003},
    )


def test_build_geometries_returns_one_geometry_per_symbol():
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    bundle = build_geometries(_ex(), symbols, now=NOW, btc_symbol="BTC/USDT:USDT",
                              beta_lookback=45)
    assert isinstance(bundle, GeometryBundle)
    assert {g.symbol for g in bundle.geometries} == set(symbols)


def test_btc_beta_is_one_and_funding_apr_is_signed():
    bundle = build_geometries(_ex(), ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
                              now=NOW, btc_symbol="BTC/USDT:USDT", beta_lookback=45)
    by = {g.symbol: g for g in bundle.geometries}
    assert by["BTC/USDT:USDT"].beta_btc == 1.0
    # SOL funding is negative -> funding_apr signed negative (carry credit visible)
    assert by["SOL/USDT:USDT"].funding_apr < 0.0
    assert by["ETH/USDT:USDT"].funding_apr > 0.0
    # mark carried through
    assert by["ETH/USDT:USDT"].mark == 3000.0


def test_funding_rate_is_clamped_sign_preserving():
    ex = _FakeExchange(
        marks={"BTC/USDT:USDT": 60000.0, "DOGE/USDT:USDT": 0.15},
        funding={"BTC/USDT:USDT": 0.0001, "DOGE/USDT:USDT": 0.5},  # 50% -> over alt cap 2%
    )
    bundle = build_geometries(ex, ["BTC/USDT:USDT", "DOGE/USDT:USDT"], now=NOW,
                              btc_symbol="BTC/USDT:USDT", beta_lookback=45)
    doge = next(g for g in bundle.geometries if g.symbol == "DOGE/USDT:USDT")
    assert doge.funding_rate == 0.02  # clamped to alt cap (PER_SYMBOL_CAP_DEFAULT), sign preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cycle_prep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'futures_fund.cycle_prep'`

- [ ] **Step 3: Write minimal implementation**

```python
# futures_fund/cycle_prep.py
"""Cycle-prep producer (Phase 8): turn exchange reads into the EXACT upstream artifacts the
control loop and reviewer consume — `GeometryBundle`, `SleeveSignal[]`, `Pair[]`, `Spread[]`.

Closes the C1 gap (alpha engine not wired to the loop's input artifacts): before Phase 8 only the
e2e test's `_seed_upstream` fixture produced these, so the desk could not build a book from market
data without hand-seeded inputs. Pure functions over a duck-typed `FuturesExchange` (the e2e fakes
it); they NEVER persist — `cycle_prep_cli.py` owns persistence.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from futures_fund.beta import beta_for_symbols
from futures_fund.contracts import CoinGeometry, GeometryBundle
from futures_fund.funding_intervals import (
    clamp_funding_rate,
    funding_apr,
    funding_interval_hours,
)


def _marks_frame(exchange, symbols: list[str]) -> dict[str, pd.Series]:
    """Per-symbol close-price series from `exchange.ohlcv` (for beta + realized vol)."""
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = exchange.ohlcv(sym)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        out[sym] = df["close"].astype(float).reset_index(drop=True)
    return out


def _realized_vol(series: pd.Series) -> float:
    """Annualized realized vol from log returns (×sqrt(365*6) for 4h candles); 0.0 if too short."""
    rets = (series / series.shift(1)).dropna()
    if len(rets) < 2:
        return 0.0
    log_r = np.log(rets.to_numpy())
    return float(np.std(log_r, ddof=1) * (365.0 * 6.0) ** 0.5)


def _momentum_20(series: pd.Series) -> float:
    """20-period close-to-close momentum; 0.0 if too short."""
    if len(series) <= 20:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-21] - 1.0)


def build_geometries(
    exchange,
    symbols: list[str],
    *,
    now: datetime,
    btc_symbol: str = "BTC/USDT:USDT",
    beta_lookback: int = 45,
) -> GeometryBundle:
    """One `CoinGeometry` per symbol from live (or faked) exchange reads.

    beta_btc <- beta.beta_for_symbols (BTC self-beta 1.0); funding_rate <- clamp_funding_rate of
    the per-symbol signed rate; funding_interval_hours <- funding_intervals.funding_interval_hours;
    funding_apr <- the SIGNED annualized carry (carry credit stays visible, §6.1). Fail-soft: a
    symbol whose reads error is skipped, never crashes the bundle.
    """
    marks_by_symbol = _marks_frame(exchange, symbols)
    betas = beta_for_symbols(marks_by_symbol, btc_symbol=btc_symbol, lookback=beta_lookback)
    geometries: list[CoinGeometry] = []
    for sym in symbols:
        series = marks_by_symbol.get(sym)
        try:
            fi = exchange.funding(sym)
            raw_rate = float(fi.current_rate)
            mark = float(fi.mark_price)
        except Exception:
            continue
        interval = funding_interval_hours(sym, exchange)
        rate = clamp_funding_rate(sym, raw_rate)
        geometries.append(CoinGeometry(
            symbol=sym,
            mark=mark,
            momentum_20=_momentum_20(series) if series is not None else 0.0,
            realized_vol=_realized_vol(series) if series is not None else 0.0,
            beta_btc=betas.get(sym, 1.0),
            beta_lookback_days=beta_lookback,
            funding_rate=rate,
            funding_interval_hours=interval,
            funding_apr=funding_apr(rate, interval),
            adv_usd=0.0,
        ))
    return GeometryBundle(geometries=geometries, as_of_ts=now)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cycle_prep.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cycle_prep.py tests/test_cycle_prep.py
git commit -m "feat(cycle_prep): build_geometries from beta+funding+market_data reads

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `cycle_prep.build_sleeves` — sleeves.json from the four sleeve builders

`control_loop_cli._load_sleeves` loads `[SleeveSignal]` from `{"sleeves": [...]}`. The four builders (`carry_signal`/`pairs_signal`/`factor_signal`/`sentiment_factor_signal`) are only called by tests. This step calls all four over geometries (+ pairs/spreads) and assigns risk budgets via `neutrality.risk_parity_budgets`.

**Files:**
- Modify: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/cycle_prep.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_cycle_prep.py`

- [ ] **Step 1: Write the failing test (append to tests/test_cycle_prep.py)**

```python
# tests/test_cycle_prep.py  (append)
from futures_fund.contracts import CoinGeometry as _CG
from futures_fund.cycle_prep import build_sleeves


def _six_geos() -> list[_CG]:
    syms = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
            "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT"]
    out = []
    for i, s in enumerate(syms):
        out.append(_CG(symbol=s, mark=100.0 + i, beta_btc=1.0,
                       funding_rate=0.0001 * (i - 2), funding_interval_hours=8.0,
                       funding_apr=0.001 * (i - 2), momentum_20=0.1 * (i - 2),
                       realized_vol=0.5, sentiment_score=0.2 * (i - 2),
                       sentiment_conf=0.8))
    return out


def test_build_sleeves_emits_the_four_named_sleeves():
    sleeves = build_sleeves(_six_geos(), pairs=[], spreads=[], now=NOW)
    names = {s.sleeve for s in sleeves}
    assert names == {"carry", "pairs", "factor", "sentiment"}


def test_risk_budgets_assigned_and_sum_to_one():
    sleeves = build_sleeves(_six_geos(), pairs=[], spreads=[], now=NOW)
    total = sum(s.risk_budget_frac for s in sleeves)
    assert abs(total - 1.0) < 1e-9


def test_sleeves_round_trip_through_the_control_loop_cli_shape():
    # control_loop_cli loads {"sleeves": [SleeveSignal-dict, ...]}; assert that shape validates.
    from futures_fund.contracts import SleeveSignal
    sleeves = build_sleeves(_six_geos(), pairs=[], spreads=[], now=NOW)
    payload = {"sleeves": [s.model_dump(mode="json") for s in sleeves]}
    reloaded = [SleeveSignal.model_validate(s) for s in payload["sleeves"]]
    assert len(reloaded) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cycle_prep.py -k build_sleeves -v`
Expected: FAIL — `ImportError: cannot import name 'build_sleeves'`

- [ ] **Step 3: Write minimal implementation (append to futures_fund/cycle_prep.py)**

```python
# futures_fund/cycle_prep.py  (append)
from futures_fund.contracts import Pair, SleeveSignal, Spread
from futures_fund.neutrality import risk_parity_budgets
from futures_fund.sleeves import (
    carry_signal,
    factor_signal,
    pairs_signal,
    sentiment_factor_signal,
)


def build_sleeves(
    geometries: list[CoinGeometry],
    pairs: list[Pair],
    spreads: list[Spread],
    *,
    now: datetime,
) -> list[SleeveSignal]:
    """Run all four alpha sleeves over the geometries (+ pairs/spreads), then assign risk-parity
    budgets across them via `neutrality.risk_parity_budgets` (the contract's single home for the
    budget split). `risk_budget_frac` starts at 0.0 on each sleeve and is filled in place by
    `risk_parity_budgets`, which sums to 1.0 across the four. Closes the C1 gap: this is the only
    producer that invokes the sleeve builders outside tests."""
    sleeves = [
        carry_signal(geometries, risk_budget_frac=0.0, now=now),
        pairs_signal(pairs, spreads, risk_budget_frac=0.0, now=now),
        factor_signal(geometries, risk_budget_frac=0.0, now=now),
        sentiment_factor_signal(geometries, risk_budget_frac=0.0, now=now),
    ]
    budgets = risk_parity_budgets(sleeves)
    return [s.model_copy(update={"risk_budget_frac": budgets[s.sleeve]}) for s in sleeves]
```

> Note: `risk_parity_budgets` mutates each `SleeveSignal.risk_budget_frac` in place AND returns the `{sleeve: frac}` map; the `model_copy(update=...)` re-stamps the returned fraction so the returned list is unambiguously budgeted (and the equal-split fallback sums to exactly 1.0 across the four).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cycle_prep.py -k build_sleeves -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cycle_prep.py tests/test_cycle_prep.py
git commit -m "feat(cycle_prep): build_sleeves runs the four sleeve builders + risk-parity budgets

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `cycle_prep.build_pairs_and_spreads` — pairs.json + spreads.json via cointegration

`reviewer.check_pair_pnl` skips every spread when `pairs=[]`. No producer builds `Pair`/`Spread`. This step enumerates candidate pairs, runs `cointegration.build_pair` (Engle-Granger + OU), FDR-adjusts the candidate ADF p-values via `fdr_adjust`, keeps survivors via `select_pairs`, and marks live spreads via `build_spread`.

**Files:**
- Modify: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/cycle_prep.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_cycle_prep.py`

- [ ] **Step 1: Write the failing test (append to tests/test_cycle_prep.py)**

```python
# tests/test_cycle_prep.py  (append)
from futures_fund.contracts import Pair, Spread
from futures_fund.cycle_prep import build_pairs_and_spreads


class _CointExchange:
    """Two cointegrated legs (y = 2x + noise) + an independent leg."""

    def __init__(self):
        rng = np.random.default_rng(7)
        n = 200
        x = np.cumsum(rng.normal(0, 1, n)) + 100.0
        noise = rng.normal(0, 0.5, n)
        self._series = {
            "AAA/USDT:USDT": pd.Series(x),
            "BBB/USDT:USDT": pd.Series(2.0 * x + noise),       # cointegrated with AAA
            "CCC/USDT:USDT": pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50.0),  # independent
        }
        self._marks = {"AAA/USDT:USDT": float(x[-1]),
                       "BBB/USDT:USDT": float(2.0 * x[-1] + noise[-1]),
                       "CCC/USDT:USDT": 50.0}

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        s = self._series[symbol]
        ts = pd.date_range("2026-01-01", periods=len(s), freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": s, "high": s, "low": s,
                             "close": s, "volume": 1.0})

    def mark_price(self, symbol):
        return self._marks[symbol]


def test_build_pairs_finds_the_cointegrated_pair():
    ex = _CointExchange()
    syms = ["AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT"]
    pairs, spreads = build_pairs_and_spreads(ex, syms, cycle=1, now=NOW,
                                             adf_pvalue_max=0.05, fdr_method="bh")
    assert all(isinstance(p, Pair) for p in pairs)
    assert all(isinstance(s, Spread) for s in spreads)
    # the AAA/BBB pair (cointegrated) survives FDR + select_pairs; one spread per kept pair
    kept_ids = {p.pair_id for p in pairs}
    assert any("AAA" in pid and "BBB" in pid for pid in kept_ids)
    assert {s.pair_id for s in spreads} == kept_ids


def test_build_pairs_round_trips_through_artifact_shape():
    ex = _CointExchange()
    pairs, spreads = build_pairs_and_spreads(
        ex, ["AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT"],
        cycle=1, now=NOW)
    pairs_payload = {"pairs": [p.model_dump(mode="json") for p in pairs]}
    spreads_payload = {"spreads": [s.model_dump(mode="json") for s in spreads]}
    assert [Pair.model_validate(p) for p in pairs_payload["pairs"]] == pairs
    assert [Spread.model_validate(s) for s in spreads_payload["spreads"]] == spreads
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cycle_prep.py -k build_pairs -v`
Expected: FAIL — `ImportError: cannot import name 'build_pairs_and_spreads'`

- [ ] **Step 3: Write minimal implementation (append to futures_fund/cycle_prep.py)**

> Import discipline: write the cointegration imports as a SINGLE statement (no duplicate `fdr_adjust` line). This block has no overlap with the Task 2 imports (Task 2 imports `beta`/`contracts`/`funding_intervals` only; cointegration/itertools/select_pairs are introduced here for the first time), so ruff F811/redefinition cannot fire.

```python
# futures_fund/cycle_prep.py  (append)
from itertools import combinations

from futures_fund.cointegration import build_pair, build_spread, fdr_adjust
from futures_fund.sleeves.pairs import select_pairs


def build_pairs_and_spreads(
    exchange,
    symbols: list[str],
    *,
    cycle: int,
    now: datetime,
    adf_pvalue_max: float = 0.05,
    fdr_method: str = "bh",
    max_candidates: int = 30,
) -> tuple[list[Pair], list[Spread]]:
    """Enumerate candidate pairs over `symbols`, Engle-Granger + OU-fit each (`build_pair`),
    FDR-correct the candidate ADF p-values (`fdr_adjust`), keep survivors (`select_pairs`), and
    mark each survivor's live spread (`build_spread`). Returns (kept_pairs, spreads). Closes the
    C1 + C2 gaps: produces the `pairs.json`/`spreads.json` the loop never had and the reviewer's
    `check_pair_pnl` needs to stop skipping every spread.

    A pair is dropped if either leg's OHLCV is unreadable (cannot test cointegration). The
    candidate set is capped at `max_candidates` (cheapest by symbol order) to bound the O(n^2)
    Engle-Granger sweep on a large universe. `now` is accepted for caller symmetry with the other
    producers; the Pair/Spread contracts carry no timestamp field."""
    series: dict[str, pd.Series] = {}
    marks: dict[str, float] = {}
    for sym in symbols:
        try:
            series[sym] = exchange.ohlcv(sym)["close"].astype(float).reset_index(drop=True)
            marks[sym] = float(exchange.mark_price(sym))
        except Exception:
            continue
    usable = [s for s in symbols if s in series and s in marks]
    candidates: list[Pair] = []
    for y_sym, x_sym in list(combinations(usable, 2))[:max_candidates]:
        try:
            candidates.append(build_pair(
                series[y_sym], series[x_sym], y_sym, x_sym, cycle=cycle,
                method="engle_granger"))
        except Exception:
            continue
    if not candidates:
        return [], []
    adj = fdr_adjust([p.adf_pvalue for p in candidates], alpha=adf_pvalue_max, method=fdr_method)
    candidates = [p.model_copy(update={"adf_pvalue_adj": a}) for p, a in zip(candidates, adj)]
    kept = select_pairs(candidates, adf_pvalue_max=adf_pvalue_max)
    spreads = [build_spread(p, marks[p.symbol_y], marks[p.symbol_x]) for p in kept]
    return kept, spreads
```

- [ ] **Step 4: Run test to verify it passes + lint**

Run: `uv run pytest tests/test_cycle_prep.py -k build_pairs -v && uv run ruff check futures_fund/cycle_prep.py`
Expected: PASS (2 passed) and `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cycle_prep.py tests/test_cycle_prep.py
git commit -m "feat(cycle_prep): build_pairs_and_spreads via build_pair/fdr_adjust/select_pairs/build_spread

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `scout_cli.py` → universe.json

SKILL.md W3 names `scout_cli.py`; it does not exist here. Adapt the reference repo's `scout_cli.py` to this repo's cadence-segmented `save_output` + the `liquidity_floor` trim.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/scout_cli.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_scout_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scout_cli.py
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
```

> Note: `UniverseSettings.min_adv_usd` defaults to 50M; the fake BTC/ETH tickers report 2e9/1e9 quote volume so both clear the liquidity floor.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scout_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.scout_cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/scout_cli.py
"""Universe Scout CLI (SKILL.md W3): scan the LIVE USD-M perp universe (top by 24h quote volume,
crypto-only) and trim to the liquidity floor -> `universe.json`. Public/keyless.

    uv run python scripts/scout_cli.py --cycle N --cadence weekly --top 30

Closes the I1 gap (scout_cli.py named in SKILL.md but absent). Writes under the cadence-segmented
cycle root (`state/<cadence>/cycle/<N>/`, CADENCE-ROOT INVARIANT) the rest of the ladder reads.
"""
from __future__ import annotations

import argparse
import json
import sys

from futures_fund.config import load_settings
from futures_fund.cycle_io import save_output
from futures_fund.exchange import build_ccxt
from futures_fund.market_data import liquidity_floor, scan_universe
from futures_fund.models import Cadence


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scan + trim the crypto-only perp universe (W3).")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    settings = load_settings()
    client = build_ccxt(settings)
    client.load_markets()
    rows = scan_universe(client, top_n=max(args.top, settings.universe.symbol_count))
    universe = liquidity_floor(
        rows, min_adv_usd=settings.universe.min_adv_usd,
        symbol_count=settings.universe.symbol_count,
    )
    save_output(args.state_dir, args.cycle, "universe", {"universe": universe}, cadence=cadence)
    print(json.dumps({"universe": universe}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scout_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/scout_cli.py tests/test_scout_cli.py
git commit -m "feat(scout_cli): scan crypto-only universe -> universe.json (SKILL W3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `cycle_prep_cli.py` — drive cycle_prep against a (faked) exchange → 4 artifacts

The driver step that calls `cycle_prep` and persists `geometries.json` + `sleeves.json` + `pairs.json` + `spreads.json` under the cadence cycle root, so `control_loop_cli` and `reviewer_cli` consume them without `_seed_upstream`.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/cycle_prep_cli.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_cycle_prep_cli.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cycle_prep_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.cycle_prep_cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/cycle_prep_cli.py
"""Cycle-prep producer CLI (Phase 8): build + persist the four upstream artifacts the control loop
and reviewer consume — geometries / sleeves / pairs / spreads — from (faked or live) exchange reads.

    uv run python scripts/cycle_prep_cli.py --cycle N --cadence weekly
    uv run python scripts/cycle_prep_cli.py --cycle N --cadence weekly --now 2026-06-11T00:00:00+00:00

Closes C1: before this, only the e2e `_seed_upstream` fixture produced these, so the desk could not
build a book from market data. Reads its symbol set from this cycle's `universe.json` (scout output)
and persists every artifact under the SAME cadence cycle root the loop/reviewer scan (CADENCE-ROOT
INVARIANT). PAPER-ONLY: the exchange is built via `FuturesExchange.from_settings` (faked in tests).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund.config import load_settings
from futures_fund.cycle_io import load_output, save_output
from futures_fund.cycle_prep import (
    build_geometries,
    build_pairs_and_spreads,
    build_sleeves,
)
from futures_fund.exchange import FuturesExchange
from futures_fund.models import Cadence

_STATE_DIR = "state"


def _parse_now(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _symbols(state_dir, cycle: int, cadence: Cadence, settings) -> list[str]:
    """Symbol set from this cycle's universe.json (scout output); fall back to settings.symbols."""
    try:
        rows = load_output(state_dir, cycle, "universe", cadence=cadence)["universe"]
        syms = [r["symbol"] for r in rows if r.get("symbol")]
        if syms:
            return syms
    except FileNotFoundError:
        pass
    return list(settings.symbols)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build + persist geometries/sleeves/pairs/spreads.")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default=_STATE_DIR)
    ap.add_argument("--now", default=None)
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence
    now = _parse_now(args.now)

    settings = load_settings()
    ex = FuturesExchange.from_settings(settings)
    symbols = _symbols(args.state_dir, args.cycle, cadence, settings)

    bundle = build_geometries(
        ex, symbols, now=now, btc_symbol=settings.beta.btc_symbol,
        beta_lookback=settings.beta.lookback_days,
    )
    pairs, spreads = build_pairs_and_spreads(ex, symbols, cycle=args.cycle, now=now)
    sleeves = build_sleeves(bundle.geometries, pairs=pairs, spreads=spreads, now=now)

    save_output(args.state_dir, args.cycle, "geometries", bundle, cadence=cadence)
    save_output(args.state_dir, args.cycle, "sleeves",
                {"sleeves": [s.model_dump(mode="json") for s in sleeves]}, cadence=cadence)
    save_output(args.state_dir, args.cycle, "pairs",
                {"pairs": [p.model_dump(mode="json") for p in pairs]}, cadence=cadence)
    save_output(args.state_dir, args.cycle, "spreads",
                {"spreads": [s.model_dump(mode="json") for s in spreads]}, cadence=cadence)
    print(json.dumps({"cycle": args.cycle, "cadence": cadence, "symbols": len(symbols),
                      "pairs": len(pairs), "spreads": len(spreads)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cycle_prep_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/cycle_prep_cli.py tests/test_cycle_prep_cli.py
git commit -m "feat(cycle_prep_cli): persist geometries/sleeves/pairs/spreads from exchange reads

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `runlock_cli.py` — acquire/release/status --owner

SKILL.md W1/W12/D1/D8 name `runlock_cli.py acquire/release --owner weekly|daily`. `futures_fund.runlock` has `try_acquire`/`release`/`single_flight` but no CLI. Adapt the reference `runlock_cli.py`. Per the CLI flag convention above, the state flag is `--state-dir` (consistent with the rest of the family).

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/runlock_cli.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_runlock_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runlock_cli.py
from __future__ import annotations

from scripts.runlock_cli import main


def test_acquire_then_release_round_trip(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert main(["acquire", "--owner", "weekly", "--state-dir", state]) == 0
    assert "ACQUIRED" in capsys.readouterr().out

    # a second acquire while held prints LOCKED (still exit 0 — caller stands down, not an error)
    assert main(["acquire", "--owner", "daily", "--state-dir", state]) == 0
    assert "LOCKED:" in capsys.readouterr().out

    assert main(["release", "--owner", "weekly", "--state-dir", state]) == 0
    assert "RELEASED" in capsys.readouterr().out

    # after release, a fresh acquire succeeds again
    assert main(["acquire", "--owner", "weekly", "--state-dir", state]) == 0
    assert "ACQUIRED" in capsys.readouterr().out


def test_status_reports_free_and_held(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert main(["status", "--state-dir", state]) == 0
    assert "FREE" in capsys.readouterr().out
    main(["acquire", "--owner", "weekly", "--state-dir", state])
    capsys.readouterr()
    assert main(["status", "--state-dir", state]) == 0
    assert "HELD:" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runlock_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.runlock_cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/runlock_cli.py
"""Single-flight run-lock CLI (SKILL.md W1/W12/D1/D8): the dual-cadence desk is orchestrated by
Claude across MANY separate CLI processes, so each cadence acquires the lock at the START of its
meeting and releases it at the END — exactly one writer at a time over the shared book.

    uv run python scripts/runlock_cli.py acquire --owner weekly   # ACQUIRED | LOCKED: <holder>
    uv run python scripts/runlock_cli.py release --owner weekly    # RELEASED
    uv run python scripts/runlock_cli.py status                    # FREE | HELD: <holder>

Closes I1 (runlock_cli.py named in SKILL.md but absent; runlock.py had no CLI). `acquire` exits 0 on
ACQUIRED, 0 on LOCKED (the caller stands down — not an error), 2 on internal error. A crashed meeting
that never releases is auto-reclaimed after `runlock.DEFAULT_STALE_AFTER_S`. The state root flag is
`--state-dir` (consistent with the rest of the Phase 8 CLI family)."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("acquire", "release", "status"):
        print("usage: runlock_cli.py acquire|release|status [--owner NAME] [--state-dir DIR]")
        return 2
    action = argv[0]
    owner = "runner"
    state_dir = "state"
    i = 1
    while i < len(argv):
        if argv[i] == "--owner" and i + 1 < len(argv):
            owner = argv[i + 1]
            i += 2
        elif argv[i] == "--state-dir" and i + 1 < len(argv):
            state_dir = argv[i + 1]
            i += 2
        else:
            i += 1
    try:
        from futures_fund import runlock
        now = datetime.now(UTC)
        if action == "acquire":
            ok, holder = runlock.try_acquire(state_dir, now, owner=owner)
            print("ACQUIRED" if ok else f"LOCKED: {json.dumps(holder)}")
            return 0
        if action == "release":
            runlock.release(state_dir)
            print("RELEASED")
            return 0
        p = Path(state_dir) / runlock.LOCK_NAME
        holder = runlock._read(p) if p.exists() else None
        print(f"HELD: {json.dumps(holder)}" if holder else "FREE")
        return 0
    except Exception as e:  # noqa: BLE001 — surface, never crash the orchestrator silently
        print(f"ERROR: runlock {action} failed: {e!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runlock_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/runlock_cli.py tests/test_runlock_cli.py
git commit -m "feat(runlock_cli): acquire/release/status --owner over futures_fund.runlock

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `due_check.py` — state --loop weekly|daily → DUE FRESH/RETRY/SKIP tokens

SKILL.md W2/D2 parse `due_check.py state --loop weekly|daily` output. Adapt the reference `due_check.py` but route through `control_loop.cadence_due` (the cadence-correct candle width per the CADENCE-ROOT INVARIANT) instead of the legacy 4h `cycle_due`. Per the CLI flag convention, the optional state root is taken as `--state-dir` (the positional `state` argument the SKILL.md ladder passes still works — it lands as the first positional).

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/due_check.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_due_check_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_due_check_cli.py
from __future__ import annotations

from scripts.due_check import main


def test_cold_start_is_due_fresh_1(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert main([state, "--loop", "weekly"]) == 0
    assert "DUE FRESH 1" in capsys.readouterr().out


def test_daily_loop_uses_the_daily_root(tmp_path, capsys, write_served_report):
    # served the daily candle containing `now`; the weekly root is untouched, so weekly is still DUE.
    from datetime import UTC, datetime
    state = tmp_path / "state"
    now = datetime(2026, 6, 11, tzinfo=UTC)
    write_served_report(state / "daily" / "cycle" / "1", served=now, tf_minutes=1440)
    # daily SKIPs (its candle is served)
    assert main([str(state), "--loop", "daily", "--now", "2026-06-11T00:00:00+00:00"]) == 0
    assert "SKIP:" in capsys.readouterr().out
    # weekly is still DUE (different root, no served report)
    assert main([str(state), "--loop", "weekly", "--now", "2026-06-11T00:00:00+00:00"]) == 0
    assert "DUE" in capsys.readouterr().out


def test_unknown_loop_errors(tmp_path, capsys):
    assert main([str(tmp_path / "state"), "--loop", "hourly"]) == 2
    assert "ERROR:" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_due_check_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.due_check'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/due_check.py
"""Multi-cadence due-gate CLI (SKILL.md W2/D2). Run as the FIRST action each poll fire; prints ONE:

    DUE FRESH <N>   -> run a brand-new cycle end-to-end; create state/<cadence>/cycle/<N>/
    DUE RETRY <N>   -> a prior dir crashed before the gate; re-run/OVERWRITE that dir
    SKIP: <reason>  -> this candle is already served; stand down (liveness ping)
    ERROR: <reason> -> internal failure (exit 2); do NOT trade

    uv run python scripts/due_check.py state --loop weekly
    uv run python scripts/due_check.py state --loop daily

The first positional argument is the state root (SKILL.md passes the literal `state`); `--state-dir`
is also accepted for consistency with the rest of the CLI family. Routes through
`control_loop.cadence_due` so the candle width is cadence-correct (weekly=10080, daily=1440) and the
root scanned is `state/<cadence>/cycle/*` (CADENCE-ROOT INVARIANT). Exit 0 for DUE*/SKIP, 2 for
ERROR. Makes ZERO exchange/network calls and ZERO writes. Closes I1."""
from __future__ import annotations

import sys
from datetime import UTC, datetime


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    loop = None
    now_raw = None
    state_dir = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--loop" and i + 1 < len(argv):
            loop = argv[i + 1]
            i += 2
            continue
        if a == "--now" and i + 1 < len(argv):
            now_raw = argv[i + 1]
            i += 2
            continue
        if a == "--state-dir" and i + 1 < len(argv):
            state_dir = argv[i + 1]
            i += 2
            continue
        rest.append(a)
        i += 1
    if state_dir is None:
        state_dir = rest[0] if rest else "state"

    try:
        from futures_fund.control_loop import cadence_due
        if loop not in ("weekly", "daily"):
            print(f"ERROR: unknown loop {loop!r}; expected weekly|daily")
            return 2
        now = (datetime.fromisoformat(now_raw.replace("Z", "+00:00"))
               if now_raw else datetime.now(UTC))
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        mode, n, reason = cadence_due(state_dir, now, loop)
    except Exception as e:  # noqa: BLE001 — fail SAFE but visible
        print(f"ERROR: due_check failed before decision: {e!r}")
        return 2

    if mode in ("FRESH", "RETRY"):
        print(f"DUE {mode} {n}")
        print(reason)
        return 0
    print(f"SKIP: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_due_check_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/due_check.py tests/test_due_check_cli.py
git commit -m "feat(due_check): cadence-aware DUE FRESH/RETRY/SKIP gate (SKILL W2/D2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `preflight.py` — fold held symbols + per-symbol briefs → context.json

SKILL.md W3 names `preflight.py` (folds held symbols, builds per-symbol briefs). This repo has no `orchestration.preflight_step`, so build a self-contained brief builder: union the scout universe with any held symbols read from the prior cycle's `report.json`, emit a minimal per-symbol brief.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/preflight.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_preflight_cli.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preflight_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.preflight'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/preflight.py
"""Preflight CLI (SKILL.md W3): fold every HELD symbol into the scout universe and emit per-symbol
briefs as the analysts' context -> `context.json`.

    uv run python scripts/preflight.py --cycle N --cadence weekly

Closes I1 (preflight.py named in SKILL.md but absent). Held symbols are resolved from the most
recent cadence cycle's executed `report.json` legs (a held leg must stay in the universe so the desk
can audit/close it even if it dropped out of the top-by-volume scan). Pure read + light assembly;
the analysts (LLM) reason over the briefs downstream.
"""
from __future__ import annotations

import argparse
import json
import sys

from futures_fund.control_loop import latest_cadence_cycle
from futures_fund.cycle_io import cycle_dir, load_output, save_output
from futures_fund.models import Cadence


def _held_symbols(state_dir, cadence: Cadence) -> list[str]:
    """Symbols from the most recent executed report's legs (the book currently held)."""
    n = latest_cadence_cycle(state_dir, cadence, "report")
    if n is None:
        return []
    path = cycle_dir(state_dir, n, cadence=cadence) / "report.json"
    try:
        executed = json.loads(path.read_text()).get("executed", [])
    except (OSError, json.JSONDecodeError):
        return []
    return [e["symbol"] for e in executed if isinstance(e, dict) and e.get("symbol")]


def build_briefs(universe: list[dict], held: list[str]) -> list[dict]:
    """One brief per symbol in (universe ∪ held). `held=True` flags positions the book carries so
    the analysts always audit them (even if they fell out of the volume-ranked scan)."""
    held_set = set(held)
    syms: list[str] = []
    for row in universe:
        s = row.get("symbol")
        if s and s not in syms:
            syms.append(s)
    for s in held:
        if s not in syms:
            syms.append(s)
    return [{"symbol": s, "held": s in held_set} for s in syms]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fold held symbols + build per-symbol briefs (W3).")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default="state")
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    try:
        universe = load_output(args.state_dir, args.cycle, "universe", cadence=cadence)["universe"]
    except FileNotFoundError:
        universe = []
    held = _held_symbols(args.state_dir, cadence)
    ctx = {"briefs": build_briefs(universe, held), "held": held}
    save_output(args.state_dir, args.cycle, "context", ctx, cadence=cadence)
    print(json.dumps(ctx, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preflight_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/preflight.py tests/test_preflight_cli.py
git commit -m "feat(preflight): fold held symbols + per-symbol briefs -> context.json (SKILL W3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: `record_lessons_cli.py` — append reflect's lessons.json into the corpus

SKILL.md W11 names `record_lessons_cli.py`. The reference uses `reflect.record_lessons`, which does NOT exist here; this repo has `lessons.append_lesson(memory_dir, fields, ts)` (`futures_fund/lessons.py:25`). Build a CLI that reads `lessons.json` and appends each via `append_lesson`.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/record_lessons_cli.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_record_lessons_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_record_lessons_cli.py
from __future__ import annotations

import json

from futures_fund.cycle_io import save_output
from futures_fund.lessons import read_lessons
from scripts.record_lessons_cli import main


def test_records_lessons_from_reflector_output(tmp_path, capsys):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    save_output(state, 1, "lessons", {"lessons": [
        {"text": "pairs stayed cointegrated; size up next time", "polarity": "enabling",
         "dimension": "cointegration_break", "importance": 6},
        {"text": "funding flipped under stress; cut crowded carry", "polarity": "restrictive",
         "dimension": "carry_thesis_miss", "importance": 7},
    ]}, cadence="weekly")

    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state),
          "--memory-dir", str(memory)])

    lessons = read_lessons(memory)
    assert len(lessons) == 2
    assert {lz.dimension for lz in lessons} == {"cointegration_break", "carry_thesis_miss"}
    out = json.loads(capsys.readouterr().out)
    assert out["appended"] == 2


def test_missing_lessons_artifact_appends_nothing(tmp_path, capsys):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state),
          "--memory-dir", str(memory)])
    assert json.loads(capsys.readouterr().out)["appended"] == 0
    assert read_lessons(memory) == []
```

> Note: the lesson field set (`text`/`polarity`/`dimension`/`importance`) and the `Lesson.dimension`/`read_lessons` round-trip must match the real `Lesson` contract; confirm against `futures_fund/lessons.py` before writing the test (the `dimension` values are illustrative — substitute valid ones if the contract enumerates them).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_record_lessons_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.record_lessons_cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/record_lessons_cli.py
"""Deterministically persist the Reflector's lessons to the corpus (SKILL.md W11). The reflect
phase must ALWAYS append — never rely on the LLM Reflector to remember.

    uv run python scripts/record_lessons_cli.py --cycle N --cadence weekly

Reads this cycle's `lessons.json` (the Reflector agent's output) and appends each lesson via
`lessons.append_lesson` (validated against the `Lesson` contract). Closes I1 (record_lessons_cli.py
named in SKILL.md but absent). A missing artifact appends nothing (a cycle that minted no lessons is
fine). Each lesson dict supplies at least `text`; `ts` is stamped here.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund.cycle_io import load_output
from futures_fund.lessons import append_lesson
from futures_fund.models import Cadence


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Append the Reflector's lessons to the corpus (W11).")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    try:
        payload = load_output(args.state_dir, args.cycle, "lessons", cadence=cadence)
    except FileNotFoundError:
        payload = {}
    raw = payload.get("lessons", []) if isinstance(payload, dict) else (payload or [])

    now = datetime.now(UTC)
    ids: list[str] = []
    for fields in raw:
        if isinstance(fields, dict) and fields.get("text"):
            ids.append(append_lesson(args.memory_dir, fields, ts=now))
    print(json.dumps({"cycle": args.cycle, "cadence": cadence, "appended": len(ids),
                      "lesson_ids": ids}, default=str))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_record_lessons_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/record_lessons_cli.py tests/test_record_lessons_cli.py
git commit -m "feat(record_lessons_cli): append Reflector lessons via lessons.append_lesson (SKILL W11)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Fix `reviewer_cli.py` — load pairs.json + reconstruct RR-capable proposals, feed `proposals=`/`pairs=`

C2: `review_cycle` already accepts `proposals=`/`pairs=` (`reviewer.py:791-792`), but `reviewer_cli.py` never loads them, so `check_rr_after_costs` returns vacuously OK and `check_pair_pnl` skips every spread. This task loads `pairs.json` (Task 6) and reconstructs RR-capable `TradeProposal[]` from the audited book + geometries via `trader_io.proposals_from_book`. **The persisted `proposals.json` is the Trader's `target_notional`-only hand-off and is intentionally NOT consumed for RR** — it carries no entry/stop/TP geometry, so reconstruction (not file-load) is the correct source. With both fed: a fabricated `Spread.realized_pnl` now FAILS `pair_pnl_attribution`, and the live RR check runs on real geometry.

**Files:**
- Modify: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/reviewer_cli.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_reviewer_cli_live_checks.py`

> **Binding-test sizing (issue #1):** the reviewer's `check_caps` re-derives per-name weight as `|target_notional| / cfg.capital_usdt` with `capital_usdt = 20000` and `per_name_cap = 0.25`, so a leg ≥ 5000 notional FAILS `per_name_cap` and vetoes the verdict. `check_deployment_floor` re-derives each side's deployment as `gross_side$ / side_budget_usdt` with `side_budget_usdt = 10000` and `deployment_floor = 0.90`, so each side needs ≥ 9000 gross. The honest book therefore uses **4 legs of 4500** (2 long / 2 short): per-name `= 4500/20000 = 0.225 ≤ 0.25` (passes) and per side `= 9000/10000 = 0.90 ≥ 0.90` (passes). The pair sleeve carries AAA(long)/BBB(short); two extra factor legs CCC(long)/DDD(short) make each side ≥ 9000 while staying under the per-name cap. All four symbols get a matching `CoinGeometry` (mark, `market_info` crypto) so `check_funding`/`check_exchange_filters`/`check_crypto_only` pass.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reviewer_cli_live_checks.py
from __future__ import annotations

import json

import pytest

from futures_fund.contracts import (
    CoinGeometry,
    GeometryBundle,
    Pair,
    Spread,
    TargetWeights,
    WeightLeg,
)
from futures_fund.cycle_io import cycle_dir, save_output

NOW = "2026-06-11T00:00:00+00:00"


def _neutral_book() -> TargetWeights:
    # 4 legs @ 4500: per-name |w| = 4500/20000 = 0.225 <= per_name_cap 0.25, and each side gross
    # 9000/side_budget 10000 = 0.90 >= deployment_floor 0.90 -> caps + deployment both PASS.
    return TargetWeights(
        legs=[
            WeightLeg(symbol="AAA/USDT:USDT", direction="long", weight=0.225,
                      target_notional=4500.0, beta_btc=1.0, sleeve="pairs",
                      pair_id="AAAUSDT__BBBUSDT"),
            WeightLeg(symbol="BBB/USDT:USDT", direction="short", weight=-0.225,
                      target_notional=4500.0, beta_btc=1.0, sleeve="pairs",
                      pair_id="AAAUSDT__BBBUSDT"),
            WeightLeg(symbol="CCC/USDT:USDT", direction="long", weight=0.225,
                      target_notional=4500.0, beta_btc=1.0, sleeve="factor"),
            WeightLeg(symbol="DDD/USDT:USDT", direction="short", weight=-0.225,
                      target_notional=4500.0, beta_btc=1.0, sleeve="factor"),
        ],
        dollar_residual=0.0, dollar_residual_frac=0.0, beta_residual=0.0,
        gross_long=9000.0, gross_short=9000.0,
        deploy_long_frac=0.9, deploy_short_frac=0.9, gross_notional=18000.0, as_of_ts=NOW,
    )


def _geos() -> GeometryBundle:
    return GeometryBundle(geometries=[
        CoinGeometry(symbol="AAA/USDT:USDT", mark=100.0, beta_btc=1.0, funding_rate=0.0,
                     market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="BBB/USDT:USDT", mark=200.0, beta_btc=1.0, funding_rate=0.0,
                     market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="CCC/USDT:USDT", mark=50.0, beta_btc=1.0, funding_rate=0.0,
                     market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="DDD/USDT:USDT", mark=25.0, beta_btc=1.0, funding_rate=0.0,
                     market_info={"underlyingType": "COIN"}),
    ], as_of_ts=NOW)


def _pair() -> Pair:
    return Pair(pair_id="AAAUSDT__BBBUSDT", symbol_y="AAA/USDT:USDT", symbol_x="BBB/USDT:USDT",
                hedge_ratio=0.5, method="engle_granger", adf_pvalue=0.01, adf_pvalue_adj=0.02,
                half_life=10.0, theta=0.07, mu=0.0, sigma_eq=1.0, formed_cycle=1)


def _seed(state, *, spread_pnl: float, qty_y: float, qty_x: float):
    save_output(state, 1, "target_weights", _neutral_book(), cadence="weekly")
    save_output(state, 1, "geometries", _geos(), cadence="weekly")
    save_output(state, 1, "pairs", {"pairs": [_pair().model_dump(mode="json")]}, cadence="weekly")
    # a live spread whose realized_pnl/leg sizing the reviewer will re-derive
    sp = Spread(pair_id="AAAUSDT__BBBUSDT", spread_value=2.0, zscore=2.0, state="short_spread",
                qty_y=qty_y, qty_x=qty_x, realized_pnl=spread_pnl)
    save_output(state, 1, "spreads", {"spreads": [sp.model_dump(mode="json")]}, cadence="weekly")
    # NOTE: proposals.json is intentionally NOT seeded — reviewer_cli reconstructs RR-capable
    # TradeProposals from the audited book + geometries (the persisted target_notional-only
    # proposals carry no entry/stop/TP geometry and are NOT consumed for RR).


def test_fabricated_spread_pnl_now_fails_the_gate(tmp_path, monkeypatch):
    # correct leg sizing (qty_x == hedge_ratio*qty_y) but a LIED-ABOUT realized_pnl: with pairs.json
    # now loaded, check_pair_pnl re-derives PnL-since-entry and the fabricated value FAILS.
    state = tmp_path / "state"
    _seed(state, spread_pnl=999999.0, qty_y=10.0, qty_x=5.0)
    monkeypatch.chdir(tmp_path)
    from scripts.reviewer_cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--cadence", "weekly", "--cycle", "1", "--state-dir", str(state)])
    assert exc.value.code == 2
    verdict = json.loads((cycle_dir(state, 1, cadence="weekly") / "reviewer.json").read_text())
    assert verdict["passed"] is False
    assert "pair_pnl_attribution" in verdict["mismatches"]


def test_honest_book_passes_with_live_pair_and_rr_checks(tmp_path, monkeypatch):
    # honest spread PnL (re-derive expected and store it) + correct hedge-ratio sizing; the RR check
    # is LIVE (proposals reconstructed) and clears MIN_RR -> verdict passes. With the 4-leg @4500
    # book the per-name cap (0.225 <= 0.25) and the deployment floor (0.90 >= 0.90) both hold, so
    # the ONLY checks gating `passed` are the now-live pair-PnL + RR ones.
    from futures_fund.reviewer import check_pair_pnl
    state = tmp_path / "state"
    # first seed with a placeholder to compute the honest pnl, then re-seed the spread
    _seed(state, spread_pnl=0.0, qty_y=10.0, qty_x=5.0)
    sp = Spread(pair_id="AAAUSDT__BBBUSDT", spread_value=2.0, zscore=2.0, state="short_spread",
                qty_y=10.0, qty_x=5.0, realized_pnl=0.0)
    expected = check_pair_pnl([sp], [_pair()])[0].expected  # re-derived honest PnL
    sp_honest = sp.model_copy(update={"realized_pnl": expected})
    save_output(state, 1, "spreads", {"spreads": [sp_honest.model_dump(mode="json")]},
                cadence="weekly")
    monkeypatch.chdir(tmp_path)
    from scripts.reviewer_cli import main

    main(["--cadence", "weekly", "--cycle", "1", "--state-dir", str(state)])  # no HALT
    verdict = json.loads((cycle_dir(state, 1, cadence="weekly") / "reviewer.json").read_text())
    assert verdict["passed"] is True
    # rr_after_costs ran on reconstructed proposals (NOT the vacuous empty-list pass)
    rr = next(c for c in verdict["checks"] if c["name"] == "rr_after_costs")
    assert rr["ok"] is True
    assert "vacuously" not in rr["detail"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reviewer_cli_live_checks.py -v`
Expected: FAIL — current `reviewer_cli` does not load pairs or reconstruct proposals, so `pair_pnl_attribution` is not in mismatches (the spread is skipped, passing vacuously) and the fabricated-PnL test does not HALT; the honest test's `rr_after_costs.detail` still says "vacuously".

- [ ] **Step 3: Write the fix**

Edit `scripts/reviewer_cli.py`. Add imports for `Pair` and the reconstruction helper, load `pairs.json`, reconstruct proposals, and pass them to `review_cycle`.

Change the existing `from futures_fund.contracts import (...)` block to add `Pair`:

```python
from futures_fund.contracts import (
    GeometryBundle,
    Pair,
    SentimentBatch,
    Spread,
    TargetWeights,
)
```

Add the reconstruction-helper import after the `review_cycle` import:

```python
from futures_fund.reviewer import review_cycle
from futures_fund.trader_io import proposals_from_book
```

After the optional-artifacts block (the `sentiment = []` branch) and BEFORE the `verdict = review_cycle(...)` call, insert:

```python
    # Pairs (C2): without pairs.json loaded, check_pair_pnl skips every spread and a fabricated
    # Spread.realized_pnl passes. Load this cycle's pairs so the reviewer re-derives spread-level
    # PnL + hedge-ratio sizing against ground truth.
    try:
        pairs = [
            Pair.model_validate(p)
            for p in load_output(args.state_dir, args.cycle, "pairs", cadence=cadence)["pairs"]
        ]
    except FileNotFoundError:
        pairs = []

    # Proposals (C2): without proposals fed, check_rr_after_costs returns vacuously OK (the RR>=2
    # floor is never enforced on the real book). The persisted proposals.json is the Trader's
    # target_notional-only hand-off and carries NO entry/stop/TP geometry, so it is intentionally
    # NOT loaded for RR; instead reconstruct RR-capable TradeProposals from the audited book's legs
    # + the geometries' marks via trader_io.proposals_from_book.
    proposals = proposals_from_book(target, geometries)
```

Then change the `review_cycle(...)` call to pass them:

```python
    verdict = review_cycle(
        args.state_dir,
        args.memory_dir,
        cycle=args.cycle,
        cadence=cadence,
        target=target,
        geometries=geometries,
        spreads=spreads,
        sentiment=sentiment,
        cfg=cfg,
        returns=None,
        pairs=pairs,
        proposals=proposals,
    )
```

- [ ] **Step 4: Run test to verify it passes (and the existing reviewer-CLI suite stays green)**

Run: `uv run pytest tests/test_reviewer_cli_live_checks.py -v && uv run pytest tests/test_reviewer.py -q`
Expected: PASS (2 passed in the new file; existing reviewer suite unchanged-green)

- [ ] **Step 5: Commit**

```bash
git add scripts/reviewer_cli.py tests/test_reviewer_cli_live_checks.py
git commit -m "fix(reviewer_cli): load pairs.json + reconstruct proposals so RR + pair-PnL checks are LIVE

Closes C2: feed pairs= and reconstructed proposals= to review_cycle. A fabricated Spread.realized_pnl
now fails pair_pnl_attribution; the RR floor is enforced on real geometry (no vacuous empty-list pass).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Tasks 12+13 (ONE atomic unit — committed together): wire producers into `run_paper_cli.py` + the no-seed E2E

> **Atomicity note (issues #3 + #7):** Task 12 adds an integration seam to `run_paper_cli.py` (production code) whose behavior is asserted only by Task 13's no-seed E2E. To honor the plan's TDD discipline (no code lands without a failing test in the same committable unit), **Tasks 12 and 13 are ONE atomic unit with ONE commit.** Task 12 has no standalone commit step; the seam + its end-to-end assertions land together. The single legitimate edit Task 12 makes to the EXISTING seeded `test_end_to_end.py` (stubbing `_run_producers` to a no-op) is also part of this unit.

---

### Task 12: Wire scout + cycle-prep into `run_paper_cli.py` (producers before the loop)

`run_paper_cli` currently requires `geometries.json`/`sleeves.json` to pre-exist (only `_seed_upstream` writes them). Insert a producer step (scout → cycle-prep) BEFORE the control-loop step, and persist `pairs.json`/`spreads.json` so the reviewer (Task 11) consumes them.

**Files:**
- Modify: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/run_paper_cli.py`
- Modify: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_end_to_end.py` (the existing seeded E2E stubs the new producer seam to a no-op so it keeps asserting on its hand-seeded artifacts).

- [ ] **Step 1: Add the producer seam (production code — its end-to-end behavior is asserted in Task 13; committed together)**

Edit `scripts/run_paper_cli.py`. Add imports near the existing imports:

```python
from scripts.cycle_prep_cli import main as cycle_prep_main
from scripts.scout_cli import main as scout_main
```

Add a producer-step function above `_run_cadence`:

```python
def _run_producers(state_dir, cadence: Cadence, cycle: int, now: datetime) -> None:
    """Step 3b — scout the universe then build the cycle's upstream artifacts (geometries / sleeves
    / pairs / spreads) BEFORE the control-loop step consumes them. Closes C1: the loop no longer
    depends on a hand-seeded `_seed_upstream`. Both CLIs are seams (monkeypatched in tests) so the
    driver's ladder runs offline against a faked exchange. Idempotent on RETRY (overwrites the
    cycle's artifacts in place)."""
    scout_main(["--cycle", str(cycle), "--cadence", cadence, "--state-dir", str(state_dir)])
    cycle_prep_main([
        "--cycle", str(cycle), "--cadence", cadence, "--state-dir", str(state_dir),
        "--now", now.isoformat(),
    ])
```

In `_run_cadence`, insert the producer call between the SKIP guard and the control-loop step:

```python
    mode, cycle, _reason = cadence_due(state_dir, now, cadence)
    if mode == "SKIP":
        return False  # candle already served -> stand down (no re-run)

    # Step 3b — producers: scout + cycle-prep write geometries/sleeves/pairs/spreads.
    _run_producers(state_dir, cadence, cycle, now)
    # Step 4a — cadence step: persist target_weights.json under state/<cadence>/cycle/<cycle>/.
    _run_control_loop_step(state_dir, cadence, cycle)
```

No change to `_write_proposals` / `_resolve_held_book` is required — the persisted `proposals.json` stays the Trader's `target_notional`-only hand-off, and the reviewer reconstructs RR-capable proposals itself (Task 11). cycle-prep writes `pairs.json`/`spreads.json` under BOTH cadence roots (the driver invokes it per cadence), so the reviewer reads them from whichever cadence root it runs in — no extra wiring needed.

- [ ] **Step 2a: Stub the producer seam to a no-op in the EXISTING seeded `paper_env` fixture (the single required edit)**

The four `tests/test_end_to_end.py` tests assert loop behavior on HAND-SEEDED `geometries.json`/`sleeves.json` (`_seed_upstream`). If the real producers run, they OVERWRITE those seeded artifacts with freshly-built ones, changing the asserted book — and cycle-prep would try to build a real ccxt client. So the seeded E2E must stub `run_paper_cli._run_producers` to a no-op, keeping its hand-seeded artifacts intact. (Task 13 introduces the dedicated no-seed E2E that exercises the REAL producers.)

In `tests/test_end_to_end.py`, extend the existing `paper_env` fixture (after the existing `gate_execute_cli.FuturesExchange.from_settings` patch, before `monkeypatch.chdir(...)`) with exactly this one line:

```python
    # Phase 8: the producers (scout + cycle-prep) are a NO-OP for the seeded E2E — these tests
    # assert behavior on _seed_upstream's hand-seeded artifacts, which the real producers would
    # overwrite. The dedicated no-seed E2E (test_end_to_end_no_seed.py) exercises the real producers.
    monkeypatch.setattr("scripts.run_paper_cli._run_producers",
                        lambda state_dir, cadence, cycle, now: None)
```

- [ ] **Step 2b: Run the seeded E2E to confirm it stays green with the no-op stub**

Run: `uv run pytest tests/test_end_to_end.py -q`
Expected: PASS (all 4 existing E2E tests green — the no-op `_run_producers` leaves the seeded artifacts untouched, so the produced book/verdict/report are exactly as before Phase 8).

- [ ] **Step 3: (No commit here — Task 12 commits together with Task 13.)**

Proceed directly to Task 13. The seam + its end-to-end assertions are committed as one unit at Task 13 Step 5.

---

### Task 13: No-seed end-to-end test — full weekly+daily run on a FAKE exchange WITHOUT `_seed_upstream`

The acceptance test for the whole phase: run `run_paper_cli.main` against a faked exchange with NO hand-seeded artifacts, and assert a feasible dollar+beta-neutral ~90%-deployed book, a passing reviewer, a non-empty report, and a recorded equity point. A second sub-test closes the LOOP-LEVEL C2 binding (issue #3): in the fully-wired run, tampering a produced spread's `realized_pnl` must HALT at the reviewer with `pair_pnl_attribution`.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_end_to_end_no_seed.py`

- [ ] **Step 1: Write the failing test**

```python
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
    """Duck-typed FuturesExchange producing deterministic beta~1 OHLCV + funding for the universe."""

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        # all names track a common BTC factor (beta~1) + idiosyncratic noise
        factor = np.cumsum(np.random.default_rng(0).normal(0, 0.01, 120))
        idio = rng.normal(0, 0.003, 120)
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
    monkeypatch.setattr("scripts.cycle_prep_cli.FuturesExchange.from_settings",
                        lambda settings: _FakeCyclePrepExchange())
    monkeypatch.setattr("scripts.gate_execute_cli.FuturesExchange.from_settings",
                        lambda settings: object())
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
    from scripts.run_paper_cli import main
    from scripts.reviewer_cli import main as reviewer_main

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
```

> **Tamper-test robustness (the no-seed build may yield zero `flat` spreads with non-zero qty):** `check_pair_pnl` re-derives `expected_pnl = side * qty_y * (spread_value - entry_spread)` where `side = 0` for a `flat` spread. If the produced spreads are all `flat` (state="flat" → side 0), the re-derived expected PnL is 0 regardless of `spread_value`, so a fabricated 1e6 realized_pnl STILL fails the attribution check (|0 − 1e6| > tol). The assertion therefore holds for any produced spread state. If `build_pairs_and_spreads` produced NO spreads (no pair survived FDR on the fake series), the `assert ... at least one spread` line fails loudly — in that case strengthen the fake `_FakeCyclePrepExchange` series so at least one cointegrated pair survives (a test-data concern; never touch the cointegration math).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_end_to_end_no_seed.py -v`
Expected: FAIL initially. Most likely failure modes to debug systematically (use superpowers:systematic-debugging if it does not pass first try):
- If the optimizer returns `feasible=False`: the built universe's betas may not be close enough to 1.0 for a neutral book under the per-name cap. `_FakeCyclePrepExchange` deliberately drives all names off a common BTC factor so betas cluster near 1.0; if still infeasible, raise the factor weight / lower the idio-noise.
- If the reviewer HALTs in the FIRST sub-test: inspect `wk/reviewer.json` `mismatches` to see which of the 17 checks failed and fix the producer that feeds it.
- If the SECOND sub-test does not HALT: confirm cycle-prep actually produced ≥1 spread (the `assert` guards this) and that `reviewer_cli` loads `pairs.json` (Task 11).

- [ ] **Step 3: Make it pass**

No new production code should be needed if Tasks 1-12 are correct — this is the integration assertion. If the optimizer is infeasible on the built universe, tune ONLY the test's fake-exchange factor model (a test-data concern), never the optimizer (protected math). If a real producer bug surfaces (e.g. a geometry field the optimizer needs is unset), fix the producer in `cycle_prep.py` with a focused edit and a unit test in `test_cycle_prep.py`.

- [ ] **Step 4: Run the full suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: PASS (all prior 568 tests + every Phase 8 test green)

- [ ] **Step 5: Commit Tasks 12 + 13 together (the seam + its end-to-end assertions, atomic)**

```bash
git add scripts/run_paper_cli.py tests/test_end_to_end.py tests/test_end_to_end_no_seed.py
git commit -m "feat(run_paper_cli): scout+cycle-prep producer step before the control loop + no-seed E2E

Closes C1 wiring: the driver builds geometries/sleeves/pairs/spreads each cycle so the loop no
longer requires hand-seeded inputs. The legacy seeded E2E stubs _run_producers to a no-op. The
no-seed E2E proves a feasible neutral ~90%-deployed book + passing reviewer end-to-end, and
asserts the LOOP-LEVEL C2 binding: a fabricated produced-spread realized_pnl HALTs at the reviewer
with pair_pnl_attribution.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Reconcile `SKILL.md` AND tighten `tests/test_skill_md.py` (one task, one commit)

> **Merged for atomicity (issue #8):** SKILL.md must name the real producers/driver/dashboard (`cycle_prep_cli.py` / `run_paper_cli.py` / `dashboard_cli.py`), and `test_skill_md.py` must assert every named CLI EXISTS on disk. The on-disk test's `in body` assertion depends on the SKILL.md edits, so the two are done in ONE task with ONE commit (the test is written to be green at commit time — no failing test lands in isolation). The `.exists()` half already holds (`scripts/cycle_prep_cli.py` from Task 6, `scripts/run_paper_cli.py` + `scripts/dashboard_cli.py` from prior phases).

**Files:**
- Modify: `/home/roberto/crypto-trade-claude-code-market-neutral/SKILL.md`
- Modify: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_skill_md.py`

- [ ] **Step 1: Add `cycle_prep_cli.py` to the weekly W3 step**

In SKILL.md, edit the W3 line to name the producer. Replace:

```
**W3 — Universe Scout + preflight.** `scout_cli.py` -> candidates; `preflight.py` audits closes,
folds in every held symbol, builds per-symbol briefs + market context -> `universe.json`.
```

with:

```
**W3 — Universe Scout + preflight + cycle-prep.** `scout_cli.py` -> `universe.json`; `preflight.py`
audits closes, folds in every held symbol, builds per-symbol briefs -> `context.json`;
`cycle_prep_cli.py --cadence weekly --cycle N` builds the cycle's geometry/sleeve/pair/spread
artifacts (`geometries.json`, `sleeves.json`, `pairs.json`, `spreads.json`) the constructor and
reviewer consume. The analysts reason over these artifacts; the optimizer owns the numbers.
```

- [ ] **Step 2: Add `cycle_prep_cli.py` to the daily D3 step**

Replace the D3 line:

```
**D3 — Sentiment refresh + recompute.** Dispatch `sentiment` (light) and recompute drift / z-scores /
funding / neutrality -> updated geometry, `sentiment.json`. The same symbol set as the weekly meeting;
trade only drift/breaches.
```

with:

```
**D3 — Sentiment refresh + cycle-prep.** Dispatch `sentiment` (light); `cycle_prep_cli.py --cadence
daily --cycle N` rebuilds the daily geometry/sleeve/pair/spread artifacts (drift / z-scores / funding /
neutrality) for the SAME symbol set as the weekly meeting; trade only drift/breaches.
```

- [ ] **Step 3: Add a "Run the desk / Read the dashboard" section before the "Between cycles" section**

Insert before `## Between cycles — monitor tripwire`:

```
## Run the whole desk (one command) + read the dashboard
The dual-cadence run is glued together by the deterministic driver `run_paper_cli.py`, which
serializes WEEKLY-then-DAILY under ONE run lock and walks both ladders (scout -> cycle-prep ->
control loop -> reviewer -> execute -> equity -> reflect):
- `uv run python scripts/run_paper_cli.py` (or `--now <ISO>` to pin the instant offline).
After a run, read the KPI dashboard (primary KPI `no_losing_month`; secondary daily Sharpe ×365):
- `uv run python scripts/dashboard_cli.py --format both`.
PAPER-ONLY: `run_paper_cli.py` never sends a live order; the execute boundary records what WOULD fill.
```

- [ ] **Step 4: Add the on-disk existence test (append to tests/test_skill_md.py) — green at commit time**

```python
# tests/test_skill_md.py  (append)
def test_skill_md_named_clis_exist_on_disk() -> None:
    """Every CLI the ladders name must EXIST as a script on disk — not merely be mentioned."""
    body = SKILL_PATH.read_text()
    expected = [
        "runlock_cli.py", "due_check.py", "scout_cli.py", "preflight.py",
        "cycle_prep_cli.py", "control_loop_cli.py", "reviewer_cli.py",
        "gate_execute_cli.py", "record_lessons_cli.py", "promote_lesson_cli.py",
        "reflect_cli.py", "monitor_cli.py", "run_paper_cli.py", "dashboard_cli.py",
    ]
    for cli in expected:
        assert cli in body, f"SKILL.md must name the provenanced CLI {cli}"
        assert (Path("scripts") / cli).exists(), f"named CLI scripts/{cli} is missing on disk"
```

- [ ] **Step 5: Run the SKILL.md test suite (now fully green) + the full suite, then commit**

Run: `uv run pytest tests/test_skill_md.py -v && uv run pytest -q`
Expected: PASS — every SKILL.md test green (ladders W1-W12 / D1-D8 still ordered, reviewer-before-execute intact, all named CLIs exist on disk), full suite green.

```bash
git add SKILL.md tests/test_skill_md.py
git commit -m "docs(SKILL): name cycle_prep_cli/run_paper_cli/dashboard_cli; assert CLIs exist on disk

Reconciles the runbook with the real producers + driver and tightens test_skill_md to require each
named CLI file to exist (not merely be mentioned). One atomic commit: the test is green as committed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: `README.md` — operating rules + how to run a cycle + read the dashboard

Spec §16 lists README among the project layout; it is absent. Write a concise operator README grounding the paper-only / LLM-proposes-code-disposes / every-cycle reviewer hard-veto / neutrality+deployment mandate, plus the run + dashboard commands.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/README.md`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_docs_exist.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_exist.py
from __future__ import annotations

from pathlib import Path


def test_readme_exists_and_names_the_run_and_dashboard_commands():
    text = Path("README.md").read_text()
    assert "run_paper_cli.py" in text, "README must show how to run a cycle"
    assert "dashboard_cli.py" in text, "README must show how to read the dashboard"
    assert "paper" in text.lower(), "README must state the desk is paper-only"
    assert "neutral" in text.lower(), "README must state the dollar+beta-neutral mandate"


def test_claude_md_exists_with_operating_rules():
    text = Path("CLAUDE.md").read_text()
    assert "live" in text.lower() and "false" in text.lower(), "CLAUDE.md must affirm live=false"
    assert "reviewer" in text.lower(), "CLAUDE.md must state the every-cycle reviewer hard-veto"
    assert "protected" in text.lower(), "CLAUDE.md must state the protected-module rule"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs_exist.py -v`
Expected: FAIL — `FileNotFoundError: [Errno 2] No such file or directory: 'README.md'`

- [ ] **Step 3: Write README.md**

```markdown
# Market-Neutral Crypto Trading Desk (paper)

An adversarial multi-agent, **dollar + beta-neutral** crypto trading desk on Binance USD-M perpetual
futures. **Paper only** — real mainnet data, simulated execution, no path to live capital. The desk
profits from relative value, funding-rate carry, cross-sectional factors, and sentiment while staying
neutral to the overall crypto market (target: **never lose a calendar month**; secondary: maximize
daily-equity Sharpe ×365).

## Architecture — "LLM proposes, code disposes"
- **Reasoning layer (LLM agents, `agents/*.md`)** reasons and proposes; it never computes final numbers.
- **Deterministic spine (`futures_fund/`, Python, unit-tested)** owns ALL math: signal computation
  (beta, cointegration, the four alpha sleeves), the dollar+beta-neutral optimizer, sizing, the risk
  gate, fee/funding/slippage accounting, execution simulation, P&L, and state.
- Every LLM output is validated against a pydantic contract before the spine consumes it; the spine
  fails loud (HALT) on contract violations.

## Mandates (non-negotiable)
- **Paper only.** `live` stays `false` forever.
- **Neutrality is a construction constraint, not telemetry.** |Σlong$ − Σshort$| within the dollar
  band AND |Σ wᵢ·βᵢ| within the beta band, with a BTC hedge leg absorbing residual beta.
- **Full two-sided deployment by default** — ≥90% of each side's budget deployed (counters the prior
  desk's all-cash / one-sided ratchet death).
- **Every cycle, the Adversarial Code & Calc Reviewer is a hard veto.** It re-derives neutrality
  residuals, funding sign/amount, pair P&L, RR-after-costs, Sharpe annualization, exchange-filter and
  sentiment-cap compliance from ground truth; `passed=False` HALTs before any fill.

## Run a cycle
```bash
uv sync
uv run python scripts/run_paper_cli.py                       # wall-clock UTC
uv run python scripts/run_paper_cli.py --now 2026-06-11T00:00:00+00:00   # pinned / offline
```
The driver serializes WEEKLY-then-DAILY under one run lock and walks both ladders: scout →
cycle-prep (geometries/sleeves/pairs/spreads) → control loop (the neutrality optimizer) → reviewer
(hard veto) → gate-execute → equity → reflect.

## Read the dashboard
```bash
uv run python scripts/dashboard_cli.py --format both
```
Primary KPI: `no_losing_month` (target 1.0). Secondary: daily Sharpe (×365). Process KPIs:
both-sides deployment rate, neutrality-residual adherence, pair-survival, carry-capture, sentiment
hit-rate, reviewer veto-rate.

## Layout
`agents/` LLM prompts · `futures_fund/` deterministic spine · `scripts/` CLIs the orchestrator
invokes · `state/` cycle artifacts + equity log · `memory/` lessons/journal/repair-journal ·
`tests/` pytest suite · `docs/` specs + plans. `SKILL.md` is the orchestrator's runbook;
`CLAUDE.md` holds the operating rules.

## Tests
```bash
uv run pytest -q
uv run ruff check .
```
```

- [ ] **Step 4: Run the README half of the test (CLAUDE.md half still fails — Task 16)**

Run: `uv run pytest tests/test_docs_exist.py::test_readme_exists_and_names_the_run_and_dashboard_commands -v`
Expected: PASS (1 passed; the README half is green. The CLAUDE.md test is still failing — fixed in Task 16.)

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_docs_exist.py
git commit -m "docs(README): operating rules + how to run a cycle + read the dashboard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: `CLAUDE.md` — operating rules (spec §16)

Spec §16 lists `CLAUDE.md`; it is absent. Write the operating-rules file (paper-only, LLM-proposes-code-disposes, every-cycle reviewer hard-veto, neutrality+deployment mandate, protected-module rule, how to run a cycle). The failing test (`test_claude_md_exists_with_operating_rules`) was already added to `tests/test_docs_exist.py` in Task 15.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/CLAUDE.md`

- [ ] **Step 1: Confirm the CLAUDE.md test is currently red**

Run: `uv run pytest tests/test_docs_exist.py::test_claude_md_exists_with_operating_rules -v`
Expected: FAIL — `FileNotFoundError: ... 'CLAUDE.md'`

- [ ] **Step 2: Write CLAUDE.md**

```markdown
# CLAUDE.md — Operating Rules

You are operating the **market-neutral crypto trading desk** (paper). Read `MISSION.md` and `SKILL.md`
before acting. These rules are non-negotiable.

## Hard rules
- **PAPER ONLY.** `live` MUST stay `false`. There is no path to real capital in this project. Never
  set `live: true`; never add a live-order code path.
- **LLM proposes, code disposes.** Agents reason and propose; the deterministic spine (`futures_fund/`)
  owns ALL math/risk/neutrality/sizing/execution and CANNOT be overridden. Never trade by gut, never
  hand-edit `state/`, never weaken a limit.
- **Neutrality + deployment mandate.** The book is dollar AND beta neutral within bands, with a BTC
  hedge leg, ≥90% deployed per side by default. Neutrality is a construction constraint, never an
  excuse to sit flat or one-sided.
- **Every-cycle reviewer is a hard veto.** `scripts/reviewer_cli.py` runs BEFORE every execute and
  is MANDATORY + non-skippable. It re-derives all 17 canonical checks from ground truth; a missing or
  failed `ReviewerVerdict.passed` HALTs (`SystemExit(2)`). The execute boundary independently refuses
  to fill without a passing verdict (`reviewer_gate_ok`). You may NOT proceed to execute on a failed
  or absent verdict.

## Protected modules (never weakened)
A fix to a protected module — `risk_gate`, `executor`, `exits`, `consolidation`, `policy`,
`liquidation`, `sizing`, `cycle` — may NEVER weaken a limit, breaker, or safety path. New logic lives
in new non-protected modules. Full `uv run pytest` must be green before any commit. HALT rather than
bypass a limit you cannot fix safely; journal every repair to `memory/repair-journal.md`.

## Run a cycle
```bash
uv run python scripts/run_paper_cli.py            # weekly-first, both cadences, one lock
uv run python scripts/dashboard_cli.py --format both
```
Per cadence the ladder is: run-lock → due-check → scout/preflight/cycle-prep → control loop →
reviewer (hard veto) → gate-execute → equity → reflect → record-lessons → release lock.

## Self-healing
On any phase error: log to `state/error-log.jsonl`, diagnose the ROOT cause (don't guess-patch), fix
the CODE properly with tests, and resume from the failed phase or degrade safely (emit a neutral
report, never fabricate). Sentiment is fail-soft (missing/stale → neutral) and never blocks the book.
```

- [ ] **Step 3: Run the docs test (now fully green)**

Run: `uv run pytest tests/test_docs_exist.py -v`
Expected: PASS (2 passed)

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): operating rules — paper-only, code-disposes, reviewer hard-veto, neutrality

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 17 (minor): `monitor_book.py` writer — give `monitor_cli.py` a producer

MINOR: `monitor_cli.py` reads `state/monitor_book.json` which has no writer. Add a small `write_monitor_book` so a between-cycle sweeper (or the executor) can persist the light book the monitor evaluates. This is a thin, low-risk producer for an existing consumer — worth doing (not YAGNI) because the monitor's neutrality tripwire is dead without it.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/monitor_book.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_monitor_book.py`

- [ ] **Step 1: Write the failing test**

```python
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
    # the book the writer produces is consumed by the monitor without a HALT (neutral, no liq breach)
    from scripts.monitor_cli import main
    marks = {"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0}
    liqs = {"BTC/USDT:USDT": 30000.0, "ETH/USDT:USDT": 6000.0}
    write_monitor_book(tmp_path / "state", _book(), marks=marks, liq_prices=liqs,
                       balance=20000.0, peak_equity=20000.0)
    assert main(["--state-dir", str(tmp_path / "state")]) == 0  # no HALT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monitor_book.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'futures_fund.monitor_book'`

- [ ] **Step 3: Write minimal implementation**

```python
# futures_fund/monitor_book.py
"""Writer for the light book `monitor_cli.py` evaluates (`state/monitor_book.json`).

The between-cycle monitor (drawdown / liq-distance / neutrality tripwire) reads a self-contained
book artifact but nothing produced it. This module shapes a `TargetWeights` book + live marks/liq
prices into the `{balance, peak_equity, legs:[{symbol, mark, liq_price, notional, beta_btc}]}` rows
`monitor_cli.check_positions` / `check_neutrality` consume, so the monitor's neutrality guard is live
between cycles. Atomic write (tmp + os.replace), mirroring `cycle_io.save_output`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from futures_fund.contracts import TargetWeights


def write_monitor_book(
    state_dir,
    book: TargetWeights,
    *,
    marks: dict[str, float],
    liq_prices: dict[str, float],
    balance: float,
    peak_equity: float,
) -> Path:
    """Persist the light monitor book from a `TargetWeights` + live marks/liq prices.

    Each non-flat leg becomes a row `{symbol, mark, liq_price, notional, beta_btc}` (notional is the
    leg's |target_notional|). A leg with no mark is skipped (the monitor cannot guard a leg whose
    liq-distance it cannot compute)."""
    legs: list[dict] = []
    for leg in book.legs:
        if abs(leg.target_notional) <= 0.0 or leg.symbol not in marks:
            continue
        legs.append({
            "symbol": leg.symbol,
            "mark": float(marks[leg.symbol]),
            "liq_price": float(liq_prices[leg.symbol]) if leg.symbol in liq_prices else None,
            "notional": abs(float(leg.target_notional)),
            "beta_btc": float(leg.beta_btc),
        })
    payload = {"balance": float(balance), "peak_equity": float(peak_equity), "legs": legs}
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "monitor_book.json"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)
    return p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_monitor_book.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/monitor_book.py tests/test_monitor_book.py
git commit -m "feat(monitor_book): writer for the light book monitor_cli evaluates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 18 (minor): `repair_cli.py` — thin CLI over `repair.apply_repair`

MINOR: `futures_fund/repair.py` is fully built (`apply_repair`/`log_error`/`record_repair`) and tested but UNWIRED — no CLI exposes it to the orchestrator's self-healing loop. Add a thin CLI so SKILL.md's "Self-healing" section has a callable entry point. (Not YAGNI: the self-healing loop in SKILL.md references the protected-module guard; without a CLI the orchestrator cannot journal a repair deterministically.)

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/scripts/repair_cli.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_repair_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repair_cli.py
from __future__ import annotations

import json

from scripts.repair_cli import main


def test_repair_cli_refuses_protected_module(tmp_path, capsys):
    rc = main(["--module", "risk_gate", "--symptom", "x", "--root-cause", "y",
               "--fix", "z", "--verification", "tests", "--memory-dir", str(tmp_path / "memory")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is False
    journal = (tmp_path / "memory" / "repair-journal.md").read_text()
    assert "REFUSED" in journal


def test_repair_cli_applies_non_protected_module(tmp_path, capsys):
    rc = main(["--module", "cycle_prep", "--symptom", "x", "--root-cause", "y",
               "--fix", "z", "--verification", "tests", "--memory-dir", str(tmp_path / "memory")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is True
    journal = (tmp_path / "memory" / "repair-journal.md").read_text()
    assert "applied" in journal
```

> Note: `repair.PROTECTED_PATHS` includes `cycle` (the cycle ENGINE), but the guard matches on `Path(module).stem`, so `cycle_prep` is NOT protected (its stem is `cycle_prep`, not `cycle`) and is applied — the test relies on that distinction.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_repair_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.repair_cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/repair_cli.py
"""Self-healing repair CLI (SKILL.md Self-healing): gate a proposed fix on the protected-module
guard and journal it (applied or REFUSED) via `repair.apply_repair`.

    uv run python scripts/repair_cli.py --module cycle_prep --symptom ... --root-cause ... \\
        --fix ... --verification "uv run pytest -q"

Closes the MINOR repair-unwired gap. A fix to a protected risk/execution module is REFUSED and
journaled as REFUSED — the self-healing loop can never silently weaken a limit. Exit 0 always (the
guard decision is in the printed JSON `applied`); 2 only on an internal error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund.repair import apply_repair


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gate + journal a self-healing repair.")
    ap.add_argument("--module", required=True)
    ap.add_argument("--symptom", required=True)
    ap.add_argument("--root-cause", required=True)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--verification", required=True)
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args(argv)
    try:
        result = apply_repair(
            args.memory_dir, module=args.module, symptom=args.symptom,
            root_cause=args.root_cause, fix=args.fix, verification=args.verification,
            ts=datetime.now(UTC),
        )
    except Exception as e:  # noqa: BLE001 — surface, never crash the orchestrator silently
        print(f"ERROR: repair failed: {e!r}")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_repair_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/repair_cli.py tests/test_repair_cli.py
git commit -m "feat(repair_cli): thin CLI over repair.apply_repair (protected-module guard + journal)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 19: Final full-suite + lint gate

Confirm the whole phase is green end-to-end and lint-clean before declaring done.

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS — all prior 568 tests + every Phase 8 test (Tasks 1-18) green, zero failures.

- [ ] **Step 2: Run ruff on the whole tree**

Run: `uv run ruff check .`
Expected: `All checks passed!` (selects E,F,I,UP,B; line-length 100). Fix any import-order (I) or unused-import (F401) findings in the new modules/CLIs, then re-run.

- [ ] **Step 3: Sanity-run the driver offline (import smoke test)**

Run: `uv run python -c "import scripts.run_paper_cli, scripts.cycle_prep_cli, scripts.scout_cli, scripts.due_check, scripts.runlock_cli, scripts.preflight, scripts.record_lessons_cli, scripts.repair_cli; print('all CLIs import')"`
Expected: prints `all CLIs import` (no ImportError).

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "chore(phase8): full-suite green + ruff clean across the integration glue

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec / gap coverage:**
- **C1 (alpha engine not wired):** Tasks 2-4 (`cycle_prep` producers via beta/funding/market_data + the four sleeve builders + cointegration build_pair/build_spread/select_pairs/fdr_adjust), Task 6 (`cycle_prep_cli` persists the EXACT shapes the loop loads), Tasks 12+13 (wired into `run_paper_cli` before the loop; no-seed E2E proves it). ✅
- **C2 (reviewer checks starved):** Task 1 (RR-capable proposals), Task 11 (`reviewer_cli` loads `pairs.json` + reconstructs proposals via `trader_io.proposals_from_book`, passes `pairs=`/`proposals=`; binding test uses a 4-leg @4500 book that PASSES caps + deployment so the now-live pair-PnL + RR checks are the only gating ones — a fabricated `Spread.realized_pnl` FAILS, an honest one PASSES). The LOOP-LEVEL C2 binding is closed in Task 13's second sub-test (a fabricated produced-spread realized_pnl HALTs the wired reviewer). ✅
- **I1 (orchestration CLIs absent):** Task 5 (`scout_cli`), Task 7 (`runlock_cli` acquire/release/status), Task 8 (`due_check` DUE FRESH/RETRY/SKIP), Task 9 (`preflight` folds held symbols), Task 10 (`record_lessons_cli` via `lessons.append_lesson`), Tasks 12+13 (`run_paper_cli` runs scout/cycle-prep). Task 14 reconciles SKILL.md AND tightens `test_skill_md.py` to assert files exist on disk — ONE atomic commit. ✅
- **MINOR:** Task 17 (`monitor_book.py` writer — done, not YAGNI), Task 18 (`repair_cli.py` wiring — done, not YAGNI). README (Task 15) + CLAUDE.md (Task 16). ✅
- **No-seed E2E:** Task 13 asserts feasible dollar+beta-neutral ~90%-deployed book + passing reviewer + recorded equity, without `_seed_upstream`. ✅

**Review-issue closure (this revision):**
1. **Binding test bug** — Task 11 `_neutral_book()` now uses 4 legs @4500 (per-name 0.225 ≤ 0.25, side 0.90 ≥ floor) with matching geometries for all four symbols + the AAA/BBB Pair/Spread; the positive C2 assertion is no longer broken by a per-name-cap veto. ✅
2. **Narrative vs implementation** — File-Structure bullet + Task 11 title/intro now state proposals are RECONSTRUCTED from the audited book + geometries (the persisted `proposals.json` is intentionally NOT consumed for RR); the dead `save_output(... "proposals" ...)` block is removed from the `_seed` helper. ✅
3. **Loop-level C2 not exercised** — Task 13's `test_fabricated_pair_pnl_halts_the_wired_loop` tampers a PRODUCED spread's `realized_pnl` in the wired path and asserts `SystemExit(2)` + `pair_pnl_attribution`. ✅
4. **Dependency-ordering self-flag** — Task 4 imports cointegration as a single `from futures_fund.cointegration import build_pair, build_spread, fdr_adjust` line; no F811 deferral. ✅
5. **Two contradictory fixture strategies** — Task 12 commits to the single `_run_producers` no-op stub (the correct one); the duplicate-fakes option is deleted. ✅
6. **Self-contradictory Expected** — Task 12 is split into Step 2a (apply the no-op patch) and Step 2b (`Run -> Expected PASS`), so the stated outcome matches the post-patch state. ✅
7. **TDD gap / untested seam** — Tasks 12+13 are declared ONE atomic unit with ONE commit (Task 13 Step 5); Task 12 has no standalone commit landing an untested seam. ✅
8. **Failing test split across tasks** — the old Tasks 14+15 are merged: SKILL.md is reconciled AND the on-disk test is added in ONE task (new Task 14) with ONE commit; the test is green as committed. ✅
9. **`--state` vs `--state-dir`** — standardized on `--state-dir` across `runlock_cli`/`due_check` (documented in the File-Structure CLI-flag-convention note + each CLI's docstring). ✅
10. **Test-name/count mismatches** — each Step-4 run is pinned to its `-k` selector or specific node id with corrected `(N passed)` counts (Task 1: 3, Task 2: 3, Task 3 `-k build_sleeves`: 3, Task 4 `-k build_pairs`: 2, Task 5: 1, etc.). ✅
11. **Equity-file path assumption** — Task 13 notes `equity_log.record_equity` writes `state/equity-history.jsonl` (the same path the existing seeded E2E asserts — verified inherited, not a new assumption). ✅

**Placeholder scan:** No "TBD"/"implement later"/"handle edge cases"/"similar to Task N". Every code step contains complete code; test bodies are real.

**Type / name consistency (re-verified against the repo):** `proposals_from_book(book, geometries, *, rr, stop_frac, horizon_hours)` (Task 1) referenced identically in Task 11/reviewer_cli. `build_geometries`/`build_sleeves`/`build_pairs_and_spreads` (Tasks 2-4) imported with matching signatures in Task 6. `cadence_due`/`latest_cadence_cycle` (`control_loop.py:66`/`:44`) used in Tasks 8/9. `append_lesson(memory_dir, fields, ts)` (`lessons.py:25`) matches Task 10. `review_cycle(..., pairs=, proposals=)` matches `reviewer.py:791-792`. `check_caps` per-name = `|notional|/cfg.capital_usdt` (=20000) vs `per_name_cap=0.25`, `check_deployment_floor` = `gross/side_budget_usdt` (=10000) vs `0.90` — the Task 11 4-leg @4500 sizing satisfies both. `check_pair_pnl` re-derives `entry_spread = mu - side*entry_z*sigma_eq`; the honest test recomputes `expected` via the real function. `save_output/load_output(..., *, cadence=)` match `cycle_io.py:34/:59`. `SleeveSignal.sleeve` ∈ {carry,pairs,factor,sentiment} matches the sleeve builders. `FundingInfo` fields (`current_rate`/`next_funding_ts`/`interval_hours`/`mark_price`/`index_price`) match `market_data.py:11`. `risk_parity_budgets(sleeves) -> dict[SleeveName,float]` matches `neutrality.py:222`. `apply_repair(...) -> {"applied": bool, "reason": str}` and `record_repair` headers `repair (applied)`/`repair (REFUSED)` match `repair.py:42/:60`; `PROTECTED_PATHS` (stem match) excludes `cycle_prep`. `clamp_funding_rate` clamps DOGE 0.5 → 0.02 (`PER_SYMBOL_CAP_DEFAULT`), BTC/ETH → 0.003 (`MAJOR_CAP`). `equity_log.record_equity(state_dir, ts, equity, cycle)` writes `state/equity-history.jsonl` (`equity_log.py:22`). All consistent.

**Known integration risk flagged for the executor:** Task 13 Step 2 flags optimizer-feasibility (and the requirement that ≥1 cointegrated spread survive for the tamper sub-test) as TEST-DATA tuning concerns — tune the fake-exchange factor model only, never the protected optimizer / cointegration math.
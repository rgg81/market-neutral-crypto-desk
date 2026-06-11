# Phase 9 — Realistic Paper-Trading P&L + Cost-Transparency Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat-constant equity (`run_paper_cli.py` line 231 `equity = settings.account_size_usdt`) with a real paper-trading ledger that fills at marks, deducts taker fees + depth slippage, settles signed funding carry between cycles, marks positions to market, writes a per-cycle cost/P&L attribution artifact, persists realized per-leg costs onto the journal so the Reflector keys on net alpha, exposes realized costs/carry as artifacts the external SKILL.md orchestrator injects into agent prompts, and proves — through the wired loop run TWICE one sim-day apart — that the equity series moves and `pnl.json` carries NON-ZERO funding.

**Architecture:** A new `futures_fund/account.py` ledger (`Position` + `PaperAccount`) REUSES `costs.py` (fees, `count_funding_events`), `funding_intervals.py` (`clamp_funding_rate`, `realized_funding`, `funding_interval_hours`) and `slippage.py` (`estimate_slippage`) — it never re-implements fee/funding/slippage math. The ledger persists atomically to a single `state/account.json` across cadences/cycles (tmp + `os.replace`, the same discipline as `cycle_io.save_output` / `equity_log.record_equity`).

**The reconcile-to-target invariant (load-bearing — fixes the multi-week double-count).** `report.json["executed"]` is the executed proposal book verbatim (`orchestration.gate_execute_step` line 51 `"executed": proposals`), and each leg carries the optimizer's per-symbol `target_notional` — for weekly this is the FULL intended book, re-emitted every weekly cycle. Because `account.json` PERSISTS across runs, an ADDITIVE `apply_fills` would increase every already-held same-direction position to ~2× on weekly cycle 2, ~3× on cycle 3, unbounded — it would never converge to the target book. **Therefore `apply_fills` RECONCILES each touched symbol to its leg's `target_notional` as the per-symbol TARGET signed notional**: it fills only `delta = target_signed_notional − current_signed_notional` (a same-direction increase, a reduce/close, or a flip). Re-sending the identical weekly book is then an exact no-op (delta 0). Daily's sparse `rebalance_trades.json` legs each carry that symbol's NEW post-rebalance `target_notional`, so the SAME reconcile path nudges only the changed symbols toward their new target — correct for both cadences, idempotent under a DUE RETRY, and convergent across weeks.

**The cadence-aware funding clock (load-bearing — fixes the prev_ts collision).** `equity_log.record_equity` keys ONLY on `cycle` (line 39) and writes a SINGLE state-root `equity-history.jsonl` with no cadence dimension: in one `main` invocation weekly records cycle 1 @ `now`, then daily records cycle 1 @ `now` and REPLACES the weekly point (same cycle number), so only one equity point survives per run and `series[-1][0]` is corrupted as a funding `prev_ts`. **Therefore funding accrual derives `prev_ts` from a per-account `last_funding_ts` field on `PaperAccount`, NOT from the equity series.** `settle_funding` advances `last_funding_ts` to `now` after it settles, so two `main` runs one sim-day apart settle the real elapsed window regardless of the cycle-key collision.

**`run_paper_cli._run_cadence`** loads the account, settles funding from the account's `last_funding_ts` to `now`, reconciles THIS cycle's executed book to target, marks to market, records the real equity, writes `pnl.json` + appends `ledger.jsonl`, **patches each executed leg's realized fees/slippage/funding/pnl onto the journal `Decision`** (so the Reflector's net-alpha keying has real inputs), and saves the account. Funding is settled BEFORE the fills so a position opened THIS cycle earns no funding for a pre-existence window.

**Agent cost-awareness is artifact-driven, injected by the EXTERNAL orchestrator (honest scope).** This repo contains NO LLM/Task/prompt-assembly code in `futures_fund/` or `scripts/`. `build_scorecard` is imported/called only by `scripts/promote_lesson_cli.py`, and `context.json` is read by no Python in the pipeline (`run_paper_cli`/`_run_cadence` never invoke `preflight.main`). The agent-facing pieces this plan ships — the `pnl` block folded into `context.json`, the cost keys on `build_scorecard`, and the per-leg costs on `reflection_input.json` — are ARTIFACTS the out-of-repo SKILL.md / Claude-Code orchestrator reads and injects into prompts. No automated test can prove an agent consumes them; the tests here prove only that the artifacts are PRODUCED with the right shape and values. The plan does NOT claim a wired Python prompt-injection path, and does NOT claim the desk provably "trades cost-aware" — only that the cost surface is now available to the orchestrator that assembles prompts.

**The reviewer/self-audit** gains an account-level invariant reconciling recorded equity and per-cycle funding against a `costs.py`/`funding_intervals.py` recompute.

**Tech Stack:** Python 3.11+, Pydantic v2 (`BaseModel`), pytest, `uv run`, ruff (`E,F,I,UP,B`). Branch `master`. Reuse: `futures_fund/costs.py`, `futures_fund/funding_intervals.py`, `futures_fund/slippage.py`, `futures_fund/cycle_io.py`, `futures_fund/equity_log.py`, `futures_fund/contracts.py` (`CoinGeometry`, `SymbolSpec`), `futures_fund/models.py` (`Direction`, `Cadence`), `futures_fund/journal.py` (`patch_outcome`, `Decision`).

**Honest architecture note (carry into the plan + the `pnl.json` `notes` field):** In the ACCELERATED live demo each tick re-reads the CURRENT marks, so cross-tick PRICE PnL is small — the curve mainly reflects FUNDING carry accrued per sim-day on held positions plus fees/slippage (on-thesis: a market-neutral carry desk's edge IS the carry net of cost). A true multi-day price-PnL curve needs a real daily cadence or PIT historical marks. Funding is NON-ZERO only when the run advances `now` across at least one funding settlement (every 8h by default); a single-`now` run settles 0 events — which is exactly why Task 16's funding proof runs `main` TWICE one sim-day apart.

**GIT SAFETY:** Implementers stay on branch `master`. Never `checkout`, `reset`, `stash`, or switch branches. Commit only the files each task names. Commit trailer on every commit:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Grounded facts (verified against the repo — reference, do not re-discover)

- `costs.py`: `trade_fee(notional, *, maker, pay_bnb=False) -> float` (≥0 USDT); `count_funding_events(entry_ts, exit_ts, interval_hours=8) -> int` (settlements in `(entry_ts, exit_ts]`). `project_funding` is COST-perspective (+ = we pay) — **do not** use it for balance credits.
- `funding_intervals.py`: `clamp_funding_rate(symbol, rate) -> float` (sign-preserving clamp to `[-cap,+cap]`); `realized_funding(notional_signed, mark, qty, rate, direction) -> float` returns `-side*mark*qty*rate`, **BALANCE-credit perspective** (short + positive rate = POSITIVE credit). `notional_signed` is unused. `funding_interval_hours(symbol, exchange) -> float` (defaults 8.0).
- `slippage.py`: `estimate_slippage(symbol, qty, reference_price, *, depth, adv_usd, half_spread_bps, k=0.1) -> float` (USDT, prefers depth, never flat); `slippage_bps(cost_usdt, notional) -> float`.
- `contracts.py`: `CoinGeometry` carries `symbol, mark, momentum_20, realized_vol, beta_btc, beta_lookback_days, funding_rate, funding_interval_hours=8.0, funding_apr, funding_cap, in_pair, pair_id, sentiment_score, sentiment_conf, adv_usd, spec: SymbolSpec | None, market_info`. `geometries.json` = `GeometryBundle.model_dump(mode="json")` = `{"geometries": [CoinGeometry...], "as_of_ts": ...}`, written by `cycle_prep_cli.main` via `save_output(state_dir, cycle, "geometries", bundle, cadence=cadence)` — per cadence cycle.
- `models.py`: `SymbolSpec(symbol, tick_size, step_size, min_notional, mmr_brackets)`; `Direction = Literal["long","short"]`; `Cadence = Literal["weekly","daily"]`.
- `cycle_io.py`: `cycle_dir(state_dir, cycle_no, *, cadence=None) -> Path` = `state/<cadence>/cycle/<n>`; `save_output(state_dir, cycle_no, name, data, *, cadence=None) -> Path` (atomic); `load_output(...) -> dict` (raises `FileNotFoundError` when absent).
- `equity_log.py`: `record_equity(state_dir, ts, equity, cycle) -> None` keys ONLY on `cycle` (REPLACE-by-cycle), one state-root `equity-history.jsonl`, NO cadence dimension; `equity_series(state_dir) -> list[tuple[str, float]]` (`[-1][0]` = last recorded ts iso). **Cross-cadence collision: weekly cycle 1 and daily cycle 1 share the same key, so the daily point overwrites the weekly point — `equity_series` is NOT a safe funding `prev_ts` source.**
- `run_paper_cli.py`: line 231 `equity = settings.account_size_usdt`; `_run_cadence(cadence, state_dir, memory_dir, now, equity)` records equity at the Step-7a seam; `_read_executed(state_dir, cadence, cycle)` reads `report.json["executed"]`; `_trade_legs` returns the full book (weekly) or `rebalance_trades.json` legs (daily). Each executed proposal: `{"symbol","direction","target_notional": abs(...),"trigger_type":"market","rationale"}` — **no fill price / qty**.
- `report.json["executed"]` = the proposal dicts verbatim (`orchestration.gate_execute_step` line 51). For WEEKLY this is the FULL target book re-emitted every weekly cycle (the multi-week double-count source the reconcile-to-target invariant fixes).
- `daily rebalance_trades.json` = `{"legs": [WeightLeg.model_dump...]}`; a flatten leg has `target_notional=0.0` (and is dropped by `_proposals_from_legs`).
- `journal.py`: `Decision(extra="allow")` has `realized_pnl, fees, funding_paid, slippage` (NOT `realized_funding`); `patch_outcome(memory_dir, *, cycle, symbol, direction, outcome: dict) -> bool` merges & re-validates (so `extra` keys like `realized_funding` round-trip); `read_all_decisions(memory_dir) -> list[dict]` (ISO strings, not datetimes). **Nothing in the repo currently patches `fees/slippage/realized_pnl` onto journal decisions — Task 8b adds that patch so `reflect_cli._cost_fields` is not inert.**
- `reflect_cli.build_reflection_input(memory_dir)` loops `for d in read_all_decisions(memory_dir):` — the loop variable is **literally `d`** (line 35). The entry dict is assembled, then appended to winners/losers (line 56). `realized_funding` is NOT a `Decision` field, but `Decision(extra="allow")` round-trips it after Task 8b's `patch_outcome`.
- `scorecard.build_scorecard(state_dir, memory_dir, *, last_n=10, ...) -> dict` is imported/called ONLY by `scripts/promote_lesson_cli.py` (reads `dsr_pvalue`). It is NOT injected by any Python in this repo — prompt injection is performed by the external SKILL.md orchestrator. **Pre-existing bug (out of scope, documented for Task 15):** `scorecard.py:129` and `dashboard.py:111` call `carry_capture_rate(memory_dir, ...)` but `improvement.carry_capture_rate(state_dir, last_n=10)` takes `state_dir` — the wrong directory. Task 15 must NOT lean on `carry_capture_rate` being meaningful; it routes carry through `context.json.pnl` instead.
- `dashboard.build_kpi_dashboard(state_dir, memory_dir, *, last_n=10) -> dict`; `scripts/dashboard_cli.py` `_ROWS` tuple drives display order; `_fmt` renders floats / `nan`→"n/a".
- `preflight.py` writes `context.json` = `{"briefs": [{"symbol","held"}], "held": [...]}` via `save_output(..., "context", ctx, cadence=cadence)`. **It is NOT invoked anywhere in `run_paper_cli`/`_run_cadence`; `context.json` is read only by `tests/test_preflight_cli.py` and (out of repo) the SKILL.md orchestrator.** `preflight.main` calling `load_settings()` with no `config.yaml` in cwd is SAFE: verified `load_settings()` returns `Settings(account_size_usdt=20000.0)` when the file is absent (it `yaml.safe_load`s `{}` and applies the field default). Tasks here still pass an explicit `default_cash` so a cold env cannot raise.
- `self_audit.py`: `_checks() -> list[tuple[str,bool,str]]` with `add(name, ok, detail)`; `run_self_audit() -> {"ok":..., "checks":[...]}`; already imports `realized_funding` and has `invariant_funding_sign_correct`.

---

## File Structure

**New files:**
- `futures_fund/account.py` — the ledger. `Position` (BaseModel) + `PaperAccount` (BaseModel, with `last_funding_ts: datetime | None`) with `apply_fills` (reconcile-to-target), `settle_funding` (advances `last_funding_ts`), `mark_to_market`, `equity`, `to_dict`/`from_dict`, and module functions `load_account(state_dir, default_cash)` / `save_account(state_dir, account)` (atomic `account.json` at the state root).
- `futures_fund/pnl_attribution.py` — pure builders: `build_cycle_pnl(...)` (the `pnl.json` record) and `append_ledger(state_dir, record) -> None` (atomic `ledger.jsonl` append).
- `tests/test_account.py` — ledger unit tests.
- `tests/test_pnl_attribution.py` — `pnl.json` / `ledger.jsonl` builder tests.
- `tests/test_account_integration.py` — the ≥2-cycle integration test (equity no longer constant; non-zero funding/fees in `pnl.json`).
- `tests/test_agent_cost_context.py` — context-bus cost-block + scorecard-keys + journal-patch artifact tests.
- `tests/fixtures/pnl_block.json` — the injected-block fixture/contract.

**Modified files:**
- `scripts/run_paper_cli.py` — wire account load → settle funding (from `last_funding_ts`) → reconcile-to-target fills → MTM → real equity → `pnl.json` + `ledger.jsonl` → patch journal costs → save account.
- `scripts/preflight.py` — fold a `pnl` block + per-symbol realized-cost into `context.json` (artifact for the external orchestrator).
- `futures_fund/scorecard.py` — add cost-transparency keys to `build_scorecard`.
- `futures_fund/dashboard.py` — add `gross_pnl/net_pnl/total_fees/total_slippage/total_funding/cost_drag_bps`.
- `scripts/dashboard_cli.py` — add the cost rows to `_ROWS`.
- `futures_fund/self_audit.py` — add `invariant_account_equity_reconciles` + `invariant_cycle_funding_reconciles` and wire into `_checks`.
- `agents/funding_carry.md`, `agents/pair_analyst.md`, `agents/trader.md`, `agents/research_manager.md`, `agents/reflector.md` — small scoped additions naming the new cost artifacts, each declaring under `## Inputs` that `context.json` is provided by the orchestrator.
- `scripts/reflect_cli.py` — add per-closed-leg `realized_funding/fees/slippage/net_pnl` to each winner/loser entry (now fed by Task 8b's journal patch).
- `tests/test_end_to_end_no_seed.py` — assert equity moves + `pnl.json`/`account.json`/`ledger.jsonl` exist (single-run, no seed) AND a NEW two-run E2E proving non-zero funding in `pnl.json`.

---

### Task 1: `Position` + `PaperAccount` skeleton (with `last_funding_ts`) + JSON round-trip

**Files:**
- Create: `futures_fund/account.py`
- Test: `tests/test_account.py`

`PaperAccount` carries a `last_funding_ts: datetime | None` — the per-account funding clock that `settle_funding` advances (the funding `prev_ts` source, NOT the cycle-collided equity series).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_account.py
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import PaperAccount, Position


def _pos(symbol="ETH/USDT:USDT", direction="long", qty=2.0, entry=2000.0):
    return Position(
        symbol=symbol, direction=direction, qty=qty, entry_price=entry,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC),
    )


def test_account_persistence_round_trip():
    acct = PaperAccount(cash=20_000.0)
    acct.positions[_pos().symbol] = _pos()
    acct.realized_pnl = 12.5
    acct.fees_paid = 3.0
    acct.slippage_paid = 1.0
    acct.funding_received = 4.0
    acct.funding_paid = 2.0
    acct.last_funding_ts = datetime(2026, 6, 10, 8, tzinfo=UTC)

    restored = PaperAccount.from_dict(acct.to_dict())

    assert restored.cash == 20_000.0
    assert restored.realized_pnl == 12.5
    assert restored.fees_paid == 3.0
    assert restored.slippage_paid == 1.0
    assert restored.funding_received == 4.0
    assert restored.funding_paid == 2.0
    assert restored.last_funding_ts == datetime(2026, 6, 10, 8, tzinfo=UTC)
    pos = restored.positions["ETH/USDT:USDT"]
    assert pos.qty == 2.0
    assert pos.entry_price == 2000.0
    assert pos.direction == "long"


def test_fresh_account_has_no_funding_clock():
    assert PaperAccount(cash=20_000.0).last_funding_ts is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py -k "round_trip or funding_clock" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.account'`

- [ ] **Step 3: Write minimal implementation**

```python
# futures_fund/account.py
"""Phase 9 — paper-trading P&L ledger (Position + PaperAccount).

REUSES the cost primitives — it NEVER re-implements fee/funding/slippage math:
  * costs.trade_fee / costs.count_funding_events
  * funding_intervals.clamp_funding_rate / funding_intervals.realized_funding
  * slippage.estimate_slippage

Funding sign convention (load-bearing): this ledger settles funding via
`funding_intervals.realized_funding`, which is BALANCE-credit perspective (a SHORT with a positive
rate RECEIVES funding -> a POSITIVE cash credit). Do NOT use `costs.project_funding` here (that is
the opposite, cost/paid perspective).

Funding clock (load-bearing): the account carries its OWN `last_funding_ts`, advanced by
`settle_funding`. The equity series is NOT a safe `prev_ts` source — `equity_log.record_equity` keys
only on `cycle`, so weekly cycle 1 and daily cycle 1 collide and the daily point overwrites the
weekly one in a single run.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from futures_fund.models import Direction


class Position(BaseModel):
    symbol: str
    direction: Direction
    qty: float                                   # absolute contract qty (>= 0)
    entry_price: float                           # avg entry (VWAP of accumulated fills)
    opened_ts: datetime
    accrued_funding: float = 0.0                 # signed, + = received, - = paid (this leg's life)
    accrued_fees: float = 0.0                    # >= 0 taker/maker fees charged to this leg
    accrued_slippage: float = 0.0               # >= 0 depth slippage charged to this leg
    realized_pnl: float = 0.0                    # signed price P&L realized on this leg so far


class PaperAccount(BaseModel):
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)
    realized_pnl: float = 0.0
    last_funding_ts: datetime | None = None       # the funding clock (NOT the equity series)
    # cumulative cost totals across the account's life
    fees_paid: float = 0.0                        # >= 0
    slippage_paid: float = 0.0                    # >= 0
    funding_received: float = 0.0                 # >= 0 (sum of positive settlements)
    funding_paid: float = 0.0                     # >= 0 (sum of |negative settlements|)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> PaperAccount:
        return cls.model_validate(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py -k "round_trip or funding_clock" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/account.py tests/test_account.py
git commit -m "feat(account): Position + PaperAccount ledger skeleton (with funding clock) + JSON round-trip

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `mark_to_market` + `equity`

**Files:**
- Modify: `futures_fund/account.py`
- Test: `tests/test_account.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_account.py — append
def test_mark_to_market_and_equity_long_and_short():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="long", qty=2.0, entry=2000.0)
    acct.positions["BTC/USDT:USDT"] = _pos(
        symbol="BTC/USDT:USDT", direction="short", qty=0.1, entry=60_000.0)

    marks = {"ETH/USDT:USDT": 2100.0, "BTC/USDT:USDT": 59_000.0}
    upnl = acct.mark_to_market(marks)

    # long: 2*(2100-2000)=200 ; short: 0.1*(60000-59000)=100
    assert upnl["ETH/USDT:USDT"] == 200.0
    assert upnl["BTC/USDT:USDT"] == 100.0
    assert acct.equity(marks) == 20_000.0 + 300.0


def test_equity_skips_symbols_missing_a_mark():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="long", qty=2.0, entry=2000.0)
    # no mark for ETH -> contributes 0, equity == cash
    assert acct.equity({}) == 20_000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py -k "mark_to_market or equity_skips" -v`
Expected: FAIL with `AttributeError: 'PaperAccount' object has no attribute 'mark_to_market'`

- [ ] **Step 3: Write minimal implementation**

Add these methods to `PaperAccount` in `futures_fund/account.py` (below `from_dict`):

```python
    def mark_to_market(self, marks: dict[str, float]) -> dict[str, float]:
        """Unrealized PnL per held symbol (skips symbols with no mark).

        long: qty*(mark-entry) ; short: qty*(entry-mark)."""
        upnl: dict[str, float] = {}
        for sym, pos in self.positions.items():
            mark = marks.get(sym)
            if mark is None:
                continue
            if pos.direction == "long":
                upnl[sym] = pos.qty * (mark - pos.entry_price)
            else:
                upnl[sym] = pos.qty * (pos.entry_price - mark)
        return upnl

    def equity(self, marks: dict[str, float]) -> float:
        """cash + sum unrealized PnL (skips symbols missing a mark)."""
        return self.cash + sum(self.mark_to_market(marks).values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py -k "mark_to_market or equity_skips" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/account.py tests/test_account.py
git commit -m "feat(account): mark_to_market + equity (cash + unrealized)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `apply_fills` — RECONCILE each touched symbol to its leg's `target_notional`

**Files:**
- Modify: `futures_fund/account.py`
- Test: `tests/test_account.py`

The executed proposals carry NO price/qty and each leg's `target_notional` is the optimizer's per-symbol TARGET. **`apply_fills` reconciles each touched symbol to that target signed notional**, filling only the delta: `target_qty = target_notional / mark`; `delta_qty = target_qty − current_signed_qty` (signed by direction). A positive delta opens/increases the same side; a negative delta reduces/closes/flips. Re-sending the identical book is a no-op. Frictions (taker fee + depth slippage) are charged on the |delta notional| actually traded — re-sending an unchanged book trades 0 and costs 0. `executed_trades` is `report.json["executed"]`; `marks` is `{symbol: mark}`; `costs` is `{symbol: CostInputs}` (ADV + half-spread; paper passes `depth=None` so `estimate_slippage` uses the fallback).

This task implements the OPEN/INCREASE and the no-op paths; Task 4 implements the REDUCE/CLOSE/FLIP delta path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_account.py — append
from futures_fund.account import CostInputs


def test_apply_fills_opens_position_charges_fee_and_slippage():
    acct = PaperAccount(cash=20_000.0)
    executed = [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}]
    marks = {"ETH/USDT:USDT": 2000.0}
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0)}

    acct.apply_fills(executed, marks, costs)

    pos = acct.positions["ETH/USDT:USDT"]
    assert pos.qty == 2.0                          # 4000 / 2000
    assert pos.direction == "long"
    # taker fee on 4000 notional = 4000 * 0.0005 = 2.0 USDT, charged to cash + accrued
    assert pos.accrued_fees == 2.0
    assert acct.fees_paid == 2.0
    assert pos.accrued_slippage > 0.0
    assert acct.slippage_paid == pos.accrued_slippage
    # cash deducted by fee + slippage only (paper-margin: the notional itself consumes no cash)
    assert acct.cash == 20_000.0 - 2.0 - pos.accrued_slippage


def test_apply_fills_resend_same_target_is_a_noop():
    """The multi-week double-count guard: re-sending the IDENTICAL book trades 0 -> 0 frictions,
    same qty (NOT doubled)."""
    acct = PaperAccount(cash=20_000.0)
    book = [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}]
    marks = {"ETH/USDT:USDT": 2000.0}
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0)}
    acct.apply_fills(book, marks, costs)
    fees_after_open, slip_after_open = acct.fees_paid, acct.slippage_paid
    acct.apply_fills(book, marks, costs)            # same target again -> reconcile to delta 0
    assert acct.positions["ETH/USDT:USDT"].qty == 2.0        # NOT 4.0 — no double-count
    assert acct.fees_paid == fees_after_open                 # 0 extra fee
    assert acct.slippage_paid == slip_after_open             # 0 extra slippage


def test_apply_fills_increase_to_a_larger_target_blends_entry_vwap():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 2000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)            # target qty 1.0 @ 2000
    # raise the target to 4400 @ mark 2200 -> target qty 2.0, delta +1.0 filled @ 2200
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4400.0}],
        {"ETH/USDT:USDT": 2200.0}, costs)
    pos = acct.positions["ETH/USDT:USDT"]
    assert abs(pos.qty - 2.0) < 1e-9               # 4400/2200
    # blended VWAP: (1.0 @ 2000 + 1.0 @ 2200) / 2.0 = 2100
    assert abs(pos.entry_price - 2100.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py -k "apply_fills_opens or resend_same or increase_to_a_larger" -v`
Expected: FAIL with `ImportError: cannot import name 'CostInputs'`

- [ ] **Step 3: Write minimal implementation**

Replace the import block at the top of `futures_fund/account.py`:

```python
# futures_fund/account.py — replace the import block
from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from futures_fund.costs import trade_fee
from futures_fund.models import Direction
from futures_fund.slippage import estimate_slippage
```

Add after the `Position` class:

```python
class CostInputs(BaseModel):
    """Per-symbol frictions the paper executor needs but the executed proposal does not carry.

    `depth` is the optional crossing-side book; in paper we leave it None so `estimate_slippage`
    uses the ADV+half-spread fallback (which is NEVER flat 2bps)."""
    adv_usd: float = 0.0
    half_spread_bps: float = 1.0
    depth: list[tuple[float, float]] | None = None
    maker: bool = False                          # paper opens are market -> taker


def _signed_qty(pos: "Position | None") -> float:
    """Current signed qty: + for a long, - for a short, 0 if flat."""
    if pos is None:
        return 0.0
    return pos.qty if pos.direction == "long" else -pos.qty
```

Add `apply_fills` (+ the friction helper + the reduce/close/flip stub) as methods on `PaperAccount`:

```python
    def apply_fills(
        self,
        executed_trades: list[dict],
        marks: dict[str, float],
        costs: dict[str, CostInputs],
        *,
        opened_ts: datetime | None = None,
    ) -> None:
        """RECONCILE each touched symbol to its leg's target_notional (signed by direction).

        Each executed leg's `target_notional` is the optimizer's per-symbol TARGET; this fills only
        `delta = target_signed_qty - current_signed_qty`, so re-sending the identical book is an
        exact no-op (delta 0, 0 frictions). A positive delta opens/increases the SAME side (blending
        entry VWAP); a negative delta reduces/closes/flips (Task 4). Fill at the mark; charge a
        taker/maker fee + depth slippage on the |delta notional| actually traded. qty is derived
        from notional/mark because the executed proposal carries no fill price/qty.

        Convergent across weeks (weekly re-emits the full book -> delta 0 on unchanged legs) and
        correct for daily (each rebalance_trades leg carries that symbol's NEW target_notional)."""
        ts = opened_ts or datetime.now(tz=UTC)
        for trade in executed_trades:
            sym = trade["symbol"]
            direction: Direction = trade["direction"]
            target_notional = abs(float(trade["target_notional"]))
            mark = marks.get(sym)
            if mark is None or mark <= 0:
                continue
            ci = costs.get(sym) or CostInputs()
            sign = 1.0 if direction == "long" else -1.0
            target_signed_qty = sign * (target_notional / mark)
            existing = self.positions.get(sym)
            current_signed_qty = _signed_qty(existing)
            delta_signed_qty = target_signed_qty - current_signed_qty
            if abs(delta_signed_qty) <= 1e-12:
                continue  # already at target -> no-op (re-sent unchanged book)
            delta_notional = abs(delta_signed_qty) * mark
            fee = trade_fee(delta_notional, maker=ci.maker)
            slip = estimate_slippage(
                sym, abs(delta_signed_qty), mark, depth=ci.depth, adv_usd=ci.adv_usd,
                half_spread_bps=ci.half_spread_bps)

            if existing is not None and (delta_signed_qty * sign) < 0:
                # delta opposes the leg's direction by magnitude -> reduce/close/flip (Task 4).
                self._reconcile_opposite(
                    existing, sym, direction, target_signed_qty, mark, fee, slip, ts)
                continue

            # same-side open/increase: fill |delta| at the mark, blend entry VWAP.
            fill_qty = abs(delta_signed_qty)
            self._charge_frictions(sym, fee, slip, existing)
            if existing is None:
                self.positions[sym] = Position(
                    symbol=sym, direction=direction, qty=fill_qty, entry_price=mark,
                    opened_ts=ts, accrued_fees=fee, accrued_slippage=slip)
            else:
                total_qty = existing.qty + fill_qty
                existing.entry_price = (
                    existing.entry_price * existing.qty + mark * fill_qty) / total_qty
                existing.qty = total_qty

    def _charge_frictions(
        self, sym: str, fee: float, slip: float, pos: "Position | None"
    ) -> None:
        self.cash -= fee + slip
        self.fees_paid += fee
        self.slippage_paid += slip
        if pos is not None:
            pos.accrued_fees += fee
            pos.accrued_slippage += slip

    def _reconcile_opposite(
        self, existing: "Position", sym: str, direction: Direction,
        target_signed_qty: float, mark: float, fee: float, slip: float, ts: datetime,
    ) -> None:
        raise NotImplementedError("reduce/close/flip implemented in Task 4")
```

> Note on the same-side branch: a same-direction leg whose target is SMALLER than the current holding produces a `delta_signed_qty` that opposes `sign` (e.g. long target 2000 vs long-held 4000 -> negative delta), so it routes to `_reconcile_opposite` (a REDUCE) — exactly what Task 4 handles. The `(delta_signed_qty * sign) < 0` test catches both "opposite-direction leg" and "same-direction shrink".

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py -k "apply_fills_opens or resend_same or increase_to_a_larger" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/account.py tests/test_account.py
git commit -m "feat(account): apply_fills RECONCILES to target_notional (fixes multi-week double-count)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `apply_fills` reduce / close / FLIP to a target, realizing P&L

**Files:**
- Modify: `futures_fund/account.py`
- Test: `tests/test_account.py`

The reconcile delta drives the held qty TOWARD the target signed qty: a smaller same-side target REDUCES (realizing P&L on the closed portion); a zero target CLOSES (pops the position); an opposite-direction target FLIPS (close the held side flat, reopen the residual at the mark to reach the target). Realized P&L uses the same long/short convention as `mark_to_market`, accumulates onto `account.realized_pnl` AND the leg's `realized_pnl`, and credits/debits cash. Frictions were already computed on |delta notional| by `apply_fills`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_account.py — append
def test_apply_fills_reduce_to_smaller_target_realizes_partial_pnl():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)            # long qty 2 @ 2000
    cash_after_open = acct.cash
    # lower target to 2200 @ mark 2200 -> target qty 1.0, reduce 1.0 @ 2200, realize 1*(2200-2000)
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 2200.0}],
        {"ETH/USDT:USDT": 2200.0}, costs)
    pos = acct.positions["ETH/USDT:USDT"]
    assert pos.direction == "long"
    assert abs(pos.qty - 1.0) < 1e-9               # 2200/2200
    assert abs(acct.realized_pnl - 1.0 * (2200.0 - 2000.0)) < 1e-6
    assert abs(pos.realized_pnl - 200.0) < 1e-6
    assert acct.cash > cash_after_open             # got the realized credit (fee>0, slip=0)


def test_apply_fills_zero_target_closes_and_pops():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)
    # target 0 at the SAME mark closes the whole 2.0 qty flat
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 0.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)
    assert "ETH/USDT:USDT" not in acct.positions
    assert abs(acct.realized_pnl) < 1e-6           # closed flat -> ~0 price pnl


def test_apply_fills_opposite_target_flips_side():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 2000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)            # long qty 1 @ 2000
    # target a SHORT 4000 @ 2000 -> target signed qty -2.0; close the +1 (flat pnl), open 2 short
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "short", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs)
    pos = acct.positions["ETH/USDT:USDT"]
    assert pos.direction == "short"
    assert abs(pos.qty - 2.0) < 1e-9               # |−2.0| target
    assert pos.entry_price == 2000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py -k "reduce_to_smaller or zero_target or opposite_target" -v`
Expected: FAIL with `NotImplementedError: reduce/close/flip implemented in Task 4`

- [ ] **Step 3: Write minimal implementation**

Replace the `_reconcile_opposite` stub body in `futures_fund/account.py`:

```python
    def _reconcile_opposite(
        self, existing: "Position", sym: str, direction: Direction,
        target_signed_qty: float, mark: float, fee: float, slip: float, ts: datetime,
    ) -> None:
        """Drive the held qty TOWARD `target_signed_qty` when the delta opposes the held side:
        reduce -> (close) -> (flip). Realize P&L on the closed portion, charge the (already-computed)
        frictions, and open the residual the other way on a flip. Frictions were sized on the FULL
        |delta notional| by `apply_fills`, so they are charged once here."""
        self._charge_frictions(sym, fee, slip, existing)
        current_signed_qty = _signed_qty(existing)
        # qty being closed on the held side = min(|delta|, held qty), capped at a full close.
        delta_signed = target_signed_qty - current_signed_qty
        closed_qty = min(abs(delta_signed), existing.qty)
        if existing.direction == "long":
            realized = closed_qty * (mark - existing.entry_price)
        else:
            realized = closed_qty * (existing.entry_price - mark)
        self.realized_pnl += realized
        existing.realized_pnl += realized
        self.cash += realized

        residual_held = existing.qty - closed_qty
        if residual_held > 1e-12:
            existing.qty = residual_held
            return
        # fully closed this side -> pop it; reopen the residual on the target side if flipping.
        self.positions.pop(sym, None)
        residual_new_qty = abs(target_signed_qty)
        if residual_new_qty > 1e-12:                # FLIP: open to reach the opposite-side target
            self.positions[sym] = Position(
                symbol=sym, direction=direction, qty=residual_new_qty, entry_price=mark,
                opened_ts=ts, accrued_fees=0.0, accrued_slippage=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py -k "reduce_to_smaller or zero_target or opposite_target" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/account.py tests/test_account.py
git commit -m "feat(account): apply_fills reduce/close/FLIP to target, realizing P&L

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `settle_funding` — signed carry between two timestamps, advancing the funding clock

**Files:**
- Modify: `futures_fund/account.py`
- Test: `tests/test_account.py`

Settle funding for every held position between `prev_ts` and `now`. Per symbol: count `count_funding_events(prev_ts, now, interval_hours)`, clamp the rate, and credit `realized_funding(0.0, mark, qty, clamped_rate, direction) * n_events` to cash (short + positive rate = positive credit). Accumulate signed `accrued_funding` per position and split into cumulative `funding_received` / `funding_paid`. **After settling, advance `self.last_funding_ts = now`** — the per-account funding clock that the wired loop reads as `prev_ts` (NOT the cycle-collided equity series).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_account.py — append
from futures_fund.costs import count_funding_events


def test_settle_funding_short_positive_rate_is_a_credit_and_advances_clock():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="short", qty=2.0, entry=2000.0)
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 11, 0, 0, tzinfo=UTC)       # 24h -> 3 settlements at 8h
    assert count_funding_events(prev, now, 8) == 3
    marks = {"ETH/USDT:USDT": 2000.0}
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, marks)

    # short + positive rate RECEIVES: realized_funding = -(-1)*2000*2*0.0005 = +2.0 per event
    expected = 3 * 2.0
    assert abs(acct.positions["ETH/USDT:USDT"].accrued_funding - expected) < 1e-9
    assert abs(acct.cash - (20_000.0 + expected)) < 1e-9
    assert abs(acct.funding_received - expected) < 1e-9
    assert acct.funding_paid == 0.0
    assert acct.last_funding_ts == now             # the funding clock advanced


def test_settle_funding_long_positive_rate_is_a_debit():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="long", qty=2.0, entry=2000.0)
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 10, 8, 1, tzinfo=UTC)       # 1 settlement at hour 8
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8},
                        {"ETH/USDT:USDT": 2000.0})
    assert abs(acct.positions["ETH/USDT:USDT"].accrued_funding - (-2.0)) < 1e-9
    assert abs(acct.cash - (20_000.0 - 2.0)) < 1e-9
    assert acct.funding_received == 0.0
    assert abs(acct.funding_paid - 2.0) < 1e-9


def test_settle_funding_no_events_still_advances_clock():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="short", qty=2.0, entry=2000.0)
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 10, 1, 0, tzinfo=UTC)       # < 8h -> 0 settlements
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8},
                        {"ETH/USDT:USDT": 2000.0})
    assert acct.cash == 20_000.0
    assert acct.positions["ETH/USDT:USDT"].accrued_funding == 0.0
    assert acct.last_funding_ts == now             # clock advances even with 0 events
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py -k "settle_funding" -v`
Expected: FAIL with `AttributeError: 'PaperAccount' object has no attribute 'settle_funding'`

- [ ] **Step 3: Write minimal implementation**

Update the imports of `futures_fund/account.py` (replace the costs import line, add the funding_intervals import):

```python
from futures_fund.costs import count_funding_events, trade_fee
from futures_fund.funding_intervals import clamp_funding_rate, realized_funding
```

Add `settle_funding` as a method on `PaperAccount`:

```python
    def settle_funding(
        self,
        prev_ts: datetime,
        now: datetime,
        funding_by_symbol: dict[str, float],
        intervals: dict[str, int],
        marks: dict[str, float],
    ) -> None:
        """Settle funding for every held position over (prev_ts, now], then ADVANCE the funding
        clock to `now`.

        Per symbol: n = count_funding_events(prev_ts, now, interval); clamp the rate; credit
        realized_funding(0, mark, qty, clamped_rate, direction) * n to cash (BALANCE-credit
        perspective: a SHORT with a positive rate RECEIVES). Accumulate signed per-position
        accrued_funding and split the total into funding_received (+) / funding_paid (|-|).
        `last_funding_ts` always moves to `now` (even with 0 events) so the next cycle's window
        starts here — the account, not the cycle-collided equity series, is the funding clock."""
        for sym, pos in self.positions.items():
            mark = marks.get(sym)
            if mark is None:
                continue
            interval = int(intervals.get(sym, 8))
            n = count_funding_events(prev_ts, now, interval)
            if n <= 0:
                continue
            rate = clamp_funding_rate(sym, funding_by_symbol.get(sym, 0.0))
            per_event = realized_funding(0.0, mark, pos.qty, rate, pos.direction)
            settled = per_event * n
            pos.accrued_funding += settled
            self.cash += settled
            if settled >= 0.0:
                self.funding_received += settled
            else:
                self.funding_paid += -settled
        self.last_funding_ts = now
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py -k "settle_funding" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/account.py tests/test_account.py
git commit -m "feat(account): settle_funding signed carry + advance per-account funding clock

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Atomic `load_account` / `save_account` at the state root

**Files:**
- Modify: `futures_fund/account.py`
- Test: `tests/test_account.py`

Single `account.json` at the state root (not per-cadence/cycle) — one account across cadences/cycles. Atomic via tmp + `os.replace`. `load_account` returns a fresh account at `default_cash` (with `last_funding_ts=None`) when the file is absent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_account.py — append
from futures_fund.account import load_account, save_account


def test_load_account_fresh_inits_at_default_cash(tmp_path):
    acct = load_account(tmp_path / "state", default_cash=20_000.0)
    assert acct.cash == 20_000.0
    assert acct.positions == {}
    assert acct.realized_pnl == 0.0
    assert acct.last_funding_ts is None


def test_save_then_load_round_trips_at_state_root(tmp_path):
    state = tmp_path / "state"
    acct = PaperAccount(cash=15_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos()
    acct.fees_paid = 7.0
    acct.last_funding_ts = datetime(2026, 6, 10, 8, tzinfo=UTC)
    save_account(state, acct)
    assert (state / "account.json").exists()
    restored = load_account(state, default_cash=99_999.0)
    assert restored.cash == 15_000.0               # NOT the default — the file wins
    assert restored.fees_paid == 7.0
    assert restored.last_funding_ts == datetime(2026, 6, 10, 8, tzinfo=UTC)
    assert restored.positions["ETH/USDT:USDT"].qty == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py -k "load_account_fresh or save_then_load" -v`
Expected: FAIL with `ImportError: cannot import name 'load_account'`

- [ ] **Step 3: Write minimal implementation**

Extend the top import block of `futures_fund/account.py`:

```python
import json
import os
from pathlib import Path
```

Append at module bottom:

```python
def _account_path(state_dir) -> Path:
    return Path(state_dir) / "account.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """tmp + os.replace — a crash mid-write leaves the PRIOR account.json intact (same discipline
    as cycle_io.save_output / equity_log.record_equity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_account(state_dir, default_cash: float) -> PaperAccount:
    """Load the single account.json at the state root, or init a fresh account at `default_cash`
    (zero positions, no funding clock) on a clean state dir — the restart-from-scratch path."""
    p = _account_path(state_dir)
    if p.exists():
        return PaperAccount.from_dict(json.loads(p.read_text()))
    return PaperAccount(cash=default_cash)


def save_account(state_dir, account: PaperAccount) -> None:
    _atomic_write_text(_account_path(state_dir), json.dumps(account.to_dict(), indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py -k "load_account_fresh or save_then_load" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full account suite + ruff**

Run: `uv run pytest tests/test_account.py -v && uv run ruff check futures_fund/account.py tests/test_account.py`
Expected: all PASS, ruff reports `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add futures_fund/account.py tests/test_account.py
git commit -m "feat(account): atomic load_account/save_account at state root

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `pnl_attribution` — the per-cycle `pnl.json` record builder

**Files:**
- Create: `futures_fund/pnl_attribution.py`
- Test: `tests/test_pnl_attribution.py`

The "know all these data" record: `opening_equity, fees_paid, slippage_paid, funding_received, funding_paid, funding_net, realized_pnl, unrealized_pnl, gross_pnl, net_pnl, closing_equity, turnover_usd`, a per-position list, and a `notes` carrying the honest accelerated-demo caveat. Totals are the account's CUMULATIVE totals. `funding_net = funding_received - funding_paid`; `gross_pnl = realized_pnl + unrealized_pnl + funding_net`; `net_pnl = gross_pnl - fees_paid - slippage_paid`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pnl_attribution.py
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import PaperAccount, Position
from futures_fund.pnl_attribution import build_cycle_pnl


def _acct():
    acct = PaperAccount(cash=20_050.0)
    acct.positions["ETH/USDT:USDT"] = Position(
        symbol="ETH/USDT:USDT", direction="short", qty=2.0, entry_price=2000.0,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC),
        accrued_funding=6.0, accrued_fees=2.0, accrued_slippage=1.0)
    acct.realized_pnl = 10.0
    acct.fees_paid = 4.0
    acct.slippage_paid = 2.0
    acct.funding_received = 6.0
    acct.funding_paid = 0.0
    return acct


def test_build_cycle_pnl_record_shape_and_arithmetic():
    acct = _acct()
    marks = {"ETH/USDT:USDT": 1950.0}              # short upnl = 2*(2000-1950)=100
    rec = build_cycle_pnl(
        acct, opening_equity=20_000.0, marks=marks, turnover_usd=4000.0,
        cycle=2, cadence="daily", now=datetime(2026, 6, 11, tzinfo=UTC))

    assert rec["opening_equity"] == 20_000.0
    assert rec["fees_paid"] == 4.0
    assert rec["slippage_paid"] == 2.0
    assert rec["funding_received"] == 6.0
    assert rec["funding_paid"] == 0.0
    assert rec["funding_net"] == 6.0
    assert rec["realized_pnl"] == 10.0
    assert rec["unrealized_pnl"] == 100.0
    assert rec["gross_pnl"] == 10.0 + 100.0 + 6.0
    assert rec["net_pnl"] == rec["gross_pnl"] - 4.0 - 2.0
    assert rec["closing_equity"] == acct.equity(marks)
    assert rec["turnover_usd"] == 4000.0
    assert rec["cycle"] == 2
    assert rec["cadence"] == "daily"
    pos = rec["positions"][0]
    assert pos["symbol"] == "ETH/USDT:USDT"
    assert pos["direction"] == "short"
    assert pos["qty"] == 2.0
    assert pos["entry"] == 2000.0
    assert pos["mark"] == 1950.0
    assert pos["unrealized"] == 100.0
    assert pos["accrued_funding"] == 6.0
    assert pos["accrued_fees"] == 2.0
    assert "funding" in rec["notes"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pnl_attribution.py::test_build_cycle_pnl_record_shape_and_arithmetic -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.pnl_attribution'`

- [ ] **Step 3: Write minimal implementation**

```python
# futures_fund/pnl_attribution.py
"""Phase 9 — per-cycle cost/P&L attribution artifact (pnl.json) + cumulative ledger.jsonl.

build_cycle_pnl is the 'know all these data' record: opening_equity, the cumulative cost totals
(fees/slippage/funding), realized + unrealized P&L, gross/net P&L, closing_equity, turnover, and a
per-position list. In the accelerated live demo cross-tick PRICE PnL is small (each tick re-reads
current marks) so the curve mainly reflects FUNDING carry + fees — on-thesis for a carry desk; a
true multi-day price-PnL curve needs real daily cadence or PIT historical marks (see `notes`).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from futures_fund.account import PaperAccount
from futures_fund.models import Cadence

_NOTES = (
    "Accelerated demo: each tick re-reads current marks, so cross-tick PRICE PnL is small while "
    "FUNDING carry accrues per sim-day on held positions; the curve mainly reflects funding carry "
    "+ fees (on-thesis). A true multi-day price-PnL curve needs real daily cadence or PIT marks. "
    "Funding is non-zero only across runs that advance `now` past a settlement (every 8h default)."
)


def build_cycle_pnl(
    account: PaperAccount,
    *,
    opening_equity: float,
    marks: dict[str, float],
    turnover_usd: float,
    cycle: int,
    cadence: Cadence,
    now: datetime,
) -> dict:
    """The per-cycle pnl.json record (cumulative cost totals + this-cycle marks)."""
    upnl_by_sym = account.mark_to_market(marks)
    unrealized = sum(upnl_by_sym.values())
    funding_net = account.funding_received - account.funding_paid
    gross_pnl = account.realized_pnl + unrealized + funding_net
    net_pnl = gross_pnl - account.fees_paid - account.slippage_paid
    positions = [
        {
            "symbol": p.symbol,
            "direction": p.direction,
            "qty": p.qty,
            "entry": p.entry_price,
            "mark": marks.get(p.symbol),
            "unrealized": upnl_by_sym.get(p.symbol),
            "accrued_funding": p.accrued_funding,
            "accrued_fees": p.accrued_fees,
        }
        for p in account.positions.values()
    ]
    return {
        "ts": now.isoformat(),
        "cycle": cycle,
        "cadence": cadence,
        "opening_equity": opening_equity,
        "fees_paid": account.fees_paid,
        "slippage_paid": account.slippage_paid,
        "funding_received": account.funding_received,
        "funding_paid": account.funding_paid,
        "funding_net": funding_net,
        "realized_pnl": account.realized_pnl,
        "unrealized_pnl": unrealized,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "closing_equity": account.equity(marks),
        "turnover_usd": turnover_usd,
        "positions": positions,
        "notes": _NOTES,
    }


def append_ledger(state_dir, record: dict) -> None:
    """Append one pnl record to the cumulative state/ledger.jsonl (atomic full-rewrite)."""
    path = Path(state_dir) / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = path.read_text() if path.exists() else ""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(prior + json.dumps(record) + "\n")
    os.replace(tmp, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pnl_attribution.py::test_build_cycle_pnl_record_shape_and_arithmetic -v`
Expected: PASS

- [ ] **Step 5: Write the ledger-append test**

```python
# tests/test_pnl_attribution.py — append
import json

from futures_fund.pnl_attribution import append_ledger


def test_append_ledger_accumulates_lines(tmp_path):
    state = tmp_path / "state"
    append_ledger(state, {"cycle": 1, "net_pnl": 1.0})
    append_ledger(state, {"cycle": 2, "net_pnl": 2.0})
    lines = (state / "ledger.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cycle"] == 1
    assert json.loads(lines[1])["net_pnl"] == 2.0
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_pnl_attribution.py -v && uv run ruff check futures_fund/pnl_attribution.py tests/test_pnl_attribution.py`
Expected: all PASS, ruff clean

- [ ] **Step 7: Commit**

```bash
git add futures_fund/pnl_attribution.py tests/test_pnl_attribution.py
git commit -m "feat(pnl): per-cycle pnl.json record + cumulative ledger.jsonl

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Wire the ledger into `run_paper_cli._run_cadence` (the central change)

**Files:**
- Modify: `scripts/run_paper_cli.py`
- Test: `tests/test_account.py` (focused helper) + Task 8a (funding-vs-fill ordering) + Task 12 (integration) + Task 16 (E2E).

Flow inside `_run_cadence`, AFTER `_run_execute` (writes `report.json`) and BEFORE the equity record: load account (default cash = the passed `equity`) → read `geometries.json` → build marks/funding/intervals/costs → `prev_ts = account.last_funding_ts or now` (the per-account funding clock, NOT the cycle-collided equity series) → `opening_equity` (pre-fill) → `settle_funding` (advances `last_funding_ts` to `now`) → `apply_fills` of `report.json["executed"]` (RECONCILE to target → no double-count) → real `equity_now` → `record_equity` (replaces the hardcoded value) → `build_cycle_pnl` → `save_output("pnl")` → `append_ledger` → `save_account`. (Task 8b adds the journal cost-patch into this same seam.)

**Funding-vs-fill ordering (load-bearing):** `settle_funding` is called BEFORE `apply_fills` so a position OPENED this cycle does not earn funding for a window it did not exist in — Task 8a pins this invariant.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_account.py — append
from scripts.run_paper_cli import _geometry_cost_maps


def test_geometry_cost_maps_from_bundle():
    bundle = {"geometries": [
        {"symbol": "ETH/USDT:USDT", "mark": 2000.0, "funding_rate": 0.0005,
         "funding_interval_hours": 8.0, "adv_usd": 5_000_000.0, "beta_btc": 1.0,
         "momentum_20": 0.0, "realized_vol": 0.0, "sentiment_score": 0.0,
         "sentiment_conf": 0.0},
    ]}
    marks, funding, intervals, costs = _geometry_cost_maps(bundle)
    assert marks["ETH/USDT:USDT"] == 2000.0
    assert funding["ETH/USDT:USDT"] == 0.0005
    assert intervals["ETH/USDT:USDT"] == 8
    assert costs["ETH/USDT:USDT"].adv_usd == 5_000_000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py::test_geometry_cost_maps_from_bundle -v`
Expected: FAIL with `ImportError: cannot import name '_geometry_cost_maps'`

- [ ] **Step 3: Write minimal implementation**

Extend the `from futures_fund...` import block of `scripts/run_paper_cli.py`:

```python
from futures_fund.account import CostInputs, load_account, save_account
from futures_fund.pnl_attribution import append_ledger, build_cycle_pnl
```

Add the pure helpers near `_read_executed`:

```python
def _geometry_cost_maps(bundle: dict) -> tuple[dict, dict, dict, dict]:
    """From a geometries.json bundle build (marks, funding_by_symbol, intervals, costs).

    marks/funding/interval come straight off each CoinGeometry; costs is a CostInputs carrier (ADV
    + a 1bps half-spread default) so the paper fill uses the slippage fallback (never flat)."""
    marks: dict[str, float] = {}
    funding: dict[str, float] = {}
    intervals: dict[str, int] = {}
    costs: dict[str, CostInputs] = {}
    for g in bundle.get("geometries", []):
        sym = g.get("symbol")
        mark = g.get("mark")
        if not sym or mark is None:
            continue
        marks[sym] = float(mark)
        funding[sym] = float(g.get("funding_rate", 0.0))
        intervals[sym] = int(g.get("funding_interval_hours", 8) or 8)
        costs[sym] = CostInputs(adv_usd=float(g.get("adv_usd", 0.0)))
    return marks, funding, intervals, costs


def _load_geometries(state_dir, cadence: Cadence, cycle: int) -> dict:
    """Best-effort read of this cycle's geometries.json (marks + funding + ADV)."""
    try:
        return load_output(state_dir, cycle, "geometries", cadence=cadence)
    except FileNotFoundError:
        return {"geometries": []}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py::test_geometry_cost_maps_from_bundle -v`
Expected: PASS

- [ ] **Step 5: Replace the equity record seam in `_run_cadence`**

Find this block in `_run_cadence`:

```python
    # Step 7a — equity point (the dashboard's return-series source) + reflect.
    equity_log.record_equity(state_dir, now, equity, cycle)
    _run_reflect(state_dir, cadence, cycle, memory_dir)
    return True
```

Replace it with:

```python
    # Step 7a — REALISTIC P&L: load the account, settle funding since the account's OWN funding
    # clock (NOT the cycle-collided equity series), reconcile THIS cycle's executed book to target
    # (weekly re-emits the full book -> delta 0 on unchanged legs; daily nudges the changed legs —
    # no double-count, convergent across weeks), mark-to-market, record the REAL equity (replaces the
    # old flat settings.account_size_usdt), write pnl.json + ledger, save the account. settle_funding
    # runs BEFORE apply_fills so a position opened this cycle earns no funding for a pre-existence
    # window (Task 8a pins this).
    account = load_account(state_dir, equity)            # `equity` is the default cash on a cold dir
    bundle = _load_geometries(state_dir, cadence, cycle)
    marks, funding_by_symbol, intervals, costs = _geometry_cost_maps(bundle)
    prev_ts = account.last_funding_ts or now             # the per-account funding clock
    opening_equity = account.equity(marks)
    account.settle_funding(prev_ts, now, funding_by_symbol, intervals, marks)
    executed = _read_executed(state_dir, cadence, cycle)
    account.apply_fills(executed, marks, costs, opened_ts=now)
    turnover = sum(abs(float(t.get("target_notional", 0.0))) for t in executed)
    equity_now = account.equity(marks)
    equity_log.record_equity(state_dir, now, equity_now, cycle)
    rec = build_cycle_pnl(
        account, opening_equity=opening_equity, marks=marks, turnover_usd=turnover,
        cycle=cycle, cadence=cadence, now=now)
    save_output(state_dir, cycle, "pnl", rec, cadence=cadence)
    append_ledger(state_dir, rec)
    save_account(state_dir, account)
    _run_reflect(state_dir, cadence, cycle, memory_dir)
    return True
```

The `equity` param of `_run_cadence` (from `settings.account_size_usdt` in `main`) is now the cold-start default cash — leave `main` passing it unchanged. Line 231 `equity = settings.account_size_usdt` in `main` remains as the default-cash source.

> Note `turnover` here is the SUM of executed leg `target_notional` (the intended book size this cycle), used as a coarse turnover proxy for the cost-drag KPI; it is NOT the |delta| actually filled. This is intentional and documented — turnover-as-book-size is the headline figure the dashboard shows; the precise traded |delta| is captured in `fees_paid`/`slippage_paid`.

- [ ] **Step 6: Run the helper test + the no-seed E2E to confirm no regression**

Run: `uv run pytest tests/test_account.py::test_geometry_cost_maps_from_bundle tests/test_end_to_end_no_seed.py -v`
Expected: PASS

- [ ] **Step 7: ruff + commit**

Run: `uv run ruff check scripts/run_paper_cli.py`
Expected: `All checks passed!`

```bash
git add scripts/run_paper_cli.py tests/test_account.py
git commit -m "feat(run): wire PaperAccount ledger into _run_cadence (real equity + pnl.json, no double-count)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8a: Pin the funding-vs-fill ordering + the multi-week no-double-count, ledger-level

**Files:**
- Create/extend: `tests/test_account_integration.py` (or append to `tests/test_account.py`)

Two invariants the wired seam relies on but does not itself assert: (a) a position opened in the SAME cycle as a `settle_funding` call earns 0 funding for that cycle (settle-before-fill); (b) running weekly cycle 1's full book then weekly cycle 2's IDENTICAL full book does NOT double qty.

- [ ] **Step 1: Write the failing/passing test**

```python
# tests/test_account_integration.py — create (or append if it exists from Task 12 ordering)
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_fund.account import CostInputs, PaperAccount


def test_position_opened_this_cycle_earns_zero_funding_this_cycle():
    """Settle-BEFORE-fill: a leg opened in the same cycle as settle_funding earns no funding for
    that cycle's window (it did not exist for the pre-existence span)."""
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=0.0)}
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 11, 0, 0, tzinfo=UTC)  # 24h -> 3 settlements IF the leg existed
    # ORDER MATTERS: settle first (no positions yet -> 0), THEN open.
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, {})
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "short", "target_notional": 4000.0}],
        {"ETH/USDT:USDT": 2000.0}, costs, opened_ts=now)
    assert acct.funding_received == 0.0            # opened AFTER settle -> no funding this cycle
    assert acct.positions["ETH/USDT:USDT"].accrued_funding == 0.0


def test_weekly_cycle2_resend_does_not_double_qty():
    """The multi-week double-count guard at the book level: re-applying the IDENTICAL full weekly
    book on cycle 2 reconciles to delta 0 -> qty unchanged, frictions unchanged."""
    acct = PaperAccount(cash=20_000.0)
    costs = {
        "ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0),
        "BTC/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0),
    }
    marks = {"ETH/USDT:USDT": 2000.0, "BTC/USDT:USDT": 60_000.0}
    book = [
        {"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 4000.0},
        {"symbol": "BTC/USDT:USDT", "direction": "short", "target_notional": 6000.0},
    ]
    acct.apply_fills(book, marks, costs, opened_ts=datetime(2026, 6, 10, tzinfo=UTC))
    qty1 = {s: p.qty for s, p in acct.positions.items()}
    fees1, slip1 = acct.fees_paid, acct.slippage_paid
    # weekly cycle 2: the SAME full book again
    acct.apply_fills(book, marks, costs, opened_ts=datetime(2026, 6, 17, tzinfo=UTC))
    qty2 = {s: p.qty for s, p in acct.positions.items()}
    assert qty2 == qty1                            # NOT doubled to ~2x notional
    assert acct.fees_paid == fees1                 # 0 extra fee on the re-send
    assert acct.slippage_paid == slip1             # 0 extra slippage on the re-send
```

- [ ] **Step 2: Run (exercises Tasks 3-5 — should PASS)**

Run: `uv run pytest tests/test_account_integration.py -k "earns_zero_funding or cycle2_resend" -v`
Expected: PASS. A FAIL pinpoints an ordering/reconcile bug — fix with superpowers:systematic-debugging before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_account_integration.py
git commit -m "test(account): pin settle-before-fill + multi-week no-double-count invariants

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8b: Patch realized per-leg costs onto the journal at close (so reflect_cli is not inert)

**Files:**
- Modify: `scripts/run_paper_cli.py`
- Test: `tests/test_account.py` (helper) + `tests/test_reflect_promote_cli.py` (round-trip, in Task 10)

`reflect_cli._cost_fields` (Task 10) reads `fees/slippage/realized_pnl/realized_funding` off journal `Decision` dicts, but NOTHING currently patches those onto the journal — so without this step `_cost_fields` returns all-0.0 and the Reflector's net-alpha keying is cosmetic. After the fills are applied (in the Step-7a seam), for each position the account now holds (or just realized on), patch its realized per-leg `fees`/`slippage`/`realized_funding`/`realized_pnl` onto the matching journal `Decision` via `journal.patch_outcome` (keyed on `(cycle, symbol, direction)`; `Decision(extra="allow")` round-trips `realized_funding`). `patch_outcome` returns `False` when no decision matches that key (a leg the journal never recorded — fail-soft, no raise).

- [ ] **Step 1: Write the failing test (pure helper)**

```python
# tests/test_account.py — append
from scripts.run_paper_cli import _leg_cost_patches


def test_leg_cost_patches_from_account():
    acct = PaperAccount(cash=20_000.0)
    p = _pos(symbol="ETH/USDT:USDT", direction="short", qty=2.0, entry=2000.0)
    p.accrued_fees = 4.0
    p.accrued_slippage = 2.0
    p.accrued_funding = 6.0
    p.realized_pnl = 12.0
    acct.positions["ETH/USDT:USDT"] = p
    patches = _leg_cost_patches(acct, cycle=1)
    assert patches == [
        ("ETH/USDT:USDT", "short",
         {"fees": 4.0, "slippage": 2.0, "realized_funding": 6.0, "realized_pnl": 12.0}),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py::test_leg_cost_patches_from_account -v`
Expected: FAIL with `ImportError: cannot import name '_leg_cost_patches'`

- [ ] **Step 3: Write minimal implementation in `scripts/run_paper_cli.py`**

Add the import:

```python
from futures_fund.journal import patch_outcome
```

Add the pure helper near `_geometry_cost_maps`:

```python
def _leg_cost_patches(account, cycle: int) -> list[tuple[str, str, dict]]:
    """Per-held-leg realized cost patches for the journal (Reflector net-alpha keying source).

    Returns (symbol, direction, outcome_dict) tuples carrying the leg's realized fees/slippage/
    funding/price-pnl. `cycle` is accepted for symmetry with the journal key (the caller pairs it
    with the cycle the leg was opened in); patches are matched on (cycle, symbol, direction)."""
    out: list[tuple[str, str, dict]] = []
    for pos in account.positions.values():
        out.append((
            pos.symbol, pos.direction,
            {
                "fees": pos.accrued_fees,
                "slippage": pos.accrued_slippage,
                "realized_funding": pos.accrued_funding,
                "realized_pnl": pos.realized_pnl,
            },
        ))
    return out
```

In the Step-7a seam (inside `_run_cadence`, after `save_account(state_dir, account)` and before `_run_reflect`), add:

```python
    # Patch each held leg's realized fees/slippage/funding/price-pnl onto the journal Decision so the
    # Reflector keys lessons on NET (after-cost) alpha. Decision is extra="allow", so realized_funding
    # round-trips. patch_outcome returns False (fail-soft) for a leg the journal never recorded.
    for sym, direction, outcome in _leg_cost_patches(account, cycle):
        try:
            patch_outcome(memory_dir, cycle=cycle, symbol=sym, direction=direction, outcome=outcome)
        except Exception as exc:  # noqa: BLE001 — cost bookkeeping must not unwind an executed cycle
            print(f"WARNING: journal cost-patch failed for {sym} {direction}: {exc!r}",
                  file=sys.stderr)
```

> Scope note: this patches the OPEN cycle's journal record with the leg's cumulative realized costs/funding so far. It is a best-effort enrichment for reflection; it is NOT the alpha-outcome close (the six alpha-vs-beta fields are patched elsewhere by the reflect/close path). Because `Decision(extra="allow")`, these keys coexist with the alpha-outcome fields without a schema bump.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py::test_leg_cost_patches_from_account -v`
Expected: PASS

- [ ] **Step 5: ruff + commit**

Run: `uv run ruff check scripts/run_paper_cli.py`
Expected: `All checks passed!`

```bash
git add scripts/run_paper_cli.py tests/test_account.py
git commit -m "feat(run): patch realized per-leg costs onto the journal at close (feeds reflect_cli)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Cost-block artifact for `context.json` (consumed by the EXTERNAL orchestrator, not wired in Python)

**Files:**
- Modify: `scripts/preflight.py`
- Create: `tests/fixtures/pnl_block.json`
- Test: `tests/test_agent_cost_context.py`

> **Scope (honest):** `context.json` is produced by `scripts/preflight.py` but is NOT read by any Python in the run pipeline — `run_paper_cli`/`_run_cadence` never invoke `preflight.main`. It is consumed by the external SKILL.md / Claude-Code orchestrator that assembles agent prompts. This task makes `preflight.py` FOLD a realized `pnl` block into `context.json` so that artifact carries cost/carry data; it does NOT wire a Python prompt-injection path (there is none in this repo). The tests prove the artifact's SHAPE and VALUES, not that an agent reads it.

`context.json` gets a top-level `"pnl"` block + per-symbol realized-cost on each brief, sourced from the persisted account + the latest `pnl.json`. The block: per-symbol realized funding carry (signed `accrued_funding`), current unrealized PnL, accrued fees, and the last rebalance's cost (fees+slippage) vs turnover. A `pnl_block.json` fixture pins the contract.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cost_context.py
from __future__ import annotations

import json
from datetime import UTC, datetime

from futures_fund.account import PaperAccount, Position, save_account
from scripts.preflight import build_pnl_block


def _seed_account(state):
    acct = PaperAccount(cash=20_010.0)
    acct.positions["ETH/USDT:USDT"] = Position(
        symbol="ETH/USDT:USDT", direction="short", qty=2.0, entry_price=2000.0,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC),
        accrued_funding=6.0, accrued_fees=2.0, accrued_slippage=1.0)
    acct.fees_paid = 4.0
    acct.slippage_paid = 2.0
    acct.funding_received = 6.0
    save_account(state, acct)
    return acct


def test_build_pnl_block_is_populated_from_the_account(tmp_path):
    state = tmp_path / "state"
    _seed_account(state)
    marks = {"ETH/USDT:USDT": 1950.0}              # short upnl = 100
    last_pnl = {"fees_paid": 4.0, "slippage_paid": 2.0, "turnover_usd": 4000.0}

    block = build_pnl_block(state, marks=marks, last_pnl=last_pnl, default_cash=20_000.0)

    assert block["equity"] == 20_010.0 + 100.0
    assert block["total_fees"] == 4.0
    assert block["total_slippage"] == 2.0
    assert block["total_funding_received"] == 6.0
    per = block["by_symbol"]["ETH/USDT:USDT"]
    assert per["unrealized"] == 100.0
    assert per["realized_funding"] == 6.0          # signed accrued_funding (+ = received)
    assert per["accrued_fees"] == 2.0
    assert block["last_rebalance_cost"] == 6.0     # fees 4 + slippage 2
    assert block["last_rebalance_turnover_usd"] == 4000.0


def test_pnl_block_fixture_contract_matches():
    fixture = json.loads(open("tests/fixtures/pnl_block.json").read())
    for key in ("equity", "total_fees", "total_slippage", "total_funding_received",
                "total_funding_paid", "last_rebalance_cost", "last_rebalance_turnover_usd",
                "by_symbol"):
        assert key in fixture
    for key in ("unrealized", "realized_funding", "accrued_fees"):
        assert key in next(iter(fixture["by_symbol"].values()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_cost_context.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_pnl_block'`

- [ ] **Step 3: Write the fixture**

```json
// tests/fixtures/pnl_block.json
{
  "equity": 20110.0,
  "total_fees": 4.0,
  "total_slippage": 2.0,
  "total_funding_received": 6.0,
  "total_funding_paid": 0.0,
  "last_rebalance_cost": 6.0,
  "last_rebalance_turnover_usd": 4000.0,
  "by_symbol": {
    "ETH/USDT:USDT": {
      "unrealized": 100.0,
      "realized_funding": 6.0,
      "accrued_fees": 2.0
    }
  }
}
```

- [ ] **Step 4: Write minimal implementation in `scripts/preflight.py`**

Add imports:

```python
from futures_fund.account import load_account
from futures_fund.config import load_settings
```

Add `build_pnl_block` before `main`:

```python
def build_pnl_block(state_dir, *, marks: dict[str, float], last_pnl: dict,
                    default_cash: float) -> dict:
    """The realized cost/carry/PnL block folded into context.json (an ARTIFACT the external SKILL.md
    orchestrator reads when assembling prompts — there is no Python prompt-injection path here).

    Per-symbol realized funding carry (signed accrued_funding, + = received), current unrealized
    PnL, and accrued fees; plus the last rebalance's cost (fees+slippage) vs its turnover so the
    trader can weigh round-trip cost against the spread edge before churning a pair."""
    acct = load_account(state_dir, default_cash)
    upnl = acct.mark_to_market(marks)
    by_symbol = {
        sym: {
            "unrealized": upnl.get(sym, 0.0),
            "realized_funding": pos.accrued_funding,
            "accrued_fees": pos.accrued_fees,
        }
        for sym, pos in acct.positions.items()
    }
    return {
        "equity": acct.equity(marks),
        "total_fees": acct.fees_paid,
        "total_slippage": acct.slippage_paid,
        "total_funding_received": acct.funding_received,
        "total_funding_paid": acct.funding_paid,
        "last_rebalance_cost": float(last_pnl.get("fees_paid", 0.0))
        + float(last_pnl.get("slippage_paid", 0.0)),
        "last_rebalance_turnover_usd": float(last_pnl.get("turnover_usd", 0.0)),
        "by_symbol": by_symbol,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_cost_context.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Fold `build_pnl_block` into `preflight.main`**

In `scripts/preflight.py`, replace the `ctx = {"briefs": ..., "held": held}` / `save_output` lines in `main` with the block below. `load_settings()` is SAFE with no `config.yaml` (verified: it returns `account_size_usdt=20000.0`), but we still pass it explicitly so a cold env cannot raise:

```python
    # Fold the realized cost/carry/PnL block (Phase 9) so the artifact context.json carries cost
    # data for the external orchestrator. load_settings() is safe with no config.yaml in cwd
    # (account_size_usdt defaults to 20000); default_cash is passed explicitly regardless.
    marks: dict[str, float] = {}
    try:
        bundle = load_output(args.state_dir, args.cycle, "geometries", cadence=cadence)
        marks = {g["symbol"]: float(g["mark"]) for g in bundle.get("geometries", [])
                 if g.get("symbol") and g.get("mark") is not None}
    except FileNotFoundError:
        marks = {}
    try:
        last_pnl = load_output(args.state_dir, args.cycle, "pnl", cadence=cadence)
    except FileNotFoundError:
        last_pnl = {}
    settings = load_settings()
    pnl_block = build_pnl_block(
        args.state_dir, marks=marks, last_pnl=last_pnl,
        default_cash=settings.account_size_usdt)
    briefs = build_briefs(universe, held)
    for b in briefs:
        per = pnl_block["by_symbol"].get(b["symbol"])
        if per is not None:
            b["pnl"] = per                          # per-symbol realized cost/carry on the brief
    ctx = {"briefs": briefs, "held": held, "pnl": pnl_block}
    save_output(args.state_dir, args.cycle, "context", ctx, cadence=cadence)
    print(json.dumps(ctx, indent=2, default=str))
```

- [ ] **Step 7: Run the preflight + cost-context suites + ruff**

Run: `uv run pytest tests/test_preflight_cli.py tests/test_agent_cost_context.py -v && uv run ruff check scripts/preflight.py tests/test_agent_cost_context.py`
Expected: PASS, ruff clean. If `test_preflight_cli.py` asserts the exact `ctx` dict, relax it to assert `ctx["briefs"]` / `ctx["held"]` (the new `"pnl"` key is additive).

- [ ] **Step 8: Commit**

```bash
git add scripts/preflight.py tests/test_agent_cost_context.py tests/fixtures/pnl_block.json
git commit -m "feat(context): fold realized cost/carry/PnL artifact block into context.json

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: `reflect_cli` — per-closed-leg realized funding/fees/slippage/net (fed by Task 8b)

**Files:**
- Modify: `scripts/reflect_cli.py`
- Test: `tests/test_reflect_promote_cli.py` (extend)

The Reflector keys lessons on net (after-cost) alpha. Add `realized_funding`, `fees`, `slippage`, `net_pnl` per winner/loser entry, read from journal `Decision` fields (`fees`, `slippage`, `realized_pnl`) and the `realized_funding` key Task 8b patches on (`Decision(extra="allow")` round-trips it). Absent fields default to `0.0`. **The loop variable in `build_reflection_input` is literally `d`** (`reflect_cli.py` line 35 `for d in read_all_decisions(memory_dir):`) — use exactly `d`, not `dec`/`rec`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reflect_promote_cli.py — append
def test_reflection_entries_carry_realized_costs():
    from scripts.reflect_cli import _cost_fields

    decision = {
        "fees": 3.0, "slippage": 1.5, "funding_paid": -2.0,
        "realized_funding": 2.0, "realized_pnl": 12.0,
    }
    costs = _cost_fields(decision)
    assert costs["fees"] == 3.0
    assert costs["slippage"] == 1.5
    assert costs["realized_funding"] == 2.0
    assert costs["net_pnl"] == 12.0 - 3.0 - 1.5    # realized_pnl net of fees+slippage


def test_cost_fields_default_zero_on_missing():
    from scripts.reflect_cli import _cost_fields
    costs = _cost_fields({})
    assert costs == {"fees": 0.0, "slippage": 0.0, "realized_funding": 0.0, "net_pnl": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reflect_promote_cli.py -k "realized_costs or cost_fields_default" -v`
Expected: FAIL with `ImportError: cannot import name '_cost_fields'`

- [ ] **Step 3: Write minimal implementation in `scripts/reflect_cli.py`**

Add `_cost_fields` near `build_reflection_input`:

```python
def _cost_fields(decision: dict) -> dict:
    """Per-closed-leg realized costs for the Reflector (net-of-cost alpha keying).

    fees/slippage are >= 0; realized_funding is signed (+ = received). net_pnl = realized_pnl minus
    fees + slippage. All default to 0.0 on a journal record that predates the cost engine. These
    fields are populated by run_paper_cli's journal cost-patch (Task 8b)."""
    fees = float(decision.get("fees") or 0.0)
    slippage = float(decision.get("slippage") or 0.0)
    realized_funding = float(decision.get("realized_funding") or 0.0)
    realized_pnl = float(decision.get("realized_pnl") or 0.0)
    return {
        "fees": fees,
        "slippage": slippage,
        "realized_funding": realized_funding,
        "net_pnl": realized_pnl - fees - slippage,
    }
```

In `build_reflection_input`, immediately after each `entry = {...}` is assembled (before the winners/losers append on line 56), merge the cost fields using the journal-record dict in scope — the loop variable is exactly `d`:

```python
        entry.update(_cost_fields(d))           # `d` is the journal decision dict (reflect_cli line 35)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_reflect_promote_cli.py -k "realized_costs or cost_fields_default" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full reflect-cli suite + ruff**

Run: `uv run pytest tests/test_reflect_promote_cli.py -v && uv run ruff check scripts/reflect_cli.py`
Expected: all PASS, ruff clean

- [ ] **Step 6: Commit**

```bash
git add scripts/reflect_cli.py tests/test_reflect_promote_cli.py
git commit -m "feat(reflect): per-closed-leg realized funding/fees/slippage/net in reflection_input

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: `build_scorecard` cost-transparency keys (artifact the orchestrator injects)

**Files:**
- Modify: `futures_fund/scorecard.py`
- Test: `tests/test_agent_cost_context.py` (extend)

> **Scope (honest):** `build_scorecard` is imported/called only by `scripts/promote_lesson_cli.py` (reads `dsr_pvalue`); it is NOT injected by any Python in this repo. The module docstring says it is "injected into every agent prompt" — that injection is performed by the external SKILL.md orchestrator, not Python. This task adds cost keys to the returned dict so that artifact carries cost data; the test proves the keys/values, not that an agent reads them.

Add cost keys read from the latest `pnl.json` (across cadences): `net_pnl`, `gross_pnl`, `total_fees`, `total_slippage`, `funding_net`, `cost_drag_bps` (= `(fees+slippage) / max(|gross|, 1) * 1e4`, or `nan` on no data).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cost_context.py — append
from futures_fund.cycle_io import save_output
from futures_fund.scorecard import _latest_pnl, build_scorecard


def test_latest_pnl_reads_the_newest_cycle(tmp_path):
    state = tmp_path / "state"
    save_output(state, 1, "pnl", {"net_pnl": 1.0, "cycle": 1,
                                  "ts": "2026-06-10T00:00:00+00:00"}, cadence="weekly")
    save_output(state, 2, "pnl", {"net_pnl": 5.0, "cycle": 2,
                                  "ts": "2026-06-11T00:00:00+00:00"}, cadence="daily")
    rec = _latest_pnl(state)
    assert rec["net_pnl"] == 5.0                    # newest ts wins


def test_scorecard_carries_cost_transparency_keys(tmp_path):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    save_output(state, 1, "pnl", {
        "net_pnl": 8.0, "gross_pnl": 14.0, "fees_paid": 4.0, "slippage_paid": 2.0,
        "funding_net": 6.0, "cycle": 1, "ts": "2026-06-10T00:00:00+00:00"}, cadence="weekly")
    sc = build_scorecard(state, memory)
    assert sc["net_pnl"] == 8.0
    assert sc["gross_pnl"] == 14.0
    assert sc["total_fees"] == 4.0
    assert sc["total_slippage"] == 2.0
    assert sc["funding_net"] == 6.0
    assert abs(sc["cost_drag_bps"] - (6.0 / 14.0 * 1e4)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_cost_context.py -k "latest_pnl or cost_transparency" -v`
Expected: FAIL with `ImportError: cannot import name '_latest_pnl'`

- [ ] **Step 3: Write minimal implementation in `futures_fund/scorecard.py`**

Extend the top imports:

```python
import json

from futures_fund.control_loop import latest_cadence_cycle
from futures_fund.cycle_io import cycle_dir
```

Add the reader:

```python
def _latest_pnl(state_dir) -> dict:
    """The newest pnl.json across both cadences (by recorded ts, else by cycle number)."""
    best: dict = {}
    best_key: tuple = ("", -1)
    for cadence in ("weekly", "daily"):
        n = latest_cadence_cycle(state_dir, cadence, "pnl")
        if n is None:
            continue
        path = cycle_dir(state_dir, n, cadence=cadence) / "pnl.json"
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        key = (str(rec.get("ts") or ""), int(rec.get("cycle") or 0))
        if key > best_key:
            best, best_key = rec, key
    return best
```

In `build_scorecard`, just above `return {`:

```python
    pnl = _latest_pnl(state_dir)
    gross = float(pnl.get("gross_pnl", 0.0))
    fees = float(pnl.get("fees_paid", 0.0))
    slip = float(pnl.get("slippage_paid", 0.0))
    cost_drag_bps = (fees + slip) / max(abs(gross), 1.0) * 1e4 if pnl else float("nan")
```

Add these keys inside the returned dict (after `"corpus": corpus_health(memory_dir),`):

```python
        "net_pnl": float(pnl.get("net_pnl", 0.0)) if pnl else float("nan"),
        "gross_pnl": gross if pnl else float("nan"),
        "total_fees": fees,
        "total_slippage": slip,
        "funding_net": float(pnl.get("funding_net", 0.0)) if pnl else float("nan"),
        "cost_drag_bps": cost_drag_bps,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_cost_context.py -k "latest_pnl or cost_transparency" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the scorecard tests + ruff**

Run: `uv run pytest tests/ -k scorecard -v ; uv run ruff check futures_fund/scorecard.py`
Expected: PASS, ruff clean

- [ ] **Step 6: Commit**

```bash
git add futures_fund/scorecard.py tests/test_agent_cost_context.py
git commit -m "feat(scorecard): add net_pnl/fees/slippage/funding_net/cost_drag cost keys

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Integration test — equity is no longer constant 20000 across ≥2 cycles

**Files:**
- Extend: `tests/test_account_integration.py` (created in Task 8a)

Prove across ≥2 cycles, ledger-level: settle funding on a held short over a sim-day, reconcile a cycle's fills, and assert recorded equity moves off 20000 and `pnl.json` carries non-zero funding/fees.

- [ ] **Step 1: Write the failing/passing test**

```python
# tests/test_account_integration.py — append
from futures_fund.pnl_attribution import build_cycle_pnl


def test_two_cycle_equity_moves_off_constant_with_funding_and_fees():
    acct = PaperAccount(cash=20_000.0)
    costs = {"ETH/USDT:USDT": CostInputs(adv_usd=5_000_000.0, half_spread_bps=1.0)}
    t0 = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)

    # cycle 1: settle (no positions -> 0), then open a short carry leg
    marks1 = {"ETH/USDT:USDT": 2000.0}
    opening1 = acct.equity(marks1)
    acct.settle_funding(t0, t0, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, marks1)
    acct.apply_fills(
        [{"symbol": "ETH/USDT:USDT", "direction": "short", "target_notional": 4000.0}],
        marks1, costs, opened_ts=t0)
    rec1 = build_cycle_pnl(acct, opening_equity=opening1, marks=marks1,
                           turnover_usd=4000.0, cycle=1, cadence="weekly", now=t0)
    assert rec1["closing_equity"] != 20_000.0      # frictions moved equity off the constant
    assert rec1["fees_paid"] > 0.0
    assert rec1["slippage_paid"] > 0.0
    assert rec1["funding_received"] == 0.0         # leg opened AFTER cycle-1 settle

    # cycle 2 (one sim-day later): settle funding (3 events at 8h) from the account clock, re-mark
    t1 = t0 + timedelta(days=1)
    marks2 = {"ETH/USDT:USDT": 2000.0}
    opening2 = acct.equity(marks2)
    acct.settle_funding(acct.last_funding_ts, t1, {"ETH/USDT:USDT": 0.0005},
                        {"ETH/USDT:USDT": 8}, marks2)
    rec2 = build_cycle_pnl(acct, opening_equity=opening2, marks=marks2,
                           turnover_usd=0.0, cycle=2, cadence="daily", now=t1)

    assert rec2["funding_received"] > 0.0          # short + positive rate = received over the day
    assert rec2["funding_net"] > 0.0
    equities = [rec1["closing_equity"], rec2["closing_equity"]]
    assert len(set(equities)) == 2                 # not a flat 20000
    assert all(e != 20_000.0 for e in equities)
    assert rec2["net_pnl"] == rec2["gross_pnl"] - rec2["fees_paid"] - rec2["slippage_paid"]
```

- [ ] **Step 2: Run test (it exercises Tasks 1-7 — should PASS)**

Run: `uv run pytest tests/test_account_integration.py -v`
Expected: PASS. A FAIL pinpoints a ledger bug to fix (use superpowers:systematic-debugging) before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_account_integration.py
git commit -m "test(account): >=2-cycle integration — equity moves off 20000 with funding+fees

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Dashboard — cost rows in `build_kpi_dashboard` + `dashboard_cli`

**Files:**
- Modify: `futures_fund/dashboard.py`
- Modify: `scripts/dashboard_cli.py`
- Test: `tests/test_dashboard.py` (extend)

Add `gross_pnl/net_pnl/total_fees/total_slippage/total_funding/cost_drag_bps` (read from the latest `pnl.json` via `scorecard._latest_pnl`) and add the display rows to `_ROWS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py — append
from futures_fund.cycle_io import save_output
from futures_fund.dashboard import build_kpi_dashboard


def test_dashboard_carries_cost_rows(tmp_path):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    save_output(state, 1, "pnl", {
        "net_pnl": 8.0, "gross_pnl": 14.0, "fees_paid": 4.0, "slippage_paid": 2.0,
        "funding_net": 6.0, "cycle": 1, "ts": "2026-06-10T00:00:00+00:00"}, cadence="weekly")
    dash = build_kpi_dashboard(state, memory)
    assert dash["net_pnl"] == 8.0
    assert dash["gross_pnl"] == 14.0
    assert dash["total_fees"] == 4.0
    assert dash["total_slippage"] == 2.0
    assert dash["total_funding"] == 6.0
    assert abs(dash["cost_drag_bps"] - (6.0 / 14.0 * 1e4)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py::test_dashboard_carries_cost_rows -v`
Expected: FAIL with `KeyError: 'net_pnl'`

- [ ] **Step 3: Write minimal implementation in `futures_fund/dashboard.py`**

Add the import:

```python
from futures_fund.scorecard import _latest_pnl
```

In `build_kpi_dashboard`, before `return {`:

```python
    pnl = _latest_pnl(state_dir)
    gross = float(pnl.get("gross_pnl", 0.0))
    fees = float(pnl.get("fees_paid", 0.0))
    slip = float(pnl.get("slippage_paid", 0.0))
    cost_drag_bps = (fees + slip) / max(abs(gross), 1.0) * 1e4 if pnl else float("nan")
```

Add inside the returned dict (after `"reviewer_veto_rate": reviewer_veto_rate(...)`):

```python
        # cost-transparency KPIs (Phase 9 — read off the latest pnl.json)
        "gross_pnl": gross if pnl else float("nan"),
        "net_pnl": float(pnl.get("net_pnl", 0.0)) if pnl else float("nan"),
        "total_fees": fees,
        "total_slippage": slip,
        "total_funding": float(pnl.get("funding_net", 0.0)) if pnl else float("nan"),
        "cost_drag_bps": cost_drag_bps,
```

> Import note: `dashboard.py` importing `scorecard._latest_pnl` is acyclic — `scorecard.py` does NOT import `dashboard.py`. Confirm no cycle with `uv run python -c "import futures_fund.dashboard"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py::test_dashboard_carries_cost_rows -v`
Expected: PASS

- [ ] **Step 5: Add the display rows to `scripts/dashboard_cli.py`**

Append to the `_ROWS` tuple (after `("reviewer_veto_rate", "Reviewer veto-rate"),`):

```python
    ("net_pnl", "Net P&L (after costs)"),
    ("gross_pnl", "Gross P&L (incl. carry)"),
    ("total_fees", "Fees paid (cumulative)"),
    ("total_slippage", "Slippage cost (cumulative)"),
    ("total_funding", "Funding (signed net, + = received)"),
    ("cost_drag_bps", "Cost drag (bps of gross)"),
```

- [ ] **Step 6: Run the dashboard suite + ruff**

Run: `uv run pytest tests/test_dashboard.py -v && uv run ruff check futures_fund/dashboard.py scripts/dashboard_cli.py`
Expected: all PASS, ruff clean

- [ ] **Step 7: Commit**

```bash
git add futures_fund/dashboard.py scripts/dashboard_cli.py tests/test_dashboard.py
git commit -m "feat(dashboard): fees/slippage/funding/net-PnL + cost-drag rows

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Reviewer / self-audit account invariants

**Files:**
- Modify: `futures_fund/self_audit.py`
- Test: `tests/test_account.py` (extend)

Two invariants: (a) recorded equity == cash + unrealized within tolerance; (b) recorded per-cycle funding equals a recompute via `realized_funding` × `count_funding_events` (extends `invariant_funding_sign_correct` to the account level).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_account.py — append
from futures_fund.self_audit import (
    invariant_account_equity_reconciles,
    invariant_cycle_funding_reconciles,
)


def test_invariant_account_equity_reconciles():
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="short", qty=2.0, entry=2000.0)
    marks = {"ETH/USDT:USDT": 1950.0}              # upnl = 100
    recorded = acct.equity(marks)                  # 20100
    assert invariant_account_equity_reconciles(acct, marks, recorded)
    assert not invariant_account_equity_reconciles(acct, marks, recorded + 5.0)


def test_invariant_cycle_funding_reconciles():
    prev = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 6, 11, 0, 0, tzinfo=UTC)
    acct = PaperAccount(cash=20_000.0)
    acct.positions["ETH/USDT:USDT"] = _pos(direction="short", qty=2.0, entry=2000.0)
    marks = {"ETH/USDT:USDT": 2000.0}
    acct.settle_funding(prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, marks)
    recorded = acct.positions["ETH/USDT:USDT"].accrued_funding   # +6.0
    assert invariant_cycle_funding_reconciles(
        acct, prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, marks, recorded)
    assert not invariant_cycle_funding_reconciles(
        acct, prev, now, {"ETH/USDT:USDT": 0.0005}, {"ETH/USDT:USDT": 8}, marks, recorded + 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_account.py -k "equity_reconciles or funding_reconciles" -v`
Expected: FAIL with `ImportError: cannot import name 'invariant_account_equity_reconciles'`

- [ ] **Step 3: Write minimal implementation in `futures_fund/self_audit.py`**

Extend imports (add the missing names; replace the existing `from futures_fund.funding_intervals import realized_funding` line with the combined import):

```python
from datetime import UTC, datetime

from futures_fund.account import PaperAccount, Position
from futures_fund.costs import count_funding_events
from futures_fund.funding_intervals import clamp_funding_rate, realized_funding
```

Add the two invariants near `invariant_funding_sign_correct`:

```python
def invariant_account_equity_reconciles(
    account: PaperAccount, marks: dict, recorded_equity: float, *, tol: float = 1e-6
) -> bool:
    """Recorded equity must equal cash + unrealized PnL within tolerance (no phantom equity)."""
    return abs(account.equity(marks) - recorded_equity) <= tol


def invariant_cycle_funding_reconciles(
    account: PaperAccount, prev_ts, now, funding_by_symbol: dict, intervals: dict,
    marks: dict, recorded_funding: float, *, tol: float = 1e-6
) -> bool:
    """Recorded per-cycle funding must equal a recompute via realized_funding x events (extends the
    funding-sign check to the account level — the reviewer's settlement-window re-derivation)."""
    total = 0.0
    for sym, pos in account.positions.items():
        mark = marks.get(sym)
        if mark is None:
            continue
        n = count_funding_events(prev_ts, now, int(intervals.get(sym, 8)))
        rate = clamp_funding_rate(sym, funding_by_symbol.get(sym, 0.0))
        total += realized_funding(0.0, mark, pos.qty, rate, pos.direction) * n
    return abs(total - recorded_funding) <= tol
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_account.py -k "equity_reconciles or funding_reconciles" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire one account check into `_checks`**

In `self_audit._checks`, after the `funding_sign_correct` add() call, append:

```python
    # 4b. ACCOUNT EQUITY RECONCILE: recorded equity == cash + unrealized (no phantom equity).
    _acct = PaperAccount(cash=20_000.0)
    _acct.positions["ETH/USDT:USDT"] = Position(
        symbol="ETH/USDT:USDT", direction="short", qty=2.0, entry_price=2000.0,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC))
    _marks = {"ETH/USDT:USDT": 1950.0}
    add("account_equity_reconciles",
        invariant_account_equity_reconciles(_acct, _marks, _acct.equity(_marks)),
        "recorded equity must equal cash + unrealized PnL")
```

- [ ] **Step 6: Run the self-audit suite + ruff**

Run: `uv run pytest tests/ -k self_audit -v && uv run ruff check futures_fund/self_audit.py`
Expected: PASS, ruff clean. Also: `uv run python -c "from futures_fund.self_audit import run_self_audit; print(run_self_audit()['ok'])"` → `True`.

- [ ] **Step 7: Commit**

```bash
git add futures_fund/self_audit.py tests/test_account.py
git commit -m "feat(self_audit): account equity + per-cycle funding reconcile invariants

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Agent prompt edits — surface the cost artifacts to the desk

**Files:**
- Modify: `agents/funding_carry.md`, `agents/pair_analyst.md`, `agents/trader.md`, `agents/research_manager.md`, `agents/reflector.md`

> **Scope (honest):** these four analyst/trader/manager prompts do NOT currently list `context.json` in their `## Inputs` (only `universe_scout.md` and `sentiment.md` mention it). Each edit therefore FIRST declares — under `## Inputs` — that `context.json` is provided by the orchestrator (so the new `context.json.pnl.*` references rest on documented input plumbing), THEN adds the cost-aware decision rule + a literal example block. `reflector.md` instead cites `reflection_input.json` (which `reflect_cli` actually produces and now carries the per-leg costs from Task 10). No edit leans on `scorecard.carry_capture_rate` being correct (it currently passes the wrong directory — a documented pre-existing bug); the carry signal is routed through `context.json.pnl.by_symbol[...].realized_funding` instead. No code.

- [ ] **Step 1: `agents/funding_carry.md` — favor names with positive realized carry**

Under `## Inputs` add:

```markdown
- `context.json` (provided by the orchestrator): the realized cost/carry/PnL block. Read
  `context.json.pnl.by_symbol[<symbol>].realized_funding` (SIGNED, + = carry RECEIVED) and
  `total_funding_received` / `total_funding_paid`. Example block:
  ```json
  {"by_symbol": {"OP/USDT:USDT": {"realized_funding": 6.0, "unrealized": 100.0, "accrued_fees": 2.0}},
   "total_funding_received": 6.0, "total_funding_paid": 0.0}
  ```
```

Under the decision/ranking section add:

```markdown
- COST-AWARE RANKING: favor names whose realized carry has actually BANKED
  (`pnl.by_symbol[...].realized_funding > 0`, i.e. a short on positive funding or a long on negative
  funding that has settled). Discount a thesis whose projected carry has NOT shown up as realized
  carry over the holding window (carry capture is leaking).
```

- [ ] **Step 2: `agents/pair_analyst.md` — cost-adjusted pair P&L**

Under `## Inputs` add:

```markdown
- `context.json` (provided by the orchestrator): per-pair realized P&L NET of costs. Sum each leg's
  `pnl.by_symbol[<symbol>].unrealized` + `realized_funding` minus `accrued_fees`. Example: a pair
  `{long A, short B}` whose legs net `+150 unrealized + 6 carry − 4 fees = +152` after costs.
```

Under the attribution section add:

```markdown
- Judge a pair on its COST-ADJUSTED P&L, not gross: a pair that looks profitable on spread move but
  bleeds it back in fees+funding is NOT a keeper.
```

- [ ] **Step 3: `agents/trader.md` — weigh round-trip cost vs edge**

Under `## Inputs` add:

```markdown
- `context.json` (provided by the orchestrator): per-leg round-trip cost context. Read
  `pnl.last_rebalance_cost` (fees+slippage of the last rebalance) and
  `pnl.last_rebalance_turnover_usd`. Example block:
  ```json
  {"last_rebalance_cost": 6.0, "last_rebalance_turnover_usd": 4000.0,
   "by_symbol": {"ETH/USDT:USDT": {"accrued_fees": 2.0, "unrealized": 100.0}}}
  ```
```

Under the trigger/stop-placement section add:

```markdown
- DO NOT CHURN: weigh the round-trip rebalance cost (fees + slippage,
  ~`last_rebalance_cost/last_rebalance_turnover_usd` in fractional terms) against the pair's expected
  spread edge. If the round-trip cost exceeds the edge a nudge would capture, HOLD rather than
  re-trade.
```

- [ ] **Step 4: `agents/research_manager.md` — cost-adjusted RR / cost drag**

The existing `## Inputs` line reads `That symbol's analyst reports and the current regime/health from context.` — extend the Inputs list with:

```markdown
- `context.json` (provided by the orchestrator) and the injected scorecard: net-of-cost RR and cost
  drag. Read the scorecard keys `net_pnl`, `gross_pnl`, `cost_drag_bps` and the `context.json.pnl`
  block. Example: `{"net_pnl": 8.0, "gross_pnl": 14.0, "cost_drag_bps": 4285.7}` — cost is eating
  ~43% of gross.
```

Under the judgement section add:

```markdown
- Judge cost-ADJUSTED pair P&L and RR-after-costs: a high `cost_drag_bps` means the book's edge is
  being consumed by frictions — bias toward fewer, higher-conviction, lower-turnover legs.
```

- [ ] **Step 5: `agents/reflector.md` — key lessons on net (after-cost) alpha**

Under `## Inputs` (the `reflection_input.json` line) add:

```markdown
- Each winner/loser entry in `reflection_input.json` now also carries `realized_funding` (signed),
  `fees`, `slippage`, and `net_pnl` (realized P&L net of fees+slippage), populated from the journal
  by the paper-run cost engine. Example entry:
  `{"symbol": "OP/USDT:USDT", "alpha_return": 0.012, "realized_funding": 6.0, "fees": 4.0,
    "slippage": 2.0, "net_pnl": 6.0}`.
```

Under the lessons section add:

```markdown
- Key lessons on NET (after-cost) alpha, not gross: a "winner" on alpha that is a loser on `net_pnl`
  (carry/fees drag) is a lesson about cost drag, not edge. Promote lessons that improved net alpha
  and flag theses whose gross edge never survived costs.
```

- [ ] **Step 6: Verify the prompt conformance tests still pass**

Run: `uv run pytest tests/test_agent_conformance.py tests/test_role_files.py -v`
Expected: PASS. If a test pins an exact `## Inputs` line set, update its expected list to include the new bullets.

- [ ] **Step 7: Commit**

```bash
git add agents/funding_carry.md agents/pair_analyst.md agents/trader.md agents/research_manager.md agents/reflector.md
git commit -m "docs(agents): surface cost artifacts (context.json.pnl + reflection net-alpha) to the desk

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: E2E — single-run equity moves + pnl.json/account.json/ledger.jsonl exist, AND a two-run NON-ZERO funding proof

**Files:**
- Modify: `tests/test_end_to_end_no_seed.py`

Two E2E assertions, both no-seed:
1. The existing `test_full_run_builds_a_neutral_deployed_book_without_seeding` (single `main` run) now also asserts `pnl.json`/`account.json`/`ledger.jsonl` exist and the equity series moves off the flat 20000 (movement from fees+slippage on the opened book; a single-`now` run settles 0 funding events).
2. A NEW test runs `run_paper_cli.main` TWICE on the SAME state-dir with two `--now` values one sim-day apart, and asserts the second cycle's `pnl.json` carries `funding_received > 0` — proving NON-ZERO funding through the wired loop (the user's headline requirement), not only via a unit test that hand-advances time.

- [ ] **Step 1: Add the single-run assertions to the existing test**

In `test_full_run_builds_a_neutral_deployed_book_without_seeding`, after the existing `eq = state / "equity-history.jsonl"` block, add:

```python
    # Phase 9 — REALISTIC P&L: per-cycle pnl.json exists and the equity series MOVES (not flat 20000)
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
```

- [ ] **Step 2: Add the two-run NON-ZERO funding E2E**

Append to `tests/test_end_to_end_no_seed.py` (reuses the `no_seed_env` fixture and the module's `NOW_ISO`/`_FUNDING`; the universe has positive-funding shorts so a held carry leg banks funding over a day):

```python
def test_two_runs_one_day_apart_prove_nonzero_funding_in_pnl(no_seed_env):
    """The user's headline requirement: pnl.json carries NON-ZERO funding through the WIRED loop.

    Run main twice on the SAME state-dir, one sim-day apart. Run 1 opens the book (its funding clock
    starts at NOW_ISO; 0 events settled). Run 2 (NOW_ISO + 1 day) settles funding over the elapsed
    day on the still-held book -> pnl.json funding fields are non-zero. Funding is NON-ZERO at the
    account level (funding_received + funding_paid > 0), independent of net sign across the book."""
    import json as _json
    from datetime import datetime, timedelta

    from scripts.run_paper_cli import main

    main(["--now", NOW_ISO])                         # run 1: open the book, clock starts here
    state = no_seed_env / "state"
    acct1 = _json.loads((state / "account.json").read_text())
    assert acct1["last_funding_ts"] is not None      # clock advanced on run 1

    next_day = (datetime.fromisoformat(NOW_ISO) + timedelta(days=1)).isoformat()
    main(["--now", next_day])                         # run 2: a full sim-day later -> funding settles

    # the weekly cycle-2 pnl.json (run 2) must carry non-zero funding over the elapsed day
    wk2 = state / "weekly" / "cycle" / "2" / "pnl.json"
    assert wk2.exists()
    rec2 = _json.loads(wk2.read_text())
    funding_activity = rec2["funding_received"] + rec2["funding_paid"]
    assert funding_activity > 0.0, "wired loop must settle non-zero funding across one sim-day"
    # and the account clock advanced to the second run instant
    acct2 = _json.loads((state / "account.json").read_text())
    assert acct2["last_funding_ts"] is not None and acct2["last_funding_ts"] != acct1["last_funding_ts"]
```

> Note: run 2 is a SEPARATE weekly cadence cycle (cycle 2) because `cadence_due` advances the weekly counter a sim-week's worth — but funding settles on the ELAPSED time between the account's `last_funding_ts` (run-1 `now`) and run-2 `now`, which is one day, so ≥3 settlements occur regardless of the cycle number. If `cadence_due` SKIPs the weekly cadence on run 2 (candle not yet due a week later — it IS due after 7 days but only 1 day passed, so weekly may SKIP), the funding settlement still happens on whichever cadence runs (DAILY is due after 1 day). In that case assert on `state / "daily" / "cycle" / "2" / "pnl.json"` instead. **Implementer: run the test, observe which cadence produced the run-2 `pnl.json` (check `cadence_due` due-ness one day later), and assert on that cadence's cycle-2 path.** The robust assertion is: across ALL cycle-2 `pnl.json` files written on run 2, at least one has `funding_received + funding_paid > 0`. Prefer this scan if the cadence due-ness is ambiguous:
>
> ```python
>     pnls = list(state.glob("*/cycle/*/pnl.json"))
>     activity = [
>         (_json.loads(p.read_text())["funding_received"] + _json.loads(p.read_text())["funding_paid"])
>         for p in pnls
>     ]
>     assert any(a > 0.0 for a in activity), "wired loop must settle non-zero funding across one sim-day"
> ```

- [ ] **Step 3: Run the no-seed E2E**

Run: `uv run pytest tests/test_end_to_end_no_seed.py -v`
Expected: PASS. If the two-run test's cadence-path assertion is ambiguous, switch to the glob-scan assertion above (already in the plan) and re-run.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all PASS. Investigate any failure with superpowers:systematic-debugging before proceeding.

- [ ] **Step 5: ruff over the whole change set**

Run: `uv run ruff check futures_fund/ scripts/ tests/`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add tests/test_end_to_end_no_seed.py
git commit -m "test(e2e): single-run equity moves + two-run one-day-apart proves NON-ZERO funding in pnl.json

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (run by the plan author)

**Review-issue coverage (all 11 fixed):**
1. **CRITICAL multi-week double-count** — `apply_fills` now RECONCILES each touched symbol to its leg's `target_notional` (per-symbol signed delta), so re-sending the weekly book is a no-op; convergent across weeks (Task 3 + Task 4). Pinned by `test_apply_fills_resend_same_target_is_a_noop` (Task 3) and `test_weekly_cycle2_resend_does_not_double_qty` (Task 8a). ✓
2. **CRITICAL preflight/context.json dead code** — Tasks 9/11 explicitly state `context.json`/`build_scorecard` are ARTIFACTS consumed by the external SKILL.md orchestrator, NOT a wired Python prompt path; `run_paper_cli` never calls `preflight.main`. No false "wired" claim remains (Architecture + Task 9/11 scope notes). ✓
3. **CRITICAL build_scorecard "injected into every prompt"** — Architecture + Task 11 acknowledge injection is external (markdown-driven), downgrade the "trades cost-aware" claim to "the cost surface is available to the orchestrator," and note tests prove only artifact shape/values. ✓
4. **HIGH prev_ts collision** — funding `prev_ts` now derives from `PaperAccount.last_funding_ts` (Task 1/5), advanced by `settle_funding`, NOT the cycle-collided `equity_series`. Task 8 reads `account.last_funding_ts or now`. ✓
5. **HIGH E2E funding requirement** — Task 16 adds a TWO-run, one-sim-day-apart E2E asserting `funding_received + funding_paid > 0` in a run-2 `pnl.json` through the wired loop. ✓
6. **MEDIUM inert reflect cost fields** — new Task 8b patches per-leg realized `fees/slippage/realized_funding/realized_pnl` onto the journal `Decision` via `patch_outcome` (Decision is `extra="allow"`), so `_cost_fields` (Task 10) has real inputs. ✓
7. **MEDIUM load_settings dependency** — verified `load_settings()` returns `account_size_usdt=20000.0` with no `config.yaml` in cwd (ran it); Task 9 documents this AND passes `default_cash` explicitly. ✓
8. **MEDIUM funding-vs-fill ordering untested** — Task 8a `test_position_opened_this_cycle_earns_zero_funding_this_cycle` pins settle-before-fill. ✓
9. **LOW agents missing context.json in Inputs** — Task 15 step-by-step adds a `## Inputs` line declaring `context.json` is provided by the orchestrator for funding_carry/pair_analyst/trader/research_manager; reflector cites `reflection_input.json`. ✓
10. **LOW reflect loop-variable name** — Task 10 states the variable is exactly `d` (reflect_cli.py line 35) and the merge line uses `d`. ✓
11. **LOW pre-existing carry_capture_rate wrong dir** — Grounded facts + Task 15 scope note document that `scorecard.py:129` and `dashboard.py:111` call `carry_capture_rate(memory_dir, ...)` against a `state_dir` signature (out of scope); Task 15 routes carry through `context.json.pnl`, not `carry_capture_rate`. ✓

**Spec coverage (unchanged, all preserved):** ledger (Tasks 1-6), `pnl.json`/`ledger.jsonl` (Task 7), wired loop (Task 8) + ordering/no-double-count (Task 8a) + journal cost-patch (Task 8b), agent cost artifacts (Tasks 9/10/11/15), dashboard (Task 13), self-audit (Task 14), integration ≥2-cycle (Task 12), two-run funding E2E (Task 16), restart-from-scratch (`load_account` at default cash, Task 6). ✓

**Placeholder scan:** No TBD/TODO; complete code in every code step; the Task-4 reduce/close/flip branch replaces the explicit `NotImplementedError` stub raised in Task 3.

**Type consistency:** `Position` fields (`symbol, direction, qty, entry_price, opened_ts, accrued_funding, accrued_fees, accrued_slippage, realized_pnl`) and `PaperAccount` fields (`cash, positions, realized_pnl, last_funding_ts, fees_paid, slippage_paid, funding_received, funding_paid`) are identical across all tasks. Methods `apply_fills/settle_funding/mark_to_market/equity/to_dict/from_dict/_reconcile_opposite/_charge_frictions` + module funcs `load_account/save_account` and the free helper `_signed_qty` used as defined. `CostInputs(adv_usd, half_spread_bps, depth, maker)` consistent (Tasks 3, 8, 12, 8a). `build_cycle_pnl/append_ledger/_latest_pnl/build_pnl_block/_cost_fields/_geometry_cost_maps/_load_geometries/_leg_cost_patches/invariant_account_equity_reconciles/invariant_cycle_funding_reconciles` referenced with their defined signatures. Funding sign held to balance-credit `realized_funding` (never `project_funding`) in Tasks 5, 8a, 12, 14. `patch_outcome(memory_dir, *, cycle, symbol, direction, outcome)` used per its real signature (Task 8b).

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-12-phase9-realistic-paper-pnl.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**2. Inline Execution** — execute tasks in this session using superpowers:executing-plans, batch execution with checkpoints.

**Which approach?**

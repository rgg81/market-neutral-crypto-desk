# Phases 3-7 Roadmap + Master Plan Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Serve as (a) the **master index** for the entire market-neutral desk plan set and (b) a **detailed task-level roadmap** for the integration phases (3-7) that sit on top of the foundation (phases 0-2). Each phase below is a concrete, file-level task breakdown with public interfaces, acceptance criteria, and test strategy. Each integration phase here is a **placeholder-free skeleton that will be expanded into a full bite-sized TDD plan just-in-time** once its predecessor lands.

**Architecture:** "LLM proposes, code disposes." A deterministic Python spine (`futures_fund/`) owns all math (signals, the neutrality optimizer, sizing, the risk gate, fee/funding/slippage accounting, P&L, state); markdown prompt files (`agents/*.md`) reason and propose, dispatched by Claude running `SKILL.md`; thin CLIs (`scripts/*.py`) are the only way the orchestrator runs spine code. Every LLM output is validated against a pydantic contract before the spine consumes it; the spine fails loud (HALT) on contract violations. An every-cycle Adversarial Code & Calc Reviewer re-derives every load-bearing number and HALTs on mismatch via a deterministic guard flag.

**Tech Stack:** Python ≥3.11, `uv`, `pydantic>=2.6`, `numpy`, `pandas`, `scipy`, `statsmodels` (ADF/Johansen/OU), `scikit-learn` (Ledoit-Wolf), `cvxpy` OR `scipy.optimize` (constrained optimizer), `ccxt` (keyless Binance USD-M), `httpx`, `pyyaml`; `pytest` + `ruff` (line-length 100, `select=["E","F","I","UP","B"]`, `futures_fund/vendor/*` exempt). State as JSON/JSONL under `state/` and `memory/`; config in `config.yaml`. Reuse verbatim/adapted from `/home/roberto/crypto-trade-claude-code-weekly` per the reusable API map.

---

## MASTER INDEX — the whole plan set

This roadmap is the **index document**. The full plan set, in dependency order, is:

| # | Plan doc (`docs/superpowers/plans/`) | Scope | Detail level | Status |
|---|---|---|---|---|
| Spec | `2026-06-11-market-neutral-desk-design.md` | Approved design spec (source of truth) | Design | Landed |
| Contract | Canonical New-Interface Contract (embedded in each plan header) | Every NET-NEW name/signature | Interface | Locked |
| P0 | `2026-06-11-phase0-scaffold-and-realism.md` | Scaffold + data + realism primitives + **all reused CLIs lifted from the weekly repo** | **Full bite-sized TDD** | Detailed (separate doc) |
| P1 | `2026-06-11-phase1-neutrality-optimizer.md` | Neutrality + portfolio optimizer (`neutrality.py`) | **Full bite-sized TDD** | Detailed (separate doc) |
| P2 | `2026-06-11-phase2-sleeves-and-pairs.md` | Four sleeves + `Pair`/`Spread` + sentiment factor | **Full bite-sized TDD** | Detailed (separate doc) |
| **P3-7** | **`2026-06-11-phases3-7-roadmap-and-index.md` (THIS DOC)** | Control loop, agents, reviewer, self-improvement, paper run | **Task-level roadmap** | This document |

**Dependency order (must build bottom-up):**

```
P0 scaffold/realism  ──►  P1 neutrality optimizer  ──►  P2 sleeves + Pair/Spread
                                                              │
                                                              ▼
                                P3 two-cadence control loop (control_loop.py, scheduling roots)
                                                              │
                                                              ▼
                                P4 agent roster + SKILL.md orchestration
                                                              │
                                                              ▼
                                P5 risk gate reuse + every-cycle reviewer + self_audit invariants
                                                              │
                                                              ▼
                                P6 self-improvement loop re-keyed on ALPHA vs BTC-beta
                                                              │
                                                              ▼
                                P7 end-to-end paper run + KPI dashboard + walk-forward
```

Each phase consumes the contracts and modules its predecessor produced. P3 requires `neutrality.optimize_book` (P1) and the four `*_signal` builders (P2). P4 requires P3's `control_loop` step functions to drive `SKILL.md`. P5 requires P4's artifacts to review. P6 requires P5's reviewer verdicts and the per-leg neutrality residuals. P7 requires all of the above to run an end-to-end loop and compute KPIs.

### Reused-CLI provenance (resolves dangling references)

This scaffold repo is **net-new**: nothing exists until a plan creates it. The weekly repo's `scripts/` (the reusable API map) supplies the *templates* for several operational CLIs the SKILL.md ladders below invoke, but each must be **created by an explicit task** — in this repo they are produced by the **Phase 0 scaffold plan (P0)**, which lifts and re-tests the operational CLIs verbatim/adapted. The table below pins which plan creates each so no name in phases 3-7 is dangling:

| CLI / module referenced in P3-7 ladders | Created by | Weekly-repo template |
|---|---|---|
| `scripts/runlock_cli.py` | **P0** (lift `runlock.single_flight` + CLI) | `scripts/runlock_cli.py` |
| `scripts/due_check.py` | **P0** (lift `scheduling.cycle_due` + CLI; extended in P3 Task 3.1 to add `weekly`/`daily` cadences) | `scripts/due_check.py` |
| `scripts/scout_cli.py` | **P0** (lift universe screen CLI; re-pointed at the crypto-only two-sided shortlist) | `scripts/scout_cli.py` |
| `scripts/preflight.py` | **P0** (lift data/connectivity preflight) | `scripts/preflight.py` |
| `scripts/record_lessons_cli.py` | **P0** (lift two-phase lessons writer; re-keyed on alpha in P6) | `scripts/record_lessons_cli.py` |
| `scripts/self_audit_cli.py` | **P0** (lift; invariants extended in P5 Task 5.5) | `scripts/self_audit_cli.py` |
| `scripts/monitor_cli.py` | **P3 Task 3.8** (NET-NEW in this repo; created below) | `scripts/monitor_cli.py` |
| `scripts/gate_execute_cli.py` | **P4 Task 4.5** (NET-NEW in this repo; created below) | `scripts/gate_execute_cli.py` |
| `scripts/reviewer_cli.py` | **P5 Task 5.4** | (no template — net-new design) |
| `scripts/reflect_cli.py`, `scripts/promote_lesson_cli.py` | **P6 Task 6.4** | `scripts/reflect_cli.py`, `scripts/promote_lesson_cli.py` |
| `scripts/control_loop_cli.py`, `scripts/dashboard_cli.py`, `scripts/walk_forward_cli.py`, `scripts/run_paper_cli.py` | **P3/P7** (net-new, created below) | (no template) |

When P3-7 are expanded just-in-time, any ladder step that *invokes* a P0 CLI is a call, not a create; the four NET-NEW-in-this-repo execute/monitor CLIs (`monitor_cli.py`, `gate_execute_cli.py`) are created by the tasks added below (3.8, 4.5).

**How to execute (subagent-driven-development):** Each plan is executed task-by-task by a worker following `superpowers:subagent-driven-development`. For each `### Task N`, the worker (or a dispatched subagent) walks the TDD step ladder: write the failing test → run it (expect FAIL) → minimal implementation → run tests (expect PASS) → `uv run ruff check .` → commit. The full `uv run pytest` suite must be green and `uv run python scripts/self_audit_cli.py` must print `SELF-AUDIT: OK` before any commit. Protected modules (`risk_gate, executor, exits, consolidation, policy, liquidation, sizing, cycle`) may never have a limit/breaker weakened. `live` stays `false` forever.

### Just-in-time planning note

Phases 3-7 below are a **task-level roadmap**, not full bite-sized TDD plans. Per the spec (§17) and the writing-plans skill, **each of phases 3-7 will be expanded into its own full bite-sized TDD plan document (like P0/P1/P2) just-in-time — only once its predecessor phase has fully landed** (all tasks committed, suite green, self-audit OK). This avoids planning against interfaces that may shift as the foundation settles. The roadmap fixes the public interfaces (from the canonical contract), the file structure, the acceptance criteria, the **exact per-file pytest command on every Run step**, and the test strategy now, so the just-in-time expansion is mechanical: it slots in the exact failing-test/implementation/commit ladders for the steps sketched here. No task here is a placeholder — every task names real files, real signatures (all from the canonical contract or defined in P0-2), real acceptance criteria, and a real test strategy.

---

## File Structure

Files created or modified across phases 3-7. One responsibility each. Net-new modules are non-protected (§15). Paths are absolute under `/home/roberto/crypto-trade-claude-code-market-neutral`.

### Phase 3 — two-cadence control loop

| Path | Create/Modify | Single responsibility |
|---|---|---|
| `futures_fund/control_loop.py` | Create | Weekly `weekly_selection` + daily `daily_rebalance` step functions; `cadence_due`, `drift_exceeded`, `neutrality_breached`, `rebalance_deltas`. |
| `futures_fund/scheduling.py` | Modify | Already multi-cadence (`cycle_due(loop=, tf_minutes=)`, root `state/<loop>/cycle/*`); add no code — `control_loop.cadence_due` wraps it with weekly/daily `tf_minutes` and `loop=cadence`. (Reuse, no weakening.) |
| `futures_fund/cycle_io.py` | Modify | Extend `cycle_dir` to accept an optional `cadence` segment → **`state/<cadence>/cycle/<N>/`** (matching `scheduling.cycle_due`'s `state/<loop>/cycle/*` root so the due-gate and the artifact writer agree). Reused atomic write unchanged. |
| `scripts/control_loop_cli.py` | Create | Thin CLI: `--cadence {weekly,daily} --cycle N`; runs the matching step function, persists `target_weights.json`, prints JSON. |
| `scripts/monitor_cli.py` | Create | **Between-cycle light risk monitor (spec §9, §19):** reads open book + marks; trips HALT on drawdown / liq-distance / **neutrality-residual** breach; notifies. Adapted from the weekly repo template, extended with a neutrality-residual trip. |
| `tests/test_control_loop.py` | Create | Cadence gating, carry-over delta trading, drift band, neutrality-breach trigger. |
| `tests/test_cycle_io_cadence.py` | Create | `cycle_dir` cadence segmentation + atomic write under both roots; **asserts the path equals `scheduling.cycle_due`'s `state/<cadence>/cycle/*` root.** |
| `tests/test_monitor.py` | Create | Monitor trips HALT on each of drawdown / liq-distance / neutrality breach; no-op when all in band. |
| `config.yaml` | Modify | `loops.weekly` / `loops.daily` blocks (already in contract Part 3). |

### Phase 4 — agent roster + SKILL.md orchestration

| Path | Create/Modify | Single responsibility |
|---|---|---|
| `agents/universe_scout.md` | Create | Crypto-only, liquidity-filtered two-sided shortlist → `WatcherOutput`. |
| `agents/funding_carry.md` | Create | Funding cross-section ranking → `list[AnalystReport]`. |
| `agents/pair_analyst.md` | Create | Candidate pairs + hedge ratio + cointegration evidence → `list[AnalystReport]`. |
| `agents/factor_analyst.md` | Create | Momentum/carry/low-vol ranking → `list[AnalystReport]`. |
| `agents/sentiment.md` | Modify | Point-in-time sentiment → `SentimentBatch`. (Adapt archived `agents/archive/sentiment.md`.) |
| `agents/technical.md` | Create | Per-leg structure/momentum/mean-reversion → `list[AnalystReport]`. (Adapt archive.) |
| `agents/derivatives.md` | Create | OI / long-short ratio / funding crowding → `list[AnalystReport]`. (Adapt archive.) |
| `agents/bull.md` | Create | Strongest case to open/keep a leg or pair → `AnalystReport`. (Adapt archive.) |
| `agents/bear.md` | Create | Strongest case to short/close, rebut Bull → `AnalystReport`. (Adapt archive.) |
| `agents/research_manager.md` | Create | 5-tier rating + falsifiable prediction → `ResearchPlan`. (Adapt archive.) |
| `agents/trader.md` | Modify | Target weights → per-leg entry/stop/TP/triggers → `TraderOutput` (`{proposals, management, triggers, cancel_triggers}`). |
| `agents/reflector.md` | Modify | Alpha-vs-beta contrastive lessons → `{lessons:[Lesson]}`. |
| `agents/neutrality_constructor.md` | Create | Doc-only deterministic role: describes the §8 optimizer; code-enforced by `neutrality.py`. No JSON. |
| `agents/risk_gate.md` | Create | Doc-only deterministic role: describes the non-overridable gate; code-enforced by `risk_gate.py`. No JSON. |
| `futures_fund/contracts.py` | Modify | Add `TraderOutput` bundle model (`proposals: list[AgentProposal]`, `management/triggers/cancel_triggers: list[dict]`) — the Trader's conformance target. (Mirrors the weekly repo's `ScalperOutput`.) |
| `futures_fund/config.py` | Modify | Extend `_default_loops()` so `Settings.loops` has `weekly`/`daily` keys → `model_for(role, loop="weekly"\|"daily")` resolves. |
| `scripts/gate_execute_cli.py` | Create | **Execute boundary (NET-NEW in this repo):** `--cadence {weekly,daily} --cycle N`; loads `proposals.json`, runs the risk gate + execution sim via `gate_execute_step(..., loop=cadence)`, persists `report.json`. Reviewer precondition wired in P5 Task 5.4. |
| `tests/fixtures/agent_examples/{universe_scout,funding_carry,pair_analyst,factor_analyst,sentiment,technical,derivatives,bull,bear,research_manager,trader,reflector}.json` | Create | One example fixture per analyst/decision agent. |
| `tests/test_role_files.py` | Modify | Pin the new roster + mandatory section structure. |
| `tests/test_agent_conformance.py` | Modify | Validate each fixture against its contract model (incl. `trader.json` → `TraderOutput`). |
| `tests/test_settings_loops.py` | Create | `model_for` resolves `weekly`/`daily` loops; `loops` has both keys. |
| `tests/test_gate_execute_cli.py` | Create | `gate_execute_cli` `--cadence` dispatch; reviewer-gate HALT wired in P5. |

### Phase 5 — risk gate reuse + every-cycle reviewer + self-audit

| Path | Create/Modify | Single responsibility |
|---|---|---|
| `futures_fund/risk_gate.py` | Modify | Lift verbatim from weekly; add `funding.unclamped_in_rr` plumbing so carry credit is visible (override the `max(0.0, funding)` clamp). Protected — no limit weakened; **RR-floor-monotonicity test required** (Task 5.1 Step 5). |
| `futures_fund/reviewer.py` | Create | `review_cycle` + the 17 canonical `check_*` functions + `reviewer_gate_ok`. Re-derives every load-bearing number; emits `ReviewerVerdict`. |
| `futures_fund/self_audit.py` | Modify | Add neutrality/funding/pair/sentiment/crypto-only invariants to `_checks()`. |
| `scripts/reviewer_cli.py` | Create | `--cadence --cycle N`; runs `review_cycle`, persists `reviewer.json`, prints JSON; `SystemExit(2)` if `passed` False. |
| `tests/test_reviewer.py` | Create | Each of the 17 canonical check ids: matched ground truth → ok; injected mismatch → fail + HALT. |
| `tests/test_self_audit.py` | Modify | Assert the new named invariants are present and pass. |
| `tests/test_risk_gate.py` | Modify | Carry-visible-in-RR test + the RR-floor-monotonicity guard (no clamp→unclamp flip from FAIL to PASS at the RR≥2 floor). |

### Phase 6 — self-improvement loop (alpha-keyed)

| Path | Create/Modify | Single responsibility |
|---|---|---|
| `futures_fund/journal.py` | Modify | Reuse two-phase journal; add a typed alpha-vs-beta outcome accessor `alpha_outcome(decision) -> AlphaOutcome` reading `alpha_return`, `beta_contribution`, `pair_cointegrated_at_exit`, `funding_thesis_matched`, `neutrality_in_band`, `sentiment_helped`. |
| `futures_fund/lessons.py` | Modify | Reuse; add new lesson tags/dimensions to the retrieval filter; keep DSR-gated promotion. |
| `futures_fund/improvement.py` | Modify | Re-point KPIs at neutral metrics; add `both_sides_deployment_rate`, `pair_survival_rate`, `carry_capture_rate`, `sentiment_hit_rate`, `reviewer_veto_rate`, `alpha_sharpe_trend`. |
| `futures_fund/scorecard.py` | Modify | Inject alpha-Sharpe + neutral KPIs (incl. `reviewer_veto_rate`, `alpha_sharpe_trend`) into agent prompts; keep two-sided warnings. |
| `futures_fund/graduation.py` | Modify | Reuse DSR gate; `walk_forward_required` enforced before trusting sleeve-param change. |
| `futures_fund/repair.py` | Modify | Self-healing code loop hooks (`memory/repair-journal.md`). |
| `scripts/reflect_cli.py` | Create | Builds `reflection_input.json`, dispatches reflector, records lessons (template: weekly repo). |
| `scripts/promote_lesson_cli.py` | Create | DSR-gated CANDIDATE→VALIDATED promotion (template: weekly repo). |
| `tests/test_improvement_neutral.py` | Create | New KPI computations from seeded state (incl. `reviewer_veto_rate`, `alpha_sharpe_trend`). |
| `tests/test_journal_alpha.py` | Create | Alpha-vs-beta outcome accessor + idempotency. |

### Phase 7 — end-to-end paper run + KPI dashboard

| Path | Create/Modify | Single responsibility |
|---|---|---|
| `futures_fund/dashboard.py` | Create | `build_kpi_dashboard` — no-losing-month, daily Sharpe×365, both-sides deployment rate, neutrality adherence, pair-survival, carry-capture, sentiment hit-rate, **reviewer veto rate**, max drawdown. |
| `futures_fund/walk_forward.py` | Create | Time-series-aware OOS validation harness over sleeve params (reuse `walk-forward-validation` skill + `vendor/overfit_detector.py`); **point-in-time inputs from `data.binance.vision` archives**. |
| `scripts/dashboard_cli.py` | Create | Prints the KPI dashboard JSON / markdown. |
| `scripts/run_paper_cli.py` | Create | End-to-end driver: run-lock → due-check → cadence step → reviewer gate → execute → equity → reflect, both cadences under one lock. |
| `tests/test_dashboard.py` | Create | Each KPI from seeded equity/journal/cycle artifacts (incl. `reviewer_veto_rate`). |
| `tests/test_walk_forward.py` | Create | OOS split correctness + DSR gate decision + PIT-provenance assertion. |
| `tests/test_end_to_end.py` | Create | Full weekly+daily cycle on a fake exchange, fully neutral book, reviewer green. |

---

## Phase 3 — Two-cadence control loop

**Depends on:** P1 `neutrality.optimize_book`, `NeutralityConfig`; P2 `*_signal` builders, `Pair`/`Spread`. **Reuses:** `scheduling.cycle_due` (root `state/<loop>/cycle/*`), `runlock.single_flight`, `cycle_io.save_output/load_output`, `state.load_account/save_account`, `equity_log.record_equity`, the weekly `monitor_cli.py`/`monitor.py` template.

**Goal:** Two scheduled cadence roots — a weekly Selection Meeting (new symbol set + target weights via the optimizer, carry-over delta trading) and a daily Rebalance Meeting (same set toward targets within a drift band) — with per-cycle state artifacts under **`state/<cadence>/cycle/<N>/`** (the SAME root `scheduling.cycle_due(loop=cadence)` reads, so the due-gate and the artifact writer never diverge), plus a between-cycle light risk monitor.

> **CADENCE-ROOT INVARIANT (binding):** the reused `scheduling.cycle_due(..., loop=cadence)` resolves its cycle root as `state/<loop>/cycle/*` (i.e. `state/weekly/cycle/N`, `state/daily/cycle/N`) — see the weekly repo's `scheduling.py` docstring. Therefore `cycle_io.cycle_dir(cadence=...)` MUST write artifacts to the **same** `state/<cadence>/cycle/<N>/` path so the due-gate reads the served-candle `report.json` from exactly where the writer put it. We do NOT use `state/cycle/<cadence>/N`; that earlier layout pointed the gate and the writer at two different directories and silently broke cadence gating + idempotency.

### Task 3.1: Cadence gating (`cadence_due`)

**Files:**
- Create: `futures_fund/control_loop.py`
- Test: `tests/test_control_loop.py`

- [ ] **Step 1: Write failing test for weekly cadence wiring** — spy on `scheduling.cycle_due` and assert `cadence_due` delegates with EXACTLY `tf_minutes=10080, loop="weekly"` (so the cycle root is `state/weekly/cycle/*`). The spy asserts the directory contract, not just the returned mode.
  ```python
  # tests/test_control_loop.py
  from datetime import datetime, timezone
  import futures_fund.control_loop as cl
  from futures_fund.control_loop import cadence_due

  def test_cadence_due_weekly_delegates_with_weekly_root(tmp_path, monkeypatch):
      seen = {}
      def fake_cycle_due(state_dir, now_utc, *, tf_minutes, loop):
          seen["tf_minutes"] = tf_minutes
          seen["loop"] = loop
          return ("FRESH", 1, "spy")
      monkeypatch.setattr(cl, "cycle_due", fake_cycle_due)
      now = datetime(2026, 6, 11, 0, 0, tzinfo=timezone.utc)
      mode, n, reason = cadence_due(tmp_path / "s", now, "weekly")
      assert seen == {"tf_minutes": 10080, "loop": "weekly"}  # root => state/weekly/cycle/*
      assert (mode, n) == ("FRESH", 1)
  ```
- [ ] **Step 2: Run it (expect FAIL — module/function missing).** `uv run pytest tests/test_control_loop.py::test_cadence_due_weekly_delegates_with_weekly_root -x` (expect FAIL: `ModuleNotFoundError`).
- [ ] **Step 3: Minimal implementation.** Implement `cadence_due` per contract §2.12 mapping `weekly→(7*1440=10080,"weekly")`, `daily→(1440,"daily")`.
  ```python
  # futures_fund/control_loop.py
  from __future__ import annotations
  from datetime import datetime
  from futures_fund.models import Cadence
  from futures_fund.scheduling import cycle_due

  _CADENCE_TF = {"weekly": 7 * 1440, "daily": 1440}  # 10080 / 1440 minutes

  def cadence_due(state_dir, now_utc: datetime, cadence: Cadence) -> tuple[str, int, str]:
      tf = _CADENCE_TF[cadence]
      # loop=cadence => cycle root state/<cadence>/cycle/* (matches cycle_io.cycle_dir)
      return cycle_due(state_dir, now_utc, tf_minutes=tf, loop=cadence)
  ```
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_control_loop.py -x` (expect PASS).
- [ ] **Step 5: Add daily-delegation + double-fire SKIP tests.** Show the actual assertions (not prose): a spy asserting `cycle_due(..., tf_minutes=1440, loop="daily")`, and an integration assertion that a served candle returns `SKIP` (cadence cannot double-fire — spec §9).
  ```python
  def test_cadence_due_daily_delegates_with_daily_root(tmp_path, monkeypatch):
      seen = {}
      def fake_cycle_due(state_dir, now_utc, *, tf_minutes, loop):
          seen.update(tf_minutes=tf_minutes, loop=loop)
          return ("FRESH", 1, "spy")
      monkeypatch.setattr(cl, "cycle_due", fake_cycle_due)
      now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
      cadence_due(tmp_path / "s", now, "daily")
      assert seen == {"tf_minutes": 1440, "loop": "daily"}  # root => state/daily/cycle/*

  def test_cadence_cannot_double_fire(tmp_path, write_served_report):
      # seed a completed report for the candle containing `now` under state/daily/cycle/1/
      now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
      write_served_report(tmp_path / "s" / "daily" / "cycle" / "1", served=now, tf_minutes=1440)
      mode, n, _ = cadence_due(tmp_path / "s", now, "daily")  # real cycle_due, no monkeypatch
      assert mode == "SKIP"
      assert n == 1
  ```
- [ ] **Step 6: Run it (expect PASS).** `uv run pytest tests/test_control_loop.py -x` (expect PASS).
- [ ] **Step 7: Commit.** `git commit -am "Phase 3: cadence_due weekly/daily roots over scheduling.cycle_due (state/<cadence>/cycle/*)"` (footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`)

### Task 3.2: Per-cadence cycle artifacts

**Files:**
- Modify: `futures_fund/cycle_io.py`
- Test: `tests/test_cycle_io_cadence.py`

- [ ] **Step 1: Write failing test** asserting `cycle_dir(state_dir, 3, cadence="weekly")` resolves to **`state/weekly/cycle/3`** — i.e. the SAME root `scheduling.cycle_due(loop="weekly")` reads — and that `save_output`/`load_output` round-trip a `TargetWeights` there atomically. The test pins the path against the due-gate root so the two can never diverge.
  ```python
  # tests/test_cycle_io_cadence.py
  from pathlib import Path
  from futures_fund.cycle_io import cycle_dir, save_output, load_output

  def test_cycle_dir_cadence_matches_due_gate_root(tmp_path):
      d = cycle_dir(tmp_path, 3, cadence="weekly")
      # MUST equal scheduling.cycle_due(loop="weekly")'s root state/weekly/cycle/3
      assert d == Path(tmp_path) / "weekly" / "cycle" / "3"

  def test_cycle_dir_no_cadence_is_back_compat(tmp_path):
      assert cycle_dir(tmp_path, 3) == Path(tmp_path) / "cycle" / "3"
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_cycle_io_cadence.py -x` (expect FAIL: `cycle_dir() got an unexpected keyword argument 'cadence'`).
- [ ] **Step 3: Minimal implementation.** Add optional `cadence: Cadence | None = None` to `cycle_dir`; when set, the root is `state/<cadence>/cycle/<n>` (NOT `state/cycle/<cadence>/n`) to match `scheduling.cycle_due`'s `state/<loop>/cycle/*`. Keep the no-cadence path (`state/cycle/<n>`) for back-compat. `save_output`/`load_output` pass `cadence` through. Atomic write (temp + `os.replace`) unchanged.
  ```python
  def cycle_dir(state_dir, cycle_no: int, *, cadence: str | None = None) -> Path:
      base = Path(state_dir)
      root = (base / cadence / "cycle") if cadence else (base / "cycle")
      return root / str(cycle_no)
  ```
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_cycle_io_cadence.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 3: cycle_io cadence segmentation state/<cadence>/cycle/<N>/ (matches cycle_due root)"`

### Task 3.3: Weekly Selection Meeting (`weekly_selection`)

**Files:**
- Modify: `futures_fund/control_loop.py`
- Test: `tests/test_control_loop.py`

- [ ] **Step 1: Write failing test** that `weekly_selection` calls `optimize_book` and returns a `TargetWeights` whose residuals are in band and whose `feasible` is True for a balanced two-sleeve fixture; persisted under `state/weekly/cycle/<N>/target_weights.json`.
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_control_loop.py::test_weekly_selection_runs_optimizer -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation.** Build geometries+sleeves → `risk_parity_budgets` → `optimize_book(..., prior_legs=prior.legs if prior else None, cfg=cfg)`; persist via `save_output(state_dir, cycle, "target_weights", tw, cadence="weekly")`; return `tw`.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_control_loop.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 3: weekly_selection runs optimizer, persists target weights"`

### Task 3.4: Carry-over delta trading (`rebalance_deltas`)

**Files:**
- Modify: `futures_fund/control_loop.py`
- Test: `tests/test_control_loop.py`

This is the trickiest bit: overlapping unchanged legs must be **excluded** so the book is not churned (spec §9: "trade only the deltas").

- [ ] **Step 1: Write failing test.** Prior book has BTC long $5k; new target has BTC long $5k (unchanged) and ETH short $5k (new). `rebalance_deltas(prior, target)` must return only the ETH leg, not BTC.
  ```python
  def test_rebalance_deltas_excludes_unchanged_overlap(make_tw):
      prior = make_tw([("BTC/USDT:USDT", "long", 5000.0)])
      target = make_tw([("BTC/USDT:USDT", "long", 5000.0),
                        ("ETH/USDT:USDT", "short", 5000.0)])
      deltas = rebalance_deltas(prior, target)
      syms = {leg.symbol for leg in deltas}
      assert syms == {"ETH/USDT:USDT"}
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_control_loop.py::test_rebalance_deltas_excludes_unchanged_overlap -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** — key by `(symbol, direction)`, emit a delta leg only when `target_notional` differs beyond a $1 epsilon (or the leg is new/removed):
  ```python
  def rebalance_deltas(prior: TargetWeights, target: TargetWeights) -> list[WeightLeg]:
      prior_by = {(l.symbol, l.direction): l for l in prior.legs}
      out: list[WeightLeg] = []
      for leg in target.legs:
          p = prior_by.get((leg.symbol, leg.direction))
          if p is None or abs(leg.target_notional - p.target_notional) > 1.0:
              out.append(leg)  # carry-over: unchanged overlap excluded
      # removed legs (in prior, absent from target) become zero-notional unwinds
      tgt_keys = {(l.symbol, l.direction) for l in target.legs}
      for (sym, d), p in prior_by.items():
          if (sym, d) not in tgt_keys:
              out.append(p.model_copy(update={"target_notional": 0.0, "weight": 0.0}))
      return out
  ```
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_control_loop.py -x` (expect PASS).
- [ ] **Step 5: Add unwind test** — leg in prior but absent from target produces a zero-notional unwind delta. `uv run pytest tests/test_control_loop.py::test_rebalance_deltas_unwinds_removed -x` (expect PASS).
- [ ] **Step 6: Commit.** `git commit -am "Phase 3: carry-over rebalance_deltas (trade only deltas, unwind removed legs)"`

### Task 3.5: Daily drift band + neutrality-breach trigger

**Files:**
- Modify: `futures_fund/control_loop.py`
- Test: `tests/test_control_loop.py`

- [ ] **Step 1: Write failing tests** for `drift_exceeded(0.5, 0.4, drift_band=0.20)` → True (25% drift > 20%) and `False` at 0.45 vs 0.4 (12.5%); and `neutrality_breached(tw, cfg)` True when `dollar_residual_frac > dollar_band` OR `|beta_residual| > beta_band`.
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_control_loop.py::test_drift_exceeded tests/test_control_loop.py::test_neutrality_breached -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** per contract:
  ```python
  def drift_exceeded(current_weight, target_weight, *, drift_band=0.20) -> bool:
      if target_weight == 0.0:
          return current_weight != 0.0
      return abs(current_weight - target_weight) / abs(target_weight) > drift_band

  def neutrality_breached(target, cfg) -> bool:
      return (target.dollar_residual_frac > cfg.dollar_band
              or abs(target.beta_residual) > cfg.beta_band)
  ```
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_control_loop.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 3: drift_exceeded + neutrality_breached daily-rebalance gates"`

### Task 3.6: Daily Rebalance Meeting (`daily_rebalance`)

**Files:**
- Modify: `futures_fund/control_loop.py`
- Test: `tests/test_control_loop.py`

- [ ] **Step 1: Write failing test** that `daily_rebalance` keeps the SAME symbol set, recomputes residuals/z/funding/sentiment, and only trades names where `drift_exceeded`, a `Spread.state == "stop"`, or `neutrality_breached` is true; an in-band book yields zero delta legs; persists under `state/daily/cycle/<N>/`.
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_control_loop.py::test_daily_rebalance_same_set -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation.** Re-run `optimize_book` against the fixed symbol set (`prior_legs=target.legs`), then `rebalance_deltas(prior=target, target=recomputed)`; force a trade when `neutrality_breached` or any `Spread.state=="stop"`; persist under `cadence="daily"`.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_control_loop.py -x` (expect PASS).
- [ ] **Step 5: Add z-stop forcing test** — a `Spread` flipped to `"stop"` forces its legs into the deltas even if drift is in-band. `uv run pytest tests/test_control_loop.py::test_daily_rebalance_zstop_forces -x` (expect PASS).
- [ ] **Step 6: Commit.** `git commit -am "Phase 3: daily_rebalance same-set toward targets within drift band"`

### Task 3.7: `control_loop_cli.py`

**Files:**
- Create: `scripts/control_loop_cli.py`
- Test: `tests/test_control_loop.py`

- [ ] **Step 1: Write failing test** invoking `main()` with `--cadence weekly --cycle 1` writes **`state/weekly/cycle/1/target_weights.json`** (the cadence-due root) and prints parseable JSON.
  ```python
  def test_cli_writes_weekly_cadence_root(tmp_path, monkeypatch, balanced_settings):
      monkeypatch.setattr("scripts.control_loop_cli.load_settings", lambda *_a, **_k: balanced_settings)
      monkeypatch.chdir(tmp_path)
      from scripts.control_loop_cli import main
      main(["--cadence", "weekly", "--cycle", "1"])
      assert (tmp_path / "state" / "weekly" / "cycle" / "1" / "target_weights.json").exists()
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_control_loop.py::test_cli_writes_weekly_cadence_root -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** — argparse (`--cadence {weekly,daily}`, `--cycle int required`), `load_settings()`, dispatch to `weekly_selection`/`daily_rebalance`, `print(json.dumps(result, indent=2, default=str))`. Fail-closed: `SystemExit(2)` if upstream artifacts (sleeves/geometry) missing.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_control_loop.py -x` (expect PASS).
- [ ] **Step 5: Lint + commit.** `uv run ruff check .` then `git commit -am "Phase 3: control_loop_cli weekly/daily entrypoint"`

### Task 3.8: Between-cycle light risk monitor (`monitor_cli.py`)

**Files:**
- Create: `scripts/monitor_cli.py`
- Test: `tests/test_monitor.py`

Spec §9 / §19: a lighter risk monitor runs between cadence cycles and can trip HALT on a drawdown / liq-distance / **neutrality-residual** breach. Adapt the weekly repo's `monitor_cli.py` + `futures_fund/monitor.py` template (drawdown + liq-distance already covered there) and **extend it with a neutrality-residual trip** using `neutrality.dollar_residual`/`beta_residual` against `NeutralityConfig` bands.

- [ ] **Step 1: Write failing test** that `check_positions(...)` (or the monitor entrypoint) trips HALT (`set_halt` called / exit non-zero) when (a) drawdown breaches `max_drawdown_tolerance`, (b) any leg's liq-distance < `2.5×`, or (c) the live book's dollar/beta residual exceeds the bands; and is a no-op (no HALT) when all three are in band.
  ```python
  def test_monitor_trips_halt_on_neutrality_breach(tmp_path, monkeypatch, imbalanced_book):
      halted = {}
      monkeypatch.setattr("scripts.monitor_cli.set_halt", lambda *_a, **_k: halted.setdefault("h", True))
      # imbalanced_book has dollar_residual_frac well above dollar_band
      from scripts.monitor_cli import main
      main([])
      assert halted.get("h") is True

  def test_monitor_noop_when_in_band(tmp_path, monkeypatch, neutral_book):
      called = {"halt": False}
      monkeypatch.setattr("scripts.monitor_cli.set_halt", lambda *_a, **_k: called.__setitem__("halt", True))
      from scripts.monitor_cli import main
      main([])
      assert called["halt"] is False
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_monitor.py -x` (expect FAIL: module missing).
- [ ] **Step 3: Minimal implementation** — adapt weekly `monitor_cli.py`: load account/positions/marks, compute equity + per-leg liq-distance (existing `check_positions`), AND compute the live book's `dollar_residual`/`beta_residual` vs `NeutralityConfig` bands; `set_halt` + notify if ANY trips; exit 0 when clean. No new limit weakened (adds a trip, never relaxes one — protected-module rule respected).
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_monitor.py -x` (expect PASS).
- [ ] **Step 5: Lint + commit.** `uv run ruff check .` then `git commit -am "Phase 3: monitor_cli between-cycle risk monitor (drawdown/liq-distance/neutrality HALT)"`

**Acceptance criteria (Phase 3):**
- `cadence_due` returns `SKIP` for an already-served candle (no double-fire) and `FRESH`/`RETRY` otherwise, **reading from `state/<cadence>/cycle/*`** (the same root `cycle_io.cycle_dir(cadence=...)` writes).
- `weekly_selection` produces a `TargetWeights` with `feasible=True` and residuals in band on a balanced fixture; symbol set + target weights persisted under `state/weekly/cycle/<N>/`.
- `daily_rebalance` keeps the same symbol set and trades only names outside the drift band, on a broken z-stop, or on a neutrality breach; persists under `state/daily/cycle/<N>/`.
- `rebalance_deltas` excludes unchanged overlapping legs and emits zero-notional unwinds for removed legs.
- `monitor_cli` trips HALT on drawdown / liq-distance / neutrality breach and is a no-op in band.
- All state writes atomic; idempotent under DUE RETRY.

**Test strategy (Phase 3):** Pure-function unit tests with `tmp_path` state dirs and hand-built `TargetWeights`/`Spread` fixtures (no network). Cadence gating tested by (a) a monkeypatch spy proving `cadence_due` delegates with the exact `tf_minutes`/`loop` so the root is `state/<cadence>/cycle/*`, and (b) an integration test seeding a `report.json` for the served candle and asserting `SKIP`. The `cycle_io` test pins `cycle_dir(cadence=...)` against the due-gate root so they can never diverge. Carry-over tested by asserting exact symbol membership of the delta set. Property test: for any prior==target, `rebalance_deltas` returns `[]` (no churn). CLI tested by direct `main()` invocation with monkeypatched `load_settings`. Monitor tested by injecting an imbalanced vs neutral book and asserting `set_halt` is/ isn't called.

---

## Phase 4 — Agent roster + SKILL.md orchestration

**Depends on:** P3 control-loop step functions (to define which CLI each phase calls); P0 operational CLIs (`runlock_cli`, `due_check`, `scout_cli`, `preflight`, `record_lessons_cli`) per the provenance table. **Reuses:** archived prompts (`agents/archive/*`), `contracts.py` models, the role-file/conformance test patterns, the weekly `gate_execute_step` + `gate_execute_cli.py` template, the weekly `ScalperOutput` model as the precedent for `TraderOutput`.

**Goal:** Write every `agents/*.md` prompt with the mandatory section structure, tie each to its pydantic output contract + fixture, define the `TraderOutput` bundle model, wire `model_for` to the `weekly`/`daily` loops, create the NET-NEW `gate_execute_cli.py` execute boundary, and write `SKILL.md` ordering weekly vs daily phases. The two deterministic doc-only roles (Neutrality Constructor, Risk Gate) have NO JSON output.

### Agent → contract map (each `.md` Output block is a literal example of its model)

| Agent file | Contract model (`futures_fund/contracts.py`) | Output shape | Cadence |
|---|---|---|---|
| `universe_scout.md` | `WatcherOutput` | `{"candidates":[Candidate]}` | weekly (refresh) |
| `funding_carry.md` | `list[AnalystReport]` | `{"reports":[AnalystReport]}` | weekly + daily |
| `pair_analyst.md` | `list[AnalystReport]` | `{"reports":[AnalystReport]}` (signals carry hedge ratio, adf p) | weekly |
| `factor_analyst.md` | `list[AnalystReport]` | `{"reports":[AnalystReport]}` | weekly |
| `sentiment.md` | `SentimentBatch` | `{"reports":[SentimentReport]}` (incl. `"MARKET"`) | weekly + daily |
| `technical.md` | `list[AnalystReport]` | `{"reports":[AnalystReport]}` | weekly |
| `derivatives.md` | `list[AnalystReport]` | `{"reports":[AnalystReport]}` | weekly |
| `bull.md` | `AnalystReport` | `{"reports":[AnalystReport]}` (`stance:"bullish"`) | weekly |
| `bear.md` | `AnalystReport` | `{"reports":[AnalystReport]}` (`stance:"bearish"`) | weekly |
| `research_manager.md` | `ResearchPlan` | `{"plans":[ResearchPlan]}` | weekly |
| `trader.md` | **`TraderOutput`** (added Task 4.0) | `{"proposals":[AgentProposal],"management":[],"triggers":[],"cancel_triggers":[]}` | weekly + daily |
| `reflector.md` | `Lesson` | `{"lessons":[Lesson]}` | weekly + daily |
| `neutrality_constructor.md` | none (doc-only) | NO JSON — code-enforced by `neutrality.py` | both |
| `risk_gate.md` | none (doc-only) | NO JSON — code-enforced by `risk_gate.py` | both |

**Sentiment Analyst contract + point-in-time rule (spec §7.1, §7.3):** `sentiment.md` outputs a `SentimentBatch` of `SentimentReport` rows, one per coin plus one with `symbol=="MARKET"`. Each report's `level ∈ SentimentLevel` maps to `s` via `sentiment_ingest.level_to_s` (`very_positive→1.0 … very_negative→-1.0`); each `SentimentSource.published_ts` MUST be `< report.as_of_ts` (point-in-time). The prompt must instruct: cite sources with timestamps, never use any source published at/after decision time, set `confidence` from source agreement, and emit a neutral report (not omission) on missing data so the spine can fail-soft.

### Worked fixture #1 — `sentiment.json` (the trickiest contract; PIT rule is load-bearing)

This is the literal `tests/fixtures/agent_examples/sentiment.json` and the `sentiment.md` Output block. Note **every `published_ts < as_of_ts`** (point-in-time), the `"MARKET"` row, and the ordinal `level` that `level_to_s` maps to `s ∈ [-1,+1]`:

```json
{
  "reports": [
    {
      "symbol": "BTC/USDT:USDT",
      "as_of_ts": "2026-06-11T00:00:00Z",
      "level": "positive",
      "confidence": 0.62,
      "rationale": "ETF inflows + constructive funding; no fresh negative catalysts.",
      "sources": [
        {"url": "https://example-cryptonews/feed/btc-etf", "title": "BTC ETF net inflow $210M", "published_ts": "2026-06-10T18:30:00Z"},
        {"url": "https://reddit.com/r/CryptoCurrency/c1", "title": "Funding stays mildly positive", "published_ts": "2026-06-10T21:05:00Z"}
      ]
    },
    {
      "symbol": "ETH/USDT:USDT",
      "as_of_ts": "2026-06-11T00:00:00Z",
      "level": "neutral",
      "confidence": 0.40,
      "rationale": "Mixed signals; no decisive catalyst before decision time.",
      "sources": [
        {"url": "https://example-cryptonews/feed/eth", "title": "ETH range-bound ahead of upgrade", "published_ts": "2026-06-10T12:00:00Z"}
      ]
    },
    {
      "symbol": "MARKET",
      "as_of_ts": "2026-06-11T00:00:00Z",
      "level": "neutral",
      "confidence": 0.55,
      "rationale": "Fear & Greed = 52 (neutral); broad tape balanced.",
      "sources": [
        {"url": "https://api.alternative.me/fng/", "title": "Fear & Greed 52", "published_ts": "2026-06-10T23:00:00Z"}
      ]
    }
  ]
}
```

The fixture test asserts, for every report and every source, `published_ts < as_of_ts`; `level ∈ SentimentLevel`; and that a `"MARKET"` row is present. The other analyst fixtures (`funding_carry.json`, `pair_analyst.json`, etc.) follow the `AnalystReport` field set: `{symbol, stance, conviction, thesis, signals:{...}, horizon}` — `pair_analyst.json.signals` additionally carries `hedge_ratio` and `adf_pvalue`; `funding_carry.json.signals` carries `signed_funding` and `funding_interval_h`.

### Worked fixture #2 — `trader.json` (`TraderOutput`; the other tricky contract)

This is the literal `tests/fixtures/agent_examples/trader.json` and the `trader.md` Output block, validated against the NEW `TraderOutput` model (Task 4.0). The Trader maps target weights → per-leg orders, **does no sizing** (notional comes from the optimizer); `management/triggers/cancel_triggers` default to `[]` (the stand-down contract requires an explicit empty `management` list):

```json
{
  "proposals": [
    {
      "symbol": "BTC/USDT:USDT",
      "direction": "long",
      "entry": 68500.0,
      "stop": 66100.0,
      "take_profit": 73200.0,
      "rationale": "Carry+factor long leg; entry at mark, stop = 2.5x liq-distance floor.",
      "trigger_type": "market"
    },
    {
      "symbol": "ETH/USDT:USDT",
      "direction": "short",
      "entry": 3580.0,
      "stop": 3720.0,
      "take_profit": 3300.0,
      "rationale": "Relative-value short vs BTC; sentiment neutral, no flip.",
      "trigger_type": "market"
    }
  ],
  "management": [],
  "triggers": [],
  "cancel_triggers": []
}
```

`AgentProposal` field names (`symbol, direction, entry, stop, take_profit, rationale, trigger_type`) are reused verbatim from the contract — the Trader fills these from the optimizer's `TargetWeights.legs`, never inventing notional.

### Task 4.0: `TraderOutput` bundle model + `model_for` loop wiring

**Files:**
- Modify: `futures_fund/contracts.py`, `futures_fund/config.py`
- Test: `tests/test_settings_loops.py`, `tests/test_agent_conformance.py`

Resolves the undefined Trader-bundle contract (the map's `(proposals bundle)` had no model) and the `model_for(loop="weekly"|"daily")` value-contract gap (the reused `loops` vocabulary is `fast`/`strategic`).

- [ ] **Step 1: Write failing test** that `TraderOutput` validates the `trader.json` fixture (`proposals: list[AgentProposal]`, `management/triggers/cancel_triggers: list[dict]` defaulting to `[]`), and that `Settings().model_for("trader", loop="weekly")` and `...loop="daily")` both resolve to a concrete model string (i.e. `"weekly"`/`"daily"` are present in `Settings.loops`).
  ```python
  # tests/test_settings_loops.py
  from futures_fund.config import Settings
  def test_model_for_resolves_weekly_daily_loops():
      s = Settings()
      assert {"weekly", "daily"} <= set(s.loops)          # loops vocabulary extended
      assert isinstance(s.model_for("trader", loop="weekly"), str)
      assert isinstance(s.model_for("reflector", loop="daily"), str)
  ```
  ```python
  # tests/test_agent_conformance.py (added case)
  from futures_fund.contracts import TraderOutput
  import json, pathlib
  def test_trader_fixture_validates_traderoutput():
      data = json.loads(pathlib.Path("tests/fixtures/agent_examples/trader.json").read_text())
      out = TraderOutput.model_validate(data)
      assert len(out.proposals) == 2
      assert out.management == []          # stand-down contract: explicit empty list
  ```
- [ ] **Step 2: Run them (expect FAIL).** `uv run pytest tests/test_settings_loops.py tests/test_agent_conformance.py::test_trader_fixture_validates_traderoutput -x` (expect FAIL: `ImportError: TraderOutput` / `weekly not in loops`).
- [ ] **Step 3: Minimal implementation.** In `contracts.py` add (mirroring the weekly repo's `ScalperOutput`):
  ```python
  class TraderOutput(BaseModel):
      """The Trader/Execution planner's bundle: gate-ready opens + management + triggers."""
      proposals: list[AgentProposal] = Field(default_factory=list)
      management: list[dict] = Field(default_factory=list)
      triggers: list[dict] = Field(default_factory=list)
      cancel_triggers: list[dict] = Field(default_factory=list)
  ```
  In `config.py` extend `_default_loops()` to include the desk's two cadences so `model_for(loop=...)` resolves:
  ```python
  def _default_loops() -> dict[str, LoopSettings]:
      return {
          "weekly": LoopSettings(timeframe="1w", quick_model="sonnet", deep_model="opus", poll_minutes=60),
          "daily":  LoopSettings(timeframe="1d", quick_model="sonnet", deep_model="opus", poll_minutes=60),
      }
  ```
  (`model_for`'s body is unchanged — it already falls back to `self.loops[loop].deep_model` when `loop in self.loops`; the per-agent `agent_models` map still wins for deciding agents.)
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_settings_loops.py tests/test_agent_conformance.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 4: TraderOutput bundle model + weekly/daily loop keys for model_for"`

### Task 4.1: Analyst roster prompts (Scout, Carry, Pair, Factor, Sentiment, Technical, Derivatives)

**Files:**
- Create: `agents/universe_scout.md`, `agents/funding_carry.md`, `agents/pair_analyst.md`, `agents/factor_analyst.md`, `agents/technical.md`, `agents/derivatives.md`
- Modify: `agents/sentiment.md`
- Create: `tests/fixtures/agent_examples/{universe_scout,funding_carry,pair_analyst,factor_analyst,sentiment,technical,derivatives}.json`
- Modify: `tests/test_role_files.py`, `tests/test_agent_conformance.py`

- [ ] **Step 1: Write failing role-file test** pinning the new roster and required sections (`# Title`, `## Mission`, `## Inputs`, `## How you think`, `## Output (return ONLY this JSON, no prose)` with a fenced ```json``` block + `## Example` — except the two doc-only roles).
- [ ] **Step 2: Write failing conformance test** validating each new fixture against its model (e.g. `sentiment.json` → `SentimentBatch`, `funding_carry.json` → list-of-`AnalystReport`), reusing the **worked `sentiment.json` fixture above verbatim**.
- [ ] **Step 3: Run them (expect FAIL — files missing).** `uv run pytest tests/test_role_files.py tests/test_agent_conformance.py -x` (expect FAIL).
- [ ] **Step 4: Write each `.md`** with the mandatory structure, adapting from `agents/archive/*` where the API map shows a fit (attribute the source). The Output block's ```json``` is a literal example of the model — for `sentiment.md` it is the worked `sentiment.json` above; for the analysts it is the `AnalystReport` field set (`pair_analyst` carries `signals.hedge_ratio`/`signals.adf_pvalue`, `funding_carry` carries `signals.signed_funding`/`signals.funding_interval_h`).
- [ ] **Step 5: Write each fixture** matching the `.md` example exactly; ensure every `SentimentSource.published_ts < as_of_ts` (use the worked fixture for `sentiment.json`).
- [ ] **Step 6: Run tests (expect PASS).** `uv run pytest tests/test_role_files.py tests/test_agent_conformance.py -x` (expect PASS).
- [ ] **Step 7: Commit.** `git commit -am "Phase 4: analyst roster prompts + fixtures + conformance (sentiment PIT enforced)"`

### Task 4.2: Debate + decision + execution + learning prompts

**Files:**
- Create: `agents/bull.md`, `agents/bear.md`, `agents/research_manager.md`
- Modify: `agents/trader.md`, `agents/reflector.md`
- Create: `tests/fixtures/agent_examples/{bull,bear,research_manager,trader,reflector}.json`

- [ ] **Step 1: Extend the role-file + conformance tests** to cover these five roles (Bull/Bear `stance` constraints, RM `ResearchPlan`, Trader bundle → **`TraderOutput`** using the worked `trader.json` above, Reflector `Lesson`).
- [ ] **Step 2: Run them (expect FAIL).** `uv run pytest tests/test_role_files.py tests/test_agent_conformance.py -x` (expect FAIL).
- [ ] **Step 3: Write/adapt the `.md` files.** Bull/Bear from archive (rebuttal requirement preserved); RM rates relative-value pairs explicitly (spec §10); Trader maps target weights → per-leg orders, **no sizing** (its Output block is the worked `trader.json` above; `management` empty list mandatory); Reflector keyed on alpha-vs-beta.
- [ ] **Step 4: Write the five fixtures** (use the worked `trader.json` for the Trader).
- [ ] **Step 5: Run tests (expect PASS).** `uv run pytest tests/test_role_files.py tests/test_agent_conformance.py -x` (expect PASS).
- [ ] **Step 6: Commit.** `git commit -am "Phase 4: debate/RM/trader/reflector prompts + fixtures (TraderOutput)"`

### Task 4.3: Deterministic doc-only roles (Neutrality Constructor, Risk Gate)

**Files:**
- Create: `agents/neutrality_constructor.md`, `agents/risk_gate.md`
- Modify: `tests/test_role_files.py`

- [ ] **Step 1: Write failing test** that these two files exist, have `# Title`/`## Mission`/`## Inputs`/`## How you think`, and are EXEMPT from the `## Output JSON` requirement (like `risk_manager`/`portfolio_manager` in the API map).
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_role_files.py -x` (expect FAIL).
- [ ] **Step 3: Write the two `.md`** describing the §8 optimizer and the non-overridable gate respectively; both state explicitly that final numbers are computed by code (`neutrality.py` / `risk_gate.py`), not the LLM.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_role_files.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 4: deterministic doc-only Neutrality Constructor + Risk Gate roles"`

### Task 4.4: SKILL.md weekly/daily orchestration

**Files:**
- Create: `SKILL.md`
- Test: `tests/test_skill_md.py`

- [ ] **Step 1: Write failing test** asserting `SKILL.md` has YAML frontmatter (`name`, `description`), a run-lock step, a due-check step per cadence, the weekly phase ladder, the daily phase ladder, and a mandatory reviewer-gate step before any execute step.
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_skill_md.py -x` (expect FAIL).
- [ ] **Step 3: Write `SKILL.md`** with the phase ladders below; each phase names exactly one CLI. Every CLI named is created by a task per the provenance table (P0 CLIs are lifted in P0; `gate_execute_cli.py` is created in Task 4.5; `reviewer_cli.py` in P5; `reflect_cli.py`/`promote_lesson_cli.py` in P6).

  **Weekly Selection Meeting (W1-W12):**
  - W1 run-lock: `runlock_cli.py acquire --owner weekly` (P0)
  - W2 due-check: `due_check.py state --loop weekly` (P0; cadence keys added in P3 Task 3.1)
  - W3 Universe Scout + preflight → `universe.json` (`scout_cli.py`, `preflight.py` — P0)
  - W4 parallel analysts (Carry, Pair, Factor, Sentiment deep, Technical, Derivatives) → `analyst_reports.json`, `sentiment.json`, geometry
  - W5 adversarial debate (Bull/Bear) → debate plan
  - W6 Research Manager 5-tier ratings → `research.json`
  - W7 Neutrality Constructor (code: `neutrality.optimize_book` via `control_loop_cli.py --cadence weekly`) → `target_weights.json`
  - W8 Trader → `proposals.json`
  - W9 **Reviewer gate (MANDATORY): `reviewer_cli.py --cadence weekly --cycle N`** — `SystemExit(2)`/HALT if `passed` False
  - W10 risk gate + execute: `gate_execute_cli.py --cadence weekly --cycle N` (Task 4.5) → `report.json`
  - W11 Reflector → `record_lessons_cli.py` (P0), `promote_lesson_cli.py` (P6)
  - W12 release lock (always): `runlock_cli.py release --owner weekly` (P0)

  **Daily Rebalance Meeting (D1-D8):**
  - D1 run-lock: `runlock_cli.py acquire --owner daily` (P0); D2 due-check: `due_check.py state --loop daily` (P0)
  - D3 sentiment refresh + drift/z/funding/neutrality recompute → geometry, `sentiment.json`
  - D4 Neutrality Constructor (`control_loop_cli.py --cadence daily`) → `target_weights.json`
  - D5 Trader deltas → `proposals.json`
  - D6 **Reviewer gate (MANDATORY): `reviewer_cli.py --cadence daily --cycle N`** — HALT on fail
  - D7 risk gate + execute: `gate_execute_cli.py --cadence daily --cycle N` (Task 4.5) → `report.json`
  - D8 reflect (light) + release lock: `reflect_cli.py` (P6), `runlock_cli.py release --owner daily` (P0)

  **Between cycles:** `monitor_cli.py` (P3 Task 3.8) runs on a faster cron and trips HALT on drawdown / liq-distance / neutrality breach.

  Model dispatch: `settings.model_for(<role>, loop=<cadence>)` where `<cadence> ∈ {weekly, daily}` — both keys present in `Settings.loops` per Task 4.0. Stand-down contract: empty `management` list mandatory. `live` stays `false`.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_skill_md.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 4: SKILL.md weekly/daily orchestration with mandatory reviewer gate"`

### Task 4.5: Execute boundary CLI (`gate_execute_cli.py`)

**Files:**
- Create: `scripts/gate_execute_cli.py`
- Test: `tests/test_gate_execute_cli.py`

NET-NEW in this repo (the weekly repo's `gate_execute_cli.py` is only a template; the contract defines `gate_execute_step(..., loop=...)` but no `--cadence` CLI). This is the execute boundary the whole reviewer-gate design depends on; W10/D7 invoke it. The reviewer precondition is wired in P5 Task 5.4 (it imports `reviewer_gate_ok` and HALTs first).

- [ ] **Step 1: Write failing test** invoking `main(["--cadence","weekly","--cycle","1"])` loads `state/weekly/cycle/1/proposals.json`, calls `gate_execute_step(ex, settings, "state", "memory", now, 1, proposals, ..., loop="weekly")`, and writes `state/weekly/cycle/1/report.json`. A second test asserts `--cadence` is required and dispatches `loop=cadence` into `gate_execute_step`.
  ```python
  def test_gate_execute_cli_dispatches_cadence(tmp_path, monkeypatch, seeded_proposals):
      seen = {}
      def fake_step(ex, settings, sd, md, now, cyc, props, **kw):
          seen["loop"] = kw.get("loop")
          return {"executed": [], "dropped": []}
      monkeypatch.setattr("scripts.gate_execute_cli.gate_execute_step", fake_step)
      monkeypatch.chdir(tmp_path)
      from scripts.gate_execute_cli import main
      main(["--cadence", "weekly", "--cycle", "1"])
      assert seen["loop"] == "weekly"
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_gate_execute_cli.py -x` (expect FAIL: module missing).
- [ ] **Step 3: Minimal implementation** — adapt the weekly `gate_execute_cli.py`: argparse `--cadence {weekly,daily}` (required) and `--cycle int` (required); `load_settings()`; `FuturesExchange.from_settings`; load `proposals.json` (+ management/triggers/cancel_triggers) from `cycle_dir(cadence=cadence)`; call `gate_execute_step(..., loop=cadence)`; persist `report.json` under the same cadence root; `print(json.dumps(report, default=str))`.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_gate_execute_cli.py -x` (expect PASS).
- [ ] **Step 5: Lint + commit.** `uv run ruff check .` then `git commit -am "Phase 4: gate_execute_cli execute boundary (--cadence weekly/daily, loop dispatch)"`

**Acceptance criteria (Phase 4):** every roster file exists with the mandatory sections; every analyst/decision fixture validates against its contract model (incl. `trader.json` → `TraderOutput`); `model_for(loop="weekly"|"daily")` resolves; `gate_execute_cli.py` exists with `--cadence` and dispatches `loop=cadence`; the two doc-only roles are JSON-exempt; `SKILL.md` orders weekly W1-W12 and daily D1-D8 with the reviewer gate as a non-skippable stage before any execute, and every CLI it names is created by a task per the provenance table; Sentiment Analyst's `SentimentBatch` includes a `"MARKET"` report and all sources are point-in-time in the fixture.

**Test strategy (Phase 4):** Reuse `test_role_files.py` (section-structure assertions) and `test_agent_conformance.py` (fixture→model validation) patterns. `TraderOutput` validated against the worked `trader.json`. `test_settings_loops.py` proves `weekly`/`daily` resolve in `model_for`. Add `test_skill_md.py` parsing the frontmatter and asserting phase/CLI presence + reviewer-gate ordering. `test_gate_execute_cli.py` proves the `--cadence`→`loop` dispatch. Sentiment point-in-time asserted by a fixture test: every `published_ts < as_of_ts`.

---

## Phase 5 — Risk gate reuse + every-cycle Adversarial Code & Calc Reviewer + self-audit

**Depends on:** P4 artifacts (`target_weights.json`, `proposals.json`, `sentiment.json`, geometry) and the `gate_execute_cli.py` execute boundary (Task 4.5). **Reuses:** `risk_gate.evaluate` and its math (direction-agnostic), `is_crypto_perp`, `metrics.sharpe`, `costs.project_funding`, cluster-heat helpers.

**Goal:** Lift the direction-agnostic risk gate; build the every-cycle reviewer that re-derives every load-bearing number and HALTs on mismatch via a deterministic flag the execute step checks; extend `self_audit.py` invariants.

### Canonical reviewer-check checklist (the 17 `ReviewerCheck.name`s → task/test)

Every canonical name is implemented by exactly one `check_*` and tested both green (matched ground truth) and red (injected mismatch). This checklist is binding for the just-in-time expansion:

| # | Canonical `ReviewerCheck.name` | `check_*` fn | Task | Test (`tests/test_reviewer.py::`) |
|---|---|---|---|---|
| 1 | `dollar_residual_in_band` | `check_dollar_neutral` | 5.2 | `test_dollar_neutral_recomputed_from_legs` |
| 2 | `beta_residual_in_band` | `check_beta_neutral` | 5.2 | `test_beta_neutral_recomputed` |
| 3 | `btc_hedge_sizing` | `check_btc_hedge` | 5.2 | `test_btc_hedge_sizing_recomputed` |
| 4 | `deployment_floor_both_sides` | `check_deployment_floor` | 5.2 | `test_deployment_floor_both_sides` |
| 5 | `per_name_cap` | `check_caps` (per-name) | 5.2 | `test_per_name_cap` |
| 6 | `cluster_cap` | `check_caps` (cluster) | 5.2 | `test_cluster_cap` |
| 7 | `funding_sign` | `check_funding` (sign) | 5.3 | `test_funding_sign_short_credit` |
| 8 | `funding_amount` | `check_funding` (amount) | 5.3 | `test_funding_amount_matches_realized` |
| 9 | `pair_pnl_attribution` | `check_pair_pnl` | 5.3 | `test_pair_pnl_at_spread_level` |
| 10 | `pair_leg_hedge_ratio` | `check_pair_pnl` (hedge ratio) | 5.3 | `test_pair_legs_sized_by_hedge_ratio` |
| 11 | `rr_after_costs` | `check_rr_after_costs` | 5.3 | `test_rr_after_costs_ge_2` |
| 12 | `sharpe_annualization` | `check_sharpe_annualization` | 5.3 | `test_sharpe_daily_365_weekly_52` |
| 13 | `exchange_filter_compliance` | `check_exchange_filters` | 5.3 | `test_exchange_filter_min_notional` |
| 14 | `sentiment_range` | `check_sentiment` (range) | 5.4 | `test_sentiment_level_s_range` |
| 15 | `sentiment_cap_respected` | `check_sentiment` (cap) | 5.4 | `test_sentiment_cap_respected` |
| 16 | `sentiment_point_in_time` | `check_sentiment` (PIT) | 5.4 | `test_sentiment_sources_pit` |
| 17 | `crypto_only_universe` | `check_crypto_only` | 5.4 | `test_crypto_only_no_tokenized_stock` |

`review_cycle` ANDs all 17 into `ReviewerVerdict.passed`; `mismatches == [c.name for c in checks if not c.ok]`.

### Task 5.1: Risk gate lift + carry-visibility fix

**Files:**
- Modify: `futures_fund/risk_gate.py`
- Test: `tests/test_risk_gate.py`

**Protected-module justification (binding):** `risk_gate.py` is protected; the rule is *never weaken a limit/breaker*. Un-clamping funding only **un-hides a real credit** (a short genuinely receives positive funding) — it adds visible truth to a cost computation, it does not relax the RR≥2 floor, the liq-distance ≥2.5× floor, or min-notional. HOWEVER, un-hiding a credit **raises RR**, which could let a marginal trade pass the RR≥2 floor it previously failed under the clamp. Step 5 adds a monotonicity guard so the gate cannot be silently *weakened* by the flag: the flag may only make an *already-RR-passing* trade more attractive, never resurrect a trade that failed the floor purely on the clamp. (If product wants the credit to count toward the floor, that is a deliberate, tested change — but the default keeps the floor decision conservative.)

- [ ] **Step 1: Write failing test** that with `funding.unclamped_in_rr=True`, a short receiving positive funding sees a NEGATIVE funding cost (credit) in the `CostEstimate`, raising RR — i.e. the `max(0.0, funding)` clamp is overridden. With the flag off, behavior is the legacy clamp.
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_risk_gate.py::test_unclamped_funding_shows_credit -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation.** Plumb an `unclamped_funding` boolean through `_build_sized` (sourced from settings via the gate path; **no signature change to `to_trade_proposal`** — a new optional kwarg on the gate path only, per contract §2.3/§2.10); when set, drop the `max(0.0, funding)` clamp and keep the signed value from `project_funding`. **Do not weaken any limit/breaker** — this only un-hides a real credit (RR floor, liq-distance, min-notional unchanged).
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_risk_gate.py -x` (expect PASS).
- [ ] **Step 5: Add the RR-floor-monotonicity guard test.** A trade that FAILED RR≥2 under the clamp must NOT pass solely because the unclamp raised RR — assert the floor decision is unchanged for that trade (or, if product intends the credit to count, assert the explicit, documented threshold). This proves the protected-module rule (no breaker weakened) is satisfied.
  ```python
  def test_unclamp_does_not_resurrect_rr_floor_failure(gate, marginal_short_with_credit):
      # marginal trade: RR = 1.9 under clamp (FAIL); credit would push RR to 2.1
      clamped = gate.evaluate(marginal_short_with_credit, unclamped_funding=False)
      unclamped = gate.evaluate(marginal_short_with_credit, unclamped_funding=True)
      assert clamped.passed is False
      # default: the floor decision is NOT weakened by un-hiding the credit
      assert unclamped.passed is False
  ```
- [ ] **Step 6: Run it (expect PASS).** `uv run pytest tests/test_risk_gate.py -x` (expect PASS).
- [ ] **Step 7: Commit.** `git commit -am "Phase 5: lift risk gate, plumb unclamped funding (carry visible in RR), RR floor not weakened"`

### Task 5.2: Neutrality + hedge + deployment + cap checks (canonical names 1-6)

**Files:**
- Create: `futures_fund/reviewer.py`
- Test: `tests/test_reviewer.py`

These checks re-derive the §5/§8 residuals from ground truth and compare to the artifact (trickiest bit: the reviewer must NOT trust `TargetWeights`' own residual fields — it recomputes from legs+betas+notionals). They cover canonical names 1-6 (`dollar_residual_in_band`, `beta_residual_in_band`, `btc_hedge_sizing`, `deployment_floor_both_sides`, `per_name_cap`, `cluster_cap`).

- [ ] **Step 1: Write failing test** for `check_dollar_neutral`: a `TargetWeights` whose stated `dollar_residual_frac` matches the recomputed `dollar_residual(weights, notionals)/side_budget` → `ReviewerCheck(name="dollar_residual_in_band", ok=True)`; an artifact with a tampered `dollar_residual_frac` (claims in-band but legs are imbalanced) → `ok=False`. Add the matched+tampered pair for each of names 2-6.
  ```python
  from futures_fund.reviewer import check_dollar_neutral
  def test_dollar_neutral_recomputed_from_legs(make_tw, cfg):
      tw = make_tw([("BTC/USDT:USDT","long",6000.0), ("ETH/USDT:USDT","short",4000.0)])
      tw.dollar_residual_frac = 0.0           # tampered claim
      chk = check_dollar_neutral(tw, cfg)
      assert chk.name == "dollar_residual_in_band"
      assert chk.ok is False                    # real residual = (6000-4000)/10000 = 0.20 > band
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_reviewer.py::test_dollar_neutral_recomputed_from_legs -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** of `check_dollar_neutral` (`dollar_residual_in_band`), `check_beta_neutral` (`beta_residual_in_band`), `check_btc_hedge` (`btc_hedge_sizing`), `check_deployment_floor` (`deployment_floor_both_sides`), `check_caps` (emits BOTH `per_name_cap` and `cluster_cap` checks) — each recomputes from legs/betas/notionals using `neutrality.dollar_residual`/`beta_residual`/`size_btc_hedge` and compares to the stated field within `cfg`-driven bands; emit `ReviewerCheck` with `expected`/`actual`.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_reviewer.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 5: reviewer checks 1-6 neutrality/hedge/deployment/caps (re-derived from ground truth)"`

### Task 5.3: Funding sign/amount, pair PnL+hedge-ratio, RR-after-costs, Sharpe annualization, exchange filters (canonical names 7-13)

**Files:**
- Modify: `futures_fund/reviewer.py`
- Test: `tests/test_reviewer.py`

- [ ] **Step 1: Write failing tests** (per the checklist rows 7-13): `check_funding` emits `funding_sign` (a short with positive funding shows a positive realized credit) AND `funding_amount` (= `funding_intervals.realized_funding(...)`); `check_pair_pnl` emits `pair_pnl_attribution` (PnL at spread level) AND `pair_leg_hedge_ratio` (legs sized by `hedge_ratio`); `check_rr_after_costs` (`rr_after_costs`) reuses `risk_gate._reward_risk`, RR≥2; `check_sharpe_annualization` (`sharpe_annualization`) — daily→365, weekly→52 (NOT the inherited 2190); `check_exchange_filters` (`exchange_filter_compliance`) — sub-`min_notional`/tick/step legs flagged.
- [ ] **Step 2: Run them (expect FAIL).** `uv run pytest tests/test_reviewer.py -k "funding or pair or rr_after or sharpe or exchange" -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation.** Each `check_*` re-derives from the realism primitives (P0): `realized_funding`, `Spread.realized_pnl`, `_reward_risk`, `metrics.sharpe(..., periods_per_year=365|52)`, `SymbolSpec` filters. Tolerances from `cfg`/`reviewer.tolerance` (1e-6). `check_funding` and `check_pair_pnl` each return TWO `ReviewerCheck`s (sign+amount; attribution+hedge-ratio).
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_reviewer.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 5: reviewer checks 7-13 funding(sign+amount)/pair(pnl+hedge)/RR/Sharpe/exchange-filter"`

### Task 5.4: Sentiment range/cap/point-in-time + crypto-only + `review_cycle` + gate flag (canonical names 14-17)

**Files:**
- Modify: `futures_fund/reviewer.py`
- Create: `scripts/reviewer_cli.py`
- Modify: `scripts/gate_execute_cli.py` (wire the reviewer precondition created in Task 4.5)
- Test: `tests/test_reviewer.py`, `tests/test_gate_execute_cli.py`

The hard-veto guard is the trickiest integration: `review_cycle` ANDs all 17 checks into `ReviewerVerdict.passed`; `reviewer_gate_ok` reads the persisted flag and the execute CLI raises `SystemExit(2)` if absent/false.

- [ ] **Step 1: Write failing tests** for: `check_sentiment` — emits `sentiment_range` (score round-trips `level↔s`), `sentiment_cap_respected` (`|Δw|≤cap` between `target_before`/`target_after`), and `sentiment_point_in_time` (every source `published_ts < as_of_ts`); `check_crypto_only` (`crypto_only_universe`) reuses `is_crypto_perp`; `review_cycle` returns `passed=False` and `"sentiment_cap_respected" in mismatches` when a sentiment tilt exceeds the 25% cap; `reviewer_gate_ok` returns False when `reviewer.json` is missing.
  ```python
  def test_review_cycle_halts_on_sentiment_cap_breach(state_dir, before, after_over_cap, ...):
      v = review_cycle(state_dir, memory_dir, cycle=1, cadence="weekly",
                       target=after_over_cap, geometries=..., spreads=[], sentiment=...,
                       cfg=cfg, returns=None)
      assert v.passed is False
      assert "sentiment_cap_respected" in v.mismatches
  ```
- [ ] **Step 2: Run them (expect FAIL).** `uv run pytest tests/test_reviewer.py -k "sentiment or crypto or review_cycle or gate_ok" -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** of `check_sentiment` (returns `sentiment_range`, `sentiment_cap_respected`, `sentiment_point_in_time`), `check_crypto_only` (`crypto_only_universe`), `review_cycle` (AND of all 17 checks, `mismatches == [c.name for c in checks if not c.ok]`), and `reviewer_gate_ok` (reads persisted `ReviewerVerdict.passed`; missing/false → False). Write `reviewer_cli.py`: `--cadence --cycle N`, persist `reviewer.json` under `cycle_dir(cadence=cadence)`, `print(json.dumps(...))`, `SystemExit(2)` if not `passed`.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_reviewer.py -x` (expect PASS).
- [ ] **Step 5: Wire the execute gate** — extend `gate_execute_cli.py` (created Task 4.5) to call `reviewer_gate_ok(cycle_dir(cadence=cadence))` FIRST and `SystemExit(2)` if false (mandatory non-skippable stage, spec §10/§12). Add a test that execute HALTs when `reviewer.json` is absent.
  ```python
  def test_execute_halts_without_reviewer_verdict(tmp_path, monkeypatch, seeded_proposals):
      monkeypatch.chdir(tmp_path)  # no reviewer.json written
      from scripts.gate_execute_cli import main
      import pytest
      with pytest.raises(SystemExit) as e:
          main(["--cadence", "weekly", "--cycle", "1"])
      assert e.value.code == 2
  ```
- [ ] **Step 6: Run it (expect PASS).** `uv run pytest tests/test_reviewer.py tests/test_gate_execute_cli.py -x` (expect PASS).
- [ ] **Step 7: Commit.** `git commit -am "Phase 5: reviewer checks 14-17, review_cycle HALT flag (all 17), execute gate wired"`

### Task 5.5: Extended self-audit invariants

**Files:**
- Modify: `futures_fund/self_audit.py`
- Test: `tests/test_self_audit.py`

> **NOTE — invariant vocabulary is intentionally separate (not a bug):** `self_audit.py`'s invariant names below (`both_sides_deployment_floor`, `funding_sign_correct`, `pair_legs_hedge_ratio_sized`, `sentiment_within_cap_range`, `no_tokenized_stock_leg`) are a **deliberately distinct, overlapping** vocabulary from the reviewer's canonical `ReviewerCheck.name`s (`deployment_floor_both_sides`, `funding_sign`, `pair_leg_hedge_ratio`, `sentiment_cap_respected`/`sentiment_range`, `crypto_only_universe`). `self_audit` is the *standing import-time invariant panel* (pure, no I/O, no cycle artifact); the reviewer is the *per-cycle artifact re-derivation*. A future worker must NOT "align" these by renaming one to match the other — they are two independent guards on overlapping properties. This note is binding.

- [ ] **Step 1: Write failing test** asserting `run_self_audit()["checks"]` contains the named invariants `dollar_residual_in_band`, `beta_residual_in_band`, `both_sides_deployment_floor`, `funding_sign_correct`, `pair_legs_hedge_ratio_sized`, `sentiment_within_cap_range`, `no_tokenized_stock_leg` — and `ok` is True for a conformant synthetic book, False for a deliberately broken one.
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_self_audit.py -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** — extend `_checks()` (pure-import, no I/O) with the seven invariants, reusing `neutrality.dollar_residual`/`beta_residual`, `funding_intervals.realized_funding`, `is_crypto_perp`, `conviction_tilt` cap logic on small synthetic inputs.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_self_audit.py -x` (expect PASS). Then `uv run python scripts/self_audit_cli.py` → `SELF-AUDIT: OK`.
- [ ] **Step 5: Commit.** `git commit -am "Phase 5: extend self_audit with neutrality/funding/pair/sentiment/crypto-only invariants (distinct vocabulary)"`

**Acceptance criteria (Phase 5):** every canonical `ReviewerCheck.name` (the 17 in the contract — see the checklist table) is implemented by exactly one `check_*` and tested both green and red; `review_cycle` ANDs them into `passed`; a single mismatch sets `passed=False` and HALTs the execute CLI with `SystemExit(2)`; risk gate shows carry as a credit under `unclamped_in_rr` **without weakening the RR≥2 floor** (monotonicity guard test green); `self_audit` covers the seven new invariants (a distinct-but-overlapping vocabulary, by design) and prints `SELF-AUDIT: OK`.

**Test strategy (Phase 5):** Each `check_*` has a matched (ok) and a tampered (fail) fixture per the 17-row checklist so the reviewer is proven to catch a real bug, not just echo the artifact. The HALT path is tested at the CLI boundary (`SystemExit(2)` on missing/false `reviewer.json`). The risk-gate monotonicity guard proves the unclamp never resurrects an RR-floor failure. Self-audit tested by asserting exact invariant names present and `ok=True` on conformant inputs, `ok=False` on a deliberately broken book.

---

## Phase 6 — Self-improvement loop re-keyed on ALPHA vs BTC-beta

**Depends on:** P5 reviewer verdicts + per-leg neutrality residuals; the weekly repo's journal/lessons/flat_journal/shadow substrate + `reflect_cli.py`/`promote_lesson_cli.py` templates. **Reuses:** `lessons.py`, `flat_journal.py`, `shadow.py`, `hitrate.py`, `improvement.py`, `scorecard.py`, `graduation.py`, `vendor/overfit_detector.py`. **Ported here (not pre-existing in this repo):** `journal.py` — the two-phase machinery is lifted/adapted faithfully from the weekly `futures_fund/journal.py` (verify+merge, exactly as Phase 3 did for `scheduling.py`) and re-keyed on `(cycle, symbol, direction)`. (The remaining `lessons.py`/`flat_journal.py`/`shadow.py`/`hitrate.py`/`improvement.py`/`scorecard.py` are likewise ported by their respective tasks when first needed.)

**Goal:** Re-key lessons/journal on alpha (return net of BTC-beta) rather than raw return; add the new lesson dimensions; keep the DSR-gated promotion; wire the self-healing code loop; re-point the improvement panel at neutral KPIs (including `reviewer_veto_rate` and `alpha_sharpe_trend`).

### Task 6.1: Alpha-vs-beta journal outcome accessor

**Files:**
- Create: `futures_fund/journal.py` (port/adapt from the weekly `futures_fund/journal.py`, verify+merge — re-keyed on `(cycle, symbol, direction)`; the file does not exist in this repo at the Phase 6 base commit)
- Test: `tests/test_journal_alpha.py`

`Decision` is `ConfigDict(extra="allow")`, so the new fields round-trip already. To make this a **genuine TDD step with a real behavioral assertion** (not a trivially-true `extra="allow"` echo), the production change is a **typed accessor** `alpha_outcome(decision) -> AlphaOutcome` that reads + validates the six alpha-vs-beta fields and raises on a missing/ill-typed field — the test below fails until that accessor exists.

- [ ] **Step 1: Write failing test** that `alpha_outcome(decision)` returns a typed `AlphaOutcome` exposing `alpha_return`, `beta_contribution`, `pair_cointegrated_at_exit`, `funding_thesis_matched`, `neutrality_in_band`, `sentiment_helped`; raises `KeyError`/`ValidationError` if a field is absent after `patch_outcome`; and `append_decision` stays idempotent per `(cycle, symbol, direction)`.
  ```python
  from futures_fund.journal import append_decision, patch_outcome, alpha_outcome
  def test_alpha_outcome_typed_accessor(tmp_path):
      append_decision(tmp_path, cycle=1, symbol="BTC/USDT:USDT", direction="long", payload={...})
      patch_outcome(tmp_path, cycle=1, symbol="BTC/USDT:USDT", direction="long",
                    outcome={"alpha_return": 0.012, "beta_contribution": -0.003,
                             "pair_cointegrated_at_exit": True, "funding_thesis_matched": True,
                             "neutrality_in_band": True, "sentiment_helped": False})
      ao = alpha_outcome(_load_one(tmp_path, 1, "BTC/USDT:USDT", "long"))
      assert ao.alpha_return == 0.012 and ao.beta_contribution == -0.003
      assert ao.sentiment_helped is False
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_journal_alpha.py -x` (expect FAIL: `ImportError: alpha_outcome`).
- [ ] **Step 3: Minimal implementation.** Port the two-phase storage machinery (`Decision`, `append_decision`, `patch_outcome`, `read_all_decisions`, `journal_file`) from the weekly `futures_fund/journal.py` (verify+merge, re-keyed on `(cycle, symbol, direction)`), then add an `AlphaOutcome` pydantic model (the six fields) and `alpha_outcome(decision) -> AlphaOutcome` that validates the decision's outcome dict. `patch_outcome` merges via `extra="allow"`; the accessor is the new behavior that would fail without code. (DRY — lift the weekly two-phase machinery for storage rather than re-inventing it; keep the public surface minimal — no speculative readers beyond what tests/consumers exercise.)
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_journal_alpha.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 6: typed alpha_outcome accessor over journal outcome fields (alpha vs BTC-beta)"`

### Task 6.2: New lesson dimensions + DSR-gated promotion

**Files:**
- Modify: `futures_fund/lessons.py`
- Test: `tests/test_lessons_neutral.py`

To make this a **real TDD step** (not a "tag strings, no code change" fixture-only task), the production change is a **tag-aware retrieval filter**: `retrieve_lessons` must surface lessons whose `dimension` tag is in the new set under the polarity quota. The test fails until `score_lesson`/the retrieval path actually reads the new `dimension` tags.

- [ ] **Step 1: Write failing test** that `retrieve_lessons(query_tags=["cointegration_break"])` ranks a lesson tagged `dimension="cointegration_break"` above an untagged one (a behavioral filter, not a round-trip), that the new dimensions (`cointegration_break`, `carry_thesis_miss`, `neutrality_breach`, `sentiment_detract`) are honored under the polarity quota, and `statistically_promote` only validates a candidate when `dsr_pvalue >= 0.95`.
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_lessons_neutral.py -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** — extend `score_lesson`'s `query_tags` matching to read the `dimension` field (the genuine code change that makes the ranking test pass); reuse `append_lesson`/`statistically_promote` unchanged (DSR gate retained). The new dimensions are valid tag values the filter now recognizes.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_lessons_neutral.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 6: dimension-aware lesson retrieval (cointegration/carry/neutrality/sentiment); DSR gate retained"`

### Task 6.3: Neutral improvement-panel KPIs

**Files:**
- Modify: `futures_fund/improvement.py`
- Test: `tests/test_improvement_neutral.py`

All new functions mirror the existing `deployment_rate(state_dir, last_n)` signature (pure, read-only, take `state_dir`/`memory_dir` + a window). Exact numerators/denominators below (the non-obvious bits per the spec):

- `both_sides_deployment_rate(state_dir, last_n) -> float` — **numerator:** cycles in the last `last_n` where BOTH `deploy_long_frac ≥ floor` AND `deploy_short_frac ≥ floor`; **denominator:** `last_n` cycles present. Guards BOTH the all-cash and the one-sided ratchet (spec §12/§19).
- `pair_survival_rate(state_dir, last_n) -> float` — **numerator:** pairs still cointegrated at their **re-test** (ADF p < 0.05 at the next weekly re-test, read from `Spread.adf_pvalue_at_retest` in the journal outcome); **denominator:** total pairs that reached a re-test in the window. `= cointegrated_at_retest / total_retested`.
- `carry_capture_rate(state_dir, last_n) -> float` — **numerator:** Σ `realized_funding` (signed, from `funding_intervals.realized_funding`) over carry legs; **denominator:** Σ `projected_funding` (signed, from `costs.project_funding` at entry). `= realized / projected`; clamp the denominator away from 0 and return `nan`/`None` when no carry legs (skip, don't divide by zero).
- `sentiment_hit_rate(memory_dir, last_n) -> float` — **numerator:** sentiment-aligned legs whose `alpha_return > 0` (sentiment said long/positive AND the leg made alpha, or said short/negative AND the short made alpha); **denominator:** legs where sentiment took a non-neutral stance (`|s| > 0`). `= sentiment_correct / sentiment_nonneutral`.
- `reviewer_veto_rate(state_dir, last_n) -> float` — **numerator:** cycles whose persisted `ReviewerVerdict.passed is False`; **denominator:** cycles with a `reviewer.json` in the window. `= vetoed / reviewed` (spec §18 process KPI).
- `alpha_sharpe_trend(state_dir, window) -> float` — rolling **alpha**-Sharpe (return net of BTC-beta, daily ×365) slope over `window` (spec §12/§18); reuse `return_trend`'s slope machinery on the alpha series.

- [ ] **Step 1: Write failing test** for each function above, computed from seeded `state/<cadence>/cycle/*/report.json` + `reviewer.json` + journal outcomes; assert exact hand-computed values (e.g. `carry_capture_rate` with realized 12.0 / projected 10.0 → 1.2; `reviewer_veto_rate` with 1 veto in 4 reviewed → 0.25; `pair_survival_rate` 3 cointegrated / 4 retested → 0.75).
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_improvement_neutral.py -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** — add the six pure read-only functions (signatures above) and fold them into `improvement_panel`, reusing `deployment_rate`/`return_trend`/`corpus_health` patterns. Guard BOTH ratchets (all-cash AND one-sided) by requiring both `deploy_long_frac` and `deploy_short_frac` ≥ floor.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_improvement_neutral.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 6: neutral improvement KPIs (both-sides deployment, pair-survival, carry-capture, sentiment hit-rate, reviewer-veto, alpha-Sharpe trend)"`

### Task 6.4: Scorecard injection + graduation walk-forward gate + self-healing loop

**Files:**
- Modify: `futures_fund/scorecard.py`, `futures_fund/graduation.py`, `futures_fund/repair.py`
- Create: `scripts/reflect_cli.py`, `scripts/promote_lesson_cli.py`
- Test: `tests/test_scorecard_neutral.py`

- [ ] **Step 1: Write failing tests** that `build_scorecard` includes `alpha_sharpe_trend` + the new neutral KPIs (incl. `reviewer_veto_rate`) and keeps two-sided warnings (an under-deployment accelerator AND a drawdown brake); `graduation_verdict` requires `walk_forward_required` OOS pass before trusting a sleeve-param change; the self-healing loop logs to `memory/repair-journal.md` and refuses to weaken a protected module.
- [ ] **Step 2: Run them (expect FAIL).** `uv run pytest tests/test_scorecard_neutral.py -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** — extend `build_scorecard` warnings (reuse the two-sided pattern; inject `alpha_sharpe_trend` + `reviewer_veto_rate`), add the OOS-required gate to `graduation_verdict`, and the repair-journal append in `repair.py`. Write `reflect_cli.py` (builds `reflection_input.json`, dispatches reflector, calls `record_lessons_cli.py`) and `promote_lesson_cli.py` (DSR-gated promotion) from the weekly templates.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_scorecard_neutral.py -x` (expect PASS). Then full suite + `uv run python scripts/self_audit_cli.py`.
- [ ] **Step 5: Commit.** `git commit -am "Phase 6: alpha scorecard (+veto-rate, alpha-Sharpe trend), walk-forward graduation gate, self-healing repair loop"`

**Acceptance criteria (Phase 6):** journal exposes a typed `alpha_outcome` accessor over the alpha-vs-beta fields; lesson retrieval is dimension-aware with DSR-gated promotion intact; improvement panel reports both-sides deployment, pair-survival, carry-capture, sentiment hit-rate, **reviewer-veto rate, and alpha-Sharpe trend**; scorecard warnings remain two-sided and inject the veto-rate + alpha-Sharpe trend; graduation requires OOS walk-forward before trusting a sleeve-param change; self-healing loop never weakens a protected module and logs every repair.

**Test strategy (Phase 6):** seed `state/`/`memory/` with real package helpers (`record_equity`, `append_decision`, `patch_outcome`, `append_lesson`) per the `tmp_path` convention; assert KPI values against hand-computed expectations (exact numerators/denominators above); assert two-sidedness of scorecard warnings (brake + accelerator both present); assert `alpha_outcome` raises on a missing field and `retrieve_lessons` ranks by `dimension`; assert `statistically_promote` blocks promotion below the DSR threshold and the graduation gate blocks without an OOS pass.

---

## Phase 7 — End-to-end paper run + KPI dashboard + full walk-forward validation

**Depends on:** all prior phases. **Reuses:** `metrics.*`, `equity_log.*`, `graduation.py`, `vendor/overfit_detector.py`, the `walk-forward-validation` skill, `runlock.single_flight`, the P0 `due_check`/`runlock_cli` CLIs, the P4 `gate_execute_cli.py`, the P5 `reviewer_cli.py`/`reviewer_gate_ok`.

**Goal:** Run a full weekly+daily loop end-to-end on a fake exchange, compute the KPI dashboard, and validate sleeve params out-of-sample on point-in-time data.

### Task 7.1: KPI dashboard

**Files:**
- Create: `futures_fund/dashboard.py`, `scripts/dashboard_cli.py`, `futures_fund/equity_log.py`
- Test: `tests/test_dashboard.py`

> **SCOPE NOTE (equity_log home):** the roadmap treats `equity_log.record_equity`/`equity_series`/`returns_series` as a PRE-EXISTING REUSE (Phase 3 daily-loop "Reuses:", line 165; Phase 6 test-strategy "real package helper", line 912; this phase's "Reuses: `equity_log.*`", line 918). In practice Phase 3 never materialized it, so the atomic, idempotent append-only equity-history log (`equity-history.jsonl` under `state/`) is created HERE in Task 7.1 and consumed by `dashboard.py` (and later `run_paper_cli.py` Step 7a). It is declared in this Create list — rather than smuggled in — so the artifact is accounted for. The module is minimal (`record_equity`/`equity_series`/`returns_series`) and clobbers nothing; downstream tasks (P7.3 Step 7a) import it as the reuse the roadmap intended.

- [ ] **Step 1: Write failing test** for `build_kpi_dashboard(state_dir, memory_dir)` returning: `no_losing_month` (fraction of calendar months positive — target 1.0), `daily_sharpe` (`metrics.sharpe(returns, periods_per_year=365)`), `both_sides_deployment_rate`, `neutrality_adherence` (fraction of cycles with residuals in band), `pair_survival`, `carry_capture`, `sentiment_hit_rate`, **`reviewer_veto_rate`** (reused from `improvement` Task 6.3), `max_drawdown`.
  ```python
  def test_dashboard_daily_sharpe_uses_365(state_dir, seeded_daily_returns):
      d = build_kpi_dashboard(state_dir, memory_dir)
      assert d["daily_sharpe"] == pytest.approx(sharpe(seeded_daily_returns, periods_per_year=365))
      assert 0.0 <= d["no_losing_month"] <= 1.0
      assert "reviewer_veto_rate" in d
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_dashboard.py -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** — `build_kpi_dashboard` reads equity series + cycle artifacts + journal outcomes; computes each KPI by reusing `metrics.sharpe`/`max_drawdown`, the `improvement` KPI functions (`both_sides_deployment_rate`, `pair_survival_rate`, `carry_capture_rate`, `sentiment_hit_rate`, `reviewer_veto_rate`), and the reviewer verdict history. `dashboard_cli.py` prints JSON + a markdown table.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_dashboard.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 7: KPI dashboard (no-losing-month, daily Sharpe x365, neutrality/pair/carry/sentiment/reviewer-veto)"`

### Task 7.2: Walk-forward validation harness (point-in-time)

**Files:**
- Create: `futures_fund/walk_forward.py`, `scripts/walk_forward_cli.py`
- Test: `tests/test_walk_forward.py`

**Point-in-time provenance (spec §11, §15 — binding):** walk-forward inputs MUST be PIT: historical klines/funding come from **`data.binance.vision` archives** (immutable daily/monthly dumps), NOT a live `exchange.py` pull that would leak post-decision revisions. `exchangeInfo` gives only currently-listed symbols → a **survivorship caveat**: the harness records which symbols were delisted/absent in the test window and excludes any symbol whose first archive date is after the IS window start (no look-ahead into later-listed names). The `returns_by_param` inputs are tagged with their archive source date so the test can assert provenance.

- [ ] **Step 1: Write failing test** that (a) `walk_forward.validate(param_grid, returns_by_param)` produces time-series-aware (no-leak) IS/OOS splits and gates a param change on OOS DSR (reuse `deflated_sharpe_pvalue`/`vendor.deflated_sharpe_ratio` with `num_trials=len(grid)`), refusing an in-sample-only grid winner; and (b) `load_pit_returns(symbol, start, end)` sources from a `data.binance.vision`-shaped archive fixture and raises/excludes a symbol whose first archive date is after `start` (survivorship guard).
  ```python
  def test_walk_forward_inputs_are_point_in_time(archive_fixture):
      # symbol "NEWCOIN" first archive date is AFTER the IS window start -> excluded (no look-ahead)
      rets = load_pit_returns("NEWCOIN/USDT:USDT", start="2025-01-01", end="2025-03-01",
                              archive_root=archive_fixture)
      assert rets is None  # survivorship caveat: later-listed name excluded from the window
  ```
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_walk_forward.py -x` (expect FAIL).
- [ ] **Step 3: Minimal implementation** — `load_pit_returns` reads the `data.binance.vision` archive layout (offline fixture in tests), enforcing the first-archive-date survivorship guard; `validate` invokes the `walk-forward-validation` skill's split logic; for each param compute OOS Sharpe; rank by OOS, apply DSR with `num_trials` = grid size; return the verdict (`promote`/`reject`). If a live PIT archive is unavailable in CI, the test uses a recorded archive fixture and the simplification is stated in the docstring.
- [ ] **Step 4: Run tests (expect PASS).** `uv run pytest tests/test_walk_forward.py -x` (expect PASS).
- [ ] **Step 5: Commit.** `git commit -am "Phase 7: walk-forward OOS validation (PIT data.binance.vision inputs, survivorship guard, DSR gate)"`

### Task 7.3: End-to-end paper run

**Files:**
- Create: `scripts/run_paper_cli.py`
- Test: `tests/test_end_to_end.py`

The orchestration is decomposed below into its natural sub-task seams (lock+due → cadence step → reviewer gate → execute → equity → reflect), each naming the exact CLI/function so the just-in-time expansion has clean boundaries. The `gate_execute` reference is resolved to the P4 `gate_execute_cli.py` / `gate_execute_step(..., loop=cadence)`.

- [ ] **Step 1: Write failing test** that a full weekly→daily run on a fake `FuturesExchange` (injected) produces a dollar+beta-neutral book within bands, a green `ReviewerVerdict`, a non-empty `report.json`, and an updated equity point — all under a single run lock.
- [ ] **Step 2: Run it (expect FAIL).** `uv run pytest tests/test_end_to_end.py -x` (expect FAIL).
- [ ] **Step 3a: Lock + due (per cadence).** `run_paper_cli.py` acquires `runlock.single_flight(owner="paper")`, then for each cadence (weekly first) calls `control_loop.cadence_due(state_dir, now, cadence)`; SKIP→continue, FRESH/RETRY→proceed. Test the lock is held and a served candle SKIPs.
- [ ] **Step 4a: Run cadence step.** Dispatch `weekly_selection` / `daily_rebalance` (via `control_loop_cli.main(["--cadence", cadence, "--cycle", str(n)])`) → persists `target_weights.json` under `state/<cadence>/cycle/<n>/`. Test the artifact exists.
- [ ] **Step 5a: Reviewer gate.** `reviewer_cli.main(["--cadence", cadence, "--cycle", str(n)])` → `reviewer.json`; if `not reviewer_gate_ok(cycle_dir(cadence=cadence))` → HALT (`SystemExit(2)`). Test a tampered book HALTs here.
- [ ] **Step 6a: Execute.** `gate_execute_cli.main(["--cadence", cadence, "--cycle", str(n)])` → `gate_execute_step(..., loop=cadence)` → `report.json`. Test `report.json` non-empty.
- [ ] **Step 7a: Equity + reflect.** `equity_log.record_equity(state_dir, now, equity, n)`; then `reflect_cli.main([...])` (light on daily). Test an equity point was appended.
- [ ] **Step 8: Wire the driver** so both cadences are serialized **weekly-first** under the one lock; `live` stays `false`. Run `uv run pytest tests/test_end_to_end.py -x` (expect PASS), then full `uv run pytest` + `uv run python scripts/self_audit_cli.py` → `SELF-AUDIT: OK`.
- [ ] **Step 9: Commit.** `git commit -am "Phase 7: end-to-end paper run driver (lock→due→step→reviewer→execute→equity→reflect, weekly-first)"`

**Acceptance criteria (Phase 7):** end-to-end run yields a neutral book within bands, a green reviewer verdict, a persisted execution report and equity point; dashboard reports all process KPIs (both-sides deployment, neutrality adherence, pair-survival, carry-capture, sentiment hit-rate, **reviewer-veto rate**) + no-losing-month + daily Sharpe×365 + max drawdown; walk-forward gate rejects an in-sample-only winner, only promotes an OOS+DSR survivor, and proves its inputs are point-in-time (`data.binance.vision` archive provenance + survivorship guard); the e2e driver serializes weekly-first under one lock with each stage naming its exact CLI/function; full suite green; `SELF-AUDIT: OK`.

**Test strategy (Phase 7):** inject a fake exchange (per the `FuturesExchange`/`from_settings` pattern) so the e2e test is deterministic and offline. Dashboard KPIs asserted against hand-seeded equity/journal/cycle fixtures (e.g. a known monthly series → exact `no_losing_month`; a seeded `reviewer.json` history → exact `reviewer_veto_rate`). Walk-forward tested with a synthetic param grid where one param wins in-sample but fails OOS (gate rejects it) AND a `data.binance.vision`-shaped archive fixture proving PIT/ survivorship handling. The e2e test asserts the reviewer gate is actually exercised (book is neutral, `ReviewerVerdict.passed is True`, and a tampered book HALTs at Step 5a).

---

## Cross-phase invariants (binding for every just-in-time expansion)

- **Cadence roots (single source of truth):** `control_loop.cadence_due` → `scheduling.cycle_due(loop="weekly", tf_minutes=10080)` / `(loop="daily", tf_minutes=1440)`; the reused gate's root is `state/<loop>/cycle/*`, so **artifacts live under `state/<cadence>/cycle/<N>/`** and `cycle_io.cycle_dir(cadence=...)` writes to exactly that path. The due-gate reader and the artifact writer MUST agree on this one root — never `state/cycle/<cadence>/N`.
- **Sentiment ordering (§7.3):** `conviction_tilt`/`apply_conviction_tilts` run BEFORE `neutrality.project_neutral`; the reviewer's `sentiment_cap_respected` compares `target_before` vs `target_after`; every `SentimentSource.published_ts < as_of_ts` (`sentiment_point_in_time`).
- **Funding sign:** SIGNED everywhere new (`realized_funding`, `project_funding`); the only clamp (`risk_gate._build_sized`) is overridden by `funding.unclamped_in_rr` — un-hiding a real credit, **without weakening the RR≥2 floor** (monotonicity guard, Task 5.1 Step 5).
- **Reviewer flag:** `ReviewerVerdict.passed` (AND of all 17 canonical checks) is the deterministic flag `reviewer_gate_ok` reads; `gate_execute_cli.py` raises `SystemExit(2)` if absent/false (mandatory non-skippable stage).
- **Reviewer vs self-audit vocabularies are intentionally distinct** (overlapping properties, two independent guards) — do not "align" them by renaming (Task 5.5 note).
- **`TargetWeights` residual fields** are exactly what `reviewer.check_*` and `self_audit` re-derive against `NeutralityConfig` bands — never trusted blindly, always recomputed.
- **Model dispatch:** `Settings.loops` carries `weekly`/`daily` keys (Task 4.0) so `model_for(role, loop=cadence)` resolves; per-agent `agent_models` still wins for deciding agents.
- **Trader contract:** `TraderOutput` (Task 4.0) is the conformance target for `trader.json` — `proposals: list[AgentProposal]` + `management/triggers/cancel_triggers: list[dict]` (empty `management` = stand-down).
- **Protected modules** (`risk_gate, executor, exits, consolidation, policy, liquidation, sizing, cycle`) never weakened; new logic lives in new non-protected modules; `monitor_cli`/reviewer ADD trips, never relax a limit; `live` stays `false` forever.
- **Point-in-time data:** any backtest/walk-forward sources `data.binance.vision` archives with a survivorship guard (Task 7.2); prompts enforce PIT in sentiment + analysts.
- **CLI provenance:** every CLI named in the SKILL.md ladders is created by a task per the provenance table (P0 lifts the operational CLIs; P3 creates `monitor_cli.py`/`control_loop_cli.py`; P4 creates `gate_execute_cli.py`; P5 `reviewer_cli.py`; P6 `reflect_cli.py`/`promote_lesson_cli.py`; P7 `dashboard_cli.py`/`walk_forward_cli.py`/`run_paper_cli.py`).
- **New deps** to add to `pyproject.toml`: `statsmodels`, `scikit-learn`, and `cvxpy` (or keep `scipy.optimize`).

---

## Just-in-time planning (reiterated, per the writing-plans skill)

Each of Phases 3-7 above is a **task-level roadmap**. Before starting a phase, expand it into its own full bite-sized TDD plan document (`docs/superpowers/plans/2026-06-11-phase<N>-<slug>.md`), like P0/P1/P2, with every failing-test/run/implement/run/commit ladder spelled out — but **only after its predecessor phase has fully landed** (all tasks committed, `uv run pytest` green, `SELF-AUDIT: OK`). The interfaces, file structure, acceptance criteria, exact per-file pytest commands, and test strategy fixed here make that expansion mechanical and placeholder-free.

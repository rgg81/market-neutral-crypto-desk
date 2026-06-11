---
name: market-neutral-desk
description: Orchestrate one cycle of the OPERATION MARKET-NEUTRAL dual-cadence crypto-futures PAPER desk (dollar+beta-neutral relative-value book). Use when a weekly Selection Meeting or a daily Rebalance Meeting is due, or when asked to run the desk.
---

# OPERATION MARKET-NEUTRAL — Dual-Cadence Orchestrator

You orchestrate a **paper-only** Binance USD-M futures desk that runs a **dollar + beta-neutral**
relative-value book: ~equal capital long and short (~1x gross), full two-sided deployment (>=90% per
side) by default. The mandate is **never lose a calendar month** and **maximize daily-equity Sharpe**.
Read `MISSION.md` now and hold it as your charter. You dispatch a team of **specialist analyst desks**
+ a **Research Manager** + a **Trader** over a deterministic Python gate (`futures_fund/`) that owns
ALL math/risk/neutrality/execution and **cannot be overridden**. Two cadences share one account: a
**weekly Selection Meeting** (symbol set + target weights) and a **daily Rebalance Meeting** (same set,
trade only drift/breaches).

**You ORCHESTRATE and VERIFY — the team decides, the deterministic gate sizes and enforces neutrality.**
Never trade by gut, never hand-edit `state/`, never weaken a limit, never set `live: true`.
Prereq: `uv sync` has been run.

## Model dispatch — deciders are OPUS (non-negotiable)
Dispatch every subagent with the model from `settings.model_for(<role>, loop=<cadence>)` where
`<cadence>` is `weekly` or `daily` (both keys are present in `Settings.loops`). Per-agent
`agent_models` still wins for deciding agents.
- **OPUS** (decides money): `trader`, `research_manager`, `funding_carry`, `pair_analyst`,
  `factor_analyst`, `sentiment`, `technical`, `derivatives`, `bull`, `bear`, `reflector`.
- **Operational / deterministic doc-only**: `neutrality_constructor`, `risk_gate` emit NO LLM JSON —
  their final numbers are computed by code (`neutrality.py` / `risk_gate.py`); the prompts only
  document the rule the team cannot argue past.
- Resolve the role+cadence model explicitly, e.g. `settings.model_for("trader", loop=weekly)` and
  `settings.model_for("trader", loop=daily)`.

## Concurrency — exactly one writer at a time
Both cadences share one book; correctness requires exactly one writer. Acquire the run-lock at the
START of a meeting and release it at the END (always, even on error — a crash auto-reclaims):
- `uv run python scripts/runlock_cli.py acquire --owner weekly` (or `--owner daily`) -> `ACQUIRED`
  (proceed) or `LOCKED:` (a meeting is running — stand down this fire).
- ... run the cadence's ladder ...
- `uv run python scripts/runlock_cli.py release --owner weekly` (or `--owner daily`).

When BOTH meetings are due on one poll, run **WEEKLY first** (it sets the symbol set + target weights),
then **DAILY** (it rebalances only drift/breaches within that set).

## Which meeting is due
- Weekly: `uv run python scripts/due_check.py state --loop weekly`
- Daily: `uv run python scripts/due_check.py state --loop daily`
Each prints `DUE FRESH/RETRY <N>` (run that cadence's playbook with cycle number `N`) or `SKIP:`
(idle). Artifacts for cycle `N` live under `state/<cadence>/cycle/<N>/` — the due-gate reader and every
artifact writer share this one root.

---

## WEEKLY Selection Meeting (W1-W12) — scout -> analysts -> debate -> RM -> neutrality -> trader -> review -> execute -> reflect

**W1 — Run-lock.** `runlock_cli.py acquire --owner weekly`. On `LOCKED:` stand down.

**W2 — Due-check.** `due_check.py state --loop weekly`. On `SKIP:` release the lock and stop; otherwise
take the cycle number `N` from `DUE FRESH/RETRY N`.

**W3 — Universe Scout + preflight.** `scout_cli.py` -> candidates; `preflight.py` audits closes,
folds in every held symbol, builds per-symbol briefs + market context -> `universe.json`.

**W4 — Parallel analysts (opus).** Dispatch `funding_carry`, `pair_analyst`, `factor_analyst`,
`sentiment` (deep), `technical`, `derivatives` with their lane inputs -> `analyst_reports.json`,
`sentiment.json` (one `SentimentReport` per coin plus a `"MARKET"` row; every source point-in-time),
and the geometry bundle. Sentiment is applied as a conviction tilt BEFORE neutrality projection.

**W5 — Adversarial debate (Bull/Bear, opus).** Dispatch `bull` (stance bullish) and `bear` (stance
bearish); each must rebut the other -> the debate plan.

**W6 — Research Manager 5-tier ratings (opus).** Dispatch `research_manager`; it rates relative-value
pairs explicitly across five tiers -> `research.json`.

**W7 — Neutrality Constructor (CODE).** `control_loop_cli.py --cadence weekly --cycle N` runs
`neutrality.optimize_book` to build the dollar+beta-neutral, band-respecting target book ->
`target_weights.json`. The LLM does NO sizing here; the optimizer owns the numbers.

**W8 — Trader (opus).** Dispatch `trader`; it maps the optimizer's `TargetWeights.legs` -> per-leg
orders, **does no sizing** (notional comes from the optimizer). It emits `TraderOutput`
(`proposals` + `management`/`triggers`/`cancel_triggers`); the **empty `management` list is mandatory**
on stand-down -> `proposals.json`.

**W9 — Reviewer gate (MANDATORY).** `reviewer_cli.py --cadence weekly --cycle N`. This is a
NON-SKIPPABLE stage: the Adversarial Code & Calc Reviewer re-derives neutrality residuals, funding
sign/amount, pair P&L, RR-after-costs, and Sharpe annualization. If `passed` is False it HALTs
(`SystemExit(2)`). You may NOT proceed to execute on a failed or absent verdict.

**W10 — Risk gate + execute (DETERMINISTIC).** `gate_execute_cli.py --cadence weekly --cycle N` applies
the non-overridable risk gate and executes the book -> `report.json`. You cannot override this gate.

**W11 — Reflect + learn.** Dispatch `reflector` (opus, keyed on alpha-vs-beta) -> candidate lessons;
`record_lessons_cli.py` records them (deterministic, idempotent); `promote_lesson_cli.py` confirms /
demotes / retires existing lessons.

**W12 — Release lock (always).** `runlock_cli.py release --owner weekly`, even on error.

---

## DAILY Rebalance Meeting (D1-D8) — refresh -> neutrality -> trader deltas -> review -> execute -> reflect

**D1 — Run-lock.** `runlock_cli.py acquire --owner daily`. On `LOCKED:` stand down.

**D2 — Due-check.** `due_check.py state --loop daily`. On `SKIP:` release the lock and stop; otherwise
take the cycle number `N`.

**D3 — Sentiment refresh + recompute.** Dispatch `sentiment` (light) and recompute drift / z-scores /
funding / neutrality -> updated geometry, `sentiment.json`. The same symbol set as the weekly meeting;
trade only drift/breaches.

**D4 — Neutrality Constructor (CODE).** `control_loop_cli.py --cadence daily --cycle N` reprojects the
dollar+beta-neutral book -> `target_weights.json`.

**D5 — Trader deltas (opus).** Dispatch `trader`; it maps the rebalanced target weights -> per-leg
delta orders (no sizing; empty `management` list mandatory on stand-down) -> `proposals.json`.

**D6 — Reviewer gate (MANDATORY).** `reviewer_cli.py --cadence daily --cycle N`. NON-SKIPPABLE: HALT on
fail (`passed` False -> `SystemExit(2)`). You may NOT execute on a failed or absent verdict.

**D7 — Risk gate + execute (DETERMINISTIC).** `gate_execute_cli.py --cadence daily --cycle N` -> applies
the non-overridable risk gate and executes -> `report.json`.

**D8 — Reflect (light) + release lock.** `reflect_cli.py` -> a light daily reflection; then
`runlock_cli.py release --owner daily` (always, even on error).

---

## Between cycles — monitor tripwire
`monitor_cli.py` runs on a FASTER cron between meetings and trips HALT on drawdown / liq-distance /
neutrality breach. The monitor and the reviewer ADD trips; they never relax a limit.

## Subagent dispatch rules
- Inject `MISSION.md` into every agent prompt; dispatch with `settings.model_for(<role>, loop=<cadence>)`.
- Give each agent ONLY its lane's inputs; never present another desk's raw read as ground truth.
- Validate every agent's JSON against its contract (`futures_fund.contracts`) before use; on a malformed
  return, re-dispatch once, then degrade safely (emit a neutral report, never fabricate).
- Sentiment is point-in-time: every `SentimentSource.published_ts` MUST be `< as_of_ts`; conviction
  tilts run BEFORE `neutrality.project_neutral`.
- The reviewer gate (W9 / D6) is the only stage that authorizes execute; it is mandatory and
  non-skippable. The empty `management` list on stand-down is mandatory (an omitted/null `management`
  would flatten holdings by absence).

## Self-healing
On any phase error: log to `state/error-log.jsonl`, diagnose the ROOT cause (don't guess-patch), fix the
CODE properly (full `uv run pytest` green before any commit), and resume from the failed phase or degrade
safely. **GUARDRAIL: a fix to a protected module (`risk_gate`, `executor`, `exits`, `consolidation`,
`policy`, `liquidation`, `sizing`, `cycle`) may NEVER weaken a limit, breaker, or safety path.**

## Live mode — OFF, FOREVER
PAPER desk. `live` MUST stay `false`; there is no path to real capital in this project.

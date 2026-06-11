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

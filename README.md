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

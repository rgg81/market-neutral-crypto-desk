# Market-Neutral Crypto Trading Desk

An **adversarial multi-agent, market-neutral trading desk** for Binance USD-M perpetual futures — **paper trading only**. It runs an equal-budget long/short book (≈ \$10k each side on a \$20k account), holds it **dollar- *and* beta-neutral** through a BTC hedge, and harvests relative-value, funding-carry, cross-sectional-factor, and sentiment edges across all market regimes. Every cycle is gated by an adversarial code-and-calculation reviewer with hard-veto power, and the desk learns from its own (cost-adjusted) results.

> ⚠️ **Paper trading only.** `live` is hard-wired to `false`; there is no order-placement code path. This is a research / educational project, **not financial advice**.

---

## Why it's interesting

- **Genuinely market-neutral, and it proves it.** The held book is reconciled to a dollar+beta-neutral target every cycle, with a BTC-perp hedge sizing residual beta to ~0. A standing self-audit invariant and a 17-check reviewer *enforce* it (a non-neutral or tampered book HALTs execution with exit code 2).
- **Realistic paper P&L.** A position ledger carried across cycles applies real Binance **maker/taker fees**, **depth-aware slippage**, and **funding settlement** (a short *receives* positive funding), then marks to the live mark price — so the equity curve, Sharpe, and drawdown are real, not a flat line.
- **"LLM proposes, code disposes."** LLM agents reason and debate; a deterministic, fully-tested Python spine owns *all* math, sizing, risk, and execution. The model can never argue past a risk limit.
- **Self-improving.** An anti-hindsight journal and DSR-gated lesson corpus, re-keyed on **alpha vs. BTC-beta** (not raw return), feed the next cycle's decisions.

---

## Architecture — two layers

```
                 ┌───────────────────── REASONING (LLM) ─────────────────────┐
                 │  agents/*.md  ·  orchestrated by SKILL.md (W1–W12 / D1–D8) │
   live Binance  │  Scout · Funding-Carry · Pair · Factor · Sentiment ·       │
   USD-M data ──▶│  Technical · Derivatives · Bull ⚔ Bear · Research Manager ·│
   (keyless)     │  Trader · Reflector                                        │
                 └───────────────┬───────────────────────────────────────────┘
                                 │ validated pydantic contracts
                 ┌───────────────▼────────────── DETERMINISTIC SPINE (futures_fund/) ────────────┐
                 │  scout → cycle-prep (geometries · sleeves · pairs)                             │
                 │       → neutrality.optimize_book  (dollar+beta neutral, HRP/shrinkage, hedge)  │
                 │       → reviewer.review_cycle     (17 checks · HARD VETO)                       │
                 │       → gate_execute              (paper fills; live=false)                     │
                 │       → account ledger            (fees · slippage · funding · mark-to-market)  │
                 │       → reflect / lessons         (alpha-keyed self-improvement)                │
                 └───────────────────────────────────────────────────────────────────────────────┘
```

- **Reasoning layer:** 14 markdown agent prompts (in `agents/`), each tied to a pydantic output contract. The adversarial Bull-vs-Bear debate is arbitrated by a Research Manager (inspired by [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)).
- **Deterministic spine** (`futures_fund/`): 42 modules + 5 alpha-sleeve modules, every public function unit-tested.

## How a cycle works

| Cadence | What it does |
|---|---|
| **Weekly Selection** (Thursdays) | Re-scan the liquid universe, re-run cointegration / factor / carry / sentiment, build the new neutral target book + weights, rotate positions paying real open/close costs. |
| **Daily Rebalance** (every day) | Keep the same symbol set; nudge back toward target only when drift exceeds a no-trade band (and hard-stop any cointegration-broken pair). No churn. |

Both cadences run under a single-flight lock and are gated on the served candle so they can never double-fire.

## The four alpha sleeves

| Sleeve | Edge |
|---|---|
| **Funding carry** | Long low/negative-funding, short high-positive-funding; the short *collects* funding. Signal bounded so it can't chase extreme/illiquid funding. |
| **Cointegration pairs** | First-class `Pair`/`Spread` objects: Engle-Granger / Johansen with rolling re-test + FDR correction, Ornstein-Uhlenbeck half-life, z-score entry ≥ 2 / exit ~ 0 / hard stop ≥ 3. |
| **Cross-sectional factor** | Rank the liquid cross-section by momentum / carry / low-vol; long top tercile, short bottom, inverse-vol weighted. |
| **Sentiment** | Point-in-time news/social sentiment as a bounded conviction tilt *and* a standalone L/S sleeve — never flips a leg's direction. |

Sleeves are risk-parity blended, then a single optimizer enforces dollar+beta neutrality, a ~90% deployment floor, per-name/cluster caps, and turnover-aware rebalancing (Ledoit-Wolf shrinkage + Hierarchical Risk Parity).

## Universe quality

Liquid, established names only: a Binance `onboardDate` listing-age floor, exclusion of extreme 24h movers, an order-book-depth floor, a 24h-volume floor, and a denylist for tokenized commodities (PAX Gold, etc.) — so new-listing pumps and gold/stock wrappers never enter the book.

---

## Quick start

```bash
uv sync                                              # install deps (Python 3.11)

# run one paper cycle (weekly Selection + daily Rebalance) on live Binance data
uv run python scripts/run_paper_cli.py

# read the KPI dashboard (no-losing-month, daily Sharpe, neutrality, fees/slippage/funding…)
uv run python scripts/dashboard_cli.py --format both

# run the standing self-audit invariant panel (neutrality, equity reconciliation, …)
uv run python scripts/self_audit_cli.py --state-dir state

uv run pytest -q                                     # the test suite
```

The full multi-agent debate path is driven by an orchestrator running `SKILL.md`; `run_paper_cli.py` runs the deterministic engine end-to-end (and is what the test suite and CI exercise).

## Project layout

```
futures_fund/      deterministic spine (optimizer, sleeves, cointegration, reviewer, account ledger, …)
agents/            14 LLM agent prompts (markdown) + their output contracts
scripts/           18 CLIs (scout, cycle-prep, control-loop, reviewer, gate-execute, dashboard, …)
tests/             65 test files
docs/superpowers/  the design spec, interface contract, and phased implementation plans
SKILL.md           the weekly/daily orchestration playbook
config.yaml        all tunables (capital, neutrality bands, caps, fees, universe filters, …)
```

## Status & testing

- **~690 passing tests**, ruff-clean. Every new behavior is TDD'd; the optimizer, reviewer, and P&L ledger have property/invariant tests.
- Built incrementally as spec → plan → TDD across ten phases, with adversarial multi-agent review at each step.

## Disclaimer

For research and education only. Paper trading on real market data with simulated execution. No live trading. Nothing here is financial advice. Use at your own risk.

---

*Inspired by [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). Built with [Claude Code](https://claude.com/claude-code).*

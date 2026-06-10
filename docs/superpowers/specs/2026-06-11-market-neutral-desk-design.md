# Market-Neutral Crypto Trading Desk — Design Spec

- **Status:** Approved (brainstorming complete) — ready for implementation planning
- **Date:** 2026-06-11
- **Project root:** `/home/roberto/crypto-trade-claude-code-market-neutral`
- **Mode:** Paper trading only (real Binance USD-M mainnet data, simulated execution)
- **Inspirations (different goals, reused infrastructure):** `TauricResearch/TradingAgents`, `/home/roberto/crypto-trade-claude-code` (daily), `/home/roberto/crypto-trade-claude-code-weekly` (weekly)

---

## 1. Mission

Build an **adversarial multi-agent, market-neutral crypto trading desk** on Binance USD-M perpetual futures (paper). Equal capital on both sides (~$10k long / ~$10k short). The desk profits from **relative value** (e.g. long BTC / short ETH when one structurally outperforms), **funding-rate carry**, **cross-sectional factors**, and **sentiment** — staying roughly neutral to the overall crypto market so it can be positive across all regimes.

**Primary success criterion:** *no losing calendar month.*
**Secondary objective:** *maximize Sharpe* on the daily equity series (annualized ×365), benchmark = cash (0).
**Hard requirements:** learn from mistakes and auto-improve; **every cycle** an agent reviews all code/calculations for zero bugs; realistic fees + slippage + funding; native cryptocurrencies only (no stocks, indexes, metals, gold coins).

---

## 2. Core decisions (locked)

| Topic | Decision |
|---|---|
| Capital | **$20k paper account, ~1× gross** → ~$10k long + ~$10k short notional |
| Deployment | **~90% floor each side** (≥ ~$9k/side deployed) + small dry-powder reserve (~$1k/side) for daily rebalancing / opportunistic adds |
| Neutrality | **Dollar + beta neutral** — equal long/short $ within a band **and** a BTC-perp hedge leg making Σ(wᵢ·βᵢ_BTC) ≈ 0; rolling β re-estimation |
| Alpha sleeves | **Four**: funding-carry · cointegration pairs · cross-sectional factor L/S · **sentiment**; risk-parity across sleeves |
| Sentiment use | **Conviction tilt + standalone sentiment factor sleeve**, as a **bounded shaper that never flips direction** |
| Universe | **Liquid large-caps** (~top 20–30 by 24h volume, min-depth floor), crypto-only USD-M perps |
| Success KPI | **No losing month** (primary) + **max Sharpe** on daily equity series (annualized ×365), benchmark = 0 |
| Turnover | **Carry-over + daily drift-band** (±20% no-trade band, L1 turnover penalty, neutrality-breach trigger) |
| Cadence | **Weekly Selection Meeting** (symbol set + target weights) + **Daily Rebalance Meeting** (same set toward targets) |
| Orchestration | **Claude/prompt-driven** (`SKILL.md`) with a **mandatory, deterministically-guarded** code/calc reviewer stage |

---

## 3. Architecture — two layers

The proven split from the inspiration desks: **"LLM proposes, code disposes."**

- **Reasoning layer (LLM agents):** markdown prompt files in `agents/`, dispatched by Claude running `SKILL.md`. Agents *reason and propose*; they never compute final numbers.
- **Deterministic spine (`futures_fund/`, Python, fully unit-tested):** owns **all** math — signal computation, neutrality optimization, sizing, risk gating, fee/funding/slippage accounting, execution simulation, P&L, and state.
- **Boundary contract:** every LLM output is validated against a pydantic schema before the spine consumes it; the spine fails loud (HALT) on contract violations.

### Reuse map (lift mostly verbatim from `crypto-trade-claude-code-weekly`)

| Component | File(s) | Use |
|---|---|---|
| Risk gate (RR≥2 after costs, leverage-as-output, liq-distance ≥2.5×, min-notional) | `futures_fund/risk_gate.py`, `sizing.py`, `liquidation.py`, `policy.py` | Direction-agnostic survival floor — reuse |
| **Signed/unclamped funding engine** (a short *receives* positive funding) | `futures_fund/costs.py` (`project_funding`, `count_funding_events`) | Core carry primitive — reuse, extend per-symbol intervals |
| Exposure measurement (gross long$ vs short$, net, tilt, risk-bearing view) | `futures_fund/portfolio.py` `book_exposure()` | Foundation of the neutrality monitor — reuse measurement, change policy |
| Correlated-as-one cluster capping (union-find on return corr) | `consolidation.py`, `portfolio_risk.py` | Prevent over-sizing correlated legs — reuse |
| Symmetric regime classifier | `regime.py` | Conviction shaper, never gates direction — reuse |
| Crypto-only universe filter (`is_crypto_perp`) | `market_data.py` | Exclude tokenized stocks/commodities/indices — reuse |
| Keyless public Binance USD-M data | `exchange.py` | Klines, mark price, funding, exchangeInfo, depth — reuse/extend |
| Market context (news RSS, Fear & Greed, Reddit, FRED) | `market_context.py` | Feed Sentiment Analyst — reuse/extend |
| Self-improvement substrate (anti-hindsight journal, DSR-gated lessons, hit-rates, scorecard, overfit detector) | `journal.py`, `lessons.py`, `flat_journal.py`, `shadow.py`, `hitrate.py`, `improvement.py`, `scorecard.py`, `graduation.py`, `vendor/overfit_detector.py` | Re-key on alpha — reuse |
| Atomic state I/O, idempotent logging | `state.py`, `equity_log.py`, `cycle_io.py` | Persistence template — reuse |
| Multi-cadence scheduler + single-flight lock | `scheduling.py` (`cycle_due`), `runlock.py` | Weekly + daily cadence roots — reuse |
| Archived market-neutral agent prompts | `agents/archive/{research_manager,watcher,bull,bear,derivatives,technical,sentiment}.md` | Starting points for the roster — reuse |
| Standing invariant panel | `self_audit.py` | Extend with neutrality/sentiment invariants — reuse |

---

## 4. Capital, budget & deployment model

- **Account equity:** $20,000 paper.
- **Gross notional target:** ~$20,000 (~1× gross leverage) = ~$10,000 long + ~$10,000 short.
- **Deployment floor:** each side ≥ **90%** of its $10k budget (≥ ~$9,000 deployed). The remaining ~$1k/side is **dry powder** held for daily rebalancing and opportunistic adds.
- **Per-name cap:** ≤ ~20–25% of a side's budget per symbol (configurable), subject to cluster caps.
- **Default-on deployment:** full two-sided deployment is the *default state*. Neutrality is a **construction constraint**, never an excuse to sit flat. (Directly counters the prior desk's documented all-cash / one-sided ratchet failure.)

All thresholds live in `config.yaml` and are tunable; values above are defaults.

---

## 5. Neutrality model (dollar + beta)

A **hard constraint** in portfolio construction, not telemetry.

- **Dollar-neutral:** |Σ long$ − Σ short$| ≤ `dollar_band` (default 2–3% of per-side budget).
- **Beta-neutral:** |Σᵢ wᵢ·βᵢ_BTC| ≤ `beta_band` (default 0.05 in equity-normalized β-$ terms), where βᵢ is each name's **rolling** beta to BTC (e.g. 30–60d, re-estimated each weekly cycle and monitored daily).
- **BTC hedge leg:** a dedicated BTC-perp position absorbs the residual portfolio beta. It is **sized jointly with the alpha legs inside the per-side budgets** (not added on top) — the optimizer solves dollar-balance, beta-neutrality, and gross ≈ $20k together, so the hedge consumes part of a side's budget rather than expanding gross exposure.
- **Stress policy:** when realized correlations spike toward 1.0 (regime classifier flags it), tighten bands and increase the BTC hedge weight; the every-cycle reviewer re-derives the residual and HALTs if out of band.

---

## 6. Alpha sleeves (four)

Each sleeve emits desired per-name tilts/positions. A **risk-parity allocator** assigns a risk budget across the four sleeves; the global optimizer (§8) merges them into one neutral book.

### 6.1 Funding carry
- Rank the cross-section by **signed funding × notional** (per-symbol settlement cadence). Long low/negative-funding names, short high-positive-funding names, delta-hedged.
- **Un-clamp** the RR-estimate funding term (the inherited `risk_gate.py` clamps it to `max(0, ·)`, hiding carry credit) so positive expected carry is visible to approve/veto.

### 6.2 Cointegration pairs (first-class `Pair`/`Spread` object)
- **Selection:** Engle-Granger (ADF p < 0.05) and/or Johansen, with **rolling re-test**; **FDR/Bonferroni** correction across the many candidate pairs to kill spurious cointegration.
- **Sizing:** hedge ratio β from the cointegrating vector; legs sized so the spread is the traded unit.
- **Signal:** Ornstein-Uhlenbeck mean reversion — **half-life = ln2/θ** as the lookback; **z-score** entry |z| ≥ 2, exit ≈ 0, **hard stop |z| ≥ 3**.
- **P&L attribution:** at the pair level (spread), not two disconnected legs.

### 6.3 Cross-sectional factor L/S
- Rank liquid names by momentum / carry / low-vol (configurable factor set). Long top tercile, short bottom tercile. Inverse-vol or value-weighted within each leg.

### 6.4 Sentiment (see §7)
- A standalone cross-sectional L/S sleeve: long high-sentiment / short low-sentiment, dollar+beta neutral.

---

## 7. Sentiment Analyst & "geometry of the coin"

### 7.1 The agent
- **Role:** gather **point-in-time** content (sources strictly timestamped before decision time) from crypto news RSS, Reddit crypto subs, crypto media sites, and the Fear & Greed index — reusing/extending `market_context.py` and the archived `agents/sentiment.md`.
- **Output (validated `SentimentReport` contract), per coin + overall market:**
  - `level ∈ {very_positive, positive, neutral, negative, very_negative}` → numeric `s ∈ {+2,+1,0,−1,−2}` normalized to [−1, +1]
  - `confidence ∈ [0,1]`, `sources[]` (citations with timestamps), one-line `rationale`
- Runs in **both cadences**: deep gather weekly, lighter refresh daily.

### 7.2 Inclusion in the coin's geometry
The per-coin **geometry** is the signal-feature bundle the constructor uses (momentum, carry/funding, vol/β, cointegration state). **Sentiment becomes first-class fields** in that bundle: `sentiment_score`, `sentiment_conf`. It acts two ways, both **bounded shapers that never flip direction**:

1. **Conviction tilt** (deterministic): scales a leg's target weight within a capped band —
   `wᵢ ← wᵢ · (1 + κ · sᵢ · confᵢ)`, clamped so |Δwᵢ| ≤ **25%**. Favors the long when positive / the short when negative. Never opens a position alone, never flips sign, never overrides neutrality or the risk gate.
2. **Sentiment factor sleeve** (§6.4): long high-sentiment / short low-sentiment, dollar+beta neutral, combined via the same risk-parity (risk-parity now spans **four** sleeves).

### 7.3 Safety
- **Neutrality preserved:** tilts and the sentiment sleeve are applied **before** the optimizer re-projects onto the dollar+beta-neutral constraint set, so sentiment cannot mathematically break neutrality or the risk gate (both computed after).
- **Decay:** a half-life (default ~3 days) decays stale scores toward neutral; refreshed each daily cycle.
- **Fail-soft:** missing / unparseable / stale sentiment → defaults to **neutral**; never blocks the book.
- **Reviewer checks:** the every-cycle reviewer and `self_audit.py` verify score is in valid ordinal/range, citations present and point-in-time, and the **influence cap is respected** (no weight moved beyond the band, no direction flipped by sentiment).

---

## 8. Portfolio construction / neutrality optimizer (the heart, net-new)

Deterministic. Inputs: the four sleeves' tilts + per-coin geometry (incl. sentiment). Solves for target weights subject to **hard constraints**:

- Dollar-neutral and beta-neutral bands (§5), with BTC hedge leg
- Deployment floor (≥90%/side) + dry-powder reserve
- Per-name and per-cluster (correlated-as-one) caps; gross ≈ $20k
- **Weighting:** Ledoit-Wolf **shrunk covariance** → **HRP** (cluster → quasi-diagonalize → recursive bisection) or risk-parity, avoiding unstable matrix inversion
- **Turnover-aware:** L1 turnover penalty + no-trade band on daily rebalance
- **Re-projection:** after sentiment tilts, project the weight vector back onto the neutrality constraint set

Implemented with `cvxpy` or `scipy.optimize`. Output: target weights → handed to the Trader for per-leg orders, then through the risk gate.

---

## 9. Two-cadence control loop (net-new)

### Weekly "Selection Meeting" (every 7 days)
Full roster + deep adversarial debate. Refresh all data; re-run cointegration tests, factor ranking, funding cross-section, sentiment deep-gather → optimizer outputs **the week's symbol set + target weights**. Move the book current → targets paying real costs, **carry-over** style (trade only the deltas; do not churn overlapping positions).

### Daily "Rebalance Meeting" (every day)
**Same symbol set.** Lighter roster. Recompute drift, z-scores, funding, sentiment refresh, neutrality residual & beta drift → trade **only** names outside the drift band, where a thesis/z-stop broke, or where neutrality breached. L1 turnover penalty.

### Scheduling
Reuse `scheduling.py cycle_due()` parameterized for two cadence roots (weekly + daily) + `runlock.py` single-flight. Daily cycle at a fixed UTC hour; weekly selection on a 7-day boundary. Every real run reaches the execution gate so cadence cannot double-fire. A lighter risk monitor (`monitor_cli.py`) runs between cycles and can trip HALT on a drawdown / liq-distance / neutrality-breach.

---

## 10. Agent roster

**Signal / analysis**
- **Universe Scout** — crypto-only, liquidity-filtered, two-sided shortlist
- **Funding-Carry Analyst** — ranks cross-section by funding sign/magnitude & basis
- **Pair/Cointegration Researcher** — candidate pairs + hedge ratio + cointegration/half-life evidence
- **Cross-sectional Factor Analyst** — momentum/carry/low-vol ranking
- **Sentiment Analyst** — §7
- **Technical Analyst** — per-leg structure/momentum/mean-reversion (reuse archive)
- **Derivatives/Positioning Analyst** — OI, long/short ratio, funding crowding (reuse archive)

**Adversarial debate**
- **Bull** — strongest case to open/keep a leg or pair
- **Bear** — strongest case to short/close; must rebut the Bull's load-bearing claims
- **Research Manager / Judge** — 5-tier rating + falsifiable prediction; rates relative-value pairs explicitly

**Deterministic (doc-only LLM, code-enforced)**
- **Neutrality & Portfolio Constructor** — the hard constraint + optimizer (§8)
- **Risk Gate** — the non-overridable survival floor (reuse)

**Execution**
- **Trader / Execution planner** — target weights → per-leg entry/stop/TP/triggers; no sizing

**Guardian (NEW, every cycle)**
- **Adversarial Code & Calc Reviewer** — re-derives neutrality residual (dollar+beta), beta-hedge sizing, funding sign/amount, pair PnL, RR-after-costs, Sharpe annualization, exchange-filter compliance, and sentiment-cap compliance against ground truth. **Hard veto → HALT on any mismatch.** A required, non-skippable SKILL stage gated by a deterministic flag the execution step checks.

**Learning**
- **Reflector** — contrastive lessons keyed on **alpha vs BTC-beta**, not raw return

Weekly cycle uses the full roster + deep debate; daily cycle uses a reduced roster (skip full re-selection; focus on rebalance, sentiment refresh, reviewer, risk gate).

---

## 11. Realism modeling (concrete defaults)

- **Fees:** taker **5.0 bps** / maker **2.0 bps** (VIP-0), optional BNB ×0.9. Taker on rebalance market fills, maker only on confirmed resting limits. *A pair round-trip = 4 fills ≈ 20 bps taker — the edge must clear this each rebalance.*
- **Funding:** source interval per-symbol from `/fapi/v1/fundingInfo` (**4h / 8h / 1h — not hardcoded 8h**); clamp realized rate to per-symbol cap/floor (BTC/ETH ±0.30%, alts ±2%); keep **signed & unclamped** in realized PnL (short receives positive funding). Settlement: `balance += −side·mark·qty·rate`. Fix the RR-estimate clamp so carry is visible to the gate.
- **Slippage:** **depth-aware** — wire the existing `vwap_fill` against an L2 depth snapshot, per-symbol; fallback `half_spread + k·√(notional/ADV)`. No flat 2 bps. Calibration anchors: BTCUSDT ~1.25 bps @ $1M, ~3.6 bps @ $5M; alts worse.
- **Mark price** (not last) for funding notional, uPnL, liquidation.
- **Liquidation / leverage:** tiered MMR brackets; leverage stays an **output** of liq-distance geometry (`choose_leverage`). At ~1× gross, liquidation risk is low but still modeled.
- **Sharpe periodicity:** daily series **×365**, weekly **×52** (the inherited 2190 (4h) would make every Sharpe/Sortino/DSR wrong).
- **Exchange filters:** tickSize / stepSize / MIN_NOTIONAL — sub-min legs rejected, not silently filled.
- **Point-in-time data:** enforce in prompts and any backtest/walk-forward (use `data.binance.vision` archives) so no post-decision info leaks; `exchangeInfo` gives only currently-listed symbols (survivorship caveat).

---

## 12. Self-improvement & zero-bug enforcement

- **Lessons/journal reused, re-keyed on alpha:** DSR-gated CANDIDATE→VALIDATED promotion; polarity-quota retrieval (force-include an enabling lesson to fight the all-cash ratchet). New dimensions: *did the pair stay cointegrated? did realized funding match the carry thesis? did the book stay dollar+beta neutral within band? did sentiment add or detract?*
- **Every-cycle Adversarial Code & Calc Reviewer** (§10): mandatory, deterministically-guarded, hard veto. This is how a prompt-driven desk achieves "zero bugs every cycle" without the skipped-stage weakness.
- **Extended `self_audit.py` invariants:** dollar-neutral residual in band · |Σwβ| in band · both sides ≥ floor · funding sign correct · pair legs sized by hedge ratio · sentiment within cap/range · no tokenized-stock/commodity leg.
- **Walk-forward + Deflated-Sharpe gate** (reuse `graduation.py`, `overfit_detector.py`, and the `walk-forward-validation` skill) before trusting any sleeve param/threshold change — stat-arb thresholds, factor lookbacks, and sentiment κ are prime overfitting targets; require OOS validation, not in-sample grid wins.
- **Improvement panel** re-pointed at neutral KPIs: rolling alpha-Sharpe trend, **both-sides deployment rate** (guard the all-cash AND one-sided-book ratchets), pair-survival rate, carry-capture rate, sentiment hit-rate.
- **Self-healing code loop:** full pytest green before any commit; protected modules never weakened; HALT if unfixable safely; `memory/repair-journal.md`.

---

## 13. Data layer

Keyless public Binance USD-M mainnet (`exchange.py`): klines, mark price, funding rate history, `exchangeInfo` (filters + per-symbol funding intervals), order-book depth (slippage). Crypto-only filter (`is_crypto_perp`, fail-closed). Market context (`market_context.py`: news RSS, Fear & Greed, Reddit, FRED) feeds the Sentiment Analyst. Point-in-time discipline throughout.

---

## 14. State, artifacts & data contracts

- **State:** atomic write (`tmp` + `os.replace`), idempotent logging on retry (reuse `state.py`, `equity_log.py`, `cycle_io.py`). Separate cycle roots for weekly vs daily.
- **Per-cycle artifacts** under `state/cycle/<cadence>/<N>/`: universe, briefs/geometry, analyst reports, sentiment reports, debate plans, proposals, optimizer output (target weights + neutrality residuals), reviewer verdict, execution report, reflection/lessons.
- **Key pydantic contracts (new or extended):** `CoinGeometry` (feature bundle incl. sentiment), `SentimentReport`, `Pair`/`Spread`, `SleeveSignal`, `TargetWeights` (with dollar/beta residuals), `ReviewerVerdict`, plus reused `AnalystReport`, `AgentProposal`, etc.

---

## 15. Testing & validation strategy

- **TDD throughout** (superpowers): write failing tests first for all new math — cointegration/ADF, OU half-life, z-score machinery, rolling beta, neutrality optimizer (dollar+beta bands, re-projection), per-symbol funding, depth slippage, sentiment tilt + cap, two-cadence scheduler, turnover band.
- **Property/invariant tests:** neutrality residuals within band; sentiment never flips direction; deployment floor honored; no tokenized-stock leg.
- **Walk-forward validation** for all sleeve params; **Deflated Sharpe** gate before trusting any change.
- Reuse the inspirations' 600+ test patterns and keep new logic in **new, non-protected modules** to limit regression risk.

---

## 16. Tech stack & project layout

Mirror the inspirations:
```
CLAUDE.md  MISSION.md  SKILL.md  config.yaml
agents/*.md          # LLM prompt files (roster)
futures_fund/        # deterministic Python spine
scripts/*.py         # CLIs the orchestrator invokes
state/               # cycle artifacts + equity log
memory/              # lessons, journal, repair-journal
tests/               # pytest suite
docs/                # specs + plans
```
Python + `uv` + `pydantic`; add `numpy` / `pandas` / `scipy` / `statsmodels` (cointegration/OU), `scikit-learn` (Ledoit-Wolf), `cvxpy` or `scipy.optimize` (constrained optimizer).

---

## 17. Phased build plan (detail in the implementation plan)

0. **Scaffold + data + realism primitives** — project skeleton, crypto-only universe, data layer, fees / per-symbol funding / depth slippage ported & re-tested; Sharpe periodicity fixed; sentiment ingestion via `market_context.py`.
1. **Neutrality + portfolio optimizer** — dollar+beta constraints, BTC hedge, shrinkage+HRP/risk-parity, deployment floor + dry powder, caps, re-projection (pure math, TDD).
2. **Four sleeves + `Pair` object + sentiment factor** — signal generators (TDD + walk-forward).
3. **Two-cadence control loop** — weekly select / daily rebalance, drift-band turnover, scheduling, state/artifacts.
4. **Agent roster + SKILL.md orchestration** — including the Sentiment Analyst and adversarial debate.
5. **Risk gate (direction-agnostic) + every-cycle Code/Calc Reviewer + self-audit invariants.**
6. **Self-improvement loop** — lessons/journal re-keyed on alpha, DSR gate, self-healing.
7. **End-to-end paper run + KPI dashboard + full walk-forward validation.**

---

## 18. Success metrics / KPIs

- **Primary:** fraction of calendar months that are positive (target: 100%).
- **Secondary:** Sharpe (daily, ×365) of the equity curve vs 0; Sortino; max drawdown (target cap configurable, e.g. ≤5%).
- **Alpha vs beta:** rolling alpha-Sharpe (return net of BTC-beta).
- **Process KPIs:** both-sides deployment rate, neutrality-residual adherence, pair-survival rate, carry-capture rate, sentiment hit-rate, reviewer veto rate.

---

## 19. Risks → mitigations

| Risk | Mitigation |
|---|---|
| All-cash / one-sided ratchet (prior desk's documented death) | Full two-sided deployment is the default; neutrality is a constraint, not a flat-excuse; deployment-rate KPI + enabling lessons |
| Dollar-neutral ≠ beta-neutral; correlations → 1 in stress | BTC hedge leg + rolling β re-estimation + stress-tightened bands |
| Cointegration breaks (depegs, regime shifts) | Rolling re-test + hard z-stop + FDR correction |
| Funding carry crowded / flips sign in stress | Continuous monitor, per-symbol settlement cadence, don't over-fit a benign funding regime |
| Cost drag on a two-sided, two-cadence book | Turnover band + L1 penalty + depth-aware per-symbol slippage |
| LLM look-ahead / hallucination (incl. sentiment) | Point-in-time data enforced; sentiment bounded/fail-soft; reviewer + invariants |
| Prompt-driven skipped/merged stages | Mandatory deterministically-guarded reviewer stage; HALT on missing flag |
| Optimizer/gate complexity & regression | New logic in new non-protected modules; TDD + DSR gate; protected modules never weakened |

---

## 20. Out of scope (YAGNI)

Live trading (paper only) · cross-exchange funding arbitrage (single-venue Binance) · spot & options markets · tokenized stocks / indexes / metals / gold coins (hard-excluded by mandate).

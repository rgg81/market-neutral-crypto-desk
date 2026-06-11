# CANONICAL NEW-INTERFACE CONTRACT — Market-Neutral Crypto Trading Desk

**Project root:** `/home/roberto/crypto-trade-claude-code-market-neutral`
**Authority:** This document is the single source of truth for every NET-NEW name. Later plan tasks reference these symbols verbatim. Spec citations are `§N`.

## 0. Reused types this contract builds on (DO NOT redefine)

All from `futures_fund/models.py` (weekly), lifted verbatim into this repo's `futures_fund/models.py`:
`Direction = Literal["long","short"]`, `RegimeQuadrant`, `HealthTier`, `Bias`, `Verdict`, `MmrBracket`, `SymbolSpec`, `TradeProposal` (validates stop-side per direction; `funding_rate`, `funding_interval_hours`, `risk_mult` present), `CostEstimate`, `SizedTrade`, `RegimeState`, `PortfolioHealth`, `RiskCaps`, `RiskDecision`.
Reused contracts from `futures_fund/contracts.py`: `AnalystReport`, `AgentProposal`, `CIOOutput`, `Candidate`, type aliases `Lean/Rating/Stance/Desk/EntryStyle`, `rating_to_direction`, `to_trade_proposal`.

**New shared type aliases** (added to `futures_fund/models.py`, imported everywhere below):

```python
SleeveName    = Literal["carry", "pairs", "factor", "sentiment"]
SentimentLevel = Literal["very_positive", "positive", "neutral", "negative", "very_negative"]
SpreadState   = Literal["flat", "long_spread", "short_spread", "stop"]   # OU position vs the traded spread
PairTestMethod = Literal["engle_granger", "johansen"]
Cadence       = Literal["weekly", "daily"]                                # control-loop / cycle root selector
```

---

## PART 1 — New pydantic models (ALL fields + types)

All in `futures_fund/contracts.py` unless noted. All `BaseModel`, strict-by-default (no `extra="allow"`) except where noted. These mirror the §14 "Key pydantic contracts" list.

### 1.1 `SentimentReport` (§7.1) — `contracts.py`
One Sentiment Analyst read for one coin (or `symbol == "MARKET"` for the overall-market read).

```python
class SentimentReport(BaseModel):
    symbol: str                                   # ccxt unified id, or "MARKET" for the market-wide read
    level: SentimentLevel                         # ordinal label
    s: float = Field(ge=-1.0, le=1.0)             # numeric score in [-1,+1] (level {+2..-2} normalized /2)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SentimentSource] = Field(default_factory=list)   # point-in-time citations
    rationale: str = ""                           # one-line
    as_of_ts: datetime                            # decision-time anchor; all sources must precede this
    decayed_s: float | None = None                # s after half-life decay toward 0 (filled by ingest, §7.3)

class SentimentSource(BaseModel):
    url: str
    published_ts: datetime                        # MUST be < owning report's as_of_ts (point-in-time)
    title: str = ""
    feed: str = ""                                 # "news_rss" | "reddit" | "fear_greed" | "media"

class SentimentBatch(BaseModel):                  # the agent's full output (validated against sentiment.md)
    reports: list[SentimentReport] = Field(default_factory=list)
```

`level`↔`s` mapping is enforced by `sentiment_ingest.level_to_s` (§2). `MARKET` report carries overall sentiment; per-coin reports feed geometry.

### 1.2 `CoinGeometry` (§7.2) — `contracts.py`
The per-coin signal-feature bundle the constructor consumes. Sentiment fields are first-class (§7.2).

```python
class CoinGeometry(BaseModel):
    symbol: str                                   # ccxt unified id
    mark: float                                   # mark price (not last) (§11)
    # --- momentum / vol / beta ---
    momentum_20: float = 0.0                      # cross-sectional momentum feature
    realized_vol: float = 0.0                     # annualized realized vol (for inverse-vol weighting)
    beta_btc: float = 1.0                         # rolling beta to BTC (beta.py, §5)
    beta_lookback_days: int = 45                  # window used for beta_btc
    # --- carry ---
    funding_rate: float = 0.0                     # current signed per-interval rate (NOT annualized)
    funding_interval_hours: float = 8.0           # per-symbol (funding_intervals.py)
    funding_apr: float = 0.0                      # signed annualized carry (rate * intervals/yr)
    funding_cap: float = 0.02                     # per-symbol clamp magnitude (BTC/ETH 0.003, alts 0.02)
    # --- cointegration state (if this coin is a pair leg) ---
    in_pair: bool = False
    pair_id: str | None = None                    # Pair.pair_id this coin participates in
    # --- sentiment (first-class, §7.2) ---
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)   # decayed s, fail-soft 0.0 (§7.3)
    sentiment_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    # --- liquidity / filters ---
    adv_usd: float = 0.0                          # 24h ADV in USDT (slippage model + universe floor)
    spec: SymbolSpec | None = None                # exchange filters (tick/step/min_notional/MMR)

class GeometryBundle(BaseModel):
    geometries: list[CoinGeometry] = Field(default_factory=list)
    as_of_ts: datetime
```

### 1.3 `Pair` / `Spread` (§6.2) — `contracts.py`
First-class cointegration pair object; P&L attributed at spread level.

```python
class Pair(BaseModel):
    pair_id: str                                  # stable id, e.g. "BTCUSDT__ETHUSDT"
    symbol_y: str                                 # dependent leg (ccxt unified)
    symbol_x: str                                 # independent / hedge leg
    hedge_ratio: float                            # beta from cointegrating vector: spread = y - beta*x
    method: PairTestMethod                        # "engle_granger" | "johansen"
    adf_pvalue: float                             # Engle-Granger ADF p (must be < adf_pvalue_max)
    adf_pvalue_adj: float | None = None           # FDR/Bonferroni-corrected p across candidate pairs
    johansen_trace_stat: float | None = None
    johansen_crit_95: float | None = None
    half_life: float                              # OU half-life in CYCLES (ln2/theta), lookback driver
    theta: float                                  # OU mean-reversion speed
    mu: float                                     # OU long-run spread mean
    sigma_eq: float                               # OU equilibrium stdev of the spread
    formed_cycle: int                             # cycle pair was selected
    cointegrated: bool = True                     # rolling re-test result (False => unwind)

class Spread(BaseModel):                          # live state of a Pair's traded spread this cycle
    pair_id: str
    spread_value: float                           # y - hedge_ratio*x at current marks
    zscore: float                                 # (spread_value - mu) / sigma_eq
    state: SpreadState                            # entry |z|>=2, exit ~0, hard stop |z|>=3 (§6.2)
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.0
    qty_y: float = 0.0                            # leg sizes so the SPREAD is the traded unit
    qty_x: float = 0.0
    notional_y: float = 0.0
    notional_x: float = 0.0
    realized_pnl: float = 0.0                     # attributed at pair level (§6.2), not per-leg
```

### 1.4 `SleeveSignal` (§6) — `contracts.py`
Each sleeve emits desired per-name tilts/weights before the optimizer merges them (§8).

```python
class SleeveTilt(BaseModel):
    symbol: str                                   # ccxt unified id
    direction: Direction                          # intended side
    target_weight: float                          # signed desired weight as fraction of side budget (pre-optimize)
    raw_score: float = 0.0                        # sleeve's unnormalized signal strength
    pair_id: str | None = None                    # set when this tilt is a pairs-sleeve leg

class SleeveSignal(BaseModel):
    sleeve: SleeveName                            # "carry" | "pairs" | "factor" | "sentiment"
    tilts: list[SleeveTilt] = Field(default_factory=list)
    risk_budget_frac: float = Field(default=0.0, ge=0.0, le=1.0)  # risk-parity share assigned to this sleeve
    diagnostics: dict = Field(default_factory=dict)              # sleeve-specific telemetry (free-form)
    as_of_ts: datetime
```

### 1.5 `TargetWeights` (§8, §5) — `contracts.py`
Optimizer output: per-symbol target weight + residuals + per-side deployment. Handed to Trader.

```python
class WeightLeg(BaseModel):
    symbol: str                                   # ccxt unified id
    direction: Direction
    weight: float                                 # signed fraction of total equity (long>0, short<0)
    target_notional: float                        # USDT notional this leg should carry
    beta_btc: float                               # beta used in the neutrality solve
    sleeve: SleeveName | Literal["hedge"]         # source sleeve, or "hedge" for the BTC hedge leg
    pair_id: str | None = None

class TargetWeights(BaseModel):
    legs: list[WeightLeg] = Field(default_factory=list)
    btc_hedge_notional: float = 0.0               # signed; the dedicated BTC-perp hedge leg (§5)
    # --- neutrality residuals (reviewer + self_audit re-derive these, §10/§12) ---
    dollar_residual: float                        # Sum(long$) - Sum(short$), USDT
    dollar_residual_frac: float                   # |dollar_residual| / per_side_budget (must be <= dollar_band)
    beta_residual: float                          # Sum_i w_i * beta_i (equity-normalized beta-$), |.|<=beta_band
    # --- deployment per side (§4) ---
    gross_long: float                             # Sum long notional, USDT
    gross_short: float                            # Sum short notional, USDT
    deploy_long_frac: float                       # gross_long / side_budget (must be >= deployment_floor)
    deploy_short_frac: float
    gross_notional: float                         # gross_long + gross_short (~ target_gross_usdt)
    turnover_l1: float = 0.0                       # L1 turnover vs prior book (daily rebalance, §9)
    feasible: bool = True                          # optimizer found a point in the constraint set
    notes: list[str] = Field(default_factory=list)
    as_of_ts: datetime
```

### 1.6 `ReviewerVerdict` (§10 Guardian, §12) — `contracts.py`
Every-cycle Adversarial Code & Calc Reviewer output. `passed=False` ⇒ orchestrator HALTs.

```python
class ReviewerCheck(BaseModel):
    name: str                                     # canonical check id (see fixed set below)
    ok: bool
    expected: float | str | None = None           # reviewer's ground-truth re-derivation
    actual: float | str | None = None             # value found in the artifact under review
    tolerance: float = 1e-6
    detail: str = ""

class ReviewerVerdict(BaseModel):
    passed: bool                                  # AND of all checks; deterministic flag the gate reads
    checks: list[ReviewerCheck] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)   # names of failed checks (== [c.name for c in checks if not c.ok])
    cycle: int
    cadence: Cadence
    reviewed_at: datetime
```

**Canonical `ReviewerCheck.name` set** (referenced verbatim by `reviewer.py` and `test_reviewer.py`):
`dollar_residual_in_band`, `beta_residual_in_band`, `btc_hedge_sizing`, `deployment_floor_both_sides`, `per_name_cap`, `cluster_cap`, `funding_sign`, `funding_amount`, `pair_leg_hedge_ratio`, `pair_pnl_attribution`, `rr_after_costs`, `sharpe_annualization`, `exchange_filter_compliance`, `sentiment_range`, `sentiment_cap_respected`, `sentiment_point_in_time`, `crypto_only_universe`.

---

## PART 2 — New module public function signatures

All under `futures_fund/`. Every module: `from __future__ import annotations`; pure where possible; fail-soft per spec. New non-protected modules (§15) — none of these are in the protected set (`risk_gate, executor, exits, consolidation, policy, liquidation, sizing, cycle`).

### 2.1 `futures_fund/beta.py` (§5) — rolling beta to BTC

```python
def log_returns(prices: pd.Series) -> pd.Series
    # Log returns of a mark-price series (aligned, NaN-dropped).

def rolling_beta(asset_returns: pd.Series, btc_returns: pd.Series, lookback: int = 45) -> float
    # OLS beta = cov(asset,btc)/var(btc) over the last `lookback` aligned points; 1.0 fallback if <10 pts or var==0.

def beta_series(asset_returns: pd.Series, btc_returns: pd.Series, lookback: int = 45) -> pd.Series
    # Rolling beta time series (for drift monitoring / reviewer re-derivation).

def beta_for_symbols(marks_by_symbol: dict[str, pd.Series], btc_symbol: str = "BTC/USDT:USDT",
                     lookback: int = 45) -> dict[str, float]
    # Per-symbol rolling beta_btc; BTC maps to 1.0 by construction.
```

### 2.2 `futures_fund/cointegration.py` (§6.2) — ADF/Johansen, OU half-life, z-score

```python
def engle_granger(y: pd.Series, x: pd.Series) -> tuple[float, float, float]
    # Returns (hedge_ratio, adf_pvalue, adf_stat): OLS y~x, ADF on residual spread. statsmodels-backed.

def johansen(frame: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict
    # Johansen trace test on a (T x n) price frame; returns {trace_stat, crit_95, hedge_ratio, rank}.

def ou_fit(spread: pd.Series) -> tuple[float, float, float]
    # Fit OU via AR(1) on the spread; returns (theta, mu, sigma_eq) (mean-reversion speed, mean, equilibrium sd).

def half_life(theta: float) -> float
    # OU half-life in cycles = ln(2)/theta; returns inf if theta <= 0 (non-mean-reverting).

def spread_value(y: float, x: float, hedge_ratio: float) -> float
    # y - hedge_ratio * x (the traded unit).

def zscore(spread_value: float, mu: float, sigma_eq: float) -> float
    # (spread_value - mu)/sigma_eq; 0.0 if sigma_eq <= 0.

def spread_state(z: float, *, entry_z: float = 2.0, exit_z: float = 0.0, stop_z: float = 3.0,
                 prev_state: SpreadState = "flat") -> SpreadState
    # OU state machine: |z|>=stop_z -> "stop"; |z|>=entry_z opens long/short_spread; |z|<=exit_z -> "flat".

def fdr_adjust(pvalues: list[float], *, alpha: float = 0.05, method: Literal["bh","bonferroni"] = "bh"
               ) -> list[float]
    # Benjamini-Hochberg (default) or Bonferroni multiple-testing correction across candidate pairs (§6.2).

def build_pair(y: pd.Series, x: pd.Series, symbol_y: str, symbol_x: str, *, cycle: int,
               method: PairTestMethod = "engle_granger") -> Pair
    # End-to-end: test + OU-fit + assemble a validated `Pair` (adf_pvalue_adj filled later by fdr_adjust).

def build_spread(pair: Pair, mark_y: float, mark_x: float, prev_state: SpreadState = "flat") -> Spread
    # Current `Spread` (value, zscore, state) from live marks + the pair's OU params.
```

### 2.3 `futures_fund/funding_intervals.py` (§11) — per-symbol interval + cap

```python
PER_SYMBOL_CAP_DEFAULT: float = 0.02              # alts default magnitude
MAJOR_CAP: float = 0.003                          # BTC/ETH magnitude (±0.30%)
_MAJORS: frozenset[str]                           # {"BTC/USDT:USDT","ETH/USDT:USDT"}

def funding_interval_hours(symbol: str, exchange) -> float
    # Per-symbol settlement interval from /fapi/v1/fundingInfo via FuturesExchange.funding(); default 8.0 on miss.

def funding_cap(symbol: str) -> float
    # Clamp magnitude for the realized rate: MAJOR_CAP for majors else PER_SYMBOL_CAP_DEFAULT (§11).

def clamp_funding_rate(symbol: str, rate: float) -> float
    # Clamp a realized rate to [-cap, +cap], SIGN-PRESERVING (carry stays signed, never zeroed) (§11).

def intervals_per_year(interval_hours: float) -> float
    # 24/interval_hours * 365 — annualization for funding_apr.

def funding_apr(rate: float, interval_hours: float) -> float
    # Signed annualized carry = rate * intervals_per_year(interval_hours).

def realized_funding(notional_signed: float, mark: float, qty: float, rate: float, direction: Direction
                     ) -> float
    # Settlement contribution to balance: -side*mark*qty*rate (short RECEIVES positive funding) (§11). Signed.
```

> Extends (does not redefine) `costs.project_funding` / `costs.count_funding_events`. The carry sleeve and reviewer use the SIGNED value; the inherited `risk_gate._build_sized` clamp is overridden via `unclamped_funding=True` plumbed through `to_trade_proposal` (see §2.10 note).

### 2.4 `futures_fund/slippage.py` (§11) — depth-aware

```python
DEFAULT_K: float = 0.1                            # sqrt-impact coefficient for the ADV fallback

def depth_slippage(levels: list[tuple[float, float]], qty: float, reference_price: float) -> float
    # Thin wrapper over costs.vwap_fill/slippage_cost against an L2 depth snapshot (USDT cost). Direction-symmetric.

def fallback_slippage(notional: float, adv_usd: float, half_spread_bps: float, *, k: float = DEFAULT_K
                      ) -> float
    # half_spread + k*sqrt(notional/ADV) model in USDT when no depth snapshot (§11). Calibrated to BTC anchors.

def estimate_slippage(symbol: str, qty: float, reference_price: float, *,
                      depth: list[tuple[float, float]] | None, adv_usd: float,
                      half_spread_bps: float, k: float = DEFAULT_K) -> float
    # Prefer depth_slippage; fall back to fallback_slippage. NEVER flat 2bps (§11). Returns USDT cost.

def slippage_bps(cost_usdt: float, notional: float) -> float
    # Convenience: cost in bps of notional (for calibration assertions ~1.25bps@$1M BTC).
```

### 2.5 `futures_fund/sentiment_ingest.py` (§7.1, §7.3) — point-in-time gather + decay

```python
LEVEL_TO_S: dict[SentimentLevel, float]           # {"very_positive":1.0,"positive":0.5,"neutral":0.0,
                                                  #  "negative":-0.5,"very_negative":-1.0}
def level_to_s(level: SentimentLevel) -> float
    # Ordinal level -> numeric s in [-1,1] ({+2..-2}/2). Enforces the §7.1 mapping.

def s_to_level(s: float) -> SentimentLevel
    # Inverse bucketing (reviewer round-trips level<->s for the sentiment_range check).

def gather_sentiment_context(http_client, settings: Settings, fred_key: str | None, *,
                             as_of: datetime) -> dict
    # Point-in-time wrapper over market_context.build_market_context; drops any source published_ts >= as_of (§7.1).

def decay_score(s: float, age_hours: float, half_life_days: float = 3.0) -> float
    # Exponential decay toward 0: s * 0.5**(age_hours/(half_life_days*24)) (§7.3).

def decay_report(report: SentimentReport, now: datetime, half_life_days: float = 3.0) -> SentimentReport
    # Returns a copy with decayed_s set from (now - as_of_ts) (§7.3).

def validate_point_in_time(report: SentimentReport) -> bool
    # True iff every source.published_ts < report.as_of_ts (reviewer sentiment_point_in_time check) (§7.3).

def fail_soft_neutral(symbol: str, now: datetime) -> SentimentReport
    # Neutral report (level="neutral", s=0, confidence=0) for missing/unparseable/stale (§7.3). Never blocks book.
```

### 2.6 `futures_fund/sleeves/carry.py` (§6.1)

```python
def carry_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                 top_frac: float = 1 / 3) -> SleeveSignal
    # Rank by SIGNED funding_apr; long low/negative-funding, short high-positive-funding, delta-hedged.
    # Carry credit is UNCLAMPED (signed) so positive expected carry is visible (§6.1). sleeve="carry".
```

### 2.7 `futures_fund/sleeves/pairs.py` (§6.2)

```python
def select_pairs(candidates: list[Pair], *, adf_pvalue_max: float = 0.05) -> list[Pair]
    # Keep pairs passing FDR-corrected ADF (adf_pvalue_adj < max) AND still cointegrated (rolling re-test).

def pairs_signal(pairs: list[Pair], spreads: list[Spread], *, risk_budget_frac: float, now: datetime
                 ) -> SleeveSignal
    # Emit per-leg tilts for active pairs (z-entry/exit/stop via Spread.state); legs sized by hedge_ratio so the
    # SPREAD is the traded unit. Each tilt carries pair_id. sleeve="pairs".
```

### 2.8 `futures_fund/sleeves/factor.py` (§6.3)

```python
def rank_factor(geometries: list[CoinGeometry], *, factor: Literal["momentum","carry","low_vol"]
                ) -> list[tuple[str, float]]
    # Cross-sectional ranking score per symbol for the chosen factor.

def factor_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                  factors: list[str] = ["momentum","carry","low_vol"], tercile: float = 1/3,
                  weighting: Literal["inverse_vol","equal"] = "inverse_vol") -> SleeveSignal
    # Long top tercile / short bottom tercile across combined factor rank; inverse-vol within each leg. sleeve="factor".
```

### 2.9 `futures_fund/sleeves/sentiment.py` (§6.4, §7.2)

```python
def sentiment_factor_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                            tercile: float = 1/3) -> SleeveSignal
    # Standalone L/S sleeve: long high (sentiment_score*sentiment_conf) / short low, dollar+beta neutral. sleeve="sentiment".

def conviction_tilt(weight: float, sentiment_score: float, sentiment_conf: float, *,
                    kappa: float = 0.5, cap: float = 0.25) -> float
    # Deterministic MAGNITUDE tilt: |w| <- |w|*(1 + kappa*sign(w)*s*conf), clamped so |Delta w| <= cap (25%).
    # sign(w) aligns s with the leg's direction so it FAVORS the long when s>0 / the short when s<0 (§7.2 prose;
    # the earlier `w*(1 + kappa*s*conf)` scalar form was wrong-for-shorts and is superseded). NEVER flips sign,
    # never opens a position alone (returns 0 if input weight is 0) (§7.2). Applied BEFORE optimizer re-projection (§7.3).

def apply_conviction_tilts(legs: list[SleeveTilt], geometries: list[CoinGeometry], *,
                           kappa: float = 0.5, cap: float = 0.25) -> list[SleeveTilt]
    # Map conviction_tilt over legs using each symbol's geometry; sign-preserving, cap-respecting.
```

> `futures_fund/sleeves/__init__.py` re-exports the four `*_signal` builders. The risk-parity allocator that assigns `risk_budget_frac` across the four sleeves lives in `neutrality.py` (`risk_parity_budgets`, §2.11) so all four are budgeted in one place.

### 2.10 `futures_fund/sentiment` plumbing note (no new module)
`to_trade_proposal` (reused) gains no signature change; the carry-visibility fix is plumbed via a new optional kwarg on the gate path documented in §2.3. The sentiment fields ride on `CoinGeometry` only — they never reach `TradeProposal`.

### 2.11 `futures_fund/neutrality.py` (§5, §8) — the optimizer (heart, net-new)

```python
def risk_parity_budgets(sleeves: list[SleeveSignal], *, cov: np.ndarray | None = None) -> dict[SleeveName, float]
    # Risk-parity (or inverse-vol) budget across the FOUR sleeves; fills SleeveSignal.risk_budget_frac. Sums to 1.0.

def merge_sleeves(sleeves: list[SleeveSignal], geometries: list[CoinGeometry]) -> dict[str, float]
    # Combine sleeve tilts (already risk-budgeted, sentiment-tilted) into one signed pre-projection weight vector.

def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray
    # Ledoit-Wolf shrunk covariance (sklearn) — stable, avoids unstable inversion (§8).

def hrp_weights(cov: np.ndarray, labels: list[str]) -> dict[str, float]
    # Hierarchical Risk Parity: cluster -> quasi-diagonalize -> recursive bisection (§8). No matrix inversion.

def project_neutral(weights: dict[str, float], betas: dict[str, float], *,
                    dollar_band: float, beta_band: float) -> dict[str, float]
    # Re-project a weight vector onto the dollar+beta-neutral constraint set (§8 re-projection). Sentiment tilts
    # are applied BEFORE this call so sentiment cannot break neutrality (§7.3).

def size_btc_hedge(weights: dict[str, float], betas: dict[str, float], *, equity: float,
                   side_budget: float) -> float
    # Signed BTC-perp hedge notional absorbing residual beta, sized INSIDE the per-side budget (not on top) (§5).

def dollar_residual(weights: dict[str, float], notionals: dict[str, float]) -> float
    # Sum(long$) - Sum(short$) in USDT.

def beta_residual(weights: dict[str, float], betas: dict[str, float]) -> float
    # Sum_i w_i * beta_i (equity-normalized beta-$).

def optimize_book(sleeves: list[SleeveSignal], geometries: list[CoinGeometry], *,
                  equity: float, prior_legs: list[WeightLeg] | None,
                  cfg: NeutralityConfig, regime: RegimeState | None = None) -> TargetWeights
    # THE solver. Merge sleeves -> apply sentiment tilts -> HRP/risk-parity weight -> enforce per-name & cluster
    # caps -> project_neutral -> size_btc_hedge -> deployment floor + dry powder -> L1 turnover penalty / no-trade
    # band (daily) -> assemble TargetWeights with residuals + per-side deployment. Stress-tightens bands when
    # regime flags correlation spike (§5). cvxpy or scipy.optimize backend. Sets feasible=False (never silently
    # un-neutral) if no point in the constraint set.

class NeutralityConfig(BaseModel):                # parsed from config.yaml `neutrality:` block (§3 below)
    capital_usdt: float = 20000.0
    target_gross_usdt: float = 20000.0
    side_budget_usdt: float = 10000.0
    deployment_floor: float = 0.90
    dry_powder_frac: float = 0.10
    per_name_cap: float = 0.25                     # fraction of a side's budget per symbol
    cluster_cap: float = 0.40                      # correlated-as-one heat cap (reuse consolidation/cluster_heat)
    dollar_band: float = 0.03                      # fraction of per-side budget
    beta_band: float = 0.05                        # equity-normalized beta-$
    drift_band: float = 0.20                       # ±20% no-trade band (daily) (§9)
    turnover_penalty: float = 0.001                # L1 turnover coefficient
    corr_threshold: float = 0.7                    # cluster union-find threshold (reuse)
    stress_band_mult: float = 0.5                  # tighten bands by this factor under correlation-spike stress
```

### 2.12 `futures_fund/control_loop.py` (§9) — weekly select + daily rebalance

```python
def cadence_due(state_dir, now_utc: datetime, cadence: Cadence) -> tuple[str, int, str]
    # Wrap scheduling.cycle_due with the right (tf_minutes, loop) per cadence:
    #   weekly -> tf_minutes=7*1440, loop="weekly"; daily -> tf_minutes=1440, loop="daily". Returns (mode,n,reason).

def weekly_selection(state_dir, geometries: list[CoinGeometry], sleeves: list[SleeveSignal], *,
                     equity: float, prior: TargetWeights | None, cfg: NeutralityConfig, cycle: int
                     ) -> TargetWeights
    # Full re-selection: new symbol set + target weights via optimize_book; carry-over (trade deltas only) (§9).

def daily_rebalance(state_dir, target: TargetWeights, geometries: list[CoinGeometry],
                    spreads: list[Spread], *, equity: float, cfg: NeutralityConfig, cycle: int
                    ) -> TargetWeights
    # SAME symbol set; recompute drift/z/funding/sentiment/neutrality residual & beta drift; trade ONLY names
    # outside drift_band, broken z-stops, or neutrality breaches; L1 turnover penalty (§9).

def drift_exceeded(current_weight: float, target_weight: float, *, drift_band: float = 0.20) -> bool
    # |current - target| / |target| > drift_band (no-trade band gate) (§9).

def neutrality_breached(target: TargetWeights, cfg: NeutralityConfig) -> bool
    # True if dollar_residual_frac > dollar_band OR |beta_residual| > beta_band (forces a daily rebalance trade).

def rebalance_deltas(prior: TargetWeights, target: TargetWeights) -> list[WeightLeg]
    # Per-symbol delta legs the Trader must execute (carry-over: overlapping unchanged legs excluded) (§9).
```

### 2.13 `futures_fund/reviewer.py` (§10 Guardian, §12) — every-cycle code/calc checks

```python
def review_cycle(state_dir, memory_dir, cycle: int, cadence: Cadence, *,
                 target: TargetWeights, geometries: list[CoinGeometry], spreads: list[Spread],
                 sentiment: list[SentimentReport], cfg: NeutralityConfig,
                 returns: list[float] | None = None) -> ReviewerVerdict
    # Re-derive from ground truth and compare to artifacts: dollar+beta residual, BTC-hedge sizing, funding
    # sign/amount, pair PnL & hedge-ratio sizing, RR-after-costs, Sharpe annualization (×365 daily/×52 weekly),
    # exchange-filter compliance, sentiment range/cap/point-in-time, crypto-only universe. passed = AND of checks.

def check_dollar_neutral(target: TargetWeights, cfg: NeutralityConfig) -> ReviewerCheck
def check_beta_neutral(target: TargetWeights, geometries: list[CoinGeometry], cfg: NeutralityConfig) -> ReviewerCheck
def check_btc_hedge(target: TargetWeights, geometries: list[CoinGeometry], cfg: NeutralityConfig) -> ReviewerCheck
def check_deployment_floor(target: TargetWeights, cfg: NeutralityConfig) -> ReviewerCheck
def check_caps(target: TargetWeights, cfg: NeutralityConfig) -> list[ReviewerCheck]   # per_name_cap + cluster_cap
def check_funding(target: TargetWeights, geometries: list[CoinGeometry]) -> list[ReviewerCheck]  # sign + amount
def check_pair_pnl(spreads: list[Spread], pairs: list[Pair]) -> list[ReviewerCheck]   # attribution + hedge ratio
def check_rr_after_costs(proposals: list[TradeProposal]) -> ReviewerCheck             # reuse risk_gate math
def check_sharpe_annualization(cadence: Cadence) -> ReviewerCheck                     # ×365 daily / ×52 weekly
def check_exchange_filters(target: TargetWeights, geometries: list[CoinGeometry]) -> list[ReviewerCheck]
def check_sentiment(sentiment: list[SentimentReport], target_before: TargetWeights,
                    target_after: TargetWeights, *, cap: float = 0.25) -> list[ReviewerCheck]  # range+cap+PIT
def check_crypto_only(geometries: list[CoinGeometry]) -> ReviewerCheck               # reuse is_crypto_perp

def reviewer_gate_ok(state_dir, cycle: int, cadence: Cadence) -> bool
    # Read persisted reviewer.json; True iff ReviewerVerdict.passed. The DETERMINISTIC flag the execution step
    # checks before any fill; missing/false => SystemExit(2)/HALT (§10, §12, mandatory non-skippable stage).
```

---

## PART 3 — `config.yaml` keys (with default values)

Extends the inherited layout (`account_size_usdt`, `timeframe`, `loops`, `agent_models`, `live: false`, `exchange`, `data`). New/overridden keys below; parsed by an extended `Settings`/`load_settings` in `futures_fund/config.py`. The `neutrality:` block parses into `NeutralityConfig`.

```yaml
# --- account / capital (§4) ---
account_size_usdt: 20000          # overrides inherited default
target_weekly: 0.05               # secondary; primary KPI is "no losing month"
max_drawdown_tolerance: 0.05      # §18 cap (tighter than weekly desk's 0.50)
live: false                       # MUST stay false forever (§16, hard rule)

# --- two-cadence loops (§9) ---
loops:
  weekly:
    timeframe: "4h"               # regime anchor candle
    regime_timeframe: "4h"
    poll_minutes: 1440
    deep_model: "opus"
    quick_model: "sonnet"
    cadence_days: 7               # weekly Selection Meeting boundary
  daily:
    timeframe: "1h"
    poll_minutes: 60
    deep_model: "sonnet"
    quick_model: "haiku"
    cadence_hour_utc: 0           # fixed UTC hour for the daily Rebalance Meeting

# --- neutrality / capital deployment (§4, §5, §8) -> NeutralityConfig ---
neutrality:
  capital_usdt: 20000
  target_gross_usdt: 20000        # ~1× gross
  side_budget_usdt: 10000         # ~$10k/side
  deployment_floor: 0.90          # ≥90%/side (≥ ~$9k)
  dry_powder_frac: 0.10           # ~$1k/side reserve
  per_name_cap: 0.25              # ≤25% of a side's budget per symbol
  cluster_cap: 0.40               # correlated-as-one heat cap
  dollar_band: 0.03               # |Σlong$−Σshort$| ≤ 3% of side budget
  beta_band: 0.05                 # |Σwᵢ·βᵢ| ≤ 0.05 (equity-normalized β-$)
  drift_band: 0.20                # ±20% daily no-trade band (§9)
  turnover_penalty: 0.001         # L1 turnover coefficient
  corr_threshold: 0.7             # cluster union-find threshold (reuse)
  stress_band_mult: 0.5           # tighten bands ×0.5 under correlation-spike stress

# --- beta estimation (§5) ---
beta:
  lookback_days: 45               # rolling β window (30–60d range)
  btc_symbol: "BTC/USDT:USDT"

# --- alpha sleeves (§6) ---
sleeves:
  risk_parity: true               # risk-parity across the four sleeves
  enabled: ["carry", "pairs", "factor", "sentiment"]
  factor:
    factors: ["momentum", "carry", "low_vol"]
    tercile: 0.3333
    weighting: "inverse_vol"
  pairs:
    adf_pvalue_max: 0.05
    fdr_method: "bh"              # Benjamini-Hochberg across candidates
    entry_z: 2.0
    exit_z: 0.0
    stop_z: 3.0
    min_half_life_cycles: 1.0
    max_half_life_cycles: 40.0
    rolling_retest_cycles: 7

# --- sentiment (§7) ---
sentiment:
  kappa: 0.5                      # κ conviction-tilt strength
  cap: 0.25                       # max |Δw| from sentiment (25%) — also the sleeve influence cap
  halflife_days: 3                # decay toward neutral
  refresh_daily: true

# --- universe (§4, §13) ---
universe:
  symbol_count: 30                # top ~20–30 by 24h volume
  min_adv_usd: 50000000           # min-depth/liquidity floor (USDT 24h ADV)
  crypto_only: true               # is_crypto_perp fail-closed

# --- fees / funding / slippage realism (§11) ---
fees:
  taker_bps: 5.0                  # 0.05% (overrides inherited TAKER_RATE)
  maker_bps: 2.0                  # 0.02%
  pay_bnb: false
  bnb_discount: 0.90
funding:
  default_interval_hours: 8       # per-symbol sourced; this is the miss default
  major_cap: 0.003                # BTC/ETH ±0.30%
  alt_cap: 0.02                   # alts ±2%
  majors: ["BTC/USDT:USDT", "ETH/USDT:USDT"]
  unclamped_in_rr: true           # fix the risk_gate carry clamp so carry is visible (§6.1, §11)
  signed_realized: true           # short receives positive funding in realized PnL
slippage:
  model: "depth"                  # depth-aware; fallback to sqrt-impact
  k: 0.1                          # sqrt-impact coefficient
  half_spread_bps_default: 1.0
  depth_levels: 20                # L2 snapshot depth
  flat_bps: null                  # explicitly NO flat 2bps (§11)

# --- metrics / annualization (§11, §18) ---
metrics:
  daily_periods_per_year: 365     # Sharpe ×365 for the daily equity series
  weekly_periods_per_year: 52     # ×52 weekly (the inherited 2190 4h factor is WRONG here)
  benchmark_return: 0.0           # cash

# --- reviewer guardian (§10, §12) ---
reviewer:
  enabled: true                   # mandatory, non-skippable
  halt_on_mismatch: true          # passed=False => HALT
  model: "opus"
  tolerance: 1e-6

# --- graduation / overfit gate (§12) ---
graduation:
  dsr_threshold: 0.95
  min_cycles: 20
  walk_forward_required: true     # OOS validation before trusting any sleeve param/threshold change
```

---

## Internal-consistency cross-reference (binding for downstream plan tasks)

- **Sleeve builders** (`carry_signal`, `pairs_signal`, `factor_signal`, `sentiment_factor_signal`) all return `SleeveSignal` with `sleeve ∈ SleeveName`; `risk_budget_frac` is assigned by `neutrality.risk_parity_budgets`, never by the sleeve itself.
- **Sentiment ordering invariant** (§7.3): `conviction_tilt`/`apply_conviction_tilts` run BEFORE `neutrality.project_neutral`; the optimizer recomputes residuals AFTER, so the reviewer's `sentiment_cap_respected` check compares `target_before` vs `target_after`.
- **Funding sign** is SIGNED everywhere new (`funding_intervals.realized_funding`, `costs.project_funding`); the only place it was clamped (`risk_gate._build_sized`) is overridden by `funding.unclamped_in_rr`.
- **`Pair.half_life`, OU params (`theta/mu/sigma_eq`)** are measured in CYCLES, consumed by `cointegration.zscore`/`spread_state` and the `pairs` config z-thresholds.
- **`TargetWeights` residual fields** (`dollar_residual_frac`, `beta_residual`, `deploy_long_frac`, `deploy_short_frac`) are exactly what `reviewer.check_*` and the extended `self_audit.py` re-derive against `NeutralityConfig` bands (§12).
- **`ReviewerVerdict.passed`** is the deterministic flag `reviewer.reviewer_gate_ok` reads; the execution CLI raises `SystemExit(2)` if absent/false (§10 mandatory stage).
- **Cadence roots:** `control_loop.cadence_due` calls `scheduling.cycle_due(loop="weekly", tf_minutes=10080)` / `(loop="daily", tf_minutes=1440)`; cycle artifacts under `state/cycle/<cadence>/<N>/` (§14).
- **New deps** the plan must add to `pyproject.toml` (§16): `statsmodels` (ADF/Johansen/OU), `scikit-learn` (Ledoit-Wolf), and `cvxpy` OR keep `scipy.optimize` (already have `scipy`).

**Relevant absolute paths** (net-new files this contract defines):
`/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/{models.py,contracts.py,beta.py,cointegration.py,funding_intervals.py,slippage.py,sentiment_ingest.py,neutrality.py,control_loop.py,reviewer.py}`,
`/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/sleeves/{__init__.py,carry.py,pairs.py,factor.py,sentiment.py}`,
`/home/roberto/crypto-trade-claude-code-market-neutral/config.yaml`.
Reused type source (lift verbatim): `/home/roberto/crypto-trade-claude-code-weekly/futures_fund/models.py`.

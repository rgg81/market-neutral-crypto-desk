# Phase 2 — Four alpha sleeves + Pair object + sentiment factor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic signal layer for the market-neutral desk — cointegration math (Engle-Granger ADF + optional Johansen, OU half-life, z-score machinery, FDR/Bonferroni), a first-class `Pair`/`Spread` object with pair-level PnL attribution, the four alpha-sleeve signal generators (carry, pairs, factor, sentiment), the risk-parity allocator that budgets the four sleeves, and a walk-forward validation harness hook — every sleeve emitting a `SleeveSignal` for the Phase 1 optimizer to consume.

**Architecture:** Pure-math Python modules under `futures_fund/` plus the four sleeve generators under `futures_fund/sleeves/`. Each module is fail-soft and side-effect-free where the contract allows. Signals flow `CoinGeometry`/`Pair`/`Spread` → sleeve `*_signal()` → `SleeveSignal` (with `SleeveTilt` legs) → `neutrality.risk_parity_budgets` (assigns each sleeve its `risk_budget_frac`) → Phase 1's `optimize_book`. The cointegration engine builds and re-tests `Pair` objects; the pairs sleeve sizes legs by the cointegrating hedge ratio so the *spread* is the traded unit. Walk-forward validation reuses the vendored Deflated-Sharpe / overfit detector before any sleeve param/threshold is trusted.

**Tech Stack:** Python 3.11, `uv`, `pydantic>=2.6`, `numpy`, `pandas`, `scipy`, `statsmodels` (ADF / Johansen / OU AR(1) fit), the vendored `overfit_detector` (DSR / Bonferroni / Holm), `pytest`, `ruff`. Reuse-by-attribution from `/home/roberto/crypto-trade-claude-code-weekly/futures_fund/` (`costs.py`, `metrics.py`, `graduation.py`, `vendor/overfit_detector.py`, `models.py` types).

---

## File Structure

Every file Phase 2 creates or modifies, with its single responsibility. Project root: `/home/roberto/crypto-trade-claude-code-market-neutral`.

| File | Create / Modify | Single responsibility |
|---|---|---|
| `futures_fund/models.py` | Modify | Add the Phase 2 shared type aliases (`SleeveName`, `SentimentLevel`, `SpreadState`, `PairTestMethod`, `Cadence`) next to the reused `Direction`/`SymbolSpec`/etc. (lifted in Phase 0). |
| `futures_fund/contracts.py` | Modify | Add the Phase 2 pydantic contracts this phase owns: `SentimentSource`, `SentimentReport`, `SentimentBatch`, `CoinGeometry`, `GeometryBundle`, `Pair`, `Spread`, `SleeveTilt`, `SleeveSignal`. (`TargetWeights`, `WeightLeg`, `ReviewerVerdict` belong to Phase 1.) |
| `futures_fund/cointegration.py` | Create | Engle-Granger ADF + Johansen, OU AR(1) fit, `half_life`, `spread_value`, `zscore`, `spread_state` machine, `fdr_adjust`, `build_pair`, `build_spread`. |
| `futures_fund/sleeves/__init__.py` | Create | Package marker; re-export the four `*_signal` builders. |
| `futures_fund/sleeves/carry.py` | Create | `carry_signal` — signed funding-rank L/S, un-clamped carry credit. |
| `futures_fund/sleeves/pairs.py` | Create | `select_pairs` (FDR-passing + still-cointegrated) + `pairs_signal` (Pair-based, hedge-ratio-sized legs). |
| `futures_fund/sleeves/factor.py` | Create | `rank_factor` + `factor_signal` — momentum/carry/low-vol cross-sectional tercile L/S, inverse-vol within leg. |
| `futures_fund/sleeves/sentiment.py` | Create | `sentiment_factor_signal` (standalone L/S sleeve) + `conviction_tilt` / `apply_conviction_tilts` (bounded per-coin tilt). |
| `futures_fund/sleeve_budget.py` | Create | `risk_parity_budgets` — risk-parity / inverse-vol budget across the four sleeves (fills `SleeveSignal.risk_budget_frac`). Implementation lives here so all sleeves can be budgeted before the optimizer exists; the canonical contract (§2.11) addresses it as `neutrality.risk_parity_budgets` — see the `neutrality.py` re-export row below. |
| `futures_fund/neutrality.py` | Create | Phase-2 stub that re-exports `risk_parity_budgets` from `sleeve_budget.py` so the canonical contract name `neutrality.risk_parity_budgets` (§2.11) resolves at the end of Phase 2. Phase 1 OWNS `neutrality.py` and will replace/extend this stub (adding `optimize_book`, beta-neutralization, etc.); Phase 2 only guarantees the `risk_parity_budgets` re-export exists. |
| `futures_fund/walk_forward.py` | Create | `walk_forward_splits` + `validate_sleeve_param` — OOS walk-forward harness hook over the vendored DSR / overfit detector, gating any sleeve param/threshold change. |
| `tests/test_cointegration.py` | Create (Test) | Tests for `cointegration.py`. |
| `tests/test_contracts_phase2.py` | Create (Test) | Tests for the Phase 2 contract models + the `level→s` invariant the sentiment sleeve relies on. |
| `tests/sleeves/__init__.py` | Create (Test) | Package marker for sleeve tests. |
| `tests/sleeves/test_carry.py` | Create (Test) | Tests for `carry_signal`. |
| `tests/sleeves/test_pairs.py` | Create (Test) | Tests for `select_pairs` + `pairs_signal`. |
| `tests/sleeves/test_factor.py` | Create (Test) | Tests for `rank_factor` + `factor_signal`. |
| `tests/sleeves/test_sentiment.py` | Create (Test) | Tests for `sentiment_factor_signal` + `conviction_tilt` + `apply_conviction_tilts`. |
| `tests/test_sleeve_budget.py` | Create (Test) | Tests for `risk_parity_budgets` (+ the `neutrality.risk_parity_budgets` re-export resolves). |
| `tests/test_walk_forward.py` | Create (Test) | Tests for `walk_forward_splits` + `validate_sleeve_param`. |
| `pyproject.toml` | Modify | Add `statsmodels>=0.14` to `[project].dependencies`. |
| `config.yaml` | Modify | Confirm the `sleeves:`, `sentiment:`, and `pairs:` blocks (defaults already specified by the contract) are present; Phase 2 reads them through the literal defaults baked into each builder's keyword args, so this is a verification-only modify. |

**Assumptions (provided by Phase 0/1, referenced verbatim, NOT re-created here):**
- `futures_fund/models.py` already contains the reused types lifted from `/home/roberto/crypto-trade-claude-code-weekly/futures_fund/models.py` (`Direction`, `SymbolSpec`, `MmrBracket`, etc.) and `from __future__ import annotations`.
- `futures_fund/contracts.py` already exists with the reused weekly contracts (`AnalystReport`, `AgentProposal`, …) and `from __future__ import annotations`.
- `futures_fund/costs.py` (with `project_funding`, `vwap_fill`, `slippage_cost`) and `futures_fund/metrics.py` (with `sharpe`, `PERIODS_PER_YEAR`) and `futures_fund/vendor/overfit_detector.py` (with `deflated_sharpe_ratio`, `bonferroni_correction`, `holm_correction`) have been lifted verbatim from the weekly repo.
- `futures_fund/graduation.py` (`deflated_sharpe_pvalue`) is present.

**Cross-phase hand-off (`neutrality.risk_parity_budgets`):** the canonical contract (§2.11) addresses the allocator as `neutrality.risk_parity_budgets`. Phase 2 implements the function in `futures_fund/sleeve_budget.py` AND ships a Phase-2 stub `futures_fund/neutrality.py` that re-exports it (Task 23a), so `from futures_fund.neutrality import risk_parity_budgets` resolves at the end of Phase 2. Phase 1 owns `neutrality.py` long-term and MUST preserve that re-export when it adds `optimize_book` / beta-neutralization; if Phase 1 rewrites `neutrality.py`, it MUST keep the `from futures_fund.sleeve_budget import risk_parity_budgets` line (or an equivalent definition) so the contract name does not regress.

If any assumption is unmet at execution time, lift the named file verbatim from `/home/roberto/crypto-trade-claude-code-weekly/futures_fund/<name>.py` (copy/adapt with attribution comment) before starting the dependent task.

---

### Task 0: Add `statsmodels` dependency + sync

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `statsmodels` to project dependencies**

In `pyproject.toml`, locate the `[project]` `dependencies` array and add `statsmodels>=0.14` to it. The block must read (preserving the inherited deps):

```toml
dependencies = [
    "pydantic>=2.6",
    "numpy>=1.26",
    "pandas>=2.1",
    "ccxt>=4.5",
    "httpx>=0.27",
    "scipy>=1.11",
    "pyyaml>=6.0",
    "statsmodels>=0.14",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves and installs `statsmodels` (and its `patsy` dep) with no errors; prints an `Installed N packages` / `Resolved` summary.

- [ ] **Step 3: Verify the import works**

Run: `uv run python -c "from statsmodels.tsa.stattools import adfuller, coint; from statsmodels.tsa.vector_ar.vecm import coint_johansen; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add statsmodels for cointegration/OU math

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 1: Phase 2 shared type aliases

**Files:**
- Modify: `futures_fund/models.py`
- Test: `tests/test_contracts_phase2.py`

- [ ] **Step 1: Write the failing test for the new aliases**

Create `tests/test_contracts_phase2.py` with:

```python
from __future__ import annotations

import typing

from futures_fund import models


def test_sleeve_name_alias_values():
    assert set(typing.get_args(models.SleeveName)) == {"carry", "pairs", "factor", "sentiment"}


def test_sentiment_level_alias_values():
    assert set(typing.get_args(models.SentimentLevel)) == {
        "very_positive", "positive", "neutral", "negative", "very_negative",
    }


def test_spread_state_alias_values():
    assert set(typing.get_args(models.SpreadState)) == {
        "flat", "long_spread", "short_spread", "stop",
    }


def test_pair_test_method_alias_values():
    assert set(typing.get_args(models.PairTestMethod)) == {"engle_granger", "johansen"}


def test_cadence_alias_values():
    assert set(typing.get_args(models.Cadence)) == {"weekly", "daily"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_phase2.py -v`
Expected: FAIL with `AttributeError: module 'futures_fund.models' has no attribute 'SleeveName'`.

- [ ] **Step 3: Add the aliases to `models.py`**

Append to `futures_fund/models.py` (after the existing `Verdict = ...` alias line, keeping all reused content intact):

```python
SleeveName = Literal["carry", "pairs", "factor", "sentiment"]
SentimentLevel = Literal["very_positive", "positive", "neutral", "negative", "very_negative"]
SpreadState = Literal["flat", "long_spread", "short_spread", "stop"]  # OU position vs the traded spread
PairTestMethod = Literal["engle_granger", "johansen"]
Cadence = Literal["weekly", "daily"]  # control-loop / cycle root selector
```

(`Literal` is already imported at the top of the file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts_phase2.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/models.py tests/test_contracts_phase2.py
git commit -m "feat: add Phase 2 shared type aliases (SleeveName, SentimentLevel, SpreadState, PairTestMethod, Cadence)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Sentiment contracts (`SentimentSource`, `SentimentReport`, `SentimentBatch`)

**Files:**
- Modify: `futures_fund/contracts.py`
- Test: `tests/test_contracts_phase2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contracts_phase2.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from futures_fund.contracts import SentimentBatch, SentimentReport, SentimentSource

_NOW = datetime(2026, 6, 11, 0, 0, tzinfo=timezone.utc)


def test_sentiment_report_valid():
    r = SentimentReport(
        symbol="BTC/USDT:USDT",
        level="positive",
        s=0.5,
        confidence=0.8,
        sources=[SentimentSource(url="http://x", published_ts=_NOW - timedelta(hours=2))],
        rationale="ETF inflows",
        as_of_ts=_NOW,
    )
    assert r.s == 0.5
    assert r.decayed_s is None
    assert r.sources[0].feed == ""


def test_sentiment_report_s_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SentimentReport(symbol="BTC/USDT:USDT", level="positive", s=1.5,
                        confidence=0.8, as_of_ts=_NOW)


def test_sentiment_report_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SentimentReport(symbol="BTC/USDT:USDT", level="neutral", s=0.0,
                        confidence=1.5, as_of_ts=_NOW)


def test_sentiment_batch_defaults_empty():
    b = SentimentBatch()
    assert b.reports == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_phase2.py -v -k sentiment`
Expected: FAIL with `ImportError: cannot import name 'SentimentReport' from 'futures_fund.contracts'`.

- [ ] **Step 3: Add the sentiment contracts**

Append to `futures_fund/contracts.py` (ensure `from datetime import datetime` and `from pydantic import BaseModel, Field` are imported at the top; add a `from futures_fund.models import SentimentLevel` import):

```python
class SentimentSource(BaseModel):
    url: str
    published_ts: datetime          # MUST be < owning report's as_of_ts (point-in-time)
    title: str = ""
    feed: str = ""                  # "news_rss" | "reddit" | "fear_greed" | "media"


class SentimentReport(BaseModel):
    symbol: str                     # ccxt unified id, or "MARKET" for the market-wide read
    level: SentimentLevel
    s: float = Field(ge=-1.0, le=1.0)               # numeric score in [-1,+1]
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SentimentSource] = Field(default_factory=list)
    rationale: str = ""
    as_of_ts: datetime
    decayed_s: float | None = None  # s after half-life decay toward 0 (filled by ingest)


class SentimentBatch(BaseModel):
    reports: list[SentimentReport] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts_phase2.py -v -k sentiment`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/contracts.py tests/test_contracts_phase2.py
git commit -m "feat: add SentimentSource/SentimentReport/SentimentBatch contracts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `CoinGeometry` + `GeometryBundle` contracts

**Files:**
- Modify: `futures_fund/contracts.py`
- Test: `tests/test_contracts_phase2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contracts_phase2.py`:

```python
from futures_fund.contracts import CoinGeometry, GeometryBundle


def test_coin_geometry_defaults():
    g = CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0)
    assert g.beta_btc == 1.0
    assert g.beta_lookback_days == 45
    assert g.funding_interval_hours == 8.0
    assert g.funding_cap == 0.02
    assert g.in_pair is False
    assert g.pair_id is None
    assert g.sentiment_score == 0.0
    assert g.sentiment_conf == 0.0
    assert g.spec is None


def test_coin_geometry_sentiment_range_enforced():
    with pytest.raises(ValidationError):
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, sentiment_score=2.0)


def test_geometry_bundle_holds_geometries():
    b = GeometryBundle(
        geometries=[CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0)],
        as_of_ts=_NOW,
    )
    assert b.geometries[0].symbol == "BTC/USDT:USDT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_phase2.py -v -k geometry`
Expected: FAIL with `ImportError: cannot import name 'CoinGeometry'`.

- [ ] **Step 3: Add the geometry contracts**

Append to `futures_fund/contracts.py` (add `from futures_fund.models import SymbolSpec` to the imports if not present):

```python
class CoinGeometry(BaseModel):
    symbol: str                                   # ccxt unified id
    mark: float                                   # mark price (not last)
    # --- momentum / vol / beta ---
    momentum_20: float = 0.0
    realized_vol: float = 0.0                     # annualized realized vol (inverse-vol weighting)
    beta_btc: float = 1.0
    beta_lookback_days: int = 45
    # --- carry ---
    funding_rate: float = 0.0                     # current signed per-interval rate (NOT annualized)
    funding_interval_hours: float = 8.0
    funding_apr: float = 0.0                      # signed annualized carry
    funding_cap: float = 0.02
    # --- cointegration state ---
    in_pair: bool = False
    pair_id: str | None = None
    # --- sentiment (first-class) ---
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)   # decayed s, fail-soft 0.0
    sentiment_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    # --- liquidity / filters ---
    adv_usd: float = 0.0
    spec: SymbolSpec | None = None


class GeometryBundle(BaseModel):
    geometries: list[CoinGeometry] = Field(default_factory=list)
    as_of_ts: datetime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts_phase2.py -v -k geometry`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/contracts.py tests/test_contracts_phase2.py
git commit -m "feat: add CoinGeometry/GeometryBundle feature-bundle contracts (sentiment first-class)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `Pair` + `Spread` contracts

**Files:**
- Modify: `futures_fund/contracts.py`
- Test: `tests/test_contracts_phase2.py`

> **Canonical `pair_id` format:** the desk uses the slash-free `"<SYMY>__<SYMX>"` form (ccxt unified ids stripped of `/` and `:`, e.g. `BTC/USDT:USDT` → `BTCUSDT`), so `"BTC/USDT:USDT"` paired with `"ETH/USDT:USDT"` yields `pair_id="BTCUSDT__ETHUSDT"`. This is the single canonical id format used everywhere `pair_id` appears in this plan (`build_pair`, `select_pairs`, `pairs_signal`, pipeline tests, and any downstream attribution). It is delimiter-safe (split on `"__"`) and matches the contract example.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contracts_phase2.py`:

```python
from futures_fund.contracts import Pair, Spread


def _pair() -> Pair:
    return Pair(
        pair_id="BTCUSDT__ETHUSDT",
        symbol_y="BTC/USDT:USDT",
        symbol_x="ETH/USDT:USDT",
        hedge_ratio=15.0,
        method="engle_granger",
        adf_pvalue=0.01,
        half_life=5.0,
        theta=0.139,
        mu=0.0,
        sigma_eq=200.0,
        formed_cycle=3,
    )


def test_pair_defaults():
    p = _pair()
    assert p.cointegrated is True
    assert p.adf_pvalue_adj is None
    assert p.johansen_trace_stat is None


def test_spread_defaults():
    s = Spread(pair_id="BTCUSDT__ETHUSDT", spread_value=400.0, zscore=2.0, state="long_spread")
    assert s.entry_z == 2.0
    assert s.exit_z == 0.0
    assert s.stop_z == 3.0
    assert s.realized_pnl == 0.0
    assert s.qty_y == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_phase2.py -v -k "pair or spread"`
Expected: FAIL with `ImportError: cannot import name 'Pair'`.

- [ ] **Step 3: Add the Pair/Spread contracts**

Append to `futures_fund/contracts.py` (add `from futures_fund.models import PairTestMethod, SpreadState` to the imports):

```python
class Pair(BaseModel):
    pair_id: str                                  # canonical slash-free id, e.g. "BTCUSDT__ETHUSDT"
    symbol_y: str                                 # dependent leg (ccxt unified id)
    symbol_x: str                                 # independent / hedge leg (ccxt unified id)
    hedge_ratio: float                            # spread = y - hedge_ratio*x
    method: PairTestMethod
    adf_pvalue: float                             # Engle-Granger ADF p (informational when method=="johansen")
    adf_pvalue_adj: float | None = None           # FDR/Bonferroni-corrected p
    johansen_trace_stat: float | None = None
    johansen_crit_95: float | None = None
    half_life: float                              # OU half-life in CYCLES (ln2/theta)
    theta: float                                  # OU mean-reversion speed
    mu: float                                     # OU long-run spread mean
    sigma_eq: float                               # OU equilibrium stdev of the spread
    formed_cycle: int
    cointegrated: bool = True                     # rolling re-test result


class Spread(BaseModel):
    pair_id: str
    spread_value: float                           # y - hedge_ratio*x at current marks
    zscore: float                                 # (spread_value - mu) / sigma_eq
    state: SpreadState
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.0
    qty_y: float = 0.0
    qty_x: float = 0.0
    notional_y: float = 0.0
    notional_x: float = 0.0
    realized_pnl: float = 0.0                     # attributed at pair level
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts_phase2.py -v -k "pair or spread"`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/contracts.py tests/test_contracts_phase2.py
git commit -m "feat: add first-class Pair/Spread cointegration contracts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `SleeveTilt` + `SleeveSignal` contracts

**Files:**
- Modify: `futures_fund/contracts.py`
- Test: `tests/test_contracts_phase2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contracts_phase2.py`:

```python
from futures_fund.contracts import SleeveSignal, SleeveTilt


def test_sleeve_tilt_defaults():
    t = SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.2)
    assert t.raw_score == 0.0
    assert t.pair_id is None


def test_sleeve_signal_valid():
    sig = SleeveSignal(
        sleeve="carry",
        tilts=[SleeveTilt(symbol="BTC/USDT:USDT", direction="short", target_weight=-0.3)],
        risk_budget_frac=0.25,
        as_of_ts=_NOW,
    )
    assert sig.sleeve == "carry"
    assert sig.tilts[0].direction == "short"
    assert sig.diagnostics == {}


def test_sleeve_signal_risk_budget_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SleeveSignal(sleeve="carry", risk_budget_frac=1.5, as_of_ts=_NOW)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_phase2.py -v -k sleeve`
Expected: FAIL with `ImportError: cannot import name 'SleeveSignal'`.

- [ ] **Step 3: Add the SleeveTilt/SleeveSignal contracts**

Append to `futures_fund/contracts.py` (add `from futures_fund.models import Direction, SleeveName` to the imports):

```python
class SleeveTilt(BaseModel):
    symbol: str                                   # ccxt unified id
    direction: Direction
    target_weight: float                          # signed desired weight (fraction of side budget)
    raw_score: float = 0.0                        # sleeve's unnormalized signal strength
    pair_id: str | None = None                    # set when this tilt is a pairs-sleeve leg


class SleeveSignal(BaseModel):
    sleeve: SleeveName
    tilts: list[SleeveTilt] = Field(default_factory=list)
    risk_budget_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    diagnostics: dict = Field(default_factory=dict)
    as_of_ts: datetime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts_phase2.py -v -k sleeve`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/contracts.py tests/test_contracts_phase2.py
git commit -m "feat: add SleeveTilt/SleeveSignal contracts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Cointegration — Engle-Granger ADF

**Files:**
- Create: `futures_fund/cointegration.py`
- Test: `tests/test_cointegration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cointegration.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from futures_fund import cointegration as co


def _cointegrated_pair(n: int = 400, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """x is a random walk; y = 2*x + stationary noise -> y and x are cointegrated."""
    rng = np.random.default_rng(seed)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0)
    noise = pd.Series(rng.normal(0, 0.5, n))
    y = 2.0 * x + noise
    return y, x


def test_engle_granger_recovers_hedge_ratio_and_rejects_unit_root():
    y, x = _cointegrated_pair()
    hedge_ratio, pvalue, stat = co.engle_granger(y, x)
    assert abs(hedge_ratio - 2.0) < 0.1          # OLS slope ~ 2.0
    assert pvalue < 0.05                          # residual is stationary -> reject unit root
    assert stat < 0.0                             # ADF stat is negative for a stationary series


def test_engle_granger_non_cointegrated_high_pvalue():
    rng = np.random.default_rng(11)
    y = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)   # two independent random walks
    _, pvalue, _ = co.engle_granger(y, x)
    assert pvalue > 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cointegration.py -v -k engle_granger`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.cointegration'`.

- [ ] **Step 3: Write the minimal implementation**

Create `futures_fund/cointegration.py` with only the imports `engle_granger` actually uses (later tasks add `import math`, `from typing import Literal`, the Johansen import, and the `Pair`/`Spread`/`PairTestMethod`/`SpreadState` contract/model imports as the functions that need them are appended — imports are introduced incrementally so each task's module is lint-clean):

> **Note (lint deferral):** Task 6 has no `ruff` step on purpose — the first cointegration lint gate is Task 12, Step 5, after all imports and functions are in place. The import list below is intentionally minimal (only what `engle_granger` references) so that even a reader who runs `ruff` against the Task-6-only module sees no F401 unused-import warnings.

```python
"""Cointegration math for the pairs sleeve: Engle-Granger ADF + Johansen, OU half-life,
z-score machinery, and FDR/Bonferroni multiple-testing correction across candidate pairs.

Adapted for the market-neutral desk; statsmodels-backed. Pure functions, fail-soft.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def engle_granger(y: pd.Series, x: pd.Series) -> tuple[float, float, float]:
    """OLS y~x then ADF on the residual spread. Returns (hedge_ratio, adf_pvalue, adf_stat).

    hedge_ratio is the OLS slope (the cointegrating beta: spread = y - hedge_ratio*x).
    A low adf_pvalue (< 0.05) means the residual is stationary -> the pair is cointegrated.
    """
    yv = pd.Series(y).reset_index(drop=True).astype(float)
    xv = pd.Series(x).reset_index(drop=True).astype(float)
    n = min(len(yv), len(xv))
    yv, xv = yv.iloc[:n], xv.iloc[:n]
    design = sm.add_constant(xv.to_numpy())
    model = sm.OLS(yv.to_numpy(), design).fit()
    hedge_ratio = float(model.params[1])
    resid = yv.to_numpy() - hedge_ratio * xv.to_numpy() - float(model.params[0])
    stat, pvalue, *_ = adfuller(resid, autolag="AIC")
    return hedge_ratio, float(pvalue), float(stat)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cointegration.py -v -k engle_granger`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cointegration.py tests/test_cointegration.py
git commit -m "feat: Engle-Granger ADF cointegration test (hedge ratio + p-value)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Cointegration — Johansen trace test

**Files:**
- Modify: `futures_fund/cointegration.py`
- Test: `tests/test_cointegration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cointegration.py` (add `import math` at the top of the test file if not already present):

```python
import math


def test_johansen_detects_cointegration_rank():
    y, x = _cointegrated_pair()
    frame = pd.DataFrame({"y": y, "x": x})
    out = co.johansen(frame)
    assert out["rank"] >= 1                        # at least one cointegrating relationship
    assert out["trace_stat"] > out["crit_95"]      # trace stat exceeds the 95% critical value
    assert math.isfinite(out["hedge_ratio"])


def test_johansen_independent_walks_rank_zero():
    rng = np.random.default_rng(3)
    a = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    b = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    out = co.johansen(pd.DataFrame({"a": a, "b": b}))
    assert out["rank"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cointegration.py -v -k johansen`
Expected: FAIL with `AttributeError: module 'futures_fund.cointegration' has no attribute 'johansen'`.

- [ ] **Step 3: Add the `johansen` function**

Append to `futures_fund/cointegration.py` (add the import at the top, next to the other statsmodels imports: `from statsmodels.tsa.vector_ar.vecm import coint_johansen`):

```python
def johansen(frame: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Johansen trace test on a (T x n) price frame.

    Returns {trace_stat, crit_95, hedge_ratio, rank}: trace_stat/crit_95 for the r=0 hypothesis,
    rank = number of cointegrating vectors at 95%, hedge_ratio normalized from the first
    eigenvector so the first column has coefficient 1 (spread = col0 - hedge_ratio*col1).
    """
    arr = frame.dropna().to_numpy(dtype=float)
    res = coint_johansen(arr, det_order, k_ar_diff)
    trace = res.lr1                                  # trace statistics, descending r
    crit_95 = res.cvt[:, 1]                          # 95% critical values column
    rank = int(sum(1 for i in range(len(trace)) if trace[i] > crit_95[i]))
    vec = res.evec[:, 0]                             # first cointegrating eigenvector
    base = vec[0] if vec[0] != 0 else 1.0
    hedge_ratio = float(-vec[1] / base) if len(vec) > 1 else 0.0
    return {
        "trace_stat": float(trace[0]),
        "crit_95": float(crit_95[0]),
        "hedge_ratio": hedge_ratio,
        "rank": rank,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cointegration.py -v -k johansen`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cointegration.py tests/test_cointegration.py
git commit -m "feat: Johansen trace test (rank + critical value + hedge ratio)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: OU fit + half-life

**Files:**
- Modify: `futures_fund/cointegration.py`
- Test: `tests/test_cointegration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cointegration.py`:

```python
def _ou_path(theta: float, mu: float, sigma: float, n: int = 2000, seed: int = 5) -> pd.Series:
    """Simulate a discrete OU process: s_{t+1} = s_t + theta*(mu - s_t) + sigma*eps."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    s[0] = mu
    for t in range(1, n):
        s[t] = s[t - 1] + theta * (mu - s[t - 1]) + sigma * rng.normal()
    return pd.Series(s)


def test_ou_fit_recovers_theta_and_mu():
    spread = _ou_path(theta=0.2, mu=5.0, sigma=0.3)
    theta, mu, sigma_eq = co.ou_fit(spread)
    assert abs(theta - 0.2) < 0.05
    assert abs(mu - 5.0) < 0.3
    assert sigma_eq > 0.0


def test_half_life_formula():
    assert abs(co.half_life(math.log(2)) - 1.0) < 1e-9     # theta = ln2 -> half-life 1 cycle
    assert abs(co.half_life(0.2) - (math.log(2) / 0.2)) < 1e-9


def test_half_life_non_mean_reverting_is_inf():
    assert co.half_life(0.0) == float("inf")
    assert co.half_life(-0.1) == float("inf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cointegration.py -v -k "ou_fit or half_life"`
Expected: FAIL with `AttributeError: module 'futures_fund.cointegration' has no attribute 'ou_fit'`.

- [ ] **Step 3: Add `ou_fit` and `half_life`**

Append to `futures_fund/cointegration.py` (add `import math` and `import numpy as np` to the top-of-file imports — `ou_fit`/`half_life` are the first functions to use them):

```python
def ou_fit(spread: pd.Series) -> tuple[float, float, float]:
    """Fit an OU process via AR(1) on the spread. Returns (theta, mu, sigma_eq).

    Discrete AR(1): s_{t+1} = a + b*s_t + eps. Then theta = 1 - b (mean-reversion speed),
    mu = a / (1 - b) (long-run mean), and sigma_eq = std(eps) / sqrt(1 - b^2) (equilibrium sd).
    """
    s = pd.Series(spread).dropna().reset_index(drop=True).astype(float).to_numpy()
    if len(s) < 3:
        return 0.0, float(s.mean()) if len(s) else 0.0, 0.0
    lagged = s[:-1]
    nxt = s[1:]
    design = sm.add_constant(lagged)
    model = sm.OLS(nxt, design).fit()
    a = float(model.params[0])
    b = float(model.params[1])
    theta = 1.0 - b
    mu = a / (1.0 - b) if abs(1.0 - b) > 1e-12 else float(s.mean())
    resid_sd = float(np.std(model.resid, ddof=2)) if len(model.resid) > 2 else 0.0
    denom = 1.0 - b * b
    sigma_eq = resid_sd / math.sqrt(denom) if denom > 0 else resid_sd
    return theta, mu, sigma_eq


def half_life(theta: float) -> float:
    """OU half-life in cycles = ln(2)/theta. inf if theta <= 0 (non-mean-reverting)."""
    if theta <= 0:
        return float("inf")
    return math.log(2) / theta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cointegration.py -v -k "ou_fit or half_life"`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cointegration.py tests/test_cointegration.py
git commit -m "feat: OU AR(1) fit (theta/mu/sigma_eq) + half-life = ln2/theta

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Spread value, z-score, and the OU state machine

**Files:**
- Modify: `futures_fund/cointegration.py`
- Test: `tests/test_cointegration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cointegration.py`:

```python
def test_spread_value():
    assert co.spread_value(100.0, 40.0, 2.0) == 100.0 - 2.0 * 40.0   # = 20.0


def test_zscore_normal():
    assert co.zscore(20.0, 10.0, 5.0) == 2.0


def test_zscore_zero_sigma_is_zero():
    assert co.zscore(20.0, 10.0, 0.0) == 0.0


def test_spread_state_transitions():
    # flat -> short_spread when z >= entry (spread rich, short the spread)
    assert co.spread_state(2.5, prev_state="flat") == "short_spread"
    # flat -> long_spread when z <= -entry (spread cheap, long the spread)
    assert co.spread_state(-2.5, prev_state="flat") == "long_spread"
    # |z| >= stop_z dominates -> stop
    assert co.spread_state(3.5, prev_state="short_spread") == "stop"
    assert co.spread_state(-3.5, prev_state="long_spread") == "stop"
    # inside exit band -> flat
    assert co.spread_state(0.0, prev_state="short_spread") == "flat"
    # between exit and entry: hold the open position
    assert co.spread_state(1.5, prev_state="short_spread") == "short_spread"
    # between exit and entry from flat: stay flat (no new entry)
    assert co.spread_state(1.5, prev_state="flat") == "flat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cointegration.py -v -k "spread_value or zscore or spread_state"`
Expected: FAIL with `AttributeError: module 'futures_fund.cointegration' has no attribute 'spread_value'`.

- [ ] **Step 3: Add `spread_value`, `zscore`, `spread_state`**

Append to `futures_fund/cointegration.py` (`spread_state` is the first function to use the `SpreadState` alias, so add `from futures_fund.models import SpreadState` to the top-of-file imports now):

```python
def spread_value(y: float, x: float, hedge_ratio: float) -> float:
    """The traded unit: y - hedge_ratio * x."""
    return float(y) - float(hedge_ratio) * float(x)


def zscore(spread_value: float, mu: float, sigma_eq: float) -> float:
    """(spread_value - mu) / sigma_eq; 0.0 if sigma_eq <= 0."""
    if sigma_eq <= 0:
        return 0.0
    return (float(spread_value) - float(mu)) / float(sigma_eq)


def spread_state(z: float, *, entry_z: float = 2.0, exit_z: float = 0.0, stop_z: float = 3.0,
                 prev_state: SpreadState = "flat") -> SpreadState:
    """OU state machine driving the traded spread.

    |z| >= stop_z  -> "stop" (hard exit).
    z >= entry_z   -> "short_spread" (spread is rich; short it for reversion).
    z <= -entry_z  -> "long_spread"  (spread is cheap; long it for reversion).
    |z| <= exit_z  -> "flat" (mean reached; close).
    Otherwise hold prev_state (no-trade hysteresis band between exit and entry).
    """
    az = abs(z)
    if az >= stop_z:
        return "stop"
    if z >= entry_z:
        return "short_spread"
    if z <= -entry_z:
        return "long_spread"
    if az <= exit_z:
        return "flat"
    return prev_state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cointegration.py -v -k "spread_value or zscore or spread_state"`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cointegration.py tests/test_cointegration.py
git commit -m "feat: spread_value/zscore + OU entry|z|>=2 / exit~0 / stop|z|>=3 state machine

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: FDR / Bonferroni multiple-testing correction

**Files:**
- Modify: `futures_fund/cointegration.py`
- Test: `tests/test_cointegration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cointegration.py`:

```python
def test_fdr_bh_is_monotone_and_ge_raw():
    raw = [0.001, 0.01, 0.03, 0.5]
    adj = co.fdr_adjust(raw, method="bh")
    assert len(adj) == 4
    assert all(a >= r - 1e-12 for a, r in zip(adj, raw))   # adjusted p >= raw p
    assert all(a <= 1.0 + 1e-12 for a in adj)


def test_fdr_bonferroni_multiplies_by_m():
    raw = [0.01, 0.02]
    adj = co.fdr_adjust(raw, method="bonferroni")
    assert abs(adj[0] - 0.02) < 1e-12              # 0.01 * 2
    assert abs(adj[1] - 0.04) < 1e-12              # 0.02 * 2


def test_fdr_empty_returns_empty():
    assert co.fdr_adjust([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cointegration.py -v -k fdr`
Expected: FAIL with `AttributeError: module 'futures_fund.cointegration' has no attribute 'fdr_adjust'`.

- [ ] **Step 3: Add `fdr_adjust`**

Append to `futures_fund/cointegration.py` (`fdr_adjust` is the first function to use `Literal`, so add `from typing import Literal` to the top-of-file imports now):

```python
def fdr_adjust(pvalues: list[float], *, alpha: float = 0.05,
               method: Literal["bh", "bonferroni"] = "bh") -> list[float]:
    """Benjamini-Hochberg (default) or Bonferroni correction across candidate pairs.

    Returns adjusted p-values in the ORIGINAL input order, each clamped to [0, 1]. BH adjustment:
    p_adj(i) = min over k>=rank(i) of (m/k * p_sorted(k)), enforced monotone non-decreasing.
    """
    m = len(pvalues)
    if m == 0:
        return []
    if method == "bonferroni":
        return [min(1.0, p * m) for p in pvalues]
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj_sorted = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):                    # rank = m..1 (largest p first)
        idx = order[rank - 1]
        val = min(1.0, pvalues[idx] * m / rank)
        prev = min(prev, val)
        adj_sorted[rank - 1] = prev
    out = [0.0] * m
    for rank, idx in enumerate(order):
        out[idx] = adj_sorted[rank]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cointegration.py -v -k fdr`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cointegration.py tests/test_cointegration.py
git commit -m "feat: FDR (Benjamini-Hochberg) + Bonferroni correction across candidate pairs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: `build_pair` end-to-end assembly

**Files:**
- Modify: `futures_fund/cointegration.py`
- Test: `tests/test_cointegration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cointegration.py`:

```python
from futures_fund.contracts import Pair


def test_build_pair_assembles_validated_pair():
    y, x = _cointegrated_pair()
    pair = co.build_pair(y, x, "BTC/USDT:USDT", "ETH/USDT:USDT", cycle=4)
    assert isinstance(pair, Pair)
    assert pair.pair_id == "BTCUSDT__ETHUSDT"      # canonical slash-free id
    assert pair.symbol_y == "BTC/USDT:USDT"
    assert pair.symbol_x == "ETH/USDT:USDT"
    assert pair.method == "engle_granger"
    assert pair.adf_pvalue < 0.05
    assert pair.adf_pvalue_adj is None             # FDR fills this later across the candidate set
    assert abs(pair.hedge_ratio - 2.0) < 0.1
    assert pair.formed_cycle == 4
    assert pair.cointegrated is True
    assert pair.half_life > 0.0


def test_build_pair_johansen_method():
    # method="johansen": hedge_ratio + johansen fields come from the Johansen result, and
    # cointegration is judged by trace_stat > crit_95 (NOT the EG ADF p, which is informational).
    y, x = _cointegrated_pair()
    pair = co.build_pair(y, x, "BTC/USDT:USDT", "ETH/USDT:USDT", cycle=4, method="johansen")
    assert pair.method == "johansen"
    assert pair.johansen_trace_stat is not None
    assert pair.johansen_crit_95 is not None
    # cointegrated derives from the trace statistic for the johansen branch
    assert pair.cointegrated == (pair.johansen_trace_stat > pair.johansen_crit_95)
    assert pair.cointegrated is True               # the simulated pair IS cointegrated
    assert pair.half_life > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cointegration.py -v -k build_pair`
Expected: FAIL with `AttributeError: module 'futures_fund.cointegration' has no attribute 'build_pair'`.

- [ ] **Step 3: Add `build_pair`**

Append to `futures_fund/cointegration.py` (`build_pair` is the first function to use the `Pair` contract and the `PairTestMethod` alias, so add `from futures_fund.contracts import Pair` and extend the models import to `from futures_fund.models import PairTestMethod, SpreadState` in the top-of-file imports now):

```python
def _canonical_pair_id(symbol_y: str, symbol_x: str) -> str:
    """Slash-free canonical pair id, e.g. ("BTC/USDT:USDT", "ETH/USDT:USDT") -> "BTCUSDT__ETHUSDT".

    Strips the ccxt delimiters ('/' and ':') so the id is split-safe on the "__" separator and
    matches the documented Pair.pair_id convention.
    """
    def _norm(sym: str) -> str:
        return sym.replace("/", "").replace(":", "")
    return f"{_norm(symbol_y)}__{_norm(symbol_x)}"


def build_pair(y: pd.Series, x: pd.Series, symbol_y: str, symbol_x: str, *, cycle: int,
               method: PairTestMethod = "engle_granger") -> Pair:
    """Test + OU-fit + assemble a validated Pair. adf_pvalue_adj is filled later by fdr_adjust
    across the full candidate set.

    For method=="engle_granger", cointegration is judged by the ADF p (< 0.05). For
    method=="johansen", hedge_ratio + johansen fields come from the Johansen result and
    cointegration is judged by trace_stat > crit_95; adf_pvalue is retained but informational.
    """
    hedge_ratio, adf_pvalue, _ = engle_granger(y, x)
    yv = pd.Series(y).reset_index(drop=True).astype(float)
    xv = pd.Series(x).reset_index(drop=True).astype(float)
    n = min(len(yv), len(xv))
    spread = yv.iloc[:n].to_numpy() - hedge_ratio * xv.iloc[:n].to_numpy()
    theta, mu, sigma_eq = ou_fit(pd.Series(spread))
    johansen_trace = johansen_crit = None
    cointegrated = adf_pvalue < 0.05
    if method == "johansen":
        jo = johansen(pd.DataFrame({"y": yv, "x": xv}))
        hedge_ratio = jo["hedge_ratio"]
        johansen_trace = jo["trace_stat"]
        johansen_crit = jo["crit_95"]
        cointegrated = jo["trace_stat"] > jo["crit_95"]   # trace-stat verdict for johansen
        # re-fit the OU spread on the Johansen-selected hedge ratio
        spread = yv.iloc[:n].to_numpy() - hedge_ratio * xv.iloc[:n].to_numpy()
        theta, mu, sigma_eq = ou_fit(pd.Series(spread))
    return Pair(
        pair_id=_canonical_pair_id(symbol_y, symbol_x),
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        hedge_ratio=hedge_ratio,
        method=method,
        adf_pvalue=adf_pvalue,
        johansen_trace_stat=johansen_trace,
        johansen_crit_95=johansen_crit,
        half_life=half_life(theta),
        theta=theta,
        mu=mu,
        sigma_eq=sigma_eq,
        formed_cycle=cycle,
        cointegrated=cointegrated,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cointegration.py -v -k build_pair`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cointegration.py tests/test_cointegration.py
git commit -m "feat: build_pair end-to-end (EG/Johansen test + OU fit -> validated Pair)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: `build_spread` live state from marks

**Files:**
- Modify: `futures_fund/cointegration.py`
- Test: `tests/test_cointegration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cointegration.py`:

```python
from futures_fund.contracts import Spread


def _btc_eth_pair() -> Pair:
    return Pair(
        pair_id="BTCUSDT__ETHUSDT",
        symbol_y="BTC/USDT:USDT", symbol_x="ETH/USDT:USDT",
        hedge_ratio=2.0, method="engle_granger", adf_pvalue=0.01,
        half_life=5.0, theta=0.139, mu=0.0, sigma_eq=10.0, formed_cycle=1,
    )


def test_build_spread_computes_value_zscore_state():
    pair = _btc_eth_pair()
    sp = co.build_spread(pair, mark_y=120.0, mark_x=49.0, prev_state="flat")
    assert isinstance(sp, Spread)
    assert sp.pair_id == pair.pair_id
    assert sp.spread_value == 120.0 - 2.0 * 49.0    # = 22.0
    assert sp.zscore == 2.2                          # (22 - 0) / 10
    assert sp.state == "short_spread"                # z >= entry_z (2.0) -> short the rich spread
    assert sp.entry_z == 2.0 and sp.exit_z == 0.0 and sp.stop_z == 3.0


def test_build_spread_hard_stop_state():
    pair = _btc_eth_pair()
    sp = co.build_spread(pair, mark_y=131.0, mark_x=49.0, prev_state="short_spread")
    assert sp.zscore == 3.3                          # (33 - 0)/10 -> |z| >= stop_z
    assert sp.state == "stop"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cointegration.py -v -k build_spread`
Expected: FAIL with `AttributeError: module 'futures_fund.cointegration' has no attribute 'build_spread'`.

- [ ] **Step 3: Add `build_spread`**

Append to `futures_fund/cointegration.py` (`build_spread` is the first function to use the `Spread` contract, so extend the contracts import to `from futures_fund.contracts import Pair, Spread`):

```python
def build_spread(pair: Pair, mark_y: float, mark_x: float,
                 prev_state: SpreadState = "flat") -> Spread:
    """Current Spread (value, zscore, state) from live marks + the pair's OU params."""
    sv = spread_value(mark_y, mark_x, pair.hedge_ratio)
    z = zscore(sv, pair.mu, pair.sigma_eq)
    state = spread_state(z, prev_state=prev_state)
    return Spread(pair_id=pair.pair_id, spread_value=sv, zscore=z, state=state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cointegration.py -v -k build_spread`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full cointegration suite + lint**

Run: `uv run pytest tests/test_cointegration.py -v && uv run ruff check futures_fund/cointegration.py`
Expected: all cointegration tests PASS; ruff prints `All checks passed!`. (This is the first cointegration lint gate; by now every front-loaded import — `math`, `numpy`, `Literal`, `coint_johansen`, `Pair`, `Spread`, `PairTestMethod`, `SpreadState` — has a using function, so no F401 unused-import warnings.)

- [ ] **Step 6: Commit**

```bash
git add futures_fund/cointegration.py tests/test_cointegration.py
git commit -m "feat: build_spread live state (value/zscore/state) from marks + Pair OU params

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Sleeves package scaffold

**Files:**
- Create: `futures_fund/sleeves/__init__.py`
- Create: `tests/sleeves/__init__.py`
- Test: `tests/sleeves/test_carry.py`

- [ ] **Step 1: Write the failing test (package importability)**

Create `tests/sleeves/__init__.py` (empty file).

Create `tests/sleeves/test_carry.py`:

```python
from __future__ import annotations


def test_sleeves_package_importable():
    import futures_fund.sleeves  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_carry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.sleeves'`.

- [ ] **Step 3: Create the package marker**

Create `futures_fund/sleeves/__init__.py`:

```python
"""Alpha-sleeve signal generators: each emits a SleeveSignal of per-name tilts the optimizer
merges into one neutral book. The four sleeves are carry, pairs, factor, and sentiment.
"""
from __future__ import annotations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_carry.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/__init__.py tests/sleeves/__init__.py tests/sleeves/test_carry.py
git commit -m "feat: scaffold futures_fund/sleeves package

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Carry sleeve — signed funding rank (un-clamped)

**Files:**
- Create: `futures_fund/sleeves/carry.py`
- Test: `tests/sleeves/test_carry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/sleeves/test_carry.py`:

```python
from datetime import datetime, timezone

from futures_fund.contracts import CoinGeometry
from futures_fund.sleeves.carry import carry_signal

_NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def _geo(symbol: str, apr: float) -> CoinGeometry:
    return CoinGeometry(symbol=symbol, mark=100.0, funding_apr=apr)


def test_carry_signal_shorts_high_funding_longs_negative_funding():
    geos = [
        _geo("A/USDT:USDT", 0.30),     # high positive carry -> crowded long -> SHORT
        _geo("B/USDT:USDT", 0.10),
        _geo("C/USDT:USDT", -0.05),
        _geo("D/USDT:USDT", -0.25),    # negative carry -> SHORTS pay us -> LONG
    ]
    sig = carry_signal(geos, risk_budget_frac=0.25, now=_NOW)
    assert sig.sleeve == "carry"
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["A/USDT:USDT"].direction == "short"
    assert by_sym["D/USDT:USDT"].direction == "long"
    # signed, un-clamped raw_score == funding_apr (carry credit visible, never zeroed)
    assert by_sym["A/USDT:USDT"].raw_score == 0.30
    assert by_sym["D/USDT:USDT"].raw_score == -0.25
    # long weights positive, short weights negative
    assert by_sym["D/USDT:USDT"].target_weight > 0
    assert by_sym["A/USDT:USDT"].target_weight < 0


def test_carry_signal_top_frac_limits_legs():
    geos = [_geo(f"{c}/USDT:USDT", apr) for c, apr in
            zip("ABCDEF", [0.3, 0.2, 0.1, -0.1, -0.2, -0.3])]
    # top_frac=1/3 matches the tercile convention used by the factor/sentiment sleeves:
    # floor(6 * 1/3) = 2 -> 2 longs + 2 shorts.
    sig = carry_signal(geos, risk_budget_frac=0.25, now=_NOW, top_frac=1 / 3)
    longs = [t for t in sig.tilts if t.direction == "long"]
    shorts = [t for t in sig.tilts if t.direction == "short"]
    assert len(longs) == 2
    assert len(shorts) == 2


def test_carry_signal_empty_geometries():
    sig = carry_signal([], risk_budget_frac=0.25, now=_NOW)
    assert sig.tilts == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_carry.py -v -k carry_signal`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.sleeves.carry'`.

- [ ] **Step 3: Write the implementation**

Create `futures_fund/sleeves/carry.py`:

```python
"""Funding-carry sleeve: rank the cross-section by SIGNED funding_apr; long the low/negative-
funding names (their shorts PAY us), short the high-positive-funding names (crowded longs).

Carry credit is UNCLAMPED and signed everywhere (unlike the inherited risk_gate clamp), so a
favorable funding edge is visible to the optimizer/gate (design spec §6.1).
"""
from __future__ import annotations

import math
from datetime import datetime

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt


def carry_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                 top_frac: float = 1 / 3) -> SleeveSignal:
    """Long low/negative funding_apr, short high-positive funding_apr, delta-hedged.

    raw_score carries the signed un-clamped funding_apr; target_weight is the per-leg signed share
    of the side budget (long > 0, short < 0), equal-weight within each side (pre-optimize).
    k = max(1, floor(n * top_frac)) names per side (top_frac is a tercile-style fraction).
    """
    ranked = sorted(geometries, key=lambda g: g.funding_apr)   # ascending: most negative first
    n = len(ranked)
    if n == 0:
        return SleeveSignal(sleeve="carry", tilts=[], risk_budget_frac=risk_budget_frac,
                            as_of_ts=now)
    k = max(1, math.floor(n * top_frac))
    longs = ranked[:k]                                          # lowest/negative funding -> LONG
    shorts = ranked[-k:]                                        # highest funding -> SHORT
    long_w = 1.0 / k
    short_w = 1.0 / k
    tilts: list[SleeveTilt] = []
    for g in longs:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="long",
                                target_weight=long_w, raw_score=g.funding_apr))
    for g in shorts:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="short",
                                target_weight=-short_w, raw_score=g.funding_apr))
    return SleeveSignal(sleeve="carry", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"n_candidates": n, "k_per_side": k}, as_of_ts=now)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_carry.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/carry.py tests/sleeves/test_carry.py
git commit -m "feat: carry sleeve (signed un-clamped funding rank L/S)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Pairs sleeve — `select_pairs`

**Files:**
- Create: `futures_fund/sleeves/pairs.py`
- Test: `tests/sleeves/test_pairs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/sleeves/test_pairs.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from futures_fund.contracts import Pair
from futures_fund.sleeves.pairs import select_pairs

_NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def _pair(pid: str, adj: float | None, *, cointegrated: bool = True) -> Pair:
    return Pair(
        pair_id=pid, symbol_y="BTC/USDT:USDT", symbol_x="ETH/USDT:USDT",
        hedge_ratio=2.0, method="engle_granger", adf_pvalue=0.01, adf_pvalue_adj=adj,
        half_life=5.0, theta=0.139, mu=0.0, sigma_eq=10.0, formed_cycle=1,
        cointegrated=cointegrated,
    )


def test_select_pairs_keeps_fdr_passing_cointegrated():
    kept = select_pairs([
        _pair("p1", 0.01),                    # passes FDR + cointegrated -> keep
        _pair("p2", 0.20),                    # fails FDR -> drop
        _pair("p3", 0.01, cointegrated=False),  # FDR ok but rolling re-test failed -> drop
        _pair("p4", None),                    # no adjusted p yet -> drop (conservative)
    ], adf_pvalue_max=0.05)
    assert [p.pair_id for p in kept] == ["p1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_pairs.py -v -k select_pairs`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.sleeves.pairs'`.

- [ ] **Step 3: Write the `select_pairs` implementation**

Create `futures_fund/sleeves/pairs.py` (Task 15 imports only `Pair`, the single symbol `select_pairs` uses; `SleeveSignal`, `SleeveTilt`, and `Spread` are added in Task 16 when `pairs_signal` first needs them — imports stay minimal per task):

```python
"""Cointegration-pairs sleeve: emit per-leg tilts for active Pairs, sized by the cointegrating
hedge ratio so the SPREAD is the traded unit. PnL is attributed at the pair level (Spread).
"""
from __future__ import annotations

from futures_fund.contracts import Pair


def select_pairs(candidates: list[Pair], *, adf_pvalue_max: float = 0.05) -> list[Pair]:
    """Keep pairs whose FDR-corrected ADF p is < adf_pvalue_max AND that are still cointegrated
    (rolling re-test passed). A pair with no adf_pvalue_adj yet is dropped (conservative)."""
    out: list[Pair] = []
    for p in candidates:
        if p.adf_pvalue_adj is None:
            continue
        if p.adf_pvalue_adj < adf_pvalue_max and p.cointegrated:
            out.append(p)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_pairs.py -v -k select_pairs`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/pairs.py tests/sleeves/test_pairs.py
git commit -m "feat: select_pairs (FDR-passing + still-cointegrated filter)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Pairs sleeve — `pairs_signal` (hedge-ratio-sized legs)

**Files:**
- Modify: `futures_fund/sleeves/pairs.py`
- Test: `tests/sleeves/test_pairs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/sleeves/test_pairs.py`:

```python
from futures_fund.contracts import Spread
from futures_fund.sleeves.pairs import pairs_signal


def _spread(pid: str, state: str, z: float) -> Spread:
    return Spread(pair_id=pid, spread_value=0.0, zscore=z, state=state)


def test_pairs_signal_short_spread_legs():
    # short_spread means: short y, long x (hedge_ratio units of x per unit of y).
    pair = _pair("p1", 0.01)
    sig = pairs_signal([pair], [_spread("p1", "short_spread", 2.5)],
                       risk_budget_frac=0.25, now=_NOW)
    assert sig.sleeve == "pairs"
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["BTC/USDT:USDT"].direction == "short"   # y leg
    assert by_sym["ETH/USDT:USDT"].direction == "long"    # x leg
    # both legs carry the pair_id (so attribution is at the pair level)
    assert all(t.pair_id == "p1" for t in sig.tilts)
    # x leg weight is hedge_ratio (2.0) times the y leg magnitude -> spread is the traded unit
    assert abs(by_sym["ETH/USDT:USDT"].target_weight) == 2.0 * abs(
        by_sym["BTC/USDT:USDT"].target_weight)


def test_pairs_signal_long_spread_flips_legs():
    pair = _pair("p1", 0.01)
    sig = pairs_signal([pair], [_spread("p1", "long_spread", -2.5)],
                       risk_budget_frac=0.25, now=_NOW)
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["BTC/USDT:USDT"].direction == "long"    # long the spread -> long y
    assert by_sym["ETH/USDT:USDT"].direction == "short"   # short hedge x


def test_pairs_signal_flat_and_stop_emit_no_legs():
    pair = _pair("p1", 0.01)
    sig = pairs_signal([pair],
                       [_spread("p1", "flat", 0.0), _spread("p1", "stop", 3.5)],
                       risk_budget_frac=0.25, now=_NOW)
    assert sig.tilts == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_pairs.py -v -k pairs_signal`
Expected: FAIL with `ImportError: cannot import name 'pairs_signal'`.

- [ ] **Step 3: Add `pairs_signal`**

Extend the imports at the top of `futures_fund/sleeves/pairs.py` to bring in the symbols `pairs_signal` newly needs (these were intentionally NOT imported in Task 15) and add `from datetime import datetime`:

```python
from datetime import datetime

from futures_fund.contracts import Pair, SleeveSignal, SleeveTilt, Spread
```

(replace the single `from futures_fund.contracts import Pair` line from Task 15 with the wider import above, and add the `datetime` import line.)

Then append to `futures_fund/sleeves/pairs.py`:

```python
def pairs_signal(pairs: list[Pair], spreads: list[Spread], *, risk_budget_frac: float,
                 now: datetime) -> SleeveSignal:
    """Emit per-leg tilts for active pairs. The spread is the traded unit:
      - short_spread: short y, long x (hedge_ratio units of x per unit of y)
      - long_spread : long y, short x
      - flat / stop : emit no legs (close handled by the optimizer/Trader)
    Each tilt carries pair_id for pair-level PnL attribution. Base y-leg weight is equal across
    active pairs (1/n_active); the x leg is scaled by the hedge ratio.
    """
    by_id = {p.pair_id: p for p in pairs}
    active = [s for s in spreads if s.state in ("long_spread", "short_spread")
              and s.pair_id in by_id]
    n = len(active)
    if n == 0:
        return SleeveSignal(sleeve="pairs", tilts=[], risk_budget_frac=risk_budget_frac,
                            as_of_ts=now)
    base_w = 1.0 / n
    tilts: list[SleeveTilt] = []
    for sp in active:
        pair = by_id[sp.pair_id]
        h = abs(pair.hedge_ratio)
        if sp.state == "short_spread":               # short y, long x
            y_dir, x_dir = "short", "long"
            y_w, x_w = -base_w, base_w * h
        else:                                        # long_spread: long y, short x
            y_dir, x_dir = "long", "short"
            y_w, x_w = base_w, -base_w * h
        tilts.append(SleeveTilt(symbol=pair.symbol_y, direction=y_dir, target_weight=y_w,
                                raw_score=sp.zscore, pair_id=pair.pair_id))
        tilts.append(SleeveTilt(symbol=pair.symbol_x, direction=x_dir, target_weight=x_w,
                                raw_score=sp.zscore, pair_id=pair.pair_id))
    return SleeveSignal(sleeve="pairs", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"n_active": n}, as_of_ts=now)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_pairs.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/pairs.py tests/sleeves/test_pairs.py
git commit -m "feat: pairs_signal (Pair-based, hedge-ratio-sized legs; spread is the traded unit)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: Factor sleeve — `rank_factor`

**Files:**
- Create: `futures_fund/sleeves/factor.py`
- Test: `tests/sleeves/test_factor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/sleeves/test_factor.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from futures_fund.contracts import CoinGeometry
from futures_fund.sleeves.factor import rank_factor

_NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def _geo(symbol, *, mom=0.0, apr=0.0, vol=0.1):
    return CoinGeometry(symbol=symbol, mark=100.0, momentum_20=mom,
                        funding_apr=apr, realized_vol=vol)


def test_rank_factor_momentum_high_first():
    geos = [_geo("A/USDT:USDT", mom=0.1), _geo("B/USDT:USDT", mom=0.3),
            _geo("C/USDT:USDT", mom=-0.2)]
    ranked = rank_factor(geos, factor="momentum")
    assert [s for s, _ in ranked] == ["B/USDT:USDT", "A/USDT:USDT", "C/USDT:USDT"]


def test_rank_factor_carry_uses_negative_funding_apr():
    # carry factor: LOW funding_apr is attractive (we get paid), so score = -funding_apr
    geos = [_geo("A/USDT:USDT", apr=0.3), _geo("B/USDT:USDT", apr=-0.3)]
    ranked = rank_factor(geos, factor="carry")
    assert ranked[0][0] == "B/USDT:USDT"          # negative funding ranks best
    assert ranked[0][1] == 0.3                     # score = -(-0.3)


def test_rank_factor_low_vol_prefers_low_realized_vol():
    geos = [_geo("A/USDT:USDT", vol=0.5), _geo("B/USDT:USDT", vol=0.1)]
    ranked = rank_factor(geos, factor="low_vol")
    assert ranked[0][0] == "B/USDT:USDT"          # lower vol ranks best
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_factor.py -v -k rank_factor`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.sleeves.factor'`.

- [ ] **Step 3: Write `rank_factor`**

Create `futures_fund/sleeves/factor.py`:

```python
"""Cross-sectional factor L/S sleeve: rank liquid names by momentum / carry / low-vol; long the
top tercile, short the bottom tercile, inverse-vol weighted within each leg (design spec §6.3).
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt


def _factor_score(g: CoinGeometry, factor: str) -> float:
    if factor == "momentum":
        return g.momentum_20
    if factor == "carry":
        return -g.funding_apr               # low/negative funding is attractive
    if factor == "low_vol":
        return -g.realized_vol              # lower vol is attractive
    raise ValueError(f"unknown factor {factor!r}")


def rank_factor(geometries: list[CoinGeometry], *,
                factor: Literal["momentum", "carry", "low_vol"]) -> list[tuple[str, float]]:
    """Cross-sectional ranking score per symbol for the chosen factor, best (highest score) first."""
    scored = [(g.symbol, _factor_score(g, factor)) for g in geometries]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_factor.py -v -k rank_factor`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/factor.py tests/sleeves/test_factor.py
git commit -m "feat: rank_factor (momentum/carry/low_vol cross-sectional scoring)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 18: Factor sleeve — `factor_signal` (tercile L/S, inverse-vol)

**Files:**
- Modify: `futures_fund/sleeves/factor.py`
- Test: `tests/sleeves/test_factor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/sleeves/test_factor.py`:

```python
from futures_fund.sleeves.factor import factor_signal


def test_factor_signal_tercile_long_short_combined():
    # 6 names with monotone momentum; tercile (1/3 of 6 = 2) -> 2 longs (top) + 2 shorts (bottom)
    geos = [_geo(f"{c}/USDT:USDT", mom=m, vol=0.1)
            for c, m in zip("ABCDEF", [0.5, 0.4, 0.1, -0.1, -0.4, -0.5])]
    sig = factor_signal(geos, risk_budget_frac=0.25, now=_NOW,
                        factors=["momentum"], tercile=1 / 3, weighting="equal")
    assert sig.sleeve == "factor"
    longs = {t.symbol for t in sig.tilts if t.direction == "long"}
    shorts = {t.symbol for t in sig.tilts if t.direction == "short"}
    assert longs == {"A/USDT:USDT", "B/USDT:USDT"}
    assert shorts == {"E/USDT:USDT", "F/USDT:USDT"}


def test_factor_signal_inverse_vol_weights_lower_vol_heavier():
    geos = [_geo("A/USDT:USDT", mom=0.5, vol=0.1),   # low vol -> heavier
            _geo("B/USDT:USDT", mom=0.4, vol=0.4),   # high vol -> lighter
            _geo("C/USDT:USDT", mom=-0.4, vol=0.2),
            _geo("D/USDT:USDT", mom=-0.5, vol=0.2)]
    sig = factor_signal(geos, risk_budget_frac=0.25, now=_NOW,
                        factors=["momentum"], tercile=0.5, weighting="inverse_vol")
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["A/USDT:USDT"].target_weight > by_sym["B/USDT:USDT"].target_weight
    # weights within the long side sum to ~1.0
    long_sum = sum(t.target_weight for t in sig.tilts if t.direction == "long")
    assert abs(long_sum - 1.0) < 1e-9


def test_factor_signal_combines_multiple_factors_by_rank():
    geos = [_geo("A/USDT:USDT", mom=0.9, apr=-0.3),   # best on both momentum & carry
            _geo("B/USDT:USDT", mom=0.1, apr=0.0),
            _geo("C/USDT:USDT", mom=-0.9, apr=0.3)]   # worst on both
    sig = factor_signal(geos, risk_budget_frac=0.25, now=_NOW,
                        factors=["momentum", "carry"], tercile=1 / 3, weighting="equal")
    longs = {t.symbol for t in sig.tilts if t.direction == "long"}
    shorts = {t.symbol for t in sig.tilts if t.direction == "short"}
    assert "A/USDT:USDT" in longs
    assert "C/USDT:USDT" in shorts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_factor.py -v -k factor_signal`
Expected: FAIL with `ImportError: cannot import name 'factor_signal'`.

- [ ] **Step 3: Add `factor_signal`**

Append to `futures_fund/sleeves/factor.py`:

```python
def _combined_rank(geometries: list[CoinGeometry], factors: list[str]) -> list[tuple[str, float]]:
    """Average rank-position across factors (0 = best). Lower combined value = stronger long."""
    agg: dict[str, float] = {g.symbol: 0.0 for g in geometries}
    for factor in factors:
        for pos, (sym, _score) in enumerate(rank_factor(geometries, factor=factor)):
            agg[sym] += pos
    combined = [(sym, agg[sym] / max(1, len(factors))) for sym in agg]
    combined.sort(key=lambda t: t[1])               # best (lowest avg rank) first
    return combined


def _inverse_vol_weights(syms: list[str], geo_by_sym: dict[str, CoinGeometry],
                         weighting: str) -> dict[str, float]:
    if weighting == "equal" or not syms:
        w = 1.0 / len(syms) if syms else 0.0
        return {s: w for s in syms}
    inv = {s: 1.0 / max(geo_by_sym[s].realized_vol, 1e-6) for s in syms}
    total = sum(inv.values())
    return {s: inv[s] / total for s in syms}


def factor_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                  factors: list[str] = ["momentum", "carry", "low_vol"], tercile: float = 1 / 3,
                  weighting: Literal["inverse_vol", "equal"] = "inverse_vol") -> SleeveSignal:
    """Long top tercile / short bottom tercile of the combined factor rank; inverse-vol (or equal)
    within each leg. target_weight is the signed within-side share (long > 0, short < 0)."""
    n = len(geometries)
    if n == 0:
        return SleeveSignal(sleeve="factor", tilts=[], risk_budget_frac=risk_budget_frac,
                            as_of_ts=now)
    geo_by_sym = {g.symbol: g for g in geometries}
    ranked = _combined_rank(geometries, factors)
    k = max(1, math.floor(n * tercile))
    long_syms = [s for s, _ in ranked[:k]]
    short_syms = [s for s, _ in ranked[-k:]]
    long_w = _inverse_vol_weights(long_syms, geo_by_sym, weighting)
    short_w = _inverse_vol_weights(short_syms, geo_by_sym, weighting)
    tilts: list[SleeveTilt] = []
    for s in long_syms:
        tilts.append(SleeveTilt(symbol=s, direction="long", target_weight=long_w[s]))
    for s in short_syms:
        tilts.append(SleeveTilt(symbol=s, direction="short", target_weight=-short_w[s]))
    return SleeveSignal(sleeve="factor", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"factors": factors, "k_per_side": k}, as_of_ts=now)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_factor.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/factor.py tests/sleeves/test_factor.py
git commit -m "feat: factor_signal (combined-rank tercile L/S, inverse-vol within leg)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 19: Sentiment — `conviction_tilt`

**Files:**
- Create: `futures_fund/sleeves/sentiment.py`
- Test: `tests/sleeves/test_sentiment.py`

- [ ] **Step 1: Write the failing test**

Create `tests/sleeves/test_sentiment.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from futures_fund.sleeves.sentiment import conviction_tilt

_NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def test_conviction_tilt_positive_sentiment_boosts_long():
    # w*(1 + kappa*s*conf) = 0.2 * (1 + 0.5*0.8*1.0) = 0.2 * 1.4 = 0.28
    assert abs(conviction_tilt(0.2, 0.8, 1.0, kappa=0.5) - 0.28) < 1e-9


def test_conviction_tilt_negative_sentiment_shrinks_long():
    # 0.2 * (1 + 0.5*(-0.8)*1.0) = 0.2 * 0.6 = 0.12
    assert abs(conviction_tilt(0.2, -0.8, 1.0, kappa=0.5) - 0.12) < 1e-9


def test_conviction_tilt_never_flips_sign():
    # huge negative sentiment cannot push a long weight negative
    out = conviction_tilt(0.2, -1.0, 1.0, kappa=5.0)
    assert out >= 0.0


def test_conviction_tilt_cap_limits_delta_to_25pct():
    # cap=0.25 -> |delta w| <= 25% of |w|, so max tilted long = 0.2 * 1.25 = 0.25
    out = conviction_tilt(0.2, 1.0, 1.0, kappa=5.0, cap=0.25)
    assert abs(out - 0.25) < 1e-9


def test_conviction_tilt_zero_weight_stays_zero():
    # sentiment never OPENS a position on its own
    assert conviction_tilt(0.0, 1.0, 1.0, kappa=0.5) == 0.0


def test_conviction_tilt_short_leg_negative_weight():
    # short leg w=-0.2, positive sentiment should SHRINK the short magnitude (toward 0)
    out = conviction_tilt(-0.2, 0.8, 1.0, kappa=0.5)
    assert -0.2 < out <= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_sentiment.py -v -k conviction_tilt`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.sleeves.sentiment'`.

- [ ] **Step 3: Write `conviction_tilt`**

Create `futures_fund/sleeves/sentiment.py`:

```python
"""Sentiment sleeve + per-coin conviction tilt (design spec §6.4, §7.2).

Two bounded shapers that NEVER flip direction and never open a position alone:
  1. conviction_tilt / apply_conviction_tilts — deterministic per-leg tilt within a +-cap band.
  2. sentiment_factor_signal — a standalone cross-sectional L/S sleeve on sentiment_score*conf.

Both run BEFORE the optimizer re-projects onto the dollar+beta-neutral set, so sentiment cannot
mathematically break neutrality or the risk gate (computed after).
"""
from __future__ import annotations

import math
from datetime import datetime

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt


def conviction_tilt(weight: float, sentiment_score: float, sentiment_conf: float, *,
                    kappa: float = 0.5, cap: float = 0.25) -> float:
    """Deterministic tilt: w*(1 + kappa*s*conf), clamped so |delta w| <= cap*|w|.

    NEVER flips sign (a tilted long stays >= 0, a tilted short stays <= 0) and NEVER opens a
    position alone (returns 0 if the input weight is 0).
    """
    if weight == 0.0:
        return 0.0
    factor = 1.0 + kappa * sentiment_score * sentiment_conf
    factor = max(1.0 - cap, min(1.0 + cap, factor))     # |delta w| <= cap*|w|
    tilted = weight * factor
    # sign-preserving guard (factor is clamped to [1-cap, 1+cap] > 0, so sign already holds; this
    # makes the invariant explicit and robust to cap >= 1)
    if weight > 0:
        return max(0.0, tilted)
    return min(0.0, tilted)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_sentiment.py -v -k conviction_tilt`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/sentiment.py tests/sleeves/test_sentiment.py
git commit -m "feat: conviction_tilt (bounded sign-preserving per-coin sentiment tilt)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 20: Sentiment — `apply_conviction_tilts`

**Files:**
- Modify: `futures_fund/sleeves/sentiment.py`
- Test: `tests/sleeves/test_sentiment.py`

> **Note (short-leg arithmetic):** the test asserts that B (a short leg, `w=-0.2`, positive sentiment `s=0.8`, `conf=1.0`) tilts to `-0.12`. `apply_conviction_tilts` signs sentiment by leg direction, so the short leg sees `signed_s = -0.8`, giving `factor = 1 + 0.5*(-0.8)*1.0 = 0.6` and `-0.2 * 0.6 = -0.12` — the short magnitude shrinks toward 0, as positive sentiment should. This callout is documentation, not an executable step; the sequence below remains 1 → 2 → 3 → 4 → 5.

- [ ] **Step 1: Write the failing test**

Append to `tests/sleeves/test_sentiment.py`:

```python
from futures_fund.contracts import CoinGeometry, SleeveTilt
from futures_fund.sleeves.sentiment import apply_conviction_tilts


def test_apply_conviction_tilts_maps_per_symbol_geometry():
    legs = [
        SleeveTilt(symbol="A/USDT:USDT", direction="long", target_weight=0.2),
        SleeveTilt(symbol="B/USDT:USDT", direction="short", target_weight=-0.2),
    ]
    geos = [
        CoinGeometry(symbol="A/USDT:USDT", mark=100.0, sentiment_score=0.8, sentiment_conf=1.0),
        CoinGeometry(symbol="B/USDT:USDT", mark=100.0, sentiment_score=0.8, sentiment_conf=1.0),
    ]
    out = apply_conviction_tilts(legs, geos, kappa=0.5, cap=0.25)
    by_sym = {t.symbol: t for t in out}
    # A: long boosted -> 0.2*1.4 = 0.28
    assert abs(by_sym["A/USDT:USDT"].target_weight - 0.28) < 1e-9
    # B: short leg, positive sentiment is UNFAVORABLE to a short, so its magnitude shrinks toward 0
    #    via signed_s = -0.8 -> factor 0.6 -> -0.2*0.6 = -0.12.
    assert abs(by_sym["B/USDT:USDT"].target_weight - (-0.12)) < 1e-9


def test_apply_conviction_tilts_missing_geometry_is_unchanged():
    legs = [SleeveTilt(symbol="Z/USDT:USDT", direction="long", target_weight=0.2)]
    out = apply_conviction_tilts(legs, [], kappa=0.5, cap=0.25)
    assert out[0].target_weight == 0.2              # no geometry -> no tilt (fail-soft neutral)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_sentiment.py -v -k apply_conviction_tilts`
Expected: FAIL with `ImportError: cannot import name 'apply_conviction_tilts'`.

- [ ] **Step 3: Add `apply_conviction_tilts`**

Append to `futures_fund/sleeves/sentiment.py`:

```python
def apply_conviction_tilts(legs: list[SleeveTilt], geometries: list[CoinGeometry], *,
                           kappa: float = 0.5, cap: float = 0.25) -> list[SleeveTilt]:
    """Map conviction_tilt over legs using each symbol's geometry. Sign-preserving, cap-respecting.

    Positive sentiment favors a LONG (boost) and disfavors a SHORT (shrink magnitude toward 0):
    the effective score is signed by the leg's direction so 'favorable to this leg' always boosts.
    A leg with no matching geometry is returned unchanged (fail-soft neutral).
    """
    geo_by_sym = {g.symbol: g for g in geometries}
    out: list[SleeveTilt] = []
    for leg in legs:
        g = geo_by_sym.get(leg.symbol)
        if g is None:
            out.append(leg.model_copy())
            continue
        # signed sentiment relative to the leg: positive sentiment is favorable to a long and
        # unfavorable to a short, so flip its sign for short legs.
        signed_s = g.sentiment_score if leg.direction == "long" else -g.sentiment_score
        new_w = conviction_tilt(leg.target_weight, signed_s, g.sentiment_conf,
                                kappa=kappa, cap=cap)
        out.append(leg.model_copy(update={"target_weight": new_w}))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_sentiment.py -v -k apply_conviction_tilts`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/sentiment.py tests/sleeves/test_sentiment.py
git commit -m "feat: apply_conviction_tilts (direction-signed, sign-preserving over legs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 21: Sentiment — `sentiment_factor_signal` (standalone L/S sleeve)

**Files:**
- Modify: `futures_fund/sleeves/sentiment.py`
- Test: `tests/sleeves/test_sentiment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/sleeves/test_sentiment.py`:

```python
from futures_fund.sleeves.sentiment import sentiment_factor_signal


def _sgeo(symbol, s, conf):
    return CoinGeometry(symbol=symbol, mark=100.0, sentiment_score=s, sentiment_conf=conf)


def test_sentiment_factor_signal_long_high_short_low():
    geos = [
        _sgeo("A/USDT:USDT", 0.9, 1.0),    # strong positive -> LONG
        _sgeo("B/USDT:USDT", 0.1, 0.5),
        _sgeo("C/USDT:USDT", -0.2, 0.5),
        _sgeo("D/USDT:USDT", -0.9, 1.0),   # strong negative -> SHORT
    ]
    sig = sentiment_factor_signal(geos, risk_budget_frac=0.25, now=_NOW, tercile=1 / 3)
    assert sig.sleeve == "sentiment"
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["A/USDT:USDT"].direction == "long"
    assert by_sym["D/USDT:USDT"].direction == "short"
    # score is sentiment_score * sentiment_conf
    assert by_sym["A/USDT:USDT"].raw_score == 0.9 * 1.0


def test_sentiment_factor_signal_empty():
    sig = sentiment_factor_signal([], risk_budget_frac=0.25, now=_NOW)
    assert sig.tilts == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_sentiment.py -v -k sentiment_factor_signal`
Expected: FAIL with `ImportError: cannot import name 'sentiment_factor_signal'`.

- [ ] **Step 3: Add `sentiment_factor_signal`**

Append to `futures_fund/sleeves/sentiment.py`:

```python
def sentiment_factor_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float,
                            now: datetime, tercile: float = 1 / 3) -> SleeveSignal:
    """Standalone cross-sectional L/S sleeve: long the highest (sentiment_score*sentiment_conf),
    short the lowest. Equal-weight within each side (the optimizer re-projects to neutral after)."""
    n = len(geometries)
    if n == 0:
        return SleeveSignal(sleeve="sentiment", tilts=[], risk_budget_frac=risk_budget_frac,
                            as_of_ts=now)
    scored = sorted(geometries, key=lambda g: g.sentiment_score * g.sentiment_conf, reverse=True)
    k = max(1, math.floor(n * tercile))
    longs = scored[:k]
    shorts = scored[-k:]
    long_w = 1.0 / k
    short_w = 1.0 / k
    tilts: list[SleeveTilt] = []
    for g in longs:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="long", target_weight=long_w,
                                raw_score=g.sentiment_score * g.sentiment_conf))
    for g in shorts:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="short", target_weight=-short_w,
                                raw_score=g.sentiment_score * g.sentiment_conf))
    return SleeveSignal(sleeve="sentiment", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"k_per_side": k}, as_of_ts=now)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_sentiment.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/sentiment.py tests/sleeves/test_sentiment.py
git commit -m "feat: sentiment_factor_signal (standalone sentiment L/S sleeve)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 22: Re-export the four sleeve builders from the package

**Files:**
- Modify: `futures_fund/sleeves/__init__.py`
- Test: `tests/sleeves/test_carry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/sleeves/test_carry.py`:

```python
def test_package_reexports_all_four_builders():
    from futures_fund.sleeves import (
        carry_signal,
        factor_signal,
        pairs_signal,
        sentiment_factor_signal,
    )
    assert callable(carry_signal)
    assert callable(pairs_signal)
    assert callable(factor_signal)
    assert callable(sentiment_factor_signal)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_carry.py -v -k reexports`
Expected: FAIL with `ImportError: cannot import name 'carry_signal' from 'futures_fund.sleeves'`.

- [ ] **Step 3: Add the re-exports**

Replace the contents of `futures_fund/sleeves/__init__.py` with:

```python
"""Alpha-sleeve signal generators: each emits a SleeveSignal of per-name tilts the optimizer
merges into one neutral book. The four sleeves are carry, pairs, factor, and sentiment.
"""
from __future__ import annotations

from futures_fund.sleeves.carry import carry_signal
from futures_fund.sleeves.factor import factor_signal
from futures_fund.sleeves.pairs import pairs_signal
from futures_fund.sleeves.sentiment import (
    apply_conviction_tilts,
    conviction_tilt,
    sentiment_factor_signal,
)

__all__ = [
    "carry_signal",
    "pairs_signal",
    "factor_signal",
    "sentiment_factor_signal",
    "conviction_tilt",
    "apply_conviction_tilts",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sleeves/test_carry.py -v -k reexports`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/__init__.py tests/sleeves/test_carry.py
git commit -m "feat: re-export the four sleeve builders + tilt helpers from sleeves package

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 23: Risk-parity allocation across the four sleeves

**Files:**
- Create: `futures_fund/sleeve_budget.py`
- Test: `tests/test_sleeve_budget.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sleeve_budget.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from futures_fund.contracts import SleeveSignal, SleeveTilt
from futures_fund.sleeve_budget import risk_parity_budgets

_NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def _sig(name: str, n_tilts: int = 1) -> SleeveSignal:
    tilts = [SleeveTilt(symbol=f"{name}{i}/USDT:USDT", direction="long", target_weight=0.1)
             for i in range(n_tilts)]
    return SleeveSignal(sleeve=name, tilts=tilts, as_of_ts=_NOW)


def test_risk_parity_budgets_equal_when_no_cov_and_all_active():
    sleeves = [_sig("carry"), _sig("pairs"), _sig("factor"), _sig("sentiment")]
    budgets = risk_parity_budgets(sleeves)
    assert set(budgets) == {"carry", "pairs", "factor", "sentiment"}
    assert all(abs(b - 0.25) < 1e-9 for b in budgets.values())
    assert abs(sum(budgets.values()) - 1.0) < 1e-9


def test_risk_parity_budgets_skip_empty_sleeves():
    # a sleeve with no tilts gets zero budget; the rest split 1.0 equally
    sleeves = [_sig("carry"), _sig("pairs", n_tilts=0), _sig("factor"), _sig("sentiment")]
    budgets = risk_parity_budgets(sleeves)
    assert budgets["pairs"] == 0.0
    assert abs(budgets["carry"] - 1 / 3) < 1e-9
    assert abs(sum(budgets.values()) - 1.0) < 1e-9


def test_risk_parity_budgets_inverse_vol_from_cov():
    # diagonal cov: variances [1, 4, 1, 1] -> inverse-vol weights ~ [1, 0.5, 1, 1]/sum
    sleeves = [_sig("carry"), _sig("pairs"), _sig("factor"), _sig("sentiment")]
    cov = np.diag([1.0, 4.0, 1.0, 1.0])
    budgets = risk_parity_budgets(sleeves, cov=cov)
    inv = np.array([1.0, 0.5, 1.0, 1.0])
    expected = inv / inv.sum()
    assert abs(budgets["pairs"] - expected[1]) < 1e-9
    assert abs(budgets["carry"] - expected[0]) < 1e-9
    assert abs(sum(budgets.values()) - 1.0) < 1e-9


def test_risk_parity_budgets_all_empty_returns_zeros():
    sleeves = [_sig("carry", 0), _sig("pairs", 0)]
    budgets = risk_parity_budgets(sleeves)
    assert all(b == 0.0 for b in budgets.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sleeve_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.sleeve_budget'`.

- [ ] **Step 3: Write `risk_parity_budgets`**

Create `futures_fund/sleeve_budget.py`:

```python
"""Risk-parity (or inverse-vol) budget allocation across the FOUR alpha sleeves.

Lives in its own module so all four sleeves can be budgeted before the Phase 1 optimizer exists;
the canonical contract (§2.11) addresses this as neutrality.risk_parity_budgets, so Phase 2 also
ships a futures_fund/neutrality.py stub (Task 23a) that re-exports this function. Budgets sum to
1.0 over the active (non-empty) sleeves and fill SleeveSignal.risk_budget_frac.
"""
from __future__ import annotations

import numpy as np

from futures_fund.contracts import SleeveSignal
from futures_fund.models import SleeveName


def risk_parity_budgets(sleeves: list[SleeveSignal],
                        *, cov: np.ndarray | None = None) -> dict[SleeveName, float]:
    """Assign each sleeve its risk budget. With no cov, active sleeves split 1.0 equally; with a
    cov (sleeve-return covariance, same order as `sleeves`), use inverse-vol (1/sigma) weights.
    Sleeves with no tilts get a 0.0 budget and are excluded from the split.
    """
    names = [s.sleeve for s in sleeves]
    active = [i for i, s in enumerate(sleeves) if s.tilts]
    out: dict[SleeveName, float] = {n: 0.0 for n in names}
    if not active:
        return out
    if cov is None:
        share = 1.0 / len(active)
        for i in active:
            out[names[i]] = share
        return out
    variances = np.diag(np.asarray(cov, dtype=float))
    inv_vol = {i: 1.0 / np.sqrt(variances[i]) for i in active if variances[i] > 0}
    total = sum(inv_vol.values())
    if total <= 0:
        share = 1.0 / len(active)
        for i in active:
            out[names[i]] = share
        return out
    for i, w in inv_vol.items():
        out[names[i]] = float(w / total)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sleeve_budget.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeve_budget.py tests/test_sleeve_budget.py
git commit -m "feat: risk_parity_budgets across the four sleeves (equal / inverse-vol)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 23a: `neutrality.py` re-export stub (contract name resolves)

**Files:**
- Create: `futures_fund/neutrality.py`
- Test: `tests/test_sleeve_budget.py`

The canonical contract (§2.11) addresses the allocator as `neutrality.risk_parity_budgets`. Phase 2 implements it in `sleeve_budget.py`; this task adds the Phase-2 `neutrality.py` stub that re-exports it so `from futures_fund.neutrality import risk_parity_budgets` resolves at the end of Phase 2 (Phase 1 owns and will extend `neutrality.py`; see the Cross-phase hand-off note in Assumptions).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sleeve_budget.py`:

```python
def test_neutrality_reexports_risk_parity_budgets():
    # the canonical contract name neutrality.risk_parity_budgets must resolve at the end of Phase 2
    from futures_fund.neutrality import risk_parity_budgets as nb
    from futures_fund.sleeve_budget import risk_parity_budgets as sb
    assert nb is sb                               # same object, not a divergent copy
    # and it actually works through the neutrality alias
    budgets = nb([_sig("carry"), _sig("pairs")])
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sleeve_budget.py -v -k neutrality`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.neutrality'`.

- [ ] **Step 3: Create the re-export stub**

Create `futures_fund/neutrality.py`:

```python
"""Neutrality / allocation surface.

Phase 1 OWNS this module (it will add optimize_book, dollar+beta-neutralization, the risk gate,
etc.). Phase 2 ships only the risk_parity_budgets re-export so the canonical contract name
`neutrality.risk_parity_budgets` (§2.11) resolves at the end of Phase 2. When Phase 1 expands this
module it MUST keep this re-export (or define an equivalent risk_parity_budgets) so the name does
not regress.
"""
from __future__ import annotations

from futures_fund.sleeve_budget import risk_parity_budgets

__all__ = ["risk_parity_budgets"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sleeve_budget.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/neutrality.py tests/test_sleeve_budget.py
git commit -m "feat: neutrality.py Phase-2 stub re-exporting risk_parity_budgets (contract name resolves)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 24: Walk-forward split harness

**Files:**
- Create: `futures_fund/walk_forward.py`
- Test: `tests/test_walk_forward.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_walk_forward.py`:

```python
from __future__ import annotations

from futures_fund.walk_forward import walk_forward_splits


def test_walk_forward_splits_anchored_expanding():
    # 100 points, 4 folds: each fold trains on a growing prefix, tests on the next chunk.
    splits = walk_forward_splits(100, n_splits=4, min_train=20)
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert train_idx.stop <= test_idx.start          # no overlap: train strictly before test
        assert train_idx.start == 0                       # anchored (expanding window)
    # test chunks are contiguous and cover the tail
    assert splits[0][1].start >= 20
    assert splits[-1][1].stop == 100


def test_walk_forward_splits_too_short_returns_empty():
    assert walk_forward_splits(10, n_splits=4, min_train=20) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walk_forward.py -v -k splits`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.walk_forward'`.

- [ ] **Step 3: Write `walk_forward_splits`**

Create `futures_fund/walk_forward.py`:

```python
"""Walk-forward validation harness hook for sleeve params/thresholds (design spec §12, §15).

Anchored (expanding-window) out-of-sample splits + a DSR/overfit gate over the vendored
overfit_detector. A sleeve param change is only trusted if it clears the OOS Deflated-Sharpe
threshold, not an in-sample grid win.
"""
from __future__ import annotations

from futures_fund.graduation import deflated_sharpe_pvalue
from futures_fund.metrics import sharpe


def walk_forward_splits(n_obs: int, *, n_splits: int = 4,
                        min_train: int = 20) -> list[tuple[range, range]]:
    """Anchored (expanding-window) walk-forward splits over a length-`n_obs` series.

    Returns a list of (train_range, test_range): each train range is a prefix [0, t) growing fold
    by fold, and the test range is the next contiguous OOS chunk. Empty list if n_obs is too short
    to leave min_train training points plus at least one test point per split.
    """
    if n_obs < min_train + n_splits:
        return []
    test_total = n_obs - min_train
    chunk = test_total // n_splits
    if chunk < 1:
        return []
    splits: list[tuple[range, range]] = []
    for k in range(n_splits):
        train_stop = min_train + k * chunk
        test_start = train_stop
        test_stop = n_obs if k == n_splits - 1 else train_stop + chunk
        splits.append((range(0, train_stop), range(test_start, test_stop)))
    return splits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_walk_forward.py -v -k splits`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: walk_forward_splits (anchored expanding-window OOS folds)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 25: Walk-forward sleeve-param validation gate

**Files:**
- Modify: `futures_fund/walk_forward.py`
- Test: `tests/test_walk_forward.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_walk_forward.py`:

```python
import numpy as np

from futures_fund.walk_forward import validate_sleeve_param


def test_validate_sleeve_param_genuine_edge_passes():
    rng = np.random.default_rng(0)
    # strong positive-mean OOS returns across folds -> should clear the DSR gate
    oos_returns = [list(rng.normal(0.02, 0.01, 40)) for _ in range(4)]
    res = validate_sleeve_param(oos_returns, num_trials=4, periods_per_year=365.0,
                                dsr_threshold=0.95)
    assert res["passed"] is True
    assert res["oos_sharpe"] > 0
    assert res["dsr_pvalue"] >= 0.95


def test_validate_sleeve_param_noise_fails():
    rng = np.random.default_rng(1)
    # zero-mean noise -> no edge -> gate rejects
    oos_returns = [list(rng.normal(0.0, 0.02, 40)) for _ in range(4)]
    res = validate_sleeve_param(oos_returns, num_trials=20, periods_per_year=365.0,
                                dsr_threshold=0.95)
    assert res["passed"] is False


def test_validate_sleeve_param_empty_fails():
    res = validate_sleeve_param([], num_trials=4, periods_per_year=365.0)
    assert res["passed"] is False
    assert res["oos_sharpe"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walk_forward.py -v -k validate_sleeve_param`
Expected: FAIL with `ImportError: cannot import name 'validate_sleeve_param'`.

- [ ] **Step 3: Add `validate_sleeve_param`**

Append to `futures_fund/walk_forward.py`:

```python
def validate_sleeve_param(oos_returns: list[list[float]], *, num_trials: int,
                          periods_per_year: float = 365.0,
                          dsr_threshold: float = 0.95) -> dict:
    """Gate a sleeve param/threshold change on out-of-sample evidence.

    `oos_returns` is one return stream per walk-forward fold. Pools the folds, computes the OOS
    Sharpe (annualized at `periods_per_year` — 365 daily / 52 weekly) and the Deflated-Sharpe
    p-value deflated for `num_trials` (the number of param candidates tried). passed iff the OOS
    Sharpe is > 0 AND the DSR p-value clears `dsr_threshold`.
    """
    pooled: list[float] = [r for fold in oos_returns for r in fold]
    if len(pooled) < 10:
        return {"passed": False, "oos_sharpe": 0.0, "dsr_pvalue": 0.0, "n_obs": len(pooled)}
    oos_sharpe = sharpe(pooled, periods_per_year=periods_per_year)
    dsr_p = deflated_sharpe_pvalue(pooled, num_trials=num_trials,
                                   periods_per_year=periods_per_year)
    passed = oos_sharpe > 0 and dsr_p >= dsr_threshold
    return {"passed": passed, "oos_sharpe": oos_sharpe, "dsr_pvalue": dsr_p, "n_obs": len(pooled)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_walk_forward.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: validate_sleeve_param (OOS Sharpe + DSR gate over walk-forward folds)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 26: Integration — full four-sleeve pipeline with risk-parity budgets

**Files:**
- Test: `tests/sleeves/test_pipeline.py` (Create)

This task wires the four sleeves + `risk_parity_budgets` + `apply_conviction_tilts` end-to-end (no new production code — it locks the cross-module contract the Phase 1 optimizer consumes).

- [ ] **Step 1: Write the failing integration test**

Create `tests/sleeves/test_pipeline.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from futures_fund.contracts import CoinGeometry, Pair, Spread
from futures_fund.sleeve_budget import risk_parity_budgets
from futures_fund.sleeves import (
    apply_conviction_tilts,
    carry_signal,
    factor_signal,
    pairs_signal,
    sentiment_factor_signal,
)

_NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def _universe() -> list[CoinGeometry]:
    rows = [
        # symbol, momentum, funding_apr, vol, sentiment, conf
        ("BTC/USDT:USDT", 0.20, -0.10, 0.30, 0.6, 0.9),
        ("ETH/USDT:USDT", 0.10, 0.05, 0.40, 0.2, 0.7),
        ("SOL/USDT:USDT", 0.30, 0.30, 0.60, 0.8, 0.8),
        ("XRP/USDT:USDT", -0.20, 0.20, 0.50, -0.7, 0.9),
        ("BNB/USDT:USDT", -0.10, -0.25, 0.35, -0.3, 0.6),
        ("ADA/USDT:USDT", -0.30, 0.10, 0.55, -0.9, 0.9),
    ]
    return [CoinGeometry(symbol=s, mark=100.0, momentum_20=m, funding_apr=f,
                         realized_vol=v, sentiment_score=sc, sentiment_conf=cf)
            for s, m, f, v, sc, cf in rows]


def _pair_and_spread() -> tuple[Pair, Spread]:
    pair = Pair(pair_id="BTCUSDT__ETHUSDT", symbol_y="BTC/USDT:USDT",
                symbol_x="ETH/USDT:USDT", hedge_ratio=2.0, method="engle_granger",
                adf_pvalue=0.01, adf_pvalue_adj=0.02, half_life=5.0, theta=0.139, mu=0.0,
                sigma_eq=10.0, formed_cycle=1)
    spread = Spread(pair_id=pair.pair_id, spread_value=25.0, zscore=2.5, state="short_spread")
    return pair, spread


def test_full_pipeline_produces_budgeted_neutral_ready_signals():
    geos = _universe()
    pair, spread = _pair_and_spread()
    sleeves = [
        carry_signal(geos, risk_budget_frac=0.0, now=_NOW),
        pairs_signal([pair], [spread], risk_budget_frac=0.0, now=_NOW),
        factor_signal(geos, risk_budget_frac=0.0, now=_NOW, factors=["momentum"], tercile=1 / 3),
        sentiment_factor_signal(geos, risk_budget_frac=0.0, now=_NOW, tercile=1 / 3),
    ]
    # all four sleeves emit at least one tilt for this universe
    assert all(s.tilts for s in sleeves)

    # risk-parity budgets sum to 1.0 across the four active sleeves
    budgets = risk_parity_budgets(sleeves)
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    assert set(budgets) == {"carry", "pairs", "factor", "sentiment"}

    # conviction tilts applied to the factor sleeve never flip direction
    tilted = apply_conviction_tilts(sleeves[2].tilts, geos, kappa=0.5, cap=0.25)
    for before, after in zip(sleeves[2].tilts, tilted):
        if before.target_weight > 0:
            assert after.target_weight >= 0
        elif before.target_weight < 0:
            assert after.target_weight <= 0
        # cap respected: |delta| <= 25% of |w|
        assert abs(after.target_weight - before.target_weight) <= 0.25 * abs(
            before.target_weight) + 1e-9


def test_pairs_sleeve_legs_carry_pair_id_for_attribution():
    pair, spread = _pair_and_spread()
    sig = pairs_signal([pair], [spread], risk_budget_frac=0.0, now=_NOW)
    assert sig.tilts                                  # short_spread -> two legs
    assert all(t.pair_id == pair.pair_id for t in sig.tilts)
    assert pair.pair_id == "BTCUSDT__ETHUSDT"         # canonical slash-free id everywhere
```

- [ ] **Step 2: Run test to verify it fails (or passes against existing code)**

Run: `uv run pytest tests/sleeves/test_pipeline.py -v`
Expected: this test exercises only already-implemented functions; it should PASS immediately. If any assertion FAILS, that signals a cross-module contract bug — debug using `superpowers:systematic-debugging` before proceeding (do not weaken the assertion).

- [ ] **Step 3: Run the complete Phase 2 suite + lint**

Run: `uv run pytest tests/ -q && uv run ruff check futures_fund/ tests/`
Expected: all Phase 2 tests PASS; ruff prints `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add tests/sleeves/test_pipeline.py
git commit -m "test: end-to-end four-sleeve pipeline (budgeted, conviction-tilted, attribution)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 27: Verify `config.yaml` sleeve/sentiment/pairs blocks present

**Files:**
- Modify: `config.yaml`
- Test: `tests/test_config_phase2.py` (Create)

The builders bake the contract defaults into their keyword args, so config is read by the Phase 3 control loop, not Phase 2 directly. This task asserts the config keys Phase 3 will read exist with the documented defaults so nothing drifts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_phase2.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

_CFG = yaml.safe_load(Path("config.yaml").read_text())


def test_sleeves_block_lists_four_sleeves():
    assert _CFG["sleeves"]["enabled"] == ["carry", "pairs", "factor", "sentiment"]
    assert _CFG["sleeves"]["risk_parity"] is True


def test_pairs_block_thresholds():
    p = _CFG["sleeves"]["pairs"]
    assert p["adf_pvalue_max"] == 0.05
    assert p["fdr_method"] == "bh"
    assert p["entry_z"] == 2.0
    assert p["exit_z"] == 0.0
    assert p["stop_z"] == 3.0


def test_sentiment_block_defaults():
    s = _CFG["sentiment"]
    assert s["kappa"] == 0.5
    assert s["cap"] == 0.25
    assert s["halflife_days"] == 3


def test_factor_block_defaults():
    f = _CFG["sleeves"]["factor"]
    assert f["factors"] == ["momentum", "carry", "low_vol"]
    assert f["weighting"] == "inverse_vol"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_phase2.py -v`
Expected: FAIL — either `FileNotFoundError: config.yaml` or `KeyError: 'sleeves'` (the block is absent).

- [ ] **Step 3: Add/confirm the config blocks**

Ensure `config.yaml` at the project root contains these blocks (append if missing; if a `config.yaml` already exists from Phase 0, merge these keys in without disturbing existing ones):

```yaml
sleeves:
  risk_parity: true
  enabled: ["carry", "pairs", "factor", "sentiment"]
  factor:
    factors: ["momentum", "carry", "low_vol"]
    tercile: 0.3333
    weighting: "inverse_vol"
  pairs:
    adf_pvalue_max: 0.05
    fdr_method: "bh"
    entry_z: 2.0
    exit_z: 0.0
    stop_z: 3.0
    min_half_life_cycles: 1.0
    max_half_life_cycles: 40.0
    rolling_retest_cycles: 7

sentiment:
  kappa: 0.5
  cap: 0.25
  halflife_days: 3
  refresh_daily: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_phase2.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add config.yaml tests/test_config_phase2.py
git commit -m "chore: pin sleeves/pairs/sentiment config blocks with Phase 2 defaults

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 28: Full-suite green + lint gate (phase exit)

**Files:** none (verification only).

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest -q`
Expected: all Phase 2 tests PASS, 0 failures. (Counts depend on Phase 0/1 tests also present in `tests/`.)

- [ ] **Step 2: Run the linter over all new code**

Run: `uv run ruff check .`
Expected: prints `All checks passed!`. If it flags import sorting (`I`) or unused imports (`F401`), run `uv run ruff check --fix .` then re-run `uv run pytest -q` to confirm still green.

- [ ] **Step 3: Confirm no placeholder/TODO leaked into shipped modules**

Run: `grep -rn "TODO\|FIXME\|NotImplemented\|placeholder" futures_fund/cointegration.py futures_fund/sleeves/ futures_fund/sleeve_budget.py futures_fund/neutrality.py futures_fund/walk_forward.py`
Expected: no matches (empty output).

- [ ] **Step 4: Commit the phase-completion marker (only if any auto-fixes were applied)**

```bash
git add -A
git commit -m "chore: Phase 2 green — four sleeves + Pair object + sentiment factor complete

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(If Step 1–3 produced no changes, skip Step 4 — there is nothing to commit.)

---

## Self-Review

**Spec coverage (each Phase 2 scope item → task):**
- cointegration.py Engle-Granger ADF → Task 6; optional Johansen → Task 7; OU half-life=ln2/θ → Task 8; z-score machinery (`spread_value`/`zscore`/`spread_state`) → Task 9; FDR/Bonferroni across candidate pairs → Task 10; `build_pair`/`build_spread` assembly → Tasks 11–12.
- First-class `Pair`/`Spread` object: hedge_ratio fields → Task 4; hedge-ratio leg sizing (spread = traded unit) → Task 16; joint entry|z|≥2 / exit~0 / hard stop|z|≥3 → Task 9 (`spread_state`), surfaced in Tasks 12 & 16; pair-level PnL attribution (`Spread.realized_pnl` + `pair_id` on every leg, canonical slash-free `pair_id` per Task 4) → Tasks 4, 16, 26; rolling re-test (`Pair.cointegrated` consumed by `select_pairs`) → Task 15; Johansen-branch cointegration judged by trace_stat > crit_95 (not the EG ADF p) → Task 11.
- Four sleeve generators in `futures_fund/sleeves/`: carry.py (signed funding rank, un-clamped) → Task 14; pairs.py (emits Pair-based SleeveSignal) → Tasks 15–16; factor.py (momentum/carry/low-vol cross-sectional tercile L/S) → Tasks 17–18; sentiment.py (cross-sectional sentiment L/S + per-coin conviction tilt) → Tasks 19–21.
- Risk-parity allocation across the four sleeves → Task 23 (`sleeve_budget.py`) + Task 23a (`neutrality.py` re-export so the contract name `neutrality.risk_parity_budgets` resolves in Phase 2).
- Fully TDD → every task is failing-test → run-fail → minimal-impl → run-pass → commit.
- Walk-forward validation harness hook (reuse graduation/overfit_detector) → Tasks 24–25 (`validate_sleeve_param` calls `graduation.deflated_sharpe_pvalue`, which wraps `vendor/overfit_detector.deflated_sharpe_ratio`).
- Produces SleeveSignals consumed by Phase 1 → `SleeveSignal`/`SleeveTilt` contracts (Task 5) + the integration test (Task 26) lock the consumed shape.

**Placeholder scan:** no "TBD/TODO/implement later"; every code step shows complete real code; every test step shows the actual assertions; the only `grep` for TODO is the negative phase-exit check in Task 28.

**Type consistency:** all referenced symbols are defined in this plan or in the cited reused modules: `CoinGeometry`, `Pair`, `Spread`, `SleeveTilt`, `SleeveSignal`, `SentimentReport` (Tasks 2–5); cointegration functions `engle_granger`/`johansen`/`ou_fit`/`half_life`/`spread_value`/`zscore`/`spread_state`/`fdr_adjust`/`build_pair`/`build_spread` (Tasks 6–12); sleeve builders `carry_signal`/`select_pairs`/`pairs_signal`/`rank_factor`/`factor_signal`/`conviction_tilt`/`apply_conviction_tilts`/`sentiment_factor_signal` (Tasks 14–21) all match the canonical contract signatures verbatim; `risk_parity_budgets` (Task 23, re-exported via `neutrality.py` in Task 23a) and `walk_forward_splits`/`validate_sleeve_param` (Tasks 24–25) match the contract. The reused names actually referenced by this Phase 2 plan are `sharpe` and `PERIODS_PER_YEAR` (from `metrics.py`) and `deflated_sharpe_pvalue` (from `graduation.py`, which transitively wraps `vendor/overfit_detector.deflated_sharpe_ratio`) — all via Task 25 — and they are used exactly as defined in the weekly source files read during planning. (`project_funding` and `cycle_due` are NOT used anywhere in Phase 2 — carry-PnL settlement and scheduling belong to Phase 1/3 — so they are deliberately not listed here.) The `spread_state` semantics (z≥+entry ⇒ short the rich spread; z≤−entry ⇒ long the cheap spread) are used identically in `build_spread` (Task 12) and `pairs_signal` (Task 16). The `pair_id` format is the single canonical slash-free `"<SYMY>__<SYMX>"` form (Task 4), produced by `build_pair` via `_canonical_pair_id` (Task 11) and asserted in that exact form everywhere it appears (Tasks 4, 11, 12, 26).

**Deviation note (intentional, documented):** the canonical contract places `risk_parity_budgets` in `neutrality.py` (§2.11), a Phase 1 module. To keep Phase 2 self-contained and executable directly (Phase 1's optimizer is out of scope here), the implementation lives in the net-new `futures_fund/sleeve_budget.py`, and Phase 2 also ships a `futures_fund/neutrality.py` stub (Task 23a) that re-exports it so `neutrality.risk_parity_budgets` resolves at the end of Phase 2 (no dangling cross-phase obligation). Phase 1 owns `neutrality.py` long-term and MUST preserve the re-export when it adds `optimize_book` / beta-neutralization (see the Cross-phase hand-off note in Assumptions). This is the single, explicitly-flagged structural divergence; the function signature and behavior match the contract exactly.

**Execution handoff:** Plan complete and saved to `docs/superpowers/plans/2026-06-11-phase2-sleeves.md`. Recommended execution: superpowers:subagent-driven-development (fresh subagent per task, two-stage review between tasks).

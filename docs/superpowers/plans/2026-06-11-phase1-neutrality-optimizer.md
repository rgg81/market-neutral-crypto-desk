# Phase 1 — Neutrality + Portfolio Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic dollar+beta-neutral portfolio optimizer (`neutrality.py`) and its rolling-BTC-beta input (`beta.py`) that turn `SleeveSignal`s + `CoinGeometry` into a `TargetWeights` book — dollar-neutral and beta-neutral within bands, with a budget-internal BTC hedge leg that is a real degree of freedom (sized to absorb the alpha legs' residual beta), a ≥90%/side deployment floor enforced (not just reported) with ≤(1−dry_powder) headroom, per-name + per-cluster caps, Ledoit-Wolf shrunk covariance → HRP/risk-parity weighting that actually shapes per-name notionals, an L1 turnover penalty + no-trade drift band applied BEFORE the final projection, and a sentiment-tilt-then-re-project step that never flips sign.

**Architecture:** Pure-math spine, no I/O. `beta.py` computes rolling OLS beta to BTC from mark-price return series. `neutrality.py` merges risk-budgeted sleeve tilts into a signed weight vector, applies sentiment conviction tilts BEFORE projection, shapes per-name weights via Ledoit-Wolf→HRP, enforces per-name/per-cluster caps (reusing the weekly desk's union-find cluster logic), applies the turnover/no-trade band against the prior book, **sizes a BTC hedge leg to absorb the alpha legs' residual beta inside the per-side budget**, projects the alpha+hedge vector onto the dollar+beta-neutral constraint set, then **scales the neutral unit book up to the per-side deployment target — a single positive scale that preserves both neutralities exactly** — landing each side in `[floor, 1−dry_powder]`, and assembles a `TargetWeights` with re-derived residuals and per-side deployment. Every step is TDD with synthetic price/beta fixtures and property tests.

**Tech Stack:** Python 3.11, `uv`, `pydantic>=2.6`, `numpy`, `pandas`, `scipy`, `scikit-learn` (Ledoit-Wolf), `pytest`, `ruff`. New types lifted verbatim from `/home/roberto/crypto-trade-claude-code-weekly/futures_fund/models.py`; cluster union-find adapted from that repo's `portfolio_risk.py`.

---

## File Structure

This phase establishes the package skeleton (if Phase 0 has not already) and builds exactly the two net-new math modules of §17 step 1, plus the four pydantic contract types they consume.

| File | Create/Modify | Single responsibility |
|---|---|---|
| `pyproject.toml` | Create/Modify | Package metadata + deps; adds `scikit-learn` for Ledoit-Wolf shrinkage. |
| `futures_fund/__init__.py` | Create | Package marker (empty). |
| `futures_fund/models.py` | Create | Lifted-verbatim reused base types (`Direction`, `SymbolSpec`, `TradeProposal`, …) + new shared type aliases (`SleeveName`, `SentimentLevel`, `SpreadState`, `PairTestMethod`, `Cadence`). |
| `futures_fund/contracts.py` | Create | Phase-1 pydantic contracts the optimizer consumes/produces: `CoinGeometry`, `GeometryBundle`, `SleeveTilt`, `SleeveSignal`, `WeightLeg`, `TargetWeights`. |
| `futures_fund/beta.py` | Create | Rolling OLS beta to BTC from mark-price return series (`log_returns`, `rolling_beta`, `beta_series`, `beta_for_symbols`). |
| `futures_fund/neutrality.py` | Create | The optimizer + `NeutralityConfig`: residual measures, Ledoit-Wolf covariance, HRP weights, risk-parity sleeve budgets, sleeve merge, conviction-tilt re-projection ordering, per-name/cluster caps, `project_neutral`, `size_btc_hedge`, deployment floor + dry powder, L1 turnover penalty, `optimize_book`. |
| `tests/__init__.py` | Create | Test package marker (empty). |
| `tests/conftest.py` | Create | Synthetic price/beta/geometry/sleeve fixtures shared across the beta + neutrality tests. |
| `tests/test_beta.py` | Create | Unit + property tests for `beta.py`. |
| `tests/test_neutrality_contracts.py` | Create | Validation tests for the new contract models. |
| `tests/test_neutrality_residuals.py` | Create | Tests for `dollar_residual`, `beta_residual`, `NeutralityConfig`. |
| `tests/test_neutrality_weighting.py` | Create | Tests for `ledoit_wolf_cov`, `hrp_weights`, `risk_parity_budgets`, `merge_sleeves`. |
| `tests/test_neutrality_caps.py` | Create | Tests for per-name + per-cluster caps and the reused cluster union-find. |
| `tests/test_neutrality_project.py` | Create | Tests for `project_neutral`, `size_btc_hedge`, conviction-tilt ordering. |
| `tests/test_neutrality_optimize.py` | Create | End-to-end + property tests for `optimize_book` (neutrality residuals in band; deployment floor honored; HRP shapes notionals; sentiment never flips direction; gross ≈ $20k). |

**Note on Phase 0:** if a prior phase already created `pyproject.toml`, `futures_fund/__init__.py`, `futures_fund/models.py`, `futures_fund/contracts.py`, or `tests/__init__.py`, treat the corresponding "Create" steps as "verify present and merge the new symbols in" — do not clobber existing content; only add the symbols this plan defines that are missing. Run `uv run pytest -q` after merging to confirm nothing regressed.

---

## Task 1: Bootstrap package, dependencies, and reused base types

**Files:**
- Create: `pyproject.toml`
- Create: `futures_fund/__init__.py`
- Create: `futures_fund/models.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "futures-fund"
version = "0.1.0"
description = "Market-neutral crypto trading desk — deterministic spine"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "numpy>=1.26",
    "pandas>=2.1",
    "ccxt>=4.5",
    "httpx>=0.27",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.ruff.lint.per-file-ignores]
"futures_fund/vendor/*" = ["E", "F", "I", "UP", "B"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["futures_fund"]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves and installs deps including `scikit-learn`; prints an "Installed N packages" / "Resolved" summary and exits 0.

- [ ] **Step 3: Create empty package markers**

Create `futures_fund/__init__.py` with a single line:

```python
"""Market-neutral crypto trading desk — deterministic spine."""
```

Create `tests/__init__.py` empty:

```python
```

- [ ] **Step 4: Create `futures_fund/models.py` with lifted base types + new shared aliases**

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Direction = Literal["long", "short"]
RegimeQuadrant = Literal[
    "low_vol_trend", "high_vol_trend", "low_vol_range", "high_vol_range", "transition"
]
HealthTier = Literal["healthy", "caution", "stressed"]
Bias = Literal["normal", "reduce", "flat"]
Verdict = Literal["approve", "resize", "veto"]

# --- new shared type aliases (market-neutral desk) ---
SleeveName = Literal["carry", "pairs", "factor", "sentiment"]
SentimentLevel = Literal[
    "very_positive", "positive", "neutral", "negative", "very_negative"
]
SpreadState = Literal["flat", "long_spread", "short_spread", "stop"]
PairTestMethod = Literal["engle_granger", "johansen"]
Cadence = Literal["weekly", "daily"]


class MmrBracket(BaseModel):
    notional_floor: float
    notional_cap: float
    mmr: float
    maint_amount: float
    max_leverage: float


class SymbolSpec(BaseModel):
    symbol: str
    tick_size: float
    step_size: float
    min_notional: float
    mmr_brackets: list[MmrBracket]

    @property
    def sorted_brackets(self) -> list[MmrBracket]:
        return sorted(self.mmr_brackets, key=lambda b: b.notional_floor)


class TradeProposal(BaseModel):
    symbol: str
    direction: Direction
    entry: float
    stop: float
    take_profits: list[float] = Field(default_factory=list)
    atr: float
    confidence: float = Field(ge=0.0, le=1.0)
    horizon_hours: float = Field(gt=0)
    funding_rate: float
    funding_interval_hours: float = Field(default=8.0, gt=0)
    risk_mult: float = 1.0

    @model_validator(mode="after")
    def _check_stop_side(self) -> TradeProposal:
        if self.direction == "long" and self.stop >= self.entry:
            raise ValueError("long stop must be below entry")
        if self.direction == "short" and self.stop <= self.entry:
            raise ValueError("short stop must be above entry")
        return self

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)


class CostEstimate(BaseModel):
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    funding: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return self.entry_fee + self.exit_fee + self.funding + self.slippage


class SizedTrade(BaseModel):
    proposal: TradeProposal
    qty: float
    notional: float
    leverage: float
    margin: float
    liq_price: float
    cost: CostEstimate


class RegimeState(BaseModel):
    quadrant: RegimeQuadrant
    trend_direction: Literal["up", "down", "neutral"] = "neutral"
    hurst: float = 0.5


class PortfolioHealth(BaseModel):
    equity: float
    peak_equity: float
    open_heat: float = 0.0
    recent_hit_rate: float = 0.5

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def tier(self) -> HealthTier:
        dd = self.drawdown_from_peak
        if dd >= 0.40:
            return "stressed"
        if dd >= 0.20:
            return "caution"
        return "healthy"


class RiskCaps(BaseModel):
    max_leverage: float
    per_trade_risk_pct: float
    max_heat: float
    bias: Bias


class RiskDecision(BaseModel):
    verdict: Verdict
    reason: str
    sized_trade: SizedTrade | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_sized(self) -> RiskDecision:
        if self.verdict in ("approve", "resize") and self.sized_trade is None:
            raise ValueError(f"verdict '{self.verdict}' requires a sized_trade")
        return self
```

- [ ] **Step 5: Verify the module imports and aliases resolve**

Run: `uv run python -c "from futures_fund.models import Direction, SleeveName, SpreadState, Cadence, TradeProposal; print('ok')"`
Expected: prints `ok` and exits 0.

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check futures_fund/models.py pyproject.toml`
Expected: `All checks passed!`

```bash
git checkout -b phase1-neutrality
git add pyproject.toml futures_fund/__init__.py futures_fund/models.py tests/__init__.py
git commit -m "feat: bootstrap package, deps, and lifted base types for market-neutral desk

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Phase-1 pydantic contracts (`CoinGeometry`, `SleeveSignal`, `TargetWeights`)

**Files:**
- Create: `futures_fund/contracts.py`
- Test: `tests/test_neutrality_contracts.py`

- [ ] **Step 1: Write the failing contract test**

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from futures_fund.contracts import (
    CoinGeometry,
    GeometryBundle,
    SleeveSignal,
    SleeveTilt,
    TargetWeights,
    WeightLeg,
)

NOW = datetime(2026, 6, 11, tzinfo=UTC)


def test_coin_geometry_defaults_and_sentiment_bounds():
    g = CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0)
    assert g.beta_btc == 1.0
    assert g.beta_lookback_days == 45
    assert g.funding_interval_hours == 8.0
    assert g.sentiment_score == 0.0
    assert g.sentiment_conf == 0.0
    assert g.in_pair is False


def test_coin_geometry_rejects_out_of_range_sentiment():
    with pytest.raises(ValidationError):
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, sentiment_score=1.5)
    with pytest.raises(ValidationError):
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, sentiment_conf=-0.1)


def test_geometry_bundle_holds_list():
    b = GeometryBundle(
        geometries=[CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0)], as_of_ts=NOW
    )
    assert len(b.geometries) == 1
    assert b.as_of_ts == NOW


def test_sleeve_signal_budget_bounds():
    tilt = SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.2)
    s = SleeveSignal(sleeve="carry", tilts=[tilt], risk_budget_frac=0.25, as_of_ts=NOW)
    assert s.sleeve == "carry"
    assert s.tilts[0].direction == "short"
    with pytest.raises(ValidationError):
        SleeveSignal(sleeve="carry", risk_budget_frac=1.5, as_of_ts=NOW)


def test_target_weights_assembles_residual_fields():
    leg = WeightLeg(
        symbol="BTC/USDT:USDT",
        direction="long",
        weight=0.45,
        target_notional=9000.0,
        beta_btc=1.0,
        sleeve="factor",
    )
    tw = TargetWeights(
        legs=[leg],
        btc_hedge_notional=-500.0,
        dollar_residual=0.0,
        dollar_residual_frac=0.0,
        beta_residual=0.01,
        gross_long=9000.0,
        gross_short=9000.0,
        deploy_long_frac=0.9,
        deploy_short_frac=0.9,
        gross_notional=18000.0,
        as_of_ts=NOW,
    )
    assert tw.feasible is True
    assert tw.turnover_l1 == 0.0
    assert tw.legs[0].sleeve == "factor"
    assert tw.btc_hedge_notional == -500.0


def test_weight_leg_allows_hedge_sleeve_literal():
    leg = WeightLeg(
        symbol="BTC/USDT:USDT",
        direction="short",
        weight=-0.05,
        target_notional=-1000.0,
        beta_btc=1.0,
        sleeve="hedge",
    )
    assert leg.sleeve == "hedge"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_neutrality_contracts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.contracts'`.

- [ ] **Step 3: Create `futures_fund/contracts.py`**

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from futures_fund.models import Direction, SleeveName, SymbolSpec


class CoinGeometry(BaseModel):
    symbol: str
    mark: float
    # momentum / vol / beta
    momentum_20: float = 0.0
    realized_vol: float = 0.0
    beta_btc: float = 1.0
    beta_lookback_days: int = 45
    # carry
    funding_rate: float = 0.0
    funding_interval_hours: float = 8.0
    funding_apr: float = 0.0
    funding_cap: float = 0.02
    # cointegration state
    in_pair: bool = False
    pair_id: str | None = None
    # sentiment (first-class)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    # liquidity / filters
    adv_usd: float = 0.0
    spec: SymbolSpec | None = None


class GeometryBundle(BaseModel):
    geometries: list[CoinGeometry] = Field(default_factory=list)
    as_of_ts: datetime


class SleeveTilt(BaseModel):
    symbol: str
    direction: Direction
    target_weight: float
    raw_score: float = 0.0
    pair_id: str | None = None


class SleeveSignal(BaseModel):
    sleeve: SleeveName
    tilts: list[SleeveTilt] = Field(default_factory=list)
    risk_budget_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    diagnostics: dict = Field(default_factory=dict)
    as_of_ts: datetime


class WeightLeg(BaseModel):
    symbol: str
    direction: Direction
    weight: float
    target_notional: float
    beta_btc: float
    sleeve: SleeveName | Literal["hedge"]
    pair_id: str | None = None


class TargetWeights(BaseModel):
    legs: list[WeightLeg] = Field(default_factory=list)
    btc_hedge_notional: float = 0.0
    # neutrality residuals
    dollar_residual: float
    dollar_residual_frac: float
    beta_residual: float
    # deployment per side
    gross_long: float
    gross_short: float
    deploy_long_frac: float
    deploy_short_frac: float
    gross_notional: float
    turnover_l1: float = 0.0
    feasible: bool = True
    notes: list[str] = Field(default_factory=list)
    as_of_ts: datetime
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_neutrality_contracts.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check futures_fund/contracts.py tests/test_neutrality_contracts.py`
Expected: `All checks passed!`

```bash
git add futures_fund/contracts.py tests/test_neutrality_contracts.py
git commit -m "feat: add Phase-1 optimizer contracts (CoinGeometry, SleeveSignal, TargetWeights)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Shared synthetic fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/conftest.py` with deterministic synthetic fixtures**

```python
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt

NOW = datetime(2026, 6, 11, tzinfo=UTC)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture
def btc_returns(rng: np.random.Generator) -> pd.Series:
    """120 synthetic BTC log-returns, mean ~0, sd ~0.02."""
    idx = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    return pd.Series(rng.normal(0.0, 0.02, size=120), index=idx)


@pytest.fixture
def beta_returns(btc_returns: pd.Series, rng: np.random.Generator):
    """Factory: build an asset return series with a KNOWN beta to BTC plus idio noise."""

    def _make(beta: float, noise_sd: float = 0.001) -> pd.Series:
        noise = pd.Series(
            rng.normal(0.0, noise_sd, size=len(btc_returns)), index=btc_returns.index
        )
        return beta * btc_returns + noise

    return _make


@pytest.fixture
def returns_frame(btc_returns: pd.Series, beta_returns) -> pd.DataFrame:
    """A 4-symbol return frame with distinct betas for covariance/HRP tests."""
    return pd.DataFrame(
        {
            "BTC/USDT:USDT": btc_returns,
            "ETH/USDT:USDT": beta_returns(1.2, 0.004),
            "SOL/USDT:USDT": beta_returns(1.5, 0.008),
            "XRP/USDT:USDT": beta_returns(0.8, 0.006),
        }
    )


@pytest.fixture
def geometries() -> list[CoinGeometry]:
    """Four coins with distinct betas, vols, funding, and sentiment."""
    return [
        CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0, beta_btc=1.0,
                     realized_vol=0.5, funding_apr=0.05, sentiment_score=0.4,
                     sentiment_conf=0.8, adv_usd=2e9),
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, beta_btc=1.2,
                     realized_vol=0.6, funding_apr=0.20, sentiment_score=-0.2,
                     sentiment_conf=0.5, adv_usd=1e9),
        CoinGeometry(symbol="SOL/USDT:USDT", mark=150.0, beta_btc=1.5,
                     realized_vol=0.9, funding_apr=0.30, sentiment_score=0.6,
                     sentiment_conf=0.9, adv_usd=4e8),
        CoinGeometry(symbol="XRP/USDT:USDT", mark=0.6, beta_btc=0.8,
                     realized_vol=0.7, funding_apr=-0.10, sentiment_score=-0.5,
                     sentiment_conf=0.7, adv_usd=3e8),
    ]


@pytest.fixture
def betas(geometries: list[CoinGeometry]) -> dict[str, float]:
    return {g.symbol: g.beta_btc for g in geometries}


@pytest.fixture
def sleeves() -> list[SleeveSignal]:
    """Two sleeves whose tilts net roughly dollar-balanced before projection.

    This is the canonical BALANCED 4-name book (SOL/XRP/BTC/ETH, betas 1.5/0.8/1.0/1.2)
    used by the optimizer property tests. It has >=3 active names on each side after the
    BTC hedge is added, so projection cannot collapse it to ~0 (see Task 11 n<=2 note)."""
    factor = SleeveSignal(
        sleeve="factor",
        tilts=[
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5, raw_score=1.0),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5, raw_score=-1.0),
        ],
        risk_budget_frac=0.5,
        as_of_ts=NOW,
    )
    carry = SleeveSignal(
        sleeve="carry",
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5, raw_score=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5, raw_score=-0.8),
        ],
        risk_budget_frac=0.5,
        as_of_ts=NOW,
    )
    return [factor, carry]
```

- [ ] **Step 2: Verify the fixtures import cleanly**

Run: `uv run python -c "import tests.conftest; print('ok')"`
Expected: prints `ok` and exits 0.

- [ ] **Step 3: Lint and commit**

Run: `uv run ruff check tests/conftest.py`
Expected: `All checks passed!`

```bash
git add tests/conftest.py
git commit -m "test: add shared synthetic price/beta/geometry/sleeve fixtures

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `beta.py` — `log_returns`

**Files:**
- Create: `futures_fund/beta.py`
- Test: `tests/test_beta.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from futures_fund.beta import log_returns


def test_log_returns_basic():
    prices = pd.Series([100.0, 110.0, 121.0])
    r = log_returns(prices)
    assert len(r) == 2
    assert np.isclose(r.iloc[0], np.log(110.0 / 100.0))
    assert np.isclose(r.iloc[1], np.log(121.0 / 110.0))


def test_log_returns_drops_nan_and_nonpositive():
    prices = pd.Series([100.0, np.nan, 121.0, 0.0, 130.0])
    r = log_returns(prices)
    # NaN and non-positive prices removed before differencing; no inf/NaN remains
    assert not r.isna().any()
    assert np.isfinite(r.to_numpy()).all()


def test_log_returns_empty_series_returns_empty():
    r = log_returns(pd.Series([], dtype=float))
    assert len(r) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_beta.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.beta'`.

- [ ] **Step 3: Create `futures_fund/beta.py` with `log_returns`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns of a mark-price series. Drops NaN/non-positive prices before
    differencing so no inf/NaN leaks into the return series."""
    clean = prices.dropna()
    clean = clean[clean > 0.0]
    if len(clean) < 2:
        return pd.Series([], dtype=float)
    return np.log(clean / clean.shift(1)).dropna()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_beta.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/beta.py tests/test_beta.py`
Expected: `All checks passed!`

```bash
git add futures_fund/beta.py tests/test_beta.py
git commit -m "feat: add beta.log_returns

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `beta.py` — `rolling_beta`

**Files:**
- Modify: `futures_fund/beta.py`
- Test: `tests/test_beta.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_beta.py`)**

```python
from futures_fund.beta import rolling_beta


def test_rolling_beta_recovers_known_beta(btc_returns, beta_returns):
    asset = beta_returns(1.3, noise_sd=0.0)  # noiseless => exact beta
    b = rolling_beta(asset, btc_returns, lookback=60)
    assert abs(b - 1.3) < 1e-6


def test_rolling_beta_uses_last_lookback_points(btc_returns, beta_returns):
    asset = beta_returns(2.0, noise_sd=0.0)
    b = rolling_beta(asset, btc_returns, lookback=30)
    assert abs(b - 2.0) < 1e-6


def test_rolling_beta_fallback_when_too_few_points(btc_returns, beta_returns):
    asset = beta_returns(1.5, noise_sd=0.0)
    b = rolling_beta(asset.iloc[:5], btc_returns.iloc[:5], lookback=45)
    assert b == 1.0


def test_rolling_beta_fallback_on_zero_variance():
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=50, freq="D", tz="UTC")
    flat_btc = pd.Series([0.0] * 50, index=idx)
    asset = pd.Series([0.01] * 50, index=idx)
    assert rolling_beta(asset, flat_btc, lookback=45) == 1.0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_beta.py -k rolling_beta -v`
Expected: FAIL with `ImportError: cannot import name 'rolling_beta'`.

- [ ] **Step 3: Add `rolling_beta` to `futures_fund/beta.py`**

```python
def rolling_beta(
    asset_returns: pd.Series, btc_returns: pd.Series, lookback: int = 45
) -> float:
    """OLS beta = cov(asset, btc) / var(btc) over the last `lookback` aligned points.
    Falls back to 1.0 if fewer than 10 aligned points or BTC variance is zero."""
    aligned = pd.concat([asset_returns, btc_returns], axis=1, join="inner").dropna()
    if len(aligned) > lookback:
        aligned = aligned.iloc[-lookback:]
    if len(aligned) < 10:
        return 1.0
    a = aligned.iloc[:, 0].to_numpy()
    b = aligned.iloc[:, 1].to_numpy()
    var_b = float(np.var(b))
    if var_b <= 0.0:
        return 1.0
    cov_ab = float(np.cov(a, b, ddof=0)[0, 1])
    return cov_ab / var_b
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_beta.py -k rolling_beta -v`
Expected: all 4 `rolling_beta` tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/beta.py tests/test_beta.py`
Expected: `All checks passed!`

```bash
git add futures_fund/beta.py tests/test_beta.py
git commit -m "feat: add beta.rolling_beta (OLS, lookback-windowed, 1.0 fallback)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `beta.py` — `beta_series` and `beta_for_symbols`

**Files:**
- Modify: `futures_fund/beta.py`
- Test: `tests/test_beta.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_beta.py`)**

```python
from futures_fund.beta import beta_for_symbols, beta_series


def test_beta_series_is_rolling_and_recovers_beta(btc_returns, beta_returns):
    asset = beta_returns(1.4, noise_sd=0.0)
    s = beta_series(asset, btc_returns, lookback=30)
    # First valid window appears once >= 10 aligned points exist
    valid = s.dropna()
    assert len(valid) > 0
    assert abs(valid.iloc[-1] - 1.4) < 1e-6


def test_beta_for_symbols_maps_btc_to_one(btc_returns, beta_returns):
    # Build price series from the noiseless return series for two symbols
    import numpy as np
    import pandas as pd

    def prices_from_returns(r):
        return pd.Series(100.0 * np.exp(r.cumsum()), index=r.index)

    btc_prices = prices_from_returns(btc_returns)
    eth_prices = prices_from_returns(beta_returns(1.2, noise_sd=0.0))
    marks = {"BTC/USDT:USDT": btc_prices, "ETH/USDT:USDT": eth_prices}
    out = beta_for_symbols(marks, btc_symbol="BTC/USDT:USDT", lookback=60)
    assert out["BTC/USDT:USDT"] == 1.0
    assert abs(out["ETH/USDT:USDT"] - 1.2) < 1e-3


def test_beta_for_symbols_missing_btc_returns_empty():
    import pandas as pd

    # No BTC series in the marks dict => cannot compute beta to an absent benchmark.
    marks = {"ETH/USDT:USDT": pd.Series([100.0, 101.0])}
    out = beta_for_symbols(marks, btc_symbol="BTC/USDT:USDT")
    assert out == {}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_beta.py -k "beta_series or beta_for_symbols" -v`
Expected: FAIL with `ImportError: cannot import name 'beta_series'`.

- [ ] **Step 3: Add `beta_series` and `beta_for_symbols` to `futures_fund/beta.py`**

```python
def beta_series(
    asset_returns: pd.Series, btc_returns: pd.Series, lookback: int = 45
) -> pd.Series:
    """Rolling beta time series for drift monitoring / reviewer re-derivation. NaN until
    at least 10 aligned points are available; thereafter the trailing-`lookback` beta."""
    aligned = pd.concat([asset_returns, btc_returns], axis=1, join="inner").dropna()
    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    out: list[float] = []
    for i in range(len(aligned)):
        lo = max(0, i + 1 - lookback)
        out.append(rolling_beta(a.iloc[lo : i + 1], b.iloc[lo : i + 1], lookback=lookback))
    series = pd.Series(out, index=aligned.index, dtype=float)
    # Windows with < 10 points produce the 1.0 fallback; mask them as NaN for monitoring.
    counts = pd.Series(range(1, len(aligned) + 1), index=aligned.index)
    return series.where(counts >= 10, other=float("nan"))


def beta_for_symbols(
    marks_by_symbol: dict[str, pd.Series],
    btc_symbol: str = "BTC/USDT:USDT",
    lookback: int = 45,
) -> dict[str, float]:
    """Per-symbol rolling beta to BTC. BTC maps to 1.0 by construction. Returns {} if the
    BTC series is missing (cannot compute beta to an absent benchmark)."""
    if btc_symbol not in marks_by_symbol:
        return {}
    btc_ret = log_returns(marks_by_symbol[btc_symbol])
    out: dict[str, float] = {}
    for symbol, prices in marks_by_symbol.items():
        if symbol == btc_symbol:
            out[symbol] = 1.0
            continue
        out[symbol] = rolling_beta(log_returns(prices), btc_ret, lookback=lookback)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_beta.py -v`
Expected: all `beta.py` tests PASS (including the earlier ones).

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/beta.py tests/test_beta.py`
Expected: `All checks passed!`

```bash
git add futures_fund/beta.py tests/test_beta.py
git commit -m "feat: add beta.beta_series and beta.beta_for_symbols

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `NeutralityConfig` + residual measures (`dollar_residual`, `beta_residual`)

**Files:**
- Create: `futures_fund/neutrality.py`
- Test: `tests/test_neutrality_residuals.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import numpy as np

from futures_fund.neutrality import NeutralityConfig, beta_residual, dollar_residual


def test_neutrality_config_defaults():
    cfg = NeutralityConfig()
    assert cfg.capital_usdt == 20000.0
    assert cfg.target_gross_usdt == 20000.0
    assert cfg.side_budget_usdt == 10000.0
    assert cfg.deployment_floor == 0.90
    assert cfg.dry_powder_frac == 0.10
    assert cfg.per_name_cap == 0.25
    assert cfg.cluster_cap == 0.40
    assert cfg.dollar_band == 0.03
    assert cfg.beta_band == 0.05
    assert cfg.drift_band == 0.20
    assert cfg.stress_band_mult == 0.5


def test_deployment_target_is_between_floor_and_dry_powder_band():
    # The enforced per-side deployment target must sit inside [floor, 1 - dry_powder].
    cfg = NeutralityConfig()
    assert cfg.deployment_floor <= cfg.deploy_target_frac <= 1.0 - cfg.dry_powder_frac


def test_dollar_residual_balanced_book_is_zero():
    notionals = {"A": 5000.0, "B": -5000.0}
    weights = {"A": 0.25, "B": -0.25}
    assert np.isclose(dollar_residual(weights, notionals), 0.0)


def test_dollar_residual_long_heavy():
    notionals = {"A": 6000.0, "B": -4000.0}
    weights = {"A": 0.3, "B": -0.2}
    # Sum(long$) - Sum(short$) = 6000 - 4000 = 2000
    assert np.isclose(dollar_residual(weights, notionals), 2000.0)


def test_beta_residual_is_weighted_beta_sum():
    weights = {"A": 0.5, "B": -0.5}
    betas = {"A": 1.0, "B": 1.0}
    assert np.isclose(beta_residual(weights, betas), 0.0)
    betas2 = {"A": 1.5, "B": 0.5}
    # 0.5*1.5 + (-0.5)*0.5 = 0.75 - 0.25 = 0.5
    assert np.isclose(beta_residual(weights, betas2), 0.5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_neutrality_residuals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'futures_fund.neutrality'`.

- [ ] **Step 3: Create `futures_fund/neutrality.py` with config + residuals**

The config gains a derived `deploy_target_frac` property (the midpoint of the `[floor, 1−dry_powder]`
band) that `optimize_book` scales each side up to — so the floor is **enforced**, not just reported,
and per-side deployment never exceeds `1 − dry_powder` (dry powder is held back; spec §4).

```python
from __future__ import annotations

from pydantic import BaseModel


class NeutralityConfig(BaseModel):
    capital_usdt: float = 20000.0
    target_gross_usdt: float = 20000.0
    side_budget_usdt: float = 10000.0
    deployment_floor: float = 0.90
    dry_powder_frac: float = 0.10
    per_name_cap: float = 0.25
    cluster_cap: float = 0.40
    dollar_band: float = 0.03
    beta_band: float = 0.05
    drift_band: float = 0.20
    turnover_penalty: float = 0.001
    corr_threshold: float = 0.7
    stress_band_mult: float = 0.5

    @property
    def deploy_target_frac(self) -> float:
        """Per-side deployment target the optimizer scales each side up to: the midpoint of
        the [deployment_floor, 1 - dry_powder_frac] band. With defaults: (0.90 + 0.90)/2 =
        0.90 — i.e. deploy at the floor while still holding the full dry-powder reserve.
        Always lands in [floor, 1 - dry_powder] so both spec-§4 constraints hold by
        construction."""
        lo = self.deployment_floor
        hi = 1.0 - self.dry_powder_frac
        return (lo + hi) / 2.0


def dollar_residual(weights: dict[str, float], notionals: dict[str, float]) -> float:
    """Sum(long$) - Sum(short$) in USDT, using signed per-symbol notionals."""
    longs = sum(n for n in notionals.values() if n > 0.0)
    shorts = sum(-n for n in notionals.values() if n < 0.0)
    return longs - shorts


def beta_residual(weights: dict[str, float], betas: dict[str, float]) -> float:
    """Sum_i w_i * beta_i (equity-normalized beta-dollar exposure)."""
    return sum(w * betas.get(sym, 1.0) for sym, w in weights.items())
```

> **Default-band note (binding):** with the spec defaults `deployment_floor=0.90` and
> `dry_powder_frac=0.10`, the band `[0.90, 0.90]` is a single point, so `deploy_target_frac=0.90`.
> That is exactly the spec §4 intent: deploy ≥90%/side AND hold ~$1k/side ($10k × 0.10) dry powder.
> `optimize_book` scales each side to this target, so `deploy_*_frac` lands at 0.90 (≥ floor ✓,
> ≤ 1 − dry_powder ✓) on a feasible book.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_neutrality_residuals.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/neutrality.py tests/test_neutrality_residuals.py`
Expected: `All checks passed!`

```bash
git add futures_fund/neutrality.py tests/test_neutrality_residuals.py
git commit -m "feat: add NeutralityConfig (+deploy_target band) + dollar/beta residual measures

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `ledoit_wolf_cov` and `hrp_weights`

**Files:**
- Modify: `futures_fund/neutrality.py`
- Test: `tests/test_neutrality_weighting.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import numpy as np

from futures_fund.neutrality import hrp_weights, ledoit_wolf_cov


def test_ledoit_wolf_cov_is_symmetric_psd(returns_frame):
    cov = ledoit_wolf_cov(returns_frame)
    n = returns_frame.shape[1]
    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T)
    # PSD: all eigenvalues non-negative (shrinkage guarantees this)
    eigs = np.linalg.eigvalsh(cov)
    assert (eigs >= -1e-12).all()


def test_hrp_weights_sum_to_one_and_positive(returns_frame):
    cov = ledoit_wolf_cov(returns_frame)
    labels = list(returns_frame.columns)
    w = hrp_weights(cov, labels)
    assert set(w.keys()) == set(labels)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > 0.0 for v in w.values())


def test_hrp_weights_low_vol_gets_more_weight(returns_frame):
    # XRP has lower idio noise (0.006) than SOL (0.008) but the dominant driver is BTC beta.
    cov = ledoit_wolf_cov(returns_frame)
    labels = list(returns_frame.columns)
    w = hrp_weights(cov, labels)
    # Highest-variance asset (SOL, beta 1.5 + most noise) must not dominate the book.
    assert w["SOL/USDT:USDT"] < 0.5


def test_hrp_weights_single_asset():
    cov = np.array([[0.04]])
    w = hrp_weights(cov, ["BTC/USDT:USDT"])
    assert w == {"BTC/USDT:USDT": 1.0}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_neutrality_weighting.py -v`
Expected: FAIL with `ImportError: cannot import name 'ledoit_wolf_cov'`.

- [ ] **Step 3: Add `ledoit_wolf_cov` and `hrp_weights` to `futures_fund/neutrality.py`**

Add imports at the top of the file (below the existing `from pydantic import BaseModel`):

```python
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
```

Then add the functions:

```python
def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance — stable, avoids unstable inversion. Drops rows with
    any NaN so the estimator sees a complete block."""
    clean = returns.dropna()
    if clean.shape[0] < 2 or clean.shape[1] == 0:
        n = returns.shape[1]
        return np.eye(n)
    return LedoitWolf().fit(clean.to_numpy()).covariance_


def _ivp(cov: np.ndarray, idx: list[int]) -> np.ndarray:
    """Inverse-variance portfolio weights for a sub-cluster (no matrix inversion)."""
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    return ivp / ivp.sum()


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    w = _ivp(cov, idx)
    sub = cov[np.ix_(idx, idx)]
    return float(w @ sub @ w)


def _quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    n = link[-1, 3]
    order = [link[-1, 0], link[-1, 1]]
    while max(order) >= n:
        new: list[int] = []
        for item in order:
            if item < n:
                new.append(item)
            else:
                row = link[item - n]
                new.append(row[0])
                new.append(row[1])
        order = new
    return order


def hrp_weights(cov: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Hierarchical Risk Parity: cluster -> quasi-diagonalize -> recursive bisection.
    No matrix inversion (only diagonal inverse-variance). Weights sum to 1.0."""
    n = len(labels)
    if n == 1:
        return {labels[0]: 1.0}
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    outer[outer == 0.0] = 1e-12
    corr = np.clip(cov / outer, -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = _quasi_diag(link)
    weights = np.ones(n)
    clusters = [sort_ix]
    while clusters:
        clusters = [
            c[j:k]
            for c in clusters
            for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            left = clusters[i]
            right = clusters[i + 1]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= 1.0 - alpha
    weights /= weights.sum()
    return {labels[i]: float(weights[i]) for i in range(n)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_neutrality_weighting.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/neutrality.py tests/test_neutrality_weighting.py`
Expected: `All checks passed!`

```bash
git add futures_fund/neutrality.py tests/test_neutrality_weighting.py
git commit -m "feat: add Ledoit-Wolf shrunk covariance + HRP weighting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `risk_parity_budgets`, `merge_sleeves`, and `apply_hrp_weights`

**Files:**
- Modify: `futures_fund/neutrality.py`
- Test: `tests/test_neutrality_weighting.py`

This task adds the sleeve budgeting/merge primitives AND `apply_hrp_weights`, the function that
makes the Task-8 HRP weights actually shape the merged book (so `ledoit_wolf_cov`→`hrp_weights` are
wired, not dead — spec §8). `apply_hrp_weights` redistributes magnitude **within each side**
(long pool, short pool) by the HRP weights, preserving every leg's sign and the per-side gross, so
it cannot break dollar-neutrality — it only changes the *relative* per-name notionals.

- [ ] **Step 1: Write the failing test (append to `tests/test_neutrality_weighting.py`)**

```python
from futures_fund.neutrality import apply_hrp_weights, merge_sleeves, risk_parity_budgets


def test_risk_parity_budgets_equal_when_no_cov(sleeves):
    budgets = risk_parity_budgets(sleeves)
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    # two sleeves, inverse-vol fallback with no cov => equal split
    assert abs(budgets["factor"] - 0.5) < 1e-9
    assert abs(budgets["carry"] - 0.5) < 1e-9


def test_risk_parity_budgets_writes_back_onto_signals(sleeves):
    budgets = risk_parity_budgets(sleeves)
    for s in sleeves:
        assert abs(s.risk_budget_frac - budgets[s.sleeve]) < 1e-9


def test_merge_sleeves_scales_tilts_by_budget(sleeves, geometries):
    risk_parity_budgets(sleeves)  # assigns 0.5 / 0.5
    merged = merge_sleeves(sleeves, geometries)
    # factor: SOL +0.5*0.5=+0.25 ; XRP -0.5*0.5=-0.25
    assert abs(merged["SOL/USDT:USDT"] - 0.25) < 1e-9
    assert abs(merged["XRP/USDT:USDT"] - (-0.25)) < 1e-9
    # carry: BTC +0.25 ; ETH -0.25
    assert abs(merged["BTC/USDT:USDT"] - 0.25) < 1e-9
    assert abs(merged["ETH/USDT:USDT"] - (-0.25)) < 1e-9


def test_merge_sleeves_sums_same_symbol_across_sleeves(geometries):
    from datetime import UTC, datetime

    from futures_fund.contracts import SleeveSignal, SleeveTilt

    now = datetime(2026, 6, 11, tzinfo=UTC)
    a = SleeveSignal(sleeve="factor", risk_budget_frac=0.5, as_of_ts=now,
                     tilts=[SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.4)])
    b = SleeveSignal(sleeve="carry", risk_budget_frac=0.5, as_of_ts=now,
                     tilts=[SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.6)])
    merged = merge_sleeves([a, b], geometries)
    # 0.4*0.5 + 0.6*0.5 = 0.5
    assert abs(merged["BTC/USDT:USDT"] - 0.5) < 1e-9


def test_apply_hrp_weights_preserves_sign_and_side_gross():
    # Two longs (A,B) + two shorts (C,D). HRP says A>>B on the long side.
    weights = {"A": 0.3, "B": 0.3, "C": -0.3, "D": -0.3}
    hrp = {"A": 0.4, "B": 0.1, "C": 0.25, "D": 0.25}  # within-side normalization happens inside
    out = apply_hrp_weights(weights, hrp)
    # signs preserved
    assert out["A"] > 0 and out["B"] > 0 and out["C"] < 0 and out["D"] < 0
    # per-side gross preserved (long pool stays 0.6, short pool stays 0.6)
    assert abs((out["A"] + out["B"]) - 0.6) < 1e-9
    assert abs((-out["C"] - out["D"]) - 0.6) < 1e-9
    # HRP actually reshapes: A gets 0.4/0.5 of the long pool, B gets 0.1/0.5
    assert abs(out["A"] - 0.6 * (0.4 / 0.5)) < 1e-9
    assert abs(out["B"] - 0.6 * (0.1 / 0.5)) < 1e-9


def test_apply_hrp_weights_noop_when_hrp_empty():
    weights = {"A": 0.3, "B": -0.3}
    assert apply_hrp_weights(weights, {}) == weights
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_neutrality_weighting.py -k "risk_parity or merge_sleeves or apply_hrp" -v`
Expected: FAIL with `ImportError: cannot import name 'risk_parity_budgets'`.

- [ ] **Step 3: Add `risk_parity_budgets`, `merge_sleeves`, and `apply_hrp_weights` to `futures_fund/neutrality.py`**

Add this import near the top (with the other `from ...` imports):

```python
from futures_fund.contracts import CoinGeometry, SleeveSignal
from futures_fund.models import SleeveName
```

Then add the functions:

```python
def risk_parity_budgets(
    sleeves: list[SleeveSignal], *, cov: np.ndarray | None = None
) -> dict[SleeveName, float]:
    """Risk-parity (or inverse-vol) budget across the sleeves; writes the result back onto
    each SleeveSignal.risk_budget_frac and returns the {sleeve: frac} map. Sums to 1.0.
    With no covariance supplied, falls back to an equal (inverse-unit-vol) split."""
    if not sleeves:
        return {}
    if cov is None or cov.shape[0] != len(sleeves):
        raw = np.ones(len(sleeves))
    else:
        vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        raw = 1.0 / vol
    fracs = raw / raw.sum()
    out: dict[SleeveName, float] = {}
    for s, f in zip(sleeves, fracs, strict=True):
        s.risk_budget_frac = float(f)
        out[s.sleeve] = float(f)
    return out


def merge_sleeves(
    sleeves: list[SleeveSignal], geometries: list[CoinGeometry]
) -> dict[str, float]:
    """Combine already-risk-budgeted sleeve tilts into one signed pre-projection weight
    vector. Each tilt's signed target_weight is scaled by its sleeve risk_budget_frac and
    summed per symbol."""
    known = {g.symbol for g in geometries}
    merged: dict[str, float] = {}
    for s in sleeves:
        for tilt in s.tilts:
            if tilt.symbol not in known:
                continue
            merged[tilt.symbol] = merged.get(tilt.symbol, 0.0) + (
                tilt.target_weight * s.risk_budget_frac
            )
    return merged


def apply_hrp_weights(
    weights: dict[str, float], hrp: dict[str, float]
) -> dict[str, float]:
    """Reshape a signed weight vector so each side's per-name split follows the HRP weights,
    WITHOUT changing any sign or either side's total gross. This is how Ledoit-Wolf -> HRP
    (Task 8) actually shapes the book (spec §8): for each side, redistribute that side's gross
    across its names in proportion to the names' HRP weights (re-normalized within the side).
    Returns `weights` unchanged if `hrp` is empty (HRP unavailable / single name)."""
    if not hrp:
        return dict(weights)
    longs = {s: w for s, w in weights.items() if w > 0.0}
    shorts = {s: w for s, w in weights.items() if w < 0.0}
    out: dict[str, float] = {s: w for s, w in weights.items() if w == 0.0}
    for side in (longs, shorts):
        if not side:
            continue
        side_gross = sum(abs(w) for w in side.values())
        sign = 1.0 if next(iter(side.values())) > 0.0 else -1.0
        hrp_side = {s: hrp.get(s, 0.0) for s in side}
        hrp_sum = sum(hrp_side.values())
        if hrp_sum <= 0.0:
            # HRP has no info for this side's names: keep the original split.
            out.update(side)
            continue
        for s in side:
            out[s] = sign * side_gross * (hrp_side[s] / hrp_sum)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_neutrality_weighting.py -v`
Expected: all weighting tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/neutrality.py tests/test_neutrality_weighting.py`
Expected: `All checks passed!`

```bash
git add futures_fund/neutrality.py tests/test_neutrality_weighting.py
git commit -m "feat: add risk_parity_budgets + merge_sleeves + apply_hrp_weights (HRP shapes the book)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Per-name and per-cluster caps (reuse cluster union-find)

**Files:**
- Modify: `futures_fund/neutrality.py`
- Test: `tests/test_neutrality_caps.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from futures_fund.neutrality import apply_cluster_cap, apply_per_name_cap


def test_per_name_cap_clamps_magnitude_preserving_sign():
    weights = {"A": 0.6, "B": -0.5, "C": 0.1}
    capped = apply_per_name_cap(weights, per_name_cap=0.25)
    assert capped["A"] == 0.25
    assert capped["B"] == -0.25
    assert capped["C"] == 0.1


def test_per_name_cap_noop_when_within_cap():
    weights = {"A": 0.2, "B": -0.1}
    capped = apply_per_name_cap(weights, per_name_cap=0.25)
    assert capped == weights


def test_cluster_cap_scales_correlated_same_side_group():
    # A and B are correlated >= 0.7 and same side (both long) => clustered together.
    weights = {"A": 0.3, "B": 0.3, "C": -0.4}
    corr = {("A", "B"): 0.9}
    capped = apply_cluster_cap(weights, corr=corr, cluster_cap=0.40, threshold=0.7)
    # cluster {A,B} combined long weight 0.6 > 0.40 => scale by 0.40/0.60
    assert abs(capped["A"] - 0.2) < 1e-9
    assert abs(capped["B"] - 0.2) < 1e-9
    # C is in its own cluster, magnitude 0.4 <= cap => unchanged
    assert abs(capped["C"] - (-0.4)) < 1e-9


def test_cluster_cap_does_not_cluster_opposite_sides():
    # A long, B short, even if correlated => not clustered (natural hedge).
    weights = {"A": 0.3, "B": -0.3}
    corr = {("A", "B"): 0.95}
    capped = apply_cluster_cap(weights, corr=corr, cluster_cap=0.40, threshold=0.7)
    assert capped == weights
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_neutrality_caps.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_per_name_cap'`.

- [ ] **Step 3: Add the cap functions to `futures_fund/neutrality.py`**

Add this import near the other imports:

```python
from collections.abc import Mapping
```

Then add the functions (the cluster union-find is adapted from
`crypto-trade-claude-code-weekly/futures_fund/portfolio_risk.py::cluster_heat`, same-direction-only):

```python
def apply_per_name_cap(
    weights: dict[str, float], *, per_name_cap: float
) -> dict[str, float]:
    """Clamp each symbol's weight magnitude to `per_name_cap`, preserving sign."""
    out: dict[str, float] = {}
    for sym, w in weights.items():
        if abs(w) > per_name_cap:
            out[sym] = per_name_cap if w > 0 else -per_name_cap
        else:
            out[sym] = w
    return out


def _corr_lookup(corr: Mapping[tuple[str, str], float], a: str, b: str) -> float:
    if (a, b) in corr:
        return corr[(a, b)]
    if (b, a) in corr:
        return corr[(b, a)]
    return 0.0


def apply_cluster_cap(
    weights: dict[str, float],
    *,
    corr: Mapping[tuple[str, str], float],
    cluster_cap: float,
    threshold: float = 0.7,
) -> dict[str, float]:
    """'Correlated-as-one' heat cap. Union-find groups SAME-SIDE symbols whose pairwise
    correlation >= threshold (a long and short in correlated names are a natural hedge and
    are NOT clustered). Scales down each cluster so its combined |weight| <= cluster_cap.
    Adapted from crypto-trade-claude-code-weekly portfolio_risk.cluster_heat."""
    syms = list(weights.keys())
    n = len(syms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    def side(w: float) -> int:
        return 1 if w > 0 else (-1 if w < 0 else 0)

    for i in range(n):
        for j in range(i + 1, n):
            if side(weights[syms[i]]) != 0 and side(weights[syms[i]]) == side(weights[syms[j]]):
                if _corr_lookup(corr, syms[i], syms[j]) >= threshold:
                    union(i, j)

    cluster_mag: dict[int, float] = {}
    for idx, sym in enumerate(syms):
        root = find(idx)
        cluster_mag[root] = cluster_mag.get(root, 0.0) + abs(weights[sym])

    out: dict[str, float] = {}
    for idx, sym in enumerate(syms):
        root = find(idx)
        mag = cluster_mag[root]
        if mag > cluster_cap and mag > 0.0:
            out[sym] = weights[sym] * (cluster_cap / mag)
        else:
            out[sym] = weights[sym]
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_neutrality_caps.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/neutrality.py tests/test_neutrality_caps.py`
Expected: `All checks passed!`

```bash
git add futures_fund/neutrality.py tests/test_neutrality_caps.py
git commit -m "feat: add per-name + per-cluster caps (reuse same-side union-find)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: `project_neutral` (dollar + beta re-projection)

**Files:**
- Modify: `futures_fund/neutrality.py`
- Test: `tests/test_neutrality_project.py`

> **Degenerate n ≤ 2 case (binding — issue fix):** with two independent constraints
> (dollar + beta), a least-norm projection of an `n`-vector lives in the
> `n − 2`-dimensional null space. For `n ≤ 2` that null space is `{0}`, so **any** 2-name
> book collapses to ~0 (e.g. `{SOL:0.5, XRP:-0.5}`, betas 1.5/0.8 → both ~1e-16). A
> non-trivial dollar+beta-neutral book therefore **requires ≥ 3 distinct active names**.
> `optimize_book` guarantees this in practice because the BTC hedge leg is appended as a
> third column before projection (so even a single alpha pair has ≥3 names), and a 2-name
> input is flagged `feasible=False` rather than silently zeroed. `project_neutral` itself
> stays a pure least-norm projector; the ≥3-name requirement is documented here and asserted
> by `test_project_neutral_three_names_retains_gross` below.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from futures_fund.neutrality import beta_residual, dollar_residual, project_neutral


def test_project_neutral_drives_dollar_residual_into_band():
    # 3 names so the neutral null space is non-trivial (n - 2 = 1 dimension).
    weights = {"A": 0.4, "B": -0.2, "C": 0.1}
    betas = {"A": 1.0, "B": 1.0, "C": 1.0}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    # equity-normalized: dollar residual = sum of signed weights
    assert abs(sum(out.values())) <= 0.03 + 1e-9


def test_project_neutral_drives_beta_residual_into_band():
    weights = {"A": 0.3, "B": -0.3, "C": 0.1}
    betas = {"A": 1.5, "B": 0.5, "C": 1.0}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    assert abs(beta_residual(out, betas)) <= 0.05 + 1e-9


def test_project_neutral_already_neutral_is_near_identity():
    weights = {"A": 0.25, "B": -0.25, "C": 0.0}
    betas = {"A": 1.0, "B": 1.0, "C": 1.0}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    assert abs(out["A"] - 0.25) < 1e-6
    assert abs(out["B"] - (-0.25)) < 1e-6


def test_project_neutral_three_names_retains_nontrivial_gross():
    # A >=3-name book must NOT collapse to ~0 after projection (it lives in the 1-dim null
    # space). This is the guard against the n<=2 degenerate collapse.
    weights = {"A": 0.5, "B": -0.3, "C": 0.2}
    betas = {"A": 1.2, "B": 0.9, "C": 1.5}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    gross = sum(abs(v) for v in out.values())
    assert gross > 0.2  # non-trivial residual book survives projection


def test_project_neutral_two_names_collapse_is_documented():
    # With exactly 2 names and 2 independent constraints the unique neutral point is ~0.
    # We assert the collapse so the optimizer's "append hedge => >=3 names" guard is justified.
    weights = {"A": 0.5, "B": -0.5}
    betas = {"A": 1.5, "B": 0.8}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    assert sum(abs(v) for v in out.values()) < 1e-6
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_neutrality_project.py -v`
Expected: FAIL with `ImportError: cannot import name 'project_neutral'`.

- [ ] **Step 3: Add `project_neutral` to `futures_fund/neutrality.py`**

```python
def project_neutral(
    weights: dict[str, float],
    betas: dict[str, float],
    *,
    dollar_band: float,
    beta_band: float,
) -> dict[str, float]:
    """Least-norm projection of a signed weight vector onto the dollar+beta-neutral
    constraint set: removes the components of the vector in the span of the dollar direction
    (all-ones) and the beta direction so Sum(w_i) ~ 0 and Sum(w_i*beta_i) ~ 0. The result
    lives in the (n - 2)-dimensional null space of the two constraints, so a NON-TRIVIAL
    neutral book requires >= 3 distinct active names (with n <= 2 the only neutral point is
    0 — see the Task 11 degenerate-case note). Sentiment tilts are applied BEFORE this call,
    so sentiment cannot break neutrality (residuals are recomputed after). `dollar_band` /
    `beta_band` are accepted for signature stability with the reviewer's re-derivation and to
    document the bands this projection targets; the exact projection drives residuals to ~0,
    well inside the bands."""
    syms = list(weights.keys())
    if not syms:
        return {}
    w = np.array([weights[s] for s in syms], dtype=float)
    b = np.array([betas.get(s, 1.0) for s in syms], dtype=float)
    ones = np.ones(len(syms))

    # Constraint matrix C (2 x n): row0 = dollar (ones), row1 = beta.
    c = np.vstack([ones, b])
    residual = c @ w  # [dollar_resid, beta_resid]
    gram = c @ c.T  # 2 x 2
    try:
        correction = c.T @ np.linalg.solve(gram, residual)
    except np.linalg.LinAlgError:
        correction = c.T @ (np.linalg.pinv(gram) @ residual)
    w_proj = w - correction
    return {syms[i]: float(w_proj[i]) for i in range(len(syms))}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_neutrality_project.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/neutrality.py tests/test_neutrality_project.py`
Expected: `All checks passed!`

```bash
git add futures_fund/neutrality.py tests/test_neutrality_project.py
git commit -m "feat: add project_neutral (least-norm dollar+beta re-projection; n>2 non-trivial)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: `size_btc_hedge` (budget-internal BTC hedge leg)

**Files:**
- Modify: `futures_fund/neutrality.py`
- Test: `tests/test_neutrality_project.py`

`size_btc_hedge` is the standalone hedge-sizing primitive: given the **alpha legs' residual
beta** it returns the signed BTC-perp notional that absorbs it. In `optimize_book` (Task 14) it is
called on the alpha legs BEFORE `project_neutral`, so the hedge is a real degree of freedom that
carries the beta — the BTC column then enters the projection together with the alpha legs. The
every-cycle reviewer re-derives the hedge by calling exactly this function on the alpha legs'
residual beta (roadmap §5.2 `check_btc_hedge`), so its contract must stay: hedge = −(residual beta)
in equity terms, clamped to one per-side budget.

- [ ] **Step 1: Write the failing test (append to `tests/test_neutrality_project.py`)**

```python
from futures_fund.neutrality import size_btc_hedge


def test_btc_hedge_absorbs_residual_beta_with_opposite_sign():
    # Net long beta => hedge must be short BTC (negative notional).
    weights = {"ALT/USDT:USDT": 0.3}
    betas = {"ALT/USDT:USDT": 1.5}  # beta residual = 0.45 (positive)
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    assert hedge < 0.0


def test_btc_hedge_zero_when_already_beta_neutral():
    weights = {"A": 0.3, "B": -0.3}
    betas = {"A": 1.0, "B": 1.0}  # residual 0
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    assert abs(hedge) < 1e-6


def test_btc_hedge_capped_inside_side_budget():
    # Huge residual beta must not size the hedge beyond the per-side budget.
    weights = {"A": 0.9}
    betas = {"A": 3.0}
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    assert abs(hedge) <= 10000.0 + 1e-6


def test_btc_hedge_short_beta_gives_long_hedge():
    weights = {"A": -0.3}
    betas = {"A": 1.5}  # residual -0.45 (net short beta)
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    assert hedge > 0.0


def test_btc_hedge_excludes_existing_btc_leg_from_residual():
    # If BTC is already a leg, its own beta is part of the residual the hedge should absorb,
    # but the hedge must size off the residual computed WITHOUT double-counting a prior hedge.
    # Here the alpha residual is +0.45 (ALT) and BTC alpha leg adds +0.1*1.0 => residual 0.55.
    weights = {"ALT/USDT:USDT": 0.3, "BTC/USDT:USDT": 0.1}
    betas = {"ALT/USDT:USDT": 1.5, "BTC/USDT:USDT": 1.0}
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    # residual beta = 0.3*1.5 + 0.1*1.0 = 0.55 ; hedge = -0.55*20000 = -11000 -> clamp -10000
    assert hedge < 0.0
    assert abs(hedge) <= 10000.0 + 1e-6
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_neutrality_project.py -k btc_hedge -v`
Expected: FAIL with `ImportError: cannot import name 'size_btc_hedge'`.

- [ ] **Step 3: Add `size_btc_hedge` to `futures_fund/neutrality.py`**

```python
def size_btc_hedge(
    weights: dict[str, float],
    betas: dict[str, float],
    *,
    equity: float,
    side_budget: float,
) -> float:
    """Signed BTC-perp hedge notional that absorbs the ALPHA legs' residual portfolio beta.
    BTC has beta 1.0, so the hedge weight equals the NEGATIVE of the residual beta of the
    legs passed in; converted to USDT via equity and clamped to fit INSIDE one per-side
    budget (never added on top). Call this on the alpha legs BEFORE project_neutral so the
    hedge is a real degree of freedom (the reviewer re-derives it the same way)."""
    resid_beta = beta_residual(weights, betas)
    hedge_weight = -resid_beta  # BTC beta == 1.0
    hedge_notional = hedge_weight * equity
    if hedge_notional > side_budget:
        hedge_notional = side_budget
    elif hedge_notional < -side_budget:
        hedge_notional = -side_budget
    return hedge_notional
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_neutrality_project.py -k btc_hedge -v`
Expected: all 5 `btc_hedge` tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/neutrality.py tests/test_neutrality_project.py`
Expected: `All checks passed!`

```bash
git add futures_fund/neutrality.py tests/test_neutrality_project.py
git commit -m "feat: add size_btc_hedge (residual-beta hedge inside per-side budget)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Conviction tilt + sentiment ordering invariant

**Files:**
- Modify: `futures_fund/neutrality.py`
- Test: `tests/test_neutrality_project.py`

This task adds the sentiment conviction-tilt primitive used by `optimize_book` (the standalone
sentiment SLEEVE and the agent plumbing land in Phase 2; the optimizer only needs the tilt math).

- [ ] **Step 1: Write the failing test (append to `tests/test_neutrality_project.py`)**

```python
from futures_fund.neutrality import apply_conviction_tilts, conviction_tilt


def test_conviction_tilt_positive_sentiment_grows_long():
    w = conviction_tilt(0.2, sentiment_score=0.8, sentiment_conf=1.0, kappa=0.5, cap=0.25)
    # 0.2*(1 + 0.5*0.8*1.0) = 0.2*1.4 = 0.28, but |delta| <= 0.25*|0.2| => clamp to 0.25
    assert w > 0.2
    assert abs(w - 0.2) <= 0.25 * 0.2 + 1e-9


def test_conviction_tilt_never_flips_sign():
    # Extreme negative sentiment on a long can only shrink it, never flip to short.
    w = conviction_tilt(0.2, sentiment_score=-1.0, sentiment_conf=1.0, kappa=5.0, cap=0.25)
    assert w >= 0.0


def test_conviction_tilt_zero_weight_stays_zero():
    # Sentiment never OPENS a position alone.
    assert conviction_tilt(0.0, sentiment_score=1.0, sentiment_conf=1.0) == 0.0


def test_conviction_tilt_respects_cap_magnitude():
    base = 0.4
    w = conviction_tilt(base, sentiment_score=1.0, sentiment_conf=1.0, kappa=10.0, cap=0.25)
    assert abs(w - base) <= 0.25 * abs(base) + 1e-9


def test_apply_conviction_tilts_maps_over_legs(geometries):
    from futures_fund.contracts import SleeveTilt

    legs = [
        SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.3),
        SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.3),
    ]
    out = apply_conviction_tilts(legs, geometries, kappa=0.5, cap=0.25)
    # SOL sentiment +0.6 conf 0.9 => long grows
    sol = next(t for t in out if t.symbol == "SOL/USDT:USDT")
    assert sol.target_weight > 0.3
    # XRP sentiment -0.5 conf 0.7 on a SHORT => negative*sentiment makes short stronger
    xrp = next(t for t in out if t.symbol == "XRP/USDT:USDT")
    assert xrp.target_weight <= -0.3
    # signs preserved
    assert sol.target_weight > 0 and xrp.target_weight < 0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_neutrality_project.py -k "conviction" -v`
Expected: FAIL with `ImportError: cannot import name 'conviction_tilt'`.

- [ ] **Step 3: Add the conviction-tilt functions to `futures_fund/neutrality.py`**

Add this import near the other `from futures_fund...` imports (extend the existing line):

```python
from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt
```

Then add the functions:

```python
def conviction_tilt(
    weight: float,
    sentiment_score: float,
    sentiment_conf: float,
    *,
    kappa: float = 0.5,
    cap: float = 0.25,
) -> float:
    """Deterministic sentiment tilt: w*(1 + kappa*s*conf), with |delta w| clamped to
    cap*|w|. NEVER flips sign, never opens a position alone (returns 0 if weight is 0).
    Applied BEFORE the optimizer re-projection (sentiment cannot break neutrality)."""
    if weight == 0.0:
        return 0.0
    raw = weight * (1.0 + kappa * sentiment_score * sentiment_conf)
    delta = raw - weight
    max_delta = cap * abs(weight)
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    tilted = weight + delta
    # never flip sign
    if (weight > 0 and tilted < 0) or (weight < 0 and tilted > 0):
        return 0.0
    return tilted


def apply_conviction_tilts(
    legs: list[SleeveTilt],
    geometries: list[CoinGeometry],
    *,
    kappa: float = 0.5,
    cap: float = 0.25,
) -> list[SleeveTilt]:
    """Map conviction_tilt over legs using each symbol's geometry; sign-preserving and
    cap-respecting. Symbols without geometry are returned unchanged."""
    geo = {g.symbol: g for g in geometries}
    out: list[SleeveTilt] = []
    for leg in legs:
        g = geo.get(leg.symbol)
        if g is None:
            out.append(leg)
            continue
        tilted = conviction_tilt(
            leg.target_weight, g.sentiment_score, g.sentiment_conf, kappa=kappa, cap=cap
        )
        out.append(leg.model_copy(update={"target_weight": tilted}))
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_neutrality_project.py -k "conviction" -v`
Expected: all 5 `conviction` tests PASS.

- [ ] **Step 5: Run the full neutrality-project file**

Run: `uv run pytest tests/test_neutrality_project.py -v`
Expected: all tests in the file PASS (project_neutral + btc_hedge + conviction).

- [ ] **Step 6: Commit**

Run: `uv run ruff check futures_fund/neutrality.py tests/test_neutrality_project.py`
Expected: `All checks passed!`

```bash
git add futures_fund/neutrality.py tests/test_neutrality_project.py
git commit -m "feat: add conviction_tilt + apply_conviction_tilts (sign-preserving, capped)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: `optimize_book` — assemble the solver

**Files:**
- Modify: `futures_fund/neutrality.py`
- Test: `tests/test_neutrality_optimize.py`

This task wires the previously-built primitives into the end-to-end solver. The ordering is the
load-bearing fix set for this plan; read it before coding:

1. **risk budgets + merge** sleeves into one signed vector.
2. **sentiment conviction tilts** (sign-preserving, capped) — BEFORE any projection.
3. **HRP shaping** — `ledoit_wolf_cov(returns)` → `hrp_weights` → `apply_hrp_weights` reshapes the
   per-name split within each side (only when a `returns`/`cov` frame is available; otherwise the
   merged split is used). This is how Ledoit-Wolf→HRP actually shapes the book (spec §8).
4. **per-name + cluster caps**.
5. **turnover / no-trade band vs the PRIOR book — applied BEFORE projection** so projection has the
   final say on neutrality; symbols absent from the prior are treated as **always-trade** (a fresh
   sub-drift-band leg is NOT snapped to 0).
6. **size the BTC hedge on the ALPHA legs' residual beta** (real DOF) and append it as a third
   column, then **project the alpha+hedge vector** onto the dollar+beta-neutral set. With the hedge
   appended there are ≥3 names, so projection yields a non-trivial neutral book.
7. **scale the neutral book to the per-side deployment target** with a single positive scalar.
   Scaling a dollar+beta-neutral vector by `k>0` preserves BOTH neutralities exactly
   (Σkw=0, Σkwβ=0) and scales each side's gross equally, so this restores gross/floor WITHOUT
   re-breaking neutrality. Target = `cfg.deploy_target_frac * side_budget` per side ⇒ deployment
   lands in `[floor, 1 − dry_powder]`.
8. **assemble `TargetWeights`** with residuals + per-side deployment; `feasible=False` (never
   silently un-neutral / under-deployed) if the bands or the floor cannot be met.

- [ ] **Step 1: Write the failing structural test**

```python
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import TargetWeights
from futures_fund.neutrality import NeutralityConfig, optimize_book

NOW = datetime(2026, 6, 11, tzinfo=UTC)


def test_optimize_book_returns_target_weights(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(
        sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg
    )
    assert isinstance(tw, TargetWeights)
    assert tw.feasible is True
    assert tw.as_of_ts is not None


def test_optimize_book_sets_per_side_deployment_and_gross(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    assert tw.gross_long > 0.0
    assert tw.gross_short > 0.0
    assert tw.gross_notional == tw.gross_long + tw.gross_short


def test_optimize_book_each_leg_has_target_notional(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    assert len(tw.legs) > 0
    for leg in tw.legs:
        assert leg.target_notional != 0.0
        assert leg.beta_btc != 0.0


def test_optimize_book_includes_hedge_leg_when_residual_beta(geometries):
    from futures_fund.contracts import SleeveSignal, SleeveTilt

    # A beta-imbalanced book: long the HIGH-beta name (SOL 1.5), short the LOW-beta name
    # (XRP 0.8). The alpha legs carry a NET LONG beta, so the BTC hedge MUST be a non-zero
    # SHORT BTC leg that absorbs it (the hedge is a real DOF, sized before projection).
    s = SleeveSignal(
        sleeve="factor",
        risk_budget_frac=1.0,
        as_of_ts=NOW,
        tilts=[
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )
    cfg = NeutralityConfig()
    tw = optimize_book([s], geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    hedge_legs = [leg for leg in tw.legs if leg.sleeve == "hedge"]
    # NON-vacuous: this beta-imbalanced book REQUIRES a materialized BTC hedge leg.
    assert tw.btc_hedge_notional < 0.0  # net long beta => short BTC hedge
    assert hedge_legs
    assert hedge_legs[0].symbol == "BTC/USDT:USDT"
    assert hedge_legs[0].direction == "short"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_neutrality_optimize.py -v`
Expected: FAIL with `ImportError: cannot import name 'optimize_book'`.

- [ ] **Step 3: Add `optimize_book` and its helpers to `futures_fund/neutrality.py`**

First, **set the final import block at the top of `futures_fund/neutrality.py` to read EXACTLY as
below** (this supersedes the partial `from futures_fund.contracts import ...` /
`from futures_fund.models import ...` lines added in Tasks 8/9/12/13 — replace them with these two
consolidated lines so there is exactly one import of each module, with every symbol the file now
uses present and none missing):

```python
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from futures_fund.contracts import (
    CoinGeometry,
    SleeveSignal,
    SleeveTilt,
    TargetWeights,
    WeightLeg,
)
from futures_fund.models import RegimeState, SleeveName
```

Then add the helpers and solver:

```python
def _apply_turnover_band(
    weights: dict[str, float],
    prior_weights: dict[str, float],
    *,
    drift_band: float,
    turnover_penalty: float,
) -> tuple[dict[str, float], float]:
    """No-trade drift band + L1 turnover penalty, applied BEFORE the final projection so the
    projection has the last say on neutrality. A symbol PRESENT in the prior whose target is
    within `drift_band` of its prior weight keeps the prior weight (no churn); otherwise it
    moves to target, shrunk toward prior by `turnover_penalty` (L1 damping). A symbol ABSENT
    from the prior (prior == 0) is ALWAYS-TRADE: it is never snapped to 0 by the band, so a
    fresh sub-drift-band leg survives the rebalance. Returns (adjusted_weights, l1_turnover)."""
    out: dict[str, float] = {}
    for sym, target in weights.items():
        if sym not in prior_weights:
            # fresh name: always trade (do not let the no-trade band delete it)
            out[sym] = target
            continue
        prior = prior_weights[sym]
        denom = abs(prior) if abs(prior) > 1e-12 else 1.0
        if abs(target - prior) / denom <= drift_band:
            out[sym] = prior
        else:
            out[sym] = target - turnover_penalty * (target - prior)
    l1 = sum(abs(out[s] - prior_weights.get(s, 0.0)) for s in out)
    return out, l1


def _scale_to_deploy_target(
    weights: dict[str, float], hedge_notional: float, *, equity: float,
    side_budget: float, deploy_target_frac: float,
) -> tuple[dict[str, float], float]:
    """Scale the projected (dollar+beta-neutral) book by a SINGLE positive scalar so the
    larger side's gross equals `deploy_target_frac * side_budget`. A positive scalar preserves
    BOTH neutralities exactly (Sum(k*w)=0, Sum(k*w*beta)=0) and scales each side's gross
    equally, so this restores the deployment floor WITHOUT re-breaking neutrality. The hedge
    notional is scaled by the same factor (it is part of the neutral vector). Returns
    (scaled_weights, scaled_hedge_notional)."""
    long_gross = sum(w for w in weights.values() if w > 0.0) * equity
    short_gross = -sum(w for w in weights.values() if w < 0.0) * equity
    # include the hedge in the side it sits on
    if hedge_notional > 0.0:
        long_gross += hedge_notional
    elif hedge_notional < 0.0:
        short_gross += -hedge_notional
    larger = max(long_gross, short_gross)
    if larger <= 0.0:
        return dict(weights), hedge_notional
    target_side_usd = deploy_target_frac * side_budget
    k = target_side_usd / larger
    return {s: w * k for s, w in weights.items()}, hedge_notional * k


def optimize_book(
    sleeves: list[SleeveSignal],
    geometries: list[CoinGeometry],
    *,
    equity: float,
    prior_legs: list[WeightLeg] | None,
    cfg: NeutralityConfig,
    regime: RegimeState | None = None,
    returns: pd.DataFrame | None = None,
) -> TargetWeights:
    """THE solver. Merge sleeves -> sentiment tilts -> HRP-shape (Ledoit-Wolf -> HRP) ->
    per-name & cluster caps -> turnover/no-trade band (vs prior) -> size BTC hedge on the
    alpha legs' residual beta (real DOF) and append it -> project alpha+hedge onto the
    dollar+beta-neutral set -> scale the neutral book to the per-side deployment target
    (single positive scalar, preserves neutrality) -> assemble TargetWeights with residuals
    + per-side deployment. Stress-tightens bands under a correlation-spike regime. Sets
    feasible=False (never silently un-neutral / under-deployed) if the bands or the floor
    cannot be met. `returns` (optional) is the per-symbol return frame used to build the
    Ledoit-Wolf covariance for HRP shaping; without it the merged split is used."""
    geo = {g.symbol: g for g in geometries}
    betas = {g.symbol: g.beta_btc for g in geometries}

    # Stress-tighten bands under a correlation-spike regime.
    band_mult = 1.0
    if regime is not None and regime.quadrant in (
        "high_vol_trend", "high_vol_range", "transition"
    ):
        band_mult = cfg.stress_band_mult
    dollar_band = cfg.dollar_band * band_mult
    beta_band = cfg.beta_band * band_mult

    # 1. assign risk budgets + merge sleeve tilts into one signed vector
    risk_parity_budgets(sleeves)
    merged = merge_sleeves(sleeves, geometries)

    # 2. apply sentiment conviction tilts BEFORE projection (sign-preserving, capped)
    tilted_tilts = apply_conviction_tilts(
        [SleeveTilt(symbol=s, direction="long" if w >= 0 else "short", target_weight=w)
         for s, w in merged.items()],
        geometries,
    )
    weights = {t.symbol: t.target_weight for t in tilted_tilts}

    # 3. HRP shaping: Ledoit-Wolf shrunk covariance -> HRP -> reshape per-name split per side
    if returns is not None and not returns.empty:
        labels = [s for s in weights if s in returns.columns]
        if len(labels) >= 2:
            cov = ledoit_wolf_cov(returns[labels])
            hrp = hrp_weights(cov, labels)
            weights = apply_hrp_weights(weights, hrp)

    # 4. per-name + cluster caps
    weights = apply_per_name_cap(weights, per_name_cap=cfg.per_name_cap)
    corr: dict[tuple[str, str], float] = {}  # no cross-corr snapshot in pure-math layer
    weights = apply_cluster_cap(
        weights, corr=corr, cluster_cap=cfg.cluster_cap, threshold=cfg.corr_threshold
    )

    # 5. turnover / no-trade band vs the prior book — BEFORE projection (fresh names always-trade)
    prior_weights = {leg.symbol: leg.weight for leg in (prior_legs or [])
                     if leg.sleeve != "hedge"}
    turnover_l1 = 0.0
    if prior_weights:
        weights, turnover_l1 = _apply_turnover_band(
            weights, prior_weights,
            drift_band=cfg.drift_band, turnover_penalty=cfg.turnover_penalty,
        )

    # 6. size the BTC hedge on the ALPHA legs' residual beta (real DOF) and append it, then
    #    project the alpha+hedge vector onto the dollar+beta-neutral set. With the hedge
    #    appended there are >= 3 names, so projection yields a non-trivial neutral book.
    hedge_notional = size_btc_hedge(
        weights, betas, equity=equity, side_budget=cfg.side_budget_usdt
    )
    proj_in = dict(weights)
    proj_betas = dict(betas)
    btc = "BTC/USDT:USDT"
    if abs(hedge_notional) > 1e-9:
        proj_in[btc] = proj_in.get(btc, 0.0) + hedge_notional / equity
        proj_betas.setdefault(btc, 1.0)
    projected = project_neutral(proj_in, proj_betas, dollar_band=dollar_band, beta_band=beta_band)
    # split the projected BTC weight back into (alpha BTC leg, hedge): the hedge keeps the
    # residual-beta share, the rest stays an alpha BTC leg. We carry the hedge as its own leg.
    hedge_weight = hedge_notional / equity if equity > 0 else 0.0
    alpha_weights = dict(projected)
    if abs(hedge_notional) > 1e-9:
        alpha_weights[btc] = projected.get(btc, 0.0) - hedge_weight
        if abs(alpha_weights[btc]) < 1e-12:
            alpha_weights.pop(btc, None)

    # 7. scale the neutral book up to the per-side deployment target (preserves neutrality)
    alpha_weights, hedge_notional = _scale_to_deploy_target(
        alpha_weights, hedge_notional, equity=equity,
        side_budget=cfg.side_budget_usdt, deploy_target_frac=cfg.deploy_target_frac,
    )

    # 8. assemble legs (alpha legs) + the hedge leg
    legs: list[WeightLeg] = []
    notionals: dict[str, float] = {}
    full_weights: dict[str, float] = {}
    full_betas: dict[str, float] = {}
    for sym, w in alpha_weights.items():
        if abs(w) < 1e-9:
            continue
        notional = w * equity
        notionals[sym] = notional
        full_weights[sym] = w
        full_betas[sym] = betas.get(sym, 1.0)
        legs.append(WeightLeg(
            symbol=sym,
            direction="long" if w > 0 else "short",
            weight=w,
            target_notional=notional,
            beta_btc=betas.get(sym, 1.0),
            sleeve=_dominant_sleeve(sym, sleeves),
        ))
    if abs(hedge_notional) > 1.0:
        hedge_w = hedge_notional / equity
        notionals["__hedge__"] = hedge_notional
        full_weights["__hedge__"] = hedge_w
        full_betas["__hedge__"] = 1.0
        legs.append(WeightLeg(
            symbol=btc,
            direction="long" if hedge_notional > 0 else "short",
            weight=hedge_w,
            target_notional=hedge_notional,
            beta_btc=1.0,
            sleeve="hedge",
        ))

    # residuals + per-side deployment (include hedge leg in dollar/beta sums)
    d_resid = dollar_residual(full_weights, notionals)
    d_resid_frac = abs(d_resid) / cfg.side_budget_usdt if cfg.side_budget_usdt > 0 else 0.0
    b_resid = beta_residual(full_weights, full_betas)
    gross_long = sum(n for n in notionals.values() if n > 0)
    gross_short = sum(-n for n in notionals.values() if n < 0)
    deploy_long = gross_long / cfg.side_budget_usdt if cfg.side_budget_usdt > 0 else 0.0
    deploy_short = gross_short / cfg.side_budget_usdt if cfg.side_budget_usdt > 0 else 0.0

    feasible = (
        d_resid_frac <= dollar_band + 1e-6
        and abs(b_resid) <= beta_band + 1e-6
        and deploy_long >= cfg.deployment_floor - 1e-6
        and deploy_short >= cfg.deployment_floor - 1e-6
        and deploy_long <= (1.0 - cfg.dry_powder_frac) + 1e-6
        and deploy_short <= (1.0 - cfg.dry_powder_frac) + 1e-6
    )
    notes: list[str] = []
    if not feasible:
        notes.append("constraint set infeasible: residual or deployment-floor breach")

    return TargetWeights(
        legs=legs,
        btc_hedge_notional=hedge_notional,
        dollar_residual=d_resid,
        dollar_residual_frac=d_resid_frac,
        beta_residual=b_resid,
        gross_long=gross_long,
        gross_short=gross_short,
        deploy_long_frac=deploy_long,
        deploy_short_frac=deploy_short,
        gross_notional=gross_long + gross_short,
        turnover_l1=turnover_l1,
        feasible=feasible,
        notes=notes,
        as_of_ts=datetime.now(UTC),
    )


def _dominant_sleeve(symbol: str, sleeves: list[SleeveSignal]) -> SleeveName:
    """The sleeve contributing the largest |budgeted tilt| to this symbol (source attribution)."""
    best: tuple[float, SleeveName] = (-1.0, "factor")
    for s in sleeves:
        for t in s.tilts:
            if t.symbol == symbol:
                contrib = abs(t.target_weight) * s.risk_budget_frac
                if contrib > best[0]:
                    best = (contrib, s.sleeve)
    return best[1]
```

> **Why step 7 cannot re-break neutrality (binding):** after step 6 the book (alpha legs +
> hedge) is dollar-neutral and beta-neutral, so `gross_long == gross_short` and
> `Σwβ == 0`. Multiplying every weight (and the hedge) by one positive `k` gives
> `Σkw = kΣw = 0` and `Σkwβ = kΣwβ = 0` — neutrality is invariant under positive scaling —
> while each side's gross scales by `k`. Choosing `k = deploy_target_frac·side_budget /
> max(long_gross, short_gross)` lands the larger side exactly on target and the (equal) other
> side with it, so both `deploy_*_frac` hit `deploy_target_frac ∈ [floor, 1−dry_powder]`.
> This is the fix for issues 1, 2, 3, and 6 — there is no post-projection gross shrink.

- [ ] **Step 4: Run the structural test to verify it passes**

Run: `uv run pytest tests/test_neutrality_optimize.py -k "optimize_book and not property" -v`
(scoped so the "4 structural tests" claim is unambiguous — the §15 PROPERTY tests are added to this
SAME file in Task 15 and are excluded here by the `not property` filter)
Expected: the 4 structural `optimize_book` tests PASS.

- [ ] **Step 5: Commit**

Run: `uv run ruff check futures_fund/neutrality.py tests/test_neutrality_optimize.py`
Expected: `All checks passed!`

```bash
git add futures_fund/neutrality.py tests/test_neutrality_optimize.py
git commit -m "feat: add optimize_book (merge->tilt->HRP->cap->turnover->hedge+project->scale-to-floor)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Property/invariant tests for `optimize_book`

**Files:**
- Modify: `tests/test_neutrality_optimize.py`

These are the §15 property tests (named `test_property_*`): neutrality residuals within band;
**deployment floor honored** (a normal book deploys ≥90%/side AND ≤(1−dry_powder)/side and is
`feasible`); **HRP shaping actually influences per-name notionals**; sentiment never flips
direction; gross ≈ $20k. The Task-14 Step-4 command excludes these via `-k '... and not property'`;
they run together with the structural tests in Step 2 below and in the Task 16 phase gate.

- [ ] **Step 1: Write the property tests (append to `tests/test_neutrality_optimize.py`)**

```python
import numpy as np
import pytest

from futures_fund.contracts import SleeveSignal, SleeveTilt


def _balanced_sleeves(now):
    """A risk-budgeted, dollar-balanced two-sleeve signal for property tests (>=3 active
    names per side after the hedge is appended, so projection is non-trivial)."""
    factor = SleeveSignal(
        sleeve="factor", risk_budget_frac=0.5, as_of_ts=now,
        tilts=[
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )
    carry = SleeveSignal(
        sleeve="carry", risk_budget_frac=0.5, as_of_ts=now,
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )
    return [factor, carry]


@pytest.mark.parametrize("seed", range(8))
def test_property_dollar_residual_within_band(seed, geometries):
    rng = np.random.default_rng(seed)
    geos = [
        g.model_copy(update={
            "beta_btc": float(rng.uniform(0.6, 1.6)),
            "sentiment_score": float(rng.uniform(-1.0, 1.0)),
            "sentiment_conf": float(rng.uniform(0.0, 1.0)),
        })
        for g in geometries
    ]
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geos, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    assert tw.dollar_residual_frac <= cfg.dollar_band + 1e-6


@pytest.mark.parametrize("seed", range(8))
def test_property_beta_residual_within_band(seed, geometries):
    rng = np.random.default_rng(seed + 100)
    geos = [
        g.model_copy(update={"beta_btc": float(rng.uniform(0.6, 1.6))})
        for g in geometries
    ]
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geos, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    assert abs(tw.beta_residual) <= cfg.beta_band + 1e-6


def test_property_deployment_floor_honored_on_balanced_book(geometries):
    # Spec §15 'deployment floor honored': a NORMAL balanced book must deploy >= floor on
    # BOTH sides AND <= (1 - dry_powder) on both sides, and be feasible. This is the direct
    # assertion the prior plan was missing (deploy ~0.766 < 0.90 made feasible always False).
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    assert tw.feasible is True
    assert tw.deploy_long_frac >= cfg.deployment_floor
    assert tw.deploy_short_frac >= cfg.deployment_floor
    # dry powder honored: never deploy beyond 1 - dry_powder_frac on either side
    assert tw.deploy_long_frac <= 1.0 - cfg.dry_powder_frac + 1e-6
    assert tw.deploy_short_frac <= 1.0 - cfg.dry_powder_frac + 1e-6


def test_property_gross_near_target_20k(geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    # gross including the hedge leg should land near the ~$20k target (within 20%). With the
    # scale-to-deploy-target step (Task 14 step 7) this now holds; do NOT loosen this bound.
    assert 0.8 * cfg.target_gross_usdt <= tw.gross_notional <= 1.2 * cfg.target_gross_usdt


def test_property_hrp_weighting_influences_per_name_notionals(geometries, returns_frame):
    # Spec §8: Ledoit-Wolf -> HRP must actually shape the book. Run the optimizer WITH a
    # returns frame (HRP active) vs WITHOUT (merged split), and assert the per-name long-side
    # notionals differ -> HRP is wired into optimize_book, not dead.
    cfg = NeutralityConfig()
    tw_plain = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                             prior_legs=None, cfg=cfg, returns=None)
    tw_hrp = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                           prior_legs=None, cfg=cfg, returns=returns_frame)

    def long_notional(tw, sym):
        for leg in tw.legs:
            if leg.symbol == sym and leg.sleeve != "hedge":
                return leg.target_notional
        return 0.0

    # SOL and BTC are the two long alpha names; HRP must redistribute between them.
    plain_sol = long_notional(tw_plain, "SOL/USDT:USDT")
    hrp_sol = long_notional(tw_hrp, "SOL/USDT:USDT")
    assert abs(hrp_sol - plain_sol) > 1.0  # HRP changed SOL's notional by > $1


def test_property_sentiment_never_flips_leg_direction(geometries):
    # Drown every leg in maximally adverse sentiment; directions must still match the
    # pre-sentiment sleeve intent.
    hostile = [
        g.model_copy(update={"sentiment_score": -1.0 if g.symbol in
                             ("SOL/USDT:USDT", "BTC/USDT:USDT") else 1.0,
                             "sentiment_conf": 1.0})
        for g in geometries
    ]
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), hostile, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    intent = {"SOL/USDT:USDT": "long", "XRP/USDT:USDT": "short",
              "BTC/USDT:USDT": "long", "ETH/USDT:USDT": "short"}
    for leg in tw.legs:
        if leg.sleeve == "hedge":
            continue
        # projection can shrink a leg to ~0 but must never flip its sign vs intent
        if abs(leg.weight) > 1e-6 and leg.symbol in intent:
            assert leg.direction == intent[leg.symbol]


def test_property_turnover_band_keeps_residuals_in_band_with_prior(geometries):
    # With a non-empty PRIOR book, the turnover/no-trade band runs BEFORE projection, so the
    # final projected book must STILL be dollar+beta neutral within band (the band can never
    # re-break neutrality, because projection has the last say). Spec §8/§9.
    cfg = NeutralityConfig()
    first = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                          prior_legs=None, cfg=cfg)
    # rebalance against the first book as prior
    second = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                           prior_legs=first.legs, cfg=cfg)
    assert second.dollar_residual_frac <= cfg.dollar_band + 1e-6
    assert abs(second.beta_residual) <= cfg.beta_band + 1e-6
    assert second.feasible is True


def test_property_new_sub_drift_band_leg_survives_rebalance(geometries):
    # A fresh name absent from the prior must NOT be snapped to 0 by the no-trade band, even
    # if its target magnitude is below drift_band. Build a prior WITHOUT SOL, then rebalance
    # a book that introduces SOL long; SOL must appear as a non-zero leg.
    from datetime import UTC, datetime

    now = datetime(2026, 6, 11, tzinfo=UTC)
    prior_sleeves = [SleeveSignal(
        sleeve="carry", risk_budget_frac=1.0, as_of_ts=now,
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )]
    cfg = NeutralityConfig()
    prior = optimize_book(prior_sleeves, geometries, equity=20000.0,
                          prior_legs=None, cfg=cfg)
    assert not any(leg.symbol == "SOL/USDT:USDT" for leg in prior.legs)
    # now rebalance with the balanced sleeves (introduces SOL long) against that prior
    after = optimize_book(_balanced_sleeves(now), geometries, equity=20000.0,
                          prior_legs=prior.legs, cfg=cfg)
    sol_legs = [leg for leg in after.legs if leg.symbol == "SOL/USDT:USDT"]
    assert sol_legs  # fresh sub-drift-band leg survived (always-trade)
    assert abs(sol_legs[0].target_notional) > 1.0


def test_property_no_silent_un_neutral_sets_feasible_flag(geometries):
    # A pathological single-name one-sided book cannot satisfy deployment floor on both
    # sides; the optimizer must flag feasible=False rather than report a fake-neutral book.
    s = SleeveSignal(
        sleeve="factor", risk_budget_frac=1.0, as_of_ts=NOW,
        tilts=[SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=1.0)],
    )
    cfg = NeutralityConfig()
    tw = optimize_book([s], geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    if tw.deploy_short_frac < cfg.deployment_floor:
        assert tw.feasible is False
        assert tw.notes
```

- [ ] **Step 2: Run the full optimize file (structural + property) to verify it passes**

Run: `uv run pytest tests/test_neutrality_optimize.py -v`
Expected: all structural + property tests PASS (≈13 tests: 4 structural + the `test_property_*`
set). If a dollar/beta/deployment property test fails, the bug is in `optimize_book`'s
project→scale ordering or the residual sums — fix there, **not** by loosening the assertion.

- [ ] **Step 3: Commit**

Run: `uv run ruff check tests/test_neutrality_optimize.py`
Expected: `All checks passed!`

```bash
git add tests/test_neutrality_optimize.py
git commit -m "test: add property tests (neutrality bands, deployment floor honored, HRP shapes notionals, gross ~20k, sentiment never flips, turnover-safe, feasible flag)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Phase gate — full suite green + lint clean

**Files:**
- (none — verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: all tests across `tests/` PASS (test_beta, test_neutrality_contracts,
test_neutrality_residuals, test_neutrality_weighting, test_neutrality_caps,
test_neutrality_project, and `tests/test_neutrality_optimize.py` — which now holds the 4
structural `optimize_book` tests PLUS the §15 `test_property_*` set added in Task 15, ≈13 tests
total in that file), 0 failures.

- [ ] **Step 2: Run the full linter**

Run: `uv run ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Confirm public surface matches the contract**

Run: `uv run python -c "from futures_fund.neutrality import (NeutralityConfig, risk_parity_budgets, merge_sleeves, apply_hrp_weights, ledoit_wolf_cov, hrp_weights, project_neutral, size_btc_hedge, dollar_residual, beta_residual, optimize_book, conviction_tilt, apply_conviction_tilts, apply_per_name_cap, apply_cluster_cap); from futures_fund.beta import log_returns, rolling_beta, beta_series, beta_for_symbols; print('surface ok')"`
Expected: prints `surface ok` and exits 0. (Every symbol named here is defined in the
canonical contract for `beta.py` / `neutrality.py` or is a net-new helper defined within this plan.)

- [ ] **Step 4: Commit the phase marker (empty/allowed if nothing staged — use --allow-empty)**

```bash
git commit --allow-empty -m "chore: Phase 1 neutrality + portfolio optimizer complete (full suite green)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (against the spec)

**Spec coverage (§5, §8, §15, §17 step 1):**
- Rolling BTC beta (§5) → Tasks 4-6 (`beta.py`).
- Dollar-neutral band + beta-neutral band (§5) → `project_neutral` (Task 11) + residual measures (Task 7) + property tests (Task 15).
- BTC hedge leg as a real DOF sized jointly inside per-side budget (§5) → `size_btc_hedge` (Task 12) called on the **alpha legs' residual beta BEFORE** `project_neutral`, appended as a third column and projected with the alpha legs (Task 14 step 6); the structural test asserts a NON-zero short BTC hedge for a beta-imbalanced (long-high-beta / short-low-beta) book.
- Deployment floor ≥90%/side + dry powder (§4) → `NeutralityConfig.deployment_floor`/`dry_powder_frac`/`deploy_target_frac` (Task 7); `optimize_book` **scales the neutral book to `deploy_target_frac·side_budget`/side** (Task 14 step 7), landing each side in `[floor, 1−dry_powder]`; property test asserts `deploy_*_frac ≥ floor` AND `≤ 1−dry_powder` AND `feasible` on a balanced book (Task 15).
- Per-name + per-cluster caps reusing consolidation cluster logic (§4, §8) → `apply_per_name_cap` + `apply_cluster_cap` adapting the weekly-desk same-side union-find (Task 10).
- Ledoit-Wolf shrunk covariance + HRP/risk-parity actually shaping the book (§8) → `ledoit_wolf_cov` + `hrp_weights` (Task 8) wired via `apply_hrp_weights` (Task 9) into `optimize_book` step 3 (Task 14); property test asserts HRP changes per-name notionals (Task 15).
- L1 turnover penalty + no-trade drift band, applied BEFORE projection, fresh names always-trade (§8, §9) → `_apply_turnover_band` (Task 14 step 5); property tests assert residuals stay in band with a non-None prior and that a fresh sub-drift-band leg survives (Task 15).
- Sentiment tilt then RE-PROJECT, ≤25% cap, never flips sign (§7.2, §7.3) → `conviction_tilt`/`apply_conviction_tilts` (Task 13) applied BEFORE `project_neutral` in `optimize_book` (Task 14); property test asserts no flip (Task 15).
- Produces `TargetWeights` from `SleeveSignal` + `CoinGeometry` (§8, §14) → contracts (Task 2), `optimize_book` (Task 14).
- Pure math, fully TDD with synthetic fixtures + property tests (§15) → every task is failing-test-first; property tests in Task 15; gross ≈ $20k asserted; the n≤2 degenerate projection case documented and tested (Task 11).

**Issue-fix trace (all 12 review issues):**
1-3. Post-projection gross shrink / floor breach → `optimize_book` step 7 scales the neutral book to `deploy_target_frac·side_budget` with a single positive scalar (neutrality-preserving); `test_property_gross_near_target_20k` and `test_property_deployment_floor_honored_on_balanced_book` assert it (no loosened bounds).
4. Vestigial hedge → hedge sized on the alpha legs' residual beta BEFORE projection and projected as a third column; structural test asserts a non-zero short BTC hedge for a beta-imbalanced book.
5. HRP never wired → `apply_hrp_weights` + `optimize_book` step 3 (Ledoit-Wolf→HRP) + `test_property_hrp_weighting_influences_per_name_notionals`.
6. Dead `dry_powder_frac` → consumed via `deploy_target_frac` (≤ 1−dry_powder) and the deployment property test's upper-bound assertion.
7. Turnover after projection → moved BEFORE projection (step 5); `test_property_turnover_band_keeps_residuals_in_band_with_prior`.
8. Fresh small legs deleted → `_apply_turnover_band` treats absent-from-prior as always-trade; `test_property_new_sub_drift_band_leg_survives_rebalance`.
9. Obfuscated beta test → replaced with a real BTC-less marks dict asserting `out == {}` (`test_beta_for_symbols_missing_btc_returns_empty`).
10. Ambiguous pass-count → Task 14 Step 4 scoped with `-k 'optimize_book and not property'`; Task 15/16 note the file now holds structural + property tests.
11. Fragile import note → Task 14 Step 3 shows the FINAL exact import block for `neutrality.py`.
12. n≤2 over-constraint → documented degenerate-case note + `test_project_neutral_two_names_collapse_is_documented` + `test_project_neutral_three_names_retains_nontrivial_gross`; the optimizer appends the hedge so ≥3 names always reach projection.

**Placeholder scan:** no TBD/TODO/"handle edge cases"/"write tests for the above" — every code step shows complete real code; every test step shows the actual assertions.

**Type consistency:** all referenced symbols (`SleeveSignal`, `SleeveTilt`, `CoinGeometry`, `WeightLeg`, `TargetWeights`, `NeutralityConfig`, `RegimeState`, `SleeveName`) are defined in Tasks 1-2 or lifted in Task 1, and match the canonical contract names/signatures verbatim. Contract function names (`risk_parity_budgets`, `merge_sleeves`, `ledoit_wolf_cov`, `hrp_weights`, `project_neutral`, `size_btc_hedge`, `dollar_residual`, `beta_residual`, `optimize_book`, `conviction_tilt`, `apply_conviction_tilts`, `log_returns`, `rolling_beta`, `beta_series`, `beta_for_symbols`) are exactly the contract's `beta.py`/`neutrality.py` signatures; `optimize_book`'s optional `returns=` parameter matches the reviewer's `review_cycle(..., returns=...)` re-derivation path in the P3-7 roadmap. `apply_hrp_weights`/`apply_per_name_cap`/`apply_cluster_cap`/`_apply_turnover_band`/`_scale_to_deploy_target`/`_dominant_sleeve` are net-new helpers fully defined within this plan (allowed: defined here, not invented from training). Out-of-Phase-1 contract members (`reviewer.check_caps`, `control_loop`, sleeve builders) are intentionally not referenced.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-11-phase1-neutrality-optimizer.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

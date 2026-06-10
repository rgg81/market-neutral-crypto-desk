# Phase 0 — Foundation: scaffold, data layer, realism primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Stand up the market-neutral desk's testable data + costs foundation: project scaffold (uv/pyproject, ruff, pytest, `config.yaml`, dir layout, `MISSION.md`), a crypto-only liquid USD-M universe filter with a liquidity floor (~top 20-30), the keyless Binance data layer (klines/mark/funding-history/exchangeInfo/depth), per-symbol funding intervals + signed/unclamped realized funding with per-symbol caps, depth-aware slippage, corrected fees, the Sharpe-periodicity fix (×365 daily / ×52 weekly), and a point-in-time sentiment ingestion source layer. **No trading logic, optimizer, or sleeves yet** — those are Phases 1-2.

**Architecture:** "LLM proposes, code disposes." This phase builds only the deterministic Python spine (`futures_fund/`) and its tests. Reused types/modules are lifted **verbatim** from `/home/roberto/crypto-trade-claude-code-weekly`; net-new modules follow the CANONICAL NEW-INTERFACE CONTRACT exactly. When a module is lifted verbatim, its tests are written against the **real** reused API (the lifted module's actual class fields and function signatures), not a re-imagined one. Every public symbol below is either reused-as-named or defined in the contract — none are invented.

**Tech Stack:** Python ≥3.11, `uv`-managed package `futures_fund/`; `pydantic>=2.6`, `numpy`, `pandas`, `scipy`, `statsmodels` (cointegration/OU — added now so the lockfile is ready for Phase 2), `scikit-learn` (Ledoit-Wolf — added now), `cvxpy` (optimizer — added now), `ccxt` (keyless Binance USD-M), `httpx`, `pyyaml`; `ruff` + `pytest` (`-q`, `testpaths=["tests"]`). All math is unit-tested TDD-first with real Binance-shaped fixture data.

---

## File Structure

Every file this phase creates or modifies, with its single responsibility. All paths are under the project root `/home/roberto/crypto-trade-claude-code-market-neutral`.

| File | Create/Modify | Single responsibility |
|---|---|---|
| `pyproject.toml` | Create | uv package metadata + deps (adds `statsmodels`, `scikit-learn`, `cvxpy`) + ruff + pytest config. |
| `.gitignore` | Modify | Ignore `.venv/`, `__pycache__/`, caches, `state/`, `.env`. |
| `MISSION.md` | Create | Charter stub for the market-neutral desk (no-losing-month primary KPI). |
| `config.yaml` | Create | Non-secret config: account, two-cadence loops, `neutrality:`, `beta:`, `sleeves:`, `sentiment:`, `universe:`, `fees:`, `funding:`, `slippage:`, `metrics:`, `reviewer:`, `graduation:`. |
| `futures_fund/__init__.py` | Create | Package marker. |
| `futures_fund/models.py` | Create | Reused domain types lifted verbatim from weekly + new shared aliases (`SleeveName`, `SentimentLevel`, `SpreadState`, `PairTestMethod`, `Cadence`). |
| `futures_fund/config.py` | Create | `Settings`/`load_settings` extended with the new config blocks (`NeutralityConfig` lives in `neutrality.py`, Phase 1; this phase parses the raw blocks into `Settings` fields). Keeps the inherited `agent_models`-first `model_for`. |
| `futures_fund/market_data.py` | Create | Binance payload parsers + `is_crypto_perp` + `scan_universe` + new `liquidity_floor` filter (lifted from weekly, extended with the ADV floor). |
| `futures_fund/exchange.py` | Create | `FuturesExchange` keyless wrapper (klines/funding/mark/exchangeInfo) + new `depth(symbol)` order-book method. |
| `futures_fund/market_context.py` | Create | Reused market-wide context assembler (news RSS / Fear&Greed / FRED / Reddit) feeding the Sentiment Analyst — **lifted verbatim** from weekly. |
| `futures_fund/vendors.py` | Create | Keyless feed fetchers + parsers + typed item models (`FearGreed`, `NewsItem`, `SocialPost`, `fetch_*`, `parse_*`, `tag_instruments`, `archive_jsonl`) that `market_context` depends on — **lifted verbatim** from weekly. |
| `futures_fund/costs.py` | Create | Fees (5bps taker / 2bps maker), `project_funding`, `count_funding_events`, `vwap_fill`, `slippage_cost` (lifted verbatim). |
| `futures_fund/funding_intervals.py` | Create (NEW) | Per-symbol funding interval + per-symbol caps (BTC/ETH ±0.30%, alts ±2%), sign-preserving clamp, signed realized funding, APR annualization. |
| `futures_fund/slippage.py` | Create (NEW) | Depth-aware slippage wiring `vwap_fill` + sqrt-impact fallback; never flat 2bps. |
| `futures_fund/metrics.py` | Create | Performance stats with the Sharpe-periodicity fix: `PERIODS_PER_YEAR_DAILY=365`, `PERIODS_PER_YEAR_WEEKLY=52`. |
| `futures_fund/sentiment_ingest.py` | Create (NEW) | Point-in-time sentiment gather over `market_context`, level↔s mapping, half-life decay, fail-soft neutral, point-in-time validation. |
| `futures_fund/contracts.py` | Create | `SentimentReport`/`SentimentSource`/`SentimentBatch` contracts (only the sentiment slice needed this phase; geometry/pair/weights contracts arrive in Phase 1). |
| `tests/__init__.py` | Create | Test package marker. |
| `tests/test_models.py` | Create | New shared aliases + reused-model validation. |
| `tests/test_config.py` | Create | `load_settings` parses the new config blocks (incl. two-cadence loops) + `model_for` agent-first resolution. |
| `tests/test_market_data.py` | Create | `scan_universe`, `is_crypto_perp`, `liquidity_floor`, parsers against Binance-shaped fixtures. |
| `tests/test_exchange.py` | Create | `FuturesExchange` keyless methods incl. `depth` against a fake ccxt client. |
| `tests/test_market_context.py` | Create | `build_market_context` degrades per-feed against a fake http client; real `FearGreed`/`NewsItem` models construct. |
| `tests/test_costs.py` | Create | Fee/funding/vwap/slippage primitives. |
| `tests/test_funding_intervals.py` | Create | Per-symbol interval, caps, sign-preserving clamp, signed realized funding, APR, clamp∘realized composition. |
| `tests/test_slippage.py` | Create | Depth-aware slippage + fallback + $1M calibration + monotonicity (never flat 2bps). |
| `tests/test_metrics.py` | Create | Sharpe ×365 daily / ×52 weekly periodicity fix. |
| `tests/test_sentiment_ingest.py` | Create | level↔s, decay, point-in-time validation, fail-soft. |
| `tests/test_contracts.py` | Create | `SentimentReport`/`SentimentSource` validation + range constraints. |

---

### Task 0: Project scaffold (pyproject, deps, ruff, pytest, MISSION, .gitignore)

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/pyproject.toml`
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/MISSION.md`
- Modify: `/home/roberto/crypto-trade-claude-code-market-neutral/.gitignore`
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/__init__.py`
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml` with deps + tooling config.** Adds `statsmodels`, `scikit-learn`, `cvxpy` to the inherited stack so the lockfile is ready for Phases 1-2.

```toml
[project]
name = "futures-fund"
version = "0.1.0"
description = "Market-neutral crypto trading desk — Binance USD-M perpetual futures (paper)"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "numpy>=1.26",
    "pandas>=2.1",
    "ccxt>=4.5",
    "httpx>=0.27",
    "scipy>=1.11",
    "statsmodels>=0.14",
    "scikit-learn>=1.4",
    "cvxpy>=1.5",
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

- [ ] **Step 2: Create the package + test markers.** `futures_fund/__init__.py` and `tests/__init__.py` are both empty files. Write each with a single trailing newline:

```python
```

(Both files are intentionally empty — write a zero-byte/newline-only file to each path.)

- [ ] **Step 3: Write `MISSION.md` charter stub.**

```markdown
# OPERATION MARKET-NEUTRAL

**We are an autonomous crypto-futures PAPER desk with one mandate: never lose a calendar month, and maximize Sharpe on the daily equity series (annualized ×365, benchmark = cash). We earn this by staying roughly neutral to the overall crypto market and harvesting RELATIVE value — relative-value pairs, funding-rate carry, cross-sectional factors, and sentiment — on Binance USD-M perpetual futures (paper).**

We run **equal capital on both sides**: ~$10k long and ~$10k short on a $20k paper account (~1× gross). Neutrality (dollar + beta) is a **hard construction constraint**, never an excuse to sit flat — full two-sided deployment (≥90% per side) is the default state. The remaining dry powder funds daily rebalancing.

We are **all-weather by construction**: because the book is market-neutral, it aims to be positive across regimes rather than betting on direction. A dedicated BTC-perp hedge leg absorbs residual beta; rolling beta is re-estimated each cycle.

We **deploy on two clocks**: a **weekly Selection Meeting** (symbol set + target weights) and a **daily Rebalance Meeting** (same set, trade only drift/breaches). We pay **realistic costs** — taker 5 bps / maker 2 bps, per-symbol signed funding, depth-aware slippage — and the edge must clear them every rebalance.

We are **paranoid about correctness**: every cycle, an Adversarial Code & Calc Reviewer re-derives neutrality residuals, funding sign/amount, pair P&L, RR-after-costs, and Sharpe annualization against ground truth, and HALTs on any mismatch.

We trade **cryptocurrencies only** — no tokenized stocks, indexes, metals, or gold coins. `live` stays `false` forever.

We remember: every decision is written down before its outcome is known and judged on alpha (return net of BTC-beta), not raw return. *We get a little sharper every cycle.*
```

- [ ] **Step 4: Update `.gitignore`.** Replace the file contents:

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
state/
memory/
.env
uv.lock
```

- [ ] **Step 5: Sync deps and verify the toolchain.** Run:

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv sync
```

Expected: `uv` resolves and installs all deps including `statsmodels`, `scikit-learn`, `cvxpy`; creates `.venv/` and `uv.lock`. Then confirm pytest collects zero tests cleanly:

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest
```

Expected: `no tests ran` (exit 5 from pytest's "no tests collected" is acceptable here) — confirms the harness is wired before any test exists.

- [ ] **Step 6: Commit the scaffold.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git checkout -b phase0-foundation && git add -A && git commit -m "$(cat <<'EOF'
Phase 0 scaffold: pyproject, deps, MISSION, gitignore, package markers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1: Reused domain types + new shared aliases (`models.py`)

Lift the reused types verbatim from `/home/roberto/crypto-trade-claude-code-weekly/futures_fund/models.py` and add the five new shared type aliases from the contract (§0).

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/models.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_models.py`

- [ ] **Step 1: Write the failing test for the new aliases + a reused model.** `tests/test_models.py`:

```python
from futures_fund.models import (
    Cadence,
    Direction,
    PairTestMethod,
    SentimentLevel,
    SleeveName,
    SpreadState,
    SymbolSpec,
    TradeProposal,
    get_args,
)


def test_new_shared_aliases_have_expected_members():
    assert set(get_args(SleeveName)) == {"carry", "pairs", "factor", "sentiment"}
    assert set(get_args(SentimentLevel)) == {
        "very_positive", "positive", "neutral", "negative", "very_negative"
    }
    assert set(get_args(SpreadState)) == {"flat", "long_spread", "short_spread", "stop"}
    assert set(get_args(PairTestMethod)) == {"engle_granger", "johansen"}
    assert set(get_args(Cadence)) == {"weekly", "daily"}
    assert set(get_args(Direction)) == {"long", "short"}


def test_trade_proposal_rejects_long_stop_above_entry():
    import pytest
    with pytest.raises(ValueError):
        TradeProposal(symbol="BTCUSDT", direction="long", entry=100.0, stop=101.0,
                      atr=1.0, confidence=0.5, horizon_hours=8.0, funding_rate=0.0001)


def test_symbol_spec_sorts_brackets_by_floor():
    from futures_fund.models import MmrBracket
    spec = SymbolSpec(symbol="BTCUSDT", tick_size=0.1, step_size=0.001, min_notional=5.0,
                      mmr_brackets=[
                          MmrBracket(notional_floor=50.0, notional_cap=100.0, mmr=0.01,
                                     maint_amount=1.0, max_leverage=50.0),
                          MmrBracket(notional_floor=0.0, notional_cap=50.0, mmr=0.005,
                                     maint_amount=0.0, max_leverage=100.0),
                      ])
    assert [b.notional_floor for b in spec.sorted_brackets] == [0.0, 50.0]
```

- [ ] **Step 2: Run the test — expect FAIL (module missing).**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_models.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.models'` (collection error / FAIL).

- [ ] **Step 3: Write `futures_fund/models.py`.** Reused types verbatim from weekly + the new aliases + a re-export of `get_args` so tests import it from one place.

```python
from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, Field, model_validator

__all__ = ["get_args"]

Direction = Literal["long", "short"]
RegimeQuadrant = Literal[
    "low_vol_trend", "high_vol_trend", "low_vol_range", "high_vol_range", "transition"
]
HealthTier = Literal["healthy", "caution", "stressed"]
Bias = Literal["normal", "reduce", "flat"]
Verdict = Literal["approve", "resize", "veto"]

# --- new shared aliases (CANONICAL CONTRACT §0) ---
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
    mmr: float                      # maintenance margin rate
    maint_amount: float             # maintenance amount offset (cum)
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
    funding_rate: float             # current/predicted per-interval funding rate
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

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_models.py -q && uv run ruff check futures_fund/models.py tests/test_models.py
```

Expected: 3 tests pass; ruff `All checks passed!`.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: reused domain types + new shared aliases (models.py)

Lift reused pydantic models verbatim from crypto-trade-claude-code-weekly
and add SleeveName/SentimentLevel/SpreadState/PairTestMethod/Cadence aliases.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Realistic fee/funding/slippage primitives (`costs.py`)

Lift `costs.py` verbatim from weekly — it already encodes 5 bps taker / 2 bps maker, signed `project_funding`, `count_funding_events`, `vwap_fill`, `slippage_cost`. These are the building blocks `funding_intervals.py` and `slippage.py` extend.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/costs.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_costs.py`

- [ ] **Step 1: Write the failing test.** `tests/test_costs.py`:

```python
from datetime import datetime, timezone

import pytest

from futures_fund.costs import (
    MAKER_RATE,
    TAKER_RATE,
    count_funding_events,
    project_funding,
    round_trip_fee,
    slippage_cost,
    trade_fee,
    vwap_fill,
)


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def test_fee_rates_are_5bps_taker_2bps_maker():
    assert TAKER_RATE == pytest.approx(0.0005)
    assert MAKER_RATE == pytest.approx(0.0002)


def test_taker_fee_is_5bps_of_notional():
    assert trade_fee(notional=10_000.0, maker=False) == pytest.approx(5.0)


def test_maker_fee_is_2bps_of_notional():
    assert trade_fee(notional=10_000.0, maker=True) == pytest.approx(2.0)


def test_round_trip_taker_in_and_out():
    assert round_trip_fee(10_000.0, maker_entry=False, maker_exit=False) == pytest.approx(10.0)


def test_count_funding_events_crossing_two_boundaries():
    n = count_funding_events(_utc(2026, 5, 29, 7, 0), _utc(2026, 5, 29, 17, 0))
    assert n == 2


def test_project_funding_short_receives_positive_rate():
    # short with a positive funding rate RECEIVES funding -> negative cost (a credit)
    cost = project_funding(notional=10_000.0, funding_rate=0.0001, direction="short", n_events=3)
    assert cost == pytest.approx(-3.0)


def test_project_funding_long_pays_positive_rate():
    cost = project_funding(notional=10_000.0, funding_rate=0.0001, direction="long", n_events=3)
    assert cost == pytest.approx(3.0)


def test_vwap_fill_walks_the_book():
    filled, vwap = vwap_fill([(100.0, 1.0), (101.0, 1.0)], qty=1.5)
    assert filled == pytest.approx(1.5)
    assert vwap == pytest.approx((100.0 * 1.0 + 101.0 * 0.5) / 1.5)


def test_slippage_cost_is_filled_times_abs_vwap_gap():
    cost = slippage_cost([(100.0, 1.0), (101.0, 1.0)], qty=1.5, reference_price=100.0)
    filled, vwap = vwap_fill([(100.0, 1.0), (101.0, 1.0)], qty=1.5)
    assert cost == pytest.approx(filled * abs(vwap - 100.0))
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_costs.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.costs'`.

- [ ] **Step 3: Write `futures_fund/costs.py`** (verbatim lift from weekly).

```python
from __future__ import annotations

from datetime import datetime, timedelta

from futures_fund.models import Direction

TAKER_RATE = 0.0005   # 0.05%
MAKER_RATE = 0.0002   # 0.02%
BNB_DISCOUNT = 0.90    # 10% off when paying fees in BNB


def trade_fee(notional: float, *, maker: bool, pay_bnb: bool = False) -> float:
    """Fee in USDT for a single fill of `notional` USDT."""
    rate = MAKER_RATE if maker else TAKER_RATE
    fee = abs(notional) * rate
    return fee * BNB_DISCOUNT if pay_bnb else fee


def round_trip_fee(
    notional: float, *, maker_entry: bool, maker_exit: bool, pay_bnb: bool = False
) -> float:
    """Entry + exit fee assuming the same notional both legs (conservative)."""
    return (
        trade_fee(notional, maker=maker_entry, pay_bnb=pay_bnb)
        + trade_fee(notional, maker=maker_exit, pay_bnb=pay_bnb)
    )


DEFAULT_FUNDING_INTERVAL_HOURS = 8  # majors default; per-symbol sourced in funding_intervals.py


def funding_boundary_hours(interval_hours: int = DEFAULT_FUNDING_INTERVAL_HOURS) -> tuple[int, ...]:
    """UTC hours at which funding settles (8h -> 0,8,16; 4h -> 0,4,8,12,16,20)."""
    return tuple(range(0, 24, interval_hours))


def count_funding_events(
    entry_ts: datetime, exit_ts: datetime,
    interval_hours: int = DEFAULT_FUNDING_INTERVAL_HOURS,
) -> int:
    """Number of funding settlements strictly within (entry_ts, exit_ts]."""
    if exit_ts <= entry_ts:
        return 0
    hours = set(funding_boundary_hours(interval_hours))
    count = 0
    cursor = entry_ts.replace(minute=0, second=0, microsecond=0)
    if cursor <= entry_ts:
        cursor += timedelta(hours=1)
    while cursor <= exit_ts:
        if cursor.hour in hours:
            count += 1
        cursor += timedelta(hours=1)
    return count


def project_funding(
    notional: float, funding_rate: float, direction: Direction, n_events: int
) -> float:
    """Projected funding cost in USDT (positive = we pay, negative = we receive)."""
    sign = 1.0 if direction == "long" else -1.0
    return abs(notional) * funding_rate * sign * n_events


def vwap_fill(levels: list[tuple[float, float]], qty: float) -> tuple[float, float]:
    """Walk price/qty `levels` (in crossing order) to fill `qty`.

    Returns (filled_qty, vwap). If depth is insufficient, returns the partial fill.
    """
    if qty <= 0 or not levels:
        return 0.0, 0.0
    remaining = qty
    cost = 0.0
    filled = 0.0
    for price, avail in levels:
        take = min(remaining, avail)
        cost += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break
    vwap = cost / filled if filled > 0 else 0.0
    return filled, vwap


def slippage_cost(
    levels: list[tuple[float, float]], qty: float, reference_price: float
) -> float:
    """USDT slippage cost: filled_qty * |vwap - reference_price|."""
    filled, vwap = vwap_fill(levels, qty)
    if filled <= 0:
        return 0.0
    return filled * abs(vwap - reference_price)
```

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_costs.py -q && uv run ruff check futures_fund/costs.py tests/test_costs.py
```

Expected: 10 tests pass; ruff clean.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: realistic fee/funding/slippage primitives (costs.py)

Lift costs.py verbatim from weekly: 5bps taker / 2bps maker, signed
project_funding (short receives positive funding), vwap_fill, slippage_cost.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Per-symbol funding intervals + caps + signed realized funding (`funding_intervals.py`)

NEW module per contract §2.3. Replaces the hardcoded-8h assumption with per-symbol intervals, applies per-symbol caps (BTC/ETH ±0.30%, alts ±2%) **sign-preserving** (never zeroed), and computes signed realized funding (a short receives positive funding) and APR annualization.

**§11 / contract §2.3 clamp-vs-realized ordering (load-bearing).** The cap is applied to the *rate* upstream (`clamp_funding_rate`); `realized_funding` then consumes the **signed** value and stays signed (never floored at ≥0). Config keys `unclamped_in_rr` / `signed_realized` reflect this: realized PnL keeps the signed, *unclamped-in-the-realized-path* contribution while the **rate** is clamped per-symbol before it is fed in. This task pins that ordering with an explicit composition test (`realized_funding(..., rate=clamp_funding_rate(symbol, raw_rate), ...)`), and pins `realized_funding`'s `notional_signed` arg as deliberately ignored (it computes from `mark*qty`).

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/funding_intervals.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_funding_intervals.py`

- [ ] **Step 1: Write the failing test.** `tests/test_funding_intervals.py`:

```python
import pytest

from futures_fund.funding_intervals import (
    MAJOR_CAP,
    PER_SYMBOL_CAP_DEFAULT,
    _MAJORS,
    clamp_funding_rate,
    funding_apr,
    funding_cap,
    funding_interval_hours,
    intervals_per_year,
    realized_funding,
)


class _FakeFundingInfo:
    """Stand-in for market_data.FundingInfo — only the `.interval_hours: float` field is read by
    funding_interval_hours (see test_funding_interval_consumes_fundinginfo_interval_hours, which
    ties this to the CONCRETE FundingInfo produced by exchange.funding() in Task 7)."""
    def __init__(self, interval_hours):
        self.interval_hours = interval_hours


class _FakeExchange:
    def __init__(self, hours):
        self._hours = hours

    def funding(self, symbol):
        if self._hours is None:
            raise RuntimeError("no funding info")
        return _FakeFundingInfo(self._hours)


def test_majors_set_contains_btc_and_eth():
    assert _MAJORS == frozenset({"BTC/USDT:USDT", "ETH/USDT:USDT"})


def test_interval_sourced_per_symbol():
    assert funding_interval_hours("SOL/USDT:USDT", _FakeExchange(4.0)) == pytest.approx(4.0)


def test_interval_defaults_to_8h_on_miss():
    assert funding_interval_hours("SOL/USDT:USDT", _FakeExchange(None)) == pytest.approx(8.0)


def test_funding_interval_consumes_fundinginfo_interval_hours():
    # END-TO-END WIRING (spec §11 / contract §2.3 "replace hardcoded 8h, source per-symbol"):
    # the interval funding_intervals consumes is EXACTLY the float on the FundingInfo that the real
    # exchange.funding() returns. Build a real FundingInfo (as parse_funding/exchange.funding do)
    # and feed it through a tiny exchange shim — funding_interval_hours must read its .interval_hours.
    from datetime import datetime, timezone

    from futures_fund.market_data import FundingInfo

    info = FundingInfo(symbol="SOL/USDT:USDT", current_rate=0.0002,
                       next_funding_ts=datetime(2026, 6, 1, tzinfo=timezone.utc),
                       interval_hours=4.0, mark_price=150.0, index_price=149.9)

    class _RealInfoExchange:
        def funding(self, symbol):
            return info

    assert funding_interval_hours("SOL/USDT:USDT", _RealInfoExchange()) == pytest.approx(4.0)


def test_funding_cap_majors_vs_alts():
    assert funding_cap("BTC/USDT:USDT") == pytest.approx(MAJOR_CAP) == pytest.approx(0.003)
    assert funding_cap("ETH/USDT:USDT") == pytest.approx(0.003)
    assert funding_cap("SOL/USDT:USDT") == pytest.approx(PER_SYMBOL_CAP_DEFAULT) == \
        pytest.approx(0.02)


def test_clamp_is_sign_preserving_and_bounded():
    # alt rate beyond +2% cap -> clamped to +0.02, sign kept
    assert clamp_funding_rate("SOL/USDT:USDT", 0.05) == pytest.approx(0.02)
    # negative beyond -2% -> -0.02
    assert clamp_funding_rate("SOL/USDT:USDT", -0.05) == pytest.approx(-0.02)
    # BTC small negative rate stays signed, never zeroed
    assert clamp_funding_rate("BTC/USDT:USDT", -0.0001) == pytest.approx(-0.0001)
    # BTC beyond +0.30% -> +0.003
    assert clamp_funding_rate("BTC/USDT:USDT", 0.01) == pytest.approx(0.003)


def test_intervals_per_year_for_8h_is_1095():
    assert intervals_per_year(8.0) == pytest.approx(24.0 / 8.0 * 365.0)


def test_funding_apr_is_signed_annualized():
    # +0.01% per 8h -> APR = 0.0001 * 1095 = 0.1095
    assert funding_apr(0.0001, 8.0) == pytest.approx(0.0001 * 1095.0)
    # negative rate -> negative APR (signed carry)
    assert funding_apr(-0.0001, 8.0) == pytest.approx(-0.0001 * 1095.0)


def test_realized_funding_short_receives_positive_rate():
    # short with positive rate RECEIVES: -side*mark*qty*rate, side(short)=-1 -> positive credit
    # contribution to BALANCE is positive (credit)
    bal = realized_funding(notional_signed=-10_000.0, mark=100.0, qty=100.0,
                           rate=0.0001, direction="short")
    assert bal == pytest.approx(+1.0)


def test_realized_funding_long_pays_positive_rate():
    bal = realized_funding(notional_signed=10_000.0, mark=100.0, qty=100.0,
                           rate=0.0001, direction="long")
    assert bal == pytest.approx(-1.0)


def test_realized_funding_ignores_notional_signed():
    # §11 dead-arg pin: notional_signed is accepted for call-site symmetry but UNUSED — the
    # contribution is derived from mark*qty. A deliberately-wrong notional_signed=0.0 must NOT
    # change the result (still -side*mark*qty*rate), so a caller cannot desync funding_amount.
    bal = realized_funding(notional_signed=0.0, mark=100.0, qty=100.0,
                           rate=0.0001, direction="short")
    assert bal == pytest.approx(+1.0)


def test_clamp_then_realized_composition_for_an_alt():
    # §11 / contract §2.3 ORDERING: cap the RATE upstream, then realized consumes the SIGNED,
    # clamped rate and stays signed. SOL raw +0.05 exceeds the +0.02 alt cap -> clamped to +0.02;
    # a SHORT then RECEIVES a credit on the clamped rate: -(-1)*mark*qty*0.02 = +mark*qty*0.02.
    raw_rate = 0.05
    clamped = clamp_funding_rate("SOL/USDT:USDT", raw_rate)
    assert clamped == pytest.approx(0.02)
    bal = realized_funding(notional_signed=-15_000.0, mark=150.0, qty=100.0,
                           rate=clamped, direction="short")
    assert bal == pytest.approx(+150.0 * 100.0 * 0.02)   # +300.0 credit, signed & on the clamped rate
    # and the realized contribution is NOT the raw (unclamped) rate's value:
    assert bal != pytest.approx(150.0 * 100.0 * raw_rate)
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_funding_intervals.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.funding_intervals'`.

- [ ] **Step 3: Write `futures_fund/funding_intervals.py`.**

```python
from __future__ import annotations

from futures_fund.models import Direction

PER_SYMBOL_CAP_DEFAULT: float = 0.02   # alts default magnitude (+-2%)
MAJOR_CAP: float = 0.003               # BTC/ETH magnitude (+-0.30%)
_MAJORS: frozenset[str] = frozenset({"BTC/USDT:USDT", "ETH/USDT:USDT"})


def funding_interval_hours(symbol: str, exchange) -> float:
    """Per-symbol settlement interval from /fapi/v1/fundingInfo via FuturesExchange.funding().

    Reads the `interval_hours: float` field on the FundingInfo that exchange.funding(symbol)
    returns (see market_data.FundingInfo / exchange.funding in Task 7). Defaults to 8.0 hours when
    the funding info is missing or the call fails (the major default).
    """
    try:
        return float(exchange.funding(symbol).interval_hours)
    except Exception:
        return 8.0


def funding_cap(symbol: str) -> float:
    """Clamp magnitude for the realized rate: MAJOR_CAP for majors else PER_SYMBOL_CAP_DEFAULT."""
    return MAJOR_CAP if symbol in _MAJORS else PER_SYMBOL_CAP_DEFAULT


def clamp_funding_rate(symbol: str, rate: float) -> float:
    """Clamp a realized rate to [-cap, +cap], SIGN-PRESERVING (carry stays signed, never zeroed)."""
    cap = funding_cap(symbol)
    if rate > cap:
        return cap
    if rate < -cap:
        return -cap
    return rate


def intervals_per_year(interval_hours: float) -> float:
    """24/interval_hours * 365 — annualization factor for funding_apr."""
    if interval_hours <= 0:
        return 0.0
    return 24.0 / interval_hours * 365.0


def funding_apr(rate: float, interval_hours: float) -> float:
    """Signed annualized carry = rate * intervals_per_year(interval_hours)."""
    return rate * intervals_per_year(interval_hours)


def realized_funding(
    notional_signed: float, mark: float, qty: float, rate: float, direction: Direction  # noqa: ARG001
) -> float:
    """Settlement contribution to BALANCE: -side*mark*qty*rate.

    side = +1 for long, -1 for short. A short (-1) with a positive `rate` RECEIVES funding, so the
    balance contribution is positive (a credit). Signed; never clamped to >= 0 here. Per §11 /
    contract §2.3 the per-symbol cap is applied to the RATE upstream via clamp_funding_rate, and
    this function consumes that signed, clamped rate.

    `notional_signed` is accepted for call-site symmetry with WeightLeg.target_notional (Phase 1)
    but is DELIBERATELY UNUSED — the contribution is derived from mark*qty so a partial fill is
    handled by the caller's `qty`. test_realized_funding_ignores_notional_signed pins this so a
    caller cannot desync the reviewer's funding_amount re-derivation by passing a wrong notional.
    """
    side = 1.0 if direction == "long" else -1.0
    return -side * mark * qty * rate
```

> Note: the `# noqa: ARG001` on `notional_signed` documents the deliberate non-use (rather than leaving it accidental); `test_realized_funding_ignores_notional_signed` pins the dead-arg behavior. The `intervals_per_year(0.0) -> 0.0` guard keeps `funding_apr` finite on a degenerate interval.

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_funding_intervals.py -q && uv run ruff check futures_fund/funding_intervals.py tests/test_funding_intervals.py
```

Expected: 12 tests pass; ruff clean.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: per-symbol funding intervals + caps + signed realized funding

funding_intervals.py: source interval per-symbol (replace hardcoded 8h, read off
the concrete FundingInfo.interval_hours from exchange.funding()), per-symbol caps
(BTC/ETH +-0.30%, alts +-2%) sign-preserving clamp, signed realized funding (short
receives positive funding) consuming the clamped rate, APR annualization. Tests pin
the clamp->realized ordering and the deliberately-ignored notional_signed arg.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Depth-aware slippage (`slippage.py`)

NEW module per contract §2.4. Wires `costs.vwap_fill`/`costs.slippage_cost` against an L2 depth snapshot, falls back to `half_spread + k·√(notional/ADV)`. Never flat 2 bps.

**On §11 calibration anchors (honest scope).** Spec §11 lists approximate anchors "BTCUSDT ~1.25 bps @ $1M, ~3.6 bps @ $5M". These two anchors are **not simultaneously satisfiable** by the spec's own `half_spread + k·√(notional/ADV)` law: a pure √-impact term grows as √5 ≈ 2.24× from $1M→$5M, while 3.6/1.25 = 2.88×, which would imply impact ∝ notional^0.74 — a faster-than-√ law the spec does not specify. Rather than assert a contradicted, untested property, this task (a) pins the **$1M** anchor exactly against a representative BTC depth proxy (`adv_usd` + `half_spread_bps` chosen so the model yields ≈1.25 bps at $1M with the config default `k=0.1`), and (b) pins the qualitative §11 requirements that ARE real: the fallback is **strictly monotone-increasing** in notional and is **never the flat 2 bps** the spec forbids. The docstring and commit make NO unsupported "calibrated to both anchors" claim.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/slippage.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_slippage.py`

- [ ] **Step 1: Write the failing test.** `tests/test_slippage.py`:

```python
import math

import pytest

from futures_fund.slippage import (
    DEFAULT_K,
    depth_slippage,
    estimate_slippage,
    fallback_slippage,
    slippage_bps,
)

# Representative BTC depth proxy: with the config default k=0.1 and half_spread=1.0 bps, this
# adv_usd makes the fallback yield EXACTLY 1.25 bps @ $1M (spec §11 BTC anchor):
#   1.25 = 1.0 + 0.1*sqrt(1e6/adv)*1e4  ->  sqrt(1e6/adv) = 2.5e-4  ->  adv = 1.6e13
_BTC_ADV_USD = 1.6e13
_BTC_HALF_SPREAD_BPS = 1.0


def test_default_k_is_point_one():
    assert DEFAULT_K == pytest.approx(0.1)


def test_depth_slippage_matches_vwap_gap():
    # buying 1.5 units: 1.0 @ 100, 0.5 @ 101 -> vwap = 100.333..., ref = 100
    levels = [(100.0, 1.0), (101.0, 1.0)]
    cost = depth_slippage(levels, qty=1.5, reference_price=100.0)
    vwap = (100.0 * 1.0 + 101.0 * 0.5) / 1.5
    assert cost == pytest.approx(1.5 * abs(vwap - 100.0))


def test_fallback_slippage_half_spread_plus_sqrt_impact():
    # notional = 1e6, adv = 1e9, half_spread = 1 bps, k = 0.1
    # cost_bps = 1.0 + 0.1*sqrt(1e6/1e9)*1e4 = 1.0 + 0.1*0.0316...*1e4
    notional, adv, hs_bps = 1_000_000.0, 1_000_000_000.0, 1.0
    cost = fallback_slippage(notional, adv, hs_bps, k=0.1)
    expected_bps = hs_bps + 0.1 * math.sqrt(notional / adv) * 1e4
    assert cost == pytest.approx(expected_bps / 1e4 * notional)


def test_fallback_zero_adv_returns_half_spread_only():
    cost = fallback_slippage(1_000_000.0, 0.0, 1.0, k=0.1)
    assert cost == pytest.approx(1.0 / 1e4 * 1_000_000.0)


def test_fallback_btc_1m_anchor_is_about_1_25_bps():
    # §11 BTC anchor PINNED: ~1.25 bps @ $1M against the representative BTC depth proxy + k=0.1.
    cost = fallback_slippage(1_000_000.0, _BTC_ADV_USD, _BTC_HALF_SPREAD_BPS, k=DEFAULT_K)
    assert slippage_bps(cost, 1_000_000.0) == pytest.approx(1.25, rel=1e-6)


def test_fallback_is_strictly_monotone_in_notional():
    # §11 requires impact to GROW with size (a larger clip costs more bps) — pin monotonicity at
    # the $1M and $5M points instead of asserting the unsatisfiable second anchor exactly.
    bps_1m = slippage_bps(
        fallback_slippage(1_000_000.0, _BTC_ADV_USD, _BTC_HALF_SPREAD_BPS, k=DEFAULT_K),
        1_000_000.0)
    bps_5m = slippage_bps(
        fallback_slippage(5_000_000.0, _BTC_ADV_USD, _BTC_HALF_SPREAD_BPS, k=DEFAULT_K),
        5_000_000.0)
    assert bps_5m > bps_1m  # $5M strictly costlier per-bp than $1M


def test_estimate_prefers_depth_when_present():
    levels = [(100.0, 100.0)]  # deep enough to fill at the ref price -> ~0 slippage
    cost = estimate_slippage("BTC/USDT:USDT", qty=1.0, reference_price=100.0,
                             depth=levels, adv_usd=1e9, half_spread_bps=1.0)
    assert cost == pytest.approx(0.0)


def test_estimate_falls_back_when_no_depth():
    cost = estimate_slippage("SOL/USDT:USDT", qty=10_000.0, reference_price=100.0,
                             depth=None, adv_usd=1e9, half_spread_bps=1.0, k=0.1)
    notional = 10_000.0 * 100.0
    assert cost == pytest.approx(fallback_slippage(notional, 1e9, 1.0, k=0.1))


def test_estimate_never_flat_two_bps():
    # a tiny fill against a deep book is essentially free, NOT a flat 2bps
    levels = [(100.0, 1_000_000.0)]
    cost = estimate_slippage("BTC/USDT:USDT", qty=1.0, reference_price=100.0,
                             depth=levels, adv_usd=1e12, half_spread_bps=1.0)
    assert cost < 0.02 * 100.0  # far below a flat 2bps of the 100 USDT notional


def test_slippage_bps_converts_cost_to_bps():
    assert slippage_bps(cost_usdt=125.0, notional=1_000_000.0) == pytest.approx(1.25)
    assert slippage_bps(cost_usdt=10.0, notional=0.0) == 0.0
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_slippage.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.slippage'`.

- [ ] **Step 3: Write `futures_fund/slippage.py`.**

```python
from __future__ import annotations

import math

from futures_fund.costs import slippage_cost

DEFAULT_K: float = 0.1   # sqrt-impact coefficient for the ADV fallback (config.yaml slippage.k)


def depth_slippage(
    levels: list[tuple[float, float]], qty: float, reference_price: float
) -> float:
    """Thin wrapper over costs.slippage_cost against an L2 depth snapshot (USDT cost).

    Direction-symmetric: `levels` are the crossing side of the book (asks to buy, bids to sell).
    """
    return slippage_cost(levels, qty, reference_price)


def fallback_slippage(
    notional: float, adv_usd: float, half_spread_bps: float, *, k: float = DEFAULT_K
) -> float:
    """half_spread + k*sqrt(notional/ADV) impact model in USDT when no depth snapshot.

    Strictly increasing in notional (a larger clip costs more bps); never returns a flat 2 bps.
    The k*sqrt term is a √-impact law, so it grows ~sqrt(notional); the per-bp cost is therefore
    monotone in size. (Spec §11's two approximate anchors are not both satisfiable by a pure
    √-law; see test_slippage.py — the $1M anchor is pinned, the $5M point is pinned for
    monotonicity, and no 'calibrated to both anchors' property is claimed here.)
    """
    notional = abs(notional)
    if adv_usd <= 0:
        impact_bps = 0.0
    else:
        impact_bps = k * math.sqrt(notional / adv_usd) * 1e4
    cost_bps = half_spread_bps + impact_bps
    return cost_bps / 1e4 * notional


def estimate_slippage(
    symbol: str, qty: float, reference_price: float, *,
    depth: list[tuple[float, float]] | None, adv_usd: float,
    half_spread_bps: float, k: float = DEFAULT_K,
) -> float:
    """Prefer depth_slippage; fall back to fallback_slippage. NEVER flat 2 bps. Returns USDT cost."""
    if depth:
        return depth_slippage(depth, qty, reference_price)
    notional = abs(qty) * reference_price
    return fallback_slippage(notional, adv_usd, half_spread_bps, k=k)


def slippage_bps(cost_usdt: float, notional: float) -> float:
    """Convenience: cost in bps of notional (for the §11 calibration / monotonicity assertions)."""
    if notional <= 0:
        return 0.0
    return cost_usdt / notional * 1e4
```

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_slippage.py -q && uv run ruff check futures_fund/slippage.py tests/test_slippage.py
```

Expected: 10 tests pass; ruff clean.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: depth-aware slippage (slippage.py)

Wire vwap_fill against an L2 depth snapshot, fall back to
half_spread + k*sqrt(notional/ADV). Never flat 2bps; the fallback is strictly
monotone in notional. Tests pin the §11 $1M BTC anchor (~1.25 bps) against a
representative depth proxy and the $5M point for monotonicity; the spec's second
anchor is not √-law-satisfiable, so no two-anchor calibration is claimed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Sharpe-periodicity fix in metrics (`metrics.py`)

The inherited `PERIODS_PER_YEAR = 2190` (4h cycles) makes every Sharpe/Sortino wrong for this desk. Per spec §11/§18, the daily equity series annualizes ×365 and the weekly ×52. Define both constants and make `sharpe`/`sortino` take an explicit `periods_per_year` with the daily default.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/metrics.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test.** `tests/test_metrics.py`:

```python
import math

import pytest

from futures_fund.metrics import (
    PERIODS_PER_YEAR_DAILY,
    PERIODS_PER_YEAR_WEEKLY,
    calmar,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
)


def test_periodicity_constants_are_365_daily_52_weekly():
    assert PERIODS_PER_YEAR_DAILY == pytest.approx(365.0)
    assert PERIODS_PER_YEAR_WEEKLY == pytest.approx(52.0)


def test_sharpe_default_annualizes_daily_x365():
    rets = [0.01, -0.005, 0.012, 0.003, -0.002, 0.008]
    import numpy as np
    arr = np.asarray(rets)
    expected = arr.mean() / arr.std(ddof=1) * math.sqrt(365.0)
    assert sharpe(rets) == pytest.approx(expected)


def test_sharpe_weekly_annualizes_x52():
    rets = [0.02, -0.01, 0.015, 0.005]
    import numpy as np
    arr = np.asarray(rets)
    expected = arr.mean() / arr.std(ddof=1) * math.sqrt(52.0)
    assert sharpe(rets, periods_per_year=PERIODS_PER_YEAR_WEEKLY) == pytest.approx(expected)


def test_sharpe_too_few_returns_is_zero():
    assert sharpe([0.01]) == 0.0


def test_sortino_uses_downside_rms_x365():
    rets = [0.01, -0.02, 0.01, -0.01]
    import numpy as np
    arr = np.asarray(rets)
    dd = math.sqrt(np.mean(np.minimum(arr, 0.0) ** 2))
    expected = arr.mean() / dd * math.sqrt(365.0)
    assert sortino(rets) == pytest.approx(expected)


def test_max_drawdown_peak_to_trough():
    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx((120.0 - 90.0) / 120.0)


def test_calmar_and_hit_rate_and_profit_factor():
    assert calmar(0.20, 0.05) == pytest.approx(4.0)
    assert hit_rate([{"realized_pnl": 1.0}, {"realized_pnl": -1.0}]) == pytest.approx(0.5)
    assert profit_factor([{"realized_pnl": 2.0}, {"realized_pnl": -1.0}]) == pytest.approx(2.0)
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_metrics.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.metrics'`.

- [ ] **Step 3: Write `futures_fund/metrics.py`** (Sharpe-periodicity fixed; daily default).

```python
from __future__ import annotations

import numpy as np

# Sharpe periodicity FIX (spec §11/§18): the daily equity series annualizes x365,
# the weekly x52. The inherited 2190 (4h) factor would make every Sharpe/Sortino/DSR wrong.
PERIODS_PER_YEAR_DAILY = 365.0
PERIODS_PER_YEAR_WEEKLY = 52.0


def sharpe(returns: list[float], periods_per_year: float = PERIODS_PER_YEAR_DAILY) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(arr.mean() / sd * np.sqrt(periods_per_year))


def trial_sharpe_std(return_streams: list[list[float]], min_obs: int = 5) -> float | None:
    """Cross-trial Sharpe dispersion (sigma_SR) for the Deflated Sharpe Ratio: the std of each
    trial's PER-PERIOD Sharpe. None when < 2 trials each with >= min_obs observations."""
    shrps = [sharpe(s, periods_per_year=1.0) for s in return_streams if len(s) >= min_obs]
    if len(shrps) < 2:
        return None
    return float(np.std(shrps, ddof=1))


def sortino(returns: list[float], periods_per_year: float = PERIODS_PER_YEAR_DAILY) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    dd = float(np.sqrt(np.mean(np.minimum(arr, 0.0) ** 2)))
    if dd == 0:
        return float("inf") if arr.mean() > 0 else 0.0
    return float(arr.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: list[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction (0 if monotonic up / too short)."""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak)
    return mdd


def calmar(annual_return: float, mdd: float) -> float:
    return annual_return / mdd if mdd > 0 else 0.0


def hit_rate(closed: list[dict]) -> float:
    if not closed:
        return 0.0
    wins = sum(1 for d in closed if d["realized_pnl"] > 0)
    return wins / len(closed)


def profit_factor(closed: list[dict]) -> float:
    gains = sum(d["realized_pnl"] for d in closed if d["realized_pnl"] > 0)
    losses = -sum(d["realized_pnl"] for d in closed if d["realized_pnl"] < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses
```

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_metrics.py -q && uv run ruff check futures_fund/metrics.py tests/test_metrics.py
```

Expected: 7 tests pass; ruff clean.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: Sharpe-periodicity fix in metrics (x365 daily / x52 weekly)

Replace the inherited 2190 (4h) annualization with PERIODS_PER_YEAR_DAILY=365
and PERIODS_PER_YEAR_WEEKLY=52 so daily-equity Sharpe/Sortino are correct.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Binance payload parsers + crypto-only universe + liquidity floor (`market_data.py`)

Lift the weekly parsers, `is_crypto_perp`, and `scan_universe` verbatim, and add a NEW `liquidity_floor` that trims the scanned universe to a min-ADV floor and top-N (the ~top 20-30 liquid-large-cap requirement, spec §4/§13).

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/market_data.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_market_data.py`

- [ ] **Step 1: Write the failing test.** `tests/test_market_data.py`:

```python
import pytest

from futures_fund.market_data import (
    is_crypto_perp,
    liquidity_floor,
    parse_funding,
    parse_ohlcv,
    parse_symbol_spec,
    scan_universe,
)

_MARKETS = {
    "BTC/USDT:USDT": {"info": {"underlyingType": "COIN", "contractType": "PERPETUAL"}},
    "DOGE/USDT:USDT": {"info": {"underlyingType": "COIN", "contractType": "PERPETUAL"}},
    "GOLD/USDT:USDT": {"info": {"underlyingType": "COMMODITY",
                                "contractType": "TRADIFI_PERPETUAL"}},
    "SOL/USDT:USDT": {"info": {"underlyingType": "COIN", "contractType": "PERPETUAL"}},
}


class _TickerClient:
    markets = _MARKETS

    def fetch_tickers(self):
        return {
            "BTC/USDT:USDT": {"quoteVolume": 1e10, "percentage": 0.1, "last": 70000.0},
            "GOLD/USDT:USDT": {"quoteVolume": 8e9, "percentage": 0.5, "last": 2300.0},  # TradFi
            "SOL/USDT:USDT": {"quoteVolume": 9e9, "percentage": -1.0, "last": 150.0},
            "DOGE/USDT:USDT": {"quoteVolume": 3e7, "percentage": -2.0, "last": 0.1},  # thin
            "ETH/USDT:USD": {"quoteVolume": 9e9, "percentage": 0.0, "last": 2000.0},  # not perp
        }


def test_is_crypto_perp_rejects_tradfi_wrapper():
    assert is_crypto_perp(_MARKETS["BTC/USDT:USDT"]) is True
    assert is_crypto_perp(_MARKETS["GOLD/USDT:USDT"]) is False
    assert is_crypto_perp({"info": {}}) is True  # metadata gap -> keep plain perp


def test_scan_universe_drops_tradfi_and_non_perp():
    rows = scan_universe(_TickerClient(), top_n=10)
    syms = [r["symbol"] for r in rows]
    assert "GOLD/USDT:USDT" not in syms      # tokenized commodity excluded
    assert "ETH/USDT:USD" not in syms        # not a USDT perp
    assert syms == ["BTC/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"]  # vol-ranked


def test_liquidity_floor_trims_thin_names_and_caps_top_n():
    rows = scan_universe(_TickerClient(), top_n=10)
    kept = liquidity_floor(rows, min_adv_usd=5e7, symbol_count=30)
    syms = [r["symbol"] for r in kept]
    assert syms == ["BTC/USDT:USDT", "SOL/USDT:USDT"]  # DOGE (3e7) below the 5e7 floor


def test_liquidity_floor_caps_to_symbol_count():
    rows = [{"symbol": f"X{i}/USDT:USDT", "vol_24h_usd": 1e9 - i, "last": 1.0,
             "chg_24h_pct": 0.0} for i in range(40)]
    kept = liquidity_floor(rows, min_adv_usd=0.0, symbol_count=30)
    assert len(kept) == 30
    assert kept[0]["symbol"] == "X0/USDT:USDT"  # most liquid first preserved


def test_parse_ohlcv_sorts_and_labels_columns():
    df = parse_ohlcv([[1700000000000, 1, 2, 0.5, 1.5, 10],
                      [1699999996400, 1, 2, 0.5, 1.5, 10]])
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].is_monotonic_increasing


def test_parse_funding_defaults_interval_8h():
    fr = {"symbol": "BTC/USDT:USDT", "fundingRate": "0.0001",
          "fundingTimestamp": 1700000000000, "markPrice": "70000", "indexPrice": "69990"}
    info = parse_funding(fr)
    assert info.interval_hours == pytest.approx(8.0)
    assert info.current_rate == pytest.approx(0.0001)
    assert info.mark_price == pytest.approx(70000.0)


def test_parse_funding_sources_interval_when_present():
    fr = {"symbol": "SOL/USDT:USDT", "fundingRate": "0.0002",
          "fundingTimestamp": 1700000000000, "markPrice": "150", "indexPrice": "149.9"}
    info = parse_funding(fr, {"info": {"fundingIntervalHours": 4}})
    assert info.interval_hours == pytest.approx(4.0)


def test_parse_symbol_spec_maps_filters_and_brackets():
    market = {"id": "BTCUSDT", "symbol": "BTC/USDT:USDT",
              "precision": {"price": 0.1, "amount": 0.001},
              "limits": {"cost": {"min": 100.0}},
              "info": {"filters": [
                  {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                  {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                  {"filterType": "MIN_NOTIONAL", "notional": "5.0"}]}}
    tiers = [{"minNotional": 0, "maxNotional": 50000, "maintenanceMarginRate": 0.004,
              "maxLeverage": 125, "info": {"cum": "0"}}]
    spec = parse_symbol_spec(market, tiers)
    assert spec.tick_size == pytest.approx(0.1)
    assert spec.step_size == pytest.approx(0.001)
    assert spec.min_notional == pytest.approx(5.0)
    assert spec.mmr_brackets[0].max_leverage == pytest.approx(125.0)
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_market_data.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.market_data'`.

- [ ] **Step 3: Write `futures_fund/market_data.py`** (verbatim weekly parsers + new `liquidity_floor`).

```python
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from pydantic import BaseModel, Field

from futures_fund.models import MmrBracket, SymbolSpec


class FundingInfo(BaseModel):
    symbol: str
    current_rate: float = Field(
        description="Current (last) funding rate, NOT a prediction "
        "(ccxt fundingRate == Binance lastFundingRate)."
    )
    next_funding_ts: datetime
    interval_hours: float
    mark_price: float
    index_price: float


def _filter_field(filters: list[dict], filter_type: str, field: str) -> float | None:
    for f in filters:
        if f.get("filterType") == filter_type and field in f:
            return float(f[field])
    return None


def parse_symbol_spec(market: dict, tiers: list[dict]) -> SymbolSpec:
    """ccxt market dict + leverage tiers -> SymbolSpec, preferring exchangeInfo filters."""
    filters = (market.get("info") or {}).get("filters") or []
    tick = _filter_field(filters, "PRICE_FILTER", "tickSize")
    step = _filter_field(filters, "LOT_SIZE", "stepSize")
    min_notional = _filter_field(filters, "MIN_NOTIONAL", "notional")
    if tick is None:
        tick = float(market["precision"]["price"])
    if step is None:
        step = float(market["precision"]["amount"])
    if min_notional is None:
        min_notional = float(market["limits"]["cost"]["min"])
    brackets = [
        MmrBracket(
            notional_floor=float(t["minNotional"]),
            notional_cap=float(t["maxNotional"]),
            mmr=float(t["maintenanceMarginRate"]),
            maint_amount=float(t["info"]["cum"]),
            max_leverage=float(t["maxLeverage"]),
        )
        for t in tiers
    ]
    return SymbolSpec(
        symbol=market["id"],
        tick_size=tick,
        step_size=step,
        min_notional=min_notional,
        mmr_brackets=brackets,
    )


# CRYPTO-ONLY desk: Binance USD-M lists TradFi-wrapper perps (gold/silver/oil COMMODITY,
# US/KR stocks EQUITY/KR_EQUITY, PREMARKET pre-IPO, INDEX baskets) that rank HIGH by 24h volume.
# `underlyingType` is COIN for the real cryptocurrencies; everything else is excluded.
_CRYPTO_UNDERLYING_TYPES = frozenset({"COIN"})


def is_crypto_perp(market: dict | None) -> bool:
    """True only for a cryptocurrency COIN perp; False for TradFi-wrapper contracts.

    Uses `underlyingType` authoritatively (COIN-only allowlist); on a metadata gap falls back to
    `contractType` so a TRADIFI_PERPETUAL is still rejected while a plain PERPETUAL is kept.
    """
    info = (market or {}).get("info") or {}
    utype = info.get("underlyingType")
    if utype:
        return utype in _CRYPTO_UNDERLYING_TYPES
    ctype = info.get("contractType")
    return ctype in (None, "", "PERPETUAL")


def scan_universe(client, top_n: int = 30) -> list[dict]:
    """Rank the live USD-M linear perp universe by 24h quote volume. Public/keyless. Returns up to
    top_n rows {symbol, last, chg_24h_pct, vol_24h_usd}, most-liquid first. Skips non-USDT-perps,
    zero vol/price, and (CRYPTO-ONLY) every non-cryptocurrency TradFi-wrapper perp."""
    tickers = client.fetch_tickers()
    markets = getattr(client, "markets", None) or {}
    rows: list[dict] = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT:USDT"):
            continue
        if not is_crypto_perp(markets.get(sym)):
            continue
        qv = t.get("quoteVolume") or 0.0
        last = t.get("last")
        if qv and last:
            rows.append({"symbol": sym, "last": float(last),
                         "chg_24h_pct": round(float(t.get("percentage") or 0.0), 2),
                         "vol_24h_usd": float(qv)})
    rows.sort(key=lambda r: r["vol_24h_usd"], reverse=True)
    return rows[:top_n]


def liquidity_floor(rows: list[dict], *, min_adv_usd: float, symbol_count: int) -> list[dict]:
    """Trim a vol-ranked universe to liquid large-caps: drop names below the 24h-ADV floor, then
    cap to `symbol_count` (the ~top 20-30 requirement, spec §4/§13). Input is assumed already
    ranked most-liquid-first by scan_universe; the floor is applied on `vol_24h_usd`."""
    kept = [r for r in rows if float(r.get("vol_24h_usd") or 0.0) >= min_adv_usd]
    return kept[:symbol_count]


def parse_ohlcv(rows: list[list]) -> pd.DataFrame:
    """ccxt OHLCV rows [[ts_ms,o,h,l,c,v], ...] -> sorted UTC-timestamped DataFrame."""
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def parse_funding(fr: dict, interval: dict | None = None) -> FundingInfo:
    interval_hours = 8.0
    if interval and (interval.get("info") or {}).get("fundingIntervalHours") is not None:
        interval_hours = float(interval["info"]["fundingIntervalHours"])
    return FundingInfo(
        symbol=fr["symbol"],
        current_rate=float(fr["fundingRate"]),
        next_funding_ts=datetime.fromtimestamp(  # noqa: UP017
            fr["fundingTimestamp"] / 1000, tz=timezone.utc),
        interval_hours=interval_hours,
        mark_price=float(fr["markPrice"]),
        index_price=float(fr["indexPrice"]),
    )


def parse_open_interest_history(rows: list[dict]) -> pd.DataFrame:
    cols = ["timestamp", "oi_amount", "oi_value"]
    recs = []
    for r in rows:
        try:
            recs.append({
                "timestamp": pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True),
                "oi_amount": float(r["openInterestAmount"]),
                "oi_value": (float(r["openInterestValue"])
                             if r.get("openInterestValue") is not None else float("nan")),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not recs:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(recs).sort_values("timestamp").reset_index(drop=True)


def parse_long_short_ratio(raw_rows: list[dict]) -> pd.DataFrame:
    cols = ["timestamp", "long_short_ratio", "long_account", "short_account"]
    recs = []
    for r in raw_rows:
        try:
            recs.append({
                "timestamp": pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True),
                "long_short_ratio": float(r["longShortRatio"]),
                "long_account": float(r["longAccount"]),
                "short_account": float(r["shortAccount"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not recs:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(recs).sort_values("timestamp").reset_index(drop=True)
```

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_market_data.py -q && uv run ruff check futures_fund/market_data.py tests/test_market_data.py
```

Expected: 9 tests pass; ruff clean.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: Binance parsers + crypto-only universe + liquidity floor

Lift parsers/is_crypto_perp/scan_universe verbatim from weekly and add
liquidity_floor (min-ADV + top-N) for the liquid large-cap universe.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Data layer — keyless `FuturesExchange` with depth (`exchange.py`)

Lift the keyless `FuturesExchange` wrapper from weekly (klines/funding/mark/exchangeInfo via `default_symbol_spec`) and add a NEW `depth(symbol)` order-book method (spec §13: order-book depth for slippage). Tests use a fake ccxt client — no network.

**Per-symbol funding-interval wiring (spec §11 / contract §2.3).** `exchange.funding(symbol)` returns a concrete `market_data.FundingInfo` whose `interval_hours: float` is exactly the value `funding_intervals.funding_interval_hours(symbol, exchange)` consumes (Task 3). `test_exchange.py` asserts `ex.funding('BTC/USDT:USDT').interval_hours == 8.0`, and Task 3's `test_funding_interval_consumes_fundinginfo_interval_hours` feeds a real `FundingInfo` through `funding_interval_hours`, so the "replace hardcoded 8h, source per-symbol" linkage is demonstrated end-to-end (real model on both sides), not only against a stand-in.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/exchange.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_exchange.py`

- [ ] **Step 1: Write the failing test.** `tests/test_exchange.py`:

```python
import pytest

from futures_fund.exchange import FuturesExchange, default_symbol_spec
from futures_fund.market_data import FundingInfo


class _FakeClient:
    markets = {"BTC/USDT:USDT": {"id": "BTCUSDT"}}
    markets_by_id = {"BTCUSDT": {"symbol": "BTC/USDT:USDT", "id": "BTCUSDT"}}

    def market(self, symbol):
        return {"id": "BTCUSDT", "symbol": "BTC/USDT:USDT",
                "precision": {"price": 0.1, "amount": 0.001},
                "limits": {"cost": {"min": 5.0}},
                "info": {"filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5.0"}]}}

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        return [[1700000000000, 70000, 70100, 69900, 70050, 12.0]]

    def fetch_funding_rate(self, symbol):
        return {"symbol": "BTC/USDT:USDT", "fundingRate": "0.0001",
                "fundingTimestamp": 1700000000000, "markPrice": "70050", "indexPrice": "70040"}

    def fetch_funding_interval(self, symbol):
        return {"info": {"fundingIntervalHours": 8}}

    def fetch_order_book(self, symbol, limit):
        return {"bids": [[70040.0, 1.5], [70030.0, 2.0]],
                "asks": [[70060.0, 1.2], [70070.0, 3.0]]}


def test_default_symbol_spec_from_public_filters():
    spec = default_symbol_spec(_FakeClient().market("BTC/USDT:USDT"))
    assert spec.symbol == "BTCUSDT"
    assert spec.tick_size == pytest.approx(0.1)
    assert spec.min_notional == pytest.approx(5.0)
    assert spec.mmr_brackets[0].max_leverage == pytest.approx(20.0)  # conservative paper bracket


def test_keyless_symbol_spec_uses_default_bracket():
    ex = FuturesExchange(_FakeClient(), keyless=True)
    spec = ex.symbol_spec("BTC/USDT:USDT")
    assert len(spec.mmr_brackets) == 1 and spec.mmr_brackets[0].mmr == pytest.approx(0.05)


def test_ohlcv_returns_parsed_frame():
    ex = FuturesExchange(_FakeClient(), keyless=True)
    df = ex.ohlcv("BTC/USDT:USDT", timeframe="4h", limit=1)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == pytest.approx(70050.0)


def test_funding_returns_concrete_fundinginfo_with_float_interval():
    ex = FuturesExchange(_FakeClient(), keyless=True)
    info = ex.funding("BTC/USDT:USDT")
    # the per-symbol-interval CONTRACT: exchange.funding() yields a real FundingInfo whose
    # interval_hours (float) is exactly what funding_intervals.funding_interval_hours consumes.
    assert isinstance(info, FundingInfo)
    assert isinstance(info.interval_hours, float)
    assert info.interval_hours == pytest.approx(8.0)
    assert info.current_rate == pytest.approx(0.0001)
    assert ex.mark_price("BTC/USDT:USDT") == pytest.approx(70050.0)


def test_funding_interval_hours_reads_exchange_funding_end_to_end():
    # END-TO-END: funding_intervals.funding_interval_hours pulls the interval straight off the
    # FundingInfo that THIS exchange.funding() produces (no stand-in) — spec §11 wiring proven.
    from futures_fund.funding_intervals import funding_interval_hours
    ex = FuturesExchange(_FakeClient(), keyless=True)
    assert funding_interval_hours("BTC/USDT:USDT", ex) == pytest.approx(8.0)


def test_depth_returns_ask_and_bid_levels():
    ex = FuturesExchange(_FakeClient(), keyless=True)
    book = ex.depth("BTC/USDT:USDT", limit=20)
    # asks (crossing side for a buy) ascending, bids (crossing side for a sell) descending
    assert book["asks"][0] == (70060.0, 1.2)
    assert book["bids"][0] == (70040.0, 1.5)


def test_depth_levels_are_price_qty_tuples():
    ex = FuturesExchange(_FakeClient(), keyless=True)
    book = ex.depth("BTC/USDT:USDT")
    for px, qty in book["asks"] + book["bids"]:
        assert isinstance(px, float) and isinstance(qty, float)
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_exchange.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.exchange'`.

- [ ] **Step 3: Write `futures_fund/exchange.py`** (weekly wrapper + new `depth`).

```python
from __future__ import annotations

import pandas as pd

from futures_fund.config import Settings
from futures_fund.market_data import (
    FundingInfo,
    _filter_field,
    parse_funding,
    parse_long_short_ratio,
    parse_ohlcv,
    parse_open_interest_history,
    parse_symbol_spec,
)
from futures_fund.models import MmrBracket, SymbolSpec


def build_ccxt(settings: Settings):
    """Construct a ccxt binanceusdm client (lazy import).

    Paper (settings.live False, default): a PUBLIC keyless mainnet client. Live: authenticated.
    """
    import ccxt

    config: dict = {"enableRateLimit": True}
    if settings.live:
        if not settings.exchange.api_key or not settings.exchange.api_secret:
            raise ValueError(
                "live=True requires BINANCE_KEY/BINANCE_SECRET; refusing to build a live client."
            )
        config["apiKey"] = settings.exchange.api_key
        config["secret"] = settings.exchange.api_secret
    return ccxt.binanceusdm(config)


def default_symbol_spec(market: dict) -> SymbolSpec:
    """Build a SymbolSpec from PUBLIC exchangeInfo only (no leverage tiers); one conservative
    MMR bracket (5% maintenance, 20x cap). Used in paper/keyless mode."""
    filters = (market.get("info") or {}).get("filters") or []
    tick = _filter_field(filters, "PRICE_FILTER", "tickSize")
    step = _filter_field(filters, "LOT_SIZE", "stepSize")
    mn = _filter_field(filters, "MIN_NOTIONAL", "notional")
    if tick is None:
        tick = float(market["precision"]["price"])
    if step is None:
        step = float(market["precision"]["amount"])
    if mn is None:
        mn = float((market.get("limits", {}).get("cost", {}) or {}).get("min") or 5.0)
    return SymbolSpec(
        symbol=market["id"], tick_size=float(tick), step_size=float(step), min_notional=float(mn),
        mmr_brackets=[MmrBracket(notional_floor=0.0, notional_cap=1e12, mmr=0.05,
                                 maint_amount=0.0, max_leverage=20.0)],
    )


class FuturesExchange:
    """Thin wrapper over a ccxt-like client. Inject a fake client in tests."""

    def __init__(self, client, keyless: bool = False):
        self.client = client
        self.keyless = keyless

    @classmethod
    def from_settings(cls, settings: Settings) -> FuturesExchange:
        ex = build_ccxt(settings)
        ex.load_markets()
        return cls(ex, keyless=not settings.live)

    def _raw_id(self, symbol: str) -> str:
        return self.client.market(symbol)["id"]

    def unified_for_raw(self, raw_id: str) -> str | None:
        by_id = getattr(self.client, "markets_by_id", None)
        if by_id and raw_id in by_id:
            m = by_id[raw_id]
            return (m[0] if isinstance(m, list) else m)["symbol"]
        for sym, mk in getattr(self.client, "markets", {}).items():
            if mk.get("id") == raw_id:
                return sym
        return None

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        market = self.client.market(symbol)
        if self.keyless:
            return default_symbol_spec(market)
        tiers = self.client.fetch_leverage_tiers([symbol])[symbol]
        return parse_symbol_spec(market, tiers)

    def ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 500) -> pd.DataFrame:
        return parse_ohlcv(self.client.fetch_ohlcv(symbol, timeframe, None, limit))

    def funding(self, symbol: str) -> FundingInfo:
        fr = self.client.fetch_funding_rate(symbol)
        try:
            interval = self.client.fetch_funding_interval(symbol)
        except Exception:
            interval = None
        return parse_funding(fr, interval)

    def open_interest_history(
        self, symbol: str, period: str = "4h", limit: int = 200
    ) -> pd.DataFrame:
        return parse_open_interest_history(
            self.client.fetch_open_interest_history(symbol, period, None, limit)
        )

    def long_short_ratio(self, symbol: str, period: str = "4h", limit: int = 200) -> pd.DataFrame:
        raw = self.client.fapiDataGetGlobalLongShortAccountRatio(
            {"symbol": self._raw_id(symbol), "period": period, "limit": limit}
        )
        return parse_long_short_ratio(raw)

    def mark_price(self, symbol: str) -> float:
        return float(self.client.fetch_funding_rate(symbol)["markPrice"])

    def depth(self, symbol: str, limit: int = 20) -> dict[str, list[tuple[float, float]]]:
        """L2 order-book snapshot for the depth-aware slippage model (spec §13).

        Returns {"bids": [(price, qty), ...] descending, "asks": [(price, qty), ...] ascending}.
        `asks` is the crossing side for a BUY, `bids` for a SELL; both are (price, qty) tuples
        suitable for costs.vwap_fill / slippage.depth_slippage.
        """
        book = self.client.fetch_order_book(symbol, limit)
        bids = [(float(p), float(q)) for p, q in (book.get("bids") or [])]
        asks = [(float(p), float(q)) for p, q in (book.get("asks") or [])]
        return {"bids": bids, "asks": asks}
```

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_exchange.py -q && uv run ruff check futures_fund/exchange.py tests/test_exchange.py
```

Expected: 7 tests pass; ruff clean.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: keyless FuturesExchange + L2 depth method (exchange.py)

Lift the keyless wrapper from weekly (klines/funding/mark/exchangeInfo) and add
depth(symbol) returning (price,qty) bid/ask levels for the slippage model. Tie
funding_intervals.funding_interval_hours to the concrete FundingInfo.interval_hours
this exchange.funding() returns (spec §11 per-symbol-interval wiring, end-to-end).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Config — `load_settings` with the new market-neutral blocks (`config.py`)

Extend `Settings`/`load_settings` to parse all the new `config.yaml` blocks from the contract (Part 3). This phase parses them into nested pydantic sub-models on `Settings`; `NeutralityConfig` itself is owned by `neutrality.py` in Phase 1, so here we parse the `neutrality:` block into a `dict` field that Phase 1 will hydrate (YAGNI — no premature model). Write `config.yaml` and the loader together.

**Superset-of-inherited-contract invariants (do NOT reduce).** Two reused surfaces are kept intact so `Settings` stays a strict superset of the inherited contract (API map "config.yaml layout" + weekly `_default_agent_models`; contract PART 3 "Extends the inherited layout (... agent_models ...)"):
1. **`agent_models: dict[str,str]`** is a first-class key resolved FIRST in `model_for` (per-agent wins, then the loop's `deep_model`, then the global). This phase has no agents yet, so the map defaults empty — but the resolution ORDER is preserved so a later `settings.model_for(role)` is not a breaking change.
2. **`loops:` two-cadence block** (weekly/daily) with the new `LoopSettings` fields (`cadence_days`, `cadence_hour_utc`, `regime_timeframe`, per-cadence `poll_minutes`) is parsed and round-trips through `_default_loops()`. These are net-new parsing surface (spec §9, contract PART 3) and are exercised TDD-first.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/config.yaml`
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/config.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_config.py`

- [ ] **Step 1: Write `config.yaml`** (contract Part 3, verbatim values).

```yaml
# --- account / capital ---
account_size_usdt: 20000
target_weekly: 0.05
max_drawdown_tolerance: 0.05
live: false

# --- two-cadence loops ---
loops:
  weekly:
    timeframe: "4h"
    regime_timeframe: "4h"
    poll_minutes: 1440
    deep_model: "opus"
    quick_model: "sonnet"
    cadence_days: 7
  daily:
    timeframe: "1h"
    poll_minutes: 60
    deep_model: "sonnet"
    quick_model: "haiku"
    cadence_hour_utc: 0

# --- per-agent model assignment (empty this phase; agents arrive in Phase 4) ---
# Kept as a first-class key so model_for() resolves per-agent FIRST (inherited contract);
# Phase 4 fills this in without a breaking change to the resolution order.
agent_models: {}

# --- neutrality / capital deployment -> NeutralityConfig (hydrated in Phase 1) ---
neutrality:
  capital_usdt: 20000
  target_gross_usdt: 20000
  side_budget_usdt: 10000
  deployment_floor: 0.90
  dry_powder_frac: 0.10
  per_name_cap: 0.25
  cluster_cap: 0.40
  dollar_band: 0.03
  beta_band: 0.05
  drift_band: 0.20
  turnover_penalty: 0.001
  corr_threshold: 0.7
  stress_band_mult: 0.5

# --- beta estimation ---
beta:
  lookback_days: 45
  btc_symbol: "BTC/USDT:USDT"

# --- alpha sleeves ---
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

# --- sentiment ---
sentiment:
  kappa: 0.5
  cap: 0.25
  halflife_days: 3
  refresh_daily: true

# --- universe ---
universe:
  symbol_count: 30
  min_adv_usd: 50000000
  crypto_only: true

# --- fees / funding / slippage realism ---
fees:
  taker_bps: 5.0
  maker_bps: 2.0
  pay_bnb: false
  bnb_discount: 0.90
funding:
  default_interval_hours: 8
  major_cap: 0.003
  alt_cap: 0.02
  majors: ["BTC/USDT:USDT", "ETH/USDT:USDT"]
  unclamped_in_rr: true
  signed_realized: true
slippage:
  model: "depth"
  k: 0.1
  half_spread_bps_default: 1.0
  depth_levels: 20
  flat_bps: null

# --- metrics / annualization ---
metrics:
  daily_periods_per_year: 365
  weekly_periods_per_year: 52
  benchmark_return: 0.0

# --- reviewer guardian ---
reviewer:
  enabled: true
  halt_on_mismatch: true
  model: "opus"
  tolerance: 1e-6

# --- graduation / overfit gate ---
graduation:
  dsr_threshold: 0.95
  min_cycles: 20
  walk_forward_required: true
```

- [ ] **Step 2: Write the failing test.** `tests/test_config.py`:

```python
from futures_fund.config import LoopSettings, Settings, _default_loops, load_settings


def test_load_settings_parses_account_and_live(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "account_size_usdt: 20000\n"
        "live: false\n"
        "max_drawdown_tolerance: 0.05\n"
    )
    s = load_settings(p)
    assert s.account_size_usdt == 20000.0
    assert s.live is False
    assert s.max_drawdown_tolerance == 0.05


def test_load_settings_parses_loops_two_cadence(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "loops:\n"
        "  weekly:\n"
        "    timeframe: \"4h\"\n"
        "    regime_timeframe: \"4h\"\n"
        "    poll_minutes: 1440\n"
        "    deep_model: \"opus\"\n"
        "    cadence_days: 7\n"
        "  daily:\n"
        "    timeframe: \"1h\"\n"
        "    poll_minutes: 60\n"
        "    deep_model: \"sonnet\"\n"
        "    cadence_hour_utc: 0\n"
    )
    s = load_settings(p)
    assert s.loops["weekly"].cadence_days == 7
    assert s.loops["weekly"].regime_timeframe == "4h"
    assert s.loops["weekly"].poll_minutes == 1440
    assert s.loops["daily"].cadence_hour_utc == 0
    assert s.loops["daily"].poll_minutes == 60
    # _default_loops() round-trips the same new fields when the block is absent
    dl = _default_loops()
    assert dl["weekly"].cadence_days == 7
    assert dl["daily"].cadence_hour_utc == 0
    assert isinstance(dl["weekly"], LoopSettings)


def test_model_for_resolves_agent_models_first_then_loop(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "agent_models:\n  sentiment: \"opus\"\n"
        "loops:\n  daily:\n    deep_model: \"sonnet\"\n    poll_minutes: 60\n"
    )
    s = load_settings(p)
    # per-agent map wins FIRST (inherited contract), regardless of loop tier
    assert s.model_for("sentiment", loop="daily") == "opus"
    # unknown role falls back to the loop's deep_model
    assert s.model_for("operational_narrator", loop="daily") == "sonnet"
    # no loop -> global deep_model default
    assert s.model_for("operational_narrator") == "opus"


def test_load_settings_parses_universe_block(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "universe:\n"
        "  symbol_count: 30\n"
        "  min_adv_usd: 50000000\n"
        "  crypto_only: true\n"
    )
    s = load_settings(p)
    assert s.universe.symbol_count == 30
    assert s.universe.min_adv_usd == 50_000_000.0
    assert s.universe.crypto_only is True


def test_load_settings_parses_fees_and_funding_and_slippage(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "fees:\n  taker_bps: 5.0\n  maker_bps: 2.0\n"
        "funding:\n  major_cap: 0.003\n  alt_cap: 0.02\n  unclamped_in_rr: true\n"
        "slippage:\n  model: depth\n  k: 0.1\n  half_spread_bps_default: 1.0\n"
    )
    s = load_settings(p)
    assert s.fees.taker_bps == 5.0
    assert s.fees.maker_bps == 2.0
    assert s.funding.major_cap == 0.003
    assert s.funding.alt_cap == 0.02
    assert s.funding.unclamped_in_rr is True
    assert s.slippage.k == 0.1
    assert s.slippage.flat_bps is None


def test_load_settings_parses_metrics_and_sentiment(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "metrics:\n  daily_periods_per_year: 365\n  weekly_periods_per_year: 52\n"
        "sentiment:\n  kappa: 0.5\n  cap: 0.25\n  halflife_days: 3\n"
    )
    s = load_settings(p)
    assert s.metrics.daily_periods_per_year == 365
    assert s.metrics.weekly_periods_per_year == 52
    assert s.sentiment.kappa == 0.5
    assert s.sentiment.cap == 0.25
    assert s.sentiment.halflife_days == 3


def test_neutrality_block_is_kept_as_raw_dict(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("neutrality:\n  side_budget_usdt: 10000\n  dollar_band: 0.03\n")
    s = load_settings(p)
    assert s.neutrality["side_budget_usdt"] == 10000
    assert s.neutrality["dollar_band"] == 0.03


def test_defaults_when_file_absent(tmp_path):
    s = load_settings(tmp_path / "nope.yaml")
    assert s.account_size_usdt == 20000.0
    assert s.universe.symbol_count == 30
    assert s.live is False
    assert s.agent_models == {}
    assert s.loops["weekly"].cadence_days == 7
```

- [ ] **Step 3: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_config.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.config'`.

- [ ] **Step 4: Write `futures_fund/config.py`** with the new sub-models. `neutrality` stays a raw dict (Phase 1 hydrates `NeutralityConfig`); `agent_models` + `loops` two-cadence are kept as first-class inherited surfaces.

```python
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ExchangeSettings(BaseModel):
    testnet: bool = True
    key_env: str = "BINANCE_KEY"
    secret_env: str = "BINANCE_SECRET"

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.key_env)

    @property
    def api_secret(self) -> str | None:
        return os.environ.get(self.secret_env)


class DataSettings(BaseModel):
    news_rss_sources: list[str] = Field(default_factory=lambda: [
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.cryptoslate.com/feed/",
        "https://bitcoinmagazine.com/feed",
        "https://cryptopotato.com/feed/",
    ])
    reddit_subreddits: list[str] = Field(
        default_factory=lambda: ["CryptoCurrency", "CryptoMarkets"])
    fred_key_env: str = "FRED_API_KEY"
    fred_series: list[str] = Field(
        default_factory=lambda: ["DTWEXBGS", "DGS10", "FEDFUNDS", "CPIAUCSL"]
    )
    archive_dir: str = "state/archive"

    @property
    def fred_api_key(self) -> str | None:
        return os.environ.get(self.fred_key_env)


class LoopSettings(BaseModel):
    """Per-cadence candle + model tier for the two-cadence desk (weekly select / daily rebalance)."""
    timeframe: str = "4h"
    regime_timeframe: str | None = None
    quick_model: str = "sonnet"
    deep_model: str = "opus"
    poll_minutes: int = 1440
    cadence_days: int | None = None
    cadence_hour_utc: int | None = None


def _default_loops() -> dict[str, LoopSettings]:
    return {
        "weekly": LoopSettings(timeframe="4h", regime_timeframe="4h", quick_model="sonnet",
                               deep_model="opus", poll_minutes=1440, cadence_days=7),
        "daily": LoopSettings(timeframe="1h", quick_model="haiku", deep_model="sonnet",
                              poll_minutes=60, cadence_hour_utc=0),
    }


class UniverseSettings(BaseModel):
    symbol_count: int = 30
    min_adv_usd: float = 50_000_000.0
    crypto_only: bool = True


class FeeSettings(BaseModel):
    taker_bps: float = 5.0
    maker_bps: float = 2.0
    pay_bnb: bool = False
    bnb_discount: float = 0.90


class FundingSettings(BaseModel):
    default_interval_hours: int = 8
    major_cap: float = 0.003
    alt_cap: float = 0.02
    majors: list[str] = Field(default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    unclamped_in_rr: bool = True
    signed_realized: bool = True


class SlippageSettings(BaseModel):
    model: str = "depth"
    k: float = 0.1
    half_spread_bps_default: float = 1.0
    depth_levels: int = 20
    flat_bps: float | None = None


class MetricsSettings(BaseModel):
    daily_periods_per_year: int = 365
    weekly_periods_per_year: int = 52
    benchmark_return: float = 0.0


class SentimentSettings(BaseModel):
    kappa: float = 0.5
    cap: float = 0.25
    halflife_days: float = 3.0
    refresh_daily: bool = True


class BetaSettings(BaseModel):
    lookback_days: int = 45
    btc_symbol: str = "BTC/USDT:USDT"


class Settings(BaseModel):
    account_size_usdt: float = 20_000.0
    timeframe: str = "4h"
    target_weekly: float = 0.05
    max_drawdown_tolerance: float = 0.05
    deep_model: str = "opus"   # global fallback tier for model_for (inherited contract)
    live: bool = False  # PAPER-ONLY desk: MUST stay false forever.
    loops: dict[str, LoopSettings] = Field(default_factory=_default_loops)
    # agent_models is a first-class inherited key resolved FIRST in model_for; empty this phase
    # (agents arrive in Phase 4) but the resolution ORDER is preserved (no breaking change later).
    agent_models: dict[str, str] = Field(default_factory=dict)
    # neutrality stays a raw dict here; Phase 1's neutrality.NeutralityConfig hydrates it.
    neutrality: dict = Field(default_factory=dict)
    beta: BetaSettings = Field(default_factory=BetaSettings)
    sleeves: dict = Field(default_factory=dict)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    fees: FeeSettings = Field(default_factory=FeeSettings)
    funding: FundingSettings = Field(default_factory=FundingSettings)
    slippage: SlippageSettings = Field(default_factory=SlippageSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    reviewer: dict = Field(default_factory=dict)
    graduation: dict = Field(default_factory=dict)
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    data: DataSettings = Field(default_factory=DataSettings)

    def model_for(self, role: str, *, loop: str | None = None) -> str:
        """Resolve the model an agent role is dispatched with (inherited contract resolution order):
        per-agent `agent_models` wins FIRST; else the loop's `deep_model`; else the global
        `deep_model`. agent_models is empty this phase, so a role resolves to the loop/global tier —
        but Phase 4 can populate it without changing this order (no breaking change)."""
        if role in self.agent_models:
            return self.agent_models[role]
        if loop and loop in self.loops:
            return self.loops[loop].deep_model
        return self.deep_model


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ WITHOUT overriding existing vars."""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k:
            continue
        loaded[k] = v
        os.environ.setdefault(k, v)
    return loaded


def load_settings(path: str | Path = "config.yaml") -> Settings:
    """Load non-secret config from YAML (defaults if file absent). Secrets come from env."""
    p = Path(path)
    load_env_file(p.parent / ".env")
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    return Settings(**(raw or {}))
```

> Note: `model_for` keeps the inherited resolution ORDER (per-agent `agent_models` → loop `deep_model` → global `deep_model`). `agent_models` defaults empty this phase because there are no agents yet, so every role currently resolves to the loop/global tier — but the key and its first-priority resolution are preserved, so Phase 4 populating the map is a pure extension, not a breaking change.

- [ ] **Step 5: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_config.py -q && uv run ruff check futures_fund/config.py tests/test_config.py
```

Expected: 8 tests pass; ruff clean.

- [ ] **Step 6: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: config.yaml + Settings with market-neutral blocks

Parse universe/fees/funding/slippage/metrics/sentiment/beta blocks into typed
sub-models; keep neutrality/sleeves/reviewer/graduation as raw dicts (Phase 1+).
Preserve the inherited agent_models-first model_for resolution and the two-cadence
loops block (cadence_days/cadence_hour_utc/regime_timeframe) — both tested.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Sentiment contracts (`contracts.py`)

NEW per contract §1.1. This phase needs only the sentiment slice (`SentimentSource`, `SentimentReport`, `SentimentBatch`) so `sentiment_ingest.py` can produce typed, range-validated, point-in-time reports. The geometry/pair/weights/reviewer contracts arrive in Phase 1 (YAGNI here).

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/contracts.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_contracts.py`

- [ ] **Step 1: Write the failing test.** `tests/test_contracts.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from futures_fund.contracts import SentimentBatch, SentimentReport, SentimentSource


def _utc(h=0):
    return datetime(2026, 6, 1, h, 0, tzinfo=timezone.utc)


def test_sentiment_report_accepts_valid_score():
    r = SentimentReport(symbol="BTC/USDT:USDT", level="positive", s=0.5,
                        confidence=0.8, as_of_ts=_utc(12))
    assert r.s == 0.5
    assert r.decayed_s is None
    assert r.sources == []


def test_sentiment_report_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        SentimentReport(symbol="BTC/USDT:USDT", level="positive", s=1.5,
                        confidence=0.8, as_of_ts=_utc(12))


def test_sentiment_report_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        SentimentReport(symbol="BTC/USDT:USDT", level="neutral", s=0.0,
                        confidence=1.2, as_of_ts=_utc(12))


def test_sentiment_source_carries_published_ts():
    src = SentimentSource(url="https://x/y", published_ts=_utc(6), title="t", feed="news_rss")
    r = SentimentReport(symbol="MARKET", level="neutral", s=0.0, confidence=0.3,
                        sources=[src], as_of_ts=_utc(12))
    assert r.sources[0].feed == "news_rss"
    assert r.sources[0].published_ts < r.as_of_ts


def test_sentiment_batch_holds_reports():
    batch = SentimentBatch(reports=[
        SentimentReport(symbol="BTC/USDT:USDT", level="very_positive", s=1.0,
                        confidence=0.9, as_of_ts=_utc(12)),
    ])
    assert len(batch.reports) == 1
    assert batch.reports[0].level == "very_positive"


def test_market_report_uses_market_symbol():
    r = SentimentReport(symbol="MARKET", level="negative", s=-0.5, confidence=0.6,
                        as_of_ts=_utc(12) + timedelta(hours=1))
    assert r.symbol == "MARKET" and r.s == -0.5
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_contracts.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.contracts'`.

- [ ] **Step 3: Write `futures_fund/contracts.py`** (sentiment slice verbatim from contract §1.1).

```python
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from futures_fund.models import SentimentLevel


class SentimentSource(BaseModel):
    url: str
    published_ts: datetime          # MUST be < owning report's as_of_ts (point-in-time)
    title: str = ""
    feed: str = ""                  # "news_rss" | "reddit" | "fear_greed" | "media"


class SentimentReport(BaseModel):
    symbol: str                     # ccxt unified id, or "MARKET" for the market-wide read
    level: SentimentLevel
    s: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SentimentSource] = Field(default_factory=list)
    rationale: str = ""
    as_of_ts: datetime              # decision-time anchor; all sources must precede this
    decayed_s: float | None = None  # s after half-life decay toward 0 (filled by ingest)


class SentimentBatch(BaseModel):
    reports: list[SentimentReport] = Field(default_factory=list)
```

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_contracts.py -q && uv run ruff check futures_fund/contracts.py tests/test_contracts.py
```

Expected: 6 tests pass; ruff clean.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: sentiment contracts (SentimentReport/Source/Batch)

Add the sentiment slice of contracts.py per the canonical contract: range-checked
score in [-1,1], confidence in [0,1], point-in-time sources, MARKET symbol read.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Market context + keyless feed vendors (`market_context.py`, `vendors.py`)

`sentiment_ingest.gather_sentiment_context` wraps `market_context.build_market_context`. Per spec §13/§7.1 ("reuse/extend `market_context.py`"), **both `market_context.py` and `vendors.py` are lifted VERBATIM** from `/home/roberto/crypto-trade-claude-code-weekly`. The richer reused `vendors.py` (typed `FearGreed`/`NewsItem`/`SocialPost` + `parse_fear_greed`/`parse_rss`/`parse_fred`/`fetch_fred_series`/`tag_instruments`/`archive_jsonl`) and the verbatim `build_market_context` (which calls `fetch_macro` WITHOUT a wrapping try/except because `fetch_macro` is internally fail-soft, and passes `symbols=settings.symbols` to `fetch_news`/`fetch_reddit`) are kept intact — downstream Phases depend on those exact symbols. `tests/test_market_context.py` is therefore written against the **REAL** reused models: `FearGreed` requires `ts`, `NewsItem` requires `published_at`+`kind`+`instruments`, and the degraded-context shapes match the verbatim assembler (`social == {"posts": [], "mentions": {}}`).

> The earlier draft of this plan invented a simplified `vendors.py` (e.g. `NewsItem(title,url,published_ts,source)`, `FearGreed(value,classification)` with no `ts`, `fetch_news(..., symbols=...)` keyword-only) and called it "lifted from weekly" — that contradicted the real module and would FAIL against a verbatim lift. This task lifts the real modules and tests them as-is.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/vendors.py`
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/market_context.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_market_context.py`

- [ ] **Step 1: Write the failing test.** `tests/test_market_context.py` — asserts the verbatim degraded-context shape AND constructs the REAL `FearGreed`/`NewsItem` models (with their required fields):

```python
from datetime import datetime, timezone

from futures_fund.config import Settings
from futures_fund.market_context import build_market_context
from futures_fund.vendors import FearGreed, NewsItem


class _FailHttp:
    """Fake http client whose every .get() raises — drives the per-feed degrade paths.
    (fetch_macro is internally fail-soft and short-circuits on a None key, so macro degrades
    to {} without the client being called.)"""
    def get(self, *args, **kwargs):
        raise RuntimeError("network disabled in tests")


def test_build_context_degrades_when_all_feeds_fail():
    s = Settings()
    ctx = build_market_context(_FailHttp(), s, fred_key=None)
    # every feed failed -> safe defaults + warnings, never raises
    assert ctx["fear_greed"] is None
    assert ctx["news"] == []
    assert ctx["macro"] == {}
    assert ctx["social"] == {"posts": [], "mentions": {}}
    assert any("Fear&Greed" in w for w in ctx["warnings"])
    assert any("news" in w for w in ctx["warnings"])
    assert any("FRED" in w for w in ctx["warnings"])
    assert "macro_labels" in ctx


def test_real_fear_greed_and_news_models_construct():
    # REAL reused models: FearGreed REQUIRES `ts`; NewsItem REQUIRES published_at+kind+instruments.
    fg = FearGreed(value=55, classification="Greed",
                   ts=datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert fg.value == 55 and fg.classification == "Greed"
    ni = NewsItem(title="t", url="u", published_at="2026-06-01T00:00:00Z",
                  source="coindesk", kind="news", instruments=["BTC"])
    assert ni.title == "t" and ni.source == "coindesk"
    assert ni.kind == "news" and ni.instruments == ["BTC"]
    # NewsItem.model_dump() carries the published_at key the context consumes
    assert ni.model_dump()["published_at"] == "2026-06-01T00:00:00Z"
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_market_context.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.market_context'`.

- [ ] **Step 3: Write `futures_fund/vendors.py`** — lifted **VERBATIM** from `/home/roberto/crypto-trade-claude-code-weekly/futures_fund/vendors.py` (typed item models with their real required fields + the keyless fetchers/parsers + `tag_instruments`/`archive_jsonl`, each fetcher fail-soft as in weekly).

```python
from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

FNG_URL = "https://api.alternative.me/fng/"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


class FearGreed(BaseModel):
    value: int
    classification: str
    ts: datetime


class NewsItem(BaseModel):
    title: str
    url: str
    published_at: str
    source: str
    kind: str
    instruments: list[str]
    summary: str = ""               # HTML-stripped article body/snippet (not just the title)
    votes_positive: int = 0
    votes_negative: int = 0


class SocialPost(BaseModel):
    """A reddit post the Sentiment analyst reads to gauge crowd CONTENT (not just an index number).
    `score` = net upvotes (the crowd's weight on the post); `summary` = the self-text snippet."""
    title: str
    summary: str = ""
    score: int = 0
    num_comments: int = 0
    source: str = ""                # the subreddit, e.g. 'CryptoCurrency'
    instruments: list[str] = []


def parse_fear_greed(payload: dict) -> FearGreed:
    d = payload["data"][0]
    return FearGreed(
        value=int(d["value"]),
        classification=d["value_classification"],
        ts=datetime.fromtimestamp(int(d["timestamp"]), tz=UTC),
    )


_ATOM = "{http://www.w3.org/2005/Atom}"
_ALIASES = {
    "BTC": ("btc", "bitcoin"), "ETH": ("eth", "ethereum"), "SOL": ("sol", "solana"),
    "BNB": ("bnb", "binance coin"), "XRP": ("xrp", "ripple"), "DOGE": ("doge", "dogecoin"),
    "ADA": ("ada", "cardano"), "AVAX": ("avax", "avalanche"),
}


def _base(symbol: str) -> str:
    # "BTC/USDT:USDT" -> "BTC"; "BTCUSDT" -> "BTC"
    s = symbol.split("/")[0]
    return s[:-4] if s.endswith("USDT") else s


def tag_instruments(title: str, symbols: list[str]) -> list[str]:
    """Which of `symbols` (bases or unified) a headline mentions, by ticker or full name."""
    t = title.lower()
    out: list[str] = []
    for sym in symbols:
        b = _base(sym)
        kws = (b.lower(),) + _ALIASES.get(b, ())
        if any(k in t for k in kws) and b not in out:
            out.append(b)
    return out


_CONTENT = "{http://purl.org/rss/1.0/modules/content/}"  # <content:encoded> full-body namespace
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_html(s: str | None, limit: int = 500) -> str:
    """Strip HTML tags, decode entities, collapse whitespace, truncate — turn an RSS body snippet
    into a plain-text summary the News analyst can read. Empty string on None."""
    if not s:
        return ""
    text = _WS_RE.sub(" ", _TAG_RE.sub(" ", html.unescape(s))).strip()
    return text[:limit].rstrip() + ("…" if len(text) > limit else "")


def _rss_text(el, tag: str) -> str | None:
    for cand in (tag, _ATOM + tag):
        e = el.find(cand)
        if e is not None:
            if e.text and e.text.strip():
                return e.text.strip()
            if e.get("href"):
                return e.get("href")
    return None


def parse_rss(content: bytes, source: str, symbols: list[str]) -> list[NewsItem]:
    """Parse an RSS/Atom feed (namespace-aware) into NewsItems. Returns [] on malformed XML."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    nodes = root.findall(".//item") or root.findall(f".//{_ATOM}entry")
    items: list[NewsItem] = []
    for n in nodes:
        title = _rss_text(n, "title")
        if not title:
            continue
        # Body: RSS <content:encoded> (full) or <description>; Atom <content>/<summary>. The body
        # often names coins the title doesn't, so tag instruments on title + body, and hand the
        # analyst the HTML-stripped snippet — not just the headline.
        raw_body = (_rss_text(n, _CONTENT + "encoded") or _rss_text(n, "encoded")
                    or _rss_text(n, "description") or _rss_text(n, "content")
                    or _rss_text(n, "summary"))
        summary = _clean_html(raw_body)
        items.append(NewsItem(
            title=title,
            url=_rss_text(n, "link") or "",
            published_at=_rss_text(n, "pubDate") or _rss_text(n, "published")
            or _rss_text(n, "updated") or "",
            source=source,
            kind="news",
            instruments=tag_instruments(f"{title} {summary}", symbols),
            summary=summary,
        ))
    return items


def fetch_news(
    client, sources: list[str], symbols: list[str], per_source: int = 10
) -> list[NewsItem]:
    """Fetch + parse multiple keyless RSS news feeds; skip any source that errors; dedupe by
    title."""
    seen: set[str] = set()
    out: list[NewsItem] = []
    for url in sources:
        try:
            r = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            src = url.split("//")[-1].split("/")[0]
            for item in parse_rss(r.content, source=src, symbols=symbols)[:per_source]:
                if item.title not in seen:
                    seen.add(item.title)
                    out.append(item)
        except Exception:
            continue  # graceful: a dead/blocked source must not break the cycle
    return out


_REDDIT_UA = "Mozilla/5.0 (TempestDesk research; keyless public-json read)"


def parse_reddit(payload: dict, subreddit: str, symbols: list[str]) -> list[SocialPost]:
    """Parse reddit's public listing JSON ({data:{children:[{data:{title,selftext,score,...}}]}})
    into SocialPosts, tagging instruments from title + self-text. Returns [] on any shape error."""
    try:
        children = (payload or {}).get("data", {}).get("children", [])
    except (AttributeError, TypeError):
        return []
    out: list[SocialPost] = []
    for ch in children:
        d = ch.get("data", {}) if isinstance(ch, dict) else {}
        title = (d.get("title") or "").strip()
        if not title:
            continue
        body = _clean_html(d.get("selftext") or "")
        out.append(SocialPost(
            title=title, summary=body,
            score=int(d.get("score") or 0), num_comments=int(d.get("num_comments") or 0),
            source=subreddit, instruments=tag_instruments(f"{title} {body}", symbols)))
    return out


def _posts_for_sub(client, sub: str, symbols: list[str], per_sub: int) -> list[SocialPost]:
    """One subreddit's posts. Tries /hot.json first (richer — carries upvote `score`), which reddit
    OFTEN 403s for keyless/datacenter reads; falls back to the /.rss Atom feed (works keyless but
    has no score). Returns [] if both fail."""
    try:
        r = client.get(f"https://www.reddit.com/r/{sub}/hot.json",
                       params={"limit": per_sub}, headers={"User-Agent": _REDDIT_UA})
        r.raise_for_status()
        posts = parse_reddit(r.json(), subreddit=sub, symbols=symbols)
        if posts:
            return posts[:per_sub]
    except Exception:
        pass
    try:
        r = client.get(f"https://www.reddit.com/r/{sub}/.rss", headers={"User-Agent": _REDDIT_UA})
        r.raise_for_status()
        return [SocialPost(title=i.title, summary=i.summary, source=sub, instruments=i.instruments)
                for i in parse_rss(r.content, source=sub, symbols=symbols)[:per_sub]]
    except Exception:
        return []


def fetch_reddit(client, subreddits: list[str], symbols: list[str], per_sub: int = 40) -> dict:
    """Keyless reddit social-sentiment scrape. Aggregates the top posts and a per-symbol mention
    count + score-weighted sum (the crowd's attention/weight per coin), so the Sentiment analyst
    reads real crowd CONTENT, not just a Fear&Greed number. Per sub it tries /hot.json then falls
    back to the /.rss Atom feed (reddit 403s the keyless JSON but serves the RSS). Graceful: a
    blocked sub is skipped; if all fail, returns {'posts': [], 'mentions': {}} and the desk caps
    conviction (the persona handles the degraded read)."""
    seen: set[str] = set()
    posts: list[SocialPost] = []
    for sub in subreddits:
        for p in _posts_for_sub(client, sub, symbols, per_sub):
            if p.title not in seen:
                seen.add(p.title)
                posts.append(p)
    posts.sort(key=lambda p: p.score, reverse=True)
    mentions: dict[str, dict] = {}
    for p in posts:
        for sym in p.instruments:
            m = mentions.setdefault(sym, {"count": 0, "score_sum": 0})
            m["count"] += 1
            m["score_sum"] += p.score
    return {"posts": [p.model_dump() for p in posts[:30]], "mentions": mentions}


def fetch_macro(client, series: list[str], api_key: str | None) -> dict[str, float]:
    """Latest value per FRED series (DXY/yields/Fed/CPI). Empty dict if no key (graceful)."""
    if not api_key:
        return {}
    out: dict[str, float] = {}
    for sid in series:
        try:
            r = client.get(FRED_URL, params={"series_id": sid, "api_key": api_key,
                                              "file_type": "json", "sort_order": "desc",
                                              "limit": 1})
            r.raise_for_status()
            # pick the latest observation by ISO date — order-independent (don't trust API order)
            vals = parse_fred(r.json())  # [(date, value)], skips "."
            if vals:
                out[sid] = max(vals, key=lambda dv: dv[0])[1]
        except Exception:
            continue
    return out


def parse_fred(payload: dict) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for o in payload.get("observations", []):
        if o["value"] == ".":  # FRED missing-value sentinel
            continue
        out.append((o["date"], float(o["value"])))
    return out


def fetch_fear_greed(client, limit: int = 1) -> FearGreed:
    r = client.get(FNG_URL, params={"limit": limit, "format": "json"})
    r.raise_for_status()
    return parse_fear_greed(r.json())


def fetch_fred_series(client, series_id: str, api_key: str, observation_start: str | None = None
                      ) -> list[tuple[str, float]]:
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "sort_order": "asc"}
    if observation_start:
        params["observation_start"] = observation_start
    r = client.get(FRED_URL, params=params)
    r.raise_for_status()
    return parse_fred(r.json())


def archive_jsonl(path, records: list[dict], key: str = "timestamp") -> int:
    """Append `records` to a JSONL file, deduping by `key` against existing rows.
    Returns the number of new rows written. Used to self-archive the 30-day-limited
    OI / long-short endpoints into durable history (spec §10)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line).get(key))
    written = 0
    with p.open("a") as f:
        for rec in records:
            k = rec.get(key)
            if k is not None and k in seen:
                continue
            f.write(json.dumps(rec, default=str) + "\n")
            if k is not None:
                seen.add(k)
            written += 1
    return written
```

- [ ] **Step 4: Write `futures_fund/market_context.py`** — lifted **VERBATIM** from weekly (calls `fetch_macro` WITHOUT a try/except because it is internally fail-soft; passes `symbols=settings.symbols` to `fetch_news`/`fetch_reddit`, which the real positional-`symbols` signatures accept by keyword).

```python
from __future__ import annotations

from futures_fund.config import Settings
from futures_fund.vendors import fetch_fear_greed, fetch_macro, fetch_news, fetch_reddit

_FRED_SERIES_LABELS = {"DTWEXBGS": "broad_dollar", "DGS10": "ust_10y",
                       "FEDFUNDS": "fed_funds", "CPIAUCSL": "cpi"}


def build_market_context(http_client, settings: Settings, fred_key: str | None) -> dict:
    """Assemble the market-wide context (news + Fear&Greed + macro) the news/sentiment/macro
    agents need. Each feed degrades independently: a failure omits it and records a warning so
    the agents cap conviction (mission §5)."""
    warnings: list[str] = []

    try:
        fg = fetch_fear_greed(http_client)
        fear_greed = {"value": fg.value, "classification": fg.classification}
    except Exception:
        fear_greed = None
        warnings.append("sentiment feed (Fear&Greed) unavailable — cap conviction")

    try:
        items = fetch_news(http_client, settings.data.news_rss_sources,
                           symbols=settings.symbols, per_source=10)
        news = [i.model_dump() for i in items]
        if not news:
            warnings.append("news feed returned no items — treat catalysts as unknown")
    except Exception:
        news = []
        warnings.append("news feed unavailable — cap conviction on catalysts")

    macro = fetch_macro(http_client, list(settings.data.fred_series), fred_key)
    if not macro:
        warnings.append("macro feed (FRED) unavailable — no DXY/yields/Fed read")

    # Reddit social-sentiment scrape (keyless): real crowd CONTENT per symbol for the Sentiment
    # analyst, beyond the single Fear&Greed number. Degrades to empty if reddit blocks the read.
    try:
        social = fetch_reddit(http_client, list(settings.data.reddit_subreddits),
                              symbols=settings.symbols)
        if not social.get("posts"):
            warnings.append("social feed (reddit) returned no posts — cap social-sentiment read")
    except Exception:
        social = {"posts": [], "mentions": {}}
        warnings.append("social feed (reddit) unavailable — cap social-sentiment read")

    return {"fear_greed": fear_greed, "news": news, "macro": macro, "social": social,
            "macro_labels": _FRED_SERIES_LABELS, "warnings": warnings}
```

- [ ] **Step 5: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_market_context.py -q && uv run ruff check futures_fund/vendors.py futures_fund/market_context.py tests/test_market_context.py
```

Expected: 2 tests pass; ruff clean.

> Why these pass against the verbatim lift: `fetch_news` and `fetch_reddit` wrap each source in `try/except` internally, so against a raising `_FailHttp` they return `[]` / `{"posts": [], "mentions": {}}` — `build_market_context` then records the news/social warnings. `fetch_fear_greed` raises (no internal guard for the whole call), caught by `build_market_context`'s try/except → `fear_greed=None` + warning. `fetch_macro` short-circuits to `{}` on `fred_key=None` (the client is never called), so `macro=={}` + the FRED warning. The `social == {"posts": [], "mentions": {}}` shape is exactly the verbatim assembler's degraded default.

- [ ] **Step 6: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: market context + keyless feed vendors (verbatim lift)

Lift vendors.py and build_market_context VERBATIM from weekly (spec §13/§7.1
reuse/extend market_context.py): typed FearGreed(value,classification,ts) /
NewsItem(...,published_at,kind,instruments) / SocialPost, the keyless
fetch_*/parse_* fetchers (each fail-soft), tag_instruments, archive_jsonl.
Tests assert the real models + the verbatim degraded-context shapes.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Sentiment ingestion source layer (`sentiment_ingest.py`)

NEW per contract §2.5. Point-in-time gather over `market_context`, the level↔s mapping, half-life decay, point-in-time validation, and fail-soft neutral. This is the source layer over `market_context.py` (spec §7.1/§7.3) — no LLM, just the deterministic plumbing the Sentiment Analyst's output is normalized and validated against.

**Files:**
- Create: `/home/roberto/crypto-trade-claude-code-market-neutral/futures_fund/sentiment_ingest.py`
- Test: `/home/roberto/crypto-trade-claude-code-market-neutral/tests/test_sentiment_ingest.py`

- [ ] **Step 1: Write the failing test.** `tests/test_sentiment_ingest.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from futures_fund.contracts import SentimentReport, SentimentSource
from futures_fund.sentiment_ingest import (
    LEVEL_TO_S,
    decay_report,
    decay_score,
    fail_soft_neutral,
    gather_sentiment_context,
    level_to_s,
    s_to_level,
    validate_point_in_time,
)


def _utc(h=0, d=1):
    return datetime(2026, 6, d, h, 0, tzinfo=timezone.utc)


def test_level_to_s_mapping_is_canonical():
    assert LEVEL_TO_S == {"very_positive": 1.0, "positive": 0.5, "neutral": 0.0,
                          "negative": -0.5, "very_negative": -1.0}
    assert level_to_s("very_positive") == 1.0
    assert level_to_s("negative") == -0.5


def test_s_to_level_round_trips_buckets():
    assert s_to_level(1.0) == "very_positive"
    assert s_to_level(0.5) == "positive"
    assert s_to_level(0.0) == "neutral"
    assert s_to_level(-0.5) == "negative"
    assert s_to_level(-1.0) == "very_negative"
    # round-trip for the reviewer's sentiment_range check
    for lvl, s in LEVEL_TO_S.items():
        assert s_to_level(s) == lvl


def test_decay_score_halves_after_one_halflife():
    # 3-day half-life -> 72h -> exactly halved
    assert decay_score(1.0, age_hours=72.0, half_life_days=3.0) == pytest.approx(0.5)
    assert decay_score(-0.8, age_hours=72.0, half_life_days=3.0) == pytest.approx(-0.4)
    # zero age -> unchanged
    assert decay_score(0.6, age_hours=0.0, half_life_days=3.0) == pytest.approx(0.6)


def test_decay_report_sets_decayed_s_from_age():
    r = SentimentReport(symbol="BTC/USDT:USDT", level="very_positive", s=1.0,
                        confidence=0.9, as_of_ts=_utc(0, d=1))
    out = decay_report(r, now=_utc(0, d=4), half_life_days=3.0)  # 72h later
    assert out.decayed_s == pytest.approx(0.5)
    assert out is not r  # returns a copy
    assert r.decayed_s is None  # original untouched


def test_validate_point_in_time_rejects_future_source():
    good = SentimentReport(symbol="BTC/USDT:USDT", level="neutral", s=0.0, confidence=0.3,
                           sources=[SentimentSource(url="u", published_ts=_utc(6, d=1))],
                           as_of_ts=_utc(12, d=1))
    bad = SentimentReport(symbol="BTC/USDT:USDT", level="neutral", s=0.0, confidence=0.3,
                          sources=[SentimentSource(url="u", published_ts=_utc(18, d=1))],
                          as_of_ts=_utc(12, d=1))
    assert validate_point_in_time(good) is True
    assert validate_point_in_time(bad) is False


def test_fail_soft_neutral_is_zeroed():
    r = fail_soft_neutral("SOL/USDT:USDT", now=_utc(12, d=1))
    assert r.level == "neutral" and r.s == 0.0 and r.confidence == 0.0
    assert r.symbol == "SOL/USDT:USDT" and r.as_of_ts == _utc(12, d=1)


def test_gather_sentiment_context_drops_future_sources():
    class _Http:
        def get(self, *a, **k):
            raise RuntimeError("network disabled")

    from futures_fund.config import Settings
    ctx = gather_sentiment_context(_Http(), Settings(), fred_key=None, as_of=_utc(12, d=1))
    # context degrades to safe defaults; the as_of anchor is recorded for downstream PIT checks
    assert ctx["as_of"] == _utc(12, d=1).isoformat()
    assert ctx["news"] == []  # no future/leaking sources slipped in
```

- [ ] **Step 2: Run the test — expect FAIL.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_sentiment_ingest.py -q
```

Expected: `ModuleNotFoundError: No module named 'futures_fund.sentiment_ingest'`.

- [ ] **Step 3: Write `futures_fund/sentiment_ingest.py`.** The point-in-time gather drops any news source whose `published_at` is at/after `as_of` (the real `NewsItem.model_dump()` exposes `published_at`).

```python
from __future__ import annotations

from datetime import datetime

from futures_fund.config import Settings
from futures_fund.contracts import SentimentReport
from futures_fund.market_context import build_market_context
from futures_fund.models import SentimentLevel

LEVEL_TO_S: dict[SentimentLevel, float] = {
    "very_positive": 1.0,
    "positive": 0.5,
    "neutral": 0.0,
    "negative": -0.5,
    "very_negative": -1.0,
}


def level_to_s(level: SentimentLevel) -> float:
    """Ordinal level -> numeric s in [-1,1] ({+2..-2}/2). Enforces the §7.1 mapping."""
    return LEVEL_TO_S[level]


def s_to_level(s: float) -> SentimentLevel:
    """Inverse bucketing (reviewer round-trips level<->s for the sentiment_range check)."""
    if s >= 0.75:
        return "very_positive"
    if s >= 0.25:
        return "positive"
    if s > -0.25:
        return "neutral"
    if s > -0.75:
        return "negative"
    return "very_negative"


def decay_score(s: float, age_hours: float, half_life_days: float = 3.0) -> float:
    """Exponential decay toward 0: s * 0.5**(age_hours/(half_life_days*24))."""
    if half_life_days <= 0:
        return s
    return s * (0.5 ** (age_hours / (half_life_days * 24.0)))


def decay_report(report: SentimentReport, now: datetime, half_life_days: float = 3.0
                 ) -> SentimentReport:
    """Return a copy with decayed_s set from (now - as_of_ts)."""
    age_hours = max(0.0, (now - report.as_of_ts).total_seconds() / 3600.0)
    decayed = decay_score(report.s, age_hours, half_life_days=half_life_days)
    return report.model_copy(update={"decayed_s": decayed})


def validate_point_in_time(report: SentimentReport) -> bool:
    """True iff every source.published_ts < report.as_of_ts (reviewer point-in-time check)."""
    return all(src.published_ts < report.as_of_ts for src in report.sources)


def fail_soft_neutral(symbol: str, now: datetime) -> SentimentReport:
    """Neutral report for missing/unparseable/stale sentiment. Never blocks the book."""
    return SentimentReport(symbol=symbol, level="neutral", s=0.0, confidence=0.0,
                           sources=[], rationale="fail-soft neutral", as_of_ts=now)


def gather_sentiment_context(http_client, settings: Settings, fred_key: str | None, *,
                             as_of: datetime) -> dict:
    """Point-in-time wrapper over market_context.build_market_context.

    Drops any news source whose published timestamp is at or after `as_of` (no post-decision
    leakage), and records the `as_of` anchor so downstream point-in-time checks can audit the
    gather. The real NewsItem.model_dump() carries `published_at`, which is the field checked.
    """
    ctx = build_market_context(http_client, settings, fred_key)
    cutoff_iso = as_of.isoformat()
    ctx["news"] = [n for n in ctx.get("news", [])
                   if not _is_future(n.get("published_at"), cutoff_iso)]
    ctx["as_of"] = cutoff_iso
    return ctx


def _is_future(published_at, cutoff_iso: str) -> bool:
    """True if a source's ISO timestamp is at/after the decision-time cutoff. Unparseable -> drop
    (treated as future) so an undated source never leaks past the point-in-time boundary."""
    if not published_at:
        return True
    try:
        return str(published_at) >= cutoff_iso
    except TypeError:
        return True
```

> Note: timestamps are compared as ISO strings — both `cutoff_iso` and the feed `published_at` are normalized ISO-8601 UTC, so lexicographic order matches chronological order; an undated/unparseable source is dropped (fail-closed against leakage), satisfying the §7.3 point-in-time guarantee. (The real `NewsItem.published_at` may be a raw RSS `pubDate` string; downstream normalization to ISO is a Phase-1 concern — here an unparseable value fails closed, never leaking.)

- [ ] **Step 4: Run the test — expect PASS.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest tests/test_sentiment_ingest.py -q && uv run ruff check futures_fund/sentiment_ingest.py tests/test_sentiment_ingest.py
```

Expected: 7 tests pass; ruff clean.

- [ ] **Step 5: Commit.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0: point-in-time sentiment ingestion source layer

sentiment_ingest.py over market_context.py: level<->s mapping, half-life decay,
point-in-time validation (drop future sources by published_at), fail-soft neutral.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Full-suite green + ruff + foundation tag

Final gate for the phase: the whole suite green, ruff clean across the package, and a commit marking the foundation complete.

**Files:** (none new — verification + commit)

- [ ] **Step 1: Run the full suite.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run pytest
```

Expected: **all tests green, zero failures, zero errors** across `test_models` (3), `test_costs` (10), `test_funding_intervals` (12), `test_slippage` (10), `test_metrics` (7), `test_market_data` (9), `test_exchange` (7), `test_config` (8), `test_contracts` (6), `test_market_context` (2), `test_sentiment_ingest` (7) — 81 tests total. (The gate is "all green"; the per-file counts are indicative, not a hard threshold.)

- [ ] **Step 2: Run ruff across the whole tree.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && uv run ruff check .
```

Expected: `All checks passed!`. If anything fails, run `uv run ruff check --fix .`, re-run pytest, and only then proceed.

- [ ] **Step 3: Verify the data+costs foundation has NO trading logic.** Confirm no optimizer/sleeve/risk-gate modules were created in this phase (those are Phases 1-5):

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && ls futures_fund/
```

Expected: exactly `__init__.py  config.py  contracts.py  costs.py  exchange.py  funding_intervals.py  market_context.py  market_data.py  metrics.py  models.py  sentiment_ingest.py  slippage.py  vendors.py` — and no `neutrality.py`, `sleeves/`, `reviewer.py`, `risk_gate.py`, or `control_loop.py` (deferred by design).

- [ ] **Step 4: Final commit marking the foundation complete.**

```bash
cd /home/roberto/crypto-trade-claude-code-market-neutral && git add -A && git commit -m "$(cat <<'EOF'
Phase 0 complete: testable data + costs foundation (no trading logic)

Scaffold, crypto-only liquid universe, keyless Binance data layer with depth,
per-symbol signed funding, depth-aware slippage, fixed Sharpe periodicity,
and point-in-time sentiment ingestion — all TDD with Binance-shaped fixtures.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase exit criteria

- `uv run pytest` is fully green; `uv run ruff check .` prints `All checks passed!`.
- `config.yaml` parses into `Settings` with every market-neutral block (universe/fees/funding/slippage/metrics/sentiment/beta typed; neutrality/sleeves/reviewer/graduation as raw dicts for Phase 1+), the two-cadence `loops` block (cadence_days/cadence_hour_utc/regime_timeframe), and the inherited `agent_models`-first `model_for` resolution.
- The data layer (`exchange.py`) exposes keyless klines/funding/mark/exchangeInfo and the new `depth` method; `exchange.funding()` returns a concrete `FundingInfo` whose `interval_hours` is exactly what `funding_intervals.funding_interval_hours` consumes (per-symbol-interval wiring proven end-to-end); the universe is crypto-only and liquidity-floored to ~top 20-30.
- Realism primitives are correct and signed: per-symbol funding intervals + sign-preserving caps (BTC/ETH ±0.30%, alts ±2%), signed realized funding (short receives positive funding) consuming the clamped rate per the §11 clamp→realized ordering, depth-aware slippage (never flat 2 bps; §11 $1M BTC anchor pinned, fallback monotone in notional), 5 bps taker / 2 bps maker fees, and Sharpe annualized ×365 daily / ×52 weekly.
- `market_context.py` and `vendors.py` are lifted verbatim from weekly (real `FearGreed`/`NewsItem`/`SocialPost` models + keyless fail-soft fetchers); sentiment ingestion enforces point-in-time discipline (future sources dropped by `published_at`), level↔s mapping, half-life decay, and fail-soft neutral.
- **No trading logic, optimizer, sleeves, risk gate, reviewer, or control loop exists yet** — those are Phases 1-7. This phase delivers only the testable data + costs foundation.

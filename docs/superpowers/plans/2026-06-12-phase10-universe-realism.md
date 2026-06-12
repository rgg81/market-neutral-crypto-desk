# Phase 10 — Liquid + Established Universe, Honest Slippage, Bounded Carry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the market-neutral PAPER desk trade only liquid+established crypto names, cost fills from the real order book (not a flat 1bps), and stop the carry sleeve from chasing extreme funding as if it were free alpha.

**Architecture:** Three additive layers over the existing Phase 8/9 pipeline. (1) A new `quality_filter` in `market_data.py` (age + 24h-mover + depth + the existing ADV floor) that the scout applies before writing a CLEAN `universe.json`, carrying per-symbol metadata (`vol_24h_usd`, `chg_24h_pct`, `onboard_date`) through to `cycle_prep`. (2) Depth-aware slippage fed end-to-end: `build_geometries` stamps real `adv_usd` and a per-symbol `depth` snapshot onto each `CoinGeometry`, `run_paper_cli._geometry_cost_maps` threads them into `CostInputs`, and `apply_fills` selects the crossing side by trade sign — the `estimate_slippage → depth_slippage → vwap_fill` chain downstream is already complete. (3) A configurable, opt-in strategy-level funding bound, factored into a NEUTRAL shared helper in `funding_intervals.py`, that both `sleeves/carry.py` and the `sleeves/factor.py` carry leg import to clamp/exclude extreme funding so a blow-off name is not maximally attractive.

**Tech Stack:** Python 3.11+, pydantic v2 models (`futures_fund/contracts.py`, `config.py`), pandas (OHLCV), ccxt `binanceusdm` (keyless; faked in tests), pytest, ruff (`select = E,F,I,UP,B`, line-length 100), `uv run` for commands.

---

## Notes for the implementer (read once before Task 1)

- **Git safety:** stay on branch `master` the entire time. NEVER `git checkout`, `git switch`, `git reset`, or create branches. Commit in place after each task's tests are green.
- **Commit trailer:** every commit message ends with a blank line then:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
- **Run tests with:** `uv run pytest ...`. Run ruff with: `uv run ruff check futures_fund scripts tests`. Keep the FULL suite green: `uv run pytest -q` after the last task.
- **Real signatures already in the repo (do NOT redefine — import/extend them):**
  - `market_data.scan_universe(client, top_n=30) -> list[dict]` returns rows `{symbol, last, chg_24h_pct, vol_24h_usd}`; it builds `market = {**(markets.get(sym) or {}), "symbol": sym}` per symbol where `markets = getattr(client, "markets", None) or {}` — so `market["info"]["onboardDate"]` is available there (ms-epoch **string**) when the fake/real client market carries it. Row-append block is `futures_fund/market_data.py:122-127`.
  - `market_data.liquidity_floor(rows, *, min_adv_usd, symbol_count) -> list[dict]` (`market_data.py:132`).
  - `market_data.is_crypto_perp(market)`, `market_data._base_symbol(market)`.
  - `exchange.FuturesExchange(client, keyless=False)` (constructor: `exchange.py:59`) and `exchange.FuturesExchange.from_settings(settings)` (classmethod, `exchange.py:64`, builds a keyless ccxt client + `load_markets()`). `FuturesExchange.depth(symbol, limit=20) -> {"bids": [(price,qty)...desc], "asks": [(price,qty)...asc]}` (float tuples; `exchange.py:106`). **It DOES have callers today:** `tests/test_exchange.py::test_depth_returns_ask_and_bid_levels` and `::test_depth_levels_are_price_qty_tuples` already exercise it. `FuturesExchange.ohlcv(symbol, timeframe="4h", limit=500) -> DataFrame` (`exchange.py:79`).
  - `slippage.estimate_slippage(symbol, qty, reference_price, *, depth, adv_usd, half_spread_bps, k=0.1)` — branches to `depth_slippage` iff `depth` is truthy, else `fallback_slippage`. `depth_slippage`→`costs.slippage_cost`→`costs.vwap_fill` prices on the FILLED qty only (see Task 8's modeling caveat).
  - `costs.slippage_cost(levels, qty, reference_price)` / `costs.vwap_fill(levels, qty)`; `costs.trade_fee(notional, *, maker)`.
  - `account.CostInputs(adv_usd=0.0, half_spread_bps=1.0, depth=None, maker=False)` (`account.py:73`) and `PaperAccount.apply_fills(executed_trades, marks, costs, *, opened_ts, opened_cycle, opened_cadence)` — computes `delta_signed_qty` (`account.py:211`), calls `estimate_slippage(..., depth=ci.depth, ...)` at `account.py:216-218`. `account.py` already imports `Field` (line 32) and `trade_fee` (line 34).
  - `contracts.CoinGeometry` fields incl. `funding_rate, funding_apr, funding_cap (=0.02, unused), adv_usd (=0.0), market_info (=None)`; the `# liquidity / filters` block is `contracts.py:119-125`. `contracts.py` imports `Field` (line 7).
  - `sleeves.carry.carry_signal(geometries, *, risk_budget_frac, now, top_frac=1/3)` ranks ascending on `g.funding_apr` (`carry.py:15-42`).
  - `sleeves.factor._factor_score(g, factor)` returns `-g.funding_apr` for `factor == "carry"` (`factor.py:13-20`); `rank_factor` (`factor.py:23-29`); `_combined_rank` (`factor.py:32-40`); `factor_signal` (`factor.py:55-78`).
  - `cycle_prep.build_geometries(exchange, symbols, *, now, btc_symbol, beta_lookback)` (signature `cycle_prep.py:66-73`; the `geometries.append(CoinGeometry(...))` block is `cycle_prep.py:94-105`) and `build_sleeves(geometries, pairs, spreads, *, now)` (`cycle_prep.py:109-128`).
  - `funding_intervals.clamp_funding_rate(symbol, rate)` (majors ±0.003, alts ±0.02), `funding_apr(rate, interval_hours)` (8h → ×1095). This module is the NEUTRAL home for the new `bounded_apr` helper (Task 10) so both sleeves import from it (no factor→carry coupling).
  - `config.Settings` with `UniverseSettings`, `SlippageSettings`, `sleeves: dict` (free-form, `config.py:128`). `config.load_settings(path=None)` reads `config.yaml`. `UniverseSettings` is `config.py:66-69`.
  - `scripts/scout_cli.py` `main` (`scout_cli.py:22-44`) currently builds `build_ccxt(settings)`, `client.load_markets()`, `scan_universe`, `liquidity_floor`, `save_output`, `print(json.dumps({"universe": ...}))`. NO `FuturesExchange` import yet.
  - `scripts/cycle_prep_cli.py` `_symbols` (`cycle_prep_cli.py:39-48`) and `main` (`cycle_prep_cli.py:51-80`); line 62 builds `ex = FuturesExchange.from_settings(settings)`, line 63 `symbols = _symbols(...)`, lines 65-68 the `build_geometries(...)` call, line 69 `build_pairs_and_spreads(ex, symbols, ...)`, line 70 `build_sleeves(...)`.
  - `scripts/run_paper_cli.py._geometry_cost_maps(bundle)` (`run_paper_cli.py:290-308`); the per-geometry cost build is `costs[sym] = CostInputs(adv_usd=float(g.get("adv_usd", 0.0)))` at line 307. `run_paper_cli` imports `CostInputs` (line 36).
- **APR scale reference (load-bearing for the carry cap default):** at the 8h interval, `funding_apr = rate × 1095`. An alt realized rate clamped to ±0.02 → ±21.9 APR; a major ±0.003 → ±3.285 APR. So a strategy cap of `±2.0` APR materially bounds the sleeve BELOW the per-symbol realized clamp, while leaving normal carry (rate ~0.0005/8h → ~0.55 APR) untouched.
- **Depth-floor semantics (load-bearing — read before Task 1/3):** the depth floor compares the FULL summed dollar notional of the top-N book levels on the THINNER side against `min_depth_usd`. `depth_ref_usd` is config DOCUMENTATION / a reference clip the slippage model is conceptually measured at — it is **NOT** a cap inside the floor function. A deep book ($500k/$1M/$5M per level) MUST clear `min_depth_usd=250_000`; a $1k/side thin book MUST fail it. (The original draft capped the summed notional at `depth_ref_usd=100k < min_depth_usd=250k`, which made the floor unsatisfiable and emptied the universe — that bug is removed here.)
- **Fail-soft is mandatory** in `build_geometries` and `quality_filter`: a missing capability (no `.depth`, a raising `.depth`, a missing `onboardDate`) degrades to a sane default (empty book / kline-age fallback / counted-but-kept), never a crash and never a silent drop.

---

## File Structure

**Modify:**
- `futures_fund/config.py` — extend `UniverseSettings` (age/mover/depth knobs). `Settings.sleeves` stays a free dict (carry cap rides in it via YAML; no schema change).
- `config.yaml` — mirror the new `universe.*` keys and add a `sleeves.carry.*` sub-block (sibling of `factor:`/`pairs:`).
- `futures_fund/market_data.py` — carry `onboard_date` into `scan_universe` rows; add `quality_filter(...)`.
- `futures_fund/contracts.py` — add `chg_24h_pct`, `onboard_date`, `depth_bids`, `depth_asks` fields to `CoinGeometry`.
- `futures_fund/exchange.py` — add `onboard_date_ms(symbol)` accessor.
- `futures_fund/funding_intervals.py` — add the neutral `bounded_apr(apr, cap)` helper.
- `futures_fund/cycle_prep.py` — `build_geometries` stamps `adv_usd`, depth, `onboard_date`, `chg_24h_pct`; accept a `universe_rows` map; thread the carry cap into `build_sleeves`.
- `futures_fund/sleeves/carry.py` — `carry_signal(..., max_abs_apr=None)` bound (imports `bounded_apr` from `funding_intervals`).
- `futures_fund/sleeves/factor.py` — mirror the carry bound in `_factor_score`/`rank_factor`/`_combined_rank`/`factor_signal` (imports `bounded_apr` from `funding_intervals`).
- `scripts/scout_cli.py` — call `quality_filter`, log per-filter drop counts.
- `scripts/cycle_prep_cli.py` — pass universe rows (with metadata) to `build_geometries`; thread carry cap.
- `scripts/run_paper_cli.py` — `_geometry_cost_maps` populates `CostInputs.depth_bids/depth_asks` + `half_spread_bps`.
- `futures_fund/account.py` — `apply_fills` selects `depth` side from `delta_signed_qty` sign.
- `tests/test_config.py` — APPEND quality-knob + carry-cap + YAML-mirror tests (add `Settings` to the existing import).
- `tests/test_market_data.py` — APPEND the onboard-date row test.
- `tests/test_scout_cli.py` — APPEND fakes + the quality-exclusion test; patch `FuturesExchange` in the original test.
- `tests/test_cycle_prep.py` — APPEND geometry-stamp + build_sleeves-cap tests.
- `tests/test_exchange.py` — APPEND the `onboard_date_ms` tests (file EXISTS with depth tests; `FuturesExchange` already imported line 3).
- `tests/test_funding_intervals.py` — APPEND the `bounded_apr` test.
- `tests/sleeves/test_carry.py` — APPEND the bound tests.
- `tests/sleeves/test_factor.py` — APPEND the bound tests.
- `tests/test_end_to_end_no_seed.py` — extend fakes (`onboardDate`, `.depth()`); keep suite green.

**Create:**
- `tests/test_quality_filter.py`
- `tests/test_depth_aware_slippage.py`
- `tests/test_universe_integration.py`

---

### Task 1: Config knobs for the quality filter + carry bound

**Files:**
- Modify: `futures_fund/config.py:66-69` (`UniverseSettings`). `Settings.sleeves` is unchanged (free dict).
- Modify: `config.yaml` — the `universe` block (lines 74-78) and add a `carry:` sub-block under `sleeves:` (sibling of `factor:`/`pairs:`).
- Test: `tests/test_config.py` (APPEND; the file ALREADY EXISTS, 8183 bytes).

- [ ] **Step 1: Write the failing test**

The file `tests/test_config.py` already exists and its top import is:
```python
from futures_fund.config import (
    DataSettings,
    ExchangeSettings,
    LoopSettings,
    _default_loops,
    load_env_file,
    load_settings,
)
```
It does NOT import `Settings`. FIRST edit that import block to add `Settings` (alphabetical position, between `LoopSettings` and `_default_loops`):

```python
from futures_fund.config import (
    DataSettings,
    ExchangeSettings,
    LoopSettings,
    Settings,
    _default_loops,
    load_env_file,
    load_settings,
)
```

THEN append these tests to the end of `tests/test_config.py`:

```python
def test_universe_settings_have_quality_knobs():
    u = Settings().universe
    assert u.min_age_days == 30
    assert u.max_abs_chg_24h_pct == 25.0
    assert u.min_depth_usd == 250_000.0
    assert u.depth_ref_usd == 100_000.0


def test_universe_quality_knobs_load_from_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "universe:\n"
        "  symbol_count: 30\n"
        "  min_adv_usd: 50000000\n"
        "  min_age_days: 45\n"
        "  max_abs_chg_24h_pct: 20\n"
        "  min_depth_usd: 300000\n"
        "  depth_ref_usd: 120000\n"
    )
    s = load_settings(cfg)
    assert s.universe.min_age_days == 45
    assert s.universe.max_abs_chg_24h_pct == 20.0
    assert s.universe.min_depth_usd == 300_000.0
    assert s.universe.depth_ref_usd == 120_000.0


def test_carry_funding_cap_default_none_and_loads():
    # default Settings() has an EMPTY sleeves dict -> no strategy cap (opt-in), so existing carry
    # behavior is unchanged.
    assert Settings().sleeves.get("carry", {}).get("max_abs_apr") is None


def test_repo_config_yaml_carry_cap_is_nested_correctly():
    # GUARD a YAML indent mistake: the carry block MUST be a sibling of factor:/pairs: under
    # sleeves:, not a child of factor:. Reads the REPO config.yaml (not a tmp fixture).
    s = load_settings("config.yaml")
    assert s.sleeves["carry"]["max_abs_apr"] == 2.0
    # factor: must still be its own sub-block (carry did not get nested inside it)
    assert "carry" not in s.sleeves.get("factor", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `test_universe_settings_have_quality_knobs` raises `AttributeError: 'UniverseSettings' object has no attribute 'min_age_days'`; `test_repo_config_yaml_carry_cap_is_nested_correctly` raises `KeyError: 'carry'`. (The `Settings` import now resolves — no `NameError`.)

- [ ] **Step 3: Write minimal implementation**

In `futures_fund/config.py` replace the `UniverseSettings` class (currently lines 66-69):

```python
class UniverseSettings(BaseModel):
    symbol_count: int = 30
    min_adv_usd: float = 50_000_000.0
    crypto_only: bool = True
    # Phase 10 quality filter (liquid + established only)
    min_age_days: int = 30                 # exclude names listed < this many days ago
    max_abs_chg_24h_pct: float = 25.0      # exclude extreme 24h movers (|chg| > this)
    min_depth_usd: float = 250_000.0       # floor on FULL top-of-book notional (thinner side)
    depth_ref_usd: float = 100_000.0       # reference clip for the slippage model (NOT a floor cap)
```

`Settings.sleeves` is already a free-form `dict` (line 128), so `sleeves.carry.max_abs_apr` flows through with no schema change — no code edit beyond the YAML in Step 4.

- [ ] **Step 4: Add the YAML mirror**

Edit `config.yaml`. Replace the `universe` block (lines 74-78), which currently reads exactly:
```yaml
# --- universe ---
universe:
  symbol_count: 30
  min_adv_usd: 50000000
  crypto_only: true
```
with:
```yaml
# --- universe (liquid + established only, Phase 10) ---
universe:
  symbol_count: 30
  min_adv_usd: 50000000
  crypto_only: true
  min_age_days: 30            # exclude symbols listed < 30 days ago (reversal/illiquidity risk)
  max_abs_chg_24h_pct: 25     # exclude extreme 24h movers (a +130% pump is not a tradeable edge)
  min_depth_usd: 250000       # floor on the FULL top-of-book notional on the thinner side
  depth_ref_usd: 100000       # reference clip the slippage model is measured at (not a floor cap)
```

Then add a `carry:` sub-block under the EXISTING `sleeves:` block. EXACT insertion point: the `sleeves:` block currently ends with the `pairs:` sub-block (`rolling_retest_cycles: 7`) before the blank line preceding `# --- sentiment ---`. Insert the `carry:` block (2-space indent so it is a SIBLING of `factor:`/`pairs:`, a CHILD of `sleeves:`) immediately AFTER `    rolling_retest_cycles: 7` and BEFORE the blank line + `# --- sentiment ---` comment. The result around that region must read:

```yaml
  pairs:
    adf_pvalue_max: 0.05
    fdr_method: "bh"
    entry_z: 2.0
    exit_z: 0.0
    stop_z: 3.0
    min_half_life_cycles: 1.0
    max_half_life_cycles: 40.0
    rolling_retest_cycles: 7
  carry:
    # Extreme funding is a REVERSAL TRAP, not free alpha: clamp the |funding_apr| the carry sleeve
    # ranks/sizes on so a blow-off name is treated as capped, not maximally attractive. null = no
    # strategy cap (per-symbol realized clamp still applies upstream). 2.0 APR ~= a 0.0018/8h rate.
    max_abs_apr: 2.0

# --- sentiment ---
```

`carry:` is indented 2 spaces (same column as `risk_parity:`, `enabled:`, `factor:`, `pairs:`), and `max_abs_apr:` is indented 4 spaces. Do NOT place it under `factor:`'s `weighting: "inverse_vol"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all new tests PASS — `test_universe_settings_have_quality_knobs`, `test_universe_quality_knobs_load_from_yaml`, `test_carry_funding_cap_default_none_and_loads` (default `Settings().sleeves == {}`), and `test_repo_config_yaml_carry_cap_is_nested_correctly` (proves the YAML indent is correct). All pre-existing config tests still PASS.

Run: `uv run ruff check futures_fund/config.py`
Expected: PASS (no output).

- [ ] **Step 6: Commit**

```bash
git add futures_fund/config.py config.yaml tests/test_config.py
git commit -m "feat(config): add universe quality + carry funding-cap knobs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Carry `onboard_date` into `scan_universe` rows

**Files:**
- Modify: `futures_fund/market_data.py:107-130` (`scan_universe`).
- Test: `tests/test_market_data.py` (append).

The row dict must additionally carry `onboard_date` (ms-epoch int or `None`) so the downstream age filter and `cycle_prep` have it without a second fetch. Pull it from the per-symbol `market["info"]` already constructed in the loop (`market = {**(markets.get(sym) or {}), "symbol": sym}`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_market_data.py`:

```python
from futures_fund.market_data import scan_universe


class _FakeOnboardClient:
    markets = {
        "BTC/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "1567965300000"}},
        "NEW/USDT:USDT": {"info": {"underlyingType": "COIN"}},  # no onboardDate -> None
    }

    def fetch_tickers(self):
        return {
            "BTC/USDT:USDT": {"last": 60000.0, "quoteVolume": 2e9, "percentage": 1.0},
            "NEW/USDT:USDT": {"last": 1.0, "quoteVolume": 1e9, "percentage": 130.0},
        }


def test_scan_universe_carries_onboard_date_ms_int_or_none():
    rows = scan_universe(_FakeOnboardClient(), top_n=10)
    by_sym = {r["symbol"]: r for r in rows}
    assert by_sym["BTC/USDT:USDT"]["onboard_date"] == 1567965300000
    assert by_sym["NEW/USDT:USDT"]["onboard_date"] is None
    # existing fields unchanged
    assert by_sym["NEW/USDT:USDT"]["chg_24h_pct"] == 130.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_market_data.py::test_scan_universe_carries_onboard_date_ms_int_or_none -v`
Expected: FAIL — `KeyError: 'onboard_date'`.

- [ ] **Step 3: Write minimal implementation**

In `futures_fund/market_data.py`, inside `scan_universe`'s loop, replace the row-append block (lines 122-127, which currently reads `qv = t.get("quoteVolume") or 0.0` … `"vol_24h_usd": float(qv)})`):

```python
        qv = t.get("quoteVolume") or 0.0
        last = t.get("last")
        if qv and last:
            raw_onboard = (market.get("info") or {}).get("onboardDate")
            try:
                onboard_ms = int(raw_onboard) if raw_onboard is not None else None
            except (TypeError, ValueError):
                onboard_ms = None
            rows.append({"symbol": sym, "last": float(last),
                         "chg_24h_pct": round(float(t.get("percentage") or 0.0), 2),
                         "vol_24h_usd": float(qv), "onboard_date": onboard_ms})
```

(`market` is already in scope in the loop as `market = {**(markets.get(sym) or {}), "symbol": sym}`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_market_data.py::test_scan_universe_carries_onboard_date_ms_int_or_none -v`
Expected: PASS.

Run: `uv run pytest tests/test_market_data.py tests/test_scout_cli.py -q`
Expected: PASS (existing tests unaffected — `onboard_date` is an additive key).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/market_data.py tests/test_market_data.py
git commit -m "feat(market_data): carry onboardDate (ms int|None) into scan_universe rows

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `quality_filter` — age + 24h-mover + depth + ADV floor (no silent truncation)

**Files:**
- Modify: `futures_fund/market_data.py` (add `quality_filter` after `liquidity_floor`, after line 137).
- Test: `tests/test_quality_filter.py` (create).

`quality_filter` runs AFTER `scan_universe`. It takes the rows, a `now`, an `exchange` that exposes `.depth(symbol)` and `.ohlcv(symbol)` (the live `FuturesExchange`; a fake in tests), and the thresholds. It applies, in order, four gates and returns `(kept_rows, drop_counts)` where `drop_counts` is a dict so the scout can log how many each filter removed.

**Depth floor (CORRECTED):** sum the FULL dollar value of all top-N book levels on the THINNER side (min of bid/ask) and require `>= min_depth_usd`. There is NO cap at `depth_ref_usd` — `depth_ref_usd` is config documentation for the slippage model, not part of the floor (so a deep $500k/$1M/$5M book clears `min_depth_usd=250k`, and a $1k thin book fails it). Missing depth (exchange has no `.depth`, it raises, or returns an empty/None book) → keep the name but record it under `"depth_unavailable"` (sane fallback: never silently drop a name just because depth could not be fetched).

**Age fallback:** if `onboard_date` is `None`, derive age from the OHLCV via `exchange.ohlcv(symbol)` — `now - earliest_kline_timestamp` is the listing-age proxy; a name whose earliest candle is more recent than `min_age_days` ago is treated as too young and dropped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_quality_filter.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from futures_fund.market_data import quality_filter

_NOW = datetime(2026, 6, 12, tzinfo=UTC)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class _FakeDepthExchange:
    """Deep books for the established names, a thin book for VELVET."""

    _DEEP = {"bids": [(100.0, 5000.0)], "asks": [(100.0, 5000.0)]}      # ~$500k/side
    _THIN = {"bids": [(1.0, 1000.0)], "asks": [(1.0, 1000.0)]}          # ~$1k/side

    def depth(self, symbol, limit=20):
        return self._THIN if symbol.startswith("VELVET") else self._DEEP

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        # default fallback frame: earliest candle is OLD (>= min_age_days) for any name that
        # reaches the kline fallback. Tests needing a YOUNG fallback subclass and override this.
        ts = pd.date_range("2025-01-01", periods=200, freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0,
                             "low": 1.0, "close": 1.0, "volume": 1.0})


def _rows():
    old = _ms(_NOW - timedelta(days=900))
    young = _ms(_NOW - timedelta(days=5))
    return [
        {"symbol": "BTC/USDT:USDT", "last": 60000.0, "chg_24h_pct": 1.0,
         "vol_24h_usd": 2e9, "onboard_date": old},
        {"symbol": "ETH/USDT:USDT", "last": 3000.0, "chg_24h_pct": -0.5,
         "vol_24h_usd": 1e9, "onboard_date": old},
        {"symbol": "SOL/USDT:USDT", "last": 150.0, "chg_24h_pct": 2.0,
         "vol_24h_usd": 8e8, "onboard_date": old},
        # VELVET: new (5d) AND a +130% pump AND a thin book -> fails THREE gates
        {"symbol": "VELVET/USDT:USDT", "last": 1.0, "chg_24h_pct": 130.0,
         "vol_24h_usd": 7e8, "onboard_date": young},
    ]


def test_velvet_excluded_majors_included():
    kept, drops = quality_filter(
        _rows(), now=_NOW, exchange=_FakeDepthExchange(),
        min_adv_usd=5e8, min_age_days=30, max_abs_chg_24h_pct=25.0,
        min_depth_usd=250_000.0, depth_ref_usd=100_000.0, symbol_count=30,
    )
    syms = [r["symbol"] for r in kept]
    assert "BTC/USDT:USDT" in syms   # ~$500k deep book CLEARS the 250k floor
    assert "ETH/USDT:USDT" in syms
    assert "SOL/USDT:USDT" in syms
    assert "VELVET/USDT:USDT" not in syms


def test_drop_counts_are_explicit_no_silent_truncation():
    kept, drops = quality_filter(
        _rows(), now=_NOW, exchange=_FakeDepthExchange(),
        min_adv_usd=5e8, min_age_days=30, max_abs_chg_24h_pct=25.0,
        min_depth_usd=250_000.0, depth_ref_usd=100_000.0, symbol_count=30,
    )
    # VELVET fails the age gate first (gates short-circuit in order), so age==1, others 0
    assert drops["age"] == 1
    assert drops["chg_24h"] == 0
    assert drops["depth"] == 0
    assert drops["adv"] == 0
    assert len(kept) == 3


def test_age_falls_back_to_klines_when_onboard_date_missing_and_keeps_old():
    # OLD via the kline fallback: earliest candle 2025-01-01 is >> min_age_days(30) -> KEPT.
    rows = [{"symbol": "OLDISH/USDT:USDT", "last": 1.0, "chg_24h_pct": 0.0,
             "vol_24h_usd": 1e9, "onboard_date": None}]
    kept, drops = quality_filter(
        rows, now=_NOW, exchange=_FakeDepthExchange(), min_adv_usd=1e8,
        min_age_days=30, max_abs_chg_24h_pct=25.0, min_depth_usd=250_000.0,
        depth_ref_usd=100_000.0, symbol_count=30,
    )
    assert [r["symbol"] for r in kept] == ["OLDISH/USDT:USDT"]
    assert drops["age"] == 0


def test_age_falls_back_to_klines_and_drops_a_genuinely_young_name():
    # The fallback's JOB: a name whose earliest candle is only ~5 days old must be DROPPED as
    # too young. This pins the young-rejection path the old fixture never exercised.
    class _YoungKlines(_FakeDepthExchange):
        def ohlcv(self, symbol, timeframe="4h", limit=500):
            ts = pd.date_range(_NOW - timedelta(days=5), periods=30, freq="4h", tz="UTC")
            return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0,
                                 "low": 1.0, "close": 1.0, "volume": 1.0})

    rows = [{"symbol": "YOUNG/USDT:USDT", "last": 1.0, "chg_24h_pct": 0.0,
             "vol_24h_usd": 1e9, "onboard_date": None}]
    kept, drops = quality_filter(
        rows, now=_NOW, exchange=_YoungKlines(), min_adv_usd=1e8,
        min_age_days=30, max_abs_chg_24h_pct=25.0, min_depth_usd=250_000.0,
        depth_ref_usd=100_000.0, symbol_count=30,
    )
    assert kept == []
    assert drops["age"] == 1


def test_depth_floor_excludes_thin_book():
    rows = [{"symbol": "VELVET/USDT:USDT", "last": 1.0, "chg_24h_pct": 0.0,
             "vol_24h_usd": 1e9, "onboard_date": _ms(_NOW - timedelta(days=900))}]
    kept, drops = quality_filter(
        rows, now=_NOW, exchange=_FakeDepthExchange(), min_adv_usd=1e8,
        min_age_days=30, max_abs_chg_24h_pct=25.0, min_depth_usd=250_000.0,
        depth_ref_usd=100_000.0, symbol_count=30,
    )
    assert kept == []
    assert drops["depth"] == 1


def test_depth_unavailable_keeps_name_and_is_counted():
    # exchange.depth raises -> the name is KEPT (not silently dropped) and counted.
    class _NoDepth(_FakeDepthExchange):
        def depth(self, symbol, limit=20):
            raise RuntimeError("no order book")

    rows = [{"symbol": "OK/USDT:USDT", "last": 1.0, "chg_24h_pct": 0.0,
             "vol_24h_usd": 1e9, "onboard_date": _ms(_NOW - timedelta(days=900))}]
    kept, drops = quality_filter(
        rows, now=_NOW, exchange=_NoDepth(), min_adv_usd=1e8,
        min_age_days=30, max_abs_chg_24h_pct=25.0, min_depth_usd=250_000.0,
        depth_ref_usd=100_000.0, symbol_count=30,
    )
    assert [r["symbol"] for r in kept] == ["OK/USDT:USDT"]
    assert drops["depth_unavailable"] == 1
    assert drops["depth"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quality_filter.py -v`
Expected: FAIL — `ImportError: cannot import name 'quality_filter' from 'futures_fund.market_data'`.

- [ ] **Step 3: Write minimal implementation**

In `futures_fund/market_data.py`, add these helpers and the function right after `liquidity_floor` (after line 137). `datetime`/`timezone` are already imported at the top (line 3); `pandas as pd` at line 5.

```python
def _book_depth_usd(levels: list[tuple[float, float]]) -> float:
    """FULL dollar notional of all `levels` (top-N book on one side). No cap — the depth floor is
    measured against this summed value, NOT clipped to depth_ref_usd."""
    acc = 0.0
    for price, qty in levels:
        acc += float(price) * float(qty)
    return acc


def _age_days(row: dict, *, now: datetime, exchange) -> float | None:
    """Listing age in days. Prefer onboard_date (ms-epoch); else derive from the earliest OHLCV
    kline timestamp (now - earliest). Returns None only when neither source is available (caller
    keeps the name, recording it under 'age_unknown' — a sane fallback, never a silent drop)."""
    onboard = row.get("onboard_date")
    if onboard is not None:
        return (now.timestamp() * 1000.0 - float(onboard)) / 86_400_000.0
    try:
        df = exchange.ohlcv(row["symbol"])
    except Exception:
        return None
    if df is None or df.empty or "timestamp" not in df:
        return None
    earliest = pd.to_datetime(df["timestamp"].iloc[0], utc=True).to_pydatetime()
    return (now - earliest).total_seconds() / 86_400.0


def quality_filter(
    rows: list[dict], *, now: datetime, exchange,
    min_adv_usd: float, min_age_days: int, max_abs_chg_24h_pct: float,
    min_depth_usd: float, depth_ref_usd: float, symbol_count: int,
) -> tuple[list[dict], dict[str, int]]:
    """'Liquid + established only': apply, in order, age -> 24h-mover -> depth -> ADV gates to a
    vol-ranked universe, then cap to symbol_count. Returns (kept_rows, drop_counts) so the scout
    can log EXACTLY how many names each gate removed (no silent truncation).

    - age: exclude names listed < min_age_days ago (onboard_date, else earliest-kline fallback);
      unknown age keeps the name (counted under 'age_unknown').
    - chg_24h: exclude |chg_24h_pct| > max_abs_chg_24h_pct (extreme movers are reversal traps).
    - depth: require the FULL top-of-book notional on the THINNER side >= min_depth_usd via
      exchange.depth(); missing/erroring/empty depth keeps the name ('depth_unavailable').
    - adv: the existing 24h-quote-volume floor (>= min_adv_usd).

    depth_ref_usd is accepted for config symmetry (it documents the slippage-model clip) but is NOT
    used as a cap inside the depth floor.
    """
    _ = depth_ref_usd  # reserved: slippage-model reference clip, not a floor cap
    drops = {"age": 0, "age_unknown": 0, "chg_24h": 0, "depth": 0,
             "depth_unavailable": 0, "adv": 0}
    kept: list[dict] = []
    for r in rows:
        age = _age_days(r, now=now, exchange=exchange)
        if age is None:
            drops["age_unknown"] += 1
        elif age < min_age_days:
            drops["age"] += 1
            continue
        if abs(float(r.get("chg_24h_pct") or 0.0)) > max_abs_chg_24h_pct:
            drops["chg_24h"] += 1
            continue
        try:
            book = exchange.depth(r["symbol"])
            bid_usd = _book_depth_usd(book.get("bids") or [])
            ask_usd = _book_depth_usd(book.get("asks") or [])
            side_usd = min(bid_usd, ask_usd)
            if side_usd <= 0.0:
                raise ValueError("empty book")
        except Exception:
            drops["depth_unavailable"] += 1
            side_usd = None
        if side_usd is not None and side_usd < min_depth_usd:
            drops["depth"] += 1
            continue
        if float(r.get("vol_24h_usd") or 0.0) < min_adv_usd:
            drops["adv"] += 1
            continue
        kept.append(r)
    return kept[:symbol_count], drops
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_quality_filter.py -v`
Expected: all seven tests PASS. In particular `test_velvet_excluded_majors_included` PASSES because the deep $500k book (`100.0 × 5000.0 = 500_000` per side) now CLEARS `min_depth_usd=250_000`, and `test_age_falls_back_to_klines_and_drops_a_genuinely_young_name` PASSES because the 5-day-old earliest kline is below `min_age_days`.

Run: `uv run ruff check futures_fund/market_data.py tests/test_quality_filter.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add futures_fund/market_data.py tests/test_quality_filter.py
git commit -m "feat(market_data): quality_filter (age+mover+depth+ADV) with explicit drop counts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire `quality_filter` into the scout, write a CLEAN universe.json + drop log

**Files:**
- Modify: `scripts/scout_cli.py`.
- Test: `tests/test_scout_cli.py` (the file EXISTS: `_FakeClient` at lines 8-23, the original test at lines 26-36).

The scout must build the keyless `FuturesExchange` (for `.depth()`/`.ohlcv()`), pass `now`, run `quality_filter` instead of `liquidity_floor`, persist the kept rows (with their `onboard_date`/`chg_24h_pct` metadata) to `universe.json`, and print a per-filter drop summary. Two INDEPENDENT fakes per test: `build_ccxt` → a fake ccxt CLIENT (markets + `fetch_tickers`, read by `scan_universe`), and `FuturesExchange.from_settings` → a fake EXCHANGE (`.depth`/`.ohlcv`, read by `quality_filter`).

- [ ] **Step 1: Write the failing test**

IMPORTANT ordering note: place the new fake classes ABOVE the original `test_scout_writes_crypto_only_universe` so the original test (which Step 4 edits to reference `_FakeQualityExchange`) sees them at collection time without relying on later-defined names. Insert the two fake classes IMMEDIATELY AFTER the existing `_FakeClient` (after line 23, before line 26's original test). Then append the new test at the END of the file.

First, add `import pandas as pd` to the top of `tests/test_scout_cli.py` (after `import json`, line 3). Then insert these two classes right after `_FakeClient` (after line 23):

```python
class _FakeQualityClient:
    markets = {
        "BTC/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "1567965300000"}},
        "ETH/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "1574840700000"}},
        "VELVET/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "9999999999999"}},
    }

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {
            "BTC/USDT:USDT": {"last": 60000.0, "quoteVolume": 2e9, "percentage": 1.0},
            "ETH/USDT:USDT": {"last": 3000.0, "quoteVolume": 1e9, "percentage": 0.5},
            # new (future onboardDate, 9999999999999 ms ~= year 2286) AND +130% pump
            "VELVET/USDT:USDT": {"last": 1.0, "quoteVolume": 9e8, "percentage": 130.0},
        }


class _FakeQualityExchange:
    def depth(self, symbol, limit=20):
        return {"bids": [(1.0, 1_000_000.0)], "asks": [(1.0, 1_000_000.0)]}  # ~$1M/side, deep

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        ts = pd.date_range("2020-01-01", periods=200, freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0,
                             "low": 1.0, "close": 1.0, "volume": 1.0})
```

Then append the new test at the END of the file:

```python
def test_scout_excludes_new_and_pumped_names(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.scout_cli.build_ccxt", lambda settings: _FakeQualityClient())
    monkeypatch.setattr(
        "scripts.scout_cli.FuturesExchange",
        type("X", (), {"from_settings": staticmethod(lambda settings: _FakeQualityExchange())}))
    from scripts.scout_cli import main

    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(tmp_path / "state"),
          "--top", "30"])
    out = json.loads((cycle_dir(tmp_path / "state", 1, cadence="weekly") / "universe.json")
                     .read_text())
    syms = [r["symbol"] for r in out["universe"]]
    assert "BTC/USDT:USDT" in syms and "ETH/USDT:USDT" in syms
    assert "VELVET/USDT:USDT" not in syms  # young + pumped -> dropped by quality_filter
    # kept rows carry the metadata cycle_prep needs (proves quality_filter ran on the fake exchange,
    # not a silently-unpatched real-network from_settings)
    btc = next(r for r in out["universe"] if r["symbol"] == "BTC/USDT:USDT")
    assert btc["onboard_date"] == 1567965300000
    assert "chg_24h_pct" in btc and "vol_24h_usd" in btc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scout_cli.py::test_scout_excludes_new_and_pumped_names -v`
Expected: FAIL — `AttributeError: <module 'scripts.scout_cli'> does not have the attribute 'FuturesExchange'` (the import is added in Step 3). NOTE: this is now an HONEST red — the depth floor is satisfiable, so the failure is the missing `FuturesExchange` wiring, not an empty universe.

- [ ] **Step 3: Write minimal implementation**

Rewrite `scripts/scout_cli.py`'s imports and `main`. Replace the import block (lines 9-19) and the `main` body (lines 22-44):

```python
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund.config import load_settings
from futures_fund.cycle_io import save_output
from futures_fund.exchange import FuturesExchange, build_ccxt
from futures_fund.market_data import quality_filter, scan_universe
from futures_fund.models import Cadence


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scan + quality-filter the crypto-only perp universe.")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    settings = load_settings()
    client = build_ccxt(settings)
    client.load_markets()
    exchange = FuturesExchange.from_settings(settings)
    now = datetime.now(UTC)

    rows = scan_universe(client, top_n=max(args.top, settings.universe.symbol_count))
    u = settings.universe
    universe, drops = quality_filter(
        rows, now=now, exchange=exchange,
        min_adv_usd=u.min_adv_usd, min_age_days=u.min_age_days,
        max_abs_chg_24h_pct=u.max_abs_chg_24h_pct, min_depth_usd=u.min_depth_usd,
        depth_ref_usd=u.depth_ref_usd, symbol_count=u.symbol_count,
    )
    save_output(args.state_dir, args.cycle, "universe", {"universe": universe}, cadence=cadence)
    print(json.dumps({
        "scanned": len(rows), "kept": len(universe), "dropped": drops,
        "universe": [r["symbol"] for r in universe],
    }, indent=2))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Patch the ORIGINAL test, then run all scout tests**

The ORIGINAL `test_scout_writes_crypto_only_universe` (lines 26-36) uses `_FakeClient` and does NOT patch `FuturesExchange`, so `FuturesExchange.from_settings(settings)` would try a real network build. Patch the exchange in that test too. Edit the original test body to add, right after the existing `build_ccxt` monkeypatch (line 27):

```python
    monkeypatch.setattr(
        "scripts.scout_cli.FuturesExchange",
        type("X", (), {"from_settings": staticmethod(lambda settings: _FakeQualityExchange())}))
```

(`_FakeQualityExchange` is now defined at module level above the original test by Step 1.) `_FakeClient`'s tickers report `percentage` 1.0/0.5/0.1 (all `|chg| <= 25`), `quoteVolume` ≥ 1e9 (clears `min_adv_usd`), and `_FakeQualityExchange.depth` is ~$1M/side (clears `min_depth_usd`) with 200 old candles. BUT `_FakeClient`'s markets carry NO `onboardDate`, so the age gate uses the KLINE FALLBACK: `_FakeQualityExchange.ohlcv` returns candles starting 2020-01-01 (>> min_age_days), so BTC/ETH survive the age gate; GOLD is still excluded by `is_crypto_perp` upstream in `scan_universe` (it never reaches `quality_filter`).

Run: `uv run pytest tests/test_scout_cli.py -v`
Expected: BOTH `test_scout_writes_crypto_only_universe` AND `test_scout_excludes_new_and_pumped_names` PASS. The new test's metadata assertions (`onboard_date`, `chg_24h_pct`, `vol_24h_usd` present) confirm `quality_filter` received the fake exchange and emitted real rows — a silently-unpatched real-network `from_settings` would have raised on `build_ccxt`/network instead.

Run: `uv run ruff check scripts/scout_cli.py tests/test_scout_cli.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/scout_cli.py tests/test_scout_cli.py
git commit -m "feat(scout): apply quality_filter, write clean universe.json + drop summary

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `CoinGeometry` carries depth, ADV-source metadata, 24h-change + onboard date

**Files:**
- Modify: `futures_fund/contracts.py:119-125` (the `# liquidity / filters` block in `CoinGeometry`).
- Test: `tests/test_cycle_prep.py` (append).

Add four additive fields so the geometry can feed both honest slippage and a future audit. `depth_bids`/`depth_asks` are the two crossing sides (kept separate so `apply_fills` can pick by sign); `onboard_date`/`chg_24h_pct` ride along from the universe row. `contracts.py` already imports `Field` (line 7).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cycle_prep.py` (it already imports `CoinGeometry as _CG`; use that alias to match the file):

```python
def test_coin_geometry_has_depth_and_quality_fields():
    g = _CG(
        symbol="BTC/USDT:USDT", mark=60000.0, adv_usd=2e9,
        depth_bids=[(60000.0, 5.0)], depth_asks=[(60001.0, 4.0)],
        onboard_date=1567965300000, chg_24h_pct=1.0,
    )
    assert g.depth_bids == [(60000.0, 5.0)]
    assert g.depth_asks == [(60001.0, 4.0)]
    assert g.onboard_date == 1567965300000
    assert g.chg_24h_pct == 1.0
    # defaults: a geometry built with no depth has empty books, not None crashes
    assert _CG(symbol="X/USDT:USDT", mark=1.0).depth_bids == []
    assert _CG(symbol="X/USDT:USDT", mark=1.0).onboard_date is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cycle_prep.py::test_coin_geometry_has_depth_and_quality_fields -v`
Expected: FAIL — `ValidationError` (unexpected keyword `depth_bids`).

- [ ] **Step 3: Write minimal implementation**

In `futures_fund/contracts.py`, in `CoinGeometry`, replace the `# liquidity / filters` block (lines 119-125):

```python
    # liquidity / filters
    adv_usd: float = 0.0
    chg_24h_pct: float = 0.0            # 24h % change carried from the universe row (audit/filter)
    onboard_date: int | None = None    # Binance onboardDate, ms-epoch (None when unavailable)
    # Phase 10 depth-aware slippage: the two crossing sides of the live L2 book at build time.
    # `depth_asks` is the crossing side for a BUY (delta>0), `depth_bids` for a SELL (delta<0).
    # Empty lists (not None) when depth was unavailable -> estimate_slippage uses the ADV fallback.
    depth_bids: list[tuple[float, float]] = Field(default_factory=list)
    depth_asks: list[tuple[float, float]] = Field(default_factory=list)
    spec: SymbolSpec | None = None
    # crypto-only universe audit: the exchange `market["info"]` (carries `underlyingType` /
    # `contractType`) the reviewer feeds to `market_data.is_crypto_perp` to reject TradFi-wrapper
    # perps (tokenized stocks / commodities / indices). None => no metadata (treated as crypto).
    market_info: dict | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cycle_prep.py::test_coin_geometry_has_depth_and_quality_fields -v`
Expected: PASS.

Run: `uv run pytest tests/test_cycle_prep.py -q`
Expected: PASS (additive fields, no existing test reads them).

- [ ] **Step 5: Commit**

```bash
git add futures_fund/contracts.py tests/test_cycle_prep.py
git commit -m "feat(contracts): CoinGeometry carries depth_bids/asks, onboard_date, chg_24h_pct

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `FuturesExchange.onboard_date_ms` accessor

**Files:**
- Modify: `futures_fund/exchange.py` (add after `depth`, after line 116).
- Test: `tests/test_exchange.py` (APPEND — the file EXISTS, 88 lines, with depth tests; `FuturesExchange` already imported on line 3).

A thin accessor so `build_geometries` could read `onboardDate` from ccxt's cached `market(sym)["info"]` without re-deriving from klines (it is a reusable building block; `quality_filter` itself reads onboard from the scout ROW, not this accessor). Fail-soft → `None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_exchange.py` (do NOT re-add the `from futures_fund.exchange import FuturesExchange` import — it is already on line 3):

```python
class _OnboardCcxt:
    def market(self, symbol):
        if symbol == "BTC/USDT:USDT":
            return {"info": {"onboardDate": "1567965300000"}}
        return {"info": {}}


def test_onboard_date_ms_parses_string_epoch():
    ex = FuturesExchange(_OnboardCcxt(), keyless=True)
    assert ex.onboard_date_ms("BTC/USDT:USDT") == 1567965300000


def test_onboard_date_ms_none_when_absent():
    ex = FuturesExchange(_OnboardCcxt(), keyless=True)
    assert ex.onboard_date_ms("FOO/USDT:USDT") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exchange.py::test_onboard_date_ms_parses_string_epoch -v`
Expected: FAIL — `AttributeError: 'FuturesExchange' object has no attribute 'onboard_date_ms'`.

- [ ] **Step 3: Write minimal implementation**

In `futures_fund/exchange.py`, add immediately after the `depth` method (after line 116, the `return {"bids": bids, "asks": asks}` line):

```python
    def onboard_date_ms(self, symbol: str) -> int | None:
        """Binance listing timestamp (ms-epoch) from the ccxt-cached market info, or None.

        ccxt exposes onboardDate only via market(sym)["info"]["onboardDate"] (a string) after
        load_markets(); fail-soft to None so cycle_prep can fall back to the earliest-kline age."""
        try:
            raw = (self.client.market(symbol).get("info") or {}).get("onboardDate")
            return int(raw) if raw is not None else None
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_exchange.py -v`
Expected: the two new tests PASS and all pre-existing exchange tests (incl. the depth tests at lines 78/86) still PASS.

Run: `uv run ruff check futures_fund/exchange.py tests/test_exchange.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add futures_fund/exchange.py tests/test_exchange.py
git commit -m "feat(exchange): onboard_date_ms accessor (ccxt market info, fail-soft)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `build_geometries` stamps real ADV + depth + onboard/chg from universe rows

**Files:**
- Modify: `futures_fund/cycle_prep.py` (`build_geometries` signature 66-73; the `geometries.append(...)` block 94-105; add a `_safe_depth` helper after line 64).
- Modify: `scripts/cycle_prep_cli.py` (`_symbols` → `_universe_rows`, lines 39-48; `main`, lines 63-68).
- Test: `tests/test_cycle_prep.py` (append).

`build_geometries` gains an optional `universe_rows: dict[str, dict] | None = None` keyed by symbol (the scout rows carrying `vol_24h_usd`, `chg_24h_pct`, `onboard_date`). For each symbol it: sets `adv_usd` from the row's `vol_24h_usd` (falls back to 0.0); fetches `exchange.depth(sym)` fail-soft and stamps `depth_bids`/`depth_asks`; copies `chg_24h_pct`/`onboard_date` from the row. **Fail-soft:** an exchange without a callable `.depth` (the no-seed fake before Task 9) yields empty books via a `getattr(exchange, "depth", None)` guard.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cycle_prep.py` (it already imports `build_geometries`, `pandas as pd`, `numpy as np`, `datetime/UTC`, `_CG`; add a `FundingInfo` import and a local `_NOW_T7` so the appended block is self-contained):

```python
from futures_fund.market_data import FundingInfo

_NOW_T7 = datetime(2026, 6, 12, tzinfo=UTC)


class _GeoExchange:
    def ohlcv(self, symbol, timeframe="4h", limit=500):
        ts = pd.date_range("2025-01-01", periods=60, freq="4h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 100.0,
                             "low": 100.0, "close": 100.0, "volume": 1.0})

    def funding(self, symbol):
        return FundingInfo(symbol=symbol, current_rate=0.0001,
                           next_funding_ts=_NOW_T7, interval_hours=8.0,
                           mark_price=100.0, index_price=100.0)

    def mark_price(self, symbol):
        return 100.0

    def depth(self, symbol, limit=20):
        return {"bids": [(99.0, 10.0)], "asks": [(101.0, 8.0)]}


def test_build_geometries_stamps_adv_depth_and_quality_metadata():
    rows = {"BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "vol_24h_usd": 2e9,
                              "chg_24h_pct": 1.5, "onboard_date": 1567965300000}}
    bundle = build_geometries(_GeoExchange(), ["BTC/USDT:USDT"], now=_NOW_T7,
                              universe_rows=rows)
    g = bundle.geometries[0]
    assert g.adv_usd == 2e9
    assert g.depth_asks == [(101.0, 8.0)]
    assert g.depth_bids == [(99.0, 10.0)]
    assert g.chg_24h_pct == 1.5
    assert g.onboard_date == 1567965300000


def test_build_geometries_fail_soft_without_depth_method():
    class _NoDepth(_GeoExchange):
        depth = None  # attribute present but not callable -> guarded
    bundle = build_geometries(_NoDepth(), ["BTC/USDT:USDT"], now=_NOW_T7)
    g = bundle.geometries[0]
    assert g.depth_bids == [] and g.depth_asks == []
    assert g.adv_usd == 0.0  # no universe_rows -> default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cycle_prep.py::test_build_geometries_stamps_adv_depth_and_quality_metadata -v`
Expected: FAIL — `TypeError: build_geometries() got an unexpected keyword argument 'universe_rows'`.

- [ ] **Step 3: Write minimal implementation — cycle_prep**

In `futures_fund/cycle_prep.py`, change the `build_geometries` signature (lines 66-73) to add the param:

```python
def build_geometries(
    exchange,
    symbols: list[str],
    *,
    now: datetime,
    btc_symbol: str = "BTC/USDT:USDT",
    beta_lookback: int = 45,
    universe_rows: dict[str, dict] | None = None,
) -> GeometryBundle:
```

Add a depth helper above the function (after line 64, before the two blank lines preceding `def build_geometries`):

```python
def _safe_depth(exchange, symbol: str) -> tuple[list, list]:
    """(bids, asks) from exchange.depth, fail-soft to ([], []) when unavailable/raising."""
    fn = getattr(exchange, "depth", None)
    if not callable(fn):
        return [], []
    try:
        book = fn(symbol)
        return list(book.get("bids") or []), list(book.get("asks") or [])
    except Exception:
        return [], []
```

Replace the `geometries.append(CoinGeometry(...))` block (lines 94-105) with one that reads the row + depth:

```python
        rows = universe_rows or {}
        row = rows.get(sym, {})
        bids, asks = _safe_depth(exchange, sym)
        geometries.append(CoinGeometry(
            symbol=sym,
            mark=mark,
            momentum_20=_momentum_20(series) if series is not None else 0.0,
            realized_vol=_realized_vol(series) if series is not None else 0.0,
            beta_btc=betas.get(sym, 1.0),
            beta_lookback_days=beta_lookback,
            funding_rate=rate,
            funding_interval_hours=interval,
            funding_apr=funding_apr(rate, interval),
            adv_usd=float(row.get("vol_24h_usd") or 0.0),
            chg_24h_pct=float(row.get("chg_24h_pct") or 0.0),
            onboard_date=row.get("onboard_date"),
            depth_bids=bids,
            depth_asks=asks,
        ))
```

- [ ] **Step 4: Write minimal implementation — cycle_prep_cli**

In `scripts/cycle_prep_cli.py`, replace `_symbols` (lines 39-48) with a version that returns the full rows:

```python
def _universe_rows(state_dir, cycle: int, cadence: Cadence, settings) -> list[dict]:
    """This cycle's universe.json rows (scout output); fall back to bare settings.symbols rows."""
    try:
        rows = load_output(state_dir, cycle, "universe", cadence=cadence)["universe"]
        if rows:
            return rows
    except FileNotFoundError:
        pass
    return [{"symbol": s} for s in settings.symbols]
```

Then replace lines 63-68 in `main` (currently `symbols = _symbols(...)` on line 63 followed by the `build_geometries(...)` call on 65-68). The replacement DEFINES `symbols`/`rows_by_sym` and calls `build_geometries`, fully removing the old `symbols = _symbols(...)` call site on line 63 (line 69's `build_pairs_and_spreads(ex, symbols, ...)` still consumes `symbols`, which the new code redefines):

```python
    rows = _universe_rows(args.state_dir, args.cycle, cadence, settings)
    symbols = [r["symbol"] for r in rows if r.get("symbol")]
    rows_by_sym = {r["symbol"]: r for r in rows if r.get("symbol")}

    bundle = build_geometries(
        ex, symbols, now=now, btc_symbol=settings.beta.btc_symbol,
        beta_lookback=settings.beta.lookback_days, universe_rows=rows_by_sym,
    )
```

(There is no other reference to the old `_symbols` FUNCTION: `tests/test_cycle_prep_cli.py` seeds `universe.json` via `save_output` and the CLI reads it — its fake's `self._symbols` attribute is unrelated. No test imports the CLI's `_symbols`. Verify the OLD `symbols = _symbols(...)` line is fully removed when you paste the replacement.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cycle_prep.py::test_build_geometries_stamps_adv_depth_and_quality_metadata tests/test_cycle_prep.py::test_build_geometries_fail_soft_without_depth_method -v`
Expected: both PASS.

Run: `uv run pytest tests/test_cycle_prep.py tests/test_cycle_prep_cli.py -q`
Expected: PASS — `test_cycle_prep_cli_writes_all_four_artifacts` still passes (it seeds `universe.json` rows `{"symbol": s}` with no metadata; `build_geometries` defaults `adv_usd=0.0`, empty depth; its `_FakeExchange` has no `.depth` method → `_safe_depth` guard returns empty books; no crash). The pre-existing `build_geometries` DOGE-clamp test (`test_cycle_prep.py:84-87`) still passes (it does not pass `universe_rows`).

Run: `uv run ruff check futures_fund/cycle_prep.py scripts/cycle_prep_cli.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add futures_fund/cycle_prep.py scripts/cycle_prep_cli.py tests/test_cycle_prep.py
git commit -m "feat(cycle_prep): stamp real ADV + depth + onboard/chg onto CoinGeometry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Thread depth into `CostInputs` and select crossing side in `apply_fills`

**Files:**
- Modify: `futures_fund/account.py:73-81` (`CostInputs`), `account.py:216-218` (the `estimate_slippage` call in `apply_fills`).
- Modify: `scripts/run_paper_cli.py:290-308` (`_geometry_cost_maps`).
- Test: `tests/test_depth_aware_slippage.py` (create).

`CostInputs` keeps BOTH book sides so the executor can pick by trade direction. Add two fields `depth_bids`/`depth_asks`; keep the legacy `depth` field for back-compat (default `None`, wins when set). `apply_fills` chooses `asks` for a BUY (`delta_signed_qty > 0`), `bids` for a SELL, and passes the chosen side to `estimate_slippage`. Set `half_spread_bps` from the top-of-book spread when both sides exist. `account.py` already imports `Field` (line 32) and `trade_fee` (line 34).

**Modeling caveat (documented, intentional, NON-blocking — pre-existing `vwap_fill` behavior):** `apply_fills` opens the FULL target delta qty, but `estimate_slippage → depth_slippage → costs.slippage_cost → costs.vwap_fill` prices slippage on the PARTIAL fill (only up to available book depth) — so for a clip that EXCEEDS the visible book, realized slippage is UNDER-stated while full size is still taken. This plan does NOT change `vwap_fill`; it surfaces the caveat in a `CostInputs` docstring line and pins it with a test (`test_over_depth_clip_is_under_costed_documented`) so a future reader does not trust the depth slippage number as exact for over-depth clips. The economic claim we DO pin is the comparative one: a thin alt costs materially more than BTC and more than 1bps.

- [ ] **Step 1: Write the failing test**

Create `tests/test_depth_aware_slippage.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import CostInputs, PaperAccount
from scripts.run_paper_cli import _geometry_cost_maps

_TS = datetime(2026, 6, 12, tzinfo=UTC)


def _btc_book():
    # deep: 1000 BTC at ~60000 each side -> a small clip is nearly free
    return ([(59999.0, 1000.0)], [(60001.0, 1000.0)])


def _thin_book():
    # thin alt: only 50 units near 100, then a steep step
    return ([(99.5, 50.0), (95.0, 50.0)], [(100.5, 50.0), (105.0, 50.0)])


def _account():
    return PaperAccount(cash=1_000_000.0)


def test_thin_book_slippage_materially_exceeds_btc_and_one_bp():
    # buy the SAME ~$50k notional on a deep BTC book vs a thin alt book
    deep_bids, deep_asks = _btc_book()
    thin_bids, thin_asks = _thin_book()

    btc_acct = _account()
    btc_acct.apply_fills(
        [{"symbol": "BTC/USDT:USDT", "direction": "long", "target_notional": 50_000.0}],
        marks={"BTC/USDT:USDT": 60000.0},
        costs={"BTC/USDT:USDT": CostInputs(depth_bids=deep_bids, depth_asks=deep_asks)},
        opened_ts=_TS, opened_cycle=1, opened_cadence="weekly",
    )
    alt_acct = _account()
    alt_acct.apply_fills(
        [{"symbol": "ALT/USDT:USDT", "direction": "long", "target_notional": 50_000.0}],
        marks={"ALT/USDT:USDT": 100.0},
        costs={"ALT/USDT:USDT": CostInputs(depth_bids=thin_bids, depth_asks=thin_asks)},
        opened_ts=_TS, opened_cycle=1, opened_cadence="weekly",
    )
    btc_slip = btc_acct.slippage_paid
    alt_slip = alt_acct.slippage_paid
    assert alt_slip > btc_slip
    # thin alt slippage is materially > 1bp of the 50k notional (1bp = 5.0 USDT)
    assert alt_slip > 5.0
    assert alt_slip > 5.0 * btc_slip  # at least an order of magnitude worse than BTC


def test_apply_fills_selects_ask_side_for_a_buy():
    # only the ASK side is thin; the BID side is deep. A BUY must cost from the thin ASK side.
    acct = _account()
    acct.apply_fills(
        [{"symbol": "ALT/USDT:USDT", "direction": "long", "target_notional": 4_000.0}],
        marks={"ALT/USDT:USDT": 100.0},
        costs={"ALT/USDT:USDT": CostInputs(
            depth_bids=[(100.0, 1_000_000.0)],            # deep bids (irrelevant for a buy)
            depth_asks=[(101.0, 10.0), (130.0, 10.0)])},  # thin asks -> real slippage
        opened_ts=_TS, opened_cycle=1, opened_cadence="weekly",
    )
    assert acct.slippage_paid > 5.0  # priced off the thin ASK side, not the deep bids


def test_over_depth_clip_is_under_costed_documented():
    # DOCUMENTED CAVEAT: vwap_fill prices slippage on the PARTIAL fill, but the position opens at
    # FULL target qty. Here the book holds only 10 units (~$1100) but we buy ~$50k. The position
    # qty is the full target (500 units) yet slippage is priced on the 10 filled units -> the
    # number is an UNDER-estimate for over-depth clips. We pin the exact under-cost so a future
    # reader treats it as a floor, not an exact cost.
    acct = _account()
    acct.apply_fills(
        [{"symbol": "ALT/USDT:USDT", "direction": "long", "target_notional": 50_000.0}],
        marks={"ALT/USDT:USDT": 100.0},
        costs={"ALT/USDT:USDT": CostInputs(
            depth_bids=[(100.0, 10.0)], depth_asks=[(110.0, 10.0)])},  # only 10 units @ +10
        opened_ts=_TS, opened_cycle=1, opened_cadence="weekly",
    )
    pos = acct.positions["ALT/USDT:USDT"]
    assert pos.qty == 500.0                       # FULL target qty opened (50_000 / 100)
    # slippage priced on the 10 filled units * (110 - 100) = 100 USDT (the partial), NOT on 500.
    assert acct.slippage_paid == 100.0            # under-costed vs the true 500-unit impact


def test_geometry_cost_maps_threads_both_book_sides():
    bundle = {"geometries": [{
        "symbol": "ALT/USDT:USDT", "mark": 100.0, "funding_rate": 0.0,
        "funding_interval_hours": 8, "adv_usd": 1e6,
        "depth_bids": [[99.0, 5.0]], "depth_asks": [[101.0, 4.0]],
    }]}
    _marks, _funding, _intervals, costs = _geometry_cost_maps(bundle)
    ci = costs["ALT/USDT:USDT"]
    assert ci.depth_asks == [(101.0, 4.0)]
    assert ci.depth_bids == [(99.0, 5.0)]
    assert ci.adv_usd == 1e6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_depth_aware_slippage.py -v`
Expected: FAIL — `ValidationError`/`TypeError` on `CostInputs(depth_bids=...)` (no such field yet).

- [ ] **Step 3: Write minimal implementation — CostInputs + apply_fills**

In `futures_fund/account.py`, extend `CostInputs` (replace lines 73-81):

```python
class CostInputs(BaseModel):
    """Per-symbol frictions the paper executor needs but the executed proposal does not carry.

    `depth_asks`/`depth_bids` are the two crossing sides of the live book; `apply_fills` selects
    the ASK side for a BUY (delta>0) and the BID side for a SELL (delta<0). When both are empty
    `estimate_slippage` uses the ADV + half-spread fallback (which is NEVER flat 2bps). `depth` is
    retained for back-compat (a pre-selected single side); it wins when set.

    CAVEAT: costs.vwap_fill prices slippage on the PARTIAL fill (up to visible book depth) while
    apply_fills opens the FULL target qty. For a clip that exceeds the book, realized slippage is
    UNDER-stated — treat depth slippage as a floor for over-depth clips, not an exact cost."""
    adv_usd: float = 0.0
    half_spread_bps: float = 1.0
    depth: list[tuple[float, float]] | None = None
    depth_bids: list[tuple[float, float]] = Field(default_factory=list)
    depth_asks: list[tuple[float, float]] = Field(default_factory=list)
    maker: bool = False                          # paper opens are market -> taker
```

In `apply_fills`, replace the `estimate_slippage` call (lines 216-218) — `delta_signed_qty` is already computed on line 211, `ci` a few lines above:

```python
            if ci.depth is not None:
                side = ci.depth                                 # pre-selected (back-compat)
            elif delta_signed_qty > 0:
                side = ci.depth_asks or None                    # BUY crosses the ASKS
            else:
                side = ci.depth_bids or None                    # SELL crosses the BIDS
            slip = estimate_slippage(
                sym, abs(delta_signed_qty), mark, depth=side, adv_usd=ci.adv_usd,
                half_spread_bps=ci.half_spread_bps)
```

- [ ] **Step 4: Write minimal implementation — _geometry_cost_maps**

In `scripts/run_paper_cli.py`, add a helper above `_geometry_cost_maps` (before line 290) and replace line 307 (`costs[sym] = CostInputs(adv_usd=float(g.get("adv_usd", 0.0)))`).

Add the helper:

```python
def _half_spread_bps(bids: list, asks: list, default: float) -> float:
    """Observed top-of-book half-spread in bps; `default` when a side is missing/degenerate."""
    if not bids or not asks:
        return default
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0 or best_ask < best_bid:
        return default
    return (best_ask - best_bid) / 2.0 / mid * 1e4
```

Replace line 307 with:

```python
        bids = [(float(p), float(q)) for p, q in (g.get("depth_bids") or [])]
        asks = [(float(p), float(q)) for p, q in (g.get("depth_asks") or [])]
        costs[sym] = CostInputs(
            adv_usd=float(g.get("adv_usd", 0.0)),
            half_spread_bps=_half_spread_bps(bids, asks, 1.0),
            depth_bids=bids, depth_asks=asks,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_depth_aware_slippage.py -v`
Expected: all five tests PASS (incl. the documented over-depth caveat test).

Run: `uv run pytest tests/test_account_integration.py tests/test_slippage.py -q`
Expected: PASS (existing `CostInputs` callers used keyword args / defaults; `depth=None` default unchanged).

Run: `uv run ruff check futures_fund/account.py scripts/run_paper_cli.py tests/test_depth_aware_slippage.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add futures_fund/account.py scripts/run_paper_cli.py tests/test_depth_aware_slippage.py
git commit -m "feat(slippage): thread real book depth into fills, cost thin alts off their book

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Extend the no-seed E2E fakes for depth + onboardDate (keep suite green)

**Files:**
- Modify: `tests/test_end_to_end_no_seed.py` (`_FakeCyclePrepExchange` ends ~line 54 after `mark_price`; `_FakeScoutClient.markets` is line 58; the `no_seed_env` fixture patches `from_settings` at lines 77-78).

**Ordering dependency (called out explicitly):** the age gate uses `onboard_date` from the scan_universe ROW, which carries it from the client market — this REQUIRES Task 2 (row propagation) shipped first. The chain is: Task 2 (`scan_universe` reads `onboardDate` into the row) → Task 3 (`quality_filter._age_days` prefers the row's `onboard_date`) → Task 9 (this fixture supplies `onboardDate` on every scout-client market). With Task 2 in place the age gate uses `onboardDate` (NOT the kline fallback) for these names; we still give a deep, calm, old market so every name clears age + mover + depth + ADV and the E2E's invariants (feasible/neutral/deployed) are unchanged.

The new scout path calls `FuturesExchange.from_settings(...).depth()`/`.ohlcv()` AND `build_geometries` calls `exchange.depth()`. The fixture already patches `futures_fund.exchange.FuturesExchange.from_settings` → `_FakeCyclePrepExchange` (lines 77-78), and `scripts.scout_cli.FuturesExchange` is the SAME class object (both `from futures_fund.exchange import FuturesExchange`), so the scout's quality filter and `build_geometries` both resolve to `_FakeCyclePrepExchange` once it has `.depth()`. The fixture's `monkeypatch.setattr("scripts.scout_cli.build_ccxt", ...)` already supplies the keyless client `_FakeScoutClient`.

- [ ] **Step 1: Run the E2E to capture the current (pre-Task-9) state**

Run: `uv run pytest tests/test_end_to_end_no_seed.py -q`
Expected after Tasks 4/7 are merged but before this task: the scout now calls `FuturesExchange.from_settings(...).depth()` and `build_geometries` calls `exchange.depth()`. `_FakeCyclePrepExchange` has no `.depth()` yet, so `build_geometries._safe_depth` returns empty books (fail-soft, no crash) and `quality_filter` records `depth_unavailable` (keeps every name, since the fake exchange's `.depth` is ABSENT → `quality_filter`'s `try/except` counts it and keeps the name). Age gate: `_FakeScoutClient` markets have NO `onboardDate` yet, so `_age_days` uses the kline fallback on `_FakeCyclePrepExchange.ohlcv` (120 candles from 2026-01-01; with `now=2026-06-11` the earliest candle is ~160 days old >> min_age_days, so names are KEPT). So the E2E likely STILL PASSES via fail-soft — capture whatever it does. Step 2 gives the quality filter REAL depth/onboard data and makes the depth gate genuinely exercised, not bypassed.

- [ ] **Step 2: Extend the fakes**

In `tests/test_end_to_end_no_seed.py`, add a `depth` method to `_FakeCyclePrepExchange` immediately after its `mark_price` method (~line 54):

```python
    def depth(self, symbol, limit=20):
        # deep, symmetric book around the mark so every name clears the min_depth_usd floor
        mark = _MARKS[symbol]
        qty = 5_000_000.0 / mark  # ~$5M per level -> full top-of-book notional >> min_depth_usd
        return {"bids": [(mark * 0.999, qty)], "asks": [(mark * 1.001, qty)]}
```

Add `onboardDate` (old) to every `_FakeScoutClient` market — replace line 58 (`markets = {s: {"info": {"underlyingType": "COIN"}} for s in _UNIVERSE}`):

```python
    # old listing (2019 epoch) so the min_age_days gate keeps every name via onboard_date (NOT the
    # kline fallback) — exercises the Task 2 row -> Task 3 gate path end to end.
    markets = {s: {"info": {"underlyingType": "COIN", "onboardDate": "1567965300000"}}
               for s in _UNIVERSE}
```

No change to the fixture body: `scripts.scout_cli.FuturesExchange.from_settings` and `futures_fund.exchange.FuturesExchange.from_settings` are the SAME class object, already patched once at lines 77-78. Now that `_FakeCyclePrepExchange` has `.depth()`, the scout's depth gate and `build_geometries` both work through it.

- [ ] **Step 3: Run the full no-seed E2E to verify it passes**

Run: `uv run pytest tests/test_end_to_end_no_seed.py -q`
Expected: all five tests PASS. The ~$5M-per-level book gives the FULL top-of-book notional `5_000_000` per side >> `min_depth_usd=250_000`, so every name clears the depth gate (now genuinely exercised, not via `depth_unavailable`). `slippage_paid > 0` still holds: fills happen at the crossing price (`mark*1.001` for a buy) vs the `mark` reference, so `depth_slippage` is non-zero, and `_half_spread_bps` reports ~10bps. `fees_paid > 0` unchanged. The `< 20_000` equity assertions and `points[-1] == account.equity(marks)` self-consistency hold (fees+slippage > 0). If a tighter equity-magnitude assertion drifts, note it only checks `< 20_000` and self-consistency — both hold.

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end_no_seed.py
git commit -m "test(e2e): extend no-seed fakes with depth + onboardDate for quality+slippage path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Bound the carry sleeve's extreme-funding signal (neutral shared helper)

**Files:**
- Modify: `futures_fund/funding_intervals.py` (add the neutral `bounded_apr` helper after `clamp_funding_rate`, ~line 36).
- Modify: `futures_fund/sleeves/carry.py:15-42` (`carry_signal`).
- Test: `tests/test_funding_intervals.py` (append the helper test) and `tests/sleeves/test_carry.py` (append the sleeve tests).

To AVOID a factor→carry import coupling, the sign-preserving APR clamp lives in `funding_intervals.py` (the module that already owns `clamp_funding_rate`/`funding_apr` and imports ONLY `futures_fund.models`), and BOTH sleeves import it from there. `carry_signal` gains an opt-in `max_abs_apr: float | None = None`. When set, the `funding_apr` used for RANKING and `raw_score` is clamped to `[-max_abs_apr, +max_abs_apr]` (sign-preserving) so an extreme-funding name ranks AT the cap alongside other capped names, not beyond them. Default `None` preserves every existing carry test (which pin `raw_score == 0.30` etc.).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_funding_intervals.py`:

```python
from futures_fund.funding_intervals import bounded_apr


def test_bounded_apr_sign_preserving_clamp():
    assert bounded_apr(20.0, 2.0) == 2.0
    assert bounded_apr(-20.0, 2.0) == -2.0
    assert bounded_apr(1.5, 2.0) == 1.5     # inside the band: unchanged
    assert bounded_apr(-1.5, 2.0) == -1.5


def test_bounded_apr_none_cap_is_unbounded():
    assert bounded_apr(20.0, None) == 20.0
    assert bounded_apr(-20.0, None) == -20.0
```

Append to `tests/sleeves/test_carry.py` (it already defines `_geo(symbol, apr)`, `_NOW`, and imports `carry_signal`):

```python
def test_carry_signal_clamps_extreme_apr_for_ranking_and_score():
    geos = [
        _geo("BLOWOFF/USDT:USDT", 20.0),   # extreme funding APR (reversal trap)
        _geo("HIGH/USDT:USDT", 1.5),
        _geo("MID/USDT:USDT", 0.0),
        _geo("LOW/USDT:USDT", -1.5),
        _geo("NEG/USDT:USDT", -20.0),      # extreme negative
    ]
    sig = carry_signal(geos, risk_budget_frac=0.25, now=_NOW, top_frac=1 / 3, max_abs_apr=2.0)
    by_sym = {t.symbol: t for t in sig.tilts}
    # the blow-off is still shorted (positive carry) but its raw_score is CAPPED at +2.0, not 20.0
    assert by_sym["BLOWOFF/USDT:USDT"].direction == "short"
    assert by_sym["BLOWOFF/USDT:USDT"].raw_score == 2.0
    assert by_sym["NEG/USDT:USDT"].raw_score == -2.0
    assert sig.diagnostics["max_abs_apr"] == 2.0


def test_carry_signal_default_no_cap_preserves_raw_apr():
    geos = [_geo("A/USDT:USDT", 20.0), _geo("B/USDT:USDT", -20.0)]
    sig = carry_signal(geos, risk_budget_frac=0.25, now=_NOW)  # max_abs_apr defaults None
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["A/USDT:USDT"].raw_score == 20.0   # unbounded baseline unchanged
    assert by_sym["B/USDT:USDT"].raw_score == -20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_funding_intervals.py::test_bounded_apr_sign_preserving_clamp tests/sleeves/test_carry.py::test_carry_signal_clamps_extreme_apr_for_ranking_and_score -v`
Expected: FAIL — `ImportError: cannot import name 'bounded_apr' from 'futures_fund.funding_intervals'` and `TypeError: carry_signal() got an unexpected keyword argument 'max_abs_apr'`.

- [ ] **Step 3: Write minimal implementation — the neutral helper**

In `futures_fund/funding_intervals.py`, add `bounded_apr` immediately after `clamp_funding_rate` (after its `return rate`, ~line 36):

```python
def bounded_apr(apr: float, cap: float | None) -> float:
    """Sign-preserving clamp of an annualized funding_apr to +-cap. cap=None -> unbounded.

    EXTREME FUNDING IS A REVERSAL TRAP, NOT FREE ALPHA: a blow-off rate that annualizes to a huge
    APR should be treated as CAPPED, ranking alongside other at-cap names — never as more
    attractive than them. The per-symbol realized RATE is already clamped upstream
    (clamp_funding_rate, majors +-0.003 / alts +-0.02); this is a STRATEGY-level signal cap on top.
    Lives here (not in a sleeve) so carry and factor both import it from a neutral module."""
    if cap is None:
        return apr
    if apr > cap:
        return cap
    if apr < -cap:
        return -cap
    return apr
```

- [ ] **Step 4: Write minimal implementation — carry_signal**

In `futures_fund/sleeves/carry.py`, add the import (after the existing `from futures_fund.contracts import ...` line):

```python
from futures_fund.funding_intervals import bounded_apr
```

Then replace `carry_signal` (lines 15-42):

```python
def carry_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                 top_frac: float = 1 / 3, max_abs_apr: float | None = None) -> SleeveSignal:
    """Long low/negative funding_apr, short high-positive funding_apr, delta-hedged.

    raw_score carries the signed funding_apr (bounded to +-max_abs_apr when set; see
    funding_intervals.bounded_apr — extreme funding is a reversal trap, not free alpha);
    target_weight is the per-leg signed share of the side budget (long > 0, short < 0),
    equal-weight within each side (pre-optimize). k = max(1, floor(n * top_frac)) names per side.
    """
    scored = [(g, bounded_apr(g.funding_apr, max_abs_apr)) for g in geometries]
    ranked = sorted(scored, key=lambda gs: gs[1])              # ascending: most negative first
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
    for g, score in longs:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="long",
                                target_weight=long_w, raw_score=score))
    for g, score in shorts:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="short",
                                target_weight=-short_w, raw_score=score))
    return SleeveSignal(sleeve="carry", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"n_candidates": n, "k_per_side": k,
                                     "max_abs_apr": max_abs_apr}, as_of_ts=now)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_funding_intervals.py tests/sleeves/test_carry.py -v`
Expected: the new helper + carry tests PASS and ALL pre-existing carry tests PASS (default `max_abs_apr=None` keeps `raw_score == funding_apr`, e.g. `test_carry_signal_shorts_high_funding_longs_negative_funding` still pins `0.30`/`-0.25`).

Run: `uv run ruff check futures_fund/funding_intervals.py futures_fund/sleeves/carry.py`
Expected: PASS (`bounded_apr` imported and used — no F401; no import cycle: `funding_intervals` imports only `futures_fund.models`).

- [ ] **Step 6: Commit**

```bash
git add futures_fund/funding_intervals.py futures_fund/sleeves/carry.py tests/test_funding_intervals.py tests/sleeves/test_carry.py
git commit -m "feat(carry): opt-in strategy cap on |funding_apr| via neutral bounded_apr helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Mirror the carry bound in the factor sleeve's carry leg

**Files:**
- Modify: `futures_fund/sleeves/factor.py:13-29` (`_factor_score`, `rank_factor`), `factor.py:32-40` (`_combined_rank`), `factor.py:55-63` (`factor_signal`).
- Test: `tests/sleeves/test_factor.py` (append).

The factor sleeve also ranks on `-g.funding_apr` for its `"carry"` factor (line 17). A clamp applied only in `carry_signal` leaves this leg unbounded. Import the SAME neutral `bounded_apr` from `funding_intervals` (NOT a private helper from carry — no factor→carry coupling) and thread an optional `max_abs_apr` through `rank_factor`/`_combined_rank`/`factor_signal`. Default `None` (no behavior change to existing factor tests).

- [ ] **Step 1: Write the failing test**

Append to `tests/sleeves/test_factor.py` (it already imports `rank_factor` and `CoinGeometry`; define a local `_g` to avoid clobbering the existing keyword-only `_geo`):

```python
def _g(sym, apr):
    return CoinGeometry(symbol=sym, mark=100.0, funding_apr=apr)


def test_rank_factor_carry_respects_max_abs_apr():
    geos = [_g("BLOWOFF/USDT:USDT", 20.0), _g("CALM/USDT:USDT", 1.0)]
    # carry score = -bounded_apr; with cap 2.0 the blow-off score is -2.0, not -20.0
    ranked = dict(rank_factor(geos, factor="carry", max_abs_apr=2.0))
    assert ranked["BLOWOFF/USDT:USDT"] == -2.0
    assert ranked["CALM/USDT:USDT"] == -1.0


def test_rank_factor_carry_default_unbounded():
    geos = [_g("BLOWOFF/USDT:USDT", 20.0)]
    ranked = dict(rank_factor(geos, factor="carry"))
    assert ranked["BLOWOFF/USDT:USDT"] == -20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sleeves/test_factor.py::test_rank_factor_carry_respects_max_abs_apr -v`
Expected: FAIL — `TypeError: rank_factor() got an unexpected keyword argument 'max_abs_apr'`.

- [ ] **Step 3: Write minimal implementation**

In `futures_fund/sleeves/factor.py`, add the import near the top (after the `from futures_fund.contracts import ...` line):

```python
from futures_fund.funding_intervals import bounded_apr
```

Replace `_factor_score` and `rank_factor` (lines 13-29):

```python
def _factor_score(g: CoinGeometry, factor: str, *, max_abs_apr: float | None = None) -> float:
    if factor == "momentum":
        return g.momentum_20
    if factor == "carry":
        return -bounded_apr(g.funding_apr, max_abs_apr)    # low/negative funding is attractive
    if factor == "low_vol":
        return -g.realized_vol                             # lower vol is attractive
    raise ValueError(f"unknown factor {factor!r}")


def rank_factor(geometries: list[CoinGeometry], *,
                factor: Literal["momentum", "carry", "low_vol"],
                max_abs_apr: float | None = None) -> list[tuple[str, float]]:
    """Cross-sectional ranking score per symbol for the factor, best (highest score) first."""
    scored = [(g.symbol, _factor_score(g, factor, max_abs_apr=max_abs_apr)) for g in geometries]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored
```

Replace `_combined_rank` (lines 32-40):

```python
def _combined_rank(geometries: list[CoinGeometry], factors: list[str],
                   *, max_abs_apr: float | None = None) -> list[tuple[str, float]]:
    """Average rank-position across factors (0 = best). Lower combined value = stronger long."""
    agg: dict[str, float] = {g.symbol: 0.0 for g in geometries}
    for factor in factors:
        for pos, (sym, _score) in enumerate(
                rank_factor(geometries, factor=factor, max_abs_apr=max_abs_apr)):
            agg[sym] += pos
    combined = [(sym, agg[sym] / max(1, len(factors))) for sym in agg]
    combined.sort(key=lambda t: t[1])               # best (lowest avg rank) first
    return combined
```

In `factor_signal`, add the param to the signature (after the `weighting=...` line, line 57) and pass it into `_combined_rank` (the `ranked = _combined_rank(geometries, factors)` line, ~line 63):

```python
                  weighting: Literal["inverse_vol", "equal"] = "inverse_vol",
                  max_abs_apr: float | None = None) -> SleeveSignal:
```

```python
    ranked = _combined_rank(geometries, factors, max_abs_apr=max_abs_apr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sleeves/test_factor.py -v`
Expected: the two new tests PASS, all existing factor tests PASS (default `None`).

Run: `uv run ruff check futures_fund/sleeves/factor.py`
Expected: PASS. (`bounded_apr` imported from the neutral `funding_intervals` module and used — no F401, no factor→carry coupling.)

- [ ] **Step 5: Commit**

```bash
git add futures_fund/sleeves/factor.py tests/sleeves/test_factor.py
git commit -m "feat(factor): mirror carry funding bound in the factor carry leg (neutral helper)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Thread the configured carry cap through `build_sleeves`

**Files:**
- Modify: `futures_fund/cycle_prep.py:109-128` (`build_sleeves`).
- Modify: `scripts/cycle_prep_cli.py` (pass the cap from settings, the `build_sleeves` call on line 70).
- Test: `tests/test_cycle_prep.py` (append; `build_sleeves` and `_CG`/`NOW` already imported/defined in that file).

`build_sleeves` gains `max_abs_apr: float | None = None`, passed into BOTH `carry_signal` and `factor_signal` so the desk's two carry exposures are bounded consistently. The CLI reads it from `settings.sleeves.get("carry", {}).get("max_abs_apr")`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cycle_prep.py` (uses the existing `NOW` and `_CG` from this file):

```python
def test_build_sleeves_threads_carry_cap_into_carry_signal():
    geos = [_CG(symbol=f"{c}/USDT:USDT", mark=100.0, funding_apr=apr)
            for c, apr in zip("ABCDEF", [20.0, 1.0, 0.5, -0.5, -1.0, -20.0], strict=True)]
    sleeves = build_sleeves(geos, pairs=[], spreads=[], now=NOW, max_abs_apr=2.0)
    carry = next(s for s in sleeves if s.sleeve == "carry")
    scores = {t.symbol: t.raw_score for t in carry.tilts}
    # the extreme +20 APR name is shorted with a CAPPED +2.0 raw_score (bounded by the cap)
    assert scores["A/USDT:USDT"] == 2.0
    assert scores["F/USDT:USDT"] == -2.0
    assert carry.diagnostics["max_abs_apr"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cycle_prep.py::test_build_sleeves_threads_carry_cap_into_carry_signal -v`
Expected: FAIL — `TypeError: build_sleeves() got an unexpected keyword argument 'max_abs_apr'`.

- [ ] **Step 3: Write minimal implementation**

In `futures_fund/cycle_prep.py`, replace `build_sleeves` (lines 109-128):

```python
def build_sleeves(
    geometries: list[CoinGeometry],
    pairs: list[Pair],
    spreads: list[Spread],
    *,
    now: datetime,
    max_abs_apr: float | None = None,
) -> list[SleeveSignal]:
    """Run all four alpha sleeves over the geometries (+ pairs/spreads), then assign risk-parity
    budgets across them via `neutrality.risk_parity_budgets` (the contract's single home for the
    budget split). `risk_budget_frac` starts at 0.0 on each sleeve and is filled in place by
    `risk_parity_budgets`, which sums to 1.0 across the four. `max_abs_apr` bounds the extreme-
    funding signal in BOTH carry exposures (the carry sleeve and the factor sleeve's carry leg)."""
    sleeves = [
        carry_signal(geometries, risk_budget_frac=0.0, now=now, max_abs_apr=max_abs_apr),
        pairs_signal(pairs, spreads, risk_budget_frac=0.0, now=now),
        factor_signal(geometries, risk_budget_frac=0.0, now=now, max_abs_apr=max_abs_apr),
        sentiment_factor_signal(geometries, risk_budget_frac=0.0, now=now),
    ]
    budgets = risk_parity_budgets(sleeves)
    return [s.model_copy(update={"risk_budget_frac": budgets[s.sleeve]}) for s in sleeves]
```

In `scripts/cycle_prep_cli.py`, replace the `build_sleeves` call (line 70):

```python
    carry_cap = (settings.sleeves.get("carry") or {}).get("max_abs_apr")
    sleeves = build_sleeves(bundle.geometries, pairs=pairs, spreads=spreads, now=now,
                            max_abs_apr=carry_cap)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cycle_prep.py -q`
Expected: PASS, including the existing `test_build_sleeves_emits_the_four_named_sleeves` (default `None`).

Run: `uv run ruff check futures_fund/cycle_prep.py scripts/cycle_prep_cli.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add futures_fund/cycle_prep.py scripts/cycle_prep_cli.py tests/test_cycle_prep.py
git commit -m "feat(cycle_prep): thread configured carry funding cap into build_sleeves

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Integration test — scout → cycle_prep produces a clean book (no young/pumped names)

**Files:**
- Test: `tests/test_universe_integration.py` (create).

Drive the scout CLI then the cycle-prep CLI through one cycle with a universe that contains a VELVET-like trap, and assert the produced `geometries.json` contains the established names but NOT VELVET, and that the kept names carry real ADV + depth metadata. This pins the scout→cycle_prep boundary end to end. With the depth-floor fix (Task 3), the deep fixtures genuinely CLEAR the floor and VELVET is dropped for the INTENDED reason (young + pumped), not because the gate empties the universe.

Two-fake patching, BOTH must hold: (a) `scripts.scout_cli.build_ccxt` → `_FakeClient` (the ccxt client `scan_universe` reads); (b) `scripts.scout_cli.FuturesExchange` → `_FakeExchange` (the class whose `from_settings` the scout calls), AND `futures_fund.exchange.FuturesExchange.from_settings` → a `_FakeExchange` (the same class object that `scripts.cycle_prep_cli` imports). Patching the class on the scout module and the `from_settings` on the canonical module covers both call sites.

- [ ] **Step 1: Write the failing test**

Create `tests/test_universe_integration.py`:

```python
from __future__ import annotations

import json

import pandas as pd

from futures_fund.cycle_io import cycle_dir, load_output
from futures_fund.market_data import FundingInfo

_NOW_ISO = "2026-06-12T00:00:00+00:00"
_OLD_ONBOARD = "1567965300000"   # 2019 -> well past min_age_days
_ESTABLISHED = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
_MARKS = {"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0,
          "SOL/USDT:USDT": 150.0, "VELVET/USDT:USDT": 1.0}


class _FakeClient:
    markets = {
        **{s: {"info": {"underlyingType": "COIN", "onboardDate": _OLD_ONBOARD}}
           for s in _ESTABLISHED},
        # VELVET: future onboardDate (too young) -> dropped by the age gate
        "VELVET/USDT:USDT": {"info": {"underlyingType": "COIN", "onboardDate": "9999999999999"}},
    }

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {
            **{s: {"last": _MARKS[s], "quoteVolume": 1e9, "percentage": 1.0}
               for s in _ESTABLISHED},
            "VELVET/USDT:USDT": {"last": 1.0, "quoteVolume": 9e8, "percentage": 130.0},
        }


class _FakeExchange:
    def depth(self, symbol, limit=20):
        mark = _MARKS[symbol]
        qty = 5_000_000.0 / mark                 # ~$5M/level -> full notional >> min_depth_usd
        return {"bids": [(mark * 0.999, qty)], "asks": [(mark * 1.001, qty)]}

    def ohlcv(self, symbol, timeframe="4h", limit=500):
        ts = pd.date_range("2025-01-01", periods=60, freq="4h", tz="UTC")
        c = _MARKS[symbol]
        return pd.DataFrame({"timestamp": ts, "open": c, "high": c,
                             "low": c, "close": c, "volume": 1.0})

    def funding(self, symbol):
        return FundingInfo(symbol=symbol, current_rate=0.0001,
                           next_funding_ts=pd.Timestamp(_NOW_ISO).to_pydatetime(),
                           interval_hours=8.0, mark_price=_MARKS[symbol],
                           index_price=_MARKS[symbol])

    def mark_price(self, symbol):
        return _MARKS[symbol]

    @staticmethod
    def from_settings(settings):
        return _FakeExchange()


def test_scout_to_cycle_prep_excludes_young_and_pumped(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setattr("scripts.scout_cli.build_ccxt", lambda settings: _FakeClient())
    monkeypatch.setattr("scripts.scout_cli.FuturesExchange", _FakeExchange)
    monkeypatch.setattr("futures_fund.exchange.FuturesExchange.from_settings",
                        staticmethod(lambda settings: _FakeExchange()))

    from scripts.cycle_prep_cli import main as cycle_prep_main
    from scripts.scout_cli import main as scout_main

    scout_main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state), "--top", "30"])
    universe = json.loads(
        (cycle_dir(state, 1, cadence="weekly") / "universe.json").read_text())["universe"]
    syms = [r["symbol"] for r in universe]
    assert set(syms) == set(_ESTABLISHED)              # VELVET dropped (young + pumped)

    cycle_prep_main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state),
                     "--now", _NOW_ISO])
    geos = load_output(state, 1, "geometries", cadence="weekly")["geometries"]
    geo_syms = [g["symbol"] for g in geos]
    assert "VELVET/USDT:USDT" not in geo_syms
    assert set(geo_syms) == set(_ESTABLISHED)
    # honest-cost prerequisites are stamped: real ADV (not 0.0) and a non-empty crossing book
    btc = next(g for g in geos if g["symbol"] == "BTC/USDT:USDT")
    assert btc["adv_usd"] == 1e9
    assert btc["depth_asks"] and btc["depth_bids"]
    assert btc["onboard_date"] == int(_OLD_ONBOARD)
```

- [ ] **Step 2: Run test to verify it passes (it is a GUARD; Tasks 2-7 must be complete)**

Run: `uv run pytest tests/test_universe_integration.py -v`
Expected: PASS if Tasks 2-7 are complete — it is an integration GUARD, not a new-feature test. If it FAILS, the failure pinpoints a wiring gap (`adv_usd == 0.0` → universe-row ADV not threaded; `VELVET` present → age/mover gate not applied; empty `depth_asks` → `_safe_depth`/`from_settings` not patched-through). Fix the indicated task's wiring; do not weaken the assertion. With the Task-3 depth fix, VELVET is dropped by the age+mover gates and the established names CLEAR the depth floor (the failure can no longer be "empty universe").

- [ ] **Step 3: Run to verify it passes**

Run: `uv run pytest tests/test_universe_integration.py -v`
Expected: PASS.

Run: `uv run ruff check tests/test_universe_integration.py`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_universe_integration.py
git commit -m "test: scout->cycle_prep integration excludes young/pumped, stamps ADV+depth

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Full-suite green + ruff clean (final gate)

**Files:** none (verification + any fixture touch-ups surfaced here).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: ALL tests PASS, including `tests/test_end_to_end_no_seed.py` (the no-seed E2E), the carry/factor sleeve tests, slippage, account integration, and the new Phase 10 test modules.

If a previously-green test broke because it admitted a pumped/young/thin name or pinned the old flat-1bps slippage, FIX THE FIXTURE (give the fake an old `onboardDate`, a deep `.depth()`, a calm `percentage`), never relax a Phase 10 assertion. The only intentionally-changed behaviors are: (1) which names enter the universe, (2) thin-name slippage being larger than BTC's and larger than 1bps, (3) extreme-funding carry being bounded. Any other test that changed value is a regression to investigate, not a fixture to loosen.

- [ ] **Step 2: Run ruff across the whole change surface**

Run: `uv run ruff check futures_fund scripts tests`
Expected: PASS (no output). If `UP`/`B`/`I` fires, apply the suggested fix (e.g. `ruff check --fix` for import sorting `I`), then re-run.

- [ ] **Step 3: Final commit (only if Steps 1-2 produced fixes)**

```bash
git add -A
git commit -m "test: keep full suite green + ruff clean for Phase 10 universe/slippage/carry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

If Steps 1-2 produced NO changes, skip this commit.

---

## Self-Review

**Review-issue closure (the 12 issues this revision fixes):**
1. **Depth-floor math** — `_book_depth_usd` now returns the FULL summed top-N notional (no `depth_ref_usd` cap; the `if acc >= depth_ref_usd: return depth_ref_usd` early return is deleted); the floor compares the thinner side vs `min_depth_usd` only; `depth_ref_usd` is decoupled (slippage-model documentation, with `_ = depth_ref_usd` acknowledging it). Deep fixtures ($500k/$1M/$5M per level) CLEAR `min_depth_usd=250k`; the $1k thin book fails it. Verified in Task 3 (`test_velvet_excluded_majors_included`, `test_depth_floor_excludes_thin_book`) and the Task 4/9/13 deep fakes. The "empty universe" failure mode is gone, so Tasks 3/4/9/13 RED steps fail for the INTENDED reason (closes issue 12 too).
2. **`tests/test_config.py` `Settings` import** — Task 1 Step 1 explicitly ADDS `Settings` to the existing `from futures_fund.config import (...)` block BEFORE appending tests (the file already exists, 8183 bytes, importing only `DataSettings, ExchangeSettings, LoopSettings, _default_loops, load_env_file, load_settings`).
3. **`FuturesExchange.depth` has callers / `tests/test_exchange.py` exists** — grounding brief corrected (depth has two callers at `test_exchange.py:78/86`); Task 6 APPENDS to the existing file, does NOT re-import `FuturesExchange` (already line 3), and uses the real `FuturesExchange(_FakeClient, keyless=True)` constructor (matching the file's style) rather than `__new__`.
4. **`_age_days` young-name fallback** — Task 3 adds `test_age_falls_back_to_klines_and_drops_a_genuinely_young_name` with an earliest kline at `_NOW - 5 days`, asserting it is DROPPED; the old "200 candles ≈ 33 days" reasoning is removed (the kept-old case is renamed `..._keeps_old` with honest reasoning).
5. **`_FakeQualityExchange` definition order** — Task 4 Step 1 inserts the fake classes IMMEDIATELY AFTER `_FakeClient` (ABOVE the original test), eliminating the forward-reference fragility; the note explains module-level names resolve at call time.
6. **Scout exchange construction (two independent fakes)** — Task 4 patches `scripts.scout_cli.build_ccxt` (client) AND `scripts.scout_cli.FuturesExchange` (exchange) separately; the new test asserts kept rows carry `onboard_date`/`chg_24h_pct`/`vol_24h_usd` metadata, which a silently-unpatched real-network `from_settings` could not produce. Task 13 uses the matched two-fake style and the same metadata assertion.
7. **Task 9 ordering dependency** — Task 9 explicitly states the Task 2 (row) → Task 3 (gate) → Task 9 (fixture) chain and that, with Task 2 shipped, the age gate uses `onboard_date` (not the kline fallback) for these names.
8. **Over-depth under-costing caveat** — Task 8 documents it in the `CostInputs` docstring AND pins it with `test_over_depth_clip_is_under_costed_documented` (full target qty opened = 500 units, slippage priced on the 10-unit partial = 100 USDT), so the number is a floor, not exact.
9. **Config-key path + YAML nesting** — Task 1 Step 4 gives the EXACT insertion point (after `pairs:`'s `rolling_retest_cycles: 7`, before the `# --- sentiment ---` comment) and 2-space indent so `carry:` is a SIBLING of `factor:`/`pairs:`; `test_repo_config_yaml_carry_cap_is_nested_correctly` loads the REPO `config.yaml` and asserts `sleeves["carry"]["max_abs_apr"] == 2.0` AND `carry` is not nested under `factor`.
10. **Import-cycle / private-helper coupling** — `bounded_apr` is hoisted into the NEUTRAL `funding_intervals.py` (which imports only `futures_fund.models`); both `carry.py` and `factor.py` import it from there, so there is no factor→carry dependency and no cycle. Task 10 adds a direct `bounded_apr` test.
11. **Task 7 line-range drift / dangling `_symbols`** — `cycle_prep_cli` `main` replacement redefines `symbols`/`rows_by_sym` and fully removes the old `symbols = _symbols(...)` line (line 63), with an explicit "verify the OLD line is removed" note; `_symbols`→`_universe_rows` is internal (no test imports the function; `test_cycle_prep_cli.py`'s `self._symbols` is an unrelated fake attribute). `build_geometries` signature is 66-73, append block 94-105 (verified against the repo).
12. **TDD red/green honesty** — every Step-2 "Expected: FAIL" is re-derived post-depth-fix: Task 4 expects `AttributeError ... FuturesExchange` (not empty-universe), Task 13 is a GUARD that passes once 2-7 land, Task 9 captures the fail-soft pre-state explicitly.

**Spec coverage:**
- (1) Universe quality filter — min listing age (onboardDate w/ kline-age fallback): Tasks 2,3,4,6 + `quality_filter._age_days`. Exclude extreme 24h movers: Task 3 `chg_24h` gate. Order-book depth floor via `exchange.depth()` (FULL notional vs `min_depth_usd`): Task 3 `depth` gate. Existing ADV floor retained: Task 3 `adv` gate. All thresholds in `config.yaml universe`: Task 1. Scout writes CLEAN `universe.json` + per-filter drop summary: Task 4. cycle_prep reads it: Task 7 (`_universe_rows`). ✔
- (2) Depth-aware slippage wired into fills — `CoinGeometry.depth_bids/asks` + real `adv_usd` (Tasks 5,7), threaded through `_geometry_cost_maps → CostInputs → apply_fills → estimate_slippage` (Task 8), direction-correct side selection by `delta_signed_qty` sign (Task 8), half-spread from observed top-of-book (Task 8), sane fallback when depth missing (empty books → ADV fallback, Tasks 7,8), documented over-depth caveat (Task 8). Thin name materially > BTC and > 1bps: Task 8. ✔
- (3) Carry bound — neutral `bounded_apr` (Task 10), `max_abs_apr` cap clamping the carry sleeve (Task 10) and the factor carry leg (Task 11), configurable via `sleeves.carry.max_abs_apr` (Task 1) threaded through `build_sleeves` (Task 12), documented as "extreme funding is a reversal trap, not free alpha". The per-symbol realized clamp (`clamp_funding_rate`) is the existing upstream layer this sits ON TOP of. ✔
- (4) Tests — VELVET excluded / majors included: Tasks 3,13. Depth slippage thin > BTC > 1bps: Task 8. Carry signal bounded vs unbounded baseline: Tasks 10,11,12. Integration scout→cycle_prep clean book: Task 13. Full suite incl. no-seed E2E green: Tasks 9,14. ✔

**Placeholder scan:** every code step shows complete code; every test step shows the assertion body; every command shows the exact `uv run pytest`/`ruff` invocation and expected PASS/FAIL. No TBD/TODO. ✔

**Type consistency:** `quality_filter` returns `(list[dict], dict[str,int])` (Tasks 3,4). `CoinGeometry.depth_bids/depth_asks` (Task 5) match `CostInputs.depth_bids/depth_asks` (Task 8), the `_geometry_cost_maps` keys `g["depth_bids"]/g["depth_asks"]` (Task 8), and the build-time stamp (Task 7). `max_abs_apr: float | None` is identical across `bounded_apr` (Task 10), `carry_signal` (Task 10), `rank_factor`/`_factor_score`/`_combined_rank`/`factor_signal` (Task 11), `build_sleeves` (Task 12). `onboard_date` (int ms | None) is consistent across `scan_universe` row (Task 2), `CoinGeometry` (Task 5), `onboard_date_ms` (Task 6), and `quality_filter._age_days` (Task 3). `_universe_rows` (Task 7) replaces `_symbols` and consumers read `[r["symbol"] for r in ...]`. ✔

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-12-phase10-universe-realism.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

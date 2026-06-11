from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt
from futures_fund.scheduling import floor_tf

NOW = datetime(2026, 6, 11, tzinfo=UTC)


@pytest.fixture
def write_served_report():
    """Seed a completed cycle's report.json that SERVES the candle containing `served`.

    The cadence due-gate (`scheduling.cycle_due`) keys off report['candle'] = floor_tf(gate-start),
    so a report carrying `candle == floor_tf(served, tf_minutes)` marks that candle as already
    served and forces SKIP for any `now` inside it. `cycle_dir` is the FULL cycle directory the gate
    scans (e.g. state/daily/cycle/1) — the writer and the due-gate reader share this one root."""

    def _write(cycle_dir, *, served: datetime, tf_minutes: int) -> Path:
        candle = floor_tf(served, tf_minutes)
        d = Path(cycle_dir)
        d.mkdir(parents=True, exist_ok=True)
        report = d / "report.json"
        report.write_text(
            json.dumps(
                {
                    "candle": candle.isoformat(),
                    "ran_at": served.isoformat(),
                }
            )
        )
        return report

    return _write


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
            SleeveTilt(
                symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5, raw_score=-1.0
            ),
        ],
        risk_budget_frac=0.5,
        as_of_ts=NOW,
    )
    carry = SleeveSignal(
        sleeve="carry",
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5, raw_score=0.5),
            SleeveTilt(
                symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5, raw_score=-0.8
            ),
        ],
        risk_budget_frac=0.5,
        as_of_ts=NOW,
    )
    return [factor, carry]

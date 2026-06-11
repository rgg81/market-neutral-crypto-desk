# futures_fund/cycle_prep.py
"""Cycle-prep producer (Phase 8): turn exchange reads into the EXACT upstream artifacts the
control loop and reviewer consume — `GeometryBundle`, `SleeveSignal[]`, `Pair[]`, `Spread[]`.

Closes the C1 gap (alpha engine not wired to the loop's input artifacts): before Phase 8 only the
e2e test's `_seed_upstream` fixture produced these, so the desk could not build a book from market
data without hand-seeded inputs. Pure functions over a duck-typed `FuturesExchange` (the e2e fakes
it); they NEVER persist — `cycle_prep_cli.py` owns persistence.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from futures_fund.beta import beta_for_symbols
from futures_fund.contracts import CoinGeometry, GeometryBundle
from futures_fund.funding_intervals import (
    clamp_funding_rate,
    funding_apr,
    funding_interval_hours,
)


def _marks_frame(exchange, symbols: list[str]) -> dict[str, pd.Series]:
    """Per-symbol close-price series from `exchange.ohlcv` (for beta + realized vol)."""
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = exchange.ohlcv(sym)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        out[sym] = df["close"].astype(float).reset_index(drop=True)
    return out


def _realized_vol(series: pd.Series) -> float:
    """Annualized realized vol from log returns (×sqrt(365*6) for 4h candles); 0.0 if too short."""
    rets = (series / series.shift(1)).dropna()
    if len(rets) < 2:
        return 0.0
    log_r = np.log(rets.to_numpy())
    return float(np.std(log_r, ddof=1) * (365.0 * 6.0) ** 0.5)


def _momentum_20(series: pd.Series) -> float:
    """20-period close-to-close momentum; 0.0 if too short."""
    if len(series) <= 20:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-21] - 1.0)


def build_geometries(
    exchange,
    symbols: list[str],
    *,
    now: datetime,
    btc_symbol: str = "BTC/USDT:USDT",
    beta_lookback: int = 45,
) -> GeometryBundle:
    """One `CoinGeometry` per symbol from live (or faked) exchange reads.

    beta_btc <- beta.beta_for_symbols (BTC self-beta 1.0); funding_rate <- clamp_funding_rate of
    the per-symbol signed rate; funding_interval_hours <- funding_intervals.funding_interval_hours;
    funding_apr <- the SIGNED annualized carry (carry credit stays visible, §6.1). Fail-soft: a
    symbol whose reads error is skipped, never crashes the bundle.
    """
    marks_by_symbol = _marks_frame(exchange, symbols)
    betas = beta_for_symbols(marks_by_symbol, btc_symbol=btc_symbol, lookback=beta_lookback)
    geometries: list[CoinGeometry] = []
    for sym in symbols:
        series = marks_by_symbol.get(sym)
        try:
            fi = exchange.funding(sym)
            raw_rate = float(fi.current_rate)
            mark = float(fi.mark_price)
        except Exception:
            continue
        interval = funding_interval_hours(sym, exchange)
        rate = clamp_funding_rate(sym, raw_rate)
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
            adv_usd=0.0,
        ))
    return GeometryBundle(geometries=geometries, as_of_ts=now)

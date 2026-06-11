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
        next_funding_ts=datetime.fromtimestamp(
            fr["fundingTimestamp"] / 1000, tz=timezone.utc),  # noqa: UP017
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

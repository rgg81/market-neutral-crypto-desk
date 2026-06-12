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

    def onboard_date_ms(self, symbol: str) -> int | None:
        """Binance listing timestamp (ms-epoch) from the ccxt-cached market info, or None.

        ccxt exposes onboardDate only via market(sym)["info"]["onboardDate"] (a string) after
        load_markets(); fail-soft to None so cycle_prep can fall back to the earliest-kline age."""
        try:
            raw = (self.client.market(symbol).get("info") or {}).get("onboardDate")
            return int(raw) if raw is not None else None
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

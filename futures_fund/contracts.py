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

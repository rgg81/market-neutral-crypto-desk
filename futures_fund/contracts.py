from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from futures_fund.models import Direction, SentimentLevel, SleeveName, SymbolSpec


class SentimentSource(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict-by-default (canonical contract PART 1)
    url: str
    published_ts: datetime          # MUST be < owning report's as_of_ts (point-in-time)
    title: str = ""
    feed: str = ""                  # "news_rss" | "reddit" | "fear_greed" | "media"


class SentimentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict-by-default (canonical contract PART 1)
    symbol: str                     # ccxt unified id, or "MARKET" for the market-wide read
    level: SentimentLevel
    s: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SentimentSource] = Field(default_factory=list)
    rationale: str = ""
    as_of_ts: datetime              # decision-time anchor; all sources must precede this
    decayed_s: float | None = None  # s after half-life decay toward 0 (filled by ingest)


class SentimentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict-by-default (canonical contract PART 1)
    reports: list[SentimentReport] = Field(default_factory=list)


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

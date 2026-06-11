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
    """Per-cadence candle + model tier for the two-cadence desk (weekly select / daily rebal)."""
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

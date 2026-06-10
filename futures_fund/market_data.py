from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


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

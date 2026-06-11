"""Alpha-sleeve signal generators: each emits a SleeveSignal of per-name tilts the optimizer
merges into one neutral book. The four sleeves are carry, pairs, factor, and sentiment.
"""
from __future__ import annotations

from futures_fund.sleeves.carry import carry_signal
from futures_fund.sleeves.factor import factor_signal
from futures_fund.sleeves.pairs import pairs_signal
from futures_fund.sleeves.sentiment import (
    apply_conviction_tilts,
    conviction_tilt,
    sentiment_factor_signal,
)

__all__ = [
    "carry_signal",
    "pairs_signal",
    "factor_signal",
    "sentiment_factor_signal",
    "conviction_tilt",
    "apply_conviction_tilts",
]

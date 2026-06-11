"""Cross-sectional factor L/S sleeve: rank liquid names by momentum / carry / low-vol; long the
top tercile, short the bottom tercile, inverse-vol weighted within each leg (design spec §6.3).
"""
from __future__ import annotations

from typing import Literal

from futures_fund.contracts import CoinGeometry


def _factor_score(g: CoinGeometry, factor: str) -> float:
    if factor == "momentum":
        return g.momentum_20
    if factor == "carry":
        return -g.funding_apr               # low/negative funding is attractive
    if factor == "low_vol":
        return -g.realized_vol              # lower vol is attractive
    raise ValueError(f"unknown factor {factor!r}")


def rank_factor(geometries: list[CoinGeometry], *,
                factor: Literal["momentum", "carry", "low_vol"]) -> list[tuple[str, float]]:
    """Cross-sectional ranking score per symbol for the factor, best (highest score) first."""
    scored = [(g.symbol, _factor_score(g, factor)) for g in geometries]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

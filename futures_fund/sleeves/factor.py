"""Cross-sectional factor L/S sleeve: rank liquid names by momentum / carry / low-vol; long the
top tercile, short the bottom tercile, inverse-vol weighted within each leg (design spec §6.3).
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt


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


def _combined_rank(geometries: list[CoinGeometry], factors: list[str]) -> list[tuple[str, float]]:
    """Average rank-position across factors (0 = best). Lower combined value = stronger long."""
    agg: dict[str, float] = {g.symbol: 0.0 for g in geometries}
    for factor in factors:
        for pos, (sym, _score) in enumerate(rank_factor(geometries, factor=factor)):
            agg[sym] += pos
    combined = [(sym, agg[sym] / max(1, len(factors))) for sym in agg]
    combined.sort(key=lambda t: t[1])               # best (lowest avg rank) first
    return combined


def _inverse_vol_weights(syms: list[str], geo_by_sym: dict[str, CoinGeometry],
                         weighting: str) -> dict[str, float]:
    if weighting == "equal" or not syms:
        w = 1.0 / len(syms) if syms else 0.0
        return {s: w for s in syms}
    inv = {s: 1.0 / max(geo_by_sym[s].realized_vol, 1e-6) for s in syms}
    total = sum(inv.values())
    return {s: inv[s] / total for s in syms}


def factor_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                  factors: list[str] = ["momentum", "carry", "low_vol"],  # noqa: B006
                  tercile: float = 1 / 3,
                  weighting: Literal["inverse_vol", "equal"] = "inverse_vol") -> SleeveSignal:
    """Long top tercile / short bottom tercile of the combined factor rank; inverse-vol (or equal)
    within each leg. target_weight is the signed within-side share (long > 0, short < 0)."""
    n = len(geometries)
    if n == 0:
        return SleeveSignal(sleeve="factor", tilts=[], risk_budget_frac=risk_budget_frac,
                            as_of_ts=now)
    geo_by_sym = {g.symbol: g for g in geometries}
    ranked = _combined_rank(geometries, factors)
    k = max(1, math.floor(n * tercile))
    long_syms = [s for s, _ in ranked[:k]]
    short_syms = [s for s, _ in ranked[-k:]]
    long_w = _inverse_vol_weights(long_syms, geo_by_sym, weighting)
    short_w = _inverse_vol_weights(short_syms, geo_by_sym, weighting)
    tilts: list[SleeveTilt] = []
    for s in long_syms:
        tilts.append(SleeveTilt(symbol=s, direction="long", target_weight=long_w[s]))
    for s in short_syms:
        tilts.append(SleeveTilt(symbol=s, direction="short", target_weight=-short_w[s]))
    return SleeveSignal(sleeve="factor", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"factors": factors, "k_per_side": k}, as_of_ts=now)

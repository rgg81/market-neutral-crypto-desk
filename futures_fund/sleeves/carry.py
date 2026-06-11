"""Funding-carry sleeve: rank the cross-section by SIGNED funding_apr; long the low/negative-
funding names (their shorts PAY us), short the high-positive-funding names (crowded longs).

Carry credit is UNCLAMPED and signed everywhere (unlike the inherited risk_gate clamp), so a
favorable funding edge is visible to the optimizer/gate (design spec §6.1).
"""
from __future__ import annotations

import math
from datetime import datetime

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt


def carry_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                 top_frac: float = 1 / 3) -> SleeveSignal:
    """Long low/negative funding_apr, short high-positive funding_apr, delta-hedged.

    raw_score carries the signed un-clamped funding_apr; target_weight is the per-leg signed share
    of the side budget (long > 0, short < 0), equal-weight within each side (pre-optimize).
    k = max(1, floor(n * top_frac)) names per side (top_frac is a tercile-style fraction).
    """
    ranked = sorted(geometries, key=lambda g: g.funding_apr)   # ascending: most negative first
    n = len(ranked)
    if n == 0:
        return SleeveSignal(sleeve="carry", tilts=[], risk_budget_frac=risk_budget_frac,
                            as_of_ts=now)
    k = max(1, math.floor(n * top_frac))
    longs = ranked[:k]                                          # lowest/negative funding -> LONG
    shorts = ranked[-k:]                                        # highest funding -> SHORT
    long_w = 1.0 / k
    short_w = 1.0 / k
    tilts: list[SleeveTilt] = []
    for g in longs:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="long",
                                target_weight=long_w, raw_score=g.funding_apr))
    for g in shorts:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="short",
                                target_weight=-short_w, raw_score=g.funding_apr))
    return SleeveSignal(sleeve="carry", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"n_candidates": n, "k_per_side": k}, as_of_ts=now)

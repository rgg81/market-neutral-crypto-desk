"""Funding-carry sleeve: rank the cross-section by SIGNED funding_apr; long the low/negative-
funding names (their shorts PAY us), short the high-positive-funding names (crowded longs).

Carry credit is UNCLAMPED and signed everywhere (unlike the inherited risk_gate clamp), so a
favorable funding edge is visible to the optimizer/gate (design spec §6.1).
"""
from __future__ import annotations

import math
from datetime import datetime

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt
from futures_fund.funding_intervals import bounded_apr


def carry_signal(geometries: list[CoinGeometry], *, risk_budget_frac: float, now: datetime,
                 top_frac: float = 1 / 3, max_abs_apr: float | None = None) -> SleeveSignal:
    """Long low/negative funding_apr, short high-positive funding_apr, delta-hedged.

    raw_score carries the signed funding_apr (bounded to +-max_abs_apr when set; see
    funding_intervals.bounded_apr — extreme funding is a reversal trap, not free alpha);
    target_weight is the per-leg signed share of the side budget (long > 0, short < 0),
    equal-weight within each side (pre-optimize). k = max(1, floor(n * top_frac)) names per side.
    """
    scored = [(g, bounded_apr(g.funding_apr, max_abs_apr)) for g in geometries]
    ranked = sorted(scored, key=lambda gs: gs[1])              # ascending: most negative first
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
    for g, score in longs:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="long",
                                target_weight=long_w, raw_score=score))
    for g, score in shorts:
        tilts.append(SleeveTilt(symbol=g.symbol, direction="short",
                                target_weight=-short_w, raw_score=score))
    return SleeveSignal(sleeve="carry", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"n_candidates": n, "k_per_side": k,
                                     "max_abs_apr": max_abs_apr}, as_of_ts=now)

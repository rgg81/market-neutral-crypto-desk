"""Sentiment sleeve + per-coin conviction tilt (design spec §6.4, §7.2).

Two bounded shapers that NEVER flip direction and never open a position alone:
  1. conviction_tilt / apply_conviction_tilts — deterministic per-leg tilt within a +-cap band.
  2. sentiment_factor_signal — a standalone cross-sectional L/S sleeve on sentiment_score*conf.

Both run BEFORE the optimizer re-projects onto the dollar+beta-neutral set, so sentiment cannot
mathematically break neutrality or the risk gate (computed after).

This module is the canonical home for ``conviction_tilt``/``apply_conviction_tilts`` (canonical
interface contract §2.9); ``neutrality.py`` re-imports them so there is exactly one definition.
"""
from __future__ import annotations

from futures_fund.contracts import CoinGeometry, SleeveTilt


def conviction_tilt(
    weight: float,
    sentiment_score: float,
    sentiment_conf: float,
    *,
    kappa: float = 0.5,
    cap: float = 0.25,
) -> float:
    """Deterministic sentiment tilt on the leg MAGNITUDE: |w| <- |w|*(1 + kappa*s_dir*conf),
    where s_dir = sign(w)*s aligns the score with the leg's OWN direction (spec §7.2). This
    "favors the long when positive / the short when negative": positive sentiment GROWS a long
    and negative sentiment GROWS a short, so the cross-sectional sentiment ordering invariant
    holds on BOTH sides (more-bullish => stronger long, more-bearish => stronger short).

    |delta w| is clamped to cap*|w|. NEVER flips sign, never opens a position alone (returns 0
    if weight is 0). Applied BEFORE the optimizer re-projection (sentiment cannot break
    neutrality)."""
    if weight == 0.0:
        return 0.0
    sign = 1.0 if weight > 0.0 else -1.0
    # s_dir is the score signed by the leg's direction: short legs read a bearish (s<0) score
    # as favorable, so the multiplier grows the short rather than shrinking it.
    s_dir = sign * sentiment_score
    delta = weight * (kappa * s_dir * sentiment_conf)
    max_delta = cap * abs(weight)
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    tilted = weight + delta
    # never flip sign
    if (weight > 0 and tilted < 0) or (weight < 0 and tilted > 0):
        return 0.0
    return tilted


def apply_conviction_tilts(
    legs: list[SleeveTilt],
    geometries: list[CoinGeometry],
    *,
    kappa: float = 0.5,
    cap: float = 0.25,
) -> list[SleeveTilt]:
    """Map conviction_tilt over legs using each symbol's geometry; sign-preserving and
    cap-respecting. Symbols without geometry are returned unchanged."""
    geo = {g.symbol: g for g in geometries}
    out: list[SleeveTilt] = []
    for leg in legs:
        g = geo.get(leg.symbol)
        if g is None:
            out.append(leg)
            continue
        tilted = conviction_tilt(
            leg.target_weight, g.sentiment_score, g.sentiment_conf, kappa=kappa, cap=cap
        )
        out.append(leg.model_copy(update={"target_weight": tilted}))
    return out

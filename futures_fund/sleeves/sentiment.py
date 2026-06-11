"""Sentiment sleeve + per-coin conviction tilt (design spec §6.4, §7.2).

Two bounded shapers that NEVER flip direction and never open a position alone:
  1. conviction_tilt / apply_conviction_tilts — deterministic per-leg tilt within a +-cap band.
  2. sentiment_factor_signal — a standalone cross-sectional L/S sleeve on sentiment_score*conf.

Both run BEFORE the optimizer re-projects onto the dollar+beta-neutral set, so sentiment cannot
mathematically break neutrality or the risk gate (computed after).
"""
from __future__ import annotations


def conviction_tilt(weight: float, sentiment_score: float, sentiment_conf: float, *,
                    kappa: float = 0.5, cap: float = 1.0) -> float:
    """Deterministic MAGNITUDE tilt: |w| <- |w|*(1 + kappa*sign(w)*s*conf).

    ``sign(w)`` aligns sentiment with the leg's direction so the tilt FAVORS the long when s>0 and
    the short when s<0 (canonical interface contract §7.2; the earlier scalar
    ``w*(1 + kappa*s*conf)`` form was wrong-for-shorts and is superseded). The fractional tilt term
    ``kappa*sign(w)*s*conf`` is clamped to ``[-cap, +cap]`` so ``|delta w| <= cap*|w|``.

    NEVER flips sign (a tilted long stays >= 0, a tilted short stays <= 0) and NEVER opens a
    position alone (returns 0 if the input weight is 0).
    """
    if weight == 0.0:
        return 0.0
    sign = 1.0 if weight > 0.0 else -1.0
    tilt = kappa * sign * sentiment_score * sentiment_conf
    tilt = max(-cap, min(cap, tilt))                    # |delta w| <= cap*|w|
    tilted = weight * (1.0 + tilt)
    # sign-preserving guard (tilt is clamped to [-cap, cap]; for cap <= 1 the sign already holds,
    # this makes the invariant explicit and robust to cap > 1)
    if weight > 0.0:
        return max(0.0, tilted)
    return min(0.0, tilted)

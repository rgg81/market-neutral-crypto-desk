from __future__ import annotations

import math

from futures_fund.costs import slippage_cost

DEFAULT_K: float = 0.1   # sqrt-impact coefficient for the ADV fallback (config.yaml slippage.k)


def depth_slippage(
    levels: list[tuple[float, float]], qty: float, reference_price: float
) -> float:
    """Thin wrapper over costs.slippage_cost against an L2 depth snapshot (USDT cost).

    Direction-symmetric: `levels` are the crossing side of the book (asks to buy, bids to sell).
    """
    return slippage_cost(levels, qty, reference_price)


def fallback_slippage(
    notional: float, adv_usd: float, half_spread_bps: float, *, k: float = DEFAULT_K
) -> float:
    """half_spread + k*sqrt(notional/ADV) impact model in USDT when no depth snapshot.

    Strictly increasing in notional (a larger clip costs more bps); never returns a flat 2 bps.
    The k*sqrt term is a √-impact law, so it grows ~sqrt(notional); the per-bp cost is therefore
    monotone in size. (Spec §11's two approximate anchors are not both satisfiable by a pure
    √-law; see test_slippage.py — the $1M anchor is pinned, the $5M point is pinned for
    monotonicity, and no 'calibrated to both anchors' property is claimed here.)
    """
    notional = abs(notional)
    if adv_usd <= 0:
        impact_bps = 0.0
    else:
        impact_bps = k * math.sqrt(notional / adv_usd) * 1e4
    cost_bps = half_spread_bps + impact_bps
    return cost_bps / 1e4 * notional


def estimate_slippage(
    symbol: str, qty: float, reference_price: float, *,
    depth: list[tuple[float, float]] | None, adv_usd: float,
    half_spread_bps: float, k: float = DEFAULT_K,
) -> float:
    """Prefer depth_slippage; fall back to fallback_slippage.

    NEVER flat 2 bps. Returns USDT cost.
    """
    if depth:
        return depth_slippage(depth, qty, reference_price)
    notional = abs(qty) * reference_price
    return fallback_slippage(notional, adv_usd, half_spread_bps, k=k)


def slippage_bps(cost_usdt: float, notional: float) -> float:
    """Convenience: cost in bps of notional (for the §11 calibration / monotonicity assertions)."""
    if notional <= 0:
        return 0.0
    return cost_usdt / notional * 1e4

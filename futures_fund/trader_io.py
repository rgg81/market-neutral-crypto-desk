"""Reconstruct gate-ready `TradeProposal`s from an optimizer `TargetWeights` book.

The Trader does NO sizing — notional comes from the optimizer (`WeightLeg.target_notional`).
But the every-cycle reviewer's `check_rr_after_costs` (and the live risk gate) require a full
`TradeProposal` with entry/stop/take_profit geometry to re-derive reward:risk. This module
maps each non-flat book leg + its `CoinGeometry` mark into a `TradeProposal` whose stop sits
`stop_frac` away on the loss side and whose nearest take-profit sits `rr*stop_frac` away on the
gain side — so the re-derived RR equals `rr` and the MIN_RR>=2 floor is actually enforced on the
real book (the prior `target_notional`-only proposal shape made the RR check pass vacuously).
"""
from __future__ import annotations

from futures_fund.contracts import CoinGeometry, TargetWeights
from futures_fund.models import TradeProposal


def proposals_from_book(
    book: TargetWeights,
    geometries: list[CoinGeometry],
    *,
    rr: float = 2.0,
    stop_frac: float = 0.02,
    horizon_hours: float = 168.0,
) -> list[TradeProposal]:
    """One `TradeProposal` per non-flat alpha/hedge leg, sized off the leg's mark.

    Zero-notional legs (carry-over unwinds/flattens) are excluded — there is nothing to OPEN.
    A leg with no matching geometry mark is skipped (cannot place geometric stops without a mark).
    stop = entry*(1 -/+ stop_frac); nearest take_profit = entry*(1 +/- rr*stop_frac), so
    `risk_gate._reward_risk` re-derives exactly `rr`.
    """
    geo = {g.symbol: g for g in geometries}
    out: list[TradeProposal] = []
    for leg in book.legs:
        if abs(leg.target_notional) <= 0.0:
            continue
        g = geo.get(leg.symbol)
        if g is None or g.mark <= 0.0:
            continue
        entry = g.mark
        if leg.direction == "long":
            stop = entry * (1.0 - stop_frac)
            tp = entry * (1.0 + rr * stop_frac)
        else:
            stop = entry * (1.0 + stop_frac)
            tp = entry * (1.0 - rr * stop_frac)
        out.append(TradeProposal(
            symbol=leg.symbol,
            direction=leg.direction,
            entry=entry,
            stop=stop,
            take_profits=[tp],
            atr=entry * stop_frac,
            confidence=0.6,
            horizon_hours=horizon_hours,
            funding_rate=g.funding_rate,
            funding_interval_hours=g.funding_interval_hours,
        ))
    return out

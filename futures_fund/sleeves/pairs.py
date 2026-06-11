"""Cointegration-pairs sleeve: emit per-leg tilts for active Pairs, sized by the cointegrating
hedge ratio so the SPREAD is the traded unit. PnL is attributed at the pair level (Spread).
"""
from __future__ import annotations

from datetime import datetime

from futures_fund.contracts import Pair, SleeveSignal, SleeveTilt, Spread


def select_pairs(candidates: list[Pair], *, adf_pvalue_max: float = 0.05) -> list[Pair]:
    """Keep pairs whose FDR-corrected ADF p is < adf_pvalue_max AND that are still cointegrated
    (rolling re-test passed). A pair with no adf_pvalue_adj yet is dropped (conservative)."""
    out: list[Pair] = []
    for p in candidates:
        if p.adf_pvalue_adj is None:
            continue
        if p.adf_pvalue_adj < adf_pvalue_max and p.cointegrated:
            out.append(p)
    return out


def pairs_signal(pairs: list[Pair], spreads: list[Spread], *, risk_budget_frac: float,
                 now: datetime) -> SleeveSignal:
    """Emit per-leg tilts for active pairs. The spread is the traded unit:
      - short_spread: short y, long x (hedge_ratio units of x per unit of y)
      - long_spread : long y, short x
      - flat / stop : emit no legs (close handled by the optimizer/Trader)
    Each tilt carries pair_id for pair-level PnL attribution. Base y-leg weight is equal across
    active pairs (1/n_active); the x leg is scaled by the hedge ratio.
    """
    by_id = {p.pair_id: p for p in pairs}
    active = [s for s in spreads if s.state in ("long_spread", "short_spread")
              and s.pair_id in by_id]
    n = len(active)
    if n == 0:
        return SleeveSignal(sleeve="pairs", tilts=[], risk_budget_frac=risk_budget_frac,
                            as_of_ts=now)
    base_w = 1.0 / n
    tilts: list[SleeveTilt] = []
    for sp in active:
        pair = by_id[sp.pair_id]
        h = abs(pair.hedge_ratio)
        if sp.state == "short_spread":               # short y, long x
            y_dir, x_dir = "short", "long"
            y_w, x_w = -base_w, base_w * h
        else:                                        # long_spread: long y, short x
            y_dir, x_dir = "long", "short"
            y_w, x_w = base_w, -base_w * h
        tilts.append(SleeveTilt(symbol=pair.symbol_y, direction=y_dir, target_weight=y_w,
                                raw_score=sp.zscore, pair_id=pair.pair_id))
        tilts.append(SleeveTilt(symbol=pair.symbol_x, direction=x_dir, target_weight=x_w,
                                raw_score=sp.zscore, pair_id=pair.pair_id))
    return SleeveSignal(sleeve="pairs", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"n_active": n}, as_of_ts=now)

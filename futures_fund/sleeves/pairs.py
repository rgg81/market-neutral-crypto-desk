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
    """Emit per-leg tilts for active pairs. The spread (= y - hedge_ratio*x) is the traded unit:
      - short_spread: short y, and short hedge_ratio units of x (x_w = base_w * hedge_ratio)
      - long_spread : long  y, and long  hedge_ratio units of x (x_w = -base_w * hedge_ratio)
      - flat / stop : emit no legs (close handled by the optimizer/Trader)
    The hedge ratio is SIGNED (Johansen routinely yields negative cointegrating coefficients;
    Engle-Granger returns the raw OLS slope). It is used WITHOUT abs() so the x leg hedges in the
    correct direction: for a negative hedge ratio the x leg flips side. The x-leg direction is
    derived from the sign of its signed weight; a zero hedge ratio leaves no x leg to size.
    Each tilt carries pair_id for pair-level PnL attribution. Base y-leg weight is equal across
    active pairs (1/n_active); the x leg is scaled by the (signed) hedge ratio.
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
        h = pair.hedge_ratio                         # SIGNED cointegrating beta (no abs())
        if sp.state == "short_spread":               # short the spread: short y, short h*x
            y_dir, y_w = "short", -base_w
            x_w = base_w * h
        else:                                        # long the spread: long y, long h*x
            y_dir, y_w = "long", base_w
            x_w = -base_w * h
        tilts.append(SleeveTilt(symbol=pair.symbol_y, direction=y_dir, target_weight=y_w,
                                raw_score=sp.zscore, pair_id=pair.pair_id))
        x_dir: str = "long" if x_w >= 0 else "short"
        tilts.append(SleeveTilt(symbol=pair.symbol_x, direction=x_dir, target_weight=x_w,
                                raw_score=sp.zscore, pair_id=pair.pair_id))
    return SleeveSignal(sleeve="pairs", tilts=tilts, risk_budget_frac=risk_budget_frac,
                        diagnostics={"n_active": n}, as_of_ts=now)

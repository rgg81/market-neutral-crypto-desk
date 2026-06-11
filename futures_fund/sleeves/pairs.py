"""Cointegration-pairs sleeve: emit per-leg tilts for active Pairs, sized by the cointegrating
hedge ratio so the SPREAD is the traded unit. PnL is attributed at the pair level (Spread).
"""
from __future__ import annotations

from futures_fund.contracts import Pair


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

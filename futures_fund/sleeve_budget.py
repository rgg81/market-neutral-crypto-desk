"""Risk-parity (or inverse-vol) budget allocation across the FOUR alpha sleeves.

Lives in its own module so all four sleeves can be budgeted before the Phase 1 optimizer exists;
the canonical contract (§2.11) addresses this as neutrality.risk_parity_budgets, so Phase 2 also
ships a futures_fund/neutrality.py stub (Task 23a) that re-exports this function. Budgets sum to
1.0 over the active (non-empty) sleeves and fill SleeveSignal.risk_budget_frac.
"""
from __future__ import annotations

import numpy as np

from futures_fund.contracts import SleeveSignal
from futures_fund.models import SleeveName


def risk_parity_budgets(sleeves: list[SleeveSignal],
                        *, cov: np.ndarray | None = None) -> dict[SleeveName, float]:
    """Assign each sleeve its risk budget. With no cov, active sleeves split 1.0 equally; with a
    cov (sleeve-return covariance, same order as `sleeves`), use inverse-vol (1/sigma) weights.
    Sleeves with no tilts get a 0.0 budget and are excluded from the split.
    """
    names = [s.sleeve for s in sleeves]
    active = [i for i, s in enumerate(sleeves) if s.tilts]
    out: dict[SleeveName, float] = {n: 0.0 for n in names}
    if not active:
        return out
    if cov is None:
        share = 1.0 / len(active)
        for i in active:
            out[names[i]] = share
        return out
    variances = np.diag(np.asarray(cov, dtype=float))
    inv_vol = {i: 1.0 / np.sqrt(variances[i]) for i in active if variances[i] > 0}
    total = sum(inv_vol.values())
    if total <= 0:
        share = 1.0 / len(active)
        for i in active:
            out[names[i]] = share
        return out
    for i, w in inv_vol.items():
        out[names[i]] = float(w / total)
    return out

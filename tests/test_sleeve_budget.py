from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from futures_fund.contracts import SleeveSignal
from futures_fund.models import SleeveName

NOW = datetime(2026, 6, 11, tzinfo=UTC)


def _sig(name: SleeveName) -> SleeveSignal:
    """Minimal SleeveSignal with no tilts; risk_parity_budgets only needs the sleeve label."""
    return SleeveSignal(sleeve=name, as_of_ts=NOW)


def test_neutrality_exposes_risk_parity_budgets():
    # The canonical contract name `neutrality.risk_parity_budgets` (§2.11) must resolve at the
    # end of Phase 2. Per the canonical interface contract the allocator lives in neutrality.py
    # (the standalone sleeve_budget module was a dead duplicate and was removed), so the contract
    # name resolves directly from neutrality rather than via a re-export stub.
    from futures_fund.neutrality import risk_parity_budgets

    # and it actually works through the canonical name
    budgets = risk_parity_budgets([_sig("carry"), _sig("pairs")])
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    # no covariance supplied => equal (inverse-unit-vol) split across the two sleeves
    assert abs(budgets["carry"] - 0.5) < 1e-9
    assert abs(budgets["pairs"] - 0.5) < 1e-9


def test_risk_parity_budgets_writes_back_onto_signals_via_contract_name():
    from futures_fund.neutrality import risk_parity_budgets

    carry, pairs = _sig("carry"), _sig("pairs")
    budgets = risk_parity_budgets([carry, pairs])
    assert abs(carry.risk_budget_frac - budgets["carry"]) < 1e-9
    assert abs(pairs.risk_budget_frac - budgets["pairs"]) < 1e-9


def test_risk_parity_budgets_good_case_inverse_vol_unchanged():
    # Good-case behavior must be unchanged: with real variances the split is inverse-vol.
    # var = [0.01, 0.04] -> vol = [0.1, 0.2] -> inv-vol [10, 5] -> [2/3, 1/3].
    from futures_fund.neutrality import risk_parity_budgets

    cov = np.diag([0.01, 0.04])
    budgets = risk_parity_budgets([_sig("carry"), _sig("pairs")], cov=cov)
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    assert abs(budgets["carry"] - 2.0 / 3.0) < 1e-9
    assert abs(budgets["pairs"] - 1.0 / 3.0) < 1e-9


def test_risk_parity_budgets_degenerate_sleeve_gets_zero_budget():
    # A zero/degenerate-variance sleeve must NOT absorb the budget via a 1e6 inv-vol spike.
    # With another sleeve carrying real variance, the dead sleeve drops to 0 budget.
    from futures_fund.neutrality import risk_parity_budgets

    carry, pairs = _sig("carry"), _sig("pairs")
    cov = np.diag([0.0, 0.04])           # carry is dead (zero variance), pairs is real
    budgets = risk_parity_budgets([carry, pairs], cov=cov)
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    assert budgets["carry"] == 0.0       # dead sleeve cannot capture the budget
    assert abs(budgets["pairs"] - 1.0) < 1e-9
    assert carry.risk_budget_frac == 0.0
    assert abs(pairs.risk_budget_frac - 1.0) < 1e-9


def test_risk_parity_budgets_all_degenerate_equal_share():
    # If ALL sleeves are degenerate (no real variance), fall back to an equal share rather than
    # letting an arbitrary inv-vol spike pick a winner.
    from futures_fund.neutrality import risk_parity_budgets

    cov = np.diag([0.0, 0.0, 0.0])
    budgets = risk_parity_budgets(
        [_sig("carry"), _sig("pairs"), _sig("factor")], cov=cov
    )
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    for name in ("carry", "pairs", "factor"):
        assert abs(budgets[name] - 1.0 / 3.0) < 1e-9

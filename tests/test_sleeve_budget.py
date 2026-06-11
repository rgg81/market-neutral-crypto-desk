from __future__ import annotations

from datetime import UTC, datetime

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

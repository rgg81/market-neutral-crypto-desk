from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from futures_fund.contracts import SleeveSignal, SleeveTilt
from futures_fund.sleeve_budget import risk_parity_budgets

_NOW = datetime(2026, 6, 11, tzinfo=UTC)


def _sig(name: str, n_tilts: int = 1) -> SleeveSignal:
    tilts = [SleeveTilt(symbol=f"{name}{i}/USDT:USDT", direction="long", target_weight=0.1)
             for i in range(n_tilts)]
    return SleeveSignal(sleeve=name, tilts=tilts, as_of_ts=_NOW)


def test_risk_parity_budgets_equal_when_no_cov_and_all_active():
    sleeves = [_sig("carry"), _sig("pairs"), _sig("factor"), _sig("sentiment")]
    budgets = risk_parity_budgets(sleeves)
    assert set(budgets) == {"carry", "pairs", "factor", "sentiment"}
    assert all(abs(b - 0.25) < 1e-9 for b in budgets.values())
    assert abs(sum(budgets.values()) - 1.0) < 1e-9


def test_risk_parity_budgets_skip_empty_sleeves():
    # a sleeve with no tilts gets zero budget; the rest split 1.0 equally
    sleeves = [_sig("carry"), _sig("pairs", n_tilts=0), _sig("factor"), _sig("sentiment")]
    budgets = risk_parity_budgets(sleeves)
    assert budgets["pairs"] == 0.0
    assert abs(budgets["carry"] - 1 / 3) < 1e-9
    assert abs(sum(budgets.values()) - 1.0) < 1e-9


def test_risk_parity_budgets_inverse_vol_from_cov():
    # diagonal cov: variances [1, 4, 1, 1] -> inverse-vol weights ~ [1, 0.5, 1, 1]/sum
    sleeves = [_sig("carry"), _sig("pairs"), _sig("factor"), _sig("sentiment")]
    cov = np.diag([1.0, 4.0, 1.0, 1.0])
    budgets = risk_parity_budgets(sleeves, cov=cov)
    inv = np.array([1.0, 0.5, 1.0, 1.0])
    expected = inv / inv.sum()
    assert abs(budgets["pairs"] - expected[1]) < 1e-9
    assert abs(budgets["carry"] - expected[0]) < 1e-9
    assert abs(sum(budgets.values()) - 1.0) < 1e-9


def test_risk_parity_budgets_all_empty_returns_zeros():
    sleeves = [_sig("carry", 0), _sig("pairs", 0)]
    budgets = risk_parity_budgets(sleeves)
    assert all(b == 0.0 for b in budgets.values())

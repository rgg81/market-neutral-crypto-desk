from __future__ import annotations

import numpy as np

from futures_fund.neutrality import (
    apply_hrp_weights,
    hrp_weights,
    ledoit_wolf_cov,
    merge_sleeves,
    risk_parity_budgets,
)


def test_ledoit_wolf_cov_is_symmetric_psd(returns_frame):
    cov = ledoit_wolf_cov(returns_frame)
    n = returns_frame.shape[1]
    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T)
    # PSD: all eigenvalues non-negative (shrinkage guarantees this)
    eigs = np.linalg.eigvalsh(cov)
    assert (eigs >= -1e-12).all()


def test_hrp_weights_sum_to_one_and_positive(returns_frame):
    cov = ledoit_wolf_cov(returns_frame)
    labels = list(returns_frame.columns)
    w = hrp_weights(cov, labels)
    assert set(w.keys()) == set(labels)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > 0.0 for v in w.values())


def test_hrp_weights_low_vol_gets_more_weight(returns_frame):
    # XRP has lower idio noise (0.006) than SOL (0.008) but the dominant driver is BTC beta.
    cov = ledoit_wolf_cov(returns_frame)
    labels = list(returns_frame.columns)
    w = hrp_weights(cov, labels)
    # Highest-variance asset (SOL, beta 1.5 + most noise) must not dominate the book.
    assert w["SOL/USDT:USDT"] < 0.5


def test_hrp_weights_single_asset():
    cov = np.array([[0.04]])
    w = hrp_weights(cov, ["BTC/USDT:USDT"])
    assert w == {"BTC/USDT:USDT": 1.0}


def test_risk_parity_budgets_equal_when_no_cov(sleeves):
    budgets = risk_parity_budgets(sleeves)
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    # two sleeves, inverse-vol fallback with no cov => equal split
    assert abs(budgets["factor"] - 0.5) < 1e-9
    assert abs(budgets["carry"] - 0.5) < 1e-9


def test_risk_parity_budgets_writes_back_onto_signals(sleeves):
    budgets = risk_parity_budgets(sleeves)
    for s in sleeves:
        assert abs(s.risk_budget_frac - budgets[s.sleeve]) < 1e-9


def test_merge_sleeves_scales_tilts_by_budget(sleeves, geometries):
    risk_parity_budgets(sleeves)  # assigns 0.5 / 0.5
    merged = merge_sleeves(sleeves, geometries)
    # factor: SOL +0.5*0.5=+0.25 ; XRP -0.5*0.5=-0.25
    assert abs(merged["SOL/USDT:USDT"] - 0.25) < 1e-9
    assert abs(merged["XRP/USDT:USDT"] - (-0.25)) < 1e-9
    # carry: BTC +0.25 ; ETH -0.25
    assert abs(merged["BTC/USDT:USDT"] - 0.25) < 1e-9
    assert abs(merged["ETH/USDT:USDT"] - (-0.25)) < 1e-9


def test_merge_sleeves_sums_same_symbol_across_sleeves(geometries):
    from datetime import UTC, datetime

    from futures_fund.contracts import SleeveSignal, SleeveTilt

    now = datetime(2026, 6, 11, tzinfo=UTC)
    a = SleeveSignal(
        sleeve="factor", risk_budget_frac=0.5, as_of_ts=now,
        tilts=[SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.4)],
    )
    b = SleeveSignal(
        sleeve="carry", risk_budget_frac=0.5, as_of_ts=now,
        tilts=[SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.6)],
    )
    merged = merge_sleeves([a, b], geometries)
    # 0.4*0.5 + 0.6*0.5 = 0.5
    assert abs(merged["BTC/USDT:USDT"] - 0.5) < 1e-9


def test_apply_hrp_weights_preserves_sign_and_side_gross():
    # Two longs (A,B) + two shorts (C,D). HRP says A>>B on the long side.
    weights = {"A": 0.3, "B": 0.3, "C": -0.3, "D": -0.3}
    hrp = {"A": 0.4, "B": 0.1, "C": 0.25, "D": 0.25}  # within-side normalization happens inside
    out = apply_hrp_weights(weights, hrp)
    # signs preserved
    assert out["A"] > 0 and out["B"] > 0 and out["C"] < 0 and out["D"] < 0
    # per-side gross preserved (long pool stays 0.6, short pool stays 0.6)
    assert abs((out["A"] + out["B"]) - 0.6) < 1e-9
    assert abs((-out["C"] - out["D"]) - 0.6) < 1e-9
    # HRP actually reshapes: A gets 0.4/0.5 of the long pool, B gets 0.1/0.5
    assert abs(out["A"] - 0.6 * (0.4 / 0.5)) < 1e-9
    assert abs(out["B"] - 0.6 * (0.1 / 0.5)) < 1e-9


def test_apply_hrp_weights_noop_when_hrp_empty():
    weights = {"A": 0.3, "B": -0.3}
    assert apply_hrp_weights(weights, {}) == weights

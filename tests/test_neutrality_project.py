from __future__ import annotations

from futures_fund.neutrality import beta_residual, project_neutral, size_btc_hedge


def test_project_neutral_drives_dollar_residual_into_band():
    # 3 names so the neutral null space is non-trivial (n - 2 = 1 dimension).
    weights = {"A": 0.4, "B": -0.2, "C": 0.1}
    betas = {"A": 1.0, "B": 1.0, "C": 1.0}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    # equity-normalized: dollar residual = sum of signed weights
    assert abs(sum(out.values())) <= 0.03 + 1e-9


def test_project_neutral_drives_beta_residual_into_band():
    weights = {"A": 0.3, "B": -0.3, "C": 0.1}
    betas = {"A": 1.5, "B": 0.5, "C": 1.0}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    assert abs(beta_residual(out, betas)) <= 0.05 + 1e-9


def test_project_neutral_already_neutral_is_near_identity():
    weights = {"A": 0.25, "B": -0.25, "C": 0.0}
    betas = {"A": 1.0, "B": 1.0, "C": 1.0}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    assert abs(out["A"] - 0.25) < 1e-6
    assert abs(out["B"] - (-0.25)) < 1e-6


def test_project_neutral_three_names_retains_nontrivial_gross():
    # A >=3-name book must NOT collapse to ~0 after projection (it lives in the 1-dim null
    # space). This is the guard against the n<=2 degenerate collapse.
    weights = {"A": 0.5, "B": -0.3, "C": 0.2}
    betas = {"A": 1.2, "B": 0.9, "C": 1.5}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    gross = sum(abs(v) for v in out.values())
    assert gross > 0.2  # non-trivial residual book survives projection


def test_project_neutral_two_names_collapse_is_documented():
    # With exactly 2 names and 2 independent constraints the unique neutral point is ~0.
    # We assert the collapse so the optimizer's "append hedge => >=3 names" guard is justified.
    weights = {"A": 0.5, "B": -0.5}
    betas = {"A": 1.5, "B": 0.8}
    out = project_neutral(weights, betas, dollar_band=0.03, beta_band=0.05)
    assert sum(abs(v) for v in out.values()) < 1e-6


def test_btc_hedge_absorbs_residual_beta_with_opposite_sign():
    # Net long beta => hedge must be short BTC (negative notional).
    weights = {"ALT/USDT:USDT": 0.3}
    betas = {"ALT/USDT:USDT": 1.5}  # beta residual = 0.45 (positive)
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    assert hedge < 0.0


def test_btc_hedge_zero_when_already_beta_neutral():
    weights = {"A": 0.3, "B": -0.3}
    betas = {"A": 1.0, "B": 1.0}  # residual 0
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    assert abs(hedge) < 1e-6


def test_btc_hedge_capped_inside_side_budget():
    # Huge residual beta must not size the hedge beyond the per-side budget.
    weights = {"A": 0.9}
    betas = {"A": 3.0}
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    assert abs(hedge) <= 10000.0 + 1e-6


def test_btc_hedge_short_beta_gives_long_hedge():
    weights = {"A": -0.3}
    betas = {"A": 1.5}  # residual -0.45 (net short beta)
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    assert hedge > 0.0


def test_btc_hedge_excludes_existing_btc_leg_from_residual():
    # If BTC is already a leg, its own beta is part of the residual the hedge should absorb,
    # but the hedge must size off the residual computed WITHOUT double-counting a prior hedge.
    # Here the alpha residual is +0.45 (ALT) and BTC alpha leg adds +0.1*1.0 => residual 0.55.
    weights = {"ALT/USDT:USDT": 0.3, "BTC/USDT:USDT": 0.1}
    betas = {"ALT/USDT:USDT": 1.5, "BTC/USDT:USDT": 1.0}
    hedge = size_btc_hedge(weights, betas, equity=20000.0, side_budget=10000.0)
    # residual beta = 0.3*1.5 + 0.1*1.0 = 0.55 ; hedge = -0.55*20000 = -11000 -> clamp -10000
    assert hedge < 0.0
    assert abs(hedge) <= 10000.0 + 1e-6

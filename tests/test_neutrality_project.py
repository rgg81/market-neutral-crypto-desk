from __future__ import annotations

from futures_fund.neutrality import beta_residual, project_neutral


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

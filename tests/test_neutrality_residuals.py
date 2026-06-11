from __future__ import annotations

import numpy as np

from futures_fund.neutrality import NeutralityConfig, beta_residual, dollar_residual


def test_neutrality_config_defaults():
    cfg = NeutralityConfig()
    assert cfg.capital_usdt == 20000.0
    assert cfg.target_gross_usdt == 20000.0
    assert cfg.side_budget_usdt == 10000.0
    assert cfg.deployment_floor == 0.90
    assert cfg.dry_powder_frac == 0.10
    assert cfg.per_name_cap == 0.25
    assert cfg.cluster_cap == 0.40
    assert cfg.dollar_band == 0.03
    assert cfg.beta_band == 0.05
    assert cfg.drift_band == 0.20
    assert cfg.stress_band_mult == 0.5


def test_deployment_target_is_between_floor_and_dry_powder_band():
    # The enforced per-side deployment target must sit inside [floor, 1 - dry_powder].
    cfg = NeutralityConfig()
    assert cfg.deployment_floor <= cfg.deploy_target_frac <= 1.0 - cfg.dry_powder_frac


def test_dollar_residual_balanced_book_is_zero():
    notionals = {"A": 5000.0, "B": -5000.0}
    weights = {"A": 0.25, "B": -0.25}
    assert np.isclose(dollar_residual(weights, notionals), 0.0)


def test_dollar_residual_long_heavy():
    notionals = {"A": 6000.0, "B": -4000.0}
    weights = {"A": 0.3, "B": -0.2}
    # Sum(long$) - Sum(short$) = 6000 - 4000 = 2000
    assert np.isclose(dollar_residual(weights, notionals), 2000.0)


def test_beta_residual_is_weighted_beta_sum():
    weights = {"A": 0.5, "B": -0.5}
    betas = {"A": 1.0, "B": 1.0}
    assert np.isclose(beta_residual(weights, betas), 0.0)
    betas2 = {"A": 1.5, "B": 0.5}
    # 0.5*1.5 + (-0.5)*0.5 = 0.75 - 0.25 = 0.5
    assert np.isclose(beta_residual(weights, betas2), 0.5)

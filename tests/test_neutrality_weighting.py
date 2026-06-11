from __future__ import annotations

import numpy as np

from futures_fund.neutrality import hrp_weights, ledoit_wolf_cov


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

from __future__ import annotations

import numpy as np

from futures_fund.walk_forward import validate_sleeve_param, walk_forward_splits


def test_walk_forward_splits_anchored_expanding():
    # 100 points, 4 folds: each fold trains on a growing prefix, tests on the next chunk.
    splits = walk_forward_splits(100, n_splits=4, min_train=20)
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert train_idx.stop <= test_idx.start          # no overlap: train strictly before test
        assert train_idx.start == 0                       # anchored (expanding window)
    # test chunks are contiguous and cover the tail
    assert splits[0][1].start >= 20
    assert splits[-1][1].stop == 100


def test_walk_forward_splits_too_short_returns_empty():
    assert walk_forward_splits(10, n_splits=4, min_train=20) == []


def test_validate_sleeve_param_genuine_edge_passes():
    rng = np.random.default_rng(0)
    # strong positive-mean OOS returns across folds -> should clear the DSR gate
    oos_returns = [list(rng.normal(0.02, 0.01, 40)) for _ in range(4)]
    res = validate_sleeve_param(oos_returns, num_trials=4, periods_per_year=365.0,
                                dsr_threshold=0.95)
    assert res["passed"] is True
    assert res["oos_sharpe"] > 0
    assert res["dsr_pvalue"] >= 0.95


def test_validate_sleeve_param_noise_fails():
    rng = np.random.default_rng(1)
    # zero-mean noise -> no edge -> gate rejects
    oos_returns = [list(rng.normal(0.0, 0.02, 40)) for _ in range(4)]
    res = validate_sleeve_param(oos_returns, num_trials=20, periods_per_year=365.0,
                                dsr_threshold=0.95)
    assert res["passed"] is False


def test_validate_sleeve_param_empty_fails():
    res = validate_sleeve_param([], num_trials=4, periods_per_year=365.0)
    assert res["passed"] is False
    assert res["oos_sharpe"] == 0.0

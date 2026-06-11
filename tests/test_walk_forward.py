from __future__ import annotations

import numpy as np
import pytest

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


def test_validate_sleeve_param_num_trials_deflates_pvalue_and_flips_pass():
    # Regression-lock the gate's core purpose: deflation for multiple testing.
    # In a REALISTIC Sharpe regime (per-period SR ~0.05, n=1000), the OOS Sharpe is the
    # same regardless of num_trials, but the Deflated-Sharpe p-value must SHRINK as more
    # param candidates are tried -- enough to flip passed True->False. (A genuinely strong
    # edge saturates dsr_p to 1.0, hiding this, which is why test_genuine_edge_passes alone
    # cannot catch a num_trials-ignoring implementation.)
    rng = np.random.default_rng(0)
    oos_returns = [list(rng.normal(0.001, 0.01, 250)) for _ in range(4)]  # 1000 pooled obs

    res_1 = validate_sleeve_param(oos_returns, num_trials=1, periods_per_year=365.0,
                                  dsr_threshold=0.95)
    res_20 = validate_sleeve_param(oos_returns, num_trials=20, periods_per_year=365.0,
                                   dsr_threshold=0.95)
    res_1000 = validate_sleeve_param(oos_returns, num_trials=1000, periods_per_year=365.0,
                                     dsr_threshold=0.95)

    # OOS Sharpe is a property of the returns, not of how many trials were tried.
    assert res_1["oos_sharpe"] > 0
    assert res_1["oos_sharpe"] == res_20["oos_sharpe"] == res_1000["oos_sharpe"]

    # Deflation: more trials -> strictly lower DSR p-value (this is the whole point of the gate).
    assert res_1["dsr_pvalue"] > res_20["dsr_pvalue"] > res_1000["dsr_pvalue"]

    # And it actually moves the gate decision: a single trial clears it, many trials do not.
    assert res_1["passed"] is True
    assert res_1000["passed"] is False


def test_validate_sleeve_param_weekly_path_threads_periods_per_year():
    # Walk-forward WEEKLY path: periods_per_year=52 must be threaded into the OOS Sharpe
    # annualization, so the SAME OOS returns yield a DIFFERENT (smaller) annualized Sharpe under
    # 52 than under 365 -- sqrt(52)/sqrt(365) ~ 0.377x. Independent oracle: the 52 Sharpe equals
    # the 365 Sharpe scaled by sqrt(52/365), computed here WITHOUT re-calling validate_sleeve_param.
    rng = np.random.default_rng(0)
    oos_returns = [list(rng.normal(0.01, 0.01, 40)) for _ in range(4)]  # same returns both calls

    res_365 = validate_sleeve_param(oos_returns, num_trials=4, periods_per_year=365.0,
                                    dsr_threshold=0.95)
    res_52 = validate_sleeve_param(oos_returns, num_trials=4, periods_per_year=52.0,
                                   dsr_threshold=0.95)

    # periods_per_year IS threaded into the Sharpe annualization: weekly differs from daily.
    assert res_52["oos_sharpe"] != res_365["oos_sharpe"]
    assert res_52["oos_sharpe"] == pytest.approx(
        res_365["oos_sharpe"] * np.sqrt(52.0 / 365.0)
    )
    assert res_52["oos_sharpe"] < res_365["oos_sharpe"]   # weekly annualizes a smaller multiplier
    # the weekly path still returns the full result dict shape with a DSR p-value present
    assert 0.0 <= res_52["dsr_pvalue"] <= 1.0
    assert res_52["n_obs"] == res_365["n_obs"] == 160


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

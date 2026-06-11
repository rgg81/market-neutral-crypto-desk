from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from futures_fund import cointegration as co
from futures_fund.contracts import Pair, Spread
from futures_fund.sleeves.pairs import select_pairs


def _cointegrated_pair(n: int = 400, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """x is a random walk; y = 2*x + stationary noise -> y and x are cointegrated."""
    rng = np.random.default_rng(seed)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0)
    noise = pd.Series(rng.normal(0, 0.5, n))
    y = 2.0 * x + noise
    return y, x


def test_engle_granger_recovers_hedge_ratio_and_rejects_unit_root():
    y, x = _cointegrated_pair()
    hedge_ratio, pvalue, stat = co.engle_granger(y, x)
    assert abs(hedge_ratio - 2.0) < 0.1          # OLS slope ~ 2.0
    assert pvalue < 0.05                          # residual is stationary -> reject unit root
    assert stat < 0.0                             # ADF stat is negative for a stationary series


def test_engle_granger_non_cointegrated_high_pvalue():
    rng = np.random.default_rng(11)
    y = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)   # two independent random walks
    _, pvalue, _ = co.engle_granger(y, x)
    assert pvalue > 0.05


def test_engle_granger_false_positive_rate_is_nominal():
    """The p-value must use the Engle-Granger cointegration null (its own critical values),
    NOT a plain ADF on the estimated-regression residual. Under the proper distribution the
    false-positive rate over independent random-walk pairs must be ~nominal 5% (well below the
    ~15-18% a residual-ADF produces). Regression-locks Fix 1 (statistical correctness)."""
    rng = np.random.default_rng(12345)
    trials = 200
    n = 250
    false_positives = 0
    for _ in range(trials):
        y = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50.0)
        x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50.0)   # independent random walks
        _, pvalue, _ = co.engle_granger(y, x)
        if pvalue < 0.05:
            false_positives += 1
    rate = false_positives / trials
    # Nominal 5% with sampling slack; a residual-ADF (the bug) sits ~0.15-0.18 here.
    assert rate <= 0.10, f"EG false-positive rate {rate:.3f} too high (proper null gives ~0.05)"


def test_johansen_detects_cointegration_rank():
    y, x = _cointegrated_pair()
    frame = pd.DataFrame({"y": y, "x": x})
    out = co.johansen(frame)
    assert out["rank"] >= 1                        # at least one cointegrating relationship
    assert out["trace_stat"] > out["crit_95"]      # trace stat exceeds the 95% critical value
    assert math.isfinite(out["hedge_ratio"])


def test_johansen_independent_walks_rank_zero():
    rng = np.random.default_rng(3)
    a = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    b = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    out = co.johansen(pd.DataFrame({"a": a, "b": b}))
    assert out["rank"] == 0


def _ou_path(theta: float, mu: float, sigma: float, n: int = 2000, seed: int = 5) -> pd.Series:
    """Simulate a discrete OU process: s_{t+1} = s_t + theta*(mu - s_t) + sigma*eps."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    s[0] = mu
    for t in range(1, n):
        s[t] = s[t - 1] + theta * (mu - s[t - 1]) + sigma * rng.normal()
    return pd.Series(s)


def test_ou_fit_recovers_theta_and_mu():
    spread = _ou_path(theta=0.2, mu=5.0, sigma=0.3)
    theta, mu, sigma_eq = co.ou_fit(spread)
    assert abs(theta - 0.2) < 0.05
    assert abs(mu - 5.0) < 0.3
    assert sigma_eq > 0.0


def test_ou_fit_flat_spread_fail_soft():
    # Constant/flat spread (stale or pinned prices): no IndexError, fail soft.
    theta, mu, sigma_eq = co.ou_fit(pd.Series([5.0] * 100))
    assert theta == 0.0
    assert abs(mu - 5.0) < 1e-9
    assert sigma_eq == 0.0


def test_half_life_formula():
    assert abs(co.half_life(math.log(2)) - 1.0) < 1e-9     # theta = ln2 -> half-life 1 cycle
    assert abs(co.half_life(0.2) - (math.log(2) / 0.2)) < 1e-9


def test_half_life_non_mean_reverting_is_inf():
    assert co.half_life(0.0) == float("inf")
    assert co.half_life(-0.1) == float("inf")


def test_spread_value():
    assert co.spread_value(100.0, 40.0, 2.0) == 100.0 - 2.0 * 40.0   # = 20.0


def test_zscore_normal():
    assert co.zscore(20.0, 10.0, 5.0) == 2.0


def test_zscore_zero_sigma_is_zero():
    assert co.zscore(20.0, 10.0, 0.0) == 0.0


def test_spread_state_transitions():
    # flat -> short_spread when z >= entry (spread rich, short the spread)
    assert co.spread_state(2.5, prev_state="flat") == "short_spread"
    # flat -> long_spread when z <= -entry (spread cheap, long the spread)
    assert co.spread_state(-2.5, prev_state="flat") == "long_spread"
    # |z| >= stop_z dominates -> stop
    assert co.spread_state(3.5, prev_state="short_spread") == "stop"
    assert co.spread_state(-3.5, prev_state="long_spread") == "stop"
    # inside exit band -> flat
    assert co.spread_state(0.0, prev_state="short_spread") == "flat"
    # between exit and entry: hold the open position
    assert co.spread_state(1.5, prev_state="short_spread") == "short_spread"
    # between exit and entry from flat: stay flat (no new entry)
    assert co.spread_state(1.5, prev_state="flat") == "flat"


def test_fdr_bh_is_monotone_and_ge_raw():
    raw = [0.001, 0.01, 0.03, 0.5]
    adj = co.fdr_adjust(raw, method="bh")
    assert len(adj) == 4
    assert all(a >= r - 1e-12 for a, r in zip(adj, raw, strict=True))   # adjusted p >= raw p
    assert all(a <= 1.0 + 1e-12 for a in adj)


def test_fdr_bonferroni_multiplies_by_m():
    raw = [0.01, 0.02]
    adj = co.fdr_adjust(raw, method="bonferroni")
    assert abs(adj[0] - 0.02) < 1e-12              # 0.01 * 2
    assert abs(adj[1] - 0.04) < 1e-12              # 0.02 * 2


def test_fdr_empty_returns_empty():
    assert co.fdr_adjust([]) == []


def test_fdr_adjust_matches_statsmodels_ground_truth():
    # I2: fdr_adjust must equal statsmodels' multipletests on BOTH paths, on a vector with TIES
    # and OUT-OF-ORDER p-values (the cases where a naive BH implementation diverges). statsmodels
    # is the independent oracle here (never re-call fdr_adjust as its own ground truth).
    raw = [0.04, 0.005, 0.04, 0.20, 0.005, 0.5, 0.13]   # ties at 0.04 / 0.005, not sorted
    bh = co.fdr_adjust(raw, method="bh")
    bh_truth = multipletests(raw, method="fdr_bh")[1]
    assert bh == pytest.approx(list(bh_truth))
    bonf = co.fdr_adjust(raw, method="bonferroni")
    bonf_truth = multipletests(raw, method="bonferroni")[1]
    assert bonf == pytest.approx(list(bonf_truth))


def test_build_pair_assembles_validated_pair():
    y, x = _cointegrated_pair()
    pair = co.build_pair(y, x, "BTC/USDT:USDT", "ETH/USDT:USDT", cycle=4)
    assert isinstance(pair, Pair)
    assert pair.pair_id == "BTCUSDT__ETHUSDT"      # canonical slash-free id
    assert pair.symbol_y == "BTC/USDT:USDT"
    assert pair.symbol_x == "ETH/USDT:USDT"
    assert pair.method == "engle_granger"
    assert pair.adf_pvalue < 0.05
    assert pair.adf_pvalue_adj is None             # FDR fills this later across the candidate set
    assert abs(pair.hedge_ratio - 2.0) < 0.1
    assert pair.formed_cycle == 4
    assert pair.cointegrated is True
    assert pair.half_life > 0.0


def test_build_pair_non_cointegrated_two_random_walks():
    # build_pair on two INDEPENDENT random walks must NOT call them cointegrated: the
    # Engle-Granger p stays > 0.05 (no spurious cointegration) and the assembled Pair flags
    # cointegrated=False.
    rng = np.random.default_rng(11)
    y = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 50.0)   # independent walk, no relationship
    pair = co.build_pair(y, x, "BTC/USDT:USDT", "ETH/USDT:USDT", cycle=2)
    assert pair.cointegrated is False
    assert pair.adf_pvalue > 0.05


def test_build_pair_johansen_method():
    # method="johansen": hedge_ratio + johansen fields come from the Johansen result, and
    # cointegration is judged by trace_stat > crit_95 (NOT the EG ADF p, which is informational).
    y, x = _cointegrated_pair()
    pair = co.build_pair(y, x, "BTC/USDT:USDT", "ETH/USDT:USDT", cycle=4, method="johansen")
    assert pair.method == "johansen"
    assert pair.johansen_trace_stat is not None
    assert pair.johansen_crit_95 is not None
    # cointegrated derives from the trace statistic for the johansen branch
    assert pair.cointegrated == (pair.johansen_trace_stat > pair.johansen_crit_95)
    assert pair.cointegrated is True               # the simulated pair IS cointegrated
    assert pair.half_life > 0.0


def _btc_eth_pair() -> Pair:
    return Pair(
        pair_id="BTCUSDT__ETHUSDT",
        symbol_y="BTC/USDT:USDT", symbol_x="ETH/USDT:USDT",
        hedge_ratio=2.0, method="engle_granger", adf_pvalue=0.01,
        half_life=5.0, theta=0.139, mu=0.0, sigma_eq=10.0, formed_cycle=1,
    )


def test_build_spread_computes_value_zscore_state():
    pair = _btc_eth_pair()
    sp = co.build_spread(pair, mark_y=120.0, mark_x=49.0, prev_state="flat")
    assert isinstance(sp, Spread)
    assert sp.pair_id == pair.pair_id
    assert sp.spread_value == 120.0 - 2.0 * 49.0    # = 22.0
    assert sp.zscore == 2.2                          # (22 - 0) / 10
    assert sp.state == "short_spread"                # z >= entry_z (2.0) -> short the rich spread
    assert sp.entry_z == 2.0 and sp.exit_z == 0.0 and sp.stop_z == 3.0


def test_build_spread_hard_stop_state():
    pair = _btc_eth_pair()
    sp = co.build_spread(pair, mark_y=131.0, mark_x=49.0, prev_state="short_spread")
    assert sp.zscore == 3.3                          # (33 - 0)/10 -> |z| >= stop_z
    assert sp.state == "stop"


def _indep_walks(seed: int, n: int = 400) -> tuple[pd.Series, pd.Series]:
    """Two INDEPENDENT random walks (no cointegrating relationship)."""
    rng = np.random.default_rng(seed)
    y = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50.0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50.0)
    return y, x


def test_fdr_chain_drops_spurious_pair_through_real_select_pairs():
    # I1: end-to-end multiple-testing guard. Across a candidate set with ONE genuinely cointegrated
    # pair and several spurious ones, fdr_adjust must lift a BORDERLINE-spurious pair (raw EG p just
    # under 0.05, so it would pass select_pairs on the RAW p) to an adjusted p ABOVE the threshold,
    # so the real select_pairs chain DROPS it and keeps only the genuine pair. All pairs are built
    # by the real build_pair so adf_pvalue / cointegrated come from the real EG test, not handwoven.
    rng = np.random.default_rng(7)
    gx = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 100.0)
    gy = 2.0 * gx + pd.Series(rng.normal(0, 0.5, 400))           # genuinely cointegrated
    genuine = co.build_pair(gy, gx, "BTC/USDT:USDT", "ETH/USDT:USDT", cycle=1)

    # seed 36 is a spurious independent-walk pair whose raw EG p lands just under 0.05 (in
    # [0.03, 0.05)) -- a false positive that the RAW threshold would wrongly keep.
    by, bx = _indep_walks(36)
    borderline = co.build_pair(by, bx, "XRP/USDT:USDT", "ADA/USDT:USDT", cycle=1)

    # extra clearly-spurious pairs to supply the multiple-testing multiplicity that inflates the
    # borderline pair's adjusted p above 0.05.
    spurious = [
        co.build_pair(*_indep_walks(100 + i), f"C{i}/USDT:USDT", f"D{i}/USDT:USDT", cycle=1)
        for i in range(4)
    ]

    candidates = [genuine, borderline, *spurious]
    # the borderline pair genuinely passes on the RAW p (this is what makes the FDR step bind)
    assert genuine.adf_pvalue < 0.05
    assert borderline.adf_pvalue < 0.05

    # run the real FDR correction over the candidate set, writing the adjusted p back onto each pair
    adj = co.fdr_adjust([p.adf_pvalue for p in candidates], method="bh")
    for p, a in zip(candidates, adj, strict=True):
        p.adf_pvalue_adj = a

    # the FDR adjustment BINDS: it pushes the borderline pair's adjusted p over the threshold
    assert borderline.adf_pvalue_adj > 0.05
    assert genuine.adf_pvalue_adj < 0.05

    kept = select_pairs(candidates, adf_pvalue_max=0.05)
    # only the genuine pair survives; the borderline false positive is dropped by FDR
    assert [p.pair_id for p in kept] == [genuine.pair_id]
    assert borderline not in kept

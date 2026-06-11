"""Walk-forward validation harness hook for sleeve params/thresholds (design spec §12, §15).

Anchored (expanding-window) out-of-sample splits + a DSR/overfit gate over the vendored
overfit_detector. A sleeve param change is only trusted if it clears the OOS Deflated-Sharpe
threshold, not an in-sample grid win.
"""
from __future__ import annotations

from futures_fund.graduation import deflated_sharpe_pvalue
from futures_fund.metrics import sharpe


def walk_forward_splits(n_obs: int, *, n_splits: int = 4,
                        min_train: int = 20) -> list[tuple[range, range]]:
    """Anchored (expanding-window) walk-forward splits over a length-`n_obs` series.

    Returns a list of (train_range, test_range): each train range is a prefix [0, t) growing fold
    by fold, and the test range is the next contiguous OOS chunk. Empty list if n_obs is too short
    to leave min_train training points plus at least one test point per split.
    """
    if n_obs < min_train + n_splits:
        return []
    test_total = n_obs - min_train
    chunk = test_total // n_splits
    if chunk < 1:
        return []
    splits: list[tuple[range, range]] = []
    for k in range(n_splits):
        train_stop = min_train + k * chunk
        test_start = train_stop
        test_stop = n_obs if k == n_splits - 1 else train_stop + chunk
        splits.append((range(0, train_stop), range(test_start, test_stop)))
    return splits


def validate_sleeve_param(oos_returns: list[list[float]], *, num_trials: int,
                          periods_per_year: float = 365.0,
                          dsr_threshold: float = 0.95) -> dict:
    """Gate a sleeve param/threshold change on out-of-sample evidence.

    `oos_returns` is one return stream per walk-forward fold. Pools the folds, computes the OOS
    Sharpe (annualized at `periods_per_year` — 365 daily / 52 weekly) and the Deflated-Sharpe
    p-value deflated for `num_trials` (the number of param candidates tried). passed iff the OOS
    Sharpe is > 0 AND the DSR p-value clears `dsr_threshold`.
    """
    pooled: list[float] = [r for fold in oos_returns for r in fold]
    if len(pooled) < 10:
        return {"passed": False, "oos_sharpe": 0.0, "dsr_pvalue": 0.0, "n_obs": len(pooled)}
    oos_sharpe = sharpe(pooled, periods_per_year=periods_per_year)
    dsr_p = deflated_sharpe_pvalue(pooled, num_trials=num_trials,
                                   periods_per_year=periods_per_year)
    passed = oos_sharpe > 0 and dsr_p >= dsr_threshold
    return {"passed": passed, "oos_sharpe": oos_sharpe, "dsr_pvalue": dsr_p, "n_obs": len(pooled)}

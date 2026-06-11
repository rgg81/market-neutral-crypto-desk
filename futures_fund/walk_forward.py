"""Walk-forward validation harness hook for sleeve params/thresholds (design spec §12, §15).

Anchored (expanding-window) out-of-sample splits + a DSR/overfit gate over the vendored
overfit_detector. A sleeve param change is only trusted if it clears the OOS Deflated-Sharpe
threshold, not an in-sample grid win.
"""
from __future__ import annotations


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

from __future__ import annotations

from futures_fund.walk_forward import walk_forward_splits


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

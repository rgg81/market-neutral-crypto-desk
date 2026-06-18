"""Per-symbol return frame for the optimizer's covariance (HRP shaping + cluster cap).

The live run called `optimize_book(... returns=None)`, so Ledoit-Wolf/HRP shaping fell back to the
merged split and the cluster cap could never bind (no covariance). `build_returns_frame` turns the
per-symbol close series cycle-prep already reads into the returns DataFrame `optimize_book`
consumes, and the json helpers persist it as a cycle artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from futures_fund.returns_frame import (
    build_returns_frame,
    frame_from_json,
    frame_to_json,
)


def _closes(n, start=100.0, step=1.0):
    return pd.Series([start + step * i for i in range(n)], dtype=float)


def test_build_returns_frame_columns_and_returns_math():
    marks = {"AAA": _closes(30), "BBB": _closes(30, start=10.0, step=0.5)}
    df = build_returns_frame(marks, min_obs=5)
    assert list(df.columns) == ["AAA", "BBB"]
    assert len(df) == 29  # 30 closes -> 29 returns
    # first AAA return: (101-100)/100 = 0.01
    assert abs(df["AAA"].iloc[0] - 0.01) < 1e-12


def test_build_returns_frame_aligns_on_most_recent_common_length():
    # AAA has more history than BBB; the frame aligns on the most-recent overlap (BBB's length).
    marks = {"AAA": _closes(50), "BBB": _closes(21)}
    df = build_returns_frame(marks, min_obs=5)
    assert len(df) == 20  # min(49, 20) returns -> aligned to BBB's 20
    assert set(df.columns) == {"AAA", "BBB"}


def test_build_returns_frame_drops_too_short_series():
    marks = {"AAA": _closes(30), "TOO_SHORT": _closes(3)}
    df = build_returns_frame(marks, min_obs=10)
    assert list(df.columns) == ["AAA"]  # TOO_SHORT (2 returns < 10) excluded


def test_build_returns_frame_empty_when_nothing_qualifies():
    df = build_returns_frame({"X": _closes(2)}, min_obs=10)
    assert df.empty


def test_json_round_trip_preserves_frame():
    marks = {"AAA": _closes(25), "BBB": _closes(25, start=5.0, step=0.2)}
    df = build_returns_frame(marks, min_obs=5)
    restored = frame_from_json(frame_to_json(df))
    assert list(restored.columns) == list(df.columns)
    assert np.allclose(restored.to_numpy(), df.to_numpy())


def test_frame_feeds_optimizer_covariance():
    # the frame must be consumable by the optimizer's Ledoit-Wolf covariance (columns = symbols).
    from futures_fund.neutrality import ledoit_wolf_cov

    rng = np.random.default_rng(3)
    marks = {s: pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.02, 60)))
             for s in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT")}
    df = build_returns_frame(marks, min_obs=20)
    cov = ledoit_wolf_cov(df[list(df.columns)])
    assert cov.shape == (3, 3)

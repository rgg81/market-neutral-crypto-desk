from __future__ import annotations

import numpy as np
import pandas as pd

from futures_fund.beta import log_returns


def test_log_returns_basic():
    prices = pd.Series([100.0, 110.0, 121.0])
    r = log_returns(prices)
    assert len(r) == 2
    assert np.isclose(r.iloc[0], np.log(110.0 / 100.0))
    assert np.isclose(r.iloc[1], np.log(121.0 / 110.0))


def test_log_returns_drops_nan_and_nonpositive():
    prices = pd.Series([100.0, np.nan, 121.0, 0.0, 130.0])
    r = log_returns(prices)
    # NaN and non-positive prices removed before differencing; no inf/NaN remains
    assert not r.isna().any()
    assert np.isfinite(r.to_numpy()).all()


def test_log_returns_empty_series_returns_empty():
    r = log_returns(pd.Series([], dtype=float))
    assert len(r) == 0

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns of a mark-price series. Drops NaN/non-positive prices before
    differencing so no inf/NaN leaks into the return series."""
    clean = prices.dropna()
    clean = clean[clean > 0.0]
    if len(clean) < 2:
        return pd.Series([], dtype=float)
    return np.log(clean / clean.shift(1)).dropna()

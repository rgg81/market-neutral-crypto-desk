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


def rolling_beta(
    asset_returns: pd.Series, btc_returns: pd.Series, lookback: int = 45
) -> float:
    """OLS beta = cov(asset, btc) / var(btc) over the last `lookback` aligned points.
    Falls back to 1.0 if fewer than 10 aligned points or BTC variance is zero."""
    aligned = pd.concat([asset_returns, btc_returns], axis=1, join="inner").dropna()
    if len(aligned) > lookback:
        aligned = aligned.iloc[-lookback:]
    if len(aligned) < 10:
        return 1.0
    a = aligned.iloc[:, 0].to_numpy()
    b = aligned.iloc[:, 1].to_numpy()
    var_b = float(np.var(b))
    if var_b <= 0.0:
        return 1.0
    cov_ab = float(np.cov(a, b, ddof=0)[0, 1])
    return cov_ab / var_b

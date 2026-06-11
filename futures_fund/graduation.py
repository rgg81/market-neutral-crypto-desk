"""Graduation / overfit gate (design spec §12).

`deflated_sharpe_pvalue` wraps the vendored Lopez de Prado Deflated Sharpe Ratio so a sleeve
param/threshold change is only trusted once its OOS Sharpe clears the DSR threshold after deflating
for multiple testing — not on an in-sample grid win. Lifted/adapted from the weekly desk, rewired
to this repo's corrected `metrics.PERIODS_PER_YEAR_DAILY` (the inherited 2190 4h factor was WRONG).
"""
from __future__ import annotations

from futures_fund.metrics import PERIODS_PER_YEAR_DAILY, sharpe
from futures_fund.vendor.overfit_detector import deflated_sharpe_ratio

DSR_THRESHOLD = 0.95


def deflated_sharpe_pvalue(returns: list[float], num_trials: int,
                           periods_per_year: float = PERIODS_PER_YEAR_DAILY,
                           sigma_sr: float | None = None) -> float:
    """Probability the desk's Sharpe is genuinely > 0 after deflating for multiple testing
    (vendored Lopez de Prado DSR). 0.0 if < 10 observations (DSR requires backtest_length >= 10).

    sigma_sr = cross-trial Sharpe dispersion (per-period units) from tracked per-trial Sharpes;
    None falls back to the single-strategy reduction (sigma_sr = the Sharpe's standard error)."""
    if len(returns) < 10:
        return 0.0
    observed = sharpe(returns, periods_per_year=1.0)
    result = deflated_sharpe_ratio(observed_sr=observed, num_trials=max(1, num_trials),
                                   backtest_length=len(returns), sigma_sr=sigma_sr)
    return float(result.dsr_pvalue)

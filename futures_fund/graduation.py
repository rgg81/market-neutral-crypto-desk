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


def graduation_verdict(n_cycles: int, sharpe: float, dsr_pvalue: float, beats_baseline: bool,
                       max_dd: float, *, min_cycles: int = 20, horizon_cycles: int = 120,
                       dsr_threshold: float = DSR_THRESHOLD,
                       walk_forward_required: bool = False,
                       walk_forward_passed: bool = False) -> dict:
    """Decide paper->live readiness (and whether a sleeve-param change is trusted). graduated only
    if ALL criteria pass; failed if past the verdict horizon without an edge; otherwise not_yet with
    the failing criteria listed.

    WALK-FORWARD GATE (Phase 6, Task 6.4 — binding): when `walk_forward_required` is True (the path
    that trusts a *sleeve-param change*), the verdict additionally demands an out-of-sample
    walk-forward pass (`walk_forward_passed`). An in-sample-only grid winner — strong IS Sharpe/DSR
    but no OOS confirmation — is REJECTED, never graduated. This guards against fitting the grid to
    the in-sample window (spec §12). The default (`walk_forward_required=False`) preserves the prior
    paper->live verdict behavior exactly."""
    reasons: list[str] = []
    if n_cycles < min_cycles:
        reasons.append(f"need >= {min_cycles} audited cycles (have {n_cycles})")
    if sharpe <= 0:
        reasons.append(f"OOS Sharpe must be > 0 (is {sharpe:.2f})")
    if dsr_pvalue < dsr_threshold:
        reasons.append(f"DSR {dsr_pvalue:.2f} < {dsr_threshold} (edge not statistically proven)")
    if not beats_baseline:
        reasons.append("must beat buy-&-hold baseline net of costs")
    if walk_forward_required and not walk_forward_passed:
        reasons.append(
            "walk-forward OOS validation required before trusting a sleeve-param change — an "
            "in-sample-only grid winner is not trusted (must pass OOS, not just in-sample)")
    if not reasons:
        return {"status": "graduated", "reasons": []}
    if n_cycles >= horizon_cycles:
        return {"status": "failed", "reasons": reasons + [
            f"verdict horizon ({horizon_cycles} cycles) reached without an edge — retire/redesign"]}
    return {"status": "not_yet", "reasons": reasons}

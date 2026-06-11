"""KPI dashboard (Phase 7, Task 7.1) — the desk's end-of-run scorecard.

A single read-only function, `build_kpi_dashboard(state_dir, memory_dir)`, that bundles the success
+ process KPIs the spec (§18) grades the market-neutral desk on. It REUSES the deterministic
building blocks — never reimplements them:

- the equity series from `equity_log` (the return-series source),
- `metrics.sharpe` / `metrics.max_drawdown` (the success metrics),
- the `improvement` process KPIs (`both_sides_deployment_rate`, `pair_survival_rate`,
  `carry_capture_rate`, `sentiment_hit_rate`, `reviewer_veto_rate` — Task 6.3),
- the `NeutralityConfig` bands (the same bands the reviewer re-derives against).

PRIMARY KPI is `no_losing_month` (fraction of calendar months that are positive; target 1.0);
SECONDARY is `daily_sharpe` (annualized ×365). Every KPI is fail-safe on a cold/thin log: the
ratio metrics collapse to 0.0 and the process KPIs to their `nan` sentinel rather than dividing by
zero — building a dashboard on an empty desk must never raise.
"""

from __future__ import annotations

from datetime import datetime

from futures_fund.equity_log import equity_series, returns_series
from futures_fund.improvement import (
    _cycle_dirs,  # internal KPI helpers, reused so the dashboard scans the SAME cadence roots
    _read_json,
    both_sides_deployment_rate,
    carry_capture_rate,
    pair_survival_rate,
    reviewer_veto_rate,
    sentiment_hit_rate,
)
from futures_fund.metrics import (
    PERIODS_PER_YEAR_DAILY,
    max_drawdown,
    sharpe,
)
from futures_fund.neutrality import NeutralityConfig


def _no_losing_month(series: list[tuple[str, float]]) -> float:
    """Fraction of calendar months whose return is positive (the PRIMARY success KPI; target 1.0).

    Chains each month's return off the prior month's CLOSE (its last equity point) — the first
    month's baseline is the earliest equity on record — so a month is "positive" iff its month-end
    equity is at least its starting equity. Returns 0.0 with fewer than two equity points (no month
    return is defined yet); a single flat month with start == end counts as non-losing (>=)."""
    if len(series) < 2:
        return 0.0
    # Last equity point within each calendar month, in chronological order.
    by_month: dict[tuple[int, int], float] = {}
    first_equity = series[0][1]
    for ts, eq in series:
        d = datetime.fromisoformat(ts)
        by_month[(d.year, d.month)] = eq  # later point in the month overwrites -> month CLOSE
    months = sorted(by_month)
    if not months:
        return 0.0
    closes = [by_month[m] for m in months]
    baselines = [first_equity, *closes[:-1]]  # each month's baseline is the prior month's close
    positive = sum(1 for close, base in zip(closes, baselines, strict=True)
                   if base > 0 and close >= base)
    return positive / len(months)


def _neutrality_adherence(state_dir, *, last_n: int = 10) -> float:
    """Fraction of the last `last_n` cycles whose persisted residuals are within the neutrality
    bands: `dollar_residual_frac <= dollar_band` AND `|beta_residual| <= beta_band`.

    Reads the same `target_weights.json` artifacts the deployment KPIs read; a cycle that ran but
    emitted no `target_weights.json` stays in the denominator and counts as NOT-adherent (it never
    proved it was neutral). Returns 0.0 when no cycle is present. (spec §18 process KPI.)"""
    cfg = NeutralityConfig()
    dirs = _cycle_dirs(state_dir)[-last_n:]
    n = len(dirs)
    if n == 0:
        return 0.0
    in_band = 0
    for d in dirs:
        tw = _read_json(d / "target_weights.json")
        if tw is None:
            continue
        dollar_frac = abs(float(tw.get("dollar_residual_frac", 0.0) or 0.0))
        beta_resid = abs(float(tw.get("beta_residual", 0.0) or 0.0))
        if dollar_frac <= cfg.dollar_band and beta_resid <= cfg.beta_band:
            in_band += 1
    return in_band / n


def build_kpi_dashboard(state_dir, memory_dir, *, last_n: int = 10) -> dict:
    """Bundle the desk's success + process KPIs for the end-of-run scorecard (spec §18).

    SUCCESS (read off the equity series in `state_dir`): `no_losing_month` (PRIMARY, fraction of
    calendar months positive), `daily_sharpe` (`metrics.sharpe` ×365), `max_drawdown`. PROCESS
    (reused from the `improvement` panel — artifact KPIs read `state_dir`, journal KPIs read
    `memory_dir`): `both_sides_deployment_rate`, `neutrality_adherence`, `pair_survival`,
    `carry_capture`, `sentiment_hit_rate`, `reviewer_veto_rate`. Pure / read-only and fail-safe on
    a cold log."""
    series = equity_series(state_dir)
    equity = [e for _, e in series]
    rets = returns_series(state_dir)
    return {
        # success KPIs
        "no_losing_month": _no_losing_month(series),
        "daily_sharpe": sharpe(rets, periods_per_year=PERIODS_PER_YEAR_DAILY),
        "max_drawdown": max_drawdown(equity),
        # process KPIs (reused verbatim from improvement / Task 6.3)
        "both_sides_deployment_rate": both_sides_deployment_rate(state_dir, last_n=last_n),
        "neutrality_adherence": _neutrality_adherence(state_dir, last_n=last_n),
        "pair_survival": pair_survival_rate(memory_dir, last_n=last_n),
        "carry_capture": carry_capture_rate(memory_dir, last_n=last_n),
        "sentiment_hit_rate": sentiment_hit_rate(memory_dir, last_n=last_n),
        "reviewer_veto_rate": reviewer_veto_rate(state_dir, last_n=last_n),
    }

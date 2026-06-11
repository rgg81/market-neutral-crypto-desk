"""The desk's statistical self-portrait, re-keyed on market-neutral ALPHA (Phase 6, Task 6.4).

`build_scorecard` is injected into EVERY agent prompt so the team reasons WITH its measured track
record. The weekly desk keyed this on raw return; the neutral desk keys it on the PROCESS KPIs the
spec (§12/§18/§19) cares about — folding in the Phase-6 `improvement.py` neutral KPIs
(`both_sides_deployment_rate`, `pair_survival_rate`, `carry_capture_rate`, `sentiment_hit_rate`,
`reviewer_veto_rate`) plus `alpha_sharpe_trend` (rolling alpha-Sharpe slope, return net of BTC-beta)
and the DSR-gated `graduation_verdict`.

The warnings are DELIBERATELY TWO-SIDED — a drawdown/risk-off BRAKE and an under-deployment
ACCELERATOR. The brake measures a REAL drawdown: it compounds the per-cycle alpha series into a
cumulative-alpha equity curve and reads its peak-to-trough decline via `metrics.max_drawdown`,
firing only past a tolerance (the desk is drawdown-tolerant, so it engages on a genuine drawdown,
not on a single -5% wiggle). Without the accelerator the injected context is a one-way ratchet that
talks the desk out of every clean neutral trade (the root cause of standing down to cash, or to a
one-sided book, for cycles on end). Lifted/adapted from the weekly `scorecard.py` (verify+merge),
re-pointed onto the neutral KPIs since this repo's substrate is the Phase-6 alpha process, not the
weekly raw-return equity panel.
"""

from __future__ import annotations

import math

from futures_fund.graduation import deflated_sharpe_pvalue, graduation_verdict
from futures_fund.improvement import (
    alpha_sharpe_trend,
    both_sides_deployment_rate,
    carry_capture_rate,
    corpus_health,
    deployment_rate,
    pair_survival_rate,
    reviewer_veto_rate,
    sentiment_hit_rate,
)
from futures_fund.journal import read_all_decisions
from futures_fund.metrics import PERIODS_PER_YEAR_DAILY, max_drawdown


def _alpha_returns(memory_dir, *, window: int) -> list[float]:
    """Per-cycle ALPHA series (return net of BTC-beta) from the journal outcomes, averaging legs
    within a cycle, oldest-first — the risk-adjusted series the DSR + alpha-Sharpe trend read."""
    by_cycle: dict[int, list[float]] = {}
    for d in read_all_decisions(memory_dir):
        if d.get("alpha_return") is None or d.get("cycle") is None:
            continue
        by_cycle.setdefault(int(d["cycle"]), []).append(float(d["alpha_return"]))
    series = [sum(v) / len(v) for _, v in sorted(by_cycle.items())]
    return series[-window:] if window else series


def _alpha_drawdown(alpha_rets: list[float]) -> float:
    """Current drawdown of the cumulative-alpha equity curve from its running peak, as a positive
    fraction. Builds an equity curve by compounding the per-cycle alpha series (start 1.0) and reads
    the trough-from-peak via `metrics.max_drawdown` — a REAL measured drawdown, not a flag."""
    if not alpha_rets:
        return 0.0
    equity = [1.0]
    for r in alpha_rets:
        equity.append(equity[-1] * (1.0 + r))
    return max_drawdown(equity)


def build_scorecard(state_dir, memory_dir, *, last_n: int = 10, weekly_target: float = 0.05,
                    min_cycles: int = 20, horizon_cycles: int = 120,
                    drawdown_brake: float = 0.05) -> dict:
    """The neutral desk's self-portrait, injected into every agent prompt.

    Bundles the Phase-6 neutral KPIs, the rolling alpha-Sharpe trend, a DSR-gated graduation verdict
    over the alpha series, and TWO-SIDED warnings (drawdown brake + under-deployment accelerator).
    Pure / read-only. The drawdown brake fires on a REAL measured drawdown of the cumulative-alpha
    equity curve past `drawdown_brake` (the desk is drawdown-tolerant, so it engages on a genuine
    peak-to-trough loss, not on a single -5% wiggle)."""
    alpha_rets = _alpha_returns(memory_dir, window=last_n)
    # DSR over the alpha series (conservative fixed trial count, like the weekly desk).
    dsr = deflated_sharpe_pvalue(alpha_rets, num_trials=10) if alpha_rets else 0.0
    period_alpha = sum(alpha_rets)
    n_cycles = len(alpha_rets)
    alpha_drawdown = _alpha_drawdown(alpha_rets)

    both_sides = both_sides_deployment_rate(state_dir, last_n=last_n)
    deploy = deployment_rate(state_dir, last_n=last_n)
    veto = reviewer_veto_rate(state_dir, last_n=last_n)
    a_trend = alpha_sharpe_trend(memory_dir, window=last_n)

    grad = graduation_verdict(n_cycles, a_trend, dsr, period_alpha > 0, 0.0,
                              min_cycles=min_cycles, horizon_cycles=horizon_cycles)

    # ---------- TWO-SIDED WARNINGS (brake + accelerator) ----------
    warnings: list[str] = []
    # --- BRAKE (risk-off): a REAL measured drawdown of the cumulative-alpha equity curve sizes the
    # book down. The desk is drawdown-tolerant, so the brake engages only past `drawdown_brake`, not
    # on a single -5% wiggle. ---
    if alpha_drawdown > drawdown_brake:
        warnings.append(
            f"in {alpha_drawdown:.0%} drawdown from the alpha peak — bias risk-off and size "
            "conservatively")
    if n_cycles >= 11 and dsr < 0.95:  # DSR only computable at >= 10 returns
        warnings.append("alpha edge not statistically proven (DSR < 0.95) — size conservatively")
    # --- ACCELERATOR (counter-signal): under-deployment opportunity cost. Fires when the book is
    # NOT both-sides-deployed (the all-cash OR one-sided ratchet) — idle / one-legged cash has
    # opportunity cost vs the neutral target. Self-silences once both sides are deployed, so it can
    # never manufacture trades. ---
    if both_sides < 1.0:
        warnings.append(
            "under-deployed: NOT both-sides-deployed across the window — an all-cash or one-sided "
            "book is not neutral and idle cash has opportunity cost vs the "
            f"{weekly_target:.0%}/week target. Do NOT stand flat (or one-legged) on a clean, "
            "edge-aligned, neutral setup that clears the gate (RR>=2 + heat). Taking it is NOT "
            "forcing.")
    # --- BRAKE (process): a high reviewer veto-rate means the guardian is repeatedly HALTing the
    # book — slow down and fix the upstream cause rather than re-proposing the same vetoed book. ---
    if not math.isnan(veto) and veto >= 0.5:
        warnings.append(
            f"reviewer vetoed {veto:.0%} of recent cycles — the guardian is repeatedly HALTing the "
            "book; fix the upstream neutrality/funding breach, do not re-propose a vetoed book")

    return {
        "n_cycles": n_cycles,
        "period_alpha": period_alpha,
        "weekly_target": weekly_target,
        "alpha_sharpe_trend": a_trend,
        "alpha_sharpe_annualization": PERIODS_PER_YEAR_DAILY,
        "dsr_pvalue": dsr,
        "alpha_drawdown": alpha_drawdown,
        "deployment_rate": deploy["deployment_rate"],
        "both_sides_deployment_rate": both_sides,
        "pair_survival_rate": pair_survival_rate(memory_dir, last_n=last_n),
        "carry_capture_rate": carry_capture_rate(memory_dir, last_n=last_n),
        "sentiment_hit_rate": sentiment_hit_rate(memory_dir, last_n=last_n),
        "reviewer_veto_rate": veto,
        "corpus": corpus_health(memory_dir),
        "graduation": grad,
        "warnings": warnings,
    }

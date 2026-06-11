"""Improvement-panel KPIs, re-pointed at the market-neutral process (Phase 6, Task 6.3).

The weekly desk's improvement panel measured raw-return deployment/return trend. The neutral desk
cares about a different question: is the *process* healthy? Each cycle we surface six pure,
read-only KPIs (all mirroring `deployment_rate(state_dir, last_n)` — no mutation, fail-safe to a
sentinel on a thin/cold log) so the team and the meta-reflection can see the trend:

- BOTH-SIDES DEPLOYMENT — fraction of recent cycles where BOTH long and short cleared the
  deployment floor. Guards BOTH failure modes at once: the all-cash ratchet AND the one-sided
  ratchet (a book that's "deployed" but only long is not neutral). (spec §12/§19)
- PAIR SURVIVAL — fraction of re-tested pairs still cointegrated (ADF p < 0.05) at their weekly
  re-test: is the cointegration thesis holding out-of-sample?
- CARRY CAPTURE — realized funding / projected funding over the carry legs: is the carry sleeve
  actually banking the carry it underwrote? (signed; skipped, never /0, when there are no legs)
- SENTIMENT HIT-RATE — of the legs where sentiment took a non-neutral stance, the fraction that
  earned ALPHA in that direction: is sentiment adding alpha or noise?
- REVIEWER VETO-RATE — fraction of reviewed cycles the reviewer vetoed (`passed is False`): a
  process KPI on how often the guardian had to HALT a book. (spec §18)
- ALPHA-SHARPE TREND — rolling ALPHA-Sharpe (return net of BTC-beta, annualized ×365) slope over
  a window: is risk-adjusted alpha trending up? (spec §12/§18)

The reused `deployment_rate`/`corpus_health`/`return_trend` weekly patterns are kept (folded into
the panel) for continuity; the six neutral KPIs are the Phase-6 additions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from futures_fund.journal import read_all_decisions
from futures_fund.metrics import PERIODS_PER_YEAR_DAILY, sharpe
from futures_fund.neutrality import NeutralityConfig

# Cadence-root invariant (§14): artifacts live under `state/<cadence>/cycle/<N>/`. The legacy
# single-loop `state/cycle/<N>/` is still scanned for back-compat. We glob across both so a KPI
# reads every cycle artifact regardless of which root the writer used.
_CADENCE_GLOBS = ("weekly/cycle/*", "daily/cycle/*", "cycle/*")


def _cycle_dirs(state_dir) -> list[Path]:
    """All `cycle/<N>/` artifact dirs across cadence roots, ordered by cycle number (ascending).

    Dedupes on resolved path so a repo that wrote under both a cadence and the legacy root is not
    double-counted. A non-numeric cycle name sorts first (-1) so it never masks a real cycle."""
    base = Path(state_dir)
    seen: dict[Path, None] = {}
    for g in _CADENCE_GLOBS:
        for d in base.glob(g):
            if d.is_dir():
                seen.setdefault(d.resolve(), None)
    return sorted(
        seen.keys(),
        key=lambda p: int(p.name) if p.name.isdigit() else -1,
    )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _decisions_by_cycle(memory_dir, last_n: int) -> list[dict]:
    """The decision outcome dicts for the most-recent `last_n` cycles, oldest-first.

    Windows on CYCLE (not on decision count): a cycle can open several legs, and the KPIs aggregate
    per-leg over the window, so we keep every leg whose cycle is among the last `last_n` cycles."""
    decisions = read_all_decisions(memory_dir)
    if not decisions:
        return []
    cycles = sorted({d.get("cycle") for d in decisions if d.get("cycle") is not None})
    keep = set(cycles[-last_n:])
    in_window = [d for d in decisions if d.get("cycle") in keep]
    return sorted(in_window, key=lambda d: (d.get("cycle", -1), d.get("symbol", "")))


# --------------------------------------------------------------------------------------------------
# Reused weekly patterns (kept for continuity; folded into the panel)
# --------------------------------------------------------------------------------------------------
def deployment_rate(state_dir, last_n: int = 10) -> dict:
    """Over the last `last_n` cycle reports: fraction that deployed risk (opened a position OR armed
    a trigger). Near-zero = the under-deployment alarm. Reads `state/<cadence>/cycle/*/report.json`.

    The production report.json (`orchestration.gate_execute_step`) records `executed` and `triggers`
    as LISTS — not `opened`/`triggers_armed` counts — so a cycle is "active" when either list is
    non-empty, and `opens` counts the executed legs.
    """
    reports = [d / "report.json" for d in _cycle_dirs(state_dir)]
    reports = [p for p in reports if p.exists()][-last_n:]
    n = len(reports)
    if n == 0:
        return {"deployment_rate": 0.0, "cycles": 0, "active": 0, "opens": 0}
    active = opens = 0
    for p in reports:
        r = _read_json(p)
        if r is None:
            continue
        executed = r.get("executed") or []
        triggers = r.get("triggers") or []
        o = len(executed)
        opens += o
        if o > 0 or len(triggers) > 0:
            active += 1
    return {"deployment_rate": round(active / n, 3), "cycles": n, "active": active, "opens": opens}


def corpus_health(memory_dir) -> dict:
    """Lessons-corpus two-sidedness: counts by polarity + validated count. `two_sided` = the corpus
    carries BOTH enabling and restrictive lessons (not a one-way 'don't' ratchet)."""
    try:
        from futures_fund.lessons import read_lessons

        lessons = read_lessons(memory_dir)
    except Exception:  # noqa: BLE001 — advisory; never break the cycle
        return {
            "total": 0, "validated": 0, "enabling": 0, "restrictive": 0, "process": 0,
            "two_sided": False,
        }
    pol = {"enabling": 0, "restrictive": 0, "process": 0}
    validated = 0
    for lz in lessons:
        p = getattr(lz, "polarity", "restrictive")
        pol[p] = pol.get(p, 0) + 1
        if getattr(lz, "state", "candidate") == "validated":
            validated += 1
    return {
        "total": len(lessons), "validated": validated, **pol,
        "two_sided": pol["enabling"] > 0 and pol["restrictive"] > 0,
    }


# --------------------------------------------------------------------------------------------------
# Neutral KPIs (Phase 6 additions)
# --------------------------------------------------------------------------------------------------
def both_sides_deployment_rate(state_dir, last_n: int = 10) -> float:
    """Fraction of the last `last_n` cycles where BOTH sides cleared the deployment floor.

    Numerator: cycles whose `target_weights.json` has `deploy_long_frac >= floor` AND
    `deploy_short_frac >= floor`. Denominator: `last_n` cycles present (a cycle missing its
    `target_weights.json` counts in the denominator but never the numerator). Guards BOTH the
    all-cash ratchet (neither side deployed) AND the one-sided ratchet (only one side deployed) —
    a "deployed" but long-only book is not neutral. (spec §12/§19)
    """
    floor = NeutralityConfig().deployment_floor
    # Window on ALL cycle dirs first, THEN look for the artifact — so a cycle that ran but emitted
    # no `target_weights.json` (a halt / all-cash ratchet, the very failure this KPI exists to
    # catch) stays in the denominator and lands as "not both-sides-deployed" rather than being
    # silently dropped. (See docstring + roadmap line 884: "denominator: last_n cycles present".)
    dirs = _cycle_dirs(state_dir)[-last_n:]
    n = len(dirs)
    if n == 0:
        return 0.0
    both = 0
    for d in dirs:
        tw = _read_json(d / "target_weights.json")
        if tw is None:
            continue
        long_frac = float(tw.get("deploy_long_frac", 0.0) or 0.0)
        short_frac = float(tw.get("deploy_short_frac", 0.0) or 0.0)
        if long_frac >= floor and short_frac >= floor:
            both += 1
    return both / n


def pair_survival_rate(state_dir, last_n: int = 10) -> float:
    """Cointegrated-at-retest / total-retested over the window.

    Numerator: pairs still cointegrated at their weekly re-test (ADF `p < 0.05`, read from the
    journal outcome's `adf_pvalue_at_retest`). Denominator: pairs that reached a re-test (i.e. the
    field is present). Returns `nan` when no pair reached a re-test (skip, don't /0)."""
    retested = [
        d for d in _decisions_by_cycle(state_dir, last_n)
        if d.get("adf_pvalue_at_retest") is not None
    ]
    if not retested:
        return math.nan
    survived = sum(1 for d in retested if float(d["adf_pvalue_at_retest"]) < 0.05)
    return survived / len(retested)


def carry_capture_rate(state_dir, last_n: int = 10) -> float:
    """Σ realized_funding / Σ projected_funding over the carry legs in the window (SIGNED).

    A carry leg is a decision whose outcome carries both `realized_funding` (from
    `funding_intervals.realized_funding`) and `projected_funding` (from `costs.project_funding` at
    entry). Returns `nan` when there are no carry legs (skip, don't /0) and clamps the denominator
    away from 0 so a zero-projected book never blows up."""
    legs = [
        d for d in _decisions_by_cycle(state_dir, last_n)
        if d.get("realized_funding") is not None and d.get("projected_funding") is not None
    ]
    if not legs:
        return math.nan
    realized = sum(float(d["realized_funding"]) for d in legs)
    projected = sum(float(d["projected_funding"]) for d in legs)
    if projected == 0.0:
        return math.nan  # nothing projected -> capture ratio is undefined, don't divide by zero
    return realized / projected


def sentiment_hit_rate(memory_dir, last_n: int = 10) -> float:
    """sentiment_correct / sentiment_nonneutral over the window.

    Numerator: legs where sentiment took a non-neutral stance that earned ALPHA in that direction —
    sentiment positive (long stance) AND `alpha_return > 0`, OR sentiment negative (short stance)
    on a short leg that made alpha (`alpha_return > 0`). Denominator: legs where sentiment was
    non-neutral (`|sentiment_score| > 0`). A neutral-sentiment leg is excluded from both. Returns
    `nan` when sentiment never took a stance (skip, don't /0)."""
    legs = [
        d for d in _decisions_by_cycle(memory_dir, last_n)
        if d.get("sentiment_score") is not None and float(d["sentiment_score"]) != 0.0
    ]
    if not legs:
        return math.nan
    correct = 0
    for d in legs:
        s = float(d["sentiment_score"])
        alpha = float(d.get("alpha_return", 0.0) or 0.0)
        direction = d.get("direction")
        # Sentiment is correct when its stance agrees with the leg's side AND the leg made alpha:
        #   s > 0 (positive/long stance) on a long leg, or s < 0 (negative/short stance) on a short
        #   leg, with alpha_return > 0.
        stance_matches_side = (s > 0 and direction == "long") or (s < 0 and direction == "short")
        if stance_matches_side and alpha > 0:
            correct += 1
    return correct / len(legs)


def reviewer_veto_rate(state_dir, last_n: int = 10) -> float:
    """vetoed / reviewed over the window.

    Numerator: cycles whose persisted `reviewer.json` has `passed is False`. Denominator: cycles
    with a `reviewer.json` present in the window. Returns `nan` when no cycle was reviewed (skip,
    don't /0). (spec §18 process KPI)"""
    dirs = [d for d in _cycle_dirs(state_dir) if (d / "reviewer.json").exists()][-last_n:]
    if not dirs:
        return math.nan
    vetoed = 0
    reviewed = 0
    for d in dirs:
        v = _read_json(d / "reviewer.json")
        if v is None:
            continue
        reviewed += 1
        if v.get("passed") is False:
            vetoed += 1
    if reviewed == 0:
        return math.nan
    return vetoed / reviewed


def alpha_sharpe_trend(state_dir, window: int = 8) -> float:
    """Rolling ALPHA-Sharpe slope over the window.

    Builds the per-cycle alpha series (`alpha_return`, return net of BTC-beta) from the journal
    outcomes (averaging legs within a cycle), computes a rolling alpha-Sharpe (annualized ×365) over
    each trailing `window` of cycles, and returns the OLS slope of that rolling-Sharpe series — a
    positive slope means risk-adjusted alpha is trending up. Returns 0.0 when there are too few
    cycles to form ≥ 2 rolling points. (spec §12/§18; reuses `metrics.sharpe`.)"""
    decisions = read_all_decisions(state_dir)
    by_cycle: dict[int, list[float]] = {}
    for d in decisions:
        if d.get("alpha_return") is None or d.get("cycle") is None:
            continue
        by_cycle.setdefault(int(d["cycle"]), []).append(float(d["alpha_return"]))
    if not by_cycle:
        return 0.0
    series = [sum(v) / len(v) for _, v in sorted(by_cycle.items())]
    if len(series) < window + 1:
        return 0.0
    rolling = [
        sharpe(series[i - window : i], periods_per_year=PERIODS_PER_YEAR_DAILY)
        for i in range(window, len(series) + 1)
    ]
    if len(rolling) < 2:
        return 0.0
    x = np.arange(len(rolling), dtype=float)
    slope = float(np.polyfit(x, np.asarray(rolling, dtype=float), 1)[0])
    return slope


# --------------------------------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------------------------------
def improvement_panel(state_dir, memory_dir, *, last_n: int = 10) -> dict:
    """Bundle the read-only improvement signals for the scorecard / meta-reflection.

    Keeps the reused weekly signals (`deployment`, `corpus`) for continuity and adds the six neutral
    process KPIs (re-keyed on alpha vs BTC-beta). The journal-backed KPIs read `memory_dir` (where
    the episodic journal lives); the artifact-backed KPIs read `state_dir`."""
    return {
        "deployment": deployment_rate(state_dir, last_n=last_n),
        "corpus": corpus_health(memory_dir),
        "both_sides_deployment_rate": both_sides_deployment_rate(state_dir, last_n=last_n),
        "pair_survival_rate": pair_survival_rate(memory_dir, last_n=last_n),
        "carry_capture_rate": carry_capture_rate(memory_dir, last_n=last_n),
        "sentiment_hit_rate": sentiment_hit_rate(memory_dir, last_n=last_n),
        "reviewer_veto_rate": reviewer_veto_rate(state_dir, last_n=last_n),
        "alpha_sharpe_trend": alpha_sharpe_trend(memory_dir, window=last_n),
    }

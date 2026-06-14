"""Deterministic lesson MINER — link 3 of the closed learning loop.

The real-cadence loop runs WITHOUT the LLM Reflector, so candidate lessons are distilled
mechanically from the alpha-keyed closed decisions ``learning.close_alpha_outcomes`` writes. Groups
closed decisions by ``(sleeve, direction)``; a cohort with a clear, repeated alpha edge yields a
two-sided candidate lesson — ``restrictive`` on persistent alpha BLEED, ``enabling`` on persistent
alpha GAIN — so the corpus self-heals symmetrically and never ratchets into an all-restrictive
never-trade state.

Confirmation tracks GENUINELY NEW evidence: a cohort only confirms its lesson when it cites a
closing decision not already in the lesson's ``provenance`` (so re-mining an unchanged corpus is a
no-op). ``statistically_promote`` then gates the candidate -> validated transition on the desk's
DSR — a noisy early edge can be a low-importance CANDIDATE (a small, bounded read-back tilt) but
never a VALIDATED standing rule until its alpha is statistically proven. Pure bookkeeping: this
proposes tilts for the optimizer to dispose of; it never sizes, trades, or relaxes a limit.
"""
from __future__ import annotations

from datetime import datetime

from futures_fund.journal import alpha_outcome, read_all_decisions
from futures_fund.lessons import (
    append_lesson,
    read_lessons,
    statistically_promote,
    update_lesson,
)


def _closed_alpha_decisions(memory_dir):
    """Every decision whose six alpha-vs-beta outcome fields are fully patched (i.e. closed)."""
    out = []
    for d in read_all_decisions(memory_dir):
        try:
            ao = alpha_outcome(d)
        except KeyError:
            continue  # still open / partially patched -> not a closed outcome
        out.append((d, ao))
    return out


def _sleeve(d: dict) -> str:
    return d.get("sleeve") or d.get("setup") or "unknown"


def _dominant_dimension(rows) -> str | None:
    """The most frequent neutral failure mode across a (losing) cohort, or None if none occurred."""
    counts = {
        "cointegration_break": 0,
        "carry_thesis_miss": 0,
        "neutrality_breach": 0,
        "sentiment_detract": 0,
    }
    for d, ao in rows:
        if not ao.pair_cointegrated_at_exit:
            counts["cointegration_break"] += 1
        if not ao.funding_thesis_matched:
            counts["carry_thesis_miss"] += 1
        if not ao.neutrality_in_band:
            counts["neutrality_breach"] += 1
        if (d.get("sentiment_score") or 0.0) != 0.0 and not ao.sentiment_helped:
            counts["sentiment_detract"] += 1
    dim, n = max(counts.items(), key=lambda kv: kv[1])
    return dim if n > 0 else None


def mine_lessons(
    memory_dir,
    *,
    now: datetime,
    dsr_pvalue: float,
    min_closes: int = 3,
    min_alpha: float = 0.001,
    skew: float = 0.6,
    importance_cap: int = 10,
) -> dict:
    """Distil candidate lessons from closed decisions; confirm + DSR-promote on NEW evidence.

    Returns a small summary ``{appended, confirmed, cohorts}`` for logging. A cohort qualifies only
    with at least ``min_closes`` closes AND a mean alpha past ``min_alpha`` in a direction that at
    least ``skew`` of the cohort agrees with — a weak/mixed cohort emits nothing."""
    rows = _closed_alpha_decisions(memory_dir)
    cohorts: dict[tuple[str, str], list] = {}
    for d, ao in rows:
        cohorts.setdefault((_sleeve(d), d["direction"]), []).append((d, ao))

    existing = read_lessons(memory_dir)
    appended = confirmed = 0
    for (sleeve, direction), crows in cohorts.items():
        n = len(crows)
        if n < min_closes:
            continue
        alphas = [ao.alpha_return for _, ao in crows]
        mean = sum(alphas) / n
        losers = sum(1 for a in alphas if a < 0)
        winners = sum(1 for a in alphas if a > 0)
        if mean <= -min_alpha and losers / n >= skew:
            polarity = "restrictive"
            dim = _dominant_dimension(crows)
            text = (f"{sleeve} {direction} legs bled alpha (mean {mean:.2%} over {n} closes) — "
                    "DOWN-WEIGHT this sleeve/side in the next book.")
        elif mean >= min_alpha and winners / n >= skew:
            polarity = "enabling"
            dim = None
            text = (f"{sleeve} {direction} legs banked alpha (mean {mean:.2%} over {n} closes) — "
                    "KEEP/UP-WEIGHT this sleeve/side.")
        else:
            continue
        tags = [sleeve, direction]
        decision_ids = sorted({d.get("id") for d, _ in crows if d.get("id")})
        importance = max(1, min(importance_cap, round(3 + abs(mean) * 1000 + (n - min_closes))))
        sig = (polarity, dim, tuple(sorted(tags)))
        match = next(
            (lz for lz in existing
             if (lz.polarity, lz.dimension, tuple(sorted(lz.tags))) == sig),
            None,
        )
        if match is None:
            append_lesson(memory_dir, {
                "text": text, "regime": None, "tags": tags, "dimension": dim,
                "importance": importance, "polarity": polarity, "state": "candidate",
                "provenance": decision_ids,
            }, ts=now)
            appended += 1
            continue
        new_ids = [i for i in decision_ids if i not in match.provenance]
        if new_ids:
            statistically_promote(memory_dir, match.id, dsr_pvalue=dsr_pvalue)
            update_lesson(
                memory_dir, match.id,
                provenance=sorted(set(match.provenance) | set(decision_ids)),
                text=text, importance=importance,
            )
            confirmed += 1
    return {"appended": appended, "confirmed": confirmed, "cohorts": len(cohorts)}

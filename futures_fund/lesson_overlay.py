"""Lesson READ-BACK overlay — link 4 of the closed learning loop.

Turns the corpus the miner maintains into a BOUNDED tilt on the next book's sleeve convictions, so a
lesson the desk actually learned measurably changes what it trades next. It is applied to the
``SleeveSignal`` tilts BEFORE ``optimize_book``, which then re-projects onto the dollar+beta-neutral
set and re-applies the per-name / cluster caps + the deployment floor — so a lesson can only
re-shape RELATIVE conviction WITHIN the alpha legs; it can NEVER break neutrality, breach a cap, or
move deployment. The learned edge proposes; the optimizer disposes.

A VALIDATED lesson (a DSR-proven standing rule) applies the full per-lesson delta; a CANDIDATE
applies a fraction of it (so an unproven edge nudges but never commits). ``restrictive`` lessons
pull a cohort's conviction DOWN, ``enabling`` lessons pull it UP, and stacked factors are clamped to
a safe band so no accumulation of lessons can dominate the optimizer's own risk shaping.
"""
from __future__ import annotations

from futures_fund.contracts import SleeveSignal

_DIRECTIONS = ("long", "short")


def lesson_tilt_factors(
    lessons,
    *,
    candidate_weight: float = 0.3,
    per_lesson_delta: float = 0.10,
    clamp: tuple[float, float] = (0.5, 1.5),
) -> dict[tuple[str, str], float]:
    """Map ``(sleeve, direction)`` -> a bounded multiplicative tilt factor from the corpus.

    The miner tags every lesson ``[sleeve, direction]``; a factor starts at 1.0 and each matching
    lesson adds ``±delta`` (restrictive negative, enabling positive), scaled by ``candidate_weight``
    for a not-yet-validated lesson, then the total is clamped. Retired and ``process`` lessons are
    ignored. A cohort with no lesson is simply absent (treated as factor 1.0 downstream)."""
    factors: dict[tuple[str, str], float] = {}
    for lz in lessons:
        if lz.state == "retired" or lz.polarity not in ("restrictive", "enabling"):
            continue
        dirs = [t for t in lz.tags if t in _DIRECTIONS]
        sleeves = [t for t in lz.tags if t not in _DIRECTIONS]
        if not dirs or not sleeves:
            continue
        key = (sleeves[0], dirs[0])
        strength = per_lesson_delta * (1.0 if lz.state == "validated" else candidate_weight)
        sign = -1.0 if lz.polarity == "restrictive" else 1.0
        factors[key] = factors.get(key, 1.0) + sign * strength
    lo, hi = clamp
    return {k: min(hi, max(lo, v)) for k, v in factors.items()}


def apply_lesson_overlay(sleeves: list[SleeveSignal], lessons, **kw) -> list[SleeveSignal]:
    """Scale each ``SleeveTilt.target_weight`` by its ``(sleeve, direction)`` lesson factor.

    Pure: returns a NEW ``SleeveSignal`` list and never mutates the inputs. With an empty corpus (or
    no matching lessons) the sleeves are returned unchanged, so this is a safe no-op until the desk
    has actually learned something. Neutrality/caps/deployment are re-enforced by ``optimize_book``
    downstream, so this only re-weights relative conviction within the alpha legs."""
    factors = lesson_tilt_factors(lessons, **kw)
    if not factors:
        return sleeves
    out: list[SleeveSignal] = []
    for s in sleeves:
        scaled = []
        for t in s.tilts:
            f = factors.get((s.sleeve, t.direction), 1.0)
            scaled.append(t.model_copy(update={"target_weight": t.target_weight * f}))
        out.append(s.model_copy(update={"tilts": scaled}))
    return out

"""Two-phase lessons corpus, ported/adapted from the weekly desk's `futures_fund.lessons`
(verify+merge) and re-keyed on market-neutral ALPHA (return net of BTC-beta, §10).

The canonical `Lesson`/`Polarity`/`LessonState` already live in `futures_fund.contracts` — this
module reuses them rather than redefining (single source of truth) and supplies the storage +
retrieval + promotion machinery. Phase 6 / Task 6.2 adds the neutral `dimension` failure modes
(`cointegration_break`, `carry_thesis_miss`, `neutrality_breach`, `sentiment_detract`): the
retrieval filter (`score_lesson`) reads the new `dimension` tag so a dimension query surfaces the
matching lesson above an untagged one. The DSR-gated promotion (`statistically_promote`) is carried
over unchanged."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from futures_fund.contracts import Lesson


def _store(memory_dir) -> Path:
    return Path(memory_dir) / "lessons" / "lessons.jsonl"


def append_lesson(memory_dir, fields: dict, ts: datetime) -> str:
    data = {**fields, "ts": ts}
    data.setdefault("id", uuid.uuid4().hex)
    lesson = Lesson.model_validate(data)
    p = _store(memory_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(lesson.model_dump_json() + "\n")
    return lesson.id


def read_lessons(memory_dir) -> list[Lesson]:
    p = _store(memory_dir)
    if not p.exists():
        return []
    return [Lesson.model_validate_json(line) for line in p.read_text().splitlines() if line.strip()]


def score_lesson(lesson: Lesson, now: datetime, query_tags: list[str],
                 w_rec: float = 1.0, w_imp: float = 1.0, w_rel: float = 1.0) -> float:
    """Generative-Agents-style score: recency (Ebbinghaus) + importance + tag relevance (Jaccard).

    Task 6.2: tag relevance now reads the neutral `dimension` failure-mode in addition to the
    free-form `tags`, so a query for a dimension (e.g. `cointegration_break`) surfaces a lesson
    carrying that `dimension` even when its `tags` do not overlap the query."""
    hours = max(0.0, (now - lesson.ts).total_seconds() / 3600.0)
    recency = 0.995 ** hours
    importance = lesson.importance / 10.0
    qt = set(query_tags)
    lt = set(lesson.tags)
    if lesson.dimension is not None:
        lt.add(lesson.dimension)
    relevance = len(qt & lt) / len(qt | lt) if (qt or lt) else 0.0
    return w_rec * recency + w_imp * importance + w_rel * relevance


def retrieve_lessons(memory_dir, now: datetime, regime: str | None,
                     query_tags: list[str], k: int = 5,
                     max_restrictive: int = 3) -> list[Lesson]:
    """Regime-filter FIRST (a lesson with regime=None applies everywhere), rank by score, then
    apply a POLARITY QUOTA so the injected set is two-sided: VALIDATED lessons (standing rules)
    are always kept; >=1 enabling lesson is force-included when any exists; and restrictive
    *fills* are capped at `max_restrictive` so a losing record's prohibitions can't monopolize
    every debate. Retired lessons excluded.

    NOTE: passing regime=None as the QUERY matches only universal (lz.regime is None) lessons,
    NOT all lessons; pass a regime string to also include lessons tagged to that regime."""
    candidates = [
        lz for lz in read_lessons(memory_dir)
        if lz.state != "retired" and (lz.regime is None or lz.regime == regime)
    ]
    candidates.sort(key=lambda lz: score_lesson(lz, now, query_tags), reverse=True)

    validated = [lz for lz in candidates if lz.state == "validated"]
    pool = [lz for lz in candidates if lz.state != "validated"]
    out: list[Lesson] = list(validated)  # standing rules are never dropped by the quota

    # The effective size floor never truncates away a validated standing rule (spec §6), so the
    # quota is applied relative to it -- NOT to k. Gating the enabling/two-sided guarantees behind
    # `len(out) < k` would silently abandon them once a desk accrues >= k validated standing rules
    # (which are by design never dropped), letting the injected set become entirely one-sided
    # restrictive -- exactly the all-restrictive never-trade ratchet this function must prevent.
    cap = max(k, len(validated))

    # Force-include the highest-scored enabling lesson whenever one exists and none is in the set
    # yet -- UNCONDITIONALLY, so the injected set stays two-sided even past the validated floor.
    forced: Lesson | None = None
    if not any(lz.polarity == "enabling" for lz in out):
        forced = next((lz for lz in pool if lz.polarity == "enabling"), None)
        if forced is not None:
            out.append(forced)

    # Fill the remaining slots up to the effective floor by score, capping restrictive FILLS
    # (validated standing rules already counted, not subject to the fill cap).
    n_restrict = 0
    for lz in pool:
        if lz in out:
            continue
        if len(out) >= cap:
            break
        if lz.polarity == "restrictive" and n_restrict >= max_restrictive:
            continue  # don't flood the debate with prohibitions
        out.append(lz)
        if lz.polarity == "restrictive":
            n_restrict += 1

    out.sort(key=lambda lz: score_lesson(lz, now, query_tags), reverse=True)
    kept = out[:cap]  # never truncate away a validated standing rule
    # If the force-included enabling lesson scored out of the top `cap` (possible when validated
    # standing rules fill the floor), pin it back in so two-sidedness survives truncation.
    if forced is not None and forced not in kept:
        kept.append(forced)
    return kept


def _write_all(memory_dir, lessons: list[Lesson]) -> None:
    p = _store(memory_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lz.model_dump_json() + "\n" for lz in lessons))


def update_lesson(memory_dir, lesson_id: str, **fields) -> bool:
    """Merge `fields` into the lesson with `lesson_id`; rewrites the store. False if not found."""
    lessons = read_lessons(memory_dir)
    hit = False
    for i, lz in enumerate(lessons):
        if lz.id == lesson_id:
            lessons[i] = lz.model_copy(update=fields)
            hit = True
    if hit:
        _write_all(memory_dir, lessons)
    return hit


def confirm_lesson(memory_dir, lesson_id: str, *, promote_threshold: int = 5) -> bool:
    """Increment a lesson's confirmation count; promote CANDIDATE -> VALIDATED at the threshold.
    (Count-based here; statistical support gates promotion additionally — see
    `statistically_promote`, spec §6.)"""
    lessons = read_lessons(memory_dir)
    hit = False
    for i, lz in enumerate(lessons):
        if lz.id == lesson_id:
            c = lz.confirmations + 1
            state = (
                "validated"
                if (lz.state == "candidate" and c >= promote_threshold)
                else lz.state
            )
            lessons[i] = lz.model_copy(update={"confirmations": c, "state": state})
            hit = True
    if hit:
        _write_all(memory_dir, lessons)
    return hit


def demote_lesson(memory_dir, lesson_id: str) -> bool:
    """Step a lesson down: VALIDATED -> CANDIDATE, CANDIDATE/RETIRED -> RETIRED.
    Used to aggressively age out stale or regime-mismatched rules (spec §6).
    Resets confirmations to 0 so a demoted lesson must re-earn promotion
    (anti-ossification, spec §6)."""
    lessons = read_lessons(memory_dir)
    hit = False
    for i, lz in enumerate(lessons):
        if lz.id == lesson_id:
            new = "candidate" if lz.state == "validated" else "retired"
            lessons[i] = lz.model_copy(update={"state": new, "confirmations": 0})
            hit = True
    if hit:
        _write_all(memory_dir, lessons)
    return hit


def retire_lesson(memory_dir, lesson_id: str) -> bool:
    return update_lesson(memory_dir, lesson_id, state="retired")


def validated_lessons(memory_dir) -> list[Lesson]:
    """The VALIDATED lessons — these act as hard vetoes / standing rules for the team."""
    return [lz for lz in read_lessons(memory_dir) if lz.state == "validated"]


def statistically_promote(memory_dir, lesson_id: str, *, dsr_pvalue: float,
                          promote_threshold: int = 5, dsr_threshold: float = 0.95) -> bool:
    """Confirm a lesson, but only allow CANDIDATE->VALIDATED promotion when the desk's edge is
    statistically proven (DSR p-value >= threshold). Below the gate the confirmation still
    counts, but the lesson stays CANDIDATE — the statistical layer over the count-based rule
    (spec §6). Returns True if the lesson was found."""
    lessons = read_lessons(memory_dir)
    hit = False
    for i, lz in enumerate(lessons):
        if lz.id == lesson_id:
            c = lz.confirmations + 1
            promote = (lz.state == "candidate" and c >= promote_threshold
                       and dsr_pvalue >= dsr_threshold)
            lessons[i] = lz.model_copy(update={"confirmations": c,
                                               "state": "validated" if promote else lz.state})
            hit = True
    if hit:
        _write_all(memory_dir, lessons)
    return hit

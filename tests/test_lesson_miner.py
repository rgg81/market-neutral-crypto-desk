"""Stage B — deterministic lesson MINER (link 3 of the closed loop).

The desk's real-cadence loop has no LLM Reflector, so lessons must be distilled MECHANICALLY from
the alpha-keyed closed decisions the capture layer now writes. `mine_lessons` groups closed
decisions by (sleeve, direction), and for a cohort with a clear, repeated alpha edge emits a
two-sided candidate lesson (restrictive on persistent alpha bleed, enabling on persistent alpha
gain). New evidence (a not-yet-seen closing decision) CONFIRMS the lesson, and DSR gates the
candidate -> validated promotion — so a noisy early edge can never become a standing rule.
"""
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.journal import append_decision, patch_outcome
from futures_fund.lesson_miner import mine_lessons
from futures_fund.lessons import read_lessons

NOW = datetime(2026, 6, 15, tzinfo=UTC)


def _close(memory_dir, *, cycle, symbol, direction, sleeve, alpha, n_cycle=None):
    """Journal an open + patch a full alpha outcome so the decision reads as 'closed'."""
    append_decision(memory_dir, cycle=cycle, symbol=symbol, direction=direction,
                    payload={"sleeve": sleeve, "setup": sleeve, "beta_btc": 1.0}, cadence="weekly")
    patch_outcome(memory_dir, cycle=cycle, symbol=symbol, direction=direction, cadence="weekly",
                  outcome={"alpha_return": alpha, "beta_contribution": 0.0,
                           "pair_cointegrated_at_exit": True, "funding_thesis_matched": alpha >= 0,
                           "neutrality_in_band": True, "sentiment_helped": alpha > 0})


def test_losing_cohort_mints_one_restrictive_lesson(tmp_path):
    mem = tmp_path / "memory"
    for i, a in enumerate([-0.02, -0.015, -0.03]):  # factor shorts bleed alpha 3x
        _close(mem, cycle=i + 1, symbol=f"X{i}/USDT:USDT", direction="short",
               sleeve="factor", alpha=a)
    summary = mine_lessons(mem, now=NOW, dsr_pvalue=0.5)
    lessons = read_lessons(mem)
    assert summary["appended"] == 1 and len(lessons) == 1
    lz = lessons[0]
    assert lz.polarity == "restrictive"
    assert set(lz.tags) >= {"factor", "short"}
    assert lz.state == "candidate"        # not promoted (no confirmations, weak DSR)
    assert len(lz.provenance) == 3        # cites the 3 closing decisions


def test_winning_cohort_mints_enabling_lesson(tmp_path):
    mem = tmp_path / "memory"
    for i, a in enumerate([0.02, 0.018, 0.025]):
        _close(mem, cycle=i + 1, symbol=f"Y{i}/USDT:USDT", direction="long",
               sleeve="carry", alpha=a)
    mine_lessons(mem, now=NOW, dsr_pvalue=0.5)
    lz = read_lessons(mem)[0]
    assert lz.polarity == "enabling"
    assert set(lz.tags) >= {"carry", "long"}


def test_no_new_evidence_is_idempotent_but_new_close_confirms(tmp_path):
    mem = tmp_path / "memory"
    for i, a in enumerate([-0.02, -0.015, -0.03]):
        _close(mem, cycle=i + 1, symbol=f"Z{i}/USDT:USDT", direction="short",
               sleeve="factor", alpha=a)
    mine_lessons(mem, now=NOW, dsr_pvalue=0.5)
    first = read_lessons(mem)[0]
    assert first.confirmations == 0
    # re-mine with NO new closes -> no append, no extra confirmation (provenance unchanged)
    s2 = mine_lessons(mem, now=NOW, dsr_pvalue=0.5)
    again = read_lessons(mem)[0]
    assert s2["appended"] == 0 and again.confirmations == 0 and len(again.provenance) == 3
    # a NEW closing decision in the same cohort -> confirmation increments, provenance grows
    _close(mem, cycle=9, symbol="Z9/USDT:USDT", direction="short", sleeve="factor", alpha=-0.04)
    s3 = mine_lessons(mem, now=NOW, dsr_pvalue=0.5)
    grown = read_lessons(mem)[0]
    assert s3["confirmed"] == 1 and grown.confirmations == 1 and len(grown.provenance) == 4


def test_dsr_gates_promotion_to_validated(tmp_path):
    mem = tmp_path / "memory"
    # seed a cohort + drive 5 confirmations on NEW evidence each time, under a WEAK DSR.
    _close(mem, cycle=0, symbol="A0/USDT:USDT", direction="short", sleeve="factor", alpha=-0.02)
    _close(mem, cycle=1, symbol="A1/USDT:USDT", direction="short", sleeve="factor", alpha=-0.02)
    _close(mem, cycle=2, symbol="A2/USDT:USDT", direction="short", sleeve="factor", alpha=-0.02)
    mine_lessons(mem, now=NOW, dsr_pvalue=0.5)
    for c in range(3, 9):  # six more new closes, weak DSR -> confirmations climb, stays candidate
        _close(mem, cycle=c, symbol=f"A{c}/USDT:USDT", direction="short",
               sleeve="factor", alpha=-0.02)
        mine_lessons(mem, now=NOW, dsr_pvalue=0.5)
    weak = read_lessons(mem)[0]
    assert weak.confirmations >= 5 and weak.state == "candidate"  # DSR gate holds it back
    # one more confirmation under a STRONG DSR -> promote to validated (standing rule)
    _close(mem, cycle=99, symbol="A99/USDT:USDT", direction="short", sleeve="factor", alpha=-0.02)
    mine_lessons(mem, now=NOW, dsr_pvalue=0.99)
    assert read_lessons(mem)[0].state == "validated"

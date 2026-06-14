"""Capstone — the learning loop actually CLOSES end-to-end (all four links compose).

capture (journal open + alpha-at-close) -> mine (DSR-gated lesson) -> read-back (overlay changes the
next book's sleeve convictions). This is the proof the loop is no longer write-only: a repeated,
DSR-proven alpha bleed on a cohort measurably down-weights that cohort in the next book.
"""
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import ClosedLeg, PaperAccount, Position
from futures_fund.contracts import SleeveSignal, SleeveTilt
from futures_fund.journal import patch_outcome
from futures_fund.learning import close_alpha_outcomes, journal_open_legs
from futures_fund.lesson_miner import mine_lessons
from futures_fund.lesson_overlay import apply_lesson_overlay
from futures_fund.lessons import read_lessons, validated_lessons

NOW = datetime(2026, 6, 15, tzinfo=UTC)
BTC = "BTC/USDT:USDT"


def _open_then_close_factor_short(mem, i, *, btc_move):
    """Capture a full open->close round-trip for a factor SHORT that LOST alpha (price rose against
    the short by more than its beta explains)."""
    sym = f"N{i}/USDT:USDT"
    acc = PaperAccount(cash=20_000.0)
    acc.positions[sym] = Position(symbol=sym, direction="short", qty=100.0, entry_price=10.0,
                                  opened_ts=NOW, opened_cycle=i, opened_cadence="weekly")
    journal_open_legs(mem, acc, cycle=i, cadence="weekly",
                      leg_meta={sym: {"beta_btc": 0.5, "sleeve": "factor", "pair_id": None,
                                      "sentiment_score": 0.0, "regime": None}},
                      marks={BTC: 100.0, sym: 10.0}, funding_by_symbol={sym: 0.0}, btc_symbol=BTC)
    # short loses $200 on price (entry_notional 1000 -> -20% net), BTC barely moved -> alpha < 0.
    closed = [ClosedLeg(symbol=sym, direction="short", opened_cycle=i, opened_cadence="weekly",
                        fees=1.0, slippage=1.0, realized_funding=0.0, realized_pnl=-200.0)]
    for cyc, cad, csym, d, outcome in close_alpha_outcomes(
            mem, closed, marks={BTC: btc_move, sym: 12.0}, btc_symbol=BTC):
        patch_outcome(mem, cycle=cyc, symbol=csym, direction=d, outcome=outcome, cadence=cad)


def test_full_loop_capture_to_readback(tmp_path):
    mem = tmp_path / "memory"
    # Six losing factor-short closes, mined under a STRONG DSR -> the lesson earns promotion.
    for i in range(8):
        _open_then_close_factor_short(mem, i, btc_move=101.0)  # BTC ~flat -> the loss is alpha
        mine_lessons(mem, now=NOW, dsr_pvalue=0.99)

    # link 3 result: a VALIDATED restrictive (factor, short) standing rule exists.
    val = validated_lessons(mem)
    assert any(lz.polarity == "restrictive" and set(lz.tags) >= {"factor", "short"} for lz in val)

    # link 4 result: the next book's factor SHORT convictions are down-weighted, longs untouched.
    sleeves = [SleeveSignal(sleeve="factor", risk_budget_frac=1.0, as_of_ts=NOW, tilts=[
        SleeveTilt(symbol="A/USDT:USDT", direction="long", target_weight=0.5),
        SleeveTilt(symbol="B/USDT:USDT", direction="short", target_weight=-0.5),
    ])]
    out = apply_lesson_overlay(sleeves, read_lessons(mem))
    by_dir = {t.direction: t.target_weight for t in out[0].tilts}
    assert by_dir["short"] == -0.5 * 0.9   # validated restrictive -> full -10% down-weight
    assert by_dir["long"] == 0.5           # the long side is untouched

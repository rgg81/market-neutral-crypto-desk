"""Stage A — deterministic learning CAPTURE (journal-at-entry + alpha-at-close).

These pin the two links that were missing from the deterministic run path (run_paper_cli runs
WITHOUT the LLM Reflector): nothing journaled a Decision at entry, so the journal stayed empty and
every reflection_input was {winners:[], losers:[], n_closed:0}. `learning.journal_open_legs` records
each held leg's entry context; `learning.close_alpha_outcomes` computes the six market-neutral ALPHA
outcome fields (return net of BTC-beta, net of costs) so a closed leg becomes a real winner/loser.
"""
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import ClosedLeg, PaperAccount, Position
from futures_fund.journal import alpha_outcome, patch_outcome, read_all_decisions
from futures_fund.learning import (
    carry_expected_sign,
    close_alpha_outcomes,
    journal_open_legs,
)
from scripts.reflect_cli import build_reflection_input

BTC = "BTC/USDT:USDT"


def _account_with(*positions: Position) -> PaperAccount:
    acc = PaperAccount(cash=20_000.0)
    for p in positions:
        acc.positions[p.symbol] = p
    return acc


def test_carry_expected_sign_follows_who_receives_funding():
    # positive rate: longs PAY shorts -> a SHORT receives (+1), a LONG pays (-1)
    assert carry_expected_sign("short", 0.01) == 1
    assert carry_expected_sign("long", 0.01) == -1
    # negative rate: shorts pay longs -> a LONG receives (+1), a SHORT pays (-1)
    assert carry_expected_sign("long", -0.01) == 1
    assert carry_expected_sign("short", -0.01) == -1
    assert carry_expected_sign("long", 0.0) == 0


def test_journal_open_legs_records_entry_context_and_is_idempotent(tmp_path):
    mem = tmp_path / "memory"
    acc = _account_with(
        Position(symbol="ETH/USDT:USDT", direction="long", qty=10.0, entry_price=100.0,
                 opened_ts=datetime(2026, 6, 14, tzinfo=UTC), opened_cycle=3,
                 opened_cadence="daily"),
    )
    leg_meta = {"ETH/USDT:USDT": {"beta_btc": 0.9, "sleeve": "factor", "pair_id": None,
                                  "sentiment_score": 0.2, "regime": "btc_up_range"}}
    n = journal_open_legs(mem, acc, cycle=3, cadence="daily", leg_meta=leg_meta,
                          marks={BTC: 50_000.0, "ETH/USDT:USDT": 100.0},
                          funding_by_symbol={"ETH/USDT:USDT": -0.01}, btc_symbol=BTC)
    assert n == 1
    decs = read_all_decisions(mem)
    assert len(decs) == 1
    d = decs[0]
    # keyed on the leg's OWN open cycle/cadence, carrying the alpha-attribution entry context
    assert (d["cycle"], d["symbol"], d["direction"], d["cadence"]) == (
        3, "ETH/USDT:USDT", "long", "daily")
    assert d["entry"] == 100.0 and d["size"] == 10.0
    assert d["beta_btc"] == 0.9 and d["btc_mark_at_entry"] == 50_000.0
    assert d["carry_expected_sign"] == 1  # long on negative funding RECEIVES
    # idempotent: re-journaling the same held book appends nothing
    journal_open_legs(mem, acc, cycle=3, cadence="daily", leg_meta=leg_meta,
                      marks={BTC: 50_000.0, "ETH/USDT:USDT": 100.0},
                      funding_by_symbol={"ETH/USDT:USDT": -0.01}, btc_symbol=BTC)
    assert len(read_all_decisions(mem)) == 1


def test_close_alpha_outcome_strips_beta_and_validates(tmp_path):
    mem = tmp_path / "memory"
    # A long ETH leg: entry notional 10*100 = 1000. Price rose 5% -> realized_pnl 50.
    acc = _account_with(
        Position(symbol="ETH/USDT:USDT", direction="long", qty=10.0, entry_price=100.0,
                 opened_ts=datetime(2026, 6, 14, tzinfo=UTC), opened_cycle=3,
                 opened_cadence="daily"),
    )
    journal_open_legs(mem, acc, cycle=3, cadence="daily",
                      leg_meta={"ETH/USDT:USDT": {"beta_btc": 1.0, "sleeve": "factor",
                                                  "pair_id": None, "sentiment_score": 0.0,
                                                  "regime": None}},
                      marks={BTC: 100.0, "ETH/USDT:USDT": 100.0},
                      funding_by_symbol={"ETH/USDT:USDT": 0.0}, btc_symbol=BTC)
    closed = [ClosedLeg(symbol="ETH/USDT:USDT", direction="long", opened_cycle=3,
                        opened_cadence="daily", fees=1.0, slippage=1.0, realized_funding=2.0,
                        realized_pnl=50.0)]
    # BTC rose 4% over the hold; net_pnl = 50+2-1-1 = 50 -> net_return 0.05.
    outs = close_alpha_outcomes(mem, closed, marks={BTC: 104.0, "ETH/USDT:USDT": 105.0},
                                btc_symbol=BTC, neutrality_in_band=True)
    assert len(outs) == 1
    cyc, cad, sym, direction, outcome = outs[0]
    assert (cyc, cad, sym, direction) == (3, "daily", "ETH/USDT:USDT", "long")
    # beta_contribution = sign(+1) * beta(1.0) * btc_return(0.04) = 0.04
    assert abs(outcome["beta_contribution"] - 0.04) < 1e-9
    # alpha = net_return(0.05) - beta(0.04) = 0.01
    assert abs(outcome["alpha_return"] - 0.01) < 1e-9
    # patching makes the Decision a fully-formed, validatable closed outcome
    assert patch_outcome(mem, cycle=cyc, symbol=sym, direction=direction, outcome=outcome,
                         cadence=cad) is True
    ao = alpha_outcome(read_all_decisions(mem)[0])
    assert abs(ao.alpha_return - 0.01) < 1e-9 and ao.neutrality_in_band is True
    # and the reflection input now finds a real winner (alpha > 0), not an empty set
    payload = build_reflection_input(mem)
    assert payload["n_closed"] == 1 and len(payload["winners"]) == 1


def test_close_without_journaled_entry_is_cost_only_failsoft(tmp_path):
    mem = tmp_path / "memory"
    # leg opened before capture was active -> no Decision -> cost-only patch, no alpha fields.
    closed = [ClosedLeg(symbol="SOL/USDT:USDT", direction="short", opened_cycle=1,
                        opened_cadence="weekly", fees=0.5, slippage=0.5, realized_funding=3.0,
                        realized_pnl=-2.0)]
    outs = close_alpha_outcomes(mem, closed, marks={BTC: 104.0}, btc_symbol=BTC)
    assert len(outs) == 1
    _, _, _, _, outcome = outs[0]
    assert "alpha_return" not in outcome  # un-attributable -> no alpha fields
    assert outcome["realized_funding"] == 3.0  # but carry/cost bookkeeping still flows

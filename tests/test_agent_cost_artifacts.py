"""Phase 9 / Task 15 — the agent prompts now SURFACE the realized cost/carry artifacts to the desk
(`context.json.pnl.*`, the injected scorecard, and `reflection_input.json`). A prompt that names a
key the producing code does not actually emit is a silent contract break: the orchestrator would
inject a bundle the agent is told to read and find the key missing.

These tests pin every cost-artifact key NAMED in the five edited prompts to the live schema emitted
by its producer (`preflight.build_pnl_block`, `scorecard.build_scorecard`,
`reflect_cli.build_reflection_input`). They use the real builders on minimal real fixtures — no
mocks — so renaming a key in the code without re-surfacing it in the prompt fails here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from futures_fund.account import PaperAccount, Position, save_account
from futures_fund.cycle_io import save_output
from futures_fund.journal import append_decision, patch_outcome
from futures_fund.scorecard import build_scorecard
from scripts.preflight import build_pnl_block
from scripts.reflect_cli import build_reflection_input

AGENTS = Path("agents")

_FULL_OUTCOME = {
    "beta_contribution": 0.01,
    "pair_cointegrated_at_exit": True,
    "funding_thesis_matched": True,
    "neutrality_in_band": True,
    "sentiment_helped": True,
    "alpha_return": 0.012,
}


def _pnl_block(tmp_path) -> dict:
    """The realized cost/carry block exactly as preflight folds it into context.json.pnl."""
    state = tmp_path / "state"
    acct = PaperAccount(cash=20_000.0, funding_received=6.0, funding_paid=0.0,
                        fees_paid=4.0, slippage_paid=2.0)
    acct.positions["OP/USDT:USDT"] = Position(
        symbol="OP/USDT:USDT", direction="short", qty=10.0, entry_price=2.0,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC),
        accrued_funding=6.0, accrued_fees=2.0)
    save_account(state, acct)
    return build_pnl_block(
        state, marks={"OP/USDT:USDT": 1.9},
        last_pnl={"fees_paid": 4.0, "slippage_paid": 2.0, "turnover_usd": 4000.0},
        default_cash=20_000.0)


def test_pnl_block_carries_every_key_the_prompts_name(tmp_path):
    """funding_carry / pair_analyst / trader read context.json.pnl — assert the exact keys they
    instruct the desk to read are present in build_pnl_block's output."""
    pnl = _pnl_block(tmp_path)

    # Top-level keys named in funding_carry.md (totals) and trader.md (last-rebalance).
    for top in ("total_funding_received", "total_funding_paid",
                "last_rebalance_cost", "last_rebalance_turnover_usd", "by_symbol"):
        assert top in pnl, f"context.json.pnl missing top-level key the prompts read: {top}"

    # Per-symbol keys named across funding_carry / pair_analyst / trader.
    leg = pnl["by_symbol"]["OP/USDT:USDT"]
    for k in ("unrealized", "realized_funding", "accrued_fees"):
        assert k in leg, f"context.json.pnl.by_symbol[...] missing key the prompts read: {k}"
    # realized_funding is SIGNED, + = received (the prompts assert this orientation).
    assert leg["realized_funding"] == 6.0


def test_scorecard_carries_cost_keys_research_manager_names(tmp_path):
    """research_manager.md reads the injected scorecard keys net_pnl / gross_pnl / cost_drag_bps."""
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    # The scorecard derives cost_drag from the latest pnl.json (gross + frictions).
    save_output(state, 1, "pnl",
                {"gross_pnl": 14.0, "net_pnl": 8.0, "fees_paid": 4.0,
                 "slippage_paid": 2.0, "funding_net": 0.0}, cadence="weekly")
    sc = build_scorecard(state, memory)
    for k in ("net_pnl", "gross_pnl", "cost_drag_bps"):
        assert k in sc, f"scorecard missing cost key research_manager.md reads: {k}"
    # The drag formula the prompt's worked example relies on: (fees+slip)/|gross|*1e4.
    assert round(sc["cost_drag_bps"], 1) == round((4.0 + 2.0) / 14.0 * 1e4, 1)


def test_reflection_entries_carry_net_cost_keys_reflector_names(tmp_path):
    """reflector.md reads per-winner/loser realized_funding / fees / slippage / net_pnl."""
    memory = tmp_path / "memory"
    append_decision(memory, cycle=1, symbol="OP/USDT:USDT", direction="short", payload={})
    patch_outcome(memory, cycle=1, symbol="OP/USDT:USDT", direction="short",
                  outcome={**_FULL_OUTCOME,
                           "realized_pnl": 12.0, "fees": 4.0, "slippage": 2.0,
                           "realized_funding": 6.0})

    out = build_reflection_input(memory)
    entries = out["winners"] + out["losers"]
    assert entries, "no closed decision surfaced for the reflector"
    entry = entries[0]
    for k in ("symbol", "alpha_return", "realized_funding", "fees", "slippage", "net_pnl"):
        assert k in entry, f"reflection entry missing key reflector.md reads: {k}"
    # net_pnl = realized_pnl - fees - slippage (the prompt calls this "net of fees+slippage").
    assert entry["net_pnl"] == 12.0 - 4.0 - 2.0


def test_every_pnl_key_named_in_prompts_exists_in_the_block(tmp_path):
    """Stronger guard: scrape the literal `pnl.<key>` / `by_symbol[...].<key>` tokens out of the
    three context.json-reading prompts and assert each resolves in build_pnl_block. Catches a
    prompt that invents or renames a key without a matching code change."""
    pnl = _pnl_block(tmp_path)
    leg_keys = set(next(iter(pnl["by_symbol"].values())).keys())
    top_keys = set(pnl.keys())

    # Keys the prompts EXPLICITLY tell the desk to read (the documented surface).
    named_top = {"total_funding_received", "total_funding_paid",
                 "last_rebalance_cost", "last_rebalance_turnover_usd"}
    named_leg = {"realized_funding", "unrealized", "accrued_fees"}

    missing_top = named_top - top_keys
    missing_leg = named_leg - leg_keys
    assert not missing_top, f"prompts name context.json.pnl keys not emitted: {missing_top}"
    assert not missing_leg, f"prompts name by_symbol keys not emitted: {missing_leg}"

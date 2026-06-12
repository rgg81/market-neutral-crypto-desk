"""Phase 6 / Task 6.4 — behavioral tests for the two net-new self-improvement CLIs.

`test_skill_md.py` only asserts these filenames appear in the SKILL.md ladder; the REAL logic is
exercised here (matching how `control_loop_cli` / `monitor_cli` are covered by dedicated tests):

  * `reflect_cli.build_reflection_input` — split closed decisions into winners/losers by realized
    ALPHA, including the `alpha_return == 0 -> loser` boundary and skip-on-incomplete-outcome.
  * `promote_lesson_cli.main` — the DSR-gated `confirm` wiring: pull the alpha-series DSR p-value
    from `build_scorecard` and hand it to `statistically_promote` (promote only past the 0.95 gate).
"""

from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.cycle_io import load_output
from futures_fund.journal import append_decision, patch_outcome
from futures_fund.lessons import append_lesson, read_lessons

NOW = datetime(2026, 6, 1, tzinfo=UTC)

# A fully-patched alpha outcome (the six alpha-vs-beta fields) parametrized only on alpha_return.
_FULL = {
    "beta_contribution": 0.01,
    "pair_cointegrated_at_exit": True,
    "funding_thesis_matched": True,
    "neutrality_in_band": True,
    "sentiment_helped": True,
}


def _close(memory_dir, *, cycle, symbol, direction, alpha_return) -> None:
    append_decision(memory_dir, cycle=cycle, symbol=symbol, direction=direction, payload={})
    patch_outcome(memory_dir, cycle=cycle, symbol=symbol, direction=direction,
                  outcome={**_FULL, "alpha_return": alpha_return})


# --------------------------------------------------------------------------------------------------
# build_reflection_input — winners/losers split by realized alpha
# --------------------------------------------------------------------------------------------------
def test_build_reflection_input_splits_by_alpha_with_zero_as_loser(tmp_path):
    from scripts.reflect_cli import build_reflection_input

    memory = tmp_path / "memory"
    _close(memory, cycle=1, symbol="BTC/USDT:USDT", direction="long", alpha_return=0.03)  # winner
    _close(memory, cycle=2, symbol="ETH/USDT:USDT", direction="short", alpha_return=-0.02)  # loser
    # Boundary: exactly zero alpha is a LOSER (strict `> 0` winner test), not a winner.
    _close(memory, cycle=3, symbol="SOL/USDT:USDT", direction="long", alpha_return=0.0)
    # An OPEN decision (no outcome patched) must be SKIPPED, not counted.
    append_decision(memory, cycle=4, symbol="XRP/USDT:USDT", direction="long", payload={})

    out = build_reflection_input(memory)

    assert {w["symbol"] for w in out["winners"]} == {"BTC/USDT:USDT"}
    assert {lo["symbol"] for lo in out["losers"]} == {"ETH/USDT:USDT", "SOL/USDT:USDT"}
    # zero-alpha leg landed in losers (independent expected for the == 0 boundary)
    assert any(lo["symbol"] == "SOL/USDT:USDT" and lo["alpha_return"] == 0.0
               for lo in out["losers"])
    # n_closed counts ONLY the three patched decisions; the open one is skipped.
    assert out["n_closed"] == 3
    assert out["n_closed"] == len(out["winners"]) + len(out["losers"])


def test_build_reflection_input_skips_partially_patched_outcome(tmp_path):
    from scripts.reflect_cli import build_reflection_input

    memory = tmp_path / "memory"
    # Patch only SOME of the six alpha-outcome fields -> alpha_outcome raises KeyError -> skipped.
    append_decision(memory, cycle=1, symbol="BTC/USDT:USDT", direction="long", payload={})
    patch_outcome(memory, cycle=1, symbol="BTC/USDT:USDT", direction="long",
                  outcome={"alpha_return": 0.05, "beta_contribution": 0.0})  # 4 fields missing
    _close(memory, cycle=2, symbol="ETH/USDT:USDT", direction="long", alpha_return=0.04)

    out = build_reflection_input(memory)
    assert out["n_closed"] == 1  # only the fully-patched ETH leg
    assert {w["symbol"] for w in out["winners"]} == {"ETH/USDT:USDT"}


def test_reflect_cli_main_persists_under_cadence_root(tmp_path, monkeypatch, capsys):
    import json

    from scripts.reflect_cli import main

    memory = tmp_path / "memory"
    _close(memory, cycle=5, symbol="BTC/USDT:USDT", direction="long", alpha_return=0.03)
    monkeypatch.chdir(tmp_path)

    main(["--cadence", "weekly", "--cycle", "5", "--memory-dir", str(memory)])

    # Persisted under the cadence-segmented cycle root (CADENCE INVARIANT), not relied on the LLM.
    persisted = load_output(tmp_path / "state", 5, "reflection_input", cadence="weekly")
    assert persisted["n_closed"] == 1
    # And stdout is the same parseable payload.
    assert json.loads(capsys.readouterr().out)["n_closed"] == 1


# --------------------------------------------------------------------------------------------------
# promote_lesson_cli — DSR-gated `confirm` path (alpha-series DSR -> statistically_promote)
# --------------------------------------------------------------------------------------------------
def _candidate_at_threshold(memory_dir) -> str:
    """A candidate lesson with 4 confirmations — one shy of the count threshold (5), so the next
    `confirm` decides promotion purely on the DSR gate."""
    return append_lesson(
        memory_dir,
        {"text": "carry edge held", "polarity": "enabling", "tags": ["carry"],
         "state": "candidate", "confirmations": 4},
        ts=NOW,
    )


def test_promote_cli_confirm_promotes_when_alpha_dsr_clears_gate(tmp_path, capsys):
    from scripts.promote_lesson_cli import main

    memory = tmp_path / "memory"
    state = tmp_path / "state"
    lid = _candidate_at_threshold(memory)
    # Seed a strong, steady WINNING alpha series (>=10 cycles) so build_scorecard's DSR >= 0.95.
    for c in range(1, 13):
        _close(memory, cycle=c, symbol="BTC/USDT:USDT", direction="long",
               alpha_return=0.02 + (0.001 if c % 2 else -0.001))

    main(["--id", lid, "--action", "confirm",
          "--state-dir", str(state), "--memory-dir", str(memory)])

    lz = next(lz for lz in read_lessons(memory) if lz.id == lid)
    # 5th confirmation WITH DSR support -> CANDIDATE promoted to VALIDATED.
    assert lz.state == "validated"
    assert lz.confirmations == 5
    assert "confirm" in capsys.readouterr().out


def test_promote_cli_confirm_stays_candidate_when_alpha_dsr_below_gate(tmp_path, capsys):
    from scripts.promote_lesson_cli import main

    memory = tmp_path / "memory"
    state = tmp_path / "state"
    lid = _candidate_at_threshold(memory)
    # Seed a LOSING alpha series -> DSR well below 0.95; the count threshold alone cannot promote.
    for c in range(1, 13):
        _close(memory, cycle=c, symbol="BTC/USDT:USDT", direction="long",
               alpha_return=-0.02 + (0.001 if c % 2 else -0.001))

    main(["--id", lid, "--action", "confirm",
          "--state-dir", str(state), "--memory-dir", str(memory)])

    lz = next(lz for lz in read_lessons(memory) if lz.id == lid)
    # Confirmation still counts (now 5) but the lesson stays CANDIDATE — the DSR gate held.
    assert lz.state == "candidate"
    assert lz.confirmations == 5


def test_reflection_entries_carry_realized_costs():
    from scripts.reflect_cli import _cost_fields

    decision = {
        "fees": 3.0, "slippage": 1.5, "funding_paid": -2.0,
        "realized_funding": 2.0, "realized_pnl": 12.0,
    }
    costs = _cost_fields(decision)
    assert costs["fees"] == 3.0
    assert costs["slippage"] == 1.5
    assert costs["realized_funding"] == 2.0
    assert costs["net_pnl"] == 12.0 - 3.0 - 1.5    # realized_pnl net of fees+slippage


def test_cost_fields_default_zero_on_missing():
    from scripts.reflect_cli import _cost_fields
    costs = _cost_fields({})
    assert costs == {"fees": 0.0, "slippage": 0.0, "realized_funding": 0.0, "net_pnl": 0.0}

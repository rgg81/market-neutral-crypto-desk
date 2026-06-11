"""Phase 6 / Task 6.4 — alpha scorecard injection + walk-forward graduation gate + self-heal loop.

The scorecard is the desk's statistical self-portrait injected into every agent prompt. Re-keyed on
the market-neutral process: it folds in the Phase-6 neutral KPIs (`both_sides_deployment_rate`,
`pair_survival_rate`, `carry_capture_rate`, `sentiment_hit_rate`, `reviewer_veto_rate`,
`alpha_sharpe_trend`) and keeps the warnings DELIBERATELY TWO-SIDED — a drawdown/risk-off brake AND
an under-deployment accelerator (so the injected context never becomes a one-way ratchet that talks
the desk out of every clean neutral trade or out of deploying both sides).

`graduation_verdict` gains the binding walk-forward gate: a sleeve-param change is only trusted once
its OOS Sharpe clears the DSR threshold — an in-sample-only grid winner is REJECTED
(`walk_forward_required`).

The self-healing repair loop (`repair.py`) logs every repair to `memory/repair-journal.md` and
REFUSES to weaken a protected (risk/execution-critical) module.
"""

from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.cycle_io import save_output
from futures_fund.graduation import graduation_verdict
from futures_fund.journal import append_decision, patch_outcome
from futures_fund.repair import (
    apply_repair,
    is_protected,
    record_repair,
)
from futures_fund.scorecard import build_scorecard

FLOOR = 0.90


def _target_weights(*, long_frac: float, short_frac: float) -> dict:
    return {
        "legs": [],
        "btc_hedge_notional": 0.0,
        "dollar_residual": 0.0,
        "dollar_residual_frac": 0.0,
        "beta_residual": 0.0,
        "gross_long": long_frac * 10000.0,
        "gross_short": short_frac * 10000.0,
        "deploy_long_frac": long_frac,
        "deploy_short_frac": short_frac,
        "gross_notional": (long_frac + short_frac) * 10000.0,
        "as_of_ts": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    }


def _report(*, cycle: int, executed: list[dict], triggers: list[dict]) -> dict:
    return {
        "loop": "weekly",
        "cycle": cycle,
        "ran_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "live": False,
        "executed": executed,
        "dropped": [],
        "management": [],
        "triggers": triggers,
        "cancel_triggers": [],
    }


def _reviewer(*, cycle: int, passed: bool) -> dict:
    return {
        "passed": passed,
        "checks": [],
        "mismatches": [] if passed else ["dollar_residual_in_band"],
        "cycle": cycle,
        "cadence": "weekly",
        "reviewed_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    }


def _seed_leg(memory_dir, *, cycle, symbol, direction, outcome) -> None:
    append_decision(memory_dir, cycle=cycle, symbol=symbol, direction=direction, payload={})
    patch_outcome(memory_dir, cycle=cycle, symbol=symbol, direction=direction, outcome=outcome)


# --------------------------------------------------------------------------------------------------
# build_scorecard — injects the neutral KPIs incl reviewer_veto_rate + alpha_sharpe_trend
# --------------------------------------------------------------------------------------------------
def test_build_scorecard_injects_neutral_kpis(tmp_path):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    # Two cycles: one both-sides deployed + passed review, one vetoed.
    save_output(state, 1, "target_weights", _target_weights(long_frac=0.95, short_frac=0.93),
                cadence="weekly")
    save_output(state, 1, "reviewer", _reviewer(cycle=1, passed=True), cadence="weekly")
    save_output(state, 2, "target_weights", _target_weights(long_frac=0.95, short_frac=0.92),
                cadence="weekly")
    save_output(state, 2, "reviewer", _reviewer(cycle=2, passed=False), cadence="weekly")
    _seed_leg(memory, cycle=1, symbol="BTC/USDT:USDT", direction="short",
              outcome={"realized_funding": 6.0, "projected_funding": 6.0,
                       "adf_pvalue_at_retest": 0.01, "alpha_return": 0.02})

    sc = build_scorecard(state, memory)
    for key in (
        "both_sides_deployment_rate",
        "pair_survival_rate",
        "carry_capture_rate",
        "sentiment_hit_rate",
        "reviewer_veto_rate",
        "alpha_sharpe_trend",
        "warnings",
    ):
        assert key in sc
    # reviewer_veto_rate: 1 veto / 2 reviewed = 0.5
    assert sc["reviewer_veto_rate"] == 0.5
    assert sc["both_sides_deployment_rate"] == 1.0
    assert sc["carry_capture_rate"] == 1.0


def test_scorecard_warnings_are_two_sided(tmp_path):
    """The warnings carry BOTH a drawdown/risk-off brake AND an under-deployment accelerator — a
    one-way ratchet (only brakes) is the bug this two-sidedness exists to prevent."""
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    # Under-deployed: several cycles all-cash / one-sided -> accelerator must fire.
    for i in range(1, 7):
        save_output(state, i, "target_weights", _target_weights(long_frac=0.0, short_frac=0.0),
                    cadence="weekly")
        save_output(state, i, "reviewer", _reviewer(cycle=i, passed=True), cadence="weekly")

    sc = build_scorecard(state, memory)
    text = " ".join(sc["warnings"]).lower()
    # Accelerator present (counter-signal against the all-cash / one-sided ratchet).
    assert "deploy" in text or "under-deployed" in text
    # And the brake side is structurally available (drawdown / risk-off vocabulary), proving the
    # warnings are not an accelerator-only ratchet either.
    assert any(
        w_key in " ".join(build_scorecard(state, memory, _force_drawdown=True)["warnings"]).lower()
        for w_key in ("drawdown", "risk-off")
    )


# --------------------------------------------------------------------------------------------------
# graduation_verdict — walk-forward OOS gate blocks an in-sample-only sleeve-param change
# --------------------------------------------------------------------------------------------------
def test_graduation_requires_walk_forward_oos_pass():
    # A param change with a strong IS Sharpe but NO OOS walk-forward pass is REJECTED.
    v = graduation_verdict(
        n_cycles=30, sharpe=2.0, dsr_pvalue=0.97, beats_baseline=True, max_dd=0.08,
        walk_forward_required=True, walk_forward_passed=False,
    )
    assert v["status"] != "graduated"
    assert any("walk" in r.lower() or "oos" in r.lower() for r in v["reasons"])


def test_graduation_graduates_with_walk_forward_oos_pass():
    v = graduation_verdict(
        n_cycles=30, sharpe=2.0, dsr_pvalue=0.97, beats_baseline=True, max_dd=0.08,
        walk_forward_required=True, walk_forward_passed=True,
    )
    assert v["status"] == "graduated"
    assert v["reasons"] == []


def test_graduation_default_does_not_require_walk_forward():
    # Back-compat: without the new flags the verdict behaves exactly as before.
    v = graduation_verdict(
        n_cycles=30, sharpe=2.0, dsr_pvalue=0.97, beats_baseline=True, max_dd=0.08,
    )
    assert v["status"] == "graduated"


# --------------------------------------------------------------------------------------------------
# self-healing loop — logs every repair; REFUSES to weaken a protected module
# --------------------------------------------------------------------------------------------------
def test_apply_repair_refuses_protected_module(tmp_path):
    assert is_protected("futures_fund/risk_gate.py") is True
    result = apply_repair(
        tmp_path, module="futures_fund/risk_gate.py",
        symptom="RR floor too strict", root_cause="risk limit", fix="relax RR floor",
        verification="n/a", ts=datetime(2026, 6, 1, tzinfo=UTC),
    )
    assert result["applied"] is False
    assert "protected" in result["reason"].lower()
    # The refusal is still journaled (audit trail of the REFUSED attempt).
    md = (tmp_path / "repair-journal.md").read_text()
    assert "risk_gate" in md and "REFUSED" in md


def test_apply_repair_allows_non_protected_module_and_logs(tmp_path):
    result = apply_repair(
        tmp_path, module="futures_fund/news.py",
        symptom="news parser crashed on dict", root_cause="dict-wrapped payload",
        fix="tolerate dict", verification="487 tests green",
        ts=datetime(2026, 6, 1, tzinfo=UTC),
    )
    assert result["applied"] is True
    md = (tmp_path / "repair-journal.md").read_text()
    assert "news parser crashed on dict" in md
    assert "Verification" in md


def test_record_repair_appends_structured_entry(tmp_path):
    record_repair(tmp_path, symptom="screen crashed on dict input",
                  root_cause="analyst reports saved dict-wrapped",
                  fix="screen_step tolerates dict", verification="487 tests green",
                  ts=datetime(2026, 6, 1, tzinfo=UTC))
    md = (tmp_path / "repair-journal.md").read_text()
    assert "Symptom" in md and "Root cause" in md and "Verification" in md
    assert "screen crashed on dict input" in md

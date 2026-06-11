"""Phase 6 / Task 6.3 — neutral improvement-panel KPIs.

The improvement panel is re-pointed from raw-return signals onto the market-neutral process KPIs
that the desk's spec (§12/§18/§19) actually cares about: are BOTH sides being deployed (guarding
the all-cash AND the one-sided ratchet), are pairs still cointegrated at their weekly re-test, is
the carry sleeve capturing the funding it projected, is sentiment adding alpha, how often does the
reviewer veto, and is the rolling ALPHA-Sharpe (return net of BTC-beta) trending up.

Each KPI is a pure, read-only function mirroring `deployment_rate(state_dir, last_n)`. The tests
seed `state/<cadence>/cycle/*/{target_weights,reviewer}.json` via the real `cycle_io.save_output`
and the journal outcomes via the real `journal.append_decision`/`patch_outcome`, then assert exact
hand-computed numerators/denominators.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from futures_fund.cycle_io import save_output
from futures_fund.improvement import (
    alpha_sharpe_trend,
    both_sides_deployment_rate,
    carry_capture_rate,
    improvement_panel,
    pair_survival_rate,
    reviewer_veto_rate,
    sentiment_hit_rate,
)
from futures_fund.journal import append_decision, patch_outcome

FLOOR = 0.90


def _target_weights(*, long_frac: float, short_frac: float) -> dict:
    """Minimal `target_weights.json` payload carrying just the deployment fractions read."""
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


def _reviewer(*, cycle: int, passed: bool) -> dict:
    return {
        "passed": passed,
        "checks": [],
        "mismatches": [] if passed else ["dollar_residual_in_band"],
        "cycle": cycle,
        "cadence": "weekly",
        "reviewed_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    }


def _seed_journal_leg(
    memory_dir,
    *,
    cycle: int,
    symbol: str,
    direction: str,
    outcome: dict,
) -> None:
    append_decision(memory_dir, cycle=cycle, symbol=symbol, direction=direction, payload={})
    patch_outcome(memory_dir, cycle=cycle, symbol=symbol, direction=direction, outcome=outcome)


# --------------------------------------------------------------------------------------------------
# both_sides_deployment_rate — BOTH sides must clear the floor (guards all-cash AND one-sided)
# --------------------------------------------------------------------------------------------------
def test_both_sides_deployment_rate_requires_both_sides(tmp_path):
    # cycle 1: both sides deployed -> counts; cycle 2: long-only ratchet -> does NOT count;
    # cycle 3: all-cash -> does NOT count; cycle 4: both deployed -> counts. 2/4 = 0.5.
    save_output(tmp_path, 1, "target_weights", _target_weights(long_frac=0.95, short_frac=0.92),
                cadence="weekly")
    save_output(tmp_path, 2, "target_weights", _target_weights(long_frac=0.95, short_frac=0.10),
                cadence="weekly")
    save_output(tmp_path, 3, "target_weights", _target_weights(long_frac=0.0, short_frac=0.0),
                cadence="weekly")
    save_output(tmp_path, 4, "target_weights", _target_weights(long_frac=0.91, short_frac=0.93),
                cadence="weekly")
    assert both_sides_deployment_rate(tmp_path, last_n=4) == 0.5


# --------------------------------------------------------------------------------------------------
# pair_survival_rate — cointegrated_at_retest / total_retested (ADF p < 0.05 at re-test)
# --------------------------------------------------------------------------------------------------
def test_pair_survival_rate(tmp_path):
    # 4 pairs reach a re-test; 3 are still cointegrated (adf_pvalue_at_retest < 0.05). 3/4 = 0.75.
    for i, p in enumerate([0.01, 0.03, 0.20, 0.04], start=1):
        _seed_journal_leg(
            tmp_path, cycle=i, symbol=f"ALT{i}/USDT:USDT", direction="long",
            outcome={"adf_pvalue_at_retest": p},
        )
    assert pair_survival_rate(tmp_path, last_n=10) == 0.75


# --------------------------------------------------------------------------------------------------
# carry_capture_rate — Σ realized_funding / Σ projected_funding (signed); nan when no carry legs
# --------------------------------------------------------------------------------------------------
def test_carry_capture_rate(tmp_path):
    # realized 12.0 / projected 10.0 -> 1.2 (split across two carry legs).
    _seed_journal_leg(tmp_path, cycle=1, symbol="BTC/USDT:USDT", direction="short",
                      outcome={"realized_funding": 7.0, "projected_funding": 6.0})
    _seed_journal_leg(tmp_path, cycle=2, symbol="ETH/USDT:USDT", direction="short",
                      outcome={"realized_funding": 5.0, "projected_funding": 4.0})
    assert carry_capture_rate(tmp_path, last_n=10) == 1.2


def test_carry_capture_rate_no_carry_legs_is_nan(tmp_path):
    # A non-carry leg (no funding fields) must be skipped, not divided-by-zero.
    _seed_journal_leg(tmp_path, cycle=1, symbol="BTC/USDT:USDT", direction="long",
                      outcome={"alpha_return": 0.01})
    assert math.isnan(carry_capture_rate(tmp_path, last_n=10))


# --------------------------------------------------------------------------------------------------
# sentiment_hit_rate — sentiment_correct / sentiment_nonneutral
# --------------------------------------------------------------------------------------------------
def test_sentiment_hit_rate(tmp_path):
    # 3 non-neutral sentiment legs; 2 made alpha in the direction sentiment took -> 2/3.
    # leg A: sentiment +0.8 (long), alpha +0.02 -> correct
    _seed_journal_leg(tmp_path, cycle=1, symbol="A/USDT:USDT", direction="long",
                      outcome={"sentiment_score": 0.8, "alpha_return": 0.02})
    # leg B: sentiment -0.6 (short), alpha +0.01 on a short -> correct
    _seed_journal_leg(tmp_path, cycle=2, symbol="B/USDT:USDT", direction="short",
                      outcome={"sentiment_score": -0.6, "alpha_return": 0.01})
    # leg C: sentiment +0.5 (long), alpha -0.01 -> wrong
    _seed_journal_leg(tmp_path, cycle=3, symbol="C/USDT:USDT", direction="long",
                      outcome={"sentiment_score": 0.5, "alpha_return": -0.01})
    # leg D: sentiment 0.0 (neutral) -> excluded from denominator entirely
    _seed_journal_leg(tmp_path, cycle=4, symbol="D/USDT:USDT", direction="long",
                      outcome={"sentiment_score": 0.0, "alpha_return": 0.05})
    assert sentiment_hit_rate(tmp_path, last_n=10) == 2 / 3


# --------------------------------------------------------------------------------------------------
# reviewer_veto_rate — vetoed / reviewed (ReviewerVerdict.passed is False)
# --------------------------------------------------------------------------------------------------
def test_reviewer_veto_rate(tmp_path):
    # 4 cycles reviewed; cycle 3 vetoed (passed=False). 1/4 = 0.25.
    for i, ok in enumerate([True, True, False, True], start=1):
        save_output(tmp_path, i, "reviewer", _reviewer(cycle=i, passed=ok), cadence="weekly")
    assert reviewer_veto_rate(tmp_path, last_n=4) == 0.25


# --------------------------------------------------------------------------------------------------
# alpha_sharpe_trend — rolling ALPHA-Sharpe slope over the window
# --------------------------------------------------------------------------------------------------
def test_alpha_sharpe_trend_rising_series_has_positive_slope(tmp_path):
    # A monotonically rising alpha series -> the rolling alpha-Sharpe slope is > 0.
    alphas = [0.001, 0.002, 0.004, 0.007, 0.011, 0.016, 0.022, 0.030]
    for i, a in enumerate(alphas, start=1):
        _seed_journal_leg(tmp_path, cycle=i, symbol="BTC/USDT:USDT", direction="long",
                          outcome={"alpha_return": a})
    slope = alpha_sharpe_trend(tmp_path, window=4)
    assert slope > 0.0


# --------------------------------------------------------------------------------------------------
# improvement_panel — folds the six neutral KPIs in (reusing deployment/corpus/returns patterns)
# --------------------------------------------------------------------------------------------------
def test_improvement_panel_includes_neutral_kpis(tmp_path):
    save_output(tmp_path, 1, "target_weights", _target_weights(long_frac=0.95, short_frac=0.93),
                cadence="weekly")
    save_output(tmp_path, 1, "reviewer", _reviewer(cycle=1, passed=True), cadence="weekly")
    _seed_journal_leg(tmp_path, cycle=1, symbol="BTC/USDT:USDT", direction="short",
                      outcome={"realized_funding": 6.0, "projected_funding": 6.0,
                               "adf_pvalue_at_retest": 0.01, "sentiment_score": 0.7,
                               "alpha_return": 0.02})
    panel = improvement_panel(tmp_path, tmp_path, last_n=5)
    for key in (
        "both_sides_deployment_rate",
        "pair_survival_rate",
        "carry_capture_rate",
        "sentiment_hit_rate",
        "reviewer_veto_rate",
        "alpha_sharpe_trend",
    ):
        assert key in panel
    assert panel["both_sides_deployment_rate"] == 1.0
    assert panel["reviewer_veto_rate"] == 0.0
    assert panel["carry_capture_rate"] == 1.0

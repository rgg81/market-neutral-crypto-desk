# tests/test_run_paper_mark_guard.py
"""LOUD GUARD for the market-neutrality bug class: a non-flat intended leg with NO mark must NOT be
silently skipped by `apply_fills` (which would leave the held book non-neutral). `run_paper_cli`
pre-checks that every non-flat `reviewed.legs` symbol has a (positive) mark BEFORE `apply_fills`,
failing LOUDLY otherwise instead of producing a silently broken book."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from futures_fund.contracts import TargetWeights, WeightLeg


def _book(legs: list[WeightLeg]) -> TargetWeights:
    return TargetWeights(
        legs=legs, feasible=True, dollar_residual=0.0, dollar_residual_frac=0.0,
        beta_residual=0.0, gross_long=1000.0, gross_short=1000.0,
        deploy_long_frac=0.9, deploy_short_frac=0.9, gross_notional=2000.0,
        btc_hedge_notional=0.0, as_of_ts=datetime(2026, 6, 11, tzinfo=UTC),
    )


def test_assert_legs_priced_raises_when_a_nonflat_leg_has_no_mark():
    from scripts.run_paper_cli import _assert_legs_priced

    book = _book([
        WeightLeg(symbol="ETH/USDT:USDT", direction="long", weight=0.5,
                  target_notional=1000.0, beta_btc=1.0, sleeve="factor"),
        # the BTC hedge leg the optimizer always appends — but its mark is MISSING (the live bug).
        WeightLeg(symbol="BTC/USDT:USDT", direction="short", weight=-0.5,
                  target_notional=-1000.0, beta_btc=1.0, sleeve="hedge"),
    ])
    marks = {"ETH/USDT:USDT": 3000.0}  # BTC mark absent -> the hedge would be silently skipped
    with pytest.raises((ValueError, RuntimeError)) as exc:
        _assert_legs_priced(book, marks)
    # the error names the unpriced symbol so the operator can see WHAT broke neutrality.
    assert "BTC/USDT:USDT" in str(exc.value)


def test_assert_legs_priced_passes_when_every_nonflat_leg_is_priced():
    from scripts.run_paper_cli import _assert_legs_priced

    book = _book([
        WeightLeg(symbol="ETH/USDT:USDT", direction="long", weight=0.5,
                  target_notional=1000.0, beta_btc=1.0, sleeve="factor"),
        WeightLeg(symbol="BTC/USDT:USDT", direction="short", weight=-0.5,
                  target_notional=-1000.0, beta_btc=1.0, sleeve="hedge"),
    ])
    marks = {"ETH/USDT:USDT": 3000.0, "BTC/USDT:USDT": 60000.0}
    _assert_legs_priced(book, marks)  # must not raise


def test_assert_legs_priced_ignores_flat_legs():
    from scripts.run_paper_cli import _assert_legs_priced

    # a zero-notional (flat) leg is nothing to OPEN, so a missing mark there is harmless.
    book = _book([
        WeightLeg(symbol="ETH/USDT:USDT", direction="long", weight=0.5,
                  target_notional=1000.0, beta_btc=1.0, sleeve="factor"),
        WeightLeg(symbol="DOGE/USDT:USDT", direction="long", weight=0.0,
                  target_notional=0.0, beta_btc=1.0, sleeve="carry"),
    ])
    marks = {"ETH/USDT:USDT": 3000.0}  # DOGE mark absent but DOGE is flat -> OK
    _assert_legs_priced(book, marks)  # must not raise

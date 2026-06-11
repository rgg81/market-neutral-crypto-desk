import pytest

from futures_fund.models import (
    MmrBracket,
    PortfolioHealth,
    RegimeState,
    SymbolSpec,
    TradeProposal,
)
from futures_fund.risk_gate import GateInputs, evaluate


def _spec(min_notional: float = 5.0):
    return SymbolSpec(
        symbol="BTCUSDT", tick_size=0.1, step_size=0.001, min_notional=min_notional,
        mmr_brackets=[
            MmrBracket(
                notional_floor=0, notional_cap=50_000, mmr=0.004,
                maint_amount=0.0, max_leverage=125,
            ),
        ],
    )


def _proposal(direction="long", entry=100.0, stop=95.0, tps=(115.0,), **over):
    base = dict(
        symbol="BTCUSDT", direction=direction, entry=entry, stop=stop,
        take_profits=list(tps), atr=2.0, confidence=0.7, horizon_hours=8,
        funding_rate=0.0001,
    )
    base.update(over)
    return TradeProposal(**base)


def _inputs(**over):
    base = dict(
        proposal=_proposal(),
        spec=_spec(),
        regime=RegimeState(quadrant="low_vol_trend"),
        health=PortfolioHealth(equity=10_000.0, peak_equity=10_000.0),
        open_positions=[],
        daily_pnl_pct=0.0, weekly_pnl_pct=0.0, monthly_pnl_pct=0.0,
    )
    base.update(over)
    return GateInputs(**base)


# ---- inherited gate behavior (faithful lift; smoke coverage so the port stays green) ----

def test_clean_trade_is_approved_and_leverage_is_output():
    d = evaluate(_inputs())
    assert d.verdict == "approve"
    assert d.sized_trade.leverage > 0
    risk = d.sized_trade.qty * abs(100.0 - 95.0) / 10_000.0
    assert risk == pytest.approx(0.030, abs=3e-3)


def test_stressed_portfolio_vetoes_new_entry():
    d = evaluate(_inputs(health=PortfolioHealth(equity=5_500.0, peak_equity=10_000.0)))
    assert d.verdict == "veto"
    assert "flat" in d.reason.lower() or "stressed" in d.reason.lower()


def test_bad_rr_is_vetoed():
    d = evaluate(_inputs(proposal=_proposal(tps=(101.0,))))
    assert d.verdict == "veto"
    assert "rr" in d.reason.lower() or "reward" in d.reason.lower()


def test_min_notional_vetoes_subminimum_trade():
    d = evaluate(_inputs(spec=_spec(min_notional=1_000_000.0)))
    assert d.verdict == "veto"
    assert "notional" in d.reason.lower()


def test_short_trade_is_approved_with_liq_above_entry():
    prop = _proposal(direction="short", entry=100.0, stop=105.0, tps=(85.0,))
    d = evaluate(_inputs(proposal=prop))
    assert d.verdict == "approve"
    assert d.sized_trade.liq_price > 100.0


# ---- Task 5.1: carry-visibility fix (unclamped funding) ----

def test_unclamped_funding_shows_credit():
    """A short receiving POSITIVE funding earns a credit. With `unclamped_funding=True` the
    legacy `max(0.0, funding)` clamp is overridden, so the `CostEstimate.funding` goes NEGATIVE
    (a credit) and total cost falls — un-hiding the real carry the short genuinely receives.
    With the flag OFF, behavior is the legacy clamp (funding floored at 0.0)."""
    # Short on a positive funding rate => the SHORT RECEIVES funding => signed cost is negative.
    prop = _proposal(direction="short", entry=100.0, stop=105.0, tps=(85.0,),
                     funding_rate=0.0010, funding_interval_hours=1.0, horizon_hours=8)
    inp = _inputs(proposal=prop)

    clamped = evaluate(inp, unclamped_funding=False)
    unclamped = evaluate(inp, unclamped_funding=True)

    assert clamped.verdict in ("approve", "resize")
    assert unclamped.verdict in ("approve", "resize")

    # Legacy clamp hides the credit at 0.0; the unclamp reveals a genuine NEGATIVE funding cost.
    assert clamped.sized_trade.cost.funding == pytest.approx(0.0)
    assert unclamped.sized_trade.cost.funding < 0.0

    # Un-hiding the credit can only LOWER total cost (more attractive trade), never raise it.
    assert unclamped.sized_trade.cost.total < clamped.sized_trade.cost.total


def test_unclamp_does_not_resurrect_rr_floor_failure():
    """Step 5 monotonicity guard (protected-module rule). A trade that FAILED the RR>=2 floor
    under the clamp must NOT pass solely because the unclamp un-hid a funding credit. The RR
    floor is a purely-geometric check (take-profit/stop) and must be unweakened by the flag."""
    # Marginal short with a generous funding credit but a sub-2R take-profit (RR < 2 -> FAIL).
    marginal_short_with_credit = _proposal(
        direction="short", entry=100.0, stop=105.0, tps=(91.0,),  # reward 9 / risk 5 = 1.8R
        funding_rate=0.0050, funding_interval_hours=1.0, horizon_hours=8,
    )
    inp = _inputs(proposal=marginal_short_with_credit)

    clamped = evaluate(inp, unclamped_funding=False)
    unclamped = evaluate(inp, unclamped_funding=True)

    assert clamped.verdict == "veto"
    assert "rr" in clamped.reason.lower()
    # Default: the floor decision is NOT weakened by un-hiding the credit.
    assert unclamped.verdict == "veto"
    assert "rr" in unclamped.reason.lower()

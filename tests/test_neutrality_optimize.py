from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from futures_fund.contracts import SleeveSignal, SleeveTilt, TargetWeights
from futures_fund.neutrality import NeutralityConfig, optimize_book

NOW = datetime(2026, 6, 11, tzinfo=UTC)


def test_optimize_book_returns_target_weights(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(
        sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg
    )
    assert isinstance(tw, TargetWeights)
    assert tw.feasible is True
    assert tw.as_of_ts is not None


def test_optimize_book_sets_per_side_deployment_and_gross(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    assert tw.gross_long > 0.0
    assert tw.gross_short > 0.0
    assert tw.gross_notional == tw.gross_long + tw.gross_short


def test_optimize_book_each_leg_has_target_notional(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    assert len(tw.legs) > 0
    for leg in tw.legs:
        assert leg.target_notional != 0.0
        assert leg.beta_btc != 0.0


def test_optimize_book_includes_hedge_leg_when_residual_beta(geometries):
    # A beta-imbalanced book: long the HIGH-beta name (SOL 1.5), short the LOW-beta name
    # (XRP 0.8). The alpha legs carry a NET LONG beta, so the BTC hedge MUST be a non-zero
    # SHORT BTC leg that absorbs it (the hedge is a real DOF, sized before projection).
    s = SleeveSignal(
        sleeve="factor",
        risk_budget_frac=1.0,
        as_of_ts=NOW,
        tilts=[
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )
    cfg = NeutralityConfig()
    tw = optimize_book([s], geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    hedge_legs = [leg for leg in tw.legs if leg.sleeve == "hedge"]
    # NON-vacuous: this beta-imbalanced book REQUIRES a materialized BTC hedge leg.
    assert tw.btc_hedge_notional < 0.0  # net long beta => short BTC hedge
    assert hedge_legs
    assert hedge_legs[0].symbol == "BTC/USDT:USDT"
    assert hedge_legs[0].direction == "short"


def _balanced_sleeves(now):
    """A risk-budgeted, dollar-balanced two-sleeve signal for property tests (>=3 active
    names per side after the hedge is appended, so projection is non-trivial)."""
    factor = SleeveSignal(
        sleeve="factor", risk_budget_frac=0.5, as_of_ts=now,
        tilts=[
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )
    carry = SleeveSignal(
        sleeve="carry", risk_budget_frac=0.5, as_of_ts=now,
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )
    return [factor, carry]


@pytest.mark.parametrize("seed", range(8))
def test_property_dollar_residual_within_band(seed, geometries):
    rng = np.random.default_rng(seed)
    geos = [
        g.model_copy(update={
            "beta_btc": float(rng.uniform(0.6, 1.6)),
            "sentiment_score": float(rng.uniform(-1.0, 1.0)),
            "sentiment_conf": float(rng.uniform(0.0, 1.0)),
        })
        for g in geometries
    ]
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geos, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    assert tw.dollar_residual_frac <= cfg.dollar_band + 1e-6


@pytest.mark.parametrize("seed", range(8))
def test_property_beta_residual_within_band(seed, geometries):
    rng = np.random.default_rng(seed + 100)
    geos = [
        g.model_copy(update={"beta_btc": float(rng.uniform(0.6, 1.6))})
        for g in geometries
    ]
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geos, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    assert abs(tw.beta_residual) <= cfg.beta_band + 1e-6


def test_property_deployment_floor_honored_on_balanced_book(geometries):
    # Spec §15 'deployment floor honored': a NORMAL balanced book must deploy >= floor on
    # BOTH sides AND <= (1 - dry_powder) on both sides, and be feasible. This is the direct
    # assertion the prior plan was missing (deploy ~0.766 < 0.90 made feasible always False).
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    assert tw.feasible is True
    assert tw.deploy_long_frac >= cfg.deployment_floor
    assert tw.deploy_short_frac >= cfg.deployment_floor
    # dry powder honored: never deploy beyond 1 - dry_powder_frac on either side
    assert tw.deploy_long_frac <= 1.0 - cfg.dry_powder_frac + 1e-6
    assert tw.deploy_short_frac <= 1.0 - cfg.dry_powder_frac + 1e-6


def test_property_gross_near_target_20k(geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    # gross including the hedge leg should land near the ~$20k target (within 20%). With the
    # scale-to-deploy-target step (Task 14 step 7) this now holds; do NOT loosen this bound.
    assert 0.8 * cfg.target_gross_usdt <= tw.gross_notional <= 1.2 * cfg.target_gross_usdt


def test_property_hrp_weighting_influences_per_name_notionals(geometries, returns_frame):
    # Spec §8: Ledoit-Wolf -> HRP must actually shape the book. Run the optimizer WITH a
    # returns frame (HRP active) vs WITHOUT (merged split), and assert the per-name long-side
    # notionals differ -> HRP is wired into optimize_book, not dead.
    cfg = NeutralityConfig()
    tw_plain = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                             prior_legs=None, cfg=cfg, returns=None)
    tw_hrp = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                           prior_legs=None, cfg=cfg, returns=returns_frame)

    def long_notional(tw, sym):
        for leg in tw.legs:
            if leg.symbol == sym and leg.sleeve != "hedge":
                return leg.target_notional
        return 0.0

    # SOL and BTC are the two long alpha names; HRP must redistribute between them.
    plain_sol = long_notional(tw_plain, "SOL/USDT:USDT")
    hrp_sol = long_notional(tw_hrp, "SOL/USDT:USDT")
    assert abs(hrp_sol - plain_sol) > 1.0  # HRP changed SOL's notional by > $1


def test_property_sentiment_never_flips_leg_direction(geometries):
    # Drown every leg in maximally adverse sentiment; directions must still match the
    # pre-sentiment sleeve intent.
    hostile = [
        g.model_copy(update={"sentiment_score": -1.0 if g.symbol in
                             ("SOL/USDT:USDT", "BTC/USDT:USDT") else 1.0,
                             "sentiment_conf": 1.0})
        for g in geometries
    ]
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), hostile, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    intent = {"SOL/USDT:USDT": "long", "XRP/USDT:USDT": "short",
              "BTC/USDT:USDT": "long", "ETH/USDT:USDT": "short"}
    for leg in tw.legs:
        if leg.sleeve == "hedge":
            continue
        # projection can shrink a leg to ~0 but must never flip its sign vs intent
        if abs(leg.weight) > 1e-6 and leg.symbol in intent:
            assert leg.direction == intent[leg.symbol]


def test_property_turnover_band_keeps_residuals_in_band_with_prior(geometries):
    # With a non-empty PRIOR book, the turnover/no-trade band runs BEFORE projection, so the
    # final projected book must STILL be dollar+beta neutral within band (the band can never
    # re-break neutrality, because projection has the last say). Spec §8/§9.
    cfg = NeutralityConfig()
    first = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                          prior_legs=None, cfg=cfg)
    # rebalance against the first book as prior
    second = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                           prior_legs=first.legs, cfg=cfg)
    assert second.dollar_residual_frac <= cfg.dollar_band + 1e-6
    assert abs(second.beta_residual) <= cfg.beta_band + 1e-6
    assert second.feasible is True


def test_property_new_sub_drift_band_leg_survives_rebalance(geometries):
    # A fresh name absent from the prior must NOT be snapped to 0 by the no-trade band, even
    # if its target magnitude is below drift_band. Build a prior WITHOUT SOL, then rebalance
    # a book that introduces SOL long; SOL must appear as a non-zero leg.
    now = datetime(2026, 6, 11, tzinfo=UTC)
    prior_sleeves = [SleeveSignal(
        sleeve="carry", risk_budget_frac=1.0, as_of_ts=now,
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )]
    cfg = NeutralityConfig()
    prior = optimize_book(prior_sleeves, geometries, equity=20000.0,
                          prior_legs=None, cfg=cfg)
    assert not any(leg.symbol == "SOL/USDT:USDT" for leg in prior.legs)
    # now rebalance with the balanced sleeves (introduces SOL long) against that prior
    after = optimize_book(_balanced_sleeves(now), geometries, equity=20000.0,
                          prior_legs=prior.legs, cfg=cfg)
    sol_legs = [leg for leg in after.legs if leg.symbol == "SOL/USDT:USDT"]
    assert sol_legs  # fresh sub-drift-band leg survived (always-trade)
    assert abs(sol_legs[0].target_notional) > 1.0


def test_property_no_silent_un_neutral_sets_feasible_flag(geometries):
    # A pathological single-name one-sided book cannot satisfy deployment floor on both
    # sides; the optimizer must flag feasible=False rather than report a fake-neutral book.
    s = SleeveSignal(
        sleeve="factor", risk_budget_frac=1.0, as_of_ts=NOW,
        tilts=[SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=1.0)],
    )
    cfg = NeutralityConfig()
    tw = optimize_book([s], geometries, equity=20000.0, prior_legs=None, cfg=cfg)
    if tw.deploy_short_frac < cfg.deployment_floor:
        assert tw.feasible is False
        assert tw.notes

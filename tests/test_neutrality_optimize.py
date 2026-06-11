from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt, TargetWeights
from futures_fund.neutrality import NeutralityConfig, optimize_book

NOW = datetime(2026, 6, 11, tzinfo=UTC)


def _broad_geometries():
    """A 6-name universe (3 long / 3 short capacity) with a BALANCED beta structure, so a
    fully-deployed dollar+beta-neutral book CAN respect the 25% per-name cap (unlike the
    adverse-beta 4-name canonical fixture)."""
    return [
        CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0, beta_btc=1.0, adv_usd=2e9),
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, beta_btc=1.1, adv_usd=1e9),
        CoinGeometry(symbol="SOL/USDT:USDT", mark=150.0, beta_btc=1.2, adv_usd=4e8),
        CoinGeometry(symbol="XRP/USDT:USDT", mark=0.6, beta_btc=1.0, adv_usd=3e8),
        CoinGeometry(symbol="ADA/USDT:USDT", mark=0.5, beta_btc=1.1, adv_usd=2e8),
        CoinGeometry(symbol="DOGE/USDT:USDT", mark=0.15, beta_btc=1.2, adv_usd=2e8),
    ]


def _broad_sleeves(now):
    """3 longs / 3 shorts so each side can spread its 90% gross across enough names to stay
    under the 25% per-name cap."""
    return [SleeveSignal(
        sleeve="factor", risk_budget_frac=1.0, as_of_ts=now,
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ADA/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
            SleeveTilt(symbol="DOGE/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )]


def test_optimize_book_returns_target_weights(sleeves, geometries):
    cfg = NeutralityConfig()
    tw = optimize_book(
        sleeves, geometries, equity=20000.0, prior_legs=None, cfg=cfg
    )
    assert isinstance(tw, TargetWeights)
    # The emitted book is ALWAYS dollar+beta neutral and fully deployed (the hard invariants).
    assert tw.dollar_residual_frac <= cfg.dollar_band + 1e-6
    assert abs(tw.beta_residual) <= cfg.beta_band + 1e-6
    # This canonical 4-name fixture has an adverse beta structure (longs avg beta 1.25 vs shorts
    # 1.0), so beta-neutralizing a 90%-deployed book forces a short leg past the 25% per-name cap.
    # No cap-respecting fully-deployed neutral book exists for so few names, so the optimizer
    # honestly reports feasible=False (it never silently breaches the cap). See
    # test_per_name_cap_respected_on_feasible_book for the breadth where the cap IS satisfiable.
    assert tw.feasible is False
    assert any("cap" in n for n in tw.notes)
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
    # BOTH sides AND <= (1 - dry_powder) on both sides. This is the direct assertion the prior
    # plan was missing (deploy ~0.766 < 0.90 made feasible always False). The emitted book is
    # ALWAYS neutral and deployed; the cap audit is a SEPARATE gate on feasible (below).
    cfg = NeutralityConfig()
    tw = optimize_book(_balanced_sleeves(NOW), geometries, equity=20000.0,
                       prior_legs=None, cfg=cfg)
    assert tw.deploy_long_frac >= cfg.deployment_floor - 1e-6
    assert tw.deploy_short_frac >= cfg.deployment_floor - 1e-6
    # dry powder honored: never deploy beyond 1 - dry_powder_frac on either side
    assert tw.deploy_long_frac <= 1.0 - cfg.dry_powder_frac + 1e-6
    assert tw.deploy_short_frac <= 1.0 - cfg.dry_powder_frac + 1e-6
    # neutrality always holds on the emitted book
    assert tw.dollar_residual_frac <= cfg.dollar_band + 1e-6
    assert abs(tw.beta_residual) <= cfg.beta_band + 1e-6
    # This adverse-beta 4-name fixture cannot also satisfy the 25% per-name cap (a short leg is
    # forced past it to beta-neutralize a 90%-deployed book), so the optimizer reports the cap
    # breach honestly via feasible=False rather than silently emitting an over-cap leg.
    assert tw.feasible is False
    assert any("cap" in n for n in tw.notes)


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
    # The turnover/no-trade band runs BEFORE projection, so the projected book stays neutral.
    assert second.dollar_residual_frac <= cfg.dollar_band + 1e-6
    assert abs(second.beta_residual) <= cfg.beta_band + 1e-6
    # feasible tracks the per-name cap, which this adverse-beta 4-name fixture cannot meet at
    # 90% deployment (see test_property_deployment_floor_honored_on_balanced_book); neutrality
    # and the turnover band are unaffected by that cap verdict.
    assert second.feasible is False


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


def test_per_name_cap_respected_on_feasible_book():
    # Spec §4/§8 (per-name cap is a HARD constraint): on a book broad enough to satisfy it, the
    # EMITTED alpha legs must each be <= per_name_cap and the optimizer must report feasible=True.
    # This is the regression guard the suite was missing: projection + the deploy scale used to
    # re-concentrate weight (a BTC leg at 0.36 vs cap 0.25) while feasible stayed True.
    cfg = NeutralityConfig()
    tw = optimize_book(_broad_sleeves(NOW), _broad_geometries(), equity=20000.0,
                       prior_legs=None, cfg=cfg)
    assert tw.feasible is True
    # neutral + deployed
    assert tw.dollar_residual_frac <= cfg.dollar_band + 1e-6
    assert abs(tw.beta_residual) <= cfg.beta_band + 1e-6
    assert tw.deploy_long_frac >= cfg.deployment_floor - 1e-6
    assert tw.deploy_short_frac >= cfg.deployment_floor - 1e-6
    # EVERY non-hedge leg respects the per-name cap on the FINAL book
    for leg in tw.legs:
        if leg.sleeve != "hedge":
            assert abs(leg.weight) <= cfg.per_name_cap + 1e-6, leg.symbol


def _cluster_geometries():
    """6 names, ALL beta == 1.0, so the BTC hedge stays ~0 and the per-side deployment falls
    entirely on the alpha legs (a non-zero hedge would otherwise soak up the long-side slack and
    keep the clustered pair off its cap). 3 long / 3 short capacity."""
    return [
        CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0, beta_btc=1.0, adv_usd=1e9),
        CoinGeometry(symbol="SOL/USDT:USDT", mark=150.0, beta_btc=1.0, adv_usd=1e9),
        CoinGeometry(symbol="ADA/USDT:USDT", mark=0.5, beta_btc=1.0, adv_usd=1e9),
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, beta_btc=1.0, adv_usd=1e9),
        CoinGeometry(symbol="XRP/USDT:USDT", mark=0.6, beta_btc=1.0, adv_usd=1e9),
        CoinGeometry(symbol="DOGE/USDT:USDT", mark=0.15, beta_btc=1.0, adv_usd=1e9),
    ]


def _cluster_returns(*, clustered: bool):
    """Per-symbol return frame for the cluster-cap test. BTC and SOL are the two longs we want the
    cluster cap to consolidate; when `clustered` they share a common driver (post-Ledoit-Wolf corr
    ~0.75 > corr_threshold), when not they are independent. ADA (the 3rd long) and DOGE (a short)
    are HIGH-vol so HRP de-weights them, concentrating each side onto the {BTC,SOL}/{ETH,XRP} pairs
    so the cluster cap actually has weight to bind on. The short side mirrors the long structure so
    the residual beta — and thus the BTC hedge — stays ~0 and does not absorb the long-side
    slack."""
    idx = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    rng = np.random.default_rng(11)
    v = 0.03
    long_driver = pd.Series(rng.normal(0.0, v, size=120), index=idx)
    short_driver = pd.Series(rng.normal(0.0, v, size=120), index=idx)

    def near(driver, k=0.999):
        return k * driver + (1 - k) * pd.Series(rng.normal(0.0, v, size=120), index=idx)

    def indep():
        return pd.Series(rng.normal(0.0, v, size=120), index=idx)

    def hi_vol():
        return pd.Series(rng.normal(0.0, 0.14, size=120), index=idx)

    return pd.DataFrame({
        "BTC/USDT:USDT": near(long_driver) if clustered else indep(),
        "SOL/USDT:USDT": near(long_driver) if clustered else indep(),
        "ADA/USDT:USDT": hi_vol(),                         # HRP-de-weighted long absorber
        "ETH/USDT:USDT": near(short_driver),
        "XRP/USDT:USDT": near(short_driver),
        "DOGE/USDT:USDT": hi_vol(),                        # HRP-de-weighted short absorber
    })


def _pair_mag(tw, a, b):
    def w(sym):
        return sum(leg.weight for leg in tw.legs if leg.symbol == sym and leg.sleeve != "hedge")
    return abs(w(a)) + abs(w(b))


def test_cluster_cap_consolidates_correlated_same_side_legs_via_returns():
    # Spec §3/§9 'correlated-as-one' cluster cap must be FUNCTIONAL inside optimize_book (the
    # solver path used to pass an empty corr map, making it inert). With 3 names per side the
    # HRP-de-weighted absorber (ADA) leaves the deploy slack on the {BTC,SOL} pair, so the cluster
    # cap is what holds the pair down. This is a real regression guard via TWO independent signals:
    # (1) effectively removing the cluster cap (raising it to 10.0) on the SAME correlated returns
    # lets the pair carry MORE -> the cap is binding, not inert; (2) dropping the correlation (the
    # corr-off counterfactual) lets the same pair carry MORE -> it is the correlation->cluster->cap
    # mechanism doing the work, and the unconstrained pair exceeds the cap, so a removed cap would
    # breach it.
    sleeves = _broad_sleeves(NOW)          # BTC/SOL/ADA long, ETH/XRP/DOGE short
    geos = _cluster_geometries()
    cfg = NeutralityConfig()

    tw = optimize_book(sleeves, geos, equity=20000.0, prior_legs=None, cfg=cfg,
                       returns=_cluster_returns(clustered=True))
    pair_clustered = _pair_mag(tw, "BTC/USDT:USDT", "SOL/USDT:USDT")
    # the correlated same-side cluster's combined weight is held at/under the cluster cap
    assert pair_clustered <= cfg.cluster_cap + 1e-6
    assert tw.feasible is True

    # (1) cap binds: with the cluster cap effectively OFF the SAME correlated pair carries more.
    cfg_no_cluster = NeutralityConfig(cluster_cap=10.0)
    tw_uncapped = optimize_book(sleeves, geos, equity=20000.0, prior_legs=None, cfg=cfg_no_cluster,
                                returns=_cluster_returns(clustered=True))
    pair_uncapped = _pair_mag(tw_uncapped, "BTC/USDT:USDT", "SOL/USDT:USDT")
    assert pair_uncapped > pair_clustered + 1e-3

    # (2) corr-off counterfactual: drop the BTC/SOL correlation (no cluster forms) and the pair
    # carries MORE, exceeding the cluster cap that constrained the correlated book.
    tw_corr_off = optimize_book(sleeves, geos, equity=20000.0, prior_legs=None, cfg=cfg,
                                returns=_cluster_returns(clustered=False))
    pair_corr_off = _pair_mag(tw_corr_off, "BTC/USDT:USDT", "SOL/USDT:USDT")
    assert pair_corr_off > pair_clustered + 1e-3
    # unclustered, the pair breaches the cap that the cluster held it under
    assert pair_corr_off > cfg.cluster_cap

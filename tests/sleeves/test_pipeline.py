from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from futures_fund.cointegration import build_pair, build_spread
from futures_fund.contracts import CoinGeometry, GeometryBundle, Pair, Spread
from futures_fund.neutrality import (
    NeutralityConfig,
    merge_sleeves,
    optimize_book,
    risk_parity_budgets,
)
from futures_fund.sleeves import (
    apply_conviction_tilts,
    carry_signal,
    factor_signal,
    pairs_signal,
    sentiment_factor_signal,
)

_NOW = datetime(2026, 6, 11, tzinfo=UTC)


def _universe() -> list[CoinGeometry]:
    rows = [
        # symbol, momentum, funding_apr, vol, sentiment, conf
        ("BTC/USDT:USDT", 0.20, -0.10, 0.30, 0.6, 0.9),
        ("ETH/USDT:USDT", 0.10, 0.05, 0.40, 0.2, 0.7),
        ("SOL/USDT:USDT", 0.30, 0.30, 0.60, 0.8, 0.8),
        ("XRP/USDT:USDT", -0.20, 0.20, 0.50, -0.7, 0.9),
        ("BNB/USDT:USDT", -0.10, -0.25, 0.35, -0.3, 0.6),
        ("ADA/USDT:USDT", -0.30, 0.10, 0.55, -0.9, 0.9),
    ]
    return [CoinGeometry(symbol=s, mark=100.0, momentum_20=m, funding_apr=f,
                         realized_vol=v, sentiment_score=sc, sentiment_conf=cf)
            for s, m, f, v, sc, cf in rows]


def _pair_and_spread() -> tuple[Pair, Spread]:
    pair = Pair(pair_id="BTCUSDT__ETHUSDT", symbol_y="BTC/USDT:USDT",
                symbol_x="ETH/USDT:USDT", hedge_ratio=2.0, method="engle_granger",
                adf_pvalue=0.01, adf_pvalue_adj=0.02, half_life=5.0, theta=0.139, mu=0.0,
                sigma_eq=10.0, formed_cycle=1)
    spread = Spread(pair_id=pair.pair_id, spread_value=25.0, zscore=2.5, state="short_spread")
    return pair, spread


def test_full_pipeline_produces_budgeted_neutral_ready_signals():
    geos = _universe()
    pair, spread = _pair_and_spread()
    sleeves = [
        carry_signal(geos, risk_budget_frac=0.0, now=_NOW),
        pairs_signal([pair], [spread], risk_budget_frac=0.0, now=_NOW),
        factor_signal(geos, risk_budget_frac=0.0, now=_NOW, factors=["momentum"], tercile=1 / 3),
        sentiment_factor_signal(geos, risk_budget_frac=0.0, now=_NOW, tercile=1 / 3),
    ]
    # all four sleeves emit at least one tilt for this universe
    assert all(s.tilts for s in sleeves)

    # risk-parity budgets sum to 1.0 across the four active sleeves
    budgets = risk_parity_budgets(sleeves)
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    assert set(budgets) == {"carry", "pairs", "factor", "sentiment"}

    # conviction tilts applied to the factor sleeve never flip direction
    tilted = apply_conviction_tilts(sleeves[2].tilts, geos, kappa=0.5, cap=0.25)
    for before, after in zip(sleeves[2].tilts, tilted, strict=True):
        if before.target_weight > 0:
            assert after.target_weight >= 0
        elif before.target_weight < 0:
            assert after.target_weight <= 0
        # cap respected: |delta| <= 25% of |w|
        assert abs(after.target_weight - before.target_weight) <= 0.25 * abs(
            before.target_weight) + 1e-9


def test_pairs_sleeve_legs_carry_pair_id_for_attribution():
    pair, spread = _pair_and_spread()
    sig = pairs_signal([pair], [spread], risk_budget_frac=0.0, now=_NOW)
    assert sig.tilts                                  # short_spread -> two legs
    assert all(t.pair_id == pair.pair_id for t in sig.tilts)
    assert pair.pair_id == "BTCUSDT__ETHUSDT"         # canonical slash-free id everywhere


def _liquid_bundle() -> GeometryBundle:
    """A 6-name liquid universe with a BALANCED beta structure (1.0-1.2) so a fully-deployed
    dollar+beta-neutral book CAN respect the 25% per-name cap, plus distinct momentum / funding /
    sentiment so all four real sleeves emit a non-trivial cross-section of tilts."""
    rows = [
        # symbol, mark, beta, momentum, funding_apr, vol, sentiment, conf
        ("BTC/USDT:USDT", 60000.0, 1.0, 0.20, -0.10, 0.30, 0.6, 0.9),
        ("ETH/USDT:USDT", 3000.0, 1.1, 0.10, 0.05, 0.40, 0.2, 0.7),
        ("SOL/USDT:USDT", 150.0, 1.2, 0.30, 0.30, 0.60, 0.8, 0.8),
        ("XRP/USDT:USDT", 0.6, 1.0, -0.20, 0.20, 0.50, -0.7, 0.9),
        ("ADA/USDT:USDT", 0.5, 1.1, -0.10, -0.25, 0.35, -0.3, 0.6),
        ("DOGE/USDT:USDT", 0.15, 1.2, -0.30, 0.10, 0.55, -0.9, 0.9),
    ]
    geos = [
        CoinGeometry(symbol=s, mark=mk, beta_btc=b, momentum_20=m, funding_apr=f,
                     realized_vol=v, sentiment_score=sc, sentiment_conf=cf, adv_usd=5e8)
        for s, mk, b, m, f, v, sc, cf in rows
    ]
    return GeometryBundle(geometries=geos, as_of_ts=_NOW)


def _real_short_spread_pair() -> tuple[Pair, Spread]:
    """A genuinely cointegrated BTC~ETH pair built by the real build_pair, plus a real Spread
    (built by build_spread) driven to z=+2.5 -> short_spread, so the pairs sleeve emits two
    signed hedge-ratio legs (short y, +h*x) rather than degenerating to no legs."""
    rng = np.random.default_rng(7)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 400)) + 100.0)
    y = 2.0 * x + pd.Series(rng.normal(0, 0.5, 400))           # cointegrated: y = 2x + stationary
    pair = build_pair(y, x, "BTC/USDT:USDT", "ETH/USDT:USDT", cycle=1)
    assert pair.cointegrated is True
    # marks chosen so spread_value = mu + 2.5*sigma_eq -> zscore == 2.5 -> short_spread
    mark_x = 100.0
    mark_y = (pair.mu + 2.5 * pair.sigma_eq) + pair.hedge_ratio * mark_x
    spread = build_spread(pair, mark_y=mark_y, mark_x=mark_x, prev_state="flat")
    assert spread.state == "short_spread"
    return pair, spread


def test_four_real_sleeves_through_optimize_book_is_neutral_and_deployed():
    # C1 (critical): the load-bearing seam. ALL FOUR real sleeve builders feed real Pair/Spread +
    # CoinGeometry output into risk_parity_budgets -> merge_sleeves -> optimize_book, and the
    # EMITTED TargetWeights must be a dollar+beta-neutral, >=90%-deployed, feasible book that also
    # preserves the pairs sleeve's signed hedge-ratio legs (pair_id stamped end-to-end).
    bundle = _liquid_bundle()
    geos = bundle.geometries
    pair, spread = _real_short_spread_pair()

    sleeves = [
        carry_signal(geos, risk_budget_frac=0.0, now=_NOW),
        pairs_signal([pair], [spread], risk_budget_frac=0.0, now=_NOW),
        factor_signal(geos, risk_budget_frac=0.0, now=_NOW, factors=["momentum"], tercile=1 / 3),
        sentiment_factor_signal(geos, risk_budget_frac=0.0, now=_NOW, tercile=1 / 3),
    ]
    # every real sleeve emits at least one tilt for this universe (non-vacuous seam)
    assert all(s.tilts for s in sleeves)
    assert {s.sleeve for s in sleeves} == {"carry", "pairs", "factor", "sentiment"}

    # budgets sum to 1.0 across the four active sleeves, and merge collapses to a signed vector
    budgets = risk_parity_budgets(sleeves)
    assert abs(sum(budgets.values()) - 1.0) < 1e-9
    merged = merge_sleeves(sleeves, geos)
    assert merged                                      # the pairs legs survive the merge collapse

    cfg = NeutralityConfig()
    tw = optimize_book(sleeves, geos, equity=20000.0, prior_legs=None, cfg=cfg)

    # the emitted book is dollar+beta neutral within the config bands ...
    assert tw.dollar_residual_frac <= cfg.dollar_band
    assert abs(tw.beta_residual) <= cfg.beta_band
    # ... fully deployed on BOTH sides (>= 90% deployment floor) ...
    assert tw.deploy_long_frac >= cfg.deployment_floor
    assert tw.deploy_short_frac >= cfg.deployment_floor
    # ... and feasible (cap-respecting neutral fully-deployed book exists for this universe).
    assert tw.feasible is True

    # the pairs sleeve's legs carry their pair_id end-to-end (pair-level PnL attribution survives
    # the merge_sleeves collapse to dict[str, float]); at least one pairs symbol is attributed.
    pair_legs = [leg for leg in tw.legs if leg.pair_id is not None]
    assert pair_legs
    assert all(leg.pair_id == pair.pair_id for leg in pair_legs)
    assert {leg.symbol for leg in pair_legs} <= {pair.symbol_y, pair.symbol_x}

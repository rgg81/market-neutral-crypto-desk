from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import CoinGeometry, Pair, Spread
from futures_fund.neutrality import risk_parity_budgets
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

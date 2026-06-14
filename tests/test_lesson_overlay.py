"""Stage C — lesson READ-BACK overlay (link 4: the loop actually closes).

`apply_lesson_overlay` tilts the next book's sleeve convictions by the corpus the miner maintains,
so a learned lesson measurably changes what the desk trades. The integration test pins the safety
contract: tilting the sleeves does NOT break the optimizer's dollar/beta neutrality — the optimizer
re-projects after, so a lesson can only re-shape relative conviction within the alpha legs.
"""
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import CoinGeometry, Lesson, SleeveSignal, SleeveTilt
from futures_fund.lesson_overlay import apply_lesson_overlay, lesson_tilt_factors
from futures_fund.neutrality import NeutralityConfig, optimize_book

NOW = datetime(2026, 6, 15, tzinfo=UTC)


def _lesson(polarity, tags, state="validated"):
    return Lesson(ts=NOW, text="t", tags=tags, polarity=polarity, state=state)


def _broad_geometries():
    return [
        CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0, beta_btc=1.0, adv_usd=2e9),
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, beta_btc=1.1, adv_usd=1e9),
        CoinGeometry(symbol="SOL/USDT:USDT", mark=150.0, beta_btc=1.2, adv_usd=4e8),
        CoinGeometry(symbol="XRP/USDT:USDT", mark=0.6, beta_btc=1.0, adv_usd=3e8),
        CoinGeometry(symbol="ADA/USDT:USDT", mark=0.5, beta_btc=1.1, adv_usd=2e8),
        CoinGeometry(symbol="DOGE/USDT:USDT", mark=0.15, beta_btc=1.2, adv_usd=2e8),
    ]


def _broad_sleeves():
    return [SleeveSignal(
        sleeve="factor", risk_budget_frac=1.0, as_of_ts=NOW,
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ADA/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
            SleeveTilt(symbol="DOGE/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )]


def test_restrictive_validated_lesson_down_weights_its_cohort_only():
    factors = lesson_tilt_factors([_lesson("restrictive", ["factor", "short"])])
    assert factors[("factor", "short")] == 0.9          # 1.0 - 0.10 (validated, full delta)
    assert ("factor", "long") not in factors            # other side untouched

    out = apply_lesson_overlay(_broad_sleeves(), [_lesson("restrictive", ["factor", "short"])])
    by_dir = {(t.symbol, t.direction): t.target_weight for t in out[0].tilts}
    assert by_dir[("ETH/USDT:USDT", "short")] == -0.5 * 0.9   # shorts scaled down
    assert by_dir[("SOL/USDT:USDT", "long")] == 0.5          # longs unchanged


def test_enabling_and_candidate_strengths_and_clamp():
    # enabling validated pushes UP by full delta; candidate by a fraction
    assert lesson_tilt_factors([_lesson("enabling", ["carry", "long"])])[("carry", "long")] == 1.1
    cand = lesson_tilt_factors([_lesson("enabling", ["carry", "long"], state="candidate")])
    assert abs(cand[("carry", "long")] - 1.03) < 1e-9        # 1 + 0.3*0.10
    # many stacked restrictive lessons can't push below the clamp floor (0.5)
    many = [_lesson("restrictive", ["factor", "short"]) for _ in range(20)]
    assert lesson_tilt_factors(many)[("factor", "short")] == 0.5


def test_empty_corpus_is_a_noop():
    sleeves = _broad_sleeves()
    assert apply_lesson_overlay(sleeves, []) is sleeves      # unchanged, same object


def test_overlay_preserves_optimizer_neutrality():
    """The safety contract: a lesson tilt re-shapes conviction, but the optimizer still emits a
    dollar+beta-neutral book (it re-projects after the overlay)."""
    cfg = NeutralityConfig()
    geoms = _broad_geometries()
    tilted = apply_lesson_overlay(_broad_sleeves(), [_lesson("restrictive", ["factor", "short"])])
    tw = optimize_book(tilted, geoms, equity=20000.0, prior_legs=None, cfg=cfg)
    assert tw.dollar_residual_frac <= cfg.dollar_band + 1e-6
    assert abs(tw.beta_residual) <= cfg.beta_band + 1e-6

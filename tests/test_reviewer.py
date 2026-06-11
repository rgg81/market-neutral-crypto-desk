from __future__ import annotations

from datetime import UTC, datetime

import pytest

from futures_fund.contracts import (
    CoinGeometry,
    Pair,
    Spread,
    TargetWeights,
    WeightLeg,
)
from futures_fund.models import MmrBracket, SymbolSpec, TradeProposal
from futures_fund.neutrality import NeutralityConfig
from futures_fund.reviewer import (
    check_beta_neutral,
    check_btc_hedge,
    check_caps,
    check_deployment_floor,
    check_dollar_neutral,
    check_exchange_filters,
    check_funding,
    check_pair_pnl,
    check_rr_after_costs,
    check_sharpe_annualization,
)

NOW = datetime(2026, 6, 11, tzinfo=UTC)


@pytest.fixture
def cfg() -> NeutralityConfig:
    """Contract-pinned bands: side_budget 10000, dollar_band 0.03, beta_band 0.05,
    deployment_floor 0.90, per_name_cap 0.25, cluster_cap 0.40."""
    return NeutralityConfig()


@pytest.fixture
def make_tw():
    """Build a `TargetWeights` from `(symbol, direction, magnitude_notional)` tuples.

    Unlike the shared conftest factory, this one lets the test STATE residual / deployment
    fields independently of the legs so the reviewer's re-derivation can be checked against a
    matched OR a TAMPERED claim. Each leg's `weight` is its signed share of total gross (sign
    from direction); `target_notional` is the positive magnitude the caller passes (the reviewer
    re-derives the signed leg notional from `direction`)."""

    def _make(
        legs: list[tuple[str, str, float]],
        *,
        betas: dict[str, float] | None = None,
        hedge_notional: float = 0.0,
        gross_long: float | None = None,
        gross_short: float | None = None,
        deploy_long_frac: float | None = None,
        deploy_short_frac: float | None = None,
        dollar_residual: float = 0.0,
        dollar_residual_frac: float = 0.0,
        beta_residual: float = 0.0,
    ) -> TargetWeights:
        betas = betas or {}
        gross = sum(abs(n) for _, _, n in legs) or 1.0
        weight_legs: list[WeightLeg] = []
        for symbol, direction, notional in legs:
            sign = 1.0 if direction == "long" else -1.0
            weight_legs.append(
                WeightLeg(
                    symbol=symbol,
                    direction=direction,
                    weight=sign * abs(notional) / gross,
                    target_notional=sign * abs(notional),
                    beta_btc=betas.get(symbol, 1.0),
                    sleeve="factor",
                )
            )
        gl = gross_long if gross_long is not None else sum(
            abs(n) for _, d, n in legs if d == "long"
        )
        gs = gross_short if gross_short is not None else sum(
            abs(n) for _, d, n in legs if d == "short"
        )
        return TargetWeights(
            legs=weight_legs,
            btc_hedge_notional=hedge_notional,
            dollar_residual=dollar_residual,
            dollar_residual_frac=dollar_residual_frac,
            beta_residual=beta_residual,
            gross_long=gl,
            gross_short=gs,
            deploy_long_frac=deploy_long_frac if deploy_long_frac is not None else 0.0,
            deploy_short_frac=deploy_short_frac if deploy_short_frac is not None else 0.0,
            gross_notional=gross,
            as_of_ts=NOW,
        )

    return _make


def _geoms(betas: dict[str, float]) -> list[CoinGeometry]:
    return [
        CoinGeometry(symbol=s, mark=100.0, beta_btc=b, adv_usd=1e9)
        for s, b in betas.items()
    ]


# --- name 1: dollar_residual_in_band -------------------------------------------------------

def test_dollar_neutral_recomputed_from_legs(make_tw, cfg):
    tw = make_tw([("BTC/USDT:USDT", "long", 6000.0), ("ETH/USDT:USDT", "short", 4000.0)])
    tw.dollar_residual_frac = 0.0  # tampered claim
    chk = check_dollar_neutral(tw, cfg)
    assert chk.name == "dollar_residual_in_band"
    assert chk.ok is False  # real residual = (6000-4000)/10000 = 0.20 > band


def test_dollar_neutral_matched_in_band(make_tw, cfg):
    # balanced legs => residual 0, well inside the 0.03 band
    tw = make_tw([("BTC/USDT:USDT", "long", 5000.0), ("ETH/USDT:USDT", "short", 5000.0)])
    chk = check_dollar_neutral(tw, cfg)
    assert chk.name == "dollar_residual_in_band"
    assert chk.ok is True


# --- name 2: beta_residual_in_band ---------------------------------------------------------

def test_beta_neutral_recomputed(make_tw, cfg):
    betas = {"BTC/USDT:USDT": 1.0, "ETH/USDT:USDT": 1.0}
    geoms = _geoms(betas)
    # both legs LONG beta-1.0 => net beta exposure = +(5000+5000)/10000 = 1.0 (way out of band),
    # but the artifact lies and claims beta_residual == 0.
    tw = make_tw(
        [("BTC/USDT:USDT", "long", 5000.0), ("ETH/USDT:USDT", "long", 5000.0)],
        betas=betas,
        beta_residual=0.0,
    )
    chk = check_beta_neutral(tw, geoms, cfg)
    assert chk.name == "beta_residual_in_band"
    assert chk.ok is False


def test_beta_neutral_matched_in_band(make_tw, cfg):
    betas = {"BTC/USDT:USDT": 1.0, "ETH/USDT:USDT": 1.0}
    geoms = _geoms(betas)
    # long beta-1.0 vs short beta-1.0, equal size => net beta exposure ~ 0
    tw = make_tw(
        [("BTC/USDT:USDT", "long", 5000.0), ("ETH/USDT:USDT", "short", 5000.0)],
        betas=betas,
    )
    chk = check_beta_neutral(tw, geoms, cfg)
    assert chk.name == "beta_residual_in_band"
    assert chk.ok is True


# --- name 3: btc_hedge_sizing --------------------------------------------------------------

def test_btc_hedge_sizing_recomputed(make_tw, cfg):
    # ALPHA legs (non-hedge): one long beta-1.5 (5000), one short beta-1.0 (5000). The residual
    # portfolio beta the hedge must absorb is re-derived by the reviewer; a tampered hedge
    # notional that does not match => ok=False.
    betas = {"ALT/USDT:USDT": 1.5, "ETH/USDT:USDT": 1.0}
    geoms = _geoms({**betas, "BTC/USDT:USDT": 1.0})
    tw = make_tw(
        [("ALT/USDT:USDT", "long", 5000.0), ("ETH/USDT:USDT", "short", 5000.0)],
        betas=betas,
        hedge_notional=99999.0,  # tampered: nowhere near the re-derived hedge size
    )
    chk = check_btc_hedge(tw, geoms, cfg)
    assert chk.name == "btc_hedge_sizing"
    assert chk.ok is False


def test_btc_hedge_sizing_matched(make_tw, cfg):
    # Build the matched case from the SAME re-derivation the reviewer uses (size_btc_hedge on the
    # alpha legs' residual beta).
    from futures_fund.neutrality import size_btc_hedge

    betas = {"ALT/USDT:USDT": 1.5, "ETH/USDT:USDT": 1.0}
    geoms = _geoms({**betas, "BTC/USDT:USDT": 1.0})
    equity = cfg.capital_usdt
    alpha_weights = {"ALT/USDT:USDT": 5000.0 / equity, "ETH/USDT:USDT": -5000.0 / equity}
    expected = size_btc_hedge(
        alpha_weights, betas, equity=equity, side_budget=cfg.side_budget_usdt
    )
    tw = make_tw(
        [("ALT/USDT:USDT", "long", 5000.0), ("ETH/USDT:USDT", "short", 5000.0)],
        betas=betas,
        hedge_notional=expected,
    )
    chk = check_btc_hedge(tw, geoms, cfg)
    assert chk.name == "btc_hedge_sizing"
    assert chk.ok is True


# --- name 4: deployment_floor_both_sides ---------------------------------------------------

def test_deployment_floor_both_sides(make_tw, cfg):
    # short side only deploys 0.50 of its budget => below the 0.90 floor => ok=False even though
    # the artifact claims both sides are at the floor.
    tw = make_tw(
        [("BTC/USDT:USDT", "long", 9000.0), ("ETH/USDT:USDT", "short", 5000.0)],
        deploy_long_frac=0.90,
        deploy_short_frac=0.90,  # tampered claim
    )
    chk = check_deployment_floor(tw, cfg)
    assert chk.name == "deployment_floor_both_sides"
    assert chk.ok is False


def test_deployment_floor_both_sides_ok(make_tw, cfg):
    # both sides deploy 0.90 of the 10000 budget => at the floor
    tw = make_tw(
        [("BTC/USDT:USDT", "long", 9000.0), ("ETH/USDT:USDT", "short", 9000.0)],
    )
    chk = check_deployment_floor(tw, cfg)
    assert chk.name == "deployment_floor_both_sides"
    assert chk.ok is True


# --- names 5 + 6: per_name_cap + cluster_cap ----------------------------------------------

def test_per_name_cap(make_tw, cfg):
    # one leg at |w| = 0.30 > per_name_cap 0.25 => per_name_cap check ok=False.
    # gross = 6000 + 2000 = 8000; long leg weight = 6000/... use equity-fraction convention.
    tw = make_tw(
        [
            ("BTC/USDT:USDT", "long", 7000.0),
            ("ETH/USDT:USDT", "short", 4000.0),
            ("SOL/USDT:USDT", "short", 3000.0),
        ],
    )
    checks = check_caps(tw, cfg)
    names = {c.name for c in checks}
    assert names == {"per_name_cap", "cluster_cap"}
    per_name = next(c for c in checks if c.name == "per_name_cap")
    # BTC notional 7000 / equity 20000 = 0.35 > 0.25 => breach
    assert per_name.ok is False


def test_per_name_cap_ok(make_tw, cfg):
    # every leg under the 0.25 equity-fraction cap (max 4000/20000 = 0.20)
    tw = make_tw(
        [
            ("BTC/USDT:USDT", "long", 4000.0),
            ("ETH/USDT:USDT", "long", 4000.0),
            ("SOL/USDT:USDT", "short", 4000.0),
            ("XRP/USDT:USDT", "short", 4000.0),
        ],
    )
    checks = check_caps(tw, cfg)
    per_name = next(c for c in checks if c.name == "per_name_cap")
    assert per_name.ok is True


def test_cluster_cap(make_tw, cfg):
    # Two same-side legs whose combined |w| exceeds cluster_cap 0.40 when treated as one cluster.
    # 5000 + 5000 = 10000 / equity 20000 = 0.50 > 0.40 => cluster breach. Provide a correlation
    # high enough to cluster them (default corr_threshold 0.7).
    corr = {("AAA/USDT:USDT", "BBB/USDT:USDT"): 0.95}
    tw = make_tw(
        [
            ("AAA/USDT:USDT", "long", 5000.0),
            ("BBB/USDT:USDT", "long", 5000.0),
            ("ETH/USDT:USDT", "short", 4000.0),
        ],
    )
    checks = check_caps(tw, cfg, corr=corr)
    cluster = next(c for c in checks if c.name == "cluster_cap")
    assert cluster.ok is False


def test_cluster_cap_ok(make_tw, cfg):
    # Uncorrelated same-side legs are not clustered, so neither exceeds the cluster cap.
    tw = make_tw(
        [
            ("AAA/USDT:USDT", "long", 4000.0),
            ("BBB/USDT:USDT", "long", 4000.0),
            ("ETH/USDT:USDT", "short", 4000.0),
            ("XRP/USDT:USDT", "short", 4000.0),
        ],
    )
    checks = check_caps(tw, cfg)  # empty corr => no clustering
    cluster = next(c for c in checks if c.name == "cluster_cap")
    assert cluster.ok is True


# === Task 5.3: canonical names 7-13 =========================================================


def _funding_geoms(specs: dict[str, dict]) -> list[CoinGeometry]:
    """CoinGeometry list carrying mark + signed funding_rate + interval for the funding check."""
    return [
        CoinGeometry(
            symbol=s,
            mark=d["mark"],
            funding_rate=d["funding_rate"],
            funding_interval_hours=d.get("interval", 8.0),
            beta_btc=1.0,
            adv_usd=1e9,
        )
        for s, d in specs.items()
    ]


# --- names 7 + 8: funding_sign + funding_amount -------------------------------------------

def test_funding_sign_short_credit(make_tw):
    # A SHORT leg with POSITIVE funding RECEIVES funding => the realized settlement is a CREDIT
    # (positive balance contribution). The reviewer re-derives the sign from realized_funding and
    # the funding_sign check must reflect that a short-on-positive-funding is a credit (ok=True),
    # while an artifact that flips the sign is caught.
    geoms = _funding_geoms({"ETH/USDT:USDT": {"mark": 100.0, "funding_rate": 0.001}})
    tw = make_tw([("ETH/USDT:USDT", "short", 5000.0)])
    checks = check_funding(tw, geoms)
    names = {c.name for c in checks}
    assert names == {"funding_sign", "funding_amount"}
    sign = next(c for c in checks if c.name == "funding_sign")
    # short + positive funding => realized credit (positive) => recomputed sign is positive
    assert sign.expected > 0
    assert sign.ok is True


def test_funding_amount_matches_realized(make_tw):
    # The funding_amount check re-derives the per-leg realized funding via
    # funding_intervals.realized_funding and totals it. A geometry whose funding_rate is internally
    # consistent matches; the amount equals the realized primitive over the leg's qty.
    from futures_fund.funding_intervals import realized_funding

    geoms = _funding_geoms({"ETH/USDT:USDT": {"mark": 100.0, "funding_rate": 0.001}})
    tw = make_tw([("ETH/USDT:USDT", "short", 5000.0)])
    checks = check_funding(tw, geoms)
    amt = next(c for c in checks if c.name == "funding_amount")
    qty = 5000.0 / 100.0
    expected = realized_funding(-5000.0, 100.0, qty, 0.001, "short")
    assert amt.expected == pytest.approx(expected)
    assert amt.ok is True


def test_funding_sign_caught_when_primitive_flips_sign(make_tw, monkeypatch):
    # ADVERSARIAL fail-path: the funding_sign check re-derives the EXPECTED sign from market physics
    # (short-on-positive-funding => credit) independently of realized_funding. If the audited
    # primitive flips its sign convention, the per-leg sign cross-check must catch it (ok=False) —
    # proving the check is falsifiable, not a self-equal pin.
    import futures_fund.reviewer as rev

    monkeypatch.setattr(
        rev, "realized_funding", lambda notional, mark, qty, rate, direction: -(
            (1.0 if direction == "long" else -1.0) * mark * qty * rate
        ) * -1.0  # deliberately wrong: flips the correct -side*mark*qty*rate sign
    )
    geoms = _funding_geoms({"ETH/USDT:USDT": {"mark": 100.0, "funding_rate": 0.001}})
    tw = make_tw([("ETH/USDT:USDT", "short", 5000.0)])
    checks = rev.check_funding(tw, geoms)
    sign = next(c for c in checks if c.name == "funding_sign")
    assert sign.ok is False


def test_funding_amount_caught_when_primitive_wrong_scale(make_tw, monkeypatch):
    # ADVERSARIAL fail-path: funding_amount compares the closed-form Σ -side·notional·rate
    # (expected) to Σ realized_funding (actual). A primitive that mis-scales (e.g. forgets a
    # factor) makes the totals diverge => ok=False.
    import futures_fund.reviewer as rev

    monkeypatch.setattr(
        rev, "realized_funding", lambda notional, mark, qty, rate, direction: (
            -(1.0 if direction == "long" else -1.0) * mark * qty * rate
        ) * 2.0  # correct sign but 2x scale => expected != actual
    )
    geoms = _funding_geoms({"ETH/USDT:USDT": {"mark": 100.0, "funding_rate": 0.001}})
    tw = make_tw([("ETH/USDT:USDT", "short", 5000.0)])
    checks = rev.check_funding(tw, geoms)
    amt = next(c for c in checks if c.name == "funding_amount")
    assert amt.ok is False


# --- names 9 + 10: pair_pnl_attribution + pair_leg_hedge_ratio ----------------------------

def _pair() -> Pair:
    return Pair(
        pair_id="AAAUSDT__BBBUSDT",
        symbol_y="AAA/USDT:USDT",
        symbol_x="BBB/USDT:USDT",
        hedge_ratio=2.0,
        method="engle_granger",
        adf_pvalue=0.01,
        half_life=5.0,
        theta=0.1,
        mu=0.0,
        sigma_eq=1.0,
        formed_cycle=1,
    )


def test_pair_pnl_at_spread_level(make_tw):  # noqa: ARG001
    # PnL is attributed at the SPREAD level: a long_spread (long y, short hedge_ratio*x) earns
    # qty_y*(spread_now - spread_entry). The reviewer re-derives the spread-level PnL from the
    # leg qtys and compares to the artifact's realized_pnl. A tampered realized_pnl is caught.
    pair = _pair()
    spread = Spread(
        pair_id=pair.pair_id,
        spread_value=3.0,
        zscore=-1.0,
        state="long_spread",
        qty_y=10.0,
        qty_x=20.0,           # = hedge_ratio(2.0) * qty_y(10.0) => sized by hedge ratio
        notional_y=1000.0,
        notional_x=2000.0,
        realized_pnl=999999.0,  # tampered claim
    )
    checks = check_pair_pnl([spread], [pair])
    names = {c.name for c in checks}
    assert names == {"pair_pnl_attribution", "pair_leg_hedge_ratio"}
    attribution = next(c for c in checks if c.name == "pair_pnl_attribution")
    assert attribution.ok is False  # 999999 nowhere near the re-derived spread PnL


def test_pair_legs_sized_by_hedge_ratio():
    # The hedge-ratio check verifies the x leg is sized at hedge_ratio * qty_y. A spread whose
    # qty_x is NOT hedge_ratio*qty_y leaves a residual beta in the pair => ok=False.
    pair = _pair()  # hedge_ratio 2.0
    spread = Spread(
        pair_id=pair.pair_id,
        spread_value=0.0,
        zscore=0.0,
        state="long_spread",
        qty_y=10.0,
        qty_x=15.0,            # should be 20.0 (2.0 * 10.0) => mis-hedged
        realized_pnl=0.0,
    )
    checks = check_pair_pnl([spread], [pair])
    hedge = next(c for c in checks if c.name == "pair_leg_hedge_ratio")
    assert hedge.ok is False


def test_pair_legs_sized_by_hedge_ratio_ok():
    pair = _pair()  # hedge_ratio 2.0
    spread = Spread(
        pair_id=pair.pair_id,
        spread_value=0.0,
        zscore=0.0,
        state="long_spread",
        qty_y=10.0,
        qty_x=20.0,            # exactly hedge_ratio * qty_y
        realized_pnl=0.0,
    )
    checks = check_pair_pnl([spread], [pair])
    hedge = next(c for c in checks if c.name == "pair_leg_hedge_ratio")
    assert hedge.ok is True


# --- name 11: rr_after_costs ---------------------------------------------------------------

def _proposal(*, entry: float, stop: float, tp: float) -> TradeProposal:
    return TradeProposal(
        symbol="ETH/USDT:USDT",
        direction="long",
        entry=entry,
        stop=stop,
        take_profits=[tp],
        atr=1.0,
        confidence=0.6,
        horizon_hours=24.0,
        funding_rate=0.0001,
    )


def test_rr_after_costs_ge_2():
    # RR re-derived via risk_gate._reward_risk must be >= 2.0. A proposal with reward = 2x risk
    # passes; one below the floor fails.
    ok_prop = _proposal(entry=100.0, stop=99.0, tp=102.0)   # reward 2, risk 1 => RR 2.0
    bad_prop = _proposal(entry=100.0, stop=99.0, tp=101.5)  # reward 1.5 => RR 1.5 < 2
    ok_chk = check_rr_after_costs([ok_prop])
    bad_chk = check_rr_after_costs([bad_prop])
    assert ok_chk.name == "rr_after_costs"
    assert ok_chk.ok is True
    assert bad_chk.ok is False


# --- name 12: sharpe_annualization ---------------------------------------------------------

def test_sharpe_daily_365_weekly_52():
    # The annualization factor must be 365 for daily and 52 for weekly cadence (NOT the inherited
    # 2190 4h factor). The reviewer re-derives the periods_per_year from the cadence.
    daily = check_sharpe_annualization("daily")
    weekly = check_sharpe_annualization("weekly")
    assert daily.name == "sharpe_annualization"
    assert daily.expected == pytest.approx(365.0)
    assert daily.actual == pytest.approx(365.0)
    assert daily.ok is True
    assert weekly.expected == pytest.approx(52.0)
    assert weekly.actual == pytest.approx(52.0)
    assert weekly.ok is True


def test_sharpe_annualization_caught_on_legacy_regression(monkeypatch):
    # ADVERSARIAL fail-path: the check compares the spec factor (expected) to the constant the
    # production metrics module actually exposes (actual). If the metrics module regresses back to
    # the inherited 2190 4h factor, the check must FAIL (ok=False) — proving expected-vs-actual is a
    # real comparison, not an always-true `!= 2190` tautology.
    import futures_fund.reviewer as rev

    monkeypatch.setattr(rev, "PERIODS_PER_YEAR_WEEKLY", 2190.0)
    chk = rev.check_sharpe_annualization("weekly")
    assert chk.actual == pytest.approx(2190.0)
    assert chk.ok is False


# --- name 13: exchange_filter_compliance ---------------------------------------------------

def _spec(
    symbol: str, *, min_notional: float, tick: float = 0.01, step: float = 0.001
) -> SymbolSpec:
    return SymbolSpec(
        symbol=symbol,
        tick_size=tick,
        step_size=step,
        min_notional=min_notional,
        mmr_brackets=[
            MmrBracket(
                notional_floor=0.0,
                notional_cap=1e9,
                mmr=0.005,
                maint_amount=0.0,
                max_leverage=20.0,
            )
        ],
    )


def _filter_geoms(specs: dict[str, tuple[float, SymbolSpec]]) -> list[CoinGeometry]:
    return [
        CoinGeometry(symbol=s, mark=mark, beta_btc=1.0, adv_usd=1e9, spec=spec)
        for s, (mark, spec) in specs.items()
    ]


def test_exchange_filter_min_notional(make_tw):
    # A leg whose notional is BELOW the exchange min_notional must be flagged non-compliant.
    geoms = _filter_geoms(
        {"ETH/USDT:USDT": (100.0, _spec("ETH/USDT:USDT", min_notional=1_000_000.0))}
    )
    tw = make_tw([("ETH/USDT:USDT", "long", 5000.0)])  # 5000 < 1,000,000 min
    checks = check_exchange_filters(tw, geoms)
    names = {c.name for c in checks}
    assert names == {"exchange_filter_compliance"}
    chk = checks[0]
    assert chk.ok is False


def test_exchange_filter_min_notional_ok(make_tw):
    geoms = _filter_geoms(
        {"ETH/USDT:USDT": (100.0, _spec("ETH/USDT:USDT", min_notional=10.0))}
    )
    tw = make_tw([("ETH/USDT:USDT", "long", 5000.0)])  # 5000 >= 10 min, qty/price on grid
    checks = check_exchange_filters(tw, geoms)
    chk = checks[0]
    assert chk.name == "exchange_filter_compliance"
    assert chk.ok is True


def test_exchange_filter_off_grid_mark_is_compliant(make_tw):
    # REGRESSION GUARD for the old over-strict logic: a realistic mark that IS on the tick grid
    # (100.07 on a 0.01 tick) divided into an arbitrary 5000 notional yields qty 49.965... which the
    # previous check wrongly flagged as 'off step_size'. After grid-rounding the executable order
    # (49.965 @ 100.07 => ~4999.998) clears min_notional, so the leg is COMPLIANT — the raw
    # notional/mark ratio landing off-grid is normal and must NOT halt a legitimate book.
    spec = _spec("ETH/USDT:USDT", min_notional=10.0, tick=0.01, step=0.001)
    geoms = _filter_geoms({"ETH/USDT:USDT": (100.07, spec)})
    tw = make_tw([("ETH/USDT:USDT", "long", 5000.0)])
    checks = check_exchange_filters(tw, geoms)
    assert checks[0].ok is True


def test_exchange_filter_rounds_to_dust(make_tw):
    # ADVERSARIAL fail-path for the grid logic: a leg whose qty floors to ZERO on a coarse step grid
    # cannot be submitted => non-compliant. notional 5 / mark 100 = 0.05 qty, floored to step 1.0
    # => 0 lots.
    geoms = _filter_geoms(
        {"ETH/USDT:USDT": (100.0, _spec("ETH/USDT:USDT", min_notional=1.0, tick=0.01, step=1.0))}
    )
    tw = make_tw([("ETH/USDT:USDT", "long", 5.0)])
    checks = check_exchange_filters(tw, geoms)
    assert checks[0].ok is False
    assert "rounds to 0" in checks[0].detail


def test_exchange_filter_below_min_after_rounding(make_tw):
    # ADVERSARIAL fail-path: a leg whose RAW notional clears min_notional but whose grid-FLOORED
    # qty drops the executable notional BELOW it => non-compliant. notional 150 / mark 100 = 1.5
    # qty; floored to step 1.0 => 1 lot => executable notional 100 < min_notional 120.
    geoms = _filter_geoms(
        {"ETH/USDT:USDT": (100.0, _spec("ETH/USDT:USDT", min_notional=120.0, tick=0.01, step=1.0))}
    )
    tw = make_tw([("ETH/USDT:USDT", "long", 150.0)])
    checks = check_exchange_filters(tw, geoms)
    assert checks[0].ok is False
    assert "executable notional" in checks[0].detail

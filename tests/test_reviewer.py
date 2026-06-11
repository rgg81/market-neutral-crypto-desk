from __future__ import annotations

from datetime import UTC, datetime

import pytest

from futures_fund.contracts import (
    CoinGeometry,
    TargetWeights,
    WeightLeg,
)
from futures_fund.neutrality import NeutralityConfig
from futures_fund.reviewer import (
    check_beta_neutral,
    check_btc_hedge,
    check_caps,
    check_deployment_floor,
    check_dollar_neutral,
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

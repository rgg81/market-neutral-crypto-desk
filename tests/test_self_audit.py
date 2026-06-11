"""The extended standing self-audit panel (Pillar 4) must PASS on a healthy market-neutral book —
every critical neutrality / funding-sign / pair-hedge / sentiment / crypto-only invariant holds.

These invariant NAMES are a deliberately distinct, overlapping vocabulary from the reviewer's
canonical ``ReviewerCheck.name``s (roadmap Task 5.5 note) — ``self_audit`` is the standing,
pure-import invariant panel; the reviewer is the per-cycle artifact re-derivation. The two are
independent guards on overlapping properties and must NOT be aligned by renaming.
"""
from futures_fund.self_audit import (
    invariant_both_sides_deployment_floor,
    invariant_funding_sign_correct,
    invariant_no_tokenized_stock_leg,
    invariant_pair_legs_hedge_ratio_sized,
    invariant_residuals_in_band,
    invariant_sentiment_within_cap_range,
    run_self_audit,
)

# The seven named invariants Task 5.5 requires the panel to carry.
_REQUIRED = {
    "dollar_residual_in_band",
    "beta_residual_in_band",
    "both_sides_deployment_floor",
    "funding_sign_correct",
    "pair_legs_hedge_ratio_sized",
    "sentiment_within_cap_range",
    "no_tokenized_stock_leg",
}


def test_self_audit_carries_the_seven_named_invariants():
    names = {c["name"] for c in run_self_audit()["checks"]}
    missing = _REQUIRED - names
    assert not missing, f"self-audit missing invariants: {missing}"


def test_self_audit_all_invariants_pass_on_conformant_book():
    res = run_self_audit()
    failed = [c["name"] for c in res["checks"] if not c["ok"]]
    assert res["ok"], f"self-audit FAILED on a conformant book: {failed} -> {res['checks']}"


# --- per-invariant: ok=True on a conformant synthetic book, False on a deliberately broken one ---


def test_residuals_in_band_conformant_vs_broken():
    # dollar-neutral + beta-neutral: longs == shorts, Sum(w*beta) ~ 0.
    good_notionals = {"A": 1000.0, "B": -1000.0}
    good_weights = {"A": 0.05, "B": -0.05}
    betas = {"A": 1.0, "B": 1.0}
    assert invariant_residuals_in_band(good_weights, good_notionals, betas)[0]
    # broken: a long with no offsetting short -> dollar residual blows past the band.
    bad_notionals = {"A": 1000.0, "B": -100.0}
    assert not invariant_residuals_in_band(good_weights, bad_notionals, betas)[0]


def test_both_sides_deployment_floor_conformant_vs_broken():
    floor = 0.90
    budget = 10000.0
    good = {"long": 9500.0, "short": 9500.0}
    assert invariant_both_sides_deployment_floor(good, budget, floor)
    bad = {"long": 9500.0, "short": 4000.0}  # short side under-deployed
    assert not invariant_both_sides_deployment_floor(bad, budget, floor)


def test_funding_sign_correct_conformant_vs_broken():
    # A short with a POSITIVE rate must RECEIVE funding (positive balance contribution).
    assert invariant_funding_sign_correct()
    # A deliberately-flipped sign convention must be caught.
    assert not invariant_funding_sign_correct(flip=True)


def test_pair_legs_hedge_ratio_sized_conformant_vs_broken():
    assert invariant_pair_legs_hedge_ratio_sized(hedge_ratio=0.8, qty_y=10.0, qty_x=8.0)
    assert not invariant_pair_legs_hedge_ratio_sized(hedge_ratio=0.8, qty_y=10.0, qty_x=5.0)


def test_sentiment_within_cap_range_conformant_vs_broken():
    # conviction_tilt clamps |dw| <= cap*|w| and never flips sign.
    assert invariant_sentiment_within_cap_range(weight=0.10, score=1.0, conf=1.0, cap=0.25)
    # a "broken" check passes an oversized claimed delta that breaches the cap.
    assert not invariant_sentiment_within_cap_range(
        weight=0.10, score=1.0, conf=1.0, cap=0.25, claimed_delta=0.05
    )


def test_no_tokenized_stock_leg_conformant_vs_broken():
    crypto = [{"info": {"underlyingType": "COIN"}}]
    assert invariant_no_tokenized_stock_leg(crypto)
    tradfi = [{"info": {"underlyingType": "EQUITY"}}]  # a tokenized-stock wrapper perp
    assert not invariant_no_tokenized_stock_leg(tradfi)

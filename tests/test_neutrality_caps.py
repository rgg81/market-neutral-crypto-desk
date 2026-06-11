from __future__ import annotations

from futures_fund.neutrality import apply_cluster_cap, apply_per_name_cap


def test_per_name_cap_clamps_magnitude_preserving_sign():
    weights = {"A": 0.6, "B": -0.5, "C": 0.1}
    capped = apply_per_name_cap(weights, per_name_cap=0.25)
    assert capped["A"] == 0.25
    assert capped["B"] == -0.25
    assert capped["C"] == 0.1


def test_per_name_cap_noop_when_within_cap():
    weights = {"A": 0.2, "B": -0.1}
    capped = apply_per_name_cap(weights, per_name_cap=0.25)
    assert capped == weights


def test_cluster_cap_scales_correlated_same_side_group():
    # A and B are correlated >= 0.7 and same side (both long) => clustered together.
    weights = {"A": 0.3, "B": 0.3, "C": -0.4}
    corr = {("A", "B"): 0.9}
    capped = apply_cluster_cap(weights, corr=corr, cluster_cap=0.40, threshold=0.7)
    # cluster {A,B} combined long weight 0.6 > 0.40 => scale by 0.40/0.60
    assert abs(capped["A"] - 0.2) < 1e-9
    assert abs(capped["B"] - 0.2) < 1e-9
    # C is in its own cluster, magnitude 0.4 <= cap => unchanged
    assert abs(capped["C"] - (-0.4)) < 1e-9


def test_cluster_cap_does_not_cluster_opposite_sides():
    # A long, B short, even if correlated => not clustered (natural hedge).
    weights = {"A": 0.3, "B": -0.3}
    corr = {("A", "B"): 0.95}
    capped = apply_cluster_cap(weights, corr=corr, cluster_cap=0.40, threshold=0.7)
    assert capped == weights

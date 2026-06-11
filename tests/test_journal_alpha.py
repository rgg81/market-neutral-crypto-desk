from __future__ import annotations

import pytest
from pydantic import ValidationError

from futures_fund.journal import (
    AlphaOutcome,
    alpha_outcome,
    append_decision,
    patch_outcome,
    read_all_decisions,
)


def _load_one(memory_dir, cycle: int, symbol: str, direction: str) -> dict:
    """Load the single decision keyed by (cycle, symbol, direction)."""
    matches = [
        r
        for r in read_all_decisions(memory_dir)
        if r.get("cycle") == cycle
        and r.get("symbol") == symbol
        and r.get("direction") == direction
    ]
    assert len(matches) == 1, f"expected exactly one decision, found {len(matches)}"
    return matches[0]


def _payload() -> dict:
    return {
        "entry": 60000.0,
        "stop": 58000.0,
        "rationale": "carry + cointegration thesis",
    }


def _outcome() -> dict:
    return {
        "alpha_return": 0.012,
        "beta_contribution": -0.003,
        "pair_cointegrated_at_exit": True,
        "funding_thesis_matched": True,
        "neutrality_in_band": True,
        "sentiment_helped": False,
    }


def test_alpha_outcome_typed_accessor(tmp_path):
    append_decision(
        tmp_path, cycle=1, symbol="BTC/USDT:USDT", direction="long", payload=_payload()
    )
    patch_outcome(
        tmp_path,
        cycle=1,
        symbol="BTC/USDT:USDT",
        direction="long",
        outcome=_outcome(),
    )
    ao = alpha_outcome(_load_one(tmp_path, 1, "BTC/USDT:USDT", "long"))
    assert isinstance(ao, AlphaOutcome)
    assert ao.alpha_return == 0.012 and ao.beta_contribution == -0.003
    assert ao.pair_cointegrated_at_exit is True
    assert ao.funding_thesis_matched is True
    assert ao.neutrality_in_band is True
    assert ao.sentiment_helped is False


def test_alpha_outcome_raises_when_field_absent(tmp_path):
    # decision exists but its outcome was never patched -> alpha fields absent
    append_decision(
        tmp_path, cycle=2, symbol="ETH/USDT:USDT", direction="short", payload=_payload()
    )
    dec = _load_one(tmp_path, 2, "ETH/USDT:USDT", "short")
    with pytest.raises((KeyError, ValidationError)):
        alpha_outcome(dec)


def test_alpha_outcome_raises_on_partial_outcome(tmp_path):
    # patch only some of the six alpha fields -> still raises (missing the rest)
    append_decision(
        tmp_path, cycle=3, symbol="SOL/USDT:USDT", direction="long", payload=_payload()
    )
    patch_outcome(
        tmp_path,
        cycle=3,
        symbol="SOL/USDT:USDT",
        direction="long",
        outcome={"alpha_return": 0.01, "beta_contribution": 0.0},
    )
    dec = _load_one(tmp_path, 3, "SOL/USDT:USDT", "long")
    with pytest.raises((KeyError, ValidationError)):
        alpha_outcome(dec)


def test_append_decision_idempotent_per_key(tmp_path):
    id1 = append_decision(
        tmp_path, cycle=4, symbol="XRP/USDT:USDT", direction="long", payload=_payload()
    )
    id2 = append_decision(
        tmp_path,
        cycle=4,
        symbol="XRP/USDT:USDT",
        direction="long",
        payload={"entry": 0.6, "stop": 0.55},
    )
    assert id1 == id2  # same (cycle, symbol, direction) -> reuse, no duplicate
    recs = [
        r
        for r in read_all_decisions(tmp_path)
        if r.get("cycle") == 4 and r.get("symbol") == "XRP/USDT:USDT"
    ]
    assert len(recs) == 1

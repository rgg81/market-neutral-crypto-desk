import json
import pathlib

import pytest
from pydantic import ValidationError

from futures_fund.contracts import (
    AgentProposal,
    AnalystReport,
    SentimentBatch,
    TraderOutput,
    WatcherOutput,
)
from futures_fund.sentiment_ingest import level_to_s

FIX = pathlib.Path("tests/fixtures/agent_examples")


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def test_universe_scout_fixture_validates_watcheroutput():
    out = WatcherOutput.model_validate(_load("universe_scout.json"))
    assert out.candidates, "scout must surface at least one candidate"
    # Two-sided shortlist: both long and short leans present (market-neutral desk).
    leans = {c.lean for c in out.candidates}
    assert "long" in leans and "short" in leans
    assert all(0.0 <= c.score <= 1.0 for c in out.candidates)


# Each analyst fixture is `{"reports": [AnalystReport, ...]}` (a LIST of reports).
@pytest.mark.parametrize(
    "name",
    [
        "funding_carry.json",
        "pair_analyst.json",
        "factor_analyst.json",
        "technical.json",
        "derivatives.json",
    ],
)
def test_analyst_fixtures_validate_list_of_analystreport(name):
    data = _load(name)
    reports = [AnalystReport.model_validate(r) for r in data["reports"]]
    assert reports
    assert all(r.stance in {"bullish", "bearish", "neutral"} for r in reports)
    assert all(0.0 <= r.conviction <= 1.0 for r in reports)


def test_funding_carry_signals_carry_signed_funding_and_interval():
    reports = [AnalystReport.model_validate(r) for r in _load("funding_carry.json")["reports"]]
    r0 = reports[0]
    assert "signed_funding" in r0.signals
    assert "funding_interval_h" in r0.signals


def test_pair_analyst_signals_carry_hedge_ratio_and_adf_pvalue():
    reports = [AnalystReport.model_validate(r) for r in _load("pair_analyst.json")["reports"]]
    r0 = reports[0]
    assert "hedge_ratio" in r0.signals
    assert "adf_pvalue" in r0.signals
    assert 0.0 <= r0.signals["adf_pvalue"] <= 1.0


def test_sentiment_fixture_validates_sentimentbatch_with_pit_and_market_row():
    data = _load("sentiment.json")
    # `level` maps to `s` via the §7.1 ordinal mapping (level_to_s); inject before validating so
    # the worked fixture stays verbatim and the load-bearing level->s mapping is exercised.
    for r in data["reports"]:
        r["s"] = level_to_s(r["level"])
    batch = SentimentBatch.model_validate(data)
    # A "MARKET" row (market-wide read) MUST be present (spec §7.1).
    assert any(rep.symbol == "MARKET" for rep in batch.reports)
    # Point-in-time: every source.published_ts strictly precedes its report.as_of_ts.
    for rep in batch.reports:
        assert rep.level in {
            "very_positive",
            "positive",
            "neutral",
            "negative",
            "very_negative",
        }
        for src in rep.sources:
            assert src.published_ts < rep.as_of_ts, (
                f"{rep.symbol}: source {src.url} not point-in-time"
            )


def test_trader_fixture_validates_traderoutput():
    data = json.loads(
        pathlib.Path("tests/fixtures/agent_examples/trader.json").read_text()
    )
    out = TraderOutput.model_validate(data)
    assert len(out.proposals) == 2
    assert out.management == []  # stand-down contract: explicit empty list


def test_trader_bad_fixture_rejected_by_stop_side_invariant():
    # The happy-path fixture only exercises valid stop/TP sides; this negative fixture
    # asserts the AgentProposal stop/TP-side invariant actually fires.
    data = json.loads(
        pathlib.Path("tests/fixtures/agent_examples/trader_bad.json").read_text()
    )
    with pytest.raises(ValidationError, match="long stop must be below entry"):
        TraderOutput.model_validate(data)


@pytest.mark.parametrize(
    ("direction", "stop", "take_profit", "match"),
    [
        ("long", 69000.0, 73200.0, "long stop must be below entry"),
        ("long", 66100.0, 68000.0, "long take_profit must be above entry"),
        ("short", 3400.0, 3300.0, "short stop must be above entry"),
        ("short", 3720.0, 3600.0, "short take_profit must be below entry"),
    ],
)
def test_agent_proposal_stop_tp_side_invariant(direction, stop, take_profit, match):
    entry = 68500.0 if direction == "long" else 3580.0
    with pytest.raises(ValidationError, match=match):
        AgentProposal(
            symbol="BTC/USDT:USDT",
            direction=direction,
            entry=entry,
            stop=stop,
            take_profit=take_profit,
        )

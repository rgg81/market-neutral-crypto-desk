import json
import pathlib

import pytest
from pydantic import ValidationError

from futures_fund.contracts import AgentProposal, TraderOutput


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

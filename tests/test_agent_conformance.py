import json
import pathlib

from futures_fund.contracts import TraderOutput


def test_trader_fixture_validates_traderoutput():
    data = json.loads(
        pathlib.Path("tests/fixtures/agent_examples/trader.json").read_text()
    )
    out = TraderOutput.model_validate(data)
    assert len(out.proposals) == 2
    assert out.management == []  # stand-down contract: explicit empty list

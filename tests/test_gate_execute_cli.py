"""Task 4.5 — execute-boundary CLI (`scripts/gate_execute_cli.py`).

NET-NEW in this repo: the execute boundary W10/D7 invoke. The CLI loads `proposals.json`
(+ management/triggers/cancel_triggers) from the SAME cadence-segmented cycle root the due-gate
scans (`state/<cadence>/cycle/<N>/`, CADENCE-ROOT INVARIANT), dispatches to `gate_execute_step`
with `loop=cadence`, and persists `report.json` under that same cadence root.

The real `gate_execute_step` (with the reviewer precondition) is wired in P5 Task 5.4; here it is
monkeypatched at the module boundary so the test pins ONLY the CLI's load/dispatch/persist
contract: `--cadence` is required and is threaded through to `gate_execute_step(..., loop=cadence)`.
"""
from __future__ import annotations

import json

import pytest

from futures_fund.cycle_io import save_output


@pytest.fixture
def seeded_proposals(tmp_path):
    """Seed `state/weekly/cycle/1/proposals.json` (the Trader hand-off) under the cadence root the
    due-gate scans, so the CLI has a real artifact to load. Returns the state dir."""
    state = tmp_path / "state"
    save_output(
        state,
        1,
        "proposals",
        {
            "proposals": [
                {
                    "symbol": "BTC/USDT:USDT",
                    "direction": "long",
                    "entry": 68500.0,
                    "stop": 66100.0,
                    "take_profit": 73200.0,
                    "rationale": "carry+factor long leg",
                    "trigger_type": "market",
                },
                {
                    "symbol": "ETH/USDT:USDT",
                    "direction": "short",
                    "entry": 3580.0,
                    "stop": 3720.0,
                    "take_profit": 3300.0,
                    "rationale": "relative-value short vs BTC",
                    "trigger_type": "market",
                },
            ],
            "management": [],
            "triggers": [],
            "cancel_triggers": [],
        },
        cadence="weekly",
    )
    return state


@pytest.fixture
def passing_reviewer(tmp_path):
    """Seed a PASSING `reviewer.json` under the same cadence cycle root so the mandatory reviewer
    precondition (wired in Task 5.4) is satisfied and the CLI proceeds to gate+execute."""
    save_output(
        tmp_path / "state",
        1,
        "reviewer",
        {"passed": True, "checks": [], "mismatches": [], "cycle": 1, "cadence": "weekly",
         "reviewed_at": "2026-06-11T00:00:00+00:00"},
        cadence="weekly",
    )
    return tmp_path / "state"


def test_gate_execute_cli_dispatches_cadence(
    tmp_path, monkeypatch, seeded_proposals, passing_reviewer
):
    seen = {}

    def fake_step(ex, settings, sd, md, now, cyc, props, **kw):
        seen["loop"] = kw.get("loop")
        return {"executed": [], "dropped": []}

    monkeypatch.setattr("scripts.gate_execute_cli.gate_execute_step", fake_step)
    monkeypatch.chdir(tmp_path)
    from scripts.gate_execute_cli import main

    main(["--cadence", "weekly", "--cycle", "1"])
    assert seen["loop"] == "weekly"


def test_gate_execute_cli_requires_cadence(tmp_path, monkeypatch, seeded_proposals):
    monkeypatch.setattr(
        "scripts.gate_execute_cli.gate_execute_step",
        lambda *a, **k: {"executed": [], "dropped": []},
    )
    monkeypatch.chdir(tmp_path)
    from scripts.gate_execute_cli import main

    # argparse exits(2) when a required option is missing.
    with pytest.raises(SystemExit):
        main(["--cycle", "1"])


def test_gate_execute_cli_loads_proposals_and_writes_report(
    tmp_path, monkeypatch, seeded_proposals, passing_reviewer
):
    seen = {}

    def fake_step(ex, settings, sd, md, now, cyc, props, **kw):
        seen["proposals"] = props
        seen["cycle"] = cyc
        return {"executed": [{"symbol": "BTC/USDT:USDT"}], "dropped": []}

    monkeypatch.setattr("scripts.gate_execute_cli.gate_execute_step", fake_step)
    # Avoid building a real ccxt exchange in the test.
    monkeypatch.setattr(
        "scripts.gate_execute_cli.FuturesExchange.from_settings", lambda settings: object()
    )
    monkeypatch.chdir(tmp_path)
    from scripts.gate_execute_cli import main

    main(["--cadence", "weekly", "--cycle", "1"])

    # The Trader's two-leg proposal list reached the step verbatim.
    assert [p["symbol"] for p in seen["proposals"]] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert seen["cycle"] == 1

    # report.json persisted under the SAME cadence root the due-gate scans.
    report_path = tmp_path / "state" / "weekly" / "cycle" / "1" / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["executed"] == [{"symbol": "BTC/USDT:USDT"}]


def test_execute_halts_without_reviewer_verdict(tmp_path, monkeypatch, seeded_proposals):
    # MANDATORY non-skippable reviewer stage (§10/§12): with no reviewer.json under the cadence
    # cycle root, the deterministic gate flag is absent => the execute CLI HALTs (SystemExit(2))
    # BEFORE any fill, even though proposals.json is present.
    monkeypatch.setattr(
        "scripts.gate_execute_cli.gate_execute_step",
        lambda *a, **k: {"executed": [], "dropped": []},
    )
    monkeypatch.setattr(
        "scripts.gate_execute_cli.FuturesExchange.from_settings", lambda settings: object()
    )
    monkeypatch.chdir(tmp_path)  # no reviewer.json written
    from scripts.gate_execute_cli import main

    with pytest.raises(SystemExit) as e:
        main(["--cadence", "weekly", "--cycle", "1"])
    assert e.value.code == 2


def test_execute_proceeds_with_passing_reviewer_verdict(
    tmp_path, monkeypatch, seeded_proposals
):
    # With a PASSING reviewer.json under the cadence cycle root, the gate flag is true => the
    # execute path runs (no HALT) and persists report.json.
    save_output(
        tmp_path / "state",
        1,
        "reviewer",
        {"passed": True, "checks": [], "mismatches": [], "cycle": 1, "cadence": "weekly",
         "reviewed_at": "2026-06-11T00:00:00+00:00"},
        cadence="weekly",
    )
    monkeypatch.setattr(
        "scripts.gate_execute_cli.gate_execute_step",
        lambda *a, **k: {"executed": [{"symbol": "BTC/USDT:USDT"}], "dropped": []},
    )
    monkeypatch.setattr(
        "scripts.gate_execute_cli.FuturesExchange.from_settings", lambda settings: object()
    )
    monkeypatch.chdir(tmp_path)
    from scripts.gate_execute_cli import main

    main(["--cadence", "weekly", "--cycle", "1"])
    report_path = tmp_path / "state" / "weekly" / "cycle" / "1" / "report.json"
    assert report_path.exists()

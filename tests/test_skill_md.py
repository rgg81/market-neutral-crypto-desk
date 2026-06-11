"""Task 4.4: SKILL.md weekly/daily orchestration.

`SKILL.md` is the orchestrator's playbook: it must carry YAML frontmatter (`name`,
`description`), acquire a run-lock, due-check per cadence, then walk the WEEKLY Selection
ladder (W1-W12) and the DAILY Rebalance ladder (D1-D8). The MANDATORY, non-skippable
reviewer gate (`reviewer_cli.py`) must appear BEFORE any execute step
(`gate_execute_cli.py`) in BOTH ladders. Every CLI named must be one created by a task per
the provenance table (no inventing CLIs).
"""

from __future__ import annotations

from pathlib import Path

import yaml

SKILL_PATH = Path("SKILL.md")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a `---`-fenced YAML frontmatter block from the markdown body."""
    assert text.startswith("---"), "SKILL.md must open with a `---` YAML frontmatter fence"
    parts = text.split("---", 2)
    assert len(parts) == 3, "SKILL.md must have a closing `---` frontmatter fence"
    front = yaml.safe_load(parts[1])
    assert isinstance(front, dict), "SKILL.md frontmatter must parse to a mapping"
    return front, parts[2]


def test_skill_md_exists() -> None:
    assert SKILL_PATH.exists(), "missing SKILL.md orchestration playbook"


def test_skill_md_has_yaml_frontmatter() -> None:
    front, _ = _split_frontmatter(SKILL_PATH.read_text())
    assert front.get("name"), "frontmatter must carry a non-empty `name`"
    assert front.get("description"), "frontmatter must carry a non-empty `description`"


def test_skill_md_has_runlock_step() -> None:
    body = SKILL_PATH.read_text()
    # The run-lock CLI is acquired (weekly + daily) and released (always).
    assert "runlock_cli.py acquire" in body, "missing run-lock acquire step"
    assert "runlock_cli.py release" in body, "missing run-lock release step"
    assert "--owner weekly" in body and "--owner daily" in body, (
        "run-lock must be owned per cadence (weekly + daily)"
    )


def test_skill_md_has_due_check_per_cadence() -> None:
    body = SKILL_PATH.read_text()
    assert "due_check.py state --loop weekly" in body, "missing weekly due-check step"
    assert "due_check.py state --loop daily" in body, "missing daily due-check step"


def test_skill_md_has_weekly_phase_ladder() -> None:
    body = SKILL_PATH.read_text()
    # Every weekly phase marker W1..W12 must be present as a bolded `**W<n> —` header, in order.
    # The bolded-header form makes the marker unambiguous (so `W1` never matches inside `W12`).
    last = -1
    for n in range(1, 13):
        marker = f"**W{n} —"
        idx = body.find(marker)
        assert idx != -1, f"missing weekly phase marker {marker!r}"
        assert idx > last, f"weekly phase W{n} out of order"
        last = idx


def test_skill_md_has_daily_phase_ladder() -> None:
    body = SKILL_PATH.read_text()
    last = -1
    for n in range(1, 9):
        marker = f"**D{n} —"
        idx = body.find(marker)
        assert idx != -1, f"missing daily phase marker {marker!r}"
        assert idx > last, f"daily phase D{n} out of order"
        last = idx


def test_skill_md_reviewer_gate_before_execute_weekly() -> None:
    """The reviewer gate (W9) is a non-skippable stage BEFORE the execute step (W10)."""
    body = SKILL_PATH.read_text()
    weekly = body[body.find("**W1 —") : body.find("**D1 —")]
    reviewer = weekly.find("reviewer_cli.py")
    execute = weekly.find("gate_execute_cli.py")
    assert reviewer != -1, "weekly ladder missing the mandatory reviewer_cli.py gate"
    assert execute != -1, "weekly ladder missing the gate_execute_cli.py execute step"
    assert reviewer < execute, "weekly reviewer gate MUST come before the execute step"
    assert "MANDATORY" in weekly, "weekly reviewer gate must be marked MANDATORY"


def test_skill_md_reviewer_gate_before_execute_daily() -> None:
    """The reviewer gate (D6) is a non-skippable stage BEFORE the execute step (D7)."""
    body = SKILL_PATH.read_text()
    daily = body[body.find("**D1 —") :]
    reviewer = daily.find("reviewer_cli.py")
    execute = daily.find("gate_execute_cli.py")
    assert reviewer != -1, "daily ladder missing the mandatory reviewer_cli.py gate"
    assert execute != -1, "daily ladder missing the gate_execute_cli.py execute step"
    assert reviewer < execute, "daily reviewer gate MUST come before the execute step"
    assert "MANDATORY" in daily, "daily reviewer gate must be marked MANDATORY"


def test_skill_md_names_only_provenanced_clis() -> None:
    """Every CLI the ladders name is created by a task per the provenance table."""
    body = SKILL_PATH.read_text()
    expected = [
        "runlock_cli.py",  # P0
        "due_check.py",  # P0
        "scout_cli.py",  # P0
        "preflight.py",  # P0
        "control_loop_cli.py",  # P3
        "reviewer_cli.py",  # P5
        "gate_execute_cli.py",  # P4 (Task 4.5)
        "record_lessons_cli.py",  # P0
        "promote_lesson_cli.py",  # P6
        "reflect_cli.py",  # P6
        "monitor_cli.py",  # P3
    ]
    for cli in expected:
        assert cli in body, f"SKILL.md must name the provenanced CLI {cli}"


def test_skill_md_dispatches_model_per_cadence() -> None:
    """Model dispatch resolves through `model_for(role, loop=cadence)` for both cadences."""
    body = SKILL_PATH.read_text()
    assert "model_for" in body, "SKILL.md must document model dispatch via model_for"
    assert "loop=weekly" in body and "loop=daily" in body, (
        "model dispatch must cover both weekly and daily cadences"
    )


def test_skill_md_keeps_live_false_and_standdown_contract() -> None:
    body = SKILL_PATH.read_text()
    assert "live" in body and "false" in body, "SKILL.md must affirm `live` stays false"
    assert "management" in body, "SKILL.md must state the empty-`management` stand-down contract"

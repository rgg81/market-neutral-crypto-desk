# tests/test_record_lessons_cli.py
from __future__ import annotations

import json

from futures_fund.cycle_io import save_output
from futures_fund.lessons import read_lessons
from scripts.record_lessons_cli import main


def test_records_lessons_from_reflector_output(tmp_path, capsys):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    save_output(state, 1, "lessons", {"lessons": [
        {"text": "pairs stayed cointegrated; size up next time", "polarity": "enabling",
         "dimension": "cointegration_break", "importance": 6},
        {"text": "funding flipped under stress; cut crowded carry", "polarity": "restrictive",
         "dimension": "carry_thesis_miss", "importance": 7},
    ]}, cadence="weekly")

    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state),
          "--memory-dir", str(memory)])

    lessons = read_lessons(memory)
    assert len(lessons) == 2
    assert {lz.dimension for lz in lessons} == {"cointegration_break", "carry_thesis_miss"}
    out = json.loads(capsys.readouterr().out)
    assert out["appended"] == 2


def test_missing_lessons_artifact_appends_nothing(tmp_path, capsys):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    main(["--cycle", "1", "--cadence", "weekly", "--state-dir", str(state),
          "--memory-dir", str(memory)])
    assert json.loads(capsys.readouterr().out)["appended"] == 0
    assert read_lessons(memory) == []

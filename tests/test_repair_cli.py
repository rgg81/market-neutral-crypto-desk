from __future__ import annotations

import json

from scripts.repair_cli import main


def test_repair_cli_refuses_protected_module(tmp_path, capsys):
    rc = main(["--module", "risk_gate", "--symptom", "x", "--root-cause", "y",
               "--fix", "z", "--verification", "tests", "--memory-dir", str(tmp_path / "memory")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is False
    journal = (tmp_path / "memory" / "repair-journal.md").read_text()
    assert "REFUSED" in journal


def test_repair_cli_applies_non_protected_module(tmp_path, capsys):
    rc = main(["--module", "cycle_prep", "--symptom", "x", "--root-cause", "y",
               "--fix", "z", "--verification", "tests", "--memory-dir", str(tmp_path / "memory")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is True
    journal = (tmp_path / "memory" / "repair-journal.md").read_text()
    assert "applied" in journal

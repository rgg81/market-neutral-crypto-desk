from __future__ import annotations

from pathlib import Path


def test_readme_exists_and_names_the_run_and_dashboard_commands():
    text = Path("README.md").read_text()
    assert "run_paper_cli.py" in text, "README must show how to run a cycle"
    assert "dashboard_cli.py" in text, "README must show how to read the dashboard"
    assert "paper" in text.lower(), "README must state the desk is paper-only"
    assert "neutral" in text.lower(), "README must state the dollar+beta-neutral mandate"

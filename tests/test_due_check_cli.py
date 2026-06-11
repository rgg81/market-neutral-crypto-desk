# tests/test_due_check_cli.py
from __future__ import annotations

from scripts.due_check import main


def test_cold_start_is_due_fresh_1(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert main([state, "--loop", "weekly"]) == 0
    assert "DUE FRESH 1" in capsys.readouterr().out


def test_daily_loop_uses_the_daily_root(tmp_path, capsys, write_served_report):
    # served the daily candle containing `now`; the weekly root is untouched, so weekly stays DUE.
    from datetime import UTC, datetime
    state = tmp_path / "state"
    now = datetime(2026, 6, 11, tzinfo=UTC)
    write_served_report(state / "daily" / "cycle" / "1", served=now, tf_minutes=1440)
    # daily SKIPs (its candle is served)
    assert main([str(state), "--loop", "daily", "--now", "2026-06-11T00:00:00+00:00"]) == 0
    assert "SKIP:" in capsys.readouterr().out
    # weekly is still DUE (different root, no served report)
    assert main([str(state), "--loop", "weekly", "--now", "2026-06-11T00:00:00+00:00"]) == 0
    assert "DUE" in capsys.readouterr().out


def test_unknown_loop_errors(tmp_path, capsys):
    assert main([str(tmp_path / "state"), "--loop", "hourly"]) == 2
    assert "ERROR:" in capsys.readouterr().out

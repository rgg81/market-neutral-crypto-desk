from datetime import UTC, datetime
from pathlib import Path

import pytest

import futures_fund.control_loop as cl
from futures_fund.control_loop import cadence_cycle_root, cadence_due


def test_cadence_due_weekly_delegates_with_weekly_root(tmp_path, monkeypatch):
    seen = {}

    def fake_cycle_due(state_dir, now_utc, *, tf_minutes, loop):
        seen["tf_minutes"] = tf_minutes
        seen["loop"] = loop
        return ("FRESH", 1, "spy")

    monkeypatch.setattr(cl, "cycle_due", fake_cycle_due)
    now = datetime(2026, 6, 11, 0, 0, tzinfo=UTC)
    mode, n, reason = cadence_due(tmp_path / "s", now, "weekly")
    assert seen == {"tf_minutes": 10080, "loop": "weekly"}  # root => state/weekly/cycle/*
    assert (mode, n) == ("FRESH", 1)


def test_cadence_due_daily_delegates_with_daily_root(tmp_path, monkeypatch):
    seen = {}

    def fake_cycle_due(state_dir, now_utc, *, tf_minutes, loop):
        seen.update(tf_minutes=tf_minutes, loop=loop)
        return ("FRESH", 1, "spy")

    monkeypatch.setattr(cl, "cycle_due", fake_cycle_due)
    now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
    cadence_due(tmp_path / "s", now, "daily")
    assert seen == {"tf_minutes": 1440, "loop": "daily"}  # root => state/daily/cycle/*


def test_cadence_cannot_double_fire(tmp_path, write_served_report):
    # seed a completed report for the candle containing `now` under state/daily/cycle/1/
    now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
    write_served_report(tmp_path / "s" / "daily" / "cycle" / "1", served=now, tf_minutes=1440)
    mode, n, _ = cadence_due(tmp_path / "s", now, "daily")  # real cycle_due, no monkeypatch
    assert mode == "SKIP"
    assert n == 1


@pytest.mark.parametrize("cadence", ["weekly", "daily"])
def test_cadence_cycle_root_is_canonical_path(tmp_path, cadence):
    # CADENCE-ROOT INVARIANT: artifacts live at state/<cadence>/cycle (never state/cycle/<cadence>).
    assert cadence_cycle_root(tmp_path / "s", cadence) == Path(tmp_path / "s") / cadence / "cycle"


@pytest.mark.parametrize("cadence,tf", [("weekly", 10080), ("daily", 1440)])
def test_cadence_root_binds_writer_to_gate_reader(tmp_path, write_served_report, cadence, tf):
    # CADENCE-ROOT INVARIANT enforced end-to-end: a report WRITTEN under cadence_cycle_root (the
    # single source of truth a future artifact writer must use) is exactly what cadence_due READS.
    # If the gate scanned a different root the seeded candle would be invisible and this would not
    # SKIP. now=12:00Z so the served candle covers it for both daily and weekly grids.
    now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
    write_dir = cadence_cycle_root(tmp_path / "s", cadence) / "1"
    write_served_report(write_dir, served=now, tf_minutes=tf)
    mode, n, _ = cadence_due(tmp_path / "s", now, cadence)  # real cycle_due, no monkeypatch
    assert (mode, n) == ("SKIP", 1)

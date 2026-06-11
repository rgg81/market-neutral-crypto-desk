from datetime import UTC, datetime
from pathlib import Path

import pytest

import futures_fund.control_loop as cl
from futures_fund.contracts import CoinGeometry, SleeveSignal, SleeveTilt, TargetWeights
from futures_fund.control_loop import (
    cadence_cycle_root,
    cadence_due,
    rebalance_deltas,
    weekly_selection,
)
from futures_fund.cycle_io import load_output
from futures_fund.neutrality import NeutralityConfig

NOW = datetime(2026, 6, 11, tzinfo=UTC)


def _broad_geometries():
    """A 6-name universe (3 long / 3 short) with a BALANCED beta structure, so a fully-deployed
    dollar+beta-neutral book CAN respect the 25% per-name cap (feasible=True)."""
    return [
        CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0, beta_btc=1.0, adv_usd=2e9),
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, beta_btc=1.1, adv_usd=1e9),
        CoinGeometry(symbol="SOL/USDT:USDT", mark=150.0, beta_btc=1.2, adv_usd=4e8),
        CoinGeometry(symbol="XRP/USDT:USDT", mark=0.6, beta_btc=1.0, adv_usd=3e8),
        CoinGeometry(symbol="ADA/USDT:USDT", mark=0.5, beta_btc=1.1, adv_usd=2e8),
        CoinGeometry(symbol="DOGE/USDT:USDT", mark=0.15, beta_btc=1.2, adv_usd=2e8),
    ]


def _broad_sleeves(now):
    """3 longs / 3 shorts so each side can spread its gross across enough names to stay under
    the per-name cap (the band-respecting, feasible book)."""
    return [SleeveSignal(
        sleeve="factor", risk_budget_frac=1.0, as_of_ts=now,
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ADA/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
            SleeveTilt(symbol="DOGE/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )]


def test_weekly_selection_runs_optimizer(tmp_path):
    cfg = NeutralityConfig()
    geometries = _broad_geometries()
    sleeves = _broad_sleeves(NOW)
    tw = weekly_selection(
        tmp_path / "s", geometries, sleeves,
        equity=20000.0, prior=None, cfg=cfg, cycle=1,
    )
    # returns a TargetWeights whose residuals are in band and which is feasible
    assert isinstance(tw, TargetWeights)
    assert tw.feasible is True
    assert tw.dollar_residual_frac <= cfg.dollar_band + 1e-6
    assert abs(tw.beta_residual) <= cfg.beta_band + 1e-6
    # persisted under state/weekly/cycle/1/target_weights.json (cadence-segmented root)
    persisted = tmp_path / "s" / "weekly" / "cycle" / "1" / "target_weights.json"
    assert persisted.exists()
    reloaded = TargetWeights.model_validate(
        load_output(tmp_path / "s", 1, "target_weights", cadence="weekly")
    )
    assert reloaded.feasible is True
    assert [leg.symbol for leg in reloaded.legs] == [leg.symbol for leg in tw.legs]


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


def test_rebalance_deltas_excludes_unchanged_overlap(make_tw):
    prior = make_tw([("BTC/USDT:USDT", "long", 5000.0)])
    target = make_tw([("BTC/USDT:USDT", "long", 5000.0),
                      ("ETH/USDT:USDT", "short", 5000.0)])
    deltas = rebalance_deltas(prior, target)
    syms = {leg.symbol for leg in deltas}
    assert syms == {"ETH/USDT:USDT"}


def test_rebalance_deltas_unwinds_removed(make_tw):
    # leg in prior but absent from target -> zero-notional unwind delta
    prior = make_tw([("BTC/USDT:USDT", "long", 5000.0),
                     ("ETH/USDT:USDT", "short", 5000.0)])
    target = make_tw([("BTC/USDT:USDT", "long", 5000.0)])
    deltas = rebalance_deltas(prior, target)
    by_sym = {leg.symbol: leg for leg in deltas}
    assert set(by_sym) == {"ETH/USDT:USDT"}  # BTC unchanged -> excluded
    unwind = by_sym["ETH/USDT:USDT"]
    assert unwind.direction == "short"
    assert unwind.target_notional == 0.0
    assert unwind.weight == 0.0


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

"""Task 7.3 — end-to-end paper run driver (`scripts/run_paper_cli.py`).

A full weekly->daily loop on a FAKE (injected) exchange, no network. The driver serializes both
cadences WEEKLY-FIRST under ONE run lock (`runlock.single_flight(owner="paper")`) and, per cadence,
walks the SKILL.md ladder seams: lock+due -> cadence step -> reviewer gate -> execute -> equity ->
reflect. Each stage names its exact CLI/function (control_loop_cli / reviewer_cli / gate_execute_cli
/ equity_log / reflect_cli), so the orchestration is the deterministic glue, not new business logic.

The test asserts the acceptance criteria (Phase 7): the produced book is dollar+beta neutral WITHIN
BANDS, the persisted `ReviewerVerdict.passed is True` (the reviewer gate is actually exercised), a
non-empty `report.json` is written, and an equity point is appended — all under a single lock that a
served candle SKIPs and a tampered (off-neutral) book HALTs at the reviewer stage.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from futures_fund.contracts import (
    CoinGeometry,
    GeometryBundle,
    SleeveSignal,
    SleeveTilt,
)
from futures_fund.cycle_io import save_output

NOW_ISO = "2026-06-11T00:00:00+00:00"


def _balanced_geometries() -> list[CoinGeometry]:
    """6-name (3 long / 3 short), balanced-beta universe — a fully-deployed dollar+beta-neutral
    book respecting the per-name cap is feasible (the proven control-loop optimizer input)."""
    return [
        CoinGeometry(symbol="BTC/USDT:USDT", mark=60000.0, beta_btc=1.0, adv_usd=2e9,
            market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="ETH/USDT:USDT", mark=3000.0, beta_btc=1.0, adv_usd=1e9,
            market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="SOL/USDT:USDT", mark=150.0, beta_btc=1.0, adv_usd=4e8,
            market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="XRP/USDT:USDT", mark=0.6, beta_btc=1.0, adv_usd=3e8,
            market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="ADA/USDT:USDT", mark=0.5, beta_btc=1.0, adv_usd=2e8,
            market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="DOGE/USDT:USDT", mark=0.15, beta_btc=1.0, adv_usd=2e8,
            market_info={"underlyingType": "COIN"}),
    ]


def _balanced_sleeves() -> list[SleeveSignal]:
    return [SleeveSignal(
        sleeve="factor", risk_budget_frac=1.0, as_of_ts=NOW_ISO,
        tilts=[
            SleeveTilt(symbol="BTC/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="SOL/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ADA/USDT:USDT", direction="long", target_weight=0.5),
            SleeveTilt(symbol="ETH/USDT:USDT", direction="short", target_weight=-0.5),
            SleeveTilt(symbol="XRP/USDT:USDT", direction="short", target_weight=-0.5),
            SleeveTilt(symbol="DOGE/USDT:USDT", direction="short", target_weight=-0.5),
        ],
    )]


def _seed_upstream(state, cycle: int = 1) -> None:
    """Seed the geometries/sleeves the control-loop step loads, under BOTH cadence roots so the
    weekly-first driver finds a feasible balanced book for each cadence at cycle `cycle`."""
    bundle = GeometryBundle(geometries=_balanced_geometries(), as_of_ts=NOW_ISO)
    for cadence in ("weekly", "daily"):
        save_output(state, cycle, "geometries", bundle, cadence=cadence)
        save_output(
            state, cycle, "sleeves",
            {"sleeves": [s.model_dump(mode="json") for s in _balanced_sleeves()]},
            cadence=cadence,
        )


@pytest.fixture
def paper_env(tmp_path, monkeypatch):
    """A chdir'd tmp workspace with seeded upstream artifacts, a fake (no-network) exchange, and a
    pinned `now` — the deterministic, offline harness for the e2e driver."""
    state = tmp_path / "state"
    _seed_upstream(state, cycle=1)
    # `Settings.live` defaults False (PAPER-ONLY desk) — no override needed.
    # Inject a fake exchange: the paper-only execute boundary records, never hits the network.
    monkeypatch.setattr(
        "scripts.gate_execute_cli.FuturesExchange.from_settings", lambda settings: object()
    )
    # Phase 8: the producers (scout + cycle-prep) are a NO-OP for the seeded E2E — these tests
    # assert behavior on _seed_upstream's hand-seeded artifacts, which the real producers would
    # overwrite. The dedicated no-seed E2E (test_end_to_end_no_seed.py) exercises the real
    # producers.
    monkeypatch.setattr("scripts.run_paper_cli._run_producers",
                        lambda state_dir, cadence, cycle, now, memory_dir="memory": None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_end_to_end_weekly_then_daily_paper_run(paper_env):
    from futures_fund.contracts import TargetWeights
    from scripts.run_paper_cli import main

    main(["--now", NOW_ISO])

    state = paper_env / "state"
    # WEEKLY first: a neutral, in-band book was produced and persisted under the weekly root.
    weekly_tw = state / "weekly" / "cycle" / "1" / "target_weights.json"
    assert weekly_tw.exists()
    tw = TargetWeights.model_validate(json.loads(weekly_tw.read_text()))
    cfg_dollar, cfg_beta = 0.03, 0.05
    assert tw.feasible is True
    assert tw.dollar_residual_frac <= cfg_dollar + 1e-6
    assert abs(tw.beta_residual) <= cfg_beta + 1e-6

    # A GREEN ReviewerVerdict was persisted under the weekly root (the gate was exercised).
    reviewer = json.loads((state / "weekly" / "cycle" / "1" / "reviewer.json").read_text())
    assert reviewer["passed"] is True

    # A non-empty execution report was written under the weekly root.
    report = json.loads((state / "weekly" / "cycle" / "1" / "report.json").read_text())
    assert report["live"] is False
    assert report["executed"]  # non-empty: the trader hand-off reached the boundary

    # An equity point was appended (the return-series source the dashboard reads).
    eq = state / "equity-history.jsonl"
    assert eq.exists()
    points = [json.loads(line) for line in eq.read_text().splitlines() if line.strip()]
    assert points  # at least one equity point recorded

    # The DAILY cadence also ran weekly-first-then-daily under the SAME lock.
    assert (state / "daily" / "cycle" / "1" / "report.json").exists()

    # The run lock is RELEASED at the end (single-flight context exited).
    assert not (state / ".run.lock").exists()


def test_end_to_end_holds_single_run_lock(paper_env, monkeypatch):
    # Step 3a: the driver runs UNDER a single run lock — while the cadence step executes, the lock
    # is HELD (a concurrent fire would stand down). We assert the lock file exists mid-run.
    from scripts import run_paper_cli

    state = paper_env / "state"
    seen = {}
    real_step = run_paper_cli._run_cadence

    def spy_step(cadence, *a, **k):
        seen.setdefault("lock_held_during", []).append((state / ".run.lock").exists())
        return real_step(cadence, *a, **k)

    monkeypatch.setattr(run_paper_cli, "_run_cadence", spy_step)
    run_paper_cli.main(["--now", NOW_ISO])
    # the lock was held for every cadence step, and released afterward
    assert seen["lock_held_during"] and all(seen["lock_held_during"])
    assert not (state / ".run.lock").exists()


def test_end_to_end_skips_served_candle(paper_env, write_served_report):
    # Step 3a: a cadence whose current candle was ALREADY served (a completed report.json) SKIPs —
    # the driver does not re-run it. Seed a served candle (for the candle containing `now`) on BOTH
    # cadence roots so the whole weekly-first run stands down cleanly.
    from scripts.run_paper_cli import main

    state = paper_env / "state"
    now = datetime(2026, 6, 11, tzinfo=UTC)
    write_served_report(state / "weekly" / "cycle" / "1", served=now, tf_minutes=10080)
    write_served_report(state / "daily" / "cycle" / "1", served=now, tf_minutes=1440)
    main(["--now", NOW_ISO])
    # Both cadences SKIPped: no reviewer.json was produced for the served candle's cycle by this run
    # (the served reports stand; the driver did not overwrite them with a fresh execute).
    assert not (state / "weekly" / "cycle" / "1" / "reviewer.json").exists()
    assert not (state / "daily" / "cycle" / "1" / "reviewer.json").exists()


def test_end_to_end_tampered_book_halts_at_reviewer(paper_env, monkeypatch):
    # Step 5a: a tampered (off-neutral) book must HALT at the reviewer gate (SystemExit(2)) BEFORE
    # any fill. We force the control-loop step to persist a book whose dollar residual breaches the
    # band, so the reviewer's dollar_residual_in_band check fails and the verdict vetoes.
    from futures_fund.contracts import TargetWeights, WeightLeg
    from futures_fund.cycle_io import save_output as _save
    from scripts import run_paper_cli

    state = paper_env / "state"

    def tamper_then_run_step(state_dir, cadence, cycle):
        # off-neutral book: a single long leg, no offsetting short -> dollar residual = 100%.
        bad = TargetWeights(
            legs=[WeightLeg(symbol="BTC/USDT:USDT", direction="long", weight=1.0,
                            target_notional=5000.0, beta_btc=1.0, sleeve="factor")],
            dollar_residual=5000.0, dollar_residual_frac=1.0, beta_residual=1.0,
            gross_long=5000.0, gross_short=0.0,
            deploy_long_frac=0.9, deploy_short_frac=0.0,
            gross_notional=5000.0, as_of_ts=NOW_ISO,
        )
        _save(state_dir, cycle, "target_weights", bad, cadence=cadence)

    monkeypatch.setattr(run_paper_cli, "_run_control_loop_step", tamper_then_run_step)
    with pytest.raises(SystemExit) as exc:
        run_paper_cli.main(["--now", NOW_ISO])
    assert exc.value.code == 2
    # HALTed at the reviewer gate: no execution report was written for the tampered book.
    assert not (state / "weekly" / "cycle" / "1" / "report.json").exists()
    # the lock is still released even on a HALT (single_flight finally-releases).
    assert not (state / ".run.lock").exists()

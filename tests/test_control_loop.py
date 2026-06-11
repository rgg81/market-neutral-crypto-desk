from datetime import UTC, datetime
from pathlib import Path

import pytest

import futures_fund.control_loop as cl
from futures_fund.contracts import (
    CoinGeometry,
    SleeveSignal,
    SleeveTilt,
    Spread,
    TargetWeights,
    WeightLeg,
)
from futures_fund.control_loop import (
    cadence_cycle_root,
    cadence_due,
    daily_rebalance,
    drift_exceeded,
    neutrality_breached,
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


def test_rebalance_deltas_emits_changed_overlap(make_tw):
    # Headline branch (§9 "trade only the deltas"): an OVERLAPPING leg (same symbol+direction
    # present in both books) whose target_notional MOVED beyond the $1 epsilon must be emitted as a
    # delta carrying the NEW notional, while a co-resident leg that did NOT move stays excluded.
    prior = make_tw([("BTC/USDT:USDT", "long", 5000.0),
                     ("ETH/USDT:USDT", "short", 5000.0)])
    target = make_tw([("BTC/USDT:USDT", "long", 5002.0),   # +$2 > epsilon -> delta
                      ("ETH/USDT:USDT", "short", 5000.0)])  # unchanged -> excluded
    deltas = rebalance_deltas(prior, target)
    by_sym = {leg.symbol: leg for leg in deltas}
    assert set(by_sym) == {"BTC/USDT:USDT"}
    assert by_sym["BTC/USDT:USDT"].target_notional == 5002.0  # carries the moved target


@pytest.mark.parametrize(
    "new_notional,expect_emitted",
    [
        (5000.5, False),  # +$0.5 <= $1 epsilon -> excluded (no churn)
        (5002.0, True),   # +$2.0  >  $1 epsilon -> emitted
    ],
)
def test_rebalance_deltas_epsilon_boundary(make_tw, new_notional, expect_emitted):
    # The $1 no-churn epsilon is the boundary: a move <= $1 is excluded, a move > $1 is emitted.
    prior = make_tw([("BTC/USDT:USDT", "long", 5000.0)])
    target = make_tw([("BTC/USDT:USDT", "long", new_notional)])
    syms = {leg.symbol for leg in rebalance_deltas(prior, target)}
    assert syms == ({"BTC/USDT:USDT"} if expect_emitted else set())


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


def _tw_with_residuals(*, dollar_residual_frac: float, beta_residual: float) -> TargetWeights:
    """A minimal `TargetWeights` whose neutrality residual fields carry the supplied values; the
    deployment/leg scaffolding is inert for `neutrality_breached` (it keys only on residuals)."""
    return TargetWeights(
        legs=[],
        dollar_residual=0.0,
        dollar_residual_frac=dollar_residual_frac,
        beta_residual=beta_residual,
        gross_long=0.0,
        gross_short=0.0,
        deploy_long_frac=0.0,
        deploy_short_frac=0.0,
        gross_notional=0.0,
        as_of_ts=NOW,
    )


def test_drift_exceeded():
    # 0.5 vs 0.4 -> 25% drift > 20% band -> True
    assert drift_exceeded(0.5, 0.4, drift_band=0.20) is True
    # 0.45 vs 0.4 -> 12.5% drift <= 20% band -> False
    assert drift_exceeded(0.45, 0.4, drift_band=0.20) is False
    # target==0 -> any nonzero current is a breach; zero current is in-band
    assert drift_exceeded(0.1, 0.0, drift_band=0.20) is True
    assert drift_exceeded(0.0, 0.0, drift_band=0.20) is False


def test_neutrality_breached():
    cfg = NeutralityConfig()  # dollar_band=0.03, beta_band=0.05
    # in-band on both axes -> not breached
    assert neutrality_breached(
        _tw_with_residuals(dollar_residual_frac=0.02, beta_residual=0.04), cfg
    ) is False
    # dollar residual frac over the dollar band -> breached
    assert neutrality_breached(
        _tw_with_residuals(dollar_residual_frac=0.04, beta_residual=0.0), cfg
    ) is True
    # |beta residual| over the beta band (negative side) -> breached
    assert neutrality_breached(
        _tw_with_residuals(dollar_residual_frac=0.0, beta_residual=-0.06), cfg
    ) is True


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


def _stop_spread(pair_id: str) -> Spread:
    """A `Spread` flipped to the z-stop state (|z| past stop_z) — its legs MUST be force-traded."""
    return Spread(pair_id=pair_id, spread_value=0.0, zscore=3.5, state="stop")


def test_daily_rebalance_same_set(tmp_path):
    # Daily Rebalance Meeting (§9): keep the SAME symbol set as the weekly target, recompute
    # residuals/z/funding/sentiment, and (with an in-band book and no z-stops) emit ZERO delta legs
    # so the carry-over book is not churned. Still persists a TargetWeights under the daily root.
    cfg = NeutralityConfig()
    geometries = _broad_geometries()
    sleeves = _broad_sleeves(NOW)
    target = weekly_selection(
        tmp_path / "s", geometries, sleeves,
        equity=20000.0, prior=None, cfg=cfg, cycle=1,
    )
    weekly_symbols = {leg.symbol for leg in target.legs}

    result = daily_rebalance(
        tmp_path / "s", target, geometries, spreads=[],
        equity=20000.0, cfg=cfg, cycle=1,
    )
    assert isinstance(result, TargetWeights)
    # SAME symbol set: the recompute introduced no new names and dropped none.
    assert {leg.symbol for leg in result.legs} <= weekly_symbols
    # in-band, no z-stop, no neutrality breach -> zero delta legs (no churn)
    assert result.legs == []
    # persisted under state/daily/cycle/1/target_weights.json (cadence-segmented daily root)
    persisted = tmp_path / "s" / "daily" / "cycle" / "1" / "target_weights.json"
    assert persisted.exists()
    reloaded = TargetWeights.model_validate(
        load_output(tmp_path / "s", 1, "target_weights", cadence="daily")
    )
    assert reloaded.legs == []


def test_daily_rebalance_zstop_flattens(tmp_path):
    # A hard z-stop (Spread.state=="stop") is the cointegration-break EXIT (§6.2; the pairs sleeve
    # treats "stop" as "emit no legs" == close). So it must force the stopped pair's legs into the
    # delta book as ZERO-notional UNWINDS (flatten the broken pair), NOT re-mark them at target
    # notional — even when every leg is individually inside its drift band (no-churn rebalance).
    cfg = NeutralityConfig()
    geometries = _broad_geometries()
    sleeves = _broad_sleeves(NOW)
    target = weekly_selection(
        tmp_path / "s", geometries, sleeves,
        equity=20000.0, prior=None, cfg=cfg, cycle=1,
    )
    # Pick two real symbols from the weekly book and bind them into a pair whose spread is stopped.
    syms = [leg.symbol for leg in target.legs]

    def _slug(sym: str) -> str:
        return sym.replace("/", "").replace(":", "")

    pair_id = f"{_slug(syms[0])}__{_slug(syms[1])}"
    # stamp the pair_id onto the two legs so the forced spread maps back to real book legs
    forced = target.model_copy(update={"legs": [
        leg.model_copy(update={"pair_id": pair_id}) if leg.symbol in (syms[0], syms[1]) else leg
        for leg in target.legs
    ]})
    prior_by_sym = {leg.symbol: leg for leg in forced.legs}

    result = daily_rebalance(
        tmp_path / "s", forced, geometries, spreads=[_stop_spread(pair_id)],
        equity=20000.0, cfg=cfg, cycle=2,
    )
    by_key = {(leg.symbol, leg.direction): leg for leg in result.legs}
    # the stopped pair's legs are forced into the delta book despite being in drift band
    assert {syms[0], syms[1]} <= {leg.symbol for leg in result.legs}
    for sym in (syms[0], syms[1]):
        # the override keys on the PRIOR leg's (symbol, direction) — the position that exists
        d = prior_by_sym[sym].direction
        flat = by_key[(sym, d)]
        # FLATTEN, not re-mark: zeroed notional + weight, same direction as the prior position
        assert flat.target_notional == 0.0, f"{sym} stopped pair should flatten, not re-mark"
        assert flat.weight == 0.0
        assert flat.direction == d


def test_daily_rebalance_neutrality_breach_forces_full_set(tmp_path, monkeypatch):
    # neutrality-breach override (§9): when the RECOMPUTED book is off-neutral (dollar/beta residual
    # past its band) the FULL recomputed leg set is forced into the delta book — even though those
    # legs are individually UNCHANGED vs the prior target (so the base carry-over rebalance_deltas
    # would emit ZERO). We hold the recomputed legs identical to the prior so ONLY the breach branch
    # can produce output, isolating the override.
    cfg = NeutralityConfig()  # dollar_band=0.03, beta_band=0.05
    geometries = _broad_geometries()
    sleeves = _broad_sleeves(NOW)
    target = weekly_selection(
        tmp_path / "s", geometries, sleeves,
        equity=20000.0, prior=None, cfg=cfg, cycle=1,
    )
    # Build a recomputed book whose legs EXACTLY equal the prior target's legs (zero base delta),
    # but whose residual fields BREACH the dollar band (frac well over 0.03).
    breached = target.model_copy(update={
        "legs": [WeightLeg(**leg.model_dump()) for leg in target.legs],
        "dollar_residual_frac": 0.50,
        "beta_residual": 0.0,
    })
    monkeypatch.setattr(cl, "optimize_book", lambda *a, **k: breached)

    result = daily_rebalance(
        tmp_path / "s", target, geometries, spreads=[],
        equity=20000.0, cfg=cfg, cycle=3,
    )
    # without the breach override the unchanged book would yield ZERO deltas; the breach forces the
    # FULL recomputed leg set (same symbol set, same directions) into the delta book.
    assert len(result.legs) == len(target.legs)
    assert {(leg.symbol, leg.direction) for leg in result.legs} == \
        {(leg.symbol, leg.direction) for leg in target.legs}

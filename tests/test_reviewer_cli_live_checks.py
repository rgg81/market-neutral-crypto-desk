# tests/test_reviewer_cli_live_checks.py
from __future__ import annotations

import json

import pytest

from futures_fund.contracts import (
    CoinGeometry,
    GeometryBundle,
    Pair,
    Spread,
    TargetWeights,
    WeightLeg,
)
from futures_fund.cycle_io import cycle_dir, save_output

NOW = "2026-06-11T00:00:00+00:00"


def _neutral_book() -> TargetWeights:
    # 4 legs @ 4500: per-name |w| = 4500/20000 = 0.225 <= per_name_cap 0.25, and each side gross
    # 9000/side_budget 10000 = 0.90 >= deployment_floor 0.90 -> caps + deployment both PASS.
    return TargetWeights(
        legs=[
            WeightLeg(symbol="AAA/USDT:USDT", direction="long", weight=0.225,
                      target_notional=4500.0, beta_btc=1.0, sleeve="pairs",
                      pair_id="AAAUSDT__BBBUSDT"),
            WeightLeg(symbol="BBB/USDT:USDT", direction="short", weight=-0.225,
                      target_notional=4500.0, beta_btc=1.0, sleeve="pairs",
                      pair_id="AAAUSDT__BBBUSDT"),
            WeightLeg(symbol="CCC/USDT:USDT", direction="long", weight=0.225,
                      target_notional=4500.0, beta_btc=1.0, sleeve="factor"),
            WeightLeg(symbol="DDD/USDT:USDT", direction="short", weight=-0.225,
                      target_notional=4500.0, beta_btc=1.0, sleeve="factor"),
        ],
        dollar_residual=0.0, dollar_residual_frac=0.0, beta_residual=0.0,
        gross_long=9000.0, gross_short=9000.0,
        deploy_long_frac=0.9, deploy_short_frac=0.9, gross_notional=18000.0, as_of_ts=NOW,
    )


def _geos() -> GeometryBundle:
    return GeometryBundle(geometries=[
        CoinGeometry(symbol="AAA/USDT:USDT", mark=100.0, beta_btc=1.0, funding_rate=0.0,
                     market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="BBB/USDT:USDT", mark=200.0, beta_btc=1.0, funding_rate=0.0,
                     market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="CCC/USDT:USDT", mark=50.0, beta_btc=1.0, funding_rate=0.0,
                     market_info={"underlyingType": "COIN"}),
        CoinGeometry(symbol="DDD/USDT:USDT", mark=25.0, beta_btc=1.0, funding_rate=0.0,
                     market_info={"underlyingType": "COIN"}),
    ], as_of_ts=NOW)


def _pair() -> Pair:
    return Pair(pair_id="AAAUSDT__BBBUSDT", symbol_y="AAA/USDT:USDT", symbol_x="BBB/USDT:USDT",
                hedge_ratio=0.5, method="engle_granger", adf_pvalue=0.01, adf_pvalue_adj=0.02,
                half_life=10.0, theta=0.07, mu=0.0, sigma_eq=1.0, formed_cycle=1)


def _seed(state, *, spread_pnl: float, qty_y: float, qty_x: float):
    save_output(state, 1, "target_weights", _neutral_book(), cadence="weekly")
    save_output(state, 1, "geometries", _geos(), cadence="weekly")
    save_output(state, 1, "pairs", {"pairs": [_pair().model_dump(mode="json")]}, cadence="weekly")
    # a live spread whose realized_pnl/leg sizing the reviewer will re-derive
    sp = Spread(pair_id="AAAUSDT__BBBUSDT", spread_value=2.0, zscore=2.0, state="short_spread",
                qty_y=qty_y, qty_x=qty_x, realized_pnl=spread_pnl)
    save_output(state, 1, "spreads", {"spreads": [sp.model_dump(mode="json")]}, cadence="weekly")
    # NOTE: proposals.json is intentionally NOT seeded — reviewer_cli reconstructs RR-capable
    # TradeProposals from the audited book + geometries (the persisted target_notional-only
    # proposals carry no entry/stop/TP geometry and are NOT consumed for RR).


def test_fabricated_spread_pnl_now_fails_the_gate(tmp_path, monkeypatch):
    # correct leg sizing (qty_x == hedge_ratio*qty_y) but a LIED-ABOUT realized_pnl: with pairs.json
    # now loaded, check_pair_pnl re-derives PnL-since-entry and the fabricated value FAILS.
    state = tmp_path / "state"
    _seed(state, spread_pnl=999999.0, qty_y=10.0, qty_x=5.0)
    monkeypatch.chdir(tmp_path)
    from scripts.reviewer_cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--cadence", "weekly", "--cycle", "1", "--state-dir", str(state)])
    assert exc.value.code == 2
    verdict = json.loads((cycle_dir(state, 1, cadence="weekly") / "reviewer.json").read_text())
    assert verdict["passed"] is False
    assert "pair_pnl_attribution" in verdict["mismatches"]


def test_honest_book_passes_with_live_pair_and_rr_checks(tmp_path, monkeypatch):
    # honest spread PnL (re-derive expected and store it) + correct hedge-ratio sizing; the RR check
    # is LIVE (proposals reconstructed) and clears MIN_RR -> verdict passes. With the 4-leg @4500
    # book the per-name cap (0.225 <= 0.25) and the deployment floor (0.90 >= 0.90) both hold, so
    # the ONLY checks gating `passed` are the now-live pair-PnL + RR ones.
    from futures_fund.reviewer import check_pair_pnl
    state = tmp_path / "state"
    # first seed with a placeholder to compute the honest pnl, then re-seed the spread
    _seed(state, spread_pnl=0.0, qty_y=10.0, qty_x=5.0)
    sp = Spread(pair_id="AAAUSDT__BBBUSDT", spread_value=2.0, zscore=2.0, state="short_spread",
                qty_y=10.0, qty_x=5.0, realized_pnl=0.0)
    expected = check_pair_pnl([sp], [_pair()])[0].expected  # re-derived honest PnL
    sp_honest = sp.model_copy(update={"realized_pnl": expected})
    save_output(state, 1, "spreads", {"spreads": [sp_honest.model_dump(mode="json")]},
                cadence="weekly")
    monkeypatch.chdir(tmp_path)
    from scripts.reviewer_cli import main

    main(["--cadence", "weekly", "--cycle", "1", "--state-dir", str(state)])  # no HALT
    verdict = json.loads((cycle_dir(state, 1, cadence="weekly") / "reviewer.json").read_text())
    assert verdict["passed"] is True
    # rr_after_costs ran on reconstructed proposals (NOT the vacuous empty-list pass)
    rr = next(c for c in verdict["checks"] if c["name"] == "rr_after_costs")
    assert rr["ok"] is True
    assert "vacuously" not in rr["detail"]

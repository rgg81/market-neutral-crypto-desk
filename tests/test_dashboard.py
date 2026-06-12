"""Phase 7 / Task 7.1 — KPI dashboard.

`build_kpi_dashboard(state_dir, memory_dir)` is the read-only end-of-run scorecard that bundles the
desk's success + process KPIs (spec §18):

- `no_losing_month`   — fraction of calendar months that are positive (PRIMARY; target 1.0), read
                        off the persisted equity series (`equity_log`).
- `daily_sharpe`      — `metrics.sharpe(returns, periods_per_year=365)` of the daily equity series.
- `sortino`           — downside-only annualized (×365) ratio of the same series.
- `max_drawdown`      — peak-to-trough decline of the equity curve.
- `both_sides_deployment_rate` / `pair_survival` / `carry_capture` / `sentiment_hit_rate` /
  `reviewer_veto_rate` — the process KPIs, REUSED verbatim from the `improvement` panel (Task 6.3).
- `neutrality_adherence` — fraction of cycles whose persisted residuals are within the
                        `NeutralityConfig` bands (dollar residual fraction ≤ dollar_band AND
                        |beta residual| ≤ beta_band).

The tests seed the equity series via the real `equity_log.record_equity`, the cycle artifacts via
the real `cycle_io.save_output`, and the journal outcomes via the real
`journal.append_decision`/`patch_outcome`, then assert exact hand-computed expectations.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from futures_fund.cycle_io import save_output
from futures_fund.dashboard import build_kpi_dashboard
from futures_fund.equity_log import record_equity
from futures_fund.journal import append_decision, patch_outcome
from futures_fund.metrics import max_drawdown, sharpe


def _target_weights(*, long_frac: float, short_frac: float,
                    dollar_residual_frac: float, beta_residual: float) -> dict:
    """Minimal `target_weights.json` payload carrying the deployment + residual fields read."""
    return {
        "legs": [],
        "btc_hedge_notional": 0.0,
        "dollar_residual": dollar_residual_frac * 10000.0,
        "dollar_residual_frac": dollar_residual_frac,
        "beta_residual": beta_residual,
        "gross_long": long_frac * 10000.0,
        "gross_short": short_frac * 10000.0,
        "deploy_long_frac": long_frac,
        "deploy_short_frac": short_frac,
        "gross_notional": (long_frac + short_frac) * 10000.0,
        "as_of_ts": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    }


def _reviewer(*, cycle: int, passed: bool) -> dict:
    return {
        "passed": passed,
        "checks": [],
        "mismatches": [] if passed else ["dollar_residual_in_band"],
        "cycle": cycle,
        "cadence": "weekly",
        "reviewed_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    }


def _seed_journal_leg(memory_dir, *, cycle, symbol, direction, outcome) -> None:
    append_decision(memory_dir, cycle=cycle, symbol=symbol, direction=direction, payload={})
    patch_outcome(memory_dir, cycle=cycle, symbol=symbol, direction=direction, outcome=outcome)


@pytest.fixture
def seeded_daily_returns() -> list[float]:
    """The day-over-day returns implied by the seeded equity series below."""
    eq = _EQUITY_POINTS
    return [eq[i] / eq[i - 1] - 1.0 for i in range(1, len(eq))]


# A daily equity curve that spans two calendar months: month 1 (Jan) ends ABOVE its start (a
# winning month), month 2 (Feb) ends BELOW its start (a losing month) -> no_losing_month == 0.5.
_EQUITY_DATES = [
    datetime(2026, 1, 1, tzinfo=UTC),
    datetime(2026, 1, 15, tzinfo=UTC),
    datetime(2026, 1, 31, tzinfo=UTC),
    datetime(2026, 2, 14, tzinfo=UTC),
    datetime(2026, 2, 28, tzinfo=UTC),
]
_EQUITY_POINTS = [20000.0, 20400.0, 20600.0, 20200.0, 20100.0]


@pytest.fixture
def state_dir(tmp_path):
    """A `state/` seeded with the equity series + 4 cycles of cadence-segmented cycle artifacts."""
    sd = tmp_path / "state"
    for i, (ts, eq) in enumerate(zip(_EQUITY_DATES, _EQUITY_POINTS, strict=True), start=1):
        record_equity(sd, ts, eq, i)
    # 4 cycles of target_weights + reviewer artifacts.
    # both-sides deployment: cycles 1,2,4 both-deployed, cycle 3 long-only -> 3/4 = 0.75.
    # neutrality adherence: cycles 1,2,3 in band, cycle 4 out of band (beta 0.09 > 0.05) -> 3/4.
    # reviewer veto: cycle 3 vetoed -> 1/4 = 0.25.
    cfgs = [
        dict(long_frac=0.95, short_frac=0.93, dollar_residual_frac=0.01, beta_residual=0.01),
        dict(long_frac=0.94, short_frac=0.92, dollar_residual_frac=0.02, beta_residual=0.02),
        dict(long_frac=0.95, short_frac=0.10, dollar_residual_frac=0.01, beta_residual=0.01),
        dict(long_frac=0.93, short_frac=0.94, dollar_residual_frac=0.02, beta_residual=0.09),
    ]
    passed = [True, True, False, True]
    for i, (cfg, ok) in enumerate(zip(cfgs, passed, strict=True), start=1):
        save_output(sd, i, "target_weights", _target_weights(**cfg), cadence="weekly")
        save_output(sd, i, "reviewer", _reviewer(cycle=i, passed=ok), cadence="weekly")
    return sd


@pytest.fixture
def memory_dir(tmp_path):
    """A `memory/` seeded with journal outcomes for the carry / pair / sentiment KPIs."""
    md = tmp_path / "memory"
    # carry: realized 12 / projected 10 -> 1.2
    _seed_journal_leg(md, cycle=1, symbol="BTC/USDT:USDT", direction="short",
                      outcome={"realized_funding": 7.0, "projected_funding": 6.0})
    _seed_journal_leg(md, cycle=2, symbol="ETH/USDT:USDT", direction="short",
                      outcome={"realized_funding": 5.0, "projected_funding": 4.0})
    # pair survival: 2 retested, both cointegrated -> 1.0
    _seed_journal_leg(md, cycle=1, symbol="SOL/USDT:USDT", direction="long",
                      outcome={"adf_pvalue_at_retest": 0.01})
    _seed_journal_leg(md, cycle=2, symbol="XRP/USDT:USDT", direction="long",
                      outcome={"adf_pvalue_at_retest": 0.03})
    # sentiment: 1 non-neutral leg, correct -> 1.0
    _seed_journal_leg(md, cycle=3, symbol="ADA/USDT:USDT", direction="long",
                      outcome={"sentiment_score": 0.7, "alpha_return": 0.02})
    return md


def test_dashboard_daily_sharpe_uses_365(state_dir, memory_dir, seeded_daily_returns):
    d = build_kpi_dashboard(state_dir, memory_dir)
    assert d["daily_sharpe"] == pytest.approx(sharpe(seeded_daily_returns, periods_per_year=365))
    assert 0.0 <= d["no_losing_month"] <= 1.0
    assert "reviewer_veto_rate" in d


def test_dashboard_no_losing_month_fraction(state_dir, memory_dir):
    # Jan is a winning month (20000 -> 20600), Feb a losing month (20600 -> 20100). 1/2 positive.
    d = build_kpi_dashboard(state_dir, memory_dir)
    assert d["no_losing_month"] == pytest.approx(0.5)


def test_dashboard_max_drawdown_matches_metrics(state_dir, memory_dir):
    d = build_kpi_dashboard(state_dir, memory_dir)
    assert d["max_drawdown"] == pytest.approx(max_drawdown(_EQUITY_POINTS))


def test_dashboard_reuses_improvement_process_kpis(state_dir, memory_dir):
    d = build_kpi_dashboard(state_dir, memory_dir)
    assert d["both_sides_deployment_rate"] == pytest.approx(0.75)
    assert d["reviewer_veto_rate"] == pytest.approx(0.25)
    assert d["carry_capture"] == pytest.approx(1.2)
    assert d["pair_survival"] == pytest.approx(1.0)
    assert d["sentiment_hit_rate"] == pytest.approx(1.0)


def test_dashboard_neutrality_adherence_counts_in_band_cycles(state_dir, memory_dir):
    # cycles 1,2,3 within bands; cycle 4 has |beta_residual|=0.09 > beta_band 0.05 -> 3/4 = 0.75.
    d = build_kpi_dashboard(state_dir, memory_dir)
    assert d["neutrality_adherence"] == pytest.approx(0.75)


def test_dashboard_has_all_kpi_keys(state_dir, memory_dir):
    d = build_kpi_dashboard(state_dir, memory_dir)
    for key in (
        "no_losing_month",
        "daily_sharpe",
        "max_drawdown",
        "both_sides_deployment_rate",
        "neutrality_adherence",
        "pair_survival",
        "carry_capture",
        "sentiment_hit_rate",
        "reviewer_veto_rate",
    ):
        assert key in d


def test_dashboard_empty_state_is_fail_safe(tmp_path):
    # A cold state/memory must not raise: KPIs collapse to safe sentinels (0.0 / nan), never /0.
    d = build_kpi_dashboard(tmp_path / "state", tmp_path / "memory")
    assert d["daily_sharpe"] == 0.0
    assert d["max_drawdown"] == 0.0
    assert d["no_losing_month"] == 0.0
    assert math.isnan(d["carry_capture"])


def test_dashboard_carries_cost_rows(tmp_path):
    state = tmp_path / "state"
    memory = tmp_path / "memory"
    save_output(state, 1, "pnl", {
        "net_pnl": 8.0, "gross_pnl": 14.0, "fees_paid": 4.0, "slippage_paid": 2.0,
        "funding_net": 6.0, "cycle": 1, "ts": "2026-06-10T00:00:00+00:00"}, cadence="weekly")
    dash = build_kpi_dashboard(state, memory)
    assert dash["net_pnl"] == 8.0
    assert dash["gross_pnl"] == 14.0
    assert dash["total_fees"] == 4.0
    assert dash["total_slippage"] == 2.0
    assert dash["total_funding"] == 6.0
    assert abs(dash["cost_drag_bps"] - (6.0 / 14.0 * 1e4)) < 1e-6

from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import Pair, Spread
from futures_fund.sleeves.pairs import pairs_signal, select_pairs

_NOW = datetime(2026, 6, 11, tzinfo=UTC)


def _pair(pid: str, adj: float | None, *, cointegrated: bool = True) -> Pair:
    return Pair(
        pair_id=pid, symbol_y="BTC/USDT:USDT", symbol_x="ETH/USDT:USDT",
        hedge_ratio=2.0, method="engle_granger", adf_pvalue=0.01, adf_pvalue_adj=adj,
        half_life=5.0, theta=0.139, mu=0.0, sigma_eq=10.0, formed_cycle=1,
        cointegrated=cointegrated,
    )


def test_select_pairs_keeps_fdr_passing_cointegrated():
    kept = select_pairs([
        _pair("p1", 0.01),                    # passes FDR + cointegrated -> keep
        _pair("p2", 0.20),                    # fails FDR -> drop
        _pair("p3", 0.01, cointegrated=False),  # FDR ok but rolling re-test failed -> drop
        _pair("p4", None),                    # no adjusted p yet -> drop (conservative)
    ], adf_pvalue_max=0.05)
    assert [p.pair_id for p in kept] == ["p1"]


def _spread(pid: str, state: str, z: float) -> Spread:
    return Spread(pair_id=pid, spread_value=0.0, zscore=z, state=state)


def test_pairs_signal_short_spread_legs():
    # short_spread means: short y, long x (hedge_ratio units of x per unit of y).
    pair = _pair("p1", 0.01)
    sig = pairs_signal([pair], [_spread("p1", "short_spread", 2.5)],
                       risk_budget_frac=0.25, now=_NOW)
    assert sig.sleeve == "pairs"
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["BTC/USDT:USDT"].direction == "short"   # y leg
    assert by_sym["ETH/USDT:USDT"].direction == "long"    # x leg
    # both legs carry the pair_id (so attribution is at the pair level)
    assert all(t.pair_id == "p1" for t in sig.tilts)
    # x leg weight is hedge_ratio (2.0) times the y leg magnitude -> spread is the traded unit
    assert abs(by_sym["ETH/USDT:USDT"].target_weight) == 2.0 * abs(
        by_sym["BTC/USDT:USDT"].target_weight)


def test_pairs_signal_long_spread_flips_legs():
    pair = _pair("p1", 0.01)
    sig = pairs_signal([pair], [_spread("p1", "long_spread", -2.5)],
                       risk_budget_frac=0.25, now=_NOW)
    by_sym = {t.symbol: t for t in sig.tilts}
    assert by_sym["BTC/USDT:USDT"].direction == "long"    # long the spread -> long y
    assert by_sym["ETH/USDT:USDT"].direction == "short"   # short hedge x


def test_pairs_signal_flat_and_stop_emit_no_legs():
    pair = _pair("p1", 0.01)
    sig = pairs_signal([pair],
                       [_spread("p1", "flat", 0.0), _spread("p1", "stop", 3.5)],
                       risk_budget_frac=0.25, now=_NOW)
    assert sig.tilts == []

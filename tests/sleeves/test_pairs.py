from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.contracts import Pair
from futures_fund.sleeves.pairs import select_pairs

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

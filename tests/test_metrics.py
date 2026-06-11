import math

import pytest

from futures_fund.metrics import (
    PERIODS_PER_YEAR_DAILY,
    PERIODS_PER_YEAR_WEEKLY,
    calmar,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
)


def test_periodicity_constants_are_365_daily_52_weekly():
    assert PERIODS_PER_YEAR_DAILY == pytest.approx(365.0)
    assert PERIODS_PER_YEAR_WEEKLY == pytest.approx(52.0)


def test_sharpe_default_annualizes_daily_x365():
    rets = [0.01, -0.005, 0.012, 0.003, -0.002, 0.008]
    import numpy as np
    arr = np.asarray(rets)
    expected = arr.mean() / arr.std(ddof=1) * math.sqrt(365.0)
    assert sharpe(rets) == pytest.approx(expected)


def test_sharpe_weekly_annualizes_x52():
    rets = [0.02, -0.01, 0.015, 0.005]
    import numpy as np
    arr = np.asarray(rets)
    expected = arr.mean() / arr.std(ddof=1) * math.sqrt(52.0)
    assert sharpe(rets, periods_per_year=PERIODS_PER_YEAR_WEEKLY) == pytest.approx(expected)


def test_sharpe_too_few_returns_is_zero():
    assert sharpe([0.01]) == 0.0


def test_sortino_uses_downside_rms_x365():
    rets = [0.01, -0.02, 0.01, -0.01]
    import numpy as np
    arr = np.asarray(rets)
    dd = math.sqrt(np.mean(np.minimum(arr, 0.0) ** 2))
    expected = arr.mean() / dd * math.sqrt(365.0)
    assert sortino(rets) == pytest.approx(expected)


def test_max_drawdown_peak_to_trough():
    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx((120.0 - 90.0) / 120.0)


def test_calmar_and_hit_rate_and_profit_factor():
    assert calmar(0.20, 0.05) == pytest.approx(4.0)
    assert hit_rate([{"realized_pnl": 1.0}, {"realized_pnl": -1.0}]) == pytest.approx(0.5)
    assert profit_factor([{"realized_pnl": 2.0}, {"realized_pnl": -1.0}]) == pytest.approx(2.0)

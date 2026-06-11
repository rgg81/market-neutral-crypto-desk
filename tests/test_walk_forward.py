from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from futures_fund.walk_forward import (
    load_pit_returns,
    validate,
    validate_sleeve_param,
    walk_forward_splits,
)


def test_walk_forward_splits_anchored_expanding():
    # 100 points, 4 folds: each fold trains on a growing prefix, tests on the next chunk.
    splits = walk_forward_splits(100, n_splits=4, min_train=20)
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert train_idx.stop <= test_idx.start          # no overlap: train strictly before test
        assert train_idx.start == 0                       # anchored (expanding window)
    # test chunks are contiguous and cover the tail
    assert splits[0][1].start >= 20
    assert splits[-1][1].stop == 100


def test_walk_forward_splits_too_short_returns_empty():
    assert walk_forward_splits(10, n_splits=4, min_train=20) == []


def test_validate_sleeve_param_genuine_edge_passes():
    rng = np.random.default_rng(0)
    # strong positive-mean OOS returns across folds -> should clear the DSR gate
    oos_returns = [list(rng.normal(0.02, 0.01, 40)) for _ in range(4)]
    res = validate_sleeve_param(oos_returns, num_trials=4, periods_per_year=365.0,
                                dsr_threshold=0.95)
    assert res["passed"] is True
    assert res["oos_sharpe"] > 0
    assert res["dsr_pvalue"] >= 0.95


def test_validate_sleeve_param_num_trials_deflates_pvalue_and_flips_pass():
    # Regression-lock the gate's core purpose: deflation for multiple testing.
    # In a REALISTIC Sharpe regime (per-period SR ~0.05, n=1000), the OOS Sharpe is the
    # same regardless of num_trials, but the Deflated-Sharpe p-value must SHRINK as more
    # param candidates are tried -- enough to flip passed True->False. (A genuinely strong
    # edge saturates dsr_p to 1.0, hiding this, which is why test_genuine_edge_passes alone
    # cannot catch a num_trials-ignoring implementation.)
    rng = np.random.default_rng(0)
    oos_returns = [list(rng.normal(0.001, 0.01, 250)) for _ in range(4)]  # 1000 pooled obs

    res_1 = validate_sleeve_param(oos_returns, num_trials=1, periods_per_year=365.0,
                                  dsr_threshold=0.95)
    res_20 = validate_sleeve_param(oos_returns, num_trials=20, periods_per_year=365.0,
                                   dsr_threshold=0.95)
    res_1000 = validate_sleeve_param(oos_returns, num_trials=1000, periods_per_year=365.0,
                                     dsr_threshold=0.95)

    # OOS Sharpe is a property of the returns, not of how many trials were tried.
    assert res_1["oos_sharpe"] > 0
    assert res_1["oos_sharpe"] == res_20["oos_sharpe"] == res_1000["oos_sharpe"]

    # Deflation: more trials -> strictly lower DSR p-value (this is the whole point of the gate).
    assert res_1["dsr_pvalue"] > res_20["dsr_pvalue"] > res_1000["dsr_pvalue"]

    # And it actually moves the gate decision: a single trial clears it, many trials do not.
    assert res_1["passed"] is True
    assert res_1000["passed"] is False


def test_validate_sleeve_param_weekly_path_threads_periods_per_year():
    # Walk-forward WEEKLY path: periods_per_year=52 must be threaded into the OOS Sharpe
    # annualization, so the SAME OOS returns yield a DIFFERENT (smaller) annualized Sharpe under
    # 52 than under 365 -- sqrt(52)/sqrt(365) ~ 0.377x. Independent oracle: the 52 Sharpe equals
    # the 365 Sharpe scaled by sqrt(52/365), computed here WITHOUT re-calling validate_sleeve_param.
    rng = np.random.default_rng(0)
    oos_returns = [list(rng.normal(0.01, 0.01, 40)) for _ in range(4)]  # same returns both calls

    res_365 = validate_sleeve_param(oos_returns, num_trials=4, periods_per_year=365.0,
                                    dsr_threshold=0.95)
    res_52 = validate_sleeve_param(oos_returns, num_trials=4, periods_per_year=52.0,
                                   dsr_threshold=0.95)

    # periods_per_year IS threaded into the Sharpe annualization: weekly differs from daily.
    assert res_52["oos_sharpe"] != res_365["oos_sharpe"]
    assert res_52["oos_sharpe"] == pytest.approx(
        res_365["oos_sharpe"] * np.sqrt(52.0 / 365.0)
    )
    assert res_52["oos_sharpe"] < res_365["oos_sharpe"]   # weekly annualizes a smaller multiplier
    # the weekly path still returns the full result dict shape with a DSR p-value present
    assert 0.0 <= res_52["dsr_pvalue"] <= 1.0
    assert res_52["n_obs"] == res_365["n_obs"] == 160


def test_validate_sleeve_param_noise_fails():
    rng = np.random.default_rng(1)
    # zero-mean noise -> no edge -> gate rejects
    oos_returns = [list(rng.normal(0.0, 0.02, 40)) for _ in range(4)]
    res = validate_sleeve_param(oos_returns, num_trials=20, periods_per_year=365.0,
                                dsr_threshold=0.95)
    assert res["passed"] is False


def test_validate_sleeve_param_empty_fails():
    res = validate_sleeve_param([], num_trials=4, periods_per_year=365.0)
    assert res["passed"] is False
    assert res["oos_sharpe"] == 0.0


# --- Task 7.2: point-in-time walk-forward harness -------------------------------------------

def _binance_vision_klines_path(root, symbol: str, day: str, interval: str = "1d"):
    """data.binance.vision USDM-futures daily-klines archive layout for `symbol` on `day`."""
    sym = symbol.split("/")[0] + "USDT"  # "NEWCOIN/USDT:USDT" -> "NEWCOINUSDT"
    d = root / "data" / "futures" / "um" / "daily" / "klines" / sym / interval
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sym}-{interval}-{day}.csv"


def _write_kline_row(path, open_time_ms: int, close_price: float) -> None:
    # Binance kline CSV columns: open_time,open,high,low,close,volume,close_time,...
    o = close_price
    row = f"{open_time_ms},{o},{o},{o},{close_price},1.0,{open_time_ms + 86_400_000},0,0,0,0,0\n"
    with open(path, "a") as fh:
        fh.write(row)


@pytest.fixture
def archive_fixture(tmp_path):
    """A data.binance.vision-shaped offline archive (no network).

    BTCUSDT is listed from 2024-12-01 (BEFORE the IS window start 2025-01-01) -> usable.
    NEWCOINUSDT's FIRST archive day is 2025-02-01 (AFTER start) -> survivorship-excluded.
    """
    root = tmp_path / "binance-vision"
    base_ms = int(datetime(2024, 12, 1, tzinfo=UTC).timestamp() * 1000)
    # BTC: 92 daily candles from 2024-12-01 (well before the window) with a gentle drift.
    for i in range(92):
        day = (datetime(2024, 12, 1, tzinfo=UTC) + timedelta(days=i)).strftime("%Y-%m-%d")
        p = _binance_vision_klines_path(root, "BTC/USDT:USDT", day)
        _write_kline_row(p, base_ms + i * 86_400_000, 100.0 * (1.001 ** i))
    # NEWCOIN: first archive day 2025-02-01 (AFTER the IS window start 2025-01-01).
    nb_ms = int(datetime(2025, 2, 1, tzinfo=UTC).timestamp() * 1000)
    for i in range(28):
        day = (datetime(2025, 2, 1, tzinfo=UTC) + timedelta(days=i)).strftime("%Y-%m-%d")
        p = _binance_vision_klines_path(root, "NEWCOIN/USDT:USDT", day)
        _write_kline_row(p, nb_ms + i * 86_400_000, 50.0 * (1.001 ** i))
    return root


def test_walk_forward_inputs_are_point_in_time(archive_fixture):
    # symbol "NEWCOIN" first archive date is AFTER the IS window start -> excluded (no look-ahead)
    rets = load_pit_returns("NEWCOIN/USDT:USDT", start="2025-01-01", end="2025-03-01",
                            archive_root=archive_fixture)
    assert rets is None  # survivorship caveat: later-listed name excluded from the window


def test_load_pit_returns_listed_before_start_is_usable(archive_fixture):
    pit = load_pit_returns("BTC/USDT:USDT", start="2025-01-01", end="2025-03-01",
                           archive_root=archive_fixture)
    assert pit is not None
    # PIT provenance tag: the returns carry their archive source date so the test can assert it.
    assert pit["archive_source"] == "data.binance.vision"
    assert pit["first_archive_date"] == "2024-12-01"  # listed before the window start
    assert pit["first_archive_date"] < "2025-01-01"
    # returns are confined to the [start, end) window and are non-empty
    rets = pit["returns"]
    assert isinstance(rets, list) and len(rets) > 10
    assert all(isinstance(r, float) for r in rets)


def test_validate_rejects_in_sample_only_winner():
    # Two params. "lucky" wins IN-SAMPLE on a single seed-specific spike but has NO OOS edge;
    # "robust" has a genuine positive drift that persists out-of-sample.
    rng = np.random.default_rng(7)
    n = 240
    robust = list(rng.normal(0.02, 0.01, n))            # persistent edge -> survives OOS
    # "lucky" = pure noise except a huge in-sample-only spike near the front (train region).
    lucky = list(rng.normal(0.0, 0.01, n))
    for j in range(5):
        lucky[j] += 0.5                                  # giant IS spike, nothing OOS

    grid = ["robust", "lucky"]
    returns_by_param = {"robust": robust, "lucky": lucky}
    res = validate(grid, returns_by_param, periods_per_year=365.0)

    # ranks by OOS, not IS: the in-sample-only winner is never promoted
    assert res["winner"] == "robust"
    assert res["verdict"] == "promote"
    assert res["num_trials"] == len(grid)               # DSR deflates for the whole grid


def test_validate_rejects_when_no_param_has_oos_edge():
    rng = np.random.default_rng(11)
    grid = ["a", "b", "c"]
    returns_by_param = {p: list(rng.normal(0.0, 0.02, 240)) for p in grid}  # all noise
    res = validate(grid, returns_by_param, periods_per_year=365.0)
    assert res["verdict"] == "reject"
    assert res["num_trials"] == 3

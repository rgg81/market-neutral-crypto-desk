"""Walk-forward validation harness hook for sleeve params/thresholds (design spec §12, §15).

Anchored (expanding-window) out-of-sample splits + a DSR/overfit gate over the vendored
overfit_detector. A sleeve param change is only trusted if it clears the OOS Deflated-Sharpe
threshold, not an in-sample grid win.

Point-in-time provenance (spec §11, §15 — binding): `load_pit_returns` sources historical klines
from an immutable `data.binance.vision` daily-klines archive (offline fixture in tests), NOT a live
`exchange.py` pull that would leak post-decision revisions. A survivorship guard excludes any symbol
whose FIRST archive date is after the in-sample window start (no look-ahead into later-listed
names).
`validate` ranks a param grid by OOS Sharpe (anchored splits) and gates promotion on the Deflated
Sharpe p-value deflated for `num_trials = len(grid)`, so an in-sample-only grid winner is rejected.
"""
from __future__ import annotations

import csv
from pathlib import Path

from futures_fund.graduation import deflated_sharpe_pvalue
from futures_fund.metrics import sharpe

#: provenance tag attached to every point-in-time return stream (spec §11, §15).
ARCHIVE_SOURCE = "data.binance.vision"


def walk_forward_splits(n_obs: int, *, n_splits: int = 4,
                        min_train: int = 20) -> list[tuple[range, range]]:
    """Anchored (expanding-window) walk-forward splits over a length-`n_obs` series.

    Returns a list of (train_range, test_range): each train range is a prefix [0, t) growing fold
    by fold, and the test range is the next contiguous OOS chunk. Empty list if n_obs is too short
    to leave min_train training points plus at least one test point per split.
    """
    if n_obs < min_train + n_splits:
        return []
    test_total = n_obs - min_train
    chunk = test_total // n_splits
    if chunk < 1:
        return []
    splits: list[tuple[range, range]] = []
    for k in range(n_splits):
        train_stop = min_train + k * chunk
        test_start = train_stop
        test_stop = n_obs if k == n_splits - 1 else train_stop + chunk
        splits.append((range(0, train_stop), range(test_start, test_stop)))
    return splits


def validate_sleeve_param(oos_returns: list[list[float]], *, num_trials: int,
                          periods_per_year: float = 365.0,
                          dsr_threshold: float = 0.95) -> dict:
    """Gate a sleeve param/threshold change on out-of-sample evidence.

    `oos_returns` is one return stream per walk-forward fold. Pools the folds, computes the OOS
    Sharpe (annualized at `periods_per_year` — 365 daily / 52 weekly) and the Deflated-Sharpe
    p-value deflated for `num_trials` (the number of param candidates tried). passed iff the OOS
    Sharpe is > 0 AND the DSR p-value clears `dsr_threshold`.
    """
    pooled: list[float] = [r for fold in oos_returns for r in fold]
    if len(pooled) < 10:
        return {"passed": False, "oos_sharpe": 0.0, "dsr_pvalue": 0.0, "n_obs": len(pooled)}
    oos_sharpe = sharpe(pooled, periods_per_year=periods_per_year)
    dsr_p = deflated_sharpe_pvalue(pooled, num_trials=num_trials,
                                   periods_per_year=periods_per_year)
    passed = oos_sharpe > 0 and dsr_p >= dsr_threshold
    return {"passed": passed, "oos_sharpe": oos_sharpe, "dsr_pvalue": dsr_p, "n_obs": len(pooled)}


def _archive_symbol(symbol: str) -> str:
    """`"NEWCOIN/USDT:USDT"` -> `"NEWCOINUSDT"` (the data.binance.vision dir/file stem)."""
    base = symbol.split("/")[0]
    return f"{base}USDT"


def load_pit_returns(symbol: str, start: str, end: str, *, archive_root,
                     interval: str = "1d") -> dict | None:
    """Load point-in-time daily returns for `symbol` over `[start, end)` from a
    `data.binance.vision` USDM-futures daily-klines archive (offline fixture in tests).

    Archive layout (immutable daily dumps, no live revisions leaking post-decision):
        <root>/data/futures/um/daily/klines/<SYMBOL>/<interval>/<SYMBOL>-<interval>-<YYYY-MM-DD>.csv

    SURVIVORSHIP GUARD (spec §11, §15 — binding): returns ``None`` when the symbol's FIRST archive
    date is AFTER `start` — a later-listed name must not leak into a window that predates its
    listing (`exchangeInfo` would only show currently-listed symbols). Otherwise returns a dict
    tagged with its PIT provenance::

        {"returns": [...], "archive_source": "data.binance.vision",
         "first_archive_date": "YYYY-MM-DD", "symbol": symbol}

    SIMPLIFICATION (stated per the roadmap): in CI no live PIT archive is reachable, so the test
    drives a recorded `data.binance.vision`-shaped fixture; the on-disk layout and the survivorship
    rule are faithful, the candle bodies are synthetic.
    """
    root = Path(archive_root)
    sym = _archive_symbol(symbol)
    kdir = root / "data" / "futures" / "um" / "daily" / "klines" / sym / interval
    if not kdir.is_dir():
        return None
    # Daily files are named <SYMBOL>-<interval>-<YYYY-MM-DD>.csv -> date is the trailing 3 tokens.
    prefix = f"{sym}-{interval}-"
    dates = sorted(f.stem[len(prefix):] for f in kdir.glob(f"{prefix}*.csv"))
    if not dates:
        return None
    first_archive_date = dates[0]
    # Survivorship guard: first listing AFTER the window start -> no look-ahead, exclude it.
    if first_archive_date > start:
        return None
    # Gather close prices for in-window days (start inclusive, end exclusive) in date order.
    closes: list[float] = []
    for day in dates:
        if day < start or day >= end:
            continue
        path = kdir / f"{sym}-{interval}-{day}.csv"
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                closes.append(float(row[4]))  # Binance kline col 4 = close
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    return {
        "returns": [float(r) for r in returns],
        "archive_source": ARCHIVE_SOURCE,
        "first_archive_date": first_archive_date,
        "symbol": symbol,
    }


def validate(param_grid: list, returns_by_param: dict, *,
             periods_per_year: float = 365.0, n_splits: int = 4, min_train: int = 20,
             dsr_threshold: float = 0.95) -> dict:
    """Walk-forward-validate a sleeve param grid out-of-sample, refusing an in-sample-only winner.

    For each param, build anchored (expanding-window) IS/OOS splits over its return stream
    (`walk_forward_splits`), pool the OOS folds, and score the param by its OOS Sharpe
    (annualized at `periods_per_year`). The grid is RANKED BY OOS — never by an in-sample fit — so
    a param that wins only on the (train-region) in-sample slice is never selected. The OOS winner
    is then gated on the Deflated-Sharpe p-value deflated for ``num_trials = len(param_grid)``
    (multiple-testing correction over the whole grid).

    Returns ``{"verdict": "promote"|"reject", "winner": <param>|None, "num_trials": len(grid),
    "oos_sharpe": float, "dsr_pvalue": float, "ranking": [(param, oos_sharpe), ...]}``. The verdict
    is ``promote`` iff the OOS winner's Sharpe > 0 AND its DSR p-value clears `dsr_threshold`.
    """
    num_trials = len(param_grid)
    scored: list[tuple] = []  # (param, oos_sharpe, pooled_oos_returns)
    for param in param_grid:
        stream = list(returns_by_param.get(param, []))
        splits = walk_forward_splits(len(stream), n_splits=n_splits, min_train=min_train)
        oos: list[float] = []
        for _train_idx, test_idx in splits:
            oos.extend(stream[test_idx.start:test_idx.stop])
        oos_sharpe = sharpe(oos, periods_per_year=periods_per_year) if oos else 0.0
        scored.append((param, oos_sharpe, oos))
    scored.sort(key=lambda t: t[1], reverse=True)  # rank by OOS Sharpe, best first
    ranking = [(p, s) for p, s, _ in scored]
    if not scored:
        return {"verdict": "reject", "winner": None, "num_trials": num_trials,
                "oos_sharpe": 0.0, "dsr_pvalue": 0.0, "ranking": []}
    winner, win_sharpe, win_oos = scored[0]
    dsr_p = (deflated_sharpe_pvalue(win_oos, num_trials=num_trials,
                                    periods_per_year=periods_per_year)
             if len(win_oos) >= 10 else 0.0)
    promote = win_sharpe > 0 and dsr_p >= dsr_threshold
    return {
        "verdict": "promote" if promote else "reject",
        "winner": winner if promote else None,
        "num_trials": num_trials,
        "oos_sharpe": win_sharpe,
        "dsr_pvalue": dsr_p,
        "ranking": ranking,
    }

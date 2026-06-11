"""Writer for the light book `monitor_cli.py` evaluates (`state/monitor_book.json`).

The between-cycle monitor (drawdown / liq-distance / neutrality tripwire) reads a self-contained
book artifact but nothing produced it. This module shapes a `TargetWeights` book + live marks/liq
prices into the `{balance, peak_equity, legs:[{symbol, mark, liq_price, notional, beta_btc}]}` rows
`monitor_cli.check_positions` / `check_neutrality` consume, so the monitor's neutrality guard is
live between cycles. Atomic write (tmp + os.replace), mirroring `cycle_io.save_output`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from futures_fund.contracts import TargetWeights


def write_monitor_book(
    state_dir,
    book: TargetWeights,
    *,
    marks: dict[str, float],
    liq_prices: dict[str, float],
    balance: float,
    peak_equity: float,
) -> Path:
    """Persist the light monitor book from a `TargetWeights` + live marks/liq prices.

    Each non-flat leg becomes a row `{symbol, mark, liq_price, notional, beta_btc}`. `notional` is
    the leg's DIRECTION-SIGNED notional (long => +|target_notional|, short => -|target_notional|),
    matching the convention `monitor_cli.check_neutrality` consumes (the light book carries no
    `direction` field, so the sign must be baked in for the dollar/beta residuals to net out). A
    leg with no mark is skipped (the monitor can't guard a leg whose liq-distance it can't compute).
    """
    legs: list[dict] = []
    for leg in book.legs:
        if abs(leg.target_notional) <= 0.0 or leg.symbol not in marks:
            continue
        mag = abs(float(leg.target_notional))
        signed_notional = mag if leg.direction == "long" else -mag
        legs.append({
            "symbol": leg.symbol,
            "mark": float(marks[leg.symbol]),
            "liq_price": float(liq_prices[leg.symbol]) if leg.symbol in liq_prices else None,
            "notional": signed_notional,
            "beta_btc": float(leg.beta_btc),
        })
    payload = {"balance": float(balance), "peak_equity": float(peak_equity), "legs": legs}
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "monitor_book.json"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)
    return p

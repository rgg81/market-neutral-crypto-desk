"""Between-cycle light risk monitor (§9 / §19) — run on a faster cron than the cadence cycles.

    uv run python scripts/monitor_cli.py

Adapts the weekly desk's `monitor_cli.py` + `futures_fund/monitor.py` template (drawdown +
liq-distance) and EXTENDS it with a neutrality-residual trip: the live book is recomputed against
the `NeutralityConfig` dollar/beta bands using `neutrality.dollar_residual` / `beta_residual`. The
monitor trips HALT (calls `set_halt`, notifies, exits non-zero) when ANY of three guards breaches —
(a) drawdown-from-peak >= `Settings.max_drawdown_tolerance`, (b) any leg's liq-distance below the
`2.5×` maintenance buffer, or (c) the live book's dollar/beta residual exceeds its band — and is a
no-op (exit 0) when all three are in band.

PROTECTED-MODULE RULE (cross-phase invariant §15): the monitor only ADDS a trip; it never relaxes a
limit. The neutrality residual is recomputed here (never trusting a stored field) exactly as the
optimizer / reviewer derive it, so the between-cycle guard and the in-cycle guard agree.

The monitor reads a self-contained live book artifact (`state/monitor_book.json`) holding the
account balance/peak plus per-leg `{symbol, mark, liq_price, notional, beta_btc}`. This keeps the
light monitor decoupled from the exchange poll: a between-cycle sweeper writes the latest marks/
positions to that file and this CLI evaluates the guards over it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from futures_fund.config import Settings, load_settings
from futures_fund.neutrality import NeutralityConfig, beta_residual, dollar_residual

_STATE_DIR = "state"

# Liq-distance maintenance buffer (spec §19 "2.5×"): a leg is flagged when its mark sits closer to
# its liquidation price than 2.5× the maintenance band (~2.5% of mark). Tighter than the weekly
# desk's 10% pre-flatten alert because a market-neutral book runs many small legs and any single
# leg approaching liquidation breaks the hedge it was paired into.
LIQ_DISTANCE_MIN = 0.025


def _neutrality_config(settings: Settings) -> NeutralityConfig:
    """Hydrate the P1 `NeutralityConfig` from `settings.neutrality` (defaults when empty)."""
    return NeutralityConfig(**(settings.neutrality or {}))


def check_positions(
    legs: list[dict],
    *,
    equity: float,
    peak_equity: float,
    max_drawdown: float,
    liq_distance_min: float = LIQ_DISTANCE_MIN,
) -> dict:
    """Cheap between-cycle safety check (adapted from the weekly `monitor.check_positions`).

    Signals HALT when EITHER guard trips: (a) drawdown-from-peak >= `max_drawdown` (the contract
    `Settings.max_drawdown_tolerance`), or (b) any leg's mark sits within `liq_distance_min` of its
    liquidation price (a leg approaching liquidation breaks the hedge it was paired into, so it must
    halt — not merely alert). Returns the alert list, the drawdown, and whether either of THESE two
    guards trips (the caller ORs in the neutrality guard).
    """
    alerts: list[str] = []
    liq_breach = False
    for leg in legs:
        mark = leg.get("mark")
        liq = leg.get("liq_price")
        if not mark or mark <= 0 or liq is None:
            continue
        dist = abs(mark - liq) / mark
        if dist <= liq_distance_min:
            liq_breach = True
            alerts.append(
                f"{leg['symbol']} within {dist:.2%} of liquidation"
                f" (mark {mark}, liq {liq})"
            )
    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
    should_halt = drawdown >= max_drawdown or liq_breach
    if should_halt:
        alerts.append(f"drawdown {drawdown:.2%} >= halt tolerance {max_drawdown:.2%}")
    return {"alerts": alerts, "should_halt": should_halt, "drawdown": drawdown}


def check_neutrality(legs: list[dict], *, equity: float, cfg: NeutralityConfig) -> dict:
    """Recompute the live book's dollar/beta residuals and compare to the `NeutralityConfig` bands.

    `dollar_residual_frac = |Sum(long$) - Sum(short$)| / side_budget` (the SAME normalization
    `optimize_book` and the reviewer use). `beta_residual = Sum_i w_i * beta_i` with
    `w_i = notional_i / equity`. Trips when `dollar_residual_frac > dollar_band` OR
    `|beta_residual| > beta_band`. Never trusts a persisted residual — always re-derives.
    """
    notionals = {leg["symbol"]: float(leg["notional"]) for leg in legs}
    weights = {sym: (n / equity if equity > 0 else 0.0) for sym, n in notionals.items()}
    betas = {leg["symbol"]: float(leg.get("beta_btc", 1.0)) for leg in legs}

    d_resid = dollar_residual(weights, notionals)
    d_resid_frac = abs(d_resid) / cfg.side_budget_usdt if cfg.side_budget_usdt > 0 else 0.0
    b_resid = beta_residual(weights, betas)

    dollar_breach = d_resid_frac > cfg.dollar_band
    beta_breach = abs(b_resid) > cfg.beta_band
    alerts: list[str] = []
    if dollar_breach:
        alerts.append(
            f"dollar_residual_frac {d_resid_frac:.2%} > dollar_band {cfg.dollar_band:.2%}"
        )
    if beta_breach:
        alerts.append(f"|beta_residual| {abs(b_resid):.4f} > beta_band {cfg.beta_band:.4f}")
    return {
        "alerts": alerts,
        "should_halt": dollar_breach or beta_breach,
        "dollar_residual": d_resid,
        "dollar_residual_frac": d_resid_frac,
        "beta_residual": b_resid,
    }


def set_halt(state_dir, halt: bool, reason: str = "") -> None:
    """Persist the HALT flag atomically to `state/halt.json` (the deterministic flag the runner
    reads to refuse to act). Adds a trip only — never relaxes a limit (protected-module rule)."""
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "halt.json"
    payload = {
        "halt": halt,
        "reason": reason if halt else "",
        "ts": datetime.now(UTC).isoformat(),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)


def notify(state_dir, message: str, ts: datetime) -> None:
    """Append a notification (the 'notify' half of auto-execute+notify). A real channel can tail
    this file."""
    p = Path(state_dir) / "notifications.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"ts": ts.isoformat(), "message": message}) + "\n")


def _load_book(state_dir: str) -> dict | None:
    """Load the live book the monitor evaluates, or None when no sweeper has written one yet."""
    p = Path(state_dir) / "monitor_book.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Between-cycle light risk monitor (drawdown / liq-distance / neutrality HALT)."
    )
    parser.add_argument("--state-dir", default=_STATE_DIR)
    args = parser.parse_args(argv)

    settings = load_settings()
    cfg = _neutrality_config(settings)
    book = _load_book(args.state_dir)
    if book is None:
        # No live book to evaluate this tick — nothing to guard, no HALT.
        print(json.dumps({"alerts": [], "should_halt": False, "note": "no monitor_book"}))
        return 0

    legs: list[dict] = book.get("legs", [])
    balance = float(book.get("balance", settings.account_size_usdt))
    peak_equity = float(book.get("peak_equity", balance))
    # Equity = cash balance + signed unrealized PnL is not tracked in the light book; the sweeper
    # writes the realized account `balance`, so equity == balance for the drawdown guard here.
    equity = balance

    pos = check_positions(
        legs,
        equity=equity,
        peak_equity=peak_equity,
        max_drawdown=settings.max_drawdown_tolerance,
    )
    neu = check_neutrality(legs, equity=equity, cfg=cfg)

    alerts = pos["alerts"] + neu["alerts"]
    should_halt = pos["should_halt"] or neu["should_halt"]
    out = {
        "alerts": alerts,
        "should_halt": should_halt,
        "drawdown": pos["drawdown"],
        "dollar_residual_frac": neu["dollar_residual_frac"],
        "beta_residual": neu["beta_residual"],
    }

    now = datetime.now(UTC)
    if should_halt:
        set_halt(args.state_dir, True, reason=f"monitor: {alerts}")
        notify(args.state_dir, f"HALT tripped by monitor: {alerts}", now)
    elif alerts:
        notify(args.state_dir, f"risk alerts: {alerts}", now)

    print(json.dumps(out, indent=2, default=str))
    return 1 if should_halt else 0


if __name__ == "__main__":
    sys.exit(main())

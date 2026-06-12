"""Cycle-prep producer CLI (Phase 8): build + persist the four upstream artifacts the control loop
and reviewer consume — geometries / sleeves / pairs / spreads — from (faked or live) exchange reads.

    uv run python scripts/cycle_prep_cli.py --cycle N --cadence weekly
    uv run python scripts/cycle_prep_cli.py --cycle N --cadence weekly --now 2026-06-11T00:00:00Z

Closes C1: before this, only the e2e `_seed_upstream` fixture produced these, so the desk could not
build a book from market data. Reads its symbol set from this cycle's `universe.json` (scout output)
and persists every artifact under the SAME cadence cycle root the loop/reviewer scan (CADENCE-ROOT
INVARIANT). PAPER-ONLY: the exchange is built via `FuturesExchange.from_settings` (faked in tests).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from futures_fund.config import load_settings
from futures_fund.cycle_io import load_output, save_output
from futures_fund.cycle_prep import (
    build_geometries,
    build_pairs_and_spreads,
    build_sleeves,
)
from futures_fund.exchange import FuturesExchange
from futures_fund.models import Cadence

_STATE_DIR = "state"


def _parse_now(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _universe_rows(state_dir, cycle: int, cadence: Cadence, settings) -> list[dict]:
    """This cycle's universe.json rows (scout output); fall back to bare settings.symbols rows."""
    try:
        rows = load_output(state_dir, cycle, "universe", cadence=cadence)["universe"]
        if rows:
            return rows
    except FileNotFoundError:
        pass
    return [{"symbol": s} for s in settings.symbols]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build + persist geometries/sleeves/pairs/spreads.")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default=_STATE_DIR)
    ap.add_argument("--now", default=None)
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence
    now = _parse_now(args.now)

    settings = load_settings()
    ex = FuturesExchange.from_settings(settings)
    rows = _universe_rows(args.state_dir, args.cycle, cadence, settings)
    symbols = [r["symbol"] for r in rows if r.get("symbol")]
    rows_by_sym = {r["symbol"]: r for r in rows if r.get("symbol")}

    bundle = build_geometries(
        ex, symbols, now=now, btc_symbol=settings.beta.btc_symbol,
        beta_lookback=settings.beta.lookback_days, universe_rows=rows_by_sym,
    )
    pairs, spreads = build_pairs_and_spreads(ex, symbols, cycle=args.cycle, now=now)
    sleeves = build_sleeves(bundle.geometries, pairs=pairs, spreads=spreads, now=now)

    save_output(args.state_dir, args.cycle, "geometries", bundle, cadence=cadence)
    save_output(args.state_dir, args.cycle, "sleeves",
                {"sleeves": [s.model_dump(mode="json") for s in sleeves]}, cadence=cadence)
    save_output(args.state_dir, args.cycle, "pairs",
                {"pairs": [p.model_dump(mode="json") for p in pairs]}, cadence=cadence)
    save_output(args.state_dir, args.cycle, "spreads",
                {"spreads": [s.model_dump(mode="json") for s in spreads]}, cadence=cadence)
    print(json.dumps({"cycle": args.cycle, "cadence": cadence, "symbols": len(symbols),
                      "pairs": len(pairs), "spreads": len(spreads)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())

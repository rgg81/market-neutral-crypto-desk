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
    build_returns,
    build_sleeves,
)
from futures_fund.exchange import FuturesExchange
from futures_fund.lesson_overlay import apply_lesson_overlay
from futures_fund.lessons import read_lessons
from futures_fund.models import Cadence
from futures_fund.returns_frame import frame_to_json

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
    ap.add_argument("--memory-dir", default="memory")
    ap.add_argument("--now", default=None)
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence
    now = _parse_now(args.now)

    settings = load_settings()
    ex = FuturesExchange.from_settings(settings)
    rows = _universe_rows(args.state_dir, args.cycle, cadence, settings)
    # ALPHA universe — the scout's tradable top-N (BTC is in here ONLY if the scout selected it).
    alpha_symbols = [r["symbol"] for r in rows if r.get("symbol")]
    rows_by_sym = {r["symbol"]: r for r in rows if r.get("symbol")}

    # The configured BTC hedge/beta symbol is INFRASTRUCTURE, not tradable alpha: `optimize_book`
    # ALWAYS appends a BTC hedge leg (the beta-neutralizing instrument) and beta is measured against
    # BTC. It must therefore ALWAYS get a priced CoinGeometry — even when the scout universe is
    # all-alts and excludes BTC — so the hedge leg's mark reaches `apply_fills` and the HELD book
    # stays market-neutral (else the hedge leg is silently skipped and the book goes net-short). We
    # price BTC for the GEOMETRY build, but keep it OUT of the alpha pair/sleeve selection below.
    btc_symbol = settings.beta.btc_symbol
    geometry_symbols = list(alpha_symbols)
    if btc_symbol not in geometry_symbols:
        geometry_symbols.append(btc_symbol)

    bundle = build_geometries(
        ex, geometry_symbols, now=now, btc_symbol=btc_symbol,
        beta_lookback=settings.beta.lookback_days, universe_rows=rows_by_sym,
    )
    # Sleeves run on the ALPHA geometries ONLY — never force the hedge-only BTC geometry into the
    # carry/factor/sentiment cross-section (not tradable alpha when absent from the universe).
    alpha_set = set(alpha_symbols)
    alpha_geometries = [g for g in bundle.geometries if g.symbol in alpha_set]
    pairs, spreads = build_pairs_and_spreads(ex, alpha_symbols, cycle=args.cycle, now=now)
    # Per-symbol return frame for the optimizer's covariance (HRP shaping + cluster cap). Built over
    # the ALPHA symbols (the cross-section the optimizer shapes); the hedge-only BTC geometry is not
    # part of the alpha covariance. Empty frame (too little history) -> optimizer uses merged split.
    returns = build_returns(ex, alpha_symbols)
    carry_cap = (settings.sleeves.get("carry") or {}).get("max_abs_apr")
    sleeves = build_sleeves(alpha_geometries, pairs=pairs, spreads=spreads, now=now,
                            max_abs_apr=carry_cap)
    # LEARNING link 4 — READ-BACK: tilt the sleeve convictions by the lessons corpus (validated
    # standing rules at full strength, candidates reduced). `optimize_book` re-projects the book
    # the dollar+beta-neutral set and re-applies caps + the deployment floor AFTER this, so the
    # overlay can only re-shape relative conviction within the alpha legs — never break neutrality.
    # No-op until the desk has actually learned something (empty corpus -> sleeves unchanged).
    sleeves = apply_lesson_overlay(sleeves, read_lessons(args.memory_dir))

    save_output(args.state_dir, args.cycle, "geometries", bundle, cadence=cadence)
    save_output(args.state_dir, args.cycle, "sleeves",
                {"sleeves": [s.model_dump(mode="json") for s in sleeves]}, cadence=cadence)
    save_output(args.state_dir, args.cycle, "pairs",
                {"pairs": [p.model_dump(mode="json") for p in pairs]}, cadence=cadence)
    save_output(args.state_dir, args.cycle, "spreads",
                {"spreads": [s.model_dump(mode="json") for s in spreads]}, cadence=cadence)
    save_output(args.state_dir, args.cycle, "returns", frame_to_json(returns), cadence=cadence)
    print(json.dumps({"cycle": args.cycle, "cadence": cadence, "symbols": len(alpha_symbols),
                      "geometries": len(bundle.geometries),
                      "pairs": len(pairs), "spreads": len(spreads)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())

"""Universe Scout CLI (SKILL.md W3): scan the LIVE USD-M perp universe (top by 24h quote volume,
crypto-only) and trim to the liquidity floor -> `universe.json`. Public/keyless.

    uv run python scripts/scout_cli.py --cycle N --cadence weekly --top 30

Closes the I1 gap (scout_cli.py named in SKILL.md but absent). Writes under the cadence-segmented
cycle root (`state/<cadence>/cycle/<N>/`, CADENCE-ROOT INVARIANT) the rest of the ladder reads.
"""
from __future__ import annotations

import argparse
import json
import sys

from futures_fund.config import load_settings
from futures_fund.cycle_io import save_output
from futures_fund.exchange import build_ccxt
from futures_fund.market_data import liquidity_floor, scan_universe
from futures_fund.models import Cadence


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scan + trim the crypto-only perp universe (W3).")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    settings = load_settings()
    client = build_ccxt(settings)
    client.load_markets()
    rows = scan_universe(client, top_n=max(args.top, settings.universe.symbol_count))
    universe = liquidity_floor(
        rows, min_adv_usd=settings.universe.min_adv_usd,
        symbol_count=settings.universe.symbol_count,
    )
    save_output(args.state_dir, args.cycle, "universe", {"universe": universe}, cadence=cadence)
    print(json.dumps({"universe": universe}, indent=2))


if __name__ == "__main__":
    sys.exit(main())

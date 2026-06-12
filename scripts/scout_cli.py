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
from datetime import UTC, datetime

from futures_fund.config import load_settings
from futures_fund.cycle_io import save_output
from futures_fund.exchange import FuturesExchange, build_ccxt
from futures_fund.market_data import quality_filter, scan_universe
from futures_fund.models import Cadence


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scan + quality-filter the crypto-only perp universe.")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "daily"], required=True)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv)
    cadence: Cadence = args.cadence

    settings = load_settings()
    client = build_ccxt(settings)
    client.load_markets()
    exchange = FuturesExchange.from_settings(settings)
    now = datetime.now(UTC)

    rows = scan_universe(client, top_n=max(args.top, settings.universe.symbol_count))
    u = settings.universe
    universe, drops = quality_filter(
        rows, now=now, exchange=exchange,
        min_adv_usd=u.min_adv_usd, min_age_days=u.min_age_days,
        max_abs_chg_24h_pct=u.max_abs_chg_24h_pct, min_depth_usd=u.min_depth_usd,
        depth_ref_usd=u.depth_ref_usd, symbol_count=u.symbol_count,
    )
    save_output(args.state_dir, args.cycle, "universe", {"universe": universe}, cadence=cadence)
    print(json.dumps({
        "scanned": len(rows), "kept": len(universe), "dropped": drops,
        "universe": [r["symbol"] for r in universe],
    }, indent=2))


if __name__ == "__main__":
    sys.exit(main())

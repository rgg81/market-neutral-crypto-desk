"""Walk-forward OOS validation CLI (Phase 7, Task 7.2).

    uv run python scripts/walk_forward_cli.py --grid grid.json
    uv run python scripts/walk_forward_cli.py --grid grid.json --periods-per-year 52

Reads a param grid + per-param return streams from a JSON file shaped as::

    {"param_grid": ["robust", "lucky"],
     "returns_by_param": {"robust": [...], "lucky": [...]}}

and prints the walk-forward verdict (`walk_forward.validate`): the OOS-ranked winner is gated on
the Deflated-Sharpe p-value deflated for `num_trials = len(grid)`, so an in-sample-only grid winner
is REJECTED. Point-in-time inputs (`data.binance.vision` archive + survivorship guard) are produced
upstream by `walk_forward.load_pit_returns`. Pure / read-only — it never mutates desk state.
(spec §12, §15.)
"""

from __future__ import annotations

import argparse
import json

from futures_fund.walk_forward import validate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Walk-forward out-of-sample validation of a sleeve param grid (DSR-gated).")
    ap.add_argument("--grid", required=True,
                    help="path to a JSON file with 'param_grid' + 'returns_by_param'")
    ap.add_argument("--periods-per-year", type=float, default=365.0,
                    help="annualization factor (365 daily / 52 weekly)")
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=20)
    ap.add_argument("--dsr-threshold", type=float, default=0.95)
    args = ap.parse_args(argv)

    with open(args.grid) as fh:
        payload = json.load(fh)
    res = validate(
        payload["param_grid"],
        payload["returns_by_param"],
        periods_per_year=args.periods_per_year,
        n_splits=args.n_splits,
        min_train=args.min_train,
        dsr_threshold=args.dsr_threshold,
    )
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()

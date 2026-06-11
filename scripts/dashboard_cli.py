"""KPI dashboard CLI (Phase 7, Task 7.1).

    uv run python scripts/dashboard_cli.py --state-dir state --memory-dir memory
    uv run python scripts/dashboard_cli.py --format md          # markdown table only
    uv run python scripts/dashboard_cli.py --format json        # JSON only (default: both)

Reads the persisted equity series + cycle artifacts + journal outcomes and prints the desk's
end-of-run KPI dashboard (`dashboard.build_kpi_dashboard`) as JSON and/or a markdown table. Pure /
read-only — it never mutates state. PRIMARY KPI is `no_losing_month`; SECONDARY is the daily Sharpe
(annualized ×365). (spec §18.)
"""

from __future__ import annotations

import argparse
import json
import math

from futures_fund.dashboard import build_kpi_dashboard

# Display order + human labels for the markdown table (primary first, then secondary, then process).
_ROWS: tuple[tuple[str, str], ...] = (
    ("no_losing_month", "No-losing-month (primary, target 1.0)"),
    ("daily_sharpe", "Daily Sharpe (×365)"),
    ("max_drawdown", "Max drawdown"),
    ("both_sides_deployment_rate", "Both-sides deployment rate"),
    ("neutrality_adherence", "Neutrality-residual adherence"),
    ("pair_survival", "Pair-survival rate"),
    ("carry_capture", "Carry-capture rate"),
    ("sentiment_hit_rate", "Sentiment hit-rate"),
    ("reviewer_veto_rate", "Reviewer veto-rate"),
)


def _fmt(v: object) -> str:
    """Render a KPI value for the markdown table; a `nan` sentinel (skipped KPI) shows as 'n/a'."""
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def to_markdown(dash: dict) -> str:
    """A two-column `| KPI | Value |` markdown table in the canonical display order."""
    lines = ["| KPI | Value |", "| --- | --- |"]
    lines += [f"| {label} | {_fmt(dash.get(key))} |" for key, label in _ROWS]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Print the market-neutral desk's KPI dashboard.")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--memory-dir", default="memory")
    ap.add_argument("--last-n", type=int, default=10,
                    help="window (in cycles) for the process KPIs")
    ap.add_argument("--format", choices=["json", "md", "both"], default="both")
    args = ap.parse_args(argv)
    dash = build_kpi_dashboard(args.state_dir, args.memory_dir, last_n=args.last_n)
    if args.format in ("json", "both"):
        print(json.dumps(dash, indent=2, default=str))
    if args.format in ("md", "both"):
        print(to_markdown(dash))


if __name__ == "__main__":
    main()

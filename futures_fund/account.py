"""Phase 9 — paper-trading P&L ledger (Position + PaperAccount).

REUSES the cost primitives — it NEVER re-implements fee/funding/slippage math:
  * costs.trade_fee / costs.count_funding_events
  * funding_intervals.clamp_funding_rate / funding_intervals.realized_funding
  * slippage.estimate_slippage

Funding sign convention (load-bearing): this ledger settles funding via
`funding_intervals.realized_funding`, which is BALANCE-credit perspective (a SHORT with a positive
rate RECEIVES funding -> a POSITIVE cash credit). Do NOT use `costs.project_funding` here (that is
the opposite, cost/paid perspective).

Funding clock (load-bearing): the account carries its OWN `last_funding_ts`, advanced by
`settle_funding`. The equity series is NOT a safe `prev_ts` source — `equity_log.record_equity` keys
only on `cycle`, so weekly cycle 1 and daily cycle 1 collide and the daily point overwrites the
weekly one in a single run.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from futures_fund.models import Direction


class Position(BaseModel):
    symbol: str
    direction: Direction
    qty: float                                   # absolute contract qty (>= 0)
    entry_price: float                           # avg entry (VWAP of accumulated fills)
    opened_ts: datetime
    accrued_funding: float = 0.0                 # signed, + = received, - = paid (this leg's life)
    accrued_fees: float = 0.0                    # >= 0 taker/maker fees charged to this leg
    accrued_slippage: float = 0.0                # >= 0 depth slippage charged to this leg
    realized_pnl: float = 0.0                    # signed price P&L realized on this leg so far


class PaperAccount(BaseModel):
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)
    realized_pnl: float = 0.0
    last_funding_ts: datetime | None = None       # the funding clock (NOT the equity series)
    # cumulative cost totals across the account's life
    fees_paid: float = 0.0                        # >= 0
    slippage_paid: float = 0.0                    # >= 0
    funding_received: float = 0.0                 # >= 0 (sum of positive settlements)
    funding_paid: float = 0.0                     # >= 0 (sum of |negative settlements|)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> PaperAccount:
        return cls.model_validate(data)

    def mark_to_market(self, marks: dict[str, float]) -> dict[str, float]:
        """Unrealized PnL per held symbol (skips symbols with no mark).

        long: qty*(mark-entry) ; short: qty*(entry-mark)."""
        upnl: dict[str, float] = {}
        for sym, pos in self.positions.items():
            mark = marks.get(sym)
            if mark is None:
                continue
            if pos.direction == "long":
                upnl[sym] = pos.qty * (mark - pos.entry_price)
            else:
                upnl[sym] = pos.qty * (pos.entry_price - mark)
        return upnl

    def equity(self, marks: dict[str, float]) -> float:
        """cash + sum unrealized PnL (skips symbols missing a mark)."""
        return self.cash + sum(self.mark_to_market(marks).values())

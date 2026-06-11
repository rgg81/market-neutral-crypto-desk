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

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from futures_fund.costs import trade_fee
from futures_fund.models import Direction
from futures_fund.slippage import estimate_slippage


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


class CostInputs(BaseModel):
    """Per-symbol frictions the paper executor needs but the executed proposal does not carry.

    `depth` is the optional crossing-side book; in paper we leave it None so `estimate_slippage`
    uses the ADV+half-spread fallback (which is NEVER flat 2bps)."""
    adv_usd: float = 0.0
    half_spread_bps: float = 1.0
    depth: list[tuple[float, float]] | None = None
    maker: bool = False                          # paper opens are market -> taker


def _signed_qty(pos: Position | None) -> float:
    """Current signed qty: + for a long, - for a short, 0 if flat."""
    if pos is None:
        return 0.0
    return pos.qty if pos.direction == "long" else -pos.qty


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

    def apply_fills(
        self,
        executed_trades: list[dict],
        marks: dict[str, float],
        costs: dict[str, CostInputs],
        *,
        opened_ts: datetime | None = None,
    ) -> None:
        """RECONCILE each touched symbol to its leg's target_notional (signed by direction).

        Each executed leg's `target_notional` is the optimizer's per-symbol TARGET; this fills only
        `delta = target_signed_qty - current_signed_qty`, so re-sending the identical book is an
        exact no-op (delta 0, 0 frictions). A positive delta opens/increases the SAME side (blending
        entry VWAP); a negative delta reduces/closes/flips (Task 4). Fill at the mark; charge a
        taker/maker fee + depth slippage on the |delta notional| actually traded. qty is derived
        from notional/mark because the executed proposal carries no fill price/qty.

        Convergent across weeks (weekly re-emits the full book -> delta 0 on unchanged legs) and
        correct for daily (each rebalance_trades leg carries that symbol's NEW target_notional)."""
        ts = opened_ts or datetime.now(tz=UTC)
        for trade in executed_trades:
            sym = trade["symbol"]
            direction: Direction = trade["direction"]
            target_notional = abs(float(trade["target_notional"]))
            mark = marks.get(sym)
            if mark is None or mark <= 0:
                continue
            ci = costs.get(sym) or CostInputs()
            sign = 1.0 if direction == "long" else -1.0
            target_signed_qty = sign * (target_notional / mark)
            existing = self.positions.get(sym)
            current_signed_qty = _signed_qty(existing)
            delta_signed_qty = target_signed_qty - current_signed_qty
            if abs(delta_signed_qty) <= 1e-12:
                continue  # already at target -> no-op (re-sent unchanged book)
            delta_notional = abs(delta_signed_qty) * mark
            fee = trade_fee(delta_notional, maker=ci.maker)
            slip = estimate_slippage(
                sym, abs(delta_signed_qty), mark, depth=ci.depth, adv_usd=ci.adv_usd,
                half_spread_bps=ci.half_spread_bps)

            if existing is not None and (delta_signed_qty * sign) < 0:
                # delta opposes the leg's direction by magnitude -> reduce/close/flip (Task 4).
                self._reconcile_opposite(
                    existing, sym, direction, target_signed_qty, mark, fee, slip, ts)
                continue

            # same-side open/increase: fill |delta| at the mark, blend entry VWAP.
            fill_qty = abs(delta_signed_qty)
            self._charge_frictions(sym, fee, slip, existing)
            if existing is None:
                self.positions[sym] = Position(
                    symbol=sym, direction=direction, qty=fill_qty, entry_price=mark,
                    opened_ts=ts, accrued_fees=fee, accrued_slippage=slip)
            else:
                total_qty = existing.qty + fill_qty
                existing.entry_price = (
                    existing.entry_price * existing.qty + mark * fill_qty) / total_qty
                existing.qty = total_qty

    def _charge_frictions(
        self, sym: str, fee: float, slip: float, pos: Position | None
    ) -> None:
        self.cash -= fee + slip
        self.fees_paid += fee
        self.slippage_paid += slip
        if pos is not None:
            pos.accrued_fees += fee
            pos.accrued_slippage += slip

    def _reconcile_opposite(
        self, existing: Position, sym: str, direction: Direction,
        target_signed_qty: float, mark: float, fee: float, slip: float, ts: datetime,
    ) -> None:
        raise NotImplementedError("reduce/close/flip implemented in Task 4")

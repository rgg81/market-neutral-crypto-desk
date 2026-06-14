"""Deterministic self-improvement CAPTURE — journal-at-entry + alpha-outcome-at-close.

Closes the first two links of the learning loop in the DETERMINISTIC run path. The desk's
real-cadence loop runs ``run_paper_cli`` (the deterministic spine), NOT the LLM Reflector, so the
two links that used to live only in the agent/orchestrator flow were never exercised: nothing
called ``journal.append_decision`` at entry, so the journal stayed empty, ``patch_outcome`` always
fail-soft no-op'd, and every ``reflection_input.json`` was ``{winners:[], losers:[], n_closed:0}``.

This module records ground truth ONLY — it never sizes, trades, or relaxes a limit (LLM proposes /
code disposes still holds; here there is no proposal at all, just bookkeeping):

  * ``journal_open_legs``   — one Phase-1 ``Decision`` per currently-held leg, keyed on the leg's
                              OWN open (cycle, cadence), carrying the entry context the alpha
                              attribution needs (entry price, size, ``beta_btc``,
                              ``btc_mark_at_entry``, the signed carry expectation, sentiment).
                              IDEMPOTENT via ``append_decision`` (re-journaling appends nothing).
  * ``close_alpha_outcomes``— for each FULLY-closed leg (``account.drain_closed_legs``), looks up
                              the opening ``Decision`` and computes the six market-neutral ALPHA
                              outcome fields: return NET of BTC-beta and NET of costs, plus the four
                              thesis booleans. A leg with no opening Decision (opened before capture
                              was live) degrades to a cost-only patch (the historical behaviour) so
                              carry capture bookkeeping still flows.

Sign convention (load-bearing): ``account`` ``realized_pnl`` is already the signed P&L of the leg
(long: qty*(exit-entry); short: qty*(entry-exit)). ``entry_notional = size*entry`` is positive, so
``net_return = net_pnl / entry_notional`` is the leg's signed return. BTC's contribution to that
return is ``sign * beta_btc * btc_return`` (sign = +1 long / -1 short); alpha is what is left after
stripping it.
"""
from __future__ import annotations

from futures_fund.journal import _find, append_decision
from futures_fund.models import Direction

# the six alpha-vs-beta outcome fields the Reflector grades on (mirror journal.ALPHA_OUTCOME_FIELDS,
# but we BUILD them here rather than read them).


def carry_expected_sign(direction: Direction, rate: float) -> int:
    """+1 if a leg of ``direction`` is EXPECTED to RECEIVE funding at this rate; -1 pay; 0 flat.

    Funding is paid BY longs TO shorts when the rate is positive, so a SHORT on a positive rate
    receives (a carry credit) and a LONG pays; symmetric for a negative rate. This is the carry
    THESIS sign — at close, ``funding_thesis_matched`` is False only when a leg expected to bank
    carry (+1) but actually paid it (realized_funding < 0)."""
    if rate == 0:
        return 0
    receives = (direction == "short" and rate > 0) or (direction == "long" and rate < 0)
    return 1 if receives else -1


def journal_open_legs(
    memory_dir,
    account,
    *,
    cycle: int,
    cadence: str,
    leg_meta: dict[str, dict],
    marks: dict[str, float],
    funding_by_symbol: dict[str, float],
    btc_symbol: str,
) -> int:
    """Journal a Phase-1 ``Decision`` for every currently-held leg; return how many were processed.

    Keyed on each Position's OWN ``opened_cycle``/``opened_cadence`` (falling back to the current
    cycle/cadence for a position that never recorded them) so a held-over leg lands on the Decision
    that opened it, not the current cycle. ``append_decision`` is idempotent per key, so this is
    safe to call every cycle: brand-new legs get a fresh Decision with the correct entry context;
    existing legs are reused untouched. ``btc_mark_at_entry`` is exact for a leg opened THIS cycle
    (the fill happened at these marks); for a one-time BACKFILL of legs opened before capture was
    live it is the current BTC mark (approximate — best-effort alpha, exact going forward)."""
    btc_mark = marks.get(btc_symbol)
    journaled = 0
    for sym, pos in account.positions.items():
        meta = leg_meta.get(sym, {})
        rate = float(funding_by_symbol.get(sym, 0.0) or 0.0)
        payload = {
            "entry": pos.entry_price,
            "size": pos.qty,
            "beta_btc": meta.get("beta_btc"),
            "sleeve": meta.get("sleeve"),
            "setup": meta.get("sleeve"),
            "pair_id": meta.get("pair_id"),
            "regime": meta.get("regime"),
            "sentiment_score": meta.get("sentiment_score", 0.0),
            "funding_at_entry": rate,
            "carry_expected_sign": carry_expected_sign(pos.direction, rate),
            "btc_mark_at_entry": btc_mark,
            "projected_funding": pos.accrued_funding,
            "rationale": f"{meta.get('sleeve') or 'unknown'} leg (deterministic capture)",
        }
        append_decision(
            memory_dir,
            cycle=pos.opened_cycle or cycle,
            symbol=sym,
            direction=pos.direction,
            payload=payload,
            cadence=pos.opened_cadence or cadence,
        )
        journaled += 1
    return journaled


def close_alpha_outcomes(
    memory_dir,
    closed_legs,
    *,
    marks: dict[str, float],
    btc_symbol: str,
    neutrality_in_band: bool = True,
    cointegrated_by_pair: dict[str, bool] | None = None,
):
    """For each FULLY-closed leg, build the (cycle, cadence, symbol, direction, outcome) patch.

    ``outcome`` always carries the realized cost fields (fees/slippage/realized_funding/
    realized_pnl) — the historical cost-only patch ``improvement.carry_capture_rate`` consumes. When
    the opening ``Decision`` is found it ALSO carries the six ALPHA outcome fields, so the Decision
    becomes "closed" for the Reflector and the reflection winners/losers split is keyed on real,
    beta-stripped, cost-net alpha. A leg whose opening Decision is missing (opened before capture
    was live) degrades to the cost-only patch — never an exception."""
    cointegrated_by_pair = cointegrated_by_pair or {}
    btc_now = marks.get(btc_symbol)
    out: list[tuple[int | None, str | None, str, str, dict]] = []
    for cl in closed_legs:
        cost = {
            "fees": cl.fees,
            "slippage": cl.slippage,
            "realized_funding": cl.realized_funding,
            "realized_pnl": cl.realized_pnl,
        }
        dec = _find(memory_dir, cl.opened_cycle, cl.symbol, cl.direction, cl.opened_cadence)
        if dec is None:
            out.append((cl.opened_cycle, cl.opened_cadence, cl.symbol, cl.direction, cost))
            continue
        entry = dec.get("entry")
        size = dec.get("size")
        beta = float(dec.get("beta_btc") or 0.0)
        btc_entry = dec.get("btc_mark_at_entry")
        sign = 1.0 if cl.direction == "long" else -1.0
        net_pnl = cl.realized_pnl + cl.realized_funding - cl.fees - cl.slippage
        entry_notional = (size * entry) if (size and entry) else None
        net_return = net_pnl / entry_notional if (entry_notional and entry_notional > 0) else 0.0
        btc_return = (
            (btc_now - btc_entry) / btc_entry
            if (btc_entry and btc_now and btc_entry > 0)
            else 0.0
        )
        beta_contribution = sign * beta * btc_return
        alpha_return = net_return - beta_contribution
        carry_expected = int(dec.get("carry_expected_sign", 0) or 0)
        funding_thesis_matched = not (carry_expected > 0 and cl.realized_funding < 0)
        pair_id = dec.get("pair_id")
        pair_cointegrated_at_exit = (
            bool(cointegrated_by_pair.get(pair_id, True)) if pair_id else True
        )
        senti = float(dec.get("sentiment_score") or 0.0)
        senti_aligned = (senti > 0 and cl.direction == "long") or (
            senti < 0 and cl.direction == "short"
        )
        outcome = {
            **cost,
            "alpha_return": alpha_return,
            "beta_contribution": beta_contribution,
            "pair_cointegrated_at_exit": pair_cointegrated_at_exit,
            "funding_thesis_matched": bool(funding_thesis_matched),
            "neutrality_in_band": bool(neutrality_in_band),
            "sentiment_helped": bool(senti_aligned and alpha_return > 0),
        }
        out.append((cl.opened_cycle, cl.opened_cadence, cl.symbol, cl.direction, outcome))
    return out

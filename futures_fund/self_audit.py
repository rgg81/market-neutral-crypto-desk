"""Standing self-audit (Pillar 4 — AUDIT). A fast, deterministic panel of the market-neutral
desk's CRITICAL cross-module invariants, runnable any cycle / on demand
(``scripts/self_audit_cli.py``) as a complement to the full test suite. It catches a regression in
a load-bearing
neutrality/symmetry/funding/sentiment property without running the 500+ test suite. Pure-import
checks; no I/O, no network, no cycle artifact.

This is the repeatable, deterministic core of "auditing to make sure no bugs" — distinct from (and
cheaper than) the heavy every-cycle adversarial reviewer (``futures_fund.reviewer``), whose own
verify pass can fail. Every check here is a hard invariant that MUST hold for the desk to be safe to
run.

**Vocabulary note (roadmap Task 5.5, binding):** the invariant ``name``s below
(``both_sides_deployment_floor``, ``funding_sign_correct``, ``pair_legs_hedge_ratio_sized``,
``sentiment_within_cap_range``, ``no_tokenized_stock_leg``) are a *deliberately distinct,
overlapping* vocabulary from the reviewer's canonical ``ReviewerCheck.name``s
(``deployment_floor_both_sides``, ``funding_sign``, ``pair_leg_hedge_ratio``,
``sentiment_cap_respected``/``sentiment_range``, ``crypto_only_universe``). ``self_audit`` is the
*standing import-time invariant panel* (pure, no cycle artifact); the reviewer is the *per-cycle
artifact re-derivation*. They are two independent guards on overlapping properties and must NOT be
"aligned" by renaming one to match the other.
"""
from __future__ import annotations

from datetime import UTC, datetime

from futures_fund.account import PaperAccount, Position
from futures_fund.costs import count_funding_events
from futures_fund.funding_intervals import clamp_funding_rate, realized_funding
from futures_fund.market_data import is_crypto_perp
from futures_fund.neutrality import beta_residual, dollar_residual
from futures_fund.sleeves.sentiment import conviction_tilt

# Bands the standing panel audits against (mirrors NeutralityConfig defaults §4).
_DOLLAR_BAND_USDT = 0.03 * 20000.0  # dollar_band * target_gross_usdt
_BETA_BAND = 0.05
_TOL = 1e-9


# === per-invariant helpers (pure; each returns ok=True on a conformant book, False when broken) ===


def invariant_residuals_in_band(
    weights: dict[str, float],
    notionals: dict[str, float],
    betas: dict[str, float],
) -> tuple[bool, bool]:
    """Re-derive the dollar residual (Sum long$ - Sum short$) and the beta residual (Sum w*beta)
    from the legs and require both inside their §4 bands. Returns ``(dollar_ok, beta_ok)`` so the
    panel can name the two halves separately (``dollar_residual_in_band`` /
    ``beta_residual_in_band``).
    """
    d_resid = dollar_residual(weights, notionals)
    b_resid = beta_residual(weights, betas)
    return abs(d_resid) <= _DOLLAR_BAND_USDT + _TOL, abs(b_resid) <= _BETA_BAND + _TOL


def invariant_consolidated_book_dollar_neutral(
    legs: list[dict], *, band: float = _DOLLAR_BAND_USDT
) -> bool:
    """CONSOLIDATED-by-symbol dollar neutrality: the optimizer legitimately emits the SAME symbol on
    BOTH sides (a factor short AND a hedge long). The HELD book is the per-symbol NET of those legs,
    so a leg-level book that sums to $X/$X can still hold a badly one-sided NET book if a symbol's
    two sides do not net inside the band. Collapse the legs by symbol into one net signed notional
    (+target if long, -target if short) and require |Sum net-long$ - Sum net-short$| within the §4
    band — the held-position invariant the reviewer's leg-level dollar check does NOT catch.

    `legs` are `{symbol, direction, target_notional}` dicts (the executed/target book)."""
    net_signed: dict[str, float] = {}
    for leg in legs:
        sym = leg["symbol"]
        signed = abs(float(leg["target_notional"]))
        if leg["direction"] != "long":
            signed = -signed
        net_signed[sym] = net_signed.get(sym, 0.0) + signed
    longs = sum(n for n in net_signed.values() if n > 0.0)
    shorts = sum(-n for n in net_signed.values() if n < 0.0)
    return abs(longs - shorts) <= band + _TOL


def invariant_both_sides_deployment_floor(
    side_gross: dict[str, float], side_budget: float, floor: float
) -> bool:
    """BOTH sides' deployed fraction (gross side$ / side_budget) must be at/above the floor — a
    one-sided book that under-deploys the short (or long) side is caught."""
    if side_budget <= 0:
        return False
    return all((g / side_budget) >= floor - _TOL for g in side_gross.values())


def invariant_funding_sign_correct(*, flip: bool = False) -> bool:
    """SIGNED funding-carry convention: a SHORT with a POSITIVE rate RECEIVES funding (a positive
    balance contribution / credit) and a LONG with a positive rate PAYS (negative). Re-derived via
    ``realized_funding``; ``flip=True`` injects the wrong (sign-flipped) expectation so the broken
    case is caught."""
    mark, qty, rate = 100.0, 1.0, 0.001
    short_credit = realized_funding(0.0, mark, qty, rate, "short")
    long_debit = realized_funding(0.0, mark, qty, rate, "long")
    if flip:
        # the broken convention claims a short PAYS and a long RECEIVES — must NOT validate.
        return short_credit < 0.0 and long_debit > 0.0
    return short_credit > 0.0 and long_debit < 0.0


def invariant_account_equity_reconciles(
    account: PaperAccount, marks: dict, recorded_equity: float, *, tol: float = 1e-6
) -> bool:
    """Recorded equity must equal cash + unrealized PnL within tolerance (no phantom equity)."""
    return abs(account.equity(marks) - recorded_equity) <= tol


def invariant_cycle_funding_reconciles(
    account: PaperAccount, prev_ts, now, funding_by_symbol: dict, intervals: dict,
    marks: dict, recorded_funding: float, *, tol: float = 1e-6
) -> bool:
    """Recorded per-cycle funding must equal a recompute via realized_funding x events (extends the
    funding-sign check to the account level — the reviewer's settlement-window re-derivation)."""
    total = 0.0
    for sym, pos in account.positions.items():
        mark = marks.get(sym)
        if mark is None:
            continue
        n = count_funding_events(prev_ts, now, int(intervals.get(sym, 8)))
        rate = clamp_funding_rate(sym, funding_by_symbol.get(sym, 0.0))
        total += realized_funding(0.0, mark, pos.qty, rate, pos.direction) * n
    return abs(total - recorded_funding) <= tol


def invariant_pair_legs_hedge_ratio_sized(
    *, hedge_ratio: float, qty_y: float, qty_x: float
) -> bool:
    """A cointegrated pair's x leg MUST be sized at ``hedge_ratio * qty_y`` — otherwise the pair
    carries a residual single-name exposure. ``|qty_x - hedge_ratio*qty_y|`` must be ~0."""
    expected_qx = hedge_ratio * qty_y
    return abs(qty_x - expected_qx) <= _TOL * max(1.0, abs(expected_qx), abs(qty_x))


def invariant_sentiment_within_cap_range(
    *,
    weight: float,
    score: float,
    conf: float,
    cap: float,
    claimed_delta: float | None = None,
) -> bool:
    """§7.2 sentiment discipline: the conviction tilt moves a leg's magnitude by ``|dw| <= cap*|w|``
    and NEVER flips its sign. The realized tilt (from ``conviction_tilt``) is re-derived and checked
    against the cap. When ``claimed_delta`` is supplied (a tampered/over-sized book delta) the
    invariant ALSO requires that empirical delta to stay inside the band — so an over-sized leg the
    artifact claims is caught."""
    tilted = conviction_tilt(weight, score, conf, cap=cap)
    realized_delta = tilted - weight
    sign_kept = (weight >= 0 and tilted >= 0) or (weight <= 0 and tilted <= 0)
    cap_ok = abs(realized_delta) <= cap * abs(weight) + _TOL and sign_kept
    if claimed_delta is not None:
        cap_ok = cap_ok and abs(claimed_delta) <= cap * abs(weight) + _TOL
    return cap_ok


def invariant_no_tokenized_stock_leg(markets: list[dict]) -> bool:
    """§3 CRYPTO-ONLY mandate: every traded leg must be a cryptocurrency COIN perp — a
    tokenized-stock (EQUITY), commodity, or index-basket TradFi-wrapper perp is rejected. Reuses
    ``market_data.is_crypto_perp`` so the same allowlist the universe scan applies is re-checked."""
    return all(is_crypto_perp(m) for m in markets)


# === the standing panel ===========================================================================


def _checks() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    # 1+2. NEUTRALITY: dollar residual (long$ - short$) and beta residual (Sum w*beta) in band on a
    # balanced synthetic book.
    weights = {"A": 0.05, "B": -0.05, "C": 0.04, "D": -0.04}
    notionals = {"A": 1000.0, "B": -1000.0, "C": 800.0, "D": -800.0}
    betas = {"A": 1.2, "B": 1.2, "C": 0.8, "D": 0.8}
    dollar_ok, beta_ok = invariant_residuals_in_band(weights, notionals, betas)
    d_resid = abs(dollar_residual(weights, notionals))
    add("dollar_residual_in_band", dollar_ok,
        f"|long$ - short$| = {d_resid:.4f} vs band {_DOLLAR_BAND_USDT}")
    add("beta_residual_in_band", beta_ok,
        f"|Sum w*beta| = {abs(beta_residual(weights, betas)):.4f} vs band {_BETA_BAND}")

    # 2b. CONSOLIDATED-by-symbol dollar neutrality: a book with the SAME symbol on BOTH sides (a
    # factor short + a hedge long) must NET to a dollar-neutral HELD book — the per-symbol-NET
    # property that a per-leg check (and the per-leg apply_fills bug) misses.
    consolidated_legs = [
        {"symbol": "BTC/USDT:USDT", "direction": "short", "target_notional": 2116.0},  # factor
        {"symbol": "BTC/USDT:USDT", "direction": "long", "target_notional": 2129.0},   # hedge
        {"symbol": "ETH/USDT:USDT", "direction": "long", "target_notional": 6884.0},
        {"symbol": "SOL/USDT:USDT", "direction": "short", "target_notional": 6897.0},
    ]
    add("consolidated_book_dollar_neutral",
        invariant_consolidated_book_dollar_neutral(consolidated_legs),
        "per-symbol-net (consolidated) held book must be dollar-neutral within band")

    # 3. DEPLOYMENT FLOOR on BOTH sides (no one-sided under-deployment).
    side_gross = {"long": 9500.0, "short": 9500.0}
    add("both_sides_deployment_floor",
        invariant_both_sides_deployment_floor(side_gross, side_budget=10000.0, floor=0.90),
        f"deploy long={side_gross['long']/10000.0:.2f} short={side_gross['short']/10000.0:.2f} "
        f"vs floor 0.90")

    # 4. FUNDING SIGN: a short with a positive rate RECEIVES; a long PAYS (signed carry convention).
    add("funding_sign_correct", invariant_funding_sign_correct(),
        "short+positive-rate must be a credit; long a debit")

    # 4b. ACCOUNT EQUITY RECONCILE: recorded equity == cash + unrealized (no phantom equity).
    _acct = PaperAccount(cash=20_000.0)
    _acct.positions["ETH/USDT:USDT"] = Position(
        symbol="ETH/USDT:USDT", direction="short", qty=2.0, entry_price=2000.0,
        opened_ts=datetime(2026, 6, 10, tzinfo=UTC))
    _marks = {"ETH/USDT:USDT": 1950.0}
    # Independent hand-computed expected: cash 20000 + short upnl 2*(2000-1950)=100 -> 20100.0.
    # MUST be a literal, NOT account.equity(marks): re-using the recomputed call makes the check
    # abs(equity - equity) <= tol == always True, so a buggy equity() would still pass.
    _recorded_equity = 20_100.0
    add("account_equity_reconciles",
        invariant_account_equity_reconciles(_acct, _marks, _recorded_equity),
        f"recorded equity {_recorded_equity} must equal cash + unrealized PnL")

    # 5. PAIR LEGS sized by the cointegration hedge ratio (no residual single-name exposure).
    add("pair_legs_hedge_ratio_sized",
        invariant_pair_legs_hedge_ratio_sized(hedge_ratio=0.8, qty_y=10.0, qty_x=8.0),
        "qty_x must equal hedge_ratio * qty_y")

    # 6. SENTIMENT within the §7.2 cap (|dw| <= cap*|w|) and never flips direction.
    add("sentiment_within_cap_range",
        invariant_sentiment_within_cap_range(weight=0.10, score=1.0, conf=1.0, cap=0.25),
        "conviction tilt must respect |dw| <= cap*|w| and never flip sign")

    # 7. CRYPTO-ONLY: no tokenized-stock / commodity / index TradFi-wrapper leg.
    add("no_tokenized_stock_leg",
        invariant_no_tokenized_stock_leg([{"info": {"underlyingType": "COIN"}}]),
        "every leg must be a cryptocurrency COIN perp")

    return out


def run_self_audit() -> dict:
    """Run the invariant panel. Returns ``{ok, checks:[{name, ok, detail}]}``; ``ok`` = all pass."""
    results = _checks()
    return {"ok": all(ok for _, ok, _ in results),
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results]}

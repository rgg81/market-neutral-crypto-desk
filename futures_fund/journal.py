"""Two-phase decision journal, re-keyed for the market-neutral desk.

Ported/adapted from the weekly reference repo. The storage machinery is identical
(monthly ``journal-YYYY-MM.jsonl`` files under ``<memory_dir>/episodic/``, an
``extra="allow"`` ``Decision`` model, idempotent appends), but the public API is
re-keyed on ``(cycle, symbol, direction)`` — the desk's natural decision identity —
rather than an opaque ``decision_id``.

Phase 6 re-keys *outcome* signals on market-neutral ALPHA (return net of BTC-beta),
not raw return. ``alpha_outcome`` is the typed accessor that reads + validates the
six alpha-vs-beta outcome fields and raises on a missing/ill-typed field.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from futures_fund.models import Direction


class Decision(BaseModel):
    """Two-phase decision record. Phase-1 fields written at decision time; Phase-2 (outcome)
    fields patched on close. ``extra="allow"`` lets Phase-B agents attach richer context and lets
    the alpha-vs-beta outcome fields round-trip without a schema bump."""

    model_config = ConfigDict(extra="allow")

    id: str
    ts: datetime
    cycle: int
    symbol: str
    direction: Direction
    # Phase-1 optional context (everything beyond the identity key is optional payload)
    entry: float | None = None
    stop: float | None = None
    take_profit: list[float] = Field(default_factory=list)
    size: float | None = None
    leverage: float | None = None
    r_multiple: float | None = None
    funding_at_entry: float | None = None
    regime: str | None = None
    setup: str | None = None
    rationale: str | None = None
    dominant_signal: str | None = None
    contributing_agents: list[str] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    # Phase-2 outcome (None until closed) — the alpha-vs-beta fields live in `extra`.
    exit_ts: datetime | None = None
    realized_pnl: float | None = None
    fees: float | None = None
    funding_paid: float | None = None
    slippage: float | None = None
    low_level_lesson: str | None = None
    high_level_lesson: str | None = None
    importance_1_10: int | None = None


class AlphaOutcome(BaseModel):
    """Typed view of the six market-neutral outcome signals patched onto a closed decision.

    Re-keys the journal's outcome on ALPHA (return net of BTC-beta) rather than raw return:
    ``alpha_return`` is what the leg actually earned net of its ``beta_contribution`` to BTC,
    and the four booleans record whether each thesis dimension (cointegration, carry/funding,
    neutrality band, sentiment) held at exit. ``alpha_outcome`` raises if any field is absent —
    so this is a real validated accessor, not an ``extra="allow"`` echo."""

    model_config = ConfigDict(extra="ignore")

    alpha_return: float
    beta_contribution: float
    pair_cointegrated_at_exit: bool
    funding_thesis_matched: bool
    neutrality_in_band: bool
    sentiment_helped: bool


ALPHA_OUTCOME_FIELDS: tuple[str, ...] = tuple(AlphaOutcome.model_fields.keys())


def alpha_outcome(decision: dict | Decision) -> AlphaOutcome:
    """Read + validate the six alpha-vs-beta outcome fields off a (closed) decision.

    Raises ``KeyError`` if any of the six fields is absent (an outcome that was never patched, or
    only partially patched) and ``pydantic.ValidationError`` if a field is present but ill-typed.
    This is the genuine Phase-6 behavior: it would fail without the model + accessor, even though
    ``Decision`` is ``extra="allow"`` and round-trips the raw values."""
    data = decision.model_dump() if isinstance(decision, Decision) else dict(decision)
    missing = [f for f in ALPHA_OUTCOME_FIELDS if f not in data or data[f] is None]
    if missing:
        raise KeyError(
            f"alpha outcome incomplete; missing fields: {sorted(missing)}"
        )
    return AlphaOutcome.model_validate(data)


def _episodic_dir(memory_dir) -> Path:
    return Path(memory_dir) / "episodic"


def journal_file(memory_dir, ts: datetime) -> Path:
    return _episodic_dir(memory_dir) / f"journal-{ts:%Y-%m}.jsonl"


def _all_files(memory_dir) -> list[Path]:
    d = _episodic_dir(memory_dir)
    return sorted(d.glob("journal-*.jsonl")) if d.exists() else []


def read_all_decisions(memory_dir) -> list[dict]:
    """All decision records as raw dicts. NOTE: datetime fields (ts, exit_ts) are ISO-8601
    STRINGS here, not datetime objects — call ``Decision.model_validate(r)`` for typed access."""
    out: list[dict] = []
    for f in _all_files(memory_dir):
        for line in f.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _find(memory_dir, cycle: int, symbol: str, direction: str) -> dict | None:
    for d in read_all_decisions(memory_dir):
        if (
            d.get("cycle") == cycle
            and d.get("symbol") == symbol
            and d.get("direction") == direction
        ):
            return d
    return None


def append_decision(
    memory_dir,
    *,
    cycle: int,
    symbol: str,
    direction: str,
    payload: dict | None = None,
    ts: datetime | None = None,
) -> str:
    """Validate and append a Phase-1 decision keyed on ``(cycle, symbol, direction)``; return id.

    IDEMPOTENT per ``(cycle, symbol, direction)``: a DUE RETRY re-running the same cycle re-journals
    the same opens — without this guard that double-counts the open in hit-rate / per-agent stats /
    reflection. If a decision for this key already exists, its id is returned and nothing is
    appended. The key is unique per cycle (one open per symbol+direction per cycle; cycle numbers
    are monotonic), so this never collides two legitimate decisions."""
    existing = _find(memory_dir, cycle, symbol, direction)
    if existing is not None:
        return existing["id"]  # already journaled this cycle's open -> reuse, don't duplicate
    data: dict = dict(payload or {})
    data.update(
        id=data.get("id") or uuid.uuid4().hex,
        ts=ts or data.get("ts") or datetime.now(UTC),
        cycle=cycle,
        symbol=symbol,
        direction=direction,
    )
    decision = Decision.model_validate(data)
    f = journal_file(memory_dir, decision.ts)
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("a") as fh:
        fh.write(decision.model_dump_json() + "\n")
    return decision.id


def patch_outcome(
    memory_dir,
    *,
    cycle: int,
    symbol: str,
    direction: str,
    outcome: dict,
) -> bool:
    """Merge Phase-2 outcome fields into the decision keyed by ``(cycle, symbol, direction)``.

    Rewrites the containing monthly file. The merged record is re-validated through ``Decision``
    (``extra="allow"``) so the alpha-vs-beta outcome fields round-trip while typed Phase-1 fields
    stay coerced. Returns ``False`` if no decision matches the key."""
    for f in _all_files(memory_dir):
        records = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        hit = False
        for r in records:
            if (
                r.get("cycle") == cycle
                and r.get("symbol") == symbol
                and r.get("direction") == direction
            ):
                merged = Decision.model_validate({**r, **outcome})
                r.clear()
                r.update(json.loads(merged.model_dump_json()))
                hit = True
                break  # the key is unique per file; stop scanning
        if hit:
            f.write_text("".join(json.dumps(r) + "\n" for r in records))
            return True
    return False

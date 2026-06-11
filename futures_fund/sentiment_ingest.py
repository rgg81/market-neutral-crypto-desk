from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from futures_fund.config import Settings
from futures_fund.contracts import SentimentReport
from futures_fund.market_context import build_market_context
from futures_fund.models import SentimentLevel

LEVEL_TO_S: dict[SentimentLevel, float] = {
    "very_positive": 1.0,
    "positive": 0.5,
    "neutral": 0.0,
    "negative": -0.5,
    "very_negative": -1.0,
}


def level_to_s(level: SentimentLevel) -> float:
    """Ordinal level -> numeric s in [-1,1] ({+2..-2}/2). Enforces the §7.1 mapping."""
    return LEVEL_TO_S[level]


def s_to_level(s: float) -> SentimentLevel:
    """Inverse bucketing (reviewer round-trips level<->s for the sentiment_range check)."""
    if s >= 0.75:
        return "very_positive"
    if s >= 0.25:
        return "positive"
    if s > -0.25:
        return "neutral"
    if s > -0.75:
        return "negative"
    return "very_negative"


def decay_score(s: float, age_hours: float, half_life_days: float = 3.0) -> float:
    """Exponential decay toward 0: s * 0.5**(age_hours/(half_life_days*24))."""
    if half_life_days <= 0:
        return s
    return s * (0.5 ** (age_hours / (half_life_days * 24.0)))


def decay_report(report: SentimentReport, now: datetime, half_life_days: float = 3.0
                 ) -> SentimentReport:
    """Return a copy with decayed_s set from (now - as_of_ts)."""
    age_hours = max(0.0, (now - report.as_of_ts).total_seconds() / 3600.0)
    decayed = decay_score(report.s, age_hours, half_life_days=half_life_days)
    return report.model_copy(update={"decayed_s": decayed})


def validate_point_in_time(report: SentimentReport) -> bool:
    """True iff every source.published_ts < report.as_of_ts (reviewer point-in-time check)."""
    return all(src.published_ts < report.as_of_ts for src in report.sources)


def fail_soft_neutral(symbol: str, now: datetime) -> SentimentReport:
    """Neutral report for missing/unparseable/stale sentiment. Never blocks the book."""
    return SentimentReport(symbol=symbol, level="neutral", s=0.0, confidence=0.0,
                           sources=[], rationale="fail-soft neutral", as_of_ts=now)


def gather_sentiment_context(http_client, settings: Settings, fred_key: str | None, *,
                             as_of: datetime) -> dict:
    """Point-in-time wrapper over market_context.build_market_context.

    Drops any news source whose published timestamp is at or after `as_of` (no post-decision
    leakage), and records the `as_of` anchor so downstream point-in-time checks can audit the
    gather. The real NewsItem.model_dump() carries `published_at`, which is the field checked.
    """
    ctx = build_market_context(http_client, settings, fred_key)
    ctx["news"] = [n for n in ctx.get("news", [])
                   if not _is_future(n.get("published_at"), as_of)]
    ctx["as_of"] = as_of.isoformat()
    return ctx


def _parse_published(published_at) -> datetime | None:
    """Parse a source's published timestamp into a tz-aware datetime.

    Real RSS `<pubDate>` is RFC-822 (e.g. 'Fri, 29 May 2026 14:20:32 +0000'); Atom and some feeds
    emit ISO-8601. Try ISO first, then RFC-822. A naive result is assumed UTC. Returns None when the
    value is empty or unparseable.
    """
    if not published_at:
        return None
    s = str(published_at).strip()
    if not s:
        return None
    for parse in (datetime.fromisoformat, parsedate_to_datetime):
        try:
            dt = parse(s)
        except (TypeError, ValueError):
            continue
        if dt is None:
            continue
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    return None


def _is_future(published_at, cutoff: datetime) -> bool:
    """True if a source's published timestamp is at/after the decision-time cutoff.

    `published_at` is the raw feed value (RFC-822 `<pubDate>` or ISO); it is parsed to a tz-aware
    datetime and compared against `cutoff` as datetimes. Unparseable/missing -> drop (treated as
    future) so an undated source never leaks past the point-in-time boundary.
    """
    dt = _parse_published(published_at)
    if dt is None:
        return True
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)
    return dt >= cutoff

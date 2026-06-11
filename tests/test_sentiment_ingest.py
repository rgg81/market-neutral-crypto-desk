from datetime import UTC, datetime

import pytest

import futures_fund.sentiment_ingest as si
from futures_fund.contracts import SentimentReport, SentimentSource
from futures_fund.sentiment_ingest import (
    LEVEL_TO_S,
    _is_future,
    decay_report,
    decay_score,
    fail_soft_neutral,
    gather_sentiment_context,
    level_to_s,
    s_to_level,
    validate_point_in_time,
)


def _utc(h=0, d=1):
    return datetime(2026, 6, d, h, 0, tzinfo=UTC)


def test_level_to_s_mapping_is_canonical():
    assert LEVEL_TO_S == {"very_positive": 1.0, "positive": 0.5, "neutral": 0.0,
                          "negative": -0.5, "very_negative": -1.0}
    assert level_to_s("very_positive") == 1.0
    assert level_to_s("negative") == -0.5


def test_s_to_level_round_trips_buckets():
    assert s_to_level(1.0) == "very_positive"
    assert s_to_level(0.5) == "positive"
    assert s_to_level(0.0) == "neutral"
    assert s_to_level(-0.5) == "negative"
    assert s_to_level(-1.0) == "very_negative"
    # round-trip for the reviewer's sentiment_range check
    for lvl, s in LEVEL_TO_S.items():
        assert s_to_level(s) == lvl


def test_decay_score_halves_after_one_halflife():
    # 3-day half-life -> 72h -> exactly halved
    assert decay_score(1.0, age_hours=72.0, half_life_days=3.0) == pytest.approx(0.5)
    assert decay_score(-0.8, age_hours=72.0, half_life_days=3.0) == pytest.approx(-0.4)
    # zero age -> unchanged
    assert decay_score(0.6, age_hours=0.0, half_life_days=3.0) == pytest.approx(0.6)


def test_decay_report_sets_decayed_s_from_age():
    r = SentimentReport(symbol="BTC/USDT:USDT", level="very_positive", s=1.0,
                        confidence=0.9, as_of_ts=_utc(0, d=1))
    out = decay_report(r, now=_utc(0, d=4), half_life_days=3.0)  # 72h later
    assert out.decayed_s == pytest.approx(0.5)
    assert out is not r  # returns a copy
    assert r.decayed_s is None  # original untouched


def test_validate_point_in_time_rejects_future_source():
    good = SentimentReport(symbol="BTC/USDT:USDT", level="neutral", s=0.0, confidence=0.3,
                           sources=[SentimentSource(url="u", published_ts=_utc(6, d=1))],
                           as_of_ts=_utc(12, d=1))
    bad = SentimentReport(symbol="BTC/USDT:USDT", level="neutral", s=0.0, confidence=0.3,
                          sources=[SentimentSource(url="u", published_ts=_utc(18, d=1))],
                          as_of_ts=_utc(12, d=1))
    assert validate_point_in_time(good) is True
    assert validate_point_in_time(bad) is False


def test_fail_soft_neutral_is_zeroed():
    r = fail_soft_neutral("SOL/USDT:USDT", now=_utc(12, d=1))
    assert r.level == "neutral" and r.s == 0.0 and r.confidence == 0.0
    assert r.symbol == "SOL/USDT:USDT" and r.as_of_ts == _utc(12, d=1)


@pytest.mark.parametrize(
    "published_at, expect_future",
    [
        # RFC-822 <pubDate> (real CoinDesk/Cointelegraph/etc. format) before the 01-Jun cutoff
        ("Fri, 29 May 2026 14:20:32 +0000", False),
        ("Fri, 29 May 2026 14:20:32 GMT", False),
        # RFC-822 after the cutoff -> future, must be dropped
        ("Wed, 03 Jun 2026 09:00:00 +0000", True),
        # ISO-8601 before / after the cutoff
        ("2026-05-29T14:20:32+00:00", False),
        ("2026-06-03T09:00:00+00:00", True),
        # naive ISO (no tz) is assumed UTC
        ("2026-05-29T14:20:32", False),
        # the cutoff instant itself counts as future (drops published_ts >= as_of, §7.1)
        ("Mon, 01 Jun 2026 00:00:00 +0000", True),
        # missing / unparseable -> treated as future so undated sources never leak
        ("", True),
        ("not a date", True),
        (None, True),
    ],
)
def test_is_future_parses_rfc822_and_iso(published_at, expect_future):
    cutoff = _utc(0, d=1)  # 2026-06-01T00:00:00+00:00
    assert _is_future(published_at, cutoff) is expect_future


def test_gather_sentiment_context_keeps_past_drops_future(monkeypatch):
    """The point-in-time filter keeps past-dated news and drops at/after-as_of leakage, across
    both RFC-822 <pubDate> and ISO timestamps — exercising the real filter, not a degraded feed."""
    news = [
        {"title": "past rfc822", "published_at": "Fri, 29 May 2026 14:20:32 +0000"},
        {"title": "past iso", "published_at": "2026-05-30T08:00:00+00:00"},
        {"title": "future rfc822", "published_at": "Wed, 03 Jun 2026 09:00:00 +0000"},
        {"title": "future iso", "published_at": "2026-06-03T09:00:00+00:00"},
        {"title": "undated", "published_at": ""},
    ]
    monkeypatch.setattr(si, "build_market_context",
                        lambda *a, **k: {"news": [dict(n) for n in news]})

    from futures_fund.config import Settings
    ctx = gather_sentiment_context(object(), Settings(), fred_key=None, as_of=_utc(0, d=1))

    assert ctx["as_of"] == _utc(0, d=1).isoformat()  # anchor recorded for downstream PIT checks
    kept = {n["title"] for n in ctx["news"]}
    assert kept == {"past rfc822", "past iso"}  # only past-dated sources survive the gate


def test_gather_sentiment_context_real_newsitem_wiring(monkeypatch):
    """End-to-end wiring guard: pins the real NewsItem.published_at -> gather_sentiment_context
    linkage. Build REAL NewsItem objects, wrap them exactly as build_market_context does
    ({"news": [i.model_dump() for i in items], ...}), and assert the strictly-before gate keeps
    the past item and drops BOTH the future item and the one EXACTLY at as_of. If published_at were
    renamed, every news item would silently drop and this test would fail (catching the dict-only
    blind spot)."""
    from futures_fund.config import Settings
    from futures_fund.vendors import NewsItem

    as_of = _utc(0, d=1)  # 2026-06-01T00:00:00+00:00 decision time

    def _item(title: str, published_at: str) -> NewsItem:
        return NewsItem(title=title, url="https://x", published_at=published_at,
                        source="src", kind="news", instruments=["BTC"], summary="s")

    items = [
        _item("past", "2026-05-29T14:20:32+00:00"),                 # clearly PAST -> kept
        _item("future", "2026-06-03T09:00:00+00:00"),               # clearly FUTURE -> dropped
        _item("exactly_at_as_of", as_of.isoformat()),               # == as_of -> dropped (strict <)
    ]
    # mirror build_market_context's real shape: news is model_dump()ed NewsItems
    ctx_shape = {"news": [i.model_dump() for i in items], "fear_greed": None, "macro": {},
                 "social": {"posts": [], "mentions": {}}, "warnings": []}
    monkeypatch.setattr(si, "build_market_context", lambda *a, **k: ctx_shape)

    ctx = gather_sentiment_context(object(), Settings(), fred_key=None, as_of=as_of)

    kept = {n["title"] for n in ctx["news"]}
    assert kept == {"past"}  # only strictly-past survives; future AND exactly-at-as_of are dropped


def test_gather_sentiment_context_degrades_safely(monkeypatch):
    """When every feed errors, build_market_context yields no news; the gather still records the
    as_of anchor and returns an empty news list (fail-soft)."""
    class _Http:
        def get(self, *a, **k):
            raise RuntimeError("network disabled")

    from futures_fund.config import Settings
    ctx = gather_sentiment_context(_Http(), Settings(), fred_key=None, as_of=_utc(12, d=1))
    assert ctx["as_of"] == _utc(12, d=1).isoformat()
    assert ctx["news"] == []

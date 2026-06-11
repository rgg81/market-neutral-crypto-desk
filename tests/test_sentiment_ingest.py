from datetime import UTC, datetime

import pytest

from futures_fund.contracts import SentimentReport, SentimentSource
from futures_fund.sentiment_ingest import (
    LEVEL_TO_S,
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


def test_gather_sentiment_context_drops_future_sources():
    class _Http:
        def get(self, *a, **k):
            raise RuntimeError("network disabled")

    from futures_fund.config import Settings
    ctx = gather_sentiment_context(_Http(), Settings(), fred_key=None, as_of=_utc(12, d=1))
    # context degrades to safe defaults; the as_of anchor is recorded for downstream PIT checks
    assert ctx["as_of"] == _utc(12, d=1).isoformat()
    assert ctx["news"] == []  # no future/leaking sources slipped in

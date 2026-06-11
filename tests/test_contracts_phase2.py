from __future__ import annotations

import typing
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from futures_fund import models
from futures_fund.contracts import SentimentBatch, SentimentReport, SentimentSource

_NOW = datetime(2026, 6, 11, 0, 0, tzinfo=UTC)


def test_sleeve_name_alias_values():
    assert set(typing.get_args(models.SleeveName)) == {"carry", "pairs", "factor", "sentiment"}


def test_sentiment_level_alias_values():
    assert set(typing.get_args(models.SentimentLevel)) == {
        "very_positive", "positive", "neutral", "negative", "very_negative",
    }


def test_spread_state_alias_values():
    assert set(typing.get_args(models.SpreadState)) == {
        "flat", "long_spread", "short_spread", "stop",
    }


def test_pair_test_method_alias_values():
    assert set(typing.get_args(models.PairTestMethod)) == {"engle_granger", "johansen"}


def test_cadence_alias_values():
    assert set(typing.get_args(models.Cadence)) == {"weekly", "daily"}


def test_sentiment_report_valid():
    r = SentimentReport(
        symbol="BTC/USDT:USDT",
        level="positive",
        s=0.5,
        confidence=0.8,
        sources=[SentimentSource(url="http://x", published_ts=_NOW - timedelta(hours=2))],
        rationale="ETF inflows",
        as_of_ts=_NOW,
    )
    assert r.s == 0.5
    assert r.decayed_s is None
    assert r.sources[0].feed == ""


def test_sentiment_report_s_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SentimentReport(symbol="BTC/USDT:USDT", level="positive", s=1.5,
                        confidence=0.8, as_of_ts=_NOW)


def test_sentiment_report_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SentimentReport(symbol="BTC/USDT:USDT", level="neutral", s=0.0,
                        confidence=1.5, as_of_ts=_NOW)


def test_sentiment_batch_defaults_empty():
    b = SentimentBatch()
    assert b.reports == []

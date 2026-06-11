from __future__ import annotations

import typing

from futures_fund import models


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

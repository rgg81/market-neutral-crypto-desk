from datetime import UTC, datetime

from futures_fund.config import Settings
from futures_fund.market_context import build_market_context
from futures_fund.vendors import FearGreed, NewsItem


class _FailHttp:
    """Fake http client whose every .get() raises — drives the per-feed degrade paths.
    (fetch_macro is internally fail-soft and short-circuits on a None key, so macro degrades
    to {} without the client being called.)"""
    def get(self, *args, **kwargs):
        raise RuntimeError("network disabled in tests")


def test_build_context_degrades_when_all_feeds_fail():
    s = Settings()
    ctx = build_market_context(_FailHttp(), s, fred_key=None)
    # every feed failed -> safe defaults + warnings, never raises
    assert ctx["fear_greed"] is None
    assert ctx["news"] == []
    assert ctx["macro"] == {}
    assert ctx["social"] == {"posts": [], "mentions": {}}
    assert any("Fear&Greed" in w for w in ctx["warnings"])
    assert any("news" in w for w in ctx["warnings"])
    assert any("FRED" in w for w in ctx["warnings"])
    assert "macro_labels" in ctx


def test_real_fear_greed_and_news_models_construct():
    # REAL reused models: FearGreed REQUIRES `ts`; NewsItem REQUIRES published_at+kind+instruments.
    fg = FearGreed(value=55, classification="Greed",
                   ts=datetime(2026, 6, 1, tzinfo=UTC))
    assert fg.value == 55 and fg.classification == "Greed"
    ni = NewsItem(title="t", url="u", published_at="2026-06-01T00:00:00Z",
                  source="coindesk", kind="news", instruments=["BTC"])
    assert ni.title == "t" and ni.source == "coindesk"
    assert ni.kind == "news" and ni.instruments == ["BTC"]
    # NewsItem.model_dump() carries the published_at key the context consumes
    assert ni.model_dump()["published_at"] == "2026-06-01T00:00:00Z"

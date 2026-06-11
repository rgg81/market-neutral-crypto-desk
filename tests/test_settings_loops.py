from futures_fund.config import Settings


def test_model_for_resolves_weekly_daily_loops():
    s = Settings()
    assert {"weekly", "daily"} <= set(s.loops)  # loops vocabulary extended
    assert isinstance(s.model_for("trader", loop="weekly"), str)
    assert isinstance(s.model_for("reflector", loop="daily"), str)

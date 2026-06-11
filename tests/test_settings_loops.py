"""Characterization tests for the (pre-existing) two-cadence `loops` vocabulary and the
`model_for` resolution ORDER. NOTE: weekly/daily loop keys and `model_for` already existed
before Phase 4 (see futures_fund/config._default_loops / Settings.model_for); these tests
pin the inherited behavior the TraderOutput wiring depends on, they do not add net-new config.
"""

from futures_fund.config import Settings


def test_loops_carry_weekly_and_daily_cadences():
    s = Settings()
    assert {"weekly", "daily"} <= set(s.loops)  # inherited two-cadence vocabulary


def test_model_for_resolves_per_loop_deep_model():
    s = Settings()
    # No agent_models override -> a role resolves to the loop's deep_model tier.
    assert s.model_for("trader", loop="weekly") == s.loops["weekly"].deep_model
    assert s.model_for("reflector", loop="daily") == s.loops["daily"].deep_model


def test_model_for_resolution_order_agent_then_loop_then_global():
    # agent_models wins FIRST, else loop deep_model, else global deep_model.
    s = Settings(
        agent_models={"trader": "haiku"},
        loops={**Settings().loops},
        deep_model="opus",
    )
    assert s.model_for("trader", loop="weekly") == "haiku"          # per-agent override wins
    assert s.model_for("reflector", loop="weekly") == s.loops["weekly"].deep_model  # loop tier
    assert s.model_for("reflector") == "opus"                       # global fallback (no loop)
    assert s.model_for("reflector", loop="unknown") == "opus"       # unknown loop -> global

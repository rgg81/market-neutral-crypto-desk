from futures_fund.config import LoopSettings, _default_loops, load_settings


def test_load_settings_parses_account_and_live(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "account_size_usdt: 20000\n"
        "live: false\n"
        "max_drawdown_tolerance: 0.05\n"
    )
    s = load_settings(p)
    assert s.account_size_usdt == 20000.0
    assert s.live is False
    assert s.max_drawdown_tolerance == 0.05


def test_load_settings_parses_loops_two_cadence(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "loops:\n"
        "  weekly:\n"
        "    timeframe: \"4h\"\n"
        "    regime_timeframe: \"4h\"\n"
        "    poll_minutes: 1440\n"
        "    deep_model: \"opus\"\n"
        "    cadence_days: 7\n"
        "  daily:\n"
        "    timeframe: \"1h\"\n"
        "    poll_minutes: 60\n"
        "    deep_model: \"sonnet\"\n"
        "    cadence_hour_utc: 0\n"
    )
    s = load_settings(p)
    assert s.loops["weekly"].cadence_days == 7
    assert s.loops["weekly"].regime_timeframe == "4h"
    assert s.loops["weekly"].poll_minutes == 1440
    assert s.loops["daily"].cadence_hour_utc == 0
    assert s.loops["daily"].poll_minutes == 60
    # _default_loops() round-trips the same new fields when the block is absent
    dl = _default_loops()
    assert dl["weekly"].cadence_days == 7
    assert dl["daily"].cadence_hour_utc == 0
    assert isinstance(dl["weekly"], LoopSettings)


def test_model_for_resolves_agent_models_first_then_loop(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "agent_models:\n  sentiment: \"opus\"\n"
        "loops:\n  daily:\n    deep_model: \"sonnet\"\n    poll_minutes: 60\n"
    )
    s = load_settings(p)
    # per-agent map wins FIRST (inherited contract), regardless of loop tier
    assert s.model_for("sentiment", loop="daily") == "opus"
    # unknown role falls back to the loop's deep_model
    assert s.model_for("operational_narrator", loop="daily") == "sonnet"
    # no loop -> global deep_model default
    assert s.model_for("operational_narrator") == "opus"


def test_load_settings_parses_universe_block(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "universe:\n"
        "  symbol_count: 30\n"
        "  min_adv_usd: 50000000\n"
        "  crypto_only: true\n"
    )
    s = load_settings(p)
    assert s.universe.symbol_count == 30
    assert s.universe.min_adv_usd == 50_000_000.0
    assert s.universe.crypto_only is True


def test_load_settings_parses_fees_and_funding_and_slippage(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "fees:\n  taker_bps: 5.0\n  maker_bps: 2.0\n"
        "funding:\n  major_cap: 0.003\n  alt_cap: 0.02\n  unclamped_in_rr: true\n"
        "slippage:\n  model: depth\n  k: 0.1\n  half_spread_bps_default: 1.0\n"
    )
    s = load_settings(p)
    assert s.fees.taker_bps == 5.0
    assert s.fees.maker_bps == 2.0
    assert s.funding.major_cap == 0.003
    assert s.funding.alt_cap == 0.02
    assert s.funding.unclamped_in_rr is True
    assert s.slippage.k == 0.1
    assert s.slippage.flat_bps is None


def test_load_settings_parses_metrics_and_sentiment(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "metrics:\n  daily_periods_per_year: 365\n  weekly_periods_per_year: 52\n"
        "sentiment:\n  kappa: 0.5\n  cap: 0.25\n  halflife_days: 3\n"
    )
    s = load_settings(p)
    assert s.metrics.daily_periods_per_year == 365
    assert s.metrics.weekly_periods_per_year == 52
    assert s.sentiment.kappa == 0.5
    assert s.sentiment.cap == 0.25
    assert s.sentiment.halflife_days == 3


def test_neutrality_block_is_kept_as_raw_dict(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("neutrality:\n  side_budget_usdt: 10000\n  dollar_band: 0.03\n")
    s = load_settings(p)
    assert s.neutrality["side_budget_usdt"] == 10000
    assert s.neutrality["dollar_band"] == 0.03


def test_defaults_when_file_absent(tmp_path):
    s = load_settings(tmp_path / "nope.yaml")
    assert s.account_size_usdt == 20000.0
    assert s.universe.symbol_count == 30
    assert s.live is False
    assert s.agent_models == {}
    assert s.loops["weekly"].cadence_days == 7

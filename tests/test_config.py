import os

from futures_fund.config import (
    DataSettings,
    ExchangeSettings,
    LoopSettings,
    Settings,
    _default_loops,
    load_env_file,
    load_settings,
)


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


def test_load_env_file_missing_returns_empty(tmp_path):
    # No .env present -> no-op, empty mapping, no env mutation.
    loaded = load_env_file(tmp_path / ".env")
    assert loaded == {}


def test_load_env_file_strips_quotes_and_skips_comments_blanks(tmp_path, monkeypatch):
    # Quote stripping, comment/blank/no-'=' skipping, and empty-key skipping.
    monkeypatch.delenv("MN_PLAIN", raising=False)
    monkeypatch.delenv("MN_DQUOTED", raising=False)
    monkeypatch.delenv("MN_SQUOTED", raising=False)
    monkeypatch.delenv("MN_EQUALS_IN_VALUE", raising=False)
    p = tmp_path / ".env"
    p.write_text(
        "# a comment line\n"
        "\n"
        "   \n"
        "no_equals_sign_here\n"
        "=orphan_value_no_key\n"
        "MN_PLAIN=bare\n"
        'MN_DQUOTED="double quoted"\n'
        "MN_SQUOTED='single quoted'\n"
        "MN_EQUALS_IN_VALUE=a=b=c\n"
    )
    loaded = load_env_file(p)
    # Returned mapping reflects only the valid, parsed lines.
    assert loaded == {
        "MN_PLAIN": "bare",
        "MN_DQUOTED": "double quoted",
        "MN_SQUOTED": "single quoted",
        "MN_EQUALS_IN_VALUE": "a=b=c",
    }
    # Quotes are stripped and split is on the FIRST '=' only.
    assert os.environ["MN_DQUOTED"] == "double quoted"
    assert os.environ["MN_SQUOTED"] == "single quoted"
    assert os.environ["MN_EQUALS_IN_VALUE"] == "a=b=c"
    # Comment / blank / no-'=' / empty-key lines never reach os.environ.
    assert "no_equals_sign_here" not in os.environ
    assert "" not in os.environ
    assert "orphan_value_no_key" not in os.environ


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch):
    # setdefault semantics: a pre-existing env var WINS over the .env value.
    monkeypatch.setenv("MN_EXISTING", "from_environment")
    p = tmp_path / ".env"
    p.write_text("MN_EXISTING=from_dotenv\n")
    loaded = load_env_file(p)
    # The return value records what the file declared...
    assert loaded["MN_EXISTING"] == "from_dotenv"
    # ...but os.environ retains the original value (no override).
    assert os.environ["MN_EXISTING"] == "from_environment"


def test_exchange_property_accessors_read_configured_env(monkeypatch):
    monkeypatch.setenv("MY_KEY_ENV", "key-123")
    monkeypatch.setenv("MY_SECRET_ENV", "secret-456")
    ex = ExchangeSettings(key_env="MY_KEY_ENV", secret_env="MY_SECRET_ENV")
    assert ex.api_key == "key-123"
    assert ex.api_secret == "secret-456"


def test_exchange_property_accessors_none_when_env_absent(monkeypatch):
    monkeypatch.delenv("ABSENT_KEY_ENV", raising=False)
    monkeypatch.delenv("ABSENT_SECRET_ENV", raising=False)
    ex = ExchangeSettings(key_env="ABSENT_KEY_ENV", secret_env="ABSENT_SECRET_ENV")
    assert ex.api_key is None
    assert ex.api_secret is None


def test_data_fred_api_key_reads_configured_env(monkeypatch):
    monkeypatch.setenv("MY_FRED_ENV", "fred-789")
    data = DataSettings(fred_key_env="MY_FRED_ENV")
    assert data.fred_api_key == "fred-789"
    monkeypatch.delenv("MY_FRED_ENV", raising=False)
    assert data.fred_api_key is None


def test_load_settings_loads_dotenv_secrets_into_accessors(tmp_path, monkeypatch):
    # End-to-end: load_settings sources .env next to config.yaml, and the
    # configured *_env accessors then read those secrets.
    monkeypatch.delenv("BINANCE_KEY", raising=False)
    monkeypatch.delenv("BINANCE_SECRET", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        'BINANCE_KEY="env-key"\n'
        "BINANCE_SECRET=env-secret\n"
        "FRED_API_KEY='fred-key'\n"
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text("account_size_usdt: 20000\n")
    s = load_settings(cfg)
    assert s.exchange.api_key == "env-key"
    assert s.exchange.api_secret == "env-secret"
    assert s.data.fred_api_key == "fred-key"


def test_universe_settings_have_quality_knobs():
    u = Settings().universe
    assert u.min_age_days == 30
    assert u.max_abs_chg_24h_pct == 25.0
    assert u.min_depth_usd == 250_000.0
    assert u.depth_ref_usd == 100_000.0


def test_universe_quality_knobs_load_from_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "universe:\n"
        "  symbol_count: 30\n"
        "  min_adv_usd: 50000000\n"
        "  min_age_days: 45\n"
        "  max_abs_chg_24h_pct: 20\n"
        "  min_depth_usd: 300000\n"
        "  depth_ref_usd: 120000\n"
    )
    s = load_settings(cfg)
    assert s.universe.min_age_days == 45
    assert s.universe.max_abs_chg_24h_pct == 20.0
    assert s.universe.min_depth_usd == 300_000.0
    assert s.universe.depth_ref_usd == 120_000.0


def test_carry_funding_cap_default_none_and_loads():
    # default Settings() has an EMPTY sleeves dict -> no strategy cap (opt-in), so existing carry
    # behavior is unchanged.
    assert Settings().sleeves.get("carry", {}).get("max_abs_apr") is None


def test_repo_config_yaml_carry_cap_is_nested_correctly():
    # GUARD a YAML indent mistake: the carry block MUST be a sibling of factor:/pairs: under
    # sleeves:, not a child of factor:. Reads the REPO config.yaml (not a tmp fixture).
    s = load_settings("config.yaml")
    assert s.sleeves["carry"]["max_abs_apr"] == 2.0
    # factor: must still be its own sub-block (carry did not get nested inside it)
    assert "carry" not in s.sleeves.get("factor", {})

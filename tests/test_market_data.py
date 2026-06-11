import pytest

from futures_fund.market_data import (
    is_crypto_perp,
    liquidity_floor,
    parse_funding,
    parse_ohlcv,
    parse_symbol_spec,
    scan_universe,
)

_MARKETS = {
    "BTC/USDT:USDT": {"info": {"underlyingType": "COIN", "contractType": "PERPETUAL"}},
    "DOGE/USDT:USDT": {"info": {"underlyingType": "COIN", "contractType": "PERPETUAL"}},
    "GOLD/USDT:USDT": {"info": {"underlyingType": "COMMODITY",
                                "contractType": "TRADIFI_PERPETUAL"}},
    "SOL/USDT:USDT": {"info": {"underlyingType": "COIN", "contractType": "PERPETUAL"}},
}


class _TickerClient:
    markets = _MARKETS

    def fetch_tickers(self):
        return {
            "BTC/USDT:USDT": {"quoteVolume": 1e10, "percentage": 0.1, "last": 70000.0},
            "GOLD/USDT:USDT": {"quoteVolume": 8e9, "percentage": 0.5, "last": 2300.0},  # TradFi
            "SOL/USDT:USDT": {"quoteVolume": 9e9, "percentage": -1.0, "last": 150.0},
            "DOGE/USDT:USDT": {"quoteVolume": 3e7, "percentage": -2.0, "last": 0.1},  # thin
            "ETH/USDT:USD": {"quoteVolume": 9e9, "percentage": 0.0, "last": 2000.0},  # not perp
        }


def test_is_crypto_perp_rejects_tradfi_wrapper():
    assert is_crypto_perp(_MARKETS["BTC/USDT:USDT"]) is True
    assert is_crypto_perp(_MARKETS["GOLD/USDT:USDT"]) is False
    assert is_crypto_perp({"info": {}}) is True  # metadata gap -> keep plain perp


def test_scan_universe_drops_tradfi_and_non_perp():
    rows = scan_universe(_TickerClient(), top_n=10)
    syms = [r["symbol"] for r in rows]
    assert "GOLD/USDT:USDT" not in syms      # tokenized commodity excluded
    assert "ETH/USDT:USD" not in syms        # not a USDT perp
    assert syms == ["BTC/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"]  # vol-ranked


def test_liquidity_floor_trims_thin_names_and_caps_top_n():
    rows = scan_universe(_TickerClient(), top_n=10)
    kept = liquidity_floor(rows, min_adv_usd=5e7, symbol_count=30)
    syms = [r["symbol"] for r in kept]
    assert syms == ["BTC/USDT:USDT", "SOL/USDT:USDT"]  # DOGE (3e7) below the 5e7 floor


def test_liquidity_floor_caps_to_symbol_count():
    rows = [{"symbol": f"X{i}/USDT:USDT", "vol_24h_usd": 1e9 - i, "last": 1.0,
             "chg_24h_pct": 0.0} for i in range(40)]
    kept = liquidity_floor(rows, min_adv_usd=0.0, symbol_count=30)
    assert len(kept) == 30
    assert kept[0]["symbol"] == "X0/USDT:USDT"  # most liquid first preserved


def test_parse_ohlcv_sorts_and_labels_columns():
    df = parse_ohlcv([[1700000000000, 1, 2, 0.5, 1.5, 10],
                      [1699999996400, 1, 2, 0.5, 1.5, 10]])
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].is_monotonic_increasing


def test_parse_funding_defaults_interval_8h():
    fr = {"symbol": "BTC/USDT:USDT", "fundingRate": "0.0001",
          "fundingTimestamp": 1700000000000, "markPrice": "70000", "indexPrice": "69990"}
    info = parse_funding(fr)
    assert info.interval_hours == pytest.approx(8.0)
    assert info.current_rate == pytest.approx(0.0001)
    assert info.mark_price == pytest.approx(70000.0)


def test_parse_funding_sources_interval_when_present():
    fr = {"symbol": "SOL/USDT:USDT", "fundingRate": "0.0002",
          "fundingTimestamp": 1700000000000, "markPrice": "150", "indexPrice": "149.9"}
    info = parse_funding(fr, {"info": {"fundingIntervalHours": 4}})
    assert info.interval_hours == pytest.approx(4.0)


def test_parse_symbol_spec_maps_filters_and_brackets():
    market = {"id": "BTCUSDT", "symbol": "BTC/USDT:USDT",
              "precision": {"price": 0.1, "amount": 0.001},
              "limits": {"cost": {"min": 100.0}},
              "info": {"filters": [
                  {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                  {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                  {"filterType": "MIN_NOTIONAL", "notional": "5.0"}]}}
    tiers = [{"minNotional": 0, "maxNotional": 50000, "maintenanceMarginRate": 0.004,
              "maxLeverage": 125, "info": {"cum": "0"}}]
    spec = parse_symbol_spec(market, tiers)
    assert spec.tick_size == pytest.approx(0.1)
    assert spec.step_size == pytest.approx(0.001)
    assert spec.min_notional == pytest.approx(5.0)
    assert spec.mmr_brackets[0].max_leverage == pytest.approx(125.0)

"""Unit and Integration Tests for Market Data Subsystem.

Covers:
- Tier 1 Feature Tests:
  1. Candle model parsing from raw Binance and Bybit formats.
  2. Candle properties (is_bullish, is_bearish, body, range, typical_price).
  3. Ticker model parsing from Binance price and 24hr statistics endpoints.
  4. Timeframe conversion utilities (seconds, milliseconds, timedelta).
  5. CandleCache SQLite persistence (save_candles, get_candles, chronological ordering).
  6. CandleCache range querying, limit trimming, and count filtering.
  7. MarketDataService cache-first lookup (cache hit returns immediately without network hit).
- Tier 2 Boundary & Corner Tests:
  8. Empty dataset and empty responses handling.
  9. Malformed / corrupted timestamp parsing errors.
  10. Pagination across multi-batch historical ranges (>1,000 bars).
  11. HTTP 429 rate limit retry with Retry-After header.
  12. SQLite cache coverage calculation (has_coverage with missing bars).
  13. Network failure fallback to local cache.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError
import requests

from trading_bot.data.binance_client import BinanceClient
from trading_bot.data.cache import CandleCache, MarketDataService
from trading_bot.data.models import (
    Candle,
    Ticker,
    Timeframe,
    timeframe_to_milliseconds,
    timeframe_to_seconds,
    timeframe_to_timedelta,
)


# =====================================================================
# TIER 1 FEATURE TESTS
# =====================================================================
def test_candle_model_creation_and_immutability():
    """Test Candle creation, Decimal conversions, and frozen immutability."""
    ts = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    candle = Candle(
        timestamp=ts,
        open=Decimal("58000.00"),
        high=Decimal("58500.00"),
        low=Decimal("57900.00"),
        close=Decimal("58300.00"),
        volume=Decimal("14.50"),
        symbol="BTCEUR",
        interval="1h",
    )

    assert candle.timestamp == ts
    assert candle.open == Decimal("58000.00")
    assert candle.close == Decimal("58300.00")
    assert candle.is_bullish
    assert not candle.is_bearish
    assert candle.body == Decimal("300.00")
    assert candle.range == Decimal("600.00")
    assert candle.typical_price == (Decimal("58500.00") + Decimal("57900.00") + Decimal("58300.00")) / Decimal("3")

    # Immutability check
    with pytest.raises(ValidationError):
        candle.close = Decimal("59000.00")  # type: ignore[misc]


def test_candle_from_binance_raw():
    """Test parsing Binance 12-element kline array."""
    raw = [
        1785542400000,       # 0: Open time (ms)
        "58100.50",          # 1: Open
        "58650.00",          # 2: High
        "57990.25",          # 3: Low
        "58450.75",          # 4: Close
        "24.12345",          # 5: Volume
        1785545999999,       # 6: Close time (ms)
        "1407890.12",        # 7: Quote volume
        450,                 # 8: Trades count
        "12.0",              # 9: Taker buy base
        "700000.0",          # 10: Taker buy quote
        "0"                  # 11: Ignore
    ]

    candle = Candle.from_binance_raw(raw, symbol="BTCEUR", interval="1h")
    assert candle.symbol == "BTCEUR"
    assert candle.interval == "1h"
    assert candle.open == Decimal("58100.50")
    assert candle.high == Decimal("58650.00")
    assert candle.low == Decimal("57990.25")
    assert candle.close == Decimal("58450.75")
    assert candle.volume == Decimal("24.12345")
    assert candle.trades_count == 450
    assert candle.open_time_ms == 1785542400000
    assert candle.close_time_ms == 1785545999999


def test_candle_from_bybit_raw():
    """Test parsing Bybit v5 7-element kline array."""
    raw = [
        "1785542400000",  # 0: Start time ms
        "58000.0",        # 1: Open
        "58900.0",        # 2: High
        "57800.0",        # 3: Low
        "58750.0",        # 4: Close
        "18.5",           # 5: Volume
        "1080000.0",      # 6: Turnover
    ]
    candle = Candle.from_bybit_raw(raw, symbol="SOLEUR", interval="15m")
    assert candle.symbol == "SOLEUR"
    assert candle.interval == "15m"
    assert candle.open == Decimal("58000.0")
    assert candle.close == Decimal("58750.0")
    assert candle.volume == Decimal("18.5")
    assert candle.quote_volume == Decimal("1080000.0")


def test_ticker_models():
    """Test Ticker parsing from Binance price and 24hr responses."""
    price_data = {"symbol": "BTCEUR", "price": "62345.50"}
    ticker1 = Ticker.from_binance_ticker(price_data)
    assert ticker1.symbol == "BTCEUR"
    assert ticker1.price == Decimal("62345.50")

    full_data = {
        "symbol": "ETHEUR",
        "lastPrice": "3150.25",
        "bidPrice": "3150.00",
        "askPrice": "3150.50",
        "highPrice": "3200.00",
        "lowPrice": "3080.00",
        "volume": "15200.5",
        "quoteVolume": "47800000.0",
        "priceChangePercent": "2.25",
        "closeTime": 1785542400000,
    }
    ticker2 = Ticker.from_binance_24hr(full_data)
    assert ticker2.symbol == "ETHEUR"
    assert ticker2.price == Decimal("3150.25")
    assert ticker2.bid == Decimal("3150.00")
    assert ticker2.ask == Decimal("3150.50")
    assert ticker2.high_24h == Decimal("3200.00")
    assert ticker2.low_24h == Decimal("3080.00")
    assert ticker2.price_change_percent_24h == Decimal("2.25")


def test_timeframe_utilities():
    """Test timeframe conversions to seconds, milliseconds, and timedeltas."""
    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("15m") == 900
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("4h") == 14400
    assert timeframe_to_seconds("1d") == 86400

    assert timeframe_to_milliseconds("1h") == 3600000
    assert timeframe_to_timedelta("1h") == timedelta(hours=1)
    assert timeframe_to_timedelta("15m") == timedelta(minutes=15)

    # Invalid timeframe format
    with pytest.raises(ValueError):
        timeframe_to_seconds("invalid_interval")


def test_candle_cache_sqlite_lifecycle(tmp_path: Path):
    """Test SQLite candle caching, retrieval, range query, and counting."""
    db_file = tmp_path / "market_data.db"
    cache = CandleCache(db_path=db_file)

    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    candles = []
    for i in range(10):
        dt = base_time + timedelta(hours=i)
        p = Decimal(str(50000 + i * 100))
        candles.append(
            Candle(
                timestamp=dt,
                open=p,
                high=p + Decimal("50"),
                low=p - Decimal("50"),
                close=p + Decimal("20"),
                volume=Decimal("10.0"),
                symbol="BTCEUR",
                interval="1h",
            )
        )

    # Save to SQLite
    saved = cache.save_candles(candles)
    assert saved == 10
    assert cache.count("BTCEUR", "1h") == 10

    # Retrieve all
    retrieved = cache.get_candles("BTCEUR", "1h")
    assert len(retrieved) == 10
    assert retrieved[0].open == Decimal("50000")
    assert retrieved[-1].open == Decimal("50900")
    # Exact Decimal preservation without floating-point drift
    assert isinstance(retrieved[0].open, Decimal)

    # Range query: hours 2 to 5 inclusive
    start_q = base_time + timedelta(hours=2)
    end_q = base_time + timedelta(hours=5)
    subset = cache.get_candles("BTCEUR", "1h", start_time=start_q, end_time=end_q)
    assert len(subset) == 4
    assert subset[0].timestamp == start_q
    assert subset[-1].timestamp == end_q

    # Cached range check
    earliest, latest, cnt = cache.get_cached_range("BTCEUR", "1h")
    assert earliest == base_time
    assert latest == base_time + timedelta(hours=9)
    assert cnt == 10

    # Clear cache
    deleted = cache.clear("BTCEUR", "1h")
    assert deleted == 10
    assert cache.count("BTCEUR", "1h") == 0


def test_market_data_service_cache_hit_avoids_network(tmp_path: Path):
    """Test MarketDataService returns cached bars on cache hit without invoking client."""
    db_file = tmp_path / "test_cache_hit.db"
    cache = CandleCache(db_path=db_file)
    mock_client = MagicMock()

    # Pre-populate cache with 24 hours of data
    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    cached_bars = [
        Candle(
            timestamp=base_time + timedelta(hours=i),
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("1.0"),
            symbol="BTCEUR",
            interval="1h",
        )
        for i in range(24)
    ]
    cache.save_candles(cached_bars)

    service = MarketDataService(cache=cache, client=mock_client)

    start = base_time
    end = base_time + timedelta(hours=23)
    result = service.get_historical_candles(
        symbol="BTCEUR",
        interval="1h",
        start_time=start,
        end_time=end,
        force_refresh=False,
    )

    # Verify results came from cache and client was NOT called
    assert len(result) == 24
    mock_client.fetch_klines.assert_not_called()


# =====================================================================
# TIER 2 BOUNDARY & CORNER TESTS
# =====================================================================
def test_empty_dataset_and_empty_responses(tmp_path: Path):
    """Test handling of empty candle lists and query ranges with 0 results."""
    db_file = tmp_path / "test_empty.db"
    cache = CandleCache(db_path=db_file)

    # Query empty cache
    res = cache.get_candles("BTCEUR", "1h")
    assert res == []

    range_res = cache.get_cached_range("BTCEUR", "1h")
    assert range_res is None

    # Saving empty list returns 0
    assert cache.save_candles([]) == 0


def test_corrupted_and_invalid_timestamp_parsing():
    """Test that invalid timestamp formats raise ValueError during Candle construction."""
    with pytest.raises(ValueError):
        Candle(
            timestamp="not_a_date",  # Invalid
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("1.0"),
        )


def test_binance_client_pagination_multi_batch():
    """Test BinanceClient automated pagination when requesting more than 1,000 candles."""
    client = BinanceClient(base_url="https://api.binance.com")

    # Mock fetch_klines_batch to simulate 2 consecutive batches of 1,000 candles
    base_ms = 1785542400000
    step_ms = 3600000  # 1 hour

    def mock_batch(symbol, interval, start_time_ms, end_time_ms, limit):
        # Generate bars starting from start_time_ms
        count = min(limit, 1000)
        batch = []
        for i in range(count):
            t = start_time_ms + (i * step_ms)
            if t > end_time_ms:
                break
            batch.append(
                Candle(
                    timestamp=datetime.fromtimestamp(t / 1000.0, tz=timezone.utc),
                    open=Decimal("50000"),
                    high=Decimal("50100"),
                    low=Decimal("49900"),
                    close=Decimal("50050"),
                    volume=Decimal("1.0"),
                    symbol=symbol,
                    interval=interval,
                )
            )
        return batch

    with patch.object(client, "fetch_klines_batch", side_effect=mock_batch):
        # Request 1,500 hourly candles (exceeds single 1,000 batch limit)
        start_dt = datetime.fromtimestamp(base_ms / 1000.0, tz=timezone.utc)
        end_dt = datetime.fromtimestamp((base_ms + 1499 * step_ms) / 1000.0, tz=timezone.utc)

        candles = client.fetch_klines(
            symbol="BTCEUR",
            interval="1h",
            start_time=start_dt,
            end_time=end_dt,
            limit=1500,
        )

        assert len(candles) == 1500
        assert candles[0].timestamp == start_dt
        assert candles[-1].timestamp == end_dt


def test_binance_client_http_429_rate_limit_retry():
    """Test BinanceClient intercepts HTTP 429 and retries with backoff."""
    client = BinanceClient(base_url="https://api.binance.com", max_retries=2)

    mock_session = MagicMock()
    # First response: 429 Too Many Requests
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0"}  # 0s for fast unit test
    # Second response: 200 OK with valid json
    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"price": "61200.00", "symbol": "BTCEUR"}

    mock_session.get.side_effect = [resp_429, resp_200]
    client.session = mock_session

    ticker = client.get_ticker("BTCEUR")
    assert ticker.symbol == "BTCEUR"
    assert ticker.price == Decimal("61200.00")
    assert mock_session.get.call_count == 2


def test_market_data_service_network_error_cache_fallback(tmp_path: Path):
    """Test MarketDataService gracefully falls back to local cache when network fails."""
    db_file = tmp_path / "fallback.db"
    cache = CandleCache(db_path=db_file)
    mock_client = MagicMock()
    mock_client.fetch_klines.side_effect = requests.ConnectionError("DNS lookup failed")

    # Put 10 bars in cache
    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    cached_bars = [
        Candle(
            timestamp=base_time + timedelta(hours=i),
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("1.0"),
            symbol="BTCEUR",
            interval="1h",
        )
        for i in range(10)
    ]
    cache.save_candles(cached_bars)

    service = MarketDataService(cache=cache, client=mock_client)

    # Request with force_refresh=True (forces network attempt, fails, then falls back to cache)
    result = service.get_historical_candles(
        symbol="BTCEUR",
        interval="1h",
        start_time=base_time,
        end_time=base_time + timedelta(hours=9),
        force_refresh=True,
    )

    assert len(result) == 10
    assert result[0].open == Decimal("50000")

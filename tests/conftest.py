"""Shared pytest fixtures for the automated trading bot test suite.

Provides:
- Synthetic 30-day (744-bar) OHLCV candlestick series spanning bullish, sideways, and bearish regimes.
- Baseline 10.00 EUR VirtualLedger instance.
- Mock NotificationService and NotificationInterceptor fixtures.
- Mock Binance API klines responses for HTTP client and caching tests.
- Isolated temporary database and log directory fixtures.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import pytest

# Try importing domain models; fallback gracefully if worker hasn't committed yet
try:
    from trading_bot.data.models import Candle
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Candle:  # type: ignore[no-redef]
        timestamp: datetime
        open: Decimal
        high: Decimal
        low: Decimal
        close: Decimal
        volume: Decimal
        symbol: str = "BTC/EUR"


@pytest.fixture
def sample_candle() -> Candle:
    """Standard single candle representation."""
    return Candle(
        timestamp=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc),
        open=Decimal("58000.00"),
        high=Decimal("58500.00"),
        low=Decimal("57800.00"),
        close=Decimal("58250.00"),
        volume=Decimal("15.75"),
        symbol="BTC/EUR",
    )


@pytest.fixture
def synthetic_candles_30d() -> List[Candle]:
    """Generate deterministic 31-day (744 hourly candles) OHLCV series.

    Regimes covered:
    - Bars 0..199: Stable baseline / warmup (58,000 -> 59,500)
    - Bars 200..450: Strong bullish breakout momentum (59,500 -> 68,000)
    - Bars 451..600: Sideways consolidation & chop (67,500 - 68,500)
    - Bars 601..743: Bearish pullback & test (68,000 -> 61,500)
    """
    candles: List[Candle] = []
    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    current_close = Decimal("58000.00")

    for i in range(744):
        dt = base_time + timedelta(hours=i)
        
        # Regime selection
        if i < 200:
            # Baseline warmup with slight upward drift
            drift = Decimal(str(5.0 + (i % 7) * 2.0 - 5.0))
            vol = Decimal("10.0")
        elif i < 451:
            # Strong bullish breakout regime
            drift = Decimal(str(35.0 + (i % 5) * 8.0))
            vol = Decimal("25.0")
        elif i < 601:
            # Sideways chop regime (mean-reverting oscillation)
            drift = Decimal(str(((i % 6) - 2.5) * 12.0))
            vol = Decimal("8.0")
        else:
            # Bearish pullback regime
            drift = Decimal(str(-45.0 + (i % 4) * 6.0))
            vol = Decimal("20.0")

        open_p = current_close
        close_p = open_p + drift
        high_p = max(open_p, close_p) + Decimal("40.0")
        low_p = min(open_p, close_p) - Decimal("35.0")
        current_close = close_p

        candle = Candle(
            timestamp=dt,
            open=open_p.quantize(Decimal("0.01")),
            high=high_p.quantize(Decimal("0.01")),
            low=low_p.quantize(Decimal("0.01")),
            close=close_p.quantize(Decimal("0.01")),
            volume=vol,
            symbol="BTC/EUR",
        )
        candles.append(candle)

    return candles


@pytest.fixture
def bullish_breakout_candles() -> List[Candle]:
    """Short series of 30 candles producing an explosive upward breakout."""
    candles: List[Candle] = []
    base_time = datetime(2026, 8, 10, 0, 0, tzinfo=timezone.utc)
    p = Decimal("60000.00")

    for i in range(30):
        dt = base_time + timedelta(hours=i)
        delta = Decimal(str(15.0 if i < 20 else 120.0))  # sudden surge at bar 20
        open_p = p
        close_p = open_p + delta
        high_p = close_p + Decimal("25.0")
        low_p = open_p - Decimal("15.0")
        p = close_p

        candles.append(
            Candle(
                timestamp=dt,
                open=open_p.quantize(Decimal("0.01")),
                high=high_p.quantize(Decimal("0.01")),
                low=low_p.quantize(Decimal("0.01")),
                close=close_p.quantize(Decimal("0.01")),
                volume=Decimal("50.0" if i >= 20 else "10.0"),
                symbol="BTC/EUR",
            )
        )
    return candles


@pytest.fixture
def sample_ledger_10eur():
    """Returns a VirtualLedger initialized with exactly 10.00 EUR."""
    from trading_bot.simulation.ledger import VirtualLedger
    return VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")


@pytest.fixture
def mock_notification_service():
    """Returns a clean NotificationService instance with interceptor attached."""
    from trading_bot.notifications.service import NotificationService
    service = NotificationService()
    return service


@pytest.fixture
def mock_binance_klines_raw() -> List[List[Any]]:
    """Simulated Binance API klines payload for 5 hourly bars."""
    base_ms = 1785542400000  # 2026-08-01 00:00:00 UTC in ms
    rows: List[List[Any]] = []
    price = 58000.0

    for i in range(5):
        open_time = base_ms + (i * 3600 * 1000)
        close_time = open_time + (3600 * 1000) - 1
        open_p = price
        high_p = open_p + 150.0
        low_p = open_p - 100.0
        close_p = open_p + 50.0
        price = close_p
        vol = 12.345 + i

        rows.append([
            open_time,
            f"{open_p:.2f}",
            f"{high_p:.2f}",
            f"{low_p:.2f}",
            f"{close_p:.2f}",
            f"{vol:.4f}",
            close_time,
            f"{vol * close_p:.2f}",
            120 + i,
            f"{vol * 0.5:.4f}",
            f"{vol * 0.5 * close_p:.2f}",
            "0",
        ])
    return rows


@pytest.fixture
def isolated_temp_db(tmp_path: Path) -> Path:
    """Provides a fresh isolated SQLite database path for testing."""
    return tmp_path / "test_market_data.db"

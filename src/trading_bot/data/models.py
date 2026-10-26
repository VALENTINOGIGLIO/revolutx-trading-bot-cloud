"""Data models for market OHLCV candles, tickers, and timeframes.

Provides high-precision Decimal-based Pydantic schemas for candlestick and
ticker data with bi-directional serialization for Binance, Bybit, and SQLite.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =====================================================================
# Timeframe Constants & Utilities
# =====================================================================
class Timeframe(str, Enum):
    """Supported candlestick timeframes."""
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MN1 = "1M"


TIMEFRAME_SECONDS_MAP: Dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,
}


def timeframe_to_seconds(interval: str) -> int:
    """Convert an interval string (e.g. '15m', '1h', '1d') to seconds."""
    if interval in TIMEFRAME_SECONDS_MAP:
        return TIMEFRAME_SECONDS_MAP[interval]
    
    # Dynamic parser: number + unit
    unit = interval[-1].lower()
    val_str = interval[:-1]
    if not val_str.isdigit():
        raise ValueError(f"Invalid timeframe format: {interval}")
    
    val = int(val_str)
    if unit == "m":
        return val * 60
    elif unit == "h":
        return val * 3600
    elif unit == "d":
        return val * 86400
    elif unit == "w":
        return val * 604800
    elif unit == "M":
        return val * 2592000
    else:
        raise ValueError(f"Unknown timeframe unit: {unit} in {interval}")


def timeframe_to_milliseconds(interval: str) -> int:
    """Convert an interval string to milliseconds."""
    return timeframe_to_seconds(interval) * 1000


def timeframe_to_timedelta(interval: str) -> timedelta:
    """Convert an interval string to a Python timedelta object."""
    return timedelta(seconds=timeframe_to_seconds(interval))


# =====================================================================
# Candlestick (OHLCV) Model
# =====================================================================
class Candle(BaseModel):
    """Immutable, high-precision OHLCV Candlestick model.

    All price and volume fields use Decimal to prevent floating point inaccuracies.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    timestamp: datetime = Field(
        description="Candle open timestamp in UTC"
    )
    open: Decimal = Field(description="Open price")
    high: Decimal = Field(description="Highest price during bar")
    low: Decimal = Field(description="Lowest price during bar")
    close: Decimal = Field(description="Close (or current) price")
    volume: Decimal = Field(description="Base asset volume traded")
    symbol: str = Field(default="", description="Trading pair symbol (e.g. BTCEUR)")
    interval: str = Field(default="", description="Bar timeframe (e.g. 1h, 15m)")
    close_time: Optional[datetime] = Field(
        default=None, description="Candle close timestamp in UTC"
    )
    quote_volume: Optional[Decimal] = Field(
        default=None, description="Quote asset volume traded (e.g. EUR volume)"
    )
    trades_count: Optional[int] = Field(
        default=None, description="Number of trades executed during the bar"
    )

    # -----------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------
    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, v: Any) -> datetime:
        """Coerce int/float (ms or sec) or datetime into timezone-aware UTC datetime."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if isinstance(v, (int, float)):
            # If timestamp > 1e11, it's in milliseconds; otherwise seconds
            ts_sec = v / 1000.0 if v > 1e11 else float(v)
            return datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        if isinstance(v, str):
            # Try parsing ISO format
            try:
                dt = datetime.fromisoformat(v)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                pass
            # Try parsing numeric string
            try:
                val = float(v)
                ts_sec = val / 1000.0 if val > 1e11 else val
                return datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            except ValueError:
                raise ValueError(f"Cannot parse timestamp: {v}")
        raise ValueError(f"Unsupported timestamp format: {type(v)}: {v}")

    @field_validator("close_time", mode="before")
    @classmethod
    def _validate_close_time(cls, v: Any) -> Optional[datetime]:
        """Coerce close_time into UTC datetime if provided."""
        if v is None:
            return None
        return cls._validate_timestamp(v)

    @field_validator("open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def _validate_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric values to Decimal with full precision."""
        if isinstance(v, Decimal):
            return v
        if isinstance(v, (int, str)):
            return Decimal(str(v))
        if isinstance(v, float):
            # Format float cleanly to avoid binary representation artifacts
            return Decimal(str(v))
        raise ValueError(f"Cannot convert {type(v)}: {v} to Decimal")

    @field_validator("quote_volume", mode="before")
    @classmethod
    def _validate_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None or v == "":
            return None
        return cls._validate_decimal(v)

    # -----------------------------------------------------------------
    # Properties & Computed Attributes
    # -----------------------------------------------------------------
    @property
    def open_time_ms(self) -> int:
        """Return candle open time in integer Unix milliseconds."""
        return int(self.timestamp.timestamp() * 1000)

    @property
    def open_time(self) -> int:
        """Convenience alias for open_time_ms."""
        return self.open_time_ms

    @property
    def close_time_ms(self) -> int:
        """Return candle close time in integer Unix milliseconds."""
        if self.close_time is not None:
            return int(self.close_time.timestamp() * 1000)
        # Fallback: compute from interval if available
        if self.interval:
            step_ms = timeframe_to_milliseconds(self.interval)
            return self.open_time_ms + step_ms - 1
        return self.open_time_ms

    @property
    def is_bullish(self) -> bool:
        """True if close >= open."""
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        """True if close < open."""
        return self.close < self.open

    @property
    def body(self) -> Decimal:
        """Absolute difference between open and close."""
        return abs(self.close - self.open)

    @property
    def range(self) -> Decimal:
        """Total high to low price range."""
        return self.high - self.low

    @property
    def typical_price(self) -> Decimal:
        """HLC/3 typical price."""
        return (self.high + self.low + self.close) / Decimal("3")

    # -----------------------------------------------------------------
    # Factory & Serialization Methods
    # -----------------------------------------------------------------
    @classmethod
    def from_binance_raw(
        cls,
        raw: list,
        symbol: str = "",
        interval: str = ""
    ) -> "Candle":
        """Construct Candle from Binance public kline array format.

        Binance format:
        [
          0: Open time (ms),
          1: Open (str),
          2: High (str),
          3: Low (str),
          4: Close (str),
          5: Volume (str),
          6: Close time (ms),
          7: Quote asset volume (str),
          8: Number of trades (int),
          9: Taker buy base asset volume,
          10: Taker buy quote asset volume,
          11: Ignore
        ]
        """
        open_time_ms = int(raw[0])
        open_dt = datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)
        close_time_ms = int(raw[6])
        close_dt = datetime.fromtimestamp(close_time_ms / 1000.0, tz=timezone.utc)

        return cls(
            timestamp=open_dt,
            open=Decimal(str(raw[1])),
            high=Decimal(str(raw[2])),
            low=Decimal(str(raw[3])),
            close=Decimal(str(raw[4])),
            volume=Decimal(str(raw[5])),
            symbol=symbol,
            interval=interval,
            close_time=close_dt,
            quote_volume=Decimal(str(raw[7])) if len(raw) > 7 else None,
            trades_count=int(raw[8]) if len(raw) > 8 else None,
        )

    @classmethod
    def from_bybit_raw(
        cls,
        raw: list,
        symbol: str = "",
        interval: str = ""
    ) -> "Candle":
        """Construct Candle from Bybit v5 kline array format.

        Bybit format:
        [0: startTime, 1: openPrice, 2: highPrice, 3: lowPrice, 4: closePrice, 5: volume, 6: turnover]
        """
        open_time_ms = int(raw[0])
        open_dt = datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)
        step_ms = timeframe_to_milliseconds(interval) if interval else 0
        close_dt = datetime.fromtimestamp((open_time_ms + step_ms - 1) / 1000.0, tz=timezone.utc) if step_ms else None

        return cls(
            timestamp=open_dt,
            open=Decimal(str(raw[1])),
            high=Decimal(str(raw[2])),
            low=Decimal(str(raw[3])),
            close=Decimal(str(raw[4])),
            volume=Decimal(str(raw[5])),
            symbol=symbol,
            interval=interval,
            close_time=close_dt,
            quote_volume=Decimal(str(raw[6])) if len(raw) > 6 else None,
            trades_count=None,
        )

    @classmethod
    def from_db_row(
        cls,
        row: Union[sqlite3.Row, tuple, dict],
        symbol: str = "",
        interval: str = ""
    ) -> "Candle":
        """Construct Candle from SQLite query row."""
        if isinstance(row, sqlite3.Row) or isinstance(row, dict):
            sym = row["symbol"] if "symbol" in row.keys() else symbol
            inv = row["interval"] if "interval" in row.keys() else interval
            open_time_ms = int(row["open_time"])
            open_dt = datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)
            
            close_time_val = row["close_time"] if "close_time" in row.keys() else None
            close_dt = (
                datetime.fromtimestamp(int(close_time_val) / 1000.0, tz=timezone.utc)
                if close_time_val is not None
                else None
            )

            return cls(
                timestamp=open_dt,
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=Decimal(str(row["volume"])),
                symbol=sym,
                interval=inv,
                close_time=close_dt,
                quote_volume=Decimal(str(row["quote_volume"])) if ("quote_volume" in row.keys() and row["quote_volume"] is not None) else None,
                trades_count=int(row["trade_count"]) if ("trade_count" in row.keys() and row["trade_count"] is not None) else None,
            )
        else:
            # Tuple layout: (symbol, interval, open_time, open, high, low, close, volume, close_time, quote_volume, trade_count)
            sym = row[0]
            inv = row[1]
            open_time_ms = int(row[2])
            open_dt = datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)
            close_time_ms = int(row[8]) if len(row) > 8 and row[8] is not None else None
            close_dt = datetime.fromtimestamp(close_time_ms / 1000.0, tz=timezone.utc) if close_time_ms else None

            return cls(
                timestamp=open_dt,
                open=Decimal(str(row[3])),
                high=Decimal(str(row[4])),
                low=Decimal(str(row[5])),
                close=Decimal(str(row[6])),
                volume=Decimal(str(row[7])),
                symbol=sym,
                interval=inv,
                close_time=close_dt,
                quote_volume=Decimal(str(row[9])) if len(row) > 9 and row[9] is not None else None,
                trades_count=int(row[10]) if len(row) > 10 and row[10] is not None else None,
            )

    def to_db_tuple(self) -> Tuple[str, str, int, float, float, float, float, float, int, Optional[float], Optional[int]]:
        """Convert Candle to a tuple suitable for SQLite insertion."""
        return (
            self.symbol,
            self.interval,
            self.open_time_ms,
            float(self.open),
            float(self.high),
            float(self.low),
            float(self.close),
            float(self.volume),
            self.close_time_ms,
            float(self.quote_volume) if self.quote_volume is not None else None,
            self.trades_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candle to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open_time_ms": self.open_time_ms,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "symbol": self.symbol,
            "interval": self.interval,
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "close_time_ms": self.close_time_ms,
            "quote_volume": str(self.quote_volume) if self.quote_volume is not None else None,
            "trades_count": self.trades_count,
        }


# =====================================================================
# Ticker Model
# =====================================================================
class Ticker(BaseModel):
    """High-precision real-time ticker model."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    symbol: str = Field(description="Trading pair symbol (e.g. BTCEUR)")
    price: Decimal = Field(description="Latest trade / mark price")
    timestamp: datetime = Field(description="Timestamp of ticker quote in UTC")
    bid: Optional[Decimal] = Field(default=None, description="Best bid price")
    ask: Optional[Decimal] = Field(default=None, description="Best ask price")
    high_24h: Optional[Decimal] = Field(default=None, description="24-hour high price")
    low_24h: Optional[Decimal] = Field(default=None, description="24-hour low price")
    volume_24h: Optional[Decimal] = Field(default=None, description="24-hour base asset volume")
    quote_volume_24h: Optional[Decimal] = Field(default=None, description="24-hour quote volume")
    price_change_percent_24h: Optional[Decimal] = Field(
        default=None, description="24-hour percentage price change"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if isinstance(v, (int, float)):
            ts_sec = v / 1000.0 if v > 1e11 else float(v)
            return datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        if isinstance(v, str):
            try:
                dt = datetime.fromisoformat(v)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                val = float(v)
                ts_sec = val / 1000.0 if val > 1e11 else val
                return datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        return datetime.now(timezone.utc)

    @field_validator(
        "price", "bid", "ask", "high_24h", "low_24h", "volume_24h",
        "quote_volume_24h", "price_change_percent_24h", mode="before"
    )
    @classmethod
    def _validate_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None or v == "":
            return None
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    @classmethod
    def from_binance_ticker(cls, data: dict) -> "Ticker":
        """Construct Ticker from Binance /api/v3/ticker/price response."""
        return cls(
            symbol=data["symbol"],
            price=Decimal(str(data["price"])),
            timestamp=datetime.now(timezone.utc),
        )

    @classmethod
    def from_binance_24hr(cls, data: dict) -> "Ticker":
        """Construct full Ticker from Binance /api/v3/ticker/24hr response."""
        close_time = data.get("closeTime")
        ts = (
            datetime.fromtimestamp(int(close_time) / 1000.0, tz=timezone.utc)
            if close_time
            else datetime.now(timezone.utc)
        )
        return cls(
            symbol=data["symbol"],
            price=Decimal(str(data["lastPrice"])),
            timestamp=ts,
            bid=Decimal(str(data["bidPrice"])) if "bidPrice" in data else None,
            ask=Decimal(str(data["askPrice"])) if "askPrice" in data else None,
            high_24h=Decimal(str(data["highPrice"])) if "highPrice" in data else None,
            low_24h=Decimal(str(data["lowPrice"])) if "lowPrice" in data else None,
            volume_24h=Decimal(str(data["volume"])) if "volume" in data else None,
            quote_volume_24h=Decimal(str(data["quoteVolume"])) if "quoteVolume" in data else None,
            price_change_percent_24h=Decimal(str(data["priceChangePercent"])) if "priceChangePercent" in data else None,
        )


# =====================================================================
# OHLCV Series Collection Helper
# =====================================================================
class OHLCVData(BaseModel):
    """Container holding a chronological sequence of Candles with accessors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str
    interval: str
    candles: List[Candle] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.candles)

    @property
    def timestamps(self) -> List[datetime]:
        return [c.timestamp for c in self.candles]

    @property
    def opens(self) -> List[Decimal]:
        return [c.open for c in self.candles]

    @property
    def highs(self) -> List[Decimal]:
        return [c.high for c in self.candles]

    @property
    def lows(self) -> List[Decimal]:
        return [c.low for c in self.candles]

    @property
    def closes(self) -> List[Decimal]:
        return [c.close for c in self.candles]

    @property
    def volumes(self) -> List[Decimal]:
        return [c.volume for c in self.candles]

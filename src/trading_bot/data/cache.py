"""SQLite-backed Market Data Cache and Unified MarketDataService.

Provides high-speed local caching of candlestick data in SQLite (data/market_data.db),
enabling deterministic, offline backtests with zero redundant network requests.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from trading_bot.config import (
    DEFAULT_DB_PATH,
    normalize_symbol,
)
from trading_bot.data.binance_client import BinanceClient
from trading_bot.data.models import (
    Candle,
    Ticker,
    timeframe_to_milliseconds,
    timeframe_to_seconds,
)

logger = logging.getLogger(__name__)


# SQL Table and Index Definitions
CREATE_CANDLES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time INTEGER NOT NULL,
    quote_volume REAL,
    trade_count INTEGER,
    PRIMARY KEY (symbol, interval, open_time)
);
"""

CREATE_CANDLES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_candles_lookup 
ON candles (symbol, interval, open_time ASC);
"""


class CandleCache:
    """Persistent SQLite cache for market candles."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initialize SQLite database connection and schema.

        Args:
            db_path: Path to SQLite database file. Defaults to config.DEFAULT_DB_PATH.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new connection configured for WAL mode and performance."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize tables and indexes."""
        with self._get_connection() as conn:
            conn.execute(CREATE_CANDLES_TABLE_SQL)
            conn.execute(CREATE_CANDLES_INDEX_SQL)
            conn.commit()

    def save_candles(self, candles: List[Candle]) -> int:
        """Persist a batch of Candle objects into SQLite.

        Uses INSERT OR REPLACE to update existing candles or insert new ones.

        Args:
            candles: List of Candle instances.

        Returns:
            Number of rows inserted/updated.
        """
        if not candles:
            return 0

        tuples = [c.to_db_tuple() for c in candles]
        insert_sql = """
        INSERT OR REPLACE INTO candles (
            symbol, interval, open_time, open, high, low, close, volume, close_time, quote_volume, trade_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self._get_connection() as conn:
            conn.executemany(insert_sql, tuples)
            conn.commit()

        return len(tuples)

    def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[Union[datetime, int]] = None,
        end_time: Optional[Union[datetime, int]] = None,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """Query candles from SQLite matching symbol, interval, and time range.

        Args:
            symbol: Trading pair symbol (e.g. 'BTCEUR').
            interval: Timeframe (e.g. '15m', '1h').
            start_time: Inclusive start (datetime or Unix ms).
            end_time: Inclusive end (datetime or Unix ms).
            limit: Maximum candles to return.

        Returns:
            List of chronologically ordered Candle objects.
        """
        norm_sym = normalize_symbol(symbol)
        conditions = ["symbol = ?", "interval = ?"]
        params: List[Any] = [norm_sym, interval]

        if start_time is not None:
            start_ms = int(start_time.timestamp() * 1000) if isinstance(start_time, datetime) else int(start_time)
            conditions.append("open_time >= ?")
            params.append(start_ms)

        if end_time is not None:
            end_ms = int(end_time.timestamp() * 1000) if isinstance(end_time, datetime) else int(end_time)
            conditions.append("open_time <= ?")
            params.append(end_ms)

        where_clause = " AND ".join(conditions)
        if start_time is None and limit is not None:
            query = f"SELECT * FROM (SELECT * FROM candles WHERE {where_clause} ORDER BY open_time DESC LIMIT {int(limit)}) ORDER BY open_time ASC"
        else:
            query = f"SELECT * FROM candles WHERE {where_clause} ORDER BY open_time ASC"
            if limit is not None:
                query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [Candle.from_db_row(row, symbol=norm_sym, interval=interval) for row in rows]

    def get_cached_range(
        self,
        symbol: str,
        interval: str
    ) -> Optional[Tuple[datetime, datetime, int]]:
        """Return (earliest_dt, latest_dt, count) for cached candles of symbol and interval."""
        norm_sym = normalize_symbol(symbol)
        query = """
        SELECT MIN(open_time), MAX(open_time), COUNT(*)
        FROM candles
        WHERE symbol = ? AND interval = ?
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (norm_sym, interval))
            row = cursor.fetchone()

        if not row or row[0] is None or row[2] == 0:
            return None

        min_dt = datetime.fromtimestamp(row[0] / 1000.0, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(row[1] / 1000.0, tz=timezone.utc)
        count = row[2]
        return min_dt, max_dt, count

    def count(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> int:
        """Count cached candles, optionally filtered by symbol and interval."""
        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(normalize_symbol(symbol))
        if interval:
            conditions.append("interval = ?")
            params.append(interval)

        query = "SELECT COUNT(*) FROM candles"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            return row[0] if row else 0

    def has_coverage(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        tolerance_missing_pct: float = 0.05
    ) -> bool:
        """Check if cached candles sufficiently cover the requested time window."""
        norm_sym = normalize_symbol(symbol)
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        step_ms = timeframe_to_milliseconds(interval)

        query = """
        SELECT MIN(open_time), MAX(open_time), COUNT(*)
        FROM candles
        WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time <= ?
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (norm_sym, interval, start_ms, end_ms))
            row = cursor.fetchone()

        if not row or row[0] is None or row[2] == 0:
            return False

        min_open, max_open, count = row[0], row[1], row[2]

        # Check boundary coverage: earliest bar within 1 step of start_time, latest within 1 step of end_time
        if min_open > (start_ms + step_ms):
            return False
        if max_open < (end_ms - step_ms):
            return False

        # Check total count against theoretical expected count
        expected_bars = max(1, int((end_ms - start_ms) / step_ms))
        min_acceptable_bars = int(expected_bars * (1.0 - tolerance_missing_pct))

        return count >= min_acceptable_bars

    def clear(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> int:
        """Delete cached candles, optionally filtered by symbol and interval."""
        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(normalize_symbol(symbol))
        if interval:
            conditions.append("interval = ?")
            params.append(interval)

        query = "DELETE FROM candles"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount


# =====================================================================
# Unified MarketDataService
# =====================================================================
class MarketDataService:
    """Unified Market Data Service fulfilling the Project Interface Contract.

    Contract:
    `MarketDataService.get_historical_candles(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> List[Candle]`

    Checks local SQLite cache first to enable instantaneous offline deterministic backtesting.
    If missing or force_refresh=True, fetches missing data from Binance REST client and caches it.
    """

    def __init__(
        self,
        cache: Optional[CandleCache] = None,
        client: Optional[BinanceClient] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize MarketDataService.

        Args:
            cache: Pre-existing CandleCache instance, or None to create default.
            client: Pre-existing BinanceClient instance, or None to create default.
            db_path: Optional custom SQLite path if cache is None.
        """
        self.cache = cache or CandleCache(db_path=db_path)
        self.client = client or BinanceClient()

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        force_refresh: bool = False
    ) -> List[Candle]:
        """Retrieve historical candles for symbol and interval in [start_time, end_time].

        Checks local cache first. If cached data covers the range, returns cached bars.
        Otherwise fetches missing klines via REST API, caches them locally, and returns them.
        """
        norm_sym = normalize_symbol(symbol)
        now_utc = datetime.now(timezone.utc)
        step_ms = timeframe_to_milliseconds(interval)
        is_live_request = (end_time >= now_utc - timedelta(milliseconds=step_ms * 2))

        # Refresh recent live klines from exchange to guarantee SQLite market_data.db stays up to the minute
        if force_refresh or is_live_request:
            try:
                recent_fetched = self.client.fetch_klines(
                    symbol=norm_sym,
                    interval=interval,
                    limit=100,
                )
                if recent_fetched:
                    self.cache.save_candles(recent_fetched)
            except Exception as exc:
                logger.warning("Could not refresh live klines for %s from exchange: %s", norm_sym, exc)

        # Step 1: Check cache coverage
        if not force_refresh and self.cache.has_coverage(norm_sym, interval, start_time, end_time):
            cached_candles = self.cache.get_candles(norm_sym, interval, start_time, end_time)
            if cached_candles:
                logger.info(
                    "Cache hit: Loaded %d candles for %s %s from %s to %s",
                    len(cached_candles), norm_sym, interval, start_time, end_time
                )
                return cached_candles

        # Step 2: Fetch from exchange client
        logger.info(
            "Cache miss / refresh: Fetching %s %s candles from exchange (%s to %s)",
            norm_sym, interval, start_time, end_time
        )
        try:
            fetched_candles = self.client.fetch_klines(
                symbol=norm_sym,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
            )
            # Step 3: Persist newly fetched candles to SQLite
            if fetched_candles:
                saved_count = self.cache.save_candles(fetched_candles)
                logger.info("Saved %d newly fetched candles to local SQLite cache", saved_count)

            # Step 4: Return queried candles from cache to ensure consistent ordering & bounds
            results = self.cache.get_candles(norm_sym, interval, start_time, end_time)
            return results or fetched_candles

        except Exception as exc:
            # Fallback: If network is offline, attempt returning whatever cached data exists
            logger.warning("Network fetch failed (%s). Checking if any cached candles exist...", exc)
            fallback_candles = self.cache.get_candles(norm_sym, interval, start_time, end_time)
            if fallback_candles:
                logger.info("Returning %d fallback candles from cache despite network error", len(fallback_candles))
                return fallback_candles
            raise ConnectionError(
                f"Failed to fetch market data for {norm_sym} {interval} and no local cache was available: {exc}"
            ) from exc

    def get_ticker(self, symbol: str) -> Ticker:
        """Fetch current ticker price."""
        return self.client.get_ticker(symbol)

    def sync_historical_data(
        self,
        symbol: str,
        interval: str,
        days: int = 30
    ) -> int:
        """Pre-populate local SQLite cache with historical data for the last N days.

        Returns number of candles synchronized into SQLite.
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        candles = self.get_historical_candles(symbol, interval, start_time, end_time, force_refresh=True)
        return len(candles)

"""Binance Public REST Client for Historical OHLCV Klines and Market Data.

Features:
- Zero authentication required (pure public REST API endpoints)
- Automated pagination exceeding Binance's 1,000 bar limit (e.g. 30+ days of 15m/1h candles)
- Dynamic rate-limiting and exponential backoff retry logic
- Mirror URL fallback and Bybit public REST fallback for maximum resilience
- Converts responses directly to high-precision Decimal-based Candle objects
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from trading_bot.config import (
    BINANCE_BASE_URL,
    BINANCE_PUBLIC_API_MIRRORS,
    BINANCE_KLINES_ENDPOINT,
    BINANCE_TICKER_PRICE_ENDPOINT,
    BINANCE_TICKER_24HR_ENDPOINT,
    BINANCE_PING_ENDPOINT,
    BYBIT_BASE_URL,
    BYBIT_KLINES_ENDPOINT,
    BYBIT_TICKERS_ENDPOINT,
    normalize_symbol,
)
from trading_bot.data.models import (
    Candle,
    Ticker,
    timeframe_to_milliseconds,
    timeframe_to_seconds,
)

# Compatibility alias: ensure Ticker exposes last_price property
if not hasattr(Ticker, "last_price"):
    setattr(Ticker, "last_price", property(lambda self: self.price))

logger = logging.getLogger(__name__)


# Mapping for Bybit interval format
BYBIT_INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}


class BinanceClient:
    """Production-grade public market data client for Binance REST API."""

    def __init__(
        self,
        base_url: str = BINANCE_BASE_URL,
        mirrors: Optional[List[str]] = None,
        timeout: float = 10.0,
        rate_limit_delay: float = 0.05,
        max_retries: int = 3,
        fallback_to_bybit: bool = True,
        session: Optional[requests.Session] = None,
    ):
        """Initialize BinanceClient.

        Args:
            base_url: Primary Binance REST API base URL.
            mirrors: Secondary Binance public mirror URLs.
            timeout: HTTP request timeout in seconds.
            rate_limit_delay: Mandatory minimum delay between successive API calls.
            max_retries: Maximum retry attempts per request before failing or falling back.
            fallback_to_bybit: If True, falls back to Bybit public API if Binance fails.
            session: Optional pre-configured requests.Session.
        """
        self.base_url = base_url.rstrip("/")
        self.mirrors = mirrors or list(BINANCE_PUBLIC_API_MIRRORS)
        if self.base_url not in self.mirrors:
            self.mirrors.insert(0, self.base_url)

        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.fallback_to_bybit = fallback_to_bybit

        self._last_request_time: float = 0.0

        if session is not None:
            self.session = session
        else:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "TradingBot-MarketData/1.0",
                "Accept": "application/json",
            })
            # Configure transport adapter with pooling
            retries = Retry(
                total=2,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

    def _rate_limit(self) -> None:
        """Enforce spacing between successive requests to prevent exchange 429 errors."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _request_binance(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute GET request with automatic mirror rotation, backoff, and rate limiting."""
        params = params or {}
        last_exception = None

        # Try primary URL followed by mirrors
        for base in self.mirrors:
            url = f"{base}{endpoint}"
            for attempt in range(1, self.max_retries + 1):
                self._rate_limit()
                try:
                    response = self.session.get(url, params=params, timeout=self.timeout)
                    
                    # Handle rate limit (429) or IP ban (418)
                    if response.status_code in (429, 418):
                        retry_after = int(response.headers.get("Retry-After", 2 * attempt))
                        logger.warning(
                            "Binance rate limit %d on %s, sleeping for %ds (attempt %d/%d)",
                            response.status_code, url, retry_after, attempt, self.max_retries
                        )
                        time.sleep(retry_after)
                        continue

                    # Handle 5xx server errors
                    if response.status_code >= 500:
                        logger.warning(
                            "Binance server error %d on %s (attempt %d/%d)",
                            response.status_code, url, attempt, self.max_retries
                        )
                        time.sleep(0.5 * attempt)
                        continue

                    # Check for rate limit usage headers
                    used_weight = response.headers.get("x-mbx-used-weight-1m")
                    if used_weight and int(used_weight) > 1000:
                        logger.warning("Binance 1m weight high: %s/1200", used_weight)
                        time.sleep(0.2)

                    response.raise_for_status()
                    return response.json()

                except (requests.RequestException, ValueError) as exc:
                    last_exception = exc
                    logger.debug("Request failed on %s (attempt %d/%d): %s", url, attempt, self.max_retries, exc)
                    time.sleep(0.2 * attempt)

        # If Binance mirrors all failed
        raise ConnectionError(f"All Binance endpoints failed for {endpoint}. Last error: {last_exception}")

    # =================================================================
    # Public API Methods
    # =================================================================
    def ping(self) -> bool:
        """Check connection to Binance public REST API."""
        try:
            res = self._request_binance(BINANCE_PING_ENDPOINT)
            return res == {} or res is not None
        except Exception as exc:
            logger.warning("Ping failed: %s", exc)
            return False

    def get_ticker(self, symbol: str) -> Ticker:
        """Fetch real-time ticker price for symbol."""
        norm_sym = normalize_symbol(symbol)
        try:
            data = self._request_binance(BINANCE_TICKER_PRICE_ENDPOINT, {"symbol": norm_sym})
            return Ticker.from_binance_ticker(data)
        except Exception as exc:
            if self.fallback_to_bybit:
                logger.info("Binance get_ticker failed, falling back to Bybit: %s", exc)
                return self._get_ticker_bybit(norm_sym)
            raise

    def get_24h_ticker(self, symbol: str) -> Ticker:
        """Fetch 24-hour rolling window ticker statistics."""
        norm_sym = normalize_symbol(symbol)
        try:
            data = self._request_binance(BINANCE_TICKER_24HR_ENDPOINT, {"symbol": norm_sym})
            return Ticker.from_binance_24hr(data)
        except Exception as exc:
            if self.fallback_to_bybit:
                logger.info("Binance get_24h_ticker failed, falling back to Bybit: %s", exc)
                return self._get_ticker_bybit(norm_sym)
            raise

    def fetch_klines_batch(
        self,
        symbol: str,
        interval: str,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000
    ) -> List[Candle]:
        """Fetch a single batch of klines from Binance (up to 1,000 candles)."""
        norm_sym = normalize_symbol(symbol)
        params: Dict[str, Any] = {
            "symbol": norm_sym,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        raw_data = self._request_binance(BINANCE_KLINES_ENDPOINT, params=params)
        return [Candle.from_binance_raw(item, symbol=norm_sym, interval=interval) for item in raw_data]

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[Union[datetime, int]] = None,
        end_time: Optional[Union[datetime, int]] = None,
        limit: Optional[int] = None,
    ) -> List[Candle]:
        """Fetch historical klines with automatic pagination.

        Can fetch arbitrarily large ranges (e.g. 30, 60, 90+ days of 15m or 1h candles)
        by issuing sequential paginated requests of 1,000 bars each.

        Args:
            symbol: Trading pair symbol (e.g. 'BTCEUR' or 'BTC/EUR').
            interval: Candlestick timeframe (e.g. '15m', '1h', '1d').
            start_time: Start timestamp (datetime or Unix milliseconds).
            end_time: End timestamp (datetime or Unix milliseconds).
            limit: Maximum total candles to retrieve (if None, all candles in range).

        Returns:
            List of chronologically sorted Candle objects.
        """
        norm_sym = normalize_symbol(symbol)
        step_ms = timeframe_to_milliseconds(interval)

        # Normalize start and end times to Unix milliseconds
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if end_time is not None:
            if isinstance(end_time, datetime):
                end_ms = int(end_time.timestamp() * 1000)
            else:
                end_ms = int(end_time)
        else:
            end_ms = now_ms

        if start_time is not None:
            if isinstance(start_time, datetime):
                start_ms = int(start_time.timestamp() * 1000)
            else:
                start_ms = int(start_time)
        else:
            if limit is not None:
                start_ms = end_ms - (limit * step_ms)
            else:
                # Default lookback: 30 days
                start_ms = end_ms - (30 * 86400 * 1000)

        if start_ms > end_ms:
            raise ValueError(f"start_time ({start_ms}) must be <= end_time ({end_ms})")

        # Attempt fetching via Binance with pagination
        try:
            return self._paginate_binance(norm_sym, interval, start_ms, end_ms, limit, step_ms)
        except Exception as exc:
            if self.fallback_to_bybit:
                logger.warning("Binance fetch_klines failed (%s). Falling back to Bybit public API...", exc)
                return self._fetch_klines_bybit(norm_sym, interval, start_ms, end_ms, limit)
            raise

    def _paginate_binance(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: Optional[int],
        step_ms: int
    ) -> List[Candle]:
        """Internal pagination loop for Binance klines."""
        candles_dict: Dict[int, Candle] = {}
        current_start = start_ms

        while current_start <= end_ms:
            # Determine batch limit
            remaining = (limit - len(candles_dict)) if limit is not None else 1000
            if remaining <= 0:
                break
            batch_limit = min(1000, remaining)

            batch = self.fetch_klines_batch(
                symbol=symbol,
                interval=interval,
                start_time_ms=current_start,
                end_time_ms=end_ms,
                limit=batch_limit,
            )

            if not batch:
                break

            for candle in batch:
                if candle.open_time_ms <= end_ms:
                    candles_dict[candle.open_time_ms] = candle

            last_open = batch[-1].open_time_ms
            next_start = last_open + step_ms

            # Break if no progress made or reached end
            if next_start <= current_start or len(batch) < batch_limit:
                break

            current_start = next_start

        sorted_candles = [candles_dict[k] for k in sorted(candles_dict.keys())]
        if limit is not None and len(sorted_candles) > limit:
            return sorted_candles[-limit:]
        return sorted_candles

    # =================================================================
    # Bybit Public REST Fallback Implementation
    # =================================================================
    def _fetch_klines_bybit(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: Optional[int]
    ) -> List[Candle]:
        """Fallback kline fetcher using Bybit v5 public market endpoint."""
        bybit_interval = BYBIT_INTERVAL_MAP.get(interval)
        if not bybit_interval:
            raise ValueError(f"Interval {interval} not supported by Bybit fallback")

        url = f"{BYBIT_BASE_URL}{BYBIT_KLINES_ENDPOINT}"
        step_ms = timeframe_to_milliseconds(interval)
        candles_dict: Dict[int, Candle] = {}
        current_start = start_ms

        while current_start <= end_ms:
            remaining = (limit - len(candles_dict)) if limit is not None else 1000
            if remaining <= 0:
                break
            batch_limit = min(1000, remaining)

            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": bybit_interval,
                "start": current_start,
                "end": end_ms,
                "limit": batch_limit,
            }
            self._rate_limit()
            res = self.session.get(url, params=params, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")

            items = data.get("result", {}).get("list", [])
            if not items:
                break

            # Bybit returns newest first, so we reverse it
            batch = [
                Candle.from_bybit_raw(item, symbol=symbol, interval=interval)
                for item in reversed(items)
            ]

            for candle in batch:
                if candle.open_time_ms <= end_ms and candle.open_time_ms >= start_ms:
                    candles_dict[candle.open_time_ms] = candle

            last_open = batch[-1].open_time_ms
            next_start = last_open + step_ms
            if next_start <= current_start or len(items) < batch_limit:
                break
            current_start = next_start

        sorted_candles = [candles_dict[k] for k in sorted(candles_dict.keys())]
        if limit is not None and len(sorted_candles) > limit:
            return sorted_candles[-limit:]
        return sorted_candles

    def _get_ticker_bybit(self, symbol: str) -> Ticker:
        """Fallback live ticker using Bybit v5 public endpoint."""
        url = f"{BYBIT_BASE_URL}{BYBIT_TICKERS_ENDPOINT}"
        params = {"category": "spot", "symbol": symbol}
        self._rate_limit()
        try:
            res = self.session.get(url, params=params, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")

            items = data.get("result", {}).get("list", [])
            if not items:
                raise ValueError(f"Symbol {symbol} not found on Bybit")

            item = items[0]
            return Ticker(
                symbol=symbol,
                price=Decimal(str(item["lastPrice"])),
                timestamp=datetime.now(timezone.utc),
                bid=Decimal(str(item["bid1Price"])) if item.get("bid1Price") else None,
                ask=Decimal(str(item["ask1Price"])) if item.get("ask1Price") else None,
                high_24h=Decimal(str(item["highPrice24h"])) if item.get("highPrice24h") else None,
                low_24h=Decimal(str(item["lowPrice24h"])) if item.get("lowPrice24h") else None,
                volume_24h=Decimal(str(item["volume24h"])) if item.get("volume24h") else None,
                price_change_percent_24h=Decimal(str(item["price24hPcnt"])) * Decimal("100") if item.get("price24hPcnt") else None,
            )
        except (requests.exceptions.RequestException, Exception) as exc:
            raise ConnectionError(f"All Binance and Bybit endpoints failed: {exc}") from exc

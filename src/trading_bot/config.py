"""Configuration module for the automated trading bot.

Defines constants, exchange parameters, default symbols, fees, slippage,
minNotional thresholds, and filesystem paths.
"""

from decimal import Decimal
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

# =====================================================================
# Filesystem Paths
# =====================================================================
# Project root directory: .../trading_bot
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DEFAULT_DB_PATH: Path = DATA_DIR / "market_data.db"
DEFAULT_TRADES_CSV_PATH: Path = DATA_DIR / "trades.csv"
STATE_DIR: Path = PROJECT_ROOT / "state"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure essential runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Core Capital & Milestone Constants
# =====================================================================
INITIAL_CAPITAL_EUR: Decimal = Decimal("10.00")
TARGET_CAPITAL_EUR: Decimal = Decimal("10000.00")
BASE_CURRENCY: str = "EUR"

# =====================================================================
# Symbols & Timeframes
# =====================================================================
DEFAULT_SYMBOLS: List[str] = ["BTCEUR", "ETHEUR", "SOLEUR"]
DISPLAY_SYMBOLS: List[str] = ["BTC/EUR", "ETH/EUR", "SOL/EUR"]

DEFAULT_TIMEFRAME: str = "1h"
FALLBACK_TIMEFRAME: str = "15m"
SUPPORTED_TIMEFRAMES: List[str] = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]

# =====================================================================
# Exchange Parameters & Micro-Capital Constraints
# =====================================================================
# Binance (Secondary for 10€ live trading due to 5.00€ minNotional, Primary for historical OHLCV)
BINANCE_BASE_URL: str = "https://api.binance.com"
BINANCE_PUBLIC_API_MIRRORS: List[str] = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]
BINANCE_KLINES_ENDPOINT: str = "/api/v3/klines"
BINANCE_TICKER_PRICE_ENDPOINT: str = "/api/v3/ticker/price"
BINANCE_TICKER_24HR_ENDPOINT: str = "/api/v3/ticker/24hr"
BINANCE_PING_ENDPOINT: str = "/api/v3/ping"

BINANCE_MIN_NOTIONAL: Decimal = Decimal("5.00")  # Minimum order notional in EUR on Binance
BINANCE_DEFAULT_MAKER_FEE: Decimal = Decimal("0.0010")  # 0.10% (VIP0)
BINANCE_DEFAULT_TAKER_FEE: Decimal = Decimal("0.0010")  # 0.10% (VIP0)
BINANCE_DEFAULT_SLIPPAGE: Decimal = Decimal("0.0005")   # 0.05% (5 bps)

# Bybit (Primary live trading exchange for 10€ micro-capital: 1.00€ minNotional)
BYBIT_BASE_URL: str = "https://api.bybit.com"
BYBIT_KLINES_ENDPOINT: str = "/v5/market/kline"
BYBIT_TICKERS_ENDPOINT: str = "/v5/market/tickers"
BYBIT_MIN_NOTIONAL: Decimal = Decimal("1.00")  # Minimum order notional in EUR on Bybit
BYBIT_DEFAULT_FEE_RATE: Decimal = Decimal("0.0010")  # 0.10%
BYBIT_DEFAULT_SLIPPAGE: Decimal = Decimal("0.0005")   # 0.05%

# Default Global Execution Assumptions
DEFAULT_FEE_RATE: Decimal = Decimal("0.0010")  # 0.10% per trade (0.20% round-trip)
DEFAULT_SLIPPAGE: Decimal = Decimal("0.0005")  # 0.05% per trade
DEFAULT_MIN_NOTIONAL: Decimal = Decimal("5.00") # Binance default

# Buffer to guarantee orders do not fail due to cash exhaustion from fees
CASH_FEE_BUFFER: Decimal = Decimal("0.0020")  # 0.20% buffer reserve

# =====================================================================
# Symbol Normalization Helpers
# =====================================================================
def normalize_symbol(symbol: str) -> str:
    """Normalize symbol string to uppercase alphanumeric (e.g. 'BTC/EUR' -> 'BTCEUR')."""
    if not symbol:
        return ""
    cleaned = symbol.replace("/", "").replace("-", "").replace("_", "").strip().upper()
    return cleaned


def to_display_symbol(symbol: str) -> str:
    """Convert normalized symbol to human-readable pair (e.g. 'BTCEUR' -> 'BTC/EUR')."""
    norm = normalize_symbol(symbol)
    for quote in ["EUR", "USDT", "USD", "BTC", "ETH"]:
        if norm.endswith(quote) and len(norm) > len(quote):
            base = norm[:-len(quote)]
            return f"{base}/{quote}"
    return norm


# =====================================================================
# Settings Configuration Model
# =====================================================================
class BotConfig(BaseModel):
    """Pydantic configuration model holding active bot settings."""

    # Capital & Currency
    initial_capital: Decimal = Field(default=INITIAL_CAPITAL_EUR, description="Starting capital in EUR")
    target_capital: Decimal = Field(default=TARGET_CAPITAL_EUR, description="Target compounding capital in EUR")
    quote_currency: str = Field(default=BASE_CURRENCY, description="Account quote currency")

    # Trading symbols & timeframe
    symbols: List[str] = Field(default_factory=lambda: list(DEFAULT_SYMBOLS), description="Active trading pairs")
    timeframe: str = Field(default=DEFAULT_TIMEFRAME, description="Active candle timeframe (e.g. '1h', '15m')")

    # Execution parameters
    fee_rate: Decimal = Field(default=DEFAULT_FEE_RATE, description="Taker fee rate (0.0010 = 0.10%)")
    slippage: Decimal = Field(default=DEFAULT_SLIPPAGE, description="Model slippage (0.0005 = 0.05%)")
    min_notional: Decimal = Field(default=DEFAULT_MIN_NOTIONAL, description="Minimum order size in EUR")

    # Storage paths
    db_path: Path = Field(default=DEFAULT_DB_PATH, description="Path to SQLite market data cache")
    trades_csv_path: Path = Field(default=DEFAULT_TRADES_CSV_PATH, description="Path to CSV trade execution log")
    state_dir: Path = Field(default=STATE_DIR, description="Directory for ledger persistence")
    logs_dir: Path = Field(default=LOGS_DIR, description="Directory for audit logs")

    # Network / Provider
    binance_base_url: str = Field(default=BINANCE_BASE_URL, description="Binance REST API base URL")
    request_timeout: float = Field(default=10.0, description="HTTP request timeout in seconds")
    rate_limit_delay: float = Field(default=0.05, description="Delay between pagination requests in seconds")

    model_config = ConfigDict(arbitrary_types_allowed=True)

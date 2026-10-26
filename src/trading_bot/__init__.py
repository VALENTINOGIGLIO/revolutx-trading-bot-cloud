"""Autonomous Trading Bot for Micro-Capital Scale-Up (10€ -> 10,000€)."""

__version__ = "0.1.0"

from trading_bot.config import BotConfig
from trading_bot.data.models import Candle, Ticker, Timeframe
from trading_bot.data.binance_client import BinanceClient
from trading_bot.data.cache import CandleCache, MarketDataService

import os
from pathlib import Path

# Extend package __path__ with cli directory for direct python -m trading_bot.backtest support
_cli_path = str(Path(__file__).resolve().parent / "cli")
if _cli_path not in __path__:
    __path__.append(_cli_path)

__all__ = [
    "__version__",
    "BotConfig",
    "Candle",
    "Ticker",
    "Timeframe",
    "BinanceClient",
    "CandleCache",
    "MarketDataService",
]

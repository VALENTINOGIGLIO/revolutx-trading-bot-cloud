"""Market Data Management CLI Command.

Inspects, pre-fetches, and synchronizes historical candlestick data
from public exchange REST endpoints into the local SQLite cache.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from trading_bot.config import (
    DEFAULT_DB_PATH,
    DEFAULT_TIMEFRAME,
    normalize_symbol,
    to_display_symbol,
)
from trading_bot.data.cache import CandleCache, MarketDataService

logger = logging.getLogger("data_cli")

EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CLI_USAGE_ERROR = 2
EXIT_DATA_FETCH_ERROR = 3


def inspect_cache(
    cache: CandleCache,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
) -> Dict[str, Any]:
    """Gather diagnostic metadata from local SQLite cache."""
    norm_sym = normalize_symbol(symbol) if symbol else None
    with cache._get_connection() as conn:
        query = """
        SELECT symbol, interval, COUNT(*), MIN(open_time), MAX(open_time)
        FROM candles
        """
        params = []
        conditions = []
        if norm_sym:
            conditions.append("symbol = ?")
            params.append(norm_sym)
        if interval:
            conditions.append("interval = ?")
            params.append(interval)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY symbol, interval ORDER BY symbol ASC, interval ASC"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

    entries = []
    total_candles = 0

    for r in rows:
        sym, inv, count, min_ms, max_ms = r[0], r[1], r[2], r[3], r[4]
        min_dt = datetime.fromtimestamp(min_ms / 1000.0, tz=timezone.utc).isoformat() if min_ms else None
        max_dt = datetime.fromtimestamp(max_ms / 1000.0, tz=timezone.utc).isoformat() if max_ms else None
        total_candles += count
        entries.append({
            "symbol": sym,
            "display_symbol": to_display_symbol(sym),
            "interval": inv,
            "candle_count": count,
            "earliest_utc": min_dt,
            "latest_utc": max_dt,
        })

    return {
        "status": "success",
        "database_path": str(cache.db_path),
        "total_candles_cached": total_candles,
        "datasets": entries,
    }


def sync_data(
    service: MarketDataService,
    symbol: str = "BTC/EUR",
    interval: str = "1h",
    days: int = 30,
) -> Dict[str, Any]:
    """Fetch and sync historical candles into SQLite cache."""
    norm_sym = normalize_symbol(symbol)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    candles = service.get_historical_candles(
        symbol=norm_sym,
        interval=interval,
        start_time=start_dt,
        end_time=end_dt,
        force_refresh=True,
    )

    return {
        "status": "success",
        "symbol": norm_sym,
        "display_symbol": to_display_symbol(norm_sym),
        "interval": interval,
        "days": days,
        "start_utc": start_dt.isoformat(),
        "end_utc": end_dt.isoformat(),
        "synced_candles": len(candles),
    }


def clear_cache(
    cache: CandleCache,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
) -> Dict[str, Any]:
    """Clear candles from cache."""
    deleted = cache.clear(symbol=symbol, interval=interval)
    return {
        "status": "success",
        "deleted_count": deleted,
        "filter_symbol": symbol,
        "filter_interval": interval,
    }


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for market data management."""
    parser = argparse.ArgumentParser(
        prog="trading_bot.data",
        description="Autonomous Compounding Trading Bot - Market Data Management",
    )

    parser.add_argument(
        "action",
        nargs="?",
        default="inspect",
        choices=["inspect", "sync", "clear"],
        help="Action to perform: inspect (default), sync, or clear",
    )
    parser.add_argument(
        "--action",
        dest="action_flag",
        choices=["inspect", "sync", "clear"],
        default=None,
        help="Alternative flag to specify action",
    )
    parser.add_argument(
        "--pair",
        "--symbol",
        dest="pair",
        type=str,
        default="BTC/EUR",
        help="Trading pair symbol (default: BTC/EUR)",
    )
    parser.add_argument(
        "--timeframe",
        "--interval",
        dest="timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help="Candle interval (default: 1h)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to sync (default: 30)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom path to SQLite database cache",
    )
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return EXIT_CLI_USAGE_ERROR if exc.code != 0 else EXIT_SUCCESS

    action = parsed.action_flag or parsed.action
    cache = CandleCache(db_path=parsed.db_path) if parsed.db_path else CandleCache()
    service = MarketDataService(cache=cache)

    try:
        if action == "inspect":
            result = inspect_cache(cache, symbol=parsed.pair if parsed.pair != "BTC/EUR" else None, interval=parsed.timeframe if parsed.timeframe != DEFAULT_TIMEFRAME else None)
        elif action == "sync":
            result = sync_data(service, symbol=parsed.pair, interval=parsed.timeframe, days=parsed.days)
        elif action == "clear":
            result = clear_cache(cache, symbol=parsed.pair, interval=parsed.timeframe)
        else:
            sys.stderr.write(f"Unknown action: {action}\n")
            return EXIT_CLI_USAGE_ERROR
    except Exception as exc:
        sys.stderr.write(f"Data operation failed: {exc}\n")
        return EXIT_DATA_FETCH_ERROR

    if parsed.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("             MARKET DATA CACHE REPORT             ")
        print("=" * 70)
        if action == "inspect":
            print(f"Database Path: {result['database_path']}")
            print(f"Total Cached Candles: {result['total_candles_cached']}")
            print("-" * 70)
            if not result["datasets"]:
                print("  No cached datasets found.")
            else:
                print(f"{'Symbol':<12} {'Interval':<10} {'Bars':<8} {'Earliest (UTC)':<22} {'Latest (UTC)':<22}")
                print("-" * 70)
                for ds in result["datasets"]:
                    e_str = ds["earliest_utc"][:16] if ds["earliest_utc"] else "-"
                    l_str = ds["latest_utc"][:16] if ds["latest_utc"] else "-"
                    print(f"{ds['display_symbol']:<12} {ds['interval']:<10} {ds['candle_count']:<8} {e_str:<22} {l_str:<22}")
        elif action == "sync":
            print(f"Successfully synced {result['synced_candles']} candles for {result['display_symbol']} ({result['interval']})")
            print(f"Period: {result['start_utc'][:10]} to {result['end_utc'][:10]} ({result['days']} days)")
        elif action == "clear":
            print(f"Deleted {result['deleted_count']} cached candles.")
        print("=" * 70)

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())

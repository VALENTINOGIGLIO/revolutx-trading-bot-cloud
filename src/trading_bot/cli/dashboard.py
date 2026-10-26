"""Terminal CLI Command for Real-Time Web Dashboard.

Runs the standalone dashboard server:
    python -m trading_bot dashboard --host 127.0.0.1 --port 8000 --trades-csv data/trades.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from typing import List, Optional

from trading_bot.config import DEFAULT_DB_PATH, DEFAULT_TRADES_CSV_PATH
from trading_bot.dashboard.server import DashboardServer

logger = logging.getLogger("DashboardCLI")

EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CLI_USAGE_ERROR = 2


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for running the trading dashboard web server."""
    parser = argparse.ArgumentParser(
        prog="trading_bot dashboard",
        description="Autonomous Trading Bot - Real-Time Web Dashboard & Visualization",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--trades-csv",
        type=str,
        default=str(DEFAULT_TRADES_CSV_PATH),
        help=f"Path to CSV trade execution log (default: {DEFAULT_TRADES_CSV_PATH})",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite market data cache (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/EUR",
        help="Default market symbol to visualize (default: BTC/EUR)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open default browser to dashboard URL on start",
    )

    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return EXIT_CLI_USAGE_ERROR if exc.code != 0 else EXIT_SUCCESS

    print("=" * 70)
    print("       AUTONOMOUS TRADING BOT - REAL-TIME WEB DASHBOARD SERVER       ")
    print("=" * 70)
    print(f"Host:       {parsed.host}")
    print(f"Port:       {parsed.port}")
    print(f"Symbol:     {parsed.symbol}")
    print(f"Trades CSV: {parsed.trades_csv}")
    print(f"Market DB:  {parsed.db_path}")
    print(f"URL:        http://{parsed.host}:{parsed.port}")
    print("=" * 70)

    try:
        server = DashboardServer(
            host=parsed.host,
            port=parsed.port,
            trades_csv=parsed.trades_csv,
            db_path=parsed.db_path,
            symbol=parsed.symbol,
        )

        if parsed.open_browser:
            webbrowser.open(f"http://{parsed.host}:{parsed.port}")

        server.run()
        return EXIT_SUCCESS
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard server stopped by user.")
        return EXIT_SUCCESS
    except Exception as exc:
        sys.stderr.write(f"Dashboard server runtime error: {exc}\n")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())

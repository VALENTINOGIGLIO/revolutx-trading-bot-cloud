"""Unified Command-Line Interface Dispatcher for Autonomous Trading Bot.

Allows executing the bot via:
    python -m trading_bot [backtest|paper|data|notify] [OPTIONS]
"""

from __future__ import annotations

import sys
from typing import List, Optional

from trading_bot.cli.backtest import main as backtest_main
from trading_bot.cli.dashboard import main as dashboard_main
from trading_bot.cli.data import main as data_main
from trading_bot.cli.notify import main as notify_main
from trading_bot.cli.paper import main as paper_main

COMMAND_DISPATCHER = {
    "backtest": backtest_main,
    "paper": paper_main,
    "data": data_main,
    "notify": notify_main,
    "dashboard": dashboard_main,
}

HELP_BANNER = """
Autonomous Compounding Trading Bot (10€ -> 10,000€)
Unified CLI Dispatcher

Usage:
    python -m trading_bot <command> [options]

Available Commands:
    backtest    Run deterministic historical backtesting on market data
    paper       Execute live paper trading simulation loop with real ticks
    data        Manage, inspect, and sync historical candlestick cache
    notify      Test and verify notification channels (Console, JSON, Telegram)
    dashboard   Run real-time local web dashboard and chart visualization

Options:
    -h, --help  Show this help message and exit
    --version   Show program version

Examples:
    python -m trading_bot backtest --start 2026-08-01 --end 2026-09-01 --pair BTC/EUR
    python -m trading_bot paper --pair BTC/EUR --iterations 10
    python -m trading_bot dashboard --port 8000
    python -m trading_bot data inspect
    python -m trading_bot notify --channel console
"""


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for unified CLI dispatcher."""
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(HELP_BANNER.strip())
        return 0

    if args[0] == "--version":
        from trading_bot import __version__
        print(f"trading_bot version {__version__}")
        return 0

    command = args[0].lower().strip()
    sub_args = args[1:]

    if command in COMMAND_DISPATCHER:
        handler = COMMAND_DISPATCHER[command]
        return handler(sub_args)
    else:
        sys.stderr.write(
            f"Error: Unknown command '{command}'.\n"
            f"Available commands: {', '.join(COMMAND_DISPATCHER.keys())}\n"
            f"Run 'python -m trading_bot --help' for usage.\n"
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())

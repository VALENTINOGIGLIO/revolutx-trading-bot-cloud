"""Terminal CLI Subsystem for Autonomous Trading Bot.

Provides command-line interfaces for:
- Backtesting historical performance (backtest.py)
- Real-time paper trading simulation (paper.py)
- Historical market data management and cache inspection (data.py)
- Notification telemetry testing (notify.py)
"""

from typing import Any

__all__ = [
    "backtest_main",
    "run_backtest",
    "paper_main",
    "run_paper",
    "data_main",
    "notify_main",
]


def __getattr__(name: str) -> Any:
    """Lazy export of CLI entrypoints to avoid circular import warnings in runpy."""
    if name in ("backtest_main", "run_backtest"):
        from trading_bot.cli.backtest import main as backtest_main, run_backtest
        return backtest_main if name == "backtest_main" else run_backtest
    elif name in ("paper_main", "run_paper"):
        from trading_bot.cli.paper import main as paper_main, run_paper
        return paper_main if name == "paper_main" else run_paper
    elif name == "data_main":
        from trading_bot.cli.data import main as data_main
        return data_main
    elif name == "notify_main":
        from trading_bot.cli.notify import main as notify_main
        return notify_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

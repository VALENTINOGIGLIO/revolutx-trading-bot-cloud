"""AMV-TrendBreak Strategy Module.

Re-exports the core AMVTrendBreakStrategy and Walk-Forward Anti-Overfitting profiles.
"""

from trading_bot.strategy.amv_strategy import (
    AMVTrendBreakStrategy,
    PositionState,
    OrderSide,
    OrderType,
    OrderSignal,
    PROFILE_GLOBAL_PARETO_OPTIMAL,
    PROFILE_4H_PLATEAU_WINNER,
    PROFILE_V2_BASELINE,
)

__all__ = [
    "AMVTrendBreakStrategy",
    "PositionState",
    "OrderSide",
    "OrderType",
    "OrderSignal",
    "PROFILE_GLOBAL_PARETO_OPTIMAL",
    "PROFILE_4H_PLATEAU_WINNER",
    "PROFILE_V2_BASELINE",
]

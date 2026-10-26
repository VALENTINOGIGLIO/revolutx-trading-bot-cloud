"""
Simulation & Ledger Subsystem for Autonomous Trading Bot.

Provides:
- VirtualLedger: Double-entry portfolio balance and position ledger.
- ExecutionSimulator: Realistic exchange matching engine with fees, slippage, and minNotional.
- PerformanceMetrics: Quantitative risk-adjusted performance evaluation suite.
"""

from trading_bot.simulation.ledger import (
    InsufficientFundsError,
    InsufficientPositionError,
    InvalidOrderError,
    Position,
    SimulationError,
    TradeRecord,
    VirtualLedger,
)
from trading_bot.simulation.metrics import PerformanceMetrics
from trading_bot.simulation.simulator import (
    EXCHANGE_MIN_NOTIONAL,
    ExecutionResult,
    ExecutionSimulator,
    OrderSide,
    OrderSignal,
    OrderType,
)

from trading_bot.simulation.revolutx_live import (
    RevolutXLiveExecutor,
    to_revolut_symbol,
    from_revolut_symbol,
)

__all__ = [
    "VirtualLedger",
    "Position",
    "TradeRecord",
    "ExecutionSimulator",
    "ExecutionResult",
    "OrderSignal",
    "OrderSide",
    "OrderType",
    "PerformanceMetrics",
    "SimulationError",
    "InsufficientFundsError",
    "InsufficientPositionError",
    "InvalidOrderError",
    "EXCHANGE_MIN_NOTIONAL",
    "RevolutXLiveExecutor",
    "to_revolut_symbol",
    "from_revolut_symbol",
]

"""Strategy subsystem for automated trading bot.

Exposes:
- Technical Indicators: EMA, ATR, Keltner Channels, ADX, Choppiness Index, RSI
- 3-Phase Compounding Position Sizer: Micro-cap acceleration (10€-100€), Fixed fractional (100€-1,000€), Quarter-Kelly (1,000€-10,000€)
- Downside Risk Guards: 8% daily rolling drawdown circuit breaker, consecutive loss throttle
- AMVTrendBreakStrategy: Adaptive Momentum & Volatility Breakout strategy
"""

from .indicators import (
    ADXResult,
    BollingerBands,
    IndicatorSnapshot,
    KeltnerChannels,
    calculate_adx,
    calculate_atr,
    calculate_bbw,
    calculate_bollinger_bands,
    calculate_choppiness,
    calculate_ema,
    calculate_keltner_channels,
    calculate_rsi,
    calculate_true_range,
    compute_all_indicators,
    compute_indicators_from_candles,
)
from .compounding import (
    CompoundingPhase,
    CompoundingPositionSizer,
    PositionSizeResult,
    RiskCheckResult,
    RiskGuard,
)
from .amv_strategy import (
    AMVTrendBreakStrategy,
    OrderSide,
    OrderSignal,
    OrderType,
    PositionState,
)
from .regime import (
    MarketRegime,
    MarketRegimeFilter,
    RegimeSnapshot,
)
from .grid_strategy import (
    GridLevel,
    MicroGridStrategy,
)
from .orchestrator import (
    MultiRegimeStrategyOrchestrator,
)

__all__ = [
    # Indicators
    "ADXResult",
    "BollingerBands",
    "IndicatorSnapshot",
    "KeltnerChannels",
    "calculate_adx",
    "calculate_atr",
    "calculate_bbw",
    "calculate_bollinger_bands",
    "calculate_choppiness",
    "calculate_ema",
    "calculate_keltner_channels",
    "calculate_rsi",
    "calculate_true_range",
    "compute_all_indicators",
    "compute_indicators_from_candles",
    # Compounding & Risk
    "CompoundingPhase",
    "CompoundingPositionSizer",
    "PositionSizeResult",
    "RiskCheckResult",
    "RiskGuard",
    # Strategy
    "AMVTrendBreakStrategy",
    "OrderSide",
    "OrderSignal",
    "OrderType",
    "PositionState",
    # Regime
    "MarketRegime",
    "MarketRegimeFilter",
    "RegimeSnapshot",
    # MicroGrid
    "GridLevel",
    "MicroGridStrategy",
    # Orchestrator
    "MultiRegimeStrategyOrchestrator",
]


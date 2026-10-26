"""Market Regime Filter with Schmitt-Trigger Dual-Threshold Hysteresis.

Classifies market state into TRENDING vs RANGING regimes using multi-indicator confirmation:
- Average Directional Index (ADX)
- Choppiness Index (CHOP)
- Bollinger Band Width (BBW) expansion vs compression

Uses dual-threshold Schmitt trigger hysteresis with K=2 bar persistence to eliminate
whipsaw transitions between Micro-Grid and Trend-Breakout strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from .indicators import compute_indicators_from_candles


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING = "TRENDING"
    RANGING = "RANGING"


@dataclass(frozen=True)
class RegimeSnapshot:
    """Immutable snapshot of the market regime filter state at a given candle."""
    regime: MarketRegime
    score: float                # Normalized trend conviction score [0.0, 1.0]
    adx: float                  # ADX 14 value
    chop: float                 # Choppiness Index 14 value
    bbw: float                  # Bollinger Band Width
    bbw_sma: float              # Moving average of BBW
    bars_in_regime: int         # Consecutive bars spent in current regime
    transition_occurred: bool = False
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketRegimeFilter:
    """Overarching Market Regime Filter using ADX, Choppiness Index, and BBW.

    Features Schmitt-trigger dual-threshold hysteresis with multi-bar persistence (K=2)
    to prevent flip-flopping between Grid and Trending strategies.
    """
    def __init__(
        self,
        adx_trend_threshold: float = 24.0,
        adx_range_threshold: float = 20.0,
        chop_trend_threshold: float = 50.0,
        chop_range_threshold: float = 56.0,
        bbw_period: int = 20,
        persistence_bars: int = 2,
    ) -> None:
        self.adx_trend_th = adx_trend_threshold
        self.adx_range_th = adx_range_threshold
        self.chop_trend_th = chop_trend_threshold
        self.chop_range_th = chop_range_threshold
        self.bbw_period = bbw_period
        self.persistence_bars = persistence_bars

        self._current_regime: MarketRegime = MarketRegime.RANGING
        self._consecutive_trend_signals: int = 0
        self._consecutive_range_signals: int = 0
        self._bars_in_regime: int = 0

    @property
    def current_regime(self) -> MarketRegime:
        """Current market regime state."""
        return self._current_regime

    def reset(self) -> None:
        """Reset regime filter state machine."""
        self._current_regime = MarketRegime.RANGING
        self._consecutive_trend_signals = 0
        self._consecutive_range_signals = 0
        self._bars_in_regime = 0

    def evaluate(self, candle: Any, history: Sequence[Any]) -> RegimeSnapshot:
        """Evaluate candle and history, updating regime state with hysteresis.

        Args:
            candle: The current Candle or candle dict.
            history: Sequence of previous Candle objects.

        Returns:
            RegimeSnapshot containing regime classification and quantitative indicators.
        """
        candle_ts = getattr(candle, "timestamp", None)
        if candle_ts is None and isinstance(candle, dict):
            candle_ts = candle.get("timestamp")

        # 1. Warm-up verification
        warmup_required = max(20, self.bbw_period)
        if len(history) < warmup_required:
            self._bars_in_regime += 1
            return RegimeSnapshot(
                regime=self._current_regime,
                score=0.0,
                adx=0.0,
                chop=50.0,
                bbw=0.0,
                bbw_sma=0.0,
                bars_in_regime=self._bars_in_regime,
                transition_occurred=False,
                timestamp=candle_ts,
            )

        # 2. Indicator computation over sliding window
        recent = list(history[-100:]) + [candle]
        df = compute_indicators_from_candles(recent)
        if df.empty or len(df) < 2:
            self._bars_in_regime += 1
            return RegimeSnapshot(
                regime=self._current_regime,
                score=0.0,
                adx=0.0,
                chop=50.0,
                bbw=0.0,
                bbw_sma=0.0,
                bars_in_regime=self._bars_in_regime,
                transition_occurred=False,
                timestamp=candle_ts,
            )

        last = df.iloc[-1]
        adx_val = float(last.get("adx_14", 0.0))
        chop_val = float(last.get("chop_14", 50.0))
        bbw_val = float(last.get("bbw", 0.0))
        bbw_sma_val = float(last.get("bbw_sma", bbw_val))

        # 3. Raw condition evaluation
        trend_trigger = (
            adx_val >= self.adx_trend_th
            and chop_val <= self.chop_trend_th
            and bbw_val >= bbw_sma_val
        )
        range_trigger = (
            adx_val < self.adx_range_th
            or chop_val >= self.chop_range_th
        )

        if trend_trigger:
            self._consecutive_trend_signals += 1
            self._consecutive_range_signals = 0
        elif range_trigger:
            self._consecutive_range_signals += 1
            self._consecutive_trend_signals = 0
        else:
            # Deadband region (retain previous counter state or reset)
            self._consecutive_trend_signals = 0
            self._consecutive_range_signals = 0

        # 4. Hysteresis state transition with persistence (K=2)
        transition = False
        if self._current_regime == MarketRegime.RANGING:
            if self._consecutive_trend_signals >= self.persistence_bars:
                self._current_regime = MarketRegime.TRENDING
                self._bars_in_regime = 0
                transition = True
        elif self._current_regime == MarketRegime.TRENDING:
            if self._consecutive_range_signals >= self.persistence_bars:
                self._current_regime = MarketRegime.RANGING
                self._bars_in_regime = 0
                transition = True

        self._bars_in_regime += 1

        # Normalized trend conviction score [0, 1]
        score_adx = min(1.0, max(0.0, (adx_val - 15.0) / 15.0))
        score_chop = min(1.0, max(0.0, (61.8 - chop_val) / 23.6))
        composite_score = float(np.clip(0.5 * score_adx + 0.5 * score_chop, 0.0, 1.0))

        return RegimeSnapshot(
            regime=self._current_regime,
            score=composite_score,
            adx=adx_val,
            chop=chop_val,
            bbw=bbw_val,
            bbw_sma=bbw_sma_val,
            bars_in_regime=self._bars_in_regime,
            transition_occurred=transition,
            timestamp=candle_ts,
            metadata={
                "consecutive_trend_signals": self._consecutive_trend_signals,
                "consecutive_range_signals": self._consecutive_range_signals,
            }
        )

    def update(self, candle: Any, history: Sequence[Any]) -> RegimeSnapshot:
        """Alias for evaluate conforming to PROJECT.md interface contract."""
        return self.evaluate(candle, history)

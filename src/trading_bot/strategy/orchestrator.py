"""Multi-Regime Strategy Orchestrator.

Unifies MarketRegimeFilter, MicroGridStrategy, and AMVTrendBreakStrategy under the standard
strategy interface contract:
    on_candle(candle: Candle, history: Sequence[Candle], ledger: VirtualLedger) -> Optional[OrderSignal]
    notify_position_closed(...) -> None

Key Responsibilities:
1. Dynamically switches between Micro-Grid (during RANGING regime) and AMV-TrendBreak (during TRENDING regime).
2. Dismantles the Micro-Grid on upward breakout or trend transition, cancelling orders and
   liquidating residual inventory into liquid cash for AMV-TrendBreak.
3. Protects active AMV positions when market transitions to RANGING, allowing trailing stops
   to manage exits gracefully before re-anchoring the grid.
4. Emits detailed [REGIME_CHANGE] logs and tracks structured transition history.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import logging
from typing import Any, Dict, List, Optional, Sequence

from .amv_strategy import AMVTrendBreakStrategy, OrderSide, OrderSignal, OrderType
from .grid_strategy import MicroGridStrategy
from .regime import MarketRegime, MarketRegimeFilter, RegimeSnapshot
from trading_bot.simulation.ledger import VirtualLedger

logger = logging.getLogger("trading_bot.strategy.orchestrator")


class MultiRegimeStrategyOrchestrator:
    """Multi-Regime hybrid trading strategy orchestrator."""

    def __init__(
        self,
        default_symbol: str = "BTC/EUR",
        exchange_min_notional: Decimal = Decimal("1.00"),
        step_size: Decimal = Decimal("0.00001"),
        regime_filter: Optional[MarketRegimeFilter] = None,
        grid_strategy: Optional[MicroGridStrategy] = None,
        trend_strategy: Optional[AMVTrendBreakStrategy] = None,
    ) -> None:
        self.default_symbol = default_symbol
        self.symbol = default_symbol
        self.exchange_min_notional = Decimal(str(exchange_min_notional))
        self.step_size = Decimal(str(step_size))

        self.regime_filter = regime_filter or MarketRegimeFilter()
        self.grid_strategy = grid_strategy or MicroGridStrategy(
            default_symbol=default_symbol,
            min_notional=self.exchange_min_notional,
            step_size=self.step_size,
        )
        self.amv_strategy = trend_strategy or AMVTrendBreakStrategy(
            default_symbol=default_symbol,
            exchange_min_notional=self.exchange_min_notional,
            step_size=self.step_size,
        )

        self.regime_transitions: List[Dict[str, Any]] = []
        self.pending_signals: List[OrderSignal] = []
        self._last_snapshot: Optional[RegimeSnapshot] = None

    @property
    def current_regime(self) -> MarketRegime:
        """Current active market regime."""
        return self.regime_filter.current_regime

    def reset(self) -> None:
        """Reset orchestrator and child strategies."""
        self.regime_filter.reset()
        self.grid_strategy.reset()
        self.regime_transitions.clear()
        self.pending_signals.clear()
        self._last_snapshot = None

    def on_candle(
        self,
        candle: Any,
        history: Sequence[Any],
        ledger: VirtualLedger,
    ) -> Optional[OrderSignal]:
        """Execute multi-regime evaluation on the incoming candle.

        Args:
            candle: The current Candle or candle dict.
            history: Historical candles sequence.
            ledger: Virtual ledger tracking cash and positions.

        Returns:
            OrderSignal if an order is triggered, or None.
        """
        # 1. Drain any pending order signals (e.g. from grid dismantling)
        if self.pending_signals:
            return self.pending_signals.pop(0)

        c_close = Decimal(str(getattr(candle, "close", candle.get("close") if isinstance(candle, dict) else 0.0)))
        c_ts = getattr(candle, "timestamp", None)
        if c_ts is None and isinstance(candle, dict):
            c_ts = candle.get("timestamp")

        # 2. Evaluate Market Regime Filter
        snapshot = self.regime_filter.evaluate(candle, history)
        self._last_snapshot = snapshot

        # 3. Handle Regime Transitions
        if snapshot.transition_occurred:
            from_regime = MarketRegime.RANGING if snapshot.regime == MarketRegime.TRENDING else MarketRegime.TRENDING
            record = {
                "from_regime": from_regime.value,
                "to_regime": snapshot.regime.value,
                "timestamp": snapshot.timestamp or c_ts,
                "adx": snapshot.adx,
                "chop": snapshot.chop,
                "bbw": snapshot.bbw,
                "close": float(c_close),
                "score": snapshot.score,
            }
            self.regime_transitions.append(record)

            ts_str = str(record["timestamp"]) if record["timestamp"] else "UNKNOWN"
            log_msg = (
                f"[REGIME_CHANGE] {ts_str}: Transitioned from {record['from_regime']} to "
                f"{record['to_regime']} | ADX: {snapshot.adx:.2f}, CHOP: {snapshot.chop:.2f}, "
                f"BBW: {snapshot.bbw:.4f}, Close: {float(c_close):.2f}"
            )
            logger.info(log_msg)


            # On transition RANGING -> TRENDING: Safely dismantle grid
            if snapshot.regime == MarketRegime.TRENDING:
                dismantle_signals = self.grid_strategy.dismantle_grid(current_price=c_close, ledger=ledger)
                if dismantle_signals:
                    sig = dismantle_signals[0]
                    self.pending_signals.extend(dismantle_signals[1:])
                    return sig

            # On transition TRENDING -> RANGING:
            elif snapshot.regime == MarketRegime.RANGING:
                pos = ledger.get_position(self.symbol)
                pos_qty = pos.quantity if pos else Decimal("0.0")
                # If no position held, initialize grid anchor
                if pos_qty <= Decimal("0.0"):
                    self.grid_strategy.reset()
                    self.grid_strategy.anchor_price = c_close

        # 4. Delegate to the active regime's strategy
        if snapshot.regime == MarketRegime.TRENDING:
            return self.amv_strategy.on_candle(candle, history, ledger)
        else:
            # In RANGING regime:
            pos = ledger.get_position(self.symbol)
            pos_qty = pos.quantity if pos else Decimal("0.0")
            # If AMV is still holding an open trend position, allow AMV to manage its trailing exit
            if pos_qty > Decimal("0.0") and len(self.grid_strategy.filled_tranches) == 0:
                return self.amv_strategy.on_candle(candle, history, ledger)

            return self.grid_strategy.on_candle(candle, history, ledger)

    def notify_position_closed(
        self,
        symbol: str,
        exit_price: Optional[Decimal] = None,
        current_equity: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Forward position closure event to child strategies."""
        self.amv_strategy.notify_position_closed(
            symbol=symbol,
            exit_price=exit_price,
            current_equity=current_equity,
            timestamp=timestamp,
        )
        # If in RANGING regime, re-anchor the grid around exit price
        if self.current_regime == MarketRegime.RANGING and exit_price is not None:
            self.grid_strategy.anchor_price = exit_price

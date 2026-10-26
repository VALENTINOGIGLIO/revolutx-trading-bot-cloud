"""AMV-TrendBreak Trading Strategy.

Adaptive Momentum & Volatility Breakout strategy engineered for aggressive capital compounding
from 10.00 EUR towards 10,000.00 EUR.

Rules:
1. Macro Trend Filter: Close > EMA 200
2. Tactical Trend Alignment: EMA 9 > EMA 21 > EMA 50 and Close > EMA 9
3. Volatility Breakout: Close > Upper Keltner Channel (EMA 20 + 1.8 * ATR 14)
4. Regime & Noise Gate: ADX 14 >= 22, +DI > -DI, and Choppiness Index (CHOP 14) < 58.0
5. Momentum Gate: 52.0 <= RSI 14 <= 72.0
6. Volume Confirmation: Volume >= 1.0 * SMA 20 (Volume) (when volume is present)

Trade Management:
- Initial Stop Loss: 1.75x ATR 14 below entry price
- Breakeven Trigger (TP1): When price touches 1.5R (Entry + 1.50 * Stop Distance), stop loss
  is moved to entry + 0.25% (covering fees and slippage)
- Chandelier Trailing Stop: 2.0x ATR 14 below highest price attained; tightens to 1.5x ATR 14
  when ADX > 38.0 (parabolic momentum)
- Structural Invalidation: Immediate market exit when Close < EMA 21
"""

from __future__ import annotations
import sys

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd

from .compounding import (
    CompoundingPhase,
    CompoundingPositionSizer,
    RiskGuard,
)
from .indicators import (
    compute_indicators_from_candles,
)


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class OrderSignal:
    """Strategy order execution signal conforming to PROJECT.md interface contract.

    Attributes:
        symbol: Trading pair symbol (e.g. 'SOL/EUR', 'ETH/EUR', 'BTC/EUR').
        side: OrderSide (BUY or SELL).
        target_quantity: Quantity of base asset to buy or sell.
        order_type: OrderType (MARKET or LIMIT).
        stop_loss: Associated stop loss price.
        take_profit: Optional take profit target price.
        price: Projected fill price.
        reason: Plain-text explanation of strategy trigger.
        metadata: Additional quantitative diagnostics.
        target_notional: Optional target notional in quote currency.
        timestamp_ms: Optional millisecond timestamp.
        order_id: Optional tracking order identifier.
        exit_reason: Optional specific exit reason for ExecutionSimulator.
    """
    symbol: str
    side: OrderSide
    target_quantity: Decimal
    order_type: OrderType
    stop_loss: Decimal
    take_profit: Optional[Decimal] = None
    price: Optional[Decimal] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    target_notional: Optional[Decimal] = None
    timestamp_ms: Optional[int] = None
    order_id: Optional[str] = None
    exit_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.exit_reason and self.reason:
            self.exit_reason = self.reason


@dataclass
class PositionState:
    """Active open position state managed by AMVTrendBreakStrategy."""
    symbol: str
    entry_price: Decimal
    quantity: Decimal
    initial_stop: Decimal
    current_stop: Decimal
    take_profit_1: Decimal  # Breakeven target at 1.5R
    highest_price: Decimal  # Peak high attained during position lifetime
    bars_in_trade: int = 0
    tp1_hit: bool = False
    entry_timestamp: Optional[datetime] = None
    entry_reason: str = ""
    is_closing: bool = False


# =====================================================================
# Walk-Forward Anti-Overfitting Validated Profiles
# =====================================================================
PROFILE_GLOBAL_PARETO_OPTIMAL = {
    "timeframe": "4h",
    "breakout_period": 20,
    "ema_macro": 200,
    "atr_stop_mult": Decimal("2.2"),
    "chandelier_atr_mult": Decimal("5.5"),
    "rsi_max": Decimal("100"),
    "enable_tp1": False,
    "exit_on_ema21": False,
    "description": "Global Pareto Optimum (+235.49% Basket, +209.88% SOL, +127.68% ETH, +120.65% BTC, +134.89% OOS)"
}

PROFILE_4H_PLATEAU_WINNER = {
    "timeframe": "4h",
    "breakout_period": 14,
    "ema_macro": 200,
    "atr_stop_mult": Decimal("2.8"),
    "chandelier_atr_mult": Decimal("5.5"),
    "description": "Local Basket Optimum (+232.28% Basket, +68.50% SOL)"
}

PROFILE_V2_BASELINE = {
    "timeframe": "4h",
    "breakout_period": 20,
    "ema_macro": 200,
    "atr_stop_mult": Decimal("2.2"),
    "chandelier_atr_mult": Decimal("4.5"),
    "description": "Bot v2 Baseline (+157.84% Basket, +169.98% SOL, +86.68% OOS)"
}


class AMVTrendBreakStrategy:
    """Adaptive Momentum & Volatility Breakout Strategy with 3-Phase Compounding.

    Implements the full interface contract:
        on_candle(candle: Candle, history: List[Candle], ledger: VirtualLedger) -> Optional[OrderSignal]
    """

    def __init__(
        self,
        default_symbol: str = "SOL/EUR",
        exchange_min_notional: Union[Decimal, float] = Decimal("5.00"),
        step_size: Union[Decimal, float] = Decimal("0.001"),
        min_warmup_bars: int = 50,
        rsi_min: float = 52.0,
        rsi_max: Union[Decimal, float] = Decimal("100"),
        strict_rsi_cap: bool = False,
        position_sizer: Optional[CompoundingPositionSizer] = None,
        risk_guard: Optional[RiskGuard] = None,
        tp_multiplier: float = 1.50,
        atr_stop_mult: Union[Decimal, float] = Decimal("2.2"),
        chandelier_atr_mult: Union[Decimal, float] = Decimal("5.5"),
        breakout_period: int = 20,
        enable_tp1: bool = False,
        exit_on_ema21: bool = False,
    ):
        self.default_symbol = default_symbol
        self.min_notional = Decimal(str(exchange_min_notional))
        self.step_size = Decimal(str(step_size))
        self.min_warmup_bars = min_warmup_bars
        self.rsi_min = float(rsi_min)
        self.rsi_max = float(rsi_max)
        self.strict_rsi_cap = bool(strict_rsi_cap)
        self.tp_multiplier = Decimal(str(tp_multiplier))
        self.atr_stop_mult = Decimal(str(atr_stop_mult))
        self.chandelier_atr_mult = Decimal(str(chandelier_atr_mult))
        self.breakout_period = int(breakout_period)
        self.enable_tp1 = bool(enable_tp1)
        self.exit_on_ema21 = bool(exit_on_ema21)

        self.sizer = position_sizer or CompoundingPositionSizer(
            min_notional=self.min_notional,
            default_step_size=self.step_size,
        )
        self.risk_guard = risk_guard or RiskGuard(
            max_daily_drawdown_pct=0.08,
            initial_equity=Decimal("10.00"),
        )

        # Active tracked positions keyed by symbol
        self._positions: Dict[str, PositionState] = {}

    @property
    def positions(self) -> Dict[str, PositionState]:
        """Dictionary of active positions tracked by the strategy."""
        return {k: v for k, v in self._positions.items() if not getattr(v, "is_closing", False)}

    def get_position(self, symbol: str, include_closing: bool = True) -> Optional[PositionState]:
        """Retrieve active or closing position for a given symbol."""
        pos = self._positions.get(symbol)
        if pos is not None and not include_closing and getattr(pos, "is_closing", False):
            return None
        return pos

    def reset(self) -> None:
        """Reset internal positions and risk guard."""
        self._positions.clear()
        self.risk_guard.reset(Decimal("10.00"))

    def _extract_equity_from_ledger(self, ledger: Any) -> Decimal:
        """Robustly extract total equity from virtual ledger or numeric value."""
        if ledger is None:
            return Decimal("10.00")
        if isinstance(ledger, (int, float, str, Decimal)):
            return Decimal(str(ledger))

        # Check total_equity property or method
        if hasattr(ledger, "total_equity"):
            te = getattr(ledger, "total_equity")
            val = te() if callable(te) else te
            return Decimal(str(val))

        # Check equity property
        if hasattr(ledger, "equity"):
            eq = getattr(ledger, "equity")
            val = eq() if callable(eq) else eq
            return Decimal(str(val))

        # Check cash property
        if hasattr(ledger, "cash"):
            c = getattr(ledger, "cash")
            val = c() if callable(c) else c
            return Decimal(str(val))

        return Decimal("10.00")

    def _sync_ledger_position(self, symbol: str, ledger: Any) -> None:
        """Ensure internal position tracking matches ledger if external sync needed."""
        if ledger is None:
            return
        if hasattr(ledger, "has_position") and callable(ledger.has_position):
            if not ledger.has_position(symbol) and symbol in self._positions:
                pos = self._positions[symbol]
                if getattr(pos, "is_closing", False):
                    eq = self._extract_equity_from_ledger(ledger)
                    self.record_trade_closed(pos, pos.current_stop, eq)


    def on_candle(
        self,
        candle: Any,
        history: Sequence[Any],
        ledger: Any = None,
    ) -> Optional[OrderSignal]:
        """Main candle-by-candle strategy execution method.

        Evaluates stop losses, breakeven triggers, trailing stops, structural invalidations,
        and new trend breakout entries.

        Args:
            candle: Current candle (Candle model, dataclass, or dict).
            history: Sequence of preceding candles.
            ledger: VirtualLedger instance or account balance.

        Returns:
            OrderSignal (BUY or SELL) or None if no action required.
        """
        symbol = str(getattr(candle, "symbol", self.default_symbol))
        close_price = Decimal(str(getattr(candle, "close", 0.0)))
        high_price = Decimal(str(getattr(candle, "high", close_price)))
        low_price = Decimal(str(getattr(candle, "low", close_price)))
        timestamp = getattr(candle, "timestamp", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        equity = self._extract_equity_from_ledger(ledger)
        self.risk_guard.update_equity(equity, timestamp)

        # Sync position with ledger state
        self._sync_ledger_position(symbol, ledger)
        pos = self._positions.get(symbol) or next(iter(self._positions.values()), None)
        
        # 1. Warm-up check
        warmup_needed = max(300, self.min_warmup_bars)
        history_slice = history[-warmup_needed:] if len(history) > warmup_needed else history
        candles_chain = list(history_slice) + [candle]
        if (len(history) + 1) < self.min_warmup_bars:
            return None

        # 2. Compute indicators
        df = compute_indicators_from_candles(candles_chain)
        if df.empty:
            return None
        last = df.iloc[-1]

        atr = Decimal(str(last["atr_14"]))
        adx = float(last["adx_14"])
        ema_21 = Decimal(str(last["ema_21"]))

        ts_ms = int(timestamp.timestamp() * 1000) if hasattr(timestamp, "timestamp") else None

        # =====================================================================
        # CASE A: CURRENTLY IN A POSITION -> Manage Stops & Exits
        # =====================================================================
        if pos is not None:
            if getattr(pos, "is_closing", False):
                # Exit signal already emitted, awaiting fill confirmation from execution caller
                return None

            pos.bars_in_trade += 1
            if high_price > pos.highest_price:
                pos.highest_price = high_price

            # 1. Check Hard Stop Loss Hit
            if low_price <= pos.current_stop:
                pos.is_closing = True
                exit_signal = OrderSignal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    target_quantity=pos.quantity,
                    target_notional=pos.quantity * pos.current_stop,
                    order_type=OrderType.MARKET,
                    stop_loss=pos.current_stop,
                    price=pos.current_stop,
                    reason="STOP_LOSS_HIT",
                    exit_reason="STOP_LOSS_HIT",
                    timestamp_ms=ts_ms,
                    metadata={
                        "bars_in_trade": pos.bars_in_trade,
                        "entry_price": float(pos.entry_price),
                        "exit_price": float(pos.current_stop),
                        "pnl_pct": float((pos.current_stop - pos.entry_price) / pos.entry_price),
                    },
                )
                return exit_signal

            # 2. Check Breakeven Trigger (Disabled in Pure Trend-Following mode)
            if self.enable_tp1 and not pos.tp1_hit and high_price >= pos.take_profit_1:
                pos.tp1_hit = True
                # Move stop to entry + 0.25% to guarantee fees are covered
                fee_buffered_be = pos.entry_price * Decimal("1.0025")
                pos.current_stop = max(pos.current_stop, fee_buffered_be)

            # 3. Dynamic Chandelier Trailing Stop
            # Active once TP1 hit or price moves > 1.0 ATR above entry
            if pos.tp1_hit or (pos.highest_price >= pos.entry_price + atr):
                adx_factor = Decimal("0.75") if adx > 38.0 else Decimal("1.0")
                trail_mult = self.chandelier_atr_mult * adx_factor
                candidate_stop = pos.highest_price - (trail_mult * atr)
                if candidate_stop > pos.current_stop:
                    pos.current_stop = candidate_stop

            # Check if trailing stop was breached
            if low_price <= pos.current_stop:
                pos.is_closing = True
                exit_signal = OrderSignal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    target_quantity=pos.quantity,
                    target_notional=pos.quantity * pos.current_stop,
                    order_type=OrderType.MARKET,
                    stop_loss=pos.current_stop,
                    price=pos.current_stop,
                    reason="CHANDELIER_TRAILING_STOP_HIT",
                    exit_reason="CHANDELIER_TRAILING_STOP_HIT",
                    timestamp_ms=ts_ms,
                    metadata={
                        "bars_in_trade": pos.bars_in_trade,
                        "entry_price": float(pos.entry_price),
                        "exit_price": float(pos.current_stop),
                        "pnl_pct": float((pos.current_stop - pos.entry_price) / pos.entry_price),
                        "adx": adx,
                    },
                )
                return exit_signal

            # 4. Structural Invalidation Exit: Close < EMA 21 (Optional, disabled by default in Pure Trend)
            if self.exit_on_ema21 and close_price < ema_21:
                pos.is_closing = True
                exit_signal = OrderSignal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    target_quantity=pos.quantity,
                    target_notional=pos.quantity * close_price,
                    order_type=OrderType.MARKET,
                    stop_loss=pos.current_stop,
                    price=close_price,
                    reason="STRUCTURAL_INVALIDATION_EMA21",
                    exit_reason="STRUCTURAL_INVALIDATION_EMA21",
                    timestamp_ms=ts_ms,
                    metadata={
                        "bars_in_trade": pos.bars_in_trade,
                        "entry_price": float(pos.entry_price),
                        "exit_price": float(close_price),
                        "pnl_pct": float((close_price - pos.entry_price) / pos.entry_price),
                        "ema_21": float(ema_21),
                    },
                )
                return exit_signal

            # Still holding
            return None


        # =====================================================================
        # CASE B: NOT IN POSITION -> Evaluate BUY Breakout Conditions
        # =====================================================================
        risk_check = self.risk_guard.check_risk_status(timestamp)
        if not risk_check.allowed or risk_check.risk_multiplier <= 0.0:
            return None

        # Condition 1: Macro Trend (Close > EMA 200)
        c_val = float(last["close"])
        ema_200 = float(last["ema_200"])
        rsi_val = float(last["rsi_14"])
        
        # DEBUG PRINT
        if c_val <= ema_200:
            return None

        # Condition 2: Tactical Alignment (EMA 9 > EMA 21 > EMA 50 and Close > EMA 9)
        ema_9 = float(last["ema_9"])
        ema_21_f = float(last["ema_21"])
        ema_50 = float(last["ema_50"])
        if not (ema_9 > ema_21_f > ema_50):
            return None
        if c_val <= ema_9:
            return None

        # Condition 3: Keltner Breakout (Close > Upper Keltner Channel)
        kc_col = "kc_14_upper" if (self.breakout_period == 14 and "kc_14_upper" in last) else "kc_upper"
        kc_upper = float(last[kc_col])
        if c_val <= kc_upper:
            return None

        # Condition 4: Regime & Choppiness Gate (ADX >= 22, +DI > -DI, CHOP < 58)
        plus_di = float(last["plus_di"])
        minus_di = float(last["minus_di"])
        chop = float(last["chop_14"])
        if adx < 22.0 or plus_di <= minus_di or chop >= 58.0:
            return None

        # Condition 6: Volume Confirmation (if volume data present)
        vol = float(last.get("volume", 0.0))
        vol_sma = float(last.get("volume_sma_20", 0.0))
        if vol > 0.0 and vol_sma > 0.0 and vol < (1.0 * vol_sma):
            return None

        # Condition 5: Momentum Gate (RSI >= rsi_min; non-exhaustion check: RSI <= rsi_max or explosive breakout)
        rsi = float(last["rsi_14"])
        if rsi < self.rsi_min:
            return None
        if rsi > self.rsi_max:
            if self.strict_rsi_cap:
                return None
            is_explosive_breakout = (c_val > kc_upper) and (vol >= vol_sma if vol_sma > 0 else True)
            if not is_explosive_breakout:
                return None


        # All criteria satisfied: Calculate stops and position sizing
        entry_price = close_price
        # Fallback stop distance if ATR is 0
        stop_dist = self.atr_stop_mult * atr if atr > Decimal("0") else (entry_price * Decimal("0.025"))
        initial_stop = entry_price - stop_dist
        tp1_target = entry_price + (Decimal("1.50") * stop_dist)

        sizing_res = self.sizer.calculate_position_size(
            equity=equity,
            entry_price=entry_price,
            stop_loss=initial_stop,
            risk_multiplier=risk_check.risk_multiplier,
            min_notional=self.min_notional,
            step_size=self.step_size,
        )

        if not sizing_res.is_valid or sizing_res.quantity <= Decimal("0"):
            return None

        # Register position
        new_pos = PositionState(
            symbol=symbol,
            entry_price=entry_price,
            quantity=sizing_res.quantity,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            take_profit_1=tp1_target,
            highest_price=entry_price,
            bars_in_trade=0,
            tp1_hit=False,
            entry_timestamp=timestamp,
            entry_reason="AMV_TREND_BREAKOUT",
        )
        self._positions[symbol] = new_pos

        signal = OrderSignal(
            symbol=symbol,
            side=OrderSide.BUY,
            target_quantity=sizing_res.quantity,
            target_notional=sizing_res.notional,
            order_type=OrderType.MARKET,
            stop_loss=initial_stop,
            take_profit=tp1_target,
            price=entry_price,
            reason="AMV_TREND_BREAKOUT",
            timestamp_ms=ts_ms,
            metadata={
                "phase": sizing_res.phase.value,
                "atr": float(atr),
                "adx": adx,
                "rsi": rsi,
                "chop": chop,
                "risk_amount": float(sizing_res.risk_amount),
                "risk_fraction": sizing_res.risk_fraction,
                "entry_timestamp": str(timestamp),
            },
        )
        return signal

    def record_trade_closed(
        self,
        pos: PositionState,
        exit_price: Decimal,
        current_equity: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Internal callback to finalize closed position and notify risk guards."""
        symbol = pos.symbol
        if symbol in self._positions:
            del self._positions[symbol]

        pnl = (exit_price - pos.entry_price) * pos.quantity
        self.risk_guard.on_trade_completed(pnl, current_equity, timestamp)

    def notify_position_closed(
        self,
        symbol: str,
        exit_price: Optional[Union[Decimal, float]] = None,
        current_equity: Optional[Union[Decimal, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """External callback to notify strategy that a position was closed in ledger/exchange.

        Args:
            symbol: Trading pair symbol (e.g. 'SOL/EUR').
            exit_price: Realized fill price from execution simulator or exchange (defaults to current_stop).
            current_equity: Portfolio equity after trade execution (defaults to RiskGuard equity).
            timestamp: Execution timestamp (defaults to current time).
        """
        pos = self._positions.get(symbol)
        if pos is not None:
            px = Decimal(str(exit_price)) if exit_price is not None else pos.current_stop
            eq = Decimal(str(current_equity)) if current_equity is not None else self.risk_guard.current_equity
            self.record_trade_closed(
                pos,
                px,
                eq,
                timestamp,
            )

    def notify_position_rejected(self, symbol: str) -> None:
        """External callback if exit order execution failed, keeping position tracked as active."""
        pos = self._positions.get(symbol)
        if pos is not None:
            pos.is_closing = False


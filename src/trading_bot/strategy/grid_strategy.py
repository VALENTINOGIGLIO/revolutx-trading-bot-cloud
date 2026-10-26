"""Micro-Grid Trading Strategy optimized for Bybit exchange constraints.

Operates primarily in RANGING/SIDEWAYS market regimes to harvest small oscillating profits.
Strictly adheres to Bybit's 1.00 EUR minNotional threshold by utilizing step-size ceiling
quantization:
    steps = ceil(target_notional / (price * step_size))
    quantity = steps * step_size
guaranteeing that every sub-order satisfies:
    order_notional = price * quantity >= 1.00 EUR
even with a 10.00 EUR total portfolio budget (enabling up to 10 active micro-levels).

Features:
- Fee-aware grid spacing (default >= 0.60% to safely exceed round-trip friction 0.20%)
- Sub-order generation and tracking
- Safe grid dismantling upon trend breakout or breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import math
from typing import Any, Dict, List, Optional, Sequence

from .amv_strategy import OrderSide, OrderSignal, OrderType
from trading_bot.simulation.ledger import VirtualLedger


@dataclass
class GridLevel:
    """Represents a single grid price level and its associated order state."""
    level_id: int
    buy_price: Decimal
    target_sell_price: Decimal
    quantity: Decimal
    notional: Decimal
    filled: bool = False
    order_signal: Optional[OrderSignal] = None


class MicroGridStrategy:
    """Micro-Grid strategy for ranging markets with strict Bybit minNotional compliance."""

    def __init__(
        self,
        default_symbol: str = "BTC/EUR",
        min_notional: Decimal = Decimal("1.00"),
        step_size: Decimal = Decimal("0.00001"),
        grid_spacing_pct: Decimal = Decimal("0.008"),   # 0.80% spacing
        max_levels: int = 10,
        stop_loss_pct: Decimal = Decimal("0.08"),       # 8.00% breakdown stop
    ) -> None:
        self.default_symbol = default_symbol
        self.symbol = default_symbol
        self.min_notional = Decimal(str(min_notional))
        self.step_size = Decimal(str(step_size))
        self.grid_spacing_pct = Decimal(str(grid_spacing_pct))
        self.max_levels = max_levels
        self.stop_loss_pct = Decimal(str(stop_loss_pct))

        self.anchor_price: Optional[Decimal] = None
        self.active_levels: Dict[int, GridLevel] = {}
        self.filled_tranches: List[GridLevel] = []
        self._total_grid_profit: Decimal = Decimal("0.00")

    def reset(self) -> None:
        """Reset grid state."""
        self.anchor_price = None
        self.active_levels.clear()
        self.filled_tranches.clear()
        self._total_grid_profit = Decimal("0.00")

    def calculate_order_quantity(
        self,
        price: Decimal,
        target_notional: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate order quantity using strict Bybit ceil quantization.

        Formula:
            steps = ceil( max(target_notional, min_notional) / (price * step_size) )
            quantity = steps * step_size

        Guarantees price * quantity >= min_notional.
        """
        target = target_notional if target_notional is not None else self.min_notional
        if target < self.min_notional:
            target = self.min_notional

        step_val = price * self.step_size
        if step_val <= Decimal("0.0"):
            return Decimal("0.0")

        # Ceil quantization to ensure notional >= min_notional
        ratio = target / step_val
        steps_int = int(ratio)
        if ratio > Decimal(str(steps_int)):
            steps_int += 1
        if steps_int < 1:
            steps_int = 1
        steps = Decimal(str(steps_int))
        qty = steps * self.step_size

        # Numerical safety check
        while qty * price < self.min_notional:
            qty += self.step_size

        return qty

    def calculate_sub_orders(
        self,
        current_price: Decimal,
        portfolio_balance: Decimal = Decimal("10.00"),
        num_levels: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate sub-orders sized at exactly or slightly above Bybit minNotional.

        Args:
            current_price: Current market price.
            portfolio_balance: Available capital (default: 10.00 EUR).
            num_levels: Optional level count override.

        Returns:
            List of sub-order dictionaries with level, price, quantity, and notional.
        """
        p_dec = Decimal(str(current_price))
        bal_dec = Decimal(str(portfolio_balance))

        if num_levels is not None:
            k_levels = num_levels
        else:
            k_levels = min(self.max_levels, max(1, int(bal_dec / self.min_notional)))

        target_per_order = max(self.min_notional, bal_dec / Decimal(str(k_levels)))
        orders: List[Dict[str, Any]] = []

        for i in range(1, k_levels + 1):
            mult = Decimal("1") - (Decimal(str(i)) * self.grid_spacing_pct)
            level_px = p_dec * mult
            qty = self.calculate_order_quantity(level_px, target_per_order)
            notional = qty * level_px
            orders.append({
                "level_id": i,
                "price": level_px,
                "quantity": qty,
                "notional": notional,
                "meets_min_notional": notional >= self.min_notional,
            })

        return orders

    def generate_grid_levels(
        self,
        current_price: Decimal,
        portfolio_balance: Decimal = Decimal("10.00"),
    ) -> List[OrderSignal]:
        """Generate a complete set of initial limit buy OrderSignals for the grid."""
        sub_orders = self.calculate_sub_orders(current_price, portfolio_balance)
        signals: List[OrderSignal] = []

        for o in sub_orders:
            sig = OrderSignal(
                symbol=self.symbol,
                side=OrderSide.BUY,
                target_quantity=o["quantity"],
                order_type=OrderType.LIMIT,
                stop_loss=current_price * (Decimal("1") - self.stop_loss_pct),
                take_profit=o["price"] * (Decimal("1") + self.grid_spacing_pct),
                price=o["price"],
                reason=f"GRID_LEVEL_{o['level_id']}_LIMIT_BUY",
                target_notional=o["notional"],
                metadata={"level_id": o["level_id"], "grid_price": float(o["price"])},
            )
            signals.append(sig)

        return signals

    def on_candle(
        self,
        candle: Any,
        history: Sequence[Any],
        ledger: VirtualLedger,
    ) -> Optional[OrderSignal]:
        """Evaluate grid logic on incoming candle and produce OrderSignal if level is triggered.

        Args:
            candle: Current candle (Candle object or dict).
            history: Historical candles sequence.
            ledger: Virtual ledger tracking cash, equity, and positions.

        Returns:
            OrderSignal if a buy or sell is triggered, or None.
        """
        c_close = Decimal(str(getattr(candle, "close", candle.get("close") if isinstance(candle, dict) else 0.0)))
        c_high = Decimal(str(getattr(candle, "high", candle.get("high") if isinstance(candle, dict) else c_close)))
        c_low = Decimal(str(getattr(candle, "low", candle.get("low") if isinstance(candle, dict) else c_close)))
        ts_ms = getattr(candle, "open_time_ms", None)

        if c_close <= Decimal("0.0"):
            return None

        # 1. Initialize anchor price if not set
        if self.anchor_price is None:
            self.anchor_price = c_close

        pos = ledger.get_position(self.symbol)
        pos_qty = pos.quantity if pos else Decimal("0.0")

        # 2. Emergency Breakdown Stop-Loss
        breakdown_price = self.anchor_price * (Decimal("1") - self.stop_loss_pct)
        if c_low <= breakdown_price and pos_qty > Decimal("0.0"):
            self.reset()
            return OrderSignal(
                symbol=self.symbol,
                side=OrderSide.SELL,
                target_quantity=pos_qty,
                order_type=OrderType.MARKET,
                stop_loss=Decimal("0.0"),
                price=c_close,
                reason="GRID_BREAKDOWN_STOP_LOSS",
                exit_reason="GRID_BREAKDOWN_STOP_LOSS",
                timestamp_ms=ts_ms,
            )

        # 3. Check Take-Profit on filled tranches
        if self.filled_tranches and pos_qty > Decimal("0.0"):
            # Check highest profit tranche first
            for i, tranche in enumerate(self.filled_tranches):
                if c_high >= tranche.target_sell_price:
                    # Sell this tranche
                    sell_qty = min(pos_qty, tranche.quantity)
                    del self.filled_tranches[i]
                    # Free up active level
                    if tranche.level_id in self.active_levels:
                        del self.active_levels[tranche.level_id]

                    # If no position left, re-anchor grid to current close
                    if len(self.filled_tranches) == 0:
                        self.anchor_price = c_close

                    return OrderSignal(
                        symbol=self.symbol,
                        side=OrderSide.SELL,
                        target_quantity=sell_qty,
                        order_type=OrderType.MARKET,
                        stop_loss=Decimal("0.0"),
                        price=c_close,
                        reason="GRID_TAKE_PROFIT",
                        exit_reason="GRID_TAKE_PROFIT",
                        timestamp_ms=ts_ms,
                        metadata={"level_id": tranche.level_id, "tp_target": float(tranche.target_sell_price)},
                    )

        # 4. Check Buy Triggers across grid levels
        available_cash = ledger.available_cash
        if available_cash >= self.min_notional:
            total_equity = ledger.total_equity({self.symbol: c_close})
            k_levels = min(self.max_levels, max(1, int(total_equity / self.min_notional)))
            target_per_order = max(self.min_notional, total_equity / Decimal(str(k_levels)))

            for i in range(1, k_levels + 1):
                level_price = self.anchor_price * (Decimal("1") - (Decimal(str(i)) * self.grid_spacing_pct))
                # Check if level is already filled or active
                if i in self.active_levels:
                    continue

                if c_low <= level_price:
                    eff_price = min(c_close, level_price) if c_close > Decimal("0.0") else level_price
                    qty = self.calculate_order_quantity(eff_price, target_per_order)
                    order_cost = qty * level_price
                    if available_cash >= order_cost:
                        # Record active level and tranche
                        tp_px = level_price * (Decimal("1") + self.grid_spacing_pct)
                        gl = GridLevel(
                            level_id=i,
                            buy_price=level_price,
                            target_sell_price=tp_px,
                            quantity=qty,
                            notional=order_cost,
                            filled=True,
                        )
                        self.active_levels[i] = gl
                        self.filled_tranches.append(gl)

                        return OrderSignal(
                            symbol=self.symbol,
                            side=OrderSide.BUY,
                            target_quantity=qty,
                            order_type=OrderType.MARKET,
                            stop_loss=breakdown_price,
                            take_profit=tp_px,
                            price=c_close,
                            reason=f"GRID_BUY_LEVEL_{i}",
                            target_notional=order_cost,
                            timestamp_ms=ts_ms,
                            metadata={"level_id": i, "level_price": float(level_price)},
                        )

        # 5. Dynamic Anchor Adjustment (if price drifts up while flat)
        if pos_qty == Decimal("0.0") and len(self.filled_tranches) == 0:
            if self.anchor_price is not None and c_close > self.anchor_price * (Decimal("1") + (Decimal("2") * self.grid_spacing_pct)):
                self.anchor_price = c_close

        return None

    def dismantle_grid(self, current_price: Decimal, ledger: VirtualLedger) -> List[OrderSignal]:
        """Safely dismantle all grid levels upon regime transition to TRENDING.

        Cancels any pending orders, liquidates residual inventory to 100% cash,
        and resets grid state.

        Args:
            current_price: Current market price.
            ledger: Virtual ledger.

        Returns:
            List of liquidation OrderSignals (if any inventory was held).
        """
        signals: List[OrderSignal] = []
        pos = ledger.get_position(self.symbol)
        pos_qty = pos.quantity if pos else Decimal("0.0")

        # Liquidate held inventory at market to free 100% cash for AMV-TrendBreak
        if pos_qty > Decimal("0.0"):
            sig = OrderSignal(
                symbol=self.symbol,
                side=OrderSide.SELL,
                target_quantity=pos_qty,
                order_type=OrderType.MARKET,
                stop_loss=Decimal("0.0"),
                price=current_price,
                reason="GRID_DISMANTLED_BREAKOUT",
                exit_reason="GRID_DISMANTLED_BREAKOUT",
            )
            signals.append(sig)

        self.reset()
        return signals

    def notify_position_closed(
        self,
        symbol: str,
        exit_price: Optional[Decimal] = None,
        current_equity: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Handle position closure notification."""
        if exit_price is not None:
            self.anchor_price = exit_price


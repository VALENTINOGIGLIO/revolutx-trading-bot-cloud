"""
Order Execution Simulator for Paper Trading and Historical Backtesting.

Models realistic exchange frictions including maker/taker fees (default 0.10%),
conservative market order slippage (default 0.05%), lot stepSize truncation,
price tickSize quantization, and exchange minimum notional rules
(e.g., 5.00 EUR on Binance, 1.00 EUR on Bybit).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from trading_bot.simulation.ledger import (
    InsufficientFundsError,
    InsufficientPositionError,
    TradeRecord,
    VirtualLedger,
)


class OrderSide(str, Enum):
    """Order direction side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class OrderSignal:
    """
    Standard order request signal emitted by trading strategies.
    Conforms to the Strategy <-> ExecutionSimulator contract.
    """

    symbol: str
    side: Union[OrderSide, str]
    target_quantity: Optional[Decimal] = None
    target_notional: Optional[Decimal] = None
    order_type: Union[OrderType, str] = OrderType.MARKET
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    timestamp_ms: Optional[int] = None
    order_id: Optional[str] = None
    exit_reason: Optional[str] = None


@dataclass
class ExecutionResult:
    """Structured result returned by ExecutionSimulator.execute()."""

    success: bool
    symbol: str
    side: str
    order_type: str
    raw_price: Decimal
    executed_price: Decimal
    quantity: Decimal
    quote_notional: Decimal
    fee: Decimal
    fee_rate: Decimal
    slippage: Decimal
    realized_pnl: Decimal = Decimal("0.00")
    trade_record: Optional[TradeRecord] = None
    error_message: Optional[str] = None

    @property
    def fill_price(self) -> Decimal:
        """Alias for executed_price."""
        return self.executed_price

    @property
    def executed_quantity(self) -> Decimal:
        """Alias for quantity."""
        return self.quantity

    @property
    def notional(self) -> Decimal:
        """Alias for quote_notional."""
        return self.quote_notional

    @property
    def portfolio_balance(self) -> Optional[Decimal]:
        """Portfolio balance immediately following execution."""
        return self.trade_record.equity_after if self.trade_record else None

    def __bool__(self) -> bool:
        """Evaluate truthiness based on execution success."""
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result to dictionary."""
        return {
            "success": self.success,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "raw_price": str(self.raw_price),
            "executed_price": str(self.executed_price),
            "quantity": str(self.quantity),
            "quote_notional": str(self.quote_notional),
            "fee": str(self.fee),
            "fee_rate": str(self.fee_rate),
            "slippage": str(self.slippage),
            "realized_pnl": str(self.realized_pnl),
            "error_message": self.error_message,
            "trade": self.trade_record.to_dict() if self.trade_record else None,
        }


# Exchange-specific default minimum order notional thresholds in quote currency
EXCHANGE_MIN_NOTIONAL: Dict[str, Decimal] = {
    "binance": Decimal("5.00"),  # Binance Spot EUR minimum notional
    "bybit": Decimal("1.00"),    # Bybit Spot EUR minimum notional
    "kraken": Decimal("0.45"),   # Kraken Spot EUR costmin
    "coinbase": Decimal("1.00"), # Coinbase Spot EUR minimum notional
    "revolutx": Decimal("1.00"), # Revolut X Spot EUR minimum notional
    "revolut_x": Decimal("1.00"),
    "revx": Decimal("1.00"),
}


class ExecutionSimulator:
    """
    Simulates exchange order matching, friction (fee & slippage), and constraint validation.
    """

    def __init__(
        self,
        fee_rate: Union[Decimal, float, str] = Decimal("0.001"),    # 0.10% VIP0 taker fee
        slippage: Union[Decimal, float, str] = Decimal("0.0005"),   # 0.05% (5 bps) market slippage
        min_notional: Optional[Union[Decimal, float, str]] = None,
        exchange: str = "binance",
        step_size: Union[Decimal, float, str] = Decimal("0.00001"), # 5 decimal places lot step
        tick_size: Optional[Union[Decimal, float, str]] = None,     # Adaptive if None, or explicit Decimal
        strict: bool = False,
        notification_service: Optional[Any] = None,
    ) -> None:
        self.exchange: str = exchange.lower()
        self.fee_rate: Decimal = Decimal(str(fee_rate))
        self.slippage: Decimal = Decimal(str(slippage))
        self.step_size: Decimal = Decimal(str(step_size))
        self.tick_size: Optional[Decimal] = Decimal(str(tick_size)) if tick_size is not None else None
        self.strict: bool = strict
        self.notification_service: Optional[Any] = notification_service

        if min_notional is not None:
            self.min_notional: Decimal = Decimal(str(min_notional))
        else:
            self.min_notional = EXCHANGE_MIN_NOTIONAL.get(self.exchange, Decimal("5.00"))

    @classmethod
    def for_exchange(
        cls,
        exchange: str,
        fee_rate: Optional[Union[Decimal, float, str]] = None,
        slippage: Optional[Union[Decimal, float, str]] = None,
        **kwargs: Any,
    ) -> ExecutionSimulator:
        """Factory constructor configured for a specific exchange."""
        ex = exchange.lower()
        default_fee = Decimal("0.001")
        default_slip = Decimal("0.0005")
        if ex == "kraken":
            default_fee = Decimal("0.0040")  # Kraken retail tier 1
        elif ex == "binance":
            default_fee = Decimal("0.0010")  # Binance VIP0
        elif ex == "bybit":
            default_fee = Decimal("0.0010")  # Bybit standard
        elif ex in ("revolutx", "revolut_x", "revx"):
            default_fee = Decimal("0.0000")  # Revolut X Limit Maker fee is 0.00%
            default_slip = Decimal("0.0000")

        fee = Decimal(str(fee_rate)) if fee_rate is not None else default_fee
        slip = Decimal(str(slippage)) if slippage is not None else default_slip
        return cls(fee_rate=fee, slippage=slip, exchange=ex, **kwargs)

    def set_notification_service(self, service: Any) -> None:
        """Set or update the notification service for event dispatching."""
        self.notification_service = service

    def calculate_fee(self, notional: Decimal) -> Decimal:
        """Calculate taker exchange fee for a given gross notional value."""
        return notional * self.fee_rate

    def calculate_slippage_price(self, raw_price: Decimal, side: Union[OrderSide, str]) -> Decimal:
        """
        Calculate executed fill price after modeling unfavorable slippage.
        BUY orders slip upwards (+slippage).
        SELL orders slip downwards (-slippage).
        """
        side_str = str(side.value if hasattr(side, "value") else side).upper()
        if side_str == "BUY":
            slipped = raw_price * (Decimal("1") + self.slippage)
        else:
            slipped = raw_price * (Decimal("1") - self.slippage)
        return self.quantize_price(slipped)

    def quantize_quantity(self, quantity: Decimal) -> Decimal:
        """Quantize quantity down to the exchange's LOT_SIZE stepSize."""
        if self.step_size <= Decimal("0.0"):
            return quantity
        steps = (quantity / self.step_size).quantize(Decimal("1"), rounding=ROUND_DOWN)
        return steps * self.step_size

    def quantize_price(self, price: Decimal) -> Decimal:
        """Quantize price to the exchange's PRICE_FILTER tickSize."""
        if self.tick_size is not None:
            effective_tick = self.tick_size
        else:
            # Adaptive price tick size based on price magnitude when not explicitly specified
            if price < Decimal("1.0"):
                effective_tick = Decimal("0.00001")
            elif price < Decimal("10.0"):
                effective_tick = Decimal("0.0001")
            else:
                effective_tick = Decimal("0.01")

        if effective_tick <= Decimal("0.0"):
            return price
        # Standard rounding for price ticks
        return (price / effective_tick).quantize(Decimal("1")) * effective_tick

    def validate_notional(self, notional: Decimal, raise_error: bool = False) -> bool:
        """
        Validate whether an order meets the exchange minimum notional requirement.
        """
        if notional < self.min_notional:
            if raise_error:
                raise ValueError(
                    f"Order notional {notional:.4f} is below minimum notional {self.min_notional:.4f}"
                )
            return False
        return True

    def simulate_market_buy(
        self,
        cash_budget: Union[Decimal, float, str],
        raw_price: Union[Decimal, float, str],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Simulate a market buy allocation from a specific cash budget.
        Returns: (quantity, executed_price, fee).
        Raises ValueError if budget cannot satisfy exchange minimum notional.
        """
        budget = Decimal(str(cash_budget))
        raw_px = Decimal(str(raw_price))
        if budget <= Decimal("0.0"):
            raise ValueError(f"Cash budget must be positive: {budget}")
        if raw_px <= Decimal("0.0"):
            raise ValueError(f"Price must be positive: {raw_px}")

        exec_px = self.calculate_slippage_price(raw_px, OrderSide.BUY)
        # Account for fee in budget: notional * (1 + fee_rate) <= cash_budget
        max_notional = budget / (Decimal("1") + self.fee_rate)

        if max_notional < self.min_notional:
            raise ValueError(
                f"Budget notional {max_notional:.4f} is below exchange minimum notional {self.min_notional:.4f}"
            )

        raw_qty = max_notional / exec_px
        qty = self.quantize_quantity(raw_qty)
        actual_notional = qty * exec_px

        if actual_notional < self.min_notional:
            raise ValueError(
                f"Quantized order notional {actual_notional:.4f} fell below minimum notional {self.min_notional:.4f}"
            )

        fee = self.calculate_fee(actual_notional)
        return qty, exec_px, fee

    def simulate_market_sell(
        self,
        quantity: Union[Decimal, float, str],
        raw_price: Union[Decimal, float, str],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Simulate a market sell of asset units.
        Returns: (quantity, executed_price, fee).
        Raises ValueError if quantity is <= 0 or notional < min_notional.
        """
        raw_qty = Decimal(str(quantity))
        raw_px = Decimal(str(raw_price))

        if raw_qty <= Decimal("0.0"):
            raise ValueError(f"Quantity must be positive: {raw_qty}")
        if raw_px <= Decimal("0.0"):
            raise ValueError(f"Price must be positive: {raw_px}")

        qty = self.quantize_quantity(raw_qty)
        if qty <= Decimal("0.0"):
            raise ValueError(f"Quantity after quantization is zero (stepSize={self.step_size})")

        exec_px = self.calculate_slippage_price(raw_px, OrderSide.SELL)
        notional = qty * exec_px

        if notional < self.min_notional:
            raise ValueError(
                f"Sell order notional {notional:.4f} is below exchange minimum notional {self.min_notional:.4f}"
            )

        fee = self.calculate_fee(notional)
        return qty, exec_px, fee

    def execute(
        self,
        order: Union[OrderSignal, Dict[str, Any], Any],
        current_price: Union[Decimal, float, str],
        ledger: VirtualLedger,
        timestamp_ms: Optional[int] = None,
        notification_service: Optional[Any] = None,
        raise_on_error: Optional[bool] = None,
    ) -> ExecutionResult:
        """
        Execute an OrderSignal against the VirtualLedger with full friction modeling.

        Interface contract:
        ExecutionSimulator.execute(order: OrderSignal, current_price: Decimal, ledger: VirtualLedger) -> ExecutionResult
        """
        should_raise = self.strict if raise_on_error is None else raise_on_error
        raw_px = Decimal(str(current_price))
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        # Extract order parameters (supports OrderSignal, Dict, or duck-typed objects)
        if isinstance(order, dict):
            symbol = str(order.get("symbol", "BTCEUR")).upper()
            side_val = str(order.get("side", "BUY")).upper()
            target_qty = Decimal(str(order["target_quantity"])) if order.get("target_quantity") is not None else None
            target_notional = Decimal(str(order["target_notional"])) if order.get("target_notional") is not None else None
            order_type = str(order.get("order_type", "MARKET")).upper()
            order_id = order.get("order_id")
            exit_reason = order.get("exit_reason") or order.get("reason")
        else:
            symbol = str(getattr(order, "symbol", "BTCEUR")).upper()
            raw_side = getattr(order, "side", "BUY")
            side_val = str(raw_side.value if hasattr(raw_side, "value") else raw_side).upper()
            target_qty = (
                Decimal(str(getattr(order, "target_quantity")))
                if getattr(order, "target_quantity", None) is not None
                else None
            )
            target_notional = (
                Decimal(str(getattr(order, "target_notional")))
                if getattr(order, "target_notional", None) is not None
                else None
            )
            raw_type = getattr(order, "order_type", "MARKET")
            order_type = str(raw_type.value if hasattr(raw_type, "value") else raw_type).upper()
            order_id = getattr(order, "order_id", None)
            exit_reason = getattr(order, "exit_reason", None) or getattr(order, "reason", None)

        if side_val == "BUY":
            return self._execute_buy(
                symbol=symbol,
                raw_price=raw_px,
                target_quantity=target_qty,
                target_notional=target_notional,
                order_type=order_type,
                order_id=order_id,
                timestamp_ms=ts,
                ledger=ledger,
                should_raise=should_raise,
                notification_service=notification_service,
            )
        elif side_val == "SELL":
            return self._execute_sell(
                symbol=symbol,
                raw_price=raw_px,
                target_quantity=target_qty,
                order_type=order_type,
                order_id=order_id,
                exit_reason=exit_reason,
                timestamp_ms=ts,
                ledger=ledger,
                should_raise=should_raise,
                notification_service=notification_service,
            )
        else:
            err = f"Unsupported order side: {side_val}"
            if should_raise:
                raise ValueError(err)
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side=side_val,
                order_type=order_type,
                raw_price=raw_px,
                executed_price=raw_px,
                quantity=Decimal("0.0"),
                quote_notional=Decimal("0.0"),
                fee=Decimal("0.0"),
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                error_message=err,
            )

    def _execute_buy(
        self,
        symbol: str,
        raw_price: Decimal,
        target_quantity: Optional[Decimal],
        target_notional: Optional[Decimal],
        order_type: str,
        order_id: Optional[str],
        timestamp_ms: int,
        ledger: VirtualLedger,
        should_raise: bool,
        notification_service: Optional[Any],
    ) -> ExecutionResult:
        """Handle execution of BUY order."""
        exec_px = self.calculate_slippage_price(raw_price, OrderSide.BUY)

        try:
            if target_quantity is not None and target_quantity > Decimal("0.0"):
                qty = self.quantize_quantity(target_quantity)
                notional = qty * exec_px
                while notional < self.min_notional and self.step_size > Decimal("0.0"):
                    cand_qty = qty + self.step_size
                    if (cand_qty * exec_px * (Decimal("1") + self.fee_rate)) > ledger.available_cash:
                        break
                    qty = cand_qty
                    notional = qty * exec_px
            elif target_notional is not None and target_notional > Decimal("0.0"):
                raw_qty = target_notional / exec_px
                qty = self.quantize_quantity(raw_qty)
                notional = qty * exec_px
                # If target_notional meets min_notional, ensure step truncation does not drop below min_notional
                if notional < self.min_notional and target_notional >= self.min_notional and self.step_size > Decimal("0.0"):
                    qty += self.step_size
                    notional = qty * exec_px

            else:
                # Default to full available cash budget (compounding all-in)
                budget = ledger.available_cash
                max_notional = budget / (Decimal("1") + self.fee_rate)
                raw_qty = max_notional / exec_px
                qty = self.quantize_quantity(raw_qty)
                notional = qty * exec_px

            if notional < self.min_notional:
                err_msg = (
                    f"Order notional {notional:.4f} is below minimum notional "
                    f"{self.min_notional:.4f} on {self.exchange.capitalize()}"
                )
                if should_raise:
                    raise ValueError(err_msg)
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="BUY",
                    order_type=order_type,
                    raw_price=raw_price,
                    executed_price=exec_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=Decimal("0.0"),
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err_msg,
                )

            fee = self.calculate_fee(notional)
            total_cost = notional + fee

            if ledger.cash < total_cost:
                err_msg = (
                    f"Insufficient cash: have {ledger.cash:.4f}, required {total_cost:.4f} "
                    f"(notional {notional:.4f} + fee {fee:.4f})"
                )
                if should_raise:
                    raise InsufficientFundsError(err_msg)
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="BUY",
                    order_type=order_type,
                    raw_price=raw_price,
                    executed_price=exec_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=fee,
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err_msg,
                )

            trade = ledger.record_buy(
                symbol=symbol,
                quantity=qty,
                executed_price=exec_px,
                fee=fee,
                timestamp_ms=timestamp_ms,
                raw_price=raw_price,
                order_type=order_type,
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                order_id=order_id,
            )

            self._dispatch_notification(trade, ledger, notification_service)

            return ExecutionResult(
                success=True,
                symbol=symbol,
                side="BUY",
                order_type=order_type,
                raw_price=raw_price,
                executed_price=exec_px,
                quantity=qty,
                quote_notional=notional,
                fee=fee,
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                realized_pnl=Decimal("0.00"),
                trade_record=trade,
            )

        except Exception as exc:
            if should_raise:
                raise
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY",
                order_type=order_type,
                raw_price=raw_price,
                executed_price=exec_px,
                quantity=Decimal("0.0"),
                quote_notional=Decimal("0.0"),
                fee=Decimal("0.0"),
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                error_message=str(exc),
            )

    def _execute_sell(
        self,
        symbol: str,
        raw_price: Decimal,
        target_quantity: Optional[Decimal],
        order_type: str,
        order_id: Optional[str],
        exit_reason: Optional[str],
        timestamp_ms: int,
        ledger: VirtualLedger,
        should_raise: bool,
        notification_service: Optional[Any],
    ) -> ExecutionResult:
        """Handle execution of SELL order."""
        exec_px = self.calculate_slippage_price(raw_price, OrderSide.SELL)
        pos = ledger.get_position(symbol)

        try:
            if target_quantity is not None and target_quantity > Decimal("0.0"):
                qty = self.quantize_quantity(target_quantity)
            else:
                qty = pos.quantity  # Sell entire position by default

            if qty <= Decimal("0.0"):
                err_msg = f"Cannot sell {symbol}: position is zero"
                if should_raise:
                    raise ValueError(err_msg)
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="SELL",
                    order_type=order_type,
                    raw_price=raw_price,
                    executed_price=exec_px,
                    quantity=Decimal("0.0"),
                    quote_notional=Decimal("0.0"),
                    fee=Decimal("0.0"),
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err_msg,
                )

            if pos.quantity < qty:
                err_msg = f"Insufficient position: holding {pos.quantity}, requested {qty}"
                if should_raise:
                    raise InsufficientPositionError(err_msg)
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="SELL",
                    order_type=order_type,
                    raw_price=raw_price,
                    executed_price=exec_px,
                    quantity=qty,
                    quote_notional=qty * exec_px,
                    fee=Decimal("0.0"),
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err_msg,
                )

            # Determine whether this order is closing an existing position or exit signal:
            # - selling entire held quantity (target_quantity is None or qty == pos.quantity or remaining dust < step_size)
            # - exit_reason is provided (e.g. STOP_LOSS_HIT, CHANDELIER_TRAILING_STOP_HIT, etc.)
            # - order_type is STOP_LOSS or TAKE_PROFIT
            is_full_close = (
                target_quantity is None
                or qty == pos.quantity
                or (pos.quantity > Decimal("0.0") and (pos.quantity - qty) < self.step_size)
            )
            is_exit_signal = bool(exit_reason) or (
                order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT, "STOP_LOSS", "TAKE_PROFIT")
            )
            is_closing_order = is_full_close or is_exit_signal

            # When closing the entire position, sweep all remaining held units to prevent trapping dust
            if is_full_close and pos.quantity > Decimal("0.0"):
                qty = pos.quantity

            notional = qty * exec_px
            fee = self.calculate_fee(notional)

            # Minimum notional validation on sell:
            # When closing an existing position on a stop-loss / exit signal (selling the entire held quantity),
            # allow the sell order to fill even if market price drops slightly below nominal minNotional threshold
            # (e.g. 4.90 EUR on Binance), preventing stranding capital or trapping dust. Closing orders must be allowed
            # to exit cleanly. Only reject new opening orders or discretionary partial sells below minNotional.
            if notional < self.min_notional and not is_closing_order:
                err_msg = (
                    f"Sell notional {notional:.4f} is below minimum notional "
                    f"{self.min_notional:.4f} on {self.exchange.capitalize()}"
                )
                if should_raise:
                    raise ValueError(err_msg)
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="SELL",
                    order_type=order_type,
                    raw_price=raw_price,
                    executed_price=exec_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=fee,
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err_msg,
                )

            trade = ledger.record_sell(
                symbol=symbol,
                quantity=qty,
                executed_price=exec_px,
                fee=fee,
                timestamp_ms=timestamp_ms,
                raw_price=raw_price,
                order_type=order_type,
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                order_id=order_id,
                exit_reason=exit_reason,
            )

            self._dispatch_notification(trade, ledger, notification_service)

            return ExecutionResult(
                success=True,
                symbol=symbol,
                side="SELL",
                order_type=order_type,
                raw_price=raw_price,
                executed_price=exec_px,
                quantity=qty,
                quote_notional=notional,
                fee=fee,
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                realized_pnl=trade.realized_pnl,
                trade_record=trade,
            )

        except Exception as exc:
            if should_raise:
                raise
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="SELL",
                order_type=order_type,
                raw_price=raw_price,
                executed_price=exec_px,
                quantity=Decimal("0.0"),
                quote_notional=Decimal("0.0"),
                fee=Decimal("0.0"),
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                error_message=str(exc),
            )

    def execute_buy(
        self,
        symbol: str,
        raw_price: Union[Decimal, float, str],
        ledger: VirtualLedger,
        quote_budget: Optional[Union[Decimal, float, str]] = None,
        quantity: Optional[Union[Decimal, float, str]] = None,
        timestamp_ms: Optional[int] = None,
        order_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Convenience method for executing a BUY order directly."""
        order = OrderSignal(
            symbol=symbol,
            side=OrderSide.BUY,
            target_quantity=Decimal(str(quantity)) if quantity is not None else None,
            target_notional=Decimal(str(quote_budget)) if quote_budget is not None else None,
            order_id=order_id,
            timestamp_ms=timestamp_ms,
        )
        return self.execute(order=order, current_price=raw_price, ledger=ledger, timestamp_ms=timestamp_ms)

    def execute_sell(
        self,
        symbol: str,
        raw_price: Union[Decimal, float, str],
        ledger: VirtualLedger,
        quantity: Optional[Union[Decimal, float, str]] = None,
        timestamp_ms: Optional[int] = None,
        order_id: Optional[str] = None,
        exit_reason: Optional[str] = None,
    ) -> ExecutionResult:
        """Convenience method for executing a SELL order directly."""
        order = OrderSignal(
            symbol=symbol,
            side=OrderSide.SELL,
            target_quantity=Decimal(str(quantity)) if quantity is not None else None,
            order_id=order_id,
            exit_reason=exit_reason,
            timestamp_ms=timestamp_ms,
        )
        return self.execute(order=order, current_price=raw_price, ledger=ledger, timestamp_ms=timestamp_ms)

    def _dispatch_notification(
        self,
        trade: TradeRecord,
        ledger: VirtualLedger,
        notification_service: Optional[Any],
    ) -> None:
        """Dispatch trade executed event if notification service is available."""
        svc = notification_service or self.notification_service
        if not svc:
            return

        try:
            from trading_bot.notifications.models import TradeExecutedPayload
            pos = ledger.get_position(trade.symbol)
            payload = TradeExecutedPayload(
                order_id=trade.order_id or trade.trade_id,
                symbol=trade.symbol,
                side=trade.side,
                order_type=trade.order_type,
                fill_price=float(trade.executed_price),
                quantity=float(trade.quantity),
                notional_value=float(trade.quote_notional),
                fee=float(trade.fee),
                fee_currency=ledger.quote_currency,
                portfolio_balance=float(trade.equity_after),
                cash_balance=float(trade.cash_after),
                position_qty_after=float(pos.quantity),
                realized_pnl=float(trade.realized_pnl),
                timestamp=trade.timestamp,
            )
            if hasattr(svc, "dispatch"):
                svc.dispatch(payload)
            elif callable(svc):
                svc(payload)
        except Exception:
            # Fallback to dictionary payload if pydantic model is unavailable
            payload_dict = {
                "symbol": trade.symbol,
                "side": trade.side,
                "fill_price": float(trade.executed_price),
                "quantity": float(trade.quantity),
                "fee": float(trade.fee),
                "portfolio_balance": float(trade.equity_after),
                "timestamp": trade.timestamp,
                "realized_pnl": float(trade.realized_pnl),
            }
            if hasattr(svc, "dispatch"):
                svc.dispatch(payload_dict)
            elif callable(svc):
                svc(payload_dict)

"""
Virtual Ledger Engine for Paper Trading and Historical Backtesting.

Double-entry bookkeeping of virtual cash, asset positions, cost basis,
realized and unrealized PnL, cumulative exchange fees, and immutable
trade execution history. Uses fixed-point Decimal arithmetic exclusively
to avoid IEEE-754 floating-point drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class SimulationError(Exception):
    """Base exception for simulation engine errors."""


class InsufficientFundsError(SimulationError, ValueError):
    """Raised when an order requires more cash than currently available."""


class InsufficientPositionError(SimulationError, ValueError):
    """Raised when an order attempts to sell more asset than currently held."""


class InvalidOrderError(SimulationError, ValueError):
    """Raised when order parameters are mathematically or logically invalid."""


@dataclass
class Position:
    """Represents an open or closed asset position in the ledger."""

    symbol: str
    quantity: Decimal = Decimal("0.0")
    average_entry_price: Decimal = Decimal("0.0")
    cost_basis: Decimal = Decimal("0.0")
    locked_quantity: Decimal = Decimal("0.0")
    last_price: Decimal = Decimal("0.0")

    def market_value(self, current_price: Optional[Decimal] = None) -> Decimal:
        """Calculate market value of the position in quote currency."""
        if self.quantity == Decimal("0.0"):
            return Decimal("0.0")
        price = (
            current_price
            if current_price is not None
            else (self.last_price if self.last_price > Decimal("0.0") else self.average_entry_price)
        )
        return self.quantity * price

    def unrealized_pnl(self, current_price: Optional[Decimal] = None) -> Decimal:
        """Calculate mark-to-market unrealized profit or loss."""
        if self.quantity == Decimal("0.0") or self.average_entry_price == Decimal("0.0"):
            return Decimal("0.0")
        price = (
            current_price
            if current_price is not None
            else (self.last_price if self.last_price > Decimal("0.0") else self.average_entry_price)
        )
        return self.quantity * (price - self.average_entry_price)

    def unrealized_pnl_pct(self, current_price: Optional[Decimal] = None) -> Decimal:
        """Calculate mark-to-market unrealized return percentage."""
        if self.quantity == Decimal("0.0") or self.average_entry_price == Decimal("0.0"):
            return Decimal("0.0")
        price = (
            current_price
            if current_price is not None
            else (self.last_price if self.last_price > Decimal("0.0") else self.average_entry_price)
        )
        return ((price - self.average_entry_price) / self.average_entry_price) * Decimal("100.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to a JSON-serializable dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "average_entry_price": str(self.average_entry_price),
            "cost_basis": str(self.cost_basis),
            "locked_quantity": str(self.locked_quantity),
            "last_price": str(self.last_price),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Position:
        """Reconstruct a Position instance from a dictionary."""
        return cls(
            symbol=data["symbol"],
            quantity=Decimal(str(data.get("quantity", "0.0"))),
            average_entry_price=Decimal(str(data.get("average_entry_price", "0.0"))),
            cost_basis=Decimal(str(data.get("cost_basis", "0.0"))),
            locked_quantity=Decimal(str(data.get("locked_quantity", "0.0"))),
            last_price=Decimal(str(data.get("last_price", "0.0"))),
        )


@dataclass
class TradeRecord:
    """Immutable audit log record for an executed trade fill."""

    trade_id: str
    timestamp_ms: int
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET' or 'LIMIT'
    executed_price: Decimal
    raw_price: Decimal
    quantity: Decimal
    quote_notional: Decimal
    fee: Decimal
    realized_pnl: Decimal
    cash_after: Decimal
    equity_after: Decimal
    fee_rate: Decimal = Decimal("0.001")
    slippage: Decimal = Decimal("0.0005")
    order_id: Optional[str] = None
    exit_reason: Optional[str] = None

    @property
    def fill_price(self) -> Decimal:
        """Alias for executed_price for API interoperability."""
        return self.executed_price

    @property
    def portfolio_balance(self) -> Decimal:
        """Alias for equity_after representing portfolio balance after trade."""
        return self.equity_after

    @property
    def timestamp(self) -> datetime:
        """Get timezone-aware UTC datetime of the trade execution."""
        return datetime.fromtimestamp(self.timestamp_ms / 1000.0, tz=timezone.utc)

    @property
    def timestamp_iso(self) -> str:
        """ISO-8601 formatted timestamp string."""
        return self.timestamp.isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade record to dictionary representation."""
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "timestamp_ms": self.timestamp_ms,
            "datetime_utc": self.timestamp_iso,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "executed_price": str(self.executed_price),
            "raw_price": str(self.raw_price),
            "quantity": str(self.quantity),
            "quote_notional": str(self.quote_notional),
            "fee": str(self.fee),
            "fee_rate": str(self.fee_rate),
            "slippage": str(self.slippage),
            "realized_pnl": str(self.realized_pnl),
            "cash_after": str(self.cash_after),
            "equity_after": str(self.equity_after),
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TradeRecord:
        """Reconstruct TradeRecord from dictionary."""
        return cls(
            trade_id=str(data["trade_id"]),
            timestamp_ms=int(data["timestamp_ms"]),
            symbol=str(data["symbol"]),
            side=str(data["side"]).upper(),
            order_type=str(data.get("order_type", "MARKET")).upper(),
            executed_price=Decimal(str(data["executed_price"])),
            raw_price=Decimal(str(data.get("raw_price", data["executed_price"]))),
            quantity=Decimal(str(data["quantity"])),
            quote_notional=Decimal(str(data["quote_notional"])),
            fee=Decimal(str(data.get("fee", "0.0"))),
            realized_pnl=Decimal(str(data.get("realized_pnl", "0.0"))),
            cash_after=Decimal(str(data["cash_after"])),
            equity_after=Decimal(str(data["equity_after"])),
            fee_rate=Decimal(str(data.get("fee_rate", "0.001"))),
            slippage=Decimal(str(data.get("slippage", "0.0005"))),
            order_id=data.get("order_id"),
            exit_reason=data.get("exit_reason"),
        )


class VirtualLedger:
    """
    Virtual Ledger tracking portfolio cash, asset holdings, and historical trades.

    Guarantees strict conservation of funds, non-negative cash balances,
    and exact Decimal math for paper trading and backtesting parity.
    """

    def __init__(
        self,
        initial_cash: Union[Decimal, float, int, str] = Decimal("10.00"),
        quote_currency: str = "EUR",
    ) -> None:
        try:
            parsed_cash = Decimal(str(initial_cash))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid initial cash: {initial_cash}") from exc

        if parsed_cash <= Decimal("0.0"):
            raise ValueError(f"Initial balance must be > 0, got {parsed_cash}")

        self.quote_currency: str = quote_currency.upper()
        self.initial_cash: Decimal = parsed_cash
        self.cash: Decimal = parsed_cash
        self.locked_cash: Decimal = Decimal("0.00")
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: Decimal = Decimal("0.00")
        self.total_fees: Decimal = Decimal("0.00")
        self.trades: List[TradeRecord] = []
        self.equity_history: List[Tuple[int, Decimal]] = []  # (timestamp_ms, equity)
        self._trade_listeners: List[Callable[[TradeRecord], None]] = []

    @property
    def available_cash(self) -> Decimal:
        """Unencumbered cash available for opening new positions."""
        return max(Decimal("0.00"), self.cash - self.locked_cash)

    @property
    def trade_count(self) -> int:
        """Total number of executed trades."""
        return len(self.trades)

    @property
    def total_trades(self) -> int:
        """Total number of executed trades."""
        return len(self.trades)

    @property
    def winning_trades_count(self) -> int:
        """Count of closed sell trades with net positive realized PnL."""
        return sum(1 for t in self.trades if t.side == "SELL" and t.realized_pnl > Decimal("0.0"))

    @property
    def losing_trades_count(self) -> int:
        """Count of closed sell trades with net negative realized PnL."""
        return sum(1 for t in self.trades if t.side == "SELL" and t.realized_pnl < Decimal("0.0"))

    def add_trade_listener(self, callback: Callable[[TradeRecord], None]) -> None:
        """Register a callback observer to be notified upon trade execution."""
        if callback not in self._trade_listeners:
            self._trade_listeners.append(callback)

    def remove_trade_listener(self, callback: Callable[[TradeRecord], None]) -> None:
        """Remove a registered trade callback observer."""
        if callback in self._trade_listeners:
            self._trade_listeners.remove(callback)

    def attach_notification_service(self, service: Any) -> None:
        """Convenience helper to forward executed trades to a NotificationService."""
        def _listener(trade: TradeRecord) -> None:
            try:
                from trading_bot.notifications.models import TradeExecutedPayload
                pos = self.get_position(trade.symbol)
                payload = TradeExecutedPayload(
                    order_id=trade.order_id or trade.trade_id,
                    symbol=trade.symbol,
                    side=trade.side,
                    order_type=trade.order_type,
                    fill_price=float(trade.executed_price),
                    quantity=float(trade.quantity),
                    notional_value=float(trade.quote_notional),
                    fee=float(trade.fee),
                    fee_currency=self.quote_currency,
                    portfolio_balance=float(trade.equity_after),
                    cash_balance=float(trade.cash_after),
                    position_qty_after=float(pos.quantity),
                    realized_pnl=float(trade.realized_pnl),
                    timestamp=trade.timestamp,
                )
                if hasattr(service, "dispatch"):
                    service.dispatch(payload)
                elif callable(service):
                    service(payload)
            except Exception:
                payload_dict = {
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "fill_price": trade.executed_price,
                    "quantity": trade.quantity,
                    "fee": trade.fee,
                    "portfolio_balance": trade.equity_after,
                    "timestamp": trade.timestamp,
                    "realized_pnl": trade.realized_pnl,
                }
                if hasattr(service, "dispatch"):
                    service.dispatch(payload_dict)
                elif callable(service):
                    service(payload_dict)

        self.add_trade_listener(_listener)

    def get_position(self, symbol: str) -> Position:
        """Retrieve existing position or create a clean zeroed position."""
        normalized_sym = symbol.upper()
        if normalized_sym not in self.positions:
            self.positions[normalized_sym] = Position(symbol=normalized_sym)
        return self.positions[normalized_sym]

    def has_position(self, symbol: str) -> bool:
        """Return True if currently holding positive base asset units."""
        normalized_sym = symbol.upper()
        return normalized_sym in self.positions and self.positions[normalized_sym].quantity > Decimal("0.0")

    def total_equity(self, current_prices: Optional[Dict[str, Union[Decimal, float, str]]] = None) -> Decimal:
        """
        Compute mark-to-market total portfolio equity.
        Total Equity = Cash + Locked Cash + Sum(Position Quantity * Mark Price).
        """
        price_map: Dict[str, Decimal] = {}
        if current_prices:
            for k, v in current_prices.items():
                price_map[k.upper()] = Decimal(str(v))

        asset_value = Decimal("0.00")
        for sym, pos in self.positions.items():
            if pos.quantity > Decimal("0.0"):
                mark_price = price_map.get(sym, pos.average_entry_price)
                asset_value += pos.quantity * mark_price

        return self.cash + self.locked_cash + asset_value

    def unrealized_pnl(self, current_prices: Optional[Dict[str, Union[Decimal, float, str]]] = None) -> Decimal:
        """Compute cumulative unrealized profit or loss across all open positions."""
        price_map: Dict[str, Decimal] = {}
        if current_prices:
            for k, v in current_prices.items():
                price_map[k.upper()] = Decimal(str(v))

        total_unrealized = Decimal("0.00")
        for sym, pos in self.positions.items():
            if pos.quantity > Decimal("0.0"):
                mark_price = price_map.get(sym, pos.average_entry_price)
                total_unrealized += pos.unrealized_pnl(mark_price)

        return total_unrealized

    def roi_pct(self, current_prices: Optional[Dict[str, Union[Decimal, float, str]]] = None) -> Decimal:
        """Calculate return on investment percentage relative to initial cash."""
        equity = self.total_equity(current_prices)
        return ((equity - self.initial_cash) / self.initial_cash) * Decimal("100.0")

    def lock_cash(self, amount: Union[Decimal, float, str]) -> None:
        """Lock cash for a pending limit buy order."""
        dec_amount = Decimal(str(amount))
        if dec_amount <= Decimal("0.0"):
            raise ValueError(f"Lock amount must be positive: {dec_amount}")
        if self.available_cash < dec_amount:
            raise InsufficientFundsError(f"Insufficient available cash to lock {dec_amount}: available {self.available_cash}")
        self.locked_cash += dec_amount

    def unlock_cash(self, amount: Union[Decimal, float, str]) -> None:
        """Release locked cash upon order cancellation or completion."""
        dec_amount = Decimal(str(amount))
        if dec_amount > self.locked_cash:
            self.locked_cash = Decimal("0.00")
        else:
            self.locked_cash -= dec_amount

    def deposit(self, amount: Union[Decimal, float, str]) -> Decimal:
        """Deposit fiat cash into the ledger and update initial cash base."""
        dec_amount = Decimal(str(amount))
        if dec_amount <= Decimal("0.0"):
            raise ValueError(f"Deposit amount must be positive: {dec_amount}")
        self.cash += dec_amount
        self.initial_cash += dec_amount
        return self.cash

    def set_cash(self, new_balance: Union[Decimal, float, str]) -> None:
        """Directly adjust cash balance to match exchange live balance."""
        dec_amount = Decimal(str(new_balance))
        if dec_amount < Decimal("0.0"):
            raise ValueError("Cash cannot be negative")
        self.cash = dec_amount

    def record_buy(
        self,
        symbol: str,
        quantity: Union[Decimal, float, str],
        executed_price: Union[Decimal, float, str],
        fee: Union[Decimal, float, str],
        timestamp_ms: int,
        raw_price: Optional[Union[Decimal, float, str]] = None,
        order_type: str = "MARKET",
        fee_rate: Union[Decimal, float, str] = Decimal("0.001"),
        slippage: Union[Decimal, float, str] = Decimal("0.0005"),
        order_id: Optional[str] = None,
    ) -> TradeRecord:
        """
        Record an executed BUY transaction.
        Deducts notional + fee from cash, updates asset position and weighted cost basis.
        """
        sym = symbol.upper()
        qty = Decimal(str(quantity))
        exec_px = Decimal(str(executed_price))
        fee_dec = Decimal(str(fee))
        raw_px = Decimal(str(raw_price)) if raw_price is not None else exec_px
        fee_rt = Decimal(str(fee_rate))
        slip = Decimal(str(slippage))

        if qty <= Decimal("0.0"):
            raise ValueError(f"Quantity must be positive: {qty}")
        if exec_px <= Decimal("0.0"):
            raise ValueError(f"Executed price must be positive: {exec_px}")
        if fee_dec < Decimal("0.0"):
            raise ValueError(f"Fee cannot be negative: {fee_dec}")

        notional = qty * exec_px
        total_cost = notional + fee_dec

        if self.cash < total_cost:
            raise InsufficientFundsError(
                f"Insufficient cash for BUY: have {self.cash:.4f} {self.quote_currency}, "
                f"need {total_cost:.4f} {self.quote_currency} (notional {notional:.4f} + fee {fee_dec:.4f})"
            )

        # Ledger state transitions
        self.cash -= total_cost
        self.total_fees += fee_dec

        pos = self.get_position(sym)
        new_qty = pos.quantity + qty
        new_cost_basis = pos.cost_basis + notional
        pos.average_entry_price = new_cost_basis / new_qty
        pos.quantity = new_qty
        pos.cost_basis = new_cost_basis
        pos.last_price = exec_px

        equity_after = self.total_equity({sym: exec_px})
        self.record_equity_snapshot(timestamp_ms, {sym: exec_px})

        record = TradeRecord(
            trade_id=f"tr_{len(self.trades) + 1:05d}",
            timestamp_ms=timestamp_ms,
            symbol=sym,
            side="BUY",
            order_type=order_type.upper(),
            executed_price=exec_px,
            raw_price=raw_px,
            quantity=qty,
            quote_notional=notional,
            fee=fee_dec,
            realized_pnl=Decimal("0.00"),
            cash_after=self.cash,
            equity_after=equity_after,
            fee_rate=fee_rt,
            slippage=slip,
            order_id=order_id,
        )
        self.trades.append(record)

        for listener in self._trade_listeners:
            listener(record)

        return record

    def record_sell(
        self,
        symbol: str,
        quantity: Union[Decimal, float, str],
        executed_price: Union[Decimal, float, str],
        fee: Union[Decimal, float, str],
        timestamp_ms: int,
        raw_price: Optional[Union[Decimal, float, str]] = None,
        order_type: str = "MARKET",
        fee_rate: Union[Decimal, float, str] = Decimal("0.001"),
        slippage: Union[Decimal, float, str] = Decimal("0.0005"),
        order_id: Optional[str] = None,
        exit_reason: Optional[str] = None,
    ) -> TradeRecord:
        """
        Record an executed SELL transaction.
        Credits net proceeds (notional - fee) to cash, computes realized PnL,
        and decrements asset holdings.
        """
        sym = symbol.upper()
        qty = Decimal(str(quantity))
        exec_px = Decimal(str(executed_price))
        fee_dec = Decimal(str(fee))
        raw_px = Decimal(str(raw_price)) if raw_price is not None else exec_px
        fee_rt = Decimal(str(fee_rate))
        slip = Decimal(str(slippage))

        if qty <= Decimal("0.0"):
            raise ValueError(f"Quantity must be positive: {qty}")
        if exec_px <= Decimal("0.0"):
            raise ValueError(f"Executed price must be positive: {exec_px}")
        if fee_dec < Decimal("0.0"):
            raise ValueError(f"Fee cannot be negative: {fee_dec}")

        pos = self.get_position(sym)
        if pos.quantity < qty:
            raise InsufficientPositionError(
                f"Insufficient position for SELL {sym}: holding {pos.quantity}, requested {qty}"
            )

        notional = qty * exec_px
        net_proceeds = notional - fee_dec
        cost_of_sold = qty * pos.average_entry_price
        trade_realized_pnl = net_proceeds - cost_of_sold

        # Ledger state transitions
        self.cash += net_proceeds
        self.total_fees += fee_dec
        self.realized_pnl += trade_realized_pnl

        pos.quantity -= qty
        pos.cost_basis -= cost_of_sold
        if pos.quantity <= Decimal("0.0"):
            pos.quantity = Decimal("0.0")
            pos.cost_basis = Decimal("0.0")
            pos.average_entry_price = Decimal("0.0")
        pos.last_price = exec_px

        equity_after = self.total_equity({sym: exec_px})
        self.record_equity_snapshot(timestamp_ms, {sym: exec_px})

        record = TradeRecord(
            trade_id=f"tr_{len(self.trades) + 1:05d}",
            timestamp_ms=timestamp_ms,
            symbol=sym,
            side="SELL",
            order_type=order_type.upper(),
            executed_price=exec_px,
            raw_price=raw_px,
            quantity=qty,
            quote_notional=notional,
            fee=fee_dec,
            realized_pnl=trade_realized_pnl,
            cash_after=self.cash,
            equity_after=equity_after,
            fee_rate=fee_rt,
            slippage=slip,
            order_id=order_id,
            exit_reason=exit_reason,
        )
        self.trades.append(record)

        for listener in self._trade_listeners:
            listener(record)

        return record

    def record_equity_snapshot(
        self,
        timestamp_ms: int,
        current_prices: Optional[Dict[str, Union[Decimal, float, str]]] = None,
    ) -> Decimal:
        """Record an equity curve point at timestamp_ms."""
        equity = self.total_equity(current_prices)
        self.equity_history.append((timestamp_ms, equity))
        return equity

    def get_summary(
        self,
        current_prices: Optional[Dict[str, Union[Decimal, float, str]]] = None,
    ) -> Dict[str, Any]:
        """Return a structured, human-readable summary of portfolio status."""
        equity = self.total_equity(current_prices)
        unrealized = self.unrealized_pnl(current_prices)
        roi = self.roi_pct(current_prices)
        return {
            "initial_cash": float(self.initial_cash),
            "cash": float(self.cash),
            "locked_cash": float(self.locked_cash),
            "total_equity": float(equity),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(unrealized),
            "total_fees": float(self.total_fees),
            "roi_pct": float(roi),
            "trade_count": len(self.trades),
            "winning_trades": self.winning_trades_count,
            "losing_trades": self.losing_trades_count,
            "quote_currency": self.quote_currency,
            "positions": {
                sym: pos.to_dict()
                for sym, pos in self.positions.items()
                if pos.quantity > Decimal("0.0")
            },
        }

    def reset(self, initial_cash: Optional[Union[Decimal, float, str]] = None) -> None:
        """Reset ledger state back to initial state."""
        start = Decimal(str(initial_cash)) if initial_cash is not None else self.initial_cash
        if start <= Decimal("0.0"):
            raise ValueError("Initial balance must be > 0")
        self.initial_cash = start
        self.cash = start
        self.locked_cash = Decimal("0.00")
        self.positions.clear()
        self.realized_pnl = Decimal("0.00")
        self.total_fees = Decimal("0.00")
        self.trades.clear()
        self.equity_history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete ledger state to JSON-compatible dictionary."""
        return {
            "quote_currency": self.quote_currency,
            "initial_cash": str(self.initial_cash),
            "cash": str(self.cash),
            "locked_cash": str(self.locked_cash),
            "realized_pnl": str(self.realized_pnl),
            "total_fees": str(self.total_fees),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "trades": [t.to_dict() for t in self.trades],
            "equity_history": [[ts, str(eq)] for ts, eq in self.equity_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VirtualLedger:
        """Recreate VirtualLedger state from dictionary."""
        ledger = cls(
            initial_cash=Decimal(data["initial_cash"]),
            quote_currency=data.get("quote_currency", "EUR"),
        )
        ledger.cash = Decimal(data["cash"])
        ledger.locked_cash = Decimal(data.get("locked_cash", "0.00"))
        ledger.realized_pnl = Decimal(data.get("realized_pnl", "0.00"))
        ledger.total_fees = Decimal(data.get("total_fees", "0.00"))

        ledger.positions = {
            k: Position.from_dict(v) for k, v in data.get("positions", {}).items()
        }
        ledger.trades = [TradeRecord.from_dict(t) for t in data.get("trades", [])]
        ledger.equity_history = [
            (int(row[0]), Decimal(str(row[1]))) for row in data.get("equity_history", [])
        ]
        return ledger

    def save_state(self, filepath: Union[str, Path]) -> None:
        """Persist ledger state to JSON file on disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_state(cls, filepath: Union[str, Path]) -> VirtualLedger:
        """Load ledger state from JSON file on disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

"""Notification subsystem data models and event schemas.

Defines strictly typed Pydantic v2 event models for trade executions,
portfolio balance compounding updates, risk alerts, and delivery statuses.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union
import uuid
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventType(str, Enum):
    """Enumeration of notification event types."""
    TRADE_EXECUTED = "TRADE_EXECUTED"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    RISK_ALERT = "RISK_ALERT"
    SYSTEM_INFO = "SYSTEM_INFO"


class OrderSide(str, Enum):
    """Enumeration of order sides."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Enumeration of order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class BaseNotificationEvent(BaseModel):
    """Base schema for all notification events."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID string for event tracking"
    )
    event_type: EventType = Field(..., description="Type of notification event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of event creation"
    )
    mode: str = Field(
        default="PAPER",
        description="Execution mode: PAPER, BACKTEST, LIVE, TEST"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible context metadata"
    )

    @model_validator(mode="before")
    @classmethod
    def convert_event_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and "event_type" in data and isinstance(data["event_type"], str):
            try:
                data["event_type"] = EventType(data["event_type"])
            except ValueError:
                pass
        return data



class TradeExecutedPayload(BaseNotificationEvent):
    """Payload representing an executed trade fill and virtual ledger state.

    Captures trade details: symbol, side, fill price, quantity, fee,
    portfolio balance, timestamp, and realized PnL.
    """
    event_type: EventType = Field(
        default=EventType.TRADE_EXECUTED,
        description="Event type identifier"
    )
    order_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique client or exchange order identifier"
    )
    symbol: str = Field(
        ...,
        description="Market trading pair e.g. 'BTC/EUR', 'ETH/EUR'"
    )
    side: OrderSide = Field(
        ...,
        description="Order execution side: BUY or SELL"
    )
    order_type: OrderType = Field(
        default=OrderType.MARKET,
        description="Order type e.g. MARKET or LIMIT"
    )
    fill_price: float = Field(
        ...,
        gt=0,
        description="Execution fill price in quote currency (EUR)"
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Executed quantity in base asset (e.g. BTC)"
    )
    notional_value: Optional[float] = Field(
        default=None,
        description="Total order value in quote currency (fill_price * quantity)"
    )
    fee: float = Field(
        default=0.0,
        ge=0,
        description="Deducted commission/trading fee in fee currency"
    )
    fee_currency: str = Field(
        default="EUR",
        description="Currency in which fee was charged"
    )
    portfolio_balance: float = Field(
        ...,
        ge=0,
        description="Total portfolio equity in EUR after trade execution"
    )
    cash_balance: Optional[float] = Field(
        default=None,
        description="Remaining unallocated cash balance in EUR"
    )
    position_qty_after: Optional[float] = Field(
        default=None,
        description="Current holding quantity of base asset after trade execution"
    )
    realized_pnl: Optional[float] = Field(
        default=None,
        description="Realized PnL in EUR (computed for closing SELL orders)"
    )
    realized_pnl_pct: Optional[float] = Field(
        default=None,
        description="Realized PnL percentage on closed position"
    )
    strategy_signal: Optional[str] = Field(
        default=None,
        description="Triggering strategy signal identifier"
    )

    @model_validator(mode="before")
    @classmethod
    def convert_types_before(cls, data: Any) -> Any:
        """Convert Decimal and string values to standard model types before validation."""
        if isinstance(data, dict):
            # Convert Decimal values to float for compatibility with Decimal inputs
            for key in ("fill_price", "quantity", "notional_value", "fee", "portfolio_balance",
                        "cash_balance", "position_qty_after", "realized_pnl", "realized_pnl_pct"):
                if key in data and isinstance(data[key], Decimal):
                    data[key] = float(data[key])
            # Normalize side if passed as string
            if "side" in data and isinstance(data["side"], str):
                try:
                    data["side"] = OrderSide(data["side"].upper())
                except ValueError:
                    pass
            # Normalize order_type if passed as string
            if "order_type" in data and isinstance(data["order_type"], str):
                try:
                    data["order_type"] = OrderType(data["order_type"].upper())
                except ValueError:
                    pass
        return data

    @model_validator(mode="after")
    def compute_derived_fields(self) -> "TradeExecutedPayload":
        """Compute notional value and default cash balance if omitted."""
        if self.notional_value is None:
            self.notional_value = round(self.fill_price * self.quantity, 6)
        return self


class PortfolioUpdatePayload(BaseNotificationEvent):
    """Payload representing periodic portfolio valuation and compounding progress.

    Tracks progress toward the 10,000€ target starting from 10€ initial capital.
    """
    event_type: EventType = Field(
        default=EventType.PORTFOLIO_UPDATE,
        description="Event type identifier"
    )
    total_equity: float = Field(
        ...,
        ge=0,
        description="Total mark-to-market portfolio equity in EUR"
    )
    cash_balance: float = Field(
        ...,
        ge=0,
        description="Available unallocated liquid cash in EUR"
    )
    unrealized_pnl: float = Field(
        default=0.0,
        description="Unrealized PnL on open positions in EUR"
    )
    realized_pnl_cumulative: float = Field(
        default=0.0,
        description="Cumulative realized PnL across all trades in EUR"
    )
    roi_pct: Optional[float] = Field(
        default=None,
        description="Total Return on Investment percentage based on initial capital"
    )
    progress_to_target_pct: Optional[float] = Field(
        default=None,
        description="Progress percentage towards 10,000€ target"
    )
    target_equity: float = Field(
        default=10000.0,
        gt=0,
        description="Target compounding goal in EUR (default: 10,000€)"
    )
    starting_equity: float = Field(
        default=10.0,
        gt=0,
        description="Initial budget in EUR (default: 10€)"
    )
    open_positions_count: int = Field(
        default=0,
        ge=0,
        description="Number of currently open asset positions"
    )
    total_trades_count: int = Field(
        default=0,
        ge=0,
        description="Total count of executed trades"
    )
    win_rate: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Win rate percentage of closed trades"
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases_and_types(cls, data: Any) -> Any:
        """Support 'equity' and 'cash' aliases and Decimal conversion."""
        if isinstance(data, dict):
            # Decimal conversions
            for key in ("total_equity", "cash_balance", "unrealized_pnl", "realized_pnl_cumulative",
                        "roi_pct", "progress_to_target_pct", "target_equity", "starting_equity", "win_rate"):
                if key in data and isinstance(data[key], Decimal):
                    data[key] = float(data[key])
            # Alias mapping
            if "equity" in data and "total_equity" not in data:
                val = data.pop("equity")
                data["total_equity"] = float(val) if isinstance(val, Decimal) else val
            if "cash" in data and "cash_balance" not in data:
                val = data.pop("cash")
                data["cash_balance"] = float(val) if isinstance(val, Decimal) else val
            if "roi" in data and "roi_pct" not in data:
                val = data.pop("roi")
                data["roi_pct"] = float(val) if isinstance(val, Decimal) else val
            if "progress" in data and "progress_to_target_pct" not in data:
                val = data.pop("progress")
                data["progress_to_target_pct"] = float(val) if isinstance(val, Decimal) else val
        return data

    @model_validator(mode="after")
    def compute_metrics(self) -> "PortfolioUpdatePayload":
        """Compute ROI percentage and progress towards 10,000€ target."""
        if self.roi_pct is None:
            if self.starting_equity > 0:
                self.roi_pct = round(
                    ((self.total_equity - self.starting_equity) / self.starting_equity) * 100.0,
                    4
                )
            else:
                self.roi_pct = 0.0

        if self.progress_to_target_pct is None:
            denom = self.target_equity - self.starting_equity
            if denom > 0:
                calc_progress = ((self.total_equity - self.starting_equity) / denom) * 100.0
                self.progress_to_target_pct = max(0.0, round(calc_progress, 4))
            else:
                self.progress_to_target_pct = 0.0
        return self

    @property
    def equity(self) -> float:
        """Alias for total_equity."""
        return self.total_equity

    @property
    def cash(self) -> float:
        """Alias for cash_balance."""
        return self.cash_balance

    @property
    def roi(self) -> float:
        """Alias for roi_pct."""
        return self.roi_pct if self.roi_pct is not None else 0.0

    @property
    def progress(self) -> float:
        """Alias for progress_to_target_pct."""
        return self.progress_to_target_pct if self.progress_to_target_pct is not None else 0.0


class RiskAlertPayload(BaseNotificationEvent):
    """Payload representing a risk warning, drawdown alert, or circuit breaker trigger."""
    event_type: EventType = Field(
        default=EventType.RISK_ALERT,
        description="Event type identifier"
    )
    severity: str = Field(
        default="WARNING",
        description="Severity level: INFO, WARNING, CRITICAL"
    )
    message: str = Field(
        ...,
        description="Descriptive alert explanation"
    )
    current_drawdown_pct: float = Field(
        default=0.0,
        description="Current portfolio drawdown percentage from peak equity"
    )
    max_drawdown_limit_pct: float = Field(
        default=8.0,
        description="Configured maximum allowable drawdown limit percentage"
    )
    action_taken: str = Field(
        default="LOG",
        description="Action executed e.g. 'CIRCUIT_BREAKER_HALT', 'REDUCE_SIZE', 'LOG'"
    )


class DeliveryStatus(BaseModel):
    """Delivery confirmation returned by notification channels."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(..., description="Whether delivery succeeded")
    channel_name: str = Field(..., description="Name of the notification channel")
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if delivery failed"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the delivery attempt"
    )

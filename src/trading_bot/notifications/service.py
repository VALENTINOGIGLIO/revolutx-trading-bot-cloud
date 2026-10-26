"""Central notification service dispatcher.

Coordinates fan-out delivery across registered notification channels
(Console, JSONL log, Telegram, and in-memory test interceptor).
Guarantees fault isolation so channel failures never disrupt trading loops.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Union

from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.channels.console import ConsoleChannel
from trading_bot.notifications.channels.csv_log import CsvTradeLogChannel, CsvTradeLogger
from trading_bot.notifications.channels.json_log import JsonLogChannel
from trading_bot.notifications.channels.telegram import TelegramChannel
from trading_bot.notifications.interceptor import NotificationInterceptor
from trading_bot.notifications.models import (
    BaseNotificationEvent,
    DeliveryStatus,
    OrderSide,
    OrderType,
    PortfolioUpdatePayload,
    RiskAlertPayload,
    TradeExecutedPayload,
)

logger = logging.getLogger("NotificationService")


class NotificationService:
    """Central event dispatcher for trading bot notifications."""

    def __init__(self, channels: Optional[List[BaseNotificationChannel]] = None):
        self._channels: Dict[str, BaseNotificationChannel] = {}
        self._lock = threading.RLock()

        if channels:
            for channel in channels:
                self.register_channel(channel)

    @property
    def channels(self) -> Dict[str, BaseNotificationChannel]:
        """Return a snapshot of registered channels."""
        with self._lock:
            return dict(self._channels)

    def register_channel(self, channel: BaseNotificationChannel) -> None:
        """Register a notification channel."""
        with self._lock:
            self._channels[channel.name] = channel
            logger.debug(f"Registered notification channel: '{channel.name}' (enabled={channel.is_enabled()})")

    def unregister_channel(self, name: str) -> Optional[BaseNotificationChannel]:
        """Unregister and return the channel by name, or None if not found."""
        with self._lock:
            channel = self._channels.pop(name, None)
            if channel:
                logger.debug(f"Unregistered notification channel: '{name}'")
            return channel

    def get_channel(self, name: str) -> Optional[BaseNotificationChannel]:
        """Retrieve a registered channel by name."""
        with self._lock:
            ch = self._channels.get(name)
            if ch is None and name in ("csv_log", "csv"):
                ch = self._channels.get("csv_trade_log")
            elif ch is None and name == "csv_trade_log":
                ch = self._channels.get("csv_log")
            return ch

    def has_channel(self, name: str) -> bool:
        """Check if a channel is registered."""
        with self._lock:
            if name in self._channels:
                return True
            if name in ("csv_log", "csv") and "csv_trade_log" in self._channels:
                return True
            if name == "csv_trade_log" and "csv_log" in self._channels:
                return True
            return False

    def clear_channels(self) -> None:
        """Unregister all channels."""
        with self._lock:
            self._channels.clear()

    def dispatch(self, event: BaseNotificationEvent) -> Dict[str, DeliveryStatus]:
        """Dispatch a notification event to all enabled channels.

        Exceptions in any channel are caught, logged, and isolated so the
        caller execution path (e.g. trading loop) is never interrupted.
        """
        results: Dict[str, DeliveryStatus] = {}

        with self._lock:
            channel_list = list(self._channels.values())

        for channel in channel_list:
            if not channel.is_enabled():
                continue

            try:
                status = channel.dispatch(event)
                # In case a custom channel returned None instead of DeliveryStatus
                if status is None:
                    status = DeliveryStatus(success=True, channel_name=channel.name)
                results[channel.name] = status
            except Exception as ex:
                logger.error(
                    f"Notification channel '{channel.name}' crashed during dispatch: {ex}",
                    exc_info=True,
                )
                results[channel.name] = DeliveryStatus(
                    success=False,
                    channel_name=channel.name,
                    error_message=str(ex),
                )

        return results

    @contextmanager
    def intercept(self, name: str = "interceptor") -> Iterator[NotificationInterceptor]:
        """Context manager to attach an in-memory interceptor for testing.

        Automatically unregisters the interceptor upon exiting the context block.

        Example:
            with notification_service.intercept() as interceptor:
                engine.execute_order(...)
                assert len(interceptor.captured_events) == 1
                assert interceptor.last_event.symbol == "BTC/EUR"
        """
        interceptor = NotificationInterceptor(service=self, name=name)
        self.register_channel(interceptor)
        try:
            yield interceptor
        finally:
            self.unregister_channel(name)

    def notify_trade(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        fill_price: float,
        quantity: float,
        fee: float = 0.0,
        portfolio_balance: float = 0.0,
        realized_pnl: Optional[float] = None,
        order_id: Optional[str] = None,
        order_type: Union[OrderType, str] = OrderType.MARKET,
        cash_balance: Optional[float] = None,
        position_qty_after: Optional[float] = None,
        fee_currency: str = "EUR",
        mode: str = "PAPER",
        strategy_signal: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        **metadata: Any,
    ) -> Dict[str, DeliveryStatus]:
        """Helper to construct and dispatch a TradeExecutedPayload."""
        ts = timestamp or datetime.now(timezone.utc)
        payload = TradeExecutedPayload(
            symbol=symbol,
            side=side if isinstance(side, OrderSide) else OrderSide(str(side).upper()),
            fill_price=fill_price,
            quantity=quantity,
            fee=fee,
            portfolio_balance=portfolio_balance,
            realized_pnl=realized_pnl,
            order_id=order_id or str(TradeExecutedPayload.model_fields["order_id"].default_factory()),
            order_type=order_type if isinstance(order_type, OrderType) else OrderType(str(order_type).upper()),
            cash_balance=cash_balance,
            position_qty_after=position_qty_after,
            fee_currency=fee_currency,
            mode=mode,
            strategy_signal=strategy_signal,
            timestamp=ts,
            metadata=metadata,
        )
        return self.dispatch(payload)

    def notify_portfolio(
        self,
        total_equity: float,
        cash_balance: float,
        unrealized_pnl: float = 0.0,
        realized_pnl_cumulative: float = 0.0,
        roi_pct: Optional[float] = None,
        progress_to_target_pct: Optional[float] = None,
        target_equity: float = 10000.0,
        starting_equity: float = 10.0,
        total_trades_count: int = 0,
        open_positions_count: int = 0,
        win_rate: float = 0.0,
        mode: str = "PAPER",
        timestamp: Optional[datetime] = None,
        **metadata: Any,
    ) -> Dict[str, DeliveryStatus]:
        """Helper to construct and dispatch a PortfolioUpdatePayload."""
        ts = timestamp or datetime.now(timezone.utc)
        payload = PortfolioUpdatePayload(
            total_equity=total_equity,
            cash_balance=cash_balance,
            unrealized_pnl=unrealized_pnl,
            realized_pnl_cumulative=realized_pnl_cumulative,
            roi_pct=roi_pct,
            progress_to_target_pct=progress_to_target_pct,
            target_equity=target_equity,
            starting_equity=starting_equity,
            total_trades_count=total_trades_count,
            open_positions_count=open_positions_count,
            win_rate=win_rate,
            mode=mode,
            timestamp=ts,
            metadata=metadata,
        )
        return self.dispatch(payload)

    def notify_risk(
        self,
        message: str,
        severity: str = "WARNING",
        current_drawdown_pct: float = 0.0,
        max_drawdown_limit_pct: float = 8.0,
        action_taken: str = "LOG",
        mode: str = "PAPER",
        timestamp: Optional[datetime] = None,
        **metadata: Any,
    ) -> Dict[str, DeliveryStatus]:
        """Helper to construct and dispatch a RiskAlertPayload."""
        ts = timestamp or datetime.now(timezone.utc)
        payload = RiskAlertPayload(
            message=message,
            severity=severity,
            current_drawdown_pct=current_drawdown_pct,
            max_drawdown_limit_pct=max_drawdown_limit_pct,
            action_taken=action_taken,
            mode=mode,
            timestamp=ts,
            metadata=metadata,
        )
        return self.dispatch(payload)

    def close(self) -> None:
        """Close all registered channels."""
        with self._lock:
            for channel in self._channels.values():
                try:
                    channel.close()
                except Exception as ex:
                    logger.warning(f"Error closing channel '{channel.name}': {ex}")


def create_default_notification_service(
    enable_console: bool = True,
    json_log_path: Optional[str] = "logs/notifications.jsonl",
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    trades_csv_path: Optional[Union[str, "Path"]] = None,
) -> NotificationService:
    """Convenience factory to initialize a NotificationService with standard channels."""
    channels: List[BaseNotificationChannel] = []

    if enable_console:
        channels.append(ConsoleChannel(enabled=True))

    if json_log_path:
        channels.append(JsonLogChannel(log_path=json_log_path, enabled=True))

    if telegram_bot_token and telegram_chat_id:
        channels.append(
            TelegramChannel(
                bot_token=telegram_bot_token,
                chat_id=telegram_chat_id,
                enabled=True,
            )
        )

    if trades_csv_path:
        channels.append(CsvTradeLogChannel(csv_path=trades_csv_path, enabled=True))

    return NotificationService(channels=channels)

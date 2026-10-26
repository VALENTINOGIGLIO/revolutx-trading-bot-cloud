"""Trading Bot Notification Subsystem.

Provides multi-channel telemetry (Console, JSONL log, Telegram) and an
in-memory interceptor for deterministic unit and integration tests.
"""

from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.channels.console import ConsoleChannel
from trading_bot.notifications.channels.json_log import JsonLogChannel
from trading_bot.notifications.channels.telegram import TelegramChannel
from trading_bot.notifications.telegram_bot import (
    TelegramAuthManager,
    TelegramBotListener,
    TelegramPushChannel,
)
from trading_bot.notifications.interceptor import NotificationInterceptor
from trading_bot.notifications.models import (
    BaseNotificationEvent,
    DeliveryStatus,
    EventType,
    OrderSide,
    OrderType,
    PortfolioUpdatePayload,
    RiskAlertPayload,
    TradeExecutedPayload,
)
from trading_bot.notifications.service import (
    NotificationService,
    create_default_notification_service,
)

__all__ = [
    # Dispatcher Service
    "NotificationService",
    "create_default_notification_service",
    # Test Interceptor
    "NotificationInterceptor",
    # Base and Channels
    "BaseNotificationChannel",
    "ConsoleChannel",
    "JsonLogChannel",
    "TelegramChannel",
    "TelegramPushChannel",
    "TelegramBotListener",
    "TelegramAuthManager",
    # Models & Enums
    "BaseNotificationEvent",
    "EventType",
    "OrderSide",
    "OrderType",
    "TradeExecutedPayload",
    "PortfolioUpdatePayload",
    "RiskAlertPayload",
    "DeliveryStatus",
]

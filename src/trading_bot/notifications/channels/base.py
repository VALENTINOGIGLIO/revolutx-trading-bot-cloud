"""Abstract base class for notification channels.

Defines the contract for pluggable notification sinks (Console, JSONL, Telegram, Interceptor).
"""

from abc import ABC, abstractmethod
from typing import Optional
from trading_bot.notifications.models import BaseNotificationEvent, DeliveryStatus


class BaseNotificationChannel(ABC):
    """Abstract base class for all notification output adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this notification channel."""
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if this channel is configured and active."""
        pass

    @abstractmethod
    def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
        """Deliver the notification event to the destination channel.

        Must trap channel-specific delivery exceptions and return a DeliveryStatus
        indicating failure instead of propagating unhandled exceptions to the caller.
        """
        pass

    def send(self, payload: BaseNotificationEvent) -> DeliveryStatus:
        """Alias for dispatch to provide backward and cross-agent API compatibility."""
        return self.dispatch(payload)

    def close(self) -> None:
        """Cleanup channel resources, flush open file buffers, or shutdown network pools."""
        pass

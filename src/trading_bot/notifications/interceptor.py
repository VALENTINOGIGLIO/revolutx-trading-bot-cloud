"""In-memory event interceptor for deterministic unit and integration testing.

Satisfies Acceptance Criteria 2 (AC2) by intercepting dispatched notification
payloads in memory without disk I/O, external network calls, or flakiness.
"""

from typing import Any, Iterator, List, Optional, Union
from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.models import (
    BaseNotificationEvent,
    DeliveryStatus,
    EventType,
    PortfolioUpdatePayload,
    RiskAlertPayload,
    TradeExecutedPayload,
)


class NotificationInterceptor(BaseNotificationChannel):
    """In-memory notification interceptor and context manager for tests."""

    def __init__(
        self,
        service: Optional[Any] = None,
        name: str = "interceptor",
        enabled: bool = True,
    ):
        self._name = name
        self._service = service
        self._enabled = enabled
        self.captured_events: List[BaseNotificationEvent] = []

    @property
    def name(self) -> str:
        return self._name

    def is_enabled(self) -> bool:
        return self._enabled

    def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
        """Capture event synchronously in memory."""
        if not self._enabled:
            return DeliveryStatus(success=True, channel_name=self.name)

        self.captured_events.append(event)
        return DeliveryStatus(success=True, channel_name=self.name)

    def clear(self) -> None:
        """Clear all captured events."""
        self.captured_events.clear()

    @property
    def last_event(self) -> Optional[BaseNotificationEvent]:
        """Return the most recently captured event, or None if queue is empty."""
        return self.captured_events[-1] if self.captured_events else None

    @property
    def trade_events(self) -> List[TradeExecutedPayload]:
        """Filter captured events for TradeExecutedPayload instances."""
        return [e for e in self.captured_events if isinstance(e, TradeExecutedPayload)]

    @property
    def portfolio_events(self) -> List[PortfolioUpdatePayload]:
        """Filter captured events for PortfolioUpdatePayload instances."""
        return [e for e in self.captured_events if isinstance(e, PortfolioUpdatePayload)]

    @property
    def risk_events(self) -> List[RiskAlertPayload]:
        """Filter captured events for RiskAlertPayload instances."""
        return [e for e in self.captured_events if isinstance(e, RiskAlertPayload)]

    def events_by_type(self, event_type: Union[EventType, str]) -> List[BaseNotificationEvent]:
        """Return all events matching a specific event type."""
        type_str = event_type.value if isinstance(event_type, EventType) else str(event_type)
        return [e for e in self.captured_events if e.event_type.value == type_str]

    @property
    def trade_count(self) -> int:
        """Count of captured trade events."""
        return len(self.trade_events)

    def __len__(self) -> int:
        return len(self.captured_events)

    def __getitem__(self, index: int) -> BaseNotificationEvent:
        return self.captured_events[index]

    def __iter__(self) -> Iterator[BaseNotificationEvent]:
        return iter(self.captured_events)

    def __enter__(self) -> "NotificationInterceptor":
        if self._service is not None:
            self._service.register_channel(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._service is not None:
            self._service.unregister_channel(self.name)

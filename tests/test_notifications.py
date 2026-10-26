"""Unit and Integration Tests for the Notification Subsystem.

Covers:
- Acceptance Criteria 2: Intercepting TradeExecutedPayload from a simulated trade in VirtualLedger.
- Tier 1 Feature Tests:
  1. TradeExecutedPayload model validation and Decimal type conversions.
  2. PortfolioUpdatePayload compounding progress calculation towards 10,000 EUR.
  3. RiskAlertPayload validation and attributes.
  4. ConsoleChannel formatting with Rich and plain ANSI fallback.
  5. JsonLogChannel file streaming, atomic writes, reading, and clearing.
  6. TelegramChannel graceful fallback when credentials are absent or invalid.
  7. TelegramChannel message markdown rendering for BUY, SELL, and Portfolio updates.
  8. NotificationService fan-out dispatching to multiple active channels.
- Tier 2 Boundary Tests:
  9. Channel fault isolation (crashing channel does not break other channels or engine).
  10. Invalid payload rejection (fill_price <= 0 or quantity <= 0 raising ValidationError).
  11. Telegram HTTP error resilience (HTTP 500 does not propagate exceptions).
"""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.channels.console import ConsoleChannel
from trading_bot.notifications.channels.json_log import JsonLogChannel
from trading_bot.notifications.channels.telegram import TelegramChannel
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
from trading_bot.notifications.service import NotificationService
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import ExecutionSimulator, OrderSignal


# =====================================================================
# MANDATORY ACCEPTANCE CRITERIA 2 TEST
# =====================================================================
def test_simulated_trade_notification_payload_intercepted():
    """
    CRITICAL MANDATORY TEST for Acceptance Criteria 2:
    Trigger a simulated trade in VirtualLedger and assert that the intercepted
    TradeExecutedPayload contains symbol, side, fill_price, quantity, fee,
    portfolio_balance, timestamp, realized_pnl.
    """
    notification_service = NotificationService()
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        exchange="binance",
        notification_service=notification_service,
    )

    with notification_service.intercept() as interceptor:
        assert len(interceptor.captured_events) == 0

        # 1. Trigger simulated BUY trade: 9.60 EUR budget at 60,000 EUR/BTC
        raw_price = Decimal("60000.00")
        buy_order = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            target_notional=Decimal("9.60"),
            order_id="ord_ac2_buy_001",
        )
        result = simulator.execute(order=buy_order, current_price=raw_price, ledger=ledger)
        assert result.success, f"BUY order execution failed: {result.error_message}"

        # Assert intercepted payload was dispatched
        assert len(interceptor.captured_events) == 1
        payload = interceptor.last_event

        # Verify exact fields required by Acceptance Criteria 2
        assert isinstance(payload, TradeExecutedPayload)
        assert payload.symbol == "BTC/EUR"
        assert payload.side == OrderSide.BUY
        assert payload.fill_price > 0.0
        assert payload.quantity > 0.0
        assert payload.fee >= 0.0
        assert payload.portfolio_balance >= 0.0
        assert payload.timestamp is not None
        assert payload.realized_pnl is not None or payload.realized_pnl == 0.0

        # Numeric verification
        # Expected slipped fill price: 60,000 * 1.0005 = 60,030.00
        assert payload.fill_price == pytest.approx(60030.00, rel=1e-3)
        assert payload.notional_value == pytest.approx(float(result.quote_notional), rel=1e-3)
        assert payload.fee == pytest.approx(float(result.fee), rel=1e-3)
        assert payload.portfolio_balance == pytest.approx(float(result.portfolio_balance), rel=1e-3)

        # 2. Trigger simulated closing SELL trade to verify realized PnL in payload
        sell_price = Decimal("65000.00")  # Profitable exit
        pos_qty = ledger.get_position("BTC/EUR").quantity
        sell_order = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.SELL,
            target_quantity=pos_qty,
            order_id="ord_ac2_sell_002",
        )
        sell_result = simulator.execute(order=sell_order, current_price=sell_price, ledger=ledger)
        assert sell_result.success

        assert len(interceptor.captured_events) == 2
        sell_payload = interceptor.last_event
        assert isinstance(sell_payload, TradeExecutedPayload)
        assert sell_payload.symbol == "BTC/EUR"
        assert sell_payload.side == OrderSide.SELL
        assert sell_payload.fill_price == pytest.approx(65000.0 * 0.9995, rel=1e-3)
        assert sell_payload.quantity == pytest.approx(float(pos_qty), rel=1e-4)
        assert sell_payload.fee > 0.0
        assert sell_payload.portfolio_balance > 10.0  # Profitable trade grew balance
        assert sell_payload.realized_pnl is not None
        assert sell_payload.realized_pnl > 0.0  # Positive profit realized


# =====================================================================
# TIER 1 FEATURE TESTS
# =====================================================================
def test_trade_executed_payload_model_validation():
    """Test TradeExecutedPayload field validation, default values, and Decimal coercion."""
    payload = TradeExecutedPayload(
        symbol="ETH/EUR",
        side=OrderSide.BUY,
        fill_price=Decimal("3000.00"),
        quantity=Decimal("0.0033"),
        fee=Decimal("0.0099"),
        portfolio_balance=Decimal("10.00"),
        cash_balance=Decimal("0.10"),
        realized_pnl=Decimal("0.00"),
    )

    assert payload.event_type == EventType.TRADE_EXECUTED
    assert payload.symbol == "ETH/EUR"
    assert payload.side == OrderSide.BUY
    assert payload.fill_price == 3000.0
    assert payload.quantity == 0.0033
    assert payload.notional_value == pytest.approx(3000.0 * 0.0033, rel=1e-4)
    assert payload.fee == 0.0099
    assert payload.portfolio_balance == 10.0
    assert payload.cash_balance == 0.10
    assert payload.fee_currency == "EUR"
    assert payload.event_id is not None
    assert payload.timestamp is not None


def test_portfolio_update_payload_compounding_math():
    """Test PortfolioUpdatePayload automatic computation of ROI and progress towards 10,000 EUR."""
    # Case 1: Initial state (10€ equity) -> 0% ROI, 0% progress
    p1 = PortfolioUpdatePayload(
        total_equity=10.0,
        cash_balance=10.0,
        target_equity=10000.0,
        starting_equity=10.0,
    )
    assert p1.roi_pct == pytest.approx(0.0)
    assert p1.progress_to_target_pct == pytest.approx(0.0)

    # Case 2: Scaled to 100€ equity -> +900% ROI, ~0.9% progress towards 10,000€
    p2 = PortfolioUpdatePayload(
        total_equity=100.0,
        cash_balance=50.0,
        unrealized_pnl=50.0,
        target_equity=10000.0,
        starting_equity=10.0,
    )
    assert p2.roi_pct == pytest.approx(900.0)
    # (100 - 10) / (10000 - 10) * 100 = 90 / 9990 * 100 = 0.9009%
    assert p2.progress_to_target_pct == pytest.approx(0.9009, rel=1e-3)

    # Case 3: Reached 10,000€ target -> 100% progress
    p3 = PortfolioUpdatePayload(
        total_equity=10000.0,
        cash_balance=10000.0,
        target_equity=10000.0,
        starting_equity=10.0,
    )
    assert p3.progress_to_target_pct == pytest.approx(100.0)


def test_risk_alert_payload_attributes():
    """Test RiskAlertPayload attributes, severity levels, and JSON serialization."""
    alert = RiskAlertPayload(
        severity="CRITICAL",
        message="Daily drawdown exceeded 8% limit",
        current_drawdown_pct=8.45,
        max_drawdown_limit_pct=8.0,
        action_taken="CIRCUIT_BREAKER_HALT",
    )

    assert alert.event_type == EventType.RISK_ALERT
    assert alert.severity == "CRITICAL"
    assert alert.current_drawdown_pct == 8.45
    assert alert.action_taken == "CIRCUIT_BREAKER_HALT"

    dump = alert.model_dump()
    assert dump["message"] == "Daily drawdown exceeded 8% limit"
    assert dump["severity"] == "CRITICAL"


def test_console_channel_dispatch(capsys):
    """Test ConsoleChannel formatting for BUY, SELL, and Portfolio updates."""
    # Use ANSI fallback mode to capture deterministic stdout lines
    channel = ConsoleChannel(enabled=True, use_rich=False)

    buy_payload = TradeExecutedPayload(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        fill_price=60000.0,
        quantity=0.00016,
        fee=0.0096,
        portfolio_balance=9.99,
    )
    status = channel.dispatch(buy_payload)
    assert status.success
    captured = capsys.readouterr().out
    assert "BTC/EUR" in captured
    assert "BUY" in captured
    assert "60,000.00" in captured

    sell_payload = TradeExecutedPayload(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        fill_price=65000.0,
        quantity=0.00016,
        fee=0.0104,
        portfolio_balance=10.78,
        realized_pnl=0.79,
    )
    status2 = channel.dispatch(sell_payload)
    assert status2.success
    captured2 = capsys.readouterr().out
    assert "SELL" in captured2
    assert "PnL: €+0.7900" in captured2


def test_json_log_channel_writes_and_reads(tmp_path: Path):
    """Test JsonLogChannel atomic JSONL writes, reading, and clearing."""
    log_file = tmp_path / "test_events.jsonl"
    channel = JsonLogChannel(log_path=log_file, enabled=True)

    event1 = TradeExecutedPayload(
        symbol="SOL/EUR",
        side=OrderSide.BUY,
        fill_price=140.0,
        quantity=0.07,
        fee=0.0098,
        portfolio_balance=9.99,
    )
    event2 = PortfolioUpdatePayload(
        total_equity=12.50,
        cash_balance=12.50,
        roi_pct=25.0,
    )

    status1 = channel.dispatch(event1)
    status2 = channel.dispatch(event2)
    assert status1.success
    assert status2.success

    assert log_file.exists()
    records = channel.read_events()
    assert len(records) == 2
    assert records[0]["symbol"] == "SOL/EUR"
    assert records[0]["side"] == "BUY"
    assert records[1]["total_equity"] == 12.50

    # Test clearing log
    channel.clear_log()
    assert len(channel.read_events()) == 0


def test_telegram_channel_fallback_when_disabled():
    """Test TelegramChannel automatic safe fallback when credentials are absent."""
    # Empty tokens
    channel_empty = TelegramChannel(bot_token="", chat_id="", enabled=True)
    assert not channel_empty.is_enabled()
    status = channel_empty.dispatch(
        TradeExecutedPayload(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            fill_price=50000.0,
            quantity=0.0002,
            portfolio_balance=10.0,
        )
    )
    assert status.success
    assert "disabled" in status.error_message.lower()

    # Placeholder / dummy token
    channel_dummy = TelegramChannel(
        bot_token="YOUR_TELEGRAM_BOT_TOKEN",
        chat_id="12345678",
        enabled=True,
    )
    assert not channel_dummy.is_enabled()
    status_dummy = channel_dummy.dispatch(
        TradeExecutedPayload(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            fill_price=50000.0,
            quantity=0.0002,
            portfolio_balance=10.0,
        )
    )
    assert status_dummy.success
    assert "disabled" in status_dummy.error_message.lower()


def test_telegram_channel_formatting():
    """Test TelegramChannel Markdown message construction."""
    channel = TelegramChannel(bot_token="12345:dummy", chat_id="999", enabled=True)

    event = TradeExecutedPayload(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        fill_price=58200.0,
        quantity=0.000165,
        fee=0.0096,
        portfolio_balance=9.99,
        realized_pnl=None,
    )
    msg = channel.format_message(event)
    assert "TRADE EXECUTED" in msg
    assert "BTC/EUR" in msg
    assert "BUY" in msg
    assert "58,200.00" in msg


def test_notification_service_fanout_and_management():
    """Test NotificationService channel registration, unregistration, and fan-out dispatch."""
    service = NotificationService()
    c1 = NotificationInterceptor(name="chan1")
    c2 = NotificationInterceptor(name="chan2")

    service.register_channel(c1)
    service.register_channel(c2)
    assert service.has_channel("chan1")
    assert service.has_channel("chan2")

    event = PortfolioUpdatePayload(
        total_equity=11.20,
        cash_balance=11.20,
        roi_pct=12.0,
    )
    results = service.dispatch(event)
    assert len(results) == 2
    assert results["chan1"].success
    assert results["chan2"].success

    assert len(c1.captured_events) == 1
    assert len(c2.captured_events) == 1
    assert c1.last_event.total_equity == 11.20

    # Unregister chan1
    service.unregister_channel("chan1")
    assert not service.has_channel("chan1")

    service.dispatch(event)
    assert len(c1.captured_events) == 1  # Unchanged
    assert len(c2.captured_events) == 2  # Received second event


# =====================================================================
# TIER 2 BOUNDARY & CORNER TESTS
# =====================================================================
def test_notification_service_channel_fault_isolation():
    """Test that a crashing channel does not crash the service or prevent delivery to other channels."""
    service = NotificationService()
    good_interceptor = NotificationInterceptor(name="good_interceptor")

    class CrashingChannel(BaseNotificationChannel):
        @property
        def name(self) -> str:
            return "crashing_channel"

        def is_enabled(self) -> bool:
            return True

        def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
            raise RuntimeError("Fatal network explosion!")

    service.register_channel(CrashingChannel())
    service.register_channel(good_interceptor)

    event = RiskAlertPayload(
        message="System warning test",
        severity="WARNING",
    )
    # Must NOT raise exception!
    results = service.dispatch(event)

    assert not results["crashing_channel"].success
    assert "Fatal network explosion!" in results["crashing_channel"].error_message
    assert results["good_interceptor"].success
    assert len(good_interceptor.captured_events) == 1


def test_invalid_payload_validation_errors():
    """Test that invalid numeric parameters in payloads raise ValidationError."""
    # fill_price must be > 0
    with pytest.raises(ValidationError):
        TradeExecutedPayload(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            fill_price=0.0,  # Invalid
            quantity=0.001,
            portfolio_balance=10.0,
        )

    # quantity must be > 0
    with pytest.raises(ValidationError):
        TradeExecutedPayload(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            fill_price=50000.0,
            quantity=-0.001,  # Invalid
            portfolio_balance=10.0,
        )

    # negative portfolio balance
    with pytest.raises(ValidationError):
        TradeExecutedPayload(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            fill_price=50000.0,
            quantity=0.001,
            portfolio_balance=-1.0,  # Invalid
        )


def test_telegram_api_http_error_resilience():
    """Test that TelegramChannel handles HTTP 500 without crashing and reports failure status."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_session.post.return_value = mock_response

    channel = TelegramChannel(
        bot_token="12345:valid_token_test",
        chat_id="987654",
        enabled=True,
        session=mock_session,
    )
    assert channel.is_enabled()

    event = TradeExecutedPayload(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        fill_price=60000.0,
        quantity=0.0001,
        portfolio_balance=10.0,
    )
    # Must not raise an exception
    status = channel.dispatch(event)
    assert not status.success
    assert "HTTP 500" in status.error_message

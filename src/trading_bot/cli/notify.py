"""Notification Channel Verification CLI Command.

Tests and verifies notification channel telemetry (Console, JSONL log, Telegram)
by constructing and dispatching synthetic event payloads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from trading_bot.notifications.channels.console import ConsoleChannel
from trading_bot.notifications.channels.json_log import JsonLogChannel
from trading_bot.notifications.channels.telegram import TelegramChannel
from trading_bot.notifications.models import (
    DeliveryStatus,
    EventType,
    OrderSide,
    OrderType,
    PortfolioUpdatePayload,
    RiskAlertPayload,
    TradeExecutedPayload,
)
from trading_bot.notifications.service import NotificationService

logger = logging.getLogger("notify_cli")

EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CLI_USAGE_ERROR = 2
EXIT_CONFIG_ERROR = 5


def dispatch_test_events(
    service: NotificationService,
    event_type: str = "all",
) -> Dict[str, Any]:
    """Dispatch synthetic test events and return delivery statuses."""
    results = {}
    now = datetime.now(timezone.utc)

    # 1. Trade Executed Event
    if event_type in ("trade", "all"):
        trade_payload = TradeExecutedPayload(
            order_id="test_ord_001",
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            fill_price=58500.00,
            quantity=0.000168,
            notional_value=9.828,
            fee=0.0098,
            fee_currency="EUR",
            portfolio_balance=10.00,
            cash_balance=0.17,
            position_qty_after=0.000168,
            realized_pnl=0.00,
            timestamp=now,
        )
        status_map = service.dispatch(trade_payload)
        results["trade_executed"] = {
            ch: status.value if hasattr(status, "value") else str(status)
            for ch, status in status_map.items()
        }

    # 2. Portfolio Update Event
    if event_type in ("portfolio", "all"):
        portfolio_payload = PortfolioUpdatePayload(
            total_equity=12.45,
            cash_balance=10.20,
            unrealized_pnl=0.35,
            realized_pnl=2.45,
            roi_pct=24.50,
            total_trades_count=14,
            open_positions_count=1,
            win_rate=64.29,
            progress_to_target_pct=0.0245,
            mode="PAPER",
            timestamp=now,
        )
        status_map = service.dispatch(portfolio_payload)
        results["portfolio_update"] = {
            ch: status.value if hasattr(status, "value") else str(status)
            for ch, status in status_map.items()
        }

    # 3. Risk Alert Event
    if event_type in ("risk", "all"):
        risk_payload = RiskAlertPayload(
            message="Simulated test risk alert: Daily drawdown threshold check passed.",
            severity="INFO",
            current_drawdown_pct=1.5,
            max_drawdown_limit_pct=8.0,
            action_taken="LOG_NOTICE",
            mode="PAPER",
            timestamp=now,
        )
        status_map = service.dispatch(risk_payload)
        results["risk_alert"] = {
            ch: status.value if hasattr(status, "value") else str(status)
            for ch, status in status_map.items()
        }

    return results


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for notification testing."""
    parser = argparse.ArgumentParser(
        prog="trading_bot.notify",
        description="Autonomous Compounding Trading Bot - Notification Telemetry Verification",
    )

    parser.add_argument(
        "--channel",
        choices=["console", "json", "telegram", "all"],
        default="console",
        help="Notification channel to test (default: console)",
    )
    parser.add_argument(
        "--test-event",
        "--event-type",
        dest="event_type",
        choices=["trade", "portfolio", "risk", "all"],
        default="all",
        help="Event type to dispatch (default: all)",
    )
    parser.add_argument(
        "--telegram-token",
        type=str,
        default=os.environ.get("TELEGRAM_BOT_TOKEN"),
        help="Telegram Bot API token (can also be read from TELEGRAM_BOT_TOKEN env)",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=str,
        default=os.environ.get("TELEGRAM_CHAT_ID"),
        help="Telegram Chat ID (can also be read from TELEGRAM_CHAT_ID env)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="logs/notifications.jsonl",
        help="Log file path for JSONL channel (default: logs/notifications.jsonl)",
    )
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="CLI output format (default: table)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return EXIT_CLI_USAGE_ERROR if exc.code != 0 else EXIT_SUCCESS

    if parsed.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Initialize channels
    channels = []
    ch_type = parsed.channel.lower()

    if ch_type in ("console", "all"):
        channels.append(ConsoleChannel(enabled=True))

    if ch_type in ("json", "all"):
        channels.append(JsonLogChannel(log_path=parsed.json_path, enabled=True))

    if ch_type in ("telegram", "all"):
        token = parsed.telegram_token
        chat_id = parsed.telegram_chat_id
        if token and chat_id:
            channels.append(TelegramChannel(bot_token=token, chat_id=chat_id, enabled=True))
        elif ch_type == "telegram":
            sys.stderr.write("Warning: Telegram channel requested but --telegram-token or --telegram-chat-id not provided. Using disabled fallback.\n")
            channels.append(TelegramChannel(bot_token="", chat_id="", enabled=False))

    service = NotificationService(channels=channels)

    try:
        results = dispatch_test_events(service=service, event_type=parsed.event_type)
    except Exception as exc:
        sys.stderr.write(f"Notification dispatch failed: {exc}\n")
        return EXIT_RUNTIME_ERROR
    finally:
        service.close()

    if parsed.output == "json":
        print(json.dumps({
            "status": "success",
            "channel_tested": parsed.channel,
            "registered_channels": list(service.channels.keys()),
            "dispatch_results": results,
        }, indent=2))
    else:
        print("=" * 70)
        print("              NOTIFICATION TELEMETRY TEST REPORT              ")
        print("=" * 70)
        print(f"Target Channel: {parsed.channel}")
        print(f"Registered Channels: {', '.join(service.channels.keys())}")
        print("-" * 70)
        for evt, status_map in results.items():
            print(f"Event: {evt}")
            for ch, st in status_map.items():
                print(f"  - [{ch}]: {st}")
        print("=" * 70)
        print("[SUCCESS] Test notifications dispatched successfully.")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())

"""Telegram Bot notification channel with graceful disabled fallback.

Formats trade executions, portfolio compounding updates, and risk alerts
into Markdown messages and sends them via the Telegram Bot API.
Gracefully deactivates when API credentials are absent or invalid.
"""

import logging
from typing import Any, Optional
from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.models import (
    BaseNotificationEvent,
    DeliveryStatus,
    OrderSide,
    PortfolioUpdatePayload,
    RiskAlertPayload,
    TradeExecutedPayload,
)

logger = logging.getLogger("TelegramChannel")


class TelegramChannel(BaseNotificationChannel):
    """Telegram Bot notification adapter with automated fallback."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_seconds: float = 4.0,
        enabled: bool = True,
        session: Optional[Any] = None,
    ):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (str(chat_id) if chat_id is not None else "").strip()
        self.timeout_seconds = timeout_seconds
        self._enabled = enabled
        self._session = session

    @property
    def name(self) -> str:
        return "telegram"

    def is_enabled(self) -> bool:
        """Return True only if explicitly enabled AND valid credentials are present."""
        if not self._enabled:
            return False
        if not self.bot_token or not self.chat_id:
            return False
        # Reject obvious placeholder or dummy tokens
        upper_tok = self.bot_token.upper()
        if upper_tok.startswith("YOUR_") or upper_tok.startswith("DUMMY") or upper_tok == "NONE":
            return False
        return True

    def enable(self) -> None:
        """Enable Telegram channel."""
        self._enabled = True

    def disable(self) -> None:
        """Disable Telegram channel."""
        self._enabled = False

    def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
        """Deliver notification to Telegram or return clean fallback if disabled."""
        if not self.is_enabled():
            logger.debug("Telegram channel disabled (credentials missing or disabled flag set). Skipping.")
            return DeliveryStatus(
                success=True,
                channel_name=self.name,
                error_message="Channel disabled: Missing credentials or inactive",
            )

        try:
            text = self.format_message(event)
            return self._send_telegram_message(text)
        except Exception as ex:
            logger.warning(f"Failed to dispatch Telegram message: {ex}", exc_info=True)
            return DeliveryStatus(
                success=False,
                channel_name=self.name,
                error_message=str(ex),
            )

    def _send_telegram_message(self, text: str) -> DeliveryStatus:
        """Execute HTTP POST request to Telegram Bot API."""
        import requests

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        client = self._session if self._session is not None else requests

        response = client.post(url, json=payload, timeout=self.timeout_seconds)
        if response.status_code == 200:
            return DeliveryStatus(success=True, channel_name=self.name)
        else:
            error_detail = f"Telegram API HTTP {response.status_code}: {response.text}"
            logger.warning(error_detail)
            return DeliveryStatus(
                success=False,
                channel_name=self.name,
                error_message=error_detail,
            )

    def format_message(self, event: BaseNotificationEvent) -> str:
        """Format an event into Markdown text for Telegram."""
        if isinstance(event, TradeExecutedPayload):
            is_buy = (event.side == OrderSide.BUY or str(event.side).upper() == "BUY")
            side_badge = "🟢 *BUY*" if is_buy else "🔴 *SELL*"
            pnl_line = ""
            if event.realized_pnl is not None:
                sign = "+" if event.realized_pnl >= 0 else ""
                pnl_line = f"\n*Realized PnL*: `€{sign}{event.realized_pnl:,.4f}`"

            return (
                f"🔔 *TRADE EXECUTED* [{event.mode}]\n"
                f"*Side*: {side_badge}\n"
                f"*Pair*: `{event.symbol}`\n"
                f"*Price*: `€{event.fill_price:,.2f}`\n"
                f"*Quantity*: `{event.quantity:.6f}` (`€{event.notional_value:.2f}`)\n"
                f"*Fee*: `€{event.fee:.4f} {event.fee_currency}`\n"
                f"*Portfolio Balance*: `€{event.portfolio_balance:,.2f}`"
                f"{pnl_line}"
            )

        elif isinstance(event, PortfolioUpdatePayload):
            net_profit = event.realized_pnl_cumulative + event.unrealized_pnl
            sign = "+" if net_profit >= 0 else ""
            return (
                f"📊 *PORTFOLIO UPDATE* [{event.mode}]\n"
                f"*Equity*: `€{event.total_equity:,.2f}`\n"
                f"*Cash*: `€{event.cash_balance:,.2f}`\n"
                f"*Guadagni Netti*: `{sign}€{net_profit:,.2f}`\n"
                f"*Total Trades*: `{event.total_trades_count}` | *Win Rate*: `{event.win_rate:.1f}%`"
            )

        elif isinstance(event, RiskAlertPayload):
            return (
                f"⚠️ *RISK ALERT* [{event.severity}]\n"
                f"{event.message}\n"
                f"*Drawdown*: `{event.current_drawdown_pct:.2f}%` (Limit: `{event.max_drawdown_limit_pct:.2f}%`)\n"
                f"*Action Taken*: `{event.action_taken}`"
            )

        return f"ℹ️ *NOTIFICATION* [{event.event_type.value}]\n`{event.model_dump_json()}`"

    def close(self) -> None:
        """Close session if applicable."""
        if self._session is not None and hasattr(self._session, "close"):
            try:
                self._session.close()
            except Exception:
                pass

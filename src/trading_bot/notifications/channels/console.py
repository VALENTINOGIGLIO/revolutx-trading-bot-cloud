"""Console notification channel with Rich UI formatting and ANSI fallback.

Displays color-coded badges (Green BUY, Red SELL), formatted trade fill summaries,
and portfolio compounding milestones in terminal outputs.
"""

import logging
import sys
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

logger = logging.getLogger("ConsoleChannel")


class ConsoleChannel(BaseNotificationChannel):
    """Rich terminal notification channel with color-coded trade telemetry."""

    def __init__(
        self,
        enabled: bool = True,
        rich_console: Optional[Any] = None,
        use_rich: bool = True,
    ):
        self._enabled = enabled
        self._use_rich = use_rich
        self._console = None

        if self._use_rich:
            try:
                from rich.console import Console
                self._console = rich_console if rich_console is not None else Console()
            except Exception as ex:
                logger.debug(f"Rich console unavailable, falling back to standard stdout: {ex}")
                self._use_rich = False
                self._console = None

    @property
    def name(self) -> str:
        return "console"

    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable console notifications."""
        self._enabled = True

    def disable(self) -> None:
        """Disable console notifications."""
        self._enabled = False

    def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
        """Render event to terminal output."""
        if not self._enabled:
            return DeliveryStatus(success=True, channel_name=self.name)

        try:
            if isinstance(event, TradeExecutedPayload):
                self._render_trade(event)
            elif isinstance(event, PortfolioUpdatePayload):
                self._render_portfolio(event)
            elif isinstance(event, RiskAlertPayload):
                self._render_risk(event)
            else:
                self._render_generic(event)

            return DeliveryStatus(success=True, channel_name=self.name)
        except Exception as ex:
            logger.error(f"Console delivery failed: {ex}", exc_info=True)
            return DeliveryStatus(
                success=False,
                channel_name=self.name,
                error_message=str(ex),
            )

    def _render_trade(self, event: TradeExecutedPayload) -> None:
        is_buy = (event.side == OrderSide.BUY or str(event.side).upper() == "BUY")
        pnl_str = ""
        if event.realized_pnl is not None:
            sign = "+" if event.realized_pnl >= 0 else ""
            pnl_str = f" | PnL: €{sign}{event.realized_pnl:,.4f}"

        if self._use_rich and self._console:
            side_tag = (
                "[bold white on green] BUY  [/bold white on green]"
                if is_buy
                else "[bold white on red] SELL [/bold white on red]"
            )
            pnl_rich = ""
            if event.realized_pnl is not None:
                pnl_color = "green" if event.realized_pnl >= 0 else "red"
                sign = "+" if event.realized_pnl >= 0 else ""
                pnl_rich = f" | PnL: [bold {pnl_color}]€{sign}{event.realized_pnl:,.4f}[/bold {pnl_color}]"

            msg = (
                f"{side_tag} [bold]{event.symbol}[/bold] @ €{event.fill_price:,.2f} "
                f"(Qty: {event.quantity:.6f}, Val: €{event.notional_value:.2f}) "
                f"| Fee: €{event.fee:.4f} "
                f"| Balance: [bold cyan]€{event.portfolio_balance:,.2f}[/bold cyan]{pnl_rich}"
            )
            self._console.print(msg)
        else:
            badge = "\033[1;32m[BUY ]\033[0m" if is_buy else "\033[1;31m[SELL]\033[0m"
            print(
                f"{badge} {event.symbol} @ €{event.fill_price:,.2f} "
                f"(Qty: {event.quantity:.6f}, Val: €{event.notional_value:.2f}) "
                f"| Fee: €{event.fee:.4f} "
                f"| Balance: €{event.portfolio_balance:,.2f}{pnl_str}",
                file=sys.stdout,
                flush=True,
            )

    def _render_portfolio(self, event: PortfolioUpdatePayload) -> None:
        net_profit = event.realized_pnl_cumulative + event.unrealized_pnl
        profit_color = "green" if net_profit >= 0 else "red"
        sign = "+" if net_profit >= 0 else ""
        if self._use_rich and self._console:
            msg = (
                f"[bold yellow]📊 PORTFOLIO UPDATE[/bold yellow] "
                f"Equity: [bold]€{event.total_equity:,.2f}[/bold] "
                f"| Cash: €{event.cash_balance:,.2f} "
                f"| Guadagni Netti: [bold {profit_color}]{sign}€{net_profit:,.2f}[/bold {profit_color}] "
                f"| Trades: {event.total_trades_count} "
                f"| Win Rate: {event.win_rate:.1f}%"
            )
            self._console.print(msg)
        else:
            print(
                f"[PORTFOLIO] Equity: €{event.total_equity:,.2f} "
                f"| Cash: €{event.cash_balance:,.2f} "
                f"| Guadagni Netti: {sign}€{net_profit:,.2f} "
                f"| Trades: {event.total_trades_count}",
                file=sys.stdout,
                flush=True,
            )

    def _render_risk(self, event: RiskAlertPayload) -> None:
        if self._use_rich and self._console:
            msg = (
                f"[bold white on red] ⚠️ RISK ALERT [{event.severity}] [/bold white on red] "
                f"{event.message} "
                f"| Drawdown: [bold]{event.current_drawdown_pct:.2f}%[/bold] (Limit: {event.max_drawdown_limit_pct:.2f}%) "
                f"| Action: [bold yellow]{event.action_taken}[/bold yellow]"
            )
            self._console.print(msg)
        else:
            print(
                f"[RISK ALERT - {event.severity}] {event.message} "
                f"| Drawdown: {event.current_drawdown_pct:.2f}% "
                f"| Action: {event.action_taken}",
                file=sys.stderr,
                flush=True,
            )

    def _render_generic(self, event: BaseNotificationEvent) -> None:
        if self._use_rich and self._console:
            self._console.print(f"[dim][{event.event_type.value}][/dim] {event.model_dump_json()}")
        else:
            print(f"[{event.event_type.value}] {event.model_dump_json()}", file=sys.stdout, flush=True)

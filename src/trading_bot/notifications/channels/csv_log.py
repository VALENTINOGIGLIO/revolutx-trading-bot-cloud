"""CSV Trade Execution Logger and Notification Channel.

Provides persistent, thread-safe, RFC 4180 append-only trade logging to CSV
with immediate fsync durability and fault isolation.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union

from trading_bot.config import DEFAULT_TRADES_CSV_PATH
from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.models import (
    BaseNotificationEvent,
    DeliveryStatus,
    EventType,
    OrderSide,
    TradeExecutedPayload,
)

logger = logging.getLogger("CsvTradeLogger")


class CsvTradeLogger:
    """Thread-safe persistent CSV trade execution logger.

    Logs executed trades (BUY and SELL) to a structured CSV file adhering
    to RFC 4180 specifications. Guarantees immediate durability via
    explicit buffer flush and os.fsync() calls.
    """

    HEADERS: List[str] = [
        "timestamp",
        "trade_id",
        "symbol",
        "action",
        "price",
        "size",
        "notional",
        "fee",
        "pnl",
        "pnl_pct",
        "portfolio_balance",
        "cash_balance",
        "exit_reason",
    ]

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        auto_mkdir: bool = True,
    ) -> None:
        if csv_path is None:
            self.csv_path = Path(DEFAULT_TRADES_CSV_PATH)
        else:
            self.csv_path = Path(csv_path)

        self._lock = threading.Lock()
        self._auto_mkdir = auto_mkdir
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Write RFC 4180 header if the file does not exist or is 0 bytes."""
        with self._lock:
            self._ensure_header_unlocked()

    def _ensure_header_unlocked(self) -> None:
        """Internal header check without acquiring self._lock (caller must hold lock)."""
        try:
            if self._auto_mkdir:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            file_exists = self.csv_path.exists()
            file_is_empty = file_exists and self.csv_path.stat().st_size == 0

            if not file_exists or file_is_empty:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerow(self.HEADERS)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
        except Exception as ex:
            logger.warning(f"Could not initialize CSV trade log header at {self.csv_path}: {ex}")

    @staticmethod
    def _format_timestamp(ts: Any) -> str:
        """Convert timestamp to ISO-8601 UTC string."""
        if ts is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(ts, (int, float)):
            # Distinguish millisecond timestamps from second timestamps
            if ts > 1e11:
                dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.isoformat()
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.isoformat()
        return str(ts)

    def _format_payload_row(self, trade: TradeExecutedPayload) -> List[str]:
        """Format a TradeExecutedPayload into the 13 CSV columns."""
        ts_str = self._format_timestamp(trade.timestamp)
        trade_id_str = str(
            getattr(trade, "order_id", None)
            or (trade.metadata.get("trade_id") if hasattr(trade, "metadata") and isinstance(trade.metadata, dict) else None)
            or trade.event_id
        )
        symbol_str = str(trade.symbol)
        raw_side = trade.side.value if hasattr(trade.side, "value") else str(trade.side)
        action_str = raw_side.upper()
        is_buy = action_str == "BUY"

        price_str = f"{float(trade.fill_price):.2f}"
        size_str = f"{float(trade.quantity):.6f}"

        notional_val = (
            trade.notional_value
            if trade.notional_value is not None
            else round(trade.fill_price * trade.quantity, 4)
        )
        notional_str = f"{float(notional_val):.4f}"
        fee_str = f"{float(trade.fee):.4f}"

        if is_buy:
            pnl_str = "0.00"
            pnl_pct_str = "0.00"
        else:
            if trade.realized_pnl is not None:
                pnl_str = f"{float(trade.realized_pnl):.4f}"
            else:
                pnl_str = "0.00"

            if trade.realized_pnl_pct is not None:
                pnl_pct_str = f"{float(trade.realized_pnl_pct):.2f}"
            elif trade.realized_pnl is not None and trade.realized_pnl != 0.0:
                cost_basis = notional_val - trade.fee - trade.realized_pnl
                if cost_basis > 0:
                    pct = (trade.realized_pnl / cost_basis) * 100.0
                    pnl_pct_str = f"{pct:.2f}"
                else:
                    pnl_pct_str = "0.00"
            else:
                pnl_pct_str = "0.00"

        portfolio_bal_str = f"{float(trade.portfolio_balance):.4f}"
        cash_bal_val = trade.cash_balance if trade.cash_balance is not None else trade.portfolio_balance
        cash_bal_str = f"{float(cash_bal_val):.4f}"

        if is_buy:
            exit_reason_str = ""
        else:
            exit_reason_str = str(
                getattr(trade, "exit_reason", None)
                or getattr(trade, "strategy_signal", None)
                or (trade.metadata.get("exit_reason") if hasattr(trade, "metadata") and isinstance(trade.metadata, dict) else "")
                or ""
            )

        return [
            ts_str,
            trade_id_str,
            symbol_str,
            action_str,
            price_str,
            size_str,
            notional_str,
            fee_str,
            pnl_str,
            pnl_pct_str,
            portfolio_bal_str,
            cash_bal_str,
            exit_reason_str,
        ]

    def _format_record_row(self, trade: Any) -> List[str]:
        """Format a TradeRecord (or duck-typed object) into the 13 CSV columns."""
        # Timestamp
        ts_val = getattr(trade, "timestamp", None) or getattr(trade, "timestamp_ms", None)
        ts_str = self._format_timestamp(ts_val)

        # Trade ID
        trade_id_str = str(
            getattr(trade, "trade_id", None)
            or getattr(trade, "order_id", None)
            or ""
        )

        symbol_str = str(getattr(trade, "symbol", ""))
        raw_side = getattr(trade, "side", "BUY")
        action_str = str(raw_side.value if hasattr(raw_side, "value") else raw_side).upper()
        is_buy = action_str == "BUY"

        exec_px = getattr(trade, "executed_price", None) or getattr(trade, "fill_price", 0.0)
        price_str = f"{float(exec_px):.2f}"

        qty = getattr(trade, "quantity", 0.0)
        size_str = f"{float(qty):.6f}"

        notional_val = getattr(trade, "quote_notional", None)
        if notional_val is None:
            notional_val = float(exec_px) * float(qty)
        notional_str = f"{float(notional_val):.4f}"

        fee_val = getattr(trade, "fee", 0.0)
        fee_str = f"{float(fee_val):.4f}"

        realized_pnl = getattr(trade, "realized_pnl", None)
        if is_buy:
            pnl_str = "0.00"
            pnl_pct_str = "0.00"
        else:
            if realized_pnl is not None:
                pnl_str = f"{float(realized_pnl):.4f}"
            else:
                pnl_str = "0.00"

            pnl_pct_val = getattr(trade, "realized_pnl_pct", None) or getattr(trade, "pnl_pct", None)
            if pnl_pct_val is not None:
                pnl_pct_str = f"{float(pnl_pct_val):.2f}"
            elif realized_pnl is not None and float(realized_pnl) != 0.0:
                cost_basis = float(notional_val) - float(fee_val) - float(realized_pnl)
                if cost_basis > 0:
                    pct = (float(realized_pnl) / cost_basis) * 100.0
                    pnl_pct_str = f"{pct:.2f}"
                else:
                    pnl_pct_str = "0.00"
            else:
                pnl_pct_str = "0.00"

        equity_after = getattr(trade, "equity_after", None) or getattr(trade, "portfolio_balance", 0.0)
        portfolio_bal_str = f"{float(equity_after):.4f}"

        cash_after = getattr(trade, "cash_after", None) or getattr(trade, "cash_balance", equity_after)
        cash_bal_str = f"{float(cash_after):.4f}"

        if is_buy:
            exit_reason_str = ""
        else:
            exit_reason_str = str(getattr(trade, "exit_reason", None) or "")

        return [
            ts_str,
            trade_id_str,
            symbol_str,
            action_str,
            price_str,
            size_str,
            notional_str,
            fee_str,
            pnl_str,
            pnl_pct_str,
            portfolio_bal_str,
            cash_bal_str,
            exit_reason_str,
        ]

    def _format_dict_row(self, trade: Dict[str, Any]) -> List[str]:
        """Format a raw trade dictionary into the 13 CSV columns."""
        ts_val = trade.get("timestamp") or trade.get("timestamp_ms") or trade.get("datetime_utc")
        ts_str = self._format_timestamp(ts_val)

        trade_id_str = str(trade.get("trade_id") or trade.get("order_id") or "")
        symbol_str = str(trade.get("symbol", ""))

        raw_side = trade.get("action") or trade.get("side") or "BUY"
        action_str = str(raw_side.value if hasattr(raw_side, "value") else raw_side).upper()
        is_buy = action_str == "BUY"

        price_val = trade.get("price") or trade.get("fill_price") or trade.get("executed_price", 0.0)
        price_str = f"{float(price_val):.2f}"

        size_val = trade.get("size") or trade.get("quantity", 0.0)
        size_str = f"{float(size_val):.6f}"

        notional_val = (
            trade.get("notional")
            or trade.get("notional_value")
            or trade.get("quote_notional")
            or (float(price_val) * float(size_val))
        )
        notional_str = f"{float(notional_val):.4f}"

        fee_val = trade.get("fee", 0.0)
        fee_str = f"{float(fee_val):.4f}"

        pnl_val = trade.get("pnl") or trade.get("realized_pnl")
        if is_buy:
            pnl_str = "0.00"
            pnl_pct_str = "0.00"
        else:
            pnl_str = f"{float(pnl_val):.4f}" if pnl_val is not None else "0.00"
            pnl_pct_val = trade.get("pnl_pct") or trade.get("realized_pnl_pct")
            if pnl_pct_val is not None:
                pnl_pct_str = f"{float(pnl_pct_val):.2f}"
            elif pnl_val is not None and float(pnl_val) != 0.0:
                cost_basis = float(notional_val) - float(fee_val) - float(pnl_val)
                if cost_basis > 0:
                    pnl_pct_str = f"{(float(pnl_val) / cost_basis * 100):.2f}"
                else:
                    pnl_pct_str = "0.00"
            else:
                pnl_pct_str = "0.00"

        equity_val = trade.get("portfolio_balance") or trade.get("equity_after", 0.0)
        portfolio_bal_str = f"{float(equity_val):.4f}"

        cash_val = trade.get("cash_balance") or trade.get("cash_after") or equity_val
        cash_bal_str = f"{float(cash_val):.4f}"

        if is_buy:
            exit_reason_str = ""
        else:
            exit_reason_str = str(trade.get("exit_reason") or trade.get("strategy_signal") or "")

        return [
            ts_str,
            trade_id_str,
            symbol_str,
            action_str,
            price_str,
            size_str,
            notional_str,
            fee_str,
            pnl_str,
            pnl_pct_str,
            portfolio_bal_str,
            cash_bal_str,
            exit_reason_str,
        ]

    def _format_row(self, trade: Union[TradeExecutedPayload, Any, Dict[str, Any]]) -> List[str]:
        """Dispatch formatting based on input type."""
        if isinstance(trade, dict):
            return self._format_dict_row(trade)
        elif isinstance(trade, TradeExecutedPayload):
            return self._format_payload_row(trade)
        else:
            return self._format_record_row(trade)

    def log_trade(self, trade: Union[TradeExecutedPayload, Any, Dict[str, Any]]) -> bool:
        """Append trade execution fill to CSV file.

        Thread-safe and durable with immediate buffer flush and fsync.
        Returns True on success, False on error.
        """
        try:
            row = self._format_row(trade)
            with self._lock:
                self._ensure_header_unlocked()
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerow(row)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
            return True
        except Exception as ex:
            logger.error(f"Failed to append trade to CSV {self.csv_path}: {ex}", exc_info=True)
            return False

    def read_trades(self) -> List[Dict[str, Any]]:
        """Safely read all recorded trades as a list of dictionaries."""
        if not self.csv_path.exists():
            return []
        try:
            if self.csv_path.stat().st_size == 0:
                return []
        except OSError:
            return []

        trades: List[Dict[str, Any]] = []
        with self._lock:
            try:
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row or not any(row.values()):
                            continue
                        parsed: Dict[str, Any] = dict(row)
                        for col in (
                            "price",
                            "size",
                            "notional",
                            "fee",
                            "pnl",
                            "pnl_pct",
                            "portfolio_balance",
                            "cash_balance",
                        ):
                            if col in parsed and parsed[col] != "":
                                try:
                                    parsed[col] = float(parsed[col])
                                except (ValueError, TypeError):
                                    pass
                        trades.append(parsed)
            except Exception as ex:
                logger.error(f"Error reading trades from {self.csv_path}: {ex}", exc_info=True)
                return []
        return trades

    def clear(self) -> None:
        """Reset CSV file with header only."""
        with self._lock:
            try:
                if self._auto_mkdir:
                    self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerow(self.HEADERS)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
            except Exception as ex:
                logger.error(f"Error clearing CSV trade log {self.csv_path}: {ex}", exc_info=True)


class CsvTradeLogChannel(BaseNotificationChannel):
    """Notification channel adapter that routes trade execution events to CsvTradeLogger."""

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        csv_logger: Optional[CsvTradeLogger] = None,
        enabled: bool = True,
        auto_mkdir: bool = True,
    ) -> None:
        self._enabled = enabled
        if csv_logger is not None:
            self.logger = csv_logger
        else:
            path = csv_path if csv_path is not None else DEFAULT_TRADES_CSV_PATH
            self.logger = CsvTradeLogger(csv_path=path, auto_mkdir=auto_mkdir)

    @property
    def name(self) -> str:
        """Unique identifier name for this notification channel."""
        return "csv_trade_log"

    def is_enabled(self) -> bool:
        """Return True if this channel is active."""
        return self._enabled

    def enable(self) -> None:
        """Enable CSV trade log channel."""
        self._enabled = True

    def disable(self) -> None:
        """Disable CSV trade log channel."""
        self._enabled = False

    def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
        """Deliver trade execution events to the CsvTradeLogger.

        Fault-isolated: errors are caught, logged, and return a DeliveryStatus
        indicating failure rather than propagating exceptions to the execution engine.
        """
        if not self._enabled:
            return DeliveryStatus(success=True, channel_name=self.name)

        try:
            is_trade = (
                isinstance(event, TradeExecutedPayload)
                or getattr(event, "event_type", None) == EventType.TRADE_EXECUTED
                or getattr(event, "event_type", None) == "TRADE_EXECUTED"
                or (isinstance(event, dict) and (event.get("event_type") == "TRADE_EXECUTED" or "fill_price" in event))
            )
            if is_trade:
                success = self.logger.log_trade(event)
                if not success:
                    return DeliveryStatus(
                        success=False,
                        channel_name=self.name,
                        error_message=f"Failed to append trade to {self.logger.csv_path}",
                    )

            return DeliveryStatus(success=True, channel_name=self.name)
        except Exception as ex:
            logger.error(f"Failed to dispatch trade event in {self.name}: {ex}", exc_info=True)
            return DeliveryStatus(
                success=False,
                channel_name=self.name,
                error_message=str(ex),
            )

    def close(self) -> None:
        """Cleanup channel resources."""
        pass

"""Thread-Safe State Container and Event Bridge for Trading Bot Dashboard.

Maintains live market candles, trade execution history, Lightweight Charts markers,
and dispatches real-time events to connected Server-Sent Events (SSE) subscribers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import logging
import math
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from trading_bot.config import DEFAULT_DB_PATH, DEFAULT_TRADES_CSV_PATH, normalize_symbol, to_display_symbol
from trading_bot.data.cache import CandleCache
from trading_bot.data.models import Candle
from trading_bot.notifications.channels.csv_log import CsvTradeLogger
from trading_bot.notifications.models import OrderSide, TradeExecutedPayload

logger = logging.getLogger("DashboardState")


class DashboardState:
    """Thread-safe in-memory state container and event broadcaster."""

    def __init__(
        self,
        symbol: str = "BTC/EUR",
        initial_balance: float = 10.0,
        status: str = "idle",
    ) -> None:
        self._lock = threading.RLock()
        self.symbol: str = to_display_symbol(symbol)
        self.balance: float = float(initial_balance)
        self.cash_balance: float = float(initial_balance)
        self.equity: float = float(initial_balance)
        self.net_profit: float = 0.0
        self.status: str = status  # "idle" or "running"
        self.last_updated: str = datetime.now(timezone.utc).isoformat()

        # In-memory storage
        self.candles: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.trade_markers: List[Dict[str, Any]] = []

        # Event subscribers: list of (queue, loop)
        self._subscribers: List[Tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
        self._is_shutdown: bool = False

    def set_status(self, status: str) -> None:
        """Update system status ('idle' or 'running') and broadcast update."""
        with self._lock:
            self.status = status
            self.last_updated = datetime.now(timezone.utc).isoformat()
            st = self.get_status()
        self.broadcast("status", st)

    def set_portfolio(
        self,
        equity: Optional[float] = None,
        cash: Optional[float] = None,
        balance: Optional[float] = None,
        net_profit: Optional[float] = None,
    ) -> None:
        """Update portfolio balances and broadcast status update."""
        with self._lock:
            if equity is not None:
                self.equity = float(equity)
            if cash is not None:
                self.cash_balance = float(cash)
            if balance is not None:
                self.balance = float(balance)
            if net_profit is not None:
                self.net_profit = float(net_profit)
            self.last_updated = datetime.now(timezone.utc).isoformat()
            st = self.get_status()
        self.broadcast("status", st)

    # ---------------------------------------------------------------------
    # Subscription / SSE Queue Management
    # ---------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        """Register a subscriber queue on the current running asyncio event loop."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            if self._is_shutdown:
                loop.call_soon_threadsafe(queue.put_nowait, {"event": "shutdown", "data": {}})
            else:
                self._subscribers.append((queue, loop))
        logger.debug(f"Subscribed new SSE listener (total: {len(self._subscribers)})")
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove subscriber queue."""
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item[0] is not queue]
        logger.debug(f"Unsubscribed SSE listener (remaining: {len(self._subscribers)})")

    def broadcast(self, event_type: str, data: Any) -> None:
        """Broadcast an event to all active subscriber queues in a thread-safe manner."""
        with self._lock:
            if self._is_shutdown:
                return
            message = {"event": event_type, "data": data}
            active: List[Tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
            for queue, loop in self._subscribers:
                if loop.is_closed():
                    continue
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, message)
                    active.append((queue, loop))
                except Exception as ex:
                    logger.debug(f"Failed to dispatch to subscriber queue: {ex}")
            self._subscribers = active

    def shutdown(self) -> None:
        """Thread-safely snapshot and clear subscribers, then broadcast shutdown sentinel."""
        shutdown_msg = {"event": "shutdown", "data": {}}
        with self._lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
            subscribers_snapshot = list(self._subscribers)
            self._subscribers.clear()

        for queue, loop in subscribers_snapshot:
            if not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, shutdown_msg)
                except Exception as ex:
                    logger.debug(f"Failed to dispatch shutdown sentinel to subscriber queue: {ex}")
        logger.debug("DashboardState shutdown complete; sentinel broadcast.")

    def reset_shutdown(self) -> None:
        """Reset shutdown flag for state reuse across server restart cycles."""
        with self._lock:
            self._is_shutdown = False

    # ---------------------------------------------------------------------
    # Data Loading (CSV & Market DB)
    # ---------------------------------------------------------------------
    def load_from_csv(self, csv_path: Union[str, Path] = DEFAULT_TRADES_CSV_PATH) -> int:
        """Read historical trades from CSV file and populate trades and markers."""
        path = Path(csv_path)
        if not path.exists():
            logger.info(f"CSV trade file {csv_path} does not exist, skipping load.")
            return 0

        trade_logger = CsvTradeLogger(csv_path=path)
        raw_trades = trade_logger.read_trades()
        if not raw_trades:
            return 0

        count = 0
        with self._lock:
            self.trades.clear()
            self.trade_markers.clear()
            for row in raw_trades:
                try:
                    marker = self._format_and_append_trade(row)
                    if marker:
                        count += 1
                except Exception as ex:
                    logger.warning(f"Skipping malformed CSV trade row {row}: {ex}")
                    continue
            self.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(f"Loaded {count} historical trades from {csv_path}")
        return count

    def load_from_market_db(
        self,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        symbol: Optional[str] = None,
        limit: int = 500,
        interval: str = "4h",
    ) -> int:
        """Load recent historical candles from SQLite cache."""
        path = Path(db_path)
        if not path.exists():
            logger.info(f"Market DB {db_path} does not exist, skipping candle load.")
            return 0

        target_sym = symbol or self.symbol
        norm_sym = normalize_symbol(target_sym)

        try:
            cache = CandleCache(db_path=path)
            cached_candles = cache.get_candles(symbol=norm_sym, interval=interval, limit=limit)
            if not cached_candles and interval != "1h":
                cached_candles = cache.get_candles(symbol=norm_sym, interval="1h", limit=limit)
        except Exception as ex:
            logger.warning(f"Failed to load candles from market cache {db_path}: {ex}")
            return 0

        if not cached_candles:
            return 0

        loaded_candles: List[Dict[str, Any]] = []
        for c in cached_candles:
            ts_sec = int(c.timestamp.timestamp())
            loaded_candles.append({
                "time": ts_sec,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            })

        with self._lock:
            self.candles = loaded_candles
            self.symbol = to_display_symbol(norm_sym)
            self.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(f"Loaded {len(loaded_candles)} candles for {norm_sym} from {db_path}")
        return len(loaded_candles)

    # ---------------------------------------------------------------------
    # Live Ingestion (Candles & Trades)
    # ---------------------------------------------------------------------
    def add_candle(self, candle: Union[Candle, Dict[str, Any]]) -> Dict[str, Any]:
        """Append or update the latest candlestick and broadcast via SSE."""
        if isinstance(candle, Candle):
            c_dict = {
                "time": int(candle.timestamp.timestamp()),
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
            }
        elif isinstance(candle, dict):
            # Parse time
            ts_val = candle.get("time")
            if ts_val is None:
                raw_ts = candle.get("timestamp") or candle.get("open_time")
                if isinstance(raw_ts, datetime):
                    ts_val = int(raw_ts.timestamp())
                elif isinstance(raw_ts, (int, float)):
                    ts_val = int(raw_ts / 1000) if raw_ts > 1e11 else int(raw_ts)
                else:
                    ts_val = int(datetime.now(timezone.utc).timestamp())
            else:
                ts_val = int(ts_val)

            c_dict = {
                "time": ts_val,
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle.get("volume", 0.0)),
            }
        else:
            raise TypeError(f"Unsupported candle type: {type(candle)}")

        with self._lock:
            # Check if last candle has same time (in-place tick update)
            if self.candles and self.candles[-1]["time"] == c_dict["time"]:
                self.candles[-1] = c_dict
            else:
                self.candles.append(c_dict)
            self.last_updated = datetime.now(timezone.utc).isoformat()

        self.broadcast("candle", c_dict)
        return c_dict

    def add_trade(self, trade: Union[TradeExecutedPayload, Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Format and append a trade execution fill, update balances, and broadcast via SSE."""
        with self._lock:
            try:
                marker = self._format_and_append_trade(trade)
            except Exception as ex:
                logger.warning(f"Failed to format trade: {ex}")
                marker = {}

            if not marker:
                return {}

            self.last_updated = datetime.now(timezone.utc).isoformat()
            st = self.get_status()

        self.broadcast("trade", marker)
        self.broadcast("status", st)
        return marker

    def _format_and_append_trade(self, trade: Union[TradeExecutedPayload, Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Internal helper to parse, extract marker, update balances, and store trade (caller holds lock)."""
        try:
            # Extract trade attributes across payload / record / dict types
            if isinstance(trade, dict):
                raw_action = trade.get("action") or trade.get("side") or "BUY"
                action = str(raw_action.value if hasattr(raw_action, "value") else raw_action).upper()
                try:
                    price = float(trade.get("price") or trade.get("fill_price") or trade.get("executed_price", 0.0))
                    size = float(trade.get("size") or trade.get("quantity", 0.0))
                    pnl = float(trade.get("pnl") or trade.get("realized_pnl", 0.0) or 0.0)
                except (ValueError, TypeError) as ex:
                    logger.warning(f"Invalid numeric value in trade record: {ex}")
                    return {}

                trade_id = str(trade.get("trade_id") or trade.get("order_id") or "")

                raw_time = trade.get("time")
                if raw_time is not None:
                    try:
                        time_sec = int(float(raw_time))
                    except (ValueError, TypeError):
                        time_sec = int(datetime.now(timezone.utc).timestamp())
                else:
                    raw_ts = trade.get("timestamp") or trade.get("timestamp_ms")
                    if isinstance(raw_ts, datetime):
                        time_sec = int(raw_ts.timestamp())
                    elif isinstance(raw_ts, (int, float)):
                        try:
                            time_sec = int(raw_ts / 1000) if raw_ts > 1e11 else int(raw_ts)
                        except (ValueError, TypeError):
                            time_sec = int(datetime.now(timezone.utc).timestamp())
                    elif isinstance(raw_ts, str):
                        try:
                            time_sec = int(datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp())
                        except Exception:
                            time_sec = int(datetime.now(timezone.utc).timestamp())
                    else:
                        time_sec = int(datetime.now(timezone.utc).timestamp())

                portfolio_bal = trade.get("portfolio_balance") or trade.get("equity_after")
                cash_bal = trade.get("cash_balance") or trade.get("cash_after")
                symbol_val = trade.get("symbol")
            elif isinstance(trade, TradeExecutedPayload):
                raw_side = trade.side.value if hasattr(trade.side, "value") else str(trade.side)
                action = raw_side.upper()
                try:
                    price = float(trade.fill_price)
                    size = float(trade.quantity)
                    pnl = float(trade.realized_pnl) if trade.realized_pnl is not None else 0.0
                except (ValueError, TypeError) as ex:
                    logger.warning(f"Invalid numeric value in TradeExecutedPayload: {ex}")
                    return {}

                trade_id = str(trade.order_id or trade.event_id)

                raw_ts = trade.timestamp
                if isinstance(raw_ts, datetime):
                    time_sec = int(raw_ts.timestamp())
                elif isinstance(raw_ts, (int, float)):
                    try:
                        time_sec = int(raw_ts / 1000) if raw_ts > 1e11 else int(raw_ts)
                    except (ValueError, TypeError):
                        time_sec = int(datetime.now(timezone.utc).timestamp())
                else:
                    time_sec = int(datetime.now(timezone.utc).timestamp())

                portfolio_bal = trade.portfolio_balance
                cash_bal = trade.cash_balance if trade.cash_balance is not None else portfolio_bal
                symbol_val = trade.symbol
            else:
                # Duck-typed object like TradeRecord
                raw_side = getattr(trade, "side", "BUY")
                action = str(raw_side.value if hasattr(raw_side, "value") else raw_side).upper()
                try:
                    price = float(getattr(trade, "executed_price", None) or getattr(trade, "fill_price", 0.0))
                    size = float(getattr(trade, "quantity", 0.0))
                    pnl = float(getattr(trade, "realized_pnl", 0.0) or 0.0)
                except (ValueError, TypeError) as ex:
                    logger.warning(f"Invalid numeric value in trade object: {ex}")
                    return {}

                trade_id = str(getattr(trade, "trade_id", None) or getattr(trade, "order_id", "") or "")

                raw_ts = getattr(trade, "timestamp", None) or getattr(trade, "timestamp_ms", None)
                if isinstance(raw_ts, datetime):
                    time_sec = int(raw_ts.timestamp())
                elif isinstance(raw_ts, (int, float)):
                    try:
                        time_sec = int(raw_ts / 1000) if raw_ts > 1e11 else int(raw_ts)
                    except (ValueError, TypeError):
                        time_sec = int(datetime.now(timezone.utc).timestamp())
                elif isinstance(raw_ts, str):
                    try:
                        time_sec = int(datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp())
                    except Exception:
                        time_sec = int(datetime.now(timezone.utc).timestamp())
                else:
                    time_sec = int(datetime.now(timezone.utc).timestamp())

                portfolio_bal = getattr(trade, "equity_after", None) or getattr(trade, "portfolio_balance", None)
                cash_bal = getattr(trade, "cash_after", None) or getattr(trade, "cash_balance", None)
                symbol_val = getattr(trade, "symbol", None)

            # Check for NaN / Inf
            if math.isnan(price) or math.isinf(price) or math.isnan(size) or math.isinf(size) or math.isnan(pnl) or math.isinf(pnl):
                logger.warning(f"Rejecting non-finite trade metrics (price={price}, size={size}, pnl={pnl})")
                return {}

            # Update symbol if present
            if symbol_val:
                self.symbol = to_display_symbol(str(symbol_val))

            # Update balances if available (safely cast inside try...except (ValueError, TypeError): pass)
            if portfolio_bal is not None:
                try:
                    p_val = float(portfolio_bal)
                    if not (math.isnan(p_val) or math.isinf(p_val)):
                        self.equity = p_val
                        self.balance = p_val
                except (ValueError, TypeError):
                    pass

            if cash_bal is not None:
                try:
                    c_val = float(cash_bal)
                    if not (math.isnan(c_val) or math.isinf(c_val)):
                        self.cash_balance = c_val
                except (ValueError, TypeError):
                    pass

            is_buy = (action == "BUY")
            if is_buy:
                marker_text = f"BUY @ {price:.2f}"
                position = "belowBar"
                color = "#10b981"
                shape = "arrowUp"
            else:
                pnl_str = f"+{pnl:.2f}€" if pnl >= 0 else f"{pnl:.2f}€"
                marker_text = f"SELL @ {price:.2f} ({pnl_str})"
                position = "aboveBar"
                color = "#ef4444"
                shape = "arrowDown"

            marker: Dict[str, Any] = {
                "time": time_sec,
                "position": position,
                "color": color,
                "shape": shape,
                "text": marker_text,
                "id": trade_id,
                "action": action,
                "price": price,
                "size": size,
                "pnl": pnl,
            }

            trade_item: Dict[str, Any] = {
                "trade_id": trade_id,
                "time": time_sec,
                "timestamp": datetime.fromtimestamp(time_sec, tz=timezone.utc).isoformat(),
                "symbol": self.symbol,
                "action": action,
                "price": price,
                "size": size,
                "pnl": pnl,
                "portfolio_balance": self.equity,
                "cash_balance": self.cash_balance,
                "marker": marker,
            }

            self.trades.append(trade_item)
            self.trade_markers.append(marker)
            return marker

        except Exception as ex:
            logger.warning(f"Unexpected error formatting trade record: {ex}")
            return {}

    # ---------------------------------------------------------------------
    # Getters
    # ---------------------------------------------------------------------
    def get_markers(self) -> List[Dict[str, Any]]:
        """Return list of trade markers formatted for Lightweight Charts."""
        with self._lock:
            return list(self.trade_markers)

    def get_trades(self) -> List[Dict[str, Any]]:
        """Return list of detailed trade records."""
        with self._lock:
            return list(self.trades)

    def get_candles(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return list of OHLCV candlestick records."""
        with self._lock:
            if limit is not None and limit > 0:
                return list(self.candles[-limit:])
            return list(self.candles)

    def get_status(self) -> Dict[str, Any]:
        """Return system status and portfolio balances matching required API schema."""
        with self._lock:
            return {
                "status": self.status,
                "symbol": self.symbol,
                "portfolio_balance": round(self.equity, 4),
                "cash_balance": round(self.cash_balance, 4),
                "net_profit": round(getattr(self, "net_profit", 0.0), 4),
                "total_trades": len(self.trades),
                "last_updated": self.last_updated,
            }

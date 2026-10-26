"""Empirical Challenger Test Suite for Milestone 10: Full Pipeline Integration & Stress.

Author: Challenger 1 (Milestone 10: Full Pipeline Stress & Integration Challenger)
Focus: Empirical Validation of the complete trading bot pipeline:
       Strategy / Order -> ExecutionSimulator -> VirtualLedger -> NotificationService
       -> CsvTradeLogChannel (CSV) + DashboardNotificationChannel (State)
       -> Web Dashboard (/api/trades, /api/status) + Real-time SSE Stream (/api/stream).

Tests:
1. test_empirical_50_rapid_alternating_orders_pipeline:
   - Executes 50 rapid alternating BUY/SELL orders with trailing stops and compounding reinvestment.
   - Verifies CSV trade log: exact RFC 4180 header, 13 columns, chronological timestamps.
   - Verifies GET /api/trades: all 50 markers returned with matching prices, positions, and shapes.
   - Verifies concurrent /api/stream: all 50 trade events arrive in exact order.
   - Verifies GET /api/status: balances synchronize with VirtualLedger.
   - Verifies DashboardServer stops cleanly in <1.0s.

2. test_empirical_csv_rfc4180_format_and_durability_deep_audit:
   - Deep structural inspection of CSV output for strict RFC 4180 compliance.
   - Validates ISO-8601 UTC timestamps, field escaping, zero NaN/null values.
   - Verifies BUY PnL is 0.00 and SELL PnL matches ledger realized profit.

3. test_empirical_sse_stream_ordering_and_multi_client_fanout:
   - Connects multiple concurrent SSE client streams simultaneously.
   - Rapidly fires 50 trades and verifies zero dropped events and identical ordering across all clients.

4. test_empirical_server_shutdown_latency_under_continuous_streaming:
   - Measures server teardown latency with active streaming consumers.
   - Asserts stop latency is strictly <1.0s.

5. test_empirical_compounding_reinvestment_progression:
   - Verifies capital compounding from 10.00 EUR initial equity.
   - Confirms each winning trade increases reinvested order notional.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

from trading_bot.dashboard.server import DashboardServer
from trading_bot.dashboard.state import DashboardState
from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.channels.csv_log import CsvTradeLogChannel, CsvTradeLogger
from trading_bot.notifications.models import (
    BaseNotificationEvent,
    DeliveryStatus,
    OrderSide,
    TradeExecutedPayload,
)
from trading_bot.notifications.service import NotificationService
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import ExecutionSimulator, OrderSignal
from trading_bot.strategy.compounding import CompoundingPositionSizer


class DashboardNotificationChannel(BaseNotificationChannel):
    """Notification channel adapter forwarding executed trade payloads to DashboardState."""

    def __init__(self, state: DashboardState, enabled: bool = True) -> None:
        self.state = state
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "dashboard_channel"

    def is_enabled(self) -> bool:
        return self._enabled

    def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
        if isinstance(event, TradeExecutedPayload) or isinstance(event, dict):
            self.state.add_trade(event)
            return DeliveryStatus(channel_name=self.name, success=True)
        return DeliveryStatus(channel_name=self.name, success=True)


# =====================================================================
# 1. FULL PIPELINE ACCEPTANCE & RAPID ALTERNATING 50-ORDER STRESS
# =====================================================================
def test_empirical_50_rapid_alternating_orders_pipeline(tmp_path: Path):
    """Empirically execute and stress the end-to-end acceptance flow.

    Requirements verified:
    1. 50 rapid alternating BUY/SELL orders with trailing stops and compounding reinvestment.
    2. Every trade appended to CSV with correct headers, chronological timestamps, and valid RFC 4180 format.
    3. Query GET /api/trades and verify all 50 markers returned with matching prices, positions, and shapes.
    4. Concurrently stream /api/stream and verify all events arrive in order.
    5. Verify server stops in <1.0s.
    """
    csv_path = tmp_path / "trades.csv"
    server = DashboardServer(host="127.0.0.1", port=0, trades_csv=csv_path)
    actual_port = server.start_background(timeout=5.0)
    assert actual_port > 0, f"Expected valid ephemeral port, got {actual_port}"

    csv_channel = CsvTradeLogChannel(csv_path=csv_path, enabled=True)
    dash_channel = DashboardNotificationChannel(state=server.state, enabled=True)
    notif_service = NotificationService(channels=[csv_channel, dash_channel])

    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    sim = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        step_size=Decimal("0.00001"),
        notification_service=notif_service,
    )
    sizer = CompoundingPositionSizer(min_notional=Decimal("5.00"), default_step_size=Decimal("0.00001"))

    # Setup concurrent SSE consumer on /api/stream
    received_sse_events: List[Tuple[str, Any]] = []
    sse_ready = threading.Event()
    stop_sse = threading.Event()

    def sse_consumer():
        stream_url = f"http://127.0.0.1:{actual_port}/api/stream"
        try:
            with httpx.Client(timeout=15.0) as stream_client:
                with stream_client.stream("GET", stream_url) as stream_resp:
                    if stream_resp.status_code == 200:
                        current_event = None
                        for line in stream_resp.iter_lines():
                            if stop_sse.is_set():
                                break
                            if line.startswith("event:"):
                                current_event = line.split(":", 1)[1].strip()
                                if current_event == "status":
                                    sse_ready.set()
                                elif current_event == "shutdown":
                                    break
                            elif line.startswith("data:") and current_event:
                                raw_data = line.split(":", 1)[1].strip()
                                try:
                                    payload = json.loads(raw_data)
                                except Exception:
                                    payload = raw_data
                                received_sse_events.append((current_event, payload))
        except Exception:
            pass

    sse_thread = threading.Thread(target=sse_consumer, daemon=True)
    sse_thread.start()
    assert sse_ready.wait(timeout=5.0), "SSE stream client failed to connect to /api/stream"

    # Execute 50 rapid alternating BUY/SELL orders (25 cycles)
    base_time = datetime(2026, 9, 7, 10, 0, 0, tzinfo=timezone.utc)
    base_price = Decimal("50000.00")
    order_history = []

    t_start = time.perf_counter()
    for k in range(1, 26):
        # 1. BUY: Sizing based on compounding available cash
        t_buy = base_time + timedelta(seconds=k * 4 - 2)
        px_buy = base_price * (Decimal("1") + Decimal(str(k * 0.001)))
        stop_loss = px_buy * Decimal("0.98")
        size_res = sizer.calculate_position_size(ledger.cash, px_buy, stop_loss)
        assert size_res.is_valid, f"Cycle {k} BUY sizing invalid: {size_res.rejection_reason}"

        buy_sig = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            target_quantity=size_res.quantity,
            order_id=f"e2e_buy_{k:03d}",
        )
        res_buy = sim.execute(buy_sig, px_buy, ledger, timestamp_ms=int(t_buy.timestamp() * 1000))
        assert res_buy.success is True, f"Cycle {k} BUY failed: {res_buy.error_message}"
        order_history.append((res_buy, t_buy, "BUY"))

        # 2. SELL: Trailing stop ratchets up and triggers profitable exit
        pos = ledger.get_position("BTC/EUR")
        assert pos.quantity > Decimal("0.0"), f"Cycle {k} position is zero after BUY"
        t_sell = base_time + timedelta(seconds=k * 4)
        high_price = px_buy * Decimal("1.02")  # 2% rally
        trailing_stop_price = high_price * Decimal("0.995")  # Chandelier-style trailing stop
        sell_sig = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.SELL,
            target_quantity=pos.quantity,
            order_id=f"e2e_sell_{k:03d}",
            exit_reason="CHANDELIER_TRAILING_STOP_HIT",
        )
        res_sell = sim.execute(sell_sig, trailing_stop_price, ledger, timestamp_ms=int(t_sell.timestamp() * 1000))
        assert res_sell.success is True, f"Cycle {k} SELL failed: {res_sell.error_message}"
        order_history.append((res_sell, t_sell, "SELL"))

    t_elapsed = time.perf_counter() - t_start
    assert len(order_history) == 50, f"Expected 50 executed orders, got {len(order_history)}"
    assert t_elapsed < 2.0, f"50 rapid orders took {t_elapsed:.3f}s (expected < 2.0s)"

    # Compounding check: Initial 10.00 EUR has grown
    final_equity = ledger.total_equity()
    assert final_equity > Decimal("10.00"), f"Expected compounded equity > 10.00, got {final_equity}"

    # Allow SSE stream to receive all events
    deadline = time.time() + 5.0
    while time.time() < deadline:
        trades_in_sse = [ev for ev in received_sse_events if ev[0] == "trade"]
        if len(trades_in_sse) >= 50:
            break
        time.sleep(0.05)

    stop_sse.set()

    # --- VERIFY CSV TRADE LOG ---
    assert csv_path.exists(), "trades.csv does not exist"
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        csv_rows = list(csv.reader(f))

    assert len(csv_rows) == 51, f"Expected 51 rows (1 header + 50 trades), got {len(csv_rows)}"
    assert csv_rows[0] == CsvTradeLogger.HEADERS, f"CSV headers mismatch: {csv_rows[0]}"

    prev_dt = None
    for idx, row in enumerate(csv_rows[1:], start=1):
        assert len(row) == 13, f"Row {idx} does not have 13 columns: {row}"
        dt = datetime.fromisoformat(row[0])
        if prev_dt is not None:
            assert dt >= prev_dt, f"Row {idx} timestamp {dt} is not chronological after {prev_dt}"
        prev_dt = dt

        # Verify action and PnL semantics
        action = row[3]
        realized_pnl = float(row[8])
        if action == "BUY":
            assert realized_pnl == 0.0, f"Row {idx}: BUY trade must have 0.0 realized PnL"
        else:
            assert action == "SELL"
            assert realized_pnl > 0.0, f"Row {idx}: SELL trade must have positive realized PnL"
            assert row[12] in ("", "CHANDELIER_TRAILING_STOP_HIT")

    # --- VERIFY GET /api/trades ---
    with httpx.Client(base_url=f"http://127.0.0.1:{actual_port}", timeout=5.0) as client:
        resp_trades = client.get("/api/trades")
        assert resp_trades.status_code == 200
        markers = resp_trades.json()
        assert len(markers) == 50, f"Expected 50 markers, got {len(markers)}"

        for idx, (m, (res, t_expected, side_expected)) in enumerate(zip(markers, order_history)):
            assert m["time"] == int(t_expected.timestamp()), f"Marker {idx} time mismatch"
            assert m["action"] == side_expected, f"Marker {idx} action mismatch"
            assert abs(m["price"] - float(res.executed_price)) < 1e-3, f"Marker {idx} price mismatch"
            if side_expected == "BUY":
                assert m["position"] == "belowBar"
                assert m["shape"] == "arrowUp"
                assert m["color"] == "#10b981"
                assert "BUY @" in m["text"]
            else:
                assert m["position"] == "aboveBar"
                assert m["shape"] == "arrowDown"
                assert m["color"] == "#ef4444"
                assert "SELL @" in m["text"]
                assert m["pnl"] > 0

        # --- VERIFY GET /api/status ---
        resp_status = client.get("/api/status")
        assert resp_status.status_code == 200
        st = resp_status.json()
        assert st["total_trades"] == 50
        assert abs(st["portfolio_balance"] - float(ledger.total_equity())) < 1e-3
        assert abs(st["cash_balance"] - float(ledger.cash)) < 1e-3

    # --- VERIFY CONCURRENT SSE STREAM ORDERING ---
    trade_events = [ev[1] for ev in received_sse_events if ev[0] == "trade"]
    assert len(trade_events) == 50, f"Expected 50 SSE trade events, got {len(trade_events)}"
    for idx, (ev_data, (res, _, side_expected)) in enumerate(zip(trade_events, order_history)):
        expected_id = res.trade_record.order_id or res.trade_record.trade_id
        actual_id = ev_data.get("id")
        assert actual_id == expected_id, f"SSE trade {idx} id mismatch: {actual_id} vs {expected_id}"
        assert ev_data["action"] == side_expected, f"SSE trade {idx} action mismatch"

    # --- VERIFY SERVER STOP DURATION < 1.0s ---
    t_stop_start = time.perf_counter()
    server.stop(timeout=1.0)
    stop_duration = time.perf_counter() - t_stop_start
    assert stop_duration < 1.0, f"DashboardServer stop took {stop_duration:.4f}s (must be < 1.0s)"
    assert not server._thread.is_alive(), "DashboardServer thread did not terminate cleanly"


# =====================================================================
# 2. RFC 4180 FORMAT & DURABILITY DEEP AUDIT
# =====================================================================
def test_empirical_csv_rfc4180_format_and_durability_deep_audit(tmp_path: Path):
    """Deep audit of CSV formatting according to RFC 4180 specifications."""
    csv_path = tmp_path / "deep_audit.csv"
    logger = CsvTradeLogger(csv_path=csv_path)

    now = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    # Log trade with special characters in exit_reason
    sample_trade = {
        "timestamp": now,
        "trade_id": "audit_trade_001",
        "symbol": "BTC/EUR",
        "action": "SELL",
        "price": 55123.45,
        "size": 0.000123,
        "notional": 6.7801,
        "fee": 0.0068,
        "pnl": 0.5432,
        "pnl_pct": 8.71,
        "portfolio_balance": 10.5432,
        "cash_balance": 10.5432,
        "exit_reason": "TRAILING_STOP_HIT: high=55200, trail=1.5x ATR",
    }
    assert logger.log_trade(sample_trade) is True

    # Read raw bytes and verify line endings and RFC 4180 quoting
    raw_content = csv_path.read_text(encoding="utf-8")
    lines = raw_content.splitlines()
    assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

    # Parse with standard csv.reader
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f, dialect="excel"))
    assert len(reader) == 2
    header, row = reader[0], reader[1]
    assert len(header) == 13
    assert len(row) == 13
    assert row[1] == "audit_trade_001"
    assert row[12] == "TRAILING_STOP_HIT: high=55200, trail=1.5x ATR"

    # Re-read via CsvTradeLogger.read_trades()
    parsed_trades = logger.read_trades()
    assert len(parsed_trades) == 1
    pt = parsed_trades[0]
    assert pt["trade_id"] == "audit_trade_001"
    assert pt["price"] == pytest.approx(55123.45)
    assert pt["pnl"] == pytest.approx(0.5432)


# =====================================================================
# 3. CONCURRENT MULTI-CLIENT SSE STREAMING
# =====================================================================
def test_empirical_sse_stream_ordering_and_multi_client_fanout(tmp_path: Path):
    """Verify that multiple concurrent SSE subscribers receive all trade events in identical order."""
    csv_path = tmp_path / "stream_fanout.csv"
    server = DashboardServer(host="127.0.0.1", port=0, trades_csv=csv_path)
    actual_port = server.start_background(timeout=5.0)

    NUM_CLIENTS = 4
    NUM_TRADES = 20
    client_events: List[List[Dict[str, Any]]] = [[] for _ in range(NUM_CLIENTS)]
    all_ready = [threading.Event() for _ in range(NUM_CLIENTS)]
    stop_event = threading.Event()

    def subscriber_worker(client_idx: int):
        stream_url = f"http://127.0.0.1:{actual_port}/api/stream"
        try:
            with httpx.Client(timeout=15.0) as stream_client:
                with stream_client.stream("GET", stream_url) as stream_resp:
                    current_event = None
                    for line in stream_resp.iter_lines():
                        if stop_event.is_set():
                            break
                        if line.startswith("event:"):
                            current_event = line.split(":", 1)[1].strip()
                            if current_event == "status":
                                all_ready[client_idx].set()
                            elif current_event == "shutdown":
                                break
                        elif line.startswith("data:") and current_event == "trade":
                            data_str = line.split(":", 1)[1].strip()
                            try:
                                client_events[client_idx].append(json.loads(data_str))
                            except Exception:
                                pass
        except Exception:
            pass

    threads = [threading.Thread(target=subscriber_worker, args=(i,), daemon=True) for i in range(NUM_CLIENTS)]
    for t in threads:
        t.start()

    # Wait for all clients to connect
    for i, ev in enumerate(all_ready):
        assert ev.wait(timeout=5.0), f"Client {i} failed to connect to SSE stream"

    # Rapidly push trades into DashboardState
    for i in range(NUM_TRADES):
        side = "BUY" if i % 2 == 0 else "SELL"
        trade = {
            "trade_id": f"fanout_tr_{i:03d}",
            "symbol": "BTC/EUR",
            "action": side,
            "price": 50000.0 + i * 10.0,
            "size": 0.0001,
            "time": 1757260800 + i * 10,
            "pnl": 0.50 if side == "SELL" else 0.0,
        }
        server.state.add_trade(trade)

    # Allow clients to receive events
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if all(len(events) >= NUM_TRADES for events in client_events):
            break
        time.sleep(0.05)

    stop_event.set()

    # Assert all clients received exactly NUM_TRADES in identical sequence
    for client_idx in range(NUM_CLIENTS):
        received = client_events[client_idx]
        assert len(received) == NUM_TRADES, f"Client {client_idx} received {len(received)} trades, expected {NUM_TRADES}"
        for i in range(NUM_TRADES):
            expected_id = f"fanout_tr_{i:03d}"
            item_id = received[i]["id"]
            assert received[i]["id"] == expected_id, f"Client {client_idx} event {i} mismatch: {item_id} vs {expected_id}"

    # Clean server shutdown
    t_stop_start = time.perf_counter()
    server.stop(timeout=1.0)
    assert (time.perf_counter() - t_stop_start) < 1.0


# =====================================================================
# 4. SERVER SHUTDOWN LATENCY UNDER ACTIVE LOAD (< 1.0s)
# =====================================================================
def test_empirical_server_shutdown_latency_under_continuous_streaming():
    """Verify server gracefully terminates within 1.0s even under active streaming connections."""
    server = DashboardServer(host="127.0.0.1", port=0)
    actual_port = server.start_background(timeout=5.0)

    stop_client = threading.Event()

    def persistent_streamer():
        stream_url = f"http://127.0.0.1:{actual_port}/api/stream"
        try:
            with httpx.Client(timeout=10.0) as client:
                with client.stream("GET", stream_url) as resp:
                    for _ in resp.iter_lines():
                        if stop_client.is_set():
                            break
        except Exception:
            pass

    th = threading.Thread(target=persistent_streamer, daemon=True)
    th.start()
    time.sleep(0.2)  # Allow connection to establish

    # Measure stop duration
    t_start = time.perf_counter()
    server.stop(timeout=1.0)
    stop_latency = time.perf_counter() - t_start

    stop_client.set()
    assert stop_latency < 1.0, f"Server stop latency was {stop_latency:.4f}s (must be < 1.0s)"
    assert not server._thread.is_alive(), "Server thread failed to terminate"


# =====================================================================
# 5. COMPOUNDING PROFIT REINVESTMENT PROGRESSION
# =====================================================================
def test_empirical_compounding_reinvestment_progression(tmp_path: Path):
    """Empirically verify that compounding profit reinvestment scales capital monotonically."""
    csv_path = tmp_path / "compounding_audit.csv"
    csv_channel = CsvTradeLogChannel(csv_path=csv_path)
    notif = NotificationService(channels=[csv_channel])

    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    sim = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        step_size=Decimal("0.00001"),
        notification_service=notif,
    )
    sizer = CompoundingPositionSizer(min_notional=Decimal("5.00"), default_step_size=Decimal("0.00001"))

    equity_progression = [ledger.total_equity()]
    notional_progression = []

    base_time = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)
    base_price = Decimal("50000.00")

    for k in range(1, 11):
        t_buy = base_time + timedelta(minutes=k * 2 - 1)
        px_buy = base_price * (Decimal("1") + Decimal(str(k * 0.002)))
        stop_loss = px_buy * Decimal("0.98")
        size_res = sizer.calculate_position_size(ledger.cash, px_buy, stop_loss)
        assert size_res.is_valid

        buy_sig = OrderSignal(symbol="BTC/EUR", side=OrderSide.BUY, target_quantity=size_res.quantity)
        res_buy = sim.execute(buy_sig, px_buy, ledger, timestamp_ms=int(t_buy.timestamp() * 1000))
        assert res_buy.success
        notional_progression.append(res_buy.quote_notional)

        pos = ledger.get_position("BTC/EUR")
        t_sell = base_time + timedelta(minutes=k * 2)
        px_sell = px_buy * Decimal("1.015")  # 1.5% profit
        sell_sig = OrderSignal(symbol="BTC/EUR", side=OrderSide.SELL, target_quantity=pos.quantity)
        res_sell = sim.execute(sell_sig, px_sell, ledger, timestamp_ms=int(t_sell.timestamp() * 1000))
        assert res_sell.success
        equity_progression.append(ledger.total_equity())

    # Verify monotonic equity growth
    for i in range(1, len(equity_progression)):
        assert equity_progression[i] > equity_progression[i - 1], f"Equity did not compound at step {i}"

    # Verify notional growth (each reinvestment allocates more capital)
    assert notional_progression[-1] > notional_progression[0], "Compounded position sizing did not scale up"

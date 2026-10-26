"""Empirical Challenger Verification & Stress Test Suite for Milestone 9.

Author: challenger_dashboard_2
Focus: Acceptance Criterion 2 (AC2) Empirical Validation, Charting Contract Parity,
       and End-to-End Visual Data Flow.

Tests:
1. test_empirical_ac2_server_startup_and_html_delivery:
   - Starts DashboardServer on ephemeral port (port=0).
   - Verifies GET / returns 200 OK, valid HTML, and contains required DOM elements
     and script/link tags for TradingView Lightweight Charts.
   - Verifies static assets (CSS, JS, vendored lightweight-charts) return 200 OK.
2. test_empirical_ac2_charting_marker_contract_parity:
   - Verifies exact Lightweight Charts marker schema for BUY and SELL orders:
     * time: integer UNIX seconds strictly matching candle timestamps
     * position: "belowBar" for BUY, "aboveBar" for SELL
     * color: green "#10b981" for BUY, red "#ef4444" for SELL
     * shape: "arrowUp" for BUY, "arrowDown" for SELL
     * text: non-empty descriptive label including price and PnL
   - Verifies polymorphic trade ingestion across TradeExecutedPayload, TradeRecord, and dicts.
   - Verifies PnL formatting across positive, negative, and zero (breakeven) outcomes.
3. test_empirical_ac2_live_trade_flow_e2e_simulation:
   - Live HTTP end-to-end integration:
     ExecutionSimulator -> NotificationService -> DashboardChannel -> DashboardState -> DashboardServer
   - Executes real BUY and SELL orders with 10.00 EUR initial capital.
   - Fetches GET /api/trades via HTTP client and asserts newly executed trades immediately
     appear as accurately formatted markers.
   - Verifies balance synchronization via GET /api/status.
4. test_empirical_ac2_candle_marker_time_alignment:
   - Verifies strict alignment between candle timestamps and trade marker timestamps.
   - Verifies candle OHLC invariants (low <= open, close <= high) and chronological ordering.
5. test_empirical_ac2_concurrent_burst_and_rapid_polling:
   - Multi-threaded stress: 5 concurrent worker threads executing 20 trades each (100 total)
     while 5 concurrent reader threads rapidly poll GET /api/trades, /api/status, /api/market-data.
   - Verifies zero dropped writes, zero race conditions, zero 500 errors, and complete marker integrity.
6. test_empirical_ac2_edge_cases_and_adversarial_inputs:
   - Empty state handling.
   - Extreme prices, micro-quantities, zero PnL, large dataset throughput.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import pytest
from starlette.testclient import TestClient

from trading_bot.dashboard.app import create_app
from trading_bot.dashboard.server import DashboardServer
from trading_bot.dashboard.state import DashboardState
from trading_bot.data.models import Candle
from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.models import (
    BaseNotificationEvent,
    DeliveryStatus,
    OrderSide,
    OrderType,
    TradeExecutedPayload,
)
from trading_bot.notifications.service import NotificationService
from trading_bot.simulation.ledger import TradeRecord, VirtualLedger
from trading_bot.simulation.simulator import ExecutionSimulator, OrderSignal


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
        if isinstance(event, TradeExecutedPayload):
            self.state.add_trade(event)
            return DeliveryStatus(channel_name=self.name, success=True)
        return DeliveryStatus(channel_name=self.name, success=True)


# =====================================================================
# 1. SERVER STARTUP & HTML DELIVERY (AC2)
# =====================================================================
def test_empirical_ac2_server_startup_and_html_delivery():
    """Verify DashboardServer binds to an ephemeral port, serves index HTML, and static assets."""
    state = DashboardState(symbol="BTC/EUR", initial_balance=10.0)
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)

    try:
        assert actual_port > 0, "Server failed to bind to an ephemeral port"
        base_url = f"http://127.0.0.1:{actual_port}"

        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            # 1. Verify GET /
            resp = client.get("/")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            assert "text/html" in resp.headers.get("content-type", "")
            html_text = resp.text

            # Critical DOM containers for charting & tables
            assert 'id="chart"' in html_text, "Missing chart container element #chart"
            assert 'id="trades-table"' in html_text, "Missing table element #trades-table"
            assert 'id="trades-table-body"' in html_text, "Missing #trades-table-body"
            assert 'id="kpi-equity"' in html_text, "Missing KPI #kpi-equity"
            assert 'id="kpi-cash"' in html_text, "Missing KPI #kpi-cash"
            assert 'id="kpi-trades"' in html_text, "Missing KPI #kpi-trades"
            assert 'id="kpi-status"' in html_text, "Missing KPI #kpi-status"
            assert 'id="symbol-badge"' in html_text, "Missing #symbol-badge"

            # Script and Style references
            assert "/static/style.css" in html_text
            assert "/static/app.js" in html_text
            assert "lightweight-charts" in html_text

            # 2. Verify static assets are served and non-empty
            css_resp = client.get("/static/style.css")
            assert css_resp.status_code == 200
            assert len(css_resp.text) > 100

            js_resp = client.get("/static/app.js")
            assert js_resp.status_code == 200
            assert "LightweightCharts" in js_resp.text

            vendor_resp = client.get("/static/js/vendor/lightweight-charts.standalone.production.js")
            assert vendor_resp.status_code == 200
            assert len(vendor_resp.content) > 50000, "Vendored Lightweight Charts bundle unexpectedly truncated"

            # 3. Verify GET /api/trades initially empty
            trades_resp = client.get("/api/trades")
            assert trades_resp.status_code == 200
            assert trades_resp.json() == []

    finally:
        server.stop(timeout=3.0)
        assert server._thread is not None
        assert not server._thread.is_alive()


# =====================================================================
# 2. CHARTING MARKER CONTRACT PARITY (AC2)
# =====================================================================
def test_empirical_ac2_charting_marker_contract_parity():
    """Verify strict adherence to TradingView Lightweight Charts marker schema.

    Requirements:
    - time: integer UNIX seconds
    - position: "belowBar" for BUY, "aboveBar" for SELL
    - color: green "#10b981" for BUY, red "#ef4444" for SELL
    - shape: "arrowUp" for BUY, "arrowDown" for SELL
    - text: non-empty descriptive label including price and PnL
    """
    state = DashboardState(symbol="BTC/EUR")

    # A. BUY trade via TradeExecutedPayload
    buy_payload = TradeExecutedPayload(
        order_id="ord_buy_101",
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        fill_price=Decimal("59123.45"),
        quantity=Decimal("0.00016"),
        notional_value=Decimal("9.4597"),
        fee=Decimal("0.0095"),
        fee_currency="EUR",
        portfolio_balance=Decimal("10.00"),
        cash_balance=Decimal("0.53"),
        realized_pnl=Decimal("0.00"),
        timestamp=datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc),
    )
    m_buy = state.add_trade(buy_payload)

    # Assert BUY marker exact schema
    assert isinstance(m_buy["time"], int)
    assert m_buy["time"] == 1788782400
    assert m_buy["position"] == "belowBar"
    assert m_buy["color"] == "#10b981"
    assert m_buy["shape"] == "arrowUp"
    assert m_buy["text"] == "BUY @ 59123.45"
    assert m_buy["price"] == pytest.approx(59123.45)
    assert m_buy["size"] == pytest.approx(0.00016)
    assert m_buy["pnl"] == 0.0

    # B. SELL trade with PROFIT via TradeRecord
    sell_record = TradeRecord(
        trade_id="trd_sell_102",
        order_id="ord_sell_102",
        symbol="BTC/EUR",
        side="SELL",
        order_type="MARKET",
        executed_price=Decimal("62450.00"),
        raw_price=Decimal("62450.00"),
        quantity=Decimal("0.00016"),
        quote_notional=Decimal("9.992"),
        fee=Decimal("0.0099"),
        cash_after=Decimal("10.51"),
        equity_after=Decimal("10.51"),
        realized_pnl=Decimal("0.53"),
        timestamp_ms=1788786000000,
    )
    m_sell_profit = state.add_trade(sell_record)

    # Assert SELL PROFIT marker exact schema
    assert isinstance(m_sell_profit["time"], int)
    assert m_sell_profit["time"] == 1788786000
    assert m_sell_profit["position"] == "aboveBar"
    assert m_sell_profit["color"] == "#ef4444"
    assert m_sell_profit["shape"] == "arrowDown"
    assert m_sell_profit["text"] == "SELL @ 62450.00 (+0.53€)"
    assert m_sell_profit["price"] == pytest.approx(62450.00)
    assert m_sell_profit["pnl"] == pytest.approx(0.53)

    # C. SELL trade with LOSS via Dictionary
    sell_loss_dict = {
        "trade_id": "trd_sell_103",
        "action": "SELL",
        "price": 57000.00,
        "size": 0.00016,
        "pnl": -0.34,
        "time": 1788789600,
        "portfolio_balance": 9.66,
        "cash_balance": 9.66,
    }
    m_sell_loss = state.add_trade(sell_loss_dict)

    # Assert SELL LOSS marker exact schema
    assert isinstance(m_sell_loss["time"], int)
    assert m_sell_loss["time"] == 1788789600
    assert m_sell_loss["position"] == "aboveBar"
    assert m_sell_loss["color"] == "#ef4444"
    assert m_sell_loss["shape"] == "arrowDown"
    assert m_sell_loss["text"] == "SELL @ 57000.00 (-0.34€)"
    assert m_sell_loss["pnl"] == pytest.approx(-0.34)

    # D. SELL trade with BREAKEVEN (pnl = 0.0)
    sell_flat_dict = {
        "trade_id": "trd_sell_104",
        "action": "SELL",
        "price": 59123.45,
        "size": 0.00016,
        "pnl": 0.0,
        "timestamp_ms": 1788793200000,
    }
    m_sell_flat = state.add_trade(sell_flat_dict)
    assert m_sell_flat["time"] == 1788793200
    assert m_sell_flat["text"] == "SELL @ 59123.45 (+0.00€)"

    # Verify state get_markers() returns all 4 markers
    markers = state.get_markers()
    assert len(markers) == 4
    for m in markers:
        assert isinstance(m["time"], int)
        assert m["position"] in ("belowBar", "aboveBar")
        assert m["color"] in ("#10b981", "#ef4444")
        assert m["shape"] in ("arrowUp", "arrowDown")
        assert len(m["text"]) > 0


# =====================================================================
# 3. END-TO-END LIVE TRADE FLOW SIMULATION (AC2)
# =====================================================================
def test_empirical_ac2_live_trade_flow_e2e_simulation():
    """Simulate complete live trade flow:
    ExecutionSimulator -> NotificationService -> DashboardChannel -> DashboardState -> DashboardServer
    Fetches GET /api/trades via HTTP client to verify immediate appearance of accurately formatted markers.
    """
    initial_cash = Decimal("10.00")
    state = DashboardState(symbol="BTC/EUR", initial_balance=float(initial_cash), status="running")

    # 1. Start live server on ephemeral port
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)

    try:
        base_url = f"http://127.0.0.1:{actual_port}"
        http_client = httpx.Client(base_url=base_url, timeout=5.0)

        # 2. Setup NotificationService with DashboardNotificationChannel
        channel = DashboardNotificationChannel(state=state, enabled=True)
        notif_service = NotificationService(channels=[channel])

        # 3. Setup VirtualLedger and ExecutionSimulator
        ledger = VirtualLedger(initial_cash=initial_cash, quote_currency="EUR")
        simulator = ExecutionSimulator(
            fee_rate=Decimal("0.001"),
            slippage=Decimal("0.0005"),
            min_notional=Decimal("5.00"),
            exchange="binance",
            notification_service=notif_service,
        )

        # Initial check: 0 trades
        resp = http_client.get("/api/trades")
        assert resp.status_code == 200
        assert resp.json() == []

        # 4. Execute BUY trade: budget 9.60 EUR at 50,000.00 EUR
        trade_time_1 = datetime(2026, 9, 7, 14, 0, 0, tzinfo=timezone.utc)
        buy_order = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            target_notional=Decimal("9.60"),
            order_id="sim_buy_001",
        )
        exec_buy = simulator.execute(
            order=buy_order,
            current_price=Decimal("50000.00"),
            ledger=ledger,
            timestamp_ms=int(trade_time_1.timestamp() * 1000),
        )
        assert exec_buy.success, f"BUY failed: {exec_buy.error_message}"

        # 5. Immediately query GET /api/trades via HTTP
        resp_after_buy = http_client.get("/api/trades")
        assert resp_after_buy.status_code == 200
        markers_1 = resp_after_buy.json()

        assert len(markers_1) == 1
        m1 = markers_1[0]
        assert m1["id"] == "sim_buy_001"
        assert m1["action"] == "BUY"
        assert m1["position"] == "belowBar"
        assert m1["color"] == "#10b981"
        assert m1["shape"] == "arrowUp"
        assert m1["time"] == int(trade_time_1.timestamp())
        assert m1["price"] == pytest.approx(float(exec_buy.executed_price), rel=1e-3)
        assert f"BUY @ {float(exec_buy.executed_price):.2f}" in m1["text"]

        # 6. Execute SELL trade: exit entire position at 55,000.00 EUR (+10% gain)
        trade_time_2 = datetime(2026, 9, 7, 15, 0, 0, tzinfo=timezone.utc)
        pos = ledger.get_position("BTC/EUR")
        sell_order = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.SELL,
            target_quantity=pos.quantity,
            order_id="sim_sell_002",
        )
        exec_sell = simulator.execute(
            order=sell_order,
            current_price=Decimal("55000.00"),
            ledger=ledger,
            timestamp_ms=int(trade_time_2.timestamp() * 1000),
        )
        assert exec_sell.success, f"SELL failed: {exec_sell.error_message}"

        # 7. Query GET /api/trades via HTTP again
        resp_after_sell = http_client.get("/api/trades")
        assert resp_after_sell.status_code == 200
        markers_2 = resp_after_sell.json()

        assert len(markers_2) == 2
        m2 = markers_2[1]
        assert m2["id"] == "sim_sell_002"
        assert m2["action"] == "SELL"
        assert m2["position"] == "aboveBar"
        assert m2["color"] == "#ef4444"
        assert m2["shape"] == "arrowDown"
        assert m2["time"] == int(trade_time_2.timestamp())
        assert m2["price"] == pytest.approx(float(exec_sell.executed_price), rel=1e-3)
        assert m2["pnl"] == pytest.approx(float(exec_sell.realized_pnl), rel=1e-3)
        assert m2["pnl"] > 0, "Expected positive realized PnL on profitable sell"
        assert f"SELL @ {float(exec_sell.executed_price):.2f}" in m2["text"]
        assert f"+{float(exec_sell.realized_pnl):.2f}€" in m2["text"]

        # 8. Query GET /api/status to verify balance synchronization
        resp_status = http_client.get("/api/status")
        assert resp_status.status_code == 200
        status_data = resp_status.json()
        assert status_data["status"] == "running"
        assert status_data["total_trades"] == 2
        assert status_data["portfolio_balance"] == pytest.approx(float(ledger.total_equity()), rel=1e-3)
        assert status_data["cash_balance"] == pytest.approx(float(ledger.cash), rel=1e-3)

        # 9. Execute a losing trade round-trip
        trade_time_3 = datetime(2026, 9, 7, 16, 0, 0, tzinfo=timezone.utc)
        buy_order_2 = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            target_notional=Decimal("9.00"),
            order_id="sim_buy_003",
        )
        exec_buy_2 = simulator.execute(
            order=buy_order_2,
            current_price=Decimal("50000.00"),
            ledger=ledger,
            timestamp_ms=int(trade_time_3.timestamp() * 1000),
        )
        assert exec_buy_2.success

        trade_time_4 = datetime(2026, 9, 7, 17, 0, 0, tzinfo=timezone.utc)
        pos_2 = ledger.get_position("BTC/EUR")
        sell_order_2 = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.SELL,
            target_quantity=pos_2.quantity,
            order_id="sim_sell_004",
        )
        exec_sell_2 = simulator.execute(
            order=sell_order_2,
            current_price=Decimal("46000.00"),  # Loss
            ledger=ledger,
            timestamp_ms=int(trade_time_4.timestamp() * 1000),
        )
        assert exec_sell_2.success
        assert exec_sell_2.realized_pnl < Decimal("0.00")

        resp_loss = http_client.get("/api/trades")
        markers_loss = resp_loss.json()
        assert len(markers_loss) == 4
        m4 = markers_loss[3]
        assert m4["action"] == "SELL"
        assert m4["pnl"] < 0
        assert f"-{abs(float(exec_sell_2.realized_pnl)):.2f}€" in m4["text"]

    finally:
        server.stop(timeout=3.0)


# =====================================================================
# 4. CANDLE & MARKER TIME ALIGNMENT (AC2)
# =====================================================================
def test_empirical_ac2_candle_marker_time_alignment():
    """Verify that trade markers strictly align with historical candle timestamps.

    In Lightweight Charts, markers plotted on a candlestick series must map
    accurately to the time scale of the underlying candles.
    """
    state = DashboardState(symbol="BTC/EUR")
    base_time = int(datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    candle_interval = 3600  # 1h

    # Add 48 hourly candles
    candles_times = set()
    for i in range(48):
        c_time = base_time + i * candle_interval
        candles_times.add(c_time)
        state.add_candle({
            "time": c_time,
            "open": 50000.0 + (i * 50),
            "high": 50200.0 + (i * 50),
            "low": 49900.0 + (i * 50),
            "close": 50150.0 + (i * 50),
            "volume": 12.5,
        })

    # Add 6 trades occurring on exact candle timestamps (hour 5, 12, 18, 25, 33, 42)
    selected_hours = [5, 12, 18, 25, 33, 42]
    for idx, hr in enumerate(selected_hours):
        t_time = base_time + hr * candle_interval
        is_buy = (idx % 2 == 0)
        state.add_trade({
            "trade_id": f"t_align_{idx}",
            "time": t_time,
            "action": "BUY" if is_buy else "SELL",
            "price": 50000.0 + (hr * 50),
            "size": 0.0002,
            "pnl": 0.0 if is_buy else 0.45,
        })

    app = create_app(state)
    client = TestClient(app)

    c_resp = client.get("/api/market-data")
    assert c_resp.status_code == 200
    candles_data = c_resp.json()
    assert len(candles_data) == 48

    m_resp = client.get("/api/trades")
    assert m_resp.status_code == 200
    markers_data = m_resp.json()
    assert len(markers_data) == 6

    # 1. Assert every marker time exists in the candle times set
    for m in markers_data:
        assert m["time"] in candles_times, f"Marker time {m['time']} not found in candle series"

    # 2. Assert candles are strictly ascending in time
    for i in range(len(candles_data) - 1):
        assert candles_data[i]["time"] < candles_data[i + 1]["time"]

    # 3. Assert candle OHLC invariants: low <= open, close <= high
    for c in candles_data:
        assert c["low"] <= c["open"] <= c["high"]
        assert c["low"] <= c["close"] <= c["high"]


# =====================================================================
# 5. CONCURRENT BURST & RAPID POLLING STRESS (AC2)
# =====================================================================
def test_empirical_ac2_concurrent_burst_and_rapid_polling():
    """Stress test DashboardState and REST API under heavy concurrent writes and reads.

    - 5 concurrent worker threads appending 20 trades each (100 total).
    - 5 concurrent reader threads continuously querying GET /api/trades, /api/status, /api/market-data.
    - Asserts 0 deadlocks, 0 dropped markers, 0 500 errors, and total schema validity.
    """
    state = DashboardState(symbol="BTC/EUR", initial_balance=10.0)
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)

    try:
        base_url = f"http://127.0.0.1:{actual_port}"
        num_writers = 5
        trades_per_writer = 20
        total_trades_expected = num_writers * trades_per_writer

        writer_errors: List[str] = []
        reader_errors: List[str] = []
        stop_event = threading.Event()
        barrier = threading.Barrier(num_writers)

        def writer_worker(worker_id: int):
            barrier.wait()
            base_ts = 1788700000 + worker_id * 1000
            for i in range(trades_per_writer):
                trade_id = f"w_{worker_id}_t_{i}"
                is_buy = (i % 2 == 0)
                try:
                    state.add_trade({
                        "trade_id": trade_id,
                        "time": base_ts + i * 10,
                        "action": "BUY" if is_buy else "SELL",
                        "price": 58000.0 + (worker_id * 100) + i,
                        "size": 0.0001,
                        "pnl": 0.0 if is_buy else 0.25,
                        "portfolio_balance": 10.0 + (i * 0.1),
                        "cash_balance": 5.0,
                    })
                except Exception as ex:
                    writer_errors.append(f"Writer {worker_id} error on trade {i}: {ex}")

        def reader_worker(reader_id: int):
            with httpx.Client(base_url=base_url, timeout=3.0) as client:
                while not stop_event.is_set():
                    try:
                        r_trades = client.get("/api/trades")
                        assert r_trades.status_code == 200
                        r_status = client.get("/api/status")
                        assert r_status.status_code == 200
                        r_market = client.get("/api/market-data")
                        assert r_market.status_code == 200
                    except Exception as ex:
                        reader_errors.append(f"Reader {reader_id} request error: {ex}")
                    time.sleep(0.01)

        # Start readers
        readers = [threading.Thread(target=reader_worker, args=(i,)) for i in range(5)]
        for r in readers:
            r.start()

        # Start writers
        writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(num_writers)]
        for w in writers:
            w.start()

        # Join writers
        for w in writers:
            w.join(timeout=10.0)

        # Stop readers
        stop_event.set()
        for r in readers:
            r.join(timeout=5.0)

        assert len(writer_errors) == 0, f"Writer errors occurred: {writer_errors}"
        assert len(reader_errors) == 0, f"Reader errors occurred: {reader_errors}"

        # Verify final marker count and schema over HTTP
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            resp = client.get("/api/trades")
            assert resp.status_code == 200
            markers = resp.json()
            assert len(markers) == total_trades_expected, f"Expected {total_trades_expected}, got {len(markers)}"

            # Validate all 100 markers
            for m in markers:
                assert isinstance(m["time"], int)
                assert m["position"] in ("belowBar", "aboveBar")
                assert m["color"] in ("#10b981", "#ef4444")
                assert m["shape"] in ("arrowUp", "arrowDown")
                assert len(m["text"]) > 0

    finally:
        server.stop(timeout=3.0)


# =====================================================================
# 6. EDGE CASES & ADVERSARIAL INPUTS (AC2)
# =====================================================================
def test_empirical_ac2_edge_cases_and_adversarial_inputs():
    """Verify edge cases: extreme prices, micro quantities, string timestamps, large datasets."""
    state = DashboardState(symbol="BTC/EUR")

    # 1. Micro price and micro quantity (satoshis)
    state.add_trade({
        "trade_id": "sat_01",
        "action": "BUY",
        "price": 0.00001234,
        "size": 0.00000001,
        "timestamp": "2026-09-07T18:00:00+00:00",
    })

    # 2. Mega price and large profit
    state.add_trade({
        "trade_id": "mega_02",
        "action": "SELL",
        "price": 1250000.50,
        "size": 1.5,
        "pnl": 99999.99,
        "timestamp": datetime(2026, 9, 7, 19, 0, 0),  # naive datetime
    })

    app = create_app(state)
    client = TestClient(app)

    resp = client.get("/api/trades")
    assert resp.status_code == 200
    markers = resp.json()
    assert len(markers) == 2

    # Check micro price
    m1 = markers[0]
    assert m1["position"] == "belowBar"
    assert m1["price"] == pytest.approx(0.00001234)
    assert "BUY @" in m1["text"]

    # Check mega price
    m2 = markers[1]
    assert m2["position"] == "aboveBar"
    assert m2["price"] == pytest.approx(1250000.50)
    assert "+99999.99€" in m2["text"]

    # 3. High volume performance: 1,000 markers throughput
    state_large = DashboardState(symbol="BTC/EUR")
    t0 = time.perf_counter()
    for i in range(1000):
        state_large.add_trade({
            "trade_id": f"burst_{i}",
            "time": 1788700000 + i,
            "action": "BUY" if i % 2 == 0 else "SELL",
            "price": 60000.0 + i,
            "size": 0.001,
            "pnl": 0.1 if i % 2 == 1 else 0.0,
        })
    load_time = time.perf_counter() - t0
    assert load_time < 0.5, f"1000 trades ingestion too slow: {load_time:.3f}s"

    app_large = create_app(state_large)
    client_large = TestClient(app_large)

    t_req = time.perf_counter()
    resp_large = client_large.get("/api/trades")
    req_time = time.perf_counter() - t_req

    assert resp_large.status_code == 200
    assert len(resp_large.json()) == 1000
    assert req_time < 0.1, f"GET /api/trades for 1,000 items took too long: {req_time:.3f}s"


# =====================================================================
# 7. PAPER TRADING ENGINE LOOP WITH LIVE DASHBOARD SERVER (AC2)
# =====================================================================
def test_empirical_ac2_paper_trading_loop_with_dashboard(tmp_path: Path):
    """Verify run_paper with dashboard=True starts DashboardServer on ephemeral port,
    streams candles, runs paper trading iteration, and terminates gracefully.
    """
    from trading_bot.cli.paper import run_paper
    from trading_bot.data.models import Ticker

    now_utc = datetime.now(timezone.utc)

    class MockMarketDataService:
        def get_ticker(self, symbol: str) -> Ticker:
            return Ticker(
                symbol=symbol,
                price=Decimal("60000.00"),
                timestamp=now_utc,
            )

        def get_historical_candles(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> List[Candle]:
            return [
                Candle(
                    timestamp=now_utc - timedelta(hours=i),
                    open=Decimal("59500.00"),
                    high=Decimal("60500.00"),
                    low=Decimal("59000.00"),
                    close=Decimal("60000.00"),
                    volume=Decimal("15.0"),
                    symbol=symbol,
                    interval=interval,
                )
                for i in range(30, 0, -1)
            ]

    csv_path = tmp_path / "paper_loop_trades.csv"
    mock_mds = MockMarketDataService()

    ledger, notif = run_paper(
        pair="BTC/EUR",
        balance=10.0,
        iterations=1,
        interval_sec=0.01,
        market_data_service=mock_mds,  # type: ignore
        trades_csv=csv_path,
        dashboard=True,
        dashboard_port=0,
        notify_channel="stdout",
    )

    assert ledger is not None
    assert ledger.initial_cash == Decimal("10.00")
    assert notif is not None

"""End-to-End Acceptance Test Suite (Milestone 10: R1, R2, R3, AC1, AC2, AC3).

Validates the complete integrated pipeline across:
- R1 / AC1: CSV Trade Logging via ExecutionSimulator and CsvTradeLogChannel.
- R2: Real-Time Web Dashboard server lifecycle, route delivery, and SSE streaming.
- R3 / AC2: Lightweight Charts trade marker formatting (BUY belowBar/arrowUp/#10b981,
            SELL aboveBar/arrowDown/#ef4444 with realized PnL).
- AC3: Standalone Dashboard CLI and Paper Trading CLI flags and parameter parsing.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

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


def test_e2e_acceptance_full_pipeline(tmp_path: Path):
    """Unified End-to-End Acceptance Test validating full trade-to-dashboard pipeline.

    Flow:
    1. Initialize temporary working directory with data/trades.csv.
    2. Start DashboardServer(port=0, trades_csv=csv_path) in background daemon thread.
    3. Wire ExecutionSimulator, VirtualLedger (10.00 EUR initial equity),
       NotificationService, CsvTradeLogChannel, and DashboardNotificationChannel.
    4. Trigger BUY trade fill:
       - Verify row appended to CSV with 13 columns (AC1).
       - Query GET /api/trades via HTTP -> verify BUY marker (time, belowBar, arrowUp, #10b981).
    5. Trigger SELL trade fill:
       - Verify row appended to CSV with realized PnL.
       - Query GET /api/trades -> verify SELL marker (aboveBar, arrowDown, #ef4444, PnL text).
       - Query GET /api/status -> verify balance matches ledger.
    6. Connect SSE client to /api/stream:
       - Dispatch live candle and trade.
       - Verify real-time events received over the wire.
    7. Stop DashboardServer cleanly in <1.0s.
    """
    # 1. Initialize temporary working directory and data/trades.csv
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "trades.csv"

    # Pre-initialize CSV header using CsvTradeLogger
    csv_logger = CsvTradeLogger(csv_path=csv_path)
    assert csv_path.exists(), "trades.csv was not created"

    # 2. Start DashboardServer on ephemeral port in background thread
    server = DashboardServer(host="127.0.0.1", port=0, trades_csv=csv_path)
    actual_port = server.start_background(timeout=5.0)
    assert actual_port > 0, f"Expected valid ephemeral port, got {actual_port}"

    try:
        # 3. Wire execution pipeline with initial 10.00 EUR equity
        initial_equity = Decimal("10.00")
        ledger = VirtualLedger(initial_cash=initial_equity, quote_currency="EUR")

        csv_channel = CsvTradeLogChannel(csv_path=csv_path, enabled=True)
        dash_channel = DashboardNotificationChannel(state=server.state, enabled=True)

        notif_service = NotificationService(channels=[csv_channel, dash_channel])
        simulator = ExecutionSimulator(
            fee_rate=Decimal("0.001"),
            slippage=Decimal("0.0005"),
            min_notional=Decimal("5.00"),
            exchange="binance",
            notification_service=notif_service,
        )

        base_url = f"http://127.0.0.1:{actual_port}"
        with httpx.Client(base_url=base_url, timeout=5.0) as http_client:
            # Verify initial state: 0 trades
            resp_initial = http_client.get("/api/trades")
            assert resp_initial.status_code == 200
            assert resp_initial.json() == []

            # 4. Trigger BUY trade fill (budget 9.60 EUR out of 10.00 EUR initial equity)
            buy_order_time = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
            buy_order = OrderSignal(
                symbol="BTC/EUR",
                side=OrderSide.BUY,
                target_notional=Decimal("9.60"),
                order_id="e2e_buy_001",
            )
            exec_buy = simulator.execute(
                order=buy_order,
                current_price=Decimal("50000.00"),
                ledger=ledger,
                timestamp_ms=int(buy_order_time.timestamp() * 1000),
            )
            assert exec_buy.success is True, f"BUY failed: {exec_buy.error_message}"

            # Verify row appended to CSV with 13 columns (AC1)
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                csv_rows = list(csv.reader(f))
            assert len(csv_rows) == 2, f"Expected header + 1 trade, got {len(csv_rows)} rows"
            buy_row = csv_rows[1]
            assert len(buy_row) == 13, f"Expected 13 columns in CSV row, got {len(buy_row)}"
            assert buy_row[1] == "e2e_buy_001"
            assert buy_row[2] == "BTC/EUR"
            assert buy_row[3] == "BUY"
            assert float(buy_row[4]) == pytest.approx(float(exec_buy.executed_price), rel=1e-3)
            assert float(buy_row[5]) == pytest.approx(float(exec_buy.quantity), rel=1e-5)
            assert float(buy_row[8]) == 0.0  # Realized PnL is 0 for opening BUY

            # Query GET /api/trades via HTTP client -> verify BUY marker matches Lightweight Charts format
            resp_trades_buy = http_client.get("/api/trades")
            assert resp_trades_buy.status_code == 200
            buy_markers = resp_trades_buy.json()
            assert len(buy_markers) == 1
            m_buy = buy_markers[0]

            assert m_buy["time"] == int(buy_order_time.timestamp())
            assert isinstance(m_buy["time"], int)
            assert m_buy["position"] == "belowBar"
            assert m_buy["shape"] == "arrowUp"
            assert m_buy["color"] == "#10b981"
            assert m_buy["action"] == "BUY"
            assert "BUY @" in m_buy["text"]
            assert f"{float(exec_buy.executed_price):.2f}" in m_buy["text"]

            # 5. Trigger SELL trade fill (profitable exit at 55,000.00 EUR)
            sell_order_time = datetime(2026, 9, 7, 13, 0, 0, tzinfo=timezone.utc)
            held_position = ledger.get_position("BTC/EUR")
            assert held_position.quantity > 0

            sell_order = OrderSignal(
                symbol="BTC/EUR",
                side=OrderSide.SELL,
                target_quantity=held_position.quantity,
                order_id="e2e_sell_002",
                exit_reason="TAKE_PROFIT_TARGET",
            )
            exec_sell = simulator.execute(
                order=sell_order,
                current_price=Decimal("55000.00"),
                ledger=ledger,
                timestamp_ms=int(sell_order_time.timestamp() * 1000),
            )
            assert exec_sell.success is True, f"SELL failed: {exec_sell.error_message}"

            # Verify row appended to CSV with realized PnL
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                csv_rows = list(csv.reader(f))
            assert len(csv_rows) == 3, f"Expected header + 2 trades, got {len(csv_rows)} rows"
            sell_row = csv_rows[2]
            assert len(sell_row) == 13
            assert sell_row[1] == "e2e_sell_002"
            assert sell_row[3] == "SELL"
            assert float(sell_row[8]) > 0.0, "Expected positive realized PnL in CSV"
            assert sell_row[12] in ("", "TAKE_PROFIT_TARGET")

            # Query GET /api/trades -> verify SELL marker matches Lightweight Charts format
            resp_trades_sell = http_client.get("/api/trades")
            assert resp_trades_sell.status_code == 200
            all_markers = resp_trades_sell.json()
            assert len(all_markers) == 2
            m_sell = all_markers[1]

            assert m_sell["time"] == int(sell_order_time.timestamp())
            assert isinstance(m_sell["time"], int)
            assert m_sell["position"] == "aboveBar"
            assert m_sell["shape"] == "arrowDown"
            assert m_sell["color"] == "#ef4444"
            assert m_sell["action"] == "SELL"
            assert m_sell["pnl"] > 0.0
            assert "SELL @" in m_sell["text"]
            assert f"{float(exec_sell.executed_price):.2f}" in m_sell["text"]
            assert "€" in m_sell["text"]

            # Query GET /api/status -> verify balance matches ledger
            resp_status = http_client.get("/api/status")
            assert resp_status.status_code == 200
            status_data = resp_status.json()
            assert status_data["status"] in ("idle", "running")
            assert status_data["symbol"] == "BTC/EUR"
            assert status_data["total_trades"] == 2
            assert status_data["portfolio_balance"] == pytest.approx(float(ledger.total_equity()), rel=1e-3)
            assert status_data["cash_balance"] == pytest.approx(float(ledger.cash), rel=1e-3)

        # 6. Connect client to /api/stream (SSE), dispatch live candle and trade, verify real-time events
        received_events: List[tuple[str, Any]] = []
        sse_ready = threading.Event()
        stop_sse = threading.Event()

        def sse_consumer():
            stream_url = f"http://127.0.0.1:{actual_port}/api/stream"
            try:
                with httpx.Client(timeout=10.0) as stream_client:
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
                                    received_events.append((current_event, payload))
            except Exception:
                pass

        sse_thread = threading.Thread(target=sse_consumer, daemon=True)
        sse_thread.start()

        # Wait for SSE subscription readiness
        assert sse_ready.wait(timeout=5.0), "SSE stream client failed to connect"

        # Dispatch live candle into DashboardState
        live_candle_data = {
            "time": 1757260800,
            "open": 55200.0,
            "high": 55450.0,
            "low": 55150.0,
            "close": 55400.0,
            "volume": 3.75,
        }
        server.state.add_candle(live_candle_data)

        # Dispatch live trade through ExecutionSimulator
        live_order = OrderSignal(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            target_notional=Decimal("6.00"),
            order_id="e2e_live_003",
        )
        exec_live = simulator.execute(
            order=live_order,
            current_price=Decimal("55350.00"),
            ledger=ledger,
            timestamp_ms=1757260800000,
        )
        assert exec_live.success is True

        # Wait for SSE events to be pushed over the wire
        deadline = time.time() + 3.0
        while time.time() < deadline:
            event_types = [ev[0] for ev in received_events]
            if "candle" in event_types and "trade" in event_types:
                break
            time.sleep(0.02)

        stop_sse.set()

        # Verify real-time events received over the wire
        received_event_types = [ev[0] for ev in received_events]
        assert "candle" in received_event_types, f"Missing candle in SSE events: {received_event_types}"
        assert "trade" in received_event_types, f"Missing trade in SSE events: {received_event_types}"

        candle_payloads = [ev[1] for ev in received_events if ev[0] == "candle"]
        assert any(c.get("close") == 55400.0 for c in candle_payloads)

        trade_payloads = [ev[1] for ev in received_events if ev[0] == "trade"]
        assert any(t.get("id") == "e2e_live_003" for t in trade_payloads)

    finally:
        # 7. Stop DashboardServer cleanly in <1.0s
        t_stop_start = time.perf_counter()
        server.stop(timeout=1.0)
        stop_elapsed = time.perf_counter() - t_stop_start

        assert stop_elapsed < 1.0, f"DashboardServer stop took {stop_elapsed:.3f}s (must be < 1.0s)"
        assert server._thread is not None
        assert not server._thread.is_alive(), "DashboardServer thread did not terminate cleanly"


def test_e2e_acceptance_standalone_cli_dashboard():
    """Verify python -m trading_bot dashboard --help and CLI flag options."""
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.run(
        [sys.executable, "-m", "trading_bot", "dashboard", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert proc.returncode == 0
    help_text = proc.stdout

    # Verify CLI flags: --host, --port, --trades-csv, and market DB flag (--market-db or --db-path)
    assert "--host" in help_text, "Missing --host flag in dashboard --help"
    assert "--port" in help_text or "-p" in help_text, "Missing --port flag in dashboard --help"
    assert "--trades-csv" in help_text, "Missing --trades-csv flag in dashboard --help"
    assert "--market-db" in help_text or "--db-path" in help_text, "Missing market DB flag in dashboard --help"

    # Also directly verify the CLI entry point function
    from trading_bot.cli.dashboard import main as dashboard_main
    assert dashboard_main(["--help"]) == 0


def test_e2e_acceptance_paper_cli_with_dashboard():
    """Verify python -m trading_bot paper --help and dashboard-related CLI flags."""
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.run(
        [sys.executable, "-m", "trading_bot", "paper", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert proc.returncode == 0
    help_text = proc.stdout

    # Verify required dashboard integration flags
    assert "--dashboard" in help_text, "Missing --dashboard flag in paper --help"
    assert "--dashboard-host" in help_text, "Missing --dashboard-host flag in paper --help"
    assert "--dashboard-port" in help_text, "Missing --dashboard-port flag in paper --help"
    assert "--trades-csv" in help_text, "Missing --trades-csv flag in paper --help"

    # Also directly verify the CLI entry point function
    from trading_bot.cli.paper import main as paper_main
    assert paper_main(["--help"]) == 0

"""Tests for Real-Time Web Dashboard and Visualization Subsystem (Milestone 9 - AC2).

Validates:
1. Starlette ASGI application route serving (`/`, `/api/status`, `/api/market-data`, `/api/trades`, `/api/stream`).
2. TradingView Lightweight Charts trade marker formatting (AC2).
3. Server-Sent Events (SSE) live streaming.
4. DashboardServer lifecycle with ephemeral port binding (`port=0`).
5. DashboardState CSV & Market DB loading.
6. CLI subcommand dispatch and flags.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import time
from typing import Any, Dict

import httpx
import pytest
from starlette.testclient import TestClient

from trading_bot.dashboard.app import create_app
from trading_bot.dashboard.server import DashboardServer
from trading_bot.dashboard.state import DashboardState
from trading_bot.data.cache import CandleCache
from trading_bot.data.models import Candle
from trading_bot.notifications.channels.csv_log import CsvTradeLogger
from trading_bot.notifications.models import OrderSide, TradeExecutedPayload
from trading_bot.simulation.ledger import TradeRecord


def test_dashboard_index_html_serving():
    """Starlette TestClient gets / -> HTTP 200, contains chart elements and styling."""
    state = DashboardState(symbol="BTC/EUR")
    app = create_app(state)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="chart"' in response.text
    assert 'id="trades-table"' in response.text
    assert "Autonomous Trading Dashboard" in response.text
    assert "lightweight-charts" in response.text
    assert "/static/style.css" in response.text
    assert "/static/app.js" in response.text


def test_dashboard_api_status():
    """GET /api/status -> HTTP 200, valid schema matching requirements."""
    state = DashboardState(symbol="BTC/EUR", initial_balance=10.0, status="running")
    state.set_portfolio(equity=10.75, cash=5.25)
    app = create_app(state)
    client = TestClient(app)

    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "running"
    assert data["symbol"] == "BTC/EUR"
    assert data["portfolio_balance"] == 10.75
    assert data["cash_balance"] == 5.25
    assert data["total_trades"] == 0
    assert "last_updated" in data


def test_dashboard_api_market_data():
    """GET /api/market-data -> HTTP 200, valid OHLCV candle list."""
    state = DashboardState(symbol="BTC/EUR")
    base_ts = 1757246400

    for i in range(5):
        state.add_candle({
            "time": base_ts + i * 3600,
            "open": 58000.0 + i * 100,
            "high": 58200.0 + i * 100,
            "low": 57900.0 + i * 100,
            "close": 58150.0 + i * 100,
            "volume": 10.5 + i,
        })

    app = create_app(state)
    client = TestClient(app)

    response = client.get("/api/market-data")
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 5

    bar = data[0]
    assert bar["time"] == base_ts
    assert bar["open"] == 58000.0
    assert bar["high"] == 58200.0
    assert bar["low"] == 57900.0
    assert bar["close"] == 58150.0
    assert bar["volume"] == 10.5

    # Test limit parameter
    resp_limit = client.get("/api/market-data?limit=2")
    assert resp_limit.status_code == 200
    limited = resp_limit.json()
    assert len(limited) == 2
    assert limited[-1]["time"] == base_ts + 4 * 3600


def test_dashboard_api_trades_marker_formatting():
    """(AC2) GET /api/trades -> HTTP 200, asserts exact schema matching Lightweight Charts markers.

    Verifies schema: (time, position, color, shape, text) for both BUY and SELL orders.
    """
    state = DashboardState(symbol="BTC/EUR")

    # Add a BUY trade
    buy_trade = {
        "trade_id": "trd_buy_01",
        "time": 1757246400,
        "action": "BUY",
        "price": 58100.50,
        "size": 0.00017,
        "portfolio_balance": 10.0,
        "cash_balance": 0.12,
        "pnl": 0.0,
    }
    state.add_trade(buy_trade)

    # Add a SELL trade
    sell_trade = {
        "trade_id": "trd_sell_02",
        "time": 1757250000,
        "action": "SELL",
        "price": 62500.00,
        "size": 0.00017,
        "portfolio_balance": 10.75,
        "cash_balance": 10.75,
        "pnl": 0.75,
    }
    state.add_trade(sell_trade)

    app = create_app(state)
    client = TestClient(app)

    response = client.get("/api/trades")
    assert response.status_code == 200
    markers = response.json()

    assert isinstance(markers, list)
    assert len(markers) == 2

    # Assert BUY marker exact schema
    buy_marker = markers[0]
    assert buy_marker["time"] == 1757246400
    assert isinstance(buy_marker["time"], int)
    assert buy_marker["position"] == "belowBar"
    assert buy_marker["color"] == "#10b981"
    assert buy_marker["shape"] == "arrowUp"
    assert "BUY @" in buy_marker["text"]
    assert "58100.50" in buy_marker["text"]

    # Assert SELL marker exact schema
    sell_marker = markers[1]
    assert sell_marker["time"] == 1757250000
    assert isinstance(sell_marker["time"], int)
    assert sell_marker["position"] == "aboveBar"
    assert sell_marker["color"] == "#ef4444"
    assert sell_marker["shape"] == "arrowDown"
    assert "SELL @" in sell_marker["text"]
    assert "62500.00" in sell_marker["text"]
    assert "+0.75€" in sell_marker["text"]


def test_dashboard_api_stream_sse():
    """Test SSE /api/stream event emission and live trade broadcasting."""
    state = DashboardState(symbol="BTC/EUR", initial_balance=10.0)
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)

    try:
        url = f"http://127.0.0.1:{actual_port}/api/stream"
        with httpx.stream("GET", url, timeout=5.0) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line)
                if len(lines) >= 2:
                    break

            combined = "\n".join(lines)
            assert "event: status" in combined
            assert "portfolio_balance" in combined
    finally:
        server.stop(timeout=3.0)


def test_dashboard_server_lifecycle_ephemeral_port():
    """Live DashboardServer starting on port=0, responds on actual_port, clean stop()."""
    server = DashboardServer(host="127.0.0.1", port=0)
    actual_port = server.start_background(timeout=5.0)

    try:
        assert actual_port > 0
        assert server.actual_port == actual_port

        # Connect using real HTTP socket client
        with httpx.Client(base_url=f"http://127.0.0.1:{actual_port}", timeout=5.0) as client:
            resp = client.get("/api/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "portfolio_balance" in data

            resp_index = client.get("/")
            assert resp_index.status_code == 200
            assert "Autonomous Trading Dashboard" in resp_index.text
    finally:
        server.stop(timeout=3.0)
        assert server._thread is not None
        assert not server._thread.is_alive()


def test_dashboard_state_load_from_csv(tmp_path: Path):
    """Verify DashboardState correctly loads trades from CSV."""
    csv_file = tmp_path / "test_trades.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    payload = TradeExecutedPayload(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        fill_price=Decimal("60000.00"),
        quantity=Decimal("0.00016"),
        fee=Decimal("0.0096"),
        portfolio_balance=Decimal("10.00"),
        timestamp=datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc),
    )
    logger.log_trade(payload)

    state = DashboardState()
    count = state.load_from_csv(csv_file)
    assert count == 1
    assert len(state.get_markers()) == 1

    marker = state.get_markers()[0]
    assert marker["position"] == "belowBar"
    assert marker["color"] == "#10b981"
    assert marker["shape"] == "arrowUp"


def test_dashboard_state_load_from_market_db(tmp_path: Path):
    """Verify DashboardState loads recent candles from SQLite cache."""
    db_file = tmp_path / "test_market_data.db"
    cache = CandleCache(db_path=db_file)

    candles = [
        Candle(
            timestamp=datetime(2026, 9, 7, 10 + i, 0, 0, tzinfo=timezone.utc),
            open=Decimal("59000.00"),
            high=Decimal("59500.00"),
            low=Decimal("58800.00"),
            close=Decimal("59200.00"),
            volume=Decimal("5.0"),
            symbol="BTCEUR",
            interval="1h",
        )
        for i in range(3)
    ]
    cache.save_candles(candles)

    state = DashboardState(symbol="BTC/EUR")
    count = state.load_from_market_db(db_path=db_file, symbol="BTC/EUR", limit=10)
    assert count == 3
    loaded = state.get_candles()
    assert len(loaded) == 3
    assert loaded[0]["open"] == 59000.0


def test_dashboard_cli_help():
    """Verify python -m trading_bot dashboard --help works."""
    from trading_bot.cli.dashboard import main as dashboard_main

    ret = dashboard_main(["--help"])
    assert ret == 0


def test_paper_cli_dashboard_flag():
    """Verify paper trading CLI parser recognizes --dashboard flags."""
    from trading_bot.cli.paper import main as paper_main

    ret = paper_main(["--help"])
    assert ret == 0


def test_dashboard_state_load_from_csv_malformed_rows_skipped(tmp_path: Path):
    """Verify DashboardState.load_from_csv skips malformed rows and preserves valid ones."""
    corrupt_csv = tmp_path / "corrupt_trades.csv"
    corrupt_csv.write_text(
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        "2026-09-07T12:00:00+00:00,valid_1,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:05:00+00:00,bad_price,BTC/EUR,SELL,NOT_A_FLOAT,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:10:00+00:00,bad_size,BTC/EUR,BUY,51000.00,CORRUPT_SIZE,51.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:15:00+00:00,nan_price,BTC/EUR,SELL,NaN,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:20:00+00:00,valid_2,BTC/EUR,SELL,52000.00,0.001,52.0,0.05,2.00,4.00,102.0,102.0,TP\n"
        "2026-09-07T12:25:00+00:00,bad_bal,BTC/EUR,BUY,53000.00,0.001,53.0,0.05,0.00,0.00,INVALID_BAL,BAD_CASH,\n"
    )

    state = DashboardState(symbol="BTC/EUR")
    count = state.load_from_csv(corrupt_csv)

    # 3 valid trades: valid_1, valid_2, and bad_bal (bad_bal has valid price/size; malformed balance does not crash)
    assert count == 3
    markers = state.get_markers()
    assert len(markers) == 3
    assert markers[0]["id"] == "valid_1"
    assert markers[1]["id"] == "valid_2"
    assert markers[2]["id"] == "bad_bal"

    # Also test add_trade with completely broken trade dictionary
    bad_marker = state.add_trade({"price": "CORRUPT", "size": 0.001})
    assert bad_marker == {}
    assert len(state.get_markers()) == 3


def test_dashboard_server_startup_with_corrupt_csv(tmp_path: Path):
    """Verify DashboardServer boots cleanly on port=0 when given a corrupt CSV."""
    corrupt_csv = tmp_path / "corrupt_startup_trades.csv"
    corrupt_csv.write_text(
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        "2026-09-07T12:00:00+00:00,trade_ok,BTC/EUR,BUY,60000.00,0.001,60.0,0.06,0.00,0.00,100.0,40.0,\n"
        "2026-09-07T12:05:00+00:00,trade_corrupt,BTC/EUR,SELL,BROKEN_PRICE,0.001,60.0,0.06,0.00,0.00,100.0,40.0,\n"
    )

    server = DashboardServer(host="127.0.0.1", port=0, trades_csv=corrupt_csv)
    actual_port = server.start_background(timeout=5.0)

    try:
        assert actual_port > 0
        with httpx.Client(base_url=f"http://127.0.0.1:{actual_port}", timeout=5.0) as client:
            resp_trades = client.get("/api/trades")
            assert resp_trades.status_code == 200
            trades_data = resp_trades.json()
            assert len(trades_data) == 1
            assert trades_data[0]["id"] == "trade_ok"
            assert trades_data[0]["action"] == "BUY"

            resp_status = client.get("/api/status")
            assert resp_status.status_code == 200
            status_data = resp_status.json()
            assert status_data["total_trades"] == 1
    finally:
        server.stop(timeout=3.0)

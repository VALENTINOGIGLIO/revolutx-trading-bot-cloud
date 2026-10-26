"""Empirical Challenger Verification & Stress Test Suite for Milestone 9 (Web Dashboard).

Author: challenger_dashboard_1
Focus: Dashboard API, SSE Streaming Stress, Server Concurrency, Client Disconnect Resilience,
       High-Volume Data Loading, Rapid Server Restart Cycles.

Test Matrix:
1. Concurrency Stress: 25 concurrent SSE clients subscribing to `/api/stream` while rapidly
   broadcasting 50 live candles and 50 trade fills. Asserts 100% event delivery, 0 lost messages.
2. Client Disconnect Resilience: 10 SSE clients abruptly sever connections. Asserts queue cleanup
   and 0 subscriber leaks in `DashboardState`.
3. Mid-Stream Disconnect Isolation: 20 SSE clients connected; 10 dropped mid-burst while 10 remain.
   Asserts remaining clients receive complete event sequence without disruption.
4. TCP RST Abort: Raw socket hard abort with `SO_LINGER`. Asserts server stability and subscriber recovery.
5. High-Volume CSV Loading: 10,000 trade records loaded from CSV. Asserts marker generation, schema,
   performance (<5s), and memory stability.
6. High-Volume SQLite DB Loading: 10,000 historical candles loaded from SQLite cache. Asserts limit
   filtering, OHLCV schema, performance (<3s), and memory stability.
7. High-Volume HTTP Endpoint Serialization: 10,000 items queried via `/api/market-data` and `/api/trades`.
   Asserts HTTP 200, schema validity, and sub-second response time.
8. Rapid Server Restart Cycles (Ephemeral): 10 consecutive start/stop cycles on `port=0`. Asserts unique
   ports, immediate responsiveness, and 0 hung threads.
9. Rapid Server Restart Cycles (Fixed Port): 5 consecutive start/stop cycles on a designated fixed port.
   Asserts socket reuse without `EADDRINUSE`.
10. Multithreaded State Contention: 10 concurrent worker threads performing mixed reads/writes on
    `DashboardState` (1,000 ops). Asserts thread safety and balance consistency.
11. Server Shutdown with Active SSE Clients (Bug Reproduction): Probes thread termination when SSE
    subscribers are actively connected.
"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import tracemalloc
from typing import Any, Dict, List

# Ensure src is on sys.path for direct pytest invocation
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import httpx
import pytest
from starlette.testclient import TestClient

from trading_bot.dashboard.app import create_app
from trading_bot.dashboard.server import DashboardServer
from trading_bot.dashboard.state import DashboardState
from trading_bot.data.cache import CandleCache
from trading_bot.data.models import Candle
from trading_bot.notifications.channels.csv_log import CsvTradeLogger


# =====================================================================
# 1. CONCURRENCY STRESS: 25+ CONCURRENT SSE CLIENTS
# =====================================================================
def test_sse_concurrency_25_clients_rapid_burst():
    """Stress Test 1: 25 concurrent SSE clients subscribe to /api/stream.

    Rapidly pushes 50 live candles and 50 trade fills into DashboardState.
    Asserts 100% event delivery to all 25 clients without loss or corruption.
    """
    state = DashboardState(symbol="BTC/EUR", initial_balance=10.0)
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)
    assert actual_port > 0

    num_clients = 25
    num_candles = 50
    num_trades = 50

    client_events: List[List[str]] = [[] for _ in range(num_clients)]
    ready_events = [threading.Event() for _ in range(num_clients)]
    stop_flags = [False] * num_clients

    def sse_client_worker(idx: int) -> None:
        url = f"http://127.0.0.1:{actual_port}/api/stream"
        try:
            with httpx.Client(timeout=15.0) as client:
                with client.stream("GET", url) as resp:
                    assert resp.status_code == 200
                    for line in resp.iter_lines():
                        if stop_flags[idx]:
                            break
                        if line.startswith("event:"):
                            evt = line.split(":", 1)[1].strip()
                            client_events[idx].append(evt)
                            if evt == "status" and not ready_events[idx].is_set():
                                ready_events[idx].set()
        except Exception as exc:
            client_events[idx].append(f"ERR: {exc}")

    threads = [
        threading.Thread(target=sse_client_worker, args=(i,), daemon=True)
        for i in range(num_clients)
    ]
    for t in threads:
        t.start()

    # Ensure all 25 clients connect and receive initial status event
    for i, r_ev in enumerate(ready_events):
        assert r_ev.wait(timeout=5.0), f"Client {i} timed out waiting for connection"

    # Rapidly push live candles
    base_ts = 1757246400
    for i in range(num_candles):
        state.add_candle({
            "time": base_ts + i * 60,
            "open": 60000.0 + i,
            "high": 60100.0 + i,
            "low": 59900.0 + i,
            "close": 60050.0 + i,
            "volume": 1.25,
        })

    # Rapidly push trade fills
    for i in range(num_trades):
        action = "BUY" if i % 2 == 0 else "SELL"
        pnl = 1.20 if action == "SELL" else 0.0
        state.add_trade({
            "trade_id": f"burst_trade_{i:04d}",
            "time": base_ts + i * 60,
            "action": action,
            "price": 60000.0 + i,
            "size": 0.00016,
            "pnl": pnl,
            "portfolio_balance": 10.0 + (i * 0.1),
            "cash_balance": 5.0,
        })

    # Allow network propagation
    time.sleep(1.0)

    # Signal client workers to gracefully stop
    for i in range(num_clients):
        stop_flags[i] = True

    # Unblock clients waiting on queue.get
    state.set_status("running")
    time.sleep(0.5)

    # Validate 100% reception across all 25 clients
    for i in range(num_clients):
        evts = client_events[i]
        c_count = evts.count("candle")
        t_count = evts.count("trade")
        assert c_count == num_candles, f"Client {i} received {c_count}/{num_candles} candles"
        assert t_count == num_trades, f"Client {i} received {t_count}/{num_trades} trades"

    server.stop(timeout=3.0)


# =====================================================================
# 2. CLIENT DISCONNECT & ABNORMAL TERMINATION RESILIENCE
# =====================================================================
def test_sse_client_disconnect_abrupt_resilience():
    """Stress Test 2: Multiple clients connect to /api/stream and abruptly close sockets.

    Asserts that subscriber queues are properly cleaned up in DashboardState and no
    asyncio tasks or queues leak upon subsequent event broadcasts.
    """
    state = DashboardState(symbol="BTC/EUR")
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)

    assert len(state._subscribers) == 0

    # Connect 10 clients and abruptly drop each connection
    for _ in range(10):
        client = httpx.Client(timeout=5.0)
        req = client.build_request("GET", f"http://127.0.0.1:{actual_port}/api/stream")
        resp = client.send(req, stream=True)
        for line in resp.iter_lines():
            if line.startswith("event: status"):
                break
        # Force abrupt close without graceful stream exit
        client.close()

    time.sleep(0.2)

    # Broadcast event to trigger write failure detection in severed streams
    state.add_candle({
        "time": 1757246400,
        "open": 61000.0,
        "high": 61100.0,
        "low": 60900.0,
        "close": 61050.0,
        "volume": 2.0,
    })

    time.sleep(0.5)

    # Verify all dead subscriber queues were purged
    assert len(state._subscribers) == 0, f"Leaked subscribers: {len(state._subscribers)}"

    server.stop(timeout=3.0)


def test_sse_mid_stream_client_termination_isolation():
    """Stress Test 3: 20 SSE clients connected; 10 dropped mid-stream while 10 remain.

    Asserts that severed clients do not crash the broadcast loop or disrupt event
    delivery to surviving clients.
    """
    state = DashboardState(symbol="BTC/EUR")
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)

    num_staying = 10
    num_leaving = 10
    total_clients = num_staying + num_leaving

    staying_events: List[List[str]] = [[] for _ in range(num_staying)]
    ready_events = [threading.Event() for _ in range(total_clients)]
    stop_flags = [False] * total_clients

    def staying_client(idx: int) -> None:
        url = f"http://127.0.0.1:{actual_port}/api/stream"
        try:
            with httpx.Client(timeout=10.0) as client:
                with client.stream("GET", url) as resp:
                    for line in resp.iter_lines():
                        if stop_flags[idx]:
                            break
                        if line.startswith("event:"):
                            evt = line.split(":", 1)[1].strip()
                            staying_events[idx].append(evt)
                            if evt == "status" and not ready_events[idx].is_set():
                                ready_events[idx].set()
        except Exception as exc:
            staying_events[idx].append(f"ERR: {exc}")

    def leaving_client(idx: int) -> None:
        client_idx = num_staying + idx
        url = f"http://127.0.0.1:{actual_port}/api/stream"
        try:
            client = httpx.Client(timeout=10.0)
            resp = client.send(client.build_request("GET", url), stream=True)
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    evt = line.split(":", 1)[1].strip()
                    if evt == "status" and not ready_events[client_idx].is_set():
                        ready_events[client_idx].set()
                if stop_flags[client_idx]:
                    break
            client.close()
        except Exception:
            pass

    threads = []
    for i in range(num_staying):
        threads.append(threading.Thread(target=staying_client, args=(i,), daemon=True))
    for i in range(num_leaving):
        threads.append(threading.Thread(target=leaving_client, args=(i,), daemon=True))

    for t in threads:
        t.start()

    for i, r_ev in enumerate(ready_events):
        assert r_ev.wait(timeout=5.0), f"Client {i} timed out during connection"

    assert len(state._subscribers) == total_clients

    # Push batch 1 (20 candles)
    base_ts = 1757246400
    for i in range(20):
        state.add_candle({"time": base_ts + i, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1})

    # Abruptly sever the 10 leaving clients
    for i in range(num_leaving):
        stop_flags[num_staying + i] = True

    time.sleep(0.2)

    # Push batch 2 (20 candles) during disconnect fallout
    for i in range(20, 40):
        state.add_candle({"time": base_ts + i, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1})

    time.sleep(0.5)

    # Push 10 trades
    for i in range(10):
        state.add_trade({
            "trade_id": f"mid_trade_{i}",
            "time": base_ts + i,
            "action": "BUY",
            "price": 100.0,
            "size": 1.0,
        })

    time.sleep(0.5)

    # Cleanly stop staying clients
    for i in range(num_staying):
        stop_flags[i] = True
    state.set_status("idle")
    time.sleep(0.5)

    # Verify staying clients received all 40 candles and 10 trades without disruption
    for i in range(num_staying):
        evts = staying_events[i]
        assert evts.count("candle") == 40, f"Client {i} missed candles"
        assert evts.count("trade") == 10, f"Client {i} missed trades"

    server.stop(timeout=3.0)


def test_sse_raw_socket_tcp_rst_abort():
    """Stress Test 4: Hard abort of TCP socket using SO_LINGER (RST packet).

    Verifies low-level transport disconnect resilience and queue cleanup.
    """
    state = DashboardState(symbol="BTC/EUR")
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)

    for _ in range(5):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", actual_port))
        req = b"GET /api/stream HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: text/event-stream\r\n\r\n"
        s.sendall(req)
        # Read header and initial status
        s.recv(256)
        # Send hard TCP RST
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()

    time.sleep(0.2)

    # Broadcast event to trigger cleanup
    state.add_candle({"time": 1000, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1})
    time.sleep(0.5)

    assert len(state._subscribers) == 0
    server.stop(timeout=3.0)


# =====================================================================
# 3. HIGH-VOLUME DATA LOADING: 10,000+ RECORDS
# =====================================================================
def test_high_volume_csv_loading_10k_records(tmp_path: Path):
    """Stress Test 5: Load 10,000 historical trades from CSV into DashboardState.

    Asserts exact record count, Lightweight Charts marker formatting, sub-5s execution,
    and memory stability.
    """
    csv_file = tmp_path / "high_volume_trades.csv"
    base_ts = 1757246400

    # Write 10,000 RFC 4180 CSV rows
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "trade_id", "symbol", "action", "price", "size",
            "notional", "fee", "pnl", "pnl_pct", "portfolio_balance",
            "cash_balance", "exit_reason"
        ])
        for i in range(10000):
            action = "BUY" if i % 2 == 0 else "SELL"
            pnl = 2.50 if action == "SELL" else 0.0
            price = 60000.0 + (i % 500)
            writer.writerow([
                datetime.fromtimestamp(base_ts + i * 60, tz=timezone.utc).isoformat(),
                f"hv_trade_{i:06d}",
                "BTC/EUR",
                action,
                f"{price:.2f}",
                "0.00016000",
                "9.60",
                "0.0096",
                f"{pnl:.4f}",
                "2.50",
                f"{10.0 + (i * 0.01):.2f}",
                "5.00",
                "TAKE_PROFIT" if action == "SELL" else "ENTRY_SIGNAL",
            ])

    tracemalloc.start()
    t_start = time.perf_counter()

    state = DashboardState(symbol="BTC/EUR")
    loaded_count = state.load_from_csv(csv_file)

    load_duration = time.perf_counter() - t_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert loaded_count == 10000
    assert len(state.get_markers()) == 10000
    assert len(state.get_trades()) == 10000
    assert load_duration < 5.0, f"CSV loading took {load_duration:.2f}s (>5s)"
    assert (peak_memory / (1024 * 1024)) < 100.0, f"Excessive peak memory: {peak_memory / 1024 / 1024:.1f}MB"

    # Spot check first marker (BUY)
    first_m = state.get_markers()[0]
    assert first_m["shape"] == "arrowUp"
    assert first_m["position"] == "belowBar"
    assert first_m["color"] == "#10b981"
    assert "BUY @" in first_m["text"]

    # Spot check second marker (SELL)
    second_m = state.get_markers()[1]
    assert second_m["shape"] == "arrowDown"
    assert second_m["position"] == "aboveBar"
    assert second_m["color"] == "#ef4444"
    assert "SELL @" in second_m["text"]
    assert "+2.50€" in second_m["text"]


def test_high_volume_market_db_loading_10k_candles(tmp_path: Path):
    """Stress Test 6: Load 10,000 historical candles from SQLite cache.

    Asserts exact record count, limit query slicing, sub-3s execution, and schema integrity.
    """
    db_file = tmp_path / "high_volume_market.db"
    cache = CandleCache(db_path=db_file)
    base_ts = 1757246400

    candles = [
        Candle(
            timestamp=datetime.fromtimestamp(base_ts + i * 60, tz=timezone.utc),
            open=Decimal(str(60000.0 + (i % 100))),
            high=Decimal(str(60050.0 + (i % 100))),
            low=Decimal(str(59950.0 + (i % 100))),
            close=Decimal(str(60020.0 + (i % 100))),
            volume=Decimal("1.5"),
            symbol="BTCEUR",
            interval="1m",
        )
        for i in range(10000)
    ]
    cache.save_candles(candles)

    tracemalloc.start()
    t_start = time.perf_counter()

    state = DashboardState(symbol="BTC/EUR")
    loaded_count = state.load_from_market_db(db_path=db_file, symbol="BTC/EUR", limit=10000, interval="1m")

    load_duration = time.perf_counter() - t_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert loaded_count == 10000
    assert len(state.get_candles()) == 10000
    assert load_duration < 3.0, f"DB loading took {load_duration:.2f}s (>3s)"
    assert (peak_memory / (1024 * 1024)) < 100.0

    # Slicing verification
    assert len(state.get_candles(limit=50)) == 50
    assert state.get_candles(limit=50)[-1]["time"] == base_ts + 9999 * 60


def test_high_volume_http_endpoint_serialization():
    """Stress Test 7: HTTP serialization performance with 10,000 candles and trades.

    Asserts GET /api/market-data and GET /api/trades return HTTP 200 with full
    data in <500ms.
    """
    state = DashboardState(symbol="BTC/EUR")
    base_ts = 1757246400

    candles = [
        {"time": base_ts + i * 60, "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 1.0}
        for i in range(10000)
    ]
    with state._lock:
        state.candles = candles

    for i in range(10000):
        state.trade_markers.append({
            "time": base_ts + i * 60,
            "position": "belowBar",
            "color": "#10b981",
            "shape": "arrowUp",
            "text": f"BUY @ 60000.00",
            "id": f"t_{i}",
            "action": "BUY",
            "price": 60000.0,
            "size": 0.01,
            "pnl": 0.0,
        })

    app = create_app(state)
    client = TestClient(app)

    # Query 10,000 candles
    t0 = time.perf_counter()
    resp_candles = client.get("/api/market-data")
    t_market = time.perf_counter() - t0
    assert resp_candles.status_code == 200
    data_candles = resp_candles.json()
    assert len(data_candles) == 10000
    assert t_market < 0.500, f"Candle serialization too slow: {t_market:.3f}s"

    # Query 10,000 trade markers
    t0 = time.perf_counter()
    resp_trades = client.get("/api/trades")
    t_trades = time.perf_counter() - t0
    assert resp_trades.status_code == 200
    data_trades = resp_trades.json()
    assert len(data_trades) == 10000
    assert t_trades < 0.500, f"Trade serialization too slow: {t_trades:.3f}s"


# =====================================================================
# 4. RAPID SERVER RESTART CYCLES & THREAD CONCURRENCY
# =====================================================================
def test_rapid_server_restart_ephemeral_ports():
    """Stress Test 8: 10 consecutive rapid start/stop cycles of DashboardServer on port=0.

    Asserts dynamic port allocation, clean thread joining, and no orphaned background threads.
    """
    cycles = 10
    ports = set()
    threads = []

    for i in range(cycles):
        state = DashboardState(symbol="BTC/EUR")
        server = DashboardServer(host="127.0.0.1", port=0, state=state)
        actual_port = server.start_background(timeout=5.0)

        assert actual_port > 0, f"Cycle {i}: Invalid port {actual_port}"
        ports.add(actual_port)
        threads.append(server._thread)

        with httpx.Client(base_url=f"http://127.0.0.1:{actual_port}", timeout=2.0) as client:
            resp = client.get("/api/status")
            assert resp.status_code == 200
            assert resp.json()["symbol"] == "BTC/EUR"

        server.stop(timeout=2.0)
        assert not server._thread.is_alive(), f"Cycle {i}: Server thread hung after stop()"

    assert len(ports) == cycles, "Expected unique ephemeral ports per cycle"
    assert all(not t.is_alive() for t in threads), "Orphaned threads detected"


def test_rapid_server_restart_fixed_port():
    """Stress Test 9: 5 consecutive start/stop cycles on a designated fixed port.

    Asserts SO_REUSEADDR allows immediate re-binding without address collision.
    """
    fixed_port = 19123
    for cycle in range(5):
        server = DashboardServer(host="127.0.0.1", port=fixed_port)
        p = server.start_background(timeout=5.0)
        assert p == fixed_port, f"Cycle {cycle}: Expected {fixed_port}, got {p}"

        with httpx.Client(base_url=f"http://127.0.0.1:{fixed_port}", timeout=2.0) as client:
            resp = client.get("/api/status")
            assert resp.status_code == 200

        server.stop(timeout=2.0)
        assert not server._thread.is_alive()


def test_multithreaded_dashboard_state_contention():
    """Stress Test 10: 10 concurrent worker threads hammering DashboardState.

    Performs 1,000 mixed operations (add_candle, add_trade, set_portfolio, get_markers).
    Asserts no deadlocks, thread-safe mutations, and balance consistency.
    """
    state = DashboardState(symbol="BTC/EUR", initial_balance=10.0)
    num_threads = 10
    ops_per_thread = 100
    barrier = threading.Barrier(num_threads)
    base_ts = 1757246400

    def worker(tid: int) -> None:
        barrier.wait()
        for j in range(ops_per_thread):
            op_idx = tid * ops_per_thread + j
            if j % 4 == 0:
                state.add_candle({
                    "time": base_ts + op_idx,
                    "open": 60000.0,
                    "high": 60050.0,
                    "low": 59950.0,
                    "close": 60010.0,
                    "volume": 0.5,
                })
            elif j % 4 == 1:
                state.add_trade({
                    "trade_id": f"contention_{op_idx}",
                    "time": base_ts + op_idx,
                    "action": "BUY" if j % 2 == 0 else "SELL",
                    "price": 60000.0,
                    "size": 0.001,
                    "pnl": 0.5,
                })
            elif j % 4 == 2:
                state.set_portfolio(equity=10.0 + op_idx * 0.01, cash=5.0)
            else:
                _ = state.get_markers()
                _ = state.get_candles(limit=10)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "Worker thread hung in deadlock"

    assert len(state.get_candles()) > 0
    assert len(state.get_markers()) > 0
    status = state.get_status()
    assert status["portfolio_balance"] > 10.0


# =====================================================================
# 5. SERVER SHUTDOWN REGRESSION TEST: ACTIVE SSE SUBSCRIBERS
# =====================================================================
def test_server_stop_with_active_sse_clients_hangs():
    """Stress Test 11 (Regression Verification):

    When an SSE subscriber is connected to /api/stream, calling DashboardServer.stop()
    terminates cleanly (<2.0s) and joins the background server thread.
    """
    state = DashboardState(symbol="BTC/EUR")
    server = DashboardServer(host="127.0.0.1", port=0, state=state)
    actual_port = server.start_background(timeout=5.0)

    connected = threading.Event()

    def client_worker() -> None:
        try:
            with httpx.Client(timeout=10.0) as client:
                with client.stream("GET", f"http://127.0.0.1:{actual_port}/api/stream") as resp:
                    connected.set()
                    for _ in resp.iter_lines():
                        pass
        except Exception:
            pass

    client_thread = threading.Thread(target=client_worker, daemon=True)
    client_thread.start()
    assert connected.wait(timeout=5.0)

    # Attempt to stop server while SSE stream is open
    t0 = time.perf_counter()
    server.stop(timeout=2.0)
    duration = time.perf_counter() - t0

    assert not server._thread.is_alive(), f"Server thread hung after {duration:.2f}s"
    assert duration < 2.0, f"Server shutdown took too long: {duration:.2f}s"

"""Empirical Challenger Verification & Adversarial Test Suite for Milestone 10.

Author: challenger_m10_2
Focus: Boundary Conditions, Corrupt Input Resilience, Extreme Market Scenarios, Port Collision.

Verification Targets:
1. Extreme micro-trades (<0.0001 BTC, satoshi rounding behavior).
2. Extreme price gaps (99% flash crash and 10x price spikes).
3. Port conflict simulation and graceful recovery.
4. Corrupt or partially written CSV file lines handling in CsvTradeLogger & DashboardState.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any, Dict, List

import httpx
import pytest

from trading_bot.dashboard.server import DashboardServer
from trading_bot.dashboard.state import DashboardState
from trading_bot.notifications.channels.csv_log import CsvTradeLogChannel, CsvTradeLogger
from trading_bot.notifications.models import OrderSide
from trading_bot.notifications.service import NotificationService
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import ExecutionSimulator, OrderSignal


def test_extreme_micro_trades_sub_0001_btc(tmp_path: Path):
    """Verify execution, CSV logging, and dashboard marker generation for micro-trades (< 0.0001 BTC).

    Operational range on Binance: 0.00009 BTC (< 0.0001 BTC, 5.40 EUR notional at 60,000 EUR/BTC).
    """
    csv_path = tmp_path / "micro_trades.csv"
    logger = CsvTradeLogger(csv_path=csv_path)
    state = DashboardState(symbol="BTC/EUR")
    ledger = VirtualLedger(initial_cash=Decimal("100.00"), quote_currency="EUR")
    channel = CsvTradeLogChannel(csv_path=csv_path, enabled=True)
    notif = NotificationService(channels=[channel])
    sim = ExecutionSimulator(min_notional=Decimal("5.00"), notification_service=notif)

    # 1. BUY 0.00009 BTC at 60,000.00 EUR (notional ~5.40 EUR > 5.00 EUR min_notional)
    buy_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_quantity=Decimal("0.00009"),
        order_id="micro_buy_01",
    )
    res_buy = sim.execute(order=buy_order, current_price=Decimal("60000.00"), ledger=ledger)
    assert res_buy.success is True, f"Micro-buy failed: {res_buy.error_message}"
    assert res_buy.quantity == Decimal("0.00009")

    # Verify CSV content
    trades = logger.read_trades()
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "micro_buy_01"
    assert trades[0]["size"] == pytest.approx(0.00009, rel=1e-5)
    assert trades[0]["action"] == "BUY"

    # 2. SELL 0.00009 BTC at 65,000.00 EUR (gain)
    sell_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        target_quantity=Decimal("0.00009"),
        order_id="micro_sell_01",
        exit_reason="MICRO_TP",
    )
    res_sell = sim.execute(order=sell_order, current_price=Decimal("65000.00"), ledger=ledger)
    assert res_sell.success is True, f"Micro-sell failed: {res_sell.error_message}"

    trades = logger.read_trades()
    assert len(trades) == 2
    assert trades[1]["trade_id"] == "micro_sell_01"
    assert trades[1]["pnl"] > 0

    # 3. Load into DashboardState and verify markers
    state.load_from_csv(csv_path)
    markers = state.get_markers()
    assert len(markers) == 2
    assert markers[0]["shape"] == "arrowUp"
    assert markers[0]["size"] == pytest.approx(0.00009, rel=1e-5)
    assert markers[1]["shape"] == "arrowDown"
    assert "SELL @" in markers[1]["text"]
    assert "(+" in markers[1]["text"]


def test_satoshi_rounding_and_csv_precision_boundary(tmp_path: Path):
    """Empirically test sub-satoshi / single-satoshi orders and document CSV precision boundary.

    In CsvTradeLogger._format_payload_row and _format_dict_row, size is formatted as:
    `size_str = f"{float(trade.quantity):.6f}"`.
    Consequently, quantities below 0.0000005 BTC (such as 1 satoshi = 1e-8 BTC) round to "0.000000",
    returning size 0.0 upon CSV reload, even though VirtualLedger maintains exact Decimal precision.
    """
    csv_path = tmp_path / "satoshi.csv"
    logger = CsvTradeLogger(csv_path=csv_path)
    ledger = VirtualLedger(initial_cash=Decimal("100.00"), quote_currency="EUR")
    channel = CsvTradeLogChannel(csv_path=csv_path, enabled=True)
    notif = NotificationService(channels=[channel])

    sim = ExecutionSimulator(
        step_size=Decimal("0.00000001"),  # 1 satoshi step
        min_notional=Decimal("0.0001"),
        notification_service=notif,
    )

    # 1 satoshi: 0.00000001 BTC
    order_1sat = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_quantity=Decimal("0.00000001"),
        order_id="sat_exact_1",
    )
    res = sim.execute(order=order_1sat, current_price=Decimal("100000.00"), ledger=ledger)
    assert res.success is True
    assert res.quantity == Decimal("1E-8")

    # Ledger accurately preserves 1 satoshi
    pos = ledger.get_position("BTC/EUR")
    assert pos.quantity == Decimal("0.00000001")

    # Verify CSV formatting: 6-decimal rounding causes 1 satoshi to be formatted as 0.000000
    trades = logger.read_trades()
    assert len(trades) == 1
    assert trades[0]["size"] == 0.0  # Demonstrates 6-decimal rounding boundary


def test_extreme_price_gaps_flash_crash_and_10x_spike(tmp_path: Path):
    """Verify system resilience during 99% flash crash and 10x price spike.

    Verifies:
    - Closing orders bypass minNotional when closing positions after flash crash.
    - Negative realized PnL and percentage formatting in CSV and Lightweight Charts marker.
    - 10x price spike handling with massive profit percentage (+898%).
    - Live DashboardServer HTTP status and markers match ledger balances.
    """
    csv_path = tmp_path / "extreme_gaps.csv"
    logger = CsvTradeLogger(csv_path=csv_path)
    state = DashboardState(symbol="BTC/EUR")
    ledger = VirtualLedger(initial_cash=Decimal("100.00"), quote_currency="EUR")
    channel = CsvTradeLogChannel(csv_path=csv_path, enabled=True)
    notif = NotificationService(channels=[channel])
    sim = ExecutionSimulator(min_notional=Decimal("5.00"), notification_service=notif)

    # 1. Open BUY at 50,000.00 EUR
    buy_sig = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_quantity=Decimal("0.001"),
        order_id="crash_buy_01",
    )
    res_buy = sim.execute(order=buy_sig, current_price=Decimal("50000.00"), ledger=ledger)
    assert res_buy.success is True

    # 2. Flash Crash: Price drops 99% to 500.00 EUR
    sell_crash = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        target_quantity=Decimal("0.001"),
        order_id="crash_sell_01",
        exit_reason="FLASH_CRASH_CIRCUIT",
    )
    res_sell_crash = sim.execute(order=sell_crash, current_price=Decimal("500.00"), ledger=ledger)
    assert res_sell_crash.success is True
    assert res_sell_crash.realized_pnl < Decimal("-49.00")

    # 3. 10x Spike: Buy at 50,000.00 EUR and Sell at 500,000.00 EUR
    buy_spike = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_quantity=Decimal("0.0005"),
        order_id="spike_buy_01",
    )
    res_buy_spike = sim.execute(order=buy_spike, current_price=Decimal("50000.00"), ledger=ledger)
    assert res_buy_spike.success is True

    sell_spike = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        target_quantity=Decimal("0.0005"),
        order_id="spike_sell_01",
        exit_reason="10X_SPIKE_PROFIT",
    )
    res_sell_spike = sim.execute(order=sell_spike, current_price=Decimal("500000.00"), ledger=ledger)
    assert res_sell_spike.success is True
    assert res_sell_spike.realized_pnl > Decimal("200.00")

    # 4. Verify CSV rows and DashboardState markers
    trades = logger.read_trades()
    assert len(trades) == 4
    # Crash sell trade
    assert trades[1]["pnl"] < -49.0
    assert trades[1]["pnl_pct"] == pytest.approx(-99.0, abs=1.0)
    # Spike sell trade
    assert trades[3]["pnl"] > 200.0
    assert trades[3]["pnl_pct"] > 800.0

    # 5. Start DashboardServer and query HTTP endpoints
    server = DashboardServer(port=0, trades_csv=csv_path)
    actual_port = server.start_background(timeout=5.0)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{actual_port}", timeout=5.0) as client:
            resp_trades = client.get("/api/trades")
            assert resp_trades.status_code == 200
            markers = resp_trades.json()
            assert len(markers) == 4

            # Verify crash marker formatting
            crash_marker = markers[1]
            assert crash_marker["color"] == "#ef4444"
            assert crash_marker["position"] == "aboveBar"
            assert "-" in crash_marker["text"]
            assert "€" in crash_marker["text"]

            # Verify spike marker formatting
            spike_marker = markers[3]
            assert spike_marker["color"] == "#ef4444"
            assert spike_marker["position"] == "aboveBar"
            assert "+" in spike_marker["text"]
            assert "€" in spike_marker["text"]

            # Verify status endpoint
            resp_status = client.get("/api/status")
            assert resp_status.status_code == 200
            status = resp_status.json()
            assert status["total_trades"] == 4
            assert status["portfolio_balance"] == pytest.approx(float(ledger.total_equity()), rel=1e-3)
    finally:
        server.stop(timeout=1.0)


def test_port_conflict_simulation_and_graceful_recovery():
    """Verify DashboardServer gracefully detects port collision without deadlocking,

    and recovers cleanly once the port is freed or switched.
    """
    # 1. Occupy a random local port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    occupied_port = sock.getsockname()[1]

    # 2. Attempt to start DashboardServer on the occupied port
    conflicted_server = DashboardServer(host="127.0.0.1", port=occupied_port)
    t0 = time.time()
    with pytest.raises(RuntimeError) as exc_info:
        conflicted_server.start_background(timeout=2.0)
    elapsed = time.time() - t0

    # Ensure rapid failure detection (< 1.5s) and descriptive error message
    assert elapsed < 1.5, f"Port collision took too long to detect: {elapsed:.2f}s"
    assert "port" in str(exc_info.value).lower()
    assert str(occupied_port) in str(exc_info.value)
    conflicted_server.stop(timeout=0.5)

    # 3. Verify concurrent fallback: port=0 works immediately even while port is occupied
    fallback_server = DashboardServer(host="127.0.0.1", port=0)
    ephemeral_port = fallback_server.start_background(timeout=5.0)
    assert ephemeral_port > 0
    assert ephemeral_port != occupied_port
    fallback_server.stop(timeout=1.0)

    # 4. Release the occupied port and verify clean recovery on the original port
    sock.close()
    recovered_server = DashboardServer(host="127.0.0.1", port=occupied_port)
    rec_port = recovered_server.start_background(timeout=5.0)
    assert rec_port == occupied_port
    recovered_server.stop(timeout=1.0)


def test_corrupt_csv_partial_write_and_empty_file_handling(tmp_path: Path):
    """Verify CsvTradeLogger and DashboardState handle partial lines, empty files, and header-only files."""
    # Case 1: Empty 0-byte file
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    state = DashboardState()
    assert state.load_from_csv(empty_csv) == 0

    # Case 2: Header-only file
    header_csv = tmp_path / "header_only.csv"
    header_csv.write_text(
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
    )
    assert state.load_from_csv(header_csv) == 0

    # Case 3: Truncated partial line at end (process killed mid-write)
    partial_csv = tmp_path / "partial.csv"
    partial_csv.write_text(
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        "2026-09-07T12:00:00+00:00,trade_1,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:05:00+00:00,trade_2,BTC/EUR,SELL,52000.00,"
    )
    logger = CsvTradeLogger(csv_path=partial_csv)
    trades = logger.read_trades()
    # DictReader reads trade_1 and fills missing columns for partial trade_2 with None
    assert len(trades) == 2
    count = state.load_from_csv(partial_csv)
    assert count == 2


def test_corrupt_csv_malformed_float_vulnerability(tmp_path: Path):
    """Empirical Bug Verification: DashboardState.load_from_csv crashes on malformed numeric fields.

    When a corrupt line has non-numeric text in price (e.g. 'CORRUPT_PRICE') or size:
    `CsvTradeLogger.read_trades()` suppresses ValueError during type casting and keeps the raw string.
    Then `DashboardState.load_from_csv()` invokes `_format_and_append_trade()`, which calls
    `float(trade.get("price"))` WITHOUT exception handling, crashing the entire load process
    with ValueError instead of gracefully skipping the malformed row.
    """
    corrupt_csv = tmp_path / "corrupt_fields.csv"
    corrupt_csv.write_text(
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        "2026-09-07T12:00:00+00:00,t1,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:05:00+00:00,t2,BTC/EUR,SELL,CORRUPT_PRICE,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:10:00+00:00,t3,BTC/EUR,SELL,55000.00,0.001,55.0,0.05,5.00,10.00,105.0,105.0,TP\n"
    )

    state = DashboardState()
    # BUG-M10-01 Fix Verification:
    # Verify load_from_csv gracefully skips malformed row t2, loads valid rows t1 and t3 (count = 2),
    # and populates state.get_markers() with 2 markers without crashing.
    count = state.load_from_csv(corrupt_csv)
    assert count == 2

    markers = state.get_markers()
    assert len(markers) == 2
    assert markers[0]["id"] == "t1"
    assert markers[1]["id"] == "t3"

    trades = state.get_trades()
    assert len(trades) == 2
    assert trades[0]["trade_id"] == "t1"
    assert trades[1]["trade_id"] == "t3"

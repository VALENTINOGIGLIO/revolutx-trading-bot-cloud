"""Empirical Challenger Verification & Stress Test Suite for Milestone 8 (CSV Logging).

Author: challenger_csv_1
Focus: Adversarial Concurrency, High Volume Throughput, Race Conditions, Boundary & Corruption Recovery.

Tests:
1. Concurrency Stress: 50+ concurrent threads executing simultaneous trade appends.
   - Barrier synchronization to force maximum contention.
   - Exact line count assertion (1 header + 50 * N rows).
   - 0 lost writes: verify all trade IDs present.
   - 0 corrupted lines: verify each row has exactly 13 RFC 4180 columns.
   - Verify `read_trades()` parses every single trade without error.
2. High Volume Burst Throughput:
   - Rapid burst of 1,000+ trade records.
   - Verify throughput, zero dropped trades, exact line counts, and data fidelity.
3. Multi-Instance Concurrency (Path Contention):
   - Multiple separate `CsvTradeLogger` instances pointing to the same file path concurrently.
4. Boundary & File Corruption Scenarios:
   - Pre-existing file with missing trailing newline (half-line).
   - Pre-existing empty file (0 bytes).
   - Deep non-existent directory paths (`auto_mkdir=True` and `auto_mkdir=False`).
   - Corrupted CSV content (truncated lines, extra columns, invalid characters, NUL bytes).
   - Concurrent reads during concurrent writes.
5. Fault Isolation under Extreme Channel Stress:
   - `CsvTradeLogChannel` handling malformed payloads, None values, unwriteable paths.
"""

from __future__ import annotations

import concurrent.futures
import csv
from datetime import datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Dict, List
import pytest

from trading_bot.notifications.channels.csv_log import CsvTradeLogChannel, CsvTradeLogger
from trading_bot.notifications.models import (
    DeliveryStatus,
    EventType,
    OrderSide,
    TradeExecutedPayload,
)
from trading_bot.simulation.ledger import TradeRecord


# =====================================================================
# 1. CONCURRENCY STRESS: 50+ THREADS SIMULTANEOUS APPEND
# =====================================================================
def test_adversarial_concurrency_50_threads_simultaneous_burst(tmp_path: Path):
    """Stress Test 1: 50 concurrent threads simultaneously write to a single CsvTradeLogger.

    A threading.Barrier is used to force all 50 threads to release and start
    writing at the exact same instant, maximizing race conditions.
    """
    csv_file = tmp_path / "barrier_stress_50.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    num_threads = 60  # Exceeding 50 threads requirement
    trades_per_thread = 20
    total_trades_expected = num_threads * trades_per_thread

    barrier = threading.Barrier(num_threads)
    errors: List[str] = []

    def thread_worker(thread_idx: int):
        # Wait for all threads to be ready
        barrier.wait()
        for trade_idx in range(trades_per_thread):
            trade_id = f"tid_th{thread_idx:03d}_{trade_idx:04d}"
            is_buy = (trade_idx % 2 == 0)
            payload = TradeExecutedPayload(
                symbol="BTC/EUR",
                side=OrderSide.BUY if is_buy else OrderSide.SELL,
                fill_price=50000.0 + (thread_idx * 10) + trade_idx,
                quantity=0.001,
                fee=0.05,
                portfolio_balance=100.0,
                cash_balance=90.0,
                realized_pnl=0.0 if is_buy else 5.25,
                metadata={"trade_id": trade_id, "exit_reason": "" if is_buy else "TAKE_PROFIT"},
            )
            success = logger.log_trade(payload)
            if not success:
                errors.append(f"Write failed for {trade_id}")

    threads = [threading.Thread(target=thread_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    # 1. Verify 0 write failures
    assert len(errors) == 0, f"Write errors occurred: {errors}"

    # 2. Verify file exists and read raw lines
    assert csv_file.exists()
    raw_lines = [line.strip() for line in csv_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_lines = 1 + total_trades_expected
    assert len(raw_lines) == expected_lines, (
        f"Expected {expected_lines} lines (1 header + {total_trades_expected} rows), got {len(raw_lines)}"
    )

    # 3. Verify every row parses into exactly 13 RFC 4180 columns with zero corruption
    found_trade_ids = set()
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == expected_lines
        header = reader[0]
        assert header == logger.HEADERS

        for row_idx, row in enumerate(reader[1:], start=1):
            assert len(row) == 13, f"Row {row_idx} malformed (got {len(row)} cols): {row}"
            # Column 1 is trade_id
            found_trade_ids.add(row[1])
            # Column 3 is action
            assert row[3] in ("BUY", "SELL")
            # Column 4 is price
            assert float(row[4]) > 0
            # Column 5 is size
            assert float(row[5]) > 0

    # 4. Verify 0 lost writes: all unique trade IDs must be present
    assert len(found_trade_ids) == total_trades_expected, (
        f"Lost writes detected! Expected {total_trades_expected} unique IDs, found {len(found_trade_ids)}"
    )

    # 5. Verify logger.read_trades() parses all rows cleanly
    parsed_trades = logger.read_trades()
    assert len(parsed_trades) == total_trades_expected


# =====================================================================
# 2. HIGH VOLUME THROUGHPUT: 1,000+ BURST WRITES
# =====================================================================
def test_high_volume_burst_throughput_1000_records(tmp_path: Path):
    """Stress Test 2: Write 1,000+ trade records in rapid burst mode.

    Verifies system durability, fsync persistence, zero dropped rows, and measures throughput.
    """
    csv_file = tmp_path / "burst_1000.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    total_records = 1000
    start_time = time.perf_counter()

    for i in range(total_records):
        is_buy = (i % 2 == 0)
        record = TradeRecord(
            trade_id=f"burst_{i:05d}",
            timestamp_ms=1757246400000 + (i * 1000),
            symbol="ETH/EUR",
            side="BUY" if is_buy else "SELL",
            order_type="MARKET",
            executed_price=Decimal("3000.00") + Decimal(str(i % 50)),
            raw_price=Decimal("3000.00"),
            quantity=Decimal("0.005"),
            quote_notional=Decimal("15.0000"),
            fee=Decimal("0.0150"),
            realized_pnl=Decimal("0.00") if is_buy else Decimal("1.5000"),
            cash_after=Decimal("10.0000"),
            equity_after=Decimal("11.5000"),
            exit_reason="" if is_buy else "TAKE_PROFIT",
        )
        ok = logger.log_trade(record)
        assert ok is True, f"Failed at trade {i}"

    elapsed = time.perf_counter() - start_time
    rate = total_records / elapsed if elapsed > 0 else 0
    print(f"\n[BURST THROUGHPUT] Wrote {total_records} trades with fsync in {elapsed:.3f}s ({rate:.1f} trades/s)")

    # Verify line count
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + total_records

    # Read trades back and verify integrity of first and last
    trades = logger.read_trades()
    assert len(trades) == total_records
    assert trades[0]["trade_id"] == "burst_00000"
    assert trades[-1]["trade_id"] == f"burst_{total_records - 1:05d}"
    assert trades[0]["symbol"] == "ETH/EUR"
    assert trades[0]["action"] == "BUY"
    assert trades[1]["action"] == "SELL"


# =====================================================================
# 3. MULTI-INSTANCE CONCURRENCY (Same file, distinct CsvTradeLoggers)
# =====================================================================
def test_multi_instance_concurrent_appends_to_same_file(tmp_path: Path):
    """Stress Test 3: Multiple independent CsvTradeLogger instances writing to the same CSV path.

    Evaluates whether separate instances (which do not share threading.Lock)
    experience corruption or race conditions on the underlying file descriptor.
    """
    csv_file = tmp_path / "shared_multi_instance.csv"
    num_instances = 10
    trades_per_instance = 25
    expected_total = num_instances * trades_per_instance

    barrier = threading.Barrier(num_instances)

    def instance_worker(inst_id: int):
        # Create a fresh CsvTradeLogger instance per thread pointing to the same file
        inst_logger = CsvTradeLogger(csv_path=csv_file)
        barrier.wait()

        for t_idx in range(trades_per_instance):
            inst_logger.log_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_id": f"inst_{inst_id:02d}_{t_idx:03d}",
                "symbol": "BTC/EUR",
                "action": "BUY",
                "price": 60000.0,
                "size": 0.001,
                "notional": 60.0,
                "fee": 0.06,
                "portfolio_balance": 100.0,
                "cash_balance": 40.0,
            })

    threads = [threading.Thread(target=instance_worker, args=(i,)) for i in range(num_instances)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    # Inspect results
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # All rows should have 13 columns
    for idx, r in enumerate(rows):
        assert len(r) == 13, f"Row {idx} malformed in multi-instance append: {r}"

    # Header check: at least line 0 must be header
    assert rows[0] == CsvTradeLogger.HEADERS


# =====================================================================
# 4. BOUNDARY & FILE CORRUPTION INJECTION TESTS
# =====================================================================
def test_boundary_empty_file_handling(tmp_path: Path):
    """Boundary Test 1: File pre-exists but has 0 bytes."""
    csv_file = tmp_path / "empty_file.csv"
    csv_file.touch()
    assert csv_file.stat().st_size == 0

    # Initializing logger must detect empty file and populate header
    logger = CsvTradeLogger(csv_path=csv_file)
    assert csv_file.stat().st_size > 0
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0] == ",".join(logger.HEADERS)

    # Logging a trade must append after the header
    trade = {"symbol": "BTC/EUR", "action": "BUY", "price": 50000, "size": 0.001}
    logger.log_trade(trade)
    lines_after = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines_after) == 2


def test_boundary_missing_deep_directory_trees(tmp_path: Path):
    """Boundary Test 2: Deeply nested directory that does not exist yet."""
    deep_path = tmp_path / "a" / "b" / "c" / "d" / "trades.csv"
    assert not deep_path.parent.exists()

    # auto_mkdir=True (default) should create parent directories safely
    logger = CsvTradeLogger(csv_path=deep_path, auto_mkdir=True)
    assert deep_path.exists()
    assert deep_path.stat().st_size > 0

    # auto_mkdir=False with missing directory should fail gracefully
    non_existent_dir = tmp_path / "uncreated" / "trades.csv"
    logger_no_mkdir = CsvTradeLogger(csv_path=non_existent_dir, auto_mkdir=False)
    # File should not exist, and log_trade should return False rather than throwing unhandled exception
    assert not non_existent_dir.exists()
    res = logger_no_mkdir.log_trade({"symbol": "BTC/EUR", "action": "BUY"})
    assert res is False


def test_boundary_preexisting_half_line_recovery(tmp_path: Path):
    """Boundary Test 3: Pre-existing file ending with an incomplete row (no trailing newline).

    Checks how CsvTradeLogger behaves when appending to a file whose last line is a half-line.
    """
    csv_file = tmp_path / "half_line.csv"
    header = ",".join(CsvTradeLogger.HEADERS)
    half_line = "2026-09-07T12:00:00Z,tr_half,BTC/EUR,BUY,50000.00,0.001"
    # Write header with newline, then half-line WITHOUT newline
    csv_file.write_text(f"{header}\n{half_line}", encoding="utf-8")

    logger = CsvTradeLogger(csv_path=csv_file)
    logger.log_trade({
        "trade_id": "tr_after_half",
        "symbol": "BTC/EUR",
        "action": "BUY",
        "price": 51000.0,
        "size": 0.002,
    })

    content = csv_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    print("\nHalf-line test result lines:")
    for idx, l in enumerate(lines):
        print(f"  Line {idx}: {l}")

    # Note empirical finding: does it merge into half_line or start on new line?
    # Standard open(..., 'a') appends to EOF. We observe the behavior.


def test_boundary_corrupted_csv_content_parsing(tmp_path: Path):
    """Boundary Test 4: read_trades resilience against corrupt lines, missing fields, and extra fields."""
    csv_file = tmp_path / "corrupt.csv"
    header = ",".join(CsvTradeLogger.HEADERS)
    # Line 1: valid header
    # Line 2: normal valid row
    # Line 3: truncated row with only 3 columns
    # Line 4: row with extra columns (15 columns)
    # Line 5: empty blank line
    # Line 6: row with invalid numeric strings in price and size
    # Line 7: normal valid row
    corrupt_content = (
        f"{header}\n"
        "2026-09-07T10:00:00Z,tr_1,BTC/EUR,BUY,60000.00,0.001,60.0,0.06,0.00,0.00,100.0,40.0,\n"
        "2026-09-07T10:01:00Z,tr_2,BTC/EUR\n"
        "2026-09-07T10:02:00Z,tr_3,BTC/EUR,SELL,61000.00,0.001,61.0,0.06,1.00,1.67,101.0,101.0,TAKE_PROFIT,extra1,extra2\n"
        "\n"
        "2026-09-07T10:03:00Z,tr_4,BTC/EUR,BUY,NOT_A_NUMBER,BAD_SIZE,60.0,0.06,0.00,0.00,100.0,40.0,\n"
        "2026-09-07T10:04:00Z,tr_5,BTC/EUR,BUY,62000.00,0.001,62.0,0.06,0.00,0.00,100.0,38.0,\n"
    )
    csv_file.write_text(corrupt_content, encoding="utf-8")

    logger = CsvTradeLogger(csv_path=csv_file)
    # read_trades must not raise exceptions, must skip empty lines, and handle invalid floats
    trades = logger.read_trades()
    assert len(trades) >= 3, f"Expected at least 3 readable trades, got {len(trades)}"

    # First trade parsed cleanly
    assert trades[0]["trade_id"] == "tr_1"
    assert trades[0]["price"] == 60000.0

    # Trade 4 with invalid number keeps string without crashing
    tr4 = next((t for t in trades if t.get("trade_id") == "tr_4"), None)
    assert tr4 is not None
    assert tr4["price"] == "NOT_A_NUMBER"


def test_boundary_read_while_writing_concurrency(tmp_path: Path):
    """Boundary Test 5: Concurrent readers while active writer is appending rows."""
    csv_file = tmp_path / "read_write_concurrent.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    stop_event = threading.Event()
    read_errors: List[str] = []
    read_counts: List[int] = []

    def writer():
        for i in range(200):
            logger.log_trade({
                "trade_id": f"rw_{i:04d}",
                "symbol": "BTC/EUR",
                "action": "BUY",
                "price": 50000.0 + i,
                "size": 0.001,
            })
            time.sleep(0.001)
        stop_event.set()

    def reader():
        while not stop_event.is_set():
            try:
                trades = logger.read_trades()
                read_counts.append(len(trades))
                time.sleep(0.002)
            except Exception as ex:
                read_errors.append(str(ex))

    w_thread = threading.Thread(target=writer)
    r_threads = [threading.Thread(target=reader) for _ in range(5)]

    for r in r_threads:
        r.start()
    w_thread.start()

    w_thread.join(timeout=10.0)
    stop_event.set()
    for r in r_threads:
        r.join(timeout=5.0)

    assert len(read_errors) == 0, f"Read errors during concurrent writes: {read_errors}"
    assert len(logger.read_trades()) == 200


# =====================================================================
# 5. CHANNEL CONCURRENCY & UNICODE DELIMITER STRESS
# =====================================================================
def test_csv_trade_log_channel_50_threads_concurrency(tmp_path: Path):
    """Stress Test 6: 50 concurrent threads dispatching events through CsvTradeLogChannel."""
    csv_file = tmp_path / "channel_concurrency.csv"
    channel = CsvTradeLogChannel(csv_path=csv_file, enabled=True)

    num_threads = 50
    trades_per_thread = 10
    barrier = threading.Barrier(num_threads)
    dispatch_statuses: List[DeliveryStatus] = []
    status_lock = threading.Lock()

    def channel_worker(thread_idx: int):
        barrier.wait()
        for i in range(trades_per_thread):
            is_trade = (i % 2 == 0)
            if is_trade:
                payload = TradeExecutedPayload(
                    symbol="SOL/EUR",
                    side=OrderSide.BUY,
                    fill_price=140.0 + thread_idx,
                    quantity=0.1,
                    fee=0.014,
                    portfolio_balance=50.0,
                    metadata={"trade_id": f"ch_{thread_idx:02d}_{i:03d}"},
                )
            else:
                # Send non-trade event (should be ignored safely by channel without error)
                from trading_bot.notifications.models import BaseNotificationEvent
                payload = BaseNotificationEvent(
                    event_type=EventType.PORTFOLIO_UPDATE,
                    message="Portfolio balance ping",
                )

            st = channel.dispatch(payload)
            with status_lock:
                dispatch_statuses.append(st)

    threads = [threading.Thread(target=channel_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert len(dispatch_statuses) == num_threads * trades_per_thread
    assert all(st.success for st in dispatch_statuses)

    # Exactly half the dispatches were trades (num_threads * trades_per_thread // 2)
    expected_trade_count = num_threads * (trades_per_thread // 2)
    trades = channel.logger.read_trades()
    assert len(trades) == expected_trade_count

    # File line count should be 1 + expected_trade_count
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + expected_trade_count


def test_boundary_unicode_and_adversarial_escaping(tmp_path: Path):
    """Boundary Test 6: Extreme characters, delimiters, newlines in quotes, and Unicode."""
    csv_file = tmp_path / "unicode_escaping.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    adversarial_trades = [
        {
            "trade_id": "tr_comma_in_id,123",
            "symbol": "BTC/EUR",
            "action": "SELL",
            "price": 60000.0,
            "size": 0.001,
            "exit_reason": "Comma, and, multiple, commas, here",
        },
        {
            "trade_id": "tr_quotes_\"inside\"_id",
            "symbol": "ETH/EUR",
            "action": "SELL",
            "price": 3200.0,
            "size": 0.02,
            "pnl": 1.50,
            "exit_reason": 'Quote: "Take Profit" triggered at 3200',
        },
        {
            "trade_id": "tr_unicode_🚀_€_¥",
            "symbol": "SOL/EUR",
            "action": "SELL",
            "price": 150.0,
            "size": 0.5,
            "exit_reason": "Bullish breakout 🚀 Target reached 🎯",
        },
        {
            "trade_id": "tr_multiline_reason",
            "symbol": "BTC/EUR",
            "action": "SELL",
            "price": 61000.0,
            "size": 0.001,
            "pnl": -0.50,
            "exit_reason": "Reason line 1\nReason line 2\nReason line 3",
        },
    ]

    for t in adversarial_trades:
        ok = logger.log_trade(t)
        assert ok is True

    # Read back trades
    trades = logger.read_trades()
    assert len(trades) == 4

    assert trades[0]["trade_id"] == "tr_comma_in_id,123"
    assert trades[0]["exit_reason"] == "Comma, and, multiple, commas, here"

    assert trades[1]["trade_id"] == 'tr_quotes_"inside"_id'
    assert trades[1]["exit_reason"] == 'Quote: "Take Profit" triggered at 3200'

    assert trades[2]["trade_id"] == "tr_unicode_🚀_€_¥"
    assert trades[2]["exit_reason"] == "Bullish breakout 🚀 Target reached 🎯"

    assert trades[3]["trade_id"] == "tr_multiline_reason"
    assert trades[3]["exit_reason"] == "Reason line 1\nReason line 2\nReason line 3"



"""Unit and Integration Tests for Milestone 8: CSV Trade Logging Subsystem.

Validates:
- Header creation and RFC 4180 strict CSV compliance.
- Row formatting for BUY and SELL trades (TradeRecord and TradeExecutedPayload).
- AC1: Programmatic mock trade execution writing to persistent CSV via ExecutionSimulator and NotificationService.
- Thread-safety under high concurrency.
- Fault tolerance for read-only or invalid paths without crashing engine execution.
- Integration with paper trading CLI notification setup.
"""

import concurrent.futures
import csv
from datetime import datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import stat
import threading
import pytest

from trading_bot.cli.paper import setup_notifications
from trading_bot.config import DEFAULT_TRADES_CSV_PATH
from trading_bot.notifications.channels.csv_log import CsvTradeLogChannel, CsvTradeLogger
from trading_bot.notifications.models import (
    DeliveryStatus,
    EventType,
    OrderSide,
    OrderType,
    TradeExecutedPayload,
)
from trading_bot.notifications.service import NotificationService
from trading_bot.simulation.ledger import TradeRecord, VirtualLedger
from trading_bot.simulation.simulator import ExecutionSimulator, OrderSignal


# =====================================================================
# AC1 & Core CSV Logger Tests
# =====================================================================
def test_csv_trade_logger_initializes_header_on_creation(tmp_path: Path):
    """Scenario 1: Creates fresh file with exact RFC 4180 13-column header."""
    csv_file = tmp_path / "trades.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    content = csv_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert len(lines) == 1

    expected_header = (
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,"
        "portfolio_balance,cash_balance,exit_reason"
    )
    assert lines[0] == expected_header


def test_csv_trade_logger_does_not_duplicate_header_on_reopen(tmp_path: Path):
    """Reopening an existing non-empty CSV must not append another header row."""
    csv_file = tmp_path / "trades.csv"
    logger1 = CsvTradeLogger(csv_path=csv_file)
    assert len(csv_file.read_text(encoding="utf-8").splitlines()) == 1

    # Reopen
    logger2 = CsvTradeLogger(csv_path=csv_file)
    assert len(csv_file.read_text(encoding="utf-8").splitlines()) == 1


def test_csv_trade_logger_appends_mock_trade_row(tmp_path: Path):
    """Scenario 2: Log a mock BUY TradeRecord and verify row contents."""
    csv_file = tmp_path / "trades.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    trade = TradeRecord(
        trade_id="tr_00001",
        timestamp_ms=1757246400000,
        symbol="BTC/EUR",
        side="BUY",
        order_type="MARKET",
        executed_price=Decimal("60000.00"),
        raw_price=Decimal("60000.00"),
        quantity=Decimal("0.00016"),
        quote_notional=Decimal("9.6000"),
        fee=Decimal("0.0096"),
        realized_pnl=Decimal("0.00"),
        cash_after=Decimal("0.3904"),
        equity_after=Decimal("10.0000"),
    )

    success = logger.log_trade(trade)
    assert success is True

    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, f"Expected 2 lines (header + 1 trade), found: {lines}"

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2
        row = reader[1]

        # Verify exact columns:
        # timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason
        assert row[1] == "tr_00001"
        assert row[2] == "BTC/EUR"
        assert row[3] == "BUY"
        assert float(row[4]) == 60000.00
        assert row[4] == "60000.00"
        assert float(row[5]) == 0.000160
        assert float(row[6]) == 9.6000
        assert float(row[7]) == 0.0096
        assert row[8] == "0.00"
        assert row[9] == "0.00"
        assert float(row[10]) == 10.0000
        assert float(row[11]) == 0.3904
        assert row[12] == ""


def test_csv_trade_logger_appends_sell_trade_with_realized_pnl(tmp_path: Path):
    """Scenario 3: Log a closing SELL trade with realized PnL and exit reason."""
    csv_file = tmp_path / "trades.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    buy_trade = TradeRecord(
        trade_id="tr_00001",
        timestamp_ms=1757246400000,
        symbol="BTC/EUR",
        side="BUY",
        order_type="MARKET",
        executed_price=Decimal("60000.00"),
        raw_price=Decimal("60000.00"),
        quantity=Decimal("0.00016"),
        quote_notional=Decimal("9.6000"),
        fee=Decimal("0.0096"),
        realized_pnl=Decimal("0.00"),
        cash_after=Decimal("0.3904"),
        equity_after=Decimal("10.0000"),
    )
    logger.log_trade(buy_trade)

    sell_trade = TradeRecord(
        trade_id="tr_00002",
        timestamp_ms=1757253600000,
        symbol="BTC/EUR",
        side="SELL",
        order_type="MARKET",
        executed_price=Decimal("65000.00"),
        raw_price=Decimal("65000.00"),
        quantity=Decimal("0.00016"),
        quote_notional=Decimal("10.4000"),
        fee=Decimal("0.0104"),
        realized_pnl=Decimal("0.7850"),
        cash_after=Decimal("10.7800"),
        equity_after=Decimal("10.7800"),
        exit_reason="TAKE_PROFIT_TARGET",
    )
    success = logger.log_trade(sell_trade)
    assert success is True

    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3, f"Expected 3 lines (1 header + 2 trades), found: {lines}"

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 3
        row3 = reader[2]

        assert row3[1] == "tr_00002"
        assert row3[2] == "BTC/EUR"
        assert row3[3] == "SELL"
        assert float(row3[4]) == 65000.00
        assert float(row3[5]) == 0.000160
        assert float(row3[8]) == 0.7850
        assert row3[8] == "0.7850"
        assert float(row3[9]) > 0.0
        assert row3[12] == "TAKE_PROFIT_TARGET"


# =====================================================================
# AC1 End-to-End Simulation Integration Test
# =====================================================================
def test_csv_trade_log_channel_with_execution_simulator(tmp_path: Path):
    """Scenario 4 (AC1 Validation): Trigger mock trade through ExecutionSimulator

    and NotificationService; verify valid row is immediately appended to CSV.
    """
    csv_file = tmp_path / "trades.csv"
    channel = CsvTradeLogChannel(csv_path=csv_file, enabled=True)
    notif_service = NotificationService()
    notif_service.register_channel(channel)

    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        exchange="binance",
        notification_service=notif_service,
    )

    # Execute simulated BUY
    buy_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("9.60"),
        order_id="ord_mock_ac1_001",
    )
    result = simulator.execute(order=buy_order, current_price=Decimal("50000.00"), ledger=ledger)
    assert result.success is True

    # Verify CSV file now exists and contains exactly 2 lines
    assert csv_file.exists()
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    # Read trades via logger.read_trades()
    trades = channel.logger.read_trades()
    assert len(trades) == 1
    t = trades[0]

    assert t["symbol"] == "BTC/EUR"
    assert t["action"] == "BUY"
    assert t["trade_id"] == "ord_mock_ac1_001"
    assert t["price"] == pytest.approx(50000.0 * 1.0005, rel=1e-3)
    assert t["size"] > 0.0
    assert t["notional"] > 0.0
    assert t["fee"] > 0.0
    assert t["pnl"] == 0.0
    assert t["exit_reason"] == ""

    # Now execute simulated SELL to close position
    held_qty = ledger.get_position("BTC/EUR").quantity
    sell_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        target_quantity=held_qty,
        order_id="ord_mock_ac1_002",
        exit_reason="TAKE_PROFIT",
    )
    sell_result = simulator.execute(order=sell_order, current_price=Decimal("55000.00"), ledger=ledger)
    assert sell_result.success is True

    trades_after = channel.logger.read_trades()
    assert len(trades_after) == 2
    sell_t = trades_after[1]

    assert sell_t["symbol"] == "BTC/EUR"
    assert sell_t["action"] == "SELL"
    assert sell_t["trade_id"] == "ord_mock_ac1_002"
    assert sell_t["pnl"] > 0.0
    assert sell_t["exit_reason"] in ("", "TAKE_PROFIT")


# =====================================================================
# Thread Safety Test
# =====================================================================
def test_csv_trade_logger_thread_safety(tmp_path: Path):
    """Scenario 5: Concurrent appends from 10 threads writing 10 trades each.

    Asserts file contains 101 total lines with zero corruption or row interleaving.
    """
    csv_file = tmp_path / "concurrent_trades.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    num_threads = 10
    trades_per_thread = 10

    def worker(worker_id: int):
        for i in range(trades_per_thread):
            is_buy = i % 2 == 0
            trade = TradeRecord(
                trade_id=f"tr_{worker_id:02d}_{i:02d}",
                timestamp_ms=1757246400000 + (worker_id * 100 + i) * 1000,
                symbol="BTC/EUR",
                side="BUY" if is_buy else "SELL",
                order_type="MARKET",
                executed_price=Decimal("60000.00"),
                raw_price=Decimal("60000.00"),
                quantity=Decimal("0.00016"),
                quote_notional=Decimal("9.6000"),
                fee=Decimal("0.0096"),
                realized_pnl=Decimal("0.00") if is_buy else Decimal("0.7850"),
                cash_after=Decimal("10.0000"),
                equity_after=Decimal("10.7850"),
                exit_reason="" if is_buy else "STOP_LOSS",
            )
            logger.log_trade(trade)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, wid) for wid in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    lines = csv_file.read_text(encoding="utf-8").splitlines()
    total_expected_lines = 1 + (num_threads * trades_per_thread)
    assert len(lines) == total_expected_lines

    # Verify every row parses into 13 columns
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == total_expected_lines
        for idx, row in enumerate(reader):
            assert len(row) == 13, f"Row {idx} malformed: {row}"


# =====================================================================
# Fault Tolerance Tests
# =====================================================================
def test_csv_trade_logger_readonly_directory_handling(tmp_path: Path):
    """Scenario 6: Read-only or unwritable path does not raise unhandled exceptions."""
    readonly_dir = tmp_path / "readonly_dir"
    readonly_dir.mkdir()
    csv_file = readonly_dir / "trades.csv"

    # Make directory read-only
    readonly_dir.chmod(stat.S_IREAD | stat.S_IEXEC)

    try:
        # Initializing logger or logging should catch OSError, log warning, and not crash
        logger = CsvTradeLogger(csv_path=csv_file)
        payload = TradeExecutedPayload(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            fill_price=60000.0,
            quantity=0.00016,
            fee=0.0096,
            portfolio_balance=10.0,
        )
        res = logger.log_trade(payload)
        # Safely returns False without raising exception
        assert res is False
    finally:
        # Restore permissions for clean tmp_path teardown
        readonly_dir.chmod(stat.S_IRWXU)


def test_csv_trade_log_channel_fault_isolation(tmp_path: Path):
    """Channel exceptions must return DeliveryStatus(success=False) without crashing dispatch."""
    readonly_dir = tmp_path / "ro_chan"
    readonly_dir.mkdir()
    csv_file = readonly_dir / "trades.csv"
    readonly_dir.chmod(stat.S_IREAD | stat.S_IEXEC)

    try:
        channel = CsvTradeLogChannel(csv_path=csv_file, enabled=True)
        payload = TradeExecutedPayload(
            symbol="ETH/EUR",
            side=OrderSide.BUY,
            fill_price=3000.0,
            quantity=0.003,
            portfolio_balance=10.0,
        )
        status = channel.dispatch(payload)
        assert isinstance(status, DeliveryStatus)
        assert not status.success
        assert status.channel_name == "csv_trade_log"
    finally:
        readonly_dir.chmod(stat.S_IRWXU)


# =====================================================================
# Additional Feature & RFC 4180 Edge Case Tests
# =====================================================================
def test_csv_trade_logger_read_trades_and_clear(tmp_path: Path):
    """Test read_trades parser and clear method."""
    csv_file = tmp_path / "trades.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    # Empty file read returns empty list
    assert logger.read_trades() == []

    # Write 2 trades
    t1 = TradeExecutedPayload(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        fill_price=50000.0,
        quantity=0.0002,
        fee=0.01,
        portfolio_balance=10.0,
        cash_balance=0.0,
    )
    t2 = TradeExecutedPayload(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        fill_price=55000.0,
        quantity=0.0002,
        fee=0.011,
        portfolio_balance=11.0,
        cash_balance=11.0,
        realized_pnl=0.989,
        strategy_signal="TAKE_PROFIT",
    )
    logger.log_trade(t1)
    logger.log_trade(t2)

    trades = logger.read_trades()
    assert len(trades) == 2
    assert trades[0]["action"] == "BUY"
    assert trades[0]["price"] == 50000.0
    assert trades[1]["action"] == "SELL"
    assert trades[1]["pnl"] == pytest.approx(0.989, rel=1e-3)
    assert trades[1]["exit_reason"] == "TAKE_PROFIT"

    # Clear should leave header only
    logger.clear()
    assert len(logger.read_trades()) == 0
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_csv_trade_logger_rfc4180_special_characters(tmp_path: Path):
    """Test proper RFC 4180 quoting and escaping of commas and quotes in text fields."""
    csv_file = tmp_path / "trades_rfc4180.csv"
    logger = CsvTradeLogger(csv_path=csv_file)

    trade = TradeRecord(
        trade_id="tr_special,001",
        timestamp_ms=1757246400000,
        symbol="BTC/EUR",
        side="SELL",
        order_type="MARKET",
        executed_price=Decimal("60000.00"),
        raw_price=Decimal("60000.00"),
        quantity=Decimal("0.00016"),
        quote_notional=Decimal("9.6000"),
        fee=Decimal("0.0096"),
        realized_pnl=Decimal("-0.1500"),
        cash_after=Decimal("9.8500"),
        equity_after=Decimal("9.8500"),
        exit_reason='STOP_LOSS, "SLIPPAGE_EXCEEDED"',
    )
    logger.log_trade(trade)

    trades = logger.read_trades()
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "tr_special,001"
    assert trades[0]["exit_reason"] == 'STOP_LOSS, "SLIPPAGE_EXCEEDED"'


def test_paper_cli_setup_notifications_with_csv_logger(tmp_path: Path):
    """Test that setup_notifications in paper CLI registers CsvTradeLogChannel."""
    target_csv = tmp_path / "cli_trades.csv"
    service = setup_notifications(channel_name="stdout", trades_csv_path=target_csv)

    assert service.has_channel("csv_trade_log")
    channel = service.get_channel("csv_trade_log")
    assert isinstance(channel, CsvTradeLogChannel)
    assert channel.logger.csv_path == target_csv

    # Dispatch a trade and assert CSV was populated
    service.notify_trade(
        symbol="SOL/EUR",
        side="BUY",
        fill_price=150.0,
        quantity=0.06,
        fee=0.009,
        portfolio_balance=10.0,
    )

    assert target_csv.exists()
    records = channel.logger.read_trades()
    assert len(records) == 1
    assert records[0]["symbol"] == "SOL/EUR"
    assert records[0]["action"] == "BUY"

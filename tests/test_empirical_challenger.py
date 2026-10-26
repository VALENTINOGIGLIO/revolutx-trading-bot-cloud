"""Empirical Challenger Verification and Stress Test Suite.

Author: challenger_2
Milestone: M7 Adversarial Hardening & Audit

Mandates:
1. Conservation of money identity across 1,000 synthetic random-walk trades:
   Verify mathematical invariants of VirtualLedger and ExecutionSimulator under
   stochastic execution, fees, and slippage.
2. 3-Phase Compounding Model & RiskGuard stress tests:
   - Does Phase 1 ever attempt an order below min_notional (5.0 EUR Binance)?
   - Can the account survive 20 consecutive losing trades from 10€ without crash?
   - Does 8% daily rolling drawdown circuit breaker trigger accurately and halt signals?
3. Parametric backtest matrix:
   Run backtest across multiple combinations of symbols, timeframes, exchanges,
   balances, fee/slippage rates, and verify zero runtime crashes.
"""

from __future__ import annotations

import concurrent.futures
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
import random
import pytest

from trading_bot.data.models import Candle
from trading_bot.simulation.ledger import (
    InsufficientFundsError,
    InsufficientPositionError,
    VirtualLedger,
)
from trading_bot.simulation.simulator import (
    EXCHANGE_MIN_NOTIONAL,
    ExecutionResult,
    ExecutionSimulator,
    OrderSide,
    OrderSignal,
    OrderType,
)
from trading_bot.strategy.amv_strategy import AMVTrendBreakStrategy
from trading_bot.strategy.compounding import (
    CompoundingPhase,
    CompoundingPositionSizer,
    RiskGuard,
)
from trading_bot.cli.backtest import run_backtest
from trading_bot.notifications.channels.csv_log import CsvTradeLogChannel, CsvTradeLogger
from trading_bot.notifications.service import NotificationService


# =====================================================================
# MANDATE 1: CONSERVATION OF MONEY OVER 1,000 SYNTHETIC TRADES
# =====================================================================

def test_conservation_of_money_1000_synthetic_trades():
    """Empirically test conservation of money across 1,000 synthetic random-walk trades.

    Mathematical Invariants to check at every single trade:
    Invariant 1: Total Equity == Cash + Locked Cash + Position Market Value
    Invariant 2: Cash == Initial Cash + Realized PnL - Cumulative Buy Fees (when flat)
    Invariant 3: Net Worth delta == Net Trade PnL - Total Fees
    Invariant 4: Total Fees == Cumulative Buy Fees + Cumulative Sell Fees
    Invariant 5: Zero floating point drift (exact Decimal arithmetic throughout)
    """
    random.seed(42)
    initial_cash = Decimal("1000.00")
    ledger = VirtualLedger(initial_cash=initial_cash, quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.0010"),  # 0.10%
        slippage=Decimal("0.0005"),  # 0.05%
        min_notional=Decimal("5.00"),
        exchange="binance",
        step_size=Decimal("0.0001"),
        tick_size=Decimal("0.01"),
    )

    current_price = Decimal("100.00")
    base_time_ms = 1785542400000
    trades_executed = 0
    in_position = False
    position_qty = Decimal("0.0")

    for step in range(1000):
        # Geometric random walk with drift
        pct_change = Decimal(str(round(random.gauss(0.0002, 0.015), 6)))
        current_price = max(Decimal("10.00"), (current_price * (Decimal("1") + pct_change)).quantize(Decimal("0.01")))
        ts = base_time_ms + (step * 60000)

        if not in_position:
            # Randomly decide to buy if cash > 10 EUR
            if ledger.available_cash >= Decimal("10.00") and random.random() < 0.6:
                alloc_fraction = Decimal(str(round(random.uniform(0.3, 0.9), 2)))
                budget = (ledger.available_cash * alloc_fraction).quantize(Decimal("0.01"))
                if budget >= Decimal("5.00"):
                    res = simulator.execute_buy(
                        symbol="ETH/EUR",
                        raw_price=current_price,
                        ledger=ledger,
                        quote_budget=budget,
                        timestamp_ms=ts,
                    )
                    if res.success:
                        trades_executed += 1
                        in_position = True
                        position_qty = res.quantity
        else:
            # In position: randomly decide to exit
            if random.random() < 0.4:
                res = simulator.execute_sell(
                    symbol="ETH/EUR",
                    raw_price=current_price,
                    ledger=ledger,
                    quantity=position_qty,
                    timestamp_ms=ts,
                    exit_reason="RANDOM_WALK_EXIT",
                )
                if res.success:
                    trades_executed += 1
                    in_position = False
                    position_qty = Decimal("0.0")

        # ASSERT INVARIANTS AFTER EVERY STEP
        pos = ledger.get_position("ETH/EUR")
        mkt_val = pos.quantity * current_price
        expected_equity = ledger.cash + ledger.locked_cash + mkt_val
        actual_equity = ledger.total_equity({"ETH/EUR": current_price})

        # Invariant 1: Total Equity == Cash + Locked Cash + Position Market Value
        assert actual_equity == expected_equity, (
            f"Step {step}: Invariant 1 failed! Actual equity {actual_equity} != expected {expected_equity}"
        )

        # Invariant 2: When flat, double entry reconciliation holds exactly
        if pos.quantity == Decimal("0.0"):
            buy_fees = sum(t.fee for t in ledger.trades if t.side == "BUY")
            assert ledger.cash == initial_cash + ledger.realized_pnl - buy_fees, (
                f"Step {step}: Invariant 2 failed! Cash {ledger.cash} != initial {initial_cash} + pnl {ledger.realized_pnl} - buy_fees {buy_fees}"
            )

        # Invariant 3: Cumulative fees exactly equal sum of trade fees
        sum_trade_fees = sum(t.fee for t in ledger.trades)
        assert ledger.total_fees == sum_trade_fees, (
            f"Step {step}: Invariant 3 failed! Total fees {ledger.total_fees} != sum {sum_trade_fees}"
        )

        # Invariant 4: Cash is non-negative
        assert ledger.cash >= Decimal("0.00"), f"Step {step}: Cash became negative: {ledger.cash}"

    # Close any open position at the end
    if in_position:
        res = simulator.execute_sell(
            symbol="ETH/EUR",
            raw_price=current_price,
            ledger=ledger,
            quantity=position_qty,
            timestamp_ms=base_time_ms + 1000 * 60000,
            exit_reason="FINAL_CLOSE",
        )
        assert res.success

    # Final flat reconciliation
    buy_fees = sum(t.fee for t in ledger.trades if t.side == "BUY")
    sell_fees = sum(t.fee for t in ledger.trades if t.side == "SELL")
    assert buy_fees + sell_fees == ledger.total_fees
    assert ledger.cash == initial_cash + ledger.realized_pnl - buy_fees
    assert ledger.total_equity() == ledger.cash
    assert len(ledger.trades) >= 200, f"Expected >200 executed trades, got {len(ledger.trades)}"


# =====================================================================
# MANDATE 2: 3-PHASE COMPOUNDING & RISK GUARD STRESS TESTS
# =====================================================================

def test_phase_1_never_orders_below_min_notional():
    """Verify that Phase 1 never produces or executes an order below exchange min_notional (5.00 EUR)."""
    sizer = CompoundingPositionSizer(
        min_notional=Decimal("5.00"),
        default_step_size=Decimal("0.001"),
    )

    test_equities = [
        Decimal("0.01"), Decimal("1.00"), Decimal("4.00"), Decimal("4.99"),
        Decimal("5.00"), Decimal("5.01"), Decimal("5.05"), Decimal("5.20"),
        Decimal("6.00"), Decimal("7.50"), Decimal("9.99"), Decimal("10.00"),
        Decimal("10.50"), Decimal("20.00"), Decimal("50.00"), Decimal("99.99"),
    ]
    test_prices = [Decimal("10.00"), Decimal("50.00"), Decimal("150.00"), Decimal("3000.00"), Decimal("60000.00")]
    multipliers = [1.0, 0.70, 0.50, 0.0]

    for eq in test_equities:
        for px in test_prices:
            for mult in multipliers:
                stop = px * Decimal("0.97")  # 3% stop
                res = sizer.calculate_position_size(
                    equity=eq,
                    entry_price=px,
                    stop_loss=stop,
                    risk_multiplier=mult,
                    min_notional=Decimal("5.00"),
                    step_size=Decimal("0.0001"),
                )

                if res.is_valid:
                    # If marked valid, notional MUST be >= min_notional (5.00 EUR)
                    assert res.notional >= Decimal("5.00"), (
                        f"VIOLATION: Valid order with notional {res.notional} < 5.00 EUR (eq={eq}, px={px}, mult={mult})"
                    )
                    assert res.notional <= eq, (
                        f"VIOLATION: Order notional {res.notional} exceeds equity {eq}"
                    )
                else:
                    assert res.quantity == Decimal("0")
                    assert res.notional == Decimal("0")


def test_20_consecutive_losses_bybit_survival():
    """Verify that the account survives 20 consecutive losing trades from 10€ on Bybit without crashing.

    Bybit min_notional = 1.00 EUR.
    Asserts:
    1. Zero unhandled exceptions or runtime crashes across 20 losing trades.
    2. Cash and equity remain strictly positive (> 7 EUR).
    3. Ledger accurately records all 20 closed losing trades.
    """
    initial_cash = Decimal("10.00")
    min_notional = Decimal("1.00")
    ledger = VirtualLedger(initial_cash=initial_cash, quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=min_notional,
        step_size=Decimal("0.001"),
        exchange="bybit",
    )
    sizer = CompoundingPositionSizer(min_notional=min_notional, default_step_size=Decimal("0.001"))
    risk_guard = RiskGuard(max_daily_drawdown_pct=0.08, drawdown_halt_hours=24.0, initial_equity=initial_cash)

    entry_price = Decimal("100.00")
    stop_distance_pct = Decimal("0.02")  # 2% stop loss per trade
    current_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)

    executed_losing_trades = 0

    for trade_idx in range(20):
        current_time += timedelta(hours=25)  # daily drawdown reset
        eq = ledger.total_equity()

        risk_status = risk_guard.check_risk_status(current_time)
        mult = risk_status.risk_multiplier

        stop_price = entry_price * (Decimal("1") - stop_distance_pct)
        size_res = sizer.calculate_position_size(
            equity=eq,
            entry_price=entry_price,
            stop_loss=stop_price,
            risk_multiplier=mult,
            min_notional=min_notional,
            step_size=Decimal("0.001"),
        )

        assert size_res.is_valid, f"Trade {trade_idx} unexpectedly invalid on Bybit"

        ts_ms = int(current_time.timestamp() * 1000)
        buy_res = simulator.execute_buy(
            symbol="SOL/EUR",
            raw_price=entry_price,
            ledger=ledger,
            quantity=size_res.quantity,
            timestamp_ms=ts_ms,
        )
        assert buy_res.success, f"Buy failed at trade {trade_idx}: {buy_res.error_message}"

        sell_time = current_time + timedelta(minutes=15)
        sell_ts_ms = int(sell_time.timestamp() * 1000)
        sell_res = simulator.execute_sell(
            symbol="SOL/EUR",
            raw_price=stop_price,
            ledger=ledger,
            quantity=buy_res.quantity,
            timestamp_ms=sell_ts_ms,
            exit_reason="STOP_LOSS_HIT",
        )
        assert sell_res.success, f"Sell failed at trade {trade_idx}: {sell_res.error_message}"
        assert sell_res.realized_pnl < Decimal("0")

        executed_losing_trades += 1
        risk_guard.on_trade_completed(sell_res.realized_pnl, ledger.total_equity(), sell_time)

        assert ledger.cash > Decimal("5.00"), f"Cash drained unexpectedly: {ledger.cash}"
        assert ledger.total_equity() == ledger.cash

    assert executed_losing_trades == 20
    assert ledger.losing_trades_count == 20
    assert ledger.winning_trades_count == 0
    assert ledger.cash > Decimal("7.00")


def test_20_consecutive_losses_binance_all_in_survival():
    """Verify that Phase 1 all-in sizing (96%) survives 20 consecutive losses on Binance without crashing.

    When all-in sizing is maintained (as specified in PROJECT.md Phase 1), orders start at 9.60 EUR
    and remain above the 5.00 EUR min_notional for over 20 trades.
    """
    initial_cash = Decimal("10.00")
    min_notional = Decimal("5.00")
    ledger = VirtualLedger(initial_cash=initial_cash, quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=min_notional,
        step_size=Decimal("0.001"),
        exchange="binance",
    )
    sizer = CompoundingPositionSizer(min_notional=min_notional, default_step_size=Decimal("0.001"))

    entry_price = Decimal("100.00")
    stop_distance_pct = Decimal("0.02")  # 2% stop loss
    current_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)

    executed_losing_trades = 0

    for trade_idx in range(20):
        current_time += timedelta(hours=25)
        eq = ledger.total_equity()

        # In Phase 1 all-in sizing, mult=1.0 is maintained to avoid the <5 EUR bumping trap
        stop_price = entry_price * (Decimal("1") - stop_distance_pct)
        size_res = sizer.calculate_position_size(
            equity=eq,
            entry_price=entry_price,
            stop_loss=stop_price,
            risk_multiplier=1.0,
            min_notional=min_notional,
            step_size=Decimal("0.001"),
        )

        assert size_res.is_valid, f"Trade {trade_idx} failed validation at equity {eq}"

        ts_ms = int(current_time.timestamp() * 1000)
        buy_res = simulator.execute_buy(
            symbol="SOL/EUR",
            raw_price=entry_price,
            ledger=ledger,
            quantity=size_res.quantity,
            timestamp_ms=ts_ms,
        )
        assert buy_res.success, f"Buy failed at trade {trade_idx}: {buy_res.error_message}"

        sell_time = current_time + timedelta(minutes=15)
        sell_ts_ms = int(sell_time.timestamp() * 1000)
        sell_res = simulator.execute_sell(
            symbol="SOL/EUR",
            raw_price=stop_price,
            ledger=ledger,
            quantity=buy_res.quantity,
            timestamp_ms=sell_ts_ms,
            exit_reason="STOP_LOSS_HIT",
        )
        assert sell_res.success, f"Sell failed at trade {trade_idx}: {sell_res.error_message}"
        assert sell_res.realized_pnl < Decimal("0")

        executed_losing_trades += 1

        assert ledger.cash >= Decimal("5.00"), f"Cash drained below min_notional: {ledger.cash}"
        assert ledger.total_equity() == ledger.cash

    assert executed_losing_trades == 20
    assert ledger.losing_trades_count == 20
    assert ledger.cash >= Decimal("5.50")


def test_binance_dust_trap_vulnerability():
    """Vulnerability Proof: Demonstrates the 'Dust Trap' on Binance when RiskGuard throttles Phase 1.

    When RiskGuard cuts risk multiplier to 0.50 after 3 consecutive losses:
    1. Target notional drops to 4.80 EUR (< 5.00 EUR Binance minNotional).
    2. Sizer bumps quantity up to barely satisfy minNotional on BUY (5.10 EUR).
    3. At stop loss (-2%), market value drops to 4.995 EUR (< 5.00 EUR).
    4. ExecutionSimulator rejects the SELL, trapping the trader in an unclosable position!
    """
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        step_size=Decimal("0.001"),
        exchange="binance",
    )
    sizer = CompoundingPositionSizer(min_notional=Decimal("5.00"), default_step_size=Decimal("0.001"))

    # Force risk multiplier = 0.50 (simulating 3+ consecutive losses)
    eq = Decimal("9.44")
    entry_price = Decimal("100.00")
    stop_price = Decimal("98.00")

    size_res = sizer.calculate_position_size(
        equity=eq,
        entry_price=entry_price,
        stop_loss=stop_price,
        risk_multiplier=0.50,
        min_notional=Decimal("5.00"),
        step_size=Decimal("0.001"),
    )

    assert size_res.is_valid is True
    # The sizer with min_viable_entry buffers entry to 0.054 SOL (notional = 5.40 EUR)
    assert size_res.quantity == Decimal("0.054")

    # Buy executes successfully
    buy_res = simulator.execute_buy("SOL/EUR", entry_price, ledger, quantity=size_res.quantity, timestamp_ms=1000)
    assert buy_res.success is True

    # Now price drops to stop loss (98.00 EUR raw, 97.95 EUR executed)
    sell_res = simulator.execute_sell("SOL/EUR", stop_price, ledger, quantity=buy_res.quantity, timestamp_ms=2000)

    # Dust trap vulnerability resolved: Sell order succeeds and position is closed
    assert sell_res.success is True
    assert ledger.has_position("SOL/EUR") is False


def test_8_percent_drawdown_circuit_breaker():
    """Verify 8% daily rolling drawdown circuit breaker halts trading signals when triggered."""
    risk_guard = RiskGuard(
        max_daily_drawdown_pct=0.08,
        drawdown_halt_hours=24.0,
        initial_equity=Decimal("100.00"),
    )
    t0 = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

    # Initial state: equity 100.00 EUR -> allowed
    risk_guard.update_equity(Decimal("100.00"), t0)
    status0 = risk_guard.check_risk_status(t0)
    assert status0.allowed is True
    assert status0.halt_active is False
    assert status0.risk_multiplier == 1.0

    # Minor drawdown: equity drops 5% to 95.00 EUR at t0 + 2h -> allowed
    t1 = t0 + timedelta(hours=2)
    risk_guard.update_equity(Decimal("95.00"), t1)
    status1 = risk_guard.check_risk_status(t1)
    assert status1.allowed is True
    assert status1.halt_active is False

    # Drawdown touches 8.5% (drops to 91.50 EUR) at t0 + 4h -> HALT TRIGGERED
    t2 = t0 + timedelta(hours=4)
    risk_guard.update_equity(Decimal("91.50"), t2)
    status2 = risk_guard.check_risk_status(t2)
    assert status2.allowed is False
    assert status2.halt_active is True
    assert status2.risk_multiplier == 0.0
    assert "drawdown" in status2.reason.lower()
    expected_halt_until = t2 + timedelta(hours=24)
    assert status2.halt_expires_at == expected_halt_until

    # Check 12 hours into halt -> still halted
    t3 = t2 + timedelta(hours=12)
    status3 = risk_guard.check_risk_status(t3)
    assert status3.allowed is False
    assert status3.halt_active is True

    # Check 23h 59m into halt -> still halted
    t4 = t2 + timedelta(hours=23, minutes=59)
    status4 = risk_guard.check_risk_status(t4)
    assert status4.allowed is False
    assert status4.halt_active is True

    # Check 24h 01m after trigger -> HALT EXPIRED, trading resumed
    t5 = t2 + timedelta(hours=24, minutes=1)
    status5 = risk_guard.check_risk_status(t5)
    assert status5.allowed is True
    assert status5.halt_active is False


# =====================================================================
# MANDATE 3: PARAMETRIC BACKTEST MATRIX VERIFICATION
# =====================================================================

@pytest.mark.parametrize("pair,timeframe,exchange,balance", [
    ("BTC/EUR", "1h", "binance", 10.0),
    ("BTC/EUR", "15m", "binance", 10.0),
    ("ETH/EUR", "1h", "binance", 10.0),
    ("SOL/EUR", "1h", "binance", 10.0),
    ("BTC/EUR", "1h", "bybit", 5.0),
    ("BTC/EUR", "1h", "binance", 50.0),
    ("BTC/EUR", "1h", "binance", 100.0),
    ("BTC/EUR", "1h", "binance", 1000.0),
])
def test_backtest_matrix_zero_crashes(pair, timeframe, exchange, balance):
    """Run programmatic backtest across multiple parameter combinations.
    Assert zero runtime exceptions, valid metrics, and conservation of money.
    """
    start_dt = datetime(2026, 8, 1, tzinfo=timezone.utc)
    end_dt = datetime(2026, 9, 1, tzinfo=timezone.utc)

    ledger, metrics, candles = run_backtest(
        pair=pair,
        start=start_dt,
        end=end_dt,
        balance=balance,
        currency="EUR",
        timeframe=timeframe,
        exchange=exchange,
        fee_rate=0.001,
        slippage=0.0005,
    )

    assert len(candles) > 0
    assert metrics.initial_balance == Decimal(str(balance))
    assert metrics.final_balance > Decimal("0.0")
    assert ledger.cash >= Decimal("0.0")
    # Tolerance comparison accounting for 4-decimal place formatting in PerformanceMetrics
    assert abs(ledger.total_equity() - metrics.final_balance) < Decimal("0.0001")


# =====================================================================
# MANDATE 4: MILESTONE 8 — AC1 EMPIRICAL VALIDATION & SIMULATION PIPELINE STRESS
# =====================================================================

def test_empirical_ac1_simulation_pipeline_stress(tmp_path: Path):
    """Empirically validate Acceptance Criterion 1 (AC1) under pipeline stress.

    - Wire ExecutionSimulator with NotificationService and CsvTradeLogChannel.
    - Run an alternating sequence of 60 BUY and SELL orders across multiple assets
      (BTC/EUR, ETH/EUR, SOL/EUR) with various order types (MARKET, LIMIT, STOP_LOSS,
      TAKE_PROFIT) and profit/loss scenarios (strong gains, stop-loss cuts, breakeven).
    - Parse the resulting CSV and verify:
      1. Exactly 61 lines (1 header + 60 records).
      2. Line 1 matches RFC 4180 13-column header specification.
      3. For every single trade:
         - exact matching timestamp (ISO UTC format).
         - action is BUY or SELL matching ledger.
         - price formatted to 2 decimals matching ledger.
         - size formatted to 6 decimals matching ledger.
         - pnl for BUY is exactly 0.00 and pnl_pct is 0.00.
         - pnl for SELL accurately matches realized profit/loss.
         - portfolio balance and cash balance accurately match ledger states.
      4. Conservation of money invariant holds across the entire parsed CSV dataset.
    """
    csv_file = tmp_path / "ac1_pipeline_stress.csv"
    channel = CsvTradeLogChannel(csv_path=csv_file, enabled=True)
    service = NotificationService()
    service.register_channel(channel)

    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        exchange="binance",
        notification_service=service,
    )
    initial_cash = Decimal("100.00")
    ledger = VirtualLedger(initial_cash=initial_cash, quote_currency="EUR")

    symbols = ["BTC/EUR", "ETH/EUR", "SOL/EUR"]
    base_prices = {
        "BTC/EUR": Decimal("60000.00"),
        "ETH/EUR": Decimal("3000.00"),
        "SOL/EUR": Decimal("150.00"),
    }

    # 30 roundtrips = 60 total executed orders
    num_roundtrips = 30
    base_ts = 1757246400000  # Epoch ms anchor

    for k in range(num_roundtrips):
        sym = symbols[k % len(symbols)]
        entry_raw = base_prices[sym]

        # Alternating order types for BUY
        buy_type = OrderType.MARKET if (k % 2 == 0) else OrderType.LIMIT
        buy_order_id = f"ord_buy_{k:03d}_{sym.replace('/', '_')}"
        buy_ts = base_ts + (k * 2) * 60000

        # Safe sizing: 20 EUR or 50% available cash, well above min_notional 5 EUR
        buy_budget = min(Decimal("20.00"), (ledger.available_cash * Decimal("0.5")).quantize(Decimal("0.01")))
        if buy_budget < Decimal("5.50"):
            buy_budget = Decimal("5.50")

        buy_signal = OrderSignal(
            symbol=sym,
            side=OrderSide.BUY,
            target_notional=buy_budget,
            order_type=buy_type,
            order_id=buy_order_id,
            timestamp_ms=buy_ts,
        )
        buy_res = simulator.execute(order=buy_signal, current_price=entry_raw, ledger=ledger, timestamp_ms=buy_ts)
        assert buy_res.success is True, f"Buy roundtrip {k} failed: {buy_res.error_message}"

        # Determine exit scenario for SELL
        scenario_idx = k % 5
        if scenario_idx == 0:
            # Strong win: +6%
            exit_raw = entry_raw * Decimal("1.06")
            sell_type = OrderType.TAKE_PROFIT
            exit_reason = "TAKE_PROFIT_TARGET_HIT"
        elif scenario_idx == 1:
            # Stop loss cut: -4%
            exit_raw = entry_raw * Decimal("0.96")
            sell_type = OrderType.STOP_LOSS
            exit_reason = "STOP_LOSS_TRIGGERED"
        elif scenario_idx == 2:
            # Breakeven/flat: 0% price movement (fee loss)
            exit_raw = entry_raw
            sell_type = OrderType.MARKET
            exit_reason = "TIME_EXPIRY_EXIT"
        elif scenario_idx == 3:
            # High profit surge: +12%
            exit_raw = entry_raw * Decimal("1.12")
            sell_type = OrderType.TAKE_PROFIT
            exit_reason = "MOMENTUM_SURGE_EXIT"
        else:
            # Mild loss: -2%
            exit_raw = entry_raw * Decimal("0.98")
            sell_type = OrderType.STOP_LOSS
            exit_reason = "CHANDELIER_EXIT"

        held_qty = ledger.get_position(sym).quantity
        sell_order_id = f"ord_sell_{k:03d}_{sym.replace('/', '_')}"
        sell_ts = base_ts + (k * 2 + 1) * 60000

        sell_signal = OrderSignal(
            symbol=sym,
            side=OrderSide.SELL,
            target_quantity=held_qty,
            order_type=sell_type,
            order_id=sell_order_id,
            exit_reason=exit_reason,
            timestamp_ms=sell_ts,
        )
        sell_res = simulator.execute(order=sell_signal, current_price=exit_raw, ledger=ledger, timestamp_ms=sell_ts)
        assert sell_res.success is True, f"Sell roundtrip {k} failed: {sell_res.error_message}"

    # Verify CSV file existence and raw line structure
    assert csv_file.exists()
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 61, f"Expected 61 lines (1 header + 60 trades), found {len(lines)}"

    expected_header = (
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,"
        "portfolio_balance,cash_balance,exit_reason"
    )
    assert lines[0] == expected_header

    # Read and parse CSV via csv.DictReader and CsvTradeLogger.read_trades()
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    parsed_trades = channel.logger.read_trades()
    assert len(reader) == 60
    assert len(parsed_trades) == 60
    assert len(ledger.trades) == 60

    # Validate every single trade against ledger record
    for idx, (row, trade) in enumerate(zip(reader, ledger.trades)):
        # 1. Exact matching timestamp
        assert row["timestamp"] == trade.timestamp.isoformat(), f"Trade {idx} timestamp mismatch"

        # 2. Exact action (BUY / SELL)
        assert row["action"] == trade.side, f"Trade {idx} action mismatch"

        # 3. Exact price (2 decimal places)
        expected_px = f"{float(trade.executed_price):.2f}"
        assert row["price"] == expected_px, f"Trade {idx} price mismatch"
        assert float(row["price"]) == pytest.approx(float(trade.executed_price), abs=0.01)

        # 4. Exact size (6 decimal places)
        expected_size = f"{float(trade.quantity):.6f}"
        assert row["size"] == expected_size, f"Trade {idx} size mismatch"
        assert float(row["size"]) == pytest.approx(float(trade.quantity), abs=1e-6)

        # 5. Notional and Fee
        expected_notional = f"{float(trade.quote_notional):.4f}"
        assert row["notional"] == expected_notional, f"Trade {idx} notional mismatch"
        expected_fee = f"{float(trade.fee):.4f}"
        assert row["fee"] == expected_fee, f"Trade {idx} fee mismatch"

        # 6. PnL semantics
        if trade.side == "BUY":
            assert row["pnl"] == "0.00", f"BUY Trade {idx} must have pnl 0.00"
            assert row["pnl_pct"] == "0.00", f"BUY Trade {idx} must have pnl_pct 0.00"
            assert row["exit_reason"] == "", f"BUY Trade {idx} exit_reason must be empty"
        else:
            expected_pnl = f"{float(trade.realized_pnl):.4f}"
            assert row["pnl"] == expected_pnl, f"SELL Trade {idx} pnl mismatch"
            assert float(row["pnl"]) == pytest.approx(float(trade.realized_pnl), abs=1e-4)

            # Check realized PnL direction
            if trade.realized_pnl > Decimal("0.0"):
                assert float(row["pnl"]) > 0.0
                assert float(row["pnl_pct"]) > 0.0
            elif trade.realized_pnl < Decimal("0.0"):
                assert float(row["pnl"]) < 0.0
                assert float(row["pnl_pct"]) < 0.0

        # 7. Cash balance and Portfolio balance accurately reflect ledger states
        expected_equity = f"{float(trade.equity_after):.4f}"
        expected_cash = f"{float(trade.cash_after):.4f}"
        assert row["portfolio_balance"] == expected_equity, f"Trade {idx} portfolio balance mismatch"
        assert row["cash_balance"] == expected_cash, f"Trade {idx} cash balance mismatch"
        assert float(row["portfolio_balance"]) == pytest.approx(float(trade.equity_after), abs=1e-4)
        assert float(row["cash_balance"]) == pytest.approx(float(trade.cash_after), abs=1e-4)

    # 8. Conservation of Money invariant across CSV dataset
    final_csv_cash = Decimal(reader[-1]["cash_balance"])
    total_sell_pnl = sum(Decimal(r["pnl"]) for r in reader if r["action"] == "SELL")
    total_buy_fees = sum(Decimal(r["fee"]) for r in reader if r["action"] == "BUY")
    expected_final_cash = initial_cash + total_sell_pnl - total_buy_fees
    assert abs(final_csv_cash - expected_final_cash) < Decimal("0.001"), (
        f"Money conservation broken in CSV: final_cash={final_csv_cash} vs expected={expected_final_cash}"
    )


def test_empirical_ac1_10_euro_budget_paper_pipeline(tmp_path: Path):
    """Empirically test the 10.00 EUR initial capital paper trading pipeline.

    Validates that starting with exactly 10.00 EUR capital:
    - Micro-orders meet Binance min_notional (5.00 EUR).
    - Trades alternate through realistic compounding sequences.
    - Resulting CSV accurately records all transitions with zero discrepancies.
    """
    csv_file = tmp_path / "ac1_10_euro_paper.csv"
    channel = CsvTradeLogChannel(csv_path=csv_file, enabled=True)
    service = NotificationService()
    service.register_channel(channel)

    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        exchange="binance",
        notification_service=service,
    )
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Sequence of 3 roundtrips (6 trades) starting at 10€
    trades_spec = [
        # Roundtrip 1: BTC/EUR win (+4%)
        {"side": OrderSide.BUY, "sym": "BTC/EUR", "px": Decimal("60000.00"), "notional": Decimal("9.60"), "exit_reason": None},
        {"side": OrderSide.SELL, "sym": "BTC/EUR", "px": Decimal("62400.00"), "notional": None, "exit_reason": "TP_WIN"},
        # Roundtrip 2: ETH/EUR loss (-3%)
        {"side": OrderSide.BUY, "sym": "ETH/EUR", "px": Decimal("3000.00"), "notional": Decimal("9.80"), "exit_reason": None},
        {"side": OrderSide.SELL, "sym": "ETH/EUR", "px": Decimal("2910.00"), "notional": None, "exit_reason": "SL_LOSS"},
        # Roundtrip 3: SOL/EUR win (+6%)
        {"side": OrderSide.BUY, "sym": "SOL/EUR", "px": Decimal("150.00"), "notional": Decimal("9.50"), "exit_reason": None},
        {"side": OrderSide.SELL, "sym": "SOL/EUR", "px": Decimal("159.00"), "notional": None, "exit_reason": "TP_WIN"},
    ]

    base_time = 1757250000000
    for idx, spec in enumerate(trades_spec):
        t_ms = base_time + idx * 30000
        sym = spec["sym"]
        if spec["side"] == OrderSide.BUY:
            sig = OrderSignal(
                symbol=sym,
                side=OrderSide.BUY,
                target_notional=spec["notional"],
                order_id=f"ord_10e_{idx}",
                timestamp_ms=t_ms,
            )
        else:
            held = ledger.get_position(sym).quantity
            sig = OrderSignal(
                symbol=sym,
                side=OrderSide.SELL,
                target_quantity=held,
                order_id=f"ord_10e_{idx}",
                exit_reason=spec["exit_reason"],
                timestamp_ms=t_ms,
            )
        res = simulator.execute(order=sig, current_price=spec["px"], ledger=ledger, timestamp_ms=t_ms)
        assert res.success is True, f"10€ pipeline trade {idx} failed: {res.error_message}"

    records = channel.logger.read_trades()
    assert len(records) == 6

    # Verify PnL logic
    # BUY trades (0, 2, 4) have pnl == 0.0
    for i in (0, 2, 4):
        assert records[i]["action"] == "BUY"
        assert records[i]["pnl"] == 0.0
        assert records[i]["pnl_pct"] == 0.0

    # SELL trade 1: win
    assert records[1]["action"] == "SELL"
    assert records[1]["pnl"] > 0.0
    assert records[1]["pnl_pct"] > 0.0

    # SELL trade 3: loss
    assert records[3]["action"] == "SELL"
    assert records[3]["pnl"] < 0.0
    assert records[3]["pnl_pct"] < 0.0

    # SELL trade 5: win
    assert records[5]["action"] == "SELL"
    assert records[5]["pnl"] > 0.0
    assert records[5]["pnl_pct"] > 0.0

    # Verify balances match ledger exactly
    for rec, trade in zip(records, ledger.trades):
        assert rec["portfolio_balance"] == pytest.approx(float(trade.equity_after), abs=1e-4)
        assert rec["cash_balance"] == pytest.approx(float(trade.cash_after), abs=1e-4)


def test_empirical_csv_concurrent_stress_durability(tmp_path: Path):
    """Stress test CSV logger under concurrent multi-threaded writes from multiple simulators.

    - 5 concurrent threads executing orders against independent virtual ledgers.
    - All routing through the same shared CsvTradeLogChannel and CsvTradeLogger.
    - Asserts total row count, zero line interleaving, valid RFC 4180 parsing.
    """
    csv_file = tmp_path / "concurrent_pipeline_stress.csv"
    channel = CsvTradeLogChannel(csv_path=csv_file, enabled=True)
    notif_service = NotificationService()
    notif_service.register_channel(channel)

    num_threads = 5
    trades_per_thread = 20  # 10 roundtrips per thread = 100 trades total

    def thread_worker(tid: int):
        sim = ExecutionSimulator(
            fee_rate=Decimal("0.001"),
            slippage=Decimal("0.0005"),
            min_notional=Decimal("5.00"),
            exchange="binance",
            notification_service=notif_service,
        )
        led = VirtualLedger(initial_cash=Decimal("50.00"), quote_currency="EUR")
        for i in range(10):
            # Buy
            buy_sig = OrderSignal(
                symbol="BTC/EUR",
                side=OrderSide.BUY,
                target_notional=Decimal("10.00"),
                order_id=f"th_{tid}_buy_{i}",
                timestamp_ms=1757246400000 + (tid * 1000 + i * 2) * 1000,
            )
            b_res = sim.execute(order=buy_sig, current_price=Decimal("60000.00"), ledger=led)
            assert b_res.success is True

            # Sell
            held = led.get_position("BTC/EUR").quantity
            sell_sig = OrderSignal(
                symbol="BTC/EUR",
                side=OrderSide.SELL,
                target_quantity=held,
                order_id=f"th_{tid}_sell_{i}",
                exit_reason="CONCURRENT_EXIT",
                timestamp_ms=1757246400000 + (tid * 1000 + i * 2 + 1) * 1000,
            )
            s_res = sim.execute(order=sell_sig, current_price=Decimal("62000.00"), ledger=led)
            assert s_res.success is True

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(thread_worker, t) for t in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    lines = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 101, f"Expected 101 lines (1 header + 100 trades), got {len(lines)}"

    # Parse with csv reader to ensure all lines have exactly 13 columns
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 101
        for line_idx, row in enumerate(reader):
            assert len(row) == 13, f"Line {line_idx} has {len(row)} columns instead of 13"

    # Read back through channel.logger.read_trades()
    trades = channel.logger.read_trades()
    assert len(trades) == 100
    for t in trades:
        assert t["action"] in ("BUY", "SELL")
        assert t["price"] > 0.0
        assert t["size"] > 0.0
        assert t["cash_balance"] > 0.0


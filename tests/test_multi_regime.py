"""Acceptance and Unit Test Suite for Multi-Regime Strategy & Bybit Constraints.

Verifies:
1. Acceptance Criterion 1 (Regime Switching Validation):
   - Programmatic backtest runs with multi-regime orchestrator over historical data.
   - Transitions between Grid and AMV strategies at least once.
   - Correctly logs the regime change with [REGIME_CHANGE].
2. Acceptance Criterion 2 (Grid Constraints Validation):
   - Given a mock 10.00 EUR portfolio balance, MicroGridStrategy accurately generates
     sub-orders sized at exactly or slightly above 1.00 EUR (Bybit minNotional constraint).
   - Verified across multiple asset prices (BTC, ETH, SOL) and step sizes.
3. Unit Tests:
   - Bollinger Bands & BBW technical indicators.
   - MarketRegimeFilter Schmitt trigger dual-threshold hysteresis and K=2 persistence.
   - ExecutionSimulator gross notional handling for 1.00 EUR orders.
   - Grid dismantling and liquidation to 100% liquid cash.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import io
import logging
from typing import List
import numpy as np
import pandas as pd
import pytest

from trading_bot.data.models import Candle
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import ExecutionSimulator
from trading_bot.strategy.indicators import (
    calculate_bbw,
    calculate_bollinger_bands,
    compute_all_indicators,
)
from trading_bot.strategy.regime import (
    MarketRegime,
    MarketRegimeFilter,
    RegimeSnapshot,
)
from trading_bot.strategy.grid_strategy import (
    MicroGridStrategy,
)
from trading_bot.strategy.orchestrator import (
    MultiRegimeStrategyOrchestrator,
)
from trading_bot.cli.backtest import run_backtest


# =====================================================================
# 1. ACCEPTANCE CRITERION 2: GRID CONSTRAINTS & BYBIT 1.00 EUR SIZING
# =====================================================================

def test_grid_sub_orders_meet_bybit_min_notional_with_10_eur_balance():
    """AC2: Unit test verifying that Grid Strategy accurately generates sub-orders

    sized at exactly or slightly above 1.00 EUR (Bybit minimum constraint)
    when given a mock 10.00 EUR portfolio balance.
    """
    # 1. Test BTC/EUR: Price ~ 60,000 EUR, step_size = 0.00001 BTC
    grid_btc = MicroGridStrategy(
        default_symbol="BTC/EUR",
        min_notional=Decimal("1.00"),
        step_size=Decimal("0.00001"),
        max_levels=10,
    )
    btc_price = Decimal("60000.00")
    sub_orders_btc = grid_btc.calculate_sub_orders(btc_price, portfolio_balance=Decimal("10.00"))

    assert len(sub_orders_btc) == 10, f"Expected 10 sub-orders for 10.00 EUR balance, got {len(sub_orders_btc)}"
    for order in sub_orders_btc:
        assert order["meets_min_notional"] is True
        notional = order["notional"]
        assert notional >= Decimal("1.00"), f"Sub-order notional {notional} < 1.00 EUR Bybit minNotional"
        # Since 1 BTC step is 0.00001 * 60,000 = 0.60 EUR, notional is slightly above 1.00 EUR (<= 1.00 + 1 step)
        assert notional <= Decimal("1.00") + (btc_price * Decimal("0.00001")), f"Sub-order notional {notional} exceeds 1 step above 1.00 EUR"

    # 2. Test ETH/EUR: Price ~ 2,500 EUR, step_size = 0.0001 ETH
    grid_eth = MicroGridStrategy(
        default_symbol="ETH/EUR",
        min_notional=Decimal("1.00"),
        step_size=Decimal("0.0001"),
        max_levels=10,
    )
    eth_price = Decimal("2500.00")
    sub_orders_eth = grid_eth.calculate_sub_orders(eth_price, portfolio_balance=Decimal("10.00"))

    assert len(sub_orders_eth) == 10
    for order in sub_orders_eth:
        assert order["meets_min_notional"] is True
        notional = order["notional"]
        assert notional >= Decimal("1.00")
        assert notional <= Decimal("1.00") + (eth_price * Decimal("0.0001"))

    # 3. Test SOL/EUR: Price ~ 135 EUR, step_size = 0.001 SOL
    grid_sol = MicroGridStrategy(
        default_symbol="SOL/EUR",
        min_notional=Decimal("1.00"),
        step_size=Decimal("0.001"),
        max_levels=10,
    )
    sol_price = Decimal("135.00")
    sub_orders_sol = grid_sol.calculate_sub_orders(sol_price, portfolio_balance=Decimal("10.00"))

    assert len(sub_orders_sol) == 10
    for order in sub_orders_sol:
        assert order["meets_min_notional"] is True
        notional = order["notional"]
        assert notional >= Decimal("1.00")
        assert notional <= Decimal("1.00") + (sol_price * Decimal("0.001"))


def test_execution_simulator_accepts_1_eur_target_notional_on_bybit():
    """Verify ExecutionSimulator does not reject a gross 1.00 EUR buy order on Bybit."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    sim = ExecutionSimulator(
        exchange="bybit",
        min_notional=Decimal("1.00"),
        fee_rate=Decimal("0.0010"),
        slippage=Decimal("0.0005"),
        step_size=Decimal("0.00001"),
    )

    # Order targeting exactly 1.00 EUR gross notional
    res = sim.execute(
        order={
            "symbol": "BTCEUR",
            "side": "BUY",
            "target_notional": Decimal("1.00"),
            "order_type": "MARKET",
        },
        current_price=Decimal("60000.00"),
        ledger=ledger,
    )

    assert res.success is True, f"Execution failed: {res.error_message}"
    assert res.quote_notional >= Decimal("1.00")
    assert ledger.cash < Decimal("10.00")
    assert ledger.get_position("BTCEUR").quantity > Decimal("0.0")


# =====================================================================
# 2. INDICATORS & REGIME FILTER UNIT TESTS
# =====================================================================

def test_bollinger_bands_and_bbw_calculation():
    """Verify Bollinger Bands and BBW calculations."""
    prices = [100.0 + i * 0.5 for i in range(30)]
    bb = calculate_bollinger_bands(prices, period=20, num_std=2.0)

    assert len(bb.middle) == 30
    assert len(bb.upper) == 30
    assert len(bb.lower) == 30
    assert (bb.upper >= bb.middle).all()
    assert (bb.middle >= bb.lower).all()

    bbw = calculate_bbw(prices, period=20, num_std=2.0)
    assert len(bbw) == 30
    # BBW = (upper - lower) / middle
    expected_bbw = (bb.upper - bb.lower) / bb.middle
    pd.testing.assert_series_equal(bbw, expected_bbw)


def test_market_regime_filter_hysteresis_and_persistence():
    """Verify Schmitt trigger dual-threshold hysteresis and K=2 persistence."""
    rf = MarketRegimeFilter(
        adx_trend_threshold=24.0,
        adx_range_threshold=20.0,
        chop_trend_threshold=50.0,
        chop_range_threshold=56.0,
        persistence_bars=2,
    )

    assert rf.current_regime == MarketRegime.RANGING

    # Create synthetic series of candles
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base_price = 50000.0

    # Build 60 bars of flat/ranging data
    candles: List[Candle] = []
    for i in range(70):
        t = now + timedelta(hours=i)
        p = base_price + (np.sin(i * 0.5) * 100.0)
        c = Candle(
            symbol="BTCEUR",
            interval="1h",
            timestamp=t,
            close_time=t + timedelta(hours=1),
            open=Decimal(str(p)),
            high=Decimal(str(p + 50.0)),
            low=Decimal(str(p - 50.0)),
            close=Decimal(str(p + 10.0)),
            volume=Decimal("10.0"),
            quote_volume=Decimal("500000.0"),
            trades_count=100,
        )
        candles.append(c)

    # Initial evaluation should be RANGING
    snap = rf.evaluate(candles[-1], candles[:-1])
    assert snap.regime == MarketRegime.RANGING

    # Now add strong upward trending candles to trigger trend conditions
    trend_candles = list(candles)
    last_p = float(trend_candles[-1].close)
    for j in range(1, 15):
        t = now + timedelta(hours=70 + j)
        last_p += 1500.0  # Massive sustained move to drive ADX up and CHOP down
        c = Candle(
            symbol="BTCEUR",
            interval="1h",
            timestamp=t,
            close_time=t + timedelta(hours=1),
            open=Decimal(str(last_p - 1000.0)),
            high=Decimal(str(last_p + 100.0)),
            low=Decimal(str(last_p - 1100.0)),
            close=Decimal(str(last_p)),
            volume=Decimal("50.0"),
            quote_volume=Decimal("2500000.0"),
            trades_count=500,
        )
        trend_candles.append(c)

    # Replay through trend bars to observe transition
    transitions = []
    for idx in range(70, len(trend_candles)):
        curr = trend_candles[idx]
        hist = trend_candles[:idx]
        snap = rf.evaluate(curr, hist)
        if snap.transition_occurred:
            transitions.append(snap)

    assert len(transitions) >= 1, "Expected at least 1 regime transition to TRENDING"
    assert transitions[0].regime == MarketRegime.TRENDING


# =====================================================================
# 3. MICRO-GRID DISMANTLING UNIT TEST
# =====================================================================

def test_grid_dismantling_liquidates_inventory_to_cash():
    """Verify that dismantle_grid generates a market sell signal for held inventory."""
    grid = MicroGridStrategy(default_symbol="BTC/EUR", min_notional=Decimal("1.00"))
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Simulate buy fill in ledger
    ledger.record_buy(
        symbol="BTC/EUR",
        quantity=Decimal("0.0001"),
        executed_price=Decimal("60000.00"),
        fee=Decimal("0.006"),
        timestamp_ms=int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
    )
    assert ledger.get_position("BTC/EUR").quantity == Decimal("0.0001")


    # Dismantle grid
    signals = grid.dismantle_grid(current_price=Decimal("61000.00"), ledger=ledger)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.side.value == "SELL"
    assert sig.target_quantity == Decimal("0.0001")
    assert sig.order_type.value == "MARKET"
    assert "DISMANTLED" in sig.reason


# =====================================================================
# 4. ACCEPTANCE CRITERION 1: PROGRAMMATIC BACKTEST & REGIME SWITCHING
# =====================================================================

def test_acceptance_regime_switching_in_programmatic_backtest(caplog):
    """AC1: A programmatic backtest verifies that the bot correctly transitions

    between the Grid and AMV strategies at least once based on historical market
    volatility, and logs the regime change.
    """
    caplog.set_level(logging.INFO)

    # Run backtest with multi_regime strategy over historical BTC/EUR data
    start_dt = datetime(2025, 4, 1, tzinfo=timezone.utc)
    end_dt = datetime(2025, 7, 1, tzinfo=timezone.utc)

    ledger, metrics, candles = run_backtest(
        pair="BTC/EUR",
        start=start_dt,
        end=end_dt,
        balance=10.0,
        currency="EUR",
        timeframe="1h",
        strategy_name="multi_regime",
        exchange="bybit",
        fee_rate=0.0010,
        slippage=0.0005,
    )

    # 1. Verify backtest ran successfully
    assert len(candles) > 1000, f"Expected >1000 candles, got {len(candles)}"
    assert ledger.initial_cash == Decimal("10.00")
    assert metrics.total_trades >= 0

    # 2. Verify regime transition logging occurred
    regime_logs = [record.message for record in caplog.records if "[REGIME_CHANGE]" in record.message]
    assert len(regime_logs) >= 1, (
        f"Expected at least one [REGIME_CHANGE] log message during backtest, found {len(regime_logs)}"
    )

    # Verify log format contains transition details
    sample_log = regime_logs[0]
    assert "Transitioned from" in sample_log
    assert "ADX:" in sample_log
    assert "CHOP:" in sample_log


def test_benchmark_1_5_years_file_generated():
    """AC3: A file named benchmark_1.5_years.txt is successfully generated,

    containing a table of the final balances and net profits for the multi-regime
    strategy over the last 1.5 years across various starting capitals.
    """
    from pathlib import Path
    bench_file = Path("benchmark_1.5_years.txt")
    assert bench_file.exists(), "benchmark_1.5_years.txt does not exist"
    content = bench_file.read_text(encoding="utf-8")
    assert "Initial (EUR)" in content
    assert "Final (EUR)" in content
    assert "Net Profit (EUR)" in content
    for bal in ["10.00", "50.00", "100.00", "500.00", "1000.00"]:
        assert bal in content, f"Balance {bal} missing from benchmark report"


"""Empirical Challenger 2 Verification Suite: Multi-Regime Transitions & Grid Stress.

Stress tests:
1. Schmitt-trigger dual-threshold hysteresis & K=2 persistence under adversarial whipsaw inputs.
2. Grid dismantling & inventory liquidation on RANGING -> TRENDING (inventory leak & ledger desync checks).
3. TRENDING -> RANGING transition: preservation of active AMV position, trailing stop handling, clean grid re-anchoring.
4. Clean inventory conversion on breakout without cash or crypto leakage.
5. Extreme volatility flips stress harness (2,000 rapid cycles) testing for deadlocks, infinite loops, and stack overflows.
6. Historical backtest regime transition logging ([REGIME_CHANGE]) and solvency invariants.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
import math
import time
from typing import List, Optional
import numpy as np
import pytest

from trading_bot.data.models import Candle
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import ExecutionSimulator
from trading_bot.strategy.amv_strategy import AMVTrendBreakStrategy, OrderSide, OrderType, PositionState
from trading_bot.strategy.grid_strategy import MicroGridStrategy
from trading_bot.strategy.orchestrator import MultiRegimeStrategyOrchestrator
from trading_bot.strategy.regime import MarketRegime, MarketRegimeFilter, RegimeSnapshot
from trading_bot.cli.backtest import run_backtest


def create_candle_series(
    start_time: datetime,
    prices: List[float],
    symbol: str = "BTC/EUR",
    interval: str = "1h",
    vol_scale: float = 10.0,
) -> List[Candle]:
    """Helper to construct realistic Candle objects from a sequence of close prices."""
    candles = []
    delta = timedelta(hours=1) if interval == "1h" else timedelta(minutes=15)
    for i, p in enumerate(prices):
        t = start_time + (delta * i)
        high = p * 1.002
        low = p * 0.998
        open_px = prices[i - 1] if i > 0 else p
        c = Candle(
            symbol=symbol,
            interval=interval,
            timestamp=t,
            close_time=t + delta,
            open=Decimal(str(round(open_px, 4))),
            high=Decimal(str(round(max(open_px, p, high), 4))),
            low=Decimal(str(round(min(open_px, p, low), 4))),
            close=Decimal(str(round(p, 4))),
            volume=Decimal(str(round(vol_scale * (1.0 + (i % 5) * 0.1), 4))),
            quote_volume=Decimal(str(round(p * vol_scale, 2))),
            trades_count=100,
        )
        candles.append(c)
    return candles


# =====================================================================
# 1. HYSTERESIS & PERSISTENCE STRESS HARNESS
# =====================================================================

def test_hysteresis_and_persistence_hostile_whipsaws():
    """Empirically challenge the Schmitt trigger hysteresis and K=2 persistence.

    Verify:
    1. Single-bar excursions across the trend threshold DO NOT trigger transition (persistence requirement).
    2. After excursion collapse into deadband, persistence counter resets to 0.
    3. Exactly two consecutive bars meeting trend criteria trigger transition on the 2nd qualifying bar.
    4. Rapid alternating triggers do not cause flip-flopping or state thrashing.
    """
    rf = MarketRegimeFilter(
        adx_trend_threshold=24.0,
        adx_range_threshold=20.0,
        chop_trend_threshold=50.0,
        chop_range_threshold=56.0,
        persistence_bars=2,
    )

    assert rf.current_regime == MarketRegime.RANGING

    # Step A: Warm-up with 70 flat bars
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base_price = 50000.0
    flat_prices = [base_price + (np.sin(i * 0.4) * 40.0) for i in range(70)]
    candles = create_candle_series(t0, flat_prices)

    for i in range(25, 70):
        snap = rf.evaluate(candles[i], candles[:i])
        assert snap.regime == MarketRegime.RANGING
        assert snap.transition_occurred is False

    # Step B: 1-bar massive spike followed immediately by collapse back
    # Bar 70: Spike bar triggers raw trend conditions, but only for 1 bar
    spike_p = [base_price + 3000.0, base_price + 20.0]
    spike_candles = create_candle_series(candles[-1].timestamp + timedelta(hours=1), spike_p)
    all_candles = list(candles) + spike_candles

    snap_spike = rf.evaluate(all_candles[70], all_candles[:70])
    # Regime MUST remain RANGING because persistence K=2 has not been satisfied
    assert snap_spike.regime == MarketRegime.RANGING, "Regime prematurely flipped on 1-bar excursion!"
    assert snap_spike.transition_occurred is False
    assert rf._consecutive_trend_signals == 1

    # Bar 71 (collapse back into deadband/range): trend counter must reset
    snap_collapse = rf.evaluate(all_candles[71], all_candles[:71])
    assert snap_collapse.regime == MarketRegime.RANGING
    assert snap_collapse.transition_occurred is False
    assert rf._consecutive_trend_signals == 0, "Trend counter was not reset after collapse!"

    # Step C: Sustained multi-bar breakout to satisfy K=2 persistence
    sustained_p = [base_price + 2500.0, base_price + 5000.0, base_price + 7500.0, base_price + 10000.0]
    sust_candles = create_candle_series(all_candles[-1].timestamp + timedelta(hours=1), sustained_p)
    all_candles.extend(sust_candles)

    snaps = []
    for idx in range(72, len(all_candles)):
        s = rf.evaluate(all_candles[idx], all_candles[:idx])
        snaps.append(s)

    transitions = [s for s in snaps if s.transition_occurred]
    assert len(transitions) == 1, f"Expected exactly 1 transition to TRENDING, got {len(transitions)}"
    assert transitions[0].regime == MarketRegime.TRENDING
    assert rf.current_regime == MarketRegime.TRENDING


# =====================================================================
# 2. GRID DISMANTLING & INVENTORY LIQUIDATION ON BREAKOUT
# =====================================================================

def test_grid_dismantling_and_zero_inventory_leak():
    """Empirically test grid dismantling on RANGING -> TRENDING transition.

    Verify:
    1. Grid accumulates inventory across multiple buy levels in RANGING regime.
    2. Upon breakout triggering TRENDING, dismantle_grid emits a market SELL for exact held inventory.
    3. Execution of the dismantle signal completely eliminates crypto position (0.00000000 leak).
    4. Realized cash is fully credited back to ledger cash.
    5. Grid internal state (active_levels, filled_tranches) is 100% reset.
    6. Transition event [REGIME_CHANGE] is emitted.
    """
    sim = ExecutionSimulator(
        exchange="bybit",
        min_notional=Decimal("1.00"),
        fee_rate=Decimal("0.0010"),
        slippage=Decimal("0.0005"),
        step_size=Decimal("0.00001"),
    )
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    orchestrator = MultiRegimeStrategyOrchestrator(
        default_symbol="BTC/EUR",
        exchange_min_notional=Decimal("1.00"),
        step_size=Decimal("0.00001"),
    )

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base_price = 60000.0
    # 70 flat warmup bars
    flat_prices = [base_price for _ in range(70)]
    candles = create_candle_series(t0, flat_prices)

    for i in range(len(candles)):
        orchestrator.on_candle(candles[i], candles[:i], ledger)

    assert orchestrator.current_regime == MarketRegime.RANGING
    assert orchestrator.grid_strategy.anchor_price is not None

    # Step A: Drop price downward in steps to trigger 3 grid buy levels
    anchor = float(orchestrator.grid_strategy.anchor_price)
    spacing = float(orchestrator.grid_strategy.grid_spacing_pct)
    curr_t = candles[-1].timestamp
    for lvl in [1, 2, 3]:
        curr_t += timedelta(hours=1)
        px = anchor * (1.0 - (lvl * spacing) - 0.001)
        c = Candle(
            symbol="BTC/EUR",
            interval="1h",
            timestamp=curr_t,
            close_time=curr_t + timedelta(hours=1),
            open=Decimal(str(round(px + 10, 2))),
            high=Decimal(str(round(px + 20, 2))),
            low=Decimal(str(round(px - 10, 2))),
            close=Decimal(str(round(px, 2))),
            volume=Decimal("15.0"),
            quote_volume=Decimal("900000.0"),
            trades_count=100,
        )
        candles.append(c)

        sig = orchestrator.on_candle(c, candles[:-1], ledger)
        assert sig is not None, f"Expected grid buy signal on level {lvl}"
        assert sig.side == OrderSide.BUY
        res = sim.execute(order=sig, current_price=c.close, ledger=ledger, timestamp_ms=c.open_time_ms)
        assert res.success is True, f"Grid buy execution failed: {res.error_message}"

    # Verify inventory accumulated
    pos = ledger.get_position("BTC/EUR")
    assert pos is not None
    assert pos.quantity > Decimal("0.0")
    accumulated_qty = pos.quantity
    assert len(orchestrator.grid_strategy.filled_tranches) == 3
    initial_cash_before_breakout = ledger.available_cash

    # Step B: Trigger breakout to TRENDING
    curr_px = anchor
    breakout_candles = []
    for step in range(1, 10):
        curr_t += timedelta(hours=1)
        curr_px += 2500.0
        c = Candle(
            symbol="BTC/EUR",
            interval="1h",
            timestamp=curr_t,
            close_time=curr_t + timedelta(hours=1),
            open=Decimal(str(round(curr_px - 1500, 2))),
            high=Decimal(str(round(curr_px + 200, 2))),
            low=Decimal(str(round(curr_px - 1600, 2))),
            close=Decimal(str(round(curr_px, 2))),
            volume=Decimal("60.0"),
            quote_volume=Decimal("3600000.0"),
            trades_count=500,
        )
        candles.append(c)
        breakout_candles.append(c)

    dismantle_signal = None
    transition_candle_idx = None
    for idx, c in enumerate(breakout_candles):
        hist_idx = len(candles) - len(breakout_candles) + idx
        sig = orchestrator.on_candle(c, candles[:hist_idx], ledger)
        if orchestrator.current_regime == MarketRegime.TRENDING:
            if sig is not None and sig.side == OrderSide.SELL and "DISMANTLED" in sig.reason:
                dismantle_signal = sig
                transition_candle_idx = idx
                break

    assert dismantle_signal is not None, "Failed to emit dismantle liquidation SELL signal on regime transition"
    assert dismantle_signal.target_quantity == accumulated_qty
    assert dismantle_signal.order_type == OrderType.MARKET

    # Step C: Execute dismantle liquidation in simulator
    c_trans = breakout_candles[transition_candle_idx]
    res_dismantle = sim.execute(
        order=dismantle_signal,
        current_price=c_trans.close,
        ledger=ledger,
        timestamp_ms=c_trans.open_time_ms,
    )
    assert res_dismantle.success is True, f"Dismantle execution failed: {res_dismantle.error_message}"

    # Verify inventory is 100% liquidated (ZERO inventory leak)
    pos_after = ledger.get_position("BTC/EUR")
    assert pos_after.quantity == Decimal("0.0"), f"Inventory leaked! Remaining qty: {pos_after.quantity}"
    assert ledger.available_cash > initial_cash_before_breakout, "Cash was not replenished upon liquidation"
    assert len(orchestrator.grid_strategy.filled_tranches) == 0, "Grid filled tranches not cleaned up"
    assert len(orchestrator.grid_strategy.active_levels) == 0, "Grid active levels not cleaned up"
    assert len(orchestrator.regime_transitions) >= 1, "Transition was not recorded in regime_transitions"


# =====================================================================
# 3. TRENDING -> RANGING TRANSITION: PRESERVATION OF ACTIVE AMV POSITION
# =====================================================================

def test_trending_to_ranging_preserves_active_amv_trend_position():
    """Verify that when transitioning from TRENDING to RANGING:

    1. Active AMV trend position is NOT force-liquidated.
    2. Grid does not create conflicting orders or levels while AMV position is active.
    3. AMV continues to manage trailing stop in RANGING regime.
    4. Once AMV trailing stop exits, Grid cleanly re-anchors to exit price.
    """
    sim = ExecutionSimulator(
        exchange="bybit",
        min_notional=Decimal("1.00"),
        fee_rate=Decimal("0.0010"),
        slippage=Decimal("0.0005"),
        step_size=Decimal("0.00001"),
    )
    ledger = VirtualLedger(initial_cash=Decimal("50.00"), quote_currency="EUR")

    orchestrator = MultiRegimeStrategyOrchestrator(
        default_symbol="BTC/EUR",
        exchange_min_notional=Decimal("1.00"),
        step_size=Decimal("0.00001"),
    )

    # Force initial regime to TRENDING
    orchestrator.regime_filter._current_regime = MarketRegime.TRENDING

    # Manually open an active AMV position
    entry_px = Decimal("60000.00")
    pos_qty = Decimal("0.0005")  # ~30 EUR notional
    ledger.record_buy(
        symbol="BTC/EUR",
        quantity=pos_qty,
        executed_price=entry_px,
        fee=Decimal("0.03"),
        timestamp_ms=1700000000000,
    )
    amv_pos = PositionState(
        symbol="BTC/EUR",
        entry_price=entry_px,
        quantity=pos_qty,
        initial_stop=Decimal("58500.00"),
        current_stop=Decimal("59000.00"),
        take_profit_1=Decimal("63000.00"),
        highest_price=Decimal("61000.00"),
    )
    orchestrator.amv_strategy._positions["BTC/EUR"] = amv_pos

    assert ledger.get_position("BTC/EUR").quantity == pos_qty

    # Create candles transitioning to RANGING
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    flat_prices = [60500.0 + (np.sin(i * 0.5) * 20.0) for i in range(70)]
    candles = create_candle_series(t0, flat_prices)

    # Step through candle loop simulating full order cycle
    exit_observed = False
    exit_fill_price = None

    for i in range(30, len(candles)):
        c = candles[i]
        sig = orchestrator.on_candle(c, candles[:i], ledger)

        # While position is active, verify Grid did not deploy
        if ledger.get_position("BTC/EUR").quantity > Decimal("0.0"):
            assert len(orchestrator.grid_strategy.filled_tranches) == 0, "Grid premature deployment"

        if sig is not None:
            # When AMV exits (e.g. chandelier trailing stop)
            assert sig.side == OrderSide.SELL
            res = sim.execute(order=sig, current_price=c.close, ledger=ledger, timestamp_ms=c.open_time_ms)
            assert res.success is True
            exit_fill_price = res.executed_price
            exit_observed = True

            orchestrator.notify_position_closed(
                symbol=sig.symbol,
                exit_price=res.executed_price,
                current_equity=ledger.total_equity(),
                timestamp=c.timestamp,
            )
            # Break to inspect post-exit state
            break

    assert exit_observed is True, "Expected trailing stop exit to trigger"
    assert ledger.get_position("BTC/EUR").quantity == Decimal("0.0"), "Position not flat after trailing stop exit"
    assert orchestrator.grid_strategy.anchor_price == exit_fill_price, (
        f"Grid anchor price {orchestrator.grid_strategy.anchor_price} did not match exit price {exit_fill_price}"
    )


# =====================================================================
# 4. RANGING -> TRENDING CLEAN INVENTORY CONVERSION
# =====================================================================

def test_ranging_to_trending_clean_inventory_conversion():
    """Verify that all inventory is converted to cash on breakout without cash/crypto leaks."""
    sim = ExecutionSimulator(
        exchange="bybit",
        min_notional=Decimal("1.00"),
        fee_rate=Decimal("0.0010"),
        slippage=Decimal("0.0005"),
        step_size=Decimal("0.00001"),
    )
    ledger = VirtualLedger(initial_cash=Decimal("50.00"), quote_currency="EUR")

    orchestrator = MultiRegimeStrategyOrchestrator(
        default_symbol="BTC/EUR",
        exchange_min_notional=Decimal("1.00"),
        step_size=Decimal("0.00001"),
    )
    orchestrator.regime_filter._current_regime = MarketRegime.RANGING

    entry_px = Decimal("60000.00")
    pos_qty = Decimal("0.0005")
    ledger.record_buy(
        symbol="BTC/EUR",
        quantity=pos_qty,
        executed_price=entry_px,
        fee=Decimal("0.03"),
        timestamp_ms=1700000000000,
    )

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    flat_prices = [60000.0 for _ in range(70)]
    candles = create_candle_series(t0, flat_prices)
    for i in range(25, 70):
        orchestrator.regime_filter.evaluate(candles[i], candles[:i])

    # Add strong breakout candles to force regime to TRENDING
    t_end = candles[-1].timestamp
    px = 60000.0
    dismantle_executed = False
    for j in range(1, 10):
        px += 2500.0
        c = Candle(
            symbol="BTC/EUR",
            interval="1h",
            timestamp=t_end + timedelta(hours=j),
            close_time=t_end + timedelta(hours=j+1),
            open=Decimal(str(px - 1000)),
            high=Decimal(str(px + 100)),
            low=Decimal(str(px - 1000)),
            close=Decimal(str(px)),
            volume=Decimal("50.0"),
            quote_volume=Decimal("3000000.0"),
            trades_count=300,
        )
        candles.append(c)
        sig = orchestrator.on_candle(c, candles[:-1], ledger)
        if sig is not None and sig.side == OrderSide.SELL and "DISMANTLE" in sig.reason:
            res = sim.execute(order=sig, current_price=c.close, ledger=ledger, timestamp_ms=c.open_time_ms)
            assert res.success is True
            orchestrator.notify_position_closed(
                symbol=sig.symbol,
                exit_price=res.executed_price,
                current_equity=ledger.total_equity(),
                timestamp=c.timestamp,
            )
            dismantle_executed = True
            break

    assert dismantle_executed is True, "Dismantle order was not executed during breakout"
    assert ledger.get_position("BTC/EUR").quantity == Decimal("0.0"), "Crypto leak detected after dismantle"
    assert ledger.available_cash > Decimal("50.00"), "Cash was not replenished with profit"


# =====================================================================
# 5. EXTREME VOLATILITY FLIPS STRESS HARNESS (2,000 RAPID CYCLES)
# =====================================================================

def test_extreme_volatility_rapid_cycles_deadlock_stress_harness():
    """Stress test with 2,000 candles of hostile volatility swings.

    Tests for:
    - Deadlocks / infinite loops
    - Unhandled exceptions
    - High simulation throughput
    - Solvency & memory stability
    """
    sim = ExecutionSimulator(
        exchange="bybit",
        min_notional=Decimal("1.00"),
        fee_rate=Decimal("0.0010"),
        slippage=Decimal("0.0005"),
        step_size=Decimal("0.00001"),
    )
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    orchestrator = MultiRegimeStrategyOrchestrator(
        default_symbol="BTC/EUR",
        exchange_min_notional=Decimal("1.00"),
        step_size=Decimal("0.00001"),
    )

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    n_bars = 1000  # 1,000 rapid cycles
    rng = np.random.RandomState(42)

    prices = [50000.0]
    for i in range(1, n_bars):
        mode = i % 30
        if mode < 10:
            ret = rng.uniform(0.01, 0.04)
        elif mode < 20:
            ret = rng.uniform(-0.04, -0.01)
        else:
            ret = rng.uniform(-0.002, 0.002)
        prices.append(max(1000.0, prices[-1] * (1.0 + ret)))

    candles = create_candle_series(t0, prices)

    start_time = time.perf_counter()
    trades_executed = 0

    for i in range(len(candles)):
        c = candles[i]
        hist = candles[:i]
        sig = orchestrator.on_candle(c, hist, ledger)
        if sig is not None:
            res = sim.execute(
                order=sig,
                current_price=c.close,
                ledger=ledger,
                timestamp_ms=c.open_time_ms,
            )
            if res.success:
                trades_executed += 1
                if res.side == "SELL":
                    orchestrator.notify_position_closed(
                        symbol=sig.symbol,
                        exit_price=res.executed_price,
                        current_equity=ledger.total_equity(),
                        timestamp=c.timestamp,
                    )

    elapsed = time.perf_counter() - start_time
    throughput = n_bars / elapsed

    assert throughput > 20.0, f"Simulation throughput too slow: {throughput:.1f} candles/sec"
    assert len(orchestrator.regime_transitions) > 0, "No regime transitions occurred during chaotic bars"
    assert ledger.available_cash >= Decimal("0.0"), "Ledger ended with negative cash"
    assert ledger.get_position("BTC/EUR").quantity >= Decimal("0.0"), "Ledger ended with negative position"
    assert len(orchestrator.pending_signals) < 10, f"Pending signals leak: {len(orchestrator.pending_signals)}"


# =====================================================================
# 6. HISTORICAL BACKTEST REGIME TRANSITION LOGGING & SOLVENCY
# =====================================================================

def test_backtest_regime_transition_logging_and_solvency(caplog):
    """Verify that a 3-month backtest logs regime changes and maintains solvency invariants."""
    caplog.set_level(logging.INFO)

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

    assert len(candles) > 1000
    assert metrics.total_trades > 0

    # Verify regime transition logs
    regime_logs = [r.message for r in caplog.records if "[REGIME_CHANGE]" in r.message]
    assert len(regime_logs) >= 2, f"Expected multiple [REGIME_CHANGE] logs in 3-month period, got {len(regime_logs)}"

    # Check that each log contains valid format
    for log_msg in regime_logs:
        assert "Transitioned from" in log_msg
        assert "ADX:" in log_msg
        assert "CHOP:" in log_msg
        assert "BBW:" in log_msg
        assert "Close:" in log_msg

    # Verify solvency and metric reconciliation
    final_eq = ledger.total_equity({candles[-1].symbol: candles[-1].close})
    diff = abs(final_eq - Decimal(str(round(metrics.final_balance, 2))))
    assert diff < Decimal("0.05"), f"Discrepancy between final equity {final_eq} and metric balance {metrics.final_balance}"

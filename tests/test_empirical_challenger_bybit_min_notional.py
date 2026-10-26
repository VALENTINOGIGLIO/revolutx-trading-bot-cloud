"""Empirical Challenger Stress Suite: Bybit minNotional & Micro-Grid Constraints.

Adversarially tests:
1. Micro-balances: 1.00 EUR, 5.00 EUR, 10.00 EUR, 10.01 EUR, 15.00 EUR.
2. Extreme prices: BTC at 10,000 EUR to 150,000 EUR, micro-caps at 0.000001 EUR to 1E-12 EUR.
3. Edge-case step sizes and price tick sizes.
4. ExecutionSimulator Bybit execution without below-minimum-notional rejections.
5. Sizing desynchronization during candle drops in on_candle.
6. Strategy-ledger state synchronization.
"""

from datetime import datetime, timezone
from decimal import Decimal
import pytest

from trading_bot.data.models import Candle
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import ExecutionSimulator
from trading_bot.strategy.amv_strategy import OrderSide
from trading_bot.strategy.grid_strategy import MicroGridStrategy


# ============================================================================
# 1. MICRO-BALANCES & SUB-ORDER GENERATION (INVARIANT CHECKS)
# ============================================================================

@pytest.mark.parametrize("balance", [
    Decimal("1.00"),
    Decimal("5.00"),
    Decimal("10.00"),
    Decimal("10.01"),
    Decimal("15.00"),
])
@pytest.mark.parametrize("price,step_size", [
    (Decimal("60000.00"), Decimal("0.00001")),  # BTC
    (Decimal("2500.00"), Decimal("0.0001")),    # ETH
    (Decimal("135.00"), Decimal("0.001")),       # SOL
    (Decimal("1.00"), Decimal("0.01")),          # Token 1
    (Decimal("0.000001"), Decimal("1.0")),       # Micro-cap
])
def test_micro_balances_sub_orders_all_meet_min_notional(balance, price, step_size):
    """Verify that for all micro balances and price/step_size pairs, EVERY sub-order has notional >= 1.00 EUR."""
    grid = MicroGridStrategy(
        default_symbol="TEST/EUR",
        min_notional=Decimal("1.00"),
        step_size=step_size,
    )
    orders = grid.calculate_sub_orders(price, portfolio_balance=balance)
    assert len(orders) >= 1
    for order in orders:
        assert order["meets_min_notional"] is True
        assert order["notional"] >= Decimal("1.00"), (
            f"Sub-order notional {order['notional']} < 1.00 EUR for bal={balance}, price={price}, step={step_size}"
        )


# ============================================================================
# 2. EXTREME PRICES: BTC 10,000 EUR TO 150,000 EUR
# ============================================================================

@pytest.mark.parametrize("btc_price", [
    Decimal("10000.00"),
    Decimal("30000.00"),
    Decimal("60000.00"),
    Decimal("100000.00"),
    Decimal("150000.00"),
])
def test_extreme_btc_prices_grid_generation_and_execution(btc_price):
    """Verify BTC across extreme price regimes (10k to 150k) executes cleanly on Bybit simulator."""
    step_size = Decimal("0.00001")
    tick_size = Decimal("0.01")
    grid = MicroGridStrategy(
        default_symbol="BTC/EUR",
        min_notional=Decimal("1.00"),
        step_size=step_size,
        max_levels=10,
    )
    signals = grid.generate_grid_levels(btc_price, portfolio_balance=Decimal("10.00"))
    assert len(signals) == 10

    sim = ExecutionSimulator(
        exchange="bybit",
        min_notional=Decimal("1.00"),
        step_size=step_size,
        tick_size=tick_size,
        fee_rate=Decimal("0.0010"),
        slippage=Decimal("0.0005"),
    )

    for sig in signals:
        ledger = VirtualLedger(initial_cash=Decimal("100.00"), quote_currency="EUR")
        res = sim.execute(order=sig, current_price=sig.price, ledger=ledger)
        assert res.success is True, f"Failed executing BTC grid order at {btc_price}: {res.error_message}"
        assert res.quote_notional >= Decimal("1.00")


# ============================================================================
# 3. DEFECT 1: SIZING MISMATCH ON CANDLE DROP IN ON_CANDLE
# ============================================================================

def test_on_candle_order_gross_notional_meets_bybit_min_notional_when_candle_drops():
    """DEFECT 1 EMPIRICAL REPRODUCTION:

    When candle triggers a grid level (c_low <= level_price) but closes lower (c_close < level_price),
    the emitted OrderSignal must still execute with gross notional >= 1.00 EUR on Bybit simulator.
    Currently, quantity is sized at level_price, causing executed notional at c_close to fall below 1.00 EUR
    and triggering a Bybit below-minimum-notional rejection.
    """
    token_price = Decimal("100.00")
    step_size = Decimal("0.0001")
    grid = MicroGridStrategy(
        default_symbol="ETH/EUR",
        min_notional=Decimal("1.00"),
        step_size=step_size,
    )
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Candle 1 anchors grid at 100.00 EUR
    c1 = Candle(
        symbol="ETH/EUR", interval="1h",
        timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        close_time=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        open=Decimal("100.00"), high=Decimal("100.00"), low=Decimal("100.00"), close=Decimal("100.00"),
        volume=Decimal("1.0"), quote_volume=Decimal("100.0"), trades_count=10,
    )
    grid.on_candle(c1, [], ledger)

    # Candle 2 drops: level 1 is at 99.20 EUR.
    # The candle low reaches 95.00 EUR and closes at 95.00 EUR.
    c2 = Candle(
        symbol="ETH/EUR", interval="1h",
        timestamp=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        close_time=datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc),
        open=Decimal("100.00"), high=Decimal("100.00"), low=Decimal("95.00"), close=Decimal("95.00"),
        volume=Decimal("1.0"), quote_volume=Decimal("95.0"), trades_count=10,
    )
    sig = grid.on_candle(c2, [c1], ledger)
    assert sig is not None, "Expected buy OrderSignal for triggered grid level 1"

    # Execute on Bybit simulator
    sim = ExecutionSimulator(
        exchange="bybit",
        min_notional=Decimal("1.00"),
        step_size=step_size,
        tick_size=Decimal("0.01"),
        fee_rate=Decimal("0.0010"),
        slippage=Decimal("0.0005"),
    )
    res = sim.execute(order=sig, current_price=sig.price, ledger=ledger)

    # EMPIRICAL ASSERTION: The order must NOT be rejected for below minimum notional!
    assert res.success is True, f"Order was rejected on Bybit: {res.error_message}"
    assert res.quote_notional >= Decimal("1.00"), f"Executed notional {res.quote_notional} < 1.00 EUR"


# ============================================================================
# 4. DEFECT 2: TICK-SIZE SKEW & BYBIT EXECUTION REJECTION
# ============================================================================

@pytest.mark.parametrize("price,step_size,tick_size", [
    (Decimal("10.00"), Decimal("0.0001"), Decimal("0.10")),
    (Decimal("5.00"), Decimal("0.0001"), Decimal("0.05")),
    (Decimal("1.00"), Decimal("0.0001"), Decimal("0.01")),
    (Decimal("100.00"), Decimal("0.00001"), Decimal("1.00")),
    (Decimal("0.10"), Decimal("0.001"), Decimal("0.001")),
    (Decimal("0.000001"), Decimal("100.0"), Decimal("0.00000001")),
])
def test_execution_simulator_never_rejects_grid_orders_due_to_tick_size_rounding(price, step_size, tick_size):
    """DEFECT 2 EMPIRICAL REPRODUCTION:

    When grid levels have prices unaligned with tick_size, quantize_price rounds down.
    If target_quantity was sized to be exactly 1.00 EUR at raw level_px,
    down-quantization lowers executed notional below 1.00 EUR.
    ExecutionSimulator._execute_buy currently lacks a step-up floor for target_quantity orders,
    failing with 'below minimum notional 1.0000 on Bybit'.
    """
    grid = MicroGridStrategy(
        default_symbol="TEST/EUR",
        min_notional=Decimal("1.00"),
        step_size=step_size,
    )
    signals = grid.generate_grid_levels(price, portfolio_balance=Decimal("10.00"))
    assert len(signals) >= 1

    sim = ExecutionSimulator(
        exchange="bybit",
        min_notional=Decimal("1.00"),
        step_size=step_size,
        tick_size=tick_size,
        fee_rate=Decimal("0.0010"),
        slippage=Decimal("0.0005"),
    )

    for sig in signals:
        ledger = VirtualLedger(initial_cash=Decimal("100.00"), quote_currency="EUR")
        res = sim.execute(order=sig, current_price=sig.price, ledger=ledger)
        assert res.success is True, (
            f"ExecutionSimulator rejected order on Bybit for price={price}, step={step_size}, tick={tick_size}: "
            f"{res.error_message}"
        )
        assert res.quote_notional >= Decimal("1.00")


# ============================================================================
# 5. DEFECT 3: IEEE-754 64-BIT FLOAT PRECISION TRUNCATION WITH MICRO-CAPS
# ============================================================================

def test_calculate_order_quantity_float_precision_underflow():
    """DEFECT 3 EMPIRICAL REPRODUCTION:

    MicroGridStrategy.calculate_order_quantity uses math.ceil(float(target / step_val)).
    When target / step_val exceeds 10^16, IEEE-754 64-bit float truncates the integer by millions of steps.
    The single step_size safety check fails to compensate for the massive truncation,
    leaving notional < 1.00 EUR.
    """
    price = Decimal("1E-12")
    step_size = Decimal("1E-12")
    grid = MicroGridStrategy(
        default_symbol="MICRO/EUR",
        min_notional=Decimal("1.00"),
        step_size=step_size,
    )
    qty = grid.calculate_order_quantity(price=price, target_notional=Decimal("1.00"))
    notional = qty * price
    assert notional >= Decimal("1.00"), (
        f"Order notional {notional} is below 1.00 EUR due to IEEE-754 float precision loss!"
    )


# ============================================================================
# 6. DEFECT 4: STRATEGY-LEDGER DESYNCHRONIZATION ON REJECTED ORDERS
# ============================================================================

def test_grid_state_does_not_desynchronize_if_simulator_rejects_order():
    """DEFECT 4 EMPIRICAL REPRODUCTION:

    MicroGridStrategy.on_candle registers active_levels[i] and appends to filled_tranches
    BEFORE the order is confirmed filled by the simulator.
    If the simulator rejects the order (e.g. for sub-notional or insufficient funds),
    the strategy must NOT mark the level as filled or active.
    """
    grid = MicroGridStrategy(
        default_symbol="ETH/EUR",
        min_notional=Decimal("1.00"),
        step_size=Decimal("0.0001"),
    )
    # Available cash is only 0.50 EUR (insufficient for 1.00 EUR order)
    ledger = VirtualLedger(initial_cash=Decimal("0.50"), quote_currency="EUR")

    c1 = Candle(
        symbol="ETH/EUR", interval="1h",
        timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        close_time=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        open=Decimal("100.00"), high=Decimal("100.00"), low=Decimal("100.00"), close=Decimal("100.00"),
        volume=Decimal("1.0"), quote_volume=Decimal("100.0"), trades_count=10,
    )
    grid.on_candle(c1, [], ledger)

    # When cash < min_notional, on_candle correctly returns None and does not create tranches
    c2 = Candle(
        symbol="ETH/EUR", interval="1h",
        timestamp=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        close_time=datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc),
        open=Decimal("100.00"), high=Decimal("100.00"), low=Decimal("95.00"), close=Decimal("95.00"),
        volume=Decimal("1.0"), quote_volume=Decimal("95.0"), trades_count=10,
    )
    sig = grid.on_candle(c2, [c1], ledger)
    assert sig is None
    assert len(grid.filled_tranches) == 0
    assert len(grid.active_levels) == 0

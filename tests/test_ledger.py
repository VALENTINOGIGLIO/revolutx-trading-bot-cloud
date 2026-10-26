"""Unit and Integration Tests for Virtual Ledger and Execution Simulation.

Covers:
- Tier 1 Feature Tests:
  1. VirtualLedger initial deposit invariant (10.00 EUR cash, 0 positions, 10.00 EUR equity).
  2. Cash deduction on BUY (notional + fee deducted, position and cost basis updated).
  3. Cash addition on SELL (net proceeds credited, realized PnL computed, position decremented).
  4. Mark-to-market valuation (unrealized PnL tracking during open position).
  5. Double-entry reconciliation invariant (Initial + Realized PnL == Current Equity).
  6. PerformanceMetrics calculation from ledger state and trade history.
- Tier 2 Boundary & Friction Tests:
  7. Zero and negative initial cash rejection (ValueError).
  8. InsufficientFundsError on BUY when required cash exceeds available cash.
  9. InsufficientPositionError on SELL when attempting to sell more than held.
  10. Exact Decimal fixed-point arithmetic (zero floating-point drift across micro-trades).
  11. Minimum notional validation (5.00 EUR Binance, 1.00 EUR Bybit).
  12. Slippage modeling (unfavorable +0.05% on BUY, -0.05% on SELL).
  13. Taker fee calculation (0.10% deducted on each fill).
  14. State serialization and restoration (save_state / load_state round-trip).
"""

from decimal import Decimal
from pathlib import Path
import pytest

from trading_bot.simulation.ledger import (
    InsufficientFundsError,
    InsufficientPositionError,
    VirtualLedger,
)
from trading_bot.simulation.metrics import PerformanceMetrics
from trading_bot.simulation.simulator import (
    ExecutionResult,
    ExecutionSimulator,
    OrderSide,
    OrderSignal,
    OrderType,
)


# =====================================================================
# TIER 1 FEATURE TESTS
# =====================================================================
def test_virtual_ledger_initial_deposit_invariant():
    """Test 10.00 EUR starting balance invariants."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    assert ledger.initial_cash == Decimal("10.00")
    assert ledger.cash == Decimal("10.00")
    assert ledger.available_cash == Decimal("10.00")
    assert ledger.locked_cash == Decimal("0.00")
    assert ledger.realized_pnl == Decimal("0.00")
    assert ledger.total_fees == Decimal("0.00")
    assert ledger.trade_count == 0
    assert ledger.total_equity() == Decimal("10.00")
    assert ledger.roi_pct() == Decimal("0.00")
    assert not ledger.has_position("BTC/EUR")


def test_virtual_ledger_cash_deduction_on_buy():
    """Test BUY trade execution: cash deduction, fee addition, and position creation."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Buy 0.0001 BTC @ 60,000 EUR (Notional = 6.00 EUR, Fee = 0.006 EUR)
    trade = ledger.record_buy(
        symbol="BTC/EUR",
        quantity=Decimal("0.0001"),
        executed_price=Decimal("60000.00"),
        fee=Decimal("0.006"),
        timestamp_ms=1785542400000,
        raw_price=Decimal("60000.00"),
    )

    expected_cost = Decimal("6.00") + Decimal("0.006")  # 6.006 EUR
    expected_remaining_cash = Decimal("10.00") - expected_cost  # 3.994 EUR

    assert trade.side == "BUY"
    assert trade.quote_notional == Decimal("6.00")
    assert trade.fee == Decimal("0.006")
    assert ledger.cash == expected_remaining_cash
    assert ledger.total_fees == Decimal("0.006")

    pos = ledger.get_position("BTC/EUR")
    assert pos.quantity == Decimal("0.0001")
    assert pos.average_entry_price == Decimal("60000.00")
    assert pos.cost_basis == Decimal("6.00")

    # Equity at current price = cash + asset value = 3.994 + 6.00 = 9.994 EUR (reflecting fee deduction)
    assert ledger.total_equity({"BTC/EUR": Decimal("60000.00")}) == Decimal("9.994")


def test_virtual_ledger_cash_addition_on_sell():
    """Test SELL trade execution: net proceeds addition, realized PnL, and position decrement."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # 1. Buy 0.0001 BTC @ 60,000 EUR (Cost: 6.00 + 0.006 fee = 6.006 EUR)
    ledger.record_buy(
        symbol="BTC/EUR",
        quantity=Decimal("0.0001"),
        executed_price=Decimal("60000.00"),
        fee=Decimal("0.006"),
        timestamp_ms=1785542400000,
    )

    # 2. Sell 0.0001 BTC @ 65,000 EUR (Gross = 6.50 EUR, Fee = 0.0065 EUR, Net = 6.4935 EUR)
    sell_trade = ledger.record_sell(
        symbol="BTC/EUR",
        quantity=Decimal("0.0001"),
        executed_price=Decimal("65000.00"),
        fee=Decimal("0.0065"),
        timestamp_ms=1785546000000,
    )

    # Realized PnL = Net Proceeds (6.4935) - Cost of Sold (6.00) = +0.4935 EUR
    expected_pnl = Decimal("6.4935") - Decimal("6.00")
    assert sell_trade.realized_pnl == expected_pnl

    # Cash = 3.994 + 6.4935 = 10.4875 EUR
    expected_cash = Decimal("3.994") + Decimal("6.4935")
    assert ledger.cash == expected_cash
    assert ledger.realized_pnl == expected_pnl
    assert ledger.total_fees == Decimal("0.006") + Decimal("0.0065")

    # Position is fully closed
    pos = ledger.get_position("BTC/EUR")
    assert pos.quantity == Decimal("0.0")
    assert pos.cost_basis == Decimal("0.0")
    assert ledger.total_equity() == expected_cash


def test_mark_to_market_unrealized_pnl():
    """Test mark-to-market valuation changes without altering realized PnL or cash."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    ledger.record_buy(
        symbol="SOL/EUR",
        quantity=Decimal("0.05"),
        executed_price=Decimal("140.00"),
        fee=Decimal("0.007"),
        timestamp_ms=1785542400000,
    )

    # Position holding: 0.05 SOL @ 140 = 7.00 EUR cost basis
    # Price rises to 160 EUR: unrealized gain = 0.05 * (160 - 140) = +1.00 EUR
    assert ledger.unrealized_pnl({"SOL/EUR": Decimal("160.00")}) == Decimal("1.00")
    assert ledger.realized_pnl == Decimal("0.00")  # Unrealized does NOT impact realized

    # Price drops to 120 EUR: unrealized loss = 0.05 * (120 - 140) = -1.00 EUR
    assert ledger.unrealized_pnl({"SOL/EUR": Decimal("120.00")}) == Decimal("-1.00")


def test_double_entry_accounting_reconciliation():
    """Verify double-entry invariant: Initial Cash + Realized PnL == Current Cash (when flat)."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Trade 1: Win
    ledger.record_buy("ETH/EUR", Decimal("0.002"), Decimal("3000.00"), Decimal("0.006"), 1)
    ledger.record_sell("ETH/EUR", Decimal("0.002"), Decimal("3200.00"), Decimal("0.0064"), 2)

    # Trade 2: Loss
    ledger.record_buy("ETH/EUR", Decimal("0.002"), Decimal("3200.00"), Decimal("0.0064"), 3)
    ledger.record_sell("ETH/EUR", Decimal("0.002"), Decimal("3100.00"), Decimal("0.0062"), 4)

    # Position flat
    assert ledger.get_position("ETH/EUR").quantity == Decimal("0.0")

    # Double-entry reconciliation:
    # On BUY: cash -= (notional + buy_fee), cost_basis += notional
    # On SELL: cash += (notional - sell_fee), realized_pnl += (notional - sell_fee - cost_basis)
    # Therefore: cash == initial_cash + realized_pnl - buy_fees
    buy_fees = sum(t.fee for t in ledger.trades if t.side == "BUY")
    assert ledger.cash == ledger.initial_cash + ledger.realized_pnl - buy_fees
    assert ledger.total_equity() == ledger.cash


def test_performance_metrics_from_ledger():
    """Test PerformanceMetrics calculation from simulated ledger execution."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Closed winning trade
    ledger.record_buy("BTC/EUR", Decimal("0.00015"), Decimal("60000.00"), Decimal("0.009"), 1000)
    ledger.record_sell("BTC/EUR", Decimal("0.00015"), Decimal("63000.00"), Decimal("0.00945"), 2000)

    metrics = PerformanceMetrics.from_ledger(ledger)

    assert metrics.initial_balance == Decimal("10.00")
    assert metrics.final_balance > Decimal("10.00")
    assert metrics.total_trades == 2
    assert metrics.closed_trades == 1
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 0
    assert metrics.win_rate_pct == 100.0
    assert metrics.profit_factor > 0.0
    assert metrics.total_return_pct > 0.0


# =====================================================================
# TIER 2 BOUNDARY & FRICTION TESTS
# =====================================================================
def test_zero_or_negative_initial_cash_rejected():
    """Test that zero or negative initial cash raises ValueError."""
    with pytest.raises(ValueError, match="Initial balance must be > 0"):
        VirtualLedger(initial_cash=Decimal("0.00"))

    with pytest.raises(ValueError, match="Initial balance must be > 0"):
        VirtualLedger(initial_cash=Decimal("-10.00"))


def test_insufficient_cash_raises_error_on_buy():
    """Test that placing a BUY order exceeding available cash raises InsufficientFundsError."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Attempt to buy 15.00 EUR worth with only 10.00 EUR cash
    with pytest.raises(InsufficientFundsError):
        ledger.record_buy(
            symbol="BTC/EUR",
            quantity=Decimal("0.00025"),
            executed_price=Decimal("60000.00"),  # 15.00 EUR
            fee=Decimal("0.015"),
            timestamp_ms=1,
        )

    # Assert cash balance remains untouched
    assert ledger.cash == Decimal("10.00")
    assert ledger.trade_count == 0


def test_insufficient_position_raises_error_on_sell():
    """Test that attempting to sell more asset units than held raises InsufficientPositionError."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Holding 0 BTC, attempt to sell 0.001 BTC
    with pytest.raises(InsufficientPositionError):
        ledger.record_sell(
            symbol="BTC/EUR",
            quantity=Decimal("0.001"),
            executed_price=Decimal("60000.00"),
            fee=Decimal("0.06"),
            timestamp_ms=1,
        )


def test_exact_decimal_precision_no_floating_drift():
    """Test that 10 consecutive micro-trades do not produce floating-point drift."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00000000"), quote_currency="EUR")

    for i in range(10):
        # Repeated small buy & sell
        ledger.record_buy("BTC/EUR", Decimal("0.00010000"), Decimal("50000.00"), Decimal("0.00500000"), i * 2)
        ledger.record_sell("BTC/EUR", Decimal("0.00010000"), Decimal("50000.00"), Decimal("0.00500000"), i * 2 + 1)

    # 10 round trips, each paying 0.01 EUR total fee
    expected_fees = Decimal("0.10000000")
    expected_final_cash = Decimal("10.00000000") - expected_fees

    assert ledger.total_fees == expected_fees
    assert ledger.cash == expected_final_cash
    # Strict Decimal comparison without approximation tolerance
    assert ledger.cash == Decimal("9.90000000")


def test_execution_simulator_min_notional_rejection():
    """Test ExecutionSimulator rejects orders below exchange minimum notional (5€ on Binance, 1€ on Bybit)."""
    binance_sim = ExecutionSimulator(min_notional=Decimal("5.00"), exchange="binance")
    bybit_sim = ExecutionSimulator(min_notional=Decimal("1.00"), exchange="bybit")
    ledger = VirtualLedger(initial_cash=Decimal("10.00"))

    # Order with notional = 3.50 EUR (< 5.00 EUR Binance limit)
    sub_notional_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("3.50"),
    )

    # Binance should reject
    binance_res = binance_sim.execute(sub_notional_order, current_price=Decimal("60000.00"), ledger=ledger)
    assert not binance_res.success
    assert "below minimum notional" in binance_res.error_message

    # Bybit (minNotional 1.00 EUR) should accept 3.50 EUR
    bybit_res = bybit_sim.execute(sub_notional_order, current_price=Decimal("60000.00"), ledger=ledger)
    assert bybit_res.success
    assert bybit_res.quote_notional > Decimal("1.00")


def test_execution_simulator_slippage_and_fee_calculation():
    """Test ExecutionSimulator models 0.05% slippage and 0.10% taker fee accurately."""
    sim = ExecutionSimulator(
        fee_rate=Decimal("0.0010"),   # 0.10%
        slippage=Decimal("0.0005"),   # 0.05%
        min_notional=Decimal("5.00"),
    )
    raw_price = Decimal("60000.00")

    # Slippage check
    buy_slipped = sim.calculate_slippage_price(raw_price, OrderSide.BUY)
    assert buy_slipped == Decimal("60030.00")  # 60,000 * 1.0005

    sell_slipped = sim.calculate_slippage_price(raw_price, OrderSide.SELL)
    assert sell_slipped == Decimal("59970.00")  # 60,000 * 0.9995

    # Fee check on 9.60 EUR notional
    fee = sim.calculate_fee(Decimal("9.60"))
    assert fee == Decimal("0.0096")


def test_ledger_state_persistence_roundtrip(tmp_path: Path):
    """Test save_state and load_state round-trip preservation of balances and trades."""
    state_file = tmp_path / "ledger_state.json"
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    ledger.record_buy("SOL/EUR", Decimal("0.05"), Decimal("140.00"), Decimal("0.007"), 1000)
    ledger.save_state(state_file)
    assert state_file.exists()

    loaded = VirtualLedger.load_state(state_file)
    assert loaded.initial_cash == ledger.initial_cash
    assert loaded.cash == ledger.cash
    assert loaded.total_fees == ledger.total_fees
    assert loaded.trade_count == 1
    assert loaded.get_position("SOL/EUR").quantity == Decimal("0.05")
    assert loaded.get_position("SOL/EUR").average_entry_price == Decimal("140.00")

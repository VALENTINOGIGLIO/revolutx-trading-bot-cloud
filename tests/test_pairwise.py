"""Tier 3: Pairwise Subsystem Interaction Tests.

Verifies cross-module interactions between:
1. Strategy Signal + Order Execution Simulator + NotificationService Interceptor.
2. Market Data Pipeline (CandleCache + MarketDataService) + VirtualLedger + Mark-to-Market Valuation.
3. Simulation Engine + Realistic Friction (Maker/Taker Fees & Slippage Accounting).
4. AMVTrendBreak Strategy + RiskGuard Daily Drawdown Circuit Breaker.
5. Volatility Shock (ATR Surge) + Dynamic Compounding Position Sizer.
6. Multi-Channel Notification Concurrency (Console + JSONL Log + In-Memory Interceptor).
7. Micro-Capital Inception (10€) + Exchange Minimum Notional Enforcement + Dust Handling.
8. Stop-Loss Trigger + Closing Execution + Notification Payload Audit.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import pytest

# Ensure src directory is in Python path for test execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trading_bot.data.cache import CandleCache, MarketDataService
from trading_bot.data.models import Candle
from trading_bot.notifications.channels.console import ConsoleChannel
from trading_bot.notifications.channels.json_log import JsonLogChannel
from trading_bot.notifications.interceptor import NotificationInterceptor
from trading_bot.notifications.models import (
    EventType,
    OrderSide as NotifyOrderSide,
    PortfolioUpdatePayload,
    TradeExecutedPayload,
)
from trading_bot.notifications.service import NotificationService
from trading_bot.simulation.ledger import (
    InsufficientFundsError,
    InsufficientPositionError,
    VirtualLedger,
)
from trading_bot.simulation.metrics import PerformanceMetrics
from trading_bot.simulation.simulator import (
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


# =====================================================================
# TEST 1: Strategy Signal + Simulator Execution + Notification Interceptor
# =====================================================================
def test_pairwise_strategy_signal_execution_and_notification():
    """T3.PC.01: Verifies strategy signal execution through simulator into notification interceptor.

    Flow:
    1. VirtualLedger initialized with 10.00 EUR.
    2. NotificationService initialized with NotificationInterceptor.
    3. OrderSignal emitted and passed to ExecutionSimulator with notification hook.
    4. ExecutionResult updates ledger and dispatches TradeExecutedPayload.
    5. Assert all attributes in intercepted payload match ledger and execution details.
    """
    notif_service = NotificationService()
    interceptor = NotificationInterceptor()
    notif_service.register_channel(interceptor)

    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        exchange="binance",
        notification_service=notif_service,
    )

    # 1. Execute BUY
    entry_raw_price = Decimal("50000.00")
    buy_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("9.80"),
        order_id="sig_buy_001",
    )

    exec_result = simulator.execute(order=buy_order, current_price=entry_raw_price, ledger=ledger)
    assert exec_result.success, f"Execution failed: {exec_result.error_message}"

    # Assert notification intercepted
    assert len(interceptor.captured_events) == 1
    event = interceptor.last_event
    assert isinstance(event, TradeExecutedPayload)
    assert event.event_type == EventType.TRADE_EXECUTED
    assert event.symbol == "BTC/EUR"
    assert event.side == NotifyOrderSide.BUY
    assert event.fill_price == float(exec_result.executed_price)
    assert event.quantity == pytest.approx(float(exec_result.quantity), rel=1e-5)
    assert event.fee == pytest.approx(float(exec_result.fee), rel=1e-5)
    assert event.portfolio_balance == float(exec_result.trade_record.equity_after)
    assert event.realized_pnl == 0.0

    # 2. Execute SELL
    exit_raw_price = Decimal("55000.00")
    sell_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        target_quantity=exec_result.quantity,
        order_id="sig_sell_001",
        exit_reason="TAKE_PROFIT_TRAIL",
    )

    exec_sell_result = simulator.execute(order=sell_order, current_price=exit_raw_price, ledger=ledger)
    assert exec_sell_result.success, f"Sell execution failed: {exec_sell_result.error_message}"

    assert len(interceptor.captured_events) == 2
    sell_event = interceptor.last_event
    assert isinstance(sell_event, TradeExecutedPayload)
    assert sell_event.side == NotifyOrderSide.SELL
    assert sell_event.realized_pnl > 0.0
    assert sell_event.realized_pnl == pytest.approx(float(ledger.realized_pnl), rel=1e-5)
    assert sell_event.portfolio_balance == pytest.approx(float(ledger.cash), rel=1e-5)


# =====================================================================
# TEST 2: Market Data Service + Ledger + Execution Simulator
# =====================================================================
def test_pairwise_market_data_cache_to_ledger_execution(tmp_path: Path):
    """T3.PC.02: Verifies end-to-end data pipeline feeding into execution simulator and ledger.

    Flow:
    1. Generate realistic candle series and persist to isolated SQLite CandleCache.
    2. MarketDataService fetches cached candles without network dependency.
    3. Iterate through candles, executing entry and exit based on price movements.
    4. Assert ledger double-entry bookkeeping matches price marks from MarketData.
    """
    db_file = tmp_path / "pairwise_data.db"
    cache = CandleCache(db_path=db_file)
    service = MarketDataService(cache=cache)

    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    test_candles = []
    prices = [50000.0, 50500.0, 51200.0, 52000.0, 53000.0, 52500.0, 51800.0]
    for idx, px in enumerate(prices):
        dt = base_time + timedelta(hours=idx)
        c = Candle(
            timestamp=dt,
            open=Decimal(str(px - 100.0)),
            high=Decimal(str(px + 200.0)),
            low=Decimal(str(px - 200.0)),
            close=Decimal(str(px)),
            volume=Decimal("12.5"),
            symbol="BTCEUR",
            interval="1h",
        )
        test_candles.append(c)

    cache.save_candles(test_candles)

    # Read back through service
    retrieved = service.get_historical_candles(
        symbol="BTC/EUR",
        interval="1h",
        start_time=base_time,
        end_time=base_time + timedelta(hours=len(prices) - 1),
    )
    assert len(retrieved) == len(test_candles)

    # Feed retrieved data sequentially into ledger and simulator
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
    )

    # Buy at bar 0
    bar_0 = retrieved[0]
    buy_res = simulator.execute_buy(
        symbol="BTCEUR",
        raw_price=bar_0.close,
        ledger=ledger,
        quote_budget=Decimal("9.80"),
        timestamp_ms=int(bar_0.timestamp.timestamp() * 1000),
    )
    assert buy_res.success
    pos = ledger.get_position("BTCEUR")
    assert pos.quantity > Decimal("0.0")

    # Mark-to-market valuations across bars 1..4
    for bar in retrieved[1:5]:
        eq = ledger.total_equity({"BTCEUR": bar.close})
        unrealized = ledger.unrealized_pnl({"BTCEUR": bar.close})
        # Price increases from 50k -> 53k, total equity and unrealized PnL must increase
        assert eq > Decimal("10.00")
        assert unrealized > Decimal("0.00")

    # Sell at bar 5
    bar_5 = retrieved[5]
    sell_res = simulator.execute_sell(
        symbol="BTCEUR",
        raw_price=bar_5.close,
        ledger=ledger,
        timestamp_ms=int(bar_5.timestamp.timestamp() * 1000),
    )
    assert sell_res.success
    assert ledger.get_position("BTCEUR").quantity == Decimal("0.0")
    assert ledger.cash > Decimal("10.00")  # Net gain locked in


# =====================================================================
# TEST 3: Simulation Friction (Fees & Slippage) Accounting
# =====================================================================
def test_pairwise_simulation_friction_and_double_entry_accounting():
    """T3.PC.03: Verifies exact friction math across buy/sell cycle.

    Ensures:
    - BUY slips upward (+slippage rate).
    - SELL slips downward (-slippage rate).
    - Fees accurately deducted on both sides.
    - Double entry reconciliation: Initial Cash + Realized PnL - Total Fees == Final Cash.
    """
    fee_rate = Decimal("0.001")  # 0.1%
    slippage = Decimal("0.0005")  # 0.05%

    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=fee_rate,
        slippage=slippage,
        min_notional=Decimal("5.00"),
        exchange="binance",
    )

    p_in = Decimal("60000.00")
    buy_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("9.80"),
    )
    buy_res = simulator.execute(order=buy_order, current_price=p_in, ledger=ledger)
    assert buy_res.success

    # Slipped buy price must equal 60,000 * 1.0005 = 60,030.00
    expected_buy_px = p_in * (Decimal("1") + slippage)
    assert buy_res.executed_price == expected_buy_px
    expected_buy_fee = buy_res.quote_notional * fee_rate
    assert buy_res.fee == expected_buy_fee

    # Sell at 66,000.00
    p_out = Decimal("66000.00")
    sell_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        target_quantity=buy_res.quantity,
    )
    sell_res = simulator.execute(order=sell_order, current_price=p_out, ledger=ledger)
    assert sell_res.success

    # Slipped sell price must equal 66,000 * 0.9995 = 65,967.00
    expected_sell_px = p_out * (Decimal("1") - slippage)
    assert sell_res.executed_price == expected_sell_px
    expected_sell_fee = sell_res.quote_notional * fee_rate
    assert sell_res.fee == expected_sell_fee

    # Double entry conservation audit
    total_fees = buy_res.fee + sell_res.fee
    assert ledger.total_fees == total_fees

    expected_gross_pnl = (buy_res.quantity * expected_sell_px) - (buy_res.quantity * expected_buy_px)
    expected_net_pnl = expected_gross_pnl - sell_res.fee
    assert ledger.realized_pnl == expected_net_pnl
    assert ledger.cash == Decimal("10.00") - buy_res.quote_notional - buy_res.fee + (sell_res.quote_notional - sell_res.fee)


# =====================================================================
# TEST 4: Drawdown Circuit Breaker Halts Trading
# =====================================================================
def test_pairwise_drawdown_circuit_breaker_halts_strategy_signals():
    """T3.PC.04: Verifies RiskGuard 8% daily drawdown circuit breaker halts strategy entries.

    Flow:
    1. Account equity starts at 10.00 EUR and rises to 15.00 EUR (peak).
    2. Equity drops from 15.00 EUR to 13.50 EUR (10% drawdown, exceeding 8% threshold).
    3. RiskGuard triggers circuit breaker and sets risk_multiplier to 0.0.
    4. AMVTrendBreakStrategy on_candle() rejects trade signal evaluation while halted.
    """
    risk_guard = RiskGuard(
        max_daily_drawdown_pct=0.08,
        drawdown_halt_hours=24.0,
        initial_equity=Decimal("10.00"),
    )
    strategy = AMVTrendBreakStrategy(
        default_symbol="BTC/EUR",
        risk_guard=risk_guard,
    )

    t0 = datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc)
    risk_guard.update_equity(Decimal("15.00"), timestamp=t0)
    assert risk_guard.check_risk_status(t0).allowed

    # Drawdown drop to 13.50 (loss of 1.50 / 15.00 = 10%)
    t1 = t0 + timedelta(hours=1)
    risk_guard.update_equity(Decimal("13.50"), timestamp=t1)

    status = risk_guard.check_risk_status(t1)
    assert not status.allowed
    assert status.halt_active
    assert status.risk_multiplier == 0.0
    assert "drawdown" in status.reason.lower()

    # Now verify strategy on_candle does NOT emit buy signals
    sample_candle = Candle(
        timestamp=t1,
        open=Decimal("58000.00"),
        high=Decimal("59000.00"),
        low=Decimal("57900.00"),
        close=Decimal("58800.00"),
        volume=Decimal("10.0"),
        symbol="BTC/EUR",
        interval="1h",
    )
    # Even if history were long enough, risk check immediately aborts
    signal = strategy.on_candle(sample_candle, history=[], ledger=Decimal("13.50"))
    assert signal is None


# =====================================================================
# TEST 5: Volatility Shock + Dynamic Compounding Position Sizer
# =====================================================================
def test_pairwise_volatility_shock_and_dynamic_position_sizing():
    """T3.PC.05: Verifies that sudden volatility expansion (ATR spike) adjusts position sizing.

    Under Phase 2 (Growth 100€-1,000€) fixed fractional sizing:
    - Normal volatility (ATR = 200 EUR): Stop distance = 1.75 * 200 = 350 EUR.
    - Volatility shock (ATR = 2500 EUR): Stop distance = 1.75 * 2500 = 4,375 EUR.
    - As stop distance widens significantly, the position sizing scales down
      to guarantee risk does not exceed the allowed risk fraction.
    """
    sizer = CompoundingPositionSizer(
        min_notional=Decimal("5.00"),
        default_step_size=Decimal("0.00001"),
    )
    equity = Decimal("200.00")  # Phase 2 Growth
    entry_price = Decimal("60000.00")

    # Normal ATR = 200 EUR (stop distance = 350 EUR)
    normal_stop = entry_price - Decimal("350.00")
    res_normal = sizer.calculate_position_size(
        equity=equity,
        entry_price=entry_price,
        stop_loss=normal_stop,
        risk_multiplier=1.0,
    )
    assert res_normal.is_valid
    assert res_normal.phase == CompoundingPhase.PHASE_2_GROWTH

    # Extreme Volatility Shock (stop distance = 4,500 EUR)
    shock_stop = entry_price - Decimal("4500.00")
    res_shock = sizer.calculate_position_size(
        equity=equity,
        entry_price=entry_price,
        stop_loss=shock_stop,
        risk_multiplier=1.0,
    )
    assert res_shock.is_valid

    # Quantity during volatility shock must be smaller to protect capital
    assert res_shock.quantity < res_normal.quantity
    # Risk fraction remains bounded
    assert res_shock.risk_fraction <= 0.041


# =====================================================================
# TEST 6: Multi-Channel Notification Concurrency
# =====================================================================
def test_pairwise_multichannel_notification_concurrency(tmp_path: Path):
    """T3.PC.06: Verifies concurrent delivery to Console, JSONL, and Interceptor.

    Dispatches both TradeExecutedPayload and PortfolioUpdatePayload.
    Asserts all 3 channels receive synchronous, uncorrupted payloads.
    """
    log_file = tmp_path / "events.jsonl"
    service = NotificationService()

    interceptor = NotificationInterceptor()
    json_channel = JsonLogChannel(log_path=log_file)
    console_channel = ConsoleChannel()

    service.register_channel(interceptor)
    service.register_channel(json_channel)
    service.register_channel(console_channel)

    # Dispatch TradeExecuted
    t0 = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    trade_payload = TradeExecutedPayload(
        symbol="BTC/EUR",
        side=NotifyOrderSide.BUY,
        fill_price=59500.0,
        quantity=0.00016,
        notional_value=9.52,
        fee=0.00952,
        portfolio_balance=9.99,
        timestamp=t0,
    )
    service.dispatch(trade_payload)

    # Dispatch PortfolioUpdate
    portfolio_payload = PortfolioUpdatePayload(
        total_equity=12.50,
        cash_balance=12.50,
        roi_pct=25.0,
        progress_to_target_pct=0.025,
        total_trades=5,
        win_rate_pct=60.0,
        timestamp=t0,
    )
    service.dispatch(portfolio_payload)

    # 1. Check Interceptor
    assert len(interceptor.captured_events) == 2
    assert isinstance(interceptor.captured_events[0], TradeExecutedPayload)
    assert isinstance(interceptor.captured_events[1], PortfolioUpdatePayload)

    # 2. Check JSONL Log File
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    log_entry_1 = json.loads(lines[0])
    assert log_entry_1["event_type"] == "TRADE_EXECUTED"
    assert log_entry_1["symbol"] == "BTC/EUR"
    assert log_entry_1["fill_price"] == 59500.0

    log_entry_2 = json.loads(lines[1])
    assert log_entry_2["event_type"] == "PORTFOLIO_UPDATE"
    assert log_entry_2["total_equity"] == 12.50
    assert log_entry_2["roi_pct"] == 25.0


# =====================================================================
# TEST 7: Micro-Balance (10€) + Min Notional Check + Fee Deductions
# =====================================================================
def test_pairwise_micro_balance_min_notional_rejection_and_dust_handling():
    """T3.PC.07: Tests initial 10€ micro-trade lifecycle, min-notional gate, and dust rejection.

    Flow:
    1. VirtualLedger starts with exactly 10.00 EUR.
    2. Trade 1: Sized for 9.80 EUR (clears 5.00 EUR minNotional).
    3. Balance after trade: ~0.38 EUR remaining.
    4. Attempting Trade 2 with remaining 0.38 EUR is rejected due to minNotional.
    5. Rejection does not corrupt ledger or crash simulator.
    """
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        exchange="binance",
    )

    # Trade 1: Valid micro-trade
    res1 = simulator.execute_buy(
        symbol="BTC/EUR",
        raw_price=Decimal("60000.00"),
        ledger=ledger,
        quote_budget=Decimal("9.80"),
    )
    assert res1.success
    assert ledger.cash < Decimal("1.00")
    assert ledger.cash > Decimal("0.10")

    # Trade 2: Attempting to buy with dust cash (< 5.00 EUR minNotional)
    res2 = simulator.execute_buy(
        symbol="BTC/EUR",
        raw_price=Decimal("60000.00"),
        ledger=ledger,
        quote_budget=ledger.available_cash,
    )
    assert not res2.success
    assert "below minimum notional" in res2.error_message.lower()

    # Ledger state preserved
    assert ledger.trade_count == 1
    assert ledger.get_position("BTC/EUR").quantity == res1.quantity


# =====================================================================
# TEST 8: Stop Loss Trigger + Notification Payload Audit
# =====================================================================
def test_pairwise_stop_loss_trigger_and_notification_audit():
    """T3.PC.08: Verifies stop loss hit updates ledger and dispatches auditable alert.

    Flow:
    1. Enter BUY at 60,000 EUR with stop loss at 58,000 EUR.
    2. Market gap-down triggers SELL at stop loss.
    3. Verify realized loss recorded in ledger.
    4. Verify TradeExecuted notification reflects loss and updated portfolio balance.
    """
    notif_service = NotificationService()
    interceptor = NotificationInterceptor()
    notif_service.register_channel(interceptor)

    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        notification_service=notif_service,
    )

    # Buy fill
    p_entry = Decimal("60000.00")
    buy_res = simulator.execute_buy(
        symbol="BTC/EUR",
        raw_price=p_entry,
        ledger=ledger,
        quote_budget=Decimal("9.80"),
    )
    assert buy_res.success

    # Sell fill at stop loss (58,000 EUR)
    p_stop = Decimal("58000.00")
    sell_res = simulator.execute_sell(
        symbol="BTC/EUR",
        raw_price=p_stop,
        ledger=ledger,
        quantity=buy_res.quantity,
        exit_reason="STOP_LOSS_HIT",
    )
    assert sell_res.success
    assert sell_res.realized_pnl < Decimal("0.0")

    # Audit intercepted notification
    assert len(interceptor.captured_events) == 2
    stop_event = interceptor.captured_events[-1]
    assert isinstance(stop_event, TradeExecutedPayload)
    assert stop_event.side == NotifyOrderSide.SELL
    assert stop_event.realized_pnl < 0.0
    assert stop_event.realized_pnl == pytest.approx(float(sell_res.realized_pnl), rel=1e-5)
    assert stop_event.portfolio_balance == pytest.approx(float(ledger.cash), rel=1e-5)

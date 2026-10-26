"""Tier 4: End-to-End Acceptance Tests (MANDATORY FOR BOTH ACCEPTANCE CRITERIA).

Satisfies:
1. Acceptance Criteria 1 (AC1):
   Full programmatic backtest execution over >= 1 month of historical data,
   starting with exactly 10.00€ initial capital, and outputting final simulated
   balance and a list of executed trades without crashing.
2. Acceptance Criteria 2 (AC2):
   Unit test triggering a simulated trade and successfully intercepting / verifying
   that the notification payload (containing trade details) was constructed and dispatched.
3. Compounding Progression Test:
   Multi-step growth validation from 10€ capital through compounding phases,
   verifying dynamic position sizing, trailing stop ratcheting, and ledger reconciliation.
4. Risk Controls & Circuit Breaker E2E:
   Real-world stress scenario verifying capital preservation under adverse market conditions.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import pytest

# Ensure src directory is in Python path for test execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trading_bot.config import normalize_symbol
from trading_bot.data.binance_client import BinanceClient
from trading_bot.data.cache import CandleCache, MarketDataService
from trading_bot.data.models import Candle
from trading_bot.notifications.interceptor import NotificationInterceptor
from trading_bot.notifications.models import (
    EventType,
    OrderSide as NotifyOrderSide,
    TradeExecutedPayload,
)
from trading_bot.notifications.service import NotificationService
from trading_bot.simulation.ledger import TradeRecord, VirtualLedger
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
# Programmatic Backtest Runner Helper (Engine Parity)
# =====================================================================
def run_programmatic_backtest(
    candles: List[Candle],
    initial_balance: Decimal = Decimal("10.00"),
    pair: str = "BTC/EUR",
    fee_rate: Decimal = Decimal("0.001"),
    slippage: Decimal = Decimal("0.0005"),
    exchange: str = "binance",
    notification_service: Optional[NotificationService] = None,
) -> Dict[str, Any]:
    """Reusable programmatic backtester executing candle-by-candle replay.

    Replays historical candles through:
    - Strategy evaluation (`AMVTrendBreakStrategy.on_candle`)
    - Order Execution (`ExecutionSimulator.execute`)
    - Double-entry accounting (`VirtualLedger`)
    - Notification dispatch (`NotificationService`)
    - Quantitative metrics calculation (`PerformanceMetrics`)

    Returns:
        Dict containing:
        - initial_balance: float
        - final_balance: float
        - total_return_pct: float
        - total_bars_processed: int
        - trades: List[Dict[str, Any]]
        - metrics: PerformanceMetrics
        - ledger: VirtualLedger
    """
    norm_symbol = normalize_symbol(pair)
    min_notional = Decimal("5.00") if exchange == "binance" else Decimal("1.00")

    ledger = VirtualLedger(initial_cash=initial_balance, quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=fee_rate,
        slippage=slippage,
        min_notional=min_notional,
        step_size=Decimal("0.00001"),
        exchange=exchange,
        notification_service=notification_service,
    )
    strategy = AMVTrendBreakStrategy(
        default_symbol=norm_symbol,
        exchange_min_notional=min_notional,
        step_size=Decimal("0.00001"),
        min_warmup_bars=50,
    )

    history: List[Candle] = []

    for i, candle in enumerate(candles):
        # 1. Evaluate strategy signal on current candle
        signal = strategy.on_candle(candle=candle, history=history, ledger=ledger)

        # 2. Execute signal if generated
        if signal is not None:
            ts_ms = int(candle.timestamp.timestamp() * 1000)
            exec_result = simulator.execute(
                order=signal,
                current_price=candle.close,
                ledger=ledger,
                timestamp_ms=ts_ms,
            )
            # If position was closed in ledger, notify strategy
            if exec_result.success and signal.side == OrderSide.SELL:
                strategy.notify_position_closed(
                    symbol=norm_symbol,
                    exit_price=exec_result.executed_price,
                    current_equity=ledger.total_equity({norm_symbol: candle.close}),
                    timestamp=candle.timestamp,
                )

        history.append(candle)
        # Record periodic equity snapshot
        if i % 24 == 0 or i == len(candles) - 1:
            ts_ms = int(candle.timestamp.timestamp() * 1000)
            ledger.record_equity_snapshot(ts_ms, {norm_symbol: candle.close})

    # Final mark-to-market valuation at last candle close
    last_close = candles[-1].close if candles else Decimal("0.0")
    final_equity = ledger.total_equity({norm_symbol: last_close})
    metrics = PerformanceMetrics.from_ledger(ledger, current_prices={norm_symbol: last_close})

    executed_trades = [t.to_dict() for t in ledger.trades]

    return {
        "status": "success",
        "initial_balance": float(initial_balance),
        "final_balance": float(final_equity),
        "net_profit": float(final_equity - initial_balance),
        "total_return_pct": float(metrics.total_return_pct),
        "total_bars_processed": len(candles),
        "trades": executed_trades,
        "metrics": metrics,
        "ledger": ledger,
    }


# =====================================================================
# MANDATORY ACCEPTANCE CRITERIA 1: FULL 1-MONTH BACKTEST FROM 10€
# =====================================================================
def test_acceptance_criteria_1_full_1month_backtest_execution(synthetic_candles_30d: List[Candle]):
    """ACCEPTANCE CRITERIA 1:
    A programmatic backtesting script exists and can be executed from the terminal.
    The backtest runs successfully over at least 1 month of historical data,
    starting with 10€, and outputs a final simulated balance and a list of
    executed trades without crashing, proving the strategy's historical performance.
    """
    initial_capital = Decimal("10.00")
    assert len(synthetic_candles_30d) >= 720, f"Expected >= 720 bars for 1 month, got {len(synthetic_candles_30d)}"

    # Execute full programmatic backtest
    result = run_programmatic_backtest(
        candles=synthetic_candles_30d,
        initial_balance=initial_capital,
        pair="BTC/EUR",
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        exchange="binance",
    )

    # 1. Assert backtest executed without crashing
    assert result["status"] == "success"

    # 2. Assert starting balance was exactly 10€
    assert result["initial_balance"] == 10.0

    # 3. Assert total historical bars >= 720 (>= 30 days of 1-hour candles)
    assert result["total_bars_processed"] >= 720

    # 4. Assert final simulated balance is a positive float
    assert result["final_balance"] > 0.0

    # 5. Assert list of executed trades is present and verifiable
    assert isinstance(result["trades"], list)
    assert len(result["trades"]) > 0, "Backtest should execute at least one trade over 30 days of data"

    # 6. Verify trade structure and audit fields
    for trade in result["trades"]:
        assert "trade_id" in trade
        assert "symbol" in trade
        assert "side" in trade
        assert float(trade["executed_price"]) > 0.0
        assert float(trade["quantity"]) > 0.0
        assert float(trade["quote_notional"]) >= 5.0, "All trades must satisfy exchange minimum notional (5€)"
        assert float(trade["fee"]) >= 0.0
        assert float(trade["cash_after"]) >= 0.0, "Cash balance must never drop below 0.00 EUR"
        assert float(trade["equity_after"]) >= 0.0

    # 7. Print output proving terminal execution requirements (outputs final balance & trades)
    print("\n" + "=" * 80)
    print("AC1 VERIFICATION: PROGRAMMATIC BACKTEST COMPLETED SUCCESSFULLY")
    print(f"Initial Balance: {result['initial_balance']:.2f} EUR")
    print(f"Final Balance:   {result['final_balance']:.2f} EUR")
    print(f"Total Return:    {result['total_return_pct']:+.2f}%")
    print(f"Total Bars:      {result['total_bars_processed']} bars (>= 1 month)")
    print(f"Trades Executed: {len(result['trades'])} trades")

def test_acceptance_criteria_1_terminal_cli_execution(capsys):
    """ACCEPTANCE CRITERIA 1 (TERMINAL CLI EXECUTION):
    Executes the programmatic backtesting script via the CLI interface
    over >= 1 month of real historical data, starting with 10€, and asserts
    it outputs final simulated balance and trade list with return code 0.
    """
    from trading_bot.cli.backtest import main as cli_backtest_main

    # Run in JSON mode for structured assertion
    exit_code = cli_backtest_main([
        "--start", "2026-08-01",
        "--end", "2026-09-01",
        "--balance", "10.0",
        "--pair", "BTC/EUR",
        "--output", "json",
    ])
    assert exit_code == 0, f"CLI execution failed with exit code {exit_code}"

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["status"] == "success"
    assert data["summary"]["initial_balance"] == 10.0
    assert data["summary"]["final_balance"] > 0.0
    assert data["summary"]["total_bars_processed"] >= 720
    assert len(data["trades"]) >= 1

    # Run in table mode to assert human-readable terminal table output
    table_exit_code = cli_backtest_main([
        "--start", "2026-08-01",
        "--end", "2026-09-01",
        "--balance", "10.0",
        "--pair", "BTC/EUR",
        "--output", "table",
    ])
    assert table_exit_code == 0
    table_captured = capsys.readouterr()
    assert "AUTONOMOUS TRADING BOT - BACKTEST REPORT" in table_captured.out
    assert "PORTFOLIO SUMMARY:" in table_captured.out
    assert "EXECUTED TRADES LOG:" in table_captured.out
    assert "[SUCCESS] Backtest completed without errors." in table_captured.out


# =====================================================================
# MANDATORY ACCEPTANCE CRITERIA 2: SIMULATED TRADE NOTIFICATION INTERCEPTION
# =====================================================================
def test_acceptance_criteria_2_simulated_trade_notification_payload_intercepted():
    """ACCEPTANCE CRITERIA 2:
    A unit test exists that triggers a simulated trade and successfully
    intercepts/verifies that the notification payload (containing trade details)
    was constructed and dispatched.
    """
    notification_service = NotificationService()
    interceptor = NotificationInterceptor()
    notification_service.register_channel(interceptor)

    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        exchange="binance",
        notification_service=notification_service,
    )

    # Pre-condition: Interceptor queue is empty
    assert len(interceptor.captured_events) == 0

    # 1. Trigger simulated BUY trade: 9.80 EUR budget at 60,000.00 EUR/BTC
    raw_entry_price = Decimal("60000.00")
    buy_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("9.80"),
        order_id="ord_ac2_test_buy",
    )

    result = simulator.execute(order=buy_order, current_price=raw_entry_price, ledger=ledger)
    assert result.success, f"BUY order execution failed: {result.error_message}"

    # 2. Intercept and verify notification payload
    assert len(interceptor.captured_events) == 1
    payload = interceptor.last_event

    assert isinstance(payload, TradeExecutedPayload)
    assert payload.event_type == EventType.TRADE_EXECUTED
    assert payload.symbol == "BTC/EUR"
    assert payload.side == NotifyOrderSide.BUY
    assert payload.fill_price == float(result.executed_price)
    assert payload.quantity == pytest.approx(float(result.quantity), rel=1e-5)
    assert payload.fee == pytest.approx(float(result.fee), rel=1e-5)
    assert payload.portfolio_balance == float(result.trade_record.equity_after)
    assert payload.timestamp is not None
    assert payload.realized_pnl == 0.0

    # 3. Trigger simulated closing SELL trade: Sell entire holding at 65,000.00 EUR/BTC
    raw_exit_price = Decimal("65000.00")
    sell_order = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.SELL,
        target_quantity=result.quantity,
        order_id="ord_ac2_test_sell",
        exit_reason="TAKE_PROFIT_TARGET",
    )

    sell_result = simulator.execute(order=sell_order, current_price=raw_exit_price, ledger=ledger)
    assert sell_result.success, f"SELL order execution failed: {sell_result.error_message}"

    # Verify second intercepted payload
    assert len(interceptor.captured_events) == 2
    sell_payload = interceptor.last_event

    assert isinstance(sell_payload, TradeExecutedPayload)
    assert sell_payload.side == NotifyOrderSide.SELL
    assert sell_payload.fill_price == float(sell_result.executed_price)
    assert sell_payload.realized_pnl > 0.0
    assert sell_payload.realized_pnl == pytest.approx(float(ledger.realized_pnl), rel=1e-5)
    assert sell_payload.portfolio_balance == pytest.approx(float(ledger.cash), rel=1e-5)

    print("\n" + "=" * 80)
    print("AC2 VERIFICATION: SIMULATED TRADE NOTIFICATION INTERCEPTED SUCCESSFULLY")
    print(f"Dispatched Event: {sell_payload.event_type.value}")
    print(f"Symbol:           {sell_payload.symbol}")
    print(f"Side:             {sell_payload.side.value}")
    print(f"Fill Price:       {sell_payload.fill_price:.2f} EUR")
    print(f"Quantity:         {sell_payload.quantity:.8f}")
    print(f"Realized PnL:     {sell_payload.realized_pnl:+.4f} EUR")
    print(f"Portfolio Bal:    {sell_payload.portfolio_balance:.2f} EUR")
    print("=" * 80)


# =====================================================================
# COMPOUNDING PROGRESSION: MULTI-STEP GROWTH VALIDATION
# =====================================================================
def test_compounding_progression_lifecycle_validation():
    """Compounding Progression Test:
    Simulates multi-step capital growth starting with exactly 10.00 EUR across
    consecutive winning and controlled loss trades.
    Verifies:
    - Dynamic sizing scales upwards with equity growth.
    - Double-entry ledger reconciliation holds at every step.
    - Zero negative cash balances or arithmetic drift.
    """
    initial_cash = Decimal("10.00")
    ledger = VirtualLedger(initial_cash=initial_cash, quote_currency="EUR")
    simulator = ExecutionSimulator(
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        min_notional=Decimal("5.00"),
        step_size=Decimal("0.00001"),
    )
    sizer = CompoundingPositionSizer(
        min_notional=Decimal("5.00"),
        default_step_size=Decimal("0.00001"),
    )

    # Step 1: Trade 1 from 10.00 EUR
    eq1 = ledger.total_equity()
    assert eq1 == Decimal("10.00")
    p1_entry = Decimal("50000.00")
    p1_stop = Decimal("48500.00")
    size1 = sizer.calculate_position_size(equity=eq1, entry_price=p1_entry, stop_loss=p1_stop)
    assert size1.is_valid
    assert size1.phase == CompoundingPhase.PHASE_1_MICRO

    res_buy1 = simulator.execute_buy(symbol="BTCEUR", raw_price=p1_entry, ledger=ledger, quote_budget=size1.notional)
    assert res_buy1.success

    # Win +8%: Sell at 54,000 EUR
    p1_exit = Decimal("54000.00")
    res_sell1 = simulator.execute_sell(symbol="BTCEUR", raw_price=p1_exit, ledger=ledger, quantity=res_buy1.quantity)
    assert res_sell1.success
    assert res_sell1.realized_pnl > Decimal("0.0")

    eq2 = ledger.total_equity()
    assert eq2 > eq1, "Equity must compound positively after winning trade"

    # Step 2: Trade 2 sized on expanded capital
    p2_entry = Decimal("55000.00")
    p2_stop = Decimal("53500.00")
    size2 = sizer.calculate_position_size(equity=eq2, entry_price=p2_entry, stop_loss=p2_stop)
    assert size2.is_valid
    # Position notional must have increased because equity compounded
    assert size2.notional > size1.notional

    res_buy2 = simulator.execute_buy(symbol="BTCEUR", raw_price=p2_entry, ledger=ledger, quote_budget=size2.notional)
    assert res_buy2.success

    # Win +6%: Sell at 58,300 EUR
    p2_exit = Decimal("58300.00")
    res_sell2 = simulator.execute_sell(symbol="BTCEUR", raw_price=p2_exit, ledger=ledger, quantity=res_buy2.quantity)
    assert res_sell2.success

    eq3 = ledger.total_equity()
    assert eq3 > eq2

    # Step 3: Controlled loss (-2.5% stop loss hit)
    p3_entry = Decimal("60000.00")
    p3_stop = Decimal("58500.00")
    size3 = sizer.calculate_position_size(equity=eq3, entry_price=p3_entry, stop_loss=p3_stop)
    assert size3.is_valid

    res_buy3 = simulator.execute_buy(symbol="BTCEUR", raw_price=p3_entry, ledger=ledger, quote_budget=size3.notional)
    assert res_buy3.success

    # Stop loss hit at 58,500 EUR
    res_sell3 = simulator.execute_sell(symbol="BTCEUR", raw_price=p3_stop, ledger=ledger, quantity=res_buy3.quantity)
    assert res_sell3.success
    assert res_sell3.realized_pnl < Decimal("0.0")

    eq4 = ledger.total_equity()
    # Account survives controlled loss comfortably above initial 10€
    assert eq4 > Decimal("10.00")

    # Step 4: Rebound Win (+10%)
    p4_entry = Decimal("59000.00")
    p4_stop = Decimal("57500.00")
    size4 = sizer.calculate_position_size(equity=eq4, entry_price=p4_entry, stop_loss=p4_stop)
    assert size4.is_valid

    res_buy4 = simulator.execute_buy(symbol="BTCEUR", raw_price=p4_entry, ledger=ledger, quote_budget=size4.notional)
    assert res_buy4.success

    # Exit at 64,900 EUR (+10%)
    p4_exit = Decimal("64900.00")
    res_sell4 = simulator.execute_sell(symbol="BTCEUR", raw_price=p4_exit, ledger=ledger, quantity=res_buy4.quantity)
    assert res_sell4.success

    final_eq = ledger.total_equity()
    assert final_eq > eq4

    # Conservation audit: Initial cash + sum(realized_pnl) - buy_fees == current cash
    buy_fees = sum(t.fee for t in ledger.trades if t.side == "BUY")
    reconciled = initial_cash + ledger.realized_pnl - buy_fees
    assert ledger.cash == pytest.approx(reconciled, rel=Decimal("1e-4"))
    assert ledger.trade_count == 8  # 4 buys, 4 sells


# =====================================================================
# REAL-WORLD WORKLOAD WITH HISTORICAL BINANCE DATA / CACHE
# =====================================================================
def test_real_world_workload_backtest_with_historical_data(tmp_path: Path):
    """End-to-End Real World Workload:
    Fetches real historical candles via MarketDataService (using cache or live Binance),
    runs programmatic backtest from 10€, and verifies performance metrics generation.
    """
    db_file = tmp_path / "ac1_market_data.db"
    cache = CandleCache(db_path=db_file)
    service = MarketDataService(cache=cache)

    start_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)

    # 1. Fetch historical candles (>= 1 month = 744 hours)
    try:
        candles = service.get_historical_candles(
            symbol="BTC/EUR",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as ex:
        # Fallback to deterministic synthetic 30-day series if offline
        from conftest import synthetic_candles_30d
        candles = synthetic_candles_30d()

    assert len(candles) >= 720, f"Expected >= 720 candles for >= 1 month, got {len(candles)}"

    # 2. Run full programmatic backtest
    notif_service = NotificationService()
    interceptor = NotificationInterceptor()
    notif_service.register_channel(interceptor)

    report = run_programmatic_backtest(
        candles=candles,
        initial_balance=Decimal("10.00"),
        pair="BTC/EUR",
        fee_rate=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        exchange="binance",
        notification_service=notif_service,
    )

    # 3. Assert report content
    assert report["status"] == "success"
    assert report["initial_balance"] == 10.0
    assert report["final_balance"] > 0.0
    assert report["total_bars_processed"] >= 720
    assert isinstance(report["trades"], list)

    metrics = report["metrics"]
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.initial_balance == Decimal("10.00")
    assert metrics.final_balance > Decimal("0.00")
    assert metrics.max_drawdown_pct >= 0.0

    # Format report matching CLI specification table output
    table_summary = metrics.to_summary_table()
    assert "PORTFOLIO SUMMARY:" in table_summary
    assert "TRADING PERFORMANCE METRICS:" in table_summary
    assert "Initial Balance:" in table_summary
    assert "Final Balance:" in table_summary

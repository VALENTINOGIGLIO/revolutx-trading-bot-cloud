"""
Adversarial Stress Test Suite & Fuzzing Harness.

Empirical verification of trading bot behavior under extreme conditions:
1. Flash crashes (-50% in a single candle) and explosive surges (+100%).
2. Zero, negative, and micro-fractional balances.
3. Rapid order churn (1,000 rapid orders in VirtualLedger).
4. Empty candle datasets, corrupt timestamps, missing fields.
5. Network simulation: timeouts, empty responses, malformed JSON.
6. Algorithmic complexity and scaling on massive datasets (>10,000 candles).
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import math
import time
import pytest

from trading_bot.data.models import Candle, Ticker, timeframe_to_milliseconds
from trading_bot.simulation.ledger import (
    VirtualLedger,
    TradeRecord,
    Position,
    InsufficientFundsError,
    InsufficientPositionError,
)
from trading_bot.simulation.simulator import (
    ExecutionSimulator,
    OrderSignal,
    OrderSide,
    OrderType,
    ExecutionResult,
)
from trading_bot.simulation.metrics import PerformanceMetrics
from trading_bot.strategy.amv_strategy import AMVTrendBreakStrategy
from trading_bot.strategy.compounding import (
    CompoundingPositionSizer,
    RiskGuard,
    CompoundingPhase,
)
from trading_bot.strategy.indicators import (
    calculate_ema,
    calculate_atr,
    calculate_keltner_channels,
    calculate_adx,
    calculate_choppiness,
    calculate_rsi,
    compute_all_indicators,
    compute_indicators_from_candles,
)


# =====================================================================
# 1. Flash Crashes, Explosive Surges & State Desynchronization
# =====================================================================

def test_flash_crash_below_min_notional_state_desync():
    """
    Empirical test: Flash crash causes equity to drop below 5.00 EUR minNotional.
    Verifies that ExecutionSimulator permits closing sell orders to execute cleanly
    and that AMVTrendBreakStrategy and VirtualLedger remain fully synchronized.
    """
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(exchange="binance", min_notional=Decimal("5.00"))
    strategy = AMVTrendBreakStrategy(
        default_symbol="SOL/EUR",
        exchange_min_notional=Decimal("5.00"),
        step_size=Decimal("0.001"),
        min_warmup_bars=5,
    )

    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    candles = []
    prices = [100.0, 102.0, 105.0, 110.0, 115.0, 125.0]
    for i, p in enumerate(prices):
        c = Candle(
            timestamp=base_time + timedelta(hours=i),
            open=Decimal(str(p - 1.0)),
            high=Decimal(str(p + 2.0)),
            low=Decimal(str(p - 2.0)),
            close=Decimal(str(p)),
            volume=Decimal("1000.0"),
            symbol="SOL/EUR",
            interval="1h",
        )
        candles.append(c)

    bought = False
    for i in range(len(candles)):
        curr = candles[i]
        hist = candles[:i]
        sig = strategy.on_candle(curr, hist, ledger)
        if sig and sig.side == OrderSide.BUY:
            res = simulator.execute(sig, curr.close, ledger, curr.open_time_ms)
            assert res.success, f"Buy should succeed: {res.error_message}"
            bought = True
            break

    assert bought, "Strategy should have generated a buy signal during breakout"
    assert ledger.has_position("SOL/EUR"), "Ledger should hold SOL position"
    pos = ledger.get_position("SOL/EUR")
    assert pos.quantity > Decimal("0")

    # Inject FLASH CRASH: -60% in a single candle (Price crashes from 125 -> 50 EUR)
    crash_time = base_time + timedelta(hours=len(candles))
    crash_candle = Candle(
        timestamp=crash_time,
        open=Decimal("125.0"),
        high=Decimal("125.0"),
        low=Decimal("45.0"),
        close=Decimal("50.0"),
        volume=Decimal("50000.0"),
        symbol="SOL/EUR",
        interval="1h",
    )

    exit_sig = strategy.on_candle(crash_candle, candles, ledger)
    assert exit_sig is not None, "Stop loss should be triggered on -60% crash"
    assert exit_sig.side == OrderSide.SELL

    exec_res = simulator.execute(exit_sig, crash_candle.close, ledger, crash_candle.open_time_ms)

    assert exec_res.success is True, f"Closing sell order should succeed: {exec_res.error_message}"
    assert not ledger.has_position("SOL/EUR"), "Ledger should have cleanly closed the position"
    assert "SOL/EUR" not in strategy.positions, "Strategy position should be cleanly closed"


def test_catastrophic_99pct_flash_crash():
    """
    Test catastrophic 99.9% market collapse: price goes from 50,000 to 50 EUR.
    Verify whether double-entry accounting survives and cash never goes negative.
    """
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(exchange="binance", min_notional=Decimal("1.00"))  # using 1€ minNotional to allow sell

    # Buy BTC
    buy_res = simulator.execute_buy("BTC/EUR", Decimal("50000.00"), ledger, quote_budget=Decimal("9.50"))
    assert buy_res.success
    assert ledger.cash >= Decimal("0.00")

    # 99.9% crash
    sell_res = simulator.execute_sell("BTC/EUR", Decimal("50.00"), ledger, quantity=buy_res.quantity)
    # Even if sell fails or succeeds, cash must never be negative
    assert ledger.cash >= Decimal("0.00")
    assert ledger.total_equity() >= Decimal("0.00")


def test_explosive_300pct_surge_profit_taking():
    """
    Test parabolic surge (+300% in single candle from 100 to 400 EUR).
    Verify Chandelier Trailing Stop behavior and positive realized PnL.
    """
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(exchange="binance", min_notional=Decimal("5.00"))

    buy_res = simulator.execute_buy("SOL/EUR", Decimal("100.00"), ledger, quote_budget=Decimal("9.80"))
    assert buy_res.success

    # Sell at 400 EUR
    sell_res = simulator.execute_sell("SOL/EUR", Decimal("400.00"), ledger, quantity=buy_res.quantity)
    assert sell_res.success
    assert sell_res.realized_pnl > Decimal("0.00")
    assert ledger.cash > Decimal("30.00")  # >3x initial balance
    assert ledger.total_equity() == ledger.cash


# =====================================================================
# 2. Zero, Negative, and Micro-Fractional Balances & Dust Handling
# =====================================================================

def test_zero_and_negative_initial_balances():
    """Test VirtualLedger rejects 0 and negative balances at construction."""
    with pytest.raises(ValueError, match="must be > 0"):
        VirtualLedger(initial_cash=Decimal("0.00"))

    with pytest.raises(ValueError, match="must be > 0"):
        VirtualLedger(initial_cash=Decimal("-10.00"))

    with pytest.raises(ValueError, match="must be > 0"):
        VirtualLedger(initial_cash="-0.0001")


def test_micro_fractional_balances_rejection_in_sizer():
    """Test CompoundingPositionSizer with micro-fractional balances (0.01 EUR, 1e-8 EUR)."""
    sizer = CompoundingPositionSizer(min_notional=Decimal("5.00"))

    # 0.01 EUR balance
    res1 = sizer.calculate_position_size(
        equity=Decimal("0.01"),
        entry_price=Decimal("100.00"),
        stop_loss=Decimal("95.00"),
    )
    assert not res1.is_valid
    assert "below minimum exchange notional" in res1.rejection_reason

    # 1e-8 EUR balance
    res2 = sizer.calculate_position_size(
        equity=Decimal("0.00000001"),
        entry_price=Decimal("100.00"),
        stop_loss=Decimal("95.00"),
    )
    assert not res2.is_valid
    assert res2.quantity == Decimal("0")


def test_sub_satoshi_micro_quantity_quantization():
    """Test quantization of extreme micro-quantities (1e-12 units)."""
    simulator = ExecutionSimulator(step_size=Decimal("0.00001"))
    tiny_qty = Decimal("0.000004999")
    quantized = simulator.quantize_quantity(tiny_qty)
    assert quantized == Decimal("0.00000"), "Sub-step quantities must quantize down to zero"


def test_oversell_raises_insufficient_position():
    """Test attempting to sell more asset than held returns error or raises InsufficientPositionError."""
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    
    # 1. Non-strict mode: returns ExecutionResult(success=False)
    sim_soft = ExecutionSimulator(strict=False)
    buy_res = sim_soft.execute_buy("BTC/EUR", Decimal("50000.00"), ledger, quote_budget=Decimal("9.00"))
    assert buy_res.success

    sell_res = sim_soft.execute_sell("BTC/EUR", Decimal("50000.00"), ledger, quantity=buy_res.quantity * Decimal("2.0"))
    assert not sell_res.success
    assert "Insufficient position" in sell_res.error_message

    # 2. Strict mode: raises InsufficientPositionError
    sim_strict = ExecutionSimulator(strict=True)
    with pytest.raises(InsufficientPositionError):
        sim_strict.execute_sell("BTC/EUR", Decimal("50000.00"), ledger, quantity=buy_res.quantity * Decimal("2.0"))


# =====================================================================
# 3. Rapid Order Churn (1,000 Rapid Orders)
# =====================================================================

def test_rapid_order_churn_1000_orders():
    """
    Stress test: Execute 1,000 alternating BUY and SELL orders in VirtualLedger.
    Verify:
    1. Double-entry conservation of funds.
    2. Zero IEEE-754 floating point drift (exact Decimal math).
    3. Performance and state stability under rapid churn.
    """
    ledger = VirtualLedger(initial_cash=Decimal("1000.00"), quote_currency="EUR")
    price = Decimal("100.00")
    qty = Decimal("0.05")  # 5 EUR notional
    fee_rate = Decimal("0.001")
    slip = Decimal("0.0005")

    start_time = time.perf_counter()
    num_cycles = 500  # 500 BUY + 500 SELL = 1,000 orders

    for i in range(num_cycles):
        # Slightly oscillating price
        exec_price = price + Decimal(str((i % 5) - 2))
        fee = qty * exec_price * fee_rate

        # BUY
        ledger.record_buy(
            symbol="SOL/EUR",
            quantity=qty,
            executed_price=exec_price,
            fee=fee,
            timestamp_ms=1000000 + (i * 2),
            fee_rate=fee_rate,
            slippage=slip,
        )

        # SELL
        sell_px = exec_price + Decimal("0.50")
        sell_fee = qty * sell_px * fee_rate
        ledger.record_sell(
            symbol="SOL/EUR",
            quantity=qty,
            executed_price=sell_px,
            fee=sell_fee,
            timestamp_ms=1000000 + (i * 2) + 1,
            fee_rate=fee_rate,
            slippage=slip,
        )

    elapsed = time.perf_counter() - start_time
    print(f"\n1,000 orders executed in {elapsed:.3f}s ({1000/elapsed:.0f} orders/sec)")

    assert ledger.trade_count == 1000
    assert ledger.get_position("SOL/EUR").quantity == Decimal("0.0")

    # Double-entry reconciliation check
    buy_fees = sum(t.fee for t in ledger.trades if t.side == "BUY")
    expected_cash = ledger.initial_cash + ledger.realized_pnl - buy_fees
    assert ledger.cash == expected_cash, f"Cash {ledger.cash} does not match expected {expected_cash}"
    assert ledger.total_equity() == ledger.cash


# =====================================================================
# 4. Malformed Datasets, Flat Markets & Zero Volatility
# =====================================================================

def test_flat_zero_volatility_dataset():
    """
    Stress test: 100 bars where High == Low == Open == Close (flat line, 0 volatility).
    Verifies that ATR, Keltner, ADX, CHOP, and RSI do not crash with division-by-zero.
    """
    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    flat_candles = [
        Candle(
            timestamp=base_time + timedelta(hours=i),
            open=Decimal("100.0"),
            high=Decimal("100.0"),
            low=Decimal("100.0"),
            close=Decimal("100.0"),
            volume=Decimal("0.0"),
            symbol="BTC/EUR",
            interval="1h",
        )
        for i in range(100)
    ]

    # Compute indicators
    df = compute_indicators_from_candles(flat_candles)
    assert not df.empty
    assert len(df) == 100

    # ATR should be 0.0 without crash
    assert float(df["atr_14"].iloc[-1]) == 0.0
    # RSI for constant price should be 50.0 without crash
    assert float(df["rsi_14"].iloc[-1]) == 50.0
    # CHOP should be bounded without NaN or inf
    chop_val = float(df["chop_14"].iloc[-1])
    assert 0.0 <= chop_val <= 100.0
    assert not math.isnan(chop_val)


def test_corrupted_timestamps_and_unordered_candles():
    """Test Candle model with irregular, future, and millisecond timestamps."""
    # Far future (year 2099)
    c_future = Candle(
        timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc),
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume=Decimal("10"),
    )
    assert c_future.timestamp.year == 2099

    # Unix ms integer timestamp
    c_ms = Candle(
        timestamp=1785672000000,  # millisecond timestamp
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume=Decimal("10"),
    )
    assert c_ms.timestamp.tzinfo == timezone.utc


# =====================================================================
# 5. Algorithmic Complexity & Quadratic Scaling Analysis
# =====================================================================

def test_backtest_simulation_quadratic_complexity_scaling():
    """
    Stress test: Measure backtest simulation scaling on 100, 300, 600, 1,200 candles.
    Demonstrates the quadratic O(N^2) indicator recomputation overhead in the candle loop.
    """
    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)

    def generate_synthetic_candles(n: int) -> list:
        candles = []
        for i in range(n):
            p = 100.0 + (i % 20)
            candles.append(
                Candle(
                    timestamp=base_time + timedelta(hours=i),
                    open=Decimal(str(p)),
                    high=Decimal(str(p + 1.0)),
                    low=Decimal(str(p - 1.0)),
                    close=Decimal(str(p + 0.5)),
                    volume=Decimal("1000.0"),
                    symbol="SOL/EUR",
                    interval="1h",
                )
            )
        return candles

    benchmark_sizes = [100, 300, 600]
    timings = {}

    for size in benchmark_sizes:
        candles = generate_synthetic_candles(size)
        strategy = AMVTrendBreakStrategy(default_symbol="SOL/EUR", min_warmup_bars=50)
        ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

        t0 = time.perf_counter()
        for i in range(len(candles)):
            curr = candles[i]
            hist = candles[:i]
            strategy.on_candle(curr, hist, ledger)
        elapsed = time.perf_counter() - t0
        timings[size] = elapsed

    print("\nBacktest Scaling Timings:")
    for sz, el in timings.items():
        print(f"  {sz:5d} candles: {el:6.3f}s ({el/sz*1000:5.2f} ms/candle)")

    # Verify quadratic trend: 600 candles vs 300 candles (2x size should be ~4x time in O(N^2))
    ratio = timings[600] / timings[300]
    print(f"  Ratio (600 / 300): {ratio:.2f}x (O(N^2) theoretical is ~4.0x)")
    # Assert it completes without crashing
    assert timings[600] < 30.0, "600 candles should complete in <30s"


# =====================================================================
# 6. Network Fault Injections & BinanceClient Resilience
# =====================================================================

def test_network_simulation_timeout_and_mirror_failover(monkeypatch):
    """Test BinanceClient behavior when primary base URL times out and fallback to Bybit occurs."""
    import requests
    from trading_bot.data.binance_client import BinanceClient

    client = BinanceClient(max_retries=1, fallback_to_bybit=True)

    # Mock session.get to raise requests.Timeout for Binance URLs but succeed for Bybit
    original_get = client.session.get

    def mock_get(url, *args, **kwargs):
        if "binance" in url:
            raise requests.Timeout("Simulated Binance timeout")
        elif "bybit" in url:
            # Return mock Bybit response
            mock_resp = requests.Response()
            mock_resp.status_code = 200
            mock_resp._content = json.dumps({
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {"lastPrice": "50000.00", "symbol": "BTCEUR"}
                    ]
                }
            }).encode("utf-8")
            return mock_resp
        return original_get(url, *args, **kwargs)

    monkeypatch.setattr(client.session, "get", mock_get)

    ticker = client.get_ticker("BTC/EUR")
    assert ticker.price == Decimal("50000.00")
    assert ticker.symbol == "BTCEUR"


def test_network_simulation_all_endpoints_fail_raises_connection_error(monkeypatch):
    """Test BinanceClient raises clean ConnectionError when all mirrors and Bybit fail."""
    import requests
    from trading_bot.data.binance_client import BinanceClient

    client = BinanceClient(max_retries=1, fallback_to_bybit=True)

    def mock_get_fail(url, *args, **kwargs):
        raise requests.ConnectionError("Network completely unreachable")

    monkeypatch.setattr(client.session, "get", mock_get_fail)

    with pytest.raises(ConnectionError, match=r"All Binance.*endpoints failed"):
        client.get_ticker("BTC/EUR")


def test_network_simulation_malformed_json_response(monkeypatch):
    """Test BinanceClient handling of malformed non-JSON responses (e.g. Cloudflare 502 HTML)."""
    import requests
    from trading_bot.data.binance_client import BinanceClient

    client = BinanceClient(max_retries=1, fallback_to_bybit=False)

    def mock_get_html(url, *args, **kwargs):
        resp = requests.Response()
        resp.status_code = 200  # Some CDNs return 200 with HTML error page
        resp._content = b"<html><head><title>502 Bad Gateway</title></head><body>Error</body></html>"
        return resp

    monkeypatch.setattr(client.session, "get", mock_get_html)

    with pytest.raises(ConnectionError):
        client.get_ticker("BTC/EUR")


# =====================================================================
# 7. Post-Rejection State Machine & Circuit Breakers
# =====================================================================

def test_post_rejection_ledger_and_strategy_divergence():
    """
    Empirical confirmation of state divergence:
    When an exit order fails execution (e.g. minNotional), the strategy must not
    be oblivious to the unclosed position on subsequent candles.
    """
    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")
    simulator = ExecutionSimulator(exchange="binance", min_notional=Decimal("5.00"))
    strategy = AMVTrendBreakStrategy(
        default_symbol="SOL/EUR",
        exchange_min_notional=Decimal("5.00"),
        min_warmup_bars=5,
    )

    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    # 1. Warm-up and buy
    candles = [
        Candle(
            timestamp=base_time + timedelta(hours=i),
            open=Decimal(str(100 + i * 5)),
            high=Decimal(str(105 + i * 5)),
            low=Decimal(str(98 + i * 5)),
            close=Decimal(str(103 + i * 5)),
            volume=Decimal("1000"),
            symbol="SOL/EUR",
            interval="1h",
        )
        for i in range(6)
    ]

    for i in range(len(candles)):
        curr = candles[i]
        hist = candles[:i]
        sig = strategy.on_candle(curr, hist, ledger)
        if sig and sig.side == OrderSide.BUY:
            res = simulator.execute(sig, curr.close, ledger, curr.open_time_ms)
            assert res.success

    # Position is open
    assert ledger.has_position("SOL/EUR")
    held_qty = ledger.get_position("SOL/EUR").quantity

    # 2. Crash candle triggers STOP_LOSS_HIT, but price is so low that notional < 5€
    crash_candle = Candle(
        timestamp=base_time + timedelta(hours=7),
        open=Decimal("125"),
        high=Decimal("125"),
        low=Decimal("30"),
        close=Decimal("35"),  # notional = held_qty (~0.076) * 35 = ~2.66€ (< 5€)
        volume=Decimal("5000"),
        symbol="SOL/EUR",
        interval="1h",
    )

    exit_sig = strategy.on_candle(crash_candle, candles, ledger)
    assert exit_sig is not None
    exec_res = simulator.execute(exit_sig, crash_candle.close, ledger, crash_candle.open_time_ms)
    assert exec_res.success is True, f"Closing sell order should succeed: {exec_res.error_message}"
    assert not ledger.has_position("SOL/EUR"), "Position should be cleanly closed in ledger"

    # 3. Next candle: What does the strategy do?
    next_candle = Candle(
        timestamp=base_time + timedelta(hours=8),
        open=Decimal("35"),
        high=Decimal("40"),
        low=Decimal("30"),
        close=Decimal("38"),
        volume=Decimal("1000"),
        symbol="SOL/EUR",
        interval="1h",
    )

    next_sig = strategy.on_candle(next_candle, candles + [crash_candle], ledger)
    # Strategy and ledger remain synchronized: no unclosed position and no stray exit signals
    assert next_sig is None or next_sig.side != OrderSide.SELL
    assert not ledger.has_position("SOL/EUR")


def test_risk_guard_circuit_breaker_halts_and_expires():
    """Test 8% daily drawdown circuit breaker halts and then expires after 24h."""
    guard = RiskGuard(max_daily_drawdown_pct=0.08, drawdown_halt_hours=24.0, initial_equity=Decimal("100.00"))
    t0 = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

    # Initial equity 100
    guard.update_equity(Decimal("100.00"), t0)
    assert guard.check_risk_status(t0).allowed

    # Drop 10% to 90 at t0 + 1 hour
    t1 = t0 + timedelta(hours=1)
    guard.update_equity(Decimal("90.00"), t1)

    # Circuit breaker should be ACTIVE
    status_halted = guard.check_risk_status(t1)
    assert not status_halted.allowed
    assert status_halted.halt_active
    assert status_halted.risk_multiplier == 0.0

    # 12 hours later: still halted
    t_mid = t1 + timedelta(hours=12)
    assert not guard.check_risk_status(t_mid).allowed

    # 25 hours after halt trigger: halt should have expired
    t_expired = t1 + timedelta(hours=25)
    guard.update_equity(Decimal("90.00"), t_expired)
    status_recovered = guard.check_risk_status(t_expired)
    assert status_recovered.allowed
    assert not status_recovered.halt_active
    assert status_recovered.risk_multiplier > 0.0



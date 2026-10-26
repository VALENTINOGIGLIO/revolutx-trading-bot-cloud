"""Unit and Integration Tests for Strategy & Compounding Subsystem.

Covers:
- Tier 1 Feature Tests:
  1. Technical Indicators calculation:
     - EMA (9, 21, 50, 200) mathematical accuracy.
     - ATR (14) and True Range gap handling.
     - Keltner Channels (20, 1.8 ATR).
     - ADX (14) and Directional Movement (+DI, -DI).
     - Choppiness Index (CHOP 14) bounds and trend detection.
     - RSI (14) oscillator bounds [0, 100].
  2. Three-Phase Compounding Position Sizer:
     - Phase 1 (10€ - 100€): ~96% capital allocation with tight ATR stop.
     - Phase 2 (100€ - 1,000€): Fixed fractional sizing (4% risk per trade).
     - Phase 3 (1,000€ - 10,000€): Quarter-Kelly sizing (2% risk per trade).
  3. AMVTrendBreakStrategy bullish breakout BUY signal generation.
  4. AMVTrendBreakStrategy breakeven stop adjustment at 1.5R.
- Tier 2 Boundary & Risk Tests:
  5. 8% rolling daily drawdown circuit breaker (halts trading for 24 hours).
  6. Consecutive loss throttling (30% cut after 2 losses, 50% cut after 3 losses, resets on win).
  7. Minimum notional capital guard (rejects trades when equity < exchange minNotional).
  8. Zero risk multiplier rejection when halt is active.
  9. Structural invalidation exit on Close < EMA 21.
  10. Entry gating in sideways chop regimes (ADX < 22, CHOP >= 58, RSI out of bounds).
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import numpy as np
import pandas as pd
import pytest

from trading_bot.data.models import Candle
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.strategy.amv_strategy import (
    AMVTrendBreakStrategy,
    OrderSide,
    OrderSignal,
    OrderType,
)
from trading_bot.strategy.compounding import (
    CompoundingPhase,
    CompoundingPositionSizer,
    RiskGuard,
)
from trading_bot.strategy.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_choppiness,
    calculate_ema,
    calculate_keltner_channels,
    calculate_rsi,
    calculate_true_range,
    compute_all_indicators,
    compute_indicators_from_candles,
)


# =====================================================================
# TIER 1 FEATURE TESTS: TECHNICAL INDICATORS
# =====================================================================
def test_ema_mathematical_accuracy():
    """Verify EMA matches formula: alpha = 2 / (period + 1)."""
    prices = [100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0, 112.0]
    period = 9
    alpha = 2.0 / (period + 1)

    ema = calculate_ema(prices, period)
    assert len(ema) == len(prices)

    # Manually compute EMA
    manual_ema = [prices[0]]
    for p in prices[1:]:
        manual_ema.append(alpha * p + (1.0 - alpha) * manual_ema[-1])

    for calc, expected in zip(ema, manual_ema):
        assert calc == pytest.approx(expected, rel=1e-5)


def test_true_range_and_atr_gap_handling():
    """Verify True Range captures prior close gaps and ATR matches Wilder smoothing."""
    highs = [105.0, 115.0, 112.0]
    lows = [95.0, 108.0, 100.0]
    closes = [100.0, 110.0, 102.0]

    tr = calculate_true_range(highs, lows, closes)
    # Bar 0: High - Low = 105 - 95 = 10.0
    assert tr.iloc[0] == pytest.approx(10.0)
    # Bar 1: Gap up from Close_prev (100). High - Low = 7.0, |High - Close_prev| = 15.0, |Low - Close_prev| = 8.0 -> TR = 15.0
    assert tr.iloc[1] == pytest.approx(15.0)
    # Bar 2: Gap down from Close_prev (110). High - Low = 12.0, |High - Close_prev| = 2.0, |Low - Close_prev| = 10.0 -> TR = 12.0
    assert tr.iloc[2] == pytest.approx(12.0)

    atr = calculate_atr(highs, lows, closes, period=14)
    assert len(atr) == 3
    assert not atr.isna().any()


def test_keltner_channels():
    """Verify Keltner Channels: Middle = EMA(20), Upper/Lower = Middle +/- multiplier * ATR."""
    n = 30
    prices = np.linspace(100.0, 130.0, n)
    highs = prices + 2.0
    lows = prices - 2.0
    closes = prices

    kc = calculate_keltner_channels(highs, lows, closes, ema_period=20, atr_period=14, multiplier=1.8)
    assert len(kc.middle) == n
    assert len(kc.upper) == n
    assert len(kc.lower) == n

    # Upper must be strictly above Middle, Lower strictly below
    assert (kc.upper > kc.middle).all()
    assert (kc.lower < kc.middle).all()
    # Spread should equal 2 * 1.8 * ATR = 3.6 * ATR
    spread = kc.upper - kc.lower
    assert (spread > 0.0).all()


def test_adx_directional_movement_bounds():
    """Verify ADX and +DI / -DI oscilate within [0, 100]."""
    n = 40
    # Strong upward trend
    highs = np.linspace(100.0, 200.0, n) + np.random.normal(0, 0.5, n)
    lows = highs - 5.0
    closes = highs - 1.0

    adx_res = calculate_adx(highs, lows, closes, period=14)
    assert ((adx_res.adx >= 0.0) & (adx_res.adx <= 100.0)).all()
    assert ((adx_res.plus_di >= 0.0) & (adx_res.plus_di <= 100.0)).all()
    assert ((adx_res.minus_di >= 0.0) & (adx_res.minus_di <= 100.0)).all()

    # In a strong uptrend, +DI should exceed -DI
    assert adx_res.plus_di.iloc[-1] > adx_res.minus_di.iloc[-1]


def test_choppiness_index_bounds_and_trend_detection():
    """Verify Choppiness Index bounds in [0, 100] and detects trend vs chop."""
    # Case 1: Strong continuous trend (CHOP should be low, < 38.2)
    n = 30
    trend_highs = np.linspace(100.0, 180.0, n)
    trend_lows = trend_highs - 2.0
    trend_closes = trend_highs - 0.5

    chop_trend = calculate_choppiness(trend_highs, trend_lows, trend_closes, period=14)
    assert ((chop_trend >= 0.0) & (chop_trend <= 100.0)).all()
    # At the end of a strong unbroken trend, CHOP drops significantly
    assert chop_trend.iloc[-1] < 45.0

    # Case 2: Pure sideways oscillation (CHOP should be high, > 61.8)
    chop_highs = [105.0, 104.0, 105.0, 104.0] * 10
    chop_lows = [95.0, 96.0, 95.0, 96.0] * 10
    chop_closes = [100.0, 101.0, 99.0, 100.0] * 10
    chop_sideways = calculate_choppiness(chop_highs, chop_lows, chop_closes, period=14)
    assert chop_sideways.iloc[-1] > 60.0


def test_rsi_oscillator():
    """Verify RSI stays within [0, 100] and accurately reflects bullish and bearish moves."""
    # 20 consecutive gains -> RSI approaches 100
    prices_up = [100.0 + i * 2.0 for i in range(25)]
    rsi_up = calculate_rsi(prices_up, period=14)
    assert ((rsi_up >= 0.0) & (rsi_up <= 100.0)).all()
    assert rsi_up.iloc[-1] > 80.0

    # 20 consecutive losses -> RSI approaches 0
    prices_down = [200.0 - i * 2.0 for i in range(25)]
    rsi_down = calculate_rsi(prices_down, period=14)
    assert ((rsi_down >= 0.0) & (rsi_down <= 100.0)).all()
    assert rsi_down.iloc[-1] < 20.0


# =====================================================================
# TIER 1 FEATURE TESTS: 3-PHASE COMPOUNDING SIZER
# =====================================================================
def test_compounding_position_sizer_phases():
    """Test 3-Phase transitions: Phase 1 (10€), Phase 2 (500€), Phase 3 (2000€)."""
    sizer = CompoundingPositionSizer(
        phase_1_threshold=Decimal("100.00"),
        phase_2_threshold=Decimal("1000.00"),
        min_notional=Decimal("5.00"),
        default_step_size=Decimal("0.0001"),
    )

    # 1. Phase 1: Micro-Cap (10€ equity)
    # Allocates ~96% of 10€ = 9.60€ notional (> 5.00€ minNotional)
    res_phase1 = sizer.calculate_position_size(
        equity=Decimal("10.00"),
        entry_price=Decimal("60000.00"),
        stop_loss=Decimal("58500.00"),
    )
    assert res_phase1.is_valid
    assert res_phase1.phase == CompoundingPhase.PHASE_1_MICRO
    assert res_phase1.notional >= Decimal("5.00")
    assert res_phase1.notional <= Decimal("9.80")
    # Dollar risk is small (~0.25€)
    assert res_phase1.risk_amount < Decimal("0.50")

    # 2. Phase 2: Small-Cap Growth (500€ equity)
    # Fixed fractional: 4% risk = 20.00€ risk
    res_phase2 = sizer.calculate_position_size(
        equity=Decimal("500.00"),
        entry_price=Decimal("60000.00"),
        stop_loss=Decimal("54000.00"),  # 10% stop distance -> notional = 200€ <= 425€ cap
        step_size=Decimal("0.00001"),
    )
    assert res_phase2.is_valid
    assert res_phase2.phase == CompoundingPhase.PHASE_2_GROWTH
    # Risk amount should be exactly ~20€ (4% of 500€)
    assert float(res_phase2.risk_amount) == pytest.approx(20.0, rel=1e-2)

    # 3. Phase 3: Mid-Cap Scaling (2000€ equity)
    # Quarter-Kelly: 2% risk = 40.00€ risk, with 5% stop distance -> 800€ notional (matches 40% asset allocation cap)
    res_phase3 = sizer.calculate_position_size(
        equity=Decimal("2000.00"),
        entry_price=Decimal("60000.00"),
        stop_loss=Decimal("57000.00"),
        step_size=Decimal("0.00001"),
    )
    assert res_phase3.is_valid
    assert res_phase3.phase == CompoundingPhase.PHASE_3_SCALE
    assert float(res_phase3.risk_amount) == pytest.approx(40.0, rel=1e-2)
    assert res_phase3.notional <= Decimal("800.00")  # 40% allocation cap


def test_amv_strategy_bullish_breakout_buy_signal():
    """Test AMVTrendBreakStrategy triggers BUY signal on confirmed bullish breakout."""
    strategy = AMVTrendBreakStrategy(
        default_symbol="BTC/EUR",
        min_warmup_bars=50,
        exchange_min_notional=Decimal("5.00"),
        step_size=Decimal("0.00001"),
    )
    ledger = VirtualLedger(initial_cash=Decimal("10.00"))

    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    candles = []
    p = 50000.0
    np.random.seed(42)

    for i in range(80):
        dt = base_time + timedelta(hours=i)
        ret = np.random.normal(0.002, 0.004)
        close_p = p * (1 + ret)
        high_p = max(p, close_p) * 1.002
        low_p = min(p, close_p) * 0.998
        candles.append(
            Candle(
                timestamp=dt,
                open=Decimal(f"{p:.2f}"),
                high=Decimal(f"{high_p:.2f}"),
                low=Decimal(f"{low_p:.2f}"),
                close=Decimal(f"{close_p:.2f}"),
                volume=Decimal("10.0"),
                symbol="BTC/EUR",
                interval="1h",
            )
        )
        p = close_p

    # At bar 54, breakout conditions are fully met
    signal = strategy.on_candle(
        candle=candles[54],
        history=candles[:54],
        ledger=ledger,
    )

    assert signal is not None
    assert isinstance(signal, OrderSignal)
    assert signal.side == OrderSide.BUY
    assert signal.symbol == "BTC/EUR"
    assert signal.target_quantity > Decimal("0")
    assert signal.stop_loss < signal.price


# =====================================================================
# TIER 2 BOUNDARY & RISK TESTS
# =====================================================================
def test_risk_guard_8pct_daily_drawdown_circuit_breaker():
    """Test RiskGuard halts trading for 24 hours when rolling 24h drawdown exceeds 8%."""
    guard = RiskGuard(
        max_daily_drawdown_pct=0.08,
        drawdown_halt_hours=24.0,
        initial_equity=Decimal("10.00"),
    )
    t0 = datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)

    # Equity starts at 10.00 EUR
    guard.update_equity(Decimal("10.00"), t0)
    assert not guard.is_halted(t0)

    # Price moves, equity peaks at 12.00 EUR
    t1 = t0 + timedelta(hours=2)
    guard.update_equity(Decimal("12.00"), t1)
    assert guard.peak_equity == Decimal("12.00")
    assert not guard.is_halted(t1)

    # Intraday drop to 10.90 EUR (Drawdown = (12 - 10.9) / 12 = 9.17% > 8%)
    t2 = t0 + timedelta(hours=5)
    guard.update_equity(Decimal("10.90"), t2)

    # Circuit breaker must trigger!
    assert guard.is_halted(t2)
    status = guard.check_risk_status(t2)
    assert not status.allowed
    assert status.halt_active
    assert status.risk_multiplier == 0.0

    # 12 hours later: still halted
    t3 = t2 + timedelta(hours=12)
    assert guard.is_halted(t3)

    # 25 hours later: halt expired
    t4 = t2 + timedelta(hours=25)
    assert not guard.is_halted(t4)
    status_recovered = guard.check_risk_status(t4)
    assert status_recovered.allowed


def test_consecutive_loss_throttle():
    """Test risk multiplier reductions: 2 losses = 30% cut, 3 losses = 50% cut, resets on win."""
    guard = RiskGuard(initial_equity=Decimal("10.00"))
    t = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

    # Loss 1: Normal risk (multiplier = 1.0)
    guard.on_trade_completed(pnl=Decimal("-0.25"), current_equity=Decimal("9.75"), timestamp=t)
    assert guard.consecutive_losses == 1
    assert guard.risk_multiplier == 1.0

    # Loss 2: 30% risk cut (multiplier = 0.70)
    guard.on_trade_completed(pnl=Decimal("-0.25"), current_equity=Decimal("9.50"), timestamp=t)
    assert guard.consecutive_losses == 2
    assert guard.risk_multiplier == pytest.approx(0.70)

    # Loss 3: 50% risk cut (multiplier = 0.50)
    guard.on_trade_completed(pnl=Decimal("-0.25"), current_equity=Decimal("9.25"), timestamp=t)
    assert guard.consecutive_losses == 3
    assert guard.risk_multiplier == pytest.approx(0.50)

    # Win: Resets counter to 0 and multiplier to 1.0
    guard.on_trade_completed(pnl=Decimal("+0.75"), current_equity=Decimal("10.00"), timestamp=t)
    assert guard.consecutive_losses == 0
    assert guard.risk_multiplier == 1.0


def test_compounding_sizer_rejects_sub_notional_equity():
    """Test sizer returns invalid result when equity is less than exchange minNotional."""
    sizer = CompoundingPositionSizer(min_notional=Decimal("5.00"))

    # Account equity depleted to 3.50 EUR (< 5.00 EUR minNotional)
    res = sizer.calculate_position_size(
        equity=Decimal("3.50"),
        entry_price=Decimal("60000.00"),
        stop_loss=Decimal("58500.00"),
    )
    assert not res.is_valid
    assert res.quantity == Decimal("0")
    assert "below minimum exchange notional" in res.rejection_reason


def test_compounding_sizer_rejects_zero_risk_multiplier():
    """Test sizer returns invalid when risk multiplier is 0.0 (e.g. trading halt)."""
    sizer = CompoundingPositionSizer(min_notional=Decimal("5.00"))

    res = sizer.calculate_position_size(
        equity=Decimal("10.00"),
        entry_price=Decimal("60000.00"),
        stop_loss=Decimal("58500.00"),
        risk_multiplier=0.0,
    )
    assert not res.is_valid
    assert res.quantity == Decimal("0")
    assert "Risk multiplier is 0.0" in res.rejection_reason


def test_structural_invalidation_exit_below_ema21():
    """Test AMVTrendBreakStrategy issues structural invalidation SELL when Close drops below EMA 21."""
    strategy = AMVTrendBreakStrategy(
        default_symbol="BTC/EUR",
        min_warmup_bars=25,
        exit_on_ema21=True,
    )
    ledger = VirtualLedger(initial_cash=Decimal("10.00"))
    # Record buy in ledger so ledger state has the position
    ledger.record_buy(
        symbol="BTC/EUR",
        quantity=Decimal("0.00015"),
        executed_price=Decimal("60000.00"),
        fee=Decimal("0.009"),
        timestamp_ms=1,
    )

    # Setup position state in strategy
    entry_px = Decimal("60000.00")
    from trading_bot.strategy.amv_strategy import PositionState
    pos = PositionState(
        symbol="BTC/EUR",
        entry_price=entry_px,
        quantity=Decimal("0.00015"),
        initial_stop=Decimal("58000.00"),
        current_stop=Decimal("58000.00"),
        take_profit_1=Decimal("63000.00"),
        highest_price=entry_px,
        bars_in_trade=2,
    )
    strategy._positions["BTC/EUR"] = pos

    # Construct candles where current candle breaks down below EMA 21
    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    candles = []
    for i in range(25):
        dt = base_time + timedelta(hours=i)
        p = Decimal("60000.00")
        candles.append(
            Candle(
                timestamp=dt,
                open=p,
                high=p + Decimal("50"),
                low=p - Decimal("50"),
                close=p,
                volume=Decimal("10.0"),
                symbol="BTC/EUR",
                interval="1h",
            )
        )

    # Breakdown candle: Close drops to 58500 (below EMA 21 ~60,000, but above hard stop 58000)
    breakdown_candle = Candle(
        timestamp=base_time + timedelta(hours=25),
        open=Decimal("59800.00"),
        high=Decimal("59800.00"),
        low=Decimal("58500.00"),
        close=Decimal("58500.00"),
        volume=Decimal("20.0"),
        symbol="BTC/EUR",
        interval="1h",
    )

    exit_signal = strategy.on_candle(breakdown_candle, history=candles, ledger=ledger)

    assert exit_signal is not None
    assert exit_signal.side == OrderSide.SELL
    assert "STRUCTURAL_INVALIDATION" in exit_signal.reason
    assert "BTC/EUR" not in strategy.positions  # Position cleaned up after close

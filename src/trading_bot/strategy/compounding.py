"""Three-Phase Compounding Engine and Downside Risk Guards.

Specifies mathematical position sizing and risk controls for scaling capital from
10.00 EUR towards 10,000.00 EUR:

1. Three-Phase Compounding Position Sizer:
   - Phase 1 (10€ to 100€): Micro-Cap Acceleration
     Allocates 95-98% capital into single high-conviction setup with tight 1.75x ATR
     stop loss, overcoming exchange minimum notional (5€) while allowing 15+
     consecutive losses without account failure.
   - Phase 2 (100€ to 1,000€): Transition & De-risking
     Fixed fractional position sizing risking 4.0% of portfolio equity per trade.
   - Phase 3 (1,000€ to 10,000€): Mid-Cap Scaling
     Quarter-Kelly sizing risking 2.0% of portfolio equity per trade with max 40%
     single-asset exposure limit.

2. Downside Risk Controls (RiskGuard):
   - 8% daily rolling drawdown circuit breaker (halts trading for 24h).
   - Consecutive loss throttle (30% risk reduction after 2 losses, 50% after 3 losses).
   - Exchange minimum notional validation.
"""

from __future__ import annotations
import sys

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class CompoundingPhase(str, Enum):
    """Phases of capital compounding from 10€ to 10,000€."""
    PHASE_1_MICRO = "PHASE_1_MICRO"       # 10€ - 100€
    PHASE_2_GROWTH = "PHASE_2_GROWTH"     # 100€ - 1,000€
    PHASE_3_SCALE = "PHASE_3_SCALE"       # 1,000€ - 10,000€+


@dataclass
class PositionSizeResult:
    """Result of compounding position size calculation."""
    quantity: Decimal
    notional: Decimal
    phase: CompoundingPhase
    risk_amount: Decimal
    risk_fraction: float
    is_valid: bool
    rejection_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskCheckResult:
    """Status of risk guard evaluation."""
    allowed: bool
    risk_multiplier: float
    reason: str
    halt_active: bool = False
    halt_expires_at: Optional[datetime] = None


class RiskGuard:
    """Automated downside risk controls and circuit breakers.

    Features:
    - 8% Rolling Daily Drawdown Limit: halts new trades for 24 hours if equity drops >=8%
      from its 24-hour rolling peak.
    - Consecutive Loss Throttle: reduces risk multiplier to 0.70 (30% cut) after 2 consecutive
      losses, and to 0.50 (50% cut) after 3+ consecutive losses. Resets to 1.0 on a win.
    """

    def __init__(
        self,
        max_daily_drawdown_pct: float = 0.08,
        drawdown_halt_hours: float = 24.0,
        initial_equity: Union[Decimal, float] = Decimal("10.00"),
    ):
        self.max_daily_drawdown_pct = float(max_daily_drawdown_pct)
        self.drawdown_halt_duration = timedelta(hours=drawdown_halt_hours)

        self._consecutive_losses: int = 0
        self._risk_multiplier: float = 1.0
        self._halt_until: Optional[datetime] = None
        self._halt_reason: str = ""

        # History of (timestamp, equity) for rolling peak tracking
        init_eq = Decimal(str(initial_equity))
        self._equity_history: List[Tuple[datetime, Decimal]] = []
        self._peak_equity: Decimal = init_eq
        self._current_equity: Decimal = init_eq

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def risk_multiplier(self) -> float:
        return self._risk_multiplier

    @property
    def peak_equity(self) -> Decimal:
        return self._peak_equity

    @property
    def current_equity(self) -> Decimal:
        return self._current_equity

    def update_equity(
        self,
        equity: Union[Decimal, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record current equity and evaluate 24-hour rolling drawdown."""
        now = timestamp or datetime.now(timezone.utc)
        eq = Decimal(str(equity))
        self._current_equity = eq

        # Store in rolling history
        self._equity_history.append((now, eq))

        # Prune history older than 24 hours
        cutoff = now - timedelta(hours=24)
        self._equity_history = [
            (t, e) for t, e in self._equity_history if t >= cutoff
        ]

        # Calculate rolling 24-hour peak
        if self._equity_history:
            rolling_peak = max(e for _, e in self._equity_history)
        else:
            rolling_peak = eq

        if rolling_peak > self._peak_equity:
            self._peak_equity = rolling_peak

        # Check rolling 24h drawdown
        if rolling_peak > Decimal("0"):
            dd_pct = float((rolling_peak - eq) / rolling_peak)
            if dd_pct >= self.max_daily_drawdown_pct:
                self._trigger_drawdown_halt(
                    now,
                    dd_pct,
                    f"Rolling 24h drawdown {dd_pct*100:.2f}% exceeded limit {self.max_daily_drawdown_pct*100:.1f}%",
                )

    def _trigger_drawdown_halt(self, timestamp: datetime, dd_pct: float, reason: str) -> None:
        """Trigger 24-hour circuit breaker."""
        self._halt_until = timestamp + self.drawdown_halt_duration
        self._halt_reason = reason
        self._risk_multiplier = 0.0

    def on_trade_completed(
        self,
        pnl: Union[Decimal, float],
        current_equity: Union[Decimal, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update consecutive loss counter, risk throttle, and equity after a closed trade."""
        now = timestamp or datetime.now(timezone.utc)
        pnl_dec = Decimal(str(pnl))
        eq_dec = Decimal(str(current_equity))

        self.update_equity(eq_dec, now)

        if pnl_dec < Decimal("0"):
            self._consecutive_losses += 1
            if self._consecutive_losses == 1:
                self._risk_multiplier = 1.0
            elif self._consecutive_losses == 2:
                self._risk_multiplier = 0.70  # 30% risk cut
            else:  # >= 3
                self._risk_multiplier = 0.50  # 50% risk cut
        elif pnl_dec > Decimal("0"):
            self._consecutive_losses = 0
            # Reset multiplier unless currently halted
            if not self.is_halted(now):
                self._risk_multiplier = 1.0

    def is_halted(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if trading halt is currently in effect."""
        if self._halt_until is None:
            return False
        now = timestamp or datetime.now(timezone.utc)
        if now >= self._halt_until:
            # Halt has expired
            self._halt_until = None
            self._halt_reason = ""
            # Restore multiplier based on loss streak
            if self._consecutive_losses == 0:
                self._risk_multiplier = 1.0
            elif self._consecutive_losses == 1:
                self._risk_multiplier = 1.0
            elif self._consecutive_losses == 2:
                self._risk_multiplier = 0.70
            else:
                self._risk_multiplier = 0.50
            return False
        return True

    def check_risk_status(self, timestamp: Optional[datetime] = None) -> RiskCheckResult:
        """Assess whether a new trade is permitted under active risk controls."""
        now = timestamp or datetime.now(timezone.utc)
        
        is_h = self.is_halted(now)
        if timestamp and timestamp.year == 2024 and timestamp.month == 9 and timestamp.day == 24 and timestamp.hour == 16:
            print(f"[{timestamp}] TOP OF check_risk_status: is_halted={is_h} halt_until={self._halt_until}", file=sys.stderr)

        if is_h:
            return RiskCheckResult(
                allowed=False,
                risk_multiplier=0.0,
                reason=f"Trading halted: {self._halt_reason} (until {self._halt_until})",
                halt_active=True,
                halt_expires_at=self._halt_until,
            )

        throttle_desc = (
            f"consecutive losses: {self._consecutive_losses}, multiplier: {self._risk_multiplier:.2f}"
            if self._consecutive_losses > 1
            else "normal risk"
        )
        
        if timestamp and timestamp.year == 2024 and timestamp.month == 9 and timestamp.day == 24 and timestamp.hour == 16:
            print(f"[{timestamp}] check_risk_status: allowed=True, r_mult={self._risk_multiplier}, is_halted={self.is_halted(now)} halt_until={self._halt_until}", file=sys.stderr)

        return RiskCheckResult(
            allowed=True,
            risk_multiplier=self._risk_multiplier,
            reason=f"Trading permitted ({throttle_desc})",
            halt_active=False,
            halt_expires_at=None,
        )

    def reset(self, initial_equity: Union[Decimal, float] = Decimal("10.00")) -> None:
        """Reset risk guard state."""
        self._consecutive_losses = 0
        self._risk_multiplier = 1.0
        self._halt_until = None
        self._halt_reason = ""
        eq = Decimal(str(initial_equity))
        self._peak_equity = eq
        self._current_equity = eq
        self._equity_history.clear()


class CompoundingPositionSizer:
    """Three-Phase Compounding Position Sizer.

    Scales position sizing through 3 distinct capital tiers:
    - Phase 1 (10€ - 100€): Allocates 96% capital with 1.75x ATR stop loss (dollar risk ~0.30€).
    - Phase 2 (100€ - 1,000€): Fixed fractional risk (4% risk per trade).
    - Phase 3 (1,000€ - 10,000€): Quarter-Kelly risk (2% risk per trade, max 40% asset allocation).
    """

    def __init__(
        self,
        phase_1_threshold: Union[Decimal, float] = Decimal("100.00"),
        phase_2_threshold: Union[Decimal, float] = Decimal("1000.00"),
        phase_1_allocation_pct: float = 0.96,
        phase_2_risk_pct: float = 0.04,
        phase_2_max_allocation_pct: float = 0.85,
        phase_3_risk_pct: float = 0.02,
        phase_3_max_allocation_pct: float = 0.40,
        min_notional: Union[Decimal, float] = Decimal("5.00"),
        default_step_size: Union[Decimal, float] = Decimal("0.001"),
        atr_multiplier: Union[Decimal, float] = Decimal("1.75"),
    ):
        self.phase_1_threshold = Decimal(str(phase_1_threshold))
        self.phase_2_threshold = Decimal(str(phase_2_threshold))
        self.phase_1_allocation_pct = float(phase_1_allocation_pct)
        self.phase_2_risk_pct = float(phase_2_risk_pct)
        self.phase_2_max_allocation_pct = float(phase_2_max_allocation_pct)
        self.phase_3_risk_pct = float(phase_3_risk_pct)
        self.phase_3_max_allocation_pct = float(phase_3_max_allocation_pct)
        self.min_notional = Decimal(str(min_notional))
        self.default_step_size = Decimal(str(default_step_size))
        self.atr_multiplier = Decimal(str(atr_multiplier))

    def get_phase(self, equity: Union[Decimal, float]) -> CompoundingPhase:
        """Determine current compounding phase based on equity."""
        eq = Decimal(str(equity))
        if eq <= self.phase_1_threshold:
            return CompoundingPhase.PHASE_1_MICRO
        elif eq <= self.phase_2_threshold:
            return CompoundingPhase.PHASE_2_GROWTH
        else:
            return CompoundingPhase.PHASE_3_SCALE

    def calculate_position_size(
        self,
        equity: Union[Decimal, float],
        entry_price: Union[Decimal, float],
        stop_loss: Union[Decimal, float],
        risk_multiplier: float = 1.0,
        min_notional: Optional[Union[Decimal, float]] = None,
        step_size: Optional[Union[Decimal, float]] = None,
    ) -> PositionSizeResult:
        """Calculate exact position quantity and notional based on equity and stops.

        Args:
            equity: Total account equity.
            entry_price: Projected entry fill price.
            stop_loss: Protective stop loss price.
            risk_multiplier: Multiplier from RiskGuard (e.g. 0.70 or 0.50 after losses).
            min_notional: Minimum order notional required by exchange (defaults to self.min_notional).
            step_size: Asset lot step size for precision quantizing (e.g. 0.001).

        Returns:
            PositionSizeResult containing quantized quantity, notional, phase, and validation.
        """
        eq = Decimal(str(equity))
        entry = Decimal(str(entry_price))
        stop = Decimal(str(stop_loss))
        min_not = Decimal(str(min_notional)) if min_notional is not None else self.min_notional
        step = Decimal(str(step_size)) if step_size is not None else self.default_step_size
        mult = Decimal(str(max(0.0, risk_multiplier)))

        # 1. Minimum Notional Guard
        if eq < min_not:
            return PositionSizeResult(
                quantity=Decimal("0"),
                notional=Decimal("0"),
                phase=self.get_phase(eq),
                risk_amount=Decimal("0"),
                risk_fraction=0.0,
                is_valid=False,
                rejection_reason=f"Account equity ({eq:.2f}) is below minimum exchange notional ({min_not:.2f})",
            )

        if mult == Decimal("0"):
            return PositionSizeResult(
                quantity=Decimal("0"),
                notional=Decimal("0"),
                phase=self.get_phase(eq),
                risk_amount=Decimal("0"),
                risk_fraction=0.0,
                is_valid=False,
                rejection_reason="Risk multiplier is 0.0 (trading halted or circuit breaker active)",
            )

        if entry <= Decimal("0"):
            return PositionSizeResult(
                quantity=Decimal("0"),
                notional=Decimal("0"),
                phase=self.get_phase(eq),
                risk_amount=Decimal("0"),
                risk_fraction=0.0,
                is_valid=False,
                rejection_reason=f"Invalid entry price: {entry}",
            )

        # Calculate stop distance
        stop_dist = abs(entry - stop)
        if stop_dist <= Decimal("0"):
            # Fallback 2% stop distance
            stop_dist = entry * Decimal("0.02")

        stop_pct = stop_dist / entry
        phase = self.get_phase(eq)

        # Binance Dust Trap Prevention:
        # Ensure that even if the trade hits its initial stop-loss, the remaining exit notional
        # will still be strictly >= min_notional.
        # e.g. min_viable_entry = min_notional / (1 - stop_loss_pct) * 1.05 (with min 6% buffer, >= 5.30 EUR on Binance 5.00 EUR min_notional)
        if stop_pct < Decimal("0.90"):
            min_viable_entry = (min_not / (Decimal("1") - stop_pct)) * Decimal("1.05")
        else:
            min_viable_entry = min_not * Decimal("1.10")
        min_viable_entry = max(min_viable_entry, min_not * Decimal("1.06"))

        # 2. Phase-specific sizing computation
        if phase == CompoundingPhase.PHASE_1_MICRO:
            # Phase 1: Micro-Cap Acceleration
            max_affordable = eq * Decimal("0.98")
            if max_affordable < min_viable_entry:
                return PositionSizeResult(
                    quantity=Decimal("0"),
                    notional=Decimal("0"),
                    phase=phase,
                    risk_amount=Decimal("0"),
                    risk_fraction=0.0,
                    is_valid=False,
                    rejection_reason=(
                        f"Account equity ({eq:.2f}) cannot afford minimum viable entry "
                        f"({min_viable_entry:.2f}) required to avoid dust trap at stop loss"
                    ),
                )

            alloc_pct = Decimal(str(self.phase_1_allocation_pct)) * mult
            target_notional = eq * alloc_pct
            # Ensure target_notional satisfies min_viable_entry to prevent dust trap
            target_notional = max(target_notional, min_viable_entry)
            target_notional = min(target_notional, max_affordable)
            raw_qty = target_notional / entry
            risk_amount = raw_qty * stop_dist
            risk_frac = float(risk_amount / eq)

        elif phase == CompoundingPhase.PHASE_2_GROWTH:
            # Phase 2: Fixed Fractional (4% risk per trade)
            risk_pct = Decimal(str(self.phase_2_risk_pct)) * mult
            risk_amount = eq * risk_pct
            target_notional = risk_amount / stop_pct
            max_notional = eq * Decimal(str(self.phase_2_max_allocation_pct))
            final_notional = min(target_notional, max_notional)
            raw_qty = final_notional / entry
            risk_frac = float(risk_amount / eq)

        else:
            # Phase 3: Quarter-Kelly (2% risk per trade, max 40% asset allocation)
            risk_pct = Decimal(str(self.phase_3_risk_pct)) * mult
            risk_amount = eq * risk_pct
            target_notional = risk_amount / stop_pct
            max_notional = eq * Decimal(str(self.phase_3_max_allocation_pct))
            final_notional = min(target_notional, max_notional)
            raw_qty = final_notional / entry
            risk_frac = float(risk_amount / eq)

        # 3. Quantize lot size to step_size (ROUND_DOWN to prevent exceeding cash balance)
        if step > Decimal("0"):
            # Round down to nearest step
            steps_count = (raw_qty / step).quantize(Decimal("1"), rounding=ROUND_DOWN)
            quantized_qty = steps_count * step
            # In Phase 1, ensure quantization does not drop below min_viable_entry if affordable
            if phase == CompoundingPhase.PHASE_1_MICRO and (quantized_qty * entry) < min_viable_entry:
                candidate_qty = quantized_qty + step
                if (candidate_qty * entry) <= (eq * Decimal("0.98")):
                    quantized_qty = candidate_qty
        else:
            quantized_qty = raw_qty

        actual_notional = quantized_qty * entry

        # 4. Validate against minimum notional
        if actual_notional < min_not:
            return PositionSizeResult(
                quantity=Decimal("0"),
                notional=Decimal("0"),
                phase=phase,
                risk_amount=Decimal("0"),
                risk_fraction=0.0,
                is_valid=False,
                rejection_reason=f"Order notional ({actual_notional:.2f}) is below minimum exchange notional ({min_not:.2f})",
            )
        elif phase == CompoundingPhase.PHASE_1_MICRO and actual_notional < min_viable_entry:
            return PositionSizeResult(
                quantity=Decimal("0"),
                notional=Decimal("0"),
                phase=phase,
                risk_amount=Decimal("0"),
                risk_fraction=0.0,
                is_valid=False,
                rejection_reason=f"Order notional ({actual_notional:.2f}) is below minimum viable entry ({min_viable_entry:.2f}) with stop-loss buffer",
            )


        return PositionSizeResult(
            quantity=quantized_qty,
            notional=actual_notional,
            phase=phase,
            risk_amount=risk_amount,
            risk_fraction=risk_frac,
            is_valid=True,
            details={
                "equity": float(eq),
                "entry_price": float(entry),
                "stop_loss": float(stop),
                "stop_distance": float(stop_dist),
                "stop_distance_pct": float(stop_pct),
                "risk_multiplier": float(mult),
            },
        )

"""
Quantitative Performance Metrics Suite for Algorithmic Trading.

Computes comprehensive portfolio and execution metrics:
- Total Return %
- Compound Annual Growth Rate (CAGR) %
- Annualized Volatility %
- Maximum Drawdown % and duration
- Sharpe Ratio (Annualized)
- Sortino Ratio (Annualized Downside Deviation)
- Win Rate %, Loss Rate %
- Profit Factor, Payoff Ratio, Mathematical Expectancy
- Progress towards 10,000 EUR compounding goal
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from trading_bot.simulation.ledger import TradeRecord, VirtualLedger


@dataclass
class PerformanceMetrics:
    """Quantitative performance summary of trading strategy and ledger."""

    initial_balance: Decimal = Decimal("10.00")
    final_balance: Decimal = Decimal("10.00")
    net_profit: Decimal = Decimal("0.00")
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    annualized_volatility_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0
    peak_equity: Decimal = Decimal("10.00")
    trough_equity: Decimal = Decimal("10.00")
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    total_trades: int = 0
    closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate_pct: float = 0.0
    loss_rate_pct: float = 0.0
    profit_factor: float = 0.0
    gross_profit: Decimal = Decimal("0.00")
    gross_loss: Decimal = Decimal("0.00")
    total_fees_paid: Decimal = Decimal("0.00")
    payoff_ratio: float = 0.0
    expectancy: float = 0.0
    progress_to_10k_pct: float = 0.0

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like attribute access (e.g. metrics['win_rate_pct'])."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Unknown metric key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Safe getter with default."""
        return getattr(self, key, default)

    @classmethod
    def empty(cls, initial_balance: Union[Decimal, float, str] = Decimal("10.00")) -> PerformanceMetrics:
        """Return a blank zeroed metrics report."""
        init = Decimal(str(initial_balance))
        return cls(
            initial_balance=init,
            final_balance=init,
            net_profit=Decimal("0.00"),
            peak_equity=init,
            trough_equity=init,
        )

    @classmethod
    def calculate(
        cls,
        equity_curve: Sequence[Union[Decimal, float, int]],
        trades: Sequence[TradeRecord],
        initial_balance: Optional[Union[Decimal, float, int]] = None,
        final_balance: Optional[Union[Decimal, float, int]] = None,
        period_days: Optional[float] = None,
        periods_per_year: float = 8760.0,  # Default 8760 hourly bars/year
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics from an equity curve and trade record sequence.
        """
        # Parse equity curve
        curve: List[float] = [float(val) for val in equity_curve]

        if not curve:
            init_val = Decimal(str(initial_balance or 10.0))
            fin_val = Decimal(str(final_balance or init_val))
            return cls(
                initial_balance=init_val,
                final_balance=fin_val,
                net_profit=fin_val - init_val,
                peak_equity=max(init_val, fin_val),
                trough_equity=min(init_val, fin_val),
            )

        init_float = float(initial_balance) if initial_balance is not None else curve[0]
        fin_float = float(final_balance) if final_balance is not None else curve[-1]

        init_dec = Decimal(f"{init_float:.4f}")
        fin_dec = Decimal(f"{fin_float:.4f}")
        net_profit_dec = fin_dec - init_dec

        # 1. Total Return %
        if init_float > 0:
            total_return = ((fin_float - init_float) / init_float) * 100.0
        else:
            total_return = 0.0

        # 2. CAGR %
        cagr = 0.0
        if period_days and period_days >= 1.0 and init_float > 0 and fin_float > 0:
            years = period_days / 365.25
            try:
                cagr = ((fin_float / init_float) ** (1.0 / years) - 1.0) * 100.0
            except (OverflowError, ZeroDivisionError, ValueError):
                cagr = 0.0

        # 3. Maximum Drawdown % & Duration
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_duration = 0
        peak = curve[0]
        trough = curve[0]

        for val in curve:
            if val > peak:
                peak = val
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration

            if val < trough:
                trough = val

            if peak > 0:
                dd = ((peak - val) / peak) * 100.0
                if dd > max_dd:
                    max_dd = dd

        # 4. Period returns, Volatility, Sharpe & Sortino
        returns: List[float] = []
        for i in range(1, len(curve)):
            prev = curve[i - 1]
            curr = curve[i]
            if prev > 0:
                returns.append((curr - prev) / prev)

        ann_vol = 0.0
        sharpe = 0.0
        sortino = 0.0

        if len(returns) >= 2:
            n = len(returns)
            mean_ret = sum(returns) / n
            variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 0.0

            if std_dev > 0:
                ann_vol = std_dev * math.sqrt(periods_per_year) * 100.0
                sharpe = (mean_ret / std_dev) * math.sqrt(periods_per_year)

            # Sortino: downside standard deviation
            downside_sq_sum = sum(min(0.0, r) ** 2 for r in returns)
            downside_dev = math.sqrt(downside_sq_sum / n) if downside_sq_sum > 0 else 0.0

            if downside_dev > 0:
                sortino = (mean_ret / downside_dev) * math.sqrt(periods_per_year)
            elif mean_ret > 0:
                sortino = 999.0  # Capped at 999.0 to avoid non-standard JSON inf
            else:
                sortino = 0.0

        # 5. Trade Statistics (Closed sell trades or all trades)
        closed_sells = [t for t in trades if t.side == "SELL"]
        total_trades_count = len(trades)
        closed_count = len(closed_sells)

        winning = [t for t in closed_sells if t.realized_pnl > Decimal("0.0")]
        losing = [t for t in closed_sells if t.realized_pnl < Decimal("0.0")]
        breakeven = [t for t in closed_sells if t.realized_pnl == Decimal("0.0")]

        winning_count = len(winning)
        losing_count = len(losing)
        breakeven_count = len(breakeven)

        win_rate = (winning_count / closed_count * 100.0) if closed_count > 0 else 0.0
        loss_rate = (losing_count / closed_count * 100.0) if closed_count > 0 else 0.0

        gross_profit = sum((t.realized_pnl for t in winning), Decimal("0.00"))
        gross_loss = sum((abs(t.realized_pnl) for t in losing), Decimal("0.00"))
        total_fees = sum((t.fee for t in trades), Decimal("0.00"))

        if gross_loss > Decimal("0.0"):
            profit_factor = float(gross_profit / gross_loss)
        elif gross_profit > Decimal("0.0"):
            profit_factor = 999.0  # Cap at 999.0 for infinite profit factor
        else:
            profit_factor = 0.0

        # Payoff Ratio & Expectancy
        avg_win = float(gross_profit / Decimal(winning_count)) if winning_count > 0 else 0.0
        avg_loss = float(gross_loss / Decimal(losing_count)) if losing_count > 0 else 0.0

        if avg_loss > 0:
            payoff_ratio = avg_win / avg_loss
        elif avg_win > 0:
            payoff_ratio = 999.0
        else:
            payoff_ratio = 0.0

        # Mathematical expectancy in currency
        p_win = win_rate / 100.0
        p_loss = loss_rate / 100.0
        expectancy = (p_win * avg_win) - (p_loss * avg_loss)

        # Progress to 10k target (from 10€ base)
        target_goal = 10000.0
        start_goal = 10.0
        progress_pct = max(0.0, min(100.0, ((fin_float - start_goal) / (target_goal - start_goal)) * 100.0))

        return cls(
            initial_balance=init_dec,
            final_balance=fin_dec,
            net_profit=net_profit_dec,
            total_return_pct=round(total_return, 2),
            cagr_pct=round(cagr, 2),
            annualized_volatility_pct=round(ann_vol, 2),
            max_drawdown_pct=round(max_dd, 2),
            max_drawdown_duration_bars=max_dd_duration,
            peak_equity=Decimal(f"{peak:.4f}"),
            trough_equity=Decimal(f"{trough:.4f}"),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            total_trades=total_trades_count,
            closed_trades=closed_count,
            winning_trades=winning_count,
            losing_trades=losing_count,
            breakeven_trades=breakeven_count,
            win_rate_pct=round(win_rate, 2),
            loss_rate_pct=round(loss_rate, 2),
            profit_factor=round(profit_factor, 2),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            total_fees_paid=total_fees,
            payoff_ratio=round(payoff_ratio, 2),
            expectancy=round(expectancy, 4),
            progress_to_10k_pct=round(progress_pct, 4),
        )

    @classmethod
    def from_ledger(
        cls,
        ledger: VirtualLedger,
        period_days: Optional[float] = None,
        periods_per_year: float = 8760.0,
        current_prices: Optional[Dict[str, Decimal]] = None,
    ) -> PerformanceMetrics:
        """
        Extract metrics directly from a VirtualLedger instance.
        """
        final_eq = ledger.total_equity(current_prices)

        # Build equity curve: start with initial_cash, add snapshots, end with final_eq
        curve: List[float] = [float(ledger.initial_cash)]
        for _, eq in ledger.equity_history:
            curve.append(float(eq))
        curve.append(float(final_eq))

        # Infer period_days from equity_history timestamps if not provided
        if period_days is None and len(ledger.equity_history) >= 2:
            start_ts = ledger.equity_history[0][0]
            end_ts = ledger.equity_history[-1][0]
            diff_ms = end_ts - start_ts
            if diff_ms > 0:
                period_days = diff_ms / (1000.0 * 86400.0)

        return cls.calculate(
            equity_curve=curve,
            trades=ledger.trades,
            initial_balance=ledger.initial_cash,
            final_balance=final_eq,
            period_days=period_days,
            periods_per_year=periods_per_year,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to clean JSON-serializable dictionary."""
        return {
            "initial_balance": float(self.initial_balance),
            "final_balance": float(self.final_balance),
            "net_profit": float(self.net_profit),
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "annualized_volatility_pct": self.annualized_volatility_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_bars": self.max_drawdown_duration_bars,
            "peak_equity": float(self.peak_equity),
            "trough_equity": float(self.trough_equity),
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "total_trades": self.total_trades,
            "closed_trades": self.closed_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate_pct": self.win_rate_pct,
            "loss_rate_pct": self.loss_rate_pct,
            "profit_factor": self.profit_factor,
            "gross_profit": float(self.gross_profit),
            "gross_loss": float(self.gross_loss),
            "total_fees_paid": float(self.total_fees_paid),
            "payoff_ratio": self.payoff_ratio,
            "expectancy": self.expectancy,
            "progress_to_10k_pct": self.progress_to_10k_pct,
        }

    def summary(self) -> str:
        """Format a concise terminal summary string."""
        return (
            f"Return: {self.total_return_pct:+.2f}% | "
            f"Net: {float(self.net_profit):+.2f} EUR | "
            f"MaxDD: -{self.max_drawdown_pct:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"WinRate: {self.win_rate_pct:.1f}% ({self.winning_trades}/{self.closed_trades}) | "
            f"PF: {self.profit_factor:.2f}"
        )

    def to_summary_table(self) -> str:
        """Generate structured ASCII report matching CLI output requirements."""
        lines = [
            "--------------------------------------------------------------------------------",
            "PORTFOLIO SUMMARY:",
            f"  Initial Balance:         {float(self.initial_balance):.2f} EUR",
            f"  Final Balance:           {float(self.final_balance):.2f} EUR",
            f"  Net Profit:              {float(self.net_profit):+.2f} EUR ({self.total_return_pct:+.2f}%)",
            f"  Progress to 10,000€:     {self.progress_to_10k_pct:.2f}%",
            "--------------------------------------------------------------------------------",
            "TRADING PERFORMANCE METRICS:",
            f"  Total Trades Executed:   {self.total_trades}",
            f"  Winning Trades:          {self.winning_trades} ({self.win_rate_pct:.2f}%)",
            f"  Losing Trades:           {self.losing_trades} ({self.loss_rate_pct:.2f}%)",
            f"  Profit Factor:           {self.profit_factor:.2f}",
            f"  Max Drawdown:            -{self.max_drawdown_pct:.2f}%",
            f"  Sharpe Ratio:            {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:           {self.sortino_ratio:.2f}",
            f"  Annualized Volatility:   {self.annualized_volatility_pct:.2f}%",
            f"  Total Fees Paid:         {float(self.total_fees_paid):.2f} EUR",
            "--------------------------------------------------------------------------------",
        ]
        return "\n".join(lines)

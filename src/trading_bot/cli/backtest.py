"""Programmatic Backtesting CLI Command.

Executes offline deterministic historical simulation candle-by-candle
using MarketDataService, VirtualLedger, AMVTrendBreakStrategy,
and ExecutionSimulator.

Fulfills Acceptance Criteria 1:
- Runs successfully over >= 1 month of historical data.
- Starts with 10.00 EUR (configurable).
- Outputs final simulated balance, trade list, and performance metrics.
- Exits with return code 0 without crashing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trading_bot.config import (
    DEFAULT_TIMEFRAME,
    normalize_symbol,
    to_display_symbol,
)
from trading_bot.data.cache import MarketDataService
from trading_bot.data.models import Candle
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.metrics import PerformanceMetrics
from trading_bot.simulation.simulator import (
    EXCHANGE_MIN_NOTIONAL,
    ExecutionSimulator,
)
from trading_bot.strategy import (
    AMVTrendBreakStrategy,
    MicroGridStrategy,
    MultiRegimeStrategyOrchestrator,
)



logger = logging.getLogger("backtest_cli")

# Standardized UNIX exit codes conforming to spec
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CLI_USAGE_ERROR = 2
EXIT_DATA_FETCH_ERROR = 3
EXIT_INSUFFICIENT_FUNDS_ERROR = 4
EXIT_CONFIG_ERROR = 5


def parse_datetime(val: str) -> datetime:
    """Parse ISO or date string into UTC datetime."""
    val = val.strip()
    # Try ISO format
    try:
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    # Try common date formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(val, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Invalid date format: '{val}'. Expected YYYY-MM-DD or ISO-8601 string.")


def resolve_step_size(symbol: str, explicit_step: Optional[float] = None) -> Decimal:
    """Resolve asset step size based on symbol or explicit override."""
    if explicit_step is not None and explicit_step > 0:
        return Decimal(str(explicit_step))
    norm = normalize_symbol(symbol)
    if "BTC" in norm:
        return Decimal("0.00001")  # Binance BTC step size
    elif "ETH" in norm:
        return Decimal("0.0001")   # Binance ETH step size
    elif "SOL" in norm:
        return Decimal("0.001")    # Binance SOL step size
    return Decimal("0.00001")


def run_backtest(
    pair: str = "BTC/EUR",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    balance: float = 10.0,
    currency: str = "EUR",
    timeframe: str = "1h",
    strategy_name: str = "amv_trendbreak",
    exchange: str = "binance",
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    step_size: Optional[float] = None,
    market_data_service: Optional[MarketDataService] = None,
) -> Tuple[VirtualLedger, PerformanceMetrics, List[Candle]]:
    """Execute programmatic historical simulation candle-by-candle.

    Returns:
        (ledger, performance_metrics, candles)
    """
    if balance <= 0:
        raise ValueError(f"Initial balance must be > 0, got {balance}")

    now_utc = datetime.now(timezone.utc)
    end_dt = end or now_utc
    start_dt = start or (end_dt - timedelta(days=31))

    if end_dt <= start_dt:
        raise ValueError(f"End date ({end_dt}) must be strictly after start date ({start_dt})")

    norm_symbol = normalize_symbol(pair)
    service = market_data_service or MarketDataService()

    # Fetch historical candles
    try:
        candles = service.get_historical_candles(
            symbol=norm_symbol,
            interval=timeframe,
            start_time=start_dt,
            end_time=end_dt,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load historical candles for {norm_symbol}: {exc}") from exc

    if not candles:
        raise RuntimeError(f"No historical candles returned for {norm_symbol} {timeframe} between {start_dt} and {end_dt}")

    # Set up simulation components
    initial_cash_dec = Decimal(str(balance))
    ledger = VirtualLedger(initial_cash=initial_cash_dec, quote_currency=currency)

    min_notional_dec = EXCHANGE_MIN_NOTIONAL.get(exchange.lower(), Decimal("5.00"))
    step_size_dec = resolve_step_size(pair, step_size)
    fee_dec = Decimal(str(fee_rate))
    slippage_dec = Decimal(str(slippage))

    simulator = ExecutionSimulator(
        fee_rate=fee_dec,
        slippage=slippage_dec,
        min_notional=min_notional_dec,
        step_size=step_size_dec,
        exchange=exchange,
    )

    strat_lower = strategy_name.lower().replace("-", "_")
    if strat_lower in ("multi_regime", "multiregime", "hybrid"):
        strategy: Any = MultiRegimeStrategyOrchestrator(
            default_symbol=to_display_symbol(pair),
            exchange_min_notional=min_notional_dec,
            step_size=step_size_dec,
        )
    elif strat_lower in ("grid", "micro_grid", "microgrid"):
        strategy = MicroGridStrategy(
            default_symbol=to_display_symbol(pair),
            min_notional=min_notional_dec,
            step_size=step_size_dec,
        )
    else:
        strategy = AMVTrendBreakStrategy(
            default_symbol=to_display_symbol(pair),
            exchange_min_notional=min_notional_dec,
            step_size=step_size_dec,
        )


    # Candle-by-candle simulation loop
    for i in range(len(candles)):
        current_candle = candles[i]
        history = candles[:i]

        signal = strategy.on_candle(current_candle, history, ledger)
        if signal is not None:
            exec_res = simulator.execute(
                order=signal,
                current_price=current_candle.close,
                ledger=ledger,
                timestamp_ms=current_candle.open_time_ms,
            )
            if exec_res.success and exec_res.side == "SELL":
                strategy.notify_position_closed(
                    symbol=signal.symbol,
                    exit_price=exec_res.executed_price,
                    current_equity=ledger.total_equity(),
                    timestamp=current_candle.timestamp,
                )

        ledger.record_equity_snapshot(
            current_candle.open_time_ms,
            {norm_symbol: current_candle.close, to_display_symbol(pair): current_candle.close}
        )

    # Calculate quantitative performance metrics
    last_price = candles[-1].close
    metrics = PerformanceMetrics.from_ledger(
        ledger=ledger,
        current_prices={norm_symbol: last_price, to_display_symbol(pair): last_price}
    )

    return ledger, metrics, candles


def format_table_report(
    pair: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    strategy_name: str,
    fee_rate: float,
    slippage: float,
    ledger: VirtualLedger,
    metrics: PerformanceMetrics,
    candles: List[Candle],
) -> str:
    """Format human-readable ASCII table report conforming to formal spec."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("                    AUTONOMOUS TRADING BOT - BACKTEST REPORT                    ")
    lines.append("=" * 80)
    lines.append(
        f"Pair: {to_display_symbol(pair)} | Timeframe: {timeframe} | "
        f"Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} ({len(candles)} bars)"
    )
    lines.append(
        f"Strategy: {strategy_name} | Fee Rate: {fee_rate * 100:.2f}% | Slippage: {slippage * 100:.2f}%"
    )
    lines.append("-" * 80)
    lines.append("PORTFOLIO SUMMARY:")
    lines.append(f"  Initial Balance:         {metrics.initial_balance:.2f} {ledger.quote_currency}")
    lines.append(f"  Final Balance:           {metrics.final_balance:.2f} {ledger.quote_currency}")
    lines.append(
        f"  Net Profit:              {metrics.net_profit:+.2f} {ledger.quote_currency} ({metrics.total_return_pct:+.2f}%)"
    )
    lines.append(f"  Progress to 10,000€:     {metrics.progress_to_10k_pct:.2f}%")
    lines.append("-" * 80)
    lines.append("TRADING PERFORMANCE METRICS:")
    lines.append(f"  Total Trades Executed:   {metrics.total_trades}")
    lines.append(f"  Winning Trades:          {metrics.winning_trades} ({metrics.win_rate_pct:.2f}%)")
    lines.append(f"  Losing Trades:           {metrics.losing_trades} ({metrics.loss_rate_pct:.2f}%)")
    lines.append(f"  Profit Factor:           {metrics.profit_factor:.2f}")
    lines.append(f"  Max Drawdown:            -{metrics.max_drawdown_pct:.2f}%")
    lines.append(f"  Sharpe Ratio:            {metrics.sharpe_ratio:.2f}")
    lines.append(f"  Total Fees Paid:         {metrics.total_fees_paid:.2f} {ledger.quote_currency}")
    lines.append("-" * 80)
    lines.append("EXECUTED TRADES LOG:")

    if not ledger.trades:
        lines.append("  (No trades executed during this time period)")
    else:
        header = f"{'#':<4} {'Side':<5} {'Timestamp (UTC)':<20} {'Price':<12} {'Qty':<10} {'Notional':<10} {'Fee':<8} {'PnL':<10} {'Exit Reason'}"
        lines.append(header)
        lines.append("-" * len(header))
        for idx, trade in enumerate(ledger.trades, start=1):
            ts_str = trade.timestamp.strftime("%Y-%m-%d %H:%M")
            pnl_str = f"{trade.realized_pnl:+.2f}" if trade.side == "SELL" else "-"
            reason = trade.exit_reason or "-"
            row = (
                f"{idx:<4} {trade.side:<5} {ts_str:<20} "
                f"{trade.executed_price:<12.2f} {trade.quantity:<10.5f} "
                f"{trade.quote_notional:<10.2f} {trade.fee:<8.4f} "
                f"{pnl_str:<10} {reason}"
            )
            lines.append(row)

    lines.append("=" * 80)
    lines.append("[SUCCESS] Backtest completed without errors.")
    return "\n".join(lines)


def format_json_report(
    pair: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    balance: float,
    currency: str,
    strategy_name: str,
    fee_rate: float,
    slippage: float,
    ledger: VirtualLedger,
    metrics: PerformanceMetrics,
    candles: List[Candle],
) -> str:
    """Format machine-readable JSON report conforming to formal spec."""
    trades_list = []
    for idx, t in enumerate(ledger.trades, start=1):
        trades_list.append({
            "trade_id": idx,
            "symbol": to_display_symbol(t.symbol),
            "side": t.side,
            "timestamp": t.timestamp_iso,
            "fill_price": float(t.executed_price),
            "quantity": float(t.quantity),
            "notional_value": float(t.quote_notional),
            "fee": float(t.fee),
            "realized_pnl": float(t.realized_pnl),
            "exit_reason": t.exit_reason,
            "cash_after": float(t.cash_after),
            "equity_after": float(t.equity_after),
        })

    report = {
        "status": "success",
        "config": {
            "pair": to_display_symbol(pair),
            "timeframe": timeframe,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "initial_balance": float(balance),
            "quote_currency": currency,
            "fee_rate": float(fee_rate),
            "slippage_rate": float(slippage),
            "strategy": strategy_name,
        },
        "summary": {
            "initial_balance": float(metrics.initial_balance),
            "final_balance": float(metrics.final_balance),
            "net_profit": float(metrics.net_profit),
            "total_return_pct": float(metrics.total_return_pct),
            "total_bars_processed": len(candles),
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate_pct": float(metrics.win_rate_pct),
            "profit_factor": float(metrics.profit_factor),
            "max_drawdown_pct": float(metrics.max_drawdown_pct),
            "sharpe_ratio": float(metrics.sharpe_ratio),
            "total_fees_paid": float(metrics.total_fees_paid),
            "progress_to_10k_pct": float(metrics.progress_to_10k_pct),
        },
        "trades": trades_list,
    }
    return json.dumps(report, indent=2)


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for historical backtesting."""
    parser = argparse.ArgumentParser(
        prog="trading_bot.backtest",
        description="Autonomous Compounding Trading Bot - Historical Backtester",
    )

    parser.add_argument(
        "--pair",
        type=str,
        default="BTC/EUR",
        help="Trading pair symbol (default: BTC/EUR)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2026-08-01",
        help="Start date in ISO format YYYY-MM-DD (default: 2026-08-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-09-01",
        help="End date in ISO format YYYY-MM-DD (default: 2026-09-01)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10.0,
        help="Starting capital in quote currency (default: 10.0 EUR)",
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="EUR",
        help="Quote currency (default: EUR)",
    )
    parser.add_argument(
        "--timeframe",
        "--interval",
        dest="timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help="Candle timeframe (default: 1h)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="amv_trendbreak",
        help="Strategy identifier: multi_regime, amv_trendbreak, grid (default: amv_trendbreak)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange rules: binance (5€ minNotional) or bybit (1€ minNotional) (default: binance)",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="Taker fee rate (default: 0.001 = 0.10%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Model slippage rate (default: 0.0005 = 0.05%%)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Asset lot step size (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format: table (formatted ASCII) or json (machine-readable) (default: table)",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Optional file path to write backtest report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as exc:
        return EXIT_CLI_USAGE_ERROR if exc.code != 0 else EXIT_SUCCESS

    if parsed_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # 1. Validate starting balance
    if parsed_args.balance <= 0.0:
        sys.stderr.write(f"Error: Initial balance must be > 0, got {parsed_args.balance}\n")
        return EXIT_INSUFFICIENT_FUNDS_ERROR

    # 2. Parse dates
    try:
        start_dt = parse_datetime(parsed_args.start)
        end_dt = parse_datetime(parsed_args.end)
    except ValueError as val_err:
        sys.stderr.write(f"Error: {val_err}\n")
        return EXIT_CLI_USAGE_ERROR

    if end_dt <= start_dt:
        sys.stderr.write(f"Error: End date ({end_dt}) must be strictly after start date ({start_dt})\n")
        return EXIT_CLI_USAGE_ERROR

    # 3. Validate exchange
    if parsed_args.exchange.lower() not in EXCHANGE_MIN_NOTIONAL:
        sys.stderr.write(f"Error: Unsupported exchange '{parsed_args.exchange}'. Choose from: {list(EXCHANGE_MIN_NOTIONAL.keys())}\n")
        return EXIT_CONFIG_ERROR

    # 4. Run historical backtest
    try:
        ledger, metrics, candles = run_backtest(
            pair=parsed_args.pair,
            start=start_dt,
            end=end_dt,
            balance=parsed_args.balance,
            currency=parsed_args.currency,
            timeframe=parsed_args.timeframe,
            strategy_name=parsed_args.strategy,
            exchange=parsed_args.exchange,
            fee_rate=parsed_args.fee,
            slippage=parsed_args.slippage,
            step_size=parsed_args.step_size,
        )
    except RuntimeError as r_err:
        sys.stderr.write(f"Data fetch error: {r_err}\n")
        return EXIT_DATA_FETCH_ERROR
    except Exception as exc:
        sys.stderr.write(f"Runtime execution error: {exc}\n")
        return EXIT_RUNTIME_ERROR

    # 5. Format output
    if parsed_args.output == "json":
        output_str = format_json_report(
            pair=parsed_args.pair,
            timeframe=parsed_args.timeframe,
            start_dt=start_dt,
            end_dt=end_dt,
            balance=parsed_args.balance,
            currency=parsed_args.currency,
            strategy_name=parsed_args.strategy,
            fee_rate=parsed_args.fee,
            slippage=parsed_args.slippage,
            ledger=ledger,
            metrics=metrics,
            candles=candles,
        )
    else:
        output_str = format_table_report(
            pair=parsed_args.pair,
            timeframe=parsed_args.timeframe,
            start_dt=start_dt,
            end_dt=end_dt,
            strategy_name=parsed_args.strategy,
            fee_rate=parsed_args.fee,
            slippage=parsed_args.slippage,
            ledger=ledger,
            metrics=metrics,
            candles=candles,
        )

    print(output_str)

    # 6. Save to report file if requested
    if parsed_args.report_file:
        try:
            report_path = Path(parsed_args.report_file)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(output_str, encoding="utf-8")
        except Exception as io_err:
            sys.stderr.write(f"Warning: Failed to write report file '{parsed_args.report_file}': {io_err}\n")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())

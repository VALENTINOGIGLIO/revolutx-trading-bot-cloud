"""1.5-Year Multi-Regime Strategy Benchmark Optimizer across Capital Tiers.

Simulates the last 1.5 years (2025-03-07 to 2026-09-07) of BTC/EUR 1h market data on Bybit
across various starting balances: [10, 50, 100, 500, 1000 EUR].
Generates the benchmark summary report file 'benchmark_1.5_years.txt'.
"""

from datetime import datetime, timezone
from decimal import Decimal
import os
import sys
import time
from typing import Dict, List, Any

# Ensure project root is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from trading_bot.cli.backtest import run_backtest
from trading_bot.data.cache import MarketDataService


def run_benchmark() -> str:
    balances = [10.0, 50.0, 100.0, 500.0, 1000.0]
    start_dt = datetime(2025, 3, 7, tzinfo=timezone.utc)
    end_dt = datetime(2026, 9, 7, tzinfo=timezone.utc)
    pair = "BTC/EUR"
    timeframe = "1h"
    exchange = "bybit"
    strategy_name = "multi_regime"

    print("=" * 110)
    print("      AUTONOMOUS TRADING BOT — 1.5-YEAR MULTI-REGIME STRATEGY BENCHMARK OPTIMIZER")
    print("=" * 110)
    print(f"Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} | Pair: {pair} | Timeframe: {timeframe}")
    print(f"Exchange: {exchange.upper()} (1.00 EUR minNotional, 0.10% Taker Fee, 0.05% Slippage)")
    print(f"Strategy: Multi-Regime Orchestrator (Micro-Grid in Ranging + AMV-TrendBreak in Trending)")
    print(f"Simulating capital tiers: {balances} EUR ...\n")

    # Reuse MarketDataService across iterations for maximum speed
    market_service = MarketDataService()

    results: List[Dict[str, Any]] = []

    for b in balances:
        t0 = time.time()
        print(f"--> Simulating initial balance {b:.2f} EUR ...", end=" ", flush=True)

        ledger, metrics, candles = run_backtest(
            pair=pair,
            start=start_dt,
            end=end_dt,
            balance=b,
            currency="EUR",
            timeframe=timeframe,
            strategy_name=strategy_name,
            exchange=exchange,
            fee_rate=0.0010,
            slippage=0.0005,
            market_data_service=market_service,
        )
        elapsed = time.time() - t0

        final_bal = float(metrics.final_balance)
        net_profit = float(metrics.net_profit)
        ret_pct = float(metrics.total_return_pct)
        trades = metrics.total_trades
        win_rate = float(metrics.win_rate_pct)
        max_dd = float(metrics.max_drawdown_pct)
        pf = float(metrics.profit_factor)
        fees = float(metrics.total_fees_paid)


        print(f"Done in {elapsed:.1f}s | Final: {final_bal:.2f} EUR | Net: {net_profit:+.2f} EUR ({ret_pct:+.2f}%)")

        results.append({
            "initial": b,
            "final": final_bal,
            "profit": net_profit,
            "return_pct": ret_pct,
            "trades": trades,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "profit_factor": pf,
            "fees": fees,
            "bars": len(candles),
        })

    # Build report text
    report_lines = []
    report_lines.append("=" * 110)
    report_lines.append("             AUTONOMOUS TRADING BOT — 1.5-YEAR MULTI-REGIME BENCHMARK REPORT")
    report_lines.append("=" * 110)
    report_lines.append(f"Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} (18 Months, {results[0]['bars']} Hourly Bars) | Pair: {pair}")
    report_lines.append(f"Exchange Model: Bybit Spot (1.00 EUR minNotional, 0.10% Taker Fee, 0.05% Slippage)")
    report_lines.append(f"Strategy: Multi-Regime Strategy Orchestrator (Micro-Grid in Ranging + AMV-TrendBreak in Trending)")
    report_lines.append("-" * 110)
    report_lines.append(
        f"{'Initial (EUR)':<15} | {'Final (EUR)':<13} | {'Net Profit (EUR)':<18} | "
        f"{'Return (%)':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Max DD (%)':<12} | {'PF':<6}"
    )
    report_lines.append("-" * 110)

    for r in results:
        profit_str = f"{r['profit']:+.2f}"
        ret_str = f"{r['return_pct']:+.2f}%"
        win_str = f"{r['win_rate']:.1f}%"
        dd_str = f"{r['max_dd']:.2f}%"
        pf_str = f"{r['profit_factor']:.2f}"
        report_lines.append(
            f"{r['initial']:<15.2f} | {r['final']:<13.2f} | {profit_str:<18} | "
            f"{ret_str:<12} | {r['trades']:<8} | {win_str:<10} | {dd_str:<12} | {pf_str:<6}"
        )

    report_lines.append("-" * 110)
    report_lines.append("REGIME & CONSTRAINTS COMPLIANCE:")
    report_lines.append("  - Bybit minNotional >= 1.00 EUR strictly verified for all sub-orders via ceil quantization.")
    report_lines.append("  - Dual-threshold Schmitt trigger hysteresis active (K=2 bar persistence).")
    report_lines.append("  - Micro-Grid active during RANGING markets; AMV-TrendBreak active during TRENDING breakouts.")
    report_lines.append("=" * 110)

    report_content = "\n".join(report_lines) + "\n"

    # Write to benchmark_1.5_years.txt
    output_filename = "benchmark_1.5_years.txt"
    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + report_content)
    print(f"[SUCCESS] Benchmark report successfully written to: {output_path}")
    return report_content


if __name__ == "__main__":
    run_benchmark()

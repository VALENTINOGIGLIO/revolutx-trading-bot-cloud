import subprocess
import re
import os

strategies = ["amv_trendbreak", "micro_grid", "multi_regime"]
pairs = ["BTC/EUR", "SOL/EUR", "DOGE/EUR"]
timeframes = ["4h"]
start_date = "2024-09-07"
end_date = "2026-09-07"
balance = 10.0

print(f"=========================================================================", flush=True)
print(f"                   QUANTITATIVE MATRIX BENCHMARK (2 YEARS)               ", flush=True)
print(f"=========================================================================", flush=True)
print(f"{'Strategy':<20} | {'Pair':<10} | {'TF':<3} | {'Net Profit':<20} | {'Max DD'}", flush=True)
print(f"-------------------------------------------------------------------------", flush=True)

env = os.environ.copy()
env["PYTHONPATH"] = "src"

for p in pairs:
    for s in strategies:
        for tf in timeframes:
            cmd = [
                "./.venv/bin/python3", "-m", "trading_bot.cli.backtest",
                "--start", start_date,
                "--end", end_date,
                "--pair", p,
                "--balance", str(balance),
                "--timeframe", tf,
                "--strategy", s
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            output = result.stdout
            
            net_profit_match = re.search(r"Net Profit:\s+([-\+\d.]+\sEUR\s\([-\+\d.]+\%\))", output)
            max_dd_match = re.search(r"Max Drawdown:\s+([-\d.]+\%)", output)
            
            net_profit = net_profit_match.group(1) if net_profit_match else "ERRORE"
            max_dd = max_dd_match.group(1) if max_dd_match else "ERRORE"
            
            print(f"{s:<20} | {p:<10} | {tf:<3} | {net_profit:<20} | {max_dd}", flush=True)

print(f"=========================================================================", flush=True)

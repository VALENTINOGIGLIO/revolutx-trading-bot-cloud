import subprocess
import json
import sys
from datetime import datetime, timedelta

start_date = "2021-09-07"
end_date = "2026-09-07"
pair = "DOGE/EUR"
timeframe = "4h"
balance = 50.0
strategy = "amv_trendbreak"

print(f"Running 5-year backtest for {pair} on {timeframe}...", flush=True)

cmd = [
    "./.venv/bin/python3", "-m", "trading_bot.cli.backtest",
    "--start", start_date,
    "--end", end_date,
    "--pair", pair,
    "--balance", str(balance),
    "--timeframe", timeframe,
    "--strategy", strategy,
    "--exchange", "bybit",
    "--output", "json"
]

import os
env = os.environ.copy()
env["PYTHONPATH"] = "src"

result = subprocess.run(cmd, capture_output=True, text=True, env=env)

if result.returncode != 0:
    print("Error running backtest:")
    print(result.stderr)
    sys.exit(1)

# The output might have some warnings before the JSON. Find the start of the JSON block.
output = result.stdout
try:
    json_start = output.find('{')
    if json_start == -1:
        raise ValueError("No JSON found")
    data = json.loads(output[json_start:])
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Raw output:", output[:500])
    sys.exit(1)

trades = data.get("trades", [])
initial_balance = balance
final_balance = trades[-1]["equity_after"] if trades else balance

print(f"\nBacktest completed! Found {len(trades)} trades.")
print(f"Final Balance: {final_balance} EUR\n")

# Calculate 6-month milestones
start_dt = datetime.fromisoformat(start_date)
current_milestone = start_dt + timedelta(days=182) # approx 6 months
milestone_idx = 1
current_equity = initial_balance

print(f"[{start_date}] INIZIO - Capitale: {initial_balance:.2f} EUR")

for trade in trades:
    # trade format usually has 'timestamp' inside it
    trade_time = datetime.fromisoformat(trade["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
    
    while trade_time > current_milestone:
        print(f"[{current_milestone.strftime('%Y-%m-%d')}] Mese {milestone_idx * 6} - Capitale: {current_equity:.2f} EUR")
        current_milestone += timedelta(days=182)
        milestone_idx += 1
        
    current_equity = trade["equity_after"]

# Print any remaining milestones up to end_date
while current_milestone < datetime.fromisoformat(end_date):
    print(f"[{current_milestone.strftime('%Y-%m-%d')}] Mese {milestone_idx * 6} - Capitale: {current_equity:.2f} EUR")
    current_milestone += timedelta(days=182)
    milestone_idx += 1

print(f"[{end_date}] FINE (5 Anni) - Capitale: {final_balance:.2f} EUR")

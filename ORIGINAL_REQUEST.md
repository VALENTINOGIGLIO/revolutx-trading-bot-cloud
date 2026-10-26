# Original User Request

## Initial Request — 2026-09-07T13:22:26Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: very large team of agents

Use a very large team of agents. An automated trading bot (accessible via terminal) that starts with a budget of 10€ and aims to autonomously scale the capital towards 10,000€. The bot will initially run in a paper-trading/simulator environment to validate its strategy, and will send notifications about its operations. The agent team will select the most suitable platform/exchange.

Working directory: ~/teamwork_projects/trading_bot
Integrity mode: development

## Requirements

### R1. Aggressive Growth Strategy and Platform Selection
The bot must connect to an exchange chosen by the team (suitable for small initial capital) and autonomously execute a highly aggressive trading strategy designed to maximize profits and speed of growth.

### R2. Paper Trading and Rigorous Historical Backtesting
The system must include a robust paper-trading mode that tracks a virtual starting balance of 10€. It must heavily backtest the strategy over historical financial data to verify it makes correct trading decisions over at least a 1-month period.

### R3. Notification System
The bot must include a notification mechanism (e.g., via Telegram or standard output/logs if external APIs require user setup) to report executed trades and portfolio updates.

## Acceptance Criteria

### Execution and Simulation
- [ ] A programmatic backtesting script exists and can be executed from the terminal.
- [ ] The backtest runs successfully over at least 1 month of historical data, starting with 10€, and outputs a final simulated balance and a list of executed trades without crashing, proving the strategy's historical performance.

### Notifications
- [ ] A unit test exists that triggers a simulated trade and successfully intercepts/verifies that the notification payload (containing trade details) was constructed and dispatched.

## Follow-up — 2026-09-07T14:59:39Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: [none — teamwork routes from the description]

A real-time local web dashboard that connects to the existing trading bot. It must display a live price chart overlaid with markers indicating exactly where the bot enters and exits trades, updating in real-time. Additionally, the bot must log all trade execution details into a CSV file.

Working directory: ~/teamwork_projects/trading_bot
Integrity mode: development

## Requirements

### R1. CSV Trade Logging
The existing trading bot must be updated to append every executed trade (timestamp, action, entry/exit price, size, profit/loss) into a persistently stored CSV file.

### R2. Real-Time Web Dashboard
A local web server must be built to serve a live-updating price chart (using open-source charting libraries like Lightweight Charts or Plotly). The chart must automatically fetch and display real-time market data alongside the bot's activities.

### R3. Trade Visualization
The web dashboard must read the CSV trade log (or connect directly to the bot's data stream) and accurately plot entry and exit markers directly on the live price chart at the exact timestamps and prices where the trades occurred.

## Acceptance Criteria

### CSV Logging Validation
- [ ] A programmatic test verifies that when the bot executes a mock trade, a correctly formatted row is immediately appended to the target CSV file.

### Dashboard and Visualization
- [ ] An automated integration test (e.g., using a headless browser or API request test) verifies that the local web server starts successfully, serves the dashboard, and exposes an endpoint that correctly returns the trade history formatted for the charting library.

## Follow-up — 2026-09-07T19:50:52Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: [none — teamwork routes from the description]

Upgrade the existing trading bot (`~/teamwork_projects/trading_bot`) to implement a Multi-Regime hybrid strategy. The bot must automatically switch between a Micro-Grid Trading strategy (for sideways/choppy markets) and the existing AMV-TrendBreak strategy (for trending markets) using a market regime filter. The bot will explicitly target the Bybit exchange to utilize its 1.00 EUR minNotional limit, allowing it to deploy a full grid even with a 10 EUR total budget.

Working directory: ~/teamwork_projects/trading_bot
Integrity mode: development

## Requirements

### R1. Regime Filter implementation
Implement an overarching Market Regime Filter (e.g., using ADX, Bollinger Band Width, or Choppiness Index) to mathematically detect if the market is currently "Trending" or "Ranging/Sideways".

### R2. Micro-Grid Trading Strategy (Bybit Optimized)
Implement a Grid Trading strategy that operates when the regime is "Ranging". It must distribute the capital across multiple buy/sell levels. It must be strictly optimized for Bybit's 1.00 EUR `minNotional` limit, ensuring a 10 EUR budget can open up to 10 micro-grid levels.

### R3. Strategy Orchestrator (Multi-Regime)
Update the core engine to dynamically switch strategies. When the regime is Ranging, deploy the Grid strategy to accumulate small profits. When a macro-trend or breakout is detected, safely dismantle the Grid (cancel open grid orders) and deploy the capital into the aggressive AMV-TrendBreak strategy.

### R4. Benchmark Optimization Report
After implementing the strategy, the team must run a benchmarking script (`optimizer.py` or similar) that simulates the last 1.5 years of market data using various starting balances (10, 50, 100, 500, 1000 EUR). The results must be logged to a final summary file.

## Acceptance Criteria

### Regime Switching Validation
- [ ] A programmatic backtest verifies that the bot correctly transitions between the Grid and AMV strategies at least once based on historical market volatility, and logs the regime change.

### Grid Constraints Validation
- [ ] A unit test verifies that the Grid Strategy accurately generates sub-orders sized at exactly or slightly above 1.00 EUR (Bybit minimum constraint) when given a mock 10.00 EUR portfolio balance.

### Benchmark Delivery
- [ ] A file named `benchmark_1.5_years.txt` is successfully generated, containing a table of the final balances and net profits for the multi-regime strategy over the last 1.5 years across various starting capitals.



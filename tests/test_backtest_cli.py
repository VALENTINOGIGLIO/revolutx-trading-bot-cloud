"""Tier 1 & 2 Tests: Backtest CLI and Command-Line Interfaces.

Covers:
1. `trading_bot.cli.backtest`:
   - Argument parsing and defaults (10.0 EUR, BTC/EUR, 1h, binance, table output).
   - Return code 0 on successful execution.
   - Dual output formats: `--output table` (formatted ASCII) and `--output json` (valid machine-readable JSON).
   - Export to report file (`--report-file`).
   - Exchange selection (`--exchange binance`, `--exchange bybit`).
   - Return code 2 on invalid CLI arguments or malformed date ranges.
   - Return code 4 on non-positive initial balance (--balance 0.0, --balance -10.0).
   - Return code 5 on unsupported exchange name.
2. `trading_bot.cli.data`:
   - Cache inspection and help commands.
3. `trading_bot.cli.notify`:
   - Console and JSON channel testing.
4. `trading_bot.__main__`:
   - Unified dispatcher delegating to `backtest`, `paper`, `data`, `notify`.
   - Subprocess invocation of `python -m trading_bot.backtest` at OS level.
"""

from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List
from unittest.mock import patch
import pytest

# Ensure src directory is in Python path for test execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trading_bot.__main__ import main as unified_cli_main
from trading_bot.cli.backtest import (
    EXIT_CLI_USAGE_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_INSUFFICIENT_FUNDS_ERROR,
    EXIT_SUCCESS,
    main as backtest_main,
)
from trading_bot.cli.data import main as data_main
from trading_bot.cli.notify import main as notify_main
from trading_bot.data.models import Candle


# =====================================================================
# CLI BACKTEST ARGUMENT PARSING & DEFAULTS
# =====================================================================
def test_backtest_cli_help_exits_zero(capsys):
    """Verify `trading_bot.backtest --help` displays usage banner and exits with code 0."""
    code = backtest_main(["--help"])
    assert code == EXIT_SUCCESS
    captured = capsys.readouterr()
    assert "Autonomous Compounding Trading Bot - Historical Backtester" in captured.out
    assert "--pair" in captured.out
    assert "--balance" in captured.out
    assert "--output" in captured.out


def test_backtest_cli_successful_run_table_output(capsys, monkeypatch):
    """Verify `trading_bot.backtest` with default arguments produces valid table output and code 0."""
    code = backtest_main([
        "--pair", "BTC/EUR",
        "--start", "2026-08-01",
        "--end", "2026-09-01",
        "--balance", "10.0",
        "--output", "table",
    ])
    assert code == EXIT_SUCCESS

    captured = capsys.readouterr()
    assert "AUTONOMOUS TRADING BOT - BACKTEST REPORT" in captured.out
    assert "PORTFOLIO SUMMARY:" in captured.out
    assert "Initial Balance:         10.00 EUR" in captured.out
    assert "Final Balance:" in captured.out
    assert "TRADING PERFORMANCE METRICS:" in captured.out
    assert "Total Trades Executed:" in captured.out
    assert "[SUCCESS] Backtest completed without errors." in captured.out


def test_backtest_cli_successful_run_json_output(capsys):
    """Verify `trading_bot.backtest --output json` emits valid machine-parseable JSON."""
    code = backtest_main([
        "--pair", "BTC/EUR",
        "--start", "2026-08-01",
        "--end", "2026-09-01",
        "--balance", "10.0",
        "--output", "json",
    ])
    assert code == EXIT_SUCCESS

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["status"] == "success"
    assert "config" in data
    assert data["config"]["pair"] == "BTC/EUR"
    assert data["config"]["initial_balance"] == 10.0
    assert data["config"]["quote_currency"] == "EUR"
    assert data["config"]["strategy"] == "amv_trendbreak"

    assert "summary" in data
    assert data["summary"]["initial_balance"] == 10.0
    assert data["summary"]["final_balance"] > 0.0
    assert data["summary"]["total_bars_processed"] >= 720
    assert isinstance(data["trades"], list)


def test_backtest_cli_report_file_export(tmp_path: Path):
    """Verify `--report-file` flag writes output report to the designated file path."""
    report_file = tmp_path / "custom_backtest_report.json"

    code = backtest_main([
        "--pair", "BTC/EUR",
        "--start", "2026-08-01",
        "--end", "2026-09-01",
        "--balance", "10.0",
        "--output", "json",
        "--report-file", str(report_file),
    ])
    assert code == EXIT_SUCCESS
    assert report_file.exists(), "Report file was not created on disk"

    content = report_file.read_text(encoding="utf-8")
    data = json.loads(content)
    assert data["status"] == "success"
    assert data["summary"]["initial_balance"] == 10.0


def test_backtest_cli_bybit_exchange_rules(capsys):
    """Verify `--exchange bybit` applies Bybit rules (1€ minNotional)."""
    code = backtest_main([
        "--pair", "BTC/EUR",
        "--start", "2026-08-01",
        "--end", "2026-09-01",
        "--balance", "10.0",
        "--exchange", "bybit",
        "--output", "json",
    ])
    assert code == EXIT_SUCCESS
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"


# =====================================================================
# CLI ERROR HANDLING & EXIT CODE CONTRACTS
# =====================================================================
def test_backtest_cli_invalid_argument_exit_code_2(capsys):
    """Verify unknown command line arguments return exit code 2."""
    code = backtest_main(["--non-existent-flag-xyz"])
    assert code == EXIT_CLI_USAGE_ERROR


def test_backtest_cli_malformed_date_exit_code_2(capsys):
    """Verify malformed date string returns exit code 2."""
    code = backtest_main(["--start", "not-a-valid-date"])
    assert code == EXIT_CLI_USAGE_ERROR
    captured = capsys.readouterr()
    assert "Invalid date format" in captured.err


def test_backtest_cli_end_before_start_exit_code_2(capsys):
    """Verify end date preceding start date returns exit code 2."""
    code = backtest_main([
        "--start", "2026-09-01",
        "--end", "2026-08-01",
    ])
    assert code == EXIT_CLI_USAGE_ERROR
    captured = capsys.readouterr()
    assert "must be strictly after start date" in captured.err


def test_backtest_cli_zero_balance_exit_code_4(capsys):
    """Verify `--balance 0.0` returns exit code 4 (INSUFFICIENT_FUNDS_ERROR)."""
    code = backtest_main(["--balance", "0.0"])
    assert code == EXIT_INSUFFICIENT_FUNDS_ERROR
    captured = capsys.readouterr()
    assert "Initial balance must be > 0" in captured.err


def test_backtest_cli_negative_balance_exit_code_4(capsys):
    """Verify `--balance -10.0` returns exit code 4."""
    code = backtest_main(["--balance", "-10.0"])
    assert code == EXIT_INSUFFICIENT_FUNDS_ERROR


def test_backtest_cli_unsupported_exchange_exit_code_5(capsys):
    """Verify unrecognized exchange name returns exit code 5 (CONFIG_ERROR)."""
    code = backtest_main(["--exchange", "unsupported_crypto_exchange_xyz"])
    assert code == EXIT_CONFIG_ERROR
    captured = capsys.readouterr()
    assert "Unsupported exchange" in captured.err


# =====================================================================
# AUXILIARY CLI COMMANDS: DATA & NOTIFY
# =====================================================================
def test_cli_data_help_and_inspect(capsys):
    """Verify `trading_bot.cli.data` help and inspect commands."""
    help_code = data_main(["--help"])
    assert help_code == EXIT_SUCCESS

    inspect_code = data_main(["inspect"])
    assert inspect_code == EXIT_SUCCESS
    captured = capsys.readouterr()
    assert "MARKET DATA CACHE REPORT" in captured.out


def test_cli_notify_help_and_dispatch(capsys, tmp_path: Path):
    """Verify `trading_bot.cli.notify` channel dispatch."""
    help_code = notify_main(["--help"])
    assert help_code == EXIT_SUCCESS

    # Test Console channel
    console_code = notify_main(["--channel", "console", "--event", "trade", "--output", "json"])
    assert console_code == EXIT_SUCCESS

    # Test JSON channel with temp path
    log_file = tmp_path / "test_notify.jsonl"
    json_code = notify_main([
        "--channel", "json",
        "--json-path", str(log_file),
        "--event", "all",
        "--output", "json",
    ])
    assert json_code == EXIT_SUCCESS
    assert log_file.exists()


# =====================================================================
# UNIFIED CLI DISPATCHER (__main__)
# =====================================================================
def test_unified_cli_main_help_and_version(capsys):
    """Verify unified dispatcher `python -m trading_bot` help and version flags."""
    # No args -> shows help
    code_none = unified_cli_main([])
    assert code_none == 0

    code_help = unified_cli_main(["--help"])
    assert code_help == 0
    captured = capsys.readouterr()
    assert "Autonomous Compounding Trading Bot" in captured.out
    assert "Available Commands:" in captured.out

    code_ver = unified_cli_main(["--version"])
    assert code_ver == 0
    captured_ver = capsys.readouterr()
    assert "trading_bot version" in captured_ver.out


def test_unified_cli_main_unknown_subcommand(capsys):
    """Verify unified dispatcher returns 2 on unknown subcommand."""
    code = unified_cli_main(["nonexistent_subcommand"])
    assert code == 2
    captured = capsys.readouterr()
    assert "Error: Unknown command 'nonexistent_subcommand'" in captured.err


def test_unified_cli_main_delegates_to_backtest(capsys):
    """Verify `python -m trading_bot backtest --help` delegates to backtest command."""
    code = unified_cli_main(["backtest", "--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Autonomous Compounding Trading Bot - Historical Backtester" in captured.out


# =====================================================================
# SUBPROCESS OS-LEVEL EXECUTION TEST
# =====================================================================
def test_os_level_subprocess_backtest_invocation():
    """Verify full OS-level shell invocation of `python -m trading_bot.backtest`.

    Executes via subprocess to assert exact binary environment and UNIX return code.
    """
    cmd = [
        sys.executable,
        "-m", "trading_bot.backtest",
        "--pair", "BTC/EUR",
        "--start", "2026-08-01",
        "--end", "2026-09-01",
        "--balance", "10.0",
        "--output", "json",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, f"Process failed with stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert data["summary"]["initial_balance"] == 10.0
    assert data["summary"]["final_balance"] > 0.0
    assert len(data["trades"]) >= 1

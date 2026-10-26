"""Empirical Fuzzing and Boundary Stress Test Suite for Milestone 10 Iteration 2.

Author: Challenger 1 (challenger_m10_fix_1)
Role: Empirical Challenger (critic, specialist)
Focus: Corrupt Input Stress, Boundary Fuzzing, Extreme String/Float Handling, Server Integration.

Test Coverage:
1. Missing columns and empty comma patterns (e.g., '50000,,,BUY,,,,').
2. Extreme floats: 1e308, 1e-308, overflow 1e309 (inf), NaN, nan, Infinity, -inf.
3. Mixed Unicode, SQL injection strings, emojis, binary/null byte sequences in string fields.
4. Non-ASCII and invalid dates/timestamps (year 99999, negative, gibberish, ISO-violating).
5. Interleaved valid and malformed rows: ensure valid rows load with exact precision while corrupt are skipped.
6. Binary garbage file handling: non-UTF8/corrupted bytes do not crash the parser.
7. Live DashboardServer integration: server boots, serves valid JSON on all endpoints without 500 or hangs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List

import httpx
import pytest

from trading_bot.dashboard.server import DashboardServer
from trading_bot.dashboard.state import DashboardState
from trading_bot.notifications.channels.csv_log import CsvTradeLogger


def test_fuzz_missing_columns_and_empty_commas(tmp_path: Path):
    """Verify load_from_csv handles rows with missing columns, empty commas, and extra commas."""
    csv_file = tmp_path / "missing_columns.csv"
    csv_content = (
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        "2026-09-07T12:00:00+00:00,v1,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        ",,,,,,,,,,,,\n"  # All commas empty
        "50000,,,BUY,,,,\n"  # Missing columns / misaligned commas
        ",,,\n"  # Truncated comma line
        "2026-09-07T12:05:00+00:00,v2,BTC/EUR,SELL,55000.00,0.001,55.0,0.05,5.00,10.00,105.0,105.0,TP\n"
        ",,,,,,,,,,,,,\n"  # Extra trailing commas
        ",,,,,,,,,,,,extra1,extra2\n"  # Excess columns beyond header
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    state = DashboardState()
    loaded_count = state.load_from_csv(csv_file)

    # Valid rows v1 and v2 must be loaded; empty/misaligned rows should be handled without raising exceptions
    assert loaded_count >= 2
    trades = state.get_trades()
    trade_ids = [t["trade_id"] for t in trades]
    assert "v1" in trade_ids
    assert "v2" in trade_ids

    # Markers check
    markers = state.get_markers()
    assert len(markers) >= 2


def test_fuzz_extreme_floats_nan_inf_boundaries(tmp_path: Path):
    """Verify rejection of NaN, Infs, and handling of extreme finite floats (1e308, 1e-308)."""
    csv_file = tmp_path / "extreme_floats.csv"
    csv_content = (
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        # 1. Valid normal row
        "2026-09-07T12:00:00+00:00,norm1,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 2. NaN in price
        "2026-09-07T12:01:00+00:00,nan_price,BTC/EUR,BUY,NaN,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 3. lowercase nan in size
        "2026-09-07T12:02:00+00:00,nan_size,BTC/EUR,BUY,50000.00,nan,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 4. Infinity in price
        "2026-09-07T12:03:00+00:00,inf_price,BTC/EUR,BUY,Infinity,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 5. -inf in pnl
        "2026-09-07T12:04:00+00:00,ninf_pnl,BTC/EUR,SELL,50000.00,0.001,50.0,0.05,-inf,-100.0,100.0,50.0,\n"
        # 6. Float overflow (1e309 overflows to inf)
        "2026-09-07T12:05:00+00:00,over_price,BTC/EUR,BUY,1e309,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 7. Extremely large finite float (1e308)
        "2026-09-07T12:06:00+00:00,huge_float,BTC/EUR,BUY,1e308,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 8. Subnormal small float (1e-308)
        "2026-09-07T12:07:00+00:00,subnorm_float,BTC/EUR,BUY,50000.00,1e-308,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 9. NaN in portfolio_balance (should not set equity to NaN)
        "2026-09-07T12:08:00+00:00,nan_bal,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,NaN,NaN,\n"
        # 10. Valid closing row
        "2026-09-07T12:09:00+00:00,norm2,BTC/EUR,SELL,55000.00,0.001,55.0,0.05,5.00,10.00,105.0,105.0,TP\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    state = DashboardState()
    count = state.load_from_csv(csv_file)

    trades = state.get_trades()
    trade_ids = [t["trade_id"] for t in trades]

    # NaN / Inf rows MUST be rejected
    assert "nan_price" not in trade_ids
    assert "nan_size" not in trade_ids
    assert "inf_price" not in trade_ids
    assert "ninf_pnl" not in trade_ids
    assert "over_price" not in trade_ids

    # Normal rows must be present
    assert "norm1" in trade_ids
    assert "norm2" in trade_ids

    # Huge float (1e308) and subnormal float (1e-308) are valid finite IEEE-754 floats
    assert "huge_float" in trade_ids
    assert "subnorm_float" in trade_ids

    # Verify no NaN in state balances
    assert not math.isnan(state.equity)
    assert not math.isinf(state.equity)
    assert not math.isnan(state.cash_balance)
    assert not math.isinf(state.cash_balance)

    # Ensure get_status produces valid JSON-serializable output
    status = state.get_status()
    json_str = json.dumps(status)
    assert "portfolio_balance" in json_str


def test_fuzz_unicode_sqli_emojis_binary_strings(tmp_path: Path):
    """Verify resilience against SQL injection, HTML/XSS, emojis, and non-ASCII text in string fields."""
    csv_file = tmp_path / "strings_fuzz.csv"
    csv_content = (
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        "2026-09-07T12:00:00+00:00,id_sql_'; DROP TABLE trades; --,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:01:00+00:00,<script>alert('xss')</script>,BTC/EUR,BUY,51000.00,0.001,51.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:02:00+00:00,emoji_🚀🔥,BTC/EUR,BUY,52000.00,0.001,52.0,0.05,0.00,0.00,100.0,50.0,🚀💎🙌\n"
        "2026-09-07T12:03:00+00:00,id_cjk_日本語,BTC/EUR,SELL,53000.00,0.001,53.0,0.05,1.00,2.00,101.0,101.0,利確\n"
        "2026-09-07T12:04:00+00:00,id_rtl_عربي,BTC/EUR,BUY,54000.00,0.001,54.0,0.05,0.00,0.00,100.0,50.0,\n"
        '2026-09-07T12:05:00+00:00,"escaped_""quotes""",BTC/EUR,SELL,55000.00,0.001,55.0,0.05,2.00,4.00,103.0,103.0,"reason with , comma"\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    state = DashboardState()
    count = state.load_from_csv(csv_file)
    assert count == 6

    markers = state.get_markers()
    assert len(markers) == 6

    # Verify JSON serialization works without issues
    json_markers = json.dumps(markers, ensure_ascii=False)
    assert "DROP TABLE" in json_markers
    assert "alert" in json_markers
    assert "🚀" in json_markers
    assert "日本語" in json_markers

    # Also test default json.dumps and roundtrip
    roundtrip = json.loads(json.dumps(markers))
    assert "DROP TABLE" in roundtrip[0]["id"]
    assert "alert" in roundtrip[1]["id"]
    assert "🚀" in roundtrip[2]["id"]
    assert "日本語" in roundtrip[3]["id"]

    trades = state.get_trades()
    json_trades = json.dumps(trades, ensure_ascii=False)
    assert "DROP TABLE" in json_trades
    assert "alert" in json_trades
    assert "🚀" in json_trades
    assert "日本語" in json_trades


def test_fuzz_invalid_and_out_of_range_timestamps(tmp_path: Path):
    """Verify timestamp parsing handles corrupted strings, overflow timestamps, and negative timestamps."""
    csv_file = tmp_path / "invalid_timestamps.csv"
    csv_content = (
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        # 1. Normal ISO timestamp
        "2026-09-07T12:00:00+00:00,t_norm,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 2. Text gibberish in timestamp
        "NOT_A_TIMESTAMP,t_gibberish,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 3. Millisecond Unix timestamp
        "1788806400000,t_epoch_ms,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 4. Beyond year 9999 / platform time_t overflow (e.g. 1e18)
        "999999999999999999,t_overflow_ts,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 5. Extremely negative timestamp (-1e18)
        "-999999999999999999,t_underflow_ts,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 6. Malformed ISO format (month 99)
        "2026-99-99T99:99:99Z,t_invalid_iso,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        # 7. Empty string in timestamp
        ",t_empty_ts,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    state = DashboardState()
    count = state.load_from_csv(csv_file)

    # Must not raise an unhandled exception
    assert count >= 1
    trades = state.get_trades()
    trade_ids = [t["trade_id"] for t in trades]
    assert "t_norm" in trade_ids

    # For validly loaded rows, timestamp must be an ISO formatted string
    for t in trades:
        assert isinstance(t["time"], int)
        assert isinstance(t["timestamp"], str)


def test_fuzz_binary_garbage_and_corrupt_bytes(tmp_path: Path):
    """Verify load_from_csv handles binary garbage and non-UTF8 bytes gracefully."""
    bin_file = tmp_path / "binary_garbage.csv"
    # Write random binary bytes with null bytes and invalid UTF-8 sequences
    raw_bytes = b"\x00\xff\xfe\x80\x81\x82\x00\x01\x02CORRUPT_BYTES\x00\x00" * 20
    bin_file.write_bytes(raw_bytes)

    state = DashboardState()
    # Should catch encoding / decoding error gracefully and return 0
    count = state.load_from_csv(bin_file)
    assert count == 0
    assert len(state.get_trades()) == 0


def test_fuzz_interleaved_valid_and_malformed_rows(tmp_path: Path):
    """Verify high-volume stress testing with interleaved malformed and valid rows."""
    csv_file = tmp_path / "interleaved_fuzz.csv"
    lines = [
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason"
    ]

    expected_valid_ids = []
    for i in range(100):
        if i % 3 == 0:
            # Valid row
            tid = f"valid_{i}"
            expected_valid_ids.append(tid)
            action = "BUY" if i % 2 == 0 else "SELL"
            pnl = 2.50 if action == "SELL" else 0.0
            lines.append(
                f"2026-09-07T12:{i:02d}:00+00:00,{tid},BTC/EUR,{action},50000.00,0.001,50.0,0.05,{pnl},5.0,100.0,50.0,"
            )
        elif i % 3 == 1:
            # Corrupted numeric field
            lines.append(
                f"2026-09-07T12:{i:02d}:00+00:00,corrupt_{i},BTC/EUR,BUY,NOT_A_PRICE,0.001,50.0,0.05,0.0,0.0,100.0,50.0,"
            )
        else:
            # NaN or Inf field
            lines.append(
                f"2026-09-07T12:{i:02d}:00+00:00,nan_{i},BTC/EUR,BUY,NaN,0.001,50.0,0.05,0.0,0.0,100.0,50.0,"
            )

    csv_file.write_text("\n".join(lines), encoding="utf-8")

    state = DashboardState()
    count = state.load_from_csv(csv_file)

    assert count == len(expected_valid_ids)
    trades = state.get_trades()
    loaded_ids = [t["trade_id"] for t in trades]
    assert loaded_ids == expected_valid_ids


def test_fuzzed_csv_live_dashboard_server_background_thread(tmp_path: Path):
    """Verify DashboardServer boots successfully with a heavily fuzzed CSV file,

    serves valid HTTP responses on all endpoints without 500 errors or hanging,
    and terminates cleanly.
    """
    csv_file = tmp_path / "server_fuzz.csv"
    csv_content = (
        "timestamp,trade_id,symbol,action,price,size,notional,fee,pnl,pnl_pct,portfolio_balance,cash_balance,exit_reason\n"
        "2026-09-07T12:00:00+00:00,t_valid_1,BTC/EUR,BUY,50000.00,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:01:00+00:00,t_nan,BTC/EUR,BUY,NaN,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "50000,,,BUY,,,,\n"
        "2026-09-07T12:02:00+00:00,t_inf,BTC/EUR,SELL,Infinity,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:03:00+00:00,t_valid_2,BTC/EUR,SELL,55000.00,0.001,55.0,0.05,5.00,10.00,105.0,105.0,TP\n"
        "2026-09-07T12:04:00+00:00,t_corrupt_price,BTC/EUR,BUY,BAD_VAL,0.001,50.0,0.05,0.00,0.00,100.0,50.0,\n"
        "2026-09-07T12:05:00+00:00,t_emoji_🚀,BTC/EUR,BUY,52000.00,0.001,52.0,0.05,0.00,0.00,100.0,50.0,🚀💎\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    # Initialize DashboardServer with fuzzed CSV on ephemeral port
    server = DashboardServer(host="127.0.0.1", port=0, trades_csv=csv_file)
    actual_port = server.start_background(timeout=5.0)

    try:
        assert actual_port > 0
        base_url = f"http://127.0.0.1:{actual_port}"

        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            # 1. Root SPA HTML
            resp_root = client.get("/")
            assert resp_root.status_code == 200
            assert "Lightweight Charts" in resp_root.text or "<html" in resp_root.text.lower()

            # 2. Status endpoint
            resp_status = client.get("/api/status")
            assert resp_status.status_code == 200
            status_data = resp_status.json()
            assert "portfolio_balance" in status_data
            assert "total_trades" in status_data
            # Total trades loaded: includes valid trades + partial fallback row
            assert status_data["total_trades"] >= 3

            # 3. Trades endpoint
            resp_trades = client.get("/api/trades")
            assert resp_trades.status_code == 200
            markers = resp_trades.json()
            assert len(markers) >= 3
            marker_ids = [m["id"] for m in markers]
            assert "t_valid_1" in marker_ids
            assert "t_valid_2" in marker_ids
            assert "t_emoji_🚀" in marker_ids

            # 4. Market data endpoint
            resp_market = client.get("/api/market-data")
            assert resp_market.status_code == 200
            assert isinstance(resp_market.json(), list)

    finally:
        server.stop(timeout=2.0)
        assert not server._thread.is_alive()

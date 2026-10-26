#!/usr/bin/env python3
"""Local Failover Watchdog for Revolut X Cloud 24/7 Trading Bot.

Runs every 3 minutes via macOS launchd:
1. Checks if the GitHub Actions 24/7 Cloud Worker is active (in_progress / queued).
2. If active: exits immediately without polling Telegram getUpdates (zero conflict).
3. If not active: triggers a new Cloud Worker on GitHub Actions AND executes a
   single local failover cycle (--iterations 1 --live) to answer any pending
   Telegram messages immediately while the cloud container boots up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO = "VALENTINOGIGLIO/revolutx-trading-bot-cloud"
PYTHON_BIN = "/opt/homebrew/bin/python3"


def get_env_with_path() -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    current_path = env.get("PATH", "")
    env["PATH"] = ":".join(extra_paths + ([current_path] if current_path else []))
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return env


def check_cloud_worker_active(env: dict[str, str]) -> tuple[bool, str]:
    try:
        res = subprocess.run(
            [
                "gh",
                "run",
                "list",
                "--repo",
                REPO,
                "--limit",
                "5",
                "--json",
                "databaseId,status,conclusion,createdAt",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        if res.returncode != 0:
            return False, f"gh run list error: {res.stderr.strip()}"
        runs = json.loads(res.stdout or "[]")
        for r in runs:
            status = (r.get("status") or "").lower()
            if status in ("in_progress", "queued", "waiting", "requested", "pending"):
                return True, f"run_id={r.get('databaseId')} status={status}"
        return False, "no active runs"
    except Exception as ex:
        return False, f"exception: {ex}"


def trigger_cloud_worker(env: dict[str, str]) -> bool:
    try:
        res = subprocess.run(
            [
                "gh",
                "workflow",
                "run",
                "cloud_bot.yml",
                "--repo",
                REPO,
                "--ref",
                "main",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        return res.returncode == 0
    except Exception:
        return False


def run_local_failover_cycle(env: dict[str, str]) -> None:
    try:
        subprocess.run(
            [
                PYTHON_BIN,
                "-m",
                "trading_bot.cli.paper",
                "--pair",
                "SOL/EUR,ETH/EUR,BTC/EUR",
                "--balance",
                "150.0",
                "--timeframe",
                "4h",
                "--exchange",
                "revolutx",
                "--iterations",
                "1",
                "--live",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=60,
        )
    except Exception as ex:
        print(f"[WATCHDOG] Local failover cycle error: {ex}", flush=True)


def main() -> int:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    env = get_env_with_path()

    is_active, info = check_cloud_worker_active(env)
    if is_active:
        print(f"[{now_str}] [WATCHDOG] Cloud 24/7 worker active ({info}). Standing by.", flush=True)
        return 0

    print(f"[{now_str}] [WATCHDOG] Cloud worker NOT active ({info}). Triggering respawn & local failover...", flush=True)
    spawned = trigger_cloud_worker(env)
    print(f"[{now_str}] [WATCHDOG] Cloud respawn triggered: {spawned}. Running 1 local failover cycle...", flush=True)
    run_local_failover_cycle(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())

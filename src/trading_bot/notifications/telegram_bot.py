"""Interactive Telegram Bot, Push Notification Service & Weekly Chart Telemetry.

Provides secure, password-protected interactive control, real-time push notifications,
on-demand technical chart rendering, and automated weekly movement reports for the
Autonomous Compounding Trading Bot.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import requests
import pandas as pd
import numpy as np

# Use non-interactive Agg backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from trading_bot.config import PROJECT_ROOT, to_display_symbol, normalize_symbol
from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.channels.telegram import TelegramChannel
from trading_bot.notifications.models import (
    BaseNotificationEvent,
    DeliveryStatus,
    OrderSide,
    PortfolioUpdatePayload,
    RiskAlertPayload,
    TradeExecutedPayload,
)
from trading_bot.notifications.admin_commands import (
    ADMIN_CHAT_ID,
    AdminCommandQueue,
    calculate_risk_reward_estimates,
    format_admin_top3_report,
)

logger = logging.getLogger("telegram_bot")


ADMIN_USERNAME = os.getenv("TELEGRAM_ADMIN_USERNAME", "").strip().lower().lstrip("@")


class TelegramAuthManager:
    """Manages persistent authorization for Telegram chat IDs."""

    def __init__(self, auth_file: Union[str, Path]):
        self.auth_file = Path(auth_file)
        self._lock = threading.Lock()
        self._users: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.auth_file.exists() or self.auth_file.stat().st_size == 0:
                self._users = {}
                return
            try:
                with open(self.auth_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._users = {str(k): v for k, v in data.items()}
                    elif isinstance(data, list):
                        self._users = {str(u): {"authenticated_at": datetime.now(timezone.utc).isoformat()} for u in data}
            except Exception as ex:
                logger.warning(f"Failed to load Telegram auth file {self.auth_file}: {ex}")
                self._users = {}

    def _save(self) -> None:
        try:
            self.auth_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.auth_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._users, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.auth_file)
        except Exception as ex:
            logger.error(f"Failed to save Telegram auth file {self.auth_file}: {ex}")

    def is_authenticated(self, chat_id: Union[int, str]) -> bool:
        with self._lock:
            return str(chat_id) in self._users

    def get_user(self, chat_id: Union[int, str]) -> Dict[str, Any]:
        with self._lock:
            return dict(self._users.get(str(chat_id), {}))

    def update_user_meta(self, chat_id: Union[int, str], info: Dict[str, Any]) -> None:
        with self._lock:
            cid = str(chat_id)
            if cid in self._users:
                self._users[cid].update(info)
                self._save()

    def is_admin(self, chat_id: Union[int, str], explicit_username: Optional[str] = None) -> bool:
        return str(chat_id).strip() == str(ADMIN_CHAT_ID).strip()

    def authenticate(self, chat_id: Union[int, str], info: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            cid = str(chat_id)
            meta = info or {}
            meta["authenticated_at"] = datetime.now(timezone.utc).isoformat()
            self._users[cid] = meta
            self._save()
            logger.info(f"Chat ID {cid} successfully authenticated.")

    def deauthenticate(self, chat_id: Union[int, str]) -> bool:
        with self._lock:
            cid = str(chat_id)
            if cid in self._users:
                del self._users[cid]
                self._save()
                logger.info(f"Chat ID {cid} logged out.")
                return True
            return False

    def get_authenticated_chat_ids(self) -> List[str]:
        with self._lock:
            return list(self._users.keys())


class TelegramApiClient:
    """Wrapper around standard Telegram Bot HTTP API with photo attachment support."""

    def __init__(self, token: str, timeout: float = 12.0):
        self.token = token.strip()
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.timeout = timeout
        self.session = requests.Session()

    def send_message(
        self,
        chat_id: Union[int, str],
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: str = "Markdown",
    ) -> bool:
        url = f"{self.base_url}/sendMessage"
        payload: Dict[str, Any] = {
            "chat_id": str(chat_id),
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return True
            logger.warning(f"Telegram sendMessage error {resp.status_code}: {resp.text}")
            return False
        except Exception as ex:
            logger.warning(f"Telegram sendMessage exception: {ex}")
            return False

    def send_photo(
        self,
        chat_id: Union[int, str],
        photo_path: Union[str, Path],
        caption: Optional[str] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: str = "Markdown",
    ) -> bool:
        """Send photo to chat via multipart/form-data."""
        url = f"{self.base_url}/sendPhoto"
        data: Dict[str, Any] = {
            "chat_id": str(chat_id),
            "parse_mode": parse_mode,
        }
        if caption:
            data["caption"] = caption
        if reply_markup is not None:
            data["reply_markup"] = json.dumps(reply_markup)

        try:
            p_file = Path(photo_path)
            if not p_file.exists():
                logger.warning(f"Cannot send non-existent photo: {photo_path}")
                return False

            with open(p_file, "rb") as f:
                files = {"photo": (p_file.name, f, "image/png")}
                resp = self.session.post(url, data=data, files=files, timeout=self.timeout + 15)
                if resp.status_code == 200:
                    return True
                logger.warning(f"Telegram sendPhoto error {resp.status_code}: {resp.text}")
                return False
        except Exception as ex:
            logger.warning(f"Telegram sendPhoto exception: {ex}")
            return False

    def delete_message(self, chat_id: Union[int, str], message_id: int) -> bool:
        url = f"{self.base_url}/deleteMessage"
        payload = {
            "chat_id": str(chat_id),
            "message_id": message_id,
        }
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def answer_callback_query(self, callback_query_id: str, text: Optional[str] = None) -> bool:
        url = f"{self.base_url}/answerCallbackQuery"
        payload: Dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def get_updates(self, offset: Optional[int] = None, timeout: int = 5) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/getUpdates"
        payload: Dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset

        try:
            resp = self.session.post(url, json=payload, timeout=timeout + 5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    return data.get("result", [])
            return []
        except Exception:
            return []


def generate_live_movements_chart(
    context: Dict[str, Any],
    output_path: Union[str, Path] = "/tmp/telegram_daily_chart.png",
    db_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Generate high-resolution multi-panel dark theme chart showing daily movements."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    db_file = Path(db_path or (PROJECT_ROOT / "data" / "market_data.db"))
    trades_csv_path = Path(context.get("trades_csv", PROJECT_ROOT / "data" / "trades.csv"))

    ledger = context.get("ledger")
    simulator = context.get("simulator")
    price_tracker = context.get("price_tracker", {})

    if simulator and hasattr(simulator, "get_wallet_balances"):
        try:
            bals = simulator.get_wallet_balances()
            eur_val = bals.get("EUR") or bals.get("eur")
            if eur_val is not None and ledger:
                ledger.cash = Decimal(str(eur_val))
        except Exception:
            pass

    current_cash = Decimal("150.00")
    if ledger and hasattr(ledger, "cash"):
        current_cash = Decimal(str(ledger.cash))

    net_profit = Decimal("0.00")
    if ledger:
        realized = getattr(ledger, "realized_pnl", Decimal("0.0"))
        unrealized = Decimal("0.0")
        if hasattr(ledger, "unrealized_pnl"):
            try:
                unrealized = ledger.unrealized_pnl(price_tracker)
            except Exception:
                unrealized = Decimal("0.0")
        net_profit = realized + unrealized

    sign_p = "+" if net_profit >= Decimal("0.0") else ""
    net_profit_str = f"{sign_p}€{net_profit:,.2f}"

    # Dark modern visual palette
    bg_dark = "#0b0f19"
    card_bg = "#111827"
    grid_color = "#1f2937"
    text_color = "#f3f4f6"
    subtext_color = "#9ca3af"

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 16), sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.25, 1.25, 1.25]}
    )
    fig.patch.set_facecolor(bg_dark)

    # --------------------------------------------------------------------------
    # Panel 1: Portfolio Equity & Realized Returns Curve
    # --------------------------------------------------------------------------
    ax0 = axes[0]
    ax0.set_facecolor(card_bg)
    ax0.tick_params(colors=subtext_color, labelsize=9)
    ax0.grid(True, linestyle="--", alpha=0.35, color=grid_color)
    for spine in ax0.spines.values():
        spine.set_color("#374151")

    # Load trades from CSV to draw progressive equity
    now_utc = datetime.now(timezone.utc)
    one_week_ago = now_utc - timedelta(days=7)

    dates: List[datetime] = []
    equity_curve: List[float] = []

    if trades_csv_path.exists() and trades_csv_path.stat().st_size > 50:
        try:
            t_df = pd.read_csv(trades_csv_path)
            if not t_df.empty and "portfolio_balance" in t_df.columns:
                t_df["ts"] = pd.to_datetime(t_df["timestamp"])
                for _, row in t_df.iterrows():
                    dates.append(row["ts"])
                    equity_curve.append(float(row["portfolio_balance"]))
        except Exception as ex:
            logger.warning(f"Could not parse trades.csv for chart: {ex}")

    if not dates:
        # Pristine capital without drawdowns
        dates = [one_week_ago, now_utc]
        equity_curve = [float(current_cash), float(current_cash)]

    ax0.plot(dates, equity_curve, color="#10b981", linewidth=2.4, label=f"Equity Portafoglio (€{current_cash:,.2f})")
    ax0.fill_between(dates, float(current_cash) * 0.95, equity_curve, color="#10b981", alpha=0.12)
    ax0.axhline(float(current_cash), color="#38bdf8", linestyle=":", alpha=0.7, label=f"Guadagni Netti: {net_profit_str}")

    ax0.set_title(
        f"PORTFOLIO EQUITY & LIQUIDITÀ (€) — REVOLUT X (0.00% MAKER FEE)\n"
        f"Capitale Totale: €{current_cash:,.2f} | Guadagni Netti: {net_profit_str} | Strategia 4h Trend Breakout",
        color=text_color, fontsize=12, fontweight="bold", pad=10, loc="left"
    )
    ax0.set_ylabel("EUR (€)", color=subtext_color, fontsize=10)
    ax0.legend(loc="upper left", facecolor=card_bg, edgecolor="#374151", labelcolor=text_color, fontsize=9)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))

    # --------------------------------------------------------------------------
    # Panels 2, 3, 4: 4h Candlestick & Technical Regimes (SOL, ETH, BTC)
    # --------------------------------------------------------------------------
    pairs_meta = [
        {"symbol": "SOLEUR", "display": "SOL/EUR", "color": "#38bdf8", "ax": axes[1]},
        {"symbol": "ETHEUR", "display": "ETH/EUR", "color": "#c084fc", "ax": axes[2]},
        {"symbol": "BTCEUR", "display": "BTC/EUR", "color": "#f59e0b", "ax": axes[3]},
    ]

    conn = None
    if db_file.exists():
        try:
            conn = sqlite3.connect(str(db_file))
        except Exception as ex:
            logger.warning(f"Could not open market db: {ex}")

    strategies: Dict[str, Any] = context.get("strategies", {})

    for item in pairs_meta:
        sym = item["symbol"]
        disp = item["display"]
        ax = item["ax"]
        c_main = item["color"]

        ax.set_facecolor(card_bg)
        ax.tick_params(colors=subtext_color, labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.35, color=grid_color)
        for spine in ax.spines.values():
            spine.set_color("#374151")

        df = pd.DataFrame()
        if conn:
            try:
                query = (
                    f"SELECT open_time, open, high, low, close, volume FROM candles "
                    f"WHERE symbol='{sym}' AND interval='4h' "
                    f"ORDER BY open_time DESC LIMIT 220"
                )
                df = pd.read_sql_query(query, conn)
            except Exception as ex:
                logger.warning(f"Error querying {sym}: {ex}")

        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
            df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
            df["donchian_high"] = df["high"].rolling(20).max()
            df["donchian_low"] = df["low"].rolling(20).min()

            # Focus on last 84 bars (14 days of 4h data)
            sub = df.tail(84).copy().reset_index(drop=True)

            last_px = sub["close"].iloc[-1]
            first_px = sub["close"].iloc[0]
            pct_chg = ((last_px - first_px) / first_px) * 100.0
            sign = "+" if pct_chg >= 0 else ""

            # Plot Price Line & Donchian Channel
            ax.plot(sub["datetime"], sub["close"], color=c_main, linewidth=2.0, label=f"{disp} Prezzo (€{last_px:,.2f})")
            ax.plot(sub["datetime"], sub["ema200"], color="#f97316", linewidth=1.4, linestyle="--", label="EMA 200 Trendline")
            ax.plot(sub["datetime"], sub["donchian_high"], color="#10b981", linewidth=1.0, linestyle=":", label="Donchian 20 High (Breakout)")
            ax.plot(sub["datetime"], sub["donchian_low"], color="#ef4444", linewidth=1.0, linestyle=":", label="Donchian 20 Low")
            ax.fill_between(sub["datetime"], sub["donchian_low"], sub["donchian_high"], color=c_main, alpha=0.06)

            # Check if active position or Stop Loss is set
            strat = strategies.get(disp) or strategies.get(sym) or strategies.get(disp.replace("/", ""))
            stop_lvl = None
            if strat and hasattr(strat, "position_state"):
                pst = strat.position_state
                if pst.state.value != "IDLE" and pst.current_stop:
                    stop_lvl = float(pst.current_stop)
                    ax.axhline(stop_lvl, color="#ef4444", linestyle="-.", linewidth=1.5, label=f"Stop Loss attivo (€{stop_lvl:,.2f})")

            chg_color = "#10b981" if pct_chg >= 0 else "#ef4444"
            ax.set_title(
                f"[4H TREND] {disp} — Ultimo: €{last_px:,.2f} ({sign}{pct_chg:.2f}% su 14gg) | Regime Trend Breakout",
                color=text_color, fontsize=11, fontweight="bold", pad=8, loc="left"
            )
            ax.set_ylabel(f"{disp} (€)", color=subtext_color, fontsize=9)
            ax.legend(loc="upper left", facecolor=card_bg, edgecolor="#374151", labelcolor=text_color, fontsize=8.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        else:
            ax.text(0.5, 0.5, f"Dati 4h per {disp} in fase di sincronizzazione...",
                    color=subtext_color, ha="center", va="center", transform=ax.transAxes)

    if conn:
        conn.close()

    fig.suptitle(
        f"AUTONOMOUS TRADING BOT — REPORT TECNICO & MOVIMENTI CICLO 4H\n"
        f"Generato il: {now_utc.strftime('%d/%m/%Y alle %H:%M:%S UTC')}",
        color=text_color, fontsize=13, fontweight="bold", y=0.995
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.32)
    fig.savefig(out_file, dpi=160, facecolor=bg_dark, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated daily movement chart at {out_file}")
    return out_file


class TelegramPushChannel(BaseNotificationChannel):
    """Notification channel that broadcasts trade and portfolio alerts to all authenticated users."""

    def __init__(self, api_client: TelegramApiClient, auth_manager: TelegramAuthManager, enabled: bool = True):
        self.api_client = api_client
        self.auth_manager = auth_manager
        self._enabled = enabled
        self._formatter = TelegramChannel(bot_token="dummy", chat_id="dummy")

    @property
    def name(self) -> str:
        return "telegram_push"

    def is_enabled(self) -> bool:
        return self._enabled and bool(self.api_client.token)

    def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
        if not self.is_enabled():
            return DeliveryStatus(success=True, channel_name=self.name, error_message="Channel disabled")

        auth_chats = self.auth_manager.get_authenticated_chat_ids()
        if not auth_chats:
            return DeliveryStatus(success=True, channel_name=self.name, error_message="No authenticated recipients")

        try:
            text = self._formatter.format_message(event)
            success_count = 0
            for cid in auth_chats:
                if self.api_client.send_message(chat_id=cid, text=text):
                    success_count += 1
            return DeliveryStatus(success=(success_count > 0), channel_name=self.name)
        except Exception as ex:
            return DeliveryStatus(success=False, channel_name=self.name, error_message=str(ex))


class TelegramBotListener:
    """Background listener that handles commands, keyboards, on-demand charts, and scheduled daily reports."""

    def __init__(
        self,
        token: str,
        password: str,
        auth_file: Union[str, Path],
        context_provider: Dict[str, Any],
        daily_state_file: Optional[Union[str, Path]] = None,
        weekly_state_file: Optional[Union[str, Path]] = None,
        report_4h_state_file: Optional[Union[str, Path]] = None,
    ):
        self.token = token.strip()
        self.password = password.strip()
        self.auth_manager = TelegramAuthManager(auth_file)
        self.api_client = TelegramApiClient(self.token)
        self.ctx = context_provider
        state_path = daily_state_file or weekly_state_file or (PROJECT_ROOT / "data" / "telegram_last_daily_report.json")
        self.daily_state_file = Path(state_path)
        self.weekly_state_file = self.daily_state_file
        h4_path = report_4h_state_file or (PROJECT_ROOT / "data" / "telegram_last_4h_report.json")
        self.report_4h_state_file = Path(h4_path)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_lock = threading.Lock()
        self._last_offset: Optional[int] = None
        self._last_chart_generated_at: float = 0.0
        self._cached_chart_path = Path("/tmp/telegram_daily_chart.png")
        self._latest_eval_summaries: List[Dict[str, Any]] = []
        self.admin_queue = AdminCommandQueue()

    def set_latest_eval_summaries(self, summaries: List[Dict[str, Any]]) -> None:
        """Cache the latest 4h candle evaluation summaries for instant admin query access."""
        if summaries:
            self._latest_eval_summaries = list(summaries)

    @property
    def push_channel(self) -> TelegramPushChannel:
        return TelegramPushChannel(self.api_client, self.auth_manager)

    def start(self) -> None:
        if not self.token:
            logger.info("TelegramBotListener: Token not provided, listener not started.")
            return
        if self._running and self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="TelegramBotDaemon", daemon=True)
        self._thread.start()
        logger.info("TelegramBotListener started in background thread with 4h evaluation cadence.")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _get_reply_keyboard(self, is_admin: bool = False) -> Dict[str, Any]:
        keyboard = [
            [{"text": "📊 Stato & Saldo"}, {"text": "📈 Prezzi & Segnali"}],
            [{"text": "📉 Grafico Movimenti"}, {"text": "📜 Ultimi Trade"}],
            [{"text": "🔄 Sync Revolut X"}],
        ]
        if is_admin:
            keyboard[2].append({"text": "🔒 Blocca"})
        return {
            "keyboard": keyboard,
            "resize_keyboard": True,
            "one_time_keyboard": False,
        }

    def _get_inline_keyboard(self, is_admin: bool = False) -> Dict[str, Any]:
        if not is_admin:
            rows = [
                [
                    {"text": "📊 Stato & Saldo", "callback_data": "cb_status"},
                    {"text": "📈 Prezzi & Segnali", "callback_data": "cb_prices"},
                ],
                [
                    {"text": "📉 Grafico Movimenti", "callback_data": "cb_chart"},
                    {"text": "📜 Ultimi Trade", "callback_data": "cb_trades"},
                ],
                [
                    {"text": "🔄 Sync Revolut X", "callback_data": "cb_sync"},
                ],
            ]
            return {"inline_keyboard": rows}

        # Dynamic Admin Keyboard: shows [⚡ Avvia] or [🛑 Chiudi] based on actual positions
        ledger = self.ctx.get("ledger")
        positions = getattr(ledger, "positions", {}) if ledger else {}

        def btn_for_coin(coin: str) -> Dict[str, str]:
            pair_disp = f"{coin}/EUR"
            pair_norm = f"{coin}EUR"
            p_obj = positions.get(pair_norm) or positions.get(pair_disp) or positions.get(coin)
            has_pos = p_obj and getattr(p_obj, "quantity", Decimal("0")) > Decimal("0")
            if has_pos:
                return {"text": f"🛑 Chiudi {coin}", "callback_data": f"cmd_force_sell_{coin}"}
            return {"text": f"⚡ Avvia {coin}", "callback_data": f"cmd_force_buy_{coin}"}

        rows = [
            [
                {"text": "📊 Stato & Saldo", "callback_data": "cb_status"},
                {"text": "📈 Prezzi & Segnali", "callback_data": "cb_prices"},
            ],
            [
                {"text": "📉 Grafico Movimenti", "callback_data": "cb_chart"},
                {"text": "📜 Ultimi Trade", "callback_data": "cb_trades"},
            ],
            [
                btn_for_coin("BTC"),
                btn_for_coin("SOL"),
                btn_for_coin("ETH"),
            ],
            [
                {"text": "🔄 Sync Revolut X", "callback_data": "cb_sync"},
                {"text": "🔒 Blocca Sessione", "callback_data": "cb_logout"},
            ],
        ]
        return {"inline_keyboard": rows}

    def _poll_and_process(self, timeout: int = 2) -> None:
        """Thread-safe single poll and dispatch of pending Telegram updates."""
        if not self.token:
            return
        with self._poll_lock:
            try:
                updates = self.api_client.get_updates(offset=self._last_offset, timeout=timeout)
                for update in updates:
                    update_id = update.get("update_id")
                    if update_id is not None:
                        self._last_offset = update_id + 1
                    try:
                        if "message" in update:
                            self._handle_message(update["message"])
                        elif "callback_query" in update:
                            self._handle_callback_query(update["callback_query"])
                    except Exception as handler_ex:
                        logger.warning(f"Error handling Telegram update {update_id}: {handler_ex}")
            except Exception as ex:
                logger.warning(f"Error polling Telegram updates: {ex}")

    def process_pending_updates_once(self) -> None:
        """Execute a single cycle of getUpdates (timeout=2) and process pending messages/callbacks."""
        self._poll_and_process(timeout=2)

    def _get_github_token(self) -> Optional[str]:
        """Retrieve GitHub token from environment or local gh CLI auth."""
        tok = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if tok and tok.strip():
            return tok.strip()
        try:
            import subprocess
            res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=4)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return None

    def _fetch_remote_4h_state(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetch remote data/telegram_last_4h_report.json from GitHub repo."""
        token = self._get_github_token()
        if not token:
            return None, None
        url = "https://api.github.com/repos/VALENTINOGIGLIO/revolutx-trading-bot-cloud/contents/data/telegram_last_4h_report.json"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=6.0)
            if resp.status_code == 200:
                body = resp.json()
                sha = body.get("sha")
                raw_b64 = body.get("content", "")
                if raw_b64:
                    import base64
                    content = base64.b64decode(raw_b64).decode("utf-8")
                    data = json.loads(content)
                    return data, sha
            elif resp.status_code == 404:
                return None, None
            else:
                logger.warning(f"GitHub fetch remote 4h state returned {resp.status_code}: {resp.text[:100]}")
        except Exception as ex:
            logger.warning(f"GitHub fetch remote 4h state failed: {ex}")
        return None, None

    def _push_remote_4h_state(self, state_dict: Dict[str, Any], sha: Optional[str] = None) -> bool:
        """Push updated 4h state to GitHub cloud repository."""
        token = self._get_github_token()
        if not token:
            return False

        if sha is None:
            _, sha = self._fetch_remote_4h_state()

        url = "https://api.github.com/repos/VALENTINOGIGLIO/revolutx-trading-bot-cloud/contents/data/telegram_last_4h_report.json"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        import base64
        payload = {
            "message": "sync: update telegram_last_4h_report.json [skip ci]",
            "content": base64.b64encode(json.dumps(state_dict, indent=2).encode("utf-8")).decode("utf-8"),
        }
        if sha:
            payload["sha"] = sha

        try:
            resp = requests.put(url, headers=headers, json=payload, timeout=8.0)
            if resp.status_code in (200, 201):
                logger.info("Successfully pushed 4h report state to GitHub cloud repository.")
                return True
            elif resp.status_code == 409:
                # Concurrent update conflict, fetch latest sha and retry once
                _, latest_sha = self._fetch_remote_4h_state()
                if latest_sha:
                    payload["sha"] = latest_sha
                    retry_resp = requests.put(url, headers=headers, json=payload, timeout=8.0)
                    return retry_resp.status_code in (200, 201)
            logger.warning(f"GitHub push remote 4h state returned {resp.status_code}: {resp.text[:100]}")
            return False
        except Exception as ex:
            logger.warning(f"GitHub push remote 4h state exception: {ex}")
            return False

    def _read_local_4h_state(self) -> Dict[str, Any]:
        """Read 4h report state from local file."""
        if self.report_4h_state_file.exists():
            try:
                with open(self.report_4h_state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as ex:
                logger.warning(f"Failed to read local 4h report state: {ex}")
        return {}

    def _save_local_4h_state(self, state_dict: Dict[str, Any]) -> None:
        """Persist 4h report state to local file atomically."""
        try:
            self.report_4h_state_file.parent.mkdir(parents=True, exist_ok=True)
            temp = self.report_4h_state_file.with_suffix(".tmp")
            with open(temp, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
            temp.replace(self.report_4h_state_file)
        except Exception as ex:
            logger.warning(f"Failed to save local 4h report state: {ex}")

    def send_4h_evaluation_report(
        self,
        candle_ts_iso: str,
        eval_summaries: List[Dict[str, Any]],
        force: bool = False,
    ) -> List[str]:
        """Send automated 4h evaluation report with 4-panel chart to authenticated users.

        Synchronizes state locally and with GitHub remote to eliminate duplicate alerts
        between local daemon and cloud server.
        """
        if eval_summaries:
            self.set_latest_eval_summaries(eval_summaries)

        auth_chats = self.auth_manager.get_authenticated_chat_ids()
        if not auth_chats:
            logger.info("No authenticated recipients for 4h evaluation report.")
            return []

        now_utc = datetime.now(timezone.utc)
        remote_sha = None

        if not force:
            # 1. Check local state (candle match and anti-spam timing: < 3h 50m / 13800s)
            local_state = self._read_local_4h_state()
            if local_state.get("last_candle_ts") == candle_ts_iso:
                logger.info(f"[4H REPORT] Already sent locally for candle {candle_ts_iso}. Skipping.")
                return []

            last_local_sent = local_state.get("last_sent_at")
            if last_local_sent:
                try:
                    dt_local = datetime.fromisoformat(last_local_sent)
                    elapsed = (now_utc - dt_local).total_seconds()
                    if elapsed < 13800:
                        logger.info(f"[4H REPORT] Local report sent {elapsed:.0f}s ago (< 13800s). Anti-spam SKIP ASSOLUTO.")
                        return []
                except Exception as ex:
                    logger.debug(f"Error parsing local last_sent_at: {ex}")

            # 2. Check remote GitHub state
            remote_state, remote_sha = self._fetch_remote_4h_state()
            if remote_state and remote_state.get("last_candle_ts") == candle_ts_iso:
                logger.info(f"[4H REPORT] Already sent by remote cloud for candle {candle_ts_iso}. Syncing local and skipping.")
                self._save_local_4h_state(remote_state)
                return []

            if remote_state and remote_state.get("last_sent_at"):
                try:
                    dt_remote = datetime.fromisoformat(remote_state.get("last_sent_at"))
                    elapsed_rem = (now_utc - dt_remote).total_seconds()
                    if elapsed_rem < 13800:
                        logger.info(f"[4H REPORT] Remote report sent {elapsed_rem:.0f}s ago (< 13800s). Anti-spam SKIP ASSOLUTO.")
                        self._save_local_4h_state(remote_state)
                        return []
                except Exception as ex:
                    logger.debug(f"Error parsing remote last_sent_at: {ex}")

            # 3. Cloud suppression: if running on GitHub Actions, do not send if local Mac already sent within 13800s
            if os.getenv("GITHUB_ACTIONS"):
                if local_state.get("sent_by") == "local" and local_state.get("last_sent_at"):
                    try:
                        dt_loc = datetime.fromisoformat(local_state.get("last_sent_at"))
                        if (now_utc - dt_loc).total_seconds() < 13800:
                            logger.info("[4H REPORT Cloud] Local Mac active and sent report recently. Cloud skips sending.")
                            return []
                    except Exception:
                        pass

        # 3. Extract portfolio metrics (strictly equity and net profit)
        ledger = self.ctx.get("ledger")
        price_tracker = self.ctx.get("price_tracker", {})
        equity = float(ledger.total_equity(price_tracker)) if ledger else 150.0
        realized = getattr(ledger, "realized_pnl", Decimal("0.0")) if ledger else Decimal("0.0")
        unrealized = Decimal("0.0")
        if ledger and hasattr(ledger, "unrealized_pnl"):
            try:
                unrealized = ledger.unrealized_pnl(price_tracker)
            except Exception:
                unrealized = Decimal("0.0")
        net_pnl = float(realized + unrealized)

        # 4. Generate refreshed 4-panel movement chart
        chart_path = self._generate_or_get_chart(force_refresh=True)

        target_order = ["BTC/EUR", "SOL/EUR", "ETH/EUR"]
        def _sort_key(item: Dict[str, Any]) -> int:
            sym = item.get("symbol", "")
            return target_order.index(sym) if sym in target_order else 99

        sorted_summaries = sorted(eval_summaries, key=_sort_key)

        # 5. Build Standard Clean Caption (for non-admin users)
        std_lines = [
            "⏱ *CONTROLLO CICLO 4H — @GILLYTRADESBOT_bot*",
            f"Candela 4H: `{candle_ts_iso}`",
            "",
            f"💰 *Capitale Totale:* €{equity:.2f}",
            f"📈 *Guadagni Netti:* {net_pnl:+.2f} €",
            "",
            "*Stato delle 3 monete (4H):*",
        ]
        for s in sorted_summaries:
            sym = s.get("symbol", "")
            px = s.get("price", 0.0)
            ema = s.get("ema_200", 0.0)
            above_ema = s.get("above_ema200", False)
            ema_badge = "✅ Sopra" if above_ema else "❌ Sotto"
            bo_lvl = s.get("breakout_level", 0.0)
            dist = s.get("dist_pct", 0.0)
            outcome = s.get("outcome", "⏸ Nessun segnale (IDLE)")

            std_lines.append(
                f"• *{sym}*: €{px:,.2f} | EMA 200: {ema_badge} (€{ema:,.2f})\n"
                f"  Breakout Keltner: €{bo_lvl:,.2f} ({dist:+.2f}%) | {outcome}"
            )
        std_lines.extend(["", "Prossima attivazione: `+4 ore`"])
        standard_caption = "\n".join(std_lines)

        # 6. Build Admin Dedicated Clean Top 3 Setup Ranking with Minimum Threshold Benchmark
        admin_text = format_admin_top3_report(
            eval_summaries=sorted_summaries,
            equity=equity,
            net_pnl=net_pnl,
            candle_ts_iso=candle_ts_iso,
        )

        # 7. Dispatching: customized per user tier
        delivered: List[str] = []
        for cid in auth_chats:
            user_info = self.auth_manager.get_user(cid)
            is_adm = self.auth_manager.is_admin(cid, explicit_username=user_info.get("username"))
            markup = self._get_inline_keyboard(is_admin=is_adm)

            if is_adm:
                # Admin receives overview chart followed immediately by full technical breakdown & action buttons
                if chart_path and Path(chart_path).exists():
                    self.api_client.send_photo(
                        chat_id=cid,
                        photo_path=chart_path,
                        caption=standard_caption,
                    )
                ok = self.api_client.send_message(
                    chat_id=cid,
                    text=admin_text,
                    reply_markup=markup,
                )
            else:
                # Regular users receive clean concise caption with no technical breakdown and standard keyboard
                ok = False
                if chart_path and Path(chart_path).exists():
                    ok = self.api_client.send_photo(
                        chat_id=cid,
                        photo_path=chart_path,
                        caption=standard_caption,
                        reply_markup=markup,
                    )
                if not ok:
                    ok = self.api_client.send_message(
                        chat_id=cid,
                        text=standard_caption,
                        reply_markup=markup,
                    )

            if ok:
                delivered.append(str(cid))
                logger.info(f"[4H REPORT] Dispatched successfully to {cid} (is_admin={is_adm})")

        # 8. Update state locally and remotely
        now_iso = now_utc.isoformat()
        sent_by = "cloud" if os.getenv("GITHUB_ACTIONS") else "local"
        new_state = {
            "last_candle_ts": candle_ts_iso,
            "last_sent_at": now_iso,
            "sent_by": sent_by,
        }
        self._save_local_4h_state(new_state)
        self._push_remote_4h_state(new_state, sha=remote_sha)

        return delivered

    def _run_loop(self) -> None:
        while self._running:
            try:
                self._poll_and_process(timeout=4)
                time.sleep(0.5)
            except Exception as ex:
                logger.warning(f"Telegram polling error: {ex}")
                time.sleep(2.0)

    def _check_and_trigger_daily_report(self) -> None:
        """Legacy placeholder for daily report scheduler (now superseded by 4h cadence)."""
        pass

    # Legacy alias
    _check_and_trigger_weekly_report = _check_and_trigger_daily_report

    def _save_daily_report_timestamp(self, dt: datetime) -> None:
        try:
            self.daily_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.daily_state_file, "w", encoding="utf-8") as f:
                json.dump({"last_sent_iso": dt.isoformat()}, f, indent=2)
        except Exception as ex:
            logger.warning(f"Could not save daily report timestamp: {ex}")

    # Legacy alias
    _save_weekly_report_timestamp = _save_daily_report_timestamp

    def send_daily_report_now(self, dt: Optional[datetime] = None) -> List[str]:
        """Broadcast daily chart report immediately to all authenticated users."""
        now_utc = dt or datetime.now(timezone.utc)
        auth_chats = self.auth_manager.get_authenticated_chat_ids()
        if not auth_chats:
            logger.info("No authenticated users to broadcast daily report to.")
            return []

        chart_path = self._generate_or_get_chart(force_refresh=True)
        caption = (
            "📅 *REPORT GRAFICO GIORNALIERO DEI MOVIMENTI*\n\n"
            "Ecco l'analisi completa dei grafici a 4h e l'andamento del capitale del Trading Bot Revolut X.\n"
            "• *EMA 200* e *Canali Donchian 20* aggiornati\n"
            "• *Commissioni*: 0.00% Maker su Revolut X\n"
            f"• *Generato il*: {now_utc.strftime('%d/%m/%Y alle %H:%M:%S UTC')}"
        )
        delivered = []
        for cid in auth_chats:
            user_info = self.auth_manager.get_user(cid)
            is_adm = self.auth_manager.is_admin(cid, explicit_username=user_info.get("username"))
            ok = self.api_client.send_photo(
                chat_id=cid,
                photo_path=chart_path,
                caption=caption,
                reply_markup=self._get_inline_keyboard(is_admin=is_adm),
            )
            if ok:
                delivered.append(str(cid))
        self._save_daily_report_timestamp(now_utc)
        return delivered

    def _generate_or_get_chart(self, force_refresh: bool = False) -> Path:
        """Generate chart or use cached version if generated within last 30s."""
        now_ts = time.time()
        if force_refresh or (now_ts - self._last_chart_generated_at > 30.0) or not self._cached_chart_path.exists():
            chart_path = generate_live_movements_chart(context=self.ctx, output_path=self._cached_chart_path)
            self._last_chart_generated_at = now_ts
            return chart_path
        return self._cached_chart_path

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        chat = msg.get("chat", {})
        from_user = msg.get("from", {})
        chat_id = chat.get("id")
        if not chat_id:
            return

        text = (msg.get("text") or "").strip()
        msg_id = msg.get("message_id")
        is_auth = self.auth_manager.is_authenticated(chat_id)

        username = (from_user.get("username") or chat.get("username") or "").strip()
        first_name = (from_user.get("first_name") or chat.get("first_name") or "").strip()
        if is_auth and (username or first_name):
            update_data: Dict[str, Any] = {}
            if username:
                update_data["username"] = username
            if first_name:
                update_data["first_name"] = first_name
            self.auth_manager.update_user_meta(chat_id, update_data)

        is_admin = self.auth_manager.is_admin(chat_id, explicit_username=username)

        # 1. Handle Password Submission
        is_password_match = bool(self.password) and (
            text == self.password
            or text == f"/login {self.password}"
            or text == f"/start {self.password}"
        )

        if is_password_match:
            if msg_id:
                self.api_client.delete_message(chat_id, msg_id)

            user_meta = {
                "first_name": first_name,
                "username": username,
            }
            self.auth_manager.authenticate(chat_id, user_meta)
            is_admin = self.auth_manager.is_admin(chat_id, explicit_username=username)

            welcome_msg = (
                "✅ *Accesso Autorizzato!*\n"
                "Benvenuto nel Pannello di Controllo Telegram del Trading Bot.\n"
                "La sessione è ora attiva con comandi interattivi, notifiche push e report grafici giornalieri automatici.\n\n"
                f"{self._format_status()}"
            )
            self.api_client.send_message(
                chat_id,
                welcome_msg,
                reply_markup=self._get_reply_keyboard(is_admin=is_admin),
            )
            return

        # 2. If Not Authenticated, prompt for password
        if not is_auth:
            prompt = (
                "🔒 *Accesso Protetto*\n"
                "Questo bot è riservato al trading con fondi reali.\n"
                "Inserisci la password per sbloccare il Trading Bot:"
            )
            self.api_client.send_message(chat_id, prompt)
            return

        # 3. Authenticated Commands
        lower_txt = text.strip().lower()
        parts = lower_txt.split()
        first_word = parts[0] if parts else ""

        # Admin Manual Override Commands: /forza btc|sol|eth, /avvia btc|sol|eth, /chiudi btc|sol|eth
        if first_word in ("/forza", "/avvia", "/chiudi", "forza", "avvia", "chiudi"):
            if not is_admin:
                self.api_client.send_message(
                    chat_id,
                    "⛔ *Accesso Negato*: Solo l'amministratore ha l'autorizzazione per avviare o chiudere posizioni manualmente.",
                    reply_markup=self._get_reply_keyboard(is_admin=False),
                )
                return

            action = "SELL" if first_word in ("/chiudi", "chiudi") else "BUY"
            if len(parts) >= 2:
                coin = parts[1].upper().replace("/EUR", "")
            else:
                self.api_client.send_message(
                    chat_id,
                    f"⚠️ Specifica la moneta: es. `{first_word} btc`, `{first_word} sol`, `{first_word} eth` oppure usa i pulsanti d'azione sottostanti.",
                    reply_markup=self._get_inline_keyboard(is_admin=True),
                )
                return

            if coin not in ("BTC", "SOL", "ETH"):
                self.api_client.send_message(
                    chat_id,
                    f"⚠️ Moneta `{coin}` non valida. Scegli tra `BTC`, `SOL`, `ETH`.",
                    reply_markup=self._get_inline_keyboard(is_admin=True),
                )
                return

            symbol = f"{coin}/EUR"
            self.admin_queue.enqueue(action=action, symbol=symbol, requested_by=chat_id)

            if action == "BUY":
                ack_msg = (
                    f"⚡ *Avvio Forzato Preso in Carico*\n"
                    f"Moneta: *{symbol}* (Bypass campanelli attivo)\n\n"
                    "Calcolo automatico Stop Loss protettivo (2.2x ATR) ed esecuzione a mercato in corso su Revolut X..."
                )
            else:
                ack_msg = (
                    f"🛑 *Chiusura Forzata Presa in Carico*\n"
                    f"Moneta: *{symbol}*\n\n"
                    "Esecuzione chiusura a mercato in corso su Revolut X..."
                )
            self.api_client.send_message(chat_id, ack_msg, reply_markup=self._get_inline_keyboard(is_admin=True))
            return

        cmd = text.lower()
        if cmd in ("/status", "📊 stato & saldo", "stato", "status"):
            resp = self._format_status()
            self.api_client.send_message(chat_id, resp, reply_markup=self._get_inline_keyboard(is_admin=is_admin))
        elif cmd in ("/chart", "/grafico", "/report", "📉 grafico movimenti", "grafico", "chart"):
            self._send_chart_response(chat_id)
        elif cmd in ("/prices", "/segnali", "/market", "📈 prezzi & segnali", "prezzi", "prices", "segnali", "market"):
            resp = self._format_prices(is_admin=is_admin)
            self.api_client.send_message(chat_id, resp, reply_markup=self._get_inline_keyboard(is_admin=is_admin))
        elif cmd in ("/trades", "📜 ultimi trade", "trade", "trades"):
            resp = self._format_trades()
            self.api_client.send_message(chat_id, resp, reply_markup=self._get_inline_keyboard(is_admin=is_admin))
        elif cmd in ("/sync", "🔄 sync revolut x", "sync"):
            resp = self._format_sync()
            self.api_client.send_message(chat_id, resp, reply_markup=self._get_inline_keyboard(is_admin=is_admin))
        elif cmd in ("/logout", "🔒 blocca", "logout", "blocca"):
            if not is_admin:
                self.api_client.send_message(
                    chat_id,
                    "⛔ *Accesso Negato*: Solo l'amministratore può bloccare le sessioni.",
                    reply_markup=self._get_reply_keyboard(is_admin=False),
                )
                return
            self.auth_manager.deauthenticate(chat_id)
            self.api_client.send_message(
                chat_id,
                "🔒 *Sessione Bloccata*\nAccesso revocato con successo. Inserisci la password per sbloccare di nuovo.",
            )
        elif cmd in ("/start", "/help", "help"):
            admin_line = (
                "\n• ⚡ `/forza btc|sol|eth` : Avvio forzato con calcolo rischio (Bypass campanelli)\n"
                "• 🛑 `/chiudi btc|sol|eth`: Chiusura manuale anticipata posizione\n"
                "• 🔒 `/logout` : Blocca la sessione (Solo Admin)"
                if is_admin else ""
            )
            resp = (
                "🤖 *Pannello Comandi Trading Bot*\n\n"
                "• 📊 `/status` : Stato live, Saldo Revolut X, Guadagni Netti ed Equity\n"
                "• 📉 `/chart`  : Invia subito il grafico aggiornato a 4h e l'equity\n"
                "• 📈 `/prices` : Prezzi attuali, posizioni e Stop Loss\n"
                "• 📜 `/trades` : Ultimi ordini eseguiti\n"
                "• 🔄 `/sync`   : Interrogazione saldi Revolut X in millisecondi"
                f"{admin_line}\n"
            )
            self.api_client.send_message(chat_id, resp, reply_markup=self._get_reply_keyboard(is_admin=is_admin))
        else:
            self.api_client.send_message(
                chat_id,
                "Comando non riconosciuto. Usa i pulsanti sottostanti:",
                reply_markup=self._get_reply_keyboard(is_admin=is_admin),
            )

    def _handle_callback_query(self, query: Dict[str, Any]) -> None:
        qid = query.get("id")
        data = query.get("data", "")
        message = query.get("message", {})
        chat = message.get("chat", {})
        from_user = query.get("from", {})
        chat_id = chat.get("id") or from_user.get("id")

        if not chat_id:
            return

        username = (from_user.get("username") or chat.get("username") or "").strip()
        first_name = (from_user.get("first_name") or chat.get("first_name") or "").strip()
        if self.auth_manager.is_authenticated(chat_id) and (username or first_name):
            update_data = {}
            if username:
                update_data["username"] = username
            if first_name:
                update_data["first_name"] = first_name
            self.auth_manager.update_user_meta(chat_id, update_data)

        is_admin = self.auth_manager.is_admin(chat_id, explicit_username=username)

        if not self.auth_manager.is_authenticated(chat_id):
            if qid:
                self.api_client.answer_callback_query(qid, text="Sessione bloccata.")
            self.api_client.send_message(chat_id, "🔒 Inserisci la password per accedere:")
            return

        if qid:
            self.api_client.answer_callback_query(qid)

        # Step 1 of 2: Ask Confirmation for Force Buy to prevent accidental taps
        if data.startswith("cmd_force_buy_"):
            if not is_admin:
                self.api_client.send_message(
                    chat_id,
                    "⛔ *Accesso Negato*: Solo l'amministratore può forzare operazioni.",
                    reply_markup=self._get_inline_keyboard(is_admin=False),
                )
                return

            coin = data.split("_")[-1].upper()
            symbol = f"{coin}/EUR"
            confirm_kb = {
                "inline_keyboard": [
                    [
                        {"text": f"✅ SÌ, COMPRA {coin} ORA", "callback_data": f"cmd_confirm_buy_{coin}"},
                        {"text": "❌ ANNULLA", "callback_data": "cmd_cancel_buy"},
                    ]
                ]
            }
            confirm_msg = (
                f"⚠️ *CONFERMA AVVIO FORZATO — 2-STEP VERIFICATION*\n\n"
                f"Sei sicuro di voler avviare l'acquisto manuale di *{symbol}*?\n"
                f"• I campanelli mancanti verranno ignorati\n"
                f"• Verrà piazzato un ordine reale a mercato su Revolut X\n"
                f"• Verrà impostato lo Stop Loss protettivo (2.2x ATR)\n\n"
                f"Tocca per confermare o annullare:"
            )
            self.api_client.send_message(chat_id, confirm_msg, reply_markup=confirm_kb)
            return

        if data == "cmd_cancel_buy":
            self.api_client.send_message(
                chat_id,
                "🛡️ *Acquisto Annullato.* Nessun ordine è stato inviato su Revolut X.",
                reply_markup=self._get_inline_keyboard(is_admin=True),
            )
            return

        # Handle Confirmed Buy and Force Sell
        if data.startswith("cmd_confirm_buy_") or data.startswith("cmd_force_sell_"):
            if not is_admin:
                self.api_client.send_message(
                    chat_id,
                    "⛔ *Accesso Negato*: Solo l'amministratore può forzare operazioni.",
                    reply_markup=self._get_inline_keyboard(is_admin=False),
                )
                return

            action = "BUY" if "buy_" in data else "SELL"
            coin = data.split("_")[-1].upper()
            symbol = f"{coin}/EUR"

            self.admin_queue.enqueue(action=action, symbol=symbol, requested_by=chat_id)

            if action == "BUY":
                ack_msg = (
                    f"⚡ *Avvio Forzato Preso in Carico*\n"
                    f"Moneta: *{symbol}* (Bypass campanelli attivo)\n\n"
                    "Calcolo automatico Stop Loss protettivo (2.2x ATR) ed esecuzione a mercato in corso su Revolut X..."
                )
            else:
                ack_msg = (
                    f"🛑 *Chiusura Forzata Presa in Carico*\n"
                    f"Moneta: *{symbol}*\n\n"
                    "Esecuzione chiusura a mercato in corso su Revolut X..."
                )
            self.api_client.send_message(chat_id, ack_msg, reply_markup=self._get_inline_keyboard(is_admin=True))
            return

        if data == "cb_status":
            self.api_client.send_message(chat_id, self._format_status(), reply_markup=self._get_inline_keyboard(is_admin=is_admin))
        elif data == "cb_chart":
            self._send_chart_response(chat_id)
        elif data == "cb_prices":
            self.api_client.send_message(chat_id, self._format_prices(is_admin=is_admin), reply_markup=self._get_inline_keyboard(is_admin=is_admin))
        elif data == "cb_trades":
            self.api_client.send_message(chat_id, self._format_trades(), reply_markup=self._get_inline_keyboard(is_admin=is_admin))
        elif data == "cb_sync":
            self.api_client.send_message(chat_id, self._format_sync(), reply_markup=self._get_inline_keyboard(is_admin=is_admin))
        elif data == "cb_logout":
            if not is_admin:
                self.api_client.send_message(
                    chat_id,
                    "⛔ *Accesso Negato*: Solo l'amministratore può bloccare le sessioni.",
                    reply_markup=self._get_inline_keyboard(is_admin=False),
                )
                return
            self.auth_manager.deauthenticate(chat_id)
            self.api_client.send_message(chat_id, "🔒 *Sessione Bloccata con successo.*")

    def _send_chart_response(self, chat_id: Union[int, str]) -> None:
        user_info = self.auth_manager.get_user(chat_id)
        is_admin = self.auth_manager.is_admin(chat_id, explicit_username=user_info.get("username"))
        self.api_client.send_message(chat_id, "⏳ *Generazione del grafico live dei movimenti in corso...*")
        try:
            chart_path = self._generate_or_get_chart(force_refresh=True)
            caption = (
                "📉 *GRAFICO MOVIMENTI LIVE (4H)*\n"
                "• Curva di Equity & Liquidità Revolut X\n"
                "• Trend 4h di SOL, ETH e BTC con EMA 200 & Donchian 20\n"
                "_Aggiornato in tempo reale._"
            )
            self.api_client.send_photo(
                chat_id=chat_id,
                photo_path=chart_path,
                caption=caption,
                reply_markup=self._get_inline_keyboard(is_admin=is_admin),
            )
        except Exception as ex:
            logger.error(f"Error rendering chart: {ex}", exc_info=True)
            self.api_client.send_message(chat_id, f"⚠️ Errore nella generazione del grafico: {ex}")

    def _format_status(self) -> str:
        ledger = self.ctx.get("ledger")
        simulator = self.ctx.get("simulator")
        mode_label = self.ctx.get("mode_label", "LIVE REAL TRADING (Revolut X - 0.00% Maker Fee)")
        price_tracker: Dict[str, Decimal] = self.ctx.get("price_tracker", {})

        wallet_lines = []
        if simulator and hasattr(simulator, "get_wallet_balances"):
            try:
                bals = simulator.get_wallet_balances()
                eur_val = bals.get("EUR") or bals.get("eur")
                if eur_val is not None and ledger:
                    ledger.cash = Decimal(str(eur_val))
                for c, amt in sorted(bals.items()):
                    if amt > Decimal("0.0"):
                        wallet_lines.append(f"  • *{c}*: `{amt:.4f}`")
            except Exception:
                pass

        cash_eur = Decimal("0.0")
        if ledger:
            cash_eur = Decimal(str(ledger.cash))

        open_pos_lines = []
        crypto_equity = Decimal("0.0")
        if ledger and hasattr(ledger, "positions"):
            for sym, pos in ledger.positions.items():
                if pos.quantity > Decimal("0.0"):
                    disp = to_display_symbol(sym)
                    px = price_tracker.get(sym) or price_tracker.get(disp) or pos.last_price or Decimal("0.0")
                    val = pos.quantity * px
                    crypto_equity += val
                    open_pos_lines.append(
                        f"  • *{disp}*: `{pos.quantity:.5f}` (~€{val:.2f}) @ entry `€{pos.average_entry_price:.2f}`"
                    )

        total_equity = cash_eur + crypto_equity
        net_profit = Decimal("0.00")
        if ledger:
            realized = getattr(ledger, "realized_pnl", Decimal("0.0"))
            unrealized = Decimal("0.0")
            if hasattr(ledger, "unrealized_pnl"):
                try:
                    unrealized = ledger.unrealized_pnl(price_tracker)
                except Exception:
                    unrealized = Decimal("0.0")
            net_profit = realized + unrealized

        sign = "+" if net_profit >= Decimal("0.0") else ""
        net_profit_str = f"{sign}€{net_profit:,.2f}"

        status_emoji = "🟢" if "LIVE" in mode_label.upper() else "🟡"
        wallet_str = "\n".join(wallet_lines) if wallet_lines else "  • Nessun saldo rilevato o in caricamento"
        pos_str = "\n".join(open_pos_lines) if open_pos_lines else "  • Nessuna posizione aperta al momento"

        return (
            f"{status_emoji} *STATO & SALDO LIVE*\n"
            f"*Modalità*: `{mode_label}`\n"
            f"*Commissioni*: `0.00% Maker (Revolut X)`\n\n"
            f"💰 *Capitale Totale*: `€{total_equity:,.2f}`\n"
            f"💵 *Cash Disponibile*: `€{cash_eur:,.2f}`\n"
            f"📈 *Guadagni Netti*: `{net_profit_str}`\n\n"
            f"🏦 *Saldi Revolut X rilevati*:\n{wallet_str}\n\n"
            f"📊 *Posizioni Aperte*:\n{pos_str}\n\n"
            f"_Aggiornato al: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}_"
        )

    def _format_prices(self, is_admin: bool = False) -> str:
        price_tracker: Dict[str, Decimal] = self.ctx.get("price_tracker", {})
        strategies: Dict[str, Any] = self.ctx.get("strategies", {})
        pair_list: List[str] = self.ctx.get("pair_list", ["BTC/EUR", "SOL/EUR", "ETH/EUR"])
        ledger = self.ctx.get("ledger")
        equity = float(ledger.total_equity(price_tracker)) if ledger else 150.0
        realized = float(ledger.realized_pnl) if ledger else 0.0
        unrealized = float(ledger.unrealized_pnl(price_tracker)) if ledger else 0.0
        net_pnl = realized + unrealized

        if not is_admin or not self._latest_eval_summaries:
            lines = ["📈 *MERCATO & SEGNALI LIVE*"]
            for p in pair_list:
                disp = to_display_symbol(p)
                px = price_tracker.get(p) or price_tracker.get(disp)
                px_str = f"€{px:,.2f}" if px else "In attesa ticker..."

                strat = strategies.get(p) or strategies.get(disp)
                state_str = "MONITORING (Breakout 4h)"
                sl_str = ""
                if strat and hasattr(strat, "position_state"):
                    pst = strat.position_state
                    state_str = f"`{pst.state.value}`"
                    if pst.state.value != "IDLE" and pst.current_stop:
                        sl_str = f" | SL: `€{pst.current_stop:,.2f}`"

                lines.append(f"• *{disp}*: `{px_str}` | {state_str}{sl_str}")

            lines.append("\n_Timeframe Strategia: 4h Trend Breakout_")
            return "\n".join(lines)

        # Admin Detailed Dynamic Ranking of Top 3 Setups
        return format_admin_top3_report(
            eval_summaries=self._latest_eval_summaries,
            equity=equity,
            net_pnl=net_pnl,
        )

    def _format_trades(self) -> str:
        trades_csv = Path(self.ctx.get("trades_csv", PROJECT_ROOT / "data" / "trades.csv"))
        if not trades_csv.exists() or trades_csv.stat().st_size <= 10:
            return (
                "📜 *STORICO TRADE*\n\n"
                "Nessun ordine eseguito finora.\n"
                "Il bot sta analizzando i grafici a 4h ed entrerà automaticamente al verificarsi del primo segnale con commissione Maker 0.00%."
            )

        rows: List[Dict[str, str]] = []
        try:
            with open(trades_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception as ex:
            return f"Errore lettura trade: {ex}"

        if not rows:
            return "📜 *STORICO TRADE*: In attesa del primo trade."

        recent = rows[-5:]
        lines = [f"📜 *ULTIMI {len(recent)} TRADE ESEGUITI*:\n"]
        for t in reversed(recent):
            side = t.get("side", "").upper()
            badge = "🟢 BUY" if side == "BUY" else "🔴 SELL"
            sym = t.get("symbol", "")
            px = t.get("fill_price") or t.get("price", "0")
            qty = t.get("quantity", "0")
            ts = t.get("timestamp", "")[:19]
            pnl = t.get("realized_pnl", "")
            pnl_info = f" | PnL: `€{pnl}`" if pnl and pnl != "0.0" and pnl != "0" else ""
            lines.append(f"{badge} *{sym}* @ `€{px}` (q: `{qty}`){pnl_info}\n  _{ts}_")

        return "\n".join(lines)

    def _format_sync(self) -> str:
        simulator = self.ctx.get("simulator")
        ledger = self.ctx.get("ledger")
        if not simulator or not hasattr(simulator, "get_wallet_balances"):
            return "🔄 *SYNC REVOLUT X*: Il simulator non è attualmente connesso all'API Revolut X Live."

        try:
            bals = simulator.get_wallet_balances()
            eur_val = bals.get("EUR", Decimal("0.0"))
            if ledger:
                ledger.cash = Decimal(str(eur_val))
            lines = ["🔄 *SINCRONIZZAZIONE REVOLUT X IN TEMPO REALE*", ""]
            lines.append(f"💶 *Euro Cash (EUR)*: `€{eur_val:,.2f}`")
            lines.append("🪙 *Saldi Crypto Rilevati*:")
            has_crypto = False
            for coin, val in sorted(bals.items()):
                if coin != "EUR" and val > Decimal("0.0"):
                    lines.append(f"  • *{coin}*: `{val:.6f}`")
                    has_crypto = True
            if not has_crypto:
                lines.append("  • _Nessuna crypto attualmente in portafoglio_")

            lines.append(f"\n_Sync completato con successo al millisecondo._")
            return "\n".join(lines)
        except Exception as ex:
            return f"⚠️ Errore durante la sincronizzazione Revolut X: {ex}"

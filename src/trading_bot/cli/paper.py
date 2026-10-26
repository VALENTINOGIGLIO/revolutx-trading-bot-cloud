"""Paper Trading Engine & Terminal CLI Command.

Runs real-time paper trading simulation driven by public exchange ticker/bar data,
tracking virtual balance starting at 10.00 EUR with dual execution parity to
the historical backtester.

Supports:
- Live ticker polling
- Virtual ledger bookkeeping
- Dynamic AMVTrendBreak strategy evaluation
- Notification telemetry dispatch across Console, JSONL log, and Telegram
- Configurable --iterations count for deterministic testing
- Graceful SIGINT / SIGTERM shutdown with summary reporting
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from trading_bot.config import (
    DEFAULT_TIMEFRAME,
    DEFAULT_TRADES_CSV_PATH,
    PROJECT_ROOT,
    normalize_symbol,
    to_display_symbol,
)
from trading_bot.data.cache import MarketDataService
from trading_bot.data.models import Candle, Ticker
from trading_bot.execution.bybit_live import BybitAPIError, BybitLiveClient, BybitLiveExecutor
from trading_bot.simulation.revolutx_live import RevolutXLiveExecutor
from trading_bot.notifications.channels.console import ConsoleChannel
from trading_bot.notifications.channels.csv_log import CsvTradeLogChannel, CsvTradeLogger
from trading_bot.notifications.channels.json_log import JsonLogChannel
from trading_bot.notifications.channels.telegram import TelegramChannel
from trading_bot.notifications.service import NotificationService
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import (
    EXCHANGE_MIN_NOTIONAL,
    ExecutionSimulator,
)
from trading_bot.strategy.amv_strategy import AMVTrendBreakStrategy, PositionState
from trading_bot.strategy.compounding import CompoundingPositionSizer, RiskGuard
from trading_bot.strategy.indicators import compute_indicators_from_candles
from trading_bot.notifications.admin_commands import (
    ADMIN_CHAT_ID,
    AdminCommandQueue,
)

logger = logging.getLogger("paper_cli")

EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CLI_USAGE_ERROR = 2
EXIT_DATA_FETCH_ERROR = 3
EXIT_INSUFFICIENT_FUNDS_ERROR = 4
EXIT_CONFIG_ERROR = 5


def load_dotenv(dotenv_path: Optional[Union[str, Path]] = None) -> None:
    """Load key-value pairs from .env into os.environ if not already set."""
    candidates = []
    if dotenv_path:
        candidates.append(Path(dotenv_path))
    candidates.extend([
        Path(".env"),
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
    ])
    for cand in candidates:
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception as ex:
                logger.warning(f"Failed to read .env from {cand}: {ex}")



def setup_notifications(
    channel_name: str,
    json_path: str = "logs/notifications.jsonl",
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    trades_csv_path: Optional[Union[str, Path]] = "data/trades.csv",
) -> NotificationService:
    """Initialize NotificationService with selected channels."""
    channels = []
    ch = channel_name.lower().strip()

    if ch in ("stdout", "console", "all"):
        channels.append(ConsoleChannel(enabled=True))

    if ch in ("json", "all"):
        channels.append(JsonLogChannel(log_path=json_path, enabled=True))

    if ch in ("telegram", "all") and telegram_token and telegram_chat_id:
        channels.append(
            TelegramChannel(
                bot_token=telegram_token,
                chat_id=telegram_chat_id,
                enabled=True,
            )
        )

    # Register CsvTradeLogChannel pointing to designated CSV path
    if trades_csv_path or ch in ("csv", "all"):
        csv_dest = trades_csv_path or DEFAULT_TRADES_CSV_PATH
        channels.append(CsvTradeLogChannel(csv_path=csv_dest, enabled=True))

    return NotificationService(channels=channels)


def execute_pending_admin_commands(
    admin_queue: AdminCommandQueue,
    simulator: Any,
    ledger: Any,
    strategies_by_pair: Dict[str, Any],
    price_tracker: Dict[str, Decimal],
    step_size_map: Dict[str, Decimal],
    tg_listener: Optional[Any],
    notif_service: Any,
    currency: str = "EUR",
) -> None:
    """Check and process pending Admin manual override commands (forced buy/sell)."""
    pending = admin_queue.get_pending()
    if not pending:
        return

    now_utc = datetime.now(timezone.utc)
    now_ms = int(now_utc.timestamp() * 1000)
    from trading_bot.strategy.amv_strategy import OrderSide, OrderSignal, OrderType, PositionState

    for cmd in pending:
        cmd_id = cmd.get("id")
        action = cmd.get("action", "BUY").upper()
        sym = cmd.get("symbol", "BTC/EUR").upper()
        p_disp = to_display_symbol(sym)
        p_norm = normalize_symbol(sym)

        strategy = strategies_by_pair.get(p_disp) or strategies_by_pair.get(p_norm)
        current_px = price_tracker.get(p_norm) or price_tracker.get(p_disp) or Decimal("0.0")

        if action == "BUY":
            if current_px <= Decimal("0.0"):
                if hasattr(simulator, "get_ticker_price"):
                    try:
                        live_p = simulator.get_ticker_price(p_norm)
                        if live_p:
                            current_px = Decimal(str(live_p))
                            price_tracker[p_norm] = current_px
                    except Exception:
                        pass

            if current_px <= Decimal("0.0"):
                admin_queue.update_status(cmd_id, "FAILED", {"error": f"Prezzo non disponibile per {sym}"})
                if tg_listener:
                    tg_listener.api_client.send_message(
                        ADMIN_CHAT_ID,
                        f"❌ Impossibile eseguire avvio forzato su {p_disp}: prezzo non disponibile."
                    )
                continue

            atr_dec = Decimal("0.0")
            if strategy and hasattr(strategy, "last_atr") and strategy.last_atr:
                atr_dec = Decimal(str(strategy.last_atr))
            if atr_dec <= Decimal("0.0"):
                atr_dec = current_px * Decimal("0.018")

            atr_stop_mult = getattr(strategy, "atr_stop_mult", Decimal("2.2"))
            initial_stop = current_px - (atr_stop_mult * atr_dec)
            tp_mult = getattr(strategy, "tp_multiplier", Decimal("1.50"))
            tp1_target = current_px + (tp_mult * (current_px - initial_stop))

            equity = ledger.total_equity(price_tracker)
            step_dec = step_size_map.get(p_norm, Decimal("0.001"))
            min_notional = getattr(strategy, "min_notional", Decimal("1.00")) if strategy else Decimal("1.00")

            if strategy and hasattr(strategy, "sizer"):
                sizing_res = strategy.sizer.calculate_position_size(
                    equity=equity,
                    entry_price=current_px,
                    stop_loss=initial_stop,
                    risk_multiplier=1.0,
                    min_notional=min_notional,
                    step_size=step_dec,
                )
                target_qty = sizing_res.quantity
                target_notional = sizing_res.notional
            else:
                target_notional = min(ledger.cash * Decimal("0.95"), equity)
                target_qty = (target_notional / current_px).quantize(step_dec)

            if target_qty <= Decimal("0.0"):
                admin_queue.update_status(cmd_id, "FAILED", {"error": "Quantità calcolata insufficiente"})
                if tg_listener:
                    tg_listener.api_client.send_message(
                        ADMIN_CHAT_ID,
                        f"❌ Impossibile eseguire avvio forzato su {p_disp}: fondi insufficienti (Cash: €{ledger.cash:.2f})."
                    )
                continue

            signal = OrderSignal(
                symbol=p_disp,
                side=OrderSide.BUY,
                target_quantity=target_qty,
                target_notional=target_notional,
                order_type=OrderType.MARKET,
                stop_loss=initial_stop,
                take_profit=tp1_target,
                price=current_px,
                reason="MANUAL_ADMIN_FORCED_BUY",
                timestamp_ms=now_ms,
                metadata={"atr": float(atr_dec), "forced_by": ADMIN_CHAT_ID},
            )

            simulator.step_size = step_dec
            exec_res = simulator.execute(
                order=signal,
                current_price=current_px,
                ledger=ledger,
                timestamp_ms=now_ms,
                notification_service=notif_service,
            )

            if exec_res.success:
                exec_qty = getattr(exec_res, "quantity", getattr(exec_res, "executed_quantity", Decimal("0")))
                if strategy is not None:
                    pos_state = PositionState(
                        symbol=p_disp,
                        entry_price=exec_res.executed_price,
                        quantity=exec_qty,
                        initial_stop=initial_stop,
                        current_stop=initial_stop,
                        take_profit_1=tp1_target,
                        highest_price=exec_res.executed_price,
                        bars_in_trade=0,
                        tp1_hit=False,
                        entry_timestamp=now_utc,
                        entry_reason="MANUAL_ADMIN_FORCED_BUY",
                    )
                    strategy._positions[p_disp] = pos_state
                    strategy._positions[p_norm] = pos_state

                admin_queue.update_status(
                    cmd_id,
                    "COMPLETED",
                    {
                        "executed_price": float(exec_res.executed_price),
                        "executed_qty": float(exec_qty),
                        "stop_loss": float(initial_stop),
                    },
                )

                chand_mult = getattr(strategy, "chandelier_atr_mult", Decimal("5.5"))
                chand_stop = exec_res.executed_price - (chand_mult * atr_dec)

                if tg_listener:
                    confirm_msg = (
                        "✅ *ACQUISTO FORZATO ESEGUITO SU REVOLUT X!*\n"
                        f"Moneta: *{p_disp}*\n"
                        f"Prezzo Eseguito: `€{exec_res.executed_price:,.2f}`\n"
                        f"Quantità: `{exec_res.executed_quantity:.6f}`\n"
                        f"Controvalore: `€{exec_res.executed_quantity * exec_res.executed_price:,.2f}`\n\n"
                        f"🛡 *Protezioni Attive da Subito:*\n"
                        f"• Hard Stop Loss (2.2x ATR): `€{initial_stop:,.2f}`\n"
                        f"• Chandelier Trailing (5.5x ATR): `€{chand_stop:,.2f}`\n\n"
                        "La posizione è ora monitorata e protetta in automatico dal trading bot!"
                    )
                    tg_listener.api_client.send_message(
                        ADMIN_CHAT_ID,
                        confirm_msg,
                        reply_markup=tg_listener._get_inline_keyboard(is_admin=True),
                    )
                print(
                    f"[{now_utc.strftime('%H:%M:%S')}] [ADMIN FORCE BUY] {p_disp} @ {exec_res.executed_price} FILLED! Stop: {initial_stop:.2f}"
                )
            else:
                admin_queue.update_status(cmd_id, "FAILED", {"error": exec_res.error_message})
                if tg_listener:
                    tg_listener.api_client.send_message(
                        ADMIN_CHAT_ID,
                        f"❌ Ordine forzato su {p_disp} rifiutato da Revolut X: {exec_res.error_message}"
                    )

        elif action == "SELL":
            pos = ledger.get_position(p_norm) or ledger.get_position(p_disp)
            qty_to_sell = pos.quantity if pos else Decimal("0.0")

            if qty_to_sell <= Decimal("0.0"):
                admin_queue.update_status(cmd_id, "FAILED", {"error": "Nessuna posizione aperta da chiudere"})
                if tg_listener:
                    tg_listener.api_client.send_message(
                        ADMIN_CHAT_ID,
                        f"⚠️ Nessuna posizione aperta trovata su {p_disp}."
                    )
                continue

            sell_sig = OrderSignal(
                symbol=p_disp,
                side=OrderSide.SELL,
                target_quantity=qty_to_sell,
                target_notional=qty_to_sell * current_px,
                order_type=OrderType.MARKET,
                price=current_px,
                reason="MANUAL_ADMIN_FORCED_SELL",
                timestamp_ms=now_ms,
            )

            step_dec = step_size_map.get(p_norm, Decimal("0.001"))
            simulator.step_size = step_dec
            exec_res = simulator.execute(
                order=sell_sig,
                current_price=current_px,
                ledger=ledger,
                timestamp_ms=now_ms,
                notification_service=notif_service,
            )

            if exec_res.success:
                if strategy is not None:
                    strategy.notify_position_closed(
                        symbol=p_disp,
                        exit_price=exec_res.executed_price,
                        current_equity=ledger.total_equity(price_tracker),
                        timestamp=now_utc,
                    )
                    strategy._positions.pop(p_disp, None)
                    strategy._positions.pop(p_norm, None)

                exec_qty = getattr(exec_res, "quantity", getattr(exec_res, "executed_quantity", Decimal("0")))
                admin_queue.update_status(
                    cmd_id,
                    "COMPLETED",
                    {
                        "executed_price": float(exec_res.executed_price),
                        "executed_qty": float(exec_qty),
                    },
                )

                if tg_listener:
                    confirm_msg = (
                        "🛑 *POSIZIONE CHIUSA MANUALMENTE SU REVOLUT X!*\n"
                        f"Moneta: *{p_disp}*\n"
                        f"Prezzo Chiusura: `€{exec_res.executed_price:,.2f}`\n"
                        f"Quantità Venduta: `{exec_qty:.6f}`\n"
                        f"Incasso Netto: `€{exec_qty * exec_res.executed_price:,.2f}`\n"
                        f"Nuovo Saldo Cash: `€{ledger.cash:,.2f}`"
                    )
                    tg_listener.api_client.send_message(
                        ADMIN_CHAT_ID,
                        confirm_msg,
                        reply_markup=tg_listener._get_inline_keyboard(is_admin=True),
                    )
                print(
                    f"[{now_utc.strftime('%H:%M:%S')}] [ADMIN FORCE SELL] {p_disp} @ {exec_res.executed_price} CLOSED!"
                )
            else:
                admin_queue.update_status(cmd_id, "FAILED", {"error": exec_res.error_message})
                if tg_listener:
                    tg_listener.api_client.send_message(
                        ADMIN_CHAT_ID,
                        f"❌ Chiusura manuale su {p_disp} rifiutata: {exec_res.error_message}"
                    )


def run_paper(
    pair: str = "BTC/EUR",
    balance: float = 10.0,
    currency: str = "EUR",
    timeframe: str = "1h",
    exchange: str = "binance",
    iterations: int = 0,
    interval_sec: float = 5.0,
    notify_channel: str = "stdout",
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    dry_run: bool = False,
    market_data_service: Optional[MarketDataService] = None,
    trades_csv: Optional[Union[str, Path]] = "data/trades.csv",
    dashboard: bool = False,
    dashboard_port: int = 8000,
    dashboard_host: str = "127.0.0.1",
    live: bool = False,
) -> Tuple[VirtualLedger, NotificationService]:
    """Execute paper or live trading engine loop."""
    load_dotenv()

    # Determine if live execution is requested
    env_mode = os.getenv("TRADING_MODE", "").strip().upper()
    is_live = bool(live or (env_mode == "LIVE"))

    if balance <= 0.0:
        raise ValueError(f"Initial balance must be > 0, got {balance}")

    pair_list = [p.strip() for p in pair.split(",") if p.strip()]
    if not pair_list:
        pair_list = ["DOGE/EUR"]

    primary_pair = pair_list[0]
    norm_primary = normalize_symbol(primary_pair)
    disp_primary = to_display_symbol(primary_pair)
    service = market_data_service or MarketDataService()

    # Set up dashboard server if enabled
    dash_server = None
    if dashboard:
        from trading_bot.dashboard.server import DashboardServer
        from trading_bot.dashboard.state import DashboardState

        primary_disp = to_display_symbol(primary_pair)
        dash_state = DashboardState(
            symbol=primary_disp,
            initial_balance=balance,
            status="running",
        )
        if trades_csv:
            dash_state.load_from_csv(trades_csv)
        dash_state.load_from_market_db(symbol=norm_primary, interval=timeframe, limit=500)

        dash_server = DashboardServer(
            host=dashboard_host,
            port=dashboard_port,
            state=dash_state,
            symbol=primary_disp,
        )
        actual_port = dash_server.start_background()
        print(f"[INFO] Dashboard web server running at http://{dashboard_host}:{actual_port}")

    # Set up notifications
    notif_service = setup_notifications(
        channel_name=notify_channel,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        trades_csv_path=trades_csv,
    )

    # Set up ledger and simulator
    initial_cash_dec = Decimal(str(balance))
    ledger = VirtualLedger(initial_cash=initial_cash_dec, quote_currency=currency)
    min_notional_dec = EXCHANGE_MIN_NOTIONAL.get(exchange.lower(), Decimal("1.00"))

    def resolve_coin_step_size(sym_norm: str) -> Decimal:
        norm = sym_norm.upper().replace("/", "").replace("-", "")
        if "FLOKI" in norm:
            return Decimal("1.0")
        elif "BONK" in norm:
            return Decimal("1.0")
        elif "DOGE" in norm:
            return Decimal("1.0")
        elif "BTC" in norm:
            return Decimal("0.00001")
        elif "ETH" in norm:
            return Decimal("0.0001")
        elif "SOL" in norm:
            return Decimal("0.001")
        elif "XRP" in norm:
            return Decimal("0.1")
        return Decimal("0.0001")

    # Limit Maker execution model for v2 (0.02% fee, 0.01% slippage)
    simulator: Any = ExecutionSimulator(
        fee_rate=Decimal("0.0002"),
        slippage=Decimal("0.0001"),
        min_notional=min_notional_dec,
        step_size=Decimal("0.0001"),
        exchange=exchange,
        notification_service=notif_service,
    )

    mode_label = "DRY RUN" if dry_run else "ACTIVE PAPER TRADING"

    # Configure Exchange Execution Engine
    if exchange.lower() in ("revolutx", "revolut_x", "revx"):
        revx_key = os.getenv("REVOLUTX_API_KEY", "").strip()
        revx_priv_path = os.getenv("REVOLUTX_PRIVATE_KEY_PATH", "keys/revolutx_private.pem").strip()
        priv_file = Path(revx_priv_path)
        if not priv_file.is_absolute():
            priv_file = PROJECT_ROOT / priv_file

        if is_live:
            if revx_key and priv_file.exists():
                print("[INFO] Initializing Revolut X Live Execution Engine (0.00% Maker Fee)...")
                try:
                    live_exec = RevolutXLiveExecutor(
                        api_key=revx_key,
                        private_key_path=priv_file,
                        live=True,
                        prefer_maker=True,
                        fee_rate=Decimal("0.0000"),
                        slippage=Decimal("0.0000"),
                        min_notional=min_notional_dec,
                        notification_service=notif_service,
                    )
                    real_bals = live_exec.get_wallet_balances()
                    real_eur = real_bals.get("EUR", Decimal("0.0"))
                    if real_eur > Decimal("0.0"):
                        ledger.cash = real_eur
                        print(f"[LIVE TRADING ATTIVO] Connesso a Revolut X API. Saldo reale disponibile: {real_eur:.2f} EUR (0.00% Commissioni Maker)")
                    else:
                        print(f"[LIVE TRADING ATTIVO] Connesso a Revolut X API. Saldo EUR non rilevato o 0.0; uso budget iniziale {ledger.cash:.2f} EUR")

                    for p_item in pair_list:
                        p_norm = normalize_symbol(p_item)
                        try:
                            p_rules = live_exec.get_instrument_rules(p_norm)
                            b_coin = p_rules.get("baseCoin", "")
                            b_qty = real_bals.get(b_coin, Decimal("0.0"))
                            if b_qty > Decimal("0.0"):
                                pos = ledger.get_position(p_norm)
                                pos.quantity = b_qty
                                print(f"[LIVE TRADING] Rilevata posizione reale su {b_coin}: {b_qty}")
                        except Exception:
                            pass

                    simulator = live_exec
                    mode_label = "LIVE REAL TRADING (REVOLUT X - 0.00% MAKER FEE)"
                except Exception as live_err:
                    print(f"[ATTENZIONE] Inizializzazione Revolut X Live fallita: {live_err}. Fallback di sicurezza a Paper Trading.")
                    simulator = RevolutXLiveExecutor(
                        live=False,
                        fee_rate=Decimal("0.0000"),
                        slippage=Decimal("0.0000"),
                        min_notional=min_notional_dec,
                        notification_service=notif_service,
                    )
                    mode_label = "PAPER TRADING (Revolut X Fallback)"
            else:
                print(
                    "[INFRASTRUTTURA LIVE PRONTA] Modalità Live abilitata con Revolut X ma REVOLUTX_API_KEY non ancora configurata in .env.\n"
                    "                             Incolla la chiave generata su Revolut X nel file .env per fare trading reale.\n"
                    "                             Opero in simulazione PAPER TRADING con condizioni Revolut X (0.00% Maker fee, 1.00€ min notional)."
                )
                simulator = RevolutXLiveExecutor(
                    live=False,
                    fee_rate=Decimal("0.0000"),
                    slippage=Decimal("0.0000"),
                    min_notional=min_notional_dec,
                    notification_service=notif_service,
                )
                mode_label = "ACTIVE PAPER TRADING (Revolut X 0.00% Maker Simulation)"
        else:
            simulator = RevolutXLiveExecutor(
                live=False,
                fee_rate=Decimal("0.0000"),
                slippage=Decimal("0.0000"),
                min_notional=min_notional_dec,
                notification_service=notif_service,
            )
            mode_label = "ACTIVE PAPER TRADING (Revolut X 0.00% Maker Simulation)"
    elif is_live:
        bybit_key = os.getenv("BYBIT_API_KEY", "").strip()
        bybit_secret = os.getenv("BYBIT_API_SECRET", "").strip()

        if bybit_key and bybit_secret:
            print("[INFO] Initializing Bybit Live Execution Engine (Spot V5)...")
            try:
                live_exec = BybitLiveExecutor(
                    api_key=bybit_key,
                    api_secret=bybit_secret,
                    exchange="bybit",
                    fee_rate=Decimal("0.0002"),
                    slippage=Decimal("0.0001"),
                    notification_service=notif_service,
                )
                real_bals = live_exec.get_wallet_balances()
                real_eur = real_bals.get("EUR", Decimal("0.0"))
                if real_eur > Decimal("0.0"):
                    ledger.cash = real_eur
                    print(f"[LIVE TRADING ATTIVO] Connesso a Bybit Spot V5. Saldo reale disponibile: {real_eur:.2f} EUR")
                else:
                    print(f"[LIVE TRADING ATTIVO] Connesso a Bybit Spot V5. Saldo EUR non rilevato o 0.0; uso budget iniziale {ledger.cash:.2f} EUR")

                for p_item in pair_list:
                    p_norm = normalize_symbol(p_item)
                    try:
                        p_rules = live_exec.get_instrument_rules(p_norm)
                        b_coin = p_rules.get("baseCoin", "")
                        b_qty = real_bals.get(b_coin, Decimal("0.0"))
                        if b_qty > Decimal("0.0"):
                            pos = ledger.get_position(p_norm)
                            pos.quantity = b_qty
                            print(f"[LIVE TRADING] Rilevata posizione reale su {b_coin}: {b_qty}")
                    except Exception:
                        pass

                simulator = live_exec
                mode_label = "LIVE REAL TRADING (BYBIT SPOT V5)"
            except Exception as live_err:
                print(f"[ATTENZIONE] Inizializzazione Bybit Live fallita: {live_err}. Fallback di sicurezza a Paper Trading.")
                mode_label = "PAPER TRADING (LIVE Fallback)"
        else:
            print(
                "[INFRASTRUTTURA LIVE PRONTA] Modalità Live abilitata ma BYBIT_API_KEY e BYBIT_API_SECRET non ancora configurati.\n"
                "                             Inseriscili nel file .env per inviare ordini con denaro reale. Opero in PAPER TRADING di sicurezza."
            )
            mode_label = "ACTIVE PAPER TRADING (Awaiting Bybit API Keys)"


    # Dedicated sizers and strategies per pair with v2 anti-whipsaw parameters
    strategies_by_pair: Dict[str, AMVTrendBreakStrategy] = {}
    is_revx_or_low_notional = (
        exchange.lower() in ("revolutx", "revolut_x", "revx")
        or min_notional_dec <= Decimal("1.00")
    )
    p1_threshold = Decimal("10.00") if is_revx_or_low_notional else Decimal("100.00")

    for p_item in pair_list:
        p_norm = normalize_symbol(p_item)
        p_disp = to_display_symbol(p_item)
        coin_step = resolve_coin_step_size(p_norm)

        pair_sizer = CompoundingPositionSizer(
            phase_1_threshold=p1_threshold,
            min_notional=min_notional_dec,
            phase_1_allocation_pct=0.3333,
            phase_2_risk_pct=0.04,
            phase_2_max_allocation_pct=0.3333,
            default_step_size=coin_step,
            atr_multiplier=Decimal("2.2"),
        )
        pair_strat = AMVTrendBreakStrategy(
            default_symbol=p_disp,
            exchange_min_notional=min_notional_dec,
            step_size=coin_step,
            position_sizer=pair_sizer,
            atr_stop_mult=Decimal("2.2"),
            chandelier_atr_mult=Decimal("5.5"),
            breakout_period=20,
        )
        strategies_by_pair[p_norm] = pair_strat
        strategies_by_pair[p_disp] = pair_strat

    # VirtualLedger persistence: restore cash and open positions from CSV if exists
    if trades_csv and Path(trades_csv).exists():
        try:
            trade_logger = CsvTradeLogger(csv_path=trades_csv)
            past_trades = trade_logger.read_trades()
            if past_trades:
                for p_clean in list(ledger.positions.keys()):
                    ledger.positions[p_clean].quantity = Decimal("0.0")
                    ledger.positions[p_clean].cost_basis = Decimal("0.0")
                for t in past_trades:
                    t_sym = normalize_symbol(t.get("symbol", ""))
                    t_act = str(t.get("action", "")).upper()
                    try:
                        t_qty = Decimal(str(t.get("size", "0.0")))
                        t_px = Decimal(str(t.get("price", "0.0")))
                    except (ValueError, TypeError):
                        continue
                    if t_sym and t_qty > Decimal("0.0"):
                        pos = ledger.get_position(t_sym)
                        if t_act == "BUY":
                            new_q = pos.quantity + t_qty
                            new_cost = pos.cost_basis + (t_qty * t_px)
                            pos.average_entry_price = new_cost / new_q
                            pos.quantity = new_q
                            pos.cost_basis = new_cost
                            pos.last_price = t_px
                        elif t_act == "SELL":
                            rem_q = max(Decimal("0.0"), pos.quantity - t_qty)
                            pos.quantity = rem_q
                            if rem_q == Decimal("0.0"):
                                pos.cost_basis = Decimal("0.0")
                                pos.average_entry_price = Decimal("0.0")

                # Restore latest cash balance if available
                last_trade = past_trades[-1]
                cash_val_raw = last_trade.get("cash_balance")
                if cash_val_raw:
                    try:
                        c_val = Decimal(str(cash_val_raw))
                        if c_val > Decimal("0.0"):
                            ledger.cash = c_val
                    except Exception:
                        pass

                # Synchronize any open positions to respective strategy instances
                for s_norm, pos in ledger.positions.items():
                    if pos.quantity > Decimal("0.0") and s_norm in strategies_by_pair:
                        strat = strategies_by_pair[s_norm]
                        stop_dist = pos.average_entry_price * Decimal("0.022")
                        restored_state = PositionState(
                            symbol=to_display_symbol(s_norm),
                            entry_price=pos.average_entry_price,
                            quantity=pos.quantity,
                            initial_stop=pos.average_entry_price - stop_dist,
                            current_stop=pos.average_entry_price - stop_dist,
                            take_profit_1=pos.average_entry_price + (stop_dist * Decimal("1.5")),
                            highest_price=pos.last_price or pos.average_entry_price,
                            bars_in_trade=1,
                            entry_timestamp=datetime.now(timezone.utc),
                            entry_reason="RESTORED_FROM_CSV",
                        )
                        strat._positions[s_norm] = restored_state
                        strat._positions[to_display_symbol(s_norm)] = restored_state
                        strat._positions[s_norm.replace("-", "").replace("/", "")] = restored_state
                logger.info(f"Reconstructed ledger from {len(past_trades)} CSV trades. Restored cash: {ledger.cash:.2f}")
        except Exception as ex:
            logger.warning(f"Error restoring ledger from {trades_csv}: {ex}")

    # Live mode: Always synchronize ledger.cash AND open positions directly with real exchange wallet
    if hasattr(simulator, "get_wallet_balances"):
        try:
            live_bals = simulator.get_wallet_balances()
            real_live_eur = live_bals.get("EUR") or live_bals.get("eur")
            if real_live_eur is not None and Decimal(str(real_live_eur)) >= Decimal("0.0"):
                ledger.cash = Decimal(str(real_live_eur))
                logger.info(f"Live balance synchronized with Revolut X: {ledger.cash} EUR")
            for p_item in pair_list:
                p_norm = normalize_symbol(p_item)
                p_disp = to_display_symbol(p_norm)
                b_coin = p_disp.split("/")[0].upper()
                b_qty = Decimal(str(live_bals.get(b_coin, Decimal("0.0"))))
                pos = ledger.get_position(p_norm)
                pos.quantity = b_qty
                if b_qty > Decimal("0.00001") and p_norm in strategies_by_pair:
                    strat = strategies_by_pair[p_norm]
                    entry_px = pos.average_entry_price if pos.average_entry_price > Decimal("0") else service.get_ticker(p_norm).price
                    pos.average_entry_price = entry_px
                    pos.cost_basis = b_qty * entry_px
                    stop_dist = entry_px * Decimal("0.022")
                    live_pos_state = PositionState(
                        symbol=p_disp,
                        entry_price=entry_px,
                        quantity=b_qty,
                        initial_stop=entry_px - stop_dist,
                        current_stop=entry_px - stop_dist,
                        take_profit_1=entry_px + (stop_dist * Decimal("1.5")),
                        highest_price=max(entry_px, pos.last_price or entry_px),
                        bars_in_trade=1,
                        entry_timestamp=datetime.now(timezone.utc),
                        entry_reason="SYNCED_FROM_REVOLUTX_WALLET",
                    )
                    strat._positions[p_norm] = live_pos_state
                    strat._positions[p_disp] = live_pos_state
                    strat._positions[p_norm.replace("-", "").replace("/", "")] = live_pos_state
        except Exception as b_ex:
            logger.warning(f"Failed to sync initial balance with Revolut X: {b_ex}")

    # Signal handling for clean exit on Ctrl-C
    running = True

    def _sig_handler(signum: int, frame: Any) -> None:
        nonlocal running
        print("\n[INFO] Interrupt signal received. Shutting down gracefully...")
        running = False

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, AttributeError):
        pass  # In some testing environments, signals cannot be registered

    pairs_display_str = ", ".join(to_display_symbol(p) for p in pair_list)
    print("=" * 70)
    print("      AUTONOMOUS TRADING BOT - MULTI-PAIR PAPER TRADING ENGINE       ")
    print("=" * 70)
    print(f"Active Pairs ({len(pair_list)}): {pairs_display_str}")
    print(f"Exchange: {exchange.capitalize()} | Timeframe: {timeframe}")
    print(f"Initial Balance: {balance:.2f} {currency} | Min Notional: {min_notional_dec:.2f} {currency}")
    active_phase_str = "Phase 2 Growth (4.0% risk, 85% cap)" if Decimal(str(balance)) > p1_threshold else f"Phase 1 Micro-Cap (alloc 45%, threshold {p1_threshold:.2f} {currency})"
    print(f"Compounding Engine: {active_phase_str}")
    print(f"Mode: {mode_label}")
    print(f"Notifications: {notify_channel} | Interval: {interval_sec}s")
    if iterations > 0:
        print(f"Target Iterations: {iterations}")
    print("-" * 70)

    current_iter = 0
    price_tracker: Dict[str, Decimal] = {}
    last_evaluated_candle_ts: Dict[str, Any] = {}
    admin_cmd_queue = AdminCommandQueue()
    step_size_map: Dict[str, Decimal] = {p: resolve_coin_step_size(p) for p in pair_list}
    for extra_sym in ["BTC/EUR", "ETH/EUR", "SOL/EUR", "DOGE/EUR", "XRP/EUR", "BTC-EUR", "ETH-EUR", "SOL-EUR"]:
        step_size_map[extra_sym] = resolve_coin_step_size(extra_sym)

    # Initialize Interactive Telegram Bot Listener and Push Channel
    tg_listener = None
    tg_token_to_use = (telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    tg_pw_to_use = os.getenv("TELEGRAM_BOT_PASSWORD", "").strip()

    if tg_token_to_use:
        try:
            from trading_bot.notifications.telegram_bot import TelegramBotListener
            auth_file = PROJECT_ROOT / "data" / "telegram_auth_users.json"
            bot_context = {
                "ledger": ledger,
                "simulator": simulator,
                "mode_label": mode_label,
                "price_tracker": price_tracker,
                "strategies": strategies_by_pair,
                "pair_list": pair_list,
                "trades_csv": trades_csv,
            }
            tg_listener = TelegramBotListener(
                token=tg_token_to_use,
                password=tg_pw_to_use,
                auth_file=auth_file,
                context_provider=bot_context,
            )
            notif_service.register_channel(tg_listener.push_channel)
            print(f"[INFO] Telegram Interactive Bot (@GILLYTRADESBOT_bot) inizializzato (avvio listener post-fetch prezzi).")
        except Exception as tg_err:
            logger.warning(f"Impossibile inizializzare TelegramBotListener: {tg_err}")

    while running:
        current_iter += 1
        now_utc = datetime.now(timezone.utc)

        try:
            # Immediate check and execution of any queued admin manual commands
            try:
                execute_pending_admin_commands(
                    admin_queue=admin_cmd_queue,
                    simulator=simulator,
                    ledger=ledger,
                    strategies_by_pair=strategies_by_pair,
                    price_tracker=price_tracker,
                    step_size_map=step_size_map,
                    tg_listener=tg_listener,
                    notif_service=notif_service,
                    currency=currency,
                )
            except Exception as adm_ex:
                logger.warning(f"Error checking admin commands at cycle start: {adm_ex}")
            # Continuous synchronization with 100% of real Revolut X balance
            if hasattr(simulator, "get_wallet_balances"):
                try:
                    w_bals = simulator.get_wallet_balances()
                    eur_val = w_bals.get("EUR") or w_bals.get("eur")
                    if eur_val is not None:
                        real_cash = Decimal(str(eur_val))
                        if real_cash != ledger.cash:
                            logger.info(f"Revolut X balance updated: {ledger.cash} -> {real_cash} EUR")
                            ledger.cash = real_cash
                except Exception as sync_err:
                    logger.debug(f"Live balance sync check: {sync_err}")

            # Check auto-deposit detector on Revolut X live/sync cycles
            if hasattr(simulator, "check_auto_deposit") and (current_iter % 6 == 1):
                try:
                    dep_amount = simulator.check_auto_deposit(ledger=ledger)
                    if dep_amount:
                        print(
                            f"[{now_utc.strftime('%H:%M:%S')}] [AUTO-DEPOSIT DETECTOR] "
                            f"Accreditati +{dep_amount:.2f} EUR! Nuovo saldo cash: {ledger.cash:.2f} EUR"
                        )
                except Exception as dep_err:
                    logger.debug(f"Auto-deposit detection check: {dep_err}")

            eval_summaries: List[Dict[str, Any]] = []

            for p_item in pair_list:
                try:
                    p_norm = normalize_symbol(p_item)
                    p_disp = to_display_symbol(p_item)
                    step_dec = resolve_coin_step_size(p_norm)
                    strategy = strategies_by_pair.get(p_norm) or strategies_by_pair.get(p_disp)

                    # 1. Fetch current ticker
                    ticker = service.get_ticker(p_norm)
                    current_price = ticker.price
                    price_tracker[p_norm] = current_price
                    price_tracker[p_disp] = current_price

                    # 2. Fetch recent historical bars to feed strategy
                    end_req = now_utc
                    start_req = end_req - timedelta(days=45)
                    recent_candles = service.get_historical_candles(
                        symbol=p_norm,
                        interval=timeframe,
                        start_time=start_req,
                        end_time=end_req,
                    )

                    if recent_candles:
                        latest_candle = recent_candles[-1]
                        history = recent_candles[:-1]
                        eff_vol = latest_candle.volume
                        if history and eff_vol < history[-1].volume:
                            eff_vol = history[-1].volume
                        # Update current forming candle with live ticker price and persist to DB
                        latest_candle = latest_candle.model_copy(update={
                            "close": current_price,
                            "high": max(latest_candle.high, current_price),
                            "low": min(latest_candle.low, current_price),
                            "volume": eff_vol,
                        })
                        service.cache.save_candles([latest_candle])
                    else:
                        latest_candle = Candle(
                            timestamp=now_utc,
                            open=current_price,
                            high=current_price,
                            low=current_price,
                            close=current_price,
                            volume=Decimal("1.0"),
                            symbol=p_norm,
                            interval=timeframe,
                        )
                        history = []
                        service.cache.save_candles([latest_candle])

                    if dashboard and dash_server and p_item == primary_pair:
                        dash_server.state.add_candle(latest_candle)

                    # 3. Strategy signal evaluation - evaluate live when 8/8 bells align or stops hit
                    c_ts = getattr(latest_candle, "timestamp", None)
                    is_new_candle = (p_norm not in last_evaluated_candle_ts) or (last_evaluated_candle_ts[p_norm] != c_ts)
                    if is_new_candle:
                        last_evaluated_candle_ts[p_norm] = c_ts

                    signal_res = None
                    if strategy is not None:
                        simulator.step_size = step_dec
                        signal_res = strategy.on_candle(latest_candle, history, ledger)

                    order_status_str = "IDLE" if not is_new_candle else "EVAL_IDLE"
                    if signal_res is not None:
                        order_status_str = f"SIGNAL: {signal_res.side} {signal_res.target_quantity or signal_res.target_notional}"
                        if not dry_run:
                            simulator.step_size = step_dec
                            exec_res = simulator.execute(
                                order=signal_res,
                                current_price=current_price,
                                ledger=ledger,
                                timestamp_ms=int(now_utc.timestamp() * 1000),
                                notification_service=notif_service,
                            )
                            if exec_res.success:
                                order_status_str += f" -> FILLED @ {exec_res.executed_price:.2f}"
                                if dashboard and dash_server:
                                    dash_server.state.add_trade(exec_res.trade_record or exec_res)
                                if exec_res.side == "SELL" and strategy is not None:
                                    strategy.notify_position_closed(
                                        symbol=signal_res.symbol,
                                        exit_price=exec_res.executed_price,
                                        current_equity=ledger.total_equity(price_tracker),
                                        timestamp=now_utc,
                                    )
                            else:
                                order_status_str += f" -> REJECTED: {exec_res.error_message}"
                                if strategy is not None:
                                    if getattr(signal_res.side, "value", str(signal_res.side)).upper() == "BUY":
                                        strategy._positions.pop(signal_res.symbol, None)
                                        strategy._positions.pop(p_norm, None)
                                        strategy._positions.pop(p_disp, None)
                                    else:
                                        strategy.notify_position_rejected(signal_res.symbol)

                    pos = ledger.get_position(p_norm)
                    if signal_res is not None or current_iter % 12 == 1:
                        print(
                            f"[{now_utc.strftime('%H:%M:%S')}] [{p_disp:<8}] "
                            f"Px: {current_price:<9.2f} | Pos: {pos.quantity:.4f} | {order_status_str}"
                        )

                    # Extract 4h technical indicators and breakout metrics for Telegram report
                    candles_chain = list(history) + [latest_candle]
                    df_ind = compute_indicators_from_candles(candles_chain)

                    px_f = float(current_price)
                    if not df_ind.empty:
                        last_ind = df_ind.iloc[-1]
                        ema_200 = float(last_ind.get("ema_200", 0.0))
                        ema_9 = float(last_ind.get("ema_9", 0.0))
                        ema_21 = float(last_ind.get("ema_21", 0.0))
                        ema_50 = float(last_ind.get("ema_50", 0.0))
                        kc_upper = float(last_ind.get("kc_upper", 0.0))
                        adx = float(last_ind.get("adx_14", 0.0))
                        plus_di = float(last_ind.get("plus_di", 0.0))
                        minus_di = float(last_ind.get("minus_di", 0.0))
                        chop = float(last_ind.get("chop_14", 0.0))
                        rsi = float(last_ind.get("rsi_14", 0.0))
                        vol = float(last_ind.get("volume", 0.0))
                        vol_sma = float(last_ind.get("volume_sma_20", 0.0))
                        atr = float(last_ind.get("atr_14", 0.0))
                        if strategy:
                            strategy.last_atr = atr
                    else:
                        ema_200 = ema_9 = ema_21 = ema_50 = kc_upper = adx = plus_di = minus_di = chop = rsi = vol = vol_sma = atr = 0.0

                    bell_ema200 = (px_f > ema_200) if ema_200 > 0 else False
                    bell_tactical = (ema_9 > ema_21 > ema_50) and (px_f > ema_9) if (ema_9 > 0 and ema_21 > 0 and ema_50 > 0) else False
                    bell_keltner = (px_f > kc_upper) if kc_upper > 0 else False
                    bell_adx = adx >= 22.0
                    bell_di = plus_di > minus_di
                    bell_chop = chop < 58.0 if chop > 0 else False
                    bell_rsi = rsi >= 52.0
                    bell_vol = (vol >= vol_sma) if (vol > 0 and vol_sma > 0) else True

                    dist_kc_eur = kc_upper - px_f
                    dist_pct = ((px_f - kc_upper) / kc_upper * 100.0) if kc_upper > 0 else 0.0
                    has_pos = pos.quantity > Decimal("0")
                    strat_pos = strategy.get_position(p_disp) or strategy.get_position(p_norm) if strategy else None
                    stop_px = float(strat_pos.current_stop) if (strat_pos and strat_pos.current_stop) else (float(pos.average_entry_price * Decimal("0.96")) if has_pos else None)

                    if signal_res is not None and getattr(signal_res.side, "value", str(signal_res.side)).upper() == "BUY":
                        outcome = "🟢 INGRESSO BUY"
                    elif signal_res is not None and getattr(signal_res.side, "value", str(signal_res.side)).upper() == "SELL":
                        outcome = "🔴 USCITA SELL"
                    elif has_pos:
                        outcome = f"🛡 Posizione aperta — Stop a €{stop_px:,.2f}" if stop_px else "🛡 Posizione aperta"
                    else:
                        outcome = "⏸ Nessun segnale (IDLE)"

                    eval_summaries.append({
                        "symbol": p_disp,
                        "price": px_f,
                        "ema_200": ema_200,
                        "above_ema200": bell_ema200,
                        "breakout_level": kc_upper,
                        "dist_pct": dist_pct,
                        "dist_kc_eur": dist_kc_eur,
                        "signal_action": signal_res.side.value if signal_res else None,
                        "has_position": has_pos,
                        "stop_price": stop_px,
                        "outcome": outcome,
                        "atr": atr,
                        "ema_9": ema_9,
                        "ema_21": ema_21,
                        "ema_50": ema_50,
                        "adx": adx,
                        "plus_di": plus_di,
                        "minus_di": minus_di,
                        "chop": chop,
                        "rsi": rsi,
                        "vol": vol,
                        "vol_sma": vol_sma,
                        "bells": {
                            "ema_200": bell_ema200,
                            "tactical": bell_tactical,
                            "keltner": bell_keltner,
                            "adx": bell_adx,
                            "di": bell_di,
                            "chop": bell_chop,
                            "rsi": bell_rsi,
                            "volume": bell_vol,
                        },
                    })

                except Exception as pair_exc:
                    logger.warning(f"Error processing pair {p_item}: {pair_exc}")
                    print(f"[{now_utc.strftime('%H:%M:%S')}] [{p_item}] Network/Data error: {pair_exc}")

            # 4. Interactive Telegram Bot 4h synchronization & evaluation report
            block_hour = (now_utc.hour // 4) * 4
            current_4h_block = now_utc.replace(hour=block_hour, minute=0, second=0, microsecond=0)
            current_4h_block_iso = current_4h_block.isoformat()

            if tg_listener is not None:
                # Update cached evaluation summaries for instant admin query access
                if eval_summaries:
                    tg_listener.set_latest_eval_summaries(eval_summaries)

                # Process pending commands / callbacks once AFTER prices and eval_summaries are populated
                tg_listener.process_pending_updates_once()

                # Start continuous background polling daemon after first data fetch completes
                if not tg_listener._running and (iterations == 0 or iterations > 1):
                    tg_listener.start()
                    print("[INFO] Telegram Interactive Bot (@GILLYTRADESBOT_bot) avviato in background 24/7.")

                # Dispatch 4H report if not already sent for this 4H candle block
                if eval_summaries:
                    try:
                        tg_listener.send_4h_evaluation_report(
                            candle_ts_iso=current_4h_block_iso,
                            eval_summaries=eval_summaries,
                            force=False,
                        )
                    except Exception as rep_err:
                        logger.warning(f"Failed to dispatch 4h evaluation report: {rep_err}")

            # 4. Mark to market portfolio valuation across all held pairs
            equity = ledger.total_equity(price_tracker)
            net_profit = (ledger.realized_pnl if ledger else Decimal("0.0")) + (ledger.unrealized_pnl(price_tracker) if ledger else Decimal("0.0"))
            if dashboard and dash_server:
                dash_server.state.set_portfolio(
                    equity=float(equity),
                    cash=float(ledger.cash),
                    net_profit=float(net_profit),
                )

            if current_iter % 6 == 1:
                held_summary = ", ".join(f"{s}:{pos.quantity:.3f}" for s, pos in ledger.positions.items() if pos.quantity > 0) or "NONE"
                net_profit = (ledger.realized_pnl if ledger else Decimal("0.0")) + (ledger.unrealized_pnl(price_tracker) if ledger else Decimal("0.0"))
                print(
                    f"[{now_utc.strftime('%H:%M:%S')}] PORTFOLIO STATUS | Equity: {equity:.2f} {currency} | "
                    f"Cash: {ledger.cash:.2f} {currency} | Guadagni Netti: {net_profit:+.2f} {currency} | Open: [{held_summary}]"
                )

            # 5. Send periodic portfolio update notification
            if current_iter % 20 == 1:
                open_cnt = sum(1 for pos in ledger.positions.values() if pos.quantity > Decimal("0"))
                notif_service.notify_portfolio(
                    total_equity=float(equity),
                    cash_balance=float(ledger.cash),
                    unrealized_pnl=float(ledger.unrealized_pnl(price_tracker)),
                    realized_pnl_cumulative=float(ledger.realized_pnl),
                    total_trades_count=len(ledger.trades),
                    open_positions_count=open_cnt,
                    mode="PAPER",
                    timestamp=now_utc,
                )

        except Exception as exc:
            logger.warning(f"Error in paper cycle {current_iter}: {exc}")
            print(f"[{now_utc.strftime('%H:%M:%S')}] Warning in cycle {current_iter}: {exc}")
            time.sleep(5)

        # Check iterations boundary
        if iterations > 0 and current_iter >= iterations:
            print(f"[INFO] Completed target {iterations} iterations.")
            break

        # Responsive interruptible sleep with sub-second admin command checks
        sleep_elapsed = 0.0
        while running and sleep_elapsed < interval_sec:
            try:
                execute_pending_admin_commands(
                    admin_queue=admin_cmd_queue,
                    simulator=simulator,
                    ledger=ledger,
                    strategies_by_pair=strategies_by_pair,
                    price_tracker=price_tracker,
                    step_size_map=step_size_map,
                    tg_listener=tg_listener,
                    notif_service=notif_service,
                    currency=currency,
                )
            except Exception as adm_ex:
                logger.warning(f"Error checking/executing admin commands: {adm_ex}")
            time.sleep(min(0.5, interval_sec - sleep_elapsed))
            sleep_elapsed += 0.5

    # Session Summary
    print("=" * 70)
    print("                    PAPER TRADING SESSION SUMMARY                    ")
    print("=" * 70)
    final_eq = ledger.total_equity()
    print(f"Initial Balance:       {balance:.2f} {currency}")
    print(f"Final Total Equity:    {final_eq:.2f} {currency}")
    print(f"Net Realized PnL:      {ledger.realized_pnl:+.2f} {currency}")
    print(f"Total Trades Executed: {len(ledger.trades)}")
    print("=" * 70)

    if dash_server:
        dash_server.state.set_status("idle")
        dash_server.stop()

    if tg_listener:
        tg_listener.stop()

    notif_service.close()
    return ledger, notif_service


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for paper trading."""
    parser = argparse.ArgumentParser(
        prog="trading_bot.paper",
        description="Autonomous Compounding Trading Bot - Real-Time Paper Trading",
    )

    parser.add_argument(
        "--pair",
        type=str,
        default="BTC/EUR",
        help="Target trading pair (default: BTC/EUR)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10.0,
        help="Initial virtual paper balance (default: 10.0 EUR)",
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="EUR",
        help="Quote currency (default: EUR)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help="Candle timeframe (default: 1h)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange rules (binance, bybit, revolutx) (default: binance)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Maximum cycles before graceful termination (0 for infinite) (default: 0)",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=5.0,
        help="Polling interval between ticker updates in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--notify-channel",
        choices=["stdout", "console", "json", "telegram", "csv", "all"],
        default="stdout",
        help="Notification delivery channel (default: stdout)",
    )
    parser.add_argument(
        "--trades-csv",
        type=str,
        default="data/trades.csv",
        help="Path to CSV trade log file (default: data/trades.csv)",
    )
    parser.add_argument(
        "--telegram-token",
        type=str,
        default=None,
        help="Telegram Bot API token",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=str,
        default=None,
        help="Telegram target chat ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate signals and pricing without placing virtual orders",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch real-time web dashboard server in background daemon thread",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8000,
        help="Port for the dashboard web server (default: 8000)",
    )
    parser.add_argument(
        "--dashboard-host",
        type=str,
        default="127.0.0.1",
        help="Host address for the dashboard web server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable real order execution on Bybit Spot V5 using API keys",
    )

    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return EXIT_CLI_USAGE_ERROR if exc.code != 0 else EXIT_SUCCESS

    if parsed.balance <= 0.0:
        sys.stderr.write(f"Error: Initial balance must be > 0, got {parsed.balance}\n")
        return EXIT_INSUFFICIENT_FUNDS_ERROR

    try:
        run_paper(
            pair=parsed.pair,
            balance=parsed.balance,
            currency=parsed.currency,
            timeframe=parsed.timeframe,
            exchange=parsed.exchange,
            iterations=parsed.iterations,
            interval_sec=parsed.interval_sec,
            notify_channel=parsed.notify_channel,
            telegram_token=parsed.telegram_token,
            telegram_chat_id=parsed.telegram_chat_id,
            dry_run=parsed.dry_run,
            trades_csv=parsed.trades_csv,
            dashboard=parsed.dashboard,
            dashboard_port=parsed.dashboard_port,
            dashboard_host=parsed.dashboard_host,
            live=parsed.live,
        )
        return EXIT_SUCCESS
    except Exception as exc:
        sys.stderr.write(f"Paper trading runtime error: {exc}\n")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())

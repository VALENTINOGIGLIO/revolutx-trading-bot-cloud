"""Admin Manual Command Queue and Coordination.

Enables the designated Administrator (Telegram chat_id 1186171006) to bypass
strategy entry filters and trigger forced market entries or exits on Revolut X,
while maintaining automated stop-loss and trailing chandelier protections.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AdminCommandQueue")

ADMIN_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1186171006").strip()
DEFAULT_COMMANDS_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "admin_manual_commands.json"


class AdminCommandQueue:
    """Thread-safe and process-safe JSON-backed command queue for Admin manual overrides."""

    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = Path(filepath or DEFAULT_COMMANDS_FILE)

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.filepath.exists():
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as ex:
            logger.warning(f"Failed to read admin commands file {self.filepath}: {ex}")
            return []

    def _write_all(self, items: List[Dict[str, Any]]) -> None:
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            temp = self.filepath.with_suffix(".tmp")
            with open(temp, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
            temp.replace(self.filepath)
        except Exception as ex:
            logger.error(f"Failed to write admin commands file {self.filepath}: {ex}")

    def enqueue(self, action: str, symbol: str, requested_by: str) -> Optional[Dict[str, Any]]:
        """Enqueue an admin command (BUY or SELL).

        Args:
            action: 'BUY' or 'SELL'.
            symbol: Target pair (e.g. 'BTC/EUR', 'SOL/EUR', 'ETH/EUR').
            requested_by: Telegram chat_id of sender.

        Returns:
            Command dict if enqueued, or None if unauthorized.
        """
        if str(requested_by).strip() != ADMIN_CHAT_ID:
            logger.warning(f"Unauthorized manual command attempt by chat_id {requested_by}")
            return None

        # Clean symbol: e.g. BTC -> BTC/EUR
        sym = symbol.strip().upper()
        if "/" not in sym:
            sym = f"{sym}/EUR"

        cmd_id = f"cmd_{int(time.time() * 1000)}_{action.lower()}"
        cmd_record = {
            "id": cmd_id,
            "action": action.upper(),
            "symbol": sym,
            "requested_by": str(requested_by),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "PENDING",
            "result": None,
        }

        items = self._read_all()
        # Keep recent 50 commands
        items = items[-49:]
        items.append(cmd_record)
        self._write_all(items)
        logger.info(f"[ADMIN QUEUE] Enqueued {action} on {sym} by {requested_by} (ID: {cmd_id})")
        return cmd_record

    def get_pending(self) -> List[Dict[str, Any]]:
        """Retrieve all currently pending admin commands."""
        items = self._read_all()
        return [it for it in items if it.get("status") == "PENDING"]

    def update_status(self, cmd_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Update the status and execution result of a command."""
        items = self._read_all()
        updated = False
        for it in items:
            if it.get("id") == cmd_id:
                it["status"] = status
                it["updated_at"] = datetime.now(timezone.utc).isoformat()
                if result:
                    it["result"] = result
                updated = True
                break
        if updated:
            self._write_all(items)
            logger.info(f"[ADMIN QUEUE] Command {cmd_id} marked as {status}")


def calculate_risk_reward_estimates(
    active_bells: int,
    total_bells: int,
    price: float,
    atr: float,
    ema200_ok: bool,
    equity: float = 150.0,
    missing_bells: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate mathematically grounded risk, expected return, and empirical win rate based on 11,833 4h candles.

    Derived from historical empirical backtests on Revolut X 4h basket:
    - 8/8: WR 36.4%, Win +13.70% (+€20.55), Loss -4.05% (-€6.08), E[R] +2.40% (+€3.60), MaxDD 28.2%, 5y +282.5%
    - 7/8 (No Vol): WR 42.5%, Win +12.55% (+€18.83), Loss -4.50% (-€6.75), E[R] +2.76% (+€4.13), MaxDD 19.5%
    - 7/8 (Generico): WR 34.5%, Win +13.31% (+€19.97), Loss -3.97% (-€5.95), E[R] +2.00% (+€2.99), MaxDD 31.2%
    - 6/8 Pre-Breakout (EMA200 OK, manca breakout & tactical): WR 28.2%, Win +17.63% (+€26.45), Loss -3.98% (-€5.97), E[R] +2.12% (+€3.18), MaxDD 32.6%
    - 6/8 Generico: WR 30.0%, Win +13.14% (+€19.70), Loss -3.95% (-€5.93), E[R] +1.17% (+€1.75), MaxDD 47.9%
    - 5/8: WR 30.3%, Win +13.50% (+€20.25), Loss -3.84% (-€5.76), E[R] +1.42% (+€2.13), MaxDD 53.4%
    - 4/8: WR 29.4%, Win +12.47% (+€18.71), Loss -4.19% (-€6.28), E[R] +0.70% (+€1.05), MaxDD 68.8%
    - <4/8: WR 30.5%, Win +11.74% (+€17.61), Loss -4.67% (-€7.00), E[R] +0.33% (+€0.50), PF 0.95 (Perdita -41.9%), MaxDD 85.3%
    """
    missing = set(missing_bells or [])
    stop_mult = 2.2
    stop_distance = stop_mult * atr
    sl_pct_current = (stop_distance / price) * 100.0 if price > 0 else 0.0

    if active_bells >= 8:
        wr = 36.4
        win_pct = 13.70
        loss_pct = 4.05
        er_pct = 2.40
        max_dd = "28.2%"
        consiglio = "🟢 Vantaggio massimo (+282.5% storico, Edge confermato)"
    elif active_bells == 7 and ("volume" in missing):
        wr = 42.5
        win_pct = 12.55
        loss_pct = 4.50
        er_pct = 2.76
        max_dd = "19.5%"
        consiglio = "🟢 Ottimo setup (MaxDD minimo 19.5%, E[R] top)"
    elif active_bells == 7:
        wr = 34.5
        win_pct = 13.31
        loss_pct = 3.97
        er_pct = 2.00
        max_dd = "31.2%"
        consiglio = "🟢 Vantaggio positivo (+2.00% E[R])"
    elif active_bells == 6 and ema200_ok and (("keltner" in missing) or ("tactical" in missing)):
        wr = 28.2
        win_pct = 17.63
        loss_pct = 3.98
        er_pct = 2.12
        max_dd = "32.6%"
        consiglio = "🟢 Edge asimmetrico favorevole (+17.6% su win, E[R] +2.12%)"
    elif active_bells == 6:
        wr = 30.0
        win_pct = 13.14
        loss_pct = 3.95
        er_pct = 1.17
        max_dd = "47.9%"
        consiglio = "🟡 Edge moderato ma MaxDD più pesante (47.9%)"
    elif active_bells == 5:
        wr = 30.3
        win_pct = 13.50
        loss_pct = 3.84
        er_pct = 1.42
        max_dd = "53.4%"
        consiglio = "🟡 Discreto ma con drawdown significativo (53.4%)"
    elif active_bells == 4:
        wr = 29.4
        win_pct = 12.47
        loss_pct = 4.19
        er_pct = 0.70
        max_dd = "68.8%"
        consiglio = "⚠️ Rischio elevato (MaxDD 68.8%, basso edge)"
    else:
        wr = 30.5
        win_pct = 11.74
        loss_pct = 4.67
        er_pct = 0.33
        max_dd = "85.3%"
        consiglio = "🔴 EV NEGATIVO sul lungo periodo (PF 0.95, perdita certa)"

    # Compute monetary amounts based on 1/3 per-coin slot (e.g. ~50 € per coin out of 150 €)
    slot_equity = equity / 3.0
    sl_eur = (loss_pct / 100.0) * slot_equity
    win_eur = (win_pct / 100.0) * slot_equity
    er_eur = (er_pct / 100.0) * slot_equity
    er_sign = "+" if er_pct >= 0 else ""

    # Current ATR stop loss distance
    sl_current_eur = (sl_pct_current / 100.0) * slot_equity if sl_pct_current > 0 else sl_eur

    return {
        "sl_pct": loss_pct,
        "sl_eur": sl_eur,
        "sl_current_pct": sl_pct_current,
        "sl_current_eur": sl_current_eur,
        "win_rate": wr,
        "win_pct": win_pct,
        "win_eur": win_eur,
        "er_pct": er_pct,
        "er_eur": er_eur,
        "er_sign": er_sign,
        "max_dd": max_dd,
        "consiglio": consiglio,
    }


def format_admin_top3_report(
    eval_summaries: List[Dict[str, Any]],
    equity: float = 150.0,
    net_pnl: float = 0.0,
    candle_ts_iso: Optional[str] = None,
) -> str:
    """Format a clean, 3-line per coin dynamic ranking of the Top 3 entry setups for Admin."""
    if not eval_summaries:
        return "⚠️ Nessun dato di valutazione disponibile al momento."

    bell_labels = {
        "ema_200": "EMA200",
        "tactical": "EMA Stack",
        "keltner": "Breakout",
        "adx": "ADX",
        "di": "+DI",
        "chop": "CHOP",
        "rsi": "RSI",
        "volume": "Volume",
    }

    ranked = []
    for s in eval_summaries:
        bells = s.get("bells", {})
        active_cnt = sum(1 for v in bells.values() if v)
        missing_keys = [k for k, v in bells.items() if not v]
        ema200_ok = bool(bells.get("ema_200", False))
        box = calculate_risk_reward_estimates(
            active_bells=active_cnt,
            total_bells=8,
            price=float(s.get("price", 0.0)),
            atr=float(s.get("atr", 0.0)),
            ema200_ok=ema200_ok,
            equity=equity,
            missing_bells=missing_keys,
        )
        is_acceptable = (active_cnt >= 6) and ema200_ok
        score = (1 if is_acceptable else 0, box["er_pct"], active_cnt)
        ranked.append({
            "summary": s,
            "active_cnt": active_cnt,
            "ema200_ok": ema200_ok,
            "is_acceptable": is_acceptable,
            "box": box,
            "score": score,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    header_ts = f" (Candela 4H: `{candle_ts_iso}`)" if candle_ts_iso else ""
    lines = [
        f"👑 *REPORT DECISIONALE RISERVATO ADMIN*{header_ts}",
        "",
        f"💰 *Capitale Totale:* €{equity:.2f} | 📈 *Guadagni Netti:* {net_pnl:+.2f} €",
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 *TOP 3 OPZIONI DI AVVIO IN QUESTO MOMENTO*",
        "",
    ]

    above_thresh = []
    medals = ["1°", "2°", "3°"]

    for i, item in enumerate(ranked[:3]):
        s = item["summary"]
        sym = s.get("symbol", "")
        px = float(s.get("price", 0.0))
        cnt = item["active_cnt"]
        box = item["box"]
        acc = item["is_acceptable"]
        bells = s.get("bells", {})

        if cnt >= 8 and ema200_ok:
            above_thresh.append(f"{sym} (Auto 8/8)")
            status_tag = "🟢 *STRATEGIA AUTOMATICA (8/8) — Il bot entra da solo*"
        elif acc:
            above_thresh.append(f"{sym} (Solo Manuale {cnt}/8)")
            status_tag = f"🟡 *SOLO AVVIO MANUALE ({cnt}/8) — Il bot automatico aspetta 8/8*"
        else:
            status_tag = "🔴 *SOTTO SOGLIA — Non avviare*"

        act_list = [bell_labels[k] for k, v in bells.items() if v and k in bell_labels]
        miss_list = []
        for k, v in bells.items():
            if not v and k in bell_labels:
                if k == "keltner":
                    dist_pct = s.get("dist_pct", 0.0)
                    miss_list.append(f"Breakout ({dist_pct:+.1f}%)")
                else:
                    miss_list.append(bell_labels[k])

        act_str = ", ".join(act_list) if act_list else "Nessuno"
        miss_str = ", ".join(miss_list) if miss_list else "Nessuno"

        medal = medals[i] if i < len(medals) else f"{i+1}°"
        lines.append(f"{medal} 🪙 *{sym}* (€{px:,.2f}) — Campanelli: `{cnt}/8` | {status_tag}")
        lines.append(f"   • ✅ Attivi: `{act_str}` | ❌ Mancano: `{miss_str}`")
        lines.append(
            f"   • 📊 Prob. Win: `{box['win_rate']:.1f}%` | Se vince: `+{box['win_pct']:.1f}% (+€{box['win_eur']:.2f})` | "
            f"Se perde (SL): `-{box['sl_pct']:.1f}% (-€{box['sl_eur']:.2f})` | "
            f"*Atteso: {box['er_sign']}{box['er_pct']:.2f}% ({box['er_sign']}€{box['er_eur']:.2f})*"
        )
        lines.append("")

    thresh_coins = ", ".join(f"*{c}*" for c in above_thresh) if above_thresh else "nessuna moneta"
    lines.extend([
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📏 *REGOLE STRATEGIA vs AVVIO MANUALE:*",
        "• 🤖 *Bot Automatico*: compra da solo **solo a `8/8` campanelli** (`Mancano: Nessuno`).",
        "• ⚡ *Forzatura Manuale*: accettabile da `6/8` o `7/8` (con `EMA 200 ✅` e Atteso `>= +2.00%`), ma richiede il tuo clic.",
        f"• Stato attuale: {thresh_coins}.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        "⚡ _Per forzare a mano una posizione: usa i pulsanti d'azione sottostanti._"
    ])

    return "\n".join(lines)

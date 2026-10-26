"""
Revolut X Crypto Exchange Live Execution & Gateway Connector.

Provides:
- RevolutXLiveExecutor: Complete ExecutionSimulator-compatible executor for Revolut X.
- Ed25519 Cryptographic Request Signing conforming to Revolut X specifications.
- Limit Maker 0.00% fee order placement (post_only) with Market Taker (0.09%) support.
- Real-time balance queries and Auto-Deposit Detector (detects incoming EUR deposits).
- Dual-mode operation: Live exchange execution or high-fidelity simulation with Revolut X friction rules.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import httpx
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from trading_bot.simulation.ledger import (
    InsufficientFundsError,
    InsufficientPositionError,
    TradeRecord,
    VirtualLedger,
    Position,
)
from trading_bot.simulation.simulator import (
    EXCHANGE_MIN_NOTIONAL,
    ExecutionResult,
    ExecutionSimulator,
    OrderSide,
    OrderSignal,
    OrderType,
)

logger = logging.getLogger("revolutx_live")

# Revolut X Official Base URL & Fee Constants
REVOLUTX_DEFAULT_BASE_URL: str = "https://revx.revolut.com/api/1.0"
REVOLUTX_MIN_NOTIONAL_EUR: Decimal = Decimal("1.00")
REVOLUTX_MAKER_FEE_RATE: Decimal = Decimal("0.0000")  # 0.00% Limit Maker fee
REVOLUTX_TAKER_FEE_RATE: Decimal = Decimal("0.0009")  # 0.09% Market Taker fee


def to_revolut_symbol(symbol: str) -> str:
    """
    Format standard or normalized symbol to Revolut X hyphenated pair.
    e.g. 'BTC/EUR' -> 'BTC-EUR', 'BTCEUR' -> 'BTC-EUR', 'DOGEEUR' -> 'DOGE-EUR'.
    """
    if not symbol:
        return ""
    clean = symbol.replace("/", "").replace("_", "").replace("-", "").strip().upper()
    for quote in ["EUR", "USD", "USDT", "GBP"]:
        if clean.endswith(quote) and len(clean) > len(quote):
            base = clean[: -len(quote)]
            return f"{base}-{quote}"
    return symbol


def from_revolut_symbol(revx_symbol: str) -> str:
    """
    Convert Revolut X hyphenated pair to normalized symbol.
    e.g. 'BTC-EUR' -> 'BTCEUR'.
    """
    return revx_symbol.replace("-", "").replace("/", "").strip().upper()


class RevolutXLiveExecutor(ExecutionSimulator):
    """
    Revolut X Crypto Exchange Connector & Execution Engine.

    Implements the ExecutionSimulator interface with full Ed25519 cryptographic signing,
    0.00% Maker fee execution, live balance syncing, and automated deposit detection.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_path: Optional[Union[str, Path]] = None,
        private_key_pem: Optional[Union[str, bytes]] = None,
        base_url: Optional[str] = None,
        live: Optional[bool] = None,
        prefer_maker: bool = True,
        fee_rate: Optional[Union[Decimal, float, str]] = None,
        slippage: Optional[Union[Decimal, float, str]] = None,
        min_notional: Optional[Union[Decimal, float, str]] = None,
        step_size: Union[Decimal, float, str] = Decimal("0.0001"),
        tick_size: Optional[Union[Decimal, float, str]] = None,
        strict: bool = False,
        notification_service: Optional[Any] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        effective_min_notional = (
            Decimal(str(min_notional)) if min_notional is not None else REVOLUTX_MIN_NOTIONAL_EUR
        )
        effective_fee = (
            Decimal(str(fee_rate))
            if fee_rate is not None
            else (REVOLUTX_MAKER_FEE_RATE if prefer_maker else REVOLUTX_TAKER_FEE_RATE)
        )
        effective_slippage = (
            Decimal(str(slippage))
            if slippage is not None
            else (Decimal("0.0000") if prefer_maker else Decimal("0.0002"))
        )

        super().__init__(
            fee_rate=effective_fee,
            slippage=effective_slippage,
            min_notional=effective_min_notional,
            exchange="revolutx",
            step_size=step_size,
            tick_size=tick_size,
            strict=strict,
            notification_service=notification_service,
        )

        self.api_key: Optional[str] = api_key or os.environ.get("REVOLUTX_API_KEY")
        self.base_url: str = (
            base_url
            or os.environ.get("REVOLUTX_BASE_URL")
            or REVOLUTX_DEFAULT_BASE_URL
        ).rstrip("/")
        if live is not None:
            self.live: bool = live
        else:
            self.live: bool = os.environ.get("TRADING_MODE", "").strip().upper() == "LIVE"
        self.prefer_maker: bool = prefer_maker

        # Auto-Deposit tracking
        self.last_seen_eur_balance: Optional[Decimal] = None

        # Load Ed25519 Private Key
        self._private_key: Optional[ed25519.Ed25519PrivateKey] = None
        self._load_private_key(private_key_path=private_key_path, private_key_pem=private_key_pem)

        # HTTP Client
        self._client: httpx.Client = http_client or httpx.Client(timeout=10.0)
        self._owns_client: bool = http_client is None

    def _load_private_key(
        self,
        private_key_path: Optional[Union[str, Path]] = None,
        private_key_pem: Optional[Union[str, bytes]] = None,
    ) -> None:
        """Load and deserialize Ed25519 private key from PEM bytes, file path, or default keys dir."""
        pem_bytes: Optional[bytes] = None

        if private_key_pem is not None:
            pem_bytes = (
                private_key_pem.encode("utf-8")
                if isinstance(private_key_pem, str)
                else private_key_pem
            )
        else:
            path_candidate = (
                private_key_path
                or os.environ.get("REVOLUTX_PRIVATE_KEY_PATH")
                or Path("keys/revolutx_private.pem")
            )
            resolved_path = Path(path_candidate)
            if not resolved_path.is_absolute():
                # Attempt relative to current working directory or keys dir
                candidate_abs = Path.cwd() / resolved_path
                if candidate_abs.exists():
                    resolved_path = candidate_abs

            if resolved_path.exists():
                try:
                    pem_bytes = resolved_path.read_bytes()
                except Exception as ex:
                    logger.warning(f"Failed to read private key at {resolved_path}: {ex}")

        if pem_bytes:
            try:
                loaded = serialization.load_pem_private_key(pem_bytes, password=None)
                if isinstance(loaded, ed25519.Ed25519PrivateKey):
                    self._private_key = loaded
                    logger.debug("Revolut X Ed25519 private key successfully loaded.")
                else:
                    logger.warning(f"Key loaded is not Ed25519PrivateKey: {type(loaded)}")
            except Exception as exc:
                logger.warning(f"Could not parse Ed25519 private key: {exc}")

    @property
    def has_auth(self) -> bool:
        """Return True if both API key and Ed25519 private key are loaded."""
        return bool(self.api_key and self._private_key)

    def sign_request(
        self,
        method: str,
        path: str,
        query: str = "",
        body: str = "",
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Sign request data using Ed25519 conforming to official Revolut X specifications.

        Message payload string to sign is strictly:
        timestamp + uppercase(method) + path + query + body (no separators)
        """
        ts_str = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
        cleaned_query = query.lstrip("?") if query else ""
        cleaned_path = path if path.startswith("/") else f"/{path}"

        # Ensure path begins with /api if not already included
        if not cleaned_path.startswith("/api"):
            cleaned_path = f"/api/1.0{cleaned_path}"

        msg_str = f"{ts_str}{method.upper()}{cleaned_path}{cleaned_query}{body}"
        msg_bytes = msg_str.encode("utf-8")

        if self._private_key is not None:
            raw_sig = self._private_key.sign(msg_bytes)
            sig_b64 = base64.b64encode(raw_sig).decode("ascii")
        else:
            sig_b64 = ""

        return ts_str, sig_b64

    def get_auth_headers(
        self,
        method: str,
        path: str,
        query: str = "",
        body: str = "",
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, str]:
        """Generate authentication headers for Revolut X REST API request."""
        ts_str, sig_b64 = self.sign_request(
            method=method,
            path=path,
            query=query,
            body=body,
            timestamp_ms=timestamp_ms,
        )
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Revx-API-Key": self.api_key or "",
            "X-Revx-Timestamp": ts_str,
            "X-Revx-Signature": sig_b64,
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        query_params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute an authenticated HTTP request against Revolut X REST API."""
        cleaned_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        path = f"/api/1.0{cleaned_endpoint}" if not cleaned_endpoint.startswith("/api") else cleaned_endpoint

        query_str = ""
        if query_params:
            query_str = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))

        body_str = ""
        if json_body is not None:
            body_str = json.dumps(json_body, separators=(",", ":"))

        headers = self.get_auth_headers(
            method=method,
            path=path,
            query=query_str,
            body=body_str,
        )

        base_host = self.base_url
        if base_host.endswith("/api/1.0"):
            base_host = base_host[:-8]
        elif base_host.endswith("/api/1.0/"):
            base_host = base_host[:-9]
        elif base_host.endswith("/api"):
            base_host = base_host[:-4]

        url = f"{base_host}{path}"
        if query_str:
            url = f"{url}?{query_str}"

        response = self._client.request(
            method=method.upper(),
            url=url,
            headers=headers,
            content=body_str.encode("utf-8") if body_str else None,
        )

        if response.status_code >= 400:
            err_msg = f"Revolut X API HTTP {response.status_code}: {response.text}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        if not response.content:
            return {}
        return response.json()

    def fetch_balances(self) -> List[Dict[str, Any]]:
        """
        Fetch all fiat and crypto account balances from Revolut X.
        Calls GET /api/1.0/balances.
        """
        res = self._request("GET", "/balances")
        if isinstance(res, list):
            return res
        return []

    def fetch_eur_balance(self) -> Decimal:
        """Retrieve current available EUR cash balance."""
        balances = self.fetch_balances()
        for b in balances:
            if b.get("currency") == "EUR":
                try:
                    return Decimal(str(b.get("available", "0.0")))
                except (InvalidOperation, TypeError):
                    return Decimal("0.0")
        return Decimal("0.0")

    def fetch_crypto_balances(self) -> Dict[str, Decimal]:
        """Retrieve dictionary of open crypto positions {currency: available_qty}."""
        balances = self.fetch_balances()
        crypto_map: Dict[str, Decimal] = {}
        for b in balances:
            curr = str(b.get("currency", "")).upper()
            if curr and curr not in ("EUR", "USD", "GBP", "USDT"):
                try:
                    qty = Decimal(str(b.get("available", "0.0")))
                    if qty > Decimal("0.0"):
                        crypto_map[curr] = qty
                except (InvalidOperation, TypeError):
                    pass
        return crypto_map

    def get_wallet_balances(self) -> Dict[str, Decimal]:
        """Return unified dictionary of EUR cash and all crypto balances."""
        bals: Dict[str, Decimal] = {}
        try:
            for b in self.fetch_balances():
                curr = str(b.get("currency", "")).upper()
                if curr:
                    try:
                        bals[curr] = Decimal(str(b.get("available", "0.0")))
                    except (InvalidOperation, TypeError):
                        pass
        except Exception as ex:
            logger.warning(f"Could not fetch Revolut X wallet balances: {ex}")
        return bals

    def get_instrument_rules(self, symbol: str) -> Dict[str, Any]:
        """Return instrument trading rules (precision, tickSize, minOrderAmt)."""
        revx_sym = to_revolut_symbol(symbol)
        parts = revx_sym.split("-")
        base_coin = parts[0] if parts else symbol
        quote_coin = parts[1] if len(parts) > 1 else "EUR"

        coin = base_coin.upper()
        if "DOGE" in coin:
            base_prec = Decimal("1.0")
            tick_size = Decimal("0.00001")
        elif "BTC" in coin:
            base_prec = Decimal("0.00001")
            tick_size = Decimal("0.01")
        elif "ETH" in coin:
            base_prec = Decimal("0.0001")
            tick_size = Decimal("0.01")
        elif "SOL" in coin:
            base_prec = Decimal("0.001")
            tick_size = Decimal("0.01")
        elif "XRP" in coin:
            base_prec = Decimal("0.1")
            tick_size = Decimal("0.0001")
        else:
            base_prec = self.step_size
            tick_size = self.tick_size or Decimal("0.0001")

        return {
            "symbol": revx_sym,
            "baseCoin": base_coin,
            "quoteCoin": quote_coin,
            "basePrecision": base_prec,
            "tickSize": tick_size,
            "minOrderAmt": self.min_notional,
        }

    def check_auto_deposit(self, ledger: Optional[VirtualLedger] = None) -> Optional[Decimal]:
        """
        Auto-Deposit Detector:
        Polls current EUR balance and compares against last known EUR balance.
        If an increase >= 0.01 EUR is detected, credits the ledger and returns the deposit delta.
        """
        try:
            current_eur = self.fetch_eur_balance()
        except Exception as exc:
            logger.warning(f"Could not check Revolut X balances for auto-deposit: {exc}")
            return None

        deposit_delta: Optional[Decimal] = None
        if self.last_seen_eur_balance is not None:
            if current_eur > self.last_seen_eur_balance:
                delta = current_eur - self.last_seen_eur_balance
                if delta >= Decimal("0.01"):
                    deposit_delta = delta
                    logger.info(
                        f"[AUTO-DEPOSIT DETECTED] Deposited +{delta:.2f} EUR! "
                        f"Previous: {self.last_seen_eur_balance:.2f} EUR | New: {current_eur:.2f} EUR"
                    )
                    if ledger is not None:
                        ledger.deposit(delta)
                        if self.notification_service:
                            try:
                                self.notification_service.notify_custom(
                                    title="Deposit Detected",
                                    message=f"Revolut X Auto-Deposit: +{delta:.2f} EUR credited to trading ledger.",
                                    level="INFO",
                                )
                            except Exception:
                                pass

        self.last_seen_eur_balance = current_eur
        return deposit_delta

    def sync_ledger(
        self,
        ledger: VirtualLedger,
        price_tracker: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """
        Full synchronization of VirtualLedger cash and crypto positions with live Revolut X balances.
        """
        # Run auto-deposit check
        self.check_auto_deposit(ledger=ledger)

        # Sync cash balance
        eur_balance = self.fetch_eur_balance()
        ledger.set_cash(eur_balance)

        # Sync crypto positions
        crypto_positions = self.fetch_crypto_balances()
        for curr, qty in crypto_positions.items():
            sym_eur = f"{curr}EUR"
            pos = ledger.get_position(sym_eur)
            if pos.quantity != qty:
                pos.quantity = qty
                if price_tracker and sym_eur in price_tracker:
                    pos.last_price = price_tracker[sym_eur]
                    if pos.average_entry_price <= Decimal("0.0"):
                        pos.average_entry_price = price_tracker[sym_eur]
                        pos.cost_basis = qty * price_tracker[sym_eur]

        return {
            "cash_eur": float(eur_balance),
            "crypto_positions": {k: float(v) for k, v in crypto_positions.items()},
        }

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        order_type: str = "LIMIT",
        post_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Submit an order to Revolut X.
        Supports 0.00% Maker Limit orders with execution_instructions=['post_only'].
        """
        revx_symbol = to_revolut_symbol(symbol)
        side_lower = side.lower()
        use_post_only = self.prefer_maker if post_only is None else post_only

        client_order_id = str(uuid.uuid4())

        if order_type.upper() == "LIMIT" or use_post_only:
            config = {
                "limit": {
                    "price": str(price),
                    "base_size": str(quantity),
                    "execution_instructions": ["post_only"] if use_post_only else ["allow_taker"],
                }
            }
        else:
            config = {
                "market": {
                    "base_size": str(quantity),
                }
            }

        payload = {
            "client_order_id": client_order_id,
            "symbol": revx_symbol,
            "side": side_lower,
            "order_configuration": config,
        }

        return self._request("POST", "/orders", json_body=payload)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an active order on Revolut X."""
        return self._request("DELETE", f"/orders/{order_id}")

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve open/active orders from Revolut X."""
        params = {}
        if symbol:
            params["symbol"] = to_revolut_symbol(symbol)
        res = self._request("GET", "/orders/active", query_params=params)
        if isinstance(res, list):
            return res
        elif isinstance(res, dict) and "orders" in res:
            return res["orders"]
        return []

    def execute(
        self,
        order: Union[OrderSignal, Dict[str, Any], Any],
        current_price: Union[Decimal, float, str],
        ledger: VirtualLedger,
        timestamp_ms: Optional[int] = None,
        notification_service: Optional[Any] = None,
        raise_on_error: Optional[bool] = None,
    ) -> ExecutionResult:
        """
        Execute an order either on live Revolut X exchange or in high-fidelity simulation mode.
        """
        if not self.live:
            # High-fidelity simulation mode with Revolut X rules (0.00% maker fee, 1.00€ min notional)
            return super().execute(
                order=order,
                current_price=current_price,
                ledger=ledger,
                timestamp_ms=timestamp_ms,
                notification_service=notification_service,
                raise_on_error=raise_on_error,
            )

        # Live Execution Path
        raw_px = Decimal(str(current_price))
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        # Extract order parameters
        if isinstance(order, dict):
            symbol = str(order.get("symbol", "BTCEUR")).upper()
            side_val = str(order.get("side", "BUY")).upper()
            target_qty = Decimal(str(order["target_quantity"])) if order.get("target_quantity") is not None else None
            target_notional = Decimal(str(order["target_notional"])) if order.get("target_notional") is not None else None
            order_type = str(order.get("order_type", "LIMIT")).upper()
            order_id = order.get("order_id")
            exit_reason = order.get("exit_reason") or order.get("reason")
        else:
            symbol = str(getattr(order, "symbol", "BTCEUR")).upper()
            raw_side = getattr(order, "side", "BUY")
            side_val = str(raw_side.value if hasattr(raw_side, "value") else raw_side).upper()
            target_qty = (
                Decimal(str(getattr(order, "target_quantity")))
                if getattr(order, "target_quantity", None) is not None
                else None
            )
            target_notional = (
                Decimal(str(getattr(order, "target_notional")))
                if getattr(order, "target_notional", None) is not None
                else None
            )
            raw_type = getattr(order, "order_type", "LIMIT")
            order_type = str(raw_type.value if hasattr(raw_type, "value") else raw_type).upper()
            order_id = getattr(order, "order_id", None)
            exit_reason = getattr(order, "exit_reason", None) or getattr(order, "reason", None)

        # Compute quantity
        if side_val == "BUY":
            if target_qty is not None and target_qty > Decimal("0.0"):
                qty = self.quantize_quantity(target_qty)
            elif target_notional is not None and target_notional > Decimal("0.0"):
                qty = self.quantize_quantity(target_notional / raw_px)
            else:
                budget = ledger.available_cash
                qty = self.quantize_quantity(budget / raw_px)

            notional = qty * raw_px
            if notional > ledger.available_cash and ledger.available_cash >= self.min_notional:
                qty = self.quantize_quantity((ledger.available_cash * Decimal("0.998")) / raw_px)
                notional = qty * raw_px
            if notional < self.min_notional:
                err = f"Revolut X Live BUY rejected: Notional {notional:.2f} EUR below min {self.min_notional:.2f} EUR"
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="BUY",
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=raw_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=Decimal("0.0"),
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err,
                )

            try:
                # Place live order on Revolut X
                resp = self.place_order(
                    symbol=symbol,
                    side="buy",
                    quantity=qty,
                    price=raw_px,
                    order_type=order_type,
                    post_only=self.prefer_maker,
                )
                exec_id = resp.get("order_id") or resp.get("id") or order_id
                fee = Decimal("0.00") if self.prefer_maker else self.calculate_fee(notional)

                trade = ledger.record_buy(
                    symbol=symbol,
                    quantity=qty,
                    executed_price=raw_px,
                    fee=fee,
                    timestamp_ms=ts,
                    raw_price=raw_px,
                    order_type=order_type,
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    order_id=exec_id,
                )

                self._dispatch_notification(trade, ledger, notification_service)

                return ExecutionResult(
                    success=True,
                    symbol=symbol,
                    side="BUY",
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=raw_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=fee,
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    realized_pnl=Decimal("0.00"),
                    trade_record=trade,
                )
            except Exception as ex:
                err = f"Revolut X Live BUY error: {ex}"
                logger.error(err)
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="BUY",
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=raw_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=Decimal("0.0"),
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err,
                )

        elif side_val == "SELL":
            pos = ledger.get_position(symbol)
            qty = self.quantize_quantity(target_qty or pos.quantity)
            notional = qty * raw_px

            if qty <= Decimal("0.0") or pos.quantity < qty:
                err = f"Revolut X Live SELL rejected: Insufficient holding (have {pos.quantity}, need {qty})"
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="SELL",
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=raw_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=Decimal("0.0"),
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err,
                )

            try:
                resp = self.place_order(
                    symbol=symbol,
                    side="sell",
                    quantity=qty,
                    price=raw_px,
                    order_type=order_type,
                    post_only=self.prefer_maker,
                )
                exec_id = resp.get("order_id") or resp.get("id") or order_id
                fee = Decimal("0.00") if self.prefer_maker else self.calculate_fee(notional)

                trade = ledger.record_sell(
                    symbol=symbol,
                    quantity=qty,
                    executed_price=raw_px,
                    fee=fee,
                    timestamp_ms=ts,
                    raw_price=raw_px,
                    order_type=order_type,
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    order_id=exec_id,
                    exit_reason=exit_reason,
                )

                self._dispatch_notification(trade, ledger, notification_service)

                return ExecutionResult(
                    success=True,
                    symbol=symbol,
                    side="SELL",
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=raw_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=fee,
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    realized_pnl=trade.realized_pnl,
                    trade_record=trade,
                )
            except Exception as ex:
                err = f"Revolut X Live SELL error: {ex}"
                logger.error(err)
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side="SELL",
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=raw_px,
                    quantity=qty,
                    quote_notional=notional,
                    fee=Decimal("0.0"),
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    error_message=err,
                )

        return ExecutionResult(
            success=False,
            symbol=symbol,
            side=side_val,
            order_type=order_type,
            raw_price=raw_px,
            executed_price=raw_px,
            quantity=Decimal("0.0"),
            quote_notional=Decimal("0.0"),
            fee=Decimal("0.0"),
            fee_rate=self.fee_rate,
            slippage=self.slippage,
            error_message=f"Unsupported order side: {side_val}",
        )

    def close(self) -> None:
        """Close underlying HTTP client if owned."""
        if self._owns_client and hasattr(self, "_client") and self._client:
            try:
                self._client.close()
            except Exception:
                pass

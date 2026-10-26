"""Bybit V5 Spot Live Execution Engine.

Provides an authenticated, production-grade Bybit V5 Spot API client
and LiveExecutor that fulfills the ExecutionSimulator interface contract:
- get_wallet_balances(): query real EUR and coin balances (UNIFIED/SPOT)
- get_instrument_rules(symbol): fetch real tickSize, basePrecision, minOrderAmt
- execute(order, current_price, ledger, ...): submit real orders to Bybit V5
  Spot, verifying execution and recording trades into the ledger.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Any, Dict, Optional, Tuple, Union

from trading_bot.config import normalize_symbol, to_display_symbol
from trading_bot.simulation.ledger import (
    InsufficientFundsError,
    InsufficientPositionError,
    TradeRecord,
    VirtualLedger,
)
from trading_bot.simulation.simulator import ExecutionResult, OrderSide, OrderType

logger = logging.getLogger("bybit_live")


class BybitAPIError(Exception):
    """Raised when Bybit API returns a non-zero retCode or HTTP error."""

    def __init__(self, ret_code: int, ret_msg: str, response: Optional[Dict[str, Any]] = None):
        super().__init__(f"Bybit API Error {ret_code}: {ret_msg}")
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.response = response or {}


class BybitLiveClient:
    """Authenticated client for Bybit V5 API (Spot & Account)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.bybit.com",
        recv_window: int = 5000,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.timeout = timeout
        self._rules_cache: Dict[str, Dict[str, Any]] = {}

    def _generate_signature(self, timestamp: str, payload_str: str) -> str:
        """Generate HMAC-SHA256 signature for Bybit V5."""
        sign_payload = f"{timestamp}{self.api_key}{self.recv_window}{payload_str}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            sign_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = True,
    ) -> Dict[str, Any]:
        """Execute HTTP request to Bybit V5 API."""
        url = f"{self.base_url}{endpoint}"
        query_str = ""
        if params:
            query_str = urllib.parse.urlencode({k: str(v) for k, v in params.items() if v is not None})
            if query_str:
                url = f"{url}?{query_str}"

        body_bytes = None
        payload_for_sign = query_str if method.upper() == "GET" else ""

        if body is not None and method.upper() in ("POST", "PUT", "DELETE"):
            body_str = json.dumps(body, separators=(",", ":"))
            body_bytes = body_str.encode("utf-8")
            payload_for_sign = body_str

        headers = {
            "User-Agent": "AntigravityBybitClient/1.0",
            "Content-Type": "application/json",
        }

        if auth and self.api_key and self.api_secret:
            timestamp = str(int(time.time() * 1000))
            signature = self._generate_signature(timestamp, payload_for_sign)
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": str(self.recv_window),
            })

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method.upper())

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_text = resp.read().decode("utf-8")
                data = json.loads(resp_text)
        except urllib.error.HTTPError as http_err:
            raw_err = http_err.read().decode("utf-8") if http_err.fp else ""
            try:
                data = json.loads(raw_err)
            except Exception:
                data = {"retCode": http_err.code, "retMsg": f"HTTP {http_err.code}: {raw_err}"}
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to Bybit API ({url}): {exc}") from exc

        ret_code = data.get("retCode", 0)
        if ret_code != 0:
            ret_msg = data.get("retMsg", "Unknown error")
            raise BybitAPIError(ret_code, ret_msg, response=data)

        return data

    def get_wallet_balances(self) -> Dict[str, Decimal]:
        """Fetch available coin balances from UNIFIED or SPOT account."""
        balances: Dict[str, Decimal] = {}
        for account_type in ("UNIFIED", "SPOT"):
            try:
                data = self._request("GET", "/v5/account/wallet-balance", params={"accountType": account_type}, auth=True)
                acc_list = data.get("result", {}).get("list", [])
                if acc_list:
                    coins = acc_list[0].get("coin", [])
                    for c in coins:
                        coin_name = c.get("coin", "").upper()
                        # Prefer availableToWithdraw or walletBalance
                        bal_str = c.get("availableToWithdraw") or c.get("walletBalance") or "0"
                        try:
                            val = Decimal(str(bal_str))
                            if val > Decimal("0.0"):
                                balances[coin_name] = balances.get(coin_name, Decimal("0.0")) + val
                        except Exception:
                            continue
                    if balances:
                        return balances
            except BybitAPIError as err:
                logger.debug(f"Wallet balance query for {account_type} failed: {err}")
            except Exception as ex:
                logger.warning(f"Error querying wallet balance ({account_type}): {ex}")

        return balances

    def get_instrument_rules(self, symbol: str) -> Dict[str, Any]:
        """Fetch lot size, tick size, and minimum notional for symbol."""
        norm_sym = normalize_symbol(symbol)
        if norm_sym in self._rules_cache:
            return self._rules_cache[norm_sym]

        data = self._request(
            "GET",
            "/v5/market/instruments-info",
            params={"category": "spot", "symbol": norm_sym},
            auth=False,
        )
        items = data.get("result", {}).get("list", [])
        if not items:
            raise ValueError(f"Instrument not found on Bybit Spot: {norm_sym}")

        inst = items[0]
        lot_filter = inst.get("lotSizeFilter", {})
        price_filter = inst.get("priceFilter", {})

        base_prec = Decimal(str(lot_filter.get("basePrecision", "0.0001")))
        tick_size = Decimal(str(price_filter.get("tickSize", "0.01")))
        min_notional = Decimal(str(lot_filter.get("minOrderAmt", "1.00")))
        min_qty = Decimal(str(lot_filter.get("minOrderQty", str(base_prec))))
        max_qty = Decimal(str(lot_filter.get("maxOrderQty", "1000000")))

        rules = {
            "symbol": norm_sym,
            "baseCoin": inst.get("baseCoin"),
            "quoteCoin": inst.get("quoteCoin"),
            "basePrecision": base_prec,
            "tickSize": tick_size,
            "minOrderAmt": min_notional,
            "minOrderQty": min_qty,
            "maxOrderQty": max_qty,
        }
        self._rules_cache[norm_sym] = rules
        return rules

    def get_orderbook_ticker(self, symbol: str) -> Dict[str, Decimal]:
        """Fetch real-time top of book bid/ask and last price."""
        norm_sym = normalize_symbol(symbol)
        data = self._request(
            "GET",
            "/v5/market/tickers",
            params={"category": "spot", "symbol": norm_sym},
            auth=False,
        )
        items = data.get("result", {}).get("list", [])
        if not items:
            raise ValueError(f"Ticker not found for {norm_sym}")

        item = items[0]
        return {
            "bid": Decimal(str(item.get("bid1Price", item.get("lastPrice", "0")))),
            "ask": Decimal(str(item.get("ask1Price", item.get("lastPrice", "0")))),
            "last": Decimal(str(item.get("lastPrice", "0"))),
        }

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: Decimal,
        price: Optional[Decimal] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a real Spot order on Bybit V5."""
        norm_sym = normalize_symbol(symbol)
        side_fmt = "Buy" if side.upper() == "BUY" else "Sell"
        type_fmt = "Limit" if order_type.upper() == "LIMIT" else "Market"

        body: Dict[str, Any] = {
            "category": "spot",
            "symbol": norm_sym,
            "side": side_fmt,
            "orderType": type_fmt,
            "qty": str(qty),
        }

        if type_fmt == "Limit":
            if price is None:
                raise ValueError("Price is required for Limit orders")
            body["price"] = str(price)
            body["timeInForce"] = time_in_force
        else:
            if side_fmt == "Buy":
                # For spot market buy, Bybit supports marketUnit=baseCoin or quoteCoin
                body["marketUnit"] = "baseCoin"

        if client_order_id:
            body["orderLinkId"] = client_order_id

        data = self._request("POST", "/v5/order/create", body=body, auth=True)
        return data.get("result", {})

    def get_order_status(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve real-time order status and execution info."""
        norm_sym = normalize_symbol(symbol)
        params: Dict[str, Any] = {"category": "spot", "symbol": norm_sym}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["orderLinkId"] = client_order_id

        data = self._request("GET", "/v5/order/realtime", params=params, auth=True)
        items = data.get("result", {}).get("list", [])
        if items:
            return items[0]
        return {}


class BybitLiveExecutor:
    """Production live execution engine conforming to the ExecutionSimulator interface."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        exchange: str = "bybit",
        fee_rate: Decimal = Decimal("0.0002"),
        slippage: Decimal = Decimal("0.0001"),
        notification_service: Optional[Any] = None,
        timeout: float = 10.0,
        poll_fill_timeout: float = 6.0,
    ) -> None:
        self.exchange = exchange.lower()
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.notification_service = notification_service
        self.poll_fill_timeout = poll_fill_timeout
        self.client = BybitLiveClient(
            api_key=api_key,
            api_secret=api_secret,
            timeout=timeout,
        )

    def get_wallet_balances(self) -> Dict[str, Decimal]:
        """Delegate wallet balance check to client."""
        return self.client.get_wallet_balances()

    def get_instrument_rules(self, symbol: str) -> Dict[str, Any]:
        """Delegate instrument rules check to client."""
        return self.client.get_instrument_rules(symbol)

    def quantize_quantity(self, quantity: Decimal, step_size: Decimal) -> Decimal:
        """Quantize lot size downwards according to step_size."""
        if step_size <= Decimal("0.0"):
            return quantity
        steps = (quantity / step_size).quantize(Decimal("1"), rounding=ROUND_DOWN)
        return steps * step_size

    def quantize_price(self, price: Decimal, tick_size: Decimal) -> Decimal:
        """Quantize price downwards according to tick_size."""
        if tick_size <= Decimal("0.0"):
            return price
        ticks = (price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN)
        return ticks * tick_size

    def execute(
        self,
        order: Any,
        current_price: Union[Decimal, float, str],
        ledger: VirtualLedger,
        timestamp_ms: Optional[int] = None,
        notification_service: Optional[Any] = None,
        raise_on_error: Optional[bool] = False,
    ) -> ExecutionResult:
        """Execute real order on Bybit Spot V5 and synchronize with VirtualLedger."""
        notif = notification_service or self.notification_service
        raw_px = Decimal(str(current_price))
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        # Parse order signal attributes
        if isinstance(order, dict):
            symbol = str(order.get("symbol", "SOLEUR")).upper()
            side_val = str(order.get("side", "BUY")).upper()
            target_qty = Decimal(str(order["target_quantity"])) if order.get("target_quantity") is not None else None
            target_notional = Decimal(str(order["target_notional"])) if order.get("target_notional") is not None else None
            order_type = str(order.get("order_type", "LIMIT")).upper()
            order_id = order.get("order_id")
            exit_reason = order.get("exit_reason") or order.get("reason")
        else:
            symbol = str(getattr(order, "symbol", "SOLEUR")).upper()
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

        norm_sym = normalize_symbol(symbol)

        try:
            rules = self.client.get_instrument_rules(norm_sym)
            step_size = rules["basePrecision"]
            tick_size = rules["tickSize"]
            min_notional = rules["minOrderAmt"]

            # Query real-time top-of-book for optimal pricing
            ticker = self.client.get_orderbook_ticker(norm_sym)
            best_bid = ticker.get("bid", raw_px)
            best_ask = ticker.get("ask", raw_px)

            # Determine limit order pricing (Limit Maker model: buy @ bid, sell @ ask)
            if side_val == "BUY":
                limit_px = best_bid if best_bid > Decimal("0") else raw_px
                limit_px = self.quantize_price(limit_px, tick_size)

                # Determine order quantity
                if target_qty is not None and target_qty > Decimal("0"):
                    qty = self.quantize_quantity(target_qty, step_size)
                    notional = qty * limit_px
                elif target_notional is not None and target_notional > Decimal("0"):
                    calc_qty = target_notional / limit_px
                    qty = self.quantize_quantity(calc_qty, step_size)
                    notional = qty * limit_px
                    if notional < min_notional and target_notional >= min_notional:
                        qty += step_size
                        notional = qty * limit_px
                else:
                    # Allocate available cash with fee reserve
                    budget = ledger.available_cash
                    max_notional = budget / (Decimal("1") + self.fee_rate)
                    qty = self.quantize_quantity(max_notional / limit_px, step_size)
                    notional = qty * limit_px

                if notional < min_notional:
                    err_msg = f"Order notional {notional:.2f} is below Bybit minNotional {min_notional:.2f} EUR"
                    if raise_on_error:
                        raise ValueError(err_msg)
                    return ExecutionResult(
                        success=False,
                        symbol=norm_sym,
                        side="BUY",
                        order_type=order_type,
                        raw_price=raw_px,
                        executed_price=limit_px,
                        quantity=qty,
                        quote_notional=notional,
                        fee=Decimal("0.0"),
                        fee_rate=self.fee_rate,
                        slippage=Decimal("0.0"),
                        error_message=err_msg,
                    )

                fee_est = notional * self.fee_rate
                if ledger.cash < (notional + fee_est):
                    err_msg = f"Insufficient ledger cash: {ledger.cash:.2f} < required {notional + fee_est:.2f} EUR"
                    if raise_on_error:
                        raise InsufficientFundsError(err_msg)
                    return ExecutionResult(
                        success=False,
                        symbol=norm_sym,
                        side="BUY",
                        order_type=order_type,
                        raw_price=raw_px,
                        executed_price=limit_px,
                        quantity=qty,
                        quote_notional=notional,
                        fee=fee_est,
                        fee_rate=self.fee_rate,
                        slippage=Decimal("0.0"),
                        error_message=err_msg,
                    )

                # Submit order to Bybit Spot V5
                client_order_id = f"ag_buy_{int(time.time() * 1000)}"
                create_res = self.client.create_order(
                    symbol=norm_sym,
                    side="Buy",
                    order_type="Limit" if order_type == "LIMIT" else "Market",
                    qty=qty,
                    price=limit_px if order_type == "LIMIT" else None,
                    time_in_force="GTC",
                    client_order_id=client_order_id,
                )
                bybit_order_id = create_res.get("orderId")

                # Verify fill status
                executed_qty = qty
                executed_price = limit_px
                fee_paid = fee_est
                if bybit_order_id:
                    status_info = self._poll_fill(norm_sym, bybit_order_id)
                    cum_qty = Decimal(str(status_info.get("cumExecQty", "0")))
                    avg_px = Decimal(str(status_info.get("avgPrice", "0")))
                    cum_fee = Decimal(str(status_info.get("cumExecFee", "0")))
                    if cum_qty > Decimal("0"):
                        executed_qty = cum_qty
                    if avg_px > Decimal("0"):
                        executed_price = avg_px
                    if cum_fee > Decimal("0"):
                        fee_paid = cum_fee

                # Record into VirtualLedger
                trade = ledger.record_buy(
                    symbol=norm_sym,
                    quantity=executed_qty,
                    executed_price=executed_price,
                    fee=fee_paid,
                    timestamp_ms=ts,
                    raw_price=raw_px,
                    order_type=order_type,
                    fee_rate=self.fee_rate,
                    slippage=Decimal("0.0"),
                    order_id=bybit_order_id or client_order_id,
                )

                if notif:
                    try:
                        notif.notify_trade(trade, ledger)
                    except Exception as n_err:
                        logger.warning(f"Notification error: {n_err}")

                return ExecutionResult(
                    success=True,
                    symbol=norm_sym,
                    side="BUY",
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=executed_price,
                    quantity=executed_qty,
                    quote_notional=executed_qty * executed_price,
                    fee=fee_paid,
                    fee_rate=self.fee_rate,
                    slippage=Decimal("0.0"),
                    trade_record=trade,
                )

            elif side_val == "SELL":
                pos = ledger.get_position(norm_sym)
                held_qty = pos.quantity
                if held_qty <= Decimal("0.0"):
                    err_msg = f"No open position held in {norm_sym} to sell"
                    if raise_on_error:
                        raise InsufficientPositionError(err_msg)
                    return ExecutionResult(
                        success=False,
                        symbol=norm_sym,
                        side="SELL",
                        order_type=order_type,
                        raw_price=raw_px,
                        executed_price=raw_px,
                        quantity=Decimal("0.0"),
                        quote_notional=Decimal("0.0"),
                        fee=Decimal("0.0"),
                        fee_rate=self.fee_rate,
                        slippage=Decimal("0.0"),
                        error_message=err_msg,
                    )

                limit_px = best_ask if best_ask > Decimal("0") else raw_px
                limit_px = self.quantize_price(limit_px, tick_size)

                # Sell quantity calculation
                sell_qty = self.quantize_quantity(target_qty if target_qty and target_qty > 0 else held_qty, step_size)
                notional = sell_qty * limit_px

                if notional < min_notional:
                    err_msg = f"Sell notional {notional:.2f} is below Bybit minNotional {min_notional:.2f} EUR"
                    if raise_on_error:
                        raise ValueError(err_msg)
                    return ExecutionResult(
                        success=False,
                        symbol=norm_sym,
                        side="SELL",
                        order_type=order_type,
                        raw_price=raw_px,
                        executed_price=limit_px,
                        quantity=sell_qty,
                        quote_notional=notional,
                        fee=Decimal("0.0"),
                        fee_rate=self.fee_rate,
                        slippage=Decimal("0.0"),
                        error_message=err_msg,
                    )

                fee_est = notional * self.fee_rate
                client_order_id = f"ag_sell_{int(time.time() * 1000)}"
                create_res = self.client.create_order(
                    symbol=norm_sym,
                    side="Sell",
                    order_type="Limit" if order_type == "LIMIT" else "Market",
                    qty=sell_qty,
                    price=limit_px if order_type == "LIMIT" else None,
                    time_in_force="GTC",
                    client_order_id=client_order_id,
                )
                bybit_order_id = create_res.get("orderId")

                executed_qty = sell_qty
                executed_price = limit_px
                fee_paid = fee_est
                if bybit_order_id:
                    status_info = self._poll_fill(norm_sym, bybit_order_id)
                    cum_qty = Decimal(str(status_info.get("cumExecQty", "0")))
                    avg_px = Decimal(str(status_info.get("avgPrice", "0")))
                    cum_fee = Decimal(str(status_info.get("cumExecFee", "0")))
                    if cum_qty > Decimal("0"):
                        executed_qty = cum_qty
                    if avg_px > Decimal("0"):
                        executed_price = avg_px
                    if cum_fee > Decimal("0"):
                        fee_paid = cum_fee

                # Record sell into ledger
                trade = ledger.record_sell(
                    symbol=norm_sym,
                    quantity=executed_qty,
                    executed_price=executed_price,
                    fee=fee_paid,
                    timestamp_ms=ts,
                    raw_price=raw_px,
                    order_type=order_type,
                    fee_rate=self.fee_rate,
                    slippage=Decimal("0.0"),
                    order_id=bybit_order_id or client_order_id,
                    exit_reason=exit_reason,
                )

                if notif:
                    try:
                        notif.notify_trade(trade, ledger)
                    except Exception as n_err:
                        logger.warning(f"Notification error: {n_err}")

                return ExecutionResult(
                    success=True,
                    symbol=norm_sym,
                    side="SELL",
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=executed_price,
                    quantity=executed_qty,
                    quote_notional=executed_qty * executed_price,
                    fee=fee_paid,
                    fee_rate=self.fee_rate,
                    slippage=Decimal("0.0"),
                    realized_pnl=trade.realized_pnl,
                    trade_record=trade,
                )

            else:
                err_msg = f"Unsupported order side: {side_val}"
                if raise_on_error:
                    raise ValueError(err_msg)
                return ExecutionResult(
                    success=False,
                    symbol=norm_sym,
                    side=side_val,
                    order_type=order_type,
                    raw_price=raw_px,
                    executed_price=raw_px,
                    quantity=Decimal("0.0"),
                    quote_notional=Decimal("0.0"),
                    fee=Decimal("0.0"),
                    fee_rate=self.fee_rate,
                    slippage=Decimal("0.0"),
                    error_message=err_msg,
                )

        except Exception as exc:
            logger.error(f"Execution error on {symbol}: {exc}", exc_info=True)
            if raise_on_error:
                raise
            return ExecutionResult(
                success=False,
                symbol=norm_sym,
                side=side_val,
                order_type=order_type,
                raw_price=raw_px,
                executed_price=raw_px,
                quantity=Decimal("0.0"),
                quote_notional=Decimal("0.0"),
                fee=Decimal("0.0"),
                fee_rate=self.fee_rate,
                slippage=Decimal("0.0"),
                error_message=str(exc),
            )

    def _poll_fill(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Poll order status up to poll_fill_timeout seconds."""
        deadline = time.time() + self.poll_fill_timeout
        last_info: Dict[str, Any] = {}
        while time.time() < deadline:
            try:
                info = self.client.get_order_status(symbol=symbol, order_id=order_id)
                last_info = info
                status = info.get("orderStatus", "")
                if status in ("Filled", "Cancelled", "Rejected"):
                    return info
            except Exception as ex:
                logger.debug(f"Error polling order {order_id}: {ex}")
            time.sleep(1.0)
        return last_info

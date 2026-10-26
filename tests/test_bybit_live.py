"""Unit and Integration Tests for Bybit Live Trading Execution Engine."""

import hashlib
import hmac
import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading_bot.execution.bybit_live import (
    BybitAPIError,
    BybitLiveClient,
    BybitLiveExecutor,
)
from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import OrderSide, OrderSignal, OrderType


def test_bybit_signature_generation():
    """Verify exact HMAC-SHA256 signature formatting conforming to Bybit V5 specifications."""
    api_key = "test_key_123"
    api_secret = "test_secret_abc"
    timestamp = "1658385579423"
    recv_window = 5000
    payload = '{"category":"spot","symbol":"SOLEUR","side":"Buy","orderType":"Limit","qty":"0.1"}'

    client = BybitLiveClient(api_key=api_key, api_secret=api_secret, recv_window=recv_window)
    sig = client._generate_signature(timestamp, payload)

    expected_sign_payload = f"{timestamp}{api_key}{recv_window}{payload}"
    expected_sig = hmac.new(
        api_secret.encode("utf-8"),
        expected_sign_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    assert sig == expected_sig
    assert len(sig) == 64


def test_bybit_wallet_balances_parsing():
    """Verify wallet balance extraction from Bybit UNIFIED account structure."""
    client = BybitLiveClient(api_key="k", api_secret="s")

    mock_resp = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {
                    "accountType": "UNIFIED",
                    "totalEquity": "150.25",
                    "coin": [
                        {"coin": "EUR", "walletBalance": "50.00", "availableToWithdraw": "50.00"},
                        {"coin": "SOL", "walletBalance": "1.25", "availableToWithdraw": "1.25"},
                        {"coin": "BTC", "walletBalance": "0.00", "availableToWithdraw": "0.00"},
                    ],
                }
            ]
        },
    }

    with patch.object(client, "_request", return_value=mock_resp):
        bals = client.get_wallet_balances()
        assert bals["EUR"] == Decimal("50.00")
        assert bals["SOL"] == Decimal("1.25")
        assert "BTC" not in bals  # Zero balances excluded


def test_bybit_instrument_rules_parsing():
    """Verify lot size and price filter parsing and caching."""
    client = BybitLiveClient(api_key="k", api_secret="s")

    mock_resp = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {
                    "symbol": "SOLEUR",
                    "baseCoin": "SOL",
                    "quoteCoin": "EUR",
                    "lotSizeFilter": {
                        "basePrecision": "0.0001",
                        "minOrderQty": "0.0001",
                        "maxOrderQty": "30960",
                        "minOrderAmt": "1",
                    },
                    "priceFilter": {
                        "tickSize": "0.01",
                    },
                }
            ]
        },
    }

    with patch.object(client, "_request", return_value=mock_resp):
        rules = client.get_instrument_rules("SOLEUR")
        assert rules["symbol"] == "SOLEUR"
        assert rules["baseCoin"] == "SOL"
        assert rules["basePrecision"] == Decimal("0.0001")
        assert rules["tickSize"] == Decimal("0.01")
        assert rules["minOrderAmt"] == Decimal("1")


def test_live_executor_quantization():
    """Verify price and quantity quantization downward to step sizes."""
    executor = BybitLiveExecutor(api_key="k", api_secret="s")

    # Quantity quantization: 0.123456 with step 0.0001 -> 0.1234
    qty = Decimal("0.123456")
    q_qty = executor.quantize_quantity(qty, Decimal("0.0001"))
    assert q_qty == Decimal("0.1234")

    # Price quantization: 92.5678 with tick 0.01 -> 92.56
    px = Decimal("92.5678")
    q_px = executor.quantize_price(px, Decimal("0.01"))
    assert q_px == Decimal("92.56")


def test_live_executor_buy_and_sell_order_execution():
    """Test full BUY and SELL execution flow against VirtualLedger."""
    executor = BybitLiveExecutor(api_key="k", api_secret="s", poll_fill_timeout=0.1)

    # Mock client methods
    executor.client.get_instrument_rules = MagicMock(return_value={
        "symbol": "SOLEUR",
        "baseCoin": "SOL",
        "quoteCoin": "EUR",
        "basePrecision": Decimal("0.001"),
        "tickSize": Decimal("0.01"),
        "minOrderAmt": Decimal("1.00"),
        "minOrderQty": Decimal("0.001"),
        "maxOrderQty": Decimal("1000"),
    })
    executor.client.get_orderbook_ticker = MagicMock(return_value={
        "bid": Decimal("100.00"),
        "ask": Decimal("100.50"),
        "last": Decimal("100.25"),
    })
    executor.client.create_order = MagicMock(return_value={"orderId": "bybit_order_1001"})
    executor.client.get_order_status = MagicMock(return_value={
        "orderStatus": "Filled",
        "cumExecQty": "0.200",
        "avgPrice": "100.00",
        "cumExecFee": "0.004",
    })

    ledger = VirtualLedger(initial_cash=Decimal("50.00"), quote_currency="EUR")

    # 1. Execute BUY of 20 EUR worth of SOL
    buy_order = OrderSignal(
        symbol="SOL/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("20.00"),
        order_type=OrderType.LIMIT,
    )

    buy_res = executor.execute(order=buy_order, current_price=Decimal("100.00"), ledger=ledger)
    assert buy_res.success is True
    assert buy_res.side == "BUY"
    assert buy_res.quantity == Decimal("0.200")
    assert buy_res.executed_price == Decimal("100.00")
    assert ledger.cash < Decimal("50.00")
    assert ledger.get_position("SOLEUR").quantity == Decimal("0.200")

    # 2. Execute SELL to close position
    executor.client.get_orderbook_ticker = MagicMock(return_value={
        "bid": Decimal("110.00"),
        "ask": Decimal("110.50"),
        "last": Decimal("110.25"),
    })
    executor.client.get_order_status = MagicMock(return_value={
        "orderStatus": "Filled",
        "cumExecQty": "0.200",
        "avgPrice": "110.50",
        "cumExecFee": "0.0044",
    })

    sell_order = OrderSignal(
        symbol="SOL/EUR",
        side=OrderSide.SELL,
        target_quantity=Decimal("0.200"),
        order_type=OrderType.LIMIT,
    )

    sell_res = executor.execute(order=sell_order, current_price=Decimal("110.50"), ledger=ledger)
    assert sell_res.success is True
    assert sell_res.side == "SELL"
    assert sell_res.realized_pnl > Decimal("0.0")
    assert ledger.get_position("SOLEUR").quantity == Decimal("0.0")
    assert ledger.cash > Decimal("50.00")  # Profitable round trip!


def test_live_executor_min_notional_rejection():
    """Verify rejection when order size is smaller than Bybit minOrderAmt."""
    executor = BybitLiveExecutor(api_key="k", api_secret="s")
    executor.client.get_instrument_rules = MagicMock(return_value={
        "symbol": "SOLEUR",
        "basePrecision": Decimal("0.001"),
        "tickSize": Decimal("0.01"),
        "minOrderAmt": Decimal("1.00"),
    })
    executor.client.get_orderbook_ticker = MagicMock(return_value={
        "bid": Decimal("100.00"),
        "ask": Decimal("100.50"),
        "last": Decimal("100.00"),
    })

    ledger = VirtualLedger(initial_cash=Decimal("50.00"), quote_currency="EUR")
    small_order = OrderSignal(
        symbol="SOL/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("0.50"),  # Less than 1.00 EUR minOrderAmt
        order_type=OrderType.LIMIT,
    )

    res = executor.execute(order=small_order, current_price=Decimal("100.00"), ledger=ledger)
    assert res.success is False
    assert "below Bybit minNotional" in res.error_message

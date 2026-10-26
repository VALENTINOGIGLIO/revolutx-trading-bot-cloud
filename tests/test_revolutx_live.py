"""
Tests for Revolut X Crypto Exchange Live Execution Engine & Gateway Connector.

Verifies:
- Ed25519 cryptographic key loading and signature generation according to Revolut X specs.
- Signature verification with public key.
- Header formatting (X-Revx-API-Key, X-Revx-Timestamp, X-Revx-Signature).
- Balance queries and Auto-Deposit Detector (detects incoming EUR deposits).
- Limit Maker order generation with 0.00% fee and post_only execution instructions.
- Full ledger bookkeeping and dual-mode execution (Live and Simulation).
"""

import base64
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import httpx
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trading_bot.simulation.ledger import VirtualLedger
from trading_bot.simulation.simulator import OrderSignal, OrderSide, OrderType
from trading_bot.simulation.revolutx_live import (
    RevolutXLiveExecutor,
    to_revolut_symbol,
    from_revolut_symbol,
    REVOLUTX_MAKER_FEE_RATE,
    REVOLUTX_TAKER_FEE_RATE,
    REVOLUTX_MIN_NOTIONAL_EUR,
)


class MockTransport(httpx.BaseTransport):
    """Custom HTTP transport to mock Revolut X API responses."""

    def __init__(self, responses: Dict[Tuple[str, str], Any]):
        super().__init__()
        self.responses = responses
        self.recorded_requests = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        key = (request.method.upper(), request.url.path)

        if key in self.responses:
            item = self.responses[key]
            if callable(item):
                status_code, body = item(request)
            else:
                status_code, body = item
            return httpx.Response(
                status_code=status_code,
                json=body if isinstance(body, (dict, list)) else None,
                content=json.dumps(body).encode() if isinstance(body, (dict, list)) else b"",
                request=request,
            )

        return httpx.Response(status_code=404, json={"error": "Not Found"}, request=request)


def test_symbol_conversion():
    assert to_revolut_symbol("BTCEUR") == "BTC-EUR"
    assert to_revolut_symbol("BTC/EUR") == "BTC-EUR"
    assert to_revolut_symbol("DOGEEUR") == "DOGE-EUR"
    assert to_revolut_symbol("SOL-EUR") == "SOL-EUR"
    assert to_revolut_symbol("ETH/USDT") == "ETH-USDT"

    assert from_revolut_symbol("BTC-EUR") == "BTCEUR"
    assert from_revolut_symbol("DOGE-EUR") == "DOGEEUR"


def test_ed25519_key_generation_and_signature_verification():
    # Generate test keypair
    priv_key = ed25519.Ed25519PrivateKey.generate()
    priv_pem = priv_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_key = priv_key.public_key()

    executor = RevolutXLiveExecutor(
        api_key="test_api_key_64_characters_long_abcdefghijklmnopqrstuvwxyz012345",
        private_key_pem=priv_pem,
        live=False,
    )

    assert executor.has_auth is True

    # Test signing format: timestamp + method + path + query + body
    ts = 1765360896219
    method = "POST"
    path = "/api/1.0/orders"
    query = ""
    body = '{"symbol":"BTC-EUR","side":"buy"}'

    ts_str, sig_b64 = executor.sign_request(
        method=method,
        path=path,
        query=query,
        body=body,
        timestamp_ms=ts,
    )

    assert ts_str == str(ts)
    raw_sig = base64.b64decode(sig_b64)
    assert len(raw_sig) == 64

    # Verify signature mathematically using Ed25519 public key
    expected_msg = f"{ts}{method}{path}{query}{body}".encode("utf-8")
    pub_key.verify(raw_sig, expected_msg)

    # Verify headers
    headers = executor.get_auth_headers(method=method, path=path, body=body, timestamp_ms=ts)
    assert headers["X-Revx-API-Key"] == "test_api_key_64_characters_long_abcdefghijklmnopqrstuvwxyz012345"
    assert headers["X-Revx-Timestamp"] == str(ts)
    assert headers["X-Revx-Signature"] == sig_b64
    assert headers["Content-Type"] == "application/json"


def test_keys_generated_in_project_keys_directory():
    """Verify generated project keys in keys/ are valid Ed25519 keys."""
    keys_dir = PROJECT_ROOT / "keys"
    priv_path = keys_dir / "revolutx_private.pem"
    pub_path = keys_dir / "revolutx_public.pem"

    assert priv_path.exists(), "revolutx_private.pem must exist"
    assert pub_path.exists(), "revolutx_public.pem must exist"

    priv_bytes = priv_path.read_bytes()
    loaded_priv = serialization.load_pem_private_key(priv_bytes, password=None)
    assert isinstance(loaded_priv, ed25519.Ed25519PrivateKey)

    pub_bytes = pub_path.read_bytes()
    loaded_pub = serialization.load_pem_public_key(pub_bytes)
    assert isinstance(loaded_pub, ed25519.Ed25519PublicKey)


def test_balance_queries_and_auto_deposit_detector():
    balances_state = [
        {"currency": "EUR", "available": "0.00"},
        {"currency": "BTC", "available": "0.0005"},
    ]

    def balances_handler(req):
        return 200, balances_state

    transport = MockTransport({("GET", "/api/1.0/balances"): balances_handler})
    client = httpx.Client(transport=transport)

    priv_key = ed25519.Ed25519PrivateKey.generate()
    priv_pem = priv_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    executor = RevolutXLiveExecutor(
        api_key="mock_key",
        private_key_pem=priv_pem,
        http_client=client,
        live=True,
    )

    ledger = VirtualLedger(initial_cash=Decimal("10.00"), quote_currency="EUR")

    # Step 1: Initial balance is 10.00 EUR
    assert executor.fetch_eur_balance() == Decimal("0.00")
    bals = executor.get_wallet_balances()
    assert bals["EUR"] == Decimal("0.00")
    assert bals["BTC"] == Decimal("0.0005")

    # Initial auto-deposit check sets baseline
    dep1 = executor.check_auto_deposit(ledger=ledger)
    assert dep1 is None
    assert ledger.cash == Decimal("10.00")

    # Step 2: User deposits 50.00 EUR from Revolut into Revolut X!
    balances_state[0]["available"] = "50.00"

    dep2 = executor.check_auto_deposit(ledger=ledger)
    assert dep2 == Decimal("50.00")
    # Ledger should automatically have been credited with 50.00 EUR!
    assert ledger.cash == Decimal("60.00")
    assert ledger.initial_cash == Decimal("60.00")

    # Step 3: Next poll with same balance -> no duplicate deposit
    dep3 = executor.check_auto_deposit(ledger=ledger)
    assert dep3 is None
    assert ledger.cash == Decimal("60.00")

    # Step 4: Full sync_ledger test
    balances_state.append({"currency": "DOGE", "available": "150.0"})
    sync_res = executor.sync_ledger(ledger=ledger)
    assert sync_res["cash_eur"] == 50.0
    assert sync_res["crypto_positions"]["DOGE"] == 150.0
    assert ledger.get_position("DOGEEUR").quantity == Decimal("150.0")


def test_limit_maker_order_execution_0_percent_fee():
    order_received = {}

    def order_handler(req):
        nonlocal order_received
        order_received = json.loads(req.content.decode("utf-8"))
        return 200, {
            "order_id": "revx_ord_123456",
            "client_order_id": order_received.get("client_order_id"),
            "status": "new",
        }

    transport = MockTransport({("POST", "/api/1.0/orders"): order_handler})
    client = httpx.Client(transport=transport)

    priv_key = ed25519.Ed25519PrivateKey.generate()
    priv_pem = priv_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    executor = RevolutXLiveExecutor(
        api_key="mock_key",
        private_key_pem=priv_pem,
        http_client=client,
        live=True,
        prefer_maker=True,  # 0.00% Maker fee
    )

    ledger = VirtualLedger(initial_cash=Decimal("50.00"), quote_currency="EUR")

    signal = OrderSignal(
        symbol="BTC/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("20.00"),
        order_type=OrderType.LIMIT,
    )

    result = executor.execute(
        order=signal,
        current_price=Decimal("80000.00"),
        ledger=ledger,
    )

    assert result.success is True
    assert result.fee == Decimal("0.00"), "Revolut X Maker fee must be 0.00%!"
    assert result.executed_price == Decimal("80000.00")
    assert result.order_type == "LIMIT"

    # Verify request payload sent to Revolut X API
    assert order_received["symbol"] == "BTC-EUR"
    assert order_received["side"] == "buy"
    limit_cfg = order_received["order_configuration"]["limit"]
    assert limit_cfg["execution_instructions"] == ["post_only"]
    assert Decimal(limit_cfg["price"]) == Decimal("80000.00")


def test_simulation_mode_with_revolut_x_friction_rules():
    """Verify that when live=False, RevolutXLiveExecutor provides high-fidelity simulation with 0.00% maker fee."""
    executor = RevolutXLiveExecutor(
        live=False,
        prefer_maker=True,
    )

    assert executor.fee_rate == Decimal("0.0000")
    assert executor.slippage == Decimal("0.0000")
    assert executor.min_notional == Decimal("1.00")

    ledger = VirtualLedger(initial_cash=Decimal("50.00"), quote_currency="EUR")

    signal = OrderSignal(
        symbol="DOGE/EUR",
        side=OrderSide.BUY,
        target_notional=Decimal("15.00"),
        order_type=OrderType.LIMIT,
    )

    res = executor.execute(
        order=signal,
        current_price=Decimal("0.1500"),
        ledger=ledger,
    )

    assert res.success is True
    assert res.fee == Decimal("0.00")
    assert res.slippage == Decimal("0.00")
    assert ledger.cash == Decimal("35.00")
    assert ledger.get_position("DOGE/EUR").quantity == Decimal("100.0")


def test_instrument_rules():
    executor = RevolutXLiveExecutor(live=False)
    btc_rules = executor.get_instrument_rules("BTC/EUR")
    assert btc_rules["symbol"] == "BTC-EUR"
    assert btc_rules["baseCoin"] == "BTC"
    assert btc_rules["quoteCoin"] == "EUR"
    assert btc_rules["minOrderAmt"] == Decimal("1.00")

    doge_rules = executor.get_instrument_rules("DOGEEUR")
    assert doge_rules["symbol"] == "DOGE-EUR"
    assert doge_rules["basePrecision"] == Decimal("1.0")


def test_paper_cli_revolutx_integration():
    from trading_bot.cli.paper import main
    exit_code = main([
        "--pair", "DOGE/EUR",
        "--balance", "50.0",
        "--exchange", "revolutx",
        "--iterations", "1",
        "--interval-sec", "0.01",
        "--dry-run",
    ])
    assert exit_code == 0


def test_revolutx_compounding_phase_at_50_eur():
    """Verify that under Revolut X (min_notional <= 1.0), 50 EUR equity uses Phase 2 Growth sizing."""
    from trading_bot.strategy.compounding import CompoundingPositionSizer, CompoundingPhase

    is_revx_or_low_notional = True
    p1_threshold = Decimal("10.00") if is_revx_or_low_notional else Decimal("100.00")

    sizer = CompoundingPositionSizer(
        phase_1_threshold=p1_threshold,
        min_notional=Decimal("1.00"),
        phase_2_risk_pct=0.04,
        phase_2_max_allocation_pct=0.85,
        default_step_size=Decimal("1.0"),
    )

    # 50.00 EUR equity must be in Phase 2 Growth
    assert sizer.get_phase(Decimal("50.00")) == CompoundingPhase.PHASE_2_GROWTH

    # Test position sizing with 50 EUR equity, entry 0.20 EUR, stop 0.19 EUR (5% stop loss)
    # 4% of 50€ = 2.00€ risk. Target notional = 2.00 / 0.05 = 40.00 EUR <= 42.50 EUR (85% cap)
    res = sizer.calculate_position_size(
        equity=Decimal("50.00"),
        entry_price=Decimal("0.20"),
        stop_loss=Decimal("0.19"),
    )
    assert res.is_valid is True
    assert res.phase == CompoundingPhase.PHASE_2_GROWTH
    assert res.risk_fraction == pytest.approx(0.04, rel=1e-2)
    assert res.notional <= Decimal("42.50")
    assert res.notional >= Decimal("39.00")



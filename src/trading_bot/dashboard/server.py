"""Dashboard Server Runner and Background Lifecycle Manager.

Wraps Uvicorn ASGI server execution in blocking mode or concurrent background
daemon thread mode with ephemeral port binding support.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Any, Optional, Union

import uvicorn

from trading_bot.dashboard.app import create_app
from trading_bot.dashboard.state import DashboardState

logger = logging.getLogger("DashboardServer")


class GracefulServer(uvicorn.Server):
    """Custom Uvicorn server coordinating application state shutdown before connection shutdown."""

    def __init__(self, config: uvicorn.Config, dash_state: Optional[DashboardState] = None) -> None:
        super().__init__(config)
        self._dash_state = dash_state

    async def shutdown(self, sockets: Optional[list] = None) -> None:
        if self._dash_state is not None:
            self._dash_state.shutdown()
        await super().shutdown(sockets=sockets)


class DashboardServer:
    """Uvicorn server runner and lifecycle manager for trading bot dashboard."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        state: Optional[DashboardState] = None,
        trades_csv: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
        symbol: str = "BTC/EUR",
    ) -> None:
        self.host: str = host
        self.port: int = port
        self._actual_port: int = port if port != 0 else 0
        self.state: DashboardState = state or DashboardState(symbol=symbol)

        if trades_csv:
            self.state.load_from_csv(trades_csv)
        if db_path:
            self.state.load_from_market_db(db_path=db_path, symbol=symbol)

        self.app = create_app(self.state)
        self._server: Optional[GracefulServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def actual_port(self) -> int:
        """Return the actual TCP port the server is bound to."""
        if self._server and getattr(self._server, "servers", None):
            for s in self._server.servers:
                for sock in getattr(s, "sockets", []):
                    try:
                        return sock.getsockname()[1]
                    except (OSError, IndexError):
                        pass
        return self._actual_port

    def run(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Run Uvicorn server synchronously (blocking)."""
        if host:
            self.host = host
        if port is not None:
            self.port = port
            self._actual_port = port

        if self.state and hasattr(self.state, "reset_shutdown"):
            self.state.reset_shutdown()

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            timeout_graceful_shutdown=1.0,
        )
        self._server = GracefulServer(config, dash_state=self.state)
        self._server.run()

    def start_background(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 5.0,
    ) -> int:
        """Start Uvicorn in a background daemon thread and wait for readiness.

        Returns the actual port bound (especially useful when port=0).
        """
        if host:
            self.host = host
        if port is not None:
            self.port = port
            self._actual_port = port

        if self.state and hasattr(self.state, "reset_shutdown"):
            self.state.reset_shutdown()

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            timeout_graceful_shutdown=1.0,
        )
        self._server = GracefulServer(config, dash_state=self.state)

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Wait for server readiness probe
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._thread.is_alive():
                raise RuntimeError(
                    f"DashboardServer background thread terminated prematurely (port {self.port} may be in use)"
                )
            if self._server.started and getattr(self._server, "servers", None):
                for s in self._server.servers:
                    for sock in getattr(s, "sockets", []):
                        try:
                            self._actual_port = sock.getsockname()[1]
                            return self._actual_port
                        except (OSError, IndexError):
                            pass
            time.sleep(0.02)

        return self.actual_port

    def stop(self, timeout: float = 3.0) -> None:
        """Gracefully shut down Uvicorn server and join background thread."""
        if self.state and hasattr(self.state, "shutdown"):
            self.state.shutdown()
        if self._server:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive() and self._server:
                logger.warning("DashboardServer thread failed to terminate within %.1fs; forcing exit", timeout)
                self._server.force_exit = True
                self._thread.join(timeout=1.0)

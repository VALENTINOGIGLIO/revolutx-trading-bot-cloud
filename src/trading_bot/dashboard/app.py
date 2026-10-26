"""Starlette ASGI Web Application for Autonomous Trading Bot Dashboard.

Serves the interactive Lightweight Charts single-page application, REST APIs
for market data and trade markers, and a Server-Sent Events (SSE) live stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from trading_bot.dashboard.state import DashboardState

logger = logging.getLogger("DashboardApp")

STATIC_DIR = Path(__file__).parent / "static"


def create_app(state: Optional[DashboardState] = None) -> Starlette:
    """Create and configure the Starlette ASGI dashboard application."""
    dash_state = state or DashboardState()

    async def index_endpoint(request: Request) -> Response:
        """Serve single-page trading dashboard index.html."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<!DOCTYPE html><html><head><title>Trading Bot Dashboard</title></head>"
            "<body><div id='chart'></div><script src='/static/js/app.js'></script></body></html>"
        )

    async def status_endpoint(request: Request) -> JSONResponse:
        """Return bot status, balances, and operational telemetry."""
        return JSONResponse(dash_state.get_status())

    async def market_data_endpoint(request: Request) -> JSONResponse:
        """Return historical OHLCV candlestick series for chart rendering."""
        limit_param = request.query_params.get("limit")
        limit: Optional[int] = None
        if limit_param:
            try:
                limit = int(limit_param)
            except ValueError:
                limit = None

        candles = dash_state.get_candles(limit=limit)
        return JSONResponse(candles)

    async def trades_endpoint(request: Request) -> JSONResponse:
        """Return trade markers formatted for TradingView Lightweight Charts (AC2)."""
        markers = dash_state.get_markers()
        return JSONResponse(markers)

    async def stream_endpoint(request: Request) -> StreamingResponse:
        """Stream real-time candle updates, trade fills, and status via Server-Sent Events."""
        queue = dash_state.subscribe()

        async def event_generator():
            try:
                # Send initial status event on connection
                init_data = json.dumps(dash_state.get_status())
                yield f"event: status\ndata: {init_data}\n\n".encode("utf-8")

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        # Wait for state broadcast with 15s ping keep-alive
                        msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                        if msg is None or (isinstance(msg, dict) and msg.get("event") == "shutdown"):
                            yield b"event: shutdown\ndata: {}\n\n"
                            break
                        evt = msg.get("event", "message")
                        data_str = json.dumps(msg.get("data", {}))
                        yield f"event: {evt}\ndata: {data_str}\n\n".encode("utf-8")
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            break
                        yield b": ping\n\n"
            except (asyncio.CancelledError, GeneratorExit):
                pass
            except Exception as ex:
                logger.debug(f"SSE stream generator error: {ex}")
            finally:
                dash_state.unsubscribe(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Route definitions
    routes = [
        Route("/", endpoint=index_endpoint, methods=["GET"]),
        Route("/api/status", endpoint=status_endpoint, methods=["GET"]),
        Route("/api/candles", endpoint=market_data_endpoint, methods=["GET"]),
        Route("/api/market-data", endpoint=market_data_endpoint, methods=["GET"]),
        Route("/api/trades", endpoint=trades_endpoint, methods=["GET"]),
        Route("/api/stream", endpoint=stream_endpoint, methods=["GET"]),
    ]

    # Mount static assets directory
    if STATIC_DIR.exists():
        routes.append(Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"))

    app = Starlette(debug=False, routes=routes)
    app.state.dashboard_state = dash_state
    return app

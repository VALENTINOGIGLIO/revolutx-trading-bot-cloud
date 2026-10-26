#!/bin/bash
# ==============================================================================
# Script per avviare il Trading Bot in modo persistente (Paper o Live)
# ==============================================================================

cd /Users/gi/teamwork_projects/trading_bot

# Usa /opt/homebrew/bin/python3 (con tutte le dipendenze installate) o .venv se presente
PYTHON_BIN="/opt/homebrew/bin/python3"
if [ -x ".venv/bin/python3" ]; then
    PYTHON_BIN="./.venv/bin/python3"
fi

export PYTHONPATH=src

# Carica le variabili d'ambiente da .env se presente
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

PAIRS="${TRADING_PAIRS:-SOL/EUR,ETH/EUR,BTC/EUR}"
BALANCE="${INITIAL_BALANCE:-50.0}"
TF="${TIMEFRAME:-4h}"
EXCH="${EXCHANGE:-revolutx}"

EXTRA_FLAGS=""
if [ "$TRADING_MODE" = "LIVE" ] || [ "$TRADING_MODE" = "live" ]; then
    EXTRA_FLAGS="--live"
fi

# Avvia il bot con caffeinate per impedire lo sleep e Dashboard Web su http://localhost:8000
exec /usr/bin/caffeinate -dims "$PYTHON_BIN" -m trading_bot.cli.paper \
    --balance "$BALANCE" \
    --pair "$PAIRS" \
    --timeframe "$TF" \
    --exchange "$EXCH" \
    --notify-channel stdout \
    --dashboard \
    --verbose \
    $EXTRA_FLAGS


#!/usr/bin/env bash
set -euo pipefail

# Launch the dashboard or API bound to localhost only by default.
#
# Security defaults:
#   - Binds to 127.0.0.1 (loopback) so the service is NOT exposed to your
#     local network or the internet. Override with HOST=0.0.0.0 only if you
#     intentionally want network access (and have set strong secrets).
#   - Loads secrets from a local .env file if present (never commit .env).

MODE="${1:-dashboard}"
HOST="${HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
API_PORT="${API_PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load local secrets if a .env file exists (KEY=VALUE lines).
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Warn if running with default/unset credentials.
DEFAULT_SECRET="dev-change-me"
if [[ "${ML_EQUITY_DASHBOARD_PASSWORD:-$DEFAULT_SECRET}" == "$DEFAULT_SECRET" \
   || "${ML_EQUITY_API_TOKEN:-$DEFAULT_SECRET}" == "$DEFAULT_SECRET" ]]; then
  echo "WARNING: using the default dev credentials." >&2
  echo "         Run 'make secrets' to generate strong secrets into .env before exposing this service." >&2
fi

if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]]; then
  echo "WARNING: binding to $HOST exposes this service beyond localhost." >&2
fi

if [[ "$MODE" == "api" ]]; then
  if ! command -v uvicorn >/dev/null 2>&1; then
    echo "uvicorn is not installed. Run: pip install -r requirements.txt"
    exit 1
  fi
  uvicorn src.serving:app --host "$HOST" --port "$API_PORT" --reload
elif [[ "$MODE" == "dashboard" ]]; then
  if ! command -v streamlit >/dev/null 2>&1; then
    echo "streamlit is not installed. Run: pip install -r requirements.txt"
    exit 1
  fi
  streamlit run dashboard.py \
    --server.port "$DASHBOARD_PORT" \
    --server.address "$HOST" \
    --browser.gatherUsageStats false
else
  echo "Usage: ./launch.sh [dashboard|api]"
  exit 1
fi

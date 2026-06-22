#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dashboard}"

if [[ "$MODE" == "api" ]]; then
  if ! command -v uvicorn >/dev/null 2>&1; then
    echo "uvicorn is not installed. Run: pip install -r requirements.txt"
    exit 1
  fi
  uvicorn src.serving:app --host 0.0.0.0 --port 8000 --reload
elif [[ "$MODE" == "dashboard" ]]; then
  if ! command -v streamlit >/dev/null 2>&1; then
    echo "streamlit is not installed. Run: pip install -r requirements.txt"
    exit 1
  fi
  streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
else
  echo "Usage: ./launch.sh [dashboard|api]"
  exit 1
fi

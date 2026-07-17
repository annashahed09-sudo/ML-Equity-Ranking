#!/usr/bin/env bash
set -euo pipefail

# Generate strong random secrets and write them to a local .env file.
# Existing values are preserved unless --force is passed.
#
# Usage:
#   ./scripts/generate_secrets.sh          # create/append missing secrets
#   ./scripts/generate_secrets.sh --force  # overwrite existing secrets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
FORCE="${1:-}"

gen() {
  # 32 bytes of randomness, URL-safe.
  python -c "import secrets; print(secrets.token_urlsafe(32))"
}

set_secret() {
  local key="$1"
  local value="$2"
  if [[ -f "$ENV_FILE" ]] && grep -q "^${key}=" "$ENV_FILE"; then
    if [[ "$FORCE" == "--force" ]]; then
      # Portable in-place replace without relying on GNU sed -i semantics.
      local tmp
      tmp="$(mktemp)"
      grep -v "^${key}=" "$ENV_FILE" > "$tmp"
      mv "$tmp" "$ENV_FILE"
      echo "${key}=${value}" >> "$ENV_FILE"
      echo "updated ${key}"
    else
      echo "kept existing ${key} (use --force to overwrite)"
    fi
  else
    echo "${key}=${value}" >> "$ENV_FILE"
    echo "added ${key}"
  fi
}

touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

set_secret "ML_EQUITY_API_TOKEN" "$(gen)"
set_secret "ML_EQUITY_DASHBOARD_PASSWORD" "$(gen)"

echo "Secrets written to $ENV_FILE (this file is gitignored)."

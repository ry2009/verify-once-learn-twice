#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SLEEP_SECS="${SLEEP_SECS:-300}"

while true; do
  echo "[transfer-refresh] tick $(date '+%Y-%m-%d %H:%M:%S')"
  scripts/refresh_transfer_metrics.sh || true
  sleep "$SLEEP_SECS"
done

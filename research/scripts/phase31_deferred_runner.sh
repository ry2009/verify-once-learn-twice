#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

MAX_ACTIVE="${MAX_ACTIVE_RUN_ABLATION:-3}"
SLEEP_SECS="${SLEEP_SECS:-60}"

echo "[phase31-deferred] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[phase31-deferred] waiting until active run_ablation <= ${MAX_ACTIVE}"

while true; do
  active="$(ps -axo command | rg '^/Library/.*/Python .*scripts/run_ablation.py --spec' | wc -l | tr -d ' ')"
  echo "[phase31-deferred] active=${active} $(date '+%Y-%m-%d %H:%M:%S')"
  if [[ "$active" -le "$MAX_ACTIVE" ]]; then
    break
  fi
  sleep "$SLEEP_SECS"
done

echo "[phase31-deferred] launching phase31 pipeline"
scripts/phase31_domains_budget1_pipeline.sh

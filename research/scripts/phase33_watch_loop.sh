#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

LOG_FILE="runs/phase33_watcher.log"
mkdir -p runs

echo "[phase33_watcher] start $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

while [[ ! -f "runs/phase33_final_summary.md" ]]; do
  if ps -axo command | rg -q '^/Library/.*/Python .*scripts/run_ablation.py --spec data/ablation_phase33_hardset_numeric_budget1_s01234567.json'; then
    echo "[phase33_watcher] phase33 ablation already running $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    sleep 120
    continue
  fi

  echo "[phase33_watcher] attempt $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
  if ! bash scripts/phase33_hardset_budget1_pipeline.sh >> "$LOG_FILE" 2>&1; then
    echo "[phase33_watcher] pipeline attempt failed $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
  fi
  sleep 300
done

echo "[phase33_watcher] done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

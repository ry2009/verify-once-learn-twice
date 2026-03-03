#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p runs
STATUS_LOG="runs/phase50_watch.log"

echo "[phase50-watch] started at $(date)" | tee -a "$STATUS_LOG"
while true; do
  out="$(python3 scripts/phase50_status.py)"
  echo "$out" > runs/phase50_status_latest.txt
  echo "$out" >> "$STATUS_LOG"
  echo "" >> "$STATUS_LOG"
  if echo "$out" | rg -q "all_complete=true"; then
    echo "[phase50-watch] all runs complete at $(date)" | tee -a "$STATUS_LOG"
    bash scripts/phase50_closeout.sh | tee -a "$STATUS_LOG"
    break
  fi
  sleep 180
done

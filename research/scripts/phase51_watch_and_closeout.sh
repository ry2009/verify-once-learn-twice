#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="runs/phase51_watchdog"
mkdir -p "$LOG_DIR"
STATUS_FILE="${LOG_DIR}/latest_status.txt"

if [ -z "${TINKER_API_KEY:-}" ]; then
  echo "TINKER_API_KEY is not set"
  exit 1
fi

while true; do
  python3 scripts/phase51_status.py | tee "$STATUS_FILE"

  if rg -q "all_complete=true" "$STATUS_FILE"; then
    echo "[phase51-watch] all runs complete; building closeout artifacts"
    bash scripts/phase51_closeout.sh | tee "${LOG_DIR}/closeout.log"
    echo "[phase51-watch] done"
    exit 0
  fi

  active="$(pgrep -af "scripts/run_ablation.py --spec data/ablation_phase5[1-3]_" | wc -l | tr -d ' ')"
  if [ "$active" -eq 0 ]; then
    echo "[phase51-watch] no active ablation process; relaunching missing runs"
    bash scripts/phase51_launch.sh | tee -a "${LOG_DIR}/relaunch.log"
  else
    echo "[phase51-watch] active run_ablation=${active}"
  fi

  sleep 120
done

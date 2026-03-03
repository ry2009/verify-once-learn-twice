#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LOG_DIR="runs/_launcher_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/endgame_publish_$(date '+%Y%m%d-%H%M%S').log"

ACTIVE_SLEEP_SECS="${ACTIVE_SLEEP_SECS:-60}"

active_ablation_count() {
  ps -axo command | rg '^/Library/.*/Python .*scripts/run_ablation.py --spec' | wc -l | tr -d ' '
}

wait_for_no_ablation() {
  while true; do
    local active
    active="$(active_ablation_count)"
    echo "[endgame] active run_ablation=${active} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    if [[ "$active" -le 0 ]]; then
      break
    fi
    sleep "$ACTIVE_SLEEP_SECS"
  done
}

run_step() {
  local name="$1"
  shift
  echo "[endgame] START ${name} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  echo "[endgame] DONE  ${name} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
}

echo "[endgame] pipeline start $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "[endgame] log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

# Do not start another training sweep while one is already active.
wait_for_no_ablation

# 1) Cross-domain costly-feedback verification.
run_step phase31 scripts/phase31_domains_budget1_pipeline.sh

# 2) Mined hardset verification from phase31 numeric traces.
run_step phase33 scripts/phase33_hardset_budget1_pipeline.sh

# 3) Transfer test on MBPP with no periodic reset schedule.
run_step phase30b scripts/phase30b_mbpp_budget1_pipeline.sh

# 4) Additional odd-domain transfer on Synth20 under the same budget.
run_step phase32 scripts/phase32_synth_budget1_pipeline.sh

# 5) Resume and complete Phase-28 rare-feedback main + extension blocks.
run_step phase28_main scripts/phase28_budget1_pipeline.sh
run_step phase28_ext scripts/phase28_budget1_ext_pipeline.sh
run_step phase28_combine scripts/phase28_combine.sh

# 6) Add focused pairwise power sweep on H12 (adaptive vs fixed-k+judge-k1).
run_step phase34_pairwise scripts/phase34_h12_pairwise_power.sh

# 7) Final paper and status refresh.
run_step refresh_transfer scripts/refresh_transfer_metrics.sh
run_step build_live_status python3 scripts/build_live_status.py
run_step latex_build bash -lc 'cd paper && latexmk -pdf -interaction=nonstopmode main_phase20_live.tex'

DONE_FILE="runs/endgame_publish_done_$(date '+%Y%m%d-%H%M%S').flag"
echo "done" > "$DONE_FILE"
echo "[endgame] complete $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "[endgame] done file: ${DONE_FILE}" | tee -a "$LOG_FILE"

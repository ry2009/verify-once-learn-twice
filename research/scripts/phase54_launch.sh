#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -z "${TINKER_API_KEY:-}" ]; then
  echo "TINKER_API_KEY is not set"
  exit 1
fi

OUT_DIR="runs/phase54_launch"
mkdir -p "$OUT_DIR"

SPECS=(
  "data/ablation_phase54_kernelbench_target50_8b_rlm_s0.json"
  "data/ablation_phase55_kernelbench_target100_8b_rlm_s0.json"
  "data/ablation_phase56_kernelbench_target50_3b_rlm_s0.json"
)

for spec in "${SPECS[@]}"; do
  if pgrep -f "scripts/run_ablation.py --spec ${spec}" >/dev/null 2>&1; then
    echo "[phase54] already running: ${spec}"
    continue
  fi
  name="$(basename "${spec}" .json)"
  log_path="${OUT_DIR}/${name}.log"
  echo "[phase54] launch ${spec} -> ${log_path}"
  nohup env \
    TINKER_API_KEY="${TINKER_API_KEY}" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    python3 scripts/run_ablation.py --spec "${spec}" \
    >"${log_path}" 2>&1 &
  sleep 1
done

echo "[phase54] active processes:"
pgrep -af "scripts/run_ablation.py --spec data/ablation_phase5[4-6]_" || true

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "${TINKER_API_KEY:-}" ]; then
  echo "TINKER_API_KEY is not set"
  exit 1
fi

OUT_DIR="runs/phase51_launch"
mkdir -p "$OUT_DIR"

SPECS=(
  "data/ablation_phase51_kernelbench_target50_8b_s0.json"
  "data/ablation_phase52_kernelbench_target100_8b_s0.json"
  "data/ablation_phase53_kernelbench_target50_3b_s0.json"
)

for spec in "${SPECS[@]}"; do
  if pgrep -f "scripts/run_ablation.py --spec ${spec}" >/dev/null 2>&1; then
    echo "[phase51] already running: ${spec}"
    continue
  fi
  name="$(basename "${spec}" .json)"
  log_path="${OUT_DIR}/${name}.log"
  echo "[phase51] launch ${spec} -> ${log_path}"
  nohup env \
    TINKER_API_KEY="${TINKER_API_KEY}" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    python3 scripts/run_ablation.py --spec "${spec}" \
    >"${log_path}" 2>&1 &
  sleep 1
done

echo "[phase51] active processes:"
pgrep -af "scripts/run_ablation.py --spec data/ablation_phase5[1-3]_" || true

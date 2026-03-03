#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="runs/phase50_kernelbench_kpass_closeout"
mkdir -p "$OUT_DIR"

RUN_GROUPS=(
  "verify_phase50_kernelbench_kpass_8b_s0"
  "verify_phase50_kernelbench_kpass_3b_s0"
)

for g in "${RUN_GROUPS[@]}"; do
  python3 scripts/summarize_group.py --run_group "$g" > "$OUT_DIR/${g}_summary.txt" || true
  python3 scripts/costly_leaderboard.py --run_group "$g" --min_tasks 10 --out_json "$OUT_DIR/${g}_leaderboard.json" > "$OUT_DIR/${g}_leaderboard.csv" || true
  python3 scripts/kernel_transfer_report.py --run_group "$g" --out_dir "$OUT_DIR/${g}_transfer" || true
  if [ -f "$OUT_DIR/${g}_leaderboard.json" ]; then
    python3 scripts/kernelllm_metric_card.py --summary_json "$OUT_DIR/${g}_leaderboard.json" --out_md "$OUT_DIR/${g}_vs_kernelllm.md" || true
  fi
done

echo "[phase50] wrote closeout artifacts to $OUT_DIR"

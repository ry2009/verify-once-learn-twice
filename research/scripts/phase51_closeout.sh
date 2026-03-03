#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="runs/phase51_kernelbench_scale_closeout"
mkdir -p "$OUT_DIR"

RUN_GROUPS=(
  "verify_phase51_kernelbench_target50_8b_s0"
  "verify_phase52_kernelbench_target100_8b_s0"
  "verify_phase53_kernelbench_target50_3b_s0"
)

for g in "${RUN_GROUPS[@]}"; do
  python3 scripts/summarize_group.py --run_group "$g" > "$OUT_DIR/${g}_summary.txt" || true
  python3 scripts/costly_leaderboard.py --run_group "$g" --min_tasks 20 --out_json "$OUT_DIR/${g}_leaderboard.json" > "$OUT_DIR/${g}_leaderboard.csv" || true
  if [ -f "$OUT_DIR/${g}_leaderboard.json" ]; then
    python3 scripts/kernelllm_metric_card.py --summary_json "$OUT_DIR/${g}_leaderboard.json" --out_md "$OUT_DIR/${g}_vs_kernelllm.md" || true
    python3 scripts/kernel_cost_compare.py --summary_json "$OUT_DIR/${g}_leaderboard.json" --out_md "$OUT_DIR/${g}_direct_cost_compare.md" --out_json "$OUT_DIR/${g}_direct_cost_compare.json" || true
  fi
done

echo "[phase51] wrote closeout artifacts to $OUT_DIR"

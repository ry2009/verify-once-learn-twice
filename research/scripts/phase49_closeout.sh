#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="runs/phase49_kernelbench_transfer_closeout"
mkdir -p "$OUT_DIR"

SMALL_GROUPS=(
  "verify_phase49_kernelbench_transfer_3b_s0"
  "verify_phase49_kernelbench_transfer_3b_s1"
)

ALL_GROUPS=(
  "verify_phase49_kernelbench_transfer_3b_s0"
  "verify_phase49_kernelbench_transfer_3b_s1"
  "verify_phase49_kernelbench_transfer_70b_s0"
  "verify_phase49_kernelbench_transfer_gptoss20b_s0"
)

for g in "${ALL_GROUPS[@]}"; do
  python3 scripts/summarize_group.py --run_group "$g" > "$OUT_DIR/${g}_summary.txt" || true
  python3 scripts/costly_leaderboard.py \
    --run_group "$g" \
    --min_tasks 10 \
    --out_json "$OUT_DIR/${g}_leaderboard.json" \
    > "$OUT_DIR/${g}_leaderboard.csv" || true
done

python3 scripts/paired_compare.py \
  --run_group "${SMALL_GROUPS[0]}" \
  --run_group "${SMALL_GROUPS[1]}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 2 \
  --feedback_budget 1 \
  --pair_key seed \
  --min_tasks 10 \
  --bootstrap_samples 20000 \
  --alpha 0.05 \
  > "$OUT_DIR/adaptive_vs_fixedk2.json" || true

python3 scripts/paired_compare.py \
  --run_group "${SMALL_GROUPS[0]}" \
  --run_group "${SMALL_GROUPS[1]}" \
  --method_a adaptive_fwb \
  --method_b inference_only \
  --feedback_budget 1 \
  --pair_key seed \
  --min_tasks 10 \
  --bootstrap_samples 20000 \
  --alpha 0.05 \
  > "$OUT_DIR/adaptive_vs_inference.json" || true

python3 scripts/kernel_transfer_report.py \
  --run_group "${ALL_GROUPS[0]}" \
  --run_group "${ALL_GROUPS[1]}" \
  --run_group "${ALL_GROUPS[2]}" \
  --run_group "${ALL_GROUPS[3]}" \
  --out_dir "$OUT_DIR"

echo "[phase49] wrote closeout artifacts to $OUT_DIR"

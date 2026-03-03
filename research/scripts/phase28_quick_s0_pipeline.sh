#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_GROUP="verify_phase28_h12_budget1_quick_s0"

echo "[phase28quick] start $(date '+%Y-%m-%d %H:%M:%S')"
python3 scripts/validate_tasks.py --tasks data/humaneval_12_expensivefb600_fail2of3_seed012.jsonl
python3 scripts/run_ablation.py --spec data/ablation_phase28_h12_budget1_quick_s0.json
python3 scripts/costly_leaderboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --out_json "runs/${RUN_GROUP}/costly_leaderboard_min12.json" \
  > "runs/${RUN_GROUP}/costly_leaderboard_min12.txt"
for k in 1 2 4; do
  python3 scripts/paired_compare.py \
    --run_group "$RUN_GROUP" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b "$k" \
    --feedback_budget 1 \
    --min_tasks 12 \
    > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj${k}_b1.json" || true
done
python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_fwb \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkfwb1_b1.json" || true
python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b inference_only \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_inference_b1.json" || true

python3 scripts/plot_pretty_dashboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --title "Phase28 quick H12 budget=1 (seed 0)" \
  --write_png

echo "[phase28quick] done $(date '+%Y-%m-%d %H:%M:%S')"

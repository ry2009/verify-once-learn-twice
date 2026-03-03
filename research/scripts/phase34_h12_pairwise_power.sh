#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
RUN_GROUP="verify_phase34_h12_budget1_pairwise_s1223"
SPEC="data/ablation_phase34_h12_budget1_pairwise_s1223.json"

echo "[phase34] start $(date '+%Y-%m-%d %H:%M:%S')"
python3 scripts/validate_tasks.py --tasks data/humaneval_12_expensivefb600_fail2of3_seed012.jsonl
python3 scripts/run_ablation.py --spec "$SPEC"
python3 scripts/costly_leaderboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --out_json "runs/${RUN_GROUP}/costly_leaderboard_min12.json" \
  > "runs/${RUN_GROUP}/costly_leaderboard_min12.txt"
python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj1_b1.json"

# Recompute combined phase28+phase34 evidence.
bash scripts/phase28_combine.sh
bash scripts/refresh_transfer_metrics.sh

echo "[phase34] done $(date '+%Y-%m-%d %H:%M:%S')"

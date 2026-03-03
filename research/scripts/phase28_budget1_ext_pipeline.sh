#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
RUN_GROUP="verify_phase28_h12_budget1_ext_s891011"

echo "[phase28ext] start $(date '+%Y-%m-%d %H:%M:%S')"
python3 scripts/validate_tasks.py --tasks data/humaneval_12_expensivefb600_fail2of3_seed012.jsonl
python3 scripts/run_ablation.py --spec data/ablation_phase28_h12_budget1_ext_s891011.json
python3 scripts/costly_leaderboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --out_json "runs/${RUN_GROUP}/costly_leaderboard_min12.json" \
  > "runs/${RUN_GROUP}/costly_leaderboard_min12.txt"
python3 scripts/plot_pretty_dashboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --title "Phase28 ext H12 budget=1 (s8-11)" \
  --write_png

echo "[phase28ext] done $(date '+%Y-%m-%d %H:%M:%S')"

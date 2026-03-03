#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_GROUP="verify_phase28_h12_budget1_s01234567"
SPEC="data/ablation_phase28_h12_budget1_s01234567.json"

echo "[phase28] start $(date '+%Y-%m-%d %H:%M:%S')"

python3 scripts/validate_tasks.py \
  --tasks data/humaneval_12_expensivefb600_fail2of3_seed012.jsonl

echo "[phase28] run sweep"
python3 scripts/run_ablation.py --spec "$SPEC"

echo "[phase28] leaderboard"
python3 scripts/costly_leaderboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --out_json "runs/${RUN_GROUP}/costly_leaderboard_min12.json" \
  > "runs/${RUN_GROUP}/costly_leaderboard_min12.txt"

echo "[phase28] paired"
python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj1_b1.json"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 2 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj2_b1.json"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 4 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj4_b1.json"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_fwb \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkfwb1_b1.json"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b inference_only \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_inference_b1.json"

python3 scripts/plot_pretty_dashboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --title "Phase28 H12 Rare-Feedback (b=1; k=1,2,4; s0-7)" \
  --write_png

python3 - <<'PY'
import json
from pathlib import Path

group = "verify_phase28_h12_budget1_s01234567"
lb = json.loads(Path(f"runs/{group}/costly_leaderboard_min12.json").read_text())
rows = lb.get("aggregate", [])
by = {r["method_variant"]: r for r in rows if int(r.get("feedback_budget", -1)) == 1}

lines = ["# Phase28 Final Summary", "", "## H12 budget=1"]
for key in ["adaptive_fwb", "fixed_k_judge_k1", "fixed_k_judge_k2", "fixed_k_judge_k4", "fixed_k_fwb_k1", "inference_only"]:
    r = by.get(key)
    if not r:
        continue
    lines.append(
        f"- {key}: success={r['success_mean']:.4f} fb={r['feedback_mean']:.4f} train={r['train_mean']:.4f} runs={r['runs']}"
    )

Path("runs/phase28_final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote runs/phase28_final_summary.md")
PY

echo "[phase28] done $(date '+%Y-%m-%d %H:%M:%S')"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_GROUP="verify_phase30b_mbpp20_budget1_s0123_nors"
MIN_TASKS=20
SPEC="data/ablation_phase30b_mbpp20_budget1_s0123_nors.json"

echo "[phase30b] start $(date '+%Y-%m-%d %H:%M:%S')"
python3 scripts/validate_tasks.py --tasks data/mbpp_20_v2_hidden70_probe.jsonl

echo "[phase30b] run ablation"
python3 scripts/run_ablation.py --spec "$SPEC"

echo "[phase30b] leaderboard + paired"
python3 scripts/costly_leaderboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks "$MIN_TASKS" \
  --out_json "runs/${RUN_GROUP}/costly_leaderboard_min${MIN_TASKS}.json" \
  > "runs/${RUN_GROUP}/costly_leaderboard_min${MIN_TASKS}.txt"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks "$MIN_TASKS" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj1_b1.json"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 2 \
  --feedback_budget 1 \
  --min_tasks "$MIN_TASKS" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj2_b1.json"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 4 \
  --feedback_budget 1 \
  --min_tasks "$MIN_TASKS" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj4_b1.json"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b fixed_k_fwb \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks "$MIN_TASKS" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkfwb1_b1.json"

python3 scripts/paired_compare.py \
  --run_group "$RUN_GROUP" \
  --method_a adaptive_fwb \
  --method_b inference_only \
  --feedback_budget 1 \
  --min_tasks "$MIN_TASKS" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_inference_b1.json"

echo "[phase30b] dashboard + paper tables"
python3 scripts/plot_pretty_dashboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks "$MIN_TASKS" \
  --title "Phase30b MBPP20 Rare-Feedback (b=1; no reset schedule; s0-3)" \
  --write_png

python3 paper/scripts/build_phase30_tables.py \
  --leaderboard "runs/${RUN_GROUP}/costly_leaderboard_min${MIN_TASKS}.json" \
  --run_group "$RUN_GROUP"

png="$(ls -dt runs/pretty_plots/*${RUN_GROUP}*/dashboard.png 2>/dev/null | head -n 1 || true)"
if [[ -n "$png" ]]; then
  mkdir -p paper/figures
  cp "$png" paper/figures/phase30_mbpp20_budget1_dashboard.png
fi

python3 - <<'PY'
import json
from pathlib import Path

run_group = "verify_phase30b_mbpp20_budget1_s0123_nors"
lb_path = Path(f"runs/{run_group}/costly_leaderboard_min20.json")
if lb_path.exists():
    lb = json.loads(lb_path.read_text())
    by = {r["method_variant"]: r for r in lb.get("aggregate", []) if int(r.get("feedback_budget", -1)) == 1}
    lines = ["# Phase30b Final Summary", "", "## MBPP20 budget=1 (no reset schedule)"]
    for key in ["adaptive_fwb", "fixed_k_judge_k1", "fixed_k_judge_k2", "fixed_k_judge_k4", "fixed_k_fwb_k1", "inference_only"]:
        r = by.get(key)
        if not r:
            continue
        lines.append(
            f"- {key}: success={r['success_mean']:.4f} fb={r['feedback_mean']:.4f} train={r['train_mean']:.4f} runs={r['runs']}"
        )
    Path("runs/phase30b_final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote runs/phase30b_final_summary.md")
PY

( cd paper && latexmk -pdf -interaction=nonstopmode main_phase20_live.tex )
python3 scripts/build_live_status.py

echo "[phase30b] done $(date '+%Y-%m-%d %H:%M:%S')"

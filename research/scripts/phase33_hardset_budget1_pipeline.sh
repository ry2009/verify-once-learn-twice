#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SRC_GROUP="verify_phase31_numeric_budget1_s01234567"
RUN_GROUP="verify_phase33_hardset_numeric_budget1_s01234567"
SPEC_PATH="data/ablation_phase33_hardset_numeric_budget1_s01234567.json"
HARDSET_TASKS="data/hardset_phase33_numeric_budget1_auto.jsonl"
HARDSET_JSON="runs/hardset_phase33_numeric_budget1_auto.json"
HARDSET_MD="runs/hardset_phase33_numeric_budget1_auto.md"

echo "[phase33] start $(date '+%Y-%m-%d %H:%M:%S')"

if ! python3 scripts/mine_adaptive_hardset.py \
  --run_group "${SRC_GROUP}" \
  --feedback_budget 1 \
  --min_tasks_per_run 10 \
  --min_shared_seeds 4 \
  --min_advantage_votes 2 \
  --top_k 20 \
  --min_selected 8 \
  --out_tasks "${HARDSET_TASKS}" \
  --out_json "${HARDSET_JSON}" \
  --out_md "${HARDSET_MD}" \
  --template_spec data/ablation_phase31_numeric_budget1_s01234567.json \
  --out_spec "${SPEC_PATH}" \
  --out_run_group "${RUN_GROUP}" \
  --out_name phase33-hardset-numeric-budget1-s01234567
then
  echo "[phase33] hardset not ready yet; waiting for more completed phase31 numeric seeds"
  exit 0
fi

TASK_COUNT="$(wc -l < "${HARDSET_TASKS}" | tr -d ' ')"
if [[ "${TASK_COUNT}" -lt 8 ]]; then
  echo "[phase33] hardset has too few tasks (${TASK_COUNT}); skipping sweep"
  exit 0
fi

python3 scripts/validate_tasks.py --tasks "${HARDSET_TASKS}"

echo "[phase33] run ablation on mined hardset (${TASK_COUNT} tasks)"
python3 scripts/run_ablation.py --spec "${SPEC_PATH}"

python3 scripts/costly_leaderboard.py \
  --run_group "${RUN_GROUP}" \
  --min_tasks "${TASK_COUNT}" \
  --out_json "runs/${RUN_GROUP}/costly_leaderboard_min${TASK_COUNT}.json" \
  > "runs/${RUN_GROUP}/costly_leaderboard_min${TASK_COUNT}.txt"

python3 scripts/paired_compare.py \
  --run_group "${RUN_GROUP}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks "${TASK_COUNT}" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj1_b1.json"

python3 scripts/paired_compare.py \
  --run_group "${RUN_GROUP}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 2 \
  --feedback_budget 1 \
  --min_tasks "${TASK_COUNT}" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj2_b1.json"

python3 scripts/paired_compare.py \
  --run_group "${RUN_GROUP}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 4 \
  --feedback_budget 1 \
  --min_tasks "${TASK_COUNT}" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj4_b1.json"

python3 scripts/paired_compare.py \
  --run_group "${RUN_GROUP}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_fwb \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks "${TASK_COUNT}" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkfwb1_b1.json"

python3 scripts/paired_compare.py \
  --run_group "${RUN_GROUP}" \
  --method_a adaptive_fwb \
  --method_b inference_only \
  --feedback_budget 1 \
  --min_tasks "${TASK_COUNT}" \
  > "runs/${RUN_GROUP}/paired_adaptive_vs_inference_b1.json"

python3 scripts/plot_pretty_dashboard.py \
  --run_group "${RUN_GROUP}" \
  --min_tasks "${TASK_COUNT}" \
  --title "Phase33 Mined Hardset (Budget=1, Adaptive vs Baselines)" \
  --write_png

python3 paper/scripts/build_phase33_tables.py \
  --leaderboard "runs/${RUN_GROUP}/costly_leaderboard_min${TASK_COUNT}.json" \
  --run_group "${RUN_GROUP}" \
  --hardset_json "${HARDSET_JSON}"

png="$(ls -dt runs/pretty_plots/*${RUN_GROUP}*/dashboard.png 2>/dev/null | head -n 1 || true)"
if [[ -n "${png}" ]]; then
  mkdir -p paper/figures
  cp "${png}" "paper/figures/phase33_hardset_budget1_dashboard.png"
fi

python3 - <<'PY'
import json
from pathlib import Path

run_group = "verify_phase33_hardset_numeric_budget1_s01234567"
cands = sorted(Path(f"runs/{run_group}").glob("costly_leaderboard_min*.json"))
if cands:
    lb = json.loads(cands[-1].read_text())
    by = {r["method_variant"]: r for r in lb.get("aggregate", []) if int(r.get("feedback_budget", -1)) == 1}
    lines = ["# Phase33 Final Summary", "", "## Mined hardset budget=1"]
    for key in ["adaptive_fwb", "fixed_k_judge_k1", "fixed_k_judge_k2", "fixed_k_judge_k4", "fixed_k_fwb_k1", "inference_only"]:
        r = by.get(key)
        if not r:
            continue
        lines.append(
            f"- {key}: success={r['success_mean']:.4f} fb={r['feedback_mean']:.4f} train={r['train_mean']:.4f} runs={r['runs']}"
        )
    Path("runs/phase33_final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote runs/phase33_final_summary.md")
PY

python3 scripts/build_live_status.py
python3 scripts/build_win_snapshot.py
python3 paper/scripts/build_live_scorecard.py
python3 scripts/build_publish_readiness.py

(
  cd paper
  latexmk -pdf -interaction=nonstopmode main_phase20_live.tex
)

echo "[phase33] done $(date '+%Y-%m-%d %H:%M:%S')"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set"
  exit 1
fi

mkdir -p runs
echo "[phase25_probe_fix] start $(date '+%Y-%m-%d %H:%M:%S')" | tee -a runs/phase25_probe_fix.log

COMMON_ARGS=(
  --tasks data/synth_stringops_80_rich_v2_hidden70.jsonl
  --run_group verify_phase25_probe_synth80_hidden70_fix_s0_b2_v2
  --ablation_tag phase25_probe_synth_hidden70_fix
  --feedback_budget 2
  --judge_mode oracle_binary
  --max_steps 20
  --max_tokens 128
  --temperature 0.2
  --no_reset_per_task
  --reset_every_n_tasks 4
  --lr 2e-5
  --seed 0
)

python3 scripts/validate_tasks.py --tasks data/synth_stringops_80_rich_v2_hidden70.jsonl | tee -a runs/phase25_probe_fix.log

python3 scripts/run_experiment.py \
  "${COMMON_ARGS[@]}" \
  --run_name adaptive_fwb-fix-b2-s0 \
  --method adaptive_fwb \
  --inner_updates 1 \
  --adaptive_judge_every_updates 1 \
  --max_adaptive_steps_per_feedback 4 \
  --teacher_resample_attempts 3 \
  --teacher_filter_with_judge \
  --teacher_filter_policy pass_or_assertion \
  | tee -a runs/phase25_probe_fix.log

python3 scripts/run_experiment.py \
  "${COMMON_ARGS[@]}" \
  --run_name fixed_k_judge-k1-fix-b2-s0 \
  --method fixed_k_judge \
  --inner_updates 1 \
  --teacher_resample_attempts 3 \
  --teacher_filter_with_judge \
  --teacher_filter_policy pass_or_assertion \
  | tee -a runs/phase25_probe_fix.log

python3 scripts/run_experiment.py \
  "${COMMON_ARGS[@]}" \
  --run_name inference_only-fix-b2-s0 \
  --method inference_only \
  | tee -a runs/phase25_probe_fix.log

python3 scripts/costly_leaderboard.py \
  --run_group verify_phase25_probe_synth80_hidden70_fix_s0_b2_v2 \
  --min_tasks 20 \
  --out_json runs/verify_phase25_probe_synth80_hidden70_fix_s0_b2_v2/costly_leaderboard_min20.json \
  > runs/verify_phase25_probe_synth80_hidden70_fix_s0_b2_v2/costly_leaderboard_min20.txt

python3 scripts/paired_compare.py \
  --run_group verify_phase25_probe_synth80_hidden70_fix_s0_b2_v2 \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 1 \
  --feedback_budget 2 \
  --min_tasks 20 \
  > runs/verify_phase25_probe_synth80_hidden70_fix_s0_b2_v2/paired_adaptive_vs_fixedkj1_min20.json || true

echo "[phase25_probe_fix] done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a runs/phase25_probe_fix.log

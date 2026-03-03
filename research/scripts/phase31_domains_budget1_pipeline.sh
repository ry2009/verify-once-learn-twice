#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[phase31] start $(date '+%Y-%m-%d %H:%M:%S')"

python3 scripts/validate_tasks.py \
  --tasks data/humaneval_80_hidden70_coarse_numeric.jsonl \
  --tasks data/humaneval_80_hidden70_coarse_string.jsonl \
  --tasks data/humaneval_80_hidden70_coarse_symbolic.jsonl

declare -a DOMAINS=("numeric:10" "string:8" "symbolic:7")

for entry in "${DOMAINS[@]}"; do
  domain="${entry%%:*}"
  min_tasks="${entry##*:}"
  spec="data/ablation_phase31_${domain}_budget1_s01234567.json"
  group="verify_phase31_${domain}_budget1_s01234567"

  echo "[phase31] run ${domain} sweep"
  python3 scripts/run_ablation.py --spec "$spec"

  echo "[phase31] leaderboard ${domain}"
  python3 scripts/costly_leaderboard.py \
    --run_group "$group" \
    --min_tasks "$min_tasks" \
    --out_json "runs/${group}/costly_leaderboard_min${min_tasks}.json" \
    > "runs/${group}/costly_leaderboard_min${min_tasks}.txt"

  echo "[phase31] paired deltas ${domain}"
  python3 scripts/paired_compare.py \
    --run_group "$group" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b 1 \
    --feedback_budget 1 \
    --min_tasks "$min_tasks" \
    > "runs/${group}/paired_adaptive_vs_fixedkj1_b1.json"
  python3 scripts/paired_compare.py \
    --run_group "$group" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b 2 \
    --feedback_budget 1 \
    --min_tasks "$min_tasks" \
    > "runs/${group}/paired_adaptive_vs_fixedkj2_b1.json"
  python3 scripts/paired_compare.py \
    --run_group "$group" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b 4 \
    --feedback_budget 1 \
    --min_tasks "$min_tasks" \
    > "runs/${group}/paired_adaptive_vs_fixedkj4_b1.json"
  python3 scripts/paired_compare.py \
    --run_group "$group" \
    --method_a adaptive_fwb \
    --method_b fixed_k_fwb \
    --inner_updates_b 1 \
    --feedback_budget 1 \
    --min_tasks "$min_tasks" \
    > "runs/${group}/paired_adaptive_vs_fixedkfwb1_b1.json"
  python3 scripts/paired_compare.py \
    --run_group "$group" \
    --method_a adaptive_fwb \
    --method_b inference_only \
    --feedback_budget 1 \
    --min_tasks "$min_tasks" \
    > "runs/${group}/paired_adaptive_vs_inference_b1.json"

  echo "[phase31] dashboard ${domain}"
  python3 scripts/plot_pretty_dashboard.py \
    --run_group "$group" \
    --min_tasks "$min_tasks" \
    --title "Phase31 ${domain} budget=1 (rare-feedback, s01234567)" \
    --write_png
done

echo "[phase31] domain matrix"
python3 scripts/domain_slice_matrix.py \
  --slice "numeric=runs/verify_phase31_numeric_budget1_s01234567/costly_leaderboard_min10.json" \
  --slice "string=runs/verify_phase31_string_budget1_s01234567/costly_leaderboard_min8.json" \
  --slice "symbolic=runs/verify_phase31_symbolic_budget1_s01234567/costly_leaderboard_min7.json" \
  --title "Phase31 Domain Transfer (Budget=1, Seeds 0-7)" \
  --write_png

python3 paper/scripts/build_phase31_tables.py

latest_matrix_dir="$(ls -dt runs/pretty_plots/*domain-slice-matrix-numeric-string-symbolic | head -n 1 || true)"
if [[ -n "$latest_matrix_dir" && -f "$latest_matrix_dir/dashboard.png" ]]; then
  mkdir -p paper/figures
  cp "$latest_matrix_dir/dashboard.png" "paper/figures/phase31_domains_budget1_matrix.png"
fi

(
  cd paper
  latexmk -pdf -interaction=nonstopmode main_phase20_live.tex
)

python3 scripts/build_live_status.py

echo "[phase31] done $(date '+%Y-%m-%d %H:%M:%S')"

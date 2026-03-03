#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="runs/phase28_combined"
mkdir -p "$OUT_DIR"

RUN_GROUPS=(
  verify_phase28_h12_budget1_s01234567
  verify_phase28_h12_budget1_ext_s891011
  verify_phase28_h12_budget1_boost_s123
  verify_phase37_h12_budget1_compute_match_s01234567
  verify_phase34_h12_budget1_pairwise_s1223
)

rg_args=()
for g in "${RUN_GROUPS[@]}"; do
  if [[ -d "runs/${g}" ]]; then
    rg_args+=(--run_group "$g")
  fi
done

if [[ ${#rg_args[@]} -eq 0 ]]; then
  echo "[phase28-combine] no run groups found" >&2
  exit 1
fi

python3 scripts/costly_leaderboard.py \
  "${rg_args[@]}" \
  --min_tasks 12 \
  --out_json "$OUT_DIR/costly_leaderboard_min12.json" \
  > "$OUT_DIR/costly_leaderboard_min12.txt"

python3 scripts/paired_compare.py \
  "${rg_args[@]}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "$OUT_DIR/paired_adaptive_vs_fixedkj1_b1.json" || true

python3 scripts/paired_compare.py \
  "${rg_args[@]}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 2 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "$OUT_DIR/paired_adaptive_vs_fixedkj2_b1.json" || true

python3 scripts/paired_compare.py \
  "${rg_args[@]}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 4 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "$OUT_DIR/paired_adaptive_vs_fixedkj4_b1.json" || true

python3 scripts/paired_compare.py \
  "${rg_args[@]}" \
  --method_a adaptive_fwb \
  --method_b fixed_k_fwb \
  --inner_updates_b 1 \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "$OUT_DIR/paired_adaptive_vs_fixedkfwb1_b1.json" || true

python3 scripts/paired_compare.py \
  "${rg_args[@]}" \
  --method_a adaptive_fwb \
  --method_b inference_only \
  --feedback_budget 1 \
  --min_tasks 12 \
  > "$OUT_DIR/paired_adaptive_vs_inference_b1.json" || true

python3 scripts/plot_pretty_dashboard.py \
  "${rg_args[@]}" \
  --min_tasks 12 \
  --title "Phase28 H12 Rare-Feedback (budget=1, combined seed blocks)" \
  --write_png

python3 paper/scripts/build_phase28_tables.py \
  --leaderboard "$OUT_DIR/costly_leaderboard_min12.json" \
  --run_group verify_phase28_h12_budget1_s01234567 \
  --paired_fixedkj1 "$OUT_DIR/paired_adaptive_vs_fixedkj1_b1.json" \
  --paired_fixedkj2 "$OUT_DIR/paired_adaptive_vs_fixedkj2_b1.json" \
  --paired_fixedkj4 "$OUT_DIR/paired_adaptive_vs_fixedkj4_b1.json" \
  --paired_fixedkfwb1 "$OUT_DIR/paired_adaptive_vs_fixedkfwb1_b1.json" \
  --paired_inference "$OUT_DIR/paired_adaptive_vs_inference_b1.json"

p=$(ls -dt runs/pretty_plots/*verify_phase28_h12_budget1*dashboard.png 2>/dev/null | head -n 1 || true)
if [[ -n "$p" ]]; then
  cp "$p" paper/figures/phase28_h12_budget1_dashboard.png
fi

( cd paper && latexmk -pdf -interaction=nonstopmode main_phase20_live.tex )
python3 scripts/build_live_status.py

echo "[phase28-combine] done"

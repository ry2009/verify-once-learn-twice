#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PHASE30_GROUP="verify_phase30b_mbpp20_budget1_s0123_nors"
PHASE32_GROUP="verify_phase32_synth20_budget1_s0123_nors"
PHASE33_GROUP_NUMERIC="verify_phase33_hardset_numeric_budget1_s01234567"
PHASE33_GROUP_H12PAIR="verify_phase33_hardset_h12pair_budget1_s1223"
PHASE34_GROUP="verify_phase34_h12_budget1_pairwise_s1223"

if [[ -d "runs/${PHASE30_GROUP}" ]]; then
  python3 scripts/costly_leaderboard.py \
    --run_group "${PHASE30_GROUP}" \
    --min_tasks 10 \
    --out_json "runs/${PHASE30_GROUP}/costly_leaderboard_partial_min10.json" \
    > "runs/${PHASE30_GROUP}/costly_leaderboard_partial_min10.txt" || true

  python3 scripts/paired_compare.py \
    --run_group "${PHASE30_GROUP}" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b 1 \
    --feedback_budget 1 \
    --min_tasks 10 \
    > "runs/${PHASE30_GROUP}/paired_adaptive_vs_fixedkj1_b1_partial_min10.json" || true

  python3 scripts/plot_pretty_dashboard.py \
    --run_group "${PHASE30_GROUP}" \
    --min_tasks 10 \
    --title "Phase30b MBPP20 Rare-Feedback (b=1; partial >=10 tasks)" \
    --write_png || true

  latest_phase30_png="$(ls -dt runs/pretty_plots/*${PHASE30_GROUP}*/dashboard.png 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_phase30_png}" ]]; then
    mkdir -p paper/figures
    cp "${latest_phase30_png}" "paper/figures/phase30_mbpp20_budget1_dashboard.png"
  fi

  python3 paper/scripts/build_phase30_tables.py \
    --leaderboard "runs/${PHASE30_GROUP}/costly_leaderboard_partial_min10.json" \
    --run_group "${PHASE30_GROUP}" || true
fi

if [[ -d "runs/${PHASE32_GROUP}" ]]; then
  python3 scripts/costly_leaderboard.py \
    --run_group "${PHASE32_GROUP}" \
    --min_tasks 10 \
    --out_json "runs/${PHASE32_GROUP}/costly_leaderboard_partial_min10.json" \
    > "runs/${PHASE32_GROUP}/costly_leaderboard_partial_min10.txt" || true

  python3 scripts/paired_compare.py \
    --run_group "${PHASE32_GROUP}" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b 1 \
    --feedback_budget 1 \
    --min_tasks 10 \
    > "runs/${PHASE32_GROUP}/paired_adaptive_vs_fixedkj1_b1_partial_min10.json" || true

  python3 scripts/plot_pretty_dashboard.py \
    --run_group "${PHASE32_GROUP}" \
    --min_tasks 10 \
    --title "Phase32 Synth20 Rare-Feedback (b=1; partial >=10 tasks)" \
    --write_png || true

  latest_phase32_png="$(ls -dt runs/pretty_plots/*${PHASE32_GROUP}*/dashboard.png 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_phase32_png}" ]]; then
    mkdir -p paper/figures
    cp "${latest_phase32_png}" "paper/figures/phase32_synth20_budget1_dashboard.png"
  fi

  python3 paper/scripts/build_phase32_tables.py \
    --leaderboard "runs/${PHASE32_GROUP}/costly_leaderboard_partial_min10.json" \
    --run_group "${PHASE32_GROUP}" || true
fi

# Mine a reproducible adaptive-win hardset from current phase31 numeric traces.
python3 scripts/mine_adaptive_hardset.py \
  --run_group verify_phase31_numeric_budget1_s01234567 \
  --feedback_budget 1 \
  --min_tasks_per_run 10 \
  --min_shared_seeds 4 \
  --min_advantage_votes 2 \
  --top_k 20 \
  --min_selected 8 \
  --out_tasks data/hardset_phase33_numeric_budget1_auto.jsonl \
  --out_json runs/hardset_phase33_numeric_budget1_auto.json \
  --out_md runs/hardset_phase33_numeric_budget1_auto.md \
  --template_spec data/ablation_phase31_numeric_budget1_s01234567.json \
  --out_spec data/ablation_phase33_hardset_numeric_budget1_s01234567.json \
  --out_run_group "${PHASE33_GROUP_NUMERIC}" \
  --out_name phase33-hardset-numeric-budget1-s01234567 || true

# Mine fallback hardset from pairwise H12 traces so phase33 can proceed even when
# numeric hardset remains underpowered.
python3 scripts/mine_adaptive_hardset.py \
  --run_group "${PHASE34_GROUP}" \
  --feedback_budget 1 \
  --min_tasks_per_run 10 \
  --min_shared_seeds 2 \
  --min_advantage_votes 1 \
  --top_k 20 \
  --min_selected 6 \
  --out_tasks data/hardset_phase33_h12pair_budget1_auto.jsonl \
  --out_json runs/hardset_phase33_h12pair_budget1_auto.json \
  --out_md runs/hardset_phase33_h12pair_budget1_auto.md \
  --template_spec data/ablation_phase34_h12_budget1_pairwise_s1223.json \
  --out_spec data/ablation_phase33_hardset_h12pair_budget1_s1223.json \
  --out_run_group "${PHASE33_GROUP_H12PAIR}" \
  --out_name phase33-hardset-h12pair-budget1-s1223 || true

PHASE33_NUMERIC_MIN_TASKS=8
if [[ -f "data/hardset_phase33_numeric_budget1_auto.jsonl" ]]; then
  PHASE33_NUMERIC_MIN_TASKS="$(wc -l < data/hardset_phase33_numeric_budget1_auto.jsonl | tr -d ' ')"
fi
if [[ "${PHASE33_NUMERIC_MIN_TASKS}" -lt 1 ]]; then
  PHASE33_NUMERIC_MIN_TASKS=1
fi

PHASE33_H12PAIR_MIN_TASKS=6
if [[ -f "data/hardset_phase33_h12pair_budget1_auto.jsonl" ]]; then
  PHASE33_H12PAIR_MIN_TASKS="$(wc -l < data/hardset_phase33_h12pair_budget1_auto.jsonl | tr -d ' ')"
fi
if [[ "${PHASE33_H12PAIR_MIN_TASKS}" -lt 1 ]]; then
  PHASE33_H12PAIR_MIN_TASKS=1
fi

PHASE33_GROUP="${PHASE33_GROUP_NUMERIC}"
PHASE33_MIN_TASKS="${PHASE33_NUMERIC_MIN_TASKS}"
PHASE33_HARDSET_JSON="runs/hardset_phase33_numeric_budget1_auto.json"
if [[ -d "runs/${PHASE33_GROUP_H12PAIR}" ]]; then
  PHASE33_GROUP="${PHASE33_GROUP_H12PAIR}"
  PHASE33_MIN_TASKS="${PHASE33_H12PAIR_MIN_TASKS}"
  PHASE33_HARDSET_JSON="runs/hardset_phase33_h12pair_budget1_auto.json"
fi

if [[ -d "runs/${PHASE33_GROUP}" ]]; then
  python3 scripts/costly_leaderboard.py \
    --run_group "${PHASE33_GROUP}" \
    --min_tasks "${PHASE33_MIN_TASKS}" \
    --out_json "runs/${PHASE33_GROUP}/costly_leaderboard_min${PHASE33_MIN_TASKS}.json" \
    > "runs/${PHASE33_GROUP}/costly_leaderboard_min${PHASE33_MIN_TASKS}.txt" || true

  python3 scripts/paired_compare.py \
    --run_group "${PHASE33_GROUP}" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b 1 \
    --feedback_budget 1 \
    --min_tasks "${PHASE33_MIN_TASKS}" \
    > "runs/${PHASE33_GROUP}/paired_adaptive_vs_fixedkj1_b1.json" || true

  python3 scripts/plot_pretty_dashboard.py \
    --run_group "${PHASE33_GROUP}" \
    --min_tasks "${PHASE33_MIN_TASKS}" \
    --title "Phase33 Mined Hardset (b=1; adaptive-win slice)" \
    --write_png || true

  latest_phase33_png="$(ls -dt runs/pretty_plots/*${PHASE33_GROUP}*/dashboard.png 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_phase33_png}" ]]; then
    mkdir -p paper/figures
    cp "${latest_phase33_png}" "paper/figures/phase33_hardset_budget1_dashboard.png"
  fi

  python3 paper/scripts/build_phase33_tables.py \
    --leaderboard "runs/${PHASE33_GROUP}/costly_leaderboard_min${PHASE33_MIN_TASKS}.json" \
    --run_group "${PHASE33_GROUP}" \
    --hardset_json "${PHASE33_HARDSET_JSON}" || true
fi

if [[ -d "runs/${PHASE34_GROUP}" ]]; then
  python3 scripts/costly_leaderboard.py \
    --run_group "${PHASE34_GROUP}" \
    --min_tasks 12 \
    --out_json "runs/${PHASE34_GROUP}/costly_leaderboard_min12.json" \
    > "runs/${PHASE34_GROUP}/costly_leaderboard_min12.txt" || true

  python3 scripts/paired_compare.py \
    --run_group "${PHASE34_GROUP}" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b 1 \
    --feedback_budget 1 \
    --min_tasks 12 \
    > "runs/${PHASE34_GROUP}/paired_adaptive_vs_fixedkj1_b1.json" || true

  python3 scripts/plot_pretty_dashboard.py \
    --run_group "${PHASE34_GROUP}" \
    --min_tasks 12 \
    --title "Phase34 H12 Pairwise Power (b=1; adaptive vs k1 judge)" \
    --write_png || true

  latest_phase34_png="$(ls -dt runs/pretty_plots/*${PHASE34_GROUP}*/dashboard.png 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_phase34_png}" ]]; then
    mkdir -p paper/figures
    cp "${latest_phase34_png}" "paper/figures/phase34_h12_pairwise_dashboard.png"
  fi

  python3 paper/scripts/build_phase34_tables.py \
    --leaderboard "runs/${PHASE34_GROUP}/costly_leaderboard_min12.json" \
    --run_group "${PHASE34_GROUP}" \
    --paired_phase28 "runs/phase28_combined/paired_adaptive_vs_fixedkj1_b1.json" || true
fi

declare -a DOMAINS=("numeric:10" "string:8" "symbolic:7")
for entry in "${DOMAINS[@]}"; do
  domain="${entry%%:*}"
  min_tasks="${entry##*:}"
  group="verify_phase31_${domain}_budget1_s01234567"

  if [[ ! -d "runs/${group}" ]]; then
    continue
  fi

  python3 scripts/costly_leaderboard.py \
    --run_group "${group}" \
    --min_tasks "${min_tasks}" \
    --out_json "runs/${group}/costly_leaderboard_min${min_tasks}.json" \
    > "runs/${group}/costly_leaderboard_min${min_tasks}.txt" || true

  python3 scripts/paired_compare.py \
    --run_group "${group}" \
    --method_a adaptive_fwb \
    --method_b fixed_k_judge \
    --inner_updates_b 1 \
    --feedback_budget 1 \
    --min_tasks "${min_tasks}" \
    > "runs/${group}/paired_adaptive_vs_fixedkj1_b1.json" || true

  python3 scripts/plot_pretty_dashboard.py \
    --run_group "${group}" \
    --min_tasks "${min_tasks}" \
    --title "Phase31 ${domain} Rare-Feedback (b=1; live)" \
    --write_png || true
done

python3 paper/scripts/build_phase31_tables.py || true

if [[ -f "runs/verify_phase31_numeric_budget1_s01234567/costly_leaderboard_min10.json" ]] && \
   [[ -f "runs/verify_phase31_string_budget1_s01234567/costly_leaderboard_min8.json" ]] && \
   [[ -f "runs/verify_phase31_symbolic_budget1_s01234567/costly_leaderboard_min7.json" ]]; then
  python3 scripts/domain_slice_matrix.py \
    --slice "numeric=runs/verify_phase31_numeric_budget1_s01234567/costly_leaderboard_min10.json" \
    --slice "string=runs/verify_phase31_string_budget1_s01234567/costly_leaderboard_min8.json" \
    --slice "symbolic=runs/verify_phase31_symbolic_budget1_s01234567/costly_leaderboard_min7.json" \
    --title "Phase31 Domain Transfer (Budget=1, Seeds 0-7)" \
    --write_png || true

  latest_matrix_dir="$(ls -dt runs/pretty_plots/*domain-slice-matrix-numeric-string-symbolic 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_matrix_dir}" && -f "${latest_matrix_dir}/dashboard.png" ]]; then
    mkdir -p paper/figures
    cp "${latest_matrix_dir}/dashboard.png" "paper/figures/phase31_domains_budget1_matrix.png"
  fi
fi

python3 scripts/build_live_status.py
python3 scripts/build_win_snapshot.py || true
python3 paper/scripts/build_live_scorecard.py || true
python3 scripts/build_publish_readiness.py || true

python3 scripts/build_ablation_status.py \
  --run_group verify_phase31_numeric_budget1_s01234567 \
  --run_group verify_phase31_string_budget1_s01234567 \
  --run_group verify_phase31_symbolic_budget1_s01234567 \
  --run_group verify_phase30b_mbpp20_budget1_s0123_nors \
  --run_group verify_phase32_synth20_budget1_s0123_nors \
  --run_group verify_phase33_hardset_numeric_budget1_s01234567 \
  --run_group verify_phase33_hardset_h12pair_budget1_s1223 \
  --run_group verify_phase34_h12_budget1_pairwise_s1223 \
  --run_group verify_phase36_h80_llama31_70b_budget1_s012 \
  --run_group verify_phase37_h12_budget1_compute_match_s01234567 \
  --run_group verify_phase38_mbppplus_synth_budget1_s012 \
  --run_group verify_phase28_h12_budget1_s01234567 \
  --run_group verify_phase28_h12_budget1_ext_s891011 \
  --out_json runs/ablation_status.json \
  --out_md runs/ablation_status.md || true

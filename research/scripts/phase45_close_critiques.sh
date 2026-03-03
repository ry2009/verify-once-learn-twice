#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="runs/phase45_close_critiques"
mkdir -p "$OUT_DIR"

phase40_groups=(
  "verify_phase40_mbppplus_compute_match_b1_s3"
  "verify_phase40_mbppplus_compute_match_b1_s4"
  "verify_phase40_mbppplus_compute_match_b1_s5"
  "verify_phase40_mbppplus_compute_match_b1_s6"
)

phase47_split_groups=(
  "verify_phase47_mbppplus_compute_match_split_s3_fixed_k_judge_k1"
  "verify_phase47_mbppplus_compute_match_split_s3_fixed_k_judge_k2"
  "verify_phase47_mbppplus_compute_match_split_s3_inference_only"
  "verify_phase47_mbppplus_compute_match_split_s4_fixed_k_judge_k1"
  "verify_phase47_mbppplus_compute_match_split_s4_fixed_k_judge_k2"
  "verify_phase47_mbppplus_compute_match_split_s4_inference_only"
  "verify_phase47_mbppplus_compute_match_split_s5_fixed_k_judge_k1"
  "verify_phase47_mbppplus_compute_match_split_s5_fixed_k_judge_k2"
  "verify_phase47_mbppplus_compute_match_split_s5_inference_only"
  "verify_phase47_mbppplus_compute_match_split_s6_fixed_k_judge_k1"
  "verify_phase47_mbppplus_compute_match_split_s6_fixed_k_judge_k2"
  "verify_phase47_mbppplus_compute_match_split_s6_inference_only"
)

phase41_groups=(
  "verify_phase41_h80_70b_retune_b1_s0"
  "verify_phase41_h80_70b_retune_b1_s1"
  "verify_phase41_h80_70b_retune_b1_s2"
  "verify_phase41_h80_70b_retune_b1_s3"
  "verify_phase41_h80_70b_retune_b1_s4"
  "verify_phase41_h80_70b_retune_b1_s5"
  "verify_phase41_h80_70b_retune_b1_s6"
  "verify_phase41_h80_70b_retune_b1_s7"
  "verify_phase41_h80_70b_retune_b1_s8"
  "verify_phase41_h80_70b_retune_b1_s9"
)

phase42_llm_groups=(
  "verify_phase42_h30_b1_judgeablation_llm_s0123456789_s0to4"
  "verify_phase42_h30_b1_judgeablation_llm_s0123456789_s5to9"
)

phase42_oracle_groups=(
  "verify_phase42_h30_b1_judgeablation_oracle_s0123456789_s0to4"
  "verify_phase42_h30_b1_judgeablation_oracle_s0123456789_s5to9"
)

phase43_groups=(
  "verify_phase43_grpo_compat_h80_b1_s0"
  "verify_phase43_grpo_compat_h80_b1_s1"
  "verify_phase43_grpo_compat_h80_b1_s2"
  "verify_phase43_grpo_compat_h80_b1_s3"
  "verify_phase43_grpo_compat_h80_b1_s4"
  "verify_phase43_grpo_compat_h80_b1_s5"
)

phase44_groups=(
  "verify_phase44_symbolic_depth_b1_s0"
  "verify_phase44_symbolic_depth_b1_s1"
  "verify_phase44_symbolic_depth_b1_s2"
  "verify_phase44_symbolic_depth_b1_s3"
  "verify_phase44_symbolic_depth_b1_s4"
  "verify_phase44_symbolic_depth_b1_s5"
  "verify_phase44_symbolic_depth_b1_s6"
  "verify_phase44_symbolic_depth_b1_s7"
)

join_groups() {
  local arr=("$@")
  local out=()
  local g
  for g in "${arr[@]}"; do
    out+=(--run_group "$g")
  done
  printf '%q ' "${out[@]}"
}

run_costly() {
  local name="$1"; shift
  # shellcheck disable=SC2206
  local args=($*)
  python3 scripts/costly_leaderboard.py \
    "${args[@]}" \
    --min_tasks 10 \
    --out_json "$OUT_DIR/${name}_leaderboard.json" \
    > "$OUT_DIR/${name}_leaderboard.csv" || true
}

run_pair() {
  local name="$1"; shift
  # shellcheck disable=SC2206
  local args=($*)
  python3 scripts/paired_compare.py \
    "${args[@]}" \
    --bootstrap_samples 20000 \
    --alpha 0.05 \
    > "$OUT_DIR/${name}.json" || true
}

run_costly "phase40" "$(join_groups "${phase40_groups[@]}" "${phase47_split_groups[@]}")"
run_costly "phase41" "$(join_groups "${phase41_groups[@]}")"
run_costly "phase42_llm" "$(join_groups "${phase42_llm_groups[@]}")"
run_costly "phase42_oracle" "$(join_groups "${phase42_oracle_groups[@]}")"
run_costly "phase43" "$(join_groups "${phase43_groups[@]}")"
run_costly "phase44" "$(join_groups "${phase44_groups[@]}")"

run_pair "phase40_adaptive_vs_k1" \
  "$(join_groups "${phase40_groups[@]}" "${phase47_split_groups[@]}")" \
  --method_a adaptive_fwb --method_b fixed_k_judge --inner_updates_b 1 --feedback_budget 1 --pair_key seed --min_tasks 10
run_pair "phase40_adaptive_vs_k2" \
  "$(join_groups "${phase40_groups[@]}" "${phase47_split_groups[@]}")" \
  --method_a adaptive_fwb --method_b fixed_k_judge --inner_updates_b 2 --feedback_budget 1 --pair_key seed --min_tasks 10

run_pair "phase41_adaptive_vs_k1" \
  "$(join_groups "${phase41_groups[@]}")" \
  --method_a adaptive_fwb --method_b fixed_k_judge --inner_updates_b 1 --feedback_budget 1 --min_tasks 40
run_pair "phase41_adaptive_vs_k2" \
  "$(join_groups "${phase41_groups[@]}")" \
  --method_a adaptive_fwb --method_b fixed_k_judge --inner_updates_b 2 --feedback_budget 1 --min_tasks 40

run_pair "phase42_llm_adaptive_vs_k2" \
  "$(join_groups "${phase42_llm_groups[@]}")" \
  --method_a adaptive_fwb --method_b fixed_k_judge --inner_updates_b 2 --feedback_budget 1 --min_tasks 10
run_pair "phase42_oracle_adaptive_vs_k2" \
  "$(join_groups "${phase42_oracle_groups[@]}")" \
  --method_a adaptive_fwb --method_b fixed_k_judge --inner_updates_b 2 --feedback_budget 1 --min_tasks 10

run_pair "phase43_adaptive_vs_fixedfwbk2" \
  "$(join_groups "${phase43_groups[@]}")" \
  --method_a adaptive_fwb --method_b fixed_k_fwb --inner_updates_b 2 --feedback_budget 1 --min_tasks 40
run_pair "phase43_adaptive_vs_k2" \
  "$(join_groups "${phase43_groups[@]}")" \
  --method_a adaptive_fwb --method_b fixed_k_judge --inner_updates_b 2 --feedback_budget 1 --min_tasks 40

python3 - <<'PY'
import json
from pathlib import Path

out = Path("runs/phase45_close_critiques")

def load(path):
    p = out / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

rows = []
for name in [
    "phase40_adaptive_vs_k1.json",
    "phase40_adaptive_vs_k2.json",
    "phase41_adaptive_vs_k1.json",
    "phase41_adaptive_vs_k2.json",
    "phase42_llm_adaptive_vs_k2.json",
    "phase42_oracle_adaptive_vs_k2.json",
    "phase43_adaptive_vs_fixedfwbk2.json",
    "phase43_adaptive_vs_k2.json",
]:
    d = load(name)
    if not d:
        continue
    s = d.get("delta_success", {})
    rows.append(
        {
            "name": name.replace(".json", ""),
            "pairs": d.get("pairs"),
            "delta": s.get("mean"),
            "ci": s.get("ci"),
            "p": s.get("p_signflip_mean_greater"),
            "wins": s.get("wins"),
            "losses": s.get("losses"),
            "ties": s.get("ties"),
        }
    )

md = ["# Phase-45 Close-Critiques Summary", ""]
md.append("| Comparison | Pairs | Delta Success | 95% CI | p(sign-flip) | Wins/Losses/Ties |")
md.append("|---|---:|---:|---|---:|---:|")
for r in rows:
    ci = r["ci"] if isinstance(r["ci"], list) and len(r["ci"]) == 2 else [None, None]
    ci_txt = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "n/a"
    md.append(
        f"| {r['name']} | {r['pairs']} | {r['delta']:.4f} | {ci_txt} | {r['p']:.4f} | "
        f"{r['wins']}/{r['losses']}/{r['ties']} |"
    )

(out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print((out / "summary.md").read_text(encoding="utf-8"))
PY

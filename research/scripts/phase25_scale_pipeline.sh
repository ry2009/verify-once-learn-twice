#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[phase25] start $(date '+%Y-%m-%d %H:%M:%S')"

python3 scripts/validate_tasks.py \
  --tasks data/synth_stringops_20_rich_v2_hidden70_probe.jsonl \
  --tasks data/mbpp_20_v2_hidden70_probe.jsonl

run_exists() {
  local group="$1"
  local run_name="$2"
  if [[ ! -d "runs/${group}" ]]; then
    return 1
  fi
  rg -l "\"run_name\"\\s*:\\s*\"${run_name}\"" "runs/${group}" -g '**/config.json' >/dev/null 2>&1
}

run_one() {
  local group="$1"
  local tasks="$2"
  local run_name="$3"
  shift 3
  if run_exists "$group" "$run_name"; then
    echo "[phase25] skip ${group}/${run_name} (already exists)"
    return
  fi
  python3 scripts/run_experiment.py \
    --tasks "$tasks" \
    --run_group "$group" \
    --ablation_tag phase25_scale_costly_feedback \
    --run_name "$run_name" \
    --judge_mode oracle_binary \
    --max_steps 20 \
    --max_tokens 128 \
    --temperature 0.2 \
    --no_reset_per_task \
    --reset_every_n_tasks 4 \
    --lr 2e-5 \
    "$@"
}

run_domain() {
  local domain="$1"
  local tasks="$2"
  local group="$3"
  local min_tasks="$4"

  echo "[phase25] run domain=${domain}"
  for seed in 0 1 2; do
    for b in 2 4; do
      run_one "$group" "$tasks" "adaptive_fwb-fix-b${b}-s${seed}" \
        --method adaptive_fwb \
        --feedback_budget "$b" \
        --seed "$seed" \
        --inner_updates 1 \
        --adaptive_judge_every_updates 1 \
        --max_adaptive_steps_per_feedback 2 \
        --teacher_resample_attempts 2 \
        --teacher_filter_policy pass_or_assertion

      for k in 1 2 4; do
        run_one "$group" "$tasks" "fixed_k_judge-k${k}-fix-b${b}-s${seed}" \
          --method fixed_k_judge \
          --feedback_budget "$b" \
          --seed "$seed" \
          --inner_updates "$k" \
          --teacher_resample_attempts 3 \
          --teacher_filter_with_judge \
          --teacher_filter_policy pass_or_assertion
      done

      run_one "$group" "$tasks" "inference_only-fix-b${b}-s${seed}" \
        --method inference_only \
        --feedback_budget "$b" \
        --seed "$seed"
    done
  done

  echo "[phase25] leaderboard domain=${domain}"
  python3 scripts/costly_leaderboard.py \
    --run_group "$group" \
    --min_tasks "$min_tasks" \
    --out_json "runs/${group}/costly_leaderboard_min${min_tasks}.json" \
    > "runs/${group}/costly_leaderboard_min${min_tasks}.txt"

  echo "[phase25] paired domain=${domain}"
  for b in 2 4; do
    for k in 1 2 4; do
      python3 scripts/paired_compare.py \
        --run_group "$group" \
        --method_a adaptive_fwb \
        --method_b fixed_k_judge \
        --inner_updates_b "$k" \
        --feedback_budget "$b" \
        --min_tasks "$min_tasks" \
        > "runs/${group}/paired_adaptive_vs_fixedkj${k}_b${b}.json"
    done
    python3 scripts/paired_compare.py \
      --run_group "$group" \
      --method_a adaptive_fwb \
      --method_b inference_only \
      --feedback_budget "$b" \
      --min_tasks "$min_tasks" \
      > "runs/${group}/paired_adaptive_vs_inference_b${b}.json"
  done

  python3 scripts/plot_pretty_dashboard.py \
    --run_group "$group" \
    --min_tasks "$min_tasks" \
    --title "Phase25 ${domain} (validated odd-domain, b=2,4; k=1,2,4; s012)" \
    --write_png
}

run_domain "synth20_hidden70" \
  "data/synth_stringops_20_rich_v2_hidden70_probe.jsonl" \
  "verify_phase25_synth20_hidden70_budget24_s012" \
  20

run_domain "mbpp20_hidden70" \
  "data/mbpp_20_v2_hidden70_probe.jsonl" \
  "verify_phase25_mbpp20_hidden70_budget24_s012" \
  20

python3 - <<'PY'
import json
from pathlib import Path

domains = [
    ("synth20_hidden70", "verify_phase25_synth20_hidden70_budget24_s012", 20),
    ("mbpp20_hidden70", "verify_phase25_mbpp20_hidden70_budget24_s012", 20),
]

lines = ["# Phase25 Final Summary", ""]
for domain, group, min_tasks in domains:
    path = Path(f"runs/{group}/costly_leaderboard_min{min_tasks}.json")
    if not path.exists():
        continue
    lb = json.loads(path.read_text())
    rows = lb.get("aggregate", [])
    lines.append(f"## {domain} ({group})")
    for b in (2, 4):
        pick = [r for r in rows if int(r.get("feedback_budget", -1)) == b]
        amap = {r["method_variant"]: r for r in pick}
        a = amap.get("adaptive_fwb")
        if not a:
            continue
        k1 = amap.get("fixed_k_judge_k1")
        k2 = amap.get("fixed_k_judge_k2")
        k4 = amap.get("fixed_k_judge_k4")
        inf = amap.get("inference_only")
        lines.append(
            f"- b={b} adaptive={a['success_mean']:.4f} fb={a['feedback_mean']:.4f} train={a['train_mean']:.4f}"
        )
        if k1:
            lines.append(f"  fixed_kj1={k1['success_mean']:.4f} fb={k1['feedback_mean']:.4f}")
        if k2:
            lines.append(f"  fixed_kj2={k2['success_mean']:.4f} fb={k2['feedback_mean']:.4f}")
        if k4:
            lines.append(f"  fixed_kj4={k4['success_mean']:.4f} fb={k4['feedback_mean']:.4f}")
        if inf:
            lines.append(f"  inference={inf['success_mean']:.4f} fb={inf['feedback_mean']:.4f}")
    lines.append("")

out = Path("runs/phase25_final_summary.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[phase25] wrote {out}")
PY

echo "[phase25] done $(date '+%Y-%m-%d %H:%M:%S')"

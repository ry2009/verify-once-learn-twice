#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[phase23] start $(date '+%Y-%m-%d %H:%M:%S')"

echo "[phase23] validate datasets"
python3 scripts/validate_tasks.py \
  --tasks data/humaneval_10_hidden70_coarse_fail2of3_seed2026.jsonl \
  --tasks data/humaneval_12_expensivefb600_fail2of3_seed012.jsonl

echo "[phase23] run h10 budget 2/4 sweep"
python3 scripts/run_ablation.py --spec data/ablation_phase23_h10_budget24_core_s01234.json

echo "[phase23] run h12 budget 2/4 sweep"
python3 scripts/run_ablation.py --spec data/ablation_phase23_h12_budget24_core_s01234.json

echo "[phase23] build leaderboards"
python3 scripts/costly_leaderboard.py \
  --run_group verify_phase23_h10_budget24_core_s01234 \
  --min_tasks 10 \
  --out_json runs/verify_phase23_h10_budget24_core_s01234/costly_leaderboard_min10.json \
  > runs/verify_phase23_h10_budget24_core_s01234/costly_leaderboard_min10.txt

python3 scripts/costly_leaderboard.py \
  --run_group verify_phase23_h12_budget24_core_s01234 \
  --min_tasks 12 \
  --out_json runs/verify_phase23_h12_budget24_core_s01234/costly_leaderboard_min12.json \
  > runs/verify_phase23_h12_budget24_core_s01234/costly_leaderboard_min12.txt

echo "[phase23] paired comparisons"
for group in verify_phase23_h10_budget24_core_s01234 verify_phase23_h12_budget24_core_s01234; do
  if [[ "$group" == "verify_phase23_h10_budget24_core_s01234" ]]; then
    min_tasks=10
  else
    min_tasks=12
  fi
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
  done
done

echo "[phase23] dashboards"
python3 scripts/plot_pretty_dashboard.py \
  --run_group verify_phase23_h10_budget24_core_s01234 \
  --min_tasks 10 \
  --title "Phase23 H10 (b=2,4; k=1,2,4; s01234)" \
  --write_png

python3 scripts/plot_pretty_dashboard.py \
  --run_group verify_phase23_h12_budget24_core_s01234 \
  --min_tasks 12 \
  --title "Phase23 H12 (b=2,4; k=1,2,4; s01234)" \
  --write_png

echo "[phase23] synth summary"
python3 - <<'PY'
import json
from pathlib import Path

def load(path: str) -> dict:
    return json.loads(Path(path).read_text())

def agg_map(lb: dict) -> dict:
    return {row["method_variant"]: row for row in lb.get("aggregate", [])}

rows = []
for group, min_tasks in [
    ("verify_phase23_h10_budget24_core_s01234", 10),
    ("verify_phase23_h12_budget24_core_s01234", 12),
]:
    lb = load(f"runs/{group}/costly_leaderboard_min{min_tasks}.json")
    agg = agg_map(lb)
    for b in (2, 4):
        def pick(key: str):
            return [
                r for r in lb.get("aggregate", [])
                if r.get("method_variant") == key and int(r.get("feedback_budget", -1)) == b
            ]
        a = pick("adaptive_fwb")
        k1 = pick("fixed_k_judge_k1")
        k2 = pick("fixed_k_judge_k2")
        k4 = pick("fixed_k_judge_k4")
        if not a:
            continue
        row = {
            "group": group,
            "budget": b,
            "adaptive_success": a[0]["success_mean"],
            "adaptive_feedback": a[0]["feedback_mean"],
            "k1_success": k1[0]["success_mean"] if k1 else None,
            "k2_success": k2[0]["success_mean"] if k2 else None,
            "k4_success": k4[0]["success_mean"] if k4 else None,
        }
        rows.append(row)

lines = []
lines.append("# Phase23 Final Summary")
lines.append("")
for r in rows:
    lines.append(f"## {r['group']} budget={r['budget']}")
    lines.append(
        f"- adaptive_success={r['adaptive_success']:.4f} adaptive_feedback={r['adaptive_feedback']:.4f}"
    )
    if r["k1_success"] is not None:
        lines.append(f"- fixed_k_judge_k1_success={r['k1_success']:.4f}")
    if r["k2_success"] is not None:
        lines.append(f"- fixed_k_judge_k2_success={r['k2_success']:.4f}")
    if r["k4_success"] is not None:
        lines.append(f"- fixed_k_judge_k4_success={r['k4_success']:.4f}")
    lines.append("")

out = Path("runs/phase23_final_summary.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[phase23] wrote {out}")
PY

echo "[phase23] done $(date '+%Y-%m-%d %H:%M:%S')"

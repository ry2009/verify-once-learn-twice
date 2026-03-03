#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[phase24] start $(date '+%Y-%m-%d %H:%M:%S')"

echo "[phase24] validate datasets"
python3 scripts/validate_tasks.py \
  --tasks data/humaneval_80_hidden70_coarse_numeric.jsonl \
  --tasks data/humaneval_80_hidden70_coarse_string.jsonl \
  --tasks data/humaneval_80_hidden70_coarse_symbolic.jsonl

declare -a DOMAINS=("numeric:10" "string:8" "symbolic:7")

for entry in "${DOMAINS[@]}"; do
  domain="${entry%%:*}"
  min_tasks="${entry##*:}"
  spec="data/ablation_phase24_${domain}_budget24_s01234.json"
  group="verify_phase24_${domain}_budget24_s01234"

  echo "[phase24] run ${domain} sweep"
  python3 scripts/run_ablation.py --spec "$spec"

  echo "[phase24] leaderboard ${domain}"
  python3 scripts/costly_leaderboard.py \
    --run_group "$group" \
    --min_tasks "$min_tasks" \
    --out_json "runs/${group}/costly_leaderboard_min${min_tasks}.json" \
    > "runs/${group}/costly_leaderboard_min${min_tasks}.txt"

  echo "[phase24] paired comparisons ${domain}"
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

  echo "[phase24] dashboard ${domain}"
  python3 scripts/plot_pretty_dashboard.py \
    --run_group "$group" \
    --min_tasks "$min_tasks" \
    --title "Phase24 ${domain} (b=2,4; k=1,2,4; s01234)" \
    --write_png
done

echo "[phase24] synth summary"
python3 - <<'PY'
import json
from pathlib import Path

domains = [
    ("numeric", 10),
    ("string", 8),
    ("symbolic", 7),
]

lines = []
lines.append("# Phase24 Final Summary")
lines.append("")
for domain, min_tasks in domains:
    group = f"verify_phase24_{domain}_budget24_s01234"
    lb = json.loads(
        Path(f"runs/{group}/costly_leaderboard_min{min_tasks}.json").read_text()
    )
    rows = lb.get("aggregate", [])
    lines.append(f"## {group}")
    for b in (2, 4):
        pick = [r for r in rows if int(r.get("feedback_budget", -1)) == b]
        amap = {r["method_variant"]: r for r in pick}
        if "adaptive_fwb" not in amap:
            continue
        a = amap["adaptive_fwb"]
        k1 = amap.get("fixed_k_judge_k1")
        k2 = amap.get("fixed_k_judge_k2")
        k4 = amap.get("fixed_k_judge_k4")
        inf = amap.get("inference_only")
        lines.append(
            f"- b={b} adaptive={a['success_mean']:.4f} fb={a['feedback_mean']:.4f}"
        )
        if k1:
            lines.append(f"  fixed_kj1={k1['success_mean']:.4f}")
        if k2:
            lines.append(f"  fixed_kj2={k2['success_mean']:.4f}")
        if k4:
            lines.append(f"  fixed_kj4={k4['success_mean']:.4f}")
        if inf:
            lines.append(f"  inference={inf['success_mean']:.4f}")
    lines.append("")

out = Path("runs/phase24_final_summary.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[phase24] wrote {out}")
PY

echo "[phase24] done $(date '+%Y-%m-%d %H:%M:%S')"

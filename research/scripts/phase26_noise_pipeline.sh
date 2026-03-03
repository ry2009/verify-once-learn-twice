#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_GROUP="verify_phase26_h12_noise_budget24_s01234"

echo "[phase26] start $(date '+%Y-%m-%d %H:%M:%S')"

python3 scripts/validate_tasks.py \
  --tasks data/humaneval_12_expensivefb600_fail2of3_seed012.jsonl

echo "[phase26] run sweep"
python3 scripts/run_ablation.py --spec data/ablation_phase26_h12_noise_budget24_s01234.json

echo "[phase26] leaderboard"
python3 scripts/costly_leaderboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --out_json "runs/${RUN_GROUP}/costly_leaderboard_min12.json" \
  > "runs/${RUN_GROUP}/costly_leaderboard_min12.txt"

echo "[phase26] paired comparisons"
for b in 2 4; do
  for noise in 0.0 0.1 0.2; do
    noise_tag="$(printf '%.1f' "$noise" | tr '.' 'p')"
    python3 scripts/paired_compare.py \
      --run_group "$RUN_GROUP" \
      --method_a adaptive_fwb \
      --method_b fixed_k_judge \
      --inner_updates_b 1 \
      --feedback_budget "$b" \
      --judge_flip_prob "$noise" \
      --min_tasks 12 \
      > "runs/${RUN_GROUP}/paired_adaptive_vs_fixedkj1_b${b}_noise${noise_tag}.json"
  done
done

python3 scripts/plot_pretty_dashboard.py \
  --run_group "$RUN_GROUP" \
  --min_tasks 12 \
  --title "Phase26 H12 Judge-Noise Robustness (b=2,4; noise=0/10/20%; s01234)" \
  --write_png

python3 - <<'PY'
import json
from pathlib import Path

group = "verify_phase26_h12_noise_budget24_s01234"
lb = json.loads(Path(f"runs/{group}/costly_leaderboard_min12.json").read_text())
rows = lb.get("aggregate", [])

def pick(method: str, budget: int, noise: float):
    for row in rows:
        if (
            row.get("method_variant") == method
            and int(row.get("feedback_budget", -1)) == budget
            and abs(float(row.get("judge_flip_prob", 0.0)) - noise) < 1e-12
        ):
            return row
    return None

lines = ["# Phase26 Final Summary", ""]
for noise in (0.0, 0.1, 0.2):
    lines.append(f"## noise={noise:.1f}")
    for b in (2, 4):
        a = pick("adaptive_fwb", b, noise)
        k1 = pick("fixed_k_judge_k1", b, noise)
        if not a or not k1:
            continue
        lines.append(
            f"- b={b} adaptive={a['success_mean']:.4f} fb={a['feedback_mean']:.4f} train={a['train_mean']:.4f}"
        )
        lines.append(
            f"- b={b} fixed_k_judge_k1={k1['success_mean']:.4f} fb={k1['feedback_mean']:.4f} train={k1['train_mean']:.4f}"
        )
    lines.append("")

Path("runs/phase26_final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote runs/phase26_final_summary.md")
PY

echo "[phase26] done $(date '+%Y-%m-%d %H:%M:%S')"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[phase22] start $(date '+%Y-%m-%d %H:%M:%S')"

echo "[phase22] validate datasets"
python3 scripts/validate_tasks.py \
  --tasks data/humaneval_10_hidden70_coarse_fail2of3_seed2026.jsonl \
  --tasks data/humaneval_12_expensivefb600_fail2of3_seed012.jsonl

echo "[phase22] run h10 sweep"
python3 scripts/run_ablation.py --spec data/ablation_phase22_h10_guardreplay_core_s01234.json

echo "[phase22] run h12 sweep"
python3 scripts/run_ablation.py --spec data/ablation_phase22_h12_guardreplay_core_s01234.json

echo "[phase22] build metrics and figures"
python3 scripts/costly_leaderboard.py \
  --run_group verify_phase22_h10_guardreplay_core_s01234 \
  --min_tasks 10 \
  --out_json runs/verify_phase22_h10_guardreplay_core_s01234/costly_leaderboard_min10.json \
  > runs/verify_phase22_h10_guardreplay_core_s01234/costly_leaderboard_min10.txt

python3 scripts/costly_leaderboard.py \
  --run_group verify_phase22_h12_guardreplay_core_s01234 \
  --min_tasks 12 \
  --out_json runs/verify_phase22_h12_guardreplay_core_s01234/costly_leaderboard_min12.json \
  > runs/verify_phase22_h12_guardreplay_core_s01234/costly_leaderboard_min12.txt

python3 scripts/paired_compare.py \
  --run_group verify_phase22_h10_guardreplay_core_s01234 \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 1 \
  --feedback_budget 2 \
  --min_tasks 10 \
  > runs/verify_phase22_h10_guardreplay_core_s01234/paired_adaptive_vs_fixedkj1.json

python3 scripts/paired_compare.py \
  --run_group verify_phase22_h12_guardreplay_core_s01234 \
  --method_a adaptive_fwb \
  --method_b fixed_k_judge \
  --inner_updates_b 1 \
  --feedback_budget 2 \
  --min_tasks 12 \
  > runs/verify_phase22_h12_guardreplay_core_s01234/paired_adaptive_vs_fixedkj1.json

python3 scripts/paired_compare.py \
  --run_group verify_phase22_h10_guardreplay_core_s01234 \
  --method_a adaptive_fwb \
  --method_b fixed_k_fwb \
  --inner_updates_b 1 \
  --feedback_budget 2 \
  --min_tasks 10 \
  > runs/verify_phase22_h10_guardreplay_core_s01234/paired_adaptive_vs_fixedkfwb1.json

python3 scripts/paired_compare.py \
  --run_group verify_phase22_h12_guardreplay_core_s01234 \
  --method_a adaptive_fwb \
  --method_b fixed_k_fwb \
  --inner_updates_b 1 \
  --feedback_budget 2 \
  --min_tasks 12 \
  > runs/verify_phase22_h12_guardreplay_core_s01234/paired_adaptive_vs_fixedkfwb1.json

python3 scripts/plot_pretty_dashboard.py \
  --run_group verify_phase22_h10_guardreplay_core_s01234 \
  --min_tasks 10 \
  --title "Phase22 H10 (s01234, b=2)" \
  --write_png

python3 scripts/plot_pretty_dashboard.py \
  --run_group verify_phase22_h12_guardreplay_core_s01234 \
  --min_tasks 12 \
  --title "Phase22 H12 (s01234, b=2)" \
  --write_png

python3 - <<'PY'
import json
from pathlib import Path

rows = []
for group, min_tasks in [
    ("verify_phase22_h10_guardreplay_core_s01234", 10),
    ("verify_phase22_h12_guardreplay_core_s01234", 12),
]:
    lb_path = Path("runs") / group / f"costly_leaderboard_min{min_tasks}.json"
    pair_path = Path("runs") / group / "paired_adaptive_vs_fixedkj1.json"
    lb = json.loads(lb_path.read_text())
    agg = {r["method_variant"]: r for r in lb["aggregate"]}
    pair = json.loads(pair_path.read_text())
    rows.append(
        {
            "group": group,
            "adaptive_success": agg.get("adaptive_fwb", {}).get("success_mean", None),
            "fixedkj1_success": agg.get("fixed_k_judge_k1", {}).get("success_mean", None),
            "adaptive_feedback": agg.get("adaptive_fwb", {}).get("feedback_mean", None),
            "fixedkj1_feedback": agg.get("fixed_k_judge_k1", {}).get("feedback_mean", None),
            "delta_success": pair.get("delta_success", {}).get("mean", None),
            "delta_feedback": pair.get("delta_feedback", {}).get("mean", None),
            "shared_seeds": pair.get("shared_seeds", []),
        }
    )

lines = []
lines.append("# Phase22 Final Summary")
lines.append("")
for r in rows:
    lines.append(f"## {r['group']}")
    lines.append(
        f"- adaptive_success={r['adaptive_success']:.4f} fixedkj1_success={r['fixedkj1_success']:.4f}"
    )
    lines.append(
        f"- adaptive_feedback={r['adaptive_feedback']:.4f} fixedkj1_feedback={r['fixedkj1_feedback']:.4f}"
    )
    lines.append(
        f"- paired_delta_success={r['delta_success']:.4f} paired_delta_feedback={r['delta_feedback']:.4f}"
    )
    lines.append(f"- shared_seeds={r['shared_seeds']}")
    lines.append("")

out = Path("runs/phase22_final_summary.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[phase22] wrote {out}")
PY

echo "[phase22] done $(date '+%Y-%m-%d %H:%M:%S')"

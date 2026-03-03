#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set. Export it first, then rerun this script." >&2
  exit 1
fi

mkdir -p runs

# Clear stale phase38 workers so only one final worker runs.
screen -S phase38_parallel_s1s2 -X quit >/dev/null 2>&1 || true
screen -S phase38_k2s2_fast -X quit >/dev/null 2>&1 || true
screen -S phase38_infer_s2_fast -X quit >/dev/null 2>&1 || true
pkill -f "run_ablation.py --spec data/ablation_phase38_mbppplus_synth_budget1_s1s2_parallel.json" >/dev/null 2>&1 || true
pkill -f "run_experiment.py --tasks data/costly_mbpp_testall_plus_synth80.jsonl --run_name fixed_k_judge-k2-b1-s2-mbppplus" >/dev/null 2>&1 || true

# Keep machine awake while this final run finishes.
if ! screen -ls | rg -q "caffeinate_guard"; then
  screen -dmS caffeinate_guard bash -lc "caffeinate -dimsu"
fi

# Optional utility loops (safe to relaunch only if missing).
if ! screen -ls | rg -q "transfer_refresh_loop"; then
  screen -dmS transfer_refresh_loop bash -lc "cd \"$ROOT_DIR\" && scripts/refresh_transfer_loop.sh > runs/transfer_refresh_loop.log 2>&1"
fi
if ! screen -ls | rg -q "phase28_combine_loop"; then
  screen -dmS phase28_combine_loop bash -lc "cd \"$ROOT_DIR\" && while true; do bash scripts/phase28_combine.sh >> runs/phase28_combine_loop.log 2>&1 || true; sleep 300; done"
fi

# Final remaining phase38 run with manifest-aware tracking.
screen -dmS phase38_k2s2_final bash -lc \
  "cd \"$ROOT_DIR\" && TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 python3 scripts/run_ablation.py --spec data/ablation_phase38_k2s2_only.json | tee -a runs/autopilot_ablation_phase38_k2s2_only.log"

echo "Started phase38 final resume worker."
echo "Check: screen -ls"
echo "Tail : tail -n 40 runs/autopilot_ablation_phase38_k2s2_only.log"

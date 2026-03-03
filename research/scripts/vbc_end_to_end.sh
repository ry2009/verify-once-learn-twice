#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-data/ablation_vbc_smoke.json}"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set"
  exit 1
fi

python3 scripts/run_ablation.py --spec "${SPEC}"

RUN_GROUP="$(python3 - <<'PY' "${SPEC}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("run_group", "verify_vbc_smoke"))
PY
)"

python3 scripts/vbc_smoke_report.py --run_group "${RUN_GROUP}"
python3 scripts/costly_leaderboard.py \
  --run_group "${RUN_GROUP}" \
  --min_tasks 1 \
  --out_json "runs/${RUN_GROUP}/leaderboard.json" \
  > "runs/${RUN_GROUP}/leaderboard.csv"

echo "VBC end-to-end complete: runs/${RUN_GROUP}"


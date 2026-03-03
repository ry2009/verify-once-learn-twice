#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPEC="data/ablation_phase30_mbpp20_budget1_s0123.json"
MAN="runs/verify_phase30_mbpp20_budget1_s0123/ablation_manifest.json"

planned=$(python3 - <<'PY'
import json
print(len(json.load(open('data/ablation_phase30_mbpp20_budget1_s0123.json')).get('runs', [])))
PY
)

while true; do
  done_n=0
  if [[ -f "$MAN" ]]; then
    done_n=$(python3 - <<'PY'
import json, os
p='runs/verify_phase30_mbpp20_budget1_s0123/ablation_manifest.json'
if os.path.exists(p):
    print(len(json.load(open(p)).get('runs', [])))
else:
    print(0)
PY
)
  fi
  if [[ "$done_n" -ge "$planned" ]]; then
    break
  fi
  sleep 60
done

bash scripts/phase30_mbpp_budget1_post.sh

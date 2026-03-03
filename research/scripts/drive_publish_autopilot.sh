#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set" >&2
  exit 1
fi

MAX_ACTIVE="${MAX_ACTIVE_RUN_ABLATION:-3}"
SLEEP_SECS="${SLEEP_SECS:-90}"
REFRESH_EVERY_SECS="${REFRESH_EVERY_SECS:-600}"
STOP_ON_PUBLISH_READY="${STOP_ON_PUBLISH_READY:-1}"

LOG_FILE="runs/drive_publish_autopilot.log"
mkdir -p runs
touch "$LOG_FILE"

declare -a SPECS=(
  "data/ablation_phase31_numeric_budget1_s01234567.json"
  "data/ablation_phase34_h12_budget1_pairwise_s1223.json"
  "data/ablation_phase28_h12_budget1_s01234567.json"
  "data/ablation_phase31_string_budget1_s01234567.json"
  "data/ablation_phase31_symbolic_budget1_s01234567.json"
  "data/ablation_phase30b_mbpp20_budget1_s0123_nors.json"
  "data/ablation_phase32_synth20_budget1_s0123_nors.json"
  "data/ablation_phase37_h12_budget1_compute_match_s01234567.json"
  "data/ablation_phase36_h80_llama31_70b_budget1_s012.json"
  "data/ablation_phase28_h12_budget1_ext_s891011.json"
)

PHASE33_NUMERIC_SPEC="data/ablation_phase33_hardset_numeric_budget1_s01234567.json"
PHASE33_NUMERIC_TASKS="data/hardset_phase33_numeric_budget1_auto.jsonl"
PHASE33_H12PAIR_SPEC="data/ablation_phase33_hardset_h12pair_budget1_s1223.json"
PHASE33_H12PAIR_TASKS="data/hardset_phase33_h12pair_budget1_auto.jsonl"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
  echo "[autopilot] $*"
  echo "[autopilot] $*" >> "$LOG_FILE"
}

active_ablation_count() {
  python3 - <<'PY'
import re
import subprocess

out = subprocess.check_output(["ps", "-axo", "command"], text=True, errors="ignore")
pat = re.compile(r"^/.*/Python scripts/run_ablation.py --spec ")
count = sum(1 for line in out.splitlines() if pat.search(line))
print(count)
PY
}

spec_total_runs() {
  local spec="$1"
  python3 - "$spec" <<'PY'
import json,sys
from pathlib import Path
spec = Path(sys.argv[1])
try:
    obj = json.loads(spec.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)
print(len(obj.get("runs", [])))
PY
}

spec_run_group() {
  local spec="$1"
  python3 - "$spec" <<'PY'
import json,sys
from pathlib import Path
spec = Path(sys.argv[1])
try:
    obj = json.loads(spec.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
print(str(obj.get("run_group","")).strip())
PY
}

spec_done_runs() {
  local spec="$1"
  local group
  group="$(spec_run_group "$spec")"
  if [[ -z "$group" ]]; then
    echo 0
    return
  fi
  local manifest="runs/${group}/ablation_manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo 0
    return
  fi
  python3 - "$manifest" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
try:
    obj=json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)
print(len(obj.get("runs", [])))
PY
}

is_spec_running() {
  local spec="$1"
  python3 - "$spec" <<'PY'
import re
import subprocess
import sys

spec = sys.argv[1]
out = subprocess.check_output(["ps", "-axo", "command"], text=True, errors="ignore")
pat = re.compile(r"^/.*/Python scripts/run_ablation.py --spec " + re.escape(spec) + r"$")
sys.exit(0 if any(pat.search(line) for line in out.splitlines()) else 1)
PY
}

launch_spec() {
  local spec="$1"
  local slug
  slug="$(basename "$spec" .json | tr -cd '[:alnum:]_-' )"
  local session="auto_${slug}_$(date '+%H%M%S')"
  local run_log="runs/autopilot_${slug}.log"
  local cmd="export TINKER_API_KEY=\"${TINKER_API_KEY}\" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1; cd \"$ROOT_DIR\"; python3 scripts/run_ablation.py --spec \"$spec\" | tee -a \"$run_log\""
  screen -dmS "$session" bash -lc "$cmd"
  log "launched ${spec} in screen=${session}"
}

try_prepare_phase33() {
  # First choice: strict numeric hardset mined from phase31 numeric.
  python3 scripts/mine_adaptive_hardset.py \
    --run_group verify_phase31_numeric_budget1_s01234567 \
    --feedback_budget 1 \
    --min_tasks_per_run 10 \
    --min_shared_seeds 4 \
    --min_advantage_votes 2 \
    --top_k 20 \
    --min_selected 8 \
    --out_tasks "${PHASE33_NUMERIC_TASKS}" \
    --out_json runs/hardset_phase33_numeric_budget1_auto.json \
    --out_md runs/hardset_phase33_numeric_budget1_auto.md \
    --template_spec data/ablation_phase31_numeric_budget1_s01234567.json \
    --out_spec "${PHASE33_NUMERIC_SPEC}" \
    --out_run_group verify_phase33_hardset_numeric_budget1_s01234567 \
    --out_name phase33-hardset-numeric-budget1-s01234567 \
    >/dev/null 2>&1 || true
  if [[ -f "${PHASE33_NUMERIC_TASKS}" ]]; then
    local n_numeric
    n_numeric="$(wc -l < "${PHASE33_NUMERIC_TASKS}" | tr -d ' ')"
    if [[ "${n_numeric}" -ge 8 ]]; then
      echo "${PHASE33_NUMERIC_SPEC}"
      return 0
    fi
  fi

  # Fallback: pairwise H12 hardset, slightly relaxed floor to avoid deadlock.
  python3 scripts/mine_adaptive_hardset.py \
    --run_group verify_phase34_h12_budget1_pairwise_s1223 \
    --feedback_budget 1 \
    --min_tasks_per_run 10 \
    --min_shared_seeds 2 \
    --min_advantage_votes 1 \
    --top_k 20 \
    --min_selected 6 \
    --out_tasks "${PHASE33_H12PAIR_TASKS}" \
    --out_json runs/hardset_phase33_h12pair_budget1_auto.json \
    --out_md runs/hardset_phase33_h12pair_budget1_auto.md \
    --template_spec data/ablation_phase34_h12_budget1_pairwise_s1223.json \
    --out_spec "${PHASE33_H12PAIR_SPEC}" \
    --out_run_group verify_phase33_hardset_h12pair_budget1_s1223 \
    --out_name phase33-hardset-h12pair-budget1-s1223 \
    >/dev/null 2>&1 || return 1
  if [[ ! -f "${PHASE33_H12PAIR_TASKS}" ]]; then
    return 1
  fi
  local n_pair
  n_pair="$(wc -l < "${PHASE33_H12PAIR_TASKS}" | tr -d ' ')"
  if [[ "${n_pair}" -ge 6 ]]; then
    echo "${PHASE33_H12PAIR_SPEC}"
    return 0
  fi
  return 1
}

all_specs_complete() {
  local spec done total
  for spec in "${SPECS[@]}"; do
    total="$(spec_total_runs "$spec")"
    done="$(spec_done_runs "$spec")"
    if [[ "$done" -lt "$total" ]]; then
      return 1
    fi
  done
  return 0
}

refresh_all() {
  bash scripts/phase28_combine.sh || true
  bash scripts/refresh_transfer_metrics.sh || true
  (
    cd paper
    latexmk -pdf -interaction=nonstopmode main_phase20_live.tex >/dev/null 2>&1 || true
  )
}

publish_decision() {
  python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/publish_readiness.json")
if not p.exists():
    print("")
else:
    try:
        obj=json.loads(p.read_text(encoding="utf-8"))
        print(str(obj.get("campaign",{}).get("decision","")))
    except Exception:
        print("")
PY
}

last_refresh=0
log "start $(ts) max_active=${MAX_ACTIVE} sleep=${SLEEP_SECS}s refresh=${REFRESH_EVERY_SECS}s"

while true; do
  now="$(date +%s)"
  active="$(active_ablation_count)"
  log "tick $(ts) active=${active}"

  # Keep phase-33 mined hardset spec updated as phase31 numeric grows.
  phase33_ready_spec="$(try_prepare_phase33 || true)"
  if [[ -n "${phase33_ready_spec}" ]]; then
    if [[ ! " ${SPECS[*]} " =~ " ${phase33_ready_spec} " ]]; then
      SPECS+=("${phase33_ready_spec}")
      log "phase33 mining ready; added ${phase33_ready_spec} to schedule"
    fi
  fi

  # Launch pending specs up to concurrency cap.
  for spec in "${SPECS[@]}"; do
    total="$(spec_total_runs "$spec")"
    done="$(spec_done_runs "$spec")"
    if [[ "$done" -ge "$total" ]]; then
      continue
    fi
    if is_spec_running "$spec"; then
      continue
    fi
    active="$(active_ablation_count)"
    if [[ "$active" -ge "$MAX_ACTIVE" ]]; then
      break
    fi
    launch_spec "$spec"
    sleep 2
  done

  # Keep artifacts live.
  if (( now - last_refresh >= REFRESH_EVERY_SECS )); then
    log "refresh artifacts"
    refresh_all
    last_refresh="$now"
  fi

  # Exit if publish-ready gate is met and requested.
  if [[ "${STOP_ON_PUBLISH_READY}" == "1" ]]; then
    decision="$(publish_decision)"
    if [[ "$decision" == "publish_ready_candidate" ]]; then
      log "publish gate reached: ${decision}"
      refresh_all
      break
    fi
  fi

  # Exit if every scheduled spec is fully complete.
  if all_specs_complete; then
    log "all scheduled specs complete"
    refresh_all
    break
  fi

  sleep "$SLEEP_SECS"
done

log "done $(ts)"

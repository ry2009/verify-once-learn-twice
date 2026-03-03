#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set"
  echo "Export it first, then rerun:"
  echo "  export TINKER_API_KEY='...'"
  exit 1
fi

mkdir -p runs/phase67_launch runs/phase66_launch

start_if_missing() {
  local screen_name="$1"
  local spec_path="$2"
  local log_path="$3"
  if screen -ls | rg -q "\\.${screen_name}[[:space:]]"; then
    echo "[skip] screen already running: ${screen_name}"
    return 0
  fi
  echo "[start] ${screen_name} -> ${spec_path}"
  screen -dmS "${screen_name}" bash -lc \
    "cd \"$ROOT_DIR\" && TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 TINKER_API_KEY=\"$TINKER_API_KEY\" python3 scripts/run_ablation.py --spec \"$spec_path\" 2>&1 | tee \"$log_path\""
}

build_phase67_resume_shards() {
  python3 - <<'PY'
import json
from pathlib import Path

master = Path("data/ablation_phase67_t1l2r1_8b_t50_s012.json")
obj = json.loads(master.read_text(encoding="utf-8"))
group = str(obj.get("run_group", ""))
max_steps = int(obj.get("max_steps", 50))
group_dir = Path("runs") / group

def run_complete(run_name: str) -> bool:
    if not group_dir.exists():
        return False
    latest = None
    for d in group_dir.iterdir():
        if not d.is_dir():
            continue
        cfg = d / "config.json"
        ev = d / "events.jsonl"
        if not cfg.exists() or not ev.exists():
            continue
        try:
            cfg_obj = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(cfg_obj.get("run_name", "")) != run_name:
            continue
        mt = ev.stat().st_mtime
        if latest is None or mt > latest[0]:
            latest = (mt, ev)
    if latest is None:
        return False
    done = 0
    has_run_complete = False
    for line in latest[1].read_text(encoding="utf-8").splitlines():
        if '"type": "task_done"' in line:
            done += 1
        if '"type": "run_complete"' in line:
            has_run_complete = True
    return has_run_complete or done >= max_steps

remaining = []
for r in obj.get("runs", []):
    rn = str(r.get("run_name", ""))
    if not run_complete(rn):
        remaining.append(r)

for i in range(3):
    shard_runs = remaining[i::3]
    out = dict(obj)
    out["name"] = f"phase67-resume-shard{i}"
    out["runs"] = shard_runs
    out_path = Path(f"data/ablation_phase67_resume_shard{i}.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{out_path}:{len(shard_runs)}")
PY
}

phase66_b4s2_done() {
  python3 - <<'PY'
import json
from pathlib import Path

base = Path("runs/verify_phase66_kernelbench_target50_8b_adaptive_acc_s012")
latest = None
for d in base.iterdir() if base.exists() else []:
    if not d.is_dir():
        continue
    cfg = d / "config.json"
    ev = d / "events.jsonl"
    if not cfg.exists() or not ev.exists():
        continue
    try:
        run_name = json.loads(cfg.read_text(encoding="utf-8")).get("run_name")
    except Exception:
        continue
    if run_name != "adaptive_fwb-b4-s2-kb66-8b":
        continue
    mt = ev.stat().st_mtime
    if latest is None or mt > latest[0]:
        latest = (mt, ev)

if latest is None:
    raise SystemExit(1)

done = 0
for line in latest[1].read_text(encoding="utf-8").splitlines():
    if '"type": "task_done"' in line:
        done += 1

raise SystemExit(0 if done >= 50 else 1)
PY
}

build_phase67_resume_shards

for i in 0 1 2; do
  shard_spec="data/ablation_phase67_resume_shard${i}.json"
  shard_runs=$(python3 - <<PY
import json
print(len(json.load(open("${shard_spec}", "r", encoding="utf-8")).get("runs", [])))
PY
)
  if [[ "${shard_runs}" == "0" ]]; then
    echo "[skip] phase67 shard${i} has no remaining runs"
    continue
  fi
  start_if_missing \
    "phase67_p_seed${i}" \
    "${shard_spec}" \
    "runs/phase67_launch/phase67_seed${i}.log"
done

if phase66_b4s2_done; then
  echo "[skip] phase66 b4-s2 already reached 50/50 task_done"
else
  start_if_missing \
    "phase66_b4s2_relaunch" \
    "data/ablation_phase66_b4s2_only_relaunch.json" \
    "runs/phase66_launch/phase66_b4s2_relaunch.log"
fi

echo
echo "[status] screens:"
screen -ls | rg "phase67_p_|phase66_b4s2_relaunch|caffeinate_guard" || true

echo
echo "[status] phase67:"
python3 scripts/phase67_status.py || true

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${1:-$(date '+%Y%m%d-%H%M%S')}"
BUNDLE_NAME="publish_bundle_${STAMP}"
OUT_DIR="artifacts/${BUNDLE_NAME}"
export BUNDLE_NAME

mkdir -p "$OUT_DIR/metrics" "$OUT_DIR/repro" "$OUT_DIR/paper" "$OUT_DIR/specs" "$OUT_DIR/logs" "$OUT_DIR/metadata"

# Rebuild all live artifacts before snapshotting.
bash scripts/refresh_transfer_metrics.sh
(
  cd paper
  latexmk -pdf -interaction=nonstopmode main_phase20_live.tex
)

# Core metrics and readiness artifacts.
for f in \
  runs/live_costly_scorecard.json \
  runs/live_costly_scorecard.md \
  runs/publish_readiness.json \
  runs/publish_readiness.md \
  runs/ablation_status.json \
  runs/ablation_status.md \
  runs/live_status.md \
  runs/adaptive_win_snapshot.md \
  runs/hardset_phase33_h12pair_budget1_auto.json \
  runs/hardset_phase33_h12pair_budget1_auto.md \
  runs/hardset_phase33_numeric_budget1_auto.json \
  runs/hardset_phase33_numeric_budget1_auto.md \
  runs/drive_publish_autopilot.log \
  runs/publish_autopilot_console.log \
  runs/progress_update_20260215_0807.md \
  runs/progress_update_20260215_0812.md \
  runs/progress_update_20260215_0814.md; do
  if [[ -f "$f" ]]; then
    cp "$f" "$OUT_DIR/metrics/"
  fi
done

# Copy paper source + built outputs.
cp paper/main_phase20_live.tex "$OUT_DIR/paper/"
cp paper/main_phase20_live.pdf "$OUT_DIR/paper/"
cp paper/references.bib "$OUT_DIR/paper/"
cp -R paper/tables "$OUT_DIR/paper/tables"
cp -R paper/figures "$OUT_DIR/paper/figures"

# Snapshot active specs and hardset task files.
for f in data/ablation_phase*.json data/hardset_phase33_*_auto.jsonl; do
  if [[ -f "$f" ]]; then
    cp "$f" "$OUT_DIR/specs/"
  fi
done

# Reproducibility manifest over all live run groups.
python3 scripts/build_repro_manifest.py \
  --run_group phase28_combined \
  --run_group verify_phase28_h12_budget1_s01234567 \
  --run_group verify_phase28_h12_budget1_ext_s891011 \
  --run_group verify_phase30b_mbpp20_budget1_s0123_nors \
  --run_group verify_phase31_numeric_budget1_s01234567 \
  --run_group verify_phase31_string_budget1_s01234567 \
  --run_group verify_phase31_symbolic_budget1_s01234567 \
  --run_group verify_phase32_synth20_budget1_s0123_nors \
  --run_group verify_phase33_hardset_numeric_budget1_s01234567 \
  --run_group verify_phase33_hardset_h12pair_budget1_s1223 \
  --run_group verify_phase34_h12_budget1_pairwise_s1223 \
  --out "$OUT_DIR/repro/repro_manifest.json"

python3 - <<'PY'
import json
import os
import platform
import subprocess
from pathlib import Path

out = Path("artifacts") / os.environ["BUNDLE_NAME"] / "metadata" / "snapshot_meta.json"
meta = {
    "generated_at_local": subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S %z"], text=True).strip(),
    "generated_at_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
}
out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

cp runs/drive_publish_autopilot.log "$OUT_DIR/logs/" 2>/dev/null || true
cp runs/publish_autopilot_console.log "$OUT_DIR/logs/" 2>/dev/null || true

TARBALL="artifacts/${BUNDLE_NAME}.tar.gz"
tar -czf "$TARBALL" -C artifacts "$BUNDLE_NAME"

echo "bundle_dir=${OUT_DIR}"
echo "bundle_tar=${TARBALL}"

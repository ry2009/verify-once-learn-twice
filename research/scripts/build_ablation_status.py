from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _ts_to_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scan_expected_totals(data_root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not data_root.exists():
        return out
    for path in sorted(data_root.glob("ablation_*.json")):
        obj = _load_json(path)
        if not obj:
            continue
        group = str(obj.get("run_group", "")).strip()
        runs = obj.get("runs", [])
        if not group or not isinstance(runs, list):
            continue
        out[group] = len(runs)
    return out


def _status_for_group(runs_root: Path, run_group: str, expected_totals: dict[str, int]) -> dict[str, Any]:
    group_dir = runs_root / run_group
    manifest_path = group_dir / "ablation_manifest.json"
    manifest = _load_json(manifest_path)
    expected_total = int(expected_totals.get(run_group, 0))
    if not manifest:
        done = 0
        # Fall back to visible run directories when manifest has not yet been
        # created/updated (for live in-flight sweeps).
        if group_dir.exists():
            seen_names: set[str] = set()
            for child in sorted(group_dir.iterdir()):
                if not child.is_dir():
                    continue
                cfg_path = child / "config.json"
                ev_path = child / "events.jsonl"
                if not cfg_path.exists() or not ev_path.exists():
                    continue
                cfg = _load_json(cfg_path)
                run_name = str((cfg or {}).get("run_name", child.name))
                seen_names.add(run_name)
            done = len(seen_names)
        completion = (done / expected_total) if expected_total > 0 else 0.0
        return {
            "run_group": run_group,
            "exists": group_dir.exists(),
            "manifest": False,
            "done": done,
            "total": expected_total,
            "completion": completion,
            "eta_seconds": None,
        }

    done = len(manifest.get("runs", []))
    total = len(manifest.get("spec", {}).get("runs", []))
    if total <= 0 and expected_total > 0:
        total = expected_total
    completion = (done / total) if total > 0 else 0.0

    # Estimate ETA from observed completion pace in this run group.
    run_dirs: list[Path] = []
    for item in manifest.get("runs", []):
        p = str(item.get("run_dir", "")).strip()
        if not p:
            continue
        rp = Path(p)
        if not rp.is_absolute():
            rp = (Path.cwd() / rp).resolve()
        if rp.exists():
            run_dirs.append(rp)

    eta_seconds = None
    pace_runs_per_sec = None
    if len(run_dirs) >= 2 and done < total:
        mtimes = sorted(os.path.getmtime(p) for p in run_dirs)
        elapsed = max(1.0, mtimes[-1] - mtimes[0])
        pace_runs_per_sec = (len(mtimes) - 1) / elapsed
        if pace_runs_per_sec > 0:
            remaining = total - done
            eta_seconds = remaining / pace_runs_per_sec

    last_run_name = None
    if done > 0:
        last = manifest["runs"][-1]
        last_run_name = last.get("run", {}).get("run_name")

    return {
        "run_group": run_group,
        "exists": True,
        "manifest": True,
        "done": done,
        "total": total,
        "completion": completion,
        "pace_runs_per_hour": None if pace_runs_per_sec is None else pace_runs_per_sec * 3600.0,
        "eta_seconds": eta_seconds,
        "last_run_name": last_run_name,
        "updated_at_epoch": time.time(),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Ablation Status")
    lines.append("")
    lines.append("| Run Group | Done/Total | Completion | Pace (runs/hr) | ETA | Last Run |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in rows:
        if not r.get("manifest", False):
            done = int(r.get("done", 0))
            total = int(r.get("total", 0))
            comp = (float(done) / float(total) * 100.0) if total > 0 else 0.0
            lines.append(f"| {r['run_group']} | {done}/{total} | {comp:.1f}% | - | - | - |")
            continue
        done = int(r.get("done", 0))
        total = int(r.get("total", 0))
        comp = float(r.get("completion", 0.0)) * 100.0
        pace = r.get("pace_runs_per_hour")
        eta = r.get("eta_seconds")
        pace_s = "-" if pace is None else f"{pace:.2f}"
        eta_s = "-" if eta is None else _ts_to_hms(float(eta))
        last = r.get("last_run_name") or "-"
        lines.append(f"| {r['run_group']} | {done}/{total} | {comp:.1f}% | {pace_s} | {eta_s} | {last} |")
    lines.append("")
    lines.append(f"_Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--run_group", action="append", required=True)
    parser.add_argument("--out_json", default="runs/ablation_status.json")
    parser.add_argument("--out_md", default="runs/ablation_status.md")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    expected_totals = _scan_expected_totals(Path(args.data_root))
    rows = [_status_for_group(runs_root, g, expected_totals) for g in args.run_group]

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(rows), encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

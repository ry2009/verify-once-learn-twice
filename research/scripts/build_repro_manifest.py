from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "events.jsonl"
    if not cfg_path.exists() or not ev_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    task_done = [ev for ev in _iter_jsonl(ev_path) if ev.get("type") == "task_done"]
    success = sum(1 for ev in task_done if ev.get("success"))
    n = len(task_done)
    return {
        "run_dir": str(run_dir),
        "run_name": str(cfg.get("run_name", run_dir.name)),
        "method": cfg.get("method"),
        "feedback_budget": cfg.get("feedback_budget"),
        "inner_updates": cfg.get("inner_updates"),
        "seed": cfg.get("seed"),
        "tasks": n,
        "success_rate": (success / n) if n else 0.0,
    }


def build_group_manifest(group: str) -> dict:
    group_dir = Path("runs") / group
    if not group_dir.exists():
        return {"run_group": group, "exists": False, "runs": []}

    manifest_path = group_dir / "ablation_manifest.json"
    runs = []
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        for item in m.get("runs", []):
            run_dir = Path(item.get("run_dir", ""))
            if not run_dir.is_absolute():
                run_dir = Path.cwd() / run_dir
            row = _read_run(run_dir)
            if row:
                runs.append(row)
    else:
        for child in sorted(group_dir.iterdir()):
            if child.is_dir():
                row = _read_run(child)
                if row:
                    runs.append(row)

    methods = sorted({str(r.get("method")) for r in runs})
    seeds = sorted({int(r.get("seed")) for r in runs if r.get("seed") is not None})
    budgets = sorted({int(r.get("feedback_budget")) for r in runs if r.get("feedback_budget") is not None})
    return {
        "run_group": group,
        "exists": True,
        "run_count": len(runs),
        "methods": methods,
        "seeds": seeds,
        "feedback_budgets": budgets,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = {
        "generated_at": os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip(),
        "run_groups": [build_group_manifest(g) for g in args.run_group],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

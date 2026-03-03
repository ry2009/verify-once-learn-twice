from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _failed_task_ids(run_dir: Path) -> Set[str]:
    events_path = run_dir / "events.jsonl"
    failed: Set[str] = set()
    passed: Set[str] = set()
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("type") != "task_done":
                continue
            task_id = str(ev.get("task_id", ""))
            if ev.get("success"):
                passed.add(task_id)
            else:
                failed.add(task_id)
    return failed - passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_path", default="data/humaneval_80.jsonl")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    tasks_path = Path(args.tasks_path)
    run_dir = Path(args.run_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(tasks_path)
    failed_ids = _failed_task_ids(run_dir)
    subset = [row for row in rows if str(row.get("task_id", "")) in failed_ids]
    _write_jsonl(out_path, subset)

    manifest = {
        "tasks_path": str(tasks_path),
        "run_dir": str(run_dir),
        "failed_tasks": len(failed_ids),
        "subset_tasks": len(subset),
        "out_path": str(out_path),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"failed task ids: {len(failed_ids)}")
    print(f"subset rows: {len(subset)}")
    print(f"wrote {out_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

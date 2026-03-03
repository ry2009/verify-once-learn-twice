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


def _task_outcomes(run_dir: Path) -> Dict[str, bool]:
    events_path = run_dir / "events.jsonl"
    outcomes: Dict[str, bool] = {}
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("type") != "task_done":
                continue
            task_id = str(ev.get("task_id", ""))
            outcomes[task_id] = bool(ev.get("success"))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_path", default="data/humaneval_80.jsonl")
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--min_fail_count", type=int, default=2)
    parser.add_argument("--require_observed_in_all", action="store_true")
    args = parser.parse_args()

    tasks_path = Path(args.tasks_path)
    run_dirs = [Path(x) for x in args.run_dir]
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(tasks_path)
    outcomes_per_run = [_task_outcomes(run_dir) for run_dir in run_dirs]
    all_task_ids: Set[str] = set()
    for outcomes in outcomes_per_run:
        all_task_ids.update(outcomes.keys())

    selected_ids: Set[str] = set()
    task_stats: Dict[str, dict] = {}
    for task_id in sorted(all_task_ids):
        observed = 0
        fail_count = 0
        pass_count = 0
        for outcomes in outcomes_per_run:
            if task_id not in outcomes:
                continue
            observed += 1
            if outcomes[task_id]:
                pass_count += 1
            else:
                fail_count += 1
        if args.require_observed_in_all and observed != len(outcomes_per_run):
            continue
        if fail_count >= args.min_fail_count:
            selected_ids.add(task_id)
        task_stats[task_id] = {
            "observed": observed,
            "fail_count": fail_count,
            "pass_count": pass_count,
        }

    subset = [row for row in rows if str(row.get("task_id", "")) in selected_ids]
    _write_jsonl(out_path, subset)

    manifest = {
        "tasks_path": str(tasks_path),
        "run_dirs": [str(x) for x in run_dirs],
        "min_fail_count": int(args.min_fail_count),
        "require_observed_in_all": bool(args.require_observed_in_all),
        "subset_tasks": len(subset),
        "selected_task_ids": sorted(selected_ids),
        "task_stats": task_stats,
        "out_path": str(out_path),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"subset rows: {len(subset)}")
    print(f"wrote {out_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

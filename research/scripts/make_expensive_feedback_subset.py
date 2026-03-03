from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class TaskStats:
    task_id: str
    observed: int = 0
    fail_count: int = 0
    pass_count: int = 0
    feedback_char_total: int = 0
    feedback_events: int = 0

    @property
    def avg_feedback_chars(self) -> float:
        if self.feedback_events == 0:
            return 0.0
        return self.feedback_char_total / self.feedback_events


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_tasks(path: Path) -> Dict[str, dict]:
    tasks: Dict[str, dict] = {}
    for row in _iter_jsonl(path):
        task_id = str(row.get("task_id", ""))
        if task_id:
            tasks[task_id] = row
    return tasks


def _read_run(events_path: Path, stats: Dict[str, TaskStats]) -> None:
    # Track whether we already logged feedback length for the current attempt.
    seen_feedback_for_task: Dict[str, bool] = {}
    for ev in _iter_jsonl(events_path):
        typ = ev.get("type")
        task_id = str(ev.get("task_id", ""))
        if not task_id:
            continue
        s = stats.setdefault(task_id, TaskStats(task_id=task_id))

        if typ == "task_start":
            s.observed += 1
            seen_feedback_for_task[task_id] = False
            continue

        if typ == "test" and not ev.get("passed", False):
            if seen_feedback_for_task.get(task_id, False):
                continue
            feedback = str(ev.get("feedback", ""))
            s.feedback_char_total += len(feedback)
            s.feedback_events += 1
            seen_feedback_for_task[task_id] = True
            continue

        if typ == "task_done":
            if ev.get("success", False):
                s.pass_count += 1
            else:
                s.fail_count += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_jsonl", required=True)
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--min_fail_count", type=int, default=2)
    parser.add_argument("--min_feedback_chars", type=float, default=700.0)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--manifest_out", required=True)
    args = parser.parse_args()

    tasks_path = Path(args.tasks_jsonl)
    tasks = _load_tasks(tasks_path)
    if not tasks:
        raise SystemExit(f"No tasks loaded from {tasks_path}")

    stats: Dict[str, TaskStats] = {}
    for run_dir in args.run_dir:
        events_path = Path(run_dir) / "events.jsonl"
        if not events_path.exists():
            raise SystemExit(f"Missing events.jsonl: {events_path}")
        _read_run(events_path, stats)

    candidates: List[TaskStats] = []
    for task_id, s in stats.items():
        if task_id not in tasks:
            continue
        if s.fail_count < args.min_fail_count:
            continue
        if s.avg_feedback_chars < args.min_feedback_chars:
            continue
        candidates.append(s)

    candidates.sort(key=lambda s: (s.fail_count, s.avg_feedback_chars), reverse=True)
    if args.top_k > 0:
        candidates = candidates[: args.top_k]

    selected_ids = [s.task_id for s in candidates]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for task_id in selected_ids:
            f.write(json.dumps(tasks[task_id], ensure_ascii=True) + "\n")

    manifest = {
        "tasks_jsonl": str(tasks_path),
        "run_dirs": args.run_dir,
        "min_fail_count": args.min_fail_count,
        "min_feedback_chars": args.min_feedback_chars,
        "top_k": args.top_k,
        "selected": [
            {
                "task_id": s.task_id,
                "observed": s.observed,
                "fail_count": s.fail_count,
                "pass_count": s.pass_count,
                "avg_feedback_chars": s.avg_feedback_chars,
                "feedback_events": s.feedback_events,
            }
            for s in candidates
        ],
    }

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(selected_ids)} tasks to {out_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

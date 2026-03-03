import argparse
import json
import os
from collections import defaultdict


def load_events(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="runs/latest")
    args = parser.parse_args()

    events_path = os.path.join(args.run_dir, "events.jsonl")
    if not os.path.exists(events_path):
        raise SystemExit(f"No events.jsonl found at {events_path}")

    task_done = {}
    for ev in load_events(events_path):
        if ev.get("type") == "task_done":
            task_done[ev["task_id"]] = ev

    if not task_done:
        print("No task_done events found")
        return

    total = len(task_done)
    success = sum(1 for v in task_done.values() if v.get("success"))
    avg_feedback = sum(v.get("feedback_calls", 0) for v in task_done.values()) / total

    by_budget = defaultdict(lambda: {"total": 0, "success": 0})
    for v in task_done.values():
        fb = int(v.get("feedback_calls", 0))
        by_budget[fb]["total"] += 1
        if v.get("success"):
            by_budget[fb]["success"] += 1

    print(f"Tasks: {total}")
    print(f"Success: {success} ({success/total:.3f})")
    print(f"Avg feedback calls: {avg_feedback:.2f}")
    print("Success by feedback calls:")
    for fb in sorted(by_budget.keys()):
        bucket = by_budget[fb]
        rate = bucket["success"] / bucket["total"] if bucket["total"] else 0.0
        print(f"  {fb}: {bucket['success']}/{bucket['total']} ({rate:.3f})")


if __name__ == "__main__":
    main()

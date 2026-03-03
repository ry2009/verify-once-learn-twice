import argparse
import json
import os


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
    parser.add_argument("--max_budget", type=int, default=8)
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

    print("budget,success_rate")
    for b in range(1, args.max_budget + 1):
        successes = 0
        total = 0
        for v in task_done.values():
            total += 1
            if v.get("success") and int(v.get("feedback_calls", 0)) <= b:
                successes += 1
        rate = successes / total if total else 0.0
        print(f"{b},{rate:.4f}")


if __name__ == "__main__":
    main()

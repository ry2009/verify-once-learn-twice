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


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: str):
    events_path = os.path.join(run_dir, "events.jsonl")
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(events_path) or not os.path.exists(config_path):
        return None
    cfg = load_config(config_path)
    task_done = []
    for ev in load_events(events_path):
        if ev.get("type") == "task_done":
            task_done.append(ev)
    if not task_done:
        return None

    total = len(task_done)
    success = sum(1 for v in task_done if v.get("success"))
    avg_feedback = sum(v.get("feedback_calls", 0) for v in task_done) / total
    avg_train_steps = sum(v.get("train_steps", 0) for v in task_done) / total
    avg_judge_calls = sum(v.get("judge_calls", 0) for v in task_done) / total
    avg_test_calls = sum(v.get("test_calls", 0) for v in task_done) / total

    return {
        "run_dir": run_dir,
        "method": cfg.get("method"),
        "feedback_budget": cfg.get("feedback_budget"),
        "inner_updates": cfg.get("inner_updates"),
        "judge_mode": cfg.get("judge_mode"),
        "judge_flip_prob": cfg.get("judge_flip_prob", 0.0),
        "seed": cfg.get("seed"),
        "success_rate": success / total,
        "avg_feedback_calls": avg_feedback,
        "avg_train_steps": avg_train_steps,
        "avg_judge_calls": avg_judge_calls,
        "avg_test_calls": avg_test_calls,
        "tasks": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", default="ablation")
    args = parser.parse_args()

    root = os.path.join("runs", args.run_group)
    if not os.path.isdir(root):
        raise SystemExit(f"Run group not found: {root}")

    summaries = []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        summary = summarize_run(run_dir)
        if summary:
            summaries.append(summary)

    if not summaries:
        print("No runs found")
        return

    by_key = defaultdict(list)
    for s in summaries:
        key = (
            s["method"],
            s["feedback_budget"],
            s["inner_updates"],
            s["judge_mode"],
            s["judge_flip_prob"],
        )
        by_key[key].append(s)

    print("method,feedback_budget,inner_updates,judge_mode,judge_flip_prob,runs,success_rate,avg_feedback,avg_train_steps,avg_judge_calls,avg_test_calls")
    for key, rows in sorted(by_key.items()):
        method, fb, inner, judge_mode, judge_flip_prob = key
        runs = len(rows)
        success_rate = sum(r["success_rate"] for r in rows) / runs
        avg_feedback = sum(r["avg_feedback_calls"] for r in rows) / runs
        avg_train_steps = sum(r["avg_train_steps"] for r in rows) / runs
        avg_judge_calls = sum(r["avg_judge_calls"] for r in rows) / runs
        avg_test_calls = sum(r["avg_test_calls"] for r in rows) / runs
        print(
            f"{method},{fb},{inner},{judge_mode},{judge_flip_prob},{runs},{success_rate:.4f},{avg_feedback:.2f},{avg_train_steps:.2f},{avg_judge_calls:.2f},{avg_test_calls:.2f}"
        )


if __name__ == "__main__":
    main()

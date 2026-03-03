from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _audit_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "events.jsonl"
    if not cfg_path.exists() or not ev_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    done_events = [ev for ev in _iter_jsonl(ev_path) if ev.get("type") == "task_done"]
    if not done_events:
        return None

    by_task: Dict[str, List[dict]] = {}
    for ev in _iter_jsonl(ev_path):
        task_id = str(ev.get("task_id", ""))
        if task_id:
            by_task.setdefault(task_id, []).append(ev)

    stale_unsuccessful = 0
    stale_successful = 0
    total = 0

    for done in done_events:
        task_id = str(done.get("task_id", ""))
        events = by_task.get(task_id, [])
        if not events:
            continue
        total += 1
        prev = None
        for ev in reversed(events):
            if ev.get("type") == "task_done":
                continue
            prev = ev
            break
        if prev is None:
            continue
        if prev.get("type") == "train":
            if done.get("success"):
                stale_successful += 1
            else:
                stale_unsuccessful += 1

    return {
        "run_dir": str(run_dir),
        "method": cfg.get("method", "unknown"),
        "inner_updates": int(cfg.get("inner_updates", 1)),
        "feedback_budget": int(cfg.get("feedback_budget", -1)),
        "seed": int(cfg.get("seed", -1)),
        "tasks": total,
        "stale_terminal_unsuccessful": stale_unsuccessful,
        "stale_terminal_successful": stale_successful,
        "stale_terminal_unsuccessful_rate": (
            stale_unsuccessful / total if total else 0.0
        ),
        "stale_terminal_any_rate": (
            (stale_unsuccessful + stale_successful) / total if total else 0.0
        ),
    }


def _collect(run_groups: List[str], min_tasks: int) -> List[dict]:
    rows: List[dict] = []
    for group in run_groups:
        root = Path("runs") / group
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            row = _audit_run(child)
            if row:
                if int(row.get("tasks", 0)) < min_tasks:
                    continue
                row["run_group"] = group
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", action="append", required=True)
    parser.add_argument("--min_tasks", type=int, default=1)
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    rows = _collect(args.run_group, min_tasks=args.min_tasks)
    if not rows:
        raise SystemExit("No runs found.")

    by_method: Dict[str, List[dict]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    print(
        "method,runs,tasks,stale_terminal_unsuccessful_rate,stale_terminal_any_rate"
    )
    for method, method_rows in sorted(by_method.items()):
        tasks = sum(int(r["tasks"]) for r in method_rows)
        stale_unsuccessful = sum(int(r["stale_terminal_unsuccessful"]) for r in method_rows)
        stale_any = sum(
            int(r["stale_terminal_unsuccessful"]) + int(r["stale_terminal_successful"])
            for r in method_rows
        )
        stale_unsuccessful_rate = stale_unsuccessful / tasks if tasks else 0.0
        stale_any_rate = stale_any / tasks if tasks else 0.0
        print(
            f"{method},{len(method_rows)},{tasks},{stale_unsuccessful_rate:.4f},{stale_any_rate:.4f}"
        )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"run_groups": args.run_group, "runs": rows}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

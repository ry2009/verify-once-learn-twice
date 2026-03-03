from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SPECS = [
    "data/ablation_phase50_kernelbench_kpass_8b_s0.json",
    "data/ablation_phase50_kernelbench_kpass_3b_s0.json",
]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_jsonl(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


@dataclass
class RunView:
    run_name: str
    method: str
    complete: bool
    task_done: int
    target: int
    state: str


def _run_target(cfg: dict) -> int:
    max_steps = int(cfg.get("max_steps", 0))
    tasks_path = Path(str(cfg.get("tasks_path", "")))
    tasks_total = _count_jsonl(tasks_path) if str(tasks_path) else 0
    target = max_steps if max_steps > 0 else tasks_total
    if target > 0 and tasks_total > 0:
        target = min(target, tasks_total)
    return target


def _status_for_run_dir(run_dir: Path) -> tuple[bool, int, int]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    target = _run_target(cfg)
    done = 0
    complete = False
    ev_path = run_dir / "events.jsonl"
    if ev_path.exists():
        for ev in _iter_jsonl(ev_path):
            if ev.get("type") == "task_done":
                done += 1
            if ev.get("type") == "run_complete":
                complete = True
    if target > 0 and done >= target:
        complete = True
    return complete, done, target


def _latest_run_dir_for_name(group_dir: Path, run_name: str) -> Path | None:
    cands: list[tuple[float, Path]] = []
    for run_dir in group_dir.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(cfg.get("run_name", "")) != run_name:
            continue
        ev_path = run_dir / "events.jsonl"
        stamp = ev_path.stat().st_mtime if ev_path.exists() else run_dir.stat().st_mtime
        cands.append((stamp, run_dir))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def main() -> None:
    all_complete = True
    for spec_path in SPECS:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        group = str(spec.get("run_group", ""))
        runs = spec.get("runs", [])
        group_dir = Path("runs") / group

        views: list[RunView] = []
        for run_cfg in runs:
            run_name = str(run_cfg.get("run_name", ""))
            method = str(run_cfg.get("method", ""))
            if not group_dir.exists():
                views.append(RunView(run_name, method, False, 0, int(spec.get("max_steps", 0)), "missing"))
                all_complete = False
                continue
            latest = _latest_run_dir_for_name(group_dir, run_name)
            if latest is None:
                views.append(RunView(run_name, method, False, 0, int(spec.get("max_steps", 0)), "pending"))
                all_complete = False
                continue
            complete, done, target = _status_for_run_dir(latest)
            state = "done" if complete else "live"
            if not complete:
                all_complete = False
            views.append(RunView(run_name, method, complete, done, target, state))

        done_count = sum(1 for v in views if v.complete)
        print(f"{group}: {done_count}/{len(views)} runs complete")
        for v in views:
            print(f"  - [{v.state}] method={v.method:<16} task_done={v.task_done:>3}/{v.target:>3} run={v.run_name}")
        print()

    print(f"all_complete={str(all_complete).lower()}")


if __name__ == "__main__":
    main()

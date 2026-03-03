from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List


@dataclass
class RunStatus:
    run_group: str
    run_dir_name: str
    logical_name: str
    method: str
    seed: int
    task_done: int
    max_steps: int
    complete: bool


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@lru_cache(maxsize=128)
def _tasks_total(tasks_path: str) -> int:
    if not tasks_path:
        return 0
    if not os.path.exists(tasks_path):
        return 0
    if tasks_path.endswith(".jsonl"):
        total = 0
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
        return total
    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return 0
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        tasks = obj.get("tasks")
        if isinstance(tasks, list):
            return len(tasks)
    return 0


def _run_status(run_dir: str, run_group: str) -> RunStatus | None:
    cfg_path = os.path.join(run_dir, "config.json")
    ev_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    task_done = 0
    complete = False
    if os.path.exists(ev_path):
        for ev in _iter_jsonl(ev_path):
            if ev.get("type") == "task_done":
                task_done += 1
            elif ev.get("type") == "run_complete":
                complete = True
    max_steps = int(cfg.get("max_steps", -1))
    tasks_path = str(cfg.get("tasks_path", "")).strip()
    tasks_total = _tasks_total(tasks_path)
    target = max_steps if max_steps > 0 else tasks_total
    if tasks_total > 0 and max_steps > 0:
        target = min(max_steps, tasks_total)
    if target > 0 and task_done >= target:
        complete = True
    return RunStatus(
        run_group=run_group,
        run_dir_name=os.path.basename(run_dir),
        logical_name=str(cfg.get("run_name", os.path.basename(run_dir))),
        method=str(cfg.get("method", "unknown")),
        seed=int(cfg.get("seed", -1)),
        task_done=task_done,
        max_steps=max_steps,
        complete=complete,
    )


def _latest_statuses(run_group: str) -> List[RunStatus]:
    root = os.path.join("runs", run_group)
    if not os.path.isdir(root):
        return []
    latest_by_name: dict[str, tuple[float, RunStatus]] = {}
    for run_dir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(run_dir):
            continue
        row = _run_status(run_dir, run_group)
        if row is None:
            continue
        key = row.logical_name
        ev_path = os.path.join(run_dir, "events.jsonl")
        stamp = os.path.getmtime(ev_path) if os.path.exists(ev_path) else os.path.getmtime(run_dir)
        prev = latest_by_name.get(key)
        if prev is None or stamp >= prev[0]:
            latest_by_name[key] = (stamp, row)
    out = [v[1] for v in latest_by_name.values()]
    out.sort(key=lambda r: (r.seed, r.method, r.logical_name))
    return out


def _print_group(run_group: str) -> None:
    rows = _latest_statuses(run_group)
    if not rows:
        print(f"{run_group}: no started runs")
        return
    done = sum(1 for r in rows if r.complete)
    print(f"{run_group}: {done}/{len(rows)} runs complete")
    for r in rows:
        flag = "done" if r.complete else "live"
        print(
            f"  - [{flag}] seed={r.seed:>2} method={r.method:<14} "
            f"task_done={r.task_done:>3} run={r.logical_name}"
        )


def main() -> None:
    groups = [
        "verify_phase41_h80_70b_retune_b1_s2",
        "verify_phase41_h80_70b_retune_b1_s3",
        "verify_phase41_h80_70b_retune_b1_s4",
        "verify_phase41_h80_70b_retune_b1_s5",
        "verify_phase41_h80_70b_retune_b1_s6",
        "verify_phase41_h80_70b_retune_b1_s7",
        "verify_phase41_h80_70b_retune_b1_s8",
        "verify_phase41_h80_70b_retune_b1_s9",
        "verify_phase42_h30_b1_judgeablation_llm_s0123456789_s0to4",
        "verify_phase42_h30_b1_judgeablation_llm_s0123456789_s5to9",
        "verify_phase42_h30_b1_judgeablation_oracle_s0123456789_s0to4",
        "verify_phase42_h30_b1_judgeablation_oracle_s0123456789_s5to9",
        "verify_phase40_mbppplus_compute_match_b1_s3",
        "verify_phase40_mbppplus_compute_match_b1_s4",
        "verify_phase40_mbppplus_compute_match_b1_s5",
        "verify_phase40_mbppplus_compute_match_b1_s6",
        "verify_phase43_grpo_compat_h80_b1_s0",
        "verify_phase43_grpo_compat_h80_b1_s1",
        "verify_phase43_grpo_compat_h80_b1_s2",
        "verify_phase43_grpo_compat_h80_b1_s3",
        "verify_phase43_grpo_compat_h80_b1_s4",
        "verify_phase43_grpo_compat_h80_b1_s5",
        "verify_phase44_symbolic_depth_b1_s0",
        "verify_phase44_symbolic_depth_b1_s1",
        "verify_phase44_symbolic_depth_b1_s2",
        "verify_phase44_symbolic_depth_b1_s3",
        "verify_phase44_symbolic_depth_b1_s4",
        "verify_phase44_symbolic_depth_b1_s5",
        "verify_phase44_symbolic_depth_b1_s6",
        "verify_phase44_symbolic_depth_b1_s7",
        "verify_phase46_h30_b1_noise_n00_s0to3",
        "verify_phase46_h30_b1_noise_n00_s4to7",
        "verify_phase46_h30_b1_noise_n10_s0to3",
        "verify_phase46_h30_b1_noise_n10_s4to7",
        "verify_phase46_h30_b1_noise_n20_s0to3",
        "verify_phase46_h30_b1_noise_n20_s4to7",
        "verify_phase47_mbppplus_compute_match_split_s3_fixed_k_judge_k1",
        "verify_phase47_mbppplus_compute_match_split_s3_fixed_k_judge_k2",
        "verify_phase47_mbppplus_compute_match_split_s3_inference_only",
        "verify_phase47_mbppplus_compute_match_split_s4_fixed_k_judge_k1",
        "verify_phase47_mbppplus_compute_match_split_s4_fixed_k_judge_k2",
        "verify_phase47_mbppplus_compute_match_split_s4_inference_only",
        "verify_phase47_mbppplus_compute_match_split_s5_fixed_k_judge_k1",
        "verify_phase47_mbppplus_compute_match_split_s5_fixed_k_judge_k2",
        "verify_phase47_mbppplus_compute_match_split_s5_inference_only",
        "verify_phase47_mbppplus_compute_match_split_s6_fixed_k_judge_k1",
        "verify_phase47_mbppplus_compute_match_split_s6_fixed_k_judge_k2",
        "verify_phase47_mbppplus_compute_match_split_s6_inference_only",
    ]
    for g in groups:
        _print_group(g)
        print()


if __name__ == "__main__":
    main()

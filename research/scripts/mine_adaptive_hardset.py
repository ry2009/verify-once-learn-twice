from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _variant(method: str, inner_updates: int) -> str:
    if method == "fixed_k_judge":
        return f"fixed_k_judge_k{inner_updates}"
    if method == "fixed_k_fwb":
        return f"fixed_k_fwb_k{inner_updates}"
    return method


def _read_task_done_map(events_path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for ev in _iter_jsonl(events_path):
        if ev.get("type") != "task_done":
            continue
        task_id = str(ev.get("task_id", "")).strip()
        if not task_id:
            continue
        out[task_id] = bool(ev.get("success", False))
    return out


def _collect_latest_runs(run_group: str) -> list[tuple[float, Path, dict[str, Any]]]:
    group_dir = Path("runs") / run_group
    manifest_path = group_dir / "ablation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = _load_json(manifest_path)
    latest: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    for item in manifest.get("runs", []):
        run = dict(item.get("run", {}))
        run_name = str(run.get("run_name", "")).strip()
        run_dir = str(item.get("run_dir", "")).strip()
        if not run_name or not run_dir:
            continue
        path = Path(run_dir)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            continue
        stamp = os.path.getmtime(path)
        prev = latest.get(run_name)
        if prev is None or stamp >= prev[0]:
            latest[run_name] = (stamp, path, run)
    return list(latest.values())


def _select_runs(
    run_group: str,
    method_a: str,
    method_b: str,
    inner_updates_b: int,
    feedback_budget: int,
    min_tasks_per_run: int,
) -> tuple[dict[int, tuple[Path, dict[str, bool]]], dict[int, tuple[Path, dict[str, bool]]], set[int]]:
    latest = _collect_latest_runs(run_group)
    a_runs: dict[int, tuple[Path, dict[str, bool]]] = {}
    b_runs: dict[int, tuple[Path, dict[str, bool]]] = {}
    task_count_seen: dict[Path, int] = {}

    for _, run_dir, run in latest:
        cfg_path = run_dir / "config.json"
        ev_path = run_dir / "events.jsonl"
        if not cfg_path.exists() or not ev_path.exists():
            continue
        cfg = _load_json(cfg_path)
        method = str(cfg.get("method", ""))
        inner = int(cfg.get("inner_updates", 1))
        budget = int(cfg.get("feedback_budget", -1))
        seed = int(cfg.get("seed", -1))
        if budget != feedback_budget or seed < 0:
            continue

        task_map = _read_task_done_map(ev_path)
        task_count_seen[run_dir] = len(task_map)
        if len(task_map) < min_tasks_per_run:
            continue

        v = _variant(method, inner)
        if v == method_a:
            a_runs[seed] = (run_dir, task_map)
        if v == _variant(method_b, inner_updates_b):
            b_runs[seed] = (run_dir, task_map)

    shared = set(a_runs.keys()) & set(b_runs.keys())
    return a_runs, b_runs, shared


def _mine_tasks(
    a_runs: dict[int, tuple[Path, dict[str, bool]]],
    b_runs: dict[int, tuple[Path, dict[str, bool]]],
    shared_seeds: set[int],
    min_shared_seeds: int,
    min_advantage_votes: int,
) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, int]] = {}
    for seed in sorted(shared_seeds):
        a_map = a_runs[seed][1]
        b_map = b_runs[seed][1]
        shared_tasks = set(a_map.keys()) & set(b_map.keys())
        for task_id in shared_tasks:
            a_ok = a_map[task_id]
            b_ok = b_map[task_id]
            row = by_task.setdefault(
                task_id,
                {
                    "support": 0,
                    "a_win": 0,
                    "b_win": 0,
                    "both_pass": 0,
                    "both_fail": 0,
                },
            )
            row["support"] += 1
            if a_ok and not b_ok:
                row["a_win"] += 1
            elif b_ok and not a_ok:
                row["b_win"] += 1
            elif a_ok and b_ok:
                row["both_pass"] += 1
            else:
                row["both_fail"] += 1

    mined: list[dict[str, Any]] = []
    for task_id, row in by_task.items():
        support = int(row["support"])
        if support < min_shared_seeds:
            continue
        a_win = int(row["a_win"])
        b_win = int(row["b_win"])
        if a_win < min_advantage_votes:
            continue
        score = a_win - b_win
        mined.append(
            {
                "task_id": task_id,
                "support": support,
                "a_win": a_win,
                "b_win": b_win,
                "both_pass": int(row["both_pass"]),
                "both_fail": int(row["both_fail"]),
                "score": score,
                "a_win_rate": a_win / support,
                "b_win_rate": b_win / support,
            }
        )

    mined.sort(
        key=lambda x: (
            int(x["score"]),
            int(x["a_win"]),
            -int(x["b_win"]),
            int(x["support"]),
            x["task_id"],
        ),
        reverse=True,
    )
    return mined


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    return [row for row in _iter_jsonl(path)]


def _resolve_tasks_path(manifest_spec: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    task_path = str(manifest_spec.get("tasks_path", "")).strip()
    if not task_path:
        raise ValueError("tasks_path not found in manifest spec and --tasks_path not provided")
    p = Path(task_path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Adaptive Hardset Mining Report")
    lines.append("")
    lines.append(f"- run_group: `{payload['run_group']}`")
    lines.append(f"- method_a: `{payload['method_a']}`")
    lines.append(f"- method_b: `{payload['method_b']}`")
    lines.append(f"- feedback_budget: `{payload['feedback_budget']}`")
    lines.append(f"- shared_seeds: `{payload['shared_seeds']}`")
    lines.append(f"- selected_tasks: `{payload['selected_count']}`")
    lines.append(f"- tasks_output: `{payload['out_tasks']}`")
    lines.append("")
    lines.append("| task_id | support | a_win | b_win | both_pass | both_fail | score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["selected"]:
        lines.append(
            f"| {row['task_id']} | {row['support']} | {row['a_win']} | {row['b_win']} | {row['both_pass']} | {row['both_fail']} | {row['score']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_spec_from_template(
    template_spec_path: Path,
    out_spec_path: Path,
    tasks_out_path: Path,
    run_group: str,
    name: str,
) -> None:
    spec = _load_json(template_spec_path)
    spec["name"] = name
    spec["run_group"] = run_group
    spec["ablation_tag"] = name.replace("-", "_")
    rel_tasks = os.path.relpath(str(tasks_out_path), str(Path.cwd()))
    spec["tasks_path"] = rel_tasks
    out_spec_path.parent.mkdir(parents=True, exist_ok=True)
    out_spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", required=True)
    parser.add_argument("--method_a", default="adaptive_fwb")
    parser.add_argument("--method_b", default="fixed_k_judge")
    parser.add_argument("--inner_updates_b", type=int, default=1)
    parser.add_argument("--feedback_budget", type=int, default=1)
    parser.add_argument("--min_tasks_per_run", type=int, default=10)
    parser.add_argument("--min_shared_seeds", type=int, default=3)
    parser.add_argument("--min_advantage_votes", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_selected", type=int, default=8)
    parser.add_argument("--tasks_path", default="")
    parser.add_argument(
        "--out_tasks",
        default="data/hardset_phase31_numeric_budget1_auto.jsonl",
    )
    parser.add_argument(
        "--out_json",
        default="runs/hardset_phase31_numeric_budget1_auto.json",
    )
    parser.add_argument(
        "--out_md",
        default="runs/hardset_phase31_numeric_budget1_auto.md",
    )
    parser.add_argument("--template_spec", default="")
    parser.add_argument("--out_spec", default="")
    parser.add_argument("--out_run_group", default="")
    parser.add_argument("--out_name", default="")
    args = parser.parse_args()

    group_dir = Path("runs") / args.run_group
    manifest_path = group_dir / "ablation_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest missing: {manifest_path}")
    manifest = _load_json(manifest_path)

    a_runs, b_runs, shared_seeds = _select_runs(
        run_group=args.run_group,
        method_a=args.method_a,
        method_b=args.method_b,
        inner_updates_b=args.inner_updates_b,
        feedback_budget=args.feedback_budget,
        min_tasks_per_run=args.min_tasks_per_run,
    )
    if not shared_seeds:
        raise SystemExit("no shared seeds between selected method pair yet")

    mined = _mine_tasks(
        a_runs=a_runs,
        b_runs=b_runs,
        shared_seeds=shared_seeds,
        min_shared_seeds=args.min_shared_seeds,
        min_advantage_votes=args.min_advantage_votes,
    )
    selected = mined[: max(0, int(args.top_k))]
    if not selected:
        raise SystemExit("no tasks matched selection thresholds")
    if len(selected) < int(args.min_selected):
        raise SystemExit(
            f"selected tasks below min_selected ({len(selected)} < {int(args.min_selected)}); "
            "wait for more completed seeds or relax thresholds"
        )

    source_tasks_path = _resolve_tasks_path(manifest.get("spec", {}), args.tasks_path or None)
    source_rows = _load_tasks(source_tasks_path)
    wanted = {str(row["task_id"]) for row in selected}
    subset = [row for row in source_rows if str(row.get("task_id", "")) in wanted]
    if len(subset) != len(wanted):
        missing = sorted(wanted - {str(row.get("task_id", "")) for row in subset})
        raise SystemExit(f"missing task ids in source tasks file: {missing[:5]}")

    out_tasks_path = Path(args.out_tasks)
    _write_jsonl(out_tasks_path, subset)

    payload = {
        "run_group": args.run_group,
        "method_a": args.method_a,
        "method_b": _variant(args.method_b, args.inner_updates_b),
        "feedback_budget": args.feedback_budget,
        "min_tasks_per_run": args.min_tasks_per_run,
        "min_shared_seeds": args.min_shared_seeds,
        "min_advantage_votes": args.min_advantage_votes,
        "shared_seeds": sorted(shared_seeds),
        "selected_count": len(selected),
        "selected_task_ids": [row["task_id"] for row in selected],
        "selected": selected,
        "out_tasks": str(out_tasks_path),
        "source_tasks_path": str(source_tasks_path),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_md(Path(args.out_md), payload)

    if args.template_spec and args.out_spec:
        out_group = args.out_run_group or "verify_phase33_hardset_numeric_budget1_s01234567"
        out_name = args.out_name or "phase33-hardset-numeric-budget1-s01234567"
        _build_spec_from_template(
            template_spec_path=Path(args.template_spec),
            out_spec_path=Path(args.out_spec),
            tasks_out_path=out_tasks_path,
            run_group=out_group,
            name=out_name,
        )

    print(f"wrote {out_tasks_path}")
    print(f"wrote {out_json}")
    print(f"wrote {args.out_md}")
    if args.template_spec and args.out_spec:
        print(f"wrote {args.out_spec}")


if __name__ == "__main__":
    main()

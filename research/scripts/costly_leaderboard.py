from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class RunRow:
    run_group: str
    run_dir: str
    run_name: str
    method: str
    method_variant: str
    feedback_budget: int
    judge_flip_prob: float
    seed: int
    tasks: int
    success_rate: float
    avg_feedback_calls: float
    avg_train_steps: float
    avg_test_calls: float


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    mu = _mean(vals)
    return (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5


def _method_variant(method: str, inner_updates: int) -> str:
    if method == "fixed_k_fwb":
        return f"fixed_k_fwb_k{inner_updates}"
    if method == "fixed_k_judge":
        return f"fixed_k_judge_k{inner_updates}"
    return method


def _label(method_variant: str) -> str:
    m = re.match(r"^fixed_k_fwb_k(\d+)$", method_variant)
    if m:
        return f"fixed_k_fwb(k={m.group(1)})"
    m = re.match(r"^fixed_k_judge_k(\d+)$", method_variant)
    if m:
        return f"fixed_k_judge(k={m.group(1)})"
    return method_variant


def _read_run(run_group: str, run_dir: Path) -> RunRow | None:
    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "events.jsonl"
    if not cfg_path.exists() or not ev_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    done = [ev for ev in _iter_jsonl(ev_path) if ev.get("type") == "task_done"]
    if not done:
        return None

    success = [1.0 if ev.get("success") else 0.0 for ev in done]
    fb = [float(ev.get("feedback_calls", 0)) for ev in done]
    tr = [float(ev.get("train_steps", 0)) for ev in done]
    tc = [float(ev.get("test_calls", 0)) for ev in done]

    method = str(cfg.get("method", "unknown"))
    inner = int(cfg.get("inner_updates", 1))
    return RunRow(
        run_group=run_group,
        run_dir=str(run_dir),
        run_name=str(cfg.get("run_name", run_dir.name)),
        method=method,
        method_variant=_method_variant(method, inner),
        feedback_budget=int(cfg.get("feedback_budget", -1)),
        judge_flip_prob=float(cfg.get("judge_flip_prob", 0.0)),
        seed=int(cfg.get("seed", -1)),
        tasks=len(done),
        success_rate=_mean(success),
        avg_feedback_calls=_mean(fb),
        avg_train_steps=_mean(tr),
        avg_test_calls=_mean(tc),
    )


def _run_dirs_for_group(root: Path) -> List[Path]:
    # Merge manifest-listed run dirs with on-disk dirs. Some long-running
    # groups accumulate stale/incomplete manifests after restarts; relying only
    # on manifest entries can silently drop valid completed runs.
    out: List[Path] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            return
        seen.add(key)
        out.append(path)

    manifest = root / "ablation_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for item in data.get("runs", []):
                rel = str(item.get("run_dir", "")).strip()
                if not rel:
                    continue
                path = Path(rel)
                if not path.is_absolute():
                    path = Path.cwd() / rel
                _push(path)
        except Exception:
            pass
    for path in sorted(root.iterdir()):
        if path.is_dir():
            _push(path)
    return out


def _collect(run_groups: List[str], min_tasks: int) -> List[RunRow]:
    # Keep one row per logical run_name (latest mtime wins), which prevents
    # duplicate counting from interrupted/restarted launches.
    dedup: Dict[Tuple[str, str], Tuple[int, float, RunRow]] = {}
    for group in run_groups:
        root = Path("runs") / group
        if not root.exists():
            continue
        for child in _run_dirs_for_group(root):
            row = _read_run(group, child)
            if row and row.tasks >= min_tasks:
                stamp = float(os.path.getmtime(child)) if child.exists() else 0.0
                key = (group, row.run_name)
                prev = dedup.get(key)
                if prev is None or (row.tasks, stamp) >= (prev[0], prev[1]):
                    dedup[key] = (row.tasks, stamp, row)
    return [row for _, _, row in dedup.values()]


def _aggregate(rows: List[RunRow], lambdas: List[float]) -> List[dict]:
    buckets: Dict[Tuple[str, int, float], List[RunRow]] = {}
    for r in rows:
        buckets.setdefault(
            (r.method_variant, r.feedback_budget, r.judge_flip_prob), []
        ).append(r)

    out: List[dict] = []
    for (method_variant, budget, judge_flip_prob), vals in sorted(
        buckets.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
    ):
        success = [r.success_rate for r in vals]
        fb = [r.avg_feedback_calls for r in vals]
        tr = [r.avg_train_steps for r in vals]
        tc = [r.avg_test_calls for r in vals]
        row = {
            "method_variant": method_variant,
            "method_label": _label(method_variant),
            "feedback_budget": budget,
            "judge_flip_prob": judge_flip_prob,
            "runs": len(vals),
            "success_mean": _mean(success),
            "success_sd": _std(success),
            "feedback_mean": _mean(fb),
            "feedback_sd": _std(fb),
            "train_mean": _mean(tr),
            "test_mean": _mean(tc),
        }
        fb_mean = float(row["feedback_mean"])
        row["success_per_feedback_mean"] = (
            float(row["success_mean"]) / fb_mean if fb_mean > 1e-12 else 0.0
        )
        for lam in lambdas:
            key = f"utility_lambda_{str(lam).replace('.', '_')}"
            row[key] = row["success_mean"] - lam * row["feedback_mean"]
        out.append(row)
    return out


def _frontier_auc(agg: List[dict]) -> Dict[str, float]:
    by_method: Dict[Tuple[str, float], List[Tuple[float, float]]] = {}
    for row in agg:
        key = (str(row["method_variant"]), float(row.get("judge_flip_prob", 0.0)))
        by_method.setdefault(key, []).append(
            (float(row["feedback_budget"]), float(row["success_mean"]))
        )
    auc: Dict[str, float] = {}
    for (method, noise), pts in by_method.items():
        pts.sort()
        area = 0.0
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            area += 0.5 * (y0 + y1) * (x1 - x0)
        key = method if abs(noise) < 1e-12 else f"{method}_noise{noise:.2f}"
        auc[key] = area
    return auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", action="append", required=True)
    parser.add_argument("--min_tasks", type=int, default=1)
    parser.add_argument("--lambda_feedback", action="append", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    rows = _collect(args.run_group, min_tasks=args.min_tasks)
    if not rows:
        raise SystemExit("No matching runs found.")

    agg = _aggregate(rows, lambdas=args.lambda_feedback)
    auc = _frontier_auc(agg)
    summary = {
        "run_groups": args.run_group,
        "min_tasks": args.min_tasks,
        "lambda_feedback": args.lambda_feedback,
        "runs": [r.__dict__ for r in rows],
        "aggregate": agg,
        "frontier_auc": auc,
    }

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "method,feedback_budget,judge_flip_prob,runs,success_mean,success_sd,feedback_mean,train_mean,test_mean"
    )
    for row in agg:
        print(
            f"{row['method_variant']},{row['feedback_budget']},{row['judge_flip_prob']:.3f},{row['runs']},"
            f"{row['success_mean']:.4f},{row['success_sd']:.4f},{row['feedback_mean']:.4f},"
            f"{row['train_mean']:.4f},{row['test_mean']:.4f}"
        )
    print("frontier_auc:")
    for method, area in sorted(auc.items()):
        print(f"  {method}: {area:.4f}")


if __name__ == "__main__":
    main()

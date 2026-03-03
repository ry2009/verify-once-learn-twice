from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _split_from_task(task_id: str) -> str:
    if task_id.startswith("KB-SOURCE/"):
        return "source"
    if task_id.startswith("KB-TARGET/"):
        return "target"
    return "other"


@dataclass
class RunMetrics:
    run_group: str
    run_dir: str
    base_model: str
    method: str
    seed: int
    n_source: int
    n_target: int
    source_final: float
    target_final: float
    source_first: float
    target_first: float
    source_lift: float
    target_lift: float
    avg_feedback_calls: float
    avg_train_steps: float
    avg_test_calls: float


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / float(len(xs)) if xs else 0.0


def _run_metrics(run_group: str, run_dir: Path) -> RunMetrics | None:
    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "events.jsonl"
    if not cfg_path.exists() or not ev_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    method = str(cfg.get("method", ""))
    seed = int(cfg.get("seed", -1))
    base_model = str(cfg.get("base_model", ""))

    first_test: dict[str, bool] = {}
    done: dict[str, bool] = {}
    feedback_calls: list[float] = []
    train_steps: list[float] = []
    test_calls: list[float] = []

    for ev in _iter_jsonl(ev_path):
        et = ev.get("type")
        task_id = str(ev.get("task_id", ""))
        if et == "test" and task_id and task_id not in first_test:
            first_test[task_id] = bool(ev.get("passed", False))
        if et == "task_done" and task_id:
            done[task_id] = bool(ev.get("success", False))
            feedback_calls.append(float(ev.get("feedback_calls", 0.0)))
            train_steps.append(float(ev.get("train_steps", 0.0)))
            test_calls.append(float(ev.get("test_calls", 0.0)))

    if not done:
        return None

    src_done = [float(v) for k, v in done.items() if _split_from_task(k) == "source"]
    tgt_done = [float(v) for k, v in done.items() if _split_from_task(k) == "target"]
    src_first = [
        float(v) for k, v in first_test.items() if _split_from_task(k) == "source" and k in done
    ]
    tgt_first = [
        float(v) for k, v in first_test.items() if _split_from_task(k) == "target" and k in done
    ]

    src_first_mean = _safe_mean(src_first)
    tgt_first_mean = _safe_mean(tgt_first)
    src_final_mean = _safe_mean(src_done)
    tgt_final_mean = _safe_mean(tgt_done)

    return RunMetrics(
        run_group=run_group,
        run_dir=str(run_dir),
        base_model=base_model,
        method=method,
        seed=seed,
        n_source=len(src_done),
        n_target=len(tgt_done),
        source_final=src_final_mean,
        target_final=tgt_final_mean,
        source_first=src_first_mean,
        target_first=tgt_first_mean,
        source_lift=src_final_mean - src_first_mean,
        target_lift=tgt_final_mean - tgt_first_mean,
        avg_feedback_calls=_safe_mean(feedback_calls),
        avg_train_steps=_safe_mean(train_steps),
        avg_test_calls=_safe_mean(test_calls),
    )


def _collect_groups(groups: list[str]) -> list[RunMetrics]:
    rows: list[RunMetrics] = []
    for group in groups:
        root = Path("runs") / group
        if not root.exists():
            continue
        latest_by_name: dict[str, tuple[float, Path]] = {}
        for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            cfg_path = run_dir / "config.json"
            if not cfg_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            run_name = str(cfg.get("run_name", ""))
            if not run_name:
                continue
            ev_path = run_dir / "events.jsonl"
            stamp = ev_path.stat().st_mtime if ev_path.exists() else run_dir.stat().st_mtime
            prev = latest_by_name.get(run_name)
            if prev is None or stamp > prev[0]:
                latest_by_name[run_name] = (stamp, run_dir)
        for _, run_dir in sorted(latest_by_name.values(), key=lambda x: x[1].name):
            row = _run_metrics(group, run_dir)
            if row is not None:
                rows.append(row)
    return rows


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.mean(xs), stats.pstdev(xs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", action="append", required=True)
    parser.add_argument(
        "--out_dir",
        default="runs/phase48_kernelbench_transfer_summary",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _collect_groups(args.run_group)
    if not runs:
        raise SystemExit("no runs found for requested groups")

    run_csv = out_dir / "run_metrics.csv"
    with run_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_group",
                "run_dir",
                "base_model",
                "method",
                "seed",
                "n_source",
                "n_target",
                "source_final",
                "target_final",
                "source_first",
                "target_first",
                "source_lift",
                "target_lift",
                "avg_feedback_calls",
                "avg_train_steps",
                "avg_test_calls",
            ]
        )
        for r in runs:
            w.writerow(
                [
                    r.run_group,
                    r.run_dir,
                    r.base_model,
                    r.method,
                    r.seed,
                    r.n_source,
                    r.n_target,
                    f"{r.source_final:.6f}",
                    f"{r.target_final:.6f}",
                    f"{r.source_first:.6f}",
                    f"{r.target_first:.6f}",
                    f"{r.source_lift:.6f}",
                    f"{r.target_lift:.6f}",
                    f"{r.avg_feedback_calls:.6f}",
                    f"{r.avg_train_steps:.6f}",
                    f"{r.avg_test_calls:.6f}",
                ]
            )

    by_method: dict[tuple[str, str], list[RunMetrics]] = defaultdict(list)
    for r in runs:
        by_method[(r.base_model, r.method)].append(r)

    agg_rows: list[dict[str, Any]] = []
    for (base_model, method), rows in sorted(by_method.items()):
        tgt_final_mean, tgt_final_sd = _mean_sd([r.target_final for r in rows])
        tgt_first_mean, tgt_first_sd = _mean_sd([r.target_first for r in rows])
        src_final_mean, src_final_sd = _mean_sd([r.source_final for r in rows])
        src_first_mean, src_first_sd = _mean_sd([r.source_first for r in rows])
        tgt_lift_mean, tgt_lift_sd = _mean_sd([r.target_lift for r in rows])
        agg_rows.append(
            {
                "base_model": base_model,
                "method": method,
                "runs": len(rows),
                "source_final_mean": src_final_mean,
                "source_final_sd": src_final_sd,
                "source_first_mean": src_first_mean,
                "source_first_sd": src_first_sd,
                "target_final_mean": tgt_final_mean,
                "target_final_sd": tgt_final_sd,
                "target_first_mean": tgt_first_mean,
                "target_first_sd": tgt_first_sd,
                "target_lift_mean": tgt_lift_mean,
                "target_lift_sd": tgt_lift_sd,
                "avg_feedback_calls_mean": _safe_mean([r.avg_feedback_calls for r in rows]),
                "avg_train_steps_mean": _safe_mean([r.avg_train_steps for r in rows]),
                "avg_test_calls_mean": _safe_mean([r.avg_test_calls for r in rows]),
            }
        )

    agg_json = out_dir / "method_aggregate.json"
    agg_json.write_text(json.dumps(agg_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    agg_csv = out_dir / "method_aggregate.csv"
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = [
            "base_model",
            "method",
            "runs",
            "source_first_mean",
            "source_final_mean",
            "target_first_mean",
            "target_final_mean",
            "target_lift_mean",
            "avg_feedback_calls_mean",
            "avg_train_steps_mean",
            "avg_test_calls_mean",
        ]
        w.writerow(cols)
        for row in agg_rows:
            w.writerow([row[c] for c in cols])

    md = ["# KernelBench Transfer Summary", ""]
    md.append("| Base model | Method | Runs | Source first | Source final | Target first | Target final | Target lift | Avg feedback | Avg train steps |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(
        agg_rows,
        key=lambda x: (x["target_final_mean"], x["target_lift_mean"]),
        reverse=True,
    ):
        md.append(
            f"| {row['base_model']} | {row['method']} | {row['runs']} | "
            f"{row['source_first_mean']:.3f} | {row['source_final_mean']:.3f} | "
            f"{row['target_first_mean']:.3f} | {row['target_final_mean']:.3f} | "
            f"{row['target_lift_mean']:.3f} | {row['avg_feedback_calls_mean']:.3f} | "
            f"{row['avg_train_steps_mean']:.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

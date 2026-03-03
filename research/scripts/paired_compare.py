from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from dataclasses import dataclass
from math import comb
from typing import Dict, Iterable, List, Tuple


@dataclass
class RunRow:
    run_group: str
    run_dir: str
    mtime: float
    method: str
    inner_updates: int
    feedback_budget: int
    judge_flip_prob: float
    seed: int
    tasks: int
    success_rate: float
    avg_feedback_calls: float
    avg_train_steps: float
    avg_test_calls: float


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: List[float]) -> float:
    return statistics.pstdev(vals) if vals else 0.0


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0.0:
        return sorted_vals[0]
    if q >= 1.0:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _bootstrap_mean_ci(
    vals: List[float],
    *,
    samples: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    if len(vals) == 1:
        return (vals[0], vals[0])

    rng = random.Random(seed)
    means: List[float] = []
    n = len(vals)
    for _ in range(max(1, samples)):
        draw = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(draw))
    means.sort()
    lo = _quantile(means, alpha / 2.0)
    hi = _quantile(means, 1.0 - alpha / 2.0)
    return (lo, hi)


def _exact_sign_test_greater(vals: List[float]) -> Tuple[int, int, int, float]:
    """Exact one-sided sign-test p-value for median delta > 0.

    Ties are excluded from n.
    """
    wins = sum(1 for v in vals if v > 0)
    losses = sum(1 for v in vals if v < 0)
    ties = len(vals) - wins - losses
    n = wins + losses
    if n == 0:
        return wins, losses, ties, 1.0
    # P(X >= wins) where X ~ Binomial(n, 0.5)
    tail = sum(comb(n, k) for k in range(wins, n + 1))
    p = tail / (2**n)
    return wins, losses, ties, float(p)


def _paired_signflip_pvalue(vals: List[float], seed: int, max_samples: int = 131072) -> float:
    """One-sided paired sign-flip p-value for mean(delta) > 0.

    Uses exact enumeration for small n, randomized Monte Carlo otherwise.
    """
    if not vals:
        return 1.0
    n = len(vals)
    observed = _mean(vals)
    tol = 1e-12
    if n <= 18:
        total = 1 << n
        ge = 0
        for mask in range(total):
            s = 0.0
            for i, v in enumerate(vals):
                s += v if ((mask >> i) & 1) else -v
            if (s / n) >= (observed - tol):
                ge += 1
        return ge / total

    rng = random.Random(seed)
    ge = 0
    total = max_samples
    for _ in range(total):
        s = 0.0
        for v in vals:
            s += v if rng.random() < 0.5 else -v
        if (s / n) >= (observed - tol):
            ge += 1
    return ge / total


def _read_run(run_group: str, run_dir: str, min_tasks: int) -> RunRow | None:
    cfg_path = os.path.join(run_dir, "config.json")
    ev_path = os.path.join(run_dir, "events.jsonl")
    if not (os.path.exists(cfg_path) and os.path.exists(ev_path)):
        return None

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    task_done = [ev for ev in _iter_jsonl(ev_path) if ev.get("type") == "task_done"]
    if len(task_done) < min_tasks:
        return None

    success = [1.0 if ev.get("success") else 0.0 for ev in task_done]
    feedback = [float(ev.get("feedback_calls", 0.0)) for ev in task_done]
    train = [float(ev.get("train_steps", 0.0)) for ev in task_done]
    tests = [float(ev.get("test_calls", 0.0)) for ev in task_done]

    return RunRow(
        run_group=run_group,
        run_dir=run_dir,
        mtime=float(os.path.getmtime(run_dir)),
        method=str(cfg.get("method", "unknown")),
        inner_updates=int(cfg.get("inner_updates", 1)),
        feedback_budget=int(cfg.get("feedback_budget", -1)),
        judge_flip_prob=float(cfg.get("judge_flip_prob", 0.0)),
        seed=int(cfg.get("seed", -1)),
        tasks=len(task_done),
        success_rate=_mean(success),
        avg_feedback_calls=_mean(feedback),
        avg_train_steps=_mean(train),
        avg_test_calls=_mean(tests),
    )


def _collect(run_groups: List[str], min_tasks: int) -> List[RunRow]:
    rows: List[RunRow] = []
    for run_group in run_groups:
        root = os.path.join("runs", run_group)
        if not os.path.isdir(root):
            raise SystemExit(f"Run group not found: {root}")
        for name in sorted(os.listdir(root)):
            run_dir = os.path.join(root, name)
            if not os.path.isdir(run_dir):
                continue
            row = _read_run(run_group, run_dir, min_tasks=min_tasks)
            if row is not None:
                rows.append(row)
    return rows


def _filter_rows(
    rows: List[RunRow],
    method: str,
    inner_updates: int | None,
    feedback_budget: int | None,
    judge_flip_prob: float | None,
    pair_key: str,
) -> Dict[Tuple[str, int] | int, RunRow]:
    out: Dict[Tuple[str, int] | int, RunRow] = {}
    for row in rows:
        if row.method != method:
            continue
        if inner_updates is not None and row.inner_updates != inner_updates:
            continue
        if feedback_budget is not None and row.feedback_budget != feedback_budget:
            continue
        if judge_flip_prob is not None and abs(row.judge_flip_prob - judge_flip_prob) > 1e-12:
            continue
        key: Tuple[str, int] | int
        if pair_key == "seed":
            key = row.seed
        else:
            key = (row.run_group, row.seed)
        prev = out.get(key)
        if prev is None:
            out[key] = row
            continue
        # Keep the row with more task coverage; break ties by newest directory.
        if (row.tasks, row.mtime) >= (prev.tasks, prev.mtime):
            out[key] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", action="append", required=True)
    parser.add_argument("--method_a", required=True)
    parser.add_argument("--method_b", required=True)
    parser.add_argument("--inner_updates_a", type=int, default=None)
    parser.add_argument("--inner_updates_b", type=int, default=None)
    parser.add_argument("--feedback_budget", type=int, default=None)
    parser.add_argument("--judge_flip_prob", type=float, default=None)
    parser.add_argument(
        "--pair_key",
        choices=["run_group_seed", "seed"],
        default="run_group_seed",
        help="Pair runs by (run_group, seed) or by seed only across run groups.",
    )
    parser.add_argument("--min_tasks", type=int, default=1)
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random_seed", type=int, default=2026)
    args = parser.parse_args()

    rows = _collect(args.run_group, min_tasks=args.min_tasks)
    a_map = _filter_rows(
        rows,
        method=args.method_a,
        inner_updates=args.inner_updates_a,
        feedback_budget=args.feedback_budget,
        judge_flip_prob=args.judge_flip_prob,
        pair_key=args.pair_key,
    )
    b_map = _filter_rows(
        rows,
        method=args.method_b,
        inner_updates=args.inner_updates_b,
        feedback_budget=args.feedback_budget,
        judge_flip_prob=args.judge_flip_prob,
        pair_key=args.pair_key,
    )

    shared_pairs = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not shared_pairs:
        raise SystemExit("No shared seeds between requested method slices")

    def deltas(metric: str) -> List[float]:
        return [
            getattr(a_map[key], metric) - getattr(b_map[key], metric)
            for key in shared_pairs
        ]

    d_success = deltas("success_rate")
    d_feedback = deltas("avg_feedback_calls")
    d_train = deltas("avg_train_steps")
    d_test = deltas("avg_test_calls")

    def stats(vals: List[float], ci_seed_offset: int) -> dict:
        ci_lo, ci_hi = _bootstrap_mean_ci(
            vals,
            samples=max(1, int(args.bootstrap_samples)),
            alpha=float(args.alpha),
            seed=int(args.random_seed) + ci_seed_offset,
        )
        wins, losses, ties, p_sign = _exact_sign_test_greater(vals)
        p_flip = _paired_signflip_pvalue(vals, seed=int(args.random_seed) + 97 + ci_seed_offset)
        return {
            "values": vals,
            "mean": _mean(vals),
            "sd": _std(vals),
            "ci": [ci_lo, ci_hi],
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "p_sign_greater": p_sign,
            "p_signflip_mean_greater": p_flip,
        }

    shared_seeds: List[int] | None = None
    shared_run_seed: List[str] | None = None
    if args.pair_key == "seed":
        shared_seeds = [int(seed) for seed in shared_pairs]
    elif len(args.run_group) == 1:
        shared_seeds = [seed for (_, seed) in shared_pairs]
    else:
        shared_run_seed = [f"{group}:{seed}" for (group, seed) in shared_pairs]

    result = {
        "run_groups": args.run_group,
        "run_group": args.run_group[0] if len(args.run_group) == 1 else None,
        "method_a": args.method_a,
        "method_b": args.method_b,
        "inner_updates_a": args.inner_updates_a,
        "inner_updates_b": args.inner_updates_b,
        "feedback_budget": args.feedback_budget,
        "judge_flip_prob": args.judge_flip_prob,
        "pair_key": args.pair_key,
        "shared_seeds": shared_seeds,
        "shared_run_seed": shared_run_seed,
        "pairs": len(shared_pairs),
        "bootstrap_samples": int(args.bootstrap_samples),
        "alpha": float(args.alpha),
        "delta_success": stats(d_success, 0),
        "delta_feedback": stats(d_feedback, 1),
        "delta_train_steps": stats(d_train, 2),
        "delta_test_calls": stats(d_test, 3),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

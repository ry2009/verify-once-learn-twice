from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_leaderboard(run_group: str, min_tasks: int, out_json: Path) -> dict:
    if out_json.exists():
        return _load(out_json)
    raise FileNotFoundError(
        f"Leaderboard missing for {run_group}. Generate with scripts/costly_leaderboard.py first."
    )


def _pick(rows: List[dict], method_variant: str, budget: int, noise: float = 0.0) -> dict | None:
    for row in rows:
        if row.get("method_variant") != method_variant:
            continue
        if int(row.get("feedback_budget", -1)) != budget:
            continue
        if abs(float(row.get("judge_flip_prob", 0.0)) - noise) > 1e-12:
            continue
        return row
    return None


def _fmt(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{x:.4f}"


def _resolve_leaderboard_path(run_group: str, min_tasks: int, allow_glob: bool = False) -> Path | None:
    lb_path = Path("runs") / run_group / f"costly_leaderboard_min{min_tasks}.json"
    if lb_path.exists():
        return lb_path
    partial = Path("runs") / run_group / f"costly_leaderboard_partial_min{min_tasks}.json"
    if partial.exists():
        return partial
    if allow_glob:
        cands = sorted(glob.glob(f"runs/{run_group}/costly_leaderboard_min*.json"))
        if cands:
            return Path(cands[-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_md", default="runs/adaptive_win_snapshot.md")
    args = parser.parse_args()

    targets: List[Tuple[str, int, List[int], bool]] = [
        ("phase28_combined", 12, [1], False),
        ("verify_phase30b_mbpp20_budget1_s0123_nors", 10, [1], False),
        ("verify_phase32_synth20_budget1_s0123_nors", 10, [1], False),
        ("verify_phase33_hardset_h12pair_budget1_s1223", 6, [1], True),
        ("verify_phase33_hardset_numeric_budget1_s01234567", 8, [1], True),
        ("verify_phase31_numeric_budget1_s01234567", 10, [1], False),
        ("verify_phase31_string_budget1_s01234567", 8, [1], False),
        ("verify_phase31_symbolic_budget1_s01234567", 7, [1], False),
        ("verify_phase22_h10_guardreplay_core_s01234", 10, [2], False),
        ("verify_phase22_h12_guardreplay_core_s01234", 12, [2], False),
        ("verify_phase20_numeric_guardreplay_s01234_b2", 10, [2], False),
        ("verify_phase20_string_guardreplay_s01234_b2", 8, [2], False),
        ("verify_phase20_symbolic_guardreplay_s01234_b2", 7, [2], False),
    ]

    lines: List[str] = []
    lines.append("# Adaptive Win Snapshot")
    lines.append("")
    lines.append("| Run Group | Budget | Adaptive Success | Baseline Success | Delta Success | Adaptive Feedback | Baseline Feedback | Delta Feedback |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for run_group, min_tasks, budgets, allow_glob in targets:
        lb_path = _resolve_leaderboard_path(run_group, min_tasks, allow_glob=allow_glob)
        if lb_path is None:
            lines.append(f"| {run_group} | NA | NA | NA | NA | NA | NA | NA |")
            continue
        lb = _ensure_leaderboard(run_group, min_tasks, lb_path)
        rows = lb.get("aggregate", [])
        for budget in budgets:
            adaptive = _pick(rows, "adaptive_fwb", budget)
            baseline = _pick(rows, "fixed_k_judge_k1", budget)
            if baseline is None:
                baseline = _pick(rows, "fixed_k_fwb_k1", budget)
            if adaptive is None or baseline is None:
                lines.append(f"| {run_group} | {budget} | NA | NA | NA | NA | NA | NA |")
                continue
            a_s = float(adaptive.get("success_mean", 0.0))
            b_s = float(baseline.get("success_mean", 0.0))
            a_f = float(adaptive.get("feedback_mean", 0.0))
            b_f = float(baseline.get("feedback_mean", 0.0))
            lines.append(
                f"| {run_group} | {budget} | {_fmt(a_s)} | {_fmt(b_s)} | {_fmt(a_s - b_s)} | {_fmt(a_f)} | {_fmt(b_f)} | {_fmt(a_f - b_f)} |"
            )

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

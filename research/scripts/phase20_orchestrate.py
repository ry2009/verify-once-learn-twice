from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SweepSpec:
    spec_path: str
    run_group: str
    min_tasks: int
    methods_expected: List[str]


SWEEPS: List[SweepSpec] = [
    SweepSpec(
        spec_path="data/ablation_phase20_numeric_guardreplay_s01234_b2.json",
        run_group="verify_phase20_numeric_guardreplay_s01234_b2",
        min_tasks=10,
        methods_expected=["adaptive_fwb", "fixed_k_judge", "fixed_k_fwb", "inference_only"],
    ),
    SweepSpec(
        spec_path="data/ablation_phase20_string_guardreplay_s01234_b2.json",
        run_group="verify_phase20_string_guardreplay_s01234_b2",
        min_tasks=8,
        methods_expected=["adaptive_fwb", "fixed_k_judge", "fixed_k_fwb", "inference_only"],
    ),
    SweepSpec(
        spec_path="data/ablation_phase20_symbolic_guardreplay_s01234_b2.json",
        run_group="verify_phase20_symbolic_guardreplay_s01234_b2",
        min_tasks=7,
        methods_expected=["adaptive_fwb", "fixed_k_judge", "fixed_k_fwb", "inference_only"],
    ),
    SweepSpec(
        spec_path="data/ablation_phase20_general_guardreplay_s0_b2.json",
        run_group="verify_phase20_general_guardreplay_s0_b2",
        min_tasks=55,
        methods_expected=["adaptive_fwb", "fixed_k_judge", "fixed_k_fwb", "inference_only"],
    ),
    SweepSpec(
        spec_path="data/ablation_phase20_mbpp_hidden70_guardreplay_s0_b2.json",
        run_group="verify_phase20_mbpp_hidden70_guardreplay_s0_b2",
        min_tasks=120,
        methods_expected=["adaptive_fwb", "fixed_k_judge", "fixed_k_fwb", "inference_only"],
    ),
    SweepSpec(
        spec_path="data/ablation_phase20_synth_hidden70_guardreplay_s0_b2.json",
        run_group="verify_phase20_synth_hidden70_guardreplay_s0_b2",
        min_tasks=80,
        methods_expected=["adaptive_fwb", "fixed_k_judge", "fixed_k_fwb", "inference_only"],
    ),
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _expected_run_names(spec_path: Path) -> List[str]:
    spec = _load_json(spec_path)
    return [str(run.get("run_name", "")) for run in spec.get("runs", []) if run.get("run_name")]


def _latest_run_dirs(run_group: str) -> Dict[str, Path]:
    root = Path("runs") / run_group
    if not root.exists():
        return {}
    latest: Dict[str, Path] = {}
    latest_mtime: Dict[str, float] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        cfg_path = child / "config.json"
        ev_path = child / "events.jsonl"
        if not cfg_path.exists() or not ev_path.exists():
            continue
        try:
            cfg = _load_json(cfg_path)
        except Exception:
            continue
        run_name = str(cfg.get("run_name", "")).strip()
        if not run_name:
            continue
        mt = child.stat().st_mtime
        if run_name not in latest_mtime or mt >= latest_mtime[run_name]:
            latest_mtime[run_name] = mt
            latest[run_name] = child
    return latest


def _task_done_count(ev_path: Path) -> int:
    if not ev_path.exists():
        return 0
    n = 0
    for ev in _iter_jsonl(ev_path):
        if ev.get("type") == "task_done":
            n += 1
    return n


def _run_completed(run_dir: Path) -> bool:
    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "events.jsonl"
    if not cfg_path.exists() or not ev_path.exists():
        return False
    try:
        cfg = _load_json(cfg_path)
    except Exception:
        return False
    max_steps = int(cfg.get("max_steps", 0))
    if max_steps <= 0:
        return False
    return _task_done_count(ev_path) >= max_steps


def _completion_status(spec: SweepSpec) -> dict:
    expected = _expected_run_names(Path(spec.spec_path))
    latest = _latest_run_dirs(spec.run_group)
    complete = [rn for rn in expected if rn in latest and _run_completed(latest[rn])]
    pending = [rn for rn in expected if rn not in complete]
    return {
        "expected": expected,
        "complete": complete,
        "pending": pending,
        "done": len(pending) == 0 and len(expected) > 0,
    }


def _launch(spec: SweepSpec, env: Dict[str, str]) -> subprocess.Popen:
    cmd = ["python3", "scripts/run_ablation.py", "--spec", spec.spec_path]
    return subprocess.Popen(
        cmd,
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _drain_output(tag: str, proc: subprocess.Popen, out_dir: Path) -> None:
    # Child processes write no stdout; nothing to drain.
    _ = (tag, proc, out_dir)


def _run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _finalize_outputs(sweeps: List[SweepSpec]) -> None:
    for sw in sweeps:
        group_dir = Path("runs") / sw.run_group
        group_dir.mkdir(parents=True, exist_ok=True)
        out_json = group_dir / f"costly_leaderboard_min{sw.min_tasks}.json"
        out_txt = group_dir / f"costly_leaderboard_min{sw.min_tasks}.txt"
        with out_txt.open("w", encoding="utf-8") as f:
            subprocess.run(
                [
                    "python3",
                    "scripts/costly_leaderboard.py",
                    "--run_group",
                    sw.run_group,
                    "--min_tasks",
                    str(sw.min_tasks),
                    "--out_json",
                    str(out_json),
                ],
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        # Build pretty dashboard with a matching task threshold.
        _run_cmd(
            [
                "python3",
                "scripts/plot_pretty_dashboard.py",
                "--run_group",
                sw.run_group,
                "--min_tasks",
                str(sw.min_tasks),
                "--title",
                f"Phase-20: {sw.run_group}",
                "--write_png",
            ]
        )

        # Pairwise reports where both methods are expected in this sweep.
        if "adaptive_fwb" in sw.methods_expected and "fixed_k_judge" in sw.methods_expected:
            out_pair = group_dir / "paired_adaptive_vs_fixedkj1.json"
            with out_pair.open("w", encoding="utf-8") as f:
                subprocess.run(
                    [
                        "python3",
                        "scripts/paired_compare.py",
                        "--run_group",
                        sw.run_group,
                        "--method_a",
                        "adaptive_fwb",
                        "--method_b",
                        "fixed_k_judge",
                        "--inner_updates_b",
                        "1",
                        "--feedback_budget",
                        "2",
                        "--min_tasks",
                        str(sw.min_tasks),
                    ],
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )

        if "adaptive_fwb" in sw.methods_expected and "fixed_k_fwb" in sw.methods_expected:
            out_pair = group_dir / "paired_adaptive_vs_fixedkfwb1.json"
            with out_pair.open("w", encoding="utf-8") as f:
                subprocess.run(
                    [
                        "python3",
                        "scripts/paired_compare.py",
                        "--run_group",
                        sw.run_group,
                        "--method_a",
                        "adaptive_fwb",
                        "--method_b",
                        "fixed_k_fwb",
                        "--inner_updates_b",
                        "1",
                        "--feedback_budget",
                        "2",
                        "--min_tasks",
                        str(sw.min_tasks),
                    ],
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )


def _read_leaderboard(path: Path) -> dict:
    return _load_json(path)


def _pick_metric_row(agg: List[dict], method_variant: str) -> dict | None:
    for row in agg:
        if str(row.get("method_variant")) == method_variant:
            return row
    return None


def _write_phase20_summary(sweeps: List[SweepSpec]) -> None:
    md_lines: List[str] = []
    md_lines.append("# Phase-20 Final Summary")
    md_lines.append("")
    md_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")

    tex_lines: List[str] = []
    tex_lines.append("\\begin{tabular}{llrrrr}")
    tex_lines.append("\\toprule")
    tex_lines.append("Group & Method & Success & Avg Feedback & Avg Train Steps & Avg Test Calls \\\\")
    tex_lines.append("\\midrule")

    for sw in sweeps:
        group_dir = Path("runs") / sw.run_group
        lb_json = group_dir / f"costly_leaderboard_min{sw.min_tasks}.json"
        if not lb_json.exists():
            continue
        data = _read_leaderboard(lb_json)
        agg = data.get("aggregate", [])
        if not agg:
            continue

        # Best by success; ties broken by lower feedback then lower train cost.
        best = sorted(
            agg,
            key=lambda r: (
                -float(r.get("success_mean", 0.0)),
                float(r.get("feedback_mean", 1e9)),
                float(r.get("train_mean", 1e9)),
            ),
        )[0]
        adaptive = _pick_metric_row(agg, "adaptive_fwb")

        md_lines.append(f"## {sw.run_group}")
        md_lines.append(f"- Best: `{best.get('method_variant')}` success={best.get('success_mean'):.4f}")
        if adaptive is not None:
            md_lines.append(
                "- Adaptive: "
                f"success={adaptive.get('success_mean'):.4f}, "
                f"feedback={adaptive.get('feedback_mean'):.4f}, "
                f"train={adaptive.get('train_mean'):.4f}, "
                f"test={adaptive.get('test_mean'):.4f}"
            )
        md_lines.append("")

        for row in sorted(agg, key=lambda r: -float(r.get("success_mean", 0.0))):
            tex_lines.append(
                f"{sw.run_group.replace('_', '\\_')} & "
                f"{str(row.get('method_variant', '')).replace('_', '\\_')} & "
                f"{float(row.get('success_mean', 0.0)):.3f} & "
                f"{float(row.get('feedback_mean', 0.0)):.3f} & "
                f"{float(row.get('train_mean', 0.0)):.3f} & "
                f"{float(row.get('test_mean', 0.0)):.3f} \\\\"
            )
        tex_lines.append("\\midrule")

    if tex_lines[-1] == "\\midrule":
        tex_lines.pop()
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")

    Path("runs/phase20_final_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    paper_tables = Path("paper/tables")
    paper_tables.mkdir(parents=True, exist_ok=True)
    (paper_tables / "phase20_final_summary.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll_s", type=int, default=30)
    parser.add_argument("--max_hours", type=float, default=24.0)
    args = parser.parse_args()

    if not os.getenv("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set")

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    procs: Dict[str, subprocess.Popen] = {}
    start = time.time()
    deadline = start + args.max_hours * 3600.0

    while True:
        all_done = True
        status_rows = []
        for sw in SWEEPS:
            st = _completion_status(sw)
            done_n = len(st["complete"])
            total_n = len(st["expected"])
            print(f"[status] {sw.run_group}: {done_n}/{total_n} runs complete", flush=True)
            status_rows.append(
                {
                    "run_group": sw.run_group,
                    "complete_runs": done_n,
                    "total_runs": total_n,
                    "done": st["done"],
                    "pending": st["pending"],
                }
            )

            if st["done"]:
                # Let any stale process end naturally.
                proc = procs.get(sw.spec_path)
                if proc is not None and proc.poll() is not None:
                    _drain_output(sw.run_group, proc, Path("runs") / sw.run_group)
                    procs.pop(sw.spec_path, None)
                continue

            all_done = False
            proc = procs.get(sw.spec_path)
            if proc is None or proc.poll() is not None:
                if proc is not None:
                    _drain_output(sw.run_group, proc, Path("runs") / sw.run_group)
                print(f"[launch] {sw.spec_path}", flush=True)
                procs[sw.spec_path] = _launch(sw, env)

        status_path = Path("runs") / "phase20_supervisor_status.json"
        status_path.write_text(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "all_done": all_done,
                    "rows": status_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if all_done:
            break
        if time.time() > deadline:
            print("[timeout] max_hours reached", flush=True)
            break
        time.sleep(args.poll_s)

    for sw in SWEEPS:
        proc = procs.get(sw.spec_path)
        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
            _drain_output(sw.run_group, proc, Path("runs") / sw.run_group)

    if all(_completion_status(sw)["done"] for sw in SWEEPS):
        print("[finalize] all sweeps complete; generating outputs", flush=True)
        _finalize_outputs(SWEEPS)
        _write_phase20_summary(SWEEPS)
        print("[finalize] done", flush=True)
    else:
        print("[finalize] skipped because not all sweeps completed", flush=True)


if __name__ == "__main__":
    main()

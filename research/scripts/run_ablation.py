import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttRL.config import ExperimentConfig, SamplingConfig
from ttRL.loop import ExperimentRunner


def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_run_dir(run_group: str, run_name: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", run_group, f"{ts}-{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_runs(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "runs" in spec:
        return spec["runs"]
    if "grid" in spec:
        return list(iter_grid(spec["grid"]))
    raise ValueError("Spec must include 'runs' or 'grid'")


def _run_name_for(run: Dict[str, Any]) -> str:
    if "run_name" in run:
        return run["run_name"]
    parts = [f"{k}-{run[k]}" for k in sorted(run.keys())]
    return "__".join(parts)


def run_one(spec: Dict[str, Any], run: Dict[str, Any]) -> str:
    run_name = _run_name_for(run)

    cfg = ExperimentConfig(
        base_model=spec.get("base_model", "meta-llama/Llama-3.1-8B-Instruct"),
        judge_model=spec.get("judge_model"),
        run_name=run_name,
        run_group=spec.get("run_group", "ablation"),
        ablation_tag=spec.get("ablation_tag", spec.get("name", "ablation")),
        method=run.get("method", spec.get("method", "adaptive_fwb")),
        tasks_path=spec.get("tasks_path", "data/tasks_min.jsonl"),
        feedback_budget=int(run.get("feedback_budget", spec.get("feedback_budget", 4))),
        inner_updates=int(run.get("inner_updates", spec.get("inner_updates", 2))),
        adaptive_judge_every_updates=int(
            run.get(
                "adaptive_judge_every_updates",
                spec.get("adaptive_judge_every_updates", 1),
            )
        ),
        max_adaptive_steps_per_feedback=int(
            run.get(
                "max_adaptive_steps_per_feedback",
                spec.get("max_adaptive_steps_per_feedback", 8),
            )
        ),
        teacher_resample_attempts=int(
            run.get(
                "teacher_resample_attempts",
                spec.get("teacher_resample_attempts", 3),
            )
        ),
        teacher_best_of_n=int(
            run.get(
                "teacher_best_of_n",
                spec.get("teacher_best_of_n", 1),
            )
        ),
        teacher_min_op_coverage=float(
            run.get(
                "teacher_min_op_coverage",
                spec.get("teacher_min_op_coverage", 0.0),
            )
        ),
        teacher_rerank_coverage_weight=float(
            run.get(
                "teacher_rerank_coverage_weight",
                spec.get("teacher_rerank_coverage_weight", 1.0),
            )
        ),
        teacher_rerank_syntax_bonus=float(
            run.get(
                "teacher_rerank_syntax_bonus",
                spec.get("teacher_rerank_syntax_bonus", 0.25),
            )
        ),
        teacher_rerank_banned_penalty=float(
            run.get(
                "teacher_rerank_banned_penalty",
                spec.get("teacher_rerank_banned_penalty", 1.0),
            )
        ),
        teacher_reject_banned_calls=bool(
            run.get(
                "teacher_reject_banned_calls",
                spec.get("teacher_reject_banned_calls", True),
            )
        ),
        teacher_filter_with_judge=bool(
            run.get(
                "teacher_filter_with_judge",
                spec.get("teacher_filter_with_judge", False),
            )
        ),
        teacher_filter_policy=run.get(
            "teacher_filter_policy",
            spec.get("teacher_filter_policy", "pass_only"),
        ),
        teacher_max_tokens=run.get(
            "teacher_max_tokens",
            spec.get("teacher_max_tokens"),
        ),
        teacher_temperature=run.get(
            "teacher_temperature",
            spec.get("teacher_temperature"),
        ),
        teacher_top_p=run.get(
            "teacher_top_p",
            spec.get("teacher_top_p"),
        ),
        teacher_use_stop=bool(
            run.get(
                "teacher_use_stop",
                spec.get("teacher_use_stop", True),
            )
        ),
        teacher_recursive_enabled=bool(
            run.get(
                "teacher_recursive_enabled",
                spec.get("teacher_recursive_enabled", False),
            )
        ),
        teacher_recursive_depth=int(
            run.get(
                "teacher_recursive_depth",
                spec.get("teacher_recursive_depth", 1),
            )
        ),
        teacher_recursive_chunk_chars=int(
            run.get(
                "teacher_recursive_chunk_chars",
                spec.get("teacher_recursive_chunk_chars", 2400),
            )
        ),
        teacher_recursive_max_chunks=int(
            run.get(
                "teacher_recursive_max_chunks",
                spec.get("teacher_recursive_max_chunks", 6),
            )
        ),
        teacher_recursive_query_tokens=int(
            run.get(
                "teacher_recursive_query_tokens",
                spec.get("teacher_recursive_query_tokens", 128),
            )
        ),
        teacher_recursive_root_tokens=int(
            run.get(
                "teacher_recursive_root_tokens",
                spec.get("teacher_recursive_root_tokens", 192),
            )
        ),
        teacher_recursive_summary_chars=int(
            run.get(
                "teacher_recursive_summary_chars",
                spec.get("teacher_recursive_summary_chars", 1200),
            )
        ),
        teacher_failure_fallback=run.get(
            "teacher_failure_fallback",
            spec.get("teacher_failure_fallback", "base"),
        ),
        judge_mode=run.get("judge_mode", spec.get("judge_mode", "llm")),
        judge_flip_prob=float(
            run.get("judge_flip_prob", spec.get("judge_flip_prob", 0.0))
        ),
        judge_votes=int(run.get("judge_votes", spec.get("judge_votes", 1))),
        judge_pass_threshold=float(
            run.get("judge_pass_threshold", spec.get("judge_pass_threshold", 1.0))
        ),
        judge_min_margin=float(
            run.get("judge_min_margin", spec.get("judge_min_margin", 0.0))
        ),
        adaptive_stop_consecutive_passes=int(
            run.get(
                "adaptive_stop_consecutive_passes",
                spec.get("adaptive_stop_consecutive_passes", 1),
            )
        ),
        reinforce_once_mode=run.get(
            "reinforce_once_mode",
            spec.get("reinforce_once_mode", "off"),
        ),
        reinforce_once_rollouts=int(
            run.get(
                "reinforce_once_rollouts",
                spec.get("reinforce_once_rollouts", 4),
            )
        ),
        reinforce_once_updates=int(
            run.get(
                "reinforce_once_updates",
                spec.get("reinforce_once_updates", 1),
            )
        ),
        reinforce_once_pass_only=bool(
            run.get(
                "reinforce_once_pass_only",
                spec.get("reinforce_once_pass_only", True),
            )
        ),
        adaptive_budget_scheduler=bool(
            run.get(
                "adaptive_budget_scheduler",
                spec.get("adaptive_budget_scheduler", False),
            )
        ),
        adaptive_budget_base=int(
            run.get(
                "adaptive_budget_base",
                spec.get("adaptive_budget_base", 1),
            )
        ),
        adaptive_budget_hard=int(
            run.get(
                "adaptive_budget_hard",
                spec.get("adaptive_budget_hard", 2),
            )
        ),
        adaptive_budget_chain=int(
            run.get(
                "adaptive_budget_chain",
                spec.get("adaptive_budget_chain", 4),
            )
        ),
        adaptive_budget_op_chain_threshold=float(
            run.get(
                "adaptive_budget_op_chain_threshold",
                spec.get("adaptive_budget_op_chain_threshold", 0.5),
            )
        ),
        verifier_budget_policy=run.get(
            "verifier_budget_policy",
            spec.get("verifier_budget_policy", "legacy"),
        ),
        verifier_budget_min_refine_steps=int(
            run.get(
                "verifier_budget_min_refine_steps",
                spec.get("verifier_budget_min_refine_steps", 1),
            )
        ),
        lr=float(run.get("lr", spec.get("lr", 1e-4))),
        lora_rank=int(run.get("lora_rank", spec.get("lora_rank", 16))),
        lora_alpha=int(run.get("lora_alpha", spec.get("lora_alpha", 32))),
        lora_dropout=float(run.get("lora_dropout", spec.get("lora_dropout", 0.05))),
        sample_every=int(run.get("sample_every", spec.get("sample_every", 1))),
        reset_per_task=bool(run.get("reset_per_task", spec.get("reset_per_task", True))),
        reset_every_n_tasks=int(
            run.get("reset_every_n_tasks", spec.get("reset_every_n_tasks", 0))
        ),
        replay_buffer_max=int(
            run.get("replay_buffer_max", spec.get("replay_buffer_max", 0))
        ),
        replay_per_update=int(
            run.get("replay_per_update", spec.get("replay_per_update", 0))
        ),
        replay_add_on_success=bool(
            run.get("replay_add_on_success", spec.get("replay_add_on_success", True))
        ),
        test_timeout_s=int(run.get("test_timeout_s", spec.get("test_timeout_s", 5))),
        max_steps=int(run.get("max_steps", spec.get("max_steps", 999999))),
        seed=int(run.get("seed", spec.get("seed", 0))),
        sampling=SamplingConfig(
            max_tokens=int(run.get("max_tokens", spec.get("max_tokens", 256))),
            temperature=float(run.get("temperature", spec.get("temperature", 0.7))),
            top_p=float(run.get("top_p", spec.get("top_p", 0.95))),
        ),
    )
    run_dir = make_run_dir(cfg.run_group, cfg.run_name)
    runner = ExperimentRunner(cfg, run_dir)
    runner.run()
    return run_dir


def _run_one_worker(spec: Dict[str, Any], run: Dict[str, Any], out_q: mp.Queue) -> None:
    try:
        run_dir = run_one(spec, run)
        out_q.put({"ok": True, "run_dir": run_dir})
    except Exception as exc:  # pragma: no cover - defensive path for live jobs
        out_q.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def run_one_with_timeout(
    spec: Dict[str, Any],
    run: Dict[str, Any],
    timeout_s: int,
    retries: int,
) -> str:
    if timeout_s <= 0:
        return run_one(spec, run)

    attempts = max(1, retries + 1)
    last_error = "unknown failure"
    for attempt in range(1, attempts + 1):
        ctx = mp.get_context("spawn")
        out_q: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_run_one_worker, args=(spec, run, out_q))
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            last_error = (
                f"timeout after {timeout_s}s (attempt {attempt}/{attempts})"
            )
            print(f"[ablation] {last_error}", flush=True)
        else:
            result = None
            try:
                if not out_q.empty():
                    result = out_q.get_nowait()
            except Exception:
                result = None
            if result and result.get("ok"):
                return str(result["run_dir"])
            if result and not result.get("ok"):
                last_error = str(result.get("error", "worker error"))
                print(f"[ablation] worker error: {last_error}", flush=True)
                tb = str(result.get("traceback", "")).strip()
                if tb:
                    print(tb, flush=True)
            else:
                last_error = f"worker exited without result (exit={proc.exitcode})"
                print(f"[ablation] {last_error}", flush=True)

        try:
            out_q.close()
        except Exception:
            pass

        if attempt < attempts:
            time.sleep(2)

    raise RuntimeError(last_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="data/ablation_min.json")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--run_timeout_s",
        type=int,
        default=0,
        help="Per-run timeout in seconds. 0 means use spec/default.",
    )
    parser.add_argument(
        "--run_retries",
        type=int,
        default=-1,
        help="Retries per run after timeout/error. -1 means use spec/default.",
    )
    args = parser.parse_args()

    if not os.getenv("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set")

    spec = load_spec(args.spec)
    runs = build_runs(spec)
    run_timeout_s = int(
        args.run_timeout_s if args.run_timeout_s > 0 else spec.get("run_timeout_s", 1800)
    )
    run_retries = int(
        args.run_retries if args.run_retries >= 0 else spec.get("run_retries", 1)
    )

    manifest = {"spec": spec, "runs": []}
    group = spec.get("run_group", "ablation")
    manifest_path = os.path.join("runs", group, "ablation_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    if not args.no_resume and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    completed_run_names = set()
    for item in manifest.get("runs", []):
        run_item = item.get("run", {})
        run_name = _run_name_for(run_item)
        completed_run_names.add(run_name)

    for run in runs:
        run_name = _run_name_for(run)
        if not args.no_resume and run_name in completed_run_names:
            print(f"Skipping completed run: {run_name}")
            continue
        print(
            f"[ablation] start run={run_name} timeout_s={run_timeout_s} retries={run_retries}",
            flush=True,
        )
        run_dir = run_one_with_timeout(
            spec=spec,
            run=run,
            timeout_s=run_timeout_s,
            retries=run_retries,
        )
        manifest["runs"].append(
            {
                "run": run,
                "run_dir": run_dir,
            }
        )
        completed_run_names.add(run_name)
        print(f"[ablation] completed run={run_name} dir={run_dir}", flush=True)
        # Persist after each run so interrupted sweeps remain auditable.
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

import argparse
import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttRL.config import ExperimentConfig, SamplingConfig
from ttRL.loop import ExperimentRunner


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", dest="tasks_path", default="data/tasks_min.jsonl")
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--run_group", default="default")
    parser.add_argument("--ablation_tag", default="none")
    parser.add_argument("--method", default="adaptive_fwb")
    parser.add_argument("--feedback_budget", type=int, default=4)
    parser.add_argument("--inner_updates", type=int, default=2)
    parser.add_argument("--adaptive_judge_every_updates", type=int, default=1)
    parser.add_argument("--max_adaptive_steps_per_feedback", type=int, default=8)
    parser.add_argument("--teacher_resample_attempts", type=int, default=3)
    parser.add_argument("--teacher_filter_with_judge", action="store_true")
    parser.add_argument(
        "--teacher_filter_policy",
        default="pass_only",
        choices=["pass_only", "pass_or_assertion"],
    )
    parser.add_argument("--teacher_max_tokens", type=int, default=None)
    parser.add_argument("--teacher_temperature", type=float, default=None)
    parser.add_argument("--teacher_top_p", type=float, default=None)
    parser.add_argument("--teacher_use_stop", action="store_true")
    parser.add_argument("--no_teacher_use_stop", action="store_true")
    parser.add_argument("--teacher_recursive_enabled", action="store_true")
    parser.add_argument("--teacher_recursive_depth", type=int, default=1)
    parser.add_argument("--teacher_recursive_chunk_chars", type=int, default=2400)
    parser.add_argument("--teacher_recursive_max_chunks", type=int, default=6)
    parser.add_argument("--teacher_recursive_query_tokens", type=int, default=128)
    parser.add_argument("--teacher_recursive_root_tokens", type=int, default=192)
    parser.add_argument("--teacher_recursive_summary_chars", type=int, default=1200)
    parser.add_argument(
        "--teacher_failure_fallback",
        default="base",
        choices=["base", "feedback"],
    )
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--judge_model", default=None)
    parser.add_argument(
        "--judge_mode",
        default="llm",
        choices=["llm", "tests", "oracle_binary"],
    )
    parser.add_argument("--judge_flip_prob", type=float, default=0.0)
    parser.add_argument("--judge_votes", type=int, default=1)
    parser.add_argument("--judge_pass_threshold", type=float, default=1.0)
    parser.add_argument("--judge_min_margin", type=float, default=0.0)
    parser.add_argument("--adaptive_stop_consecutive_passes", type=int, default=1)
    parser.add_argument(
        "--reinforce_once_mode",
        default="off",
        choices=["off", "always", "fail_only"],
    )
    parser.add_argument("--reinforce_once_rollouts", type=int, default=4)
    parser.add_argument("--reinforce_once_updates", type=int, default=1)
    parser.add_argument("--reinforce_once_pass_only", action="store_true")
    parser.add_argument("--no_reinforce_once_pass_only", action="store_true")
    parser.add_argument("--adaptive_budget_scheduler", action="store_true")
    parser.add_argument("--adaptive_budget_base", type=int, default=1)
    parser.add_argument("--adaptive_budget_hard", type=int, default=2)
    parser.add_argument("--adaptive_budget_chain", type=int, default=4)
    parser.add_argument("--adaptive_budget_op_chain_threshold", type=float, default=0.5)
    parser.add_argument(
        "--verifier_budget_policy",
        default="legacy",
        choices=["legacy", "vbc_v1"],
    )
    parser.add_argument("--verifier_budget_min_refine_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--reset_every_n_tasks", type=int, default=0)
    parser.add_argument("--replay_buffer_max", type=int, default=0)
    parser.add_argument("--replay_per_update", type=int, default=0)
    parser.add_argument("--replay_add_on_success", action="store_true")
    parser.add_argument("--no_replay_add_on_success", action="store_true")
    parser.add_argument("--reset_per_task", action="store_true")
    parser.add_argument("--no_reset_per_task", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--test_timeout_s", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    reset_per_task = True
    if args.no_reset_per_task:
        reset_per_task = False
    if args.reset_per_task:
        reset_per_task = True
    replay_add_on_success = True
    if args.no_replay_add_on_success:
        replay_add_on_success = False
    if args.replay_add_on_success:
        replay_add_on_success = True
    reinforce_once_pass_only = True
    if args.no_reinforce_once_pass_only:
        reinforce_once_pass_only = False
    if args.reinforce_once_pass_only:
        reinforce_once_pass_only = True
    teacher_use_stop = True
    if args.no_teacher_use_stop:
        teacher_use_stop = False
    if args.teacher_use_stop:
        teacher_use_stop = True

    cfg = ExperimentConfig(
        base_model=args.base_model,
        judge_model=args.judge_model,
        run_name=args.run_name,
        run_group=args.run_group,
        ablation_tag=args.ablation_tag,
        method=args.method,
        tasks_path=args.tasks_path,
        feedback_budget=args.feedback_budget,
        inner_updates=args.inner_updates,
        adaptive_judge_every_updates=args.adaptive_judge_every_updates,
        max_adaptive_steps_per_feedback=args.max_adaptive_steps_per_feedback,
        teacher_resample_attempts=args.teacher_resample_attempts,
        teacher_filter_with_judge=args.teacher_filter_with_judge,
        teacher_filter_policy=args.teacher_filter_policy,
        teacher_max_tokens=args.teacher_max_tokens,
        teacher_temperature=args.teacher_temperature,
        teacher_top_p=args.teacher_top_p,
        teacher_use_stop=teacher_use_stop,
        teacher_recursive_enabled=args.teacher_recursive_enabled,
        teacher_recursive_depth=args.teacher_recursive_depth,
        teacher_recursive_chunk_chars=args.teacher_recursive_chunk_chars,
        teacher_recursive_max_chunks=args.teacher_recursive_max_chunks,
        teacher_recursive_query_tokens=args.teacher_recursive_query_tokens,
        teacher_recursive_root_tokens=args.teacher_recursive_root_tokens,
        teacher_recursive_summary_chars=args.teacher_recursive_summary_chars,
        teacher_failure_fallback=args.teacher_failure_fallback,
        judge_mode=args.judge_mode,
        judge_flip_prob=args.judge_flip_prob,
        judge_votes=args.judge_votes,
        judge_pass_threshold=args.judge_pass_threshold,
        judge_min_margin=args.judge_min_margin,
        adaptive_stop_consecutive_passes=args.adaptive_stop_consecutive_passes,
        reinforce_once_mode=args.reinforce_once_mode,
        reinforce_once_rollouts=args.reinforce_once_rollouts,
        reinforce_once_updates=args.reinforce_once_updates,
        reinforce_once_pass_only=reinforce_once_pass_only,
        adaptive_budget_scheduler=args.adaptive_budget_scheduler,
        adaptive_budget_base=args.adaptive_budget_base,
        adaptive_budget_hard=args.adaptive_budget_hard,
        adaptive_budget_chain=args.adaptive_budget_chain,
        adaptive_budget_op_chain_threshold=args.adaptive_budget_op_chain_threshold,
        verifier_budget_policy=args.verifier_budget_policy,
        verifier_budget_min_refine_steps=args.verifier_budget_min_refine_steps,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        sample_every=args.sample_every,
        reset_per_task=reset_per_task,
        reset_every_n_tasks=args.reset_every_n_tasks,
        replay_buffer_max=args.replay_buffer_max,
        replay_per_update=args.replay_per_update,
        replay_add_on_success=replay_add_on_success,
        test_timeout_s=args.test_timeout_s,
        max_steps=args.max_steps,
        seed=args.seed,
        sampling=SamplingConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )
    return cfg


def make_run_dir(run_group: str, run_name: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", run_group, f"{ts}-{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join("runs", "latest")
    if os.path.islink(latest) or os.path.exists(latest):
        try:
            os.remove(latest)
        except OSError:
            pass
    try:
        os.symlink(os.path.abspath(run_dir), latest)
    except OSError:
        pass
    return run_dir


def main() -> None:
    if not os.getenv("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set")
    cfg = parse_args()
    run_dir = make_run_dir(cfg.run_group, cfg.run_name)
    runner = ExperimentRunner(cfg, run_dir)
    runner.run()


if __name__ == "__main__":
    main()

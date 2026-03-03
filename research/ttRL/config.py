from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SamplingConfig:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Optional[List[str]] = field(
        default_factory=lambda: [
            "\n\ndef ",
            "\n\nclass ",
            "\nif __name__",
            "\n\n# Test",
        ]
    )


@dataclass
class ExperimentConfig:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    judge_model: Optional[str] = None
    run_name: str = "run"
    run_group: str = "default"
    ablation_tag: str = "none"
    method: str = "adaptive_fwb"
    tasks_path: str = "data/tasks_min.jsonl"
    feedback_budget: int = 4
    inner_updates: int = 2
    adaptive_judge_every_updates: int = 1
    max_adaptive_steps_per_feedback: int = 8
    teacher_resample_attempts: int = 3
    teacher_best_of_n: int = 1
    teacher_min_op_coverage: float = 0.0
    teacher_rerank_coverage_weight: float = 1.0
    teacher_rerank_syntax_bonus: float = 0.25
    teacher_rerank_banned_penalty: float = 1.0
    teacher_reject_banned_calls: bool = True
    teacher_filter_with_judge: bool = False
    # pass_only: require teacher to fully pass verifier.
    # pass_or_assertion: also allow assertion-only failures (reject hard runtime/syntax failures).
    teacher_filter_policy: str = "pass_only"
    teacher_max_tokens: Optional[int] = None
    teacher_temperature: Optional[float] = None
    teacher_top_p: Optional[float] = None
    teacher_use_stop: bool = True
    # RLM-style recursive context analysis before teacher sampling.
    teacher_recursive_enabled: bool = False
    teacher_recursive_depth: int = 1
    teacher_recursive_chunk_chars: int = 2400
    teacher_recursive_max_chunks: int = 6
    teacher_recursive_query_tokens: int = 128
    teacher_recursive_root_tokens: int = 192
    teacher_recursive_summary_chars: int = 1200
    # Fallback when all teacher samples are invalid.
    # base: sample from pi(y|x); feedback: sample from pi(y|x,f).
    teacher_failure_fallback: str = "base"
    # llm: LLM judge decides pass/fail
    # tests: verifier provides rich feedback and consumes feedback budget
    # oracle_binary: verifier gives pass/fail only and does NOT consume feedback budget
    judge_mode: str = "llm"
    judge_flip_prob: float = 0.0
    # Number of independent judge samples for uncertainty-aware voting.
    judge_votes: int = 1
    # PASS requires pass_rate >= threshold across judge votes.
    judge_pass_threshold: float = 1.0
    # If vote margin < min_margin, treat as uncertain and do not accept stop.
    judge_min_margin: float = 0.0
    # Require this many consecutive judged PASS outcomes before adaptive stop.
    adaptive_stop_consecutive_passes: int = 1
    # Optional post-loop reinforcement-style update:
    # off: disabled
    # always: run after every task
    # fail_only: run only if Learn-Twice did not already solve the task
    reinforce_once_mode: str = "off"
    reinforce_once_rollouts: int = 4
    reinforce_once_updates: int = 1
    # If true, only train from PASS candidates in reinforce-once stage.
    reinforce_once_pass_only: bool = True
    adaptive_budget_scheduler: bool = False
    adaptive_budget_base: int = 1
    adaptive_budget_hard: int = 2
    adaptive_budget_chain: int = 4
    adaptive_budget_op_chain_threshold: float = 0.5
    # Drop-in verifier-budget policy controller.
    # legacy: current scheduler behavior
    # vbc_v1: policy object with explicit VERIFY/REFINE/STOP actions
    verifier_budget_policy: str = "legacy"
    verifier_budget_min_refine_steps: int = 1
    reset_per_task: bool = True
    # If >0 and reset_per_task=False, reset adapter every N tasks.
    reset_every_n_tasks: int = 0
    max_steps: int = 999999
    lr: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    sample_every: int = 1
    # Replay regularization to reduce cross-task drift.
    replay_buffer_max: int = 0
    replay_per_update: int = 0
    replay_add_on_success: bool = True
    seed: int = 0
    test_timeout_s: int = 5
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    judge_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig(max_tokens=128, temperature=0.0, top_p=1.0))

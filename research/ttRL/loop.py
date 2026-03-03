from __future__ import annotations

import ast
import random
import re
from typing import Optional

from .config import ExperimentConfig, SamplingConfig
from .data import Task, load_tasks
from .eval import run_tests
from .feedback import normalize_completion
from .logging import ExperimentLogger
from .prompts import base_prompt, feedback_prompt, judge_prompt
from .tinker_api import TinkerRunner
from .verifier_budget_policy import VerifierBudgetPolicy

_CALL_PATTERN = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
_BINOP_TOKEN = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.MatMult: "matmul",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
}
_BANNED_CALL_NAMES = {"model", "Model", "get_inputs", "get_init_inputs"}
_HARD_FAILURE_TOKENS = (
    "shape mismatch",
    "size mismatch",
    "dimension out of range",
    "mat1 and mat2 shapes cannot be multiplied",
    "runtimeerror",
    "typeerror",
    "indexerror",
    "attributeerror",
    "nameerror",
    "banned_call",
)


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        self.logger = ExperimentLogger(run_dir, cfg)
        random.seed(cfg.seed)
        self.global_step = 0
        self.tinker: Optional[TinkerRunner] = None
        self._replay_buffer: list[tuple[str, str]] = []
        self._required_ops_cache: dict[str, list[str]] = {}
        self._recursive_hint_cache: dict[tuple[str, str], str] = {}
        self._budget_policy = VerifierBudgetPolicy(
            mode=str(self.cfg.verifier_budget_policy or "legacy"),
            scheduler_enabled=bool(self.cfg.adaptive_budget_scheduler),
            base_budget=int(self.cfg.adaptive_budget_base),
            hard_budget=int(self.cfg.adaptive_budget_hard),
            chain_budget=int(self.cfg.adaptive_budget_chain),
            op_chain_threshold=float(self.cfg.adaptive_budget_op_chain_threshold),
            min_refine_steps_before_verify=int(
                self.cfg.verifier_budget_min_refine_steps
            ),
        )

    def _get_runner(self) -> TinkerRunner:
        if self.tinker is None:
            self.tinker = TinkerRunner(self.cfg)
        return self.tinker

    def _reset_runner(self) -> None:
        self.tinker = TinkerRunner(self.cfg)

    def run(self) -> None:
        tasks = load_tasks(self.cfg.tasks_path)
        for idx, task in enumerate(tasks):
            should_reset = self.cfg.reset_per_task or self.tinker is None
            if (
                not self.cfg.reset_per_task
                and self.cfg.reset_every_n_tasks > 0
                and idx > 0
                and idx % self.cfg.reset_every_n_tasks == 0
            ):
                should_reset = True
                self.logger.log(
                    "reset_schedule",
                    {
                        "task_index": idx,
                        "reset_every_n_tasks": self.cfg.reset_every_n_tasks,
                    },
                )
            if should_reset:
                self._reset_runner()
            self._log_task_start(task)
            if self.cfg.method in {"adaptive_fwb", "adaptive_fwb_rlm"}:
                if self.cfg.method == "adaptive_fwb_rlm":
                    self.cfg.teacher_recursive_enabled = True
                self._run_task_adaptive_fwb(task)
            elif self.cfg.method == "fixed_k_fwb":
                self._run_task_fixed_k(task)
            elif self.cfg.method == "fixed_k_judge":
                self._run_task_fixed_k_judge(task)
            elif self.cfg.method == "adaptive_resample":
                self._run_task_adaptive_resample(task)
            elif self.cfg.method == "inference_only":
                self._run_task_inference_only(task)
            elif self.cfg.method == "resample_only":
                self._run_task_resample_only(task)
            else:
                raise ValueError(f"Unknown method: {self.cfg.method}")
            if idx + 1 >= self.cfg.max_steps:
                break

    def _log_task_start(self, task: Task) -> None:
        self._train_steps = 0
        self._samples_student = 0
        self._samples_teacher = 0
        self._judge_calls = 0
        self._test_calls = 0
        self._reinforce_tests = 0
        self._reinforce_trains = 0
        self.logger.log(
            "task_start",
            {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "test": task.test,
                "entry_point": task.entry_point,
            },
        )

    def _sample(self, prompt: str) -> str:
        runner = self._get_runner()
        self._samples_student += 1
        return runner.sample_one(prompt, self.cfg.sampling)

    def _sample_teacher(self, prompt: str) -> str:
        runner = self._get_runner()
        self._samples_teacher += 1
        return runner.sample_one(prompt, self._teacher_sampling_cfg())

    def _sample_teacher_with_cfg(self, prompt: str, sampling_cfg: SamplingConfig) -> str:
        runner = self._get_runner()
        self._samples_teacher += 1
        return runner.sample_one(prompt, sampling_cfg)

    def _teacher_sampling_cfg(self) -> SamplingConfig:
        if (
            self.cfg.teacher_max_tokens is None
            and self.cfg.teacher_temperature is None
            and self.cfg.teacher_top_p is None
            and self.cfg.teacher_use_stop
        ):
            return self.cfg.sampling
        stop_cfg = (
            list(self.cfg.sampling.stop)
            if self.cfg.teacher_use_stop and self.cfg.sampling.stop
            else None
        )
        return SamplingConfig(
            max_tokens=(
                int(self.cfg.teacher_max_tokens)
                if self.cfg.teacher_max_tokens is not None
                else self.cfg.sampling.max_tokens
            ),
            temperature=(
                float(self.cfg.teacher_temperature)
                if self.cfg.teacher_temperature is not None
                else self.cfg.sampling.temperature
            ),
            top_p=(
                float(self.cfg.teacher_top_p)
                if self.cfg.teacher_top_p is not None
                else self.cfg.sampling.top_p
            ),
            stop=stop_cfg,
        )

    def _recursive_sampling_cfg(self, max_tokens: int) -> SamplingConfig:
        base = self._teacher_sampling_cfg()
        return SamplingConfig(
            max_tokens=max(16, int(max_tokens)),
            temperature=base.temperature,
            top_p=base.top_p,
            stop=base.stop,
        )

    def _split_text_chunks(self, text: str, chunk_chars: int, max_chunks: int) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        out: list[str] = []
        i = 0
        size = max(400, int(chunk_chars))
        limit = max(1, int(max_chunks))
        while i < len(cleaned) and len(out) < limit:
            end = min(len(cleaned), i + size)
            if end < len(cleaned):
                cut = cleaned.rfind("\n", i, end)
                if cut <= i:
                    cut = end
            else:
                cut = end
            chunk = cleaned[i:cut].strip()
            if chunk:
                out.append(chunk)
            if cut <= i:
                break
            i = cut
        return out

    def _recursive_context_hints(
        self,
        task: Task,
        prompt: str,
        feedback: str,
        attempt: int,
    ) -> str:
        cache_key = (task.task_id, feedback)
        cached = self._recursive_hint_cache.get(cache_key)
        if cached is not None:
            return cached

        depth = max(1, int(self.cfg.teacher_recursive_depth))
        chunk_chars = max(400, int(self.cfg.teacher_recursive_chunk_chars))
        max_chunks = max(1, int(self.cfg.teacher_recursive_max_chunks))
        query_cfg = self._recursive_sampling_cfg(
            int(self.cfg.teacher_recursive_query_tokens)
        )
        root_cfg = self._recursive_sampling_cfg(
            int(self.cfg.teacher_recursive_root_tokens)
        )
        context = task.prompt.strip()
        if not context:
            self._recursive_hint_cache[cache_key] = ""
            return ""

        current = context
        notes: list[str] = []
        for d in range(depth):
            chunks = self._split_text_chunks(current, chunk_chars, max_chunks)
            if not chunks:
                break
            layer_notes: list[str] = []
            for idx, chunk in enumerate(chunks):
                q = (
                    "You are assisting a Python code fix under limited context.\n"
                    "From the task chunk and failure feedback, extract only cues that are likely useful "
                    "to repair the function body.\n"
                    "Return up to 5 short bullet lines with: operators, tensor-shape hints, failure causes, "
                    "and critical variable names. No prose.\n\n"
                    f"Chunk index: {idx}\n"
                    f"Failure feedback:\n{feedback}\n\n"
                    f"Task chunk:\n{chunk}\n"
                )
                note_raw = self._sample_teacher_with_cfg(q, query_cfg)
                note = note_raw.strip()
                if note:
                    layer_notes.append(note)
                    notes.append(note)
                self.logger.log(
                    "rlm_recursive_note",
                    {
                        "task_id": task.task_id,
                        "attempt": attempt,
                        "depth": d + 1,
                        "chunk_index": idx,
                        "chunk_chars": len(chunk),
                        "note": note,
                    },
                )
            if not layer_notes:
                break
            current = "\n".join(layer_notes)

        if not notes:
            self._recursive_hint_cache[cache_key] = ""
            return ""

        root_prompt = (
            "You are a root planner for recursive context analysis.\n"
            "Compress the notes into a compact patch guide for fixing a Python function body.\n"
            "Return plain text with 3 sections:\n"
            "OPS: comma-separated operators/calls\n"
            "BUGS: concise failure causes\n"
            "PATCH: concrete edit hints\n\n"
            f"Original task prompt:\n{prompt}\n\n"
            f"Failure feedback:\n{feedback}\n\n"
            f"Recursive notes:\n{chr(10).join(notes)}\n"
        )
        summary = self._sample_teacher_with_cfg(root_prompt, root_cfg).strip()
        if int(self.cfg.teacher_recursive_summary_chars) > 0:
            summary = summary[: int(self.cfg.teacher_recursive_summary_chars)].strip()
        self.logger.log(
            "rlm_recursive_summary",
            {
                "task_id": task.task_id,
                "attempt": attempt,
                "depth": depth,
                "notes_count": len(notes),
                "summary": summary,
            },
        )
        self._recursive_hint_cache[cache_key] = summary
        return summary

    def _sample_judge(self, prompt: str) -> str:
        runner = self._get_runner()
        self._judge_calls += 1
        return runner.sample_one_judge(prompt, self.cfg.judge_sampling)

    def _train_pair(self, prompt: str, completion: str) -> None:
        runner = self._get_runner()
        pairs = [(prompt, completion)]
        replay_k = max(0, int(self.cfg.replay_per_update))
        if replay_k > 0 and self._replay_buffer:
            replay_k = min(replay_k, len(self._replay_buffer))
            pairs.extend(random.sample(self._replay_buffer, replay_k))
            self.logger.log(
                "replay_train",
                {
                    "task_id": getattr(self, "_task", None).task_id
                    if getattr(self, "_task", None) is not None
                    else None,
                    "replay_pairs": replay_k,
                    "buffer_size": len(self._replay_buffer),
                },
            )
        runner.train_on_pairs(pairs)
        self.global_step += 1
        self._train_steps += 1
        if self.global_step % self.cfg.sample_every == 0:
            runner.refresh_sampling_client(name=f"step-{self.global_step}")

    def _add_replay_pair(self, task: Task, prompt: str, completion: str, success: bool) -> None:
        if not success or not self.cfg.replay_add_on_success:
            return
        if self.cfg.replay_buffer_max <= 0:
            return
        clean = normalize_completion(completion, entry_point=task.entry_point).strip()
        if not clean:
            return
        self._replay_buffer.append((prompt, clean))
        if len(self._replay_buffer) > self.cfg.replay_buffer_max:
            overflow = len(self._replay_buffer) - self.cfg.replay_buffer_max
            if overflow > 0:
                self._replay_buffer = self._replay_buffer[overflow:]
        self.logger.log(
            "replay_add",
            {
                "task_id": task.task_id,
                "buffer_size": len(self._replay_buffer),
                "replay_buffer_max": self.cfg.replay_buffer_max,
            },
        )

    def _run_reinforce_once(
        self,
        task: Task,
        prompt: str,
        success: bool,
        success_completion: str,
    ) -> tuple[bool, str]:
        mode = str(self.cfg.reinforce_once_mode or "off").lower()
        if mode not in {"always", "fail_only"}:
            return success, success_completion
        if mode == "fail_only" and success:
            return success, success_completion

        rollouts = max(1, int(self.cfg.reinforce_once_rollouts))
        updates = max(0, int(self.cfg.reinforce_once_updates))
        pass_only = bool(self.cfg.reinforce_once_pass_only)

        self.logger.log(
            "reinforce_once_start",
            {
                "task_id": task.task_id,
                "mode": mode,
                "rollouts": rollouts,
                "updates": updates,
                "pass_only": pass_only,
                "pre_success": success,
            },
        )

        best_completion = ""
        best_passed = False
        best_score = -1.0
        for r in range(rollouts):
            candidate_raw = self._sample(prompt)
            candidate = normalize_completion(candidate_raw, entry_point=task.entry_point)
            result = run_tests(task, candidate, timeout_s=self.cfg.test_timeout_s)
            self._test_calls += 1
            self._reinforce_tests += 1
            score = 1.0 if result.passed else 0.0
            self.logger.log(
                "reinforce_once_candidate",
                {
                    "task_id": task.task_id,
                    "rollout_idx": r,
                    "passed": result.passed,
                    "score": score,
                    "feedback": result.feedback,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "text": candidate_raw,
                    "normalized": candidate,
                },
            )
            if score > best_score:
                best_score = score
                best_passed = bool(result.passed)
                best_completion = candidate
            # Early exit as soon as we have a positive reward sample.
            if pass_only and result.passed:
                break

        if not best_completion:
            self.logger.log(
                "reinforce_once_skip",
                {
                    "task_id": task.task_id,
                    "reason": "no_candidate",
                    "mode": mode,
                },
            )
            return success, success_completion

        if pass_only and not best_passed:
            self.logger.log(
                "reinforce_once_skip",
                {
                    "task_id": task.task_id,
                    "reason": "no_pass_candidate",
                    "mode": mode,
                },
            )
            return success, success_completion

        for u in range(updates):
            self._train_pair(prompt, best_completion)
            self._reinforce_trains += 1
            self.logger.log(
                "reinforce_once_train",
                {
                    "task_id": task.task_id,
                    "update_idx": u,
                    "step": self.global_step,
                    "passed_candidate": best_passed,
                },
            )

        if best_passed:
            success = True
            success_completion = best_completion
        return success, success_completion

    def _reference_lines(self, task: Task) -> list[str]:
        lines = task.prompt.splitlines()
        in_ref = False
        out: list[str] = []
        for raw in lines:
            stripped = raw.strip()
            if "Reference forward body" in stripped:
                in_ref = True
                continue
            if not in_ref:
                continue
            if stripped.startswith("def "):
                break
            if not stripped.startswith("#"):
                continue
            body = stripped[1:].strip()
            if not body:
                continue
            low = body.lower()
            if body in {'"""', "'''"} or low.startswith(('"""', "'''")):
                continue
            if low.startswith(":param") or low.startswith(":return"):
                continue
            if low.startswith("args:") or low.startswith("returns:"):
                continue
            out.append(body)
        return out

    def _ops_from_line(self, line: str) -> list[str]:
        tokens: list[str] = []
        stripped = line.strip()
        if not stripped:
            return tokens
        low = stripped.lower()

        if low.startswith("for "):
            tokens.append("for")
        elif low.startswith("while "):
            tokens.append("while")
        elif low.startswith("if "):
            tokens.append("if")
        elif low.startswith("elif "):
            tokens.append("elif")
        elif low.startswith("try:"):
            tokens.append("try")
        elif low.startswith("except"):
            tokens.append("except")
        elif low.startswith("with "):
            tokens.append("with")

        try:
            parsed = ast.parse(stripped)
        except SyntaxError:
            parsed = None

        if parsed is not None:
            class _V(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.ops: list[str] = []

                def _call_name(self, node: ast.AST) -> str:
                    if isinstance(node, ast.Name):
                        return node.id
                    if isinstance(node, ast.Attribute):
                        base = self._call_name(node.value)
                        return f"{base}.{node.attr}" if base else node.attr
                    return ""

                def visit_Call(self, node: ast.Call) -> None:
                    name = self._call_name(node.func)
                    if name:
                        self.ops.append(name.lower())
                    self.generic_visit(node)

                def visit_BinOp(self, node: ast.BinOp) -> None:
                    token = _BINOP_TOKEN.get(type(node.op))
                    if token:
                        self.ops.append(token)
                    self.visit(node.left)
                    self.visit(node.right)

                def visit_AugAssign(self, node: ast.AugAssign) -> None:
                    token = _BINOP_TOKEN.get(type(node.op))
                    if token:
                        self.ops.append(token)
                    self.visit(node.value)

            visitor = _V()
            visitor.visit(parsed)
            tokens.extend(visitor.ops)
        else:
            for match in _CALL_PATTERN.finditer(stripped):
                name = match.group(1).lower()
                if name:
                    tokens.append(name)

        return tokens

    def _required_ops(self, task: Task) -> list[str]:
        cached = self._required_ops_cache.get(task.task_id)
        if cached is not None:
            return cached
        lines = self._reference_lines(task)
        ops: list[str] = []
        for line in lines:
            ops.extend(self._ops_from_line(line))
        self._required_ops_cache[task.task_id] = ops
        return ops

    def _candidate_ops(self, completion: str) -> list[str]:
        ops: list[str] = []
        for raw in completion.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ops.extend(self._ops_from_line(line))
        return ops

    def _op_coverage(
        self, task: Task, completion: str
    ) -> tuple[float, int, int, list[str], list[str]]:
        required = self._required_ops(task)
        candidate = self._candidate_ops(completion)
        if not required:
            return 1.0, 0, 0, required, candidate
        cursor = 0
        matched = 0
        for op in required:
            found_at = -1
            for idx in range(cursor, len(candidate)):
                if candidate[idx] == op:
                    found_at = idx
                    break
            if found_at >= 0:
                matched += 1
                cursor = found_at + 1
        coverage = matched / float(len(required))
        return coverage, matched, len(required), required, candidate

    def _first_banned_call(self, completion: str) -> str:
        try:
            tree = ast.parse(f"def _cand(model, *args, **kwargs):\n{completion}\n")
        except SyntaxError:
            return ""
        banned = ""

        class _V(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                nonlocal banned
                if banned:
                    return
                if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALL_NAMES:
                    banned = node.func.id
                    return
                self.generic_visit(node)

        _V().visit(tree)
        return banned

    def _classify_failure(self, task: Task, feedback: str, candidate: str) -> str:
        low = (feedback or "").lower()
        if any(tok in low for tok in _HARD_FAILURE_TOKENS):
            return "hard_runtime"
        cov, _m, _n, _req, _cand = self._op_coverage(task, candidate)
        threshold = float(self._budget_policy.op_chain_threshold)
        if cov < threshold:
            return "op_chain"
        return "soft"

    def _initial_budget_limit(self) -> int:
        return self._budget_policy.initial_limit(
            max_budget=int(self.cfg.feedback_budget)
        )

    def _maybe_escalate_budget(
        self,
        task: Task,
        current_limit: int,
        feedback: str,
        candidate: str,
    ) -> int:
        max_budget = int(self.cfg.feedback_budget)

        kind = self._classify_failure(task, feedback, candidate)
        target, reason = self._budget_policy.maybe_escalate(
            current_limit=current_limit,
            max_budget=max_budget,
            failure_kind=kind,
        )
        if target != current_limit:
            self.logger.log(
                "budget_escalate",
                {
                    "task_id": task.task_id,
                    "from_budget": current_limit,
                    "to_budget": target,
                    "max_budget": max_budget,
                    "reason": reason,
                    "policy": self._budget_policy.mode,
                },
            )
        return target

    def _is_valid_teacher(self, task: Task, completion: str) -> bool:
        if not completion.strip():
            return False
        if completion.strip() == "pass":
            return False
        if completion.strip() == "return":
            return False
        if "PASS" in completion or "FAIL" in completion:
            return False
        low = completion.lower()
        if "corrected function body" in low or "step 1" in low:
            return False
        lines = [ln.strip() for ln in completion.splitlines() if ln.strip()]
        if len(lines) > 20:
            return False
        if len(lines) >= 4:
            # Only apply the duplicate-line filter on longer samples; many correct
            # teacher bodies are legitimately one-liners.
            uniq = set(lines)
            if len(uniq) <= max(1, len(lines) // 4):
                return False
        return_count = sum(1 for ln in lines if ln.startswith("return"))
        if return_count > 6:
            return False
        try:
            ast.parse(f"{task.prompt}\n{completion}\n")
        except SyntaxError:
            return False
        return True

    def _sample_teacher_completion(
        self, task: Task, prompt: str, feedback: str, attempt: int
    ) -> tuple[str, str] | None:
        best_of_n = max(1, int(self.cfg.teacher_best_of_n))
        min_cov = max(0.0, float(self.cfg.teacher_min_op_coverage))
        teacher_prompt = feedback_prompt(prompt, feedback)
        if self.cfg.teacher_recursive_enabled:
            hints = self._recursive_context_hints(
                task=task,
                prompt=prompt,
                feedback=feedback,
                attempt=attempt,
            )
            if hints:
                teacher_prompt = (
                    f"{teacher_prompt}\n\n"
                    "Recursive context hints (RLM-style):\n"
                    f"{hints}\n\n"
                    "Use the hints only if they help satisfy tests.\n"
                    "Corrected function body:"
                )
        for teacher_try in range(self.cfg.teacher_resample_attempts):
            best: tuple[float, str, str, float, int, int, str] | None = None
            for rank in range(best_of_n):
                teacher_raw = self._sample_teacher(teacher_prompt)
                teacher = normalize_completion(teacher_raw, entry_point=task.entry_point)
                valid = self._is_valid_teacher(task, teacher)
                coverage, matched, total, _required_ops, candidate_ops = self._op_coverage(
                    task, teacher
                )
                banned_call = self._first_banned_call(teacher)
                banned = bool(banned_call)
                syntax_bonus = (
                    float(self.cfg.teacher_rerank_syntax_bonus) if valid else 0.0
                )
                score = (
                    float(self.cfg.teacher_rerank_coverage_weight) * coverage
                    + syntax_bonus
                    - float(self.cfg.teacher_rerank_banned_penalty) * (1.0 if banned else 0.0)
                )
                self.logger.log(
                    "teacher_candidate",
                    {
                        "task_id": task.task_id,
                        "attempt": attempt,
                        "teacher_try": teacher_try,
                        "candidate_rank": rank,
                        "valid": valid,
                        "op_coverage": coverage,
                        "op_matched": matched,
                        "op_total": total,
                        "op_required_preview": _required_ops[:16],
                        "op_candidate_preview": candidate_ops[:16],
                        "banned_call": banned_call,
                        "score": score,
                        "text": teacher_raw,
                        "normalized": teacher,
                        "feedback": feedback,
                    },
                )
                if not valid:
                    continue
                cand = (score, teacher_raw, teacher, coverage, matched, total, banned_call)
                if best is None or cand[0] > best[0]:
                    best = cand

            if best is None:
                self.logger.log(
                    "teacher_reject",
                    {
                        "task_id": task.task_id,
                        "attempt": attempt,
                        "teacher_try": teacher_try,
                        "reason": "invalid_teacher",
                        "feedback": feedback,
                    },
                )
                continue

            score, teacher_raw, teacher, coverage, matched, total, banned_call = best
            if self.cfg.teacher_reject_banned_calls and banned_call:
                self.logger.log(
                    "teacher_reject",
                    {
                        "task_id": task.task_id,
                        "attempt": attempt,
                        "teacher_try": teacher_try,
                        "reason": "teacher_banned_call",
                        "banned_call": banned_call,
                        "score": score,
                        "op_coverage": coverage,
                        "op_matched": matched,
                        "op_total": total,
                        "text": teacher_raw,
                        "normalized": teacher,
                        "feedback": feedback,
                    },
                )
                continue

            if coverage < min_cov:
                self.logger.log(
                    "teacher_reject",
                    {
                        "task_id": task.task_id,
                        "attempt": attempt,
                        "teacher_try": teacher_try,
                        "reason": "teacher_low_op_coverage",
                        "op_coverage": coverage,
                        "op_matched": matched,
                        "op_total": total,
                        "min_required": min_cov,
                        "score": score,
                        "text": teacher_raw,
                        "normalized": teacher,
                        "feedback": feedback,
                    },
                )
                continue

            if self.cfg.teacher_filter_with_judge:
                if self.cfg.judge_mode == "llm":
                    j_prompt = judge_prompt(prompt, feedback, teacher)
                    j_out = self._sample_judge(j_prompt).strip().upper()
                    labels = re.findall(r"\b(PASS|FAIL)\b", j_out)
                    teacher_ok = bool(labels) and labels[-1] == "PASS"
                    if not teacher_ok:
                        self.logger.log(
                            "teacher_reject",
                            {
                                "task_id": task.task_id,
                                "attempt": attempt,
                                "teacher_try": teacher_try,
                                "reason": "teacher_judge_fail",
                                "text": teacher_raw,
                                "normalized": teacher,
                                "feedback": feedback,
                                "judge_raw": j_out,
                                "score": score,
                                "op_coverage": coverage,
                            },
                        )
                        continue
                elif self.cfg.judge_mode == "oracle_binary":
                    # Oracle teacher filtering with policy control.
                    result = run_tests(
                        task,
                        teacher,
                        timeout_s=self.cfg.test_timeout_s,
                    )
                    self._judge_calls += 1
                    self._test_calls += 1
                    teacher_ok, reject_reason = self._oracle_teacher_ok(result)
                    if not teacher_ok:
                        self.logger.log(
                            "teacher_reject",
                            {
                                "task_id": task.task_id,
                                "attempt": attempt,
                                "teacher_try": teacher_try,
                                "reason": reject_reason,
                                "text": teacher_raw,
                                "normalized": teacher,
                                "feedback": feedback,
                                "oracle_feedback": result.feedback,
                                "returncode": result.returncode,
                                "score": score,
                                "op_coverage": coverage,
                            },
                        )
                        continue

            self.logger.log(
                "sample",
                {
                    "task_id": task.task_id,
                    "role": "teacher",
                    "attempt": attempt,
                    "teacher_try": teacher_try,
                    "teacher_best_of_n": best_of_n,
                    "teacher_score": score,
                    "op_coverage": coverage,
                    "op_matched": matched,
                    "op_total": total,
                    "text": teacher_raw,
                    "normalized": teacher,
                    "feedback": feedback,
                },
            )
            return teacher_raw, teacher
        return None

    def _oracle_teacher_ok(self, result) -> tuple[bool, str]:
        if result.passed:
            return True, "teacher_oracle_pass"
        if self.cfg.teacher_filter_policy == "pass_only":
            return False, "teacher_oracle_fail"

        # pass_or_assertion: keep likely-near-miss assertion failures, reject
        # hard runtime/syntax failures that tend to poison training.
        text = f"{result.feedback}\n{result.stderr}".lower()
        hard_fail_tokens = (
            "syntaxerror",
            "indentationerror",
            "nameerror",
            "typeerror",
            "attributeerror",
            "modulenotfounderror",
            "importerror",
        )
        if any(tok in text for tok in hard_fail_tokens):
            return False, "teacher_oracle_hard_fail"
        if "timeout" in text or result.returncode == 124:
            return False, "teacher_oracle_timeout"
        if "assertionerror" in text:
            return True, "teacher_oracle_assertion"
        return False, "teacher_oracle_non_assert_fail"

    def _judge(self, prompt: str, feedback: str, candidate: str) -> tuple[bool, bool, bool]:
        if self.cfg.judge_mode == "tests":
            result = run_tests(self._task, candidate, timeout_s=self.cfg.test_timeout_s)
            self._feedback_calls += 1
            self._test_calls += 1
            self.logger.log(
                "test",
                {
                    "task_id": self._task.task_id,
                    "passed": result.passed,
                    "feedback": result.feedback,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "feedback_calls": self._feedback_calls,
                },
            )
            if result.passed:
                self._last_feedback = ""
            else:
                self._last_feedback = result.feedback
            self.logger.log(
                "judge_decision",
                {
                    "task_id": self._task.task_id,
                    "source": "tests",
                    "decision": "PASS" if result.passed else "FAIL",
                    "verified": True,
                    "uncertain": False,
                    "flipped": False,
                    "judge_flip_prob": 0.0,
                },
            )
            return result.passed, True, False

        if self.cfg.judge_mode == "oracle_binary":
            # Cheap binary judge signal from verifier.
            # Optional flip noise lets us stress-test stopping robustness while
            # still tracking true success via explicit verifier checks on PASS.
            result = run_tests(self._task, candidate, timeout_s=self.cfg.test_timeout_s)
            self._judge_calls += 1
            self._test_calls += 1
            raw_decision = bool(result.passed)
            decision = raw_decision
            flipped = False
            if self.cfg.judge_flip_prob > 0.0 and random.random() < self.cfg.judge_flip_prob:
                decision = not decision
                flipped = True
            self.logger.log(
                "judge_oracle",
                {
                    "task_id": self._task.task_id,
                    "decision": "PASS" if decision else "FAIL",
                    "raw_decision": "PASS" if raw_decision else "FAIL",
                    "flipped": flipped,
                    "judge_flip_prob": self.cfg.judge_flip_prob,
                    "feedback": result.feedback,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
            )
            # Under synthetic flip noise, we force PASS verification in caller
            # (verified=False) so aggregate success is still grounded in tests.
            verified = self.cfg.judge_flip_prob <= 0.0
            self.logger.log(
                "judge_decision",
                {
                    "task_id": self._task.task_id,
                    "source": "oracle_binary",
                    "decision": "PASS" if decision else "FAIL",
                    "verified": verified,
                    "uncertain": False,
                    "flipped": flipped,
                    "judge_flip_prob": self.cfg.judge_flip_prob,
                },
            )
            return decision, verified, False

        j_prompt = judge_prompt(prompt, feedback, candidate)
        votes = max(1, int(self.cfg.judge_votes))
        pass_votes = 0
        raw_pass_votes = 0
        flipped_votes = 0
        raw_outputs = []
        for _ in range(votes):
            j_out = self._sample_judge(j_prompt).strip().upper()
            raw_outputs.append(j_out)
            labels = re.findall(r"\b(PASS|FAIL)\b", j_out)
            raw_decision = bool(labels) and labels[-1] == "PASS"
            if raw_decision:
                raw_pass_votes += 1
            decision = raw_decision
            if self.cfg.judge_flip_prob > 0.0 and random.random() < self.cfg.judge_flip_prob:
                decision = not decision
                flipped_votes += 1
            if decision:
                pass_votes += 1
        pass_rate = pass_votes / float(votes)
        margin = abs(2 * pass_votes - votes) / float(votes)
        uncertain = margin < float(self.cfg.judge_min_margin)
        decision = pass_rate >= float(self.cfg.judge_pass_threshold) and not uncertain
        self.logger.log(
            "judge",
            {
                "task_id": self._task.task_id,
                "decision": "PASS" if decision else "FAIL",
                "raw_pass_votes": raw_pass_votes,
                "pass_votes": pass_votes,
                "votes": votes,
                "pass_rate": pass_rate,
                "margin": margin,
                "uncertain": uncertain,
                "judge_pass_threshold": self.cfg.judge_pass_threshold,
                "judge_min_margin": self.cfg.judge_min_margin,
                "flipped_votes": flipped_votes,
                "judge_flip_prob": self.cfg.judge_flip_prob,
                "raw": raw_outputs[-1] if raw_outputs else "",
            },
        )
        self.logger.log(
            "judge_decision",
            {
                "task_id": self._task.task_id,
                "source": "llm",
                "decision": "PASS" if decision else "FAIL",
                "verified": False,
                "uncertain": uncertain,
                "flipped": flipped_votes > 0,
                "judge_flip_prob": self.cfg.judge_flip_prob,
            },
        )
        return decision, False, uncertain

    def _run_task_adaptive_fwb(self, task: Task) -> None:
        self._task = task
        self._feedback_calls = 0
        self._last_feedback = ""
        prompt = base_prompt(task.prompt)
        attempts = 0
        success = False
        success_completion = ""
        consecutive_passes = 0
        required_passes = max(1, int(self.cfg.adaptive_stop_consecutive_passes))
        updates_per_cycle = max(1, self.cfg.inner_updates)
        judge_every_updates = max(1, self.cfg.adaptive_judge_every_updates)
        budget_limit = self._initial_budget_limit()
        self.logger.log(
            "budget_start",
            {
                "task_id": task.task_id,
                "initial_budget_limit": budget_limit,
                "max_feedback_budget": int(self.cfg.feedback_budget),
                "scheduler_enabled": bool(self.cfg.adaptive_budget_scheduler),
                "policy": self._budget_policy.mode,
            },
        )

        while True:
            if attempts == 0 and not self._last_feedback:
                if self._feedback_calls >= budget_limit:
                    break
                action = self._budget_policy.decide_action(
                    budget_used=self._feedback_calls,
                    budget_limit=budget_limit,
                    has_feedback=False,
                    steps_since_feedback=0,
                    max_steps_per_feedback=int(self.cfg.max_adaptive_steps_per_feedback),
                    failure_kind="soft",
                )
                self.logger.log(
                    "budget_action",
                    {
                        "task_id": task.task_id,
                        "stage": "initial",
                        "action": action.value,
                        "budget_used": self._feedback_calls,
                        "budget_limit": budget_limit,
                        "policy": self._budget_policy.mode,
                    },
                )
                if action.value == "stop":
                    break
                completion = self._sample(prompt)
                self.logger.log(
                    "sample",
                    {
                        "task_id": task.task_id,
                        "role": "student",
                        "attempt": attempts,
                        "text": completion,
                    },
                )
                result = run_tests(task, completion, timeout_s=self.cfg.test_timeout_s)
                self._feedback_calls += 1
                self._test_calls += 1
                self.logger.log(
                    "test",
                    {
                        "task_id": task.task_id,
                        "passed": result.passed,
                        "feedback": result.feedback,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                        "feedback_calls": self._feedback_calls,
                    },
                )
                if result.passed:
                    success = True
                    success_completion = normalize_completion(
                        completion, entry_point=task.entry_point
                    )
                    break
                completion_norm = normalize_completion(completion, entry_point=task.entry_point)
                budget_limit = self._maybe_escalate_budget(
                    task, budget_limit, result.feedback, completion_norm
                )
                self._last_feedback = result.feedback

            if not self._last_feedback:
                break

            steps_since_feedback = 0
            last_student = ""
            stop_cycle = False
            while not stop_cycle and not success:
                for _ in range(updates_per_cycle):
                    if (
                        steps_since_feedback
                        >= self.cfg.max_adaptive_steps_per_feedback
                    ):
                        stop_cycle = True
                        break

                    teacher_out = self._sample_teacher_completion(
                        task=task,
                        prompt=prompt,
                        feedback=self._last_feedback,
                        attempt=attempts,
                    )
                    if teacher_out is None:
                        if self._feedback_calls >= budget_limit:
                            self.logger.log(
                                "train_skip",
                                {
                                    "task_id": task.task_id,
                                    "attempt": attempts,
                                    "reason": "no_valid_teacher_no_budget",
                                    "fallback": self.cfg.teacher_failure_fallback,
                                },
                            )
                            stop_cycle = True
                            break
                        # If all teacher attempts are invalid, refresh feedback directly.
                        fallback_mode = self.cfg.teacher_failure_fallback
                        fallback_prompt = prompt
                        if fallback_mode == "feedback":
                            fallback_prompt = feedback_prompt(prompt, self._last_feedback)
                        self.logger.log(
                            "train_skip",
                            {
                                "task_id": task.task_id,
                                "attempt": attempts,
                                "reason": "no_valid_teacher",
                                "fallback": fallback_mode,
                            },
                        )
                        student = self._sample(fallback_prompt)
                        last_student = student
                        self.logger.log(
                            "sample",
                            {
                                "task_id": task.task_id,
                                "role": "student",
                                "attempt": attempts,
                                "text": student,
                                "fallback": fallback_mode,
                            },
                        )
                        result = run_tests(task, student, timeout_s=self.cfg.test_timeout_s)
                        self._feedback_calls += 1
                        self._test_calls += 1
                        self.logger.log(
                            "test",
                            {
                                "task_id": task.task_id,
                                "passed": result.passed,
                                "feedback": result.feedback,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "returncode": result.returncode,
                                "feedback_calls": self._feedback_calls,
                            },
                        )
                        attempts += 1
                        if result.passed:
                            success = True
                            success_completion = normalize_completion(
                                student, entry_point=task.entry_point
                            )
                            stop_cycle = True
                            break
                        student_norm = normalize_completion(student, entry_point=task.entry_point)
                        budget_limit = self._maybe_escalate_budget(
                            task, budget_limit, result.feedback, student_norm
                        )
                        self._last_feedback = result.feedback
                        steps_since_feedback = 0
                        consecutive_passes = 0
                        stop_cycle = True
                        break

                    _, teacher = teacher_out
                    self._train_pair(prompt, teacher)
                    self.logger.log(
                        "train",
                        {
                            "task_id": task.task_id,
                            "step": self.global_step,
                        },
                    )
                    steps_since_feedback += 1

                    should_judge = (
                        steps_since_feedback % judge_every_updates == 0
                        or steps_since_feedback
                        >= self.cfg.max_adaptive_steps_per_feedback
                    )
                    if not should_judge:
                        continue

                    student = self._sample(prompt)
                    last_student = student
                    self.logger.log(
                        "sample",
                        {
                            "task_id": task.task_id,
                            "role": "student",
                            "attempt": attempts,
                            "text": student,
                        },
                    )

                    student_norm = normalize_completion(student, entry_point=task.entry_point)
                    decision, verified, _uncertain = self._judge(
                        prompt,
                        self._last_feedback,
                        student_norm,
                    )
                    attempts += 1
                    if decision:
                        consecutive_passes += 1
                    else:
                        consecutive_passes = 0
                    if decision and consecutive_passes < required_passes:
                        self.logger.log(
                            "stop_guard",
                            {
                                "task_id": task.task_id,
                                "attempt": attempts,
                                "consecutive_passes": consecutive_passes,
                                "required_passes": required_passes,
                                "reason": "insufficient_consecutive_passes",
                            },
                        )
                        continue

                    if decision and verified:
                        success = True
                        success_completion = student_norm
                        stop_cycle = True
                        break
                    if decision and not verified:
                        if self._feedback_calls >= budget_limit:
                            stop_cycle = True
                            break
                        result = run_tests(task, student, timeout_s=self.cfg.test_timeout_s)
                        self._feedback_calls += 1
                        self._test_calls += 1
                        self.logger.log(
                            "test",
                            {
                                "task_id": task.task_id,
                                "passed": result.passed,
                                "feedback": result.feedback,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "returncode": result.returncode,
                                "feedback_calls": self._feedback_calls,
                            },
                        )
                        if result.passed:
                            success = True
                            success_completion = student_norm
                            stop_cycle = True
                            break
                        budget_limit = self._maybe_escalate_budget(
                            task, budget_limit, result.feedback, student_norm
                        )
                        self._last_feedback = result.feedback
                        steps_since_feedback = 0
                        consecutive_passes = 0
                        stop_cycle = True
                        break

                if success or stop_cycle:
                    break

                if steps_since_feedback >= self.cfg.max_adaptive_steps_per_feedback:
                    # Avoid infinite judge-fail loops by forcing a verifier refresh.
                    if self._feedback_calls >= budget_limit or not last_student:
                        stop_cycle = True
                        break
                    failure_kind = self._classify_failure(
                        task,
                        self._last_feedback,
                        normalize_completion(last_student, entry_point=task.entry_point),
                    )
                    action = self._budget_policy.decide_action(
                        budget_used=self._feedback_calls,
                        budget_limit=budget_limit,
                        has_feedback=bool(self._last_feedback),
                        steps_since_feedback=steps_since_feedback,
                        max_steps_per_feedback=int(self.cfg.max_adaptive_steps_per_feedback),
                        failure_kind=failure_kind,
                    )
                    self.logger.log(
                        "budget_action",
                        {
                            "task_id": task.task_id,
                            "stage": "forced_recheck",
                            "action": action.value,
                            "failure_kind": failure_kind,
                            "budget_used": self._feedback_calls,
                            "budget_limit": budget_limit,
                            "policy": self._budget_policy.mode,
                        },
                    )
                    if action.value != "verify":
                        stop_cycle = True
                        break
                    result = run_tests(
                        task, last_student, timeout_s=self.cfg.test_timeout_s
                    )
                    self._feedback_calls += 1
                    self._test_calls += 1
                    self.logger.log(
                        "forced_recheck",
                        {
                            "task_id": task.task_id,
                            "passed": result.passed,
                            "feedback": result.feedback,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "returncode": result.returncode,
                            "feedback_calls": self._feedback_calls,
                            "steps_since_feedback": steps_since_feedback,
                        },
                    )
                    if result.passed:
                        success = True
                        success_completion = normalize_completion(
                            last_student, entry_point=task.entry_point
                        )
                        break
                    last_norm = normalize_completion(last_student, entry_point=task.entry_point)
                    budget_limit = self._maybe_escalate_budget(
                        task, budget_limit, result.feedback, last_norm
                    )
                    self._last_feedback = result.feedback
                    consecutive_passes = 0
                    stop_cycle = True

            if success or self._feedback_calls >= budget_limit:
                break

        success, success_completion = self._run_reinforce_once(
            task=task,
            prompt=prompt,
            success=success,
            success_completion=success_completion,
        )
        self._add_replay_pair(task, prompt, success_completion, success)
        self.logger.log(
            "task_done",
            {
                "task_id": task.task_id,
                "success": success,
                "attempts": attempts,
                "feedback_calls": self._feedback_calls,
                "train_steps": self._train_steps,
                "judge_calls": self._judge_calls,
                "test_calls": self._test_calls,
                "samples_student": self._samples_student,
                "samples_teacher": self._samples_teacher,
                "effective_feedback_budget": budget_limit,
                "reinforce_once_mode": self.cfg.reinforce_once_mode,
                "reinforce_tests": self._reinforce_tests,
                "reinforce_trains": self._reinforce_trains,
            },
        )

    def _run_task_fixed_k(self, task: Task) -> None:
        self._task = task
        self._feedback_calls = 0
        prompt = base_prompt(task.prompt)
        attempts = 0
        success = False
        success_completion = ""

        while self._feedback_calls < self.cfg.feedback_budget:
            completion = self._sample(prompt)
            self.logger.log(
                "sample",
                {
                    "task_id": task.task_id,
                    "role": "student",
                    "attempt": attempts,
                    "text": completion,
                },
            )
            result = run_tests(task, completion, timeout_s=self.cfg.test_timeout_s)
            self._feedback_calls += 1
            self._test_calls += 1
            self.logger.log(
                "test",
                {
                    "task_id": task.task_id,
                    "passed": result.passed,
                    "feedback": result.feedback,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "feedback_calls": self._feedback_calls,
                },
            )
            if result.passed:
                success = True
                success_completion = normalize_completion(
                    completion, entry_point=task.entry_point
                )
                break

            feedback = result.feedback
            for _ in range(self.cfg.inner_updates):
                teacher_out = self._sample_teacher_completion(
                    task=task,
                    prompt=prompt,
                    feedback=feedback,
                    attempt=attempts,
                )
                if teacher_out is None:
                    self.logger.log(
                        "train_skip",
                        {
                            "task_id": task.task_id,
                            "attempt": attempts,
                            "reason": "no_valid_teacher",
                        },
                    )
                    attempts += 1
                    continue
                _, teacher = teacher_out
                self._train_pair(prompt, teacher)
                self.logger.log(
                    "train",
                    {
                        "task_id": task.task_id,
                        "step": self.global_step,
                    },
                )
                attempts += 1

        self._add_replay_pair(task, prompt, success_completion, success)
        self.logger.log(
            "task_done",
            {
                "task_id": task.task_id,
                "success": success,
                "attempts": attempts,
                "feedback_calls": self._feedback_calls,
                "train_steps": self._train_steps,
                "judge_calls": self._judge_calls,
                "test_calls": self._test_calls,
                "samples_student": self._samples_student,
                "samples_teacher": self._samples_teacher,
            },
        )

    def _run_task_fixed_k_judge(self, task: Task) -> None:
        self._task = task
        self._feedback_calls = 0
        prompt = base_prompt(task.prompt)
        attempts = 0
        success = False
        success_completion = ""
        consecutive_passes = 0
        required_passes = max(1, int(self.cfg.adaptive_stop_consecutive_passes))

        while self._feedback_calls < self.cfg.feedback_budget and not success:
            completion = self._sample(prompt)
            self.logger.log(
                "sample",
                {
                    "task_id": task.task_id,
                    "role": "student",
                    "attempt": attempts,
                    "text": completion,
                },
            )
            result = run_tests(task, completion, timeout_s=self.cfg.test_timeout_s)
            self._feedback_calls += 1
            self._test_calls += 1
            self.logger.log(
                "test",
                {
                    "task_id": task.task_id,
                    "passed": result.passed,
                    "feedback": result.feedback,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "feedback_calls": self._feedback_calls,
                },
            )
            if result.passed:
                success = True
                success_completion = normalize_completion(
                    completion, entry_point=task.entry_point
                )
                break

            feedback = result.feedback
            any_update = False
            for _ in range(self.cfg.inner_updates):
                teacher_out = self._sample_teacher_completion(
                    task=task,
                    prompt=prompt,
                    feedback=feedback,
                    attempt=attempts,
                )
                if teacher_out is None:
                    self.logger.log(
                        "train_skip",
                        {
                            "task_id": task.task_id,
                            "attempt": attempts,
                            "reason": "no_valid_teacher",
                        },
                    )
                    attempts += 1
                    continue
                _, teacher = teacher_out
                self._train_pair(prompt, teacher)
                self.logger.log(
                    "train",
                    {
                        "task_id": task.task_id,
                        "step": self.global_step,
                    },
                )
                attempts += 1
                any_update = True

            if not any_update:
                continue

            # Evaluate the post-update policy with the same cheap judge channel used
            # by adaptive methods so fixed-k is not penalized by missing end-of-cycle checks.
            student = self._sample(prompt)
            self.logger.log(
                "sample",
                {
                    "task_id": task.task_id,
                    "role": "student",
                    "attempt": attempts,
                    "text": student,
                },
            )
            student_norm = normalize_completion(student, entry_point=task.entry_point)
            decision, verified, _uncertain = self._judge(
                prompt,
                feedback,
                student_norm,
            )
            attempts += 1
            if decision:
                consecutive_passes += 1
            else:
                consecutive_passes = 0
            if decision and consecutive_passes < required_passes:
                self.logger.log(
                    "stop_guard",
                    {
                        "task_id": task.task_id,
                        "attempt": attempts,
                        "consecutive_passes": consecutive_passes,
                        "required_passes": required_passes,
                        "reason": "insufficient_consecutive_passes",
                    },
                )
                continue
            if decision and verified:
                success = True
                success_completion = student_norm
                break
            if decision and not verified and self._feedback_calls < self.cfg.feedback_budget:
                verify = run_tests(task, student, timeout_s=self.cfg.test_timeout_s)
                self._feedback_calls += 1
                self._test_calls += 1
                self.logger.log(
                    "test",
                    {
                        "task_id": task.task_id,
                        "passed": verify.passed,
                        "feedback": verify.feedback,
                        "stdout": verify.stdout,
                        "stderr": verify.stderr,
                        "returncode": verify.returncode,
                        "feedback_calls": self._feedback_calls,
                    },
                )
                if verify.passed:
                    success = True
                    success_completion = student_norm
                    break

        self._add_replay_pair(task, prompt, success_completion, success)
        self.logger.log(
            "task_done",
            {
                "task_id": task.task_id,
                "success": success,
                "attempts": attempts,
                "feedback_calls": self._feedback_calls,
                "train_steps": self._train_steps,
                "judge_calls": self._judge_calls,
                "test_calls": self._test_calls,
                "samples_student": self._samples_student,
                "samples_teacher": self._samples_teacher,
            },
        )

    def _run_task_inference_only(self, task: Task) -> None:
        self._task = task
        self._feedback_calls = 0
        prompt = base_prompt(task.prompt)
        attempts = 0
        success = False
        success_completion = ""
        candidate = None

        while self._feedback_calls < self.cfg.feedback_budget:
            if candidate is None:
                candidate = self._sample(prompt)
            self.logger.log(
                "sample",
                {
                    "task_id": task.task_id,
                    "role": "student",
                    "attempt": attempts,
                    "text": candidate,
                },
            )
            result = run_tests(task, candidate, timeout_s=self.cfg.test_timeout_s)
            self._feedback_calls += 1
            self._test_calls += 1
            self.logger.log(
                "test",
                {
                    "task_id": task.task_id,
                    "passed": result.passed,
                    "feedback": result.feedback,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "feedback_calls": self._feedback_calls,
                },
            )
            if result.passed:
                success = True
                success_completion = normalize_completion(
                    candidate, entry_point=task.entry_point
                )
                break

            feedback = result.feedback
            candidate = self._sample(feedback_prompt(prompt, feedback))
            attempts += 1

        self._add_replay_pair(task, prompt, success_completion, success)
        self.logger.log(
            "task_done",
            {
                "task_id": task.task_id,
                "success": success,
                "attempts": attempts,
                "feedback_calls": self._feedback_calls,
                "train_steps": self._train_steps,
                "judge_calls": self._judge_calls,
                "test_calls": self._test_calls,
                "samples_student": self._samples_student,
                "samples_teacher": self._samples_teacher,
            },
        )

    def _run_task_adaptive_resample(self, task: Task) -> None:
        self._task = task
        self._feedback_calls = 0
        self._last_feedback = ""
        prompt = base_prompt(task.prompt)
        attempts = 0
        success = False
        success_completion = ""
        consecutive_passes = 0
        required_passes = max(1, int(self.cfg.adaptive_stop_consecutive_passes))
        updates_per_cycle = max(1, self.cfg.inner_updates)

        while True:
            if attempts == 0 and not self._last_feedback:
                if self._feedback_calls >= self.cfg.feedback_budget:
                    break
                completion = self._sample(prompt)
                self.logger.log(
                    "sample",
                    {
                        "task_id": task.task_id,
                        "role": "student",
                        "attempt": attempts,
                        "text": completion,
                    },
                )
                result = run_tests(task, completion, timeout_s=self.cfg.test_timeout_s)
                self._feedback_calls += 1
                self._test_calls += 1
                self.logger.log(
                    "test",
                    {
                        "task_id": task.task_id,
                        "passed": result.passed,
                        "feedback": result.feedback,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                        "feedback_calls": self._feedback_calls,
                    },
                )
                if result.passed:
                    success = True
                    success_completion = normalize_completion(
                        completion, entry_point=task.entry_point
                    )
                    break
                self._last_feedback = result.feedback

            if not self._last_feedback:
                break

            steps_since_feedback = 0
            last_student = ""
            stop_cycle = False
            while not stop_cycle and not success:
                for _ in range(updates_per_cycle):
                    if (
                        steps_since_feedback
                        >= self.cfg.max_adaptive_steps_per_feedback
                    ):
                        stop_cycle = True
                        break

                    student = self._sample(prompt)
                    last_student = student
                    self.logger.log(
                        "sample",
                        {
                            "task_id": task.task_id,
                            "role": "student",
                            "attempt": attempts,
                            "text": student,
                        },
                    )

                    student_norm = normalize_completion(student, entry_point=task.entry_point)
                    decision, verified, _uncertain = self._judge(
                        prompt,
                        self._last_feedback,
                        student_norm,
                    )
                    attempts += 1
                    steps_since_feedback += 1
                    if decision:
                        consecutive_passes += 1
                    else:
                        consecutive_passes = 0
                    if decision and consecutive_passes < required_passes:
                        self.logger.log(
                            "stop_guard",
                            {
                                "task_id": task.task_id,
                                "attempt": attempts,
                                "consecutive_passes": consecutive_passes,
                                "required_passes": required_passes,
                                "reason": "insufficient_consecutive_passes",
                            },
                        )
                        continue

                    if decision and verified:
                        success = True
                        success_completion = student_norm
                        stop_cycle = True
                        break
                    if decision and not verified:
                        if self._feedback_calls >= self.cfg.feedback_budget:
                            stop_cycle = True
                            break
                        result = run_tests(task, student, timeout_s=self.cfg.test_timeout_s)
                        self._feedback_calls += 1
                        self._test_calls += 1
                        self.logger.log(
                            "test",
                            {
                                "task_id": task.task_id,
                                "passed": result.passed,
                                "feedback": result.feedback,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "returncode": result.returncode,
                                "feedback_calls": self._feedback_calls,
                            },
                        )
                        if result.passed:
                            success = True
                            success_completion = student_norm
                            stop_cycle = True
                            break
                        self._last_feedback = result.feedback
                        consecutive_passes = 0
                        stop_cycle = True
                        break

                if success or stop_cycle:
                    break

                if steps_since_feedback >= self.cfg.max_adaptive_steps_per_feedback:
                    if self._feedback_calls >= self.cfg.feedback_budget or not last_student:
                        stop_cycle = True
                        break
                    result = run_tests(
                        task, last_student, timeout_s=self.cfg.test_timeout_s
                    )
                    self._feedback_calls += 1
                    self._test_calls += 1
                    self.logger.log(
                        "forced_recheck",
                        {
                            "task_id": task.task_id,
                            "passed": result.passed,
                            "feedback": result.feedback,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "returncode": result.returncode,
                            "feedback_calls": self._feedback_calls,
                            "steps_since_feedback": steps_since_feedback,
                        },
                    )
                    if result.passed:
                        success = True
                        success_completion = normalize_completion(
                            last_student, entry_point=task.entry_point
                        )
                        break
                    self._last_feedback = result.feedback
                    consecutive_passes = 0
                    stop_cycle = True

            if success or self._feedback_calls >= self.cfg.feedback_budget:
                break

        self._add_replay_pair(task, prompt, success_completion, success)
        self.logger.log(
            "task_done",
            {
                "task_id": task.task_id,
                "success": success,
                "attempts": attempts,
                "feedback_calls": self._feedback_calls,
                "train_steps": self._train_steps,
                "judge_calls": self._judge_calls,
                "test_calls": self._test_calls,
                "samples_student": self._samples_student,
                "samples_teacher": self._samples_teacher,
            },
        )

    def _run_task_resample_only(self, task: Task) -> None:
        self._task = task
        self._feedback_calls = 0
        prompt = base_prompt(task.prompt)
        attempts = 0
        success = False
        success_completion = ""

        while self._feedback_calls < self.cfg.feedback_budget:
            completion = self._sample(prompt)
            self.logger.log(
                "sample",
                {
                    "task_id": task.task_id,
                    "role": "student",
                    "attempt": attempts,
                    "text": completion,
                },
            )
            result = run_tests(task, completion, timeout_s=self.cfg.test_timeout_s)
            self._feedback_calls += 1
            self._test_calls += 1
            self.logger.log(
                "test",
                {
                    "task_id": task.task_id,
                    "passed": result.passed,
                    "feedback": result.feedback,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "feedback_calls": self._feedback_calls,
                },
            )
            if result.passed:
                success = True
                success_completion = normalize_completion(
                    completion, entry_point=task.entry_point
                )
                break
            attempts += 1

        self._add_replay_pair(task, prompt, success_completion, success)
        self.logger.log(
            "task_done",
            {
                "task_id": task.task_id,
                "success": success,
                "attempts": attempts,
                "feedback_calls": self._feedback_calls,
                "train_steps": self._train_steps,
                "judge_calls": self._judge_calls,
                "test_calls": self._test_calls,
                "samples_student": self._samples_student,
                "samples_teacher": self._samples_teacher,
            },
        )

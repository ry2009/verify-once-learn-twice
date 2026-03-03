from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from ttRL.config import ExperimentConfig
from ttRL.data import Task
from ttRL.eval import TestResult
from ttRL.loop import ExperimentRunner
from ttRL.verifier_budget_policy import BudgetAction, VerifierBudgetPolicy


class VerifierBudgetPolicyTests(unittest.TestCase):
    def test_initial_limit_respects_base_budget(self) -> None:
        policy = VerifierBudgetPolicy(
            mode="vbc_v1",
            scheduler_enabled=True,
            base_budget=1,
            hard_budget=2,
            chain_budget=4,
        )
        self.assertEqual(policy.initial_limit(max_budget=4), 1)
        self.assertEqual(policy.initial_limit(max_budget=1), 1)

    def test_escalates_for_hard_and_chain_failures(self) -> None:
        policy = VerifierBudgetPolicy(
            mode="vbc_v1",
            scheduler_enabled=True,
            base_budget=1,
            hard_budget=2,
            chain_budget=4,
        )
        hard_limit, hard_reason = policy.maybe_escalate(
            current_limit=1, max_budget=4, failure_kind="hard_runtime"
        )
        chain_limit, chain_reason = policy.maybe_escalate(
            current_limit=1, max_budget=4, failure_kind="op_chain"
        )
        self.assertEqual((hard_limit, hard_reason), (2, "hard_runtime"))
        self.assertEqual((chain_limit, chain_reason), (4, "op_chain"))

    def test_decide_action_prefers_verify_when_no_feedback_or_hard_failure(self) -> None:
        policy = VerifierBudgetPolicy(
            mode="vbc_v1",
            scheduler_enabled=True,
            min_refine_steps_before_verify=2,
        )
        no_feedback = policy.decide_action(
            budget_used=0,
            budget_limit=1,
            has_feedback=False,
            steps_since_feedback=0,
            max_steps_per_feedback=8,
            failure_kind="soft",
        )
        hard_failure = policy.decide_action(
            budget_used=0,
            budget_limit=2,
            has_feedback=True,
            steps_since_feedback=1,
            max_steps_per_feedback=8,
            failure_kind="hard_runtime",
        )
        self.assertEqual(no_feedback, BudgetAction.VERIFY)
        self.assertEqual(hard_failure, BudgetAction.VERIFY)

    def test_decide_action_stops_when_budget_exhausted(self) -> None:
        policy = VerifierBudgetPolicy(mode="vbc_v1", scheduler_enabled=True)
        action = policy.decide_action(
            budget_used=2,
            budget_limit=2,
            has_feedback=True,
            steps_since_feedback=3,
            max_steps_per_feedback=8,
            failure_kind="soft",
        )
        self.assertEqual(action, BudgetAction.STOP)

    def test_runner_initial_budget_uses_policy(self) -> None:
        cfg = ExperimentConfig(
            run_name="unit-vbc",
            run_group="unit",
            method="adaptive_fwb",
            feedback_budget=4,
            adaptive_budget_scheduler=True,
            adaptive_budget_base=1,
            verifier_budget_policy="vbc_v1",
            max_steps=1,
        )
        runner = ExperimentRunner(cfg, run_dir="/tmp/ttrl-vbc-unit")
        self.assertEqual(runner._initial_budget_limit(), 1)

    @patch("ttRL.loop.run_tests")
    def test_adaptive_loop_runs_with_vbc_policy(self, run_tests_mock) -> None:
        run_tests_mock.side_effect = [
            TestResult(
                passed=False,
                feedback="RuntimeError: shape mismatch",
                stdout="",
                stderr="",
                returncode=1,
            ),
            TestResult(
                passed=True,
                feedback="",
                stdout="",
                stderr="",
                returncode=0,
            ),
        ]
        cfg = ExperimentConfig(
            run_name="unit-vbc-e2e",
            run_group="unit",
            method="adaptive_fwb",
            judge_mode="oracle_binary",
            feedback_budget=2,
            adaptive_budget_scheduler=True,
            adaptive_budget_base=1,
            adaptive_budget_hard=2,
            adaptive_budget_chain=2,
            verifier_budget_policy="vbc_v1",
            max_steps=1,
        )
        run_dir = tempfile.mkdtemp(prefix="ttrl-vbc-e2e-")
        runner = ExperimentRunner(cfg, run_dir=run_dir)
        runner._sample = lambda _prompt: "return x"  # type: ignore[method-assign]
        runner._sample_teacher_completion = lambda **_kwargs: ("return x", "return x")  # type: ignore[method-assign]
        runner._train_pair = lambda _prompt, _target: None  # type: ignore[method-assign]
        runner._run_reinforce_once = (
            lambda task, prompt, success, success_completion: (success, success_completion)
        )  # type: ignore[method-assign]
        task = Task(
            task_id="unit/task",
            prompt="def f(x):",
            test="assert f(1) == 1",
            entry_point="f",
        )
        runner._log_task_start(task)
        runner._run_task_adaptive_fwb(task)

        events = []
        with Path(run_dir, "events.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        budget_start = [ev for ev in events if ev.get("type") == "budget_start"]
        escalations = [ev for ev in events if ev.get("type") == "budget_escalate"]
        done = [ev for ev in events if ev.get("type") == "task_done"]
        self.assertTrue(budget_start)
        self.assertEqual(budget_start[-1].get("policy"), "vbc_v1")
        self.assertTrue(escalations)
        self.assertEqual(escalations[-1].get("to_budget"), 2)
        self.assertTrue(done)
        self.assertTrue(done[-1].get("success"))


if __name__ == "__main__":
    unittest.main()

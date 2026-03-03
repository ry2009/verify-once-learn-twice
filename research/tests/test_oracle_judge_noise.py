from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from ttRL.config import ExperimentConfig
from ttRL.data import Task
from ttRL.eval import TestResult
from ttRL.loop import ExperimentRunner


class OracleJudgeNoiseTests(unittest.TestCase):
    def _runner(self, judge_flip_prob: float) -> ExperimentRunner:
        cfg = ExperimentConfig(
            run_name="unit-oracle-noise",
            run_group="unit",
            method="adaptive_fwb",
            judge_mode="oracle_binary",
            judge_flip_prob=judge_flip_prob,
            feedback_budget=1,
            max_steps=1,
        )
        run_dir = tempfile.mkdtemp(prefix="ttrl-unit-")
        runner = ExperimentRunner(cfg, run_dir)
        runner._task = Task(
            task_id="unit/task",
            prompt="def f(x):",
            test="assert f(1) == 1",
            entry_point="f",
        )
        runner._judge_calls = 0
        runner._test_calls = 0
        return runner

    @patch("ttRL.loop.run_tests")
    def test_oracle_without_noise_is_verified(self, run_tests_mock) -> None:
        run_tests_mock.return_value = TestResult(
            passed=True,
            feedback="",
            stdout="",
            stderr="",
            returncode=0,
        )
        runner = self._runner(judge_flip_prob=0.0)
        decision, verified, uncertain = runner._judge("p", "f", "return x")
        self.assertTrue(decision)
        self.assertTrue(verified)
        self.assertFalse(uncertain)

    @patch("ttRL.loop.run_tests")
    def test_oracle_with_flip_noise_can_flip_decision(self, run_tests_mock) -> None:
        run_tests_mock.return_value = TestResult(
            passed=True,
            feedback="",
            stdout="",
            stderr="",
            returncode=0,
        )
        runner = self._runner(judge_flip_prob=1.0)
        decision, verified, uncertain = runner._judge("p", "f", "return x")
        self.assertFalse(decision)
        self.assertFalse(verified)
        self.assertFalse(uncertain)

    @patch("ttRL.loop.random.random", return_value=0.99)
    @patch("ttRL.loop.run_tests")
    def test_oracle_with_noise_requires_verification_even_without_flip(
        self, run_tests_mock, _random_mock
    ) -> None:
        run_tests_mock.return_value = TestResult(
            passed=True,
            feedback="",
            stdout="",
            stderr="",
            returncode=0,
        )
        runner = self._runner(judge_flip_prob=0.5)
        decision, verified, uncertain = runner._judge("p", "f", "return x")
        self.assertTrue(decision)
        self.assertFalse(verified)
        self.assertFalse(uncertain)


if __name__ == "__main__":
    unittest.main()

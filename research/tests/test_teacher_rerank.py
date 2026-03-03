from __future__ import annotations

import tempfile
import unittest

from ttRL.config import ExperimentConfig
from ttRL.data import Task
from ttRL.loop import ExperimentRunner


def _kernel_task() -> Task:
    prompt = (
        "import torch\n\n"
        "# Reference forward body (self -> model):\n"
        "# x = model.conv(x)\n"
        "# x = torch.relu(x)\n"
        "# x = x + model.bias\n"
        "# return x\n\n"
        "def optimized_forward(model, x):\n"
        "    "
    )
    return Task(
        task_id="KB-UNIT/2/1",
        prompt=prompt,
        test="",
        entry_point="optimized_forward",
    )


class TeacherRerankTests(unittest.TestCase):
    def _runner(self, **overrides) -> ExperimentRunner:
        kwargs = dict(
            run_name="unit-teacher-rerank",
            run_group="unit",
            method="adaptive_fwb",
            feedback_budget=4,
        )
        kwargs.update(overrides)
        cfg = ExperimentConfig(**kwargs)
        run_dir = tempfile.mkdtemp(prefix="ttrl-unit-")
        return ExperimentRunner(cfg, run_dir)

    def test_op_coverage_scores_ordered_operator_chain(self) -> None:
        runner = self._runner()
        task = _kernel_task()
        hi = "x = model.conv(x)\nx = torch.relu(x)\nx = x + model.bias\nreturn x"
        lo = "x = model.conv(x)\nreturn x"
        cov_hi, _, _, _, _ = runner._op_coverage(task, hi)
        cov_lo, _, _, _, _ = runner._op_coverage(task, lo)
        self.assertGreater(cov_hi, cov_lo)
        self.assertGreaterEqual(cov_hi, 0.99)
        self.assertLess(cov_lo, 0.5)

    def test_teacher_best_of_n_prefers_high_coverage_candidate(self) -> None:
        runner = self._runner(
            teacher_best_of_n=3,
            teacher_resample_attempts=1,
            teacher_min_op_coverage=0.5,
        )
        task = _kernel_task()
        samples = iter(
            [
                "x = model.conv(x)\nreturn x",
                "x = model.conv(x)\nx = torch.relu(x)\nx = x + model.bias\nreturn x",
                "return x",
            ]
        )
        runner._sample_teacher = lambda _prompt: next(samples)  # type: ignore[assignment]
        chosen = runner._sample_teacher_completion(
            task=task,
            prompt=task.prompt,
            feedback="shape mismatch",
            attempt=0,
        )
        self.assertIsNotNone(chosen)
        assert chosen is not None
        _, body = chosen
        self.assertIn("torch.relu", body)
        self.assertIn("model.bias", body)

    def test_teacher_min_coverage_filter_can_reject_low_quality_pool(self) -> None:
        runner = self._runner(
            teacher_best_of_n=2,
            teacher_resample_attempts=1,
            teacher_min_op_coverage=0.95,
        )
        task = _kernel_task()
        samples = iter(["x = model.conv(x)\nreturn x", "return x"])
        runner._sample_teacher = lambda _prompt: next(samples)  # type: ignore[assignment]
        chosen = runner._sample_teacher_completion(
            task=task,
            prompt=task.prompt,
            feedback="shape mismatch",
            attempt=0,
        )
        self.assertIsNone(chosen)

    def test_budget_scheduler_escalates_on_hard_and_chain_failures(self) -> None:
        runner = self._runner(
            adaptive_budget_scheduler=True,
            feedback_budget=4,
            adaptive_budget_base=1,
            adaptive_budget_hard=2,
            adaptive_budget_chain=4,
            adaptive_budget_op_chain_threshold=0.8,
        )
        task = _kernel_task()
        self.assertEqual(runner._initial_budget_limit(), 1)
        hard = runner._maybe_escalate_budget(
            task,
            current_limit=1,
            feedback="RuntimeError: shape mismatch",
            candidate="x = model.conv(x)\nreturn x",
        )
        self.assertEqual(hard, 2)
        chain = runner._maybe_escalate_budget(
            task,
            current_limit=2,
            feedback="AssertionError: value mismatch",
            candidate="x = model.conv(x)\nreturn x",
        )
        self.assertEqual(chain, 4)


if __name__ == "__main__":
    unittest.main()

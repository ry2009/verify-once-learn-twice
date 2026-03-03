from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Tuple

from .feedback import normalize_completion
from .data import Task


@dataclass
class TestResult:
    passed: bool
    feedback: str
    stdout: str
    stderr: str
    returncode: int


def _run_python_file(path: str, timeout_s: int) -> Tuple[int, str, str]:
    last_err = None
    for exe in ("python", "python3"):
        try:
            proc = subprocess.run(
                [exe, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={"PYTHONHASHSEED": "0", **os.environ},
            )
            return proc.returncode, proc.stdout, proc.stderr
        except FileNotFoundError as exc:
            last_err = exc
            continue
    raise RuntimeError("No python interpreter found") from last_err


def run_tests(task: Task, completion: str, timeout_s: int = 5) -> TestResult:
    code = normalize_completion(completion, entry_point=task.entry_point)
    test_code = task.test
    # HumanEval-style tasks define check(candidate) and expect explicit invocation.
    if task.entry_point and "def check(" in test_code:
        check_call = rf"check\(\s*{re.escape(task.entry_point)}\s*\)"
        if not re.search(check_call, test_code):
            test_code = f"{test_code}\ncheck({task.entry_point})\n"

    full = f"{task.prompt}\n{code}\n\n{test_code}\n"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(full)
        try:
            rc, out, err = _run_python_file(path, timeout_s)
        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                feedback="Timeout expired",
                stdout="",
                stderr="Timeout expired",
                returncode=124,
            )

    passed = rc == 0
    feedback = err.strip() if err.strip() else out.strip()
    if not feedback:
        feedback = "Tests failed" if not passed else ""

    return TestResult(
        passed=passed,
        feedback=feedback,
        stdout=out,
        stderr=err,
        returncode=rc,
    )

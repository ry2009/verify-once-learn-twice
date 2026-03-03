from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class SyntaxIssue:
    task_id: str
    line: int
    msg: str


@dataclass
class ValidationReport:
    tasks_path: str
    total_tasks: int
    syntax_ok: int
    syntax_bad: int
    issues: List[SyntaxIssue]


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_program(prompt: str, test: str) -> str:
    # Use a minimal body to validate prompt/test syntax compatibility.
    return f"{prompt}\n    pass\n\n{test}\n"


def validate_tasks(path: Path, max_issues: int) -> ValidationReport:
    total = 0
    ok = 0
    bad = 0
    issues: List[SyntaxIssue] = []

    for idx, row in enumerate(_iter_jsonl(path)):
        total += 1
        task_id = str(row.get("task_id", f"row_{idx}"))
        prompt = str(row.get("prompt", ""))
        test = str(row.get("test", ""))
        code = _build_program(prompt, test)
        try:
            ast.parse(code)
            ok += 1
        except SyntaxError as exc:
            bad += 1
            if len(issues) < max_issues:
                issues.append(
                    SyntaxIssue(
                        task_id=task_id,
                        line=int(exc.lineno or 0),
                        msg=str(exc.msg),
                    )
                )

    return ValidationReport(
        tasks_path=str(path),
        total_tasks=total,
        syntax_ok=ok,
        syntax_bad=bad,
        issues=issues,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", action="append", required=True, help="Path to JSONL tasks file.")
    parser.add_argument("--max_issues", type=int, default=20)
    parser.add_argument("--out_json", default="", help="Optional path to write full JSON report.")
    args = parser.parse_args()

    reports: List[ValidationReport] = []
    failed = False
    for tasks_path in args.tasks:
        report = validate_tasks(Path(tasks_path), max_issues=int(args.max_issues))
        reports.append(report)
        print(
            f"{report.tasks_path}: total={report.total_tasks} ok={report.syntax_ok} bad={report.syntax_bad}"
        )
        for issue in report.issues:
            print(f"  - {issue.task_id}: line={issue.line} msg={issue.msg}")
        if report.syntax_bad > 0:
            failed = True

    if args.out_json:
        payload = {
            "reports": [
                {
                    **asdict(r),
                    "issues": [asdict(i) for i in r.issues],
                }
                for r in reports
            ]
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")

    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

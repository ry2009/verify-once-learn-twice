from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset


def _function_header(code: str) -> Tuple[str, str]:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            # Build a stable signature from AST rather than raw text.
            args = ast.unparse(node.args)
            return name, f"def {name}({args}):"
    raise ValueError("No top-level function definition found in MBPP code field")


def _prompt_from_text(header: str, text: str) -> str:
    desc = text.strip().replace('"""', '\"\"\"')
    return f"{header}\n    \"\"\"{desc}\"\"\"\n"


def _rewrite_assert(assert_line: str, entry_point: str) -> str:
    line = assert_line.strip()
    if not line.startswith("assert "):
        return ""
    # Replace direct calls to entry_point(...) with candidate(...).
    pat = re.compile(rf"\b{re.escape(entry_point)}\s*\(")
    rewritten = pat.sub("candidate(", line)
    if rewritten != line:
        return rewritten
    # Fallback for MBPP rows where tests call an alias/helper name directly.
    # We only rewrite if the assert expression starts with a plain function call.
    expr = line[len("assert ") :]
    m = re.match(r"([A-Za-z_]\w*)\s*\(", expr)
    if m and m.group(1) != "candidate":
        return "assert candidate(" + expr[m.end() :]
    return rewritten


def _build_test(
    entry_point: str,
    test_setup_code: str,
    asserts: Iterable[str],
) -> str:
    lines: List[str] = []
    lines.append("METADATA = {}")
    lines.append("")
    lines.append("def check(candidate):")
    setup = test_setup_code.strip("\n")
    if setup:
        for line in setup.splitlines():
            lines.append(f"    {line}")

    kept = 0
    for raw in asserts:
        rewritten = _rewrite_assert(raw, entry_point)
        if not rewritten:
            continue
        lines.append(f"    {rewritten}")
        kept += 1

    if kept == 0:
        lines.append("    pass")

    lines.append("")
    lines.append("try:")
    lines.append(f"    check({entry_point})")
    lines.append("except Exception:")
    lines.append("    raise AssertionError('TESTS_FAILED')")
    lines.append("")
    return "\n".join(lines)


def _row_to_task(row: dict, include_challenge: bool) -> dict:
    entry_point, header = _function_header(str(row["code"]))
    public_asserts = list(row.get("test_list", []))
    challenge_asserts = list(row.get("challenge_test_list", []))
    asserts = public_asserts + challenge_asserts if include_challenge else public_asserts
    return {
        "task_id": f"MBPP/{int(row['task_id'])}",
        "prompt": _prompt_from_text(header, str(row["text"])),
        "test": _build_test(
            entry_point=entry_point,
            test_setup_code=str(row.get("test_setup_code", "")),
            asserts=asserts,
        ),
        "entry_point": entry_point,
    }


def _reference_passes(task: dict, reference_code: str, timeout_s: int = 8) -> bool:
    full = f"{reference_code}\n\n{task['test']}\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prog.py"
        path.write_text(full, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["python3", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return proc.returncode == 0
        except Exception:
            return False


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="data/mbpp_test_public.jsonl")
    parser.add_argument("--out_all_tests", default="data/mbpp_test_all.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--validate_reference", action="store_true")
    args = parser.parse_args()

    ds = load_dataset("mbpp", split=args.split)
    rows = list(ds)
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    public_tasks: List[dict] = []
    all_tasks: List[dict] = []
    dropped_public = 0
    dropped_all = 0
    for row in rows:
        t_pub = _row_to_task(row, include_challenge=False)
        t_all = _row_to_task(row, include_challenge=True)
        if args.validate_reference:
            code = str(row["code"])
            keep_pub = _reference_passes(t_pub, code)
            keep_all = _reference_passes(t_all, code)
            if keep_pub:
                public_tasks.append(t_pub)
            else:
                dropped_public += 1
            if keep_all:
                all_tasks.append(t_all)
            else:
                dropped_all += 1
        else:
            public_tasks.append(t_pub)
            all_tasks.append(t_all)

    out_public = Path(args.out)
    out_all = Path(args.out_all_tests)
    out_public.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_public, public_tasks)
    _write_jsonl(out_all, all_tasks)

    manifest = {
        "dataset": "mbpp",
        "split": args.split,
        "rows": len(rows),
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
        "limit": args.limit,
        "validate_reference": bool(args.validate_reference),
        "rows_public_kept": len(public_tasks),
        "rows_all_kept": len(all_tasks),
        "rows_public_dropped": dropped_public,
        "rows_all_dropped": dropped_all,
        "out_public": str(out_public),
        "out_all_tests": str(out_all),
    }
    manifest_path = out_public.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {out_public}")
    print(f"wrote {out_all}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


@dataclass
class KBRow:
    code: str
    level: int
    name: str
    problem_id: int
    forward_args: list[str]
    forward_body: str


def _iter_levels(levels: list[int]) -> Iterable[dict[str, Any]]:
    ds = load_dataset("ScalingIntelligence/KernelBench")
    for level in levels:
        split_name = f"level_{level}"
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            yield row


def _extract_forward(code: str) -> tuple[list[str], str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    model_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            model_cls = node
            break
    if model_cls is None:
        return None

    forward = None
    for node in model_cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            forward = node
            break
    if forward is None:
        return None

    args = [a.arg for a in forward.args.args][1:]  # drop self
    segs: list[str] = []
    for stmt in forward.body:
        src = ast.get_source_segment(code, stmt)
        if src:
            segs.append(src)
    if not segs:
        try:
            segs = [ast.unparse(stmt) for stmt in forward.body]
        except Exception:
            return None
    body = "\n".join(segs).strip()
    if not body:
        return None
    return args, body


def _build_prompt(name: str, level: int, args: list[str], forward_body: str) -> str:
    sig_args = ", ".join(["model"] + args) if args else "model, *inputs"
    hint = re.sub(r"\bself\.", "model.", forward_body)
    hint = "\n".join(f"# {line}" for line in hint.splitlines())
    return (
        "import torch\n\n"
        f"# KernelBench task: {name} (level={level})\n"
        "# Implement the same computation as the reference Model.forward.\n"
        "# You may read model parameters/submodules via `model.<...>`.\n"
        "# Do NOT call model(...) directly and do NOT call helper generators.\n"
        "# Reference forward body (self -> model):\n"
        f"{hint}\n\n"
        f"def optimized_forward({sig_args}):\n"
        "    "
    )


def _build_test(code: str) -> str:
    # Keep verifier strict but stable on CPU by clipping large tensors.
    return (
        "import ast\n"
        "import inspect\n"
        "import time\n"
        "import torch\n\n"
        f"{code}\n\n"
        "torch.set_num_threads(1)\n\n"
        "def _clip_tensor(x, max_b=2):\n"
        "    if not isinstance(x, torch.Tensor):\n"
        "        return x\n"
        "    y = x\n"
        "    if y.ndim >= 1 and y.shape[0] > max_b:\n"
        "        y = y[:max_b]\n"
        "    return y.contiguous()\n\n"
        "def _clip_obj(v):\n"
        "    if isinstance(v, torch.Tensor):\n"
        "        return _clip_tensor(v)\n"
        "    if isinstance(v, tuple):\n"
        "        return tuple(_clip_obj(x) for x in v)\n"
        "    if isinstance(v, list):\n"
        "        return [_clip_obj(x) for x in v]\n"
        "    if isinstance(v, dict):\n"
        "        return {k: _clip_obj(x) for k, x in v.items()}\n"
        "    return v\n\n"
        "def _assert_close(a, b):\n"
        "    if isinstance(a, torch.Tensor):\n"
        "        assert isinstance(b, torch.Tensor), f'type mismatch: {type(a)} vs {type(b)}'\n"
        "        assert a.shape == b.shape, f'shape mismatch: {a.shape} vs {b.shape}'\n"
        "        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)\n"
        "        return\n"
        "    if isinstance(a, tuple):\n"
        "        assert isinstance(b, tuple) and len(a) == len(b), 'tuple mismatch'\n"
        "        for x, y in zip(a, b):\n"
        "            _assert_close(x, y)\n"
        "        return\n"
        "    if isinstance(a, list):\n"
        "        assert isinstance(b, list) and len(a) == len(b), 'list mismatch'\n"
        "        for x, y in zip(a, b):\n"
        "            _assert_close(x, y)\n"
        "        return\n"
        "    if isinstance(a, dict):\n"
        "        assert isinstance(b, dict) and set(a.keys()) == set(b.keys()), 'dict key mismatch'\n"
        "        for k in a:\n"
        "            _assert_close(a[k], b[k])\n"
        "        return\n"
        "    if isinstance(a, float):\n"
        "        assert abs(float(a) - float(b)) <= 1e-5, f'float mismatch: {a} vs {b}'\n"
        "        return\n"
        "    assert a == b, f'value mismatch: {a} vs {b}'\n\n"
        "def _timed(fn, n=1):\n"
        "    fn()\n"
        "    t0 = time.perf_counter()\n"
        "    for _ in range(n):\n"
        "        fn()\n"
        "    return (time.perf_counter() - t0) / float(n)\n\n"
        "def _banned_call_name(src):\n"
        "    try:\n"
        "        tree = ast.parse(src)\n"
        "    except SyntaxError:\n"
        "        return ''\n"
        "    banned = ''\n"
        "    class _V(ast.NodeVisitor):\n"
        "        def visit_Call(self, node):\n"
        "            nonlocal banned\n"
        "            if banned:\n"
        "                return\n"
        "            f = node.func\n"
        "            if isinstance(f, ast.Name) and f.id in {'model', 'Model', 'get_inputs', 'get_init_inputs'}:\n"
        "                banned = f.id\n"
        "                return\n"
        "            self.generic_visit(node)\n"
        "    _V().visit(tree)\n"
        "    return banned\n\n"
        "def check(candidate):\n"
        "    src = inspect.getsource(candidate)\n"
        "    bad = _banned_call_name(src)\n"
        "    assert not bad, f'BANNED_CALL:{bad}'\n"
        "    for seed in (0, 1):\n"
        "        torch.manual_seed(seed)\n"
        "        init_inputs = get_init_inputs()\n"
        "        model = Model(*init_inputs)\n"
        "        model.eval()\n"
        "        with torch.no_grad():\n"
        "            raw_inputs = get_inputs()\n"
        "            if not isinstance(raw_inputs, (list, tuple)):\n"
        "                raw_inputs = [raw_inputs]\n"
        "            inputs = [_clip_obj(x) for x in raw_inputs]\n"
        "            ref = model(*inputs)\n"
        "            out = candidate(model, *inputs)\n"
        "            _assert_close(ref, out)\n"
        "            ref_t = _timed(lambda: model(*inputs), n=1)\n"
        "            cand_t = _timed(lambda: candidate(model, *inputs), n=1)\n"
        "            lim = max(0.08, ref_t * 6.0)\n"
        "            assert cand_t <= lim, f'PERF_FAIL:cand={cand_t:.6f}:ref={ref_t:.6f}:lim={lim:.6f}'\n\n"
        "check(optimized_forward)\n"
    )


def _to_kb_rows(levels: list[int], max_code_chars: int) -> list[KBRow]:
    out: list[KBRow] = []
    for row in _iter_levels(levels):
        code = str(row.get("code", ""))
        if not code or len(code) > max_code_chars:
            continue
        extracted = _extract_forward(code)
        if not extracted:
            continue
        args, body = extracted
        out.append(
            KBRow(
                code=code,
                level=int(row.get("level", 0)),
                name=str(row.get("name", "")),
                problem_id=int(row.get("problem_id", -1)),
                forward_args=args,
                forward_body=body,
            )
        )
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source_levels", default="2")
    parser.add_argument("--target_levels", default="3")
    parser.add_argument("--n_source", type=int, default=24)
    parser.add_argument("--n_target", type=int, default=12)
    parser.add_argument("--max_code_chars", type=int, default=14000)
    parser.add_argument(
        "--out_all",
        default="data/kernelbench_transfer_l2to3_36.jsonl",
    )
    parser.add_argument(
        "--out_source",
        default="data/kernelbench_transfer_l2_source24.jsonl",
    )
    parser.add_argument(
        "--out_target",
        default="data/kernelbench_transfer_l3_target12.jsonl",
    )
    parser.add_argument(
        "--out_manifest",
        default="data/kernelbench_transfer_l2to3_36.manifest.json",
    )
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    source_levels = [int(x) for x in args.source_levels.split(",") if x.strip()]
    target_levels = [int(x) for x in args.target_levels.split(",") if x.strip()]

    source_pool = _to_kb_rows(source_levels, max_code_chars=int(args.max_code_chars))
    target_pool = _to_kb_rows(target_levels, max_code_chars=int(args.max_code_chars))
    if len(source_pool) < int(args.n_source):
        raise SystemExit(
            f"not enough source tasks: have={len(source_pool)} need={int(args.n_source)}"
        )
    if len(target_pool) < int(args.n_target):
        raise SystemExit(
            f"not enough target tasks: have={len(target_pool)} need={int(args.n_target)}"
        )

    rng.shuffle(source_pool)
    rng.shuffle(target_pool)
    chosen_source = source_pool[: int(args.n_source)]
    chosen_target = target_pool[: int(args.n_target)]

    used = {row.problem_id for row in chosen_source}
    chosen_target = [row for row in chosen_target if row.problem_id not in used]
    if len(chosen_target) < int(args.n_target):
        # refill from target pool with non-overlap
        for row in target_pool[int(args.n_target) :]:
            if row.problem_id in used:
                continue
            chosen_target.append(row)
            if len(chosen_target) >= int(args.n_target):
                break
    if len(chosen_target) < int(args.n_target):
        raise SystemExit("unable to build non-overlapping target set")

    def to_task(row: KBRow, split: str) -> dict[str, Any]:
        task_id = f"KB-{split.upper()}/{row.level}/{row.problem_id}"
        prompt = _build_prompt(
            name=row.name,
            level=row.level,
            args=row.forward_args,
            forward_body=row.forward_body,
        )
        test = _build_test(row.code)
        return {
            "task_id": task_id,
            "prompt": prompt,
            "entry_point": "optimized_forward",
            "test": test,
            "kernelbench_name": row.name,
            "kernelbench_level": row.level,
            "kernelbench_problem_id": row.problem_id,
            "transfer_split": split,
        }

    source_tasks = [to_task(row, "source") for row in chosen_source]
    target_tasks = [to_task(row, "target") for row in chosen_target]
    all_tasks = source_tasks + target_tasks

    out_all = Path(args.out_all)
    out_source = Path(args.out_source)
    out_target = Path(args.out_target)
    out_manifest = Path(args.out_manifest)

    _write_jsonl(out_all, all_tasks)
    _write_jsonl(out_source, source_tasks)
    _write_jsonl(out_target, target_tasks)

    manifest = {
        "seed": int(args.seed),
        "source_levels": source_levels,
        "target_levels": target_levels,
        "n_source": int(args.n_source),
        "n_target": int(args.n_target),
        "out_all": str(out_all),
        "out_source": str(out_source),
        "out_target": str(out_target),
        "source_problem_ids": [r.problem_id for r in chosen_source],
        "target_problem_ids": [r.problem_id for r in chosen_target],
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()

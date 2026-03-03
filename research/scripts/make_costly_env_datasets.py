from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _task_seed(task_id: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{task_id}|{base_seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _validate_python_snippet(task_id: str, label: str, code: str) -> None:
    try:
        compile(code, f"<{task_id}:{label}>", "exec")
    except SyntaxError as exc:
        msg = (
            f"invalid transformed test for {task_id} ({label}): "
            f"{exc.msg} at line {exc.lineno}"
        )
        raise ValueError(msg) from exc


def _assert_blocks(test: str) -> List[Tuple[int, int]]:
    lines = test.splitlines()
    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped.startswith("assert "):
            i += 1
            continue

        start = i
        base_indent = len(line) - len(stripped)
        balance = 0
        j = i
        while j < len(lines):
            cur = lines[j]
            cur_stripped = cur.strip()
            balance += cur.count("(") + cur.count("[") + cur.count("{")
            balance -= cur.count(")") + cur.count("]") + cur.count("}")

            next_j = j + 1
            if next_j >= len(lines):
                j += 1
                break
            nxt = lines[next_j]
            nxt_stripped = nxt.strip()
            nxt_indent = len(nxt) - len(nxt.lstrip())

            continues = False
            if cur.rstrip().endswith("\\"):
                continues = True
            elif balance > 0 and nxt_indent > base_indent:
                continues = True
            elif balance > 0 and nxt_stripped:
                continues = True

            if continues:
                j += 1
                continue

            j += 1
            break

        blocks.append((start, j))
        i = j
    return blocks


def _strip_explicit_check_call(test: str, entry_point: str) -> str:
    if not entry_point:
        return test
    lines = test.splitlines()
    needle = f"check({entry_point})".replace(" ", "")
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        compact = line.replace(" ", "")
        if compact != needle:
            out.append(line)
            i += 1
            continue

        # Drop the explicit check call itself.
        i += 1

        # If we removed a "check(...)" inside a simple "try/except" wrapper,
        # remove the now-empty wrapper to avoid invalid Python in coarse tests.
        if out and out[-1].strip() == "try:":
            out.pop()
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].lstrip().startswith("except "):
                except_indent = len(lines[i]) - len(lines[i].lstrip())
                i += 1
                while i < len(lines):
                    cur = lines[i]
                    stripped = cur.strip()
                    if not stripped:
                        i += 1
                        continue
                    indent = len(cur) - len(cur.lstrip())
                    if indent <= except_indent:
                        break
                    i += 1
        # Continue scanning without appending removed lines.
    return "\n".join(out)


def _inject_pass_if_no_assert(lines: List[str]) -> List[str]:
    if any(line.lstrip().startswith("assert ") for line in lines):
        return lines
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def check("):
            indent = " " * (len(line) - len(line.lstrip()) + 4)
            return lines[: i + 1] + [f"{indent}pass"] + lines[i + 1 :]
    return lines


def _filter_asserts(test: str, keep_ids: set[int], blocks: List[Tuple[int, int]]) -> str:
    lines = test.splitlines()
    drop_lines: set[int] = set()
    for block_idx, (start, end) in enumerate(blocks):
        if block_idx in keep_ids:
            continue
        for i in range(start, end):
            drop_lines.add(i)

    out: List[str] = []
    for i, line in enumerate(lines):
        if i in drop_lines:
            continue
        out.append(line)
    out = _inject_pass_if_no_assert(out)
    return "\n".join(out).rstrip() + "\n"


def _split_assert_ids(assert_ids: List[int], task_id: str, hidden_ratio: float, seed: int) -> Tuple[set[int], set[int]]:
    if not assert_ids:
        return set(), set()
    # Deterministic shuffle per task to make splits reproducible.
    keyed = [(_task_seed(f"{task_id}:{idx}", seed), idx) for idx in assert_ids]
    keyed.sort()
    ordered = [idx for _, idx in keyed]

    if len(assert_ids) == 1:
        # Duplicate the single assert into both views to keep both tests meaningful.
        only = {assert_ids[0]}
        return only, only

    hidden_n = int(round(len(assert_ids) * hidden_ratio))
    hidden_n = max(1, min(len(assert_ids) - 1, hidden_n))
    hidden = set(ordered[:hidden_n])
    public = set(ordered[hidden_n:])
    return public, hidden


def _to_coarse_test(test: str, entry_point: str) -> str:
    base = _strip_explicit_check_call(test, entry_point).rstrip()
    if not entry_point:
        return base + "\n"
    coarse_tail = (
        "\n\ntry:\n"
        f"    check({entry_point})\n"
        "except Exception:\n"
        "    raise AssertionError('TESTS_FAILED')\n"
    )
    return base + coarse_tail


def build_env_rows(rows: Sequence[dict], hidden_ratio: float, seed: int) -> Dict[str, List[dict]]:
    hidden_rows: List[dict] = []
    public_rows: List[dict] = []
    hidden_coarse_rows: List[dict] = []
    full_coarse_rows: List[dict] = []

    for row in rows:
        task_id = str(row.get("task_id", ""))
        test = str(row.get("test", ""))
        entry_point = str(row.get("entry_point", ""))
        blocks = _assert_blocks(test)
        assert_ids = list(range(len(blocks)))
        public_ids, hidden_ids = _split_assert_ids(assert_ids, task_id, hidden_ratio, seed)

        public_test = _filter_asserts(test, public_ids, blocks) if assert_ids else test
        hidden_test = _filter_asserts(test, hidden_ids, blocks) if assert_ids else test
        hidden_coarse_test = _to_coarse_test(hidden_test, entry_point)
        full_coarse_test = _to_coarse_test(test, entry_point)

        _validate_python_snippet(task_id, "public", public_test)
        _validate_python_snippet(task_id, "hidden", hidden_test)
        _validate_python_snippet(task_id, "hidden_coarse", hidden_coarse_test)
        _validate_python_snippet(task_id, "full_coarse", full_coarse_test)

        common = dict(row)
        common["test_public"] = public_test
        common["test_hidden"] = hidden_test

        public_row = dict(common)
        public_row["test"] = public_test
        public_rows.append(public_row)

        hidden_row = dict(common)
        hidden_row["test"] = hidden_test
        hidden_rows.append(hidden_row)

        hidden_coarse = dict(common)
        hidden_coarse["test"] = hidden_coarse_test
        hidden_coarse_rows.append(hidden_coarse)

        full_coarse = dict(common)
        full_coarse["test"] = full_coarse_test
        full_coarse_rows.append(full_coarse)

    return {
        "hidden": hidden_rows,
        "public": public_rows,
        "hidden_coarse": hidden_coarse_rows,
        "full_coarse": full_coarse_rows,
    }


def build_manifest(rows: Sequence[dict], hidden_ratio: float, seed: int) -> dict:
    assert_counts = [_assert_blocks(str(r.get("test", ""))) for r in rows]
    counts = [len(x) for x in assert_counts]
    n = len(counts) if counts else 1
    return {
        "tasks": len(rows),
        "hidden_ratio": hidden_ratio,
        "seed": seed,
        "assert_count_min": min(counts) if counts else 0,
        "assert_count_max": max(counts) if counts else 0,
        "assert_count_mean": sum(counts) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/humaneval_80.jsonl")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--hidden_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(in_path)
    env_rows = build_env_rows(rows, hidden_ratio=args.hidden_ratio, seed=args.seed)

    stem = in_path.stem
    pct = int(round(100 * args.hidden_ratio))
    hidden_path = out_dir / f"{stem}_hidden{pct}.jsonl"
    public_path = out_dir / f"{stem}_public{100 - pct}.jsonl"
    hidden_coarse_path = out_dir / f"{stem}_hidden{pct}_coarse.jsonl"
    full_coarse_path = out_dir / f"{stem}_coarse.jsonl"
    manifest_path = out_dir / f"{stem}_costly_env_manifest.json"

    _write_jsonl(hidden_path, env_rows["hidden"])
    _write_jsonl(public_path, env_rows["public"])
    _write_jsonl(hidden_coarse_path, env_rows["hidden_coarse"])
    _write_jsonl(full_coarse_path, env_rows["full_coarse"])

    manifest = build_manifest(rows, hidden_ratio=args.hidden_ratio, seed=args.seed)
    manifest.update(
        {
            "in_path": str(in_path),
            "hidden_path": str(hidden_path),
            "public_path": str(public_path),
            "hidden_coarse_path": str(hidden_coarse_path),
            "full_coarse_path": str(full_coarse_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {hidden_path}")
    print(f"wrote {public_path}")
    print(f"wrote {hidden_coarse_path}")
    print(f"wrote {full_coarse_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

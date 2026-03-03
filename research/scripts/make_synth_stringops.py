from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence


VOWELS = set("aeiouAEIOU")


@dataclass
class Op:
    name: str
    desc: str
    fn: Callable[[str], str]


def _reverse() -> Op:
    return Op("reverse", "Reverse the entire string.", lambda s: s[::-1])


def _drop_vowels() -> Op:
    return Op(
        "drop_vowels",
        "Remove all vowels (aeiouAEIOU).",
        lambda s: "".join(ch for ch in s if ch not in VOWELS),
    )


def _replace_space_dash() -> Op:
    return Op(
        "replace_space_dash",
        "Replace spaces with '-'.",
        lambda s: s.replace(" ", "-"),
    )


def _duplicate_chars() -> Op:
    return Op(
        "duplicate_chars",
        "Duplicate every character once (e.g., 'ab' -> 'aabb').",
        lambda s: "".join(ch + ch for ch in s),
    )


def _collapse_runs() -> Op:
    def fn(s: str) -> str:
        out: List[str] = []
        prev = None
        for ch in s:
            if ch != prev:
                out.append(ch)
            prev = ch
        return "".join(out)

    return Op(
        "collapse_runs",
        "Collapse consecutive duplicate characters into one.",
        fn,
    )


def _rotate_left(k: int) -> Op:
    def fn(s: str) -> str:
        if not s:
            return s
        kk = k % len(s)
        return s[kk:] + s[:kk]

    return Op(
        f"rotate_left_{k}",
        f"Rotate the string left by {k} positions (cyclic).",
        fn,
    )


def _every_nth(n: int) -> Op:
    return Op(
        f"every_{n}th",
        f"Keep every {n}th character starting at index 0.",
        lambda s: s[::n],
    )


def _caesar(k: int) -> Op:
    def shift_char(ch: str) -> str:
        if "a" <= ch <= "z":
            return chr((ord(ch) - ord("a") + k) % 26 + ord("a"))
        if "A" <= ch <= "Z":
            return chr((ord(ch) - ord("A") + k) % 26 + ord("A"))
        return ch

    return Op(
        f"caesar_{k}",
        f"Shift alphabetic characters by {k} with Caesar cipher; keep non-letters unchanged.",
        lambda s: "".join(shift_char(ch) for ch in s),
    )


def _ops_catalog() -> List[Op]:
    ops = [
        _reverse(),
        _drop_vowels(),
        _replace_space_dash(),
        _duplicate_chars(),
        _collapse_runs(),
        _rotate_left(1),
        _rotate_left(2),
        _every_nth(2),
        _every_nth(3),
        _caesar(1),
        _caesar(2),
    ]
    return ops


def _rand_input(rnd: random.Random) -> str:
    alphabet = string.ascii_letters + string.digits + "   _-"
    n = rnd.randint(0, 18)
    return "".join(rnd.choice(alphabet) for _ in range(n))


def _safe_samples(rnd: random.Random, n: int) -> List[str]:
    # Include edge cases then randomized strings.
    samples = ["", "a", "AbC", "hello world", "aaabbb", "Z9 z"]
    while len(samples) < n:
        samples.append(_rand_input(rnd))
    return samples[:n]


def _apply_chain(s: str, chain: Sequence[Op]) -> str:
    out = s
    for op in chain:
        out = op.fn(out)
    return out


def _build_prompt(entry_point: str, chain: Sequence[Op]) -> str:
    lines: List[str] = []
    lines.append(f"def {entry_point}(s: str) -> str:")
    lines.append('    """Apply these steps in order:')
    for i, op in enumerate(chain, start=1):
        lines.append(f"    {i}) {op.desc}")
    lines.append('    Return the transformed string.')
    lines.append('    """')
    lines.append("")
    return "\n".join(lines)


def _build_test(
    entry_point: str,
    chain: Sequence[Op],
    samples: Sequence[str],
    rich_feedback: bool,
) -> str:
    lines: List[str] = []
    lines.append("METADATA = {}")
    lines.append("")
    lines.append("def check(candidate):")
    for i, s in enumerate(samples):
        y = _apply_chain(s, chain)
        if rich_feedback:
            lines.append(f"    got_{i} = candidate({s!r})")
            lines.append(
                "    assert got_"
                f"{i} == {y!r}, "
                f"\"input={s!r} expected={y!r} got={{got_{i}!r}}\""
            )
        else:
            lines.append(f"    assert candidate({s!r}) == {y!r}")
    lines.append("")
    lines.append("try:")
    lines.append(f"    check({entry_point})")
    lines.append("except Exception:")
    lines.append("    raise AssertionError('TESTS_FAILED')")
    lines.append("")
    return "\n".join(lines)


def _make_rows(
    num_tasks: int,
    seed: int,
    asserts_per_task: int,
    rich_feedback: bool,
) -> List[dict]:
    rnd = random.Random(seed)
    catalog = _ops_catalog()
    rows: List[dict] = []
    for idx in range(num_tasks):
        entry_point = f"stringop_{idx:03d}"
        # Choose 2-3 distinct ops for each task.
        chain_len = rnd.choice([2, 3])
        chain = rnd.sample(catalog, chain_len)
        samples = _safe_samples(rnd, asserts_per_task)
        row = {
            "task_id": f"SynthStr/{idx}",
            "prompt": _build_prompt(entry_point, chain),
            "test": _build_test(entry_point, chain, samples, rich_feedback=rich_feedback),
            "entry_point": entry_point,
            "meta": {
                "ops": [op.name for op in chain],
            },
        }
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synth_stringops_80.jsonl")
    parser.add_argument("--num_tasks", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--asserts_per_task", type=int, default=12)
    parser.add_argument("--no_rich_feedback", action="store_true")
    args = parser.parse_args()

    rows = _make_rows(
        num_tasks=args.num_tasks,
        seed=args.seed,
        asserts_per_task=args.asserts_per_task,
        rich_feedback=not args.no_rich_feedback,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_path, rows)

    manifest = {
        "dataset": "synth_stringops",
        "num_tasks": args.num_tasks,
        "seed": args.seed,
        "asserts_per_task": args.asserts_per_task,
        "rich_feedback": not args.no_rich_feedback,
        "out": str(out_path),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {out_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

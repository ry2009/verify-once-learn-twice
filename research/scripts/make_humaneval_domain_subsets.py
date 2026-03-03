from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _domain(entry_point: str, prompt: str) -> str:
    text = f"{entry_point}\n{prompt}".lower()
    if any(x in text for x in ["music", "paren", "prefix", "sort_numbers", "parse_"]):
        return "symbolic"
    if any(x in text for x in ["factor", "divisor", "closest", "rescale", "prime"]):
        return "numeric"
    if any(x in text for x in ["palindrome", "xor", "substring", "sequence"]):
        return "string"
    return "general"


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/humaneval_80.jsonl")
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = _read_jsonl(args.in_path)

    by_domain: Dict[str, List[dict]] = {
        "symbolic": [],
        "numeric": [],
        "string": [],
        "general": [],
    }
    for row in rows:
        d = _domain(row.get("entry_point", ""), row.get("prompt", ""))
        by_domain[d].append(row)

    stem = os.path.splitext(os.path.basename(args.in_path))[0]
    for d, drows in by_domain.items():
        out_path = os.path.join(args.out_dir, f"{stem}_{d}.jsonl")
        _write_jsonl(out_path, drows)
        print(f"{d}: {len(drows)} -> {out_path}")


if __name__ == "__main__":
    main()

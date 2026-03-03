#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict


def _iter_event_files(run_group: str):
    group_dir = Path("runs") / run_group
    if not group_dir.exists():
        return
    for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        events = run_dir / "events.jsonl"
        if events.exists():
            yield events


def _summarize_group(run_group: str) -> Dict[str, float]:
    stats = {
        "events": 0,
        "pass_decisions": 0,
        "fail_decisions": 0,
        "raw_pass": 0,
        "raw_fail": 0,
        "flipped": 0,
        "verified_pass": 0,
        "verified_fail": 0,
    }
    for events_path in _iter_event_files(run_group):
        with events_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "judge_oracle":
                    continue
                stats["events"] += 1
                decision = event.get("decision")
                raw = event.get("raw_decision")
                if decision == "PASS":
                    stats["pass_decisions"] += 1
                else:
                    stats["fail_decisions"] += 1
                if raw == "PASS":
                    stats["raw_pass"] += 1
                else:
                    stats["raw_fail"] += 1
                if decision != raw:
                    stats["flipped"] += 1
                if decision == "PASS":
                    if raw == "PASS":
                        stats["verified_pass"] += 1
                    else:
                        stats["verified_fail"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize judge noise stats for oracle-judged runs.")
    parser.add_argument("--run_group", action="append", required=True, help="Run group under runs/")
    parser.add_argument("--out_json", default="runs/judge_noise_summary.json")
    args = parser.parse_args()

    rows = []
    for group in args.run_group:
        stats = _summarize_group(group)
        total = stats["events"] or 1
        rows.append(
            {
                "run_group": group,
                **stats,
                "flip_rate": stats["flipped"] / total,
                "pass_rate": stats["pass_decisions"] / total,
                "verified_pass_rate": (
                    stats["verified_pass"] / stats["pass_decisions"] if stats["pass_decisions"] else 0.0
                ),
            }
        )

    Path(args.out_json).write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    print("| Run Group | Judge Events | Flip Rate | Pass Rate | Verified PASS Rate |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['run_group']} | {row['events']} | {row['flip_rate']:.3f} | "
            f"{row['pass_rate']:.3f} | {row['verified_pass_rate']:.3f} |"
        )


if __name__ == "__main__":
    main()

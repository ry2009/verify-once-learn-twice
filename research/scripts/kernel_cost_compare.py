from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(summary: dict[str, Any], method: str) -> list[dict[str, Any]]:
    rows = [
        r
        for r in summary.get("aggregate", [])
        if str(r.get("method_variant", "")) == method
        and float(r.get("judge_flip_prob", 0.0)) == 0.0
    ]
    rows.sort(key=lambda x: int(x.get("feedback_budget", 0)))
    return rows


def _spc(row: dict[str, Any]) -> float:
    succ = float(row.get("success_mean", 0.0))
    fb = float(row.get("feedback_mean", 0.0))
    return succ / fb if fb > 1e-12 else 0.0


def _min_calls_to_hit(rows: list[dict[str, Any]], target_success: float) -> float | None:
    feasible = [
        float(r.get("feedback_mean", 0.0))
        for r in rows
        if float(r.get("success_mean", 0.0)) >= target_success
    ]
    if not feasible:
        return None
    return min(feasible)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--method_a", default="adaptive_fwb")
    ap.add_argument("--method_b", default="resample_only")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    summary = _load(Path(args.summary_json))
    a_rows = _rows(summary, args.method_a)
    b_rows = _rows(summary, args.method_b)

    lines: list[str] = []
    lines.append("# Direct Cost Comparability")
    lines.append("")
    lines.append(
        f"- Methods: `{args.method_a}` vs `{args.method_b}`"
    )
    lines.append("")
    lines.append("## Success and Efficiency")
    lines.append("")
    lines.append("| Method | Budget | Success | Avg feedback calls | Success / feedback call |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in a_rows + b_rows:
        lines.append(
            f"| {row.get('method_variant')} | {int(row.get('feedback_budget', 0))} | "
            f"{float(row.get('success_mean', 0.0)):.3f} | {float(row.get('feedback_mean', 0.0)):.3f} | {_spc(row):.3f} |"
        )

    lines.append("")
    lines.append("## Feedback Calls To Hit Target Success")
    lines.append("")
    lines.append(
        f"| Target source ({args.method_b}) | Target success | {args.method_b} calls | {args.method_a} min calls | Reduction (x) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    cmp_rows: list[dict[str, Any]] = []
    for row in b_rows:
        target_success = float(row.get("success_mean", 0.0))
        b_calls = float(row.get("feedback_mean", 0.0))
        a_calls = _min_calls_to_hit(a_rows, target_success)
        budget = int(row.get("feedback_budget", 0))
        cmp_row = {
            "target_budget": budget,
            "target_success": target_success,
            "source_calls": b_calls,
            "adaptive_calls": a_calls,
            "reduction_x": (b_calls / a_calls) if (a_calls is not None and a_calls > 1e-12) else None,
        }
        cmp_rows.append(cmp_row)
        if a_calls is None:
            lines.append(
                f"| {args.method_b}@b{budget} | {target_success:.3f} | {b_calls:.3f} | n/a | n/a |"
            )
        else:
            lines.append(
                f"| {args.method_b}@b{budget} | {target_success:.3f} | {b_calls:.3f} | "
                f"{a_calls:.3f} | {cmp_row['reduction_x']:.2f}x |"
            )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))

    if args.out_json:
        payload = {
            "method_a": args.method_a,
            "method_b": args.method_b,
            "summary_json": str(Path(args.summary_json)),
            "rows_a": a_rows,
            "rows_b": b_rows,
            "calls_to_target": cmp_rows,
        }
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

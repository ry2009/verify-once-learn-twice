from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KERNELLLM_REF = {
    "model": "ScalingIntelligence/KernelLLM-8B-Instruct",
    "source": "https://huggingface.co/ScalingIntelligence/KernelLLM-8B-Instruct",
    "kernelbench_l1_pass_at": {"1": 20.2, "10": 51.8, "20": 57.1},
    "training_examples": 25000,
    "reported_gpu_hours": 192,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_method(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in summary.get("aggregate", []):
        mv = str(row.get("method_variant", ""))
        budget = int(row.get("feedback_budget", -1))
        key = f"{mv}@b{budget}"
        out[key] = row
    return out


def _pick(rows: dict[str, dict[str, Any]], method: str, budget: int) -> dict[str, Any] | None:
    r = rows.get(f"{method}@b{budget}")
    if not r:
        return None
    return r


def _success_per_feedback(row: dict[str, Any]) -> float:
    fb = float(row.get("feedback_mean", 0.0))
    succ = float(row.get("success_mean", 0.0))
    if fb <= 1e-12:
        return 0.0
    return succ / fb


def _calls_to_hit(rows: list[dict[str, Any]], target_success: float) -> float | None:
    viable = [
        float(r.get("feedback_mean", 0.0))
        for r in rows
        if float(r.get("success_mean", 0.0)) >= target_success
    ]
    if not viable:
        return None
    return min(viable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    summary = _load_json(Path(args.summary_json))
    rows = _rows_by_method(summary)

    md: list[str] = []
    md.append("# KernelLLM Comparison Card")
    md.append("")
    md.append("## KernelLLM Published Reference")
    md.append("")
    md.append(f"- Model: `{KERNELLLM_REF['model']}`")
    md.append(f"- Source: {KERNELLLM_REF['source']}")
    md.append(
        "- KernelBench L1 pass@1/10/20: "
        f"{KERNELLLM_REF['kernelbench_l1_pass_at']['1']}/"
        f"{KERNELLLM_REF['kernelbench_l1_pass_at']['10']}/"
        f"{KERNELLLM_REF['kernelbench_l1_pass_at']['20']}"
    )
    md.append(
        f"- Reported training footprint: {KERNELLLM_REF['training_examples']} examples, "
        f"{KERNELLLM_REF['reported_gpu_hours']} GPU-hours"
    )
    md.append("")
    md.append("## Our k-Pass Proxy (KernelBench transfer target-12)")
    md.append("")
    md.append("- Note: this is a different task/eval setup than KernelBench-L1 Triton pass@k, so use as cost/efficiency proxy only.")
    md.append("")
    md.append("| Method | Budget | Success | Avg feedback calls | Success / feedback |")
    md.append("|---|---:|---:|---:|---:|")
    resample_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    for b in [1, 5, 10, 20]:
        r = _pick(rows, "resample_only", b)
        if r is not None:
            resample_rows.append(r)
            md.append(
                f"| resample_only | {b} | {float(r.get('success_mean', 0.0)):.3f} | "
                f"{float(r.get('feedback_mean', 0.0)):.3f} | {_success_per_feedback(r):.3f} |"
            )
    for b in [1, 2, 4]:
        a = _pick(rows, "adaptive_fwb", b)
        if a is not None:
            adaptive_rows.append(a)
            md.append(
                f"| adaptive_fwb | {b} | {float(a.get('success_mean', 0.0)):.3f} | "
                f"{float(a.get('feedback_mean', 0.0)):.3f} | {_success_per_feedback(a):.3f} |"
            )
    inf = _pick(rows, "inference_only", 1)
    if inf is not None:
        md.append(
            f"| inference_only | 1 | {float(inf.get('success_mean', 0.0)):.3f} | "
            f"{float(inf.get('feedback_mean', 0.0)):.3f} | {_success_per_feedback(inf):.3f} |"
        )
    md.append("")
    md.append("### Feedback Calls To Hit Resample Targets")
    md.append("")
    md.append("| Target source | Target success | Source feedback calls | Adaptive min calls to hit target | Call reduction (x) |")
    md.append("|---|---:|---:|---:|---:|")
    for r in sorted(resample_rows, key=lambda x: int(x.get("feedback_budget", 0))):
        tgt = float(r.get("success_mean", 0.0))
        src_calls = float(r.get("feedback_mean", 0.0))
        adaptive_calls = _calls_to_hit(adaptive_rows, tgt)
        if adaptive_calls is None or adaptive_calls <= 1e-12:
            md.append(
                f"| resample_only@b{int(r.get('feedback_budget', 0))} | {tgt:.3f} | "
                f"{src_calls:.3f} | n/a | n/a |"
            )
            continue
        reduction = src_calls / adaptive_calls
        md.append(
            f"| resample_only@b{int(r.get('feedback_budget', 0))} | {tgt:.3f} | "
            f"{src_calls:.3f} | {adaptive_calls:.3f} | {reduction:.2f}x |"
        )

    md.append("")
    md.append("## Win Conditions vs KernelLLM")
    md.append("")
    md.append("- Cost win: match/exceed a resample pass@k point with lower feedback budget.")
    md.append("- Speed win: higher success per feedback call (`success_mean / feedback_mean`).")
    md.append("- Size win: use smaller base model while preserving acceptable success.")

    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

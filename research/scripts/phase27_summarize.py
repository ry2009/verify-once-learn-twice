from __future__ import annotations

import json
from pathlib import Path

TAGS = [
    "phase27a_h12_m4c1_s012_b2",
    "phase27b_h12_m2c1_s012_b2",
    "phase27c_h12_m2c2_s012_b2",
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_agg(lb: dict | None, method_variant: str, budget: int = 2) -> dict | None:
    if not lb:
        return None
    for row in lb.get("aggregate", []):
        if row.get("method_variant") == method_variant and int(row.get("feedback_budget", -1)) == budget:
            return row
    return None


def main() -> None:
    rows = []
    for tag in TAGS:
        group = f"verify_{tag}"
        lb = _load(Path(f"runs/{group}/costly_leaderboard_min12.json"))
        pair = _load(Path(f"runs/{group}/paired_adaptive_vs_fixed_k1.json"))

        a = _pick_agg(lb, "adaptive_fwb")
        k1 = _pick_agg(lb, "fixed_k_judge_k1")
        inf = _pick_agg(lb, "inference_only")

        row = {
            "tag": tag,
            "group": group,
            "adaptive_success": float(a.get("success_mean", 0.0)) if a else None,
            "adaptive_feedback": float(a.get("feedback_mean", 0.0)) if a else None,
            "k1_success": float(k1.get("success_mean", 0.0)) if k1 else None,
            "k1_feedback": float(k1.get("feedback_mean", 0.0)) if k1 else None,
            "inference_success": float(inf.get("success_mean", 0.0)) if inf else None,
            "delta_success": (
                float(pair.get("delta_success", {}).get("mean", 0.0)) if pair else None
            ),
            "delta_feedback": (
                float(pair.get("delta_feedback", {}).get("mean", 0.0)) if pair else None
            ),
            "paired_p_signflip": (
                float(pair.get("delta_success", {}).get("p_signflip_mean_greater", 1.0))
                if pair
                else None
            ),
        }
        rows.append(row)

    out_dir = Path("runs/phase27_tune")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    md_lines = ["# Phase27 Tune Summary", ""]
    md_lines.append("| tag | adaptive_success | k1_success | delta_success | delta_feedback | p_signflip |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        def f(x):
            return "-" if x is None else f"{x:.4f}"
        md_lines.append(
            f"| {r['tag']} | {f(r['adaptive_success'])} | {f(r['k1_success'])} | {f(r['delta_success'])} | {f(r['delta_feedback'])} | {f(r['paired_p_signflip'])} |"
        )

    ready = [r for r in rows if r["delta_success"] is not None]
    if ready:
        best = max(ready, key=lambda x: x["delta_success"])
        md_lines.append("")
        md_lines.append(
            f"Best available delta_success: {best['tag']} ({best['delta_success']:+.4f})"
        )

    out_md = out_dir / "summary.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Optional


def _load(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_any(paths: list[str]) -> Optional[dict]:
    for path in paths:
        p = Path(path)
        if "*" in path:
            cands = sorted(glob.glob(path))
            if cands:
                p = Path(cands[-1])
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def _pick(lb: dict | None, method: str, budget: int, noise: float | None = None) -> Optional[dict]:
    if not lb:
        return None
    for row in lb.get("aggregate", []):
        if row.get("method_variant") != method:
            continue
        if int(row.get("feedback_budget", -1)) != budget:
            continue
        if noise is not None and abs(float(row.get("judge_flip_prob", 0.0)) - noise) > 1e-12:
            continue
        return row
    return None


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def _fmt_row(v: dict | None) -> str:
    if not v:
        return "-"
    s = _fmt(v.get("success_mean"))
    r = int(v.get("runs", 0))
    return f"{s} (n={r})"


def main() -> None:
    lines = ["# Live Status", ""]

    lb22h12 = _load("runs/verify_phase22_h12_guardreplay_core_s01234/costly_leaderboard_min12.json")
    if lb22h12:
        a = _pick(lb22h12, "adaptive_fwb", 2)
        k1 = _pick(lb22h12, "fixed_k_judge_k1", 2)
        f1 = _pick(lb22h12, "fixed_k_fwb_k1", 2)
        lines += [
            "## Phase22 H12 (complete)",
            f"- adaptive success={_fmt(a.get('success_mean') if a else None)} feedback={_fmt(a.get('feedback_mean') if a else None)}",
            f"- fixed_k_judge_k1 success={_fmt(k1.get('success_mean') if k1 else None)}",
            f"- fixed_k_fwb_k1 success={_fmt(f1.get('success_mean') if f1 else None)}",
            "",
        ]

    lb27a = _load("runs/verify_phase27a_h12_m4c1_s012_b2/costly_leaderboard_partial_min12.json")
    if lb27a:
        a = _pick(lb27a, "adaptive_fwb", 2)
        k1 = _pick(lb27a, "fixed_k_judge_k1", 2)
        lines += [
            "## Phase27a tune (complete)",
            f"- adaptive success={_fmt(a.get('success_mean') if a else None)}",
            f"- fixed_k_judge_k1 success={_fmt(k1.get('success_mean') if k1 else None)}",
            "",
        ]

    lb28 = _load("runs/phase28_combined/costly_leaderboard_min12.json")
    if lb28:
        a = _pick(lb28, "adaptive_fwb", 1)
        k1 = _pick(lb28, "fixed_k_judge_k1", 1)
        k2 = _pick(lb28, "fixed_k_judge_k2", 1)
        k4 = _pick(lb28, "fixed_k_judge_k4", 1)
        f1 = _pick(lb28, "fixed_k_fwb_k1", 1)
        inf = _pick(lb28, "inference_only", 1)
        lines += [
            "## Phase28 budget=1 (combined, live)",
            f"- adaptive success={_fmt(a.get('success_mean') if a else None)}",
            f"- fixed_k_judge_k1 success={_fmt(k1.get('success_mean') if k1 else None)}",
            f"- fixed_k_judge_k2 success={_fmt(k2.get('success_mean') if k2 else None)}",
            f"- fixed_k_judge_k4 success={_fmt(k4.get('success_mean') if k4 else None)}",
            f"- fixed_k_fwb_k1 success={_fmt(f1.get('success_mean') if f1 else None)}",
            f"- inference_only success={_fmt(inf.get('success_mean') if inf else None)}",
            "",
        ]

    lb23 = _load("runs/verify_phase23_h10_budget24_core_s01234/costly_leaderboard_partial_min10.json")
    if lb23:
        a2 = _pick(lb23, "adaptive_fwb", 2)
        k12 = _pick(lb23, "fixed_k_judge_k1", 2)
        lines += [
            "## Phase23 H10 budget=2 (partial)",
            f"- adaptive success={_fmt(a2.get('success_mean') if a2 else None)}",
            f"- fixed_k_judge_k1 success={_fmt(k12.get('success_mean') if k12 else None)}",
            "",
        ]

    lb26 = _load("runs/verify_phase26_h12_noise_budget24_s01234/costly_leaderboard_partial_min12.json")
    if lb26:
        a40 = _pick(lb26, "adaptive_fwb", 4, noise=0.0)
        k140 = _pick(lb26, "fixed_k_judge_k1", 4, noise=0.0)
        a41 = _pick(lb26, "adaptive_fwb", 4, noise=0.1)
        k141 = _pick(lb26, "fixed_k_judge_k1", 4, noise=0.1)
        lines += [
            "## Phase26 H12 noise robustness (partial)",
            f"- b=4 noise=0.0 adaptive={_fmt(a40.get('success_mean') if a40 else None)} fixed_k_judge_k1={_fmt(k140.get('success_mean') if k140 else None)}",
            f"- b=4 noise=0.1 adaptive={_fmt(a41.get('success_mean') if a41 else None)} fixed_k_judge_k1={_fmt(k141.get('success_mean') if k141 else None)}",
            "",
        ]

    lb30 = _load("runs/verify_phase30_mbpp20_budget1_s0123/costly_leaderboard_min20.json")
    if not lb30:
        lb30 = _load("runs/verify_phase30b_mbpp20_budget1_s0123_nors/costly_leaderboard_min20.json")
    if not lb30:
        lb30 = _load("runs/verify_phase30b_mbpp20_budget1_s0123_nors/costly_leaderboard_partial_min10.json")
    if lb30:
        a = _pick(lb30, "adaptive_fwb", 1)
        k1 = _pick(lb30, "fixed_k_judge_k1", 1)
        k2 = _pick(lb30, "fixed_k_judge_k2", 1)
        k4 = _pick(lb30, "fixed_k_judge_k4", 1)
        f1 = _pick(lb30, "fixed_k_fwb_k1", 1)
        inf = _pick(lb30, "inference_only", 1)
        lines += [
            "## Phase30 MBPP20 budget=1 (live)",
            f"- adaptive success={_fmt_row(a)}",
            f"- fixed_k_judge_k1 success={_fmt_row(k1)}",
            f"- fixed_k_judge_k2 success={_fmt_row(k2)}",
            f"- fixed_k_judge_k4 success={_fmt_row(k4)}",
            f"- fixed_k_fwb_k1 success={_fmt_row(f1)}",
            f"- inference_only success={_fmt_row(inf)}",
            "",
        ]

    lb31n = _load("runs/verify_phase31_numeric_budget1_s01234567/costly_leaderboard_min10.json")
    lb31s = _load("runs/verify_phase31_string_budget1_s01234567/costly_leaderboard_min8.json")
    lb31y = _load("runs/verify_phase31_symbolic_budget1_s01234567/costly_leaderboard_min7.json")
    if lb31n or lb31s or lb31y:
        lines += ["## Phase31 domain transfer budget=1 (live)"]
        for domain, lb in [("numeric", lb31n), ("string", lb31s), ("symbolic", lb31y)]:
            if not lb:
                lines.append(f"- {domain}: pending")
                continue
            a = _pick(lb, "adaptive_fwb", 1)
            k1 = _pick(lb, "fixed_k_judge_k1", 1)
            lines.append(
                f"- {domain}: adaptive={_fmt_row(a)} "
                f"fixed_k_judge_k1={_fmt_row(k1)}"
            )
        lines.append("")

    lb32 = _load("runs/verify_phase32_synth20_budget1_s0123_nors/costly_leaderboard_min20.json")
    if not lb32:
        lb32 = _load("runs/verify_phase32_synth20_budget1_s0123_nors/costly_leaderboard_partial_min10.json")
    if lb32:
        a = _pick(lb32, "adaptive_fwb", 1)
        k1 = _pick(lb32, "fixed_k_judge_k1", 1)
        k2 = _pick(lb32, "fixed_k_judge_k2", 1)
        k4 = _pick(lb32, "fixed_k_judge_k4", 1)
        f1 = _pick(lb32, "fixed_k_fwb_k1", 1)
        inf = _pick(lb32, "inference_only", 1)
        lines += [
            "## Phase32 Synth20 budget=1 (live)",
            f"- adaptive success={_fmt_row(a)}",
            f"- fixed_k_judge_k1 success={_fmt_row(k1)}",
            f"- fixed_k_judge_k2 success={_fmt_row(k2)}",
            f"- fixed_k_judge_k4 success={_fmt_row(k4)}",
            f"- fixed_k_fwb_k1 success={_fmt_row(f1)}",
            f"- inference_only success={_fmt_row(inf)}",
            "",
        ]

    lb33 = _load_any(
        [
            "runs/verify_phase33_hardset_h12pair_budget1_s1223/costly_leaderboard_min*.json",
            "runs/verify_phase33_hardset_numeric_budget1_s01234567/costly_leaderboard_min*.json",
        ]
    )
    if lb33:
        a = _pick(lb33, "adaptive_fwb", 1)
        k1 = _pick(lb33, "fixed_k_judge_k1", 1)
        k2 = _pick(lb33, "fixed_k_judge_k2", 1)
        k4 = _pick(lb33, "fixed_k_judge_k4", 1)
        f1 = _pick(lb33, "fixed_k_fwb_k1", 1)
        inf = _pick(lb33, "inference_only", 1)
        lines += [
            "## Phase33 Mined Hardset budget=1 (live)",
            f"- adaptive success={_fmt_row(a)}",
            f"- fixed_k_judge_k1 success={_fmt_row(k1)}",
            f"- fixed_k_judge_k2 success={_fmt_row(k2)}",
            f"- fixed_k_judge_k4 success={_fmt_row(k4)}",
            f"- fixed_k_fwb_k1 success={_fmt_row(f1)}",
            f"- inference_only success={_fmt_row(inf)}",
            "",
        ]

    lb34 = _load("runs/verify_phase34_h12_budget1_pairwise_s1223/costly_leaderboard_min12.json")
    if lb34:
        a = _pick(lb34, "adaptive_fwb", 1)
        k1 = _pick(lb34, "fixed_k_judge_k1", 1)
        lines += [
            "## Phase34 H12 pairwise power budget=1 (live)",
            f"- adaptive success={_fmt_row(a)}",
            f"- fixed_k_judge_k1 success={_fmt_row(k1)}",
            "",
        ]

    out = Path("runs/live_status.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

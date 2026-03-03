from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DOMAIN_CFG: list[dict[str, Any]] = [
    {
        "domain": "Phase28 H12 Combined",
        "run_group": "phase28_combined",
        "expected_runs": 12,
        "paired_candidates": ["paired_adaptive_vs_fixedkj1_b1.json"],
    },
    {
        "domain": "Phase30b MBPP20",
        "run_group": "verify_phase30b_mbpp20_budget1_s0123_nors",
        "expected_runs": 4,
        "paired_candidates": [
            "paired_adaptive_vs_fixedkj1_b1.json",
            "paired_adaptive_vs_fixedkj1_b1_partial_min10.json",
        ],
    },
    {
        "domain": "Phase31 Numeric",
        "run_group": "verify_phase31_numeric_budget1_s01234567",
        "expected_runs": 8,
        "paired_candidates": ["paired_adaptive_vs_fixedkj1_b1.json"],
    },
    {
        "domain": "Phase31 String",
        "run_group": "verify_phase31_string_budget1_s01234567",
        "expected_runs": 8,
        "paired_candidates": ["paired_adaptive_vs_fixedkj1_b1.json"],
    },
    {
        "domain": "Phase31 Symbolic",
        "run_group": "verify_phase31_symbolic_budget1_s01234567",
        "expected_runs": 8,
        "paired_candidates": ["paired_adaptive_vs_fixedkj1_b1.json"],
    },
    {
        "domain": "Phase32 Synth20",
        "run_group": "verify_phase32_synth20_budget1_s0123_nors",
        "expected_runs": 4,
        "paired_candidates": [
            "paired_adaptive_vs_fixedkj1_b1.json",
            "paired_adaptive_vs_fixedkj1_b1_partial_min10.json",
        ],
    },
    {
        "domain": "Phase33 Mined Hardset",
        "run_group": "verify_phase33_hardset_h12pair_budget1_s1223",
        "expected_runs": 12,
        "paired_candidates": ["paired_adaptive_vs_fixedkj1_b1.json"],
    },
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v: float | None, d: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{d}f}"


def _find_scorecard_row(rows: list[dict[str, Any]], domain: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("domain")) == domain:
            return row
    return None


def _find_paired(run_group: str, candidates: list[str]) -> tuple[Path | None, dict[str, Any] | None]:
    for name in candidates:
        p = Path("runs") / run_group / name
        obj = _load_json(p)
        if obj is not None:
            return p, obj
    return None, None


def _domain_eval(score_row: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "domain": cfg["domain"],
        "run_group": cfg["run_group"],
        "expected_runs": int(cfg["expected_runs"]),
        "adaptive_success": None,
        "baseline_success": None,
        "delta_success": None,
        "success_ratio": None,
        "adaptive_runs": None,
        "baseline_runs": None,
        "is_complete": False,
        "paired_file": None,
        "paired_p": None,
        "paired_ci_low": None,
        "paired_ci_high": None,
        "paired_pairs": None,
        "paired_mean_delta": None,
        "strong_win": False,
        "interim_strong": False,
        "positive": False,
        "negative": False,
    }

    if score_row is not None:
        a = score_row.get("adaptive_success")
        b = score_row.get("baseline_success")
        d = score_row.get("delta_success")
        na = score_row.get("adaptive_runs")
        nb = score_row.get("baseline_runs")
        out["adaptive_success"] = None if a is None else float(a)
        out["baseline_success"] = None if b is None else float(b)
        out["delta_success"] = None if d is None else float(d)
        out["adaptive_runs"] = None if na is None else int(na)
        out["baseline_runs"] = None if nb is None else int(nb)
        if out["baseline_success"] is not None and out["baseline_success"] > 0:
            out["success_ratio"] = float(out["adaptive_success"]) / float(out["baseline_success"])
        if out["adaptive_runs"] is not None and out["baseline_runs"] is not None:
            out["is_complete"] = min(int(out["adaptive_runs"]), int(out["baseline_runs"])) >= int(cfg["expected_runs"])
        if out["delta_success"] is not None:
            out["positive"] = out["delta_success"] > 0
            out["negative"] = out["delta_success"] < 0

    p_path, p_obj = _find_paired(cfg["run_group"], list(cfg["paired_candidates"]))
    if p_path is not None and p_obj is not None:
        out["paired_file"] = str(p_path)
        ds = p_obj.get("delta_success", {})
        ci = ds.get("ci", [None, None])
        out["paired_p"] = None if ds.get("p_signflip_mean_greater") is None else float(ds["p_signflip_mean_greater"])
        out["paired_ci_low"] = None if ci[0] is None else float(ci[0])
        out["paired_ci_high"] = None if ci[1] is None else float(ci[1])
        out["paired_pairs"] = None if p_obj.get("pairs") is None else int(p_obj.get("pairs"))
        out["paired_mean_delta"] = None if ds.get("mean") is None else float(ds.get("mean"))

    # Strong win criterion aligned with user's stated bar: >=1.3x success at same budget.
    if (
        out["is_complete"]
        and out["success_ratio"] is not None
        and out["success_ratio"] >= 1.3
        and out["paired_p"] is not None
        and out["paired_p"] <= 0.10
        and out["paired_ci_low"] is not None
        and out["paired_ci_low"] > 0.0
    ):
        out["strong_win"] = True

    # Interim strong signal: allow incomplete domains to be surfaced when
    # effect size and paired significance are already strong on available seeds.
    if (
        out["success_ratio"] is not None
        and out["success_ratio"] >= 1.3
        and out["paired_p"] is not None
        and out["paired_p"] <= 0.05
        and out["paired_ci_low"] is not None
        and out["paired_ci_low"] > 0.0
        and out["paired_pairs"] is not None
        and out["paired_pairs"] >= 5
    ):
        out["interim_strong"] = True

    return out


def _campaign_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if bool(r.get("is_complete"))]
    complete_positive = [r for r in complete if bool(r.get("positive"))]
    complete_negative = [r for r in complete if bool(r.get("negative"))]
    strong = [r for r in complete if bool(r.get("strong_win"))]
    interim_strong = [r for r in rows if bool(r.get("interim_strong"))]

    partial_positive = [
        r
        for r in rows
        if (not bool(r.get("is_complete"))) and bool(r.get("positive"))
    ]

    # Domain-targeted publish readiness gate:
    # (a) at least one strong complete domain win,
    # (b) at least 3 complete domains overall,
    # (c) complete positives >= complete negatives.
    publish_ready = (
        len(strong) >= 1
        and len(complete) >= 3
        and len(complete_positive) >= len(complete_negative)
    )

    if publish_ready:
        decision = "publish_ready_candidate"
        rationale = "At least one complete strong win domain with sufficient complete cross-domain coverage."
    elif len(interim_strong) >= 1:
        decision = "strong_signal_incomplete"
        rationale = "At least one domain shows strong paired signal but completion coverage gates are not yet satisfied."
    elif len(partial_positive) >= 2:
        decision = "promising_incomplete"
        rationale = "Positive signals exist but coverage/strength gates are not yet satisfied."
    else:
        decision = "not_ready"
        rationale = "Insufficient completed evidence for domain-targeted SOTA claim."

    return {
        "decision": decision,
        "rationale": rationale,
        "counts": {
            "domains_total": len(rows),
            "domains_complete": len(complete),
            "domains_complete_positive": len(complete_positive),
            "domains_complete_negative": len(complete_negative),
            "domains_strong_win": len(strong),
            "domains_interim_strong": len(interim_strong),
            "domains_partial_positive": len(partial_positive),
        },
        "strong_win_domains": [r["domain"] for r in strong],
        "interim_strong_domains": [r["domain"] for r in interim_strong],
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Publish Readiness")
    lines.append("")
    c = payload["campaign"]
    lines.append(f"- decision: `{c['decision']}`")
    lines.append(f"- rationale: {c['rationale']}")
    lines.append(
        f"- counts: complete={c['counts']['domains_complete']}, "
        f"complete_positive={c['counts']['domains_complete_positive']}, "
        f"complete_negative={c['counts']['domains_complete_negative']}, "
        f"strong={c['counts']['domains_strong_win']}, "
        f"interim_strong={c['counts']['domains_interim_strong']}"
    )
    lines.append("")
    lines.append("| Domain | Complete | A | B | Delta | Ratio | p(one-sided) | CI low | Strong | Interim |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in payload["domains"]:
        lines.append(
            f"| {r['domain']} | {('yes' if r['is_complete'] else 'no')} | "
            f"{_fmt(r['adaptive_success'])} | {_fmt(r['baseline_success'])} | {_fmt(r['delta_success'])} | "
            f"{_fmt(r['success_ratio'], d=3)} | {_fmt(r['paired_p'], d=3)} | {_fmt(r['paired_ci_low'])} | "
            f"{('yes' if r['strong_win'] else 'no')} | {('yes' if r['interim_strong'] else 'no')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _latex(payload: dict[str, Any]) -> str:
    c = payload["campaign"]
    decision_tex = str(c["decision"]).replace("_", "\\_")
    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Domain & Complete & A & B & $\\Delta$ & Ratio & $p$ & CI$_{low}$ & Strong & Interim \\\\")
    lines.append("\\midrule")
    for r in payload["domains"]:
        lines.append(
            f"{str(r['domain']).replace('_', '\\_')} & "
            f"{('yes' if r['is_complete'] else 'no')} & "
            f"{_fmt(r['adaptive_success'])} & "
            f"{_fmt(r['baseline_success'])} & "
            f"{_fmt(r['delta_success'])} & "
            f"{_fmt(r['success_ratio'], d=3)} & "
            f"{_fmt(r['paired_p'], d=3)} & "
            f"{_fmt(r['paired_ci_low'])} & "
            f"{('yes' if r['strong_win'] else 'no')} & "
            f"{('yes' if r['interim_strong'] else 'no')} \\\\"
        )
    lines.append("\\midrule")
    lines.append(
        f"\\multicolumn{{10}}{{l}}{{Decision: \\texttt{{{decision_tex}}}; "
        f"complete={c['counts']['domains_complete']}, strong={c['counts']['domains_strong_win']}, "
        f"interim={c['counts']['domains_interim_strong']}}} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard_json", default="runs/live_costly_scorecard.json")
    parser.add_argument("--out_json", default="runs/publish_readiness.json")
    parser.add_argument("--out_md", default="runs/publish_readiness.md")
    parser.add_argument("--out_tex", default="paper/tables/publish_readiness.tex")
    args = parser.parse_args()

    scorecard = _load_json(Path(args.scorecard_json))
    rows = list(scorecard.get("rows", [])) if scorecard else []
    domain_rows = [_domain_eval(_find_scorecard_row(rows, cfg["domain"]), cfg) for cfg in DOMAIN_CFG]
    campaign = _campaign_decision(domain_rows)

    payload = {
        "domains": domain_rows,
        "campaign": campaign,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(payload), encoding="utf-8")

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(_latex(payload), encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()

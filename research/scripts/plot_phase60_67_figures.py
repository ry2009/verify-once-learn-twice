from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

STYLE = {
    "paper_bg": "#ffffff",
    "plot_bg": "#ffffff",
    "font": "#0f172a",
    "grid": "#d6dee8",
    "axis": "#64748b",
}

METHODS = [
    ("adaptive_fwb", "Adaptive", "#1f77b4", "diamond"),
    ("resample_only", "Resample-only", "#ff7f0e", "circle"),
    ("inference_only", "Inference-only", "#7f8c8d", "x"),
    ("fixed_k_judge_k2", "Fixed-k+Judge (k=2)", "#d62728", "square"),
]


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rows_by_method(payload: dict[str, Any], method: str) -> list[dict[str, Any]]:
    rows = [r for r in payload.get("aggregate", []) if str(r.get("method_variant", "")) == method]
    rows.sort(key=lambda r: int(r.get("feedback_budget", 0)))
    return rows


def _style_axes(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=STYLE["paper_bg"],
        plot_bgcolor=STYLE["plot_bg"],
        font=dict(color=STYLE["font"], size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#d1d5db",
            borderwidth=1,
        ),
        margin=dict(l=70, r=30, t=100, b=70),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=STYLE["grid"],
        linecolor=STYLE["axis"],
        mirror=True,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=STYLE["grid"],
        linecolor=STYLE["axis"],
        mirror=True,
        zeroline=False,
    )


def _save(fig: go.Figure, out_prefix: Path, width: int, height: int) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_prefix.with_suffix(".html")), include_plotlyjs="cdn")
    try:
        fig.write_image(str(out_prefix.with_suffix(".png")), width=width, height=height, scale=2)
        fig.write_image(str(out_prefix.with_suffix(".svg")), width=width, height=height)
    except Exception as exc:
        print(f"PNG/SVG export skipped for {out_prefix.name}: {exc}")


def _add_frontier_trace(
    fig: go.Figure,
    payload: dict[str, Any],
    row: int,
    col: int,
    x_key: str,
    show_legend: bool,
) -> None:
    for method, label, color, symbol in METHODS:
        rows = _rows_by_method(payload, method)
        if not rows:
            continue
        xs = [float(r[x_key]) for r in rows]
        ys = [float(r["success_mean"]) for r in rows]
        txt = [f"b={int(r['feedback_budget'])}" for r in rows]
        mode = "markers" if len(rows) == 1 else "lines+markers+text"
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode=mode,
                text=txt if "text" in mode else None,
                textposition="top center",
                name=label,
                showlegend=show_legend,
                line=dict(color=color, width=2.8),
                marker=dict(color=color, size=10, symbol=symbol),
            ),
            row=row,
            col=col,
        )


def plot_frontiers(
    t50_8b: dict[str, Any],
    t50_3b: dict[str, Any],
    t100_8b: dict[str, Any],
    out_prefix: Path,
) -> None:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Target-50 (8B)", "Target-50 (3B)", "Target-100 (8B)"),
    )
    _add_frontier_trace(fig, t50_8b, row=1, col=1, x_key="feedback_mean", show_legend=True)
    _add_frontier_trace(fig, t50_3b, row=1, col=2, x_key="feedback_mean", show_legend=False)
    _add_frontier_trace(fig, t100_8b, row=1, col=3, x_key="feedback_mean", show_legend=False)
    fig.update_layout(
        title="KernelBench Phase60-65: Success vs Feedback Budget Cost",
        width=1700,
        height=620,
    )
    fig.update_xaxes(title_text="Avg feedback calls / task", row=1, col=1)
    fig.update_xaxes(title_text="Avg feedback calls / task", row=1, col=2)
    fig.update_xaxes(title_text="Avg feedback calls / task", row=1, col=3)
    fig.update_yaxes(title_text="Success rate", row=1, col=1, range=[0.0, 0.75])
    fig.update_yaxes(range=[0.0, 0.75], row=1, col=2)
    fig.update_yaxes(range=[0.0, 0.75], row=1, col=3)
    _style_axes(fig)
    _save(fig, out_prefix, 1700, 620)


def plot_full_cost(
    t50_8b: dict[str, Any],
    t50_3b: dict[str, Any],
    t100_8b: dict[str, Any],
    out_prefix: Path,
) -> None:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Target-50 (8B)", "Target-50 (3B)", "Target-100 (8B)"),
    )
    _add_frontier_trace(fig, t50_8b, row=1, col=1, x_key="test_mean", show_legend=True)
    _add_frontier_trace(fig, t50_3b, row=1, col=2, x_key="test_mean", show_legend=False)
    _add_frontier_trace(fig, t100_8b, row=1, col=3, x_key="test_mean", show_legend=False)
    fig.update_layout(
        title="KernelBench Phase60-65: Success vs Full External Cost (test_calls)",
        width=1700,
        height=620,
    )
    fig.update_xaxes(title_text="Avg test calls / task", row=1, col=1)
    fig.update_xaxes(title_text="Avg test calls / task", row=1, col=2)
    fig.update_xaxes(title_text="Avg test calls / task", row=1, col=3)
    fig.update_yaxes(title_text="Success rate", row=1, col=1, range=[0.0, 0.75])
    fig.update_yaxes(range=[0.0, 0.75], row=1, col=2)
    fig.update_yaxes(range=[0.0, 0.75], row=1, col=3)
    _style_axes(fig)
    _save(fig, out_prefix, 1700, 620)


def _find_row(payload: dict[str, Any], method: str, budget: int) -> dict[str, Any] | None:
    for row in payload.get("aggregate", []):
        if str(row.get("method_variant", "")) == method and int(row.get("feedback_budget", -1)) == budget:
            return row
    return None


def plot_adaptive_deltas(
    t50_8b: dict[str, Any],
    t50_3b: dict[str, Any],
    t100_8b: dict[str, Any],
    out_prefix: Path,
) -> None:
    datasets = [
        ("T50-8B", t50_8b),
        ("T50-3B", t50_3b),
        ("T100-8B", t100_8b),
    ]
    budgets = [1, 2, 4]
    colors = {"T50-8B": "#1f77b4", "T50-3B": "#2ca02c", "T100-8B": "#9467bd"}

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Adaptive - Resample Success Delta", "Adaptive - Resample Full-Cost Delta"),
    )
    for label, payload in datasets:
        x = []
        ds = []
        dt = []
        for b in budgets:
            a = _find_row(payload, "adaptive_fwb", b)
            r = _find_row(payload, "resample_only", b)
            if not a or not r:
                continue
            x.append(b)
            ds.append(float(a["success_mean"]) - float(r["success_mean"]))
            dt.append(float(a["test_mean"]) - float(r["test_mean"]))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ds,
                mode="lines+markers",
                name=label,
                marker=dict(size=10, color=colors[label]),
                line=dict(width=2.5, color=colors[label]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=dt,
                mode="lines+markers",
                name=label,
                showlegend=False,
                marker=dict(size=10, color=colors[label]),
                line=dict(width=2.5, color=colors[label]),
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8", row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8", row=1, col=2)
    fig.update_layout(
        title="Adaptive Advantage Profile Across Budgets (Phase60-65)",
        width=1400,
        height=620,
    )
    fig.update_xaxes(title_text="Feedback budget", tickvals=[1, 2, 4], row=1, col=1)
    fig.update_xaxes(title_text="Feedback budget", tickvals=[1, 2, 4], row=1, col=2)
    fig.update_yaxes(title_text="Success delta", row=1, col=1)
    fig.update_yaxes(title_text="Test-call delta", row=1, col=2)
    _style_axes(fig)
    _save(fig, out_prefix, 1400, 620)


def plot_phase66_vs_base(phase66: dict[str, Any], t50_8b: dict[str, Any], out_prefix: Path) -> None:
    budgets = [2, 4]
    base_s: list[float] = []
    tuned_s: list[float] = []
    base_t: list[float] = []
    tuned_t: list[float] = []
    for b in budgets:
        base = _find_row(t50_8b, "adaptive_fwb", b)
        tuned = _find_row(phase66, "adaptive_fwb", b)
        base_s.append(float(base["success_mean"]) if base else math.nan)
        tuned_s.append(float(tuned["success_mean"]) if tuned else math.nan)
        base_t.append(float(base["test_mean"]) if base else math.nan)
        tuned_t.append(float(tuned["test_mean"]) if tuned else math.nan)

    x = [f"b={b}" for b in budgets]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Success (base adaptive vs tuned adaptive)", "Full cost test_calls (lower better)"),
    )
    fig.add_trace(go.Bar(x=x, y=base_s, name="Base adaptive (phase60/61)", marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=tuned_s, name="Tuned adaptive (phase66)", marker_color="#17becf"), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=base_t, name="Base adaptive (phase60/61)", marker_color="#1f77b4", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=x, y=tuned_t, name="Tuned adaptive (phase66)", marker_color="#17becf", showlegend=False), row=1, col=2)
    fig.update_layout(
        barmode="group",
        title="Phase66 Adaptive Accuracy Push vs Phase60/61 Adaptive Baseline",
        width=1350,
        height=620,
    )
    fig.update_yaxes(title_text="Success rate", range=[0.0, 0.8], row=1, col=1)
    fig.update_yaxes(title_text="Avg test calls", row=1, col=2)
    _style_axes(fig)
    _save(fig, out_prefix, 1350, 620)


def _phase67_mode_rows(phase67: dict[str, Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in phase67.get("runs", []):
        name = str(row.get("run_name", ""))
        budget = int(row.get("feedback_budget", -1))
        if budget <= 0:
            continue
        if "r1always" in name:
            mode = "always"
        elif "r1fail" in name:
            mode = "fail_only"
        else:
            mode = "off"
        buckets.setdefault((budget, mode), []).append(row)

    out: list[dict[str, Any]] = []
    for (budget, mode), rows in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        success = [float(r.get("success_rate", 0.0)) for r in rows]
        feedback = [float(r.get("avg_feedback_calls", 0.0)) for r in rows]
        tests = [float(r.get("avg_test_calls", 0.0)) for r in rows]
        out.append(
            {
                "budget": budget,
                "mode": mode,
                "runs": len(rows),
                "success_mean": stats.mean(success) if success else 0.0,
                "success_sd": stats.pstdev(success) if len(success) > 1 else 0.0,
                "feedback_mean": stats.mean(feedback) if feedback else 0.0,
                "test_mean": stats.mean(tests) if tests else 0.0,
            }
        )
    return out


def plot_phase67_modes(phase67: dict[str, Any], out_prefix: Path, csv_out: Path) -> list[dict[str, Any]]:
    rows = _phase67_mode_rows(phase67)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8") as f:
        f.write("budget,mode,runs,success_mean,success_sd,feedback_mean,test_mean\n")
        for r in rows:
            f.write(
                f"{r['budget']},{r['mode']},{r['runs']},{r['success_mean']:.6f},"
                f"{r['success_sd']:.6f},{r['feedback_mean']:.6f},{r['test_mean']:.6f}\n"
            )

    mode_order = ["off", "always", "fail_only"]
    colors = {2: "#1f77b4", 4: "#ff7f0e"}

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Success (higher better)", "Feedback calls (lower better)", "Full cost test_calls (lower better)"),
    )
    for budget in [2, 4]:
        vals = {r["mode"]: r for r in rows if int(r["budget"]) == budget}
        xs = mode_order
        success = [vals.get(m, {}).get("success_mean", math.nan) for m in xs]
        sd = [vals.get(m, {}).get("success_sd", 0.0) for m in xs]
        feedback = [vals.get(m, {}).get("feedback_mean", math.nan) for m in xs]
        tests = [vals.get(m, {}).get("test_mean", math.nan) for m in xs]

        fig.add_trace(
            go.Bar(
                x=xs,
                y=success,
                error_y=dict(type="data", array=sd, visible=True),
                name=f"b={budget}",
                marker_color=colors[budget],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=xs,
                y=feedback,
                name=f"b={budget}",
                showlegend=False,
                marker_color=colors[budget],
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=xs,
                y=tests,
                name=f"b={budget}",
                showlegend=False,
                marker_color=colors[budget],
            ),
            row=1,
            col=3,
        )
    fig.update_layout(
        barmode="group",
        title="Phase67 T1-L2-R1 Ablation (off vs always vs fail_only)",
        width=1700,
        height=620,
    )
    fig.update_yaxes(title_text="Success rate", range=[0.0, 0.8], row=1, col=1)
    fig.update_yaxes(title_text="Avg feedback calls", row=1, col=2)
    fig.update_yaxes(title_text="Avg test calls", row=1, col=3)
    _style_axes(fig)
    _save(fig, out_prefix, 1700, 620)
    return rows


def write_summary(
    out_path: Path,
    t50_8b: dict[str, Any],
    t50_3b: dict[str, Any],
    t100_8b: dict[str, Any],
    phase66: dict[str, Any],
    phase67_rows: list[dict[str, Any]],
) -> None:
    def _triple(payload: dict[str, Any], budget: int) -> tuple[float, float, float]:
        a = _find_row(payload, "adaptive_fwb", budget)
        r = _find_row(payload, "resample_only", budget)
        i = _find_row(payload, "inference_only", 1)
        return (
            float(a["success_mean"]) if a else math.nan,
            float(r["success_mean"]) if r else math.nan,
            float(i["success_mean"]) if i else math.nan,
        )

    lines = [
        "# Phase60-67 KernelBench Figure Pack Summary",
        "",
        "## Success snapshots (Adaptive / Resample / Inference)",
        "",
    ]
    for name, payload in [("Target-50 8B", t50_8b), ("Target-50 3B", t50_3b), ("Target-100 8B", t100_8b)]:
        a1, r1, i1 = _triple(payload, 1)
        a2, r2, _ = _triple(payload, 2)
        a4, r4, _ = _triple(payload, 4)
        lines.append(
            f"- {name}: b1 {a1:.3f}/{r1:.3f}/{i1:.3f}, b2 {a2:.3f}/{r2:.3f}, b4 {a4:.3f}/{r4:.3f}"
        )

    base_b2 = _find_row(t50_8b, "adaptive_fwb", 2)
    base_b4 = _find_row(t50_8b, "adaptive_fwb", 4)
    tuned_b2 = _find_row(phase66, "adaptive_fwb", 2)
    tuned_b4 = _find_row(phase66, "adaptive_fwb", 4)
    lines.extend(
        [
            "",
            "## Phase66 tuned-adaptive vs phase60/61 base-adaptive",
            "",
            f"- b2 success: {float(base_b2['success_mean']):.3f} -> {float(tuned_b2['success_mean']):.3f}",
            f"- b4 success: {float(base_b4['success_mean']):.3f} -> {float(tuned_b4['success_mean']):.3f}",
            f"- b2 test_calls: {float(base_b2['test_mean']):.3f} -> {float(tuned_b2['test_mean']):.3f}",
            f"- b4 test_calls: {float(base_b4['test_mean']):.3f} -> {float(tuned_b4['test_mean']):.3f}",
        ]
    )

    lines.extend(["", "## Phase67 mode ablation (mean across seeds)", ""])
    for budget in [2, 4]:
        part = [r for r in phase67_rows if int(r["budget"]) == budget]
        part.sort(key=lambda x: x["success_mean"], reverse=True)
        for r in part:
            lines.append(
                f"- b{budget} {r['mode']}: success={r['success_mean']:.3f}, feedback={r['feedback_mean']:.3f}, test_calls={r['test_mean']:.3f}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--leaderboard_t50_8b",
        default="runs/phase60_67_closeout/phase60_61_target50_8b_leaderboard.json",
    )
    ap.add_argument(
        "--leaderboard_t50_3b",
        default="runs/phase60_67_closeout/phase62_63_target50_3b_leaderboard.json",
    )
    ap.add_argument(
        "--leaderboard_t100_8b",
        default="runs/phase60_67_closeout/phase64_65_target100_8b_leaderboard.json",
    )
    ap.add_argument(
        "--leaderboard_phase66",
        default="runs/phase60_67_closeout/phase66_adaptive_sweep_8b_leaderboard.json",
    )
    ap.add_argument(
        "--leaderboard_phase67",
        default="runs/phase60_67_closeout/phase67_t1l2r1_8b_leaderboard.json",
    )
    ap.add_argument("--out_dir", default="artifacts/phase60_67_paper_figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t50_8b = _load_json(args.leaderboard_t50_8b)
    t50_3b = _load_json(args.leaderboard_t50_3b)
    t100_8b = _load_json(args.leaderboard_t100_8b)
    phase66 = _load_json(args.leaderboard_phase66)
    phase67 = _load_json(args.leaderboard_phase67)

    plot_frontiers(t50_8b, t50_3b, t100_8b, out_dir / "fig1_frontiers")
    plot_full_cost(t50_8b, t50_3b, t100_8b, out_dir / "fig2_full_cost")
    plot_adaptive_deltas(t50_8b, t50_3b, t100_8b, out_dir / "fig3_adaptive_delta_vs_resample")
    plot_phase66_vs_base(phase66, t50_8b, out_dir / "fig4_phase66_vs_base")
    phase67_rows = plot_phase67_modes(
        phase67,
        out_dir / "fig5_phase67_modes",
        out_dir / "phase67_mode_aggregate.csv",
    )
    write_summary(
        out_dir / "summary.md",
        t50_8b,
        t50_3b,
        t100_8b,
        phase66,
        phase67_rows,
    )

    print(f"Wrote phase60-67 figure pack to {out_dir}")


if __name__ == "__main__":
    main()

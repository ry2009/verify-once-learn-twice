from __future__ import annotations

import argparse
import json
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

COLORS = {
    "adaptive": "#1f77b4",
    "resample": "#ff7f0e",
    "inference": "#7f8c8d",
    "model_8b": "#1f77b4",
    "model_3b": "#2ca02c",
    "first": "#94a3b8",
    "final": "#2563eb",
}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_rows(path: str) -> list[dict[str, Any]]:
    data = _load_json(path)
    return list(data.get("aggregate", []))


def _rows_by_method(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    out = [r for r in rows if str(r.get("method_variant", "")) == method]
    out.sort(key=lambda r: int(r.get("feedback_budget", 0)))
    return out


def _fmt_budget_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [f"b={int(r.get('feedback_budget', 0))}" for r in rows]


def _style_axes(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=STYLE["paper_bg"],
        plot_bgcolor=STYLE["plot_bg"],
        font=dict(color=STYLE["font"], size=15),
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
        margin=dict(l=70, r=30, t=90, b=70),
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


def plot_frontier(rows: list[dict[str, Any]], model_label: str, out_prefix: Path) -> None:
    resample = _rows_by_method(rows, "resample_only")
    adaptive = _rows_by_method(rows, "adaptive_fwb")
    inference = _rows_by_method(rows, "inference_only")

    fig = go.Figure()

    if resample:
        fig.add_trace(
            go.Scatter(
                x=[float(r["feedback_mean"]) for r in resample],
                y=[float(r["success_mean"]) for r in resample],
                mode="lines+markers+text",
                text=_fmt_budget_labels(resample),
                textposition="top center",
                name="Resample-only",
                line=dict(color=COLORS["resample"], width=3),
                marker=dict(size=11, color=COLORS["resample"], symbol="circle"),
            )
        )

    if adaptive:
        fig.add_trace(
            go.Scatter(
                x=[float(r["feedback_mean"]) for r in adaptive],
                y=[float(r["success_mean"]) for r in adaptive],
                mode="lines+markers+text",
                text=_fmt_budget_labels(adaptive),
                textposition="bottom center",
                name="Adaptive FWB",
                line=dict(color=COLORS["adaptive"], width=3),
                marker=dict(size=11, color=COLORS["adaptive"], symbol="diamond"),
            )
        )

    if inference:
        r = inference[0]
        fig.add_trace(
            go.Scatter(
                x=[float(r["feedback_mean"])],
                y=[float(r["success_mean"])],
                mode="markers+text",
                text=["b=1"],
                textposition="top right",
                name="Inference-only",
                marker=dict(size=12, color=COLORS["inference"], symbol="x"),
            )
        )

    fig.update_layout(
        title=f"KernelBench Target-12 Frontier ({model_label})",
        xaxis_title="Average feedback calls per task",
        yaxis_title="Success rate",
        width=980,
        height=620,
    )
    fig.update_yaxes(range=[0.0, 0.55])
    _style_axes(fig)
    _save(fig, out_prefix, 980, 620)


def plot_efficiency(rows_8b: list[dict[str, Any]], rows_3b: list[dict[str, Any]], out_prefix: Path) -> None:
    order = [
        ("inference_only", 1, "Inf b1"),
        ("resample_only", 1, "Res b1"),
        ("resample_only", 5, "Res b5"),
        ("resample_only", 10, "Res b10"),
        ("resample_only", 20, "Res b20"),
        ("adaptive_fwb", 1, "Ada b1"),
        ("adaptive_fwb", 2, "Ada b2"),
        ("adaptive_fwb", 4, "Ada b4"),
    ]

    def _series(rows: list[dict[str, Any]]) -> tuple[list[str], list[float], list[float]]:
        by = {(str(r.get("method_variant")), int(r.get("feedback_budget", -1))): r for r in rows}
        labels, succ, spf = [], [], []
        for m, b, label in order:
            r = by.get((m, b))
            if not r:
                continue
            s = float(r["success_mean"])
            f = max(1e-9, float(r["feedback_mean"]))
            labels.append(label)
            succ.append(s)
            spf.append(s / f)
        return labels, succ, spf

    labels8, succ8, spf8 = _series(rows_8b)
    labels3, succ3, spf3 = _series(rows_3b)
    labels = labels8 if len(labels8) >= len(labels3) else labels3

    map8 = {l: (s, e) for l, s, e in zip(labels8, succ8, spf8)}
    map3 = {l: (s, e) for l, s, e in zip(labels3, succ3, spf3)}
    s8 = [map8.get(l, (None, None))[0] for l in labels]
    s3 = [map3.get(l, (None, None))[0] for l in labels]
    e8 = [map8.get(l, (None, None))[1] for l in labels]
    e3 = [map3.get(l, (None, None))[1] for l in labels]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Success rate", "Success per feedback call"))
    fig.add_trace(go.Bar(x=labels, y=s8, name="8B", marker_color=COLORS["model_8b"]), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=s3, name="3B", marker_color=COLORS["model_3b"]), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=e8, name="8B", marker_color=COLORS["model_8b"], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=e3, name="3B", marker_color=COLORS["model_3b"], showlegend=False), row=1, col=2)

    fig.update_layout(
        barmode="group",
        title="KernelBench Cost-Efficiency Sweep (Phase-50)",
        width=1400,
        height=620,
    )
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(range=[0.0, 0.55], row=1, col=1)
    _style_axes(fig)
    _save(fig, out_prefix, 1400, 620)


def plot_transfer_lift(agg_8b: list[dict[str, Any]], agg_3b: list[dict[str, Any]], out_prefix: Path) -> None:
    def _prep(rows: list[dict[str, Any]]) -> tuple[list[str], list[float], list[float], list[float]]:
        order = ["inference_only", "resample_only", "adaptive_fwb"]
        by = {str(r.get("method")): r for r in rows}
        labels, first, final, lift = [], [], [], []
        for m in order:
            if m not in by:
                continue
            r = by[m]
            labels.append(m.replace("_", "-"))
            first.append(float(r.get("target_first_mean", 0.0)))
            final.append(float(r.get("target_final_mean", 0.0)))
            lift.append(float(r.get("target_lift_mean", 0.0)))
        return labels, first, final, lift

    l8, f8, t8, lift8 = _prep(agg_8b)
    l3, f3, t3, lift3 = _prep(agg_3b)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("8B: first vs final", "3B: first vs final", "Target lift (final-first)"),
    )

    fig.add_trace(go.Bar(x=l8, y=f8, name="8B first", marker_color=COLORS["first"]), row=1, col=1)
    fig.add_trace(go.Bar(x=l8, y=t8, name="8B final", marker_color=COLORS["final"]), row=1, col=1)
    fig.add_trace(go.Bar(x=l3, y=f3, name="3B first", marker_color=COLORS["first"], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=l3, y=t3, name="3B final", marker_color=COLORS["final"], showlegend=False), row=1, col=2)

    fig.add_trace(
        go.Bar(x=["inference", "resample", "adaptive"], y=lift8, name="8B lift", marker_color=COLORS["model_8b"]),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(x=["inference", "resample", "adaptive"], y=lift3, name="3B lift", marker_color=COLORS["model_3b"]),
        row=1,
        col=3,
    )

    fig.update_layout(
        barmode="group",
        title="Transfer Internalization: First-Pass vs Final Success",
        width=1700,
        height=620,
    )
    fig.update_yaxes(range=[0.0, 0.55], row=1, col=1)
    fig.update_yaxes(range=[0.0, 0.55], row=1, col=2)
    _style_axes(fig)
    _save(fig, out_prefix, 1700, 620)


def _best_equal_or_better_cost(rows: list[dict[str, Any]]) -> dict[str, float]:
    res = _rows_by_method(rows, "resample_only")
    ada = _rows_by_method(rows, "adaptive_fwb")
    if not res or not ada:
        return {}
    best_res = max(res, key=lambda r: float(r["success_mean"]))
    target = float(best_res["success_mean"])
    adaptive_hit = [r for r in ada if float(r["success_mean"]) >= target]
    if adaptive_hit:
        best_ada = min(adaptive_hit, key=lambda r: float(r["feedback_mean"]))
    else:
        best_ada = max(ada, key=lambda r: float(r["success_mean"]))
    return {
        "target_success": target,
        "resample_feedback": float(best_res["feedback_mean"]),
        "adaptive_feedback": float(best_ada["feedback_mean"]),
        "resample_success": float(best_res["success_mean"]),
        "adaptive_success": float(best_ada["success_mean"]),
    }


def plot_cost_to_target(rows_8b: list[dict[str, Any]], rows_3b: list[dict[str, Any]], out_prefix: Path) -> dict[str, dict[str, float]]:
    m8 = _best_equal_or_better_cost(rows_8b)
    m3 = _best_equal_or_better_cost(rows_3b)

    fig = go.Figure()
    models = []
    ada_vals = []
    res_vals = []
    if m8:
        models.append("8B")
        ada_vals.append(m8["adaptive_feedback"])
        res_vals.append(m8["resample_feedback"])
    if m3:
        models.append("3B")
        ada_vals.append(m3["adaptive_feedback"])
        res_vals.append(m3["resample_feedback"])

    fig.add_trace(go.Bar(x=models, y=res_vals, name="Resample-only", marker_color=COLORS["resample"]))
    fig.add_trace(go.Bar(x=models, y=ada_vals, name="Adaptive FWB", marker_color=COLORS["adaptive"]))

    fig.update_layout(
        barmode="group",
        title="Feedback Cost to Match/Beat Best Resample Success",
        xaxis_title="Model",
        yaxis_title="Average feedback calls",
        width=960,
        height=620,
    )
    _style_axes(fig)
    _save(fig, out_prefix, 960, 620)
    return {"8b": m8, "3b": m3}


def write_summary(out_dir: Path, m8: dict[str, float], m3: dict[str, float]) -> None:
    def _ratio(m: dict[str, float]) -> str:
        if not m:
            return "n/a"
        a = max(1e-9, float(m["adaptive_feedback"]))
        r = float(m["resample_feedback"]) / a
        return f"{r:.2f}x"

    lines = [
        "# Phase-50 Figure Pack",
        "",
        "This folder contains legible light-theme figures for the new KernelBench paper section.",
        "",
        "## Key cost-efficiency takeaways",
        "",
        f"- 8B feedback ratio (resample/adaptive at matched-or-better success): {_ratio(m8)}",
        f"- 3B feedback ratio (resample/adaptive at matched-or-better success): {_ratio(m3)}",
        "",
        "## Files",
        "",
        "- `fig1_frontier_8b.(png|svg|html)`",
        "- `fig2_frontier_3b.(png|svg|html)`",
        "- `fig3_efficiency_bars.(png|svg|html)`",
        "- `fig4_transfer_lift.(png|svg|html)`",
        "- `fig5_cost_to_target.(png|svg|html)`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--leaderboard_8b",
        default="runs/phase50_kernelbench_kpass_closeout/verify_phase50_kernelbench_kpass_8b_s0_leaderboard.json",
    )
    ap.add_argument(
        "--leaderboard_3b",
        default="runs/phase50_kernelbench_kpass_closeout/verify_phase50_kernelbench_kpass_3b_s0_leaderboard.json",
    )
    ap.add_argument(
        "--transfer_agg_8b",
        default="runs/phase50_kernelbench_kpass_closeout/verify_phase50_kernelbench_kpass_8b_s0_transfer/method_aggregate.json",
    )
    ap.add_argument(
        "--transfer_agg_3b",
        default="runs/phase50_kernelbench_kpass_closeout/verify_phase50_kernelbench_kpass_3b_s0_transfer/method_aggregate.json",
    )
    ap.add_argument("--out_dir", default="artifacts/phase50_paper_figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_8b = _aggregate_rows(args.leaderboard_8b)
    rows_3b = _aggregate_rows(args.leaderboard_3b)
    agg_8b = _load_json(args.transfer_agg_8b)
    agg_3b = _load_json(args.transfer_agg_3b)

    plot_frontier(rows_8b, "Llama-3.1-8B-Instruct", out_dir / "fig1_frontier_8b")
    plot_frontier(rows_3b, "Llama-3.2-3B", out_dir / "fig2_frontier_3b")
    plot_efficiency(rows_8b, rows_3b, out_dir / "fig3_efficiency_bars")
    plot_transfer_lift(agg_8b, agg_3b, out_dir / "fig4_transfer_lift")
    cost = plot_cost_to_target(rows_8b, rows_3b, out_dir / "fig5_cost_to_target")
    write_summary(out_dir, cost.get("8b", {}), cost.get("3b", {}))

    print(f"Wrote figure pack to {out_dir}")


if __name__ == "__main__":
    main()

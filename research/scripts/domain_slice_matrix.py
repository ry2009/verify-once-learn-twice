from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = {
    "adaptive_fwb": "#1f77b4",
    "fixed_k_fwb_k1": "#ff7f0e",
    "fixed_k_fwb_k2": "#2ca02c",
    "fixed_k_judge_k1": "#d62728",
    "fixed_k_judge_k2": "#9467bd",
    "inference_only": "#17becf",
}

THEMES = {
    "light": {
        "paper_bg": "#ffffff",
        "plot_bg": "#ffffff",
        "font_color": "#0f172a",
        "grid_color": "rgba(148, 163, 184, 0.28)",
        "axis_line": "#64748b",
        "legend_bg": "rgba(255,255,255,0.92)",
        "legend_border": "rgba(100, 116, 139, 0.35)",
    },
    "dark": {
        "paper_bg": "#0b1020",
        "plot_bg": "#121a2b",
        "font_color": "#e8eefc",
        "grid_color": "rgba(194, 211, 255, 0.12)",
        "axis_line": "#9aa8cc",
        "legend_bg": "rgba(11,16,32,0.45)",
        "legend_border": "rgba(154,168,204,0.30)",
    },
}


@dataclass
class Cell:
    domain: str
    method: str
    runs: int
    success_mean: float
    success_sd: float
    feedback_mean: float
    train_mean: float
    test_mean: float


def _slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip("-")


def _method_label(method: str) -> str:
    if method == "adaptive_fwb":
        return "Adaptive Stop"
    if method == "fixed_k_fwb_k1":
        return "Fixed-k (k=1)"
    if method == "fixed_k_fwb_k2":
        return "Fixed-k (k=2)"
    if method == "fixed_k_judge_k1":
        return "Fixed-k+Judge (k=1)"
    if method == "fixed_k_judge_k2":
        return "Fixed-k+Judge (k=2)"
    if method == "inference_only":
        return "Inference-only"
    return method


def _read_leaderboard(path: str, domain: str) -> List[Cell]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    cells: List[Cell] = []
    for row in j.get("aggregate", []):
        cells.append(
            Cell(
                domain=domain,
                method=str(row["method_variant"]),
                runs=int(row["runs"]),
                success_mean=float(row["success_mean"]),
                success_sd=float(row["success_sd"]),
                feedback_mean=float(row["feedback_mean"]),
                train_mean=float(row["train_mean"]),
                test_mean=float(row["test_mean"]),
            )
        )
    return cells


def _load_cells(specs: Sequence[str]) -> List[Cell]:
    cells: List[Cell] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --slice format: {spec} (expected domain=path)")
        domain, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"Leaderboard JSON not found: {path}")
        cells.extend(_read_leaderboard(path, domain=domain))
    return cells


def _tabular(cells: Sequence[Cell], methods: Sequence[str], domains: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("domain,method,runs,success_mean,success_sd,feedback_mean,train_mean,test_mean")
    by_key: Dict[Tuple[str, str], Cell] = {(c.domain, c.method): c for c in cells}
    for d in domains:
        for m in methods:
            c = by_key.get((d, m))
            if c is None:
                lines.append(f"{d},{m},0,,,,,")
                continue
            lines.append(
                f"{d},{m},{c.runs},{c.success_mean:.4f},{c.success_sd:.4f},"
                f"{c.feedback_mean:.4f},{c.train_mean:.4f},{c.test_mean:.4f}"
            )
    return "\n".join(lines) + "\n"


def _build_fig(cells: Sequence[Cell], title: str, theme_name: str) -> go.Figure:
    style = THEMES.get(theme_name, THEMES["light"])
    domains = sorted(set(c.domain for c in cells))
    preferred = [
        "adaptive_fwb",
        "fixed_k_fwb_k1",
        "fixed_k_fwb_k2",
        "fixed_k_judge_k1",
        "fixed_k_judge_k2",
        "inference_only",
    ]
    observed = sorted(set(c.method for c in cells))
    methods = [m for m in preferred if m in observed] + [m for m in observed if m not in preferred]

    by_key: Dict[Tuple[str, str], Cell] = {(c.domain, c.method): c for c in cells}

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Success by domain",
            "Adaptive advantage vs fixed-k (k=1)",
            "Feedback calls by domain",
            "Train updates by domain",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.22,
    )

    # Panel 1: grouped bars for success.
    for method in methods:
        ys: List[float] = []
        errs: List[float] = []
        for d in domains:
            c = by_key.get((d, method))
            ys.append(c.success_mean if c else 0.0)
            errs.append(c.success_sd if c else 0.0)
        fig.add_trace(
            go.Bar(
                x=domains,
                y=ys,
                name=_method_label(method),
                marker=dict(color=PALETTE.get(method, "#cccccc")),
                error_y=dict(type="data", array=errs),
                hovertemplate=(
                    "domain=%{x}<br>success=%{y:.3f}<extra>" + _method_label(method) + "</extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Panel 2: adaptive minus fixed-k1 delta.
    delta_y: List[float] = []
    delta_err: List[float] = []
    for d in domains:
        a = by_key.get((d, "adaptive_fwb"))
        f = by_key.get((d, "fixed_k_fwb_k1"))
        if a is None or f is None:
            delta_y.append(0.0)
            delta_err.append(0.0)
            continue
        delta_y.append(a.success_mean - f.success_mean)
        delta_err.append((a.success_sd**2 + f.success_sd**2) ** 0.5)
    fig.add_trace(
        go.Bar(
            x=domains,
            y=delta_y,
            marker=dict(color=["#58d1c9" if v >= 0 else "#f58db2" for v in delta_y]),
            error_y=dict(type="data", array=delta_err),
            hovertemplate="domain=%{x}<br>delta=%{y:.3f}<extra>Adaptive - Fixed-k1</extra>",
            name="Adaptive - Fixed-k1",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0.0, line_width=1.1, line_color=style["axis_line"], row=1, col=2)

    # Panel 3: feedback means.
    for method in methods:
        ys = []
        for d in domains:
            c = by_key.get((d, method))
            ys.append(c.feedback_mean if c else 0.0)
        fig.add_trace(
            go.Scatter(
                x=domains,
                y=ys,
                mode="lines+markers",
                line=dict(color=PALETTE.get(method, "#cccccc"), width=3, shape="spline"),
                marker=dict(size=9),
                name=_method_label(method) + " feedback",
                showlegend=False,
                hovertemplate="domain=%{x}<br>feedback=%{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Panel 4: train means.
    for method in methods:
        ys = []
        for d in domains:
            c = by_key.get((d, method))
            ys.append(c.train_mean if c else 0.0)
        fig.add_trace(
            go.Scatter(
                x=domains,
                y=ys,
                mode="lines+markers",
                line=dict(color=PALETTE.get(method, "#cccccc"), width=3, shape="spline"),
                marker=dict(size=9),
                name=_method_label(method) + " train",
                showlegend=False,
                hovertemplate="domain=%{x}<br>train_steps=%{y:.3f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    axis_grid = dict(
        showgrid=True,
        gridcolor=style["grid_color"],
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=style["axis_line"],
        automargin=True,
        tickfont=dict(size=13),
        title_font=dict(size=15),
    )
    fig.update_xaxes(**axis_grid, row=1, col=1, title=None, tickangle=-20)
    fig.update_xaxes(**axis_grid, row=1, col=2, title=None, tickangle=-20)
    fig.update_xaxes(**axis_grid, row=2, col=1, title="Domain", tickangle=-20, title_standoff=10)
    fig.update_xaxes(**axis_grid, row=2, col=2, title="Domain", tickangle=-20, title_standoff=10)

    fig.update_yaxes(**axis_grid, row=1, col=1, title="Success", range=[0, 1], title_standoff=10)
    fig.update_yaxes(**axis_grid, row=1, col=2, title="Delta success", title_standoff=10)
    fig.update_yaxes(**axis_grid, row=2, col=1, title="Avg feedback calls", title_standoff=10)
    fig.update_yaxes(**axis_grid, row=2, col=2, title="Avg train updates", title_standoff=10)

    fig.update_annotations(font=dict(size=15, color=style["font_color"]))

    fig.update_layout(
        barmode="group",
        template="plotly_white" if theme_name == "light" else "plotly_dark",
        title=dict(text=title, x=0.01, y=0.995, xanchor="left", yanchor="top", font=dict(size=26)),
        paper_bgcolor=style["paper_bg"],
        plot_bgcolor=style["plot_bg"],
        font=dict(
            family="Space Grotesk, Avenir Next, Segoe UI, sans-serif",
            size=13,
            color=style["font_color"],
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            bgcolor=style["legend_bg"],
            bordercolor=style["legend_border"],
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=95, r=380, t=125, b=90),
        width=2000,
        height=1250,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slice",
        action="append",
        required=True,
        help="domain=path_to_costly_leaderboard_json",
    )
    parser.add_argument("--title", default="Domain Slice Matrix")
    parser.add_argument("--out_dir", default="runs/pretty_plots")
    parser.add_argument("--theme", choices=sorted(THEMES.keys()), default="light")
    parser.add_argument("--png_width", type=int, default=2200)
    parser.add_argument("--png_height", type=int, default=1400)
    parser.add_argument("--png_scale", type=float, default=1.0)
    parser.add_argument("--write_png", action="store_true")
    args = parser.parse_args()

    cells = _load_cells(args.slice)
    if not cells:
        raise SystemExit("No cells loaded")

    domains = sorted(set(c.domain for c in cells))
    preferred = [
        "adaptive_fwb",
        "fixed_k_fwb_k1",
        "fixed_k_fwb_k2",
        "fixed_k_judge_k1",
        "fixed_k_judge_k2",
        "inference_only",
    ]
    observed = sorted(set(c.method for c in cells))
    methods = [m for m in preferred if m in observed] + [m for m in observed if m not in preferred]

    out_tag = _slug("domain-slice-matrix-" + "-".join(domains))
    out_path = os.path.join(args.out_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-{out_tag}")
    os.makedirs(out_path, exist_ok=True)

    csv_path = os.path.join(out_path, "matrix.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(_tabular(cells, methods=methods, domains=domains))

    fig = _build_fig(cells, title=args.title, theme_name=args.theme)
    html_path = os.path.join(out_path, "dashboard.html")
    fig.write_html(html_path, include_plotlyjs="cdn")

    png_path = os.path.join(out_path, "dashboard.png")
    png_ok = False
    if args.write_png:
        try:
            fig.write_image(
                png_path,
                width=args.png_width,
                height=args.png_height,
                scale=args.png_scale,
            )
            png_ok = True
        except Exception as exc:
            print(f"Warning: PNG export failed ({exc})")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domains": domains,
        "methods": methods,
        "theme": args.theme,
        "png_width": args.png_width,
        "png_height": args.png_height,
        "png_scale": args.png_scale,
        "rows": [c.__dict__ for c in cells],
        "output_html": html_path,
        "output_png": png_path if png_ok else None,
        "output_csv": csv_path,
    }
    summary_path = os.path.join(out_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Saved dashboard: {html_path}")
    if png_ok:
        print(f"Saved PNG: {png_path}")
    print(f"Saved matrix CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

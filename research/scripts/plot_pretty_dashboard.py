from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#17becf",  # cyan
]

METHOD_LABEL = {
    "adaptive_fwb": "Adaptive Stop",
    "fixed_k_fwb": "Fixed-k",
    "fixed_k_judge": "Fixed-k+Judge",
    "inference_only": "Inference-only",
    "resample_only": "Resample-only",
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
        "marker_edge": "rgba(15, 23, 42, 0.40)",
        "diag_line": "rgba(71, 85, 105, 0.55)",
    },
    "dark": {
        "paper_bg": "#0b1020",
        "plot_bg": "#121a2b",
        "font_color": "#e8eefc",
        "grid_color": "rgba(194, 211, 255, 0.12)",
        "axis_line": "#9aa8cc",
        "legend_bg": "rgba(11,16,32,0.45)",
        "legend_border": "rgba(154,168,204,0.30)",
        "marker_edge": "rgba(255, 255, 255, 0.60)",
        "diag_line": "rgba(220,230,250,0.40)",
    },
}


@dataclass
class RunMetrics:
    run_group: str
    run_dir: str
    run_name: str
    method: str
    method_variant: str
    inner_updates: int
    feedback_budget: int
    judge_flip_prob: float
    seed: int
    tasks: int
    success_rate: float
    avg_feedback_calls: float
    avg_train_steps: float
    avg_test_calls: float
    cum_success: List[float]


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip("-")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return statistics.pstdev(values)


def _jitter(vals: Sequence[float], amount: float = 0.02) -> List[float]:
    if not vals:
        return []
    n = len(vals)
    if n == 1:
        return [float(vals[0])]
    center = (n - 1) / 2.0
    return [float(v) + amount * (i - center) / center for i, v in enumerate(vals)]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _method_variant(method: str, inner_updates: int) -> str:
    if method == "fixed_k_fwb":
        return f"fixed_k_fwb_k{inner_updates}"
    if method == "fixed_k_judge":
        return f"fixed_k_judge_k{inner_updates}"
    return method


def _read_run(run_group: str, run_dir: str, include_partial: bool, min_tasks: int) -> RunMetrics | None:
    config_path = os.path.join(run_dir, "config.json")
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(config_path) or not os.path.exists(events_path):
        return None

    cfg = _load_json(config_path)
    task_done = [ev for ev in _iter_jsonl(events_path) if ev.get("type") == "task_done"]
    if not task_done:
        return None
    if not include_partial and len(task_done) < min_tasks:
        return None

    success_flags = [1 if ev.get("success") else 0 for ev in task_done]
    feedback_calls = [float(ev.get("feedback_calls", 0)) for ev in task_done]
    train_steps = [float(ev.get("train_steps", 0)) for ev in task_done]
    test_calls = [float(ev.get("test_calls", 0)) for ev in task_done]

    cum_success = []
    running = 0.0
    for idx, flag in enumerate(success_flags, start=1):
        running += flag
        cum_success.append(running / idx)

    method = str(cfg.get("method", "unknown"))
    inner_updates = int(cfg.get("inner_updates", 1))

    return RunMetrics(
        run_group=run_group,
        run_dir=run_dir,
        run_name=os.path.basename(run_dir),
        method=method,
        method_variant=_method_variant(method, inner_updates),
        inner_updates=inner_updates,
        feedback_budget=int(cfg.get("feedback_budget", -1)),
        judge_flip_prob=float(cfg.get("judge_flip_prob", 0.0)),
        seed=int(cfg.get("seed", -1)),
        tasks=len(task_done),
        success_rate=_mean(success_flags),
        avg_feedback_calls=_mean(feedback_calls),
        avg_train_steps=_mean(train_steps),
        avg_test_calls=_mean(test_calls),
        cum_success=cum_success,
    )


def _discover_groups(runs_root: str, requested: Sequence[str]) -> List[str]:
    if requested:
        return list(requested)
    groups = []
    for name in sorted(os.listdir(runs_root)):
        full = os.path.join(runs_root, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("verify_"):
            groups.append(name)
    return groups


def _collect_runs(
    runs_root: str,
    run_groups: Sequence[str],
    include_partial: bool,
    min_tasks: int,
) -> List[RunMetrics]:
    # Keep only the latest directory for each logical run_name to avoid
    # duplicate counting from resumed/interrupted sweeps.
    dedup: Dict[Tuple[str, str], Tuple[float, RunMetrics]] = {}
    for group in run_groups:
        group_root = os.path.join(runs_root, group)
        if not os.path.isdir(group_root):
            continue
        run_dirs: List[str] = []
        manifest_path = os.path.join(group_root, "ablation_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = _load_json(manifest_path)
                seen_paths: set[str] = set()
                for item in manifest.get("runs", []):
                    p = str(item.get("run_dir", "")).strip()
                    if not p:
                        continue
                    if not os.path.isabs(p):
                        p = os.path.join(os.getcwd(), p)
                    norm = os.path.realpath(p)
                    if norm in seen_paths:
                        continue
                    seen_paths.add(norm)
                    run_dirs.append(p)
            except Exception:
                run_dirs = []
        if not run_dirs:
            for name in sorted(os.listdir(group_root)):
                run_dir = os.path.join(group_root, name)
                if os.path.isdir(run_dir):
                    run_dirs.append(run_dir)
        for run_dir in run_dirs:
            row = _read_run(group, run_dir, include_partial, min_tasks)
            if row:
                key = (group, row.run_name)
                stamp = os.path.getmtime(run_dir) if os.path.exists(run_dir) else 0.0
                prev = dedup.get(key)
                if prev is None or stamp >= prev[0]:
                    dedup[key] = (stamp, row)
    return [row for _, row in dedup.values()]


def _key(run: RunMetrics) -> Tuple[str, int, float]:
    return (run.method_variant, run.feedback_budget, run.judge_flip_prob)


def _label(method: str, feedback_budget: int, judge_flip_prob: float) -> str:
    k_match = re.match(r"^fixed_k_fwb_k(\d+)$", method)
    if k_match:
        base = f"Fixed-k (k={k_match.group(1)}, b={feedback_budget})"
    else:
        k_match = re.match(r"^fixed_k_judge_k(\d+)$", method)
        if k_match:
            base = f"Fixed-k+Judge (k={k_match.group(1)}, b={feedback_budget})"
        else:
            base = f"{METHOD_LABEL.get(method, method)} (b={feedback_budget})"
    if judge_flip_prob and judge_flip_prob > 0.0:
        pct = int(round(100 * judge_flip_prob))
        return f"{base} [noise={pct}%]"
    return base


def _aggregate(runs: Sequence[RunMetrics]) -> List[dict]:
    by_key: Dict[Tuple[str, int, float], List[RunMetrics]] = {}
    for run in runs:
        by_key.setdefault(_key(run), []).append(run)

    rows = []
    for (method, budget, judge_flip_prob), items in sorted(
        by_key.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])
    ):
        rows.append(
            {
                "method": method,
                "base_method": items[0].method if items else method,
                "feedback_budget": budget,
                "judge_flip_prob": judge_flip_prob,
                "runs": len(items),
                "success_mean": _mean([x.success_rate for x in items]),
                "success_sd": _std([x.success_rate for x in items]),
                "feedback_mean": _mean([x.avg_feedback_calls for x in items]),
                "feedback_sd": _std([x.avg_feedback_calls for x in items]),
                "train_mean": _mean([x.avg_train_steps for x in items]),
                "train_sd": _std([x.avg_train_steps for x in items]),
                "test_mean": _mean([x.avg_test_calls for x in items]),
                "test_sd": _std([x.avg_test_calls for x in items]),
            }
        )
    return rows


def _cumulative_stats(items: Sequence[RunMetrics]) -> Tuple[List[int], List[float], List[float]]:
    if not items:
        return [], [], []
    max_tasks = max(x.tasks for x in items)
    xs = list(range(1, max_tasks + 1))
    means: List[float] = []
    sds: List[float] = []
    for idx in range(max_tasks):
        vals = [run.cum_success[idx] for run in items if len(run.cum_success) > idx]
        means.append(_mean(vals))
        sds.append(_std(vals))
    return xs, means, sds


def _build_figure(runs: Sequence[RunMetrics], title: str, theme_name: str) -> go.Figure:
    style = THEMES.get(theme_name, THEMES["light"])
    by_key: Dict[Tuple[str, int, float], List[RunMetrics]] = {}
    for run in runs:
        by_key.setdefault(_key(run), []).append(run)
    keys = sorted(by_key.keys(), key=lambda x: (x[1], x[0]))
    color_map = {k: PALETTE[idx % len(PALETTE)] for idx, k in enumerate(keys)}

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cumulative success by task",
            "Success vs feedback calls",
            "Success vs train updates",
            "Success frontier by feedback budget",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.22,
    )

    # Panel 1: cumulative success traces with uncertainty bands.
    for key in keys:
        items = by_key[key]
        color = color_map[key]
        label = _label(*key)
        draw_raw_runs = len(items) <= 6
        if draw_raw_runs:
            for run in items:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, run.tasks + 1)),
                        y=run.cum_success,
                        mode="lines",
                        line=dict(color=_rgba(color, 0.20), width=1.0, shape="spline"),
                        showlegend=False,
                        hovertemplate=(
                            f"{label}<br>run={run.run_name}<br>"
                            "task=%{x}<br>cum_success=%{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

        xs, means, sds = _cumulative_stats(items)
        upper = [min(1.0, m + s) for m, s in zip(means, sds)]
        lower = [max(0.0, m - s) for m, s in zip(means, sds)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=label,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_rgba(color, 0.16),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=label,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines",
                name=label,
                line=dict(color=color, width=3.0, shape="spline"),
                legendgroup=label,
                hovertemplate=f"{label}<br>task=%{{x}}<br>mean=%{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Panel 2 and 3: run-level scatter + centroid.
    for key in keys:
        items = by_key[key]
        color = color_map[key]
        label = _label(*key)

        x_feedback = [r.avg_feedback_calls for r in items]
        x_train = [r.avg_train_steps for r in items]
        y_success = [r.success_rate for r in items]
        x_feedback_plot = _jitter(x_feedback, amount=0.028)
        x_train_plot = _jitter(x_train, amount=0.03)
        hover_text = [
            (
                f"{label}<br>run={r.run_name}<br>seed={r.seed}"
                f"<br>success={r.success_rate:.3f}<br>feedback={r.avg_feedback_calls:.2f}"
                f"<br>train_steps={r.avg_train_steps:.2f}"
            )
            for r in items
        ]

        fig.add_trace(
            go.Scatter(
                x=x_feedback_plot,
                y=y_success,
                mode="markers",
                marker=dict(
                    size=8,
                    color=_rgba(color, 0.75),
                    line=dict(color=style["marker_edge"], width=1),
                ),
                showlegend=False,
                legendgroup=label,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[_mean(x_feedback)],
                y=[_mean(y_success)],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color=color,
                    line=dict(color=style["marker_edge"], width=1.3),
                ),
                error_x=dict(type="data", array=[_std(x_feedback)], color=color, thickness=1.1),
                error_y=dict(type="data", array=[_std(y_success)], color=color, thickness=1.1),
                showlegend=False,
                legendgroup=label,
                hovertemplate=(
                    f"{label}<br>mean success=%{{y:.3f}}"
                    f"<br>mean feedback=%{{x:.2f}}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=x_train_plot,
                y=y_success,
                mode="markers",
                marker=dict(
                    size=8,
                    color=_rgba(color, 0.75),
                    line=dict(color=style["marker_edge"], width=1),
                ),
                showlegend=False,
                legendgroup=label,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[_mean(x_train)],
                y=[_mean(y_success)],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color=color,
                    line=dict(color=style["marker_edge"], width=1.3),
                ),
                error_x=dict(type="data", array=[_std(x_train)], color=color, thickness=1.1),
                error_y=dict(type="data", array=[_std(y_success)], color=color, thickness=1.1),
                showlegend=False,
                legendgroup=label,
                hovertemplate=(
                    f"{label}<br>mean success=%{{y:.3f}}"
                    f"<br>mean train steps=%{{x:.2f}}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    # Panel 4: frontier by method over budgets.
    method_noise_keys = sorted(set((run.method_variant, run.judge_flip_prob) for run in runs))
    method_colors = {
        k: PALETTE[idx % len(PALETTE)] for idx, k in enumerate(method_noise_keys)
    }
    for method, judge_flip_prob in method_noise_keys:
        points = []
        for (m, budget, noise), items in by_key.items():
            if m != method or abs(noise - judge_flip_prob) > 1e-12:
                continue
            points.append((budget, _mean([x.success_rate for x in items]), _std([x.success_rate for x in items])))
        points.sort(key=lambda x: x[0])
        if not points:
            continue
        budgets = [p[0] for p in points]
        means = [p[1] for p in points]
        sds = [p[2] for p in points]
        k_match = re.match(r"^fixed_k_fwb_k(\d+)$", method)
        if k_match:
            pretty_name = f"Fixed-k (k={k_match.group(1)})"
        else:
            pretty_name = METHOD_LABEL.get(method, method)
        if judge_flip_prob and judge_flip_prob > 0.0:
            pretty_name = f"{pretty_name} [noise={int(round(100 * judge_flip_prob))}%]"
        fig.add_trace(
            go.Scatter(
                x=budgets,
                y=means,
                mode="lines+markers",
                line=dict(color=method_colors[(method, judge_flip_prob)], width=2.5, shape="spline"),
                marker=dict(
                    size=9,
                    color=method_colors[(method, judge_flip_prob)],
                    line=dict(color=style["marker_edge"], width=1),
                ),
                error_y=dict(type="data", array=sds, color=method_colors[(method, judge_flip_prob)], thickness=1.1),
                name=f"{pretty_name} frontier",
                showlegend=False,
                hovertemplate=(
                    f"{pretty_name}<br>budget=%{{x}}"
                    "<br>mean success=%{y:.3f}<extra></extra>"
                ),
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
    fig.update_xaxes(**axis_grid, row=1, col=1, title=None)
    fig.update_xaxes(**axis_grid, row=1, col=2, title=None)
    fig.update_xaxes(**axis_grid, row=2, col=1, title="Avg train updates per task", title_standoff=12)
    fig.update_xaxes(**axis_grid, row=2, col=2, title="Feedback budget", title_standoff=12)

    fig.update_yaxes(**axis_grid, row=1, col=1, title="Cumulative success", range=[0, 1], title_standoff=10)
    fig.update_yaxes(**axis_grid, row=1, col=2, title="Success rate", range=[0, 1], title_standoff=10)
    fig.update_yaxes(**axis_grid, row=2, col=1, title="Success rate", range=[0, 1], title_standoff=10)
    fig.update_yaxes(**axis_grid, row=2, col=2, title="Mean success rate", range=[0, 1], title_standoff=10)

    fig.update_annotations(font=dict(size=15, color=style["font_color"]))

    fig.update_layout(
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
        margin=dict(l=110, r=430, t=120, b=85),
        width=2000,
        height=1250,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--run_group", action="append", default=[])
    parser.add_argument("--include_partial", action="store_true")
    parser.add_argument("--min_tasks", type=int, default=30)
    parser.add_argument("--title", default="")
    parser.add_argument("--out_dir", default="runs/pretty_plots")
    parser.add_argument("--theme", choices=sorted(THEMES.keys()), default="light")
    parser.add_argument("--png_width", type=int, default=2200)
    parser.add_argument("--png_height", type=int, default=1400)
    parser.add_argument("--png_scale", type=float, default=1.0)
    parser.add_argument("--write_png", action="store_true")
    args = parser.parse_args()

    run_groups = _discover_groups(args.runs_root, args.run_group)
    if not run_groups:
        raise SystemExit("No run groups found. Pass --run_group or create runs/verify_* first.")

    runs = _collect_runs(
        runs_root=args.runs_root,
        run_groups=run_groups,
        include_partial=args.include_partial,
        min_tasks=args.min_tasks,
    )
    if not runs:
        raise SystemExit("No matching completed runs found for requested groups.")

    title = args.title
    if not title:
        title = f"ttRL Verifiable-RL Dashboard ({len(runs)} runs)"

    out_tag = _slug("-".join(run_groups))
    out_path = os.path.join(args.out_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-{out_tag}")
    os.makedirs(out_path, exist_ok=True)

    fig = _build_figure(runs, title=title, theme_name=args.theme)
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
            print(f"PNG export skipped (install kaleido for static export): {exc}")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_groups": run_groups,
        "run_count": len(runs),
        "theme": args.theme,
        "png_width": args.png_width,
        "png_height": args.png_height,
        "png_scale": args.png_scale,
        "runs": [asdict(x) for x in runs],
        "aggregate": _aggregate(runs),
        "output_html": html_path,
        "output_png": png_path if png_ok else None,
    }
    summary_path = os.path.join(out_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved dashboard: {html_path}")
    if png_ok:
        print(f"Saved PNG: {png_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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
class TaskOutcome:
    task_id: str
    entry_point: str
    domain: str
    adaptive_rate: float
    fixed_rate: float

    @property
    def delta(self) -> float:
        return self.adaptive_rate - self.fixed_rate


def _slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip("-")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _domain_label(entry_point: str, prompt: str) -> str:
    ep = (entry_point or "").lower()
    p = (prompt or "").lower()
    text = f"{ep}\n{p}"
    if any(x in text for x in ["music", "paren", "prefix", "sort_numbers", "parse_"]):
        return "Symbolic Parsing"
    if any(x in text for x in ["factor", "divisor", "closest", "rescale", "prime"]):
        return "Numeric Reasoning"
    if any(x in text for x in ["palindrome", "xor", "substring", "sequence"]):
        return "String Logic"
    return "General Program Synthesis"


def _collect_outcomes(run_groups: List[str], fixed_inner_updates: int | None = None) -> List[TaskOutcome]:
    if not run_groups:
        raise SystemExit("At least one --run_group is required")

    # method -> seed -> task_id -> success
    successes: Dict[str, Dict[int, Dict[str, int]]] = {"adaptive_fwb": {}, "fixed_k_fwb": {}}
    task_meta: Dict[str, Tuple[str, str]] = {}

    for run_group in run_groups:
        root = os.path.join("runs", run_group)
        if not os.path.isdir(root):
            raise SystemExit(f"Run group not found: {root}")

        for name in sorted(os.listdir(root)):
            run_dir = os.path.join(root, name)
            if not os.path.isdir(run_dir):
                continue
            cfg_path = os.path.join(run_dir, "config.json")
            ev_path = os.path.join(run_dir, "events.jsonl")
            if not (os.path.exists(cfg_path) and os.path.exists(ev_path)):
                continue

            cfg = _load_json(cfg_path)
            method = cfg.get("method")
            if method not in successes:
                continue
            if method == "fixed_k_fwb" and fixed_inner_updates is not None:
                if int(cfg.get("inner_updates", 1)) != fixed_inner_updates:
                    continue

            seed = int(cfg.get("seed", -1))
            run_outcomes = successes[method].setdefault(seed, {})
            for ev in _iter_jsonl(ev_path):
                if ev.get("type") == "task_start":
                    task_meta.setdefault(
                        ev["task_id"],
                        (ev.get("entry_point", ""), ev.get("prompt", "")),
                    )
                if ev.get("type") == "task_done":
                    run_outcomes[ev["task_id"]] = 1 if ev.get("success") else 0

    if not successes["adaptive_fwb"] or not successes["fixed_k_fwb"]:
        raise SystemExit("Need both adaptive_fwb and fixed_k_fwb runs in selected run groups")

    adaptive_tasks = set()
    fixed_tasks = set()
    for seed_map in successes["adaptive_fwb"].values():
        adaptive_tasks.update(seed_map.keys())
    for seed_map in successes["fixed_k_fwb"].values():
        fixed_tasks.update(seed_map.keys())
    task_ids = sorted(adaptive_tasks & fixed_tasks, key=lambda x: int(x.split("/")[-1]))

    outcomes: List[TaskOutcome] = []
    for task_id in task_ids:
        a_vals = [seed_map.get(task_id, 0) for seed_map in successes["adaptive_fwb"].values()]
        f_vals = [seed_map.get(task_id, 0) for seed_map in successes["fixed_k_fwb"].values()]
        entry_point, prompt = task_meta.get(task_id, ("", ""))
        outcomes.append(
            TaskOutcome(
                task_id=task_id,
                entry_point=entry_point,
                domain=_domain_label(entry_point, prompt),
                adaptive_rate=sum(a_vals) / len(a_vals),
                fixed_rate=sum(f_vals) / len(f_vals),
            )
        )
    return outcomes


def _group_slug(run_groups: List[str]) -> str:
    if len(run_groups) == 1:
        return run_groups[0]
    joined = "__".join(sorted(run_groups))
    return joined[:180]


def _task_label(task_id: str, entry_point: str) -> str:
    task_num = task_id.split("/")[-1] if "/" in task_id else task_id
    return f"H{task_num}: {entry_point}"


def _ranked_task_subset(ranked: List[TaskOutcome], max_task_bars: int) -> List[TaskOutcome]:
    if max_task_bars <= 0 or len(ranked) <= max_task_bars:
        return ranked
    hi = max_task_bars // 2
    lo = max_task_bars - hi
    selected = ranked[:hi] + ranked[-lo:]
    return sorted(selected, key=lambda x: x.delta, reverse=True)


def _ranked_task_subset_nonzero(
    ranked: List[TaskOutcome],
    max_task_bars: int,
    delta_eps: float,
    include_ties: bool,
) -> List[TaskOutcome]:
    nonzero = [x for x in ranked if abs(x.delta) > delta_eps]
    ties = [x for x in ranked if abs(x.delta) <= delta_eps]
    if not nonzero:
        return _ranked_task_subset(ranked, max_task_bars=max_task_bars)

    positives = sorted([x for x in nonzero if x.delta > 0], key=lambda x: x.delta, reverse=True)
    negatives = sorted([x for x in nonzero if x.delta < 0], key=lambda x: x.delta)

    if max_task_bars <= 0:
        out = negatives + list(reversed(positives))
        return out

    half = max_task_bars // 2
    take_pos = min(len(positives), half)
    take_neg = min(len(negatives), max_task_bars - take_pos)
    remaining = max_task_bars - (take_pos + take_neg)
    if remaining > 0:
        extra_pos = min(remaining, len(positives) - take_pos)
        take_pos += extra_pos
        remaining -= extra_pos
    if remaining > 0:
        extra_neg = min(remaining, len(negatives) - take_neg)
        take_neg += extra_neg
        remaining -= extra_neg

    picked = positives[:take_pos] + negatives[:take_neg]
    picked = sorted(picked, key=lambda x: x.delta)

    if include_ties and remaining > 0 and ties:
        picked.extend(ties[:remaining])
        picked = sorted(picked, key=lambda x: x.delta)
    return picked


def _make_figure(
    outcomes: List[TaskOutcome],
    title: str,
    theme_name: str,
    max_task_bars: int,
    delta_eps: float,
    include_ties: bool,
) -> go.Figure:
    style = THEMES.get(theme_name, THEMES["light"])
    domains = sorted(set(x.domain for x in outcomes))
    color_map = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(domains)}

    # Sort tasks by delta descending for panel 1.
    ranked = sorted(outcomes, key=lambda x: x.delta, reverse=True)
    ranked_view = _ranked_task_subset_nonzero(
        ranked,
        max_task_bars=max_task_bars,
        delta_eps=delta_eps,
        include_ties=include_ties,
    )
    task_labels = [_task_label(x.task_id, x.entry_point) for x in ranked_view]
    deltas = [x.delta for x in ranked_view]
    bar_colors = [color_map[x.domain] for x in ranked_view]
    top_count = len([x for x in ranked_view if x.delta > delta_eps])
    bottom_count = len([x for x in ranked_view if x.delta < -delta_eps])
    win_all = len([x for x in outcomes if x.delta > delta_eps])
    lose_all = len([x for x in outcomes if x.delta < -delta_eps])
    tie_all = len(outcomes) - win_all - lose_all

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Task delta (Adaptive - Fixed, + is better): {top_count} wins / {bottom_count} losses shown",
            "Per-task success: adaptive vs fixed-k",
            "Domain mean delta",
            "Task success distribution by method",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.22,
    )

    # Panel 1: task-level deltas.
    fig.add_trace(
        go.Bar(
            x=deltas,
            y=task_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{x:+.2f}" for x in deltas],
            textposition="outside",
            textfont=dict(size=10, color=style["font_color"]),
            cliponaxis=False,
            hovertemplate="%{y}<br>delta=%{x:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0.0, line_width=1.1, line_color=style["axis_line"], row=1, col=1)

    # Panel 2: adaptive vs fixed scatter.
    for domain in domains:
        xs = [x.fixed_rate for x in outcomes if x.domain == domain]
        ys = [x.adaptive_rate for x in outcomes if x.domain == domain]
        labels = [f"{x.task_id} · {x.entry_point}" for x in outcomes if x.domain == domain]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=domain,
                marker=dict(
                    size=9,
                    color=color_map[domain],
                    line=dict(color=style["marker_edge"], width=1),
                ),
                text=labels,
                hovertemplate="%{text}<br>fixed=%{x:.2f}<br>adaptive=%{y:.2f}<extra></extra>",
                legendgroup=domain,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=style["diag_line"], width=1.5, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Panel 3: domain mean deltas.
    domain_rows = []
    for domain in domains:
        vals = [x.delta for x in outcomes if x.domain == domain]
        domain_rows.append(
            (
                domain,
                sum(vals) / len(vals),
                statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            )
        )
    domain_rows.sort(key=lambda x: x[1], reverse=True)
    fig.add_trace(
        go.Bar(
            x=[x[0] for x in domain_rows],
            y=[x[1] for x in domain_rows],
            marker=dict(color=[color_map[x[0]] for x in domain_rows]),
            error_y=dict(type="data", array=[x[2] for x in domain_rows]),
            hovertemplate="%{x}<br>mean delta=%{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_width=1.1, line_color=style["axis_line"], row=2, col=1)

    # Panel 4: box distributions by method.
    fig.add_trace(
        go.Box(
            y=[x.adaptive_rate for x in outcomes],
            name="Adaptive",
            marker_color="#1f77b4",
            boxmean=True,
            showlegend=False,
            hovertemplate="adaptive per-task success=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Box(
            y=[x.fixed_rate for x in outcomes],
            name="Fixed-k",
            marker_color="#ff7f0e",
            boxmean=True,
            showlegend=False,
            hovertemplate="fixed per-task success=%{y:.2f}<extra></extra>",
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
    fig.update_xaxes(**axis_grid, row=2, col=1, title="Domain", tickangle=-20, title_standoff=10)
    fig.update_xaxes(**axis_grid, row=2, col=2, title="Method", title_standoff=10)
    fig.update_yaxes(**axis_grid, row=1, col=1, title="Task", title_standoff=10)
    fig.update_yaxes(**axis_grid, row=1, col=2, title="Adaptive success", range=[0, 1], title_standoff=10)
    fig.update_yaxes(**axis_grid, row=2, col=1, title="Mean delta", title_standoff=10)
    fig.update_yaxes(**axis_grid, row=2, col=2, title="Per-task success", range=[0, 1], title_standoff=10)

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
        margin=dict(l=250, r=380, t=125, b=90),
        width=2100,
        height=1250,
    )
    fig.add_annotation(
        x=0.01,
        y=1.03,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=f"All tasks: {win_all} wins, {lose_all} losses, {tie_all} ties (|delta| <= {delta_eps:g})",
        font=dict(size=12, color=style["font_color"]),
        xanchor="left",
        align="left",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", action="append", required=True)
    parser.add_argument("--fixed_inner_updates", type=int, default=None)
    parser.add_argument("--max_task_bars", type=int, default=20)
    parser.add_argument("--delta_eps", type=float, default=1e-6)
    parser.add_argument("--include_ties", action="store_true")
    parser.add_argument("--title", default="")
    parser.add_argument("--out_dir", default="runs/pretty_plots")
    parser.add_argument("--theme", choices=sorted(THEMES.keys()), default="light")
    parser.add_argument("--png_width", type=int, default=2200)
    parser.add_argument("--png_height", type=int, default=1400)
    parser.add_argument("--png_scale", type=float, default=1.0)
    parser.add_argument("--write_png", action="store_true")
    args = parser.parse_args()

    outcomes = _collect_outcomes(
        args.run_group, fixed_inner_updates=args.fixed_inner_updates
    )
    if not outcomes:
        raise SystemExit("No comparable task outcomes found")

    title = args.title or f"Domain Breakout: {_group_slug(args.run_group)}"
    out_tag = _slug(_group_slug(args.run_group))
    out_path = os.path.join(args.out_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-{out_tag}-domain-breakout")
    os.makedirs(out_path, exist_ok=True)

    fig = _make_figure(
        outcomes,
        title=title,
        theme_name=args.theme,
        max_task_bars=args.max_task_bars,
        delta_eps=args.delta_eps,
        include_ties=args.include_ties,
    )
    html_path = os.path.join(out_path, "dashboard.html")
    fig.write_html(html_path, include_plotlyjs="cdn")

    png_ok = False
    png_path = os.path.join(out_path, "dashboard.png")
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
            print(f"PNG export skipped (install kaleido): {exc}")

    ranked = sorted(outcomes, key=lambda x: x.delta, reverse=True)
    top_wins = [
        {
            "task_id": x.task_id,
            "entry_point": x.entry_point,
            "domain": x.domain,
            "adaptive_rate": x.adaptive_rate,
            "fixed_rate": x.fixed_rate,
            "delta": x.delta,
        }
        for x in ranked[:10]
    ]
    top_losses = [
        {
            "task_id": x.task_id,
            "entry_point": x.entry_point,
            "domain": x.domain,
            "adaptive_rate": x.adaptive_rate,
            "fixed_rate": x.fixed_rate,
            "delta": x.delta,
        }
        for x in sorted(outcomes, key=lambda x: x.delta)[:10]
    ]

    domain_stats: Dict[str, Dict[str, float]] = {}
    for domain in sorted(set(x.domain for x in outcomes)):
        vals = [x.delta for x in outcomes if x.domain == domain]
        domain_stats[domain] = {
            "n_tasks": len(vals),
            "mean_delta": sum(vals) / len(vals),
            "sd_delta": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_groups": args.run_group,
        "fixed_inner_updates": args.fixed_inner_updates,
        "max_task_bars": args.max_task_bars,
        "delta_eps": args.delta_eps,
        "include_ties": args.include_ties,
        "theme": args.theme,
        "png_width": args.png_width,
        "png_height": args.png_height,
        "png_scale": args.png_scale,
        "tasks": len(outcomes),
        "top_adaptive_wins": top_wins,
        "top_adaptive_losses": top_losses,
        "domain_stats": domain_stats,
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

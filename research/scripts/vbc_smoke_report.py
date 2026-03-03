from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _run_dirs(group_dir: Path) -> list[Path]:
    manifest = group_dir / "ablation_manifest.json"
    out: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            return
        seen.add(key)
        out.append(path)

    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for item in data.get("runs", []):
            rel = str(item.get("run_dir", "")).strip()
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = Path.cwd() / rel
            _add(path)
    for child in sorted(group_dir.iterdir()):
        if child.is_dir():
            _add(child)
    return out


def _summarize(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "events.jsonl"
    if not cfg_path.exists() or not ev_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    done: list[dict] = []
    action_counts: dict[str, int] = {}
    for ev in _iter_jsonl(ev_path):
        typ = str(ev.get("type", ""))
        if typ == "task_done":
            done.append(ev)
        elif typ == "budget_action":
            action = str(ev.get("action", ""))
            action_counts[action] = action_counts.get(action, 0) + 1

    if not done:
        return None
    n = len(done)
    return {
        "run_name": str(cfg.get("run_name", run_dir.name)),
        "policy": str(cfg.get("verifier_budget_policy", "legacy")),
        "tasks": n,
        "success": sum(1 for ev in done if ev.get("success")) / n,
        "feedback": sum(float(ev.get("feedback_calls", 0.0)) for ev in done) / n,
        "train_steps": sum(float(ev.get("train_steps", 0.0)) for ev in done) / n,
        "test_calls": sum(float(ev.get("test_calls", 0.0)) for ev in done) / n,
        "budget_action_events": sum(action_counts.values()),
        "verify_actions": action_counts.get("verify", 0),
        "refine_actions": action_counts.get("refine", 0),
        "stop_actions": action_counts.get("stop", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_group", default="verify_vbc_smoke")
    ap.add_argument(
        "--out_md",
        default="",
    )
    args = ap.parse_args()

    group_dir = Path("runs") / args.run_group
    if not group_dir.exists():
        raise SystemExit(f"run group not found: {group_dir}")

    rows = []
    for run_dir in _run_dirs(group_dir):
        row = _summarize(run_dir)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: (r["policy"], r["run_name"]))
    if not rows:
        raise SystemExit("no completed runs found")

    out_md = (
        Path(args.out_md)
        if args.out_md
        else group_dir / "vbc_smoke_report.md"
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VBC Smoke Report",
        "",
        f"- run_group: `{args.run_group}`",
        "",
        "| Run | Policy | Tasks | Success | Avg feedback | Avg train steps | Avg test calls | budget_action events | verify | refine | stop |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_name']} | {r['policy']} | {r['tasks']} | {r['success']:.3f} | "
            f"{r['feedback']:.3f} | {r['train_steps']:.3f} | {r['test_calls']:.3f} | "
            f"{r['budget_action_events']} | {r['verify_actions']} | {r['refine_actions']} | {r['stop_actions']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()


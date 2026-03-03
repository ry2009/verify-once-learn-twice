import argparse
import json


def load_from_human_eval():
    from human_eval.data import read_problems

    problems = read_problems()
    for task_id, p in problems.items():
        yield {
            "task_id": task_id,
            "prompt": p["prompt"],
            "test": p["test"],
            "entry_point": p["entry_point"],
        }


def load_from_datasets():
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    for i, row in enumerate(ds):
        yield {
            "task_id": row.get("task_id", f"HumanEval/{i}"),
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/humaneval.jsonl")
    args = parser.parse_args()

    loader = None
    try:
        loader = load_from_human_eval
        list(loader())
    except Exception:
        loader = None

    if loader is None:
        try:
            loader = load_from_datasets
            list(loader())
        except Exception as exc:
            raise SystemExit(
                "Unable to load HumanEval. Install `human_eval` or `datasets`."
            ) from exc

    with open(args.out, "w", encoding="utf-8") as f:
        for row in loader():
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Task:
    task_id: str
    prompt: str
    test: str
    entry_point: str


def load_tasks(path: str) -> List[Task]:
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(
                Task(
                    task_id=obj["task_id"],
                    prompt=obj["prompt"],
                    test=obj["test"],
                    entry_point=obj.get("entry_point", ""),
                )
            )
    return tasks


def iter_tasks(path: str) -> Iterable[Task]:
    for task in load_tasks(path):
        yield task

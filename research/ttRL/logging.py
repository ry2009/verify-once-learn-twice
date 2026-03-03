from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict


class ExperimentLogger:
    def __init__(self, run_dir: str, config: Any):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.events_path = os.path.join(run_dir, "events.jsonl")
        self._write_json(os.path.join(run_dir, "config.json"), asdict(config))
        self._write_json(
            os.path.join(run_dir, "meta.json"),
            {
                "run_dir": run_dir,
                "created_at": time.time(),
            },
        )

    def _write_json(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "time": time.time(),
            "type": event_type,
            **payload,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

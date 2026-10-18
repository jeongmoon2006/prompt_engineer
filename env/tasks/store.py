"""Task store: loads env/tasks/*.jsonl and serves Tasks (without ground truth)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from shared.schemas import Task


class TaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._tasks: dict[str, Task] = {}
        self._gt: dict[str, dict[str, Any]] = {}
        self._split: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                t = Task.model_validate(rec["task"])
                self._tasks[t.task_id] = t
                self._gt[t.task_id] = rec["ground_truth"]
                self._split[t.task_id] = rec["split"]

    def __len__(self) -> int:
        return len(self._tasks)

    def ground_truth(self, task_id: str) -> dict[str, Any] | None:
        return self._gt.get(task_id)

    def list(self, split: str | None = None, n: int | None = None, seed: int | None = None) -> list[Task]:
        ids = [i for i in self._tasks if split is None or self._split[i] == split]
        if seed is not None:
            random.Random(seed).shuffle(ids)
        if n is not None:
            ids = ids[:n]
        return [self._tasks[i] for i in ids]

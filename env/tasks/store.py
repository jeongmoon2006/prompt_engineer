"""Task store: loads env/tasks/*.jsonl per domain and serves Tasks (without ground truth)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from shared.schemas import Task


class TaskStore:
    def __init__(self, files: dict[str, Path] | Path) -> None:
        if isinstance(files, Path):
            files = {"toy": files}
        self.files = files
        self._tasks: dict[str, Task] = {}
        self._gt: dict[str, dict[str, Any]] = {}
        self._split: dict[str, str] = {}
        self._domain: dict[str, str] = {}
        for domain, path in files.items():
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                t = Task.model_validate(rec["task"])
                t.context.setdefault("domain", domain)
                self._tasks[t.task_id] = t
                self._gt[t.task_id] = rec["ground_truth"]
                self._split[t.task_id] = rec["split"]
                self._domain[t.task_id] = domain

    def __len__(self) -> int:
        return len(self._tasks)

    def domains(self) -> list[str]:
        return sorted({d for d in self._domain.values()})

    def ground_truth(self, task_id: str) -> dict[str, Any] | None:
        return self._gt.get(task_id)

    def domain_of(self, task_id: str) -> str | None:
        return self._domain.get(task_id)

    def list(self, domain: str | None = None, split: str | None = None, n: int | None = None, seed: int | None = None) -> list[Task]:
        ids = [i for i in self._tasks if (split is None or self._split[i] == split) and (domain is None or self._domain[i] == domain)]
        if seed is not None:
            random.Random(seed).shuffle(ids)
        if n is not None:
            ids = ids[:n]
        return [self._tasks[i] for i in ids]

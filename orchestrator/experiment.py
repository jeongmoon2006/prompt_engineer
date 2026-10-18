"""Per-run experiment directory: frozen configs + JSONL logs of every (task, bundle, trace, reward)."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from shared.schemas import PromptBundle, Reward, Task, Trace


class ExperimentLogger:
    def __init__(self, experiments_dir: Path, run_id: str | None = None, config_files: list[Path] = ()) -> None:
        self.run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        self.dir = Path(experiments_dir) / self.run_id
        (self.dir / "config").mkdir(parents=True, exist_ok=True)
        (self.dir / "traces").mkdir(exist_ok=True)
        for f in config_files:
            if Path(f).exists():
                shutil.copy(f, self.dir / "config" / Path(f).name)
        self._traj = (self.dir / "traces" / "trajectories.jsonl").open("a", encoding="utf-8")
        self._metrics = (self.dir / "metrics.jsonl").open("a", encoding="utf-8")

    def write_config(self, name: str, data: dict[str, Any]) -> None:
        (self.dir / "config" / name).write_text(json.dumps(data, indent=2, default=str))

    def log_trajectory(self, step: int, task: Task, bundle: PromptBundle, trace: Trace, reward: Reward) -> None:
        rec = {"step": step, "task": task.model_dump(), "bundle": bundle.model_dump(),
               "trace": trace.model_dump(), "reward": reward.model_dump()}
        self._traj.write(json.dumps(rec, default=str) + "\n")
        self._traj.flush()

    def log_metrics(self, rec: dict[str, Any]) -> None:
        self._metrics.write(json.dumps(rec, default=str) + "\n")
        self._metrics.flush()

    def close(self) -> None:
        self._traj.close()
        self._metrics.close()

"""staging-env HTTP server: POST /evaluate, GET /tasks, GET /health."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from env.reward import compute_reward, load_config
from env.tasks.store import TaskStore
from shared.schemas import EvaluateRequest, EvaluateResponse, HealthResponse, Task

SERVICE = "staging-env"
ROOT = Path(__file__).resolve().parent
TASK_FILE = Path(os.environ.get("ENV_TASK_FILE", ROOT / "tasks" / "toy_v1.jsonl"))

app = FastAPI(title=SERVICE)
CFG = load_config()
STORE = TaskStore(TASK_FILE)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, service=SERVICE)


@app.get("/tasks", response_model=list[Task])
async def tasks(
    split: str | None = Query(default=None, pattern="^(train|heldout)$"),
    n: int | None = Query(default=None, ge=1),
    seed: int | None = None,
) -> list[Task]:
    """Tasks for the orchestrator. Ground truth is never included."""
    return STORE.list(split=split, n=n, seed=seed)


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    gt = STORE.ground_truth(req.task.task_id)
    if gt is None:
        raise HTTPException(status_code=404, detail=f"unknown task_id {req.task.task_id!r}")
    bad = [t.task_id for t in req.traces if t.task_id != req.task.task_id]
    if bad:
        raise HTTPException(status_code=422, detail=f"trace task_id mismatch: {bad}")
    return EvaluateResponse(rewards=[compute_reward(req.task, t, gt, CFG) for t in req.traces])

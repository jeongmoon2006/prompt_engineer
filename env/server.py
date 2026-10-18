"""staging-env HTTP server: POST /evaluate, GET /tasks, GET /domain_info, GET /health."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from env.reward import compute_reward, load_config
from env.tasks.store import TaskStore
from shared.schemas import EvaluateRequest, EvaluateResponse, HealthResponse, Task

SERVICE = "staging-env"
ROOT = Path(__file__).resolve().parent

app = FastAPI(title=SERVICE)
CFG = load_config()
STORE = TaskStore({d: ROOT / "tasks" / spec["file"] for d, spec in CFG["domains"].items()})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, service=SERVICE)


@app.get("/domain_info")
async def domain_info(domain: str = Query(default="toy")) -> dict:
    """What an optimizer needs to know about a task family (used to build the meta-prompt)."""
    spec = CFG["domains"].get(domain)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown domain {domain!r}; have {sorted(CFG['domains'])}")
    n_tasks = len(STORE.list(domain=domain))
    return {"domain": domain, "fields": spec["fields"], "n_max": spec["n_max_tool_calls"],
            "description": spec["description"], "n_tasks": n_tasks, "loaded": n_tasks > 0}


@app.get("/tasks", response_model=list[Task])
async def tasks(
    domain: str | None = None,
    split: str | None = Query(default=None, pattern="^(train|heldout)$"),
    n: int | None = Query(default=None, ge=1),
    seed: int | None = None,
) -> list[Task]:
    """Tasks for the orchestrator. Ground truth is never included."""
    if domain is not None and domain not in CFG["domains"]:
        raise HTTPException(status_code=404, detail=f"unknown domain {domain!r}")
    return STORE.list(domain=domain, split=split, n=n, seed=seed)


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    gt = STORE.ground_truth(req.task.task_id)
    if gt is None:
        raise HTTPException(status_code=404, detail=f"unknown task_id {req.task.task_id!r}")
    bad = [t.task_id for t in req.traces if t.task_id != req.task.task_id]
    if bad:
        raise HTTPException(status_code=422, detail=f"trace task_id mismatch: {bad}")
    task = req.task.model_copy(deep=True)
    task.context.setdefault("domain", STORE.domain_of(req.task.task_id))
    return EvaluateResponse(rewards=[compute_reward(task, t, gt, CFG) for t in req.traces])

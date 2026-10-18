"""rl-orchestrator HTTP server: POST /step, GET /status, GET /health.

Nothing ML-related is imported at module load so that the server starts on a
CPU-only host (smoke tests must not require torch or a GPU).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from orchestrator.clients import EnvClient, HarnessClient
from orchestrator.trainer import Trainer, load_config
from shared.schemas import HealthResponse, StepRequest, StepResult

SERVICE = "rl-orchestrator"

app = FastAPI(title=SERVICE)
_trainer: Trainer | None = None


def get_trainer() -> Trainer:
    global _trainer
    if _trainer is None:
        cfg = load_config()
        _trainer = Trainer(cfg, HarnessClient(cfg["services"]["harness_url"]), EnvClient(cfg["services"]["env_url"]),
                           run_id=os.environ.get("RUN_ID"))
    return _trainer


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, service=SERVICE)


@app.get("/status")
async def status() -> dict:
    cfg = load_config()
    t = _trainer
    return {
        "service": SERVICE,
        "step": t.step_idx if t else 0,
        "run_id": t.logger.run_id if t else None,
        "tokens_total": t.tokens_spent if t else 0,
        "best_reward": (t.best[0] if t and t.best else None),
        "harness_url": cfg["services"]["harness_url"],
        "env_url": cfg["services"]["env_url"],
    }


@app.post("/step", response_model=StepResult)
async def step(req: StepRequest) -> StepResult:
    try:
        return await get_trainer().step(req.n_tasks)
    except RuntimeError as e:  # cost guard
        raise HTTPException(status_code=409, detail=str(e))

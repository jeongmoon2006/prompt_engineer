"""rl-orchestrator HTTP server: POST /step, GET /status, GET /health.

Nothing ML-related is imported at module load so that the server starts on a
CPU-only host (smoke tests must not require torch or a GPU).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from shared.schemas import HealthResponse, StepRequest, StepResult

SERVICE = "rl-orchestrator"

app = FastAPI(title=SERVICE)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, service=SERVICE)


@app.get("/status")
async def status() -> dict:
    return {
        "service": SERVICE,
        "step": 0,
        "harness_url": os.environ.get("HARNESS_URL", "http://localhost:8001"),
        "env_url": os.environ.get("ENV_URL", "http://localhost:8002"),
    }


@app.post("/step", response_model=StepResult)
async def step(req: StepRequest) -> StepResult:
    # Implemented in Phase 2 (trainer.py).
    raise HTTPException(status_code=501, detail="step is implemented in Phase 2")

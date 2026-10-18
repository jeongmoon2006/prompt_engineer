"""staging-env HTTP server: POST /evaluate, GET /health."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from shared.schemas import EvaluateRequest, EvaluateResponse, HealthResponse

SERVICE = "staging-env"

app = FastAPI(title=SERVICE)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, service=SERVICE)


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    # Implemented in Phase 1 (verifiers + reward.py).
    raise HTTPException(status_code=501, detail="evaluate is implemented in Phase 1")

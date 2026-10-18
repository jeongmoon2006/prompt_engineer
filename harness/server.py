"""agent-harness HTTP server: POST /run_batch, GET /health."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from shared.schemas import HealthResponse, RunBatchRequest, RunBatchResponse

SERVICE = "agent-harness"

app = FastAPI(title=SERVICE)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, service=SERVICE)


@app.post("/run_batch", response_model=RunBatchResponse)
async def run_batch(req: RunBatchRequest) -> RunBatchResponse:
    # Implemented in Phase 1 (harness.py agent loop).
    raise HTTPException(status_code=501, detail="run_batch is implemented in Phase 1")

"""agent-harness HTTP server: POST /run_batch, GET /prompts, GET /health."""

from __future__ import annotations

from fastapi import FastAPI

from harness.harness import Harness
from shared.schemas import HealthResponse, RunBatchRequest, RunBatchResponse

SERVICE = "agent-harness"

app = FastAPI(title=SERVICE)
_harness: Harness | None = None


def get_harness() -> Harness:
    global _harness
    if _harness is None:
        _harness = Harness()
    return _harness


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, service=SERVICE)


@app.get("/prompts")
async def prompts() -> dict[str, str]:
    """Baseline (hand-written) prompt slots, for optimizers to start from."""
    return get_harness().baseline


@app.post("/run_batch", response_model=RunBatchResponse)
async def run_batch(req: RunBatchRequest) -> RunBatchResponse:
    traces = await get_harness().run_batch(req.task, req.bundles)
    return RunBatchResponse(traces=traces)

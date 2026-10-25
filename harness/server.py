"""agent-harness HTTP server: POST /run_batch, GET /prompts, GET /config, GET /health."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query

from harness.harness import DEFAULT_DOMAIN, Harness
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
async def prompts(domain: str = Query(default=DEFAULT_DOMAIN)) -> dict[str, str]:
    """Baseline (hand-written) prompt slots for a domain, for optimizers to start from."""
    h = get_harness()
    if domain not in h.baselines:
        raise HTTPException(status_code=404, detail=f"unknown domain {domain!r}; have {sorted(h.baselines)}")
    return h.baselines[domain]


@app.get("/config")
async def config() -> dict:
    """Effective model + loop settings (no secrets), so an optimizer can freeze them into its run directory."""
    h = get_harness()
    return {"model": asdict(h.client.cfg), "loop": h.cfg["loop"], "slots": h.cfg.get("slots", [])}


@app.post("/run_batch", response_model=RunBatchResponse)
async def run_batch(req: RunBatchRequest) -> RunBatchResponse:
    traces = await get_harness().run_batch(req.task, req.bundles)
    return RunBatchResponse(traces=traces)

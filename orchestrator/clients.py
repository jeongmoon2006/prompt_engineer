"""Async HTTP clients for the harness and environment services (the only cross-container channel)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from shared.schemas import EvaluateRequest, EvaluateResponse, PromptBundle, Reward, RunBatchRequest, RunBatchResponse, Task, Trace


class ServiceClient:
    def __init__(self, base_url: str, timeout_s: float = 600.0, retries: int = 3, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                r = await self._http.request(method, path, **kw)
                if r.status_code in (429, 502, 503, 504) and attempt < self.retries:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                return r
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt >= self.retries or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code not in (429, 502, 503, 504)):
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    async def health(self) -> bool:
        try:
            return (await self._http.get("/health")).json().get("ok") is True
        except httpx.HTTPError:
            return False


class HarnessClient(ServiceClient):
    async def baseline_prompts(self, domain: str = "toy") -> dict[str, str]:
        return (await self._request("GET", "/prompts", params={"domain": domain})).json()

    async def run_batch(self, task: Task, bundles: list[PromptBundle]) -> list[Trace]:
        body = RunBatchRequest(task=task, bundles=bundles).model_dump()
        return RunBatchResponse.model_validate((await self._request("POST", "/run_batch", json=body)).json()).traces


class EnvClient(ServiceClient):
    async def tasks(self, domain: str | None = None, split: str | None = None, n: int | None = None, seed: int | None = None) -> list[Task]:
        params = {k: v for k, v in {"domain": domain, "split": split, "n": n, "seed": seed}.items() if v is not None}
        return [Task.model_validate(t) for t in (await self._request("GET", "/tasks", params=params)).json()]

    async def domain_info(self, domain: str = "toy") -> dict[str, Any]:
        return (await self._request("GET", "/domain_info", params={"domain": domain})).json()

    async def evaluate(self, task: Task, traces: list[Trace]) -> list[Reward]:
        body = EvaluateRequest(task=task, traces=traces).model_dump()
        return EvaluateResponse.model_validate((await self._request("POST", "/evaluate", json=body)).json()).rewards

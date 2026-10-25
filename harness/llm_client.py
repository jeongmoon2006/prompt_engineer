"""Remote-model client for the frozen harness model.

Providers:
  anthropic  - Claude via the official SDK (AsyncAnthropic). The SDK retries 429/5xx and
               connection errors with exponential backoff (max_retries).
  mock       - deterministic offline stand-in used by CPU smoke tests and CI. It makes one
               lookup call then emits JSON; it is NOT an oracle and is intentionally imperfect.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic


@dataclass
class LLMResponse:
    content: list[Any]                 # raw content blocks (passed back verbatim on the next turn)
    text: str
    tool_calls: list[dict[str, Any]]   # [{id, name, input}]
    stop_reason: str
    stop_category: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: float
    request_id: str | None = None


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-haiku-5-5"
    effort: str = "low"
    thinking: str = "disabled"  # adaptive | disabled
    max_tokens: int = 1024
    max_retries: int = 5
    timeout_s: float = 60.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, defaults: dict[str, Any]) -> LLMConfig:
        return cls(
            provider=os.environ.get("HARNESS_PROVIDER", defaults.get("provider", "anthropic")),
            model=os.environ.get("HARNESS_MODEL", defaults.get("model", "claude-haiku-5-5")),
            effort=defaults.get("effort", "low"),
            thinking=os.environ.get("HARNESS_THINKING", defaults.get("thinking", "disabled")),
            max_tokens=int(defaults.get("max_tokens", 1024)),
            max_retries=int(defaults.get("max_retries", 5)),
            timeout_s=float(defaults.get("timeout_s", 60)),
        )


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client: anthropic.AsyncAnthropic | None = None
        if cfg.provider == "anthropic":
            # Reads ANTHROPIC_API_KEY from the environment; never hard-code keys.
            self._client = anthropic.AsyncAnthropic(max_retries=cfg.max_retries, timeout=cfg.timeout_s)
        elif cfg.provider != "mock":
            raise ValueError(f"unknown HARNESS_PROVIDER {cfg.provider!r}")

    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        if self.cfg.provider == "mock":
            return await self._mock(messages, tools)
        assert self._client is not None
        t0 = time.perf_counter()
        resp = await self._client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"},
            output_config={"effort": self.cfg.effort},
            # "disabled" is accepted at effort <= high on Haiku 5.5; otherwise adaptive (the default).
            thinking={"type": "disabled"} if self.cfg.thinking == "disabled" else {"type": "adaptive"},
            # Haiku 5.5: no temperature/top_p (non-default values are rejected).
        )
        latency = (time.perf_counter() - t0) * 1000
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "input": b.input} for b in resp.content if b.type == "tool_use"]
        category = None
        if resp.stop_reason == "refusal" and resp.stop_details is not None:
            category = getattr(resp.stop_details, "category", None)
        return LLMResponse(
            content=[b.model_dump() for b in resp.content],
            text=text,
            tool_calls=calls,
            stop_reason=resp.stop_reason or "unknown",
            stop_category=category,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            latency_ms=latency,
            request_id=resp._request_id,
        )

    # ------------------------------------------------------------------ mock
    async def _mock(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        await asyncio.sleep(0)
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        approx_in = sum(len(json.dumps(m, default=str)) for m in messages) // 4
        if n_assistant == 0 and tools:
            arg = {"field": "temperature"} if tools[0]["name"] == "lookup" else {"expression": "2+2"}
            calls = [{"id": "toolu_mock_1", "name": tools[0]["name"], "input": arg}]
            content = [{"type": "tool_use", **c} for c in calls]
            return LLMResponse(content, "", calls, "tool_use", None, approx_in, 20, 1.0)
        # Crude extraction from whatever text is in the conversation (not an oracle).
        blob = json.dumps(messages, default=str)
        sid = re.search(r"\b((?:S-|LN-|BX|P)\d{4}[A-Za-z0-9-]*)", blob)
        temp = re.search(r"(\d+(?:\.\d+)?) K\b", blob)
        conc = re.search(r"(\d+(?:\.\d+)?) mM", blob)
        meth = re.search(r"\b(DLS|NMR|UV-Vis|SAXS|ITC|cryo-EM|FTIR|HPLC)\b", blob)
        out = {
            "sample_id": sid.group(1) if sid else "unknown",
            "temperature_K": float(temp.group(1)) if temp else 0.0,
            "concentration_mM": float(conc.group(1)) if conc else 0.0,
            "method": meth.group(1) if meth else "unknown",
        }
        text = json.dumps(out)
        return LLMResponse([{"type": "text", "text": text}], text, [], "end_turn", None, approx_in, len(text) // 4, 1.0)

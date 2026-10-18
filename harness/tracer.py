"""Structured trace capture for one trajectory."""

from __future__ import annotations

from typing import Any

from shared.schemas import Trace


class Tracer:
    def __init__(self, task_id: str, bundle_id: str) -> None:
        self.task_id = task_id
        self.bundle_id = bundle_id
        self.steps: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.error: str | None = None

    def user(self, content: str) -> None:
        self.steps.append({"role": "user", "content": content, "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0})

    def assistant(self, text: str, tool_calls: list[dict[str, Any]], tokens_in: int, tokens_out: int,
                  latency_ms: float, stop_reason: str, request_id: str | None = None) -> None:
        step: dict[str, Any] = {
            "role": "assistant", "content": text, "tokens_in": tokens_in, "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 2), "stop_reason": stop_reason,
        }
        if tool_calls:
            step["tool_call"] = tool_calls
        if request_id:
            step["request_id"] = request_id
        self.steps.append(step)
        self.total_tokens += tokens_in + tokens_out
        self.total_latency_ms += latency_ms

    def tool_result(self, results: list[dict[str, Any]]) -> None:
        self.steps.append({"role": "tool", "content": "", "tool_result": results,
                           "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0})

    def finish(self, final_output: str) -> Trace:
        return Trace(
            task_id=self.task_id, bundle_id=self.bundle_id, steps=self.steps, final_output=final_output,
            total_tokens=self.total_tokens, total_latency_ms=round(self.total_latency_ms, 2), error=self.error,
        )

"""Cross-container data contracts.

This is the ONLY module that all three containers import. Everything that
crosses an HTTP boundary is defined here as a pydantic v2 model.

Freeze policy (WORK_INSTRUCTIONS §3): frozen at the end of Phase 1. Later
phases may add *optional* fields but must never rename or remove any.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------


class Task(BaseModel):
    task_id: str
    description: str  # what the agent is asked to do
    context: dict[str, Any] = Field(default_factory=dict)  # task-specific inputs


class PromptBundle(BaseModel):
    bundle_id: str
    slots: dict[str, str]  # slot_name -> prompt text, e.g. {"system": ..., "planner": ...}


class Trace(BaseModel):
    task_id: str
    bundle_id: str
    # [{role, content, tool_call?, tool_result?, tokens_in, tokens_out, latency_ms}]
    steps: list[dict[str, Any]] = Field(default_factory=list)
    final_output: str = ""
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    error: str | None = None


class Reward(BaseModel):
    task_id: str
    bundle_id: str
    total: float
    components: dict[str, float]  # e.g. {"correctness": 0.8, "format": 1.0, "cost": -0.12}
    # Natural-language failure summary. Unused by GRPO; populated for later GEPA use.
    feedback: str | None = None


class StepResult(BaseModel):
    """Summary returned by the orchestrator after one GRPO iteration."""

    step: int
    mean_reward: float
    std_reward: float = 0.0
    kl: float = 0.0
    loss: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)  # ratio stats, clip frac, VRAM, ...


# ---------------------------------------------------------------------------
# HTTP request / response envelopes
# ---------------------------------------------------------------------------


class RunBatchRequest(BaseModel):
    task: Task
    bundles: list[PromptBundle]


class RunBatchResponse(BaseModel):
    traces: list[Trace]


class EvaluateRequest(BaseModel):
    task: Task
    traces: list[Trace]


class EvaluateResponse(BaseModel):
    rewards: list[Reward]


class StepRequest(BaseModel):
    n_tasks: int = Field(default=8, ge=1)


class HealthResponse(BaseModel):
    ok: bool = True
    service: str

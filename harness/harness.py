"""Agent loop with prompt-slot injection.

Slots (k = 3) and where each is injected:
  system       -> API system prompt
  planner      -> user-turn instruction preceding the notebook entry
  tool_policy  -> description of the `lookup` tool

The loop is append-only (required for Haiku 5.5's thinking blocks). Tool calls are
served locally from task.context["lookup"]; the model never sees ground truth.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
import yaml

from harness.llm_client import LLMClient, LLMConfig
from harness.tracer import Tracer
from shared.schemas import PromptBundle, Task, Trace

ROOT = Path(__file__).resolve().parent
TOOL_NAME = "lookup"


def load_config(path: Path = ROOT / "config.yaml") -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_baseline_prompts(prompts_dir: Path = ROOT / "prompts") -> dict[str, str]:
    slots: dict[str, str] = {}
    for f in sorted(prompts_dir.glob("*.yaml")):
        d = yaml.safe_load(f.read_text())
        slots[d["slot"]] = d["text"].strip()
    return slots


def build_tool(task: Task, tool_policy: str) -> dict[str, Any]:
    fields = task.context.get("lookup_fields") or sorted(task.context.get("lookup", {}).keys())
    return {
        "name": TOOL_NAME,
        "description": tool_policy,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"field": {"type": "string", "enum": fields}},
            "required": ["field"],
            "additionalProperties": False,
        },
    }


class Harness:
    def __init__(self, cfg: dict[str, Any] | None = None, client: LLMClient | None = None) -> None:
        self.cfg = cfg or load_config()
        self.client = client or LLMClient(LLMConfig.from_env(self.cfg["model_defaults"]))
        self.baseline = load_baseline_prompts()
        import os
        conc = int(os.environ.get("HARNESS_MAX_CONCURRENCY", self.cfg["loop"]["max_concurrency"]))
        self._sem = asyncio.Semaphore(conc)

    def resolve_slots(self, bundle: PromptBundle) -> dict[str, str]:
        slots = dict(self.baseline)
        slots.update({k: v for k, v in bundle.slots.items() if k in slots})
        return slots

    async def run_batch(self, task: Task, bundles: list[PromptBundle]) -> list[Trace]:
        return list(await asyncio.gather(*(self.run_one(task, b) for b in bundles)))

    async def run_one(self, task: Task, bundle: PromptBundle) -> Trace:
        async with self._sem:
            return await self._run(task, bundle)

    async def _run(self, task: Task, bundle: PromptBundle) -> Trace:
        slots = self.resolve_slots(bundle)
        loop_cfg = self.cfg["loop"]
        n_max = int(task.context.get("n_max_tool_calls", 4))
        tracer = Tracer(task.task_id, bundle.bundle_id)
        tool = build_tool(task, slots["tool_policy"])
        lookup: dict[str, str] = task.context.get("lookup", {})

        user_text = f"{slots['planner']}\n\nTask: {task.description}\n\nNotebook entry:\n{task.context.get('notebook_text', '')}"
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
        tracer.user(user_text)
        final_text = ""
        n_calls = 0

        for _turn in range(int(loop_cfg["max_turns"])):
            try:
                resp = await self.client.complete(slots["system"], messages, [tool])
            except anthropic.APIStatusError as e:  # non-retryable 4xx or retries exhausted
                tracer.error = f"api_error:{e.status_code}:{e.message[:200]}"
                break
            except anthropic.APIConnectionError as e:
                tracer.error = f"connection_error:{str(e)[:200]}"
                break
            tracer.assistant(resp.text, resp.tool_calls, resp.tokens_in, resp.tokens_out,
                             resp.latency_ms, resp.stop_reason, resp.request_id)
            if resp.text:
                final_text = resp.text
            if resp.stop_reason == "refusal":
                tracer.error = f"refusal:{resp.stop_category}"
                break
            if tracer.total_tokens > int(loop_cfg["token_budget"]):
                tracer.error = "token_budget_exceeded"
                break
            if not resp.tool_calls:
                if resp.stop_reason == "max_tokens" and not resp.text:
                    tracer.error = "max_tokens_without_output"
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for call in resp.tool_calls:
                n_calls += 1
                field = str(call["input"].get("field", "")) if isinstance(call["input"], dict) else ""
                if call["name"] != TOOL_NAME:
                    out, err = f"unknown tool {call['name']!r}", True
                elif n_calls > n_max:
                    out, err = f"lookup budget exhausted ({n_max} calls max)", True
                elif field in lookup:
                    out, err = lookup[field], False
                else:
                    out, err = f"no metadata recorded for field {field!r}", True
                results.append({"type": "tool_result", "tool_use_id": call["id"], "content": out, "is_error": err})
            tracer.tool_result([{"tool_use_id": r["tool_use_id"], "field": c["input"], "content": r["content"],
                                 "is_error": r["is_error"]} for r, c in zip(results, resp.tool_calls)])
            messages.append({"role": "user", "content": results})
        else:
            tracer.error = tracer.error or "max_turns_exceeded"
        return tracer.finish(final_text)

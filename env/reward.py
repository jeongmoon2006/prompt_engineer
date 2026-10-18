"""Composite reward. Deterministic: identical (task, trace, ground truth, config) -> identical Reward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from env.verifiers.correctness import score_correctness
from env.verifiers.cost import score_cost
from env.verifiers.format import score_format
from env.verifiers.tool_efficiency import score_tool_efficiency
from shared.schemas import Reward, Task, Trace

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def compute_reward(task: Task, trace: Trace, ground_truth: dict[str, Any], cfg: dict[str, Any]) -> Reward:
    rc = cfg["reward"]
    weights: dict[str, float] = rc["weights"]
    fields: list[str] = task.context.get("output_fields", list(ground_truth.keys()))
    n_max: int = int(task.context.get("n_max_tool_calls", rc["n_max_tool_calls"]))

    fmt, fmt_fb = score_format(trace.final_output, fields)
    corr, per_field, corr_fb = score_correctness(trace.final_output, ground_truth, rc["numeric_scales"])
    eff, n_calls = score_tool_efficiency(trace.steps, n_max)
    cost = score_cost(trace.total_tokens, int(rc["token_budget"]), float(rc["cost_lambda"]))

    components = {"correctness": corr, "format": fmt, "tool_efficiency": eff, "cost": cost}
    total = sum(weights[k] * components[k] for k in weights)

    fb_parts = []
    if trace.error:
        fb_parts.append(f"harness error: {trace.error}")
    if fmt_fb:
        fb_parts.append(fmt_fb)
    if corr_fb:
        fb_parts.append(corr_fb)
    fb_parts.append(f"{n_calls}/{n_max} tool calls, {trace.total_tokens} tokens")
    return Reward(
        task_id=trace.task_id,
        bundle_id=trace.bundle_id,
        total=round(total, 6),
        components={k: round(v, 6) for k, v in components.items()},
        feedback="; ".join(fb_parts),
    )

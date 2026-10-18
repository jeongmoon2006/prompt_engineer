"""tool_efficiency: 1 - tool_calls / N_max, clipped to [0, 1]."""

from __future__ import annotations

from typing import Any


def count_tool_calls(steps: list[dict[str, Any]]) -> int:
    n = 0
    for s in steps:
        tc = s.get("tool_call")
        if tc is None:
            continue
        n += len(tc) if isinstance(tc, list) else 1
    return n


def score_tool_efficiency(steps: list[dict[str, Any]], n_max: int) -> tuple[float, int]:
    calls = count_tool_calls(steps)
    if n_max <= 0:
        return 0.0, calls
    return max(0.0, min(1.0, 1.0 - calls / n_max)), calls

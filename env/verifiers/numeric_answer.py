"""correctness for single-number answers (GSM8K): exact numeric match (tolerance 1e-6) -> 1 else 0."""

from __future__ import annotations

import math
import re
from typing import Any

from env.verifiers.parsing import extract_json_object


def parse_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace(",", "").replace("$", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None
    return None


def score_numeric_answer(final_output: str, truth: float, field: str = "answer") -> tuple[float, str]:
    obj = extract_json_object(final_output)
    if obj is None or field not in obj:
        return 0.0, f"{field} missing"
    p = parse_number(obj[field])
    if p is None or not math.isfinite(p):
        return 0.0, f"{field} not numeric ({obj[field]!r})"
    ok = math.isclose(p, float(truth), rel_tol=1e-6, abs_tol=1e-6)
    return (1.0, "") if ok else (0.0, f"{field}={p:g} vs truth {float(truth):g}")

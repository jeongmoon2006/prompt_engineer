"""correctness: fraction of fields correct; numeric fields get partial credit exp(-|delta|/scale)."""

from __future__ import annotations

import math
from typing import Any

from env.verifiers.parsing import extract_json_object


def _to_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip().split()[0])
        except (ValueError, IndexError):
            return None
    return None


def _norm_str(v: Any) -> str:
    return str(v).strip().lower().replace(" ", "").replace("_", "-")


def score_correctness(
    final_output: str,
    ground_truth: dict[str, Any],
    numeric_scales: dict[str, float],
) -> tuple[float, dict[str, float], str]:
    """Returns (fraction, per-field scores, feedback)."""
    obj = extract_json_object(final_output) or {}
    per_field: dict[str, float] = {}
    problems: list[str] = []
    for field, truth in ground_truth.items():
        pred = obj.get(field)
        if pred is None:
            per_field[field] = 0.0
            problems.append(f"{field} missing")
            continue
        if field in numeric_scales:
            p = _to_float(pred)
            if p is None or not math.isfinite(p):
                per_field[field] = 0.0
                problems.append(f"{field} not numeric ({pred!r})")
            else:
                s = math.exp(-abs(p - float(truth)) / numeric_scales[field])
                per_field[field] = s
                if s < 0.999:
                    problems.append(f"{field}={p:g} vs truth {float(truth):g}")
        else:
            ok = _norm_str(pred) == _norm_str(truth)
            per_field[field] = 1.0 if ok else 0.0
            if not ok:
                problems.append(f"{field}={pred!r} vs truth {truth!r}")
    frac = sum(per_field.values()) / max(len(per_field), 1)
    return frac, per_field, "; ".join(problems)

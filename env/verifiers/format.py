"""format: 1.0 bare JSON object with exactly the required keys and numeric types;
0.75 same but wrapped in a code fence or prose; 0.5 parseable JSON object with wrong keys/types; 0 otherwise."""

from __future__ import annotations

import json
from typing import Any

from env.verifiers.parsing import extract_json_object

# Legacy default (toy domain) used only when the caller passes no numeric_fields.
NUMERIC_FIELDS = {"temperature_K", "concentration_mM"}


def _is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v.strip())
            return True
        except ValueError:
            return False
    return False


def _is_bare_json(text: str) -> bool:
    try:
        return isinstance(json.loads(text.strip()), dict)
    except (json.JSONDecodeError, ValueError):
        return False


def score_format(final_output: str, required_fields: list[str],
                 numeric_fields: set[str] | list[str] | None = None) -> tuple[float, str]:
    numeric = set(NUMERIC_FIELDS if numeric_fields is None else numeric_fields)
    obj = extract_json_object(final_output)
    if obj is None:
        return 0.0, "output is not a JSON object"
    missing = [f for f in required_fields if f not in obj]
    extra = [k for k in obj if k not in required_fields]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing keys {missing}")
        if extra:
            parts.append(f"unexpected keys {extra}")
        return 0.5, "JSON parsed but schema mismatch: " + ", ".join(parts)
    bad_types = [f for f in required_fields if f in numeric and not _is_number(obj[f])]
    if bad_types:
        return 0.5, f"non-numeric value for {bad_types}"
    if not _is_bare_json(final_output):
        return 0.75, "JSON wrapped in code fence or prose (bare JSON expected)"
    return 1.0, ""

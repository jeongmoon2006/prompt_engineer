"""format: 1.0 valid JSON with the required keys, 0.5 parseable JSON object with wrong keys, 0 otherwise."""

from __future__ import annotations

from typing import Any

from env.verifiers.parsing import extract_json_object

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


def score_format(final_output: str, required_fields: list[str]) -> tuple[float, str]:
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
    bad_types = [f for f in required_fields if f in NUMERIC_FIELDS and not _is_number(obj[f])]
    if bad_types:
        return 0.5, f"non-numeric value for {bad_types}"
    return 1.0, ""

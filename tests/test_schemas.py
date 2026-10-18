"""Shared schemas round-trip through JSON and reject malformed payloads."""

import pytest
from pydantic import ValidationError

from shared.schemas import PromptBundle, Reward, Task, Trace


def test_roundtrip() -> None:
    r = Reward(task_id="t", bundle_id="b", total=0.7,
               components={"correctness": 0.8, "format": 1.0, "cost": -0.1})
    assert Reward.model_validate_json(r.model_dump_json()) == r
    t = Trace(task_id="t", bundle_id="b", steps=[{"role": "user", "content": "hi"}],
              final_output="{}", total_tokens=3, total_latency_ms=1.5)
    assert Trace.model_validate_json(t.model_dump_json()) == t
    assert Task(task_id="t", description="d").context == {}
    assert PromptBundle(bundle_id="b", slots={}).slots == {}


def test_reward_requires_total() -> None:
    with pytest.raises(ValidationError):
        Reward(task_id="t", bundle_id="b", components={})  # type: ignore[call-arg]

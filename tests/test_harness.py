"""Harness agent loop with the offline mock provider: slot injection, tool serving, trace fidelity."""

import pytest
from fastapi.testclient import TestClient

from harness.harness import Harness, build_tool, load_baseline_prompts
from harness.llm_client import LLMClient, LLMConfig
from shared.schemas import PromptBundle, Task


def mock_harness() -> Harness:
    return Harness(client=LLMClient(LLMConfig(provider="mock")))


def task() -> Task:
    return Task(task_id="t1", description="extract", context={
        "notebook_text": "Sample S-0042 measured at 298.15 K, 2.5 mM, by DLS.",
        "lookup": {"temperature": "Measurement temperature: 298.15 K", "method": "Technique: DLS"},
        "lookup_fields": ["temperature", "method", "sample_id"],
        "n_max_tool_calls": 4, "output_fields": ["sample_id", "temperature_K", "concentration_mM", "method"],
    })


def test_baseline_prompts_have_three_slots() -> None:
    assert set(load_baseline_prompts()) == {"system", "planner", "tool_policy"}


def test_slot_resolution_overrides_only_known_slots() -> None:
    h = mock_harness()
    slots = h.resolve_slots(PromptBundle(bundle_id="b", slots={"planner": "P!", "bogus": "x"}))
    assert slots["planner"] == "P!" and slots["system"] == h.baseline["system"] and "bogus" not in slots


def test_tool_definition_uses_tool_policy_slot() -> None:
    tool = build_tool(task(), "MY POLICY")
    assert tool["description"] == "MY POLICY" and tool["strict"] is True
    assert tool["input_schema"]["properties"]["field"]["enum"] == ["temperature", "method", "sample_id"]


async def test_run_batch_traces_are_faithful() -> None:
    h = mock_harness()
    bundles = [PromptBundle(bundle_id=f"b{i}", slots={}) for i in range(3)]
    traces = await h.run_batch(task(), bundles)
    assert [t.bundle_id for t in traces] == ["b0", "b1", "b2"]
    for t in traces:
        assert t.error is None
        roles = [s["role"] for s in t.steps]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert t.steps[1]["tool_call"][0]["name"] == "lookup"
        assert t.steps[2]["tool_result"][0]["content"] == "Measurement temperature: 298.15 K"
        assert t.total_tokens == sum(s["tokens_in"] + s["tokens_out"] for s in t.steps) > 0
        assert '"sample_id": "S-0042"' in t.final_output


async def test_unknown_lookup_field_is_error_result() -> None:
    h = mock_harness()
    tk = task()
    tk.context["lookup"] = {}
    tr = (await h.run_batch(tk, [PromptBundle(bundle_id="b", slots={})]))[0]
    assert tr.steps[2]["tool_result"][0]["is_error"] is True


def test_server_prompts_and_run_batch() -> None:
    import harness.server as srv
    srv._harness = mock_harness()
    c = TestClient(srv.app)
    assert set(c.get("/prompts").json()) == {"system", "planner", "tool_policy"}
    r = c.post("/run_batch", json={"task": task().model_dump(), "bundles": [PromptBundle(bundle_id="b", slots={}).model_dump()]})
    assert r.status_code == 200 and len(r.json()["traces"]) == 1

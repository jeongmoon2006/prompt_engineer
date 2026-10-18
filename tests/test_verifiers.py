"""Unit tests for each reward component on hand-written traces, plus determinism."""

import math

import pytest

from env.reward import compute_reward, load_config
from env.verifiers.correctness import score_correctness
from env.verifiers.cost import score_cost
from env.verifiers.format import score_format
from env.verifiers.parsing import extract_json_object
from env.verifiers.tool_efficiency import score_tool_efficiency
from shared.schemas import Task, Trace

FIELDS = ["sample_id", "temperature_K", "concentration_mM", "method"]
GT = {"sample_id": "S-0001", "temperature_K": 298.15, "concentration_mM": 2.5, "method": "DLS"}
SCALES = {"temperature_K": 5.0, "concentration_mM": 2.0}
GOOD = '{"sample_id": "S-0001", "temperature_K": 298.15, "concentration_mM": 2.5, "method": "DLS"}'


def test_extract_json_variants() -> None:
    assert extract_json_object(GOOD)["method"] == "DLS"
    assert extract_json_object(f"Here you go:\n```json\n{GOOD}\n```")["method"] == "DLS"
    assert extract_json_object(f"The record is {GOOD} as requested.")["method"] == "DLS"
    assert extract_json_object("[1,2,3]") is None
    assert extract_json_object("") is None
    assert extract_json_object("not json {oops") is None


def test_format_scores() -> None:
    assert score_format(GOOD, FIELDS)[0] == 1.0
    assert score_format(f"```json\n{GOOD}\n```", FIELDS)[0] == 0.75
    assert score_format(f"Here is the record: {GOOD}", FIELDS)[0] == 0.75
    assert score_format('{"sample_id": "x", "temp": 1}', FIELDS)[0] == 0.5
    assert score_format(GOOD.replace("}", ', "extra": 1}'), FIELDS)[0] == 0.5
    assert score_format('{"sample_id": "S", "temperature_K": "hot", "concentration_mM": 1, "method": "x"}', FIELDS)[0] == 0.5
    assert score_format("no json here", FIELDS)[0] == 0.0


def test_correctness_exact_and_partial() -> None:
    frac, per, fb = score_correctness(GOOD, GT, SCALES)
    assert frac == 1.0 and fb == ""
    off = '{"sample_id": "S-0001", "temperature_K": 303.15, "concentration_mM": 2.5, "method": "DLS"}'
    frac, per, fb = score_correctness(off, GT, SCALES)
    assert math.isclose(per["temperature_K"], math.exp(-1.0))
    assert math.isclose(frac, (3 + math.exp(-1.0)) / 4)
    assert "temperature_K" in fb
    wrong = '{"sample_id": "S-0002", "temperature_K": "n/a", "concentration_mM": 2.5, "method": "nmr"}'
    frac, per, _ = score_correctness(wrong, GT, SCALES)
    assert per["sample_id"] == 0.0 and per["temperature_K"] == 0.0 and per["method"] == 0.0
    assert math.isclose(frac, 0.25)
    assert score_correctness("garbage", GT, SCALES)[0] == 0.0
    # numeric strings and case-insensitive strings get credit
    assert score_correctness(GOOD.replace("298.15", '"298.15"').replace('"DLS"', '"dls"'), GT, SCALES)[0] == 1.0


def test_tool_efficiency() -> None:
    steps = [{"role": "assistant", "tool_call": [{"id": "a"}, {"id": "b"}]}, {"role": "tool"}, {"role": "assistant"}]
    assert score_tool_efficiency(steps, 4) == (0.5, 2)
    assert score_tool_efficiency([], 4) == (1.0, 0)
    many = [{"role": "assistant", "tool_call": [{"id": str(i)} for i in range(7)]}]
    assert score_tool_efficiency(many, 4) == (0.0, 7)


def test_cost() -> None:
    assert score_cost(2000, 4000, 1.0) == -0.5
    assert score_cost(0, 4000, 1.0) == 0.0
    assert score_cost(100, 0, 1.0) == 0.0


def _task() -> Task:
    return Task(task_id="t", description="d", context={"output_fields": FIELDS, "n_max_tool_calls": 4})


def _trace(output: str, calls: int = 1, tokens: int = 800) -> Trace:
    steps = [{"role": "assistant", "content": "", "tool_call": [{"id": str(i)} for i in range(calls)]}] if calls else []
    return Trace(task_id="t", bundle_id="b", steps=steps, final_output=output, total_tokens=tokens)


def test_composite_reward_and_weights() -> None:
    cfg = load_config()
    r = compute_reward(_task(), _trace(GOOD, calls=1, tokens=800), GT, cfg)
    assert set(r.components) == {"correctness", "format", "tool_efficiency", "cost"}
    w = cfg["reward"]["weights"]
    expected = w["correctness"] * 1.0 + w["format"] * 1.0 + w["tool_efficiency"] * 0.75 + w["cost"] * (-800 / 4000)
    assert math.isclose(r.total, expected, abs_tol=1e-6)
    assert all(math.isfinite(v) for v in r.components.values())
    assert r.feedback and "1/4 tool calls" in r.feedback


def test_reward_feedback_on_failure() -> None:
    cfg = load_config()
    t = _trace("", calls=0, tokens=100)
    t.error = "refusal:general_harms"
    r = compute_reward(_task(), t, GT, cfg)
    assert r.components["correctness"] == 0.0 and r.components["format"] == 0.0
    assert "harness error: refusal" in r.feedback


def test_determinism() -> None:
    cfg = load_config()
    tr = _trace(GOOD.replace("2.5", "3.1"), calls=2, tokens=1234)
    a = compute_reward(_task(), tr, GT, cfg)
    b = compute_reward(_task(), Trace.model_validate(tr.model_dump()), GT, cfg)
    assert a == b

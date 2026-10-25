"""Environment HTTP: /tasks hides ground truth; /evaluate is deterministic and validates ids."""

from fastapi.testclient import TestClient

from env.server import app
from shared.schemas import Trace

c = TestClient(app)


def test_tasks_endpoint() -> None:
    r = c.get("/tasks", params={"split": "heldout", "n": 3, "seed": 1})
    assert r.status_code == 200 and len(r.json()) == 3
    assert all("ground_truth" not in t["context"] for t in r.json())
    assert c.get("/tasks", params={"split": "bogus"}).status_code == 422


def test_evaluate_roundtrip_and_determinism() -> None:
    task = c.get("/tasks", params={"n": 1}).json()[0]
    trace = Trace(task_id=task["task_id"], bundle_id="b", final_output='{"sample_id": "x", "temperature_K": 1, "concentration_mM": 1, "method": "x"}',
                  steps=[], total_tokens=500).model_dump()
    r1 = c.post("/evaluate", json={"task": task, "traces": [trace, trace]})
    assert r1.status_code == 200
    rewards = r1.json()["rewards"]
    assert rewards[0] == rewards[1]
    assert rewards[0]["components"]["format"] == 1.0
    r2 = c.post("/evaluate", json={"task": task, "traces": [trace]})
    assert r2.json()["rewards"][0] == rewards[0]


def test_evaluate_rejects_unknown_or_mismatched() -> None:
    task = c.get("/tasks", params={"n": 1}).json()[0]
    bad_task = dict(task, task_id="nope")
    tr = Trace(task_id="nope", bundle_id="b").model_dump()
    assert c.post("/evaluate", json={"task": bad_task, "traces": [tr]}).status_code == 404
    tr2 = Trace(task_id="other", bundle_id="b").model_dump()
    assert c.post("/evaluate", json={"task": task, "traces": [tr2]}).status_code == 422


def test_config_endpoint_exposes_reward_weights() -> None:
    conf = c.get("/config").json()
    assert set(conf["reward"]["weights"]) == {"correctness", "format", "tool_efficiency", "cost"}
    assert conf["domains"]["gsm8k"]["verifier"] == "numeric_answer" and conf["domains"]["toy"]["verifier"] == "fields"


def test_domain_info_and_domain_filter() -> None:
    info = c.get("/domain_info", params={"domain": "toy"}).json()
    assert info["fields"] == ["sample_id", "temperature_K", "concentration_mM", "method"] and info["n_max"] == 4 and info["loaded"]
    assert c.get("/domain_info", params={"domain": "nope"}).status_code == 404
    assert c.get("/tasks", params={"domain": "nope"}).status_code == 404
    ts = c.get("/tasks", params={"domain": "toy", "n": 2}).json()
    assert all(t["context"]["domain"] == "toy" for t in ts)

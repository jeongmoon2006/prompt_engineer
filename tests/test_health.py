"""Phase 0: every service exposes GET /health -> {"ok": true}; stubs return 501."""

import pytest
from fastapi.testclient import TestClient

from env.server import app as env_app
from harness.server import app as harness_app
from orchestrator.server import app as orch_app
from shared.schemas import PromptBundle, Task, Trace

APPS = {"rl-orchestrator": orch_app, "agent-harness": harness_app, "staging-env": env_app}


@pytest.mark.parametrize("name", list(APPS))
def test_health(name: str) -> None:
    r = TestClient(APPS[name]).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == name


def test_stubs_return_501_until_implemented() -> None:
    task = Task(task_id="t0", description="x").model_dump()
    bundle = PromptBundle(bundle_id="b0", slots={"system": "s"}).model_dump()
    trace = Trace(task_id="t0", bundle_id="b0").model_dump()
    assert TestClient(harness_app).post("/run_batch", json={"task": task, "bundles": [bundle]}).status_code == 501
    assert TestClient(env_app).post("/evaluate", json={"task": task, "traces": [trace]}).status_code == 501
    assert TestClient(orch_app).post("/step", json={"n_tasks": 1}).status_code == 501


def test_request_validation() -> None:
    assert TestClient(harness_app).post("/run_batch", json={"bad": 1}).status_code == 422

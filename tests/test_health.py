"""Phase 0: every service exposes GET /health -> {"ok": true}; stubs return 501."""

import pytest
from fastapi.testclient import TestClient

from env.server import app as env_app
from harness.server import app as harness_app
from orchestrator.server import app as orch_app

APPS = {"rl-orchestrator": orch_app, "agent-harness": harness_app, "staging-env": env_app}


@pytest.mark.parametrize("name", list(APPS))
def test_health(name: str) -> None:
    r = TestClient(APPS[name]).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == name


def test_step_requires_live_services_or_fails_cleanly() -> None:
    # /step is real now; without reachable services it must not hang or 500 silently.
    import orchestrator.server as srv
    srv._trainer = None
    assert TestClient(orch_app).post("/step", json={"n_tasks": 0}).status_code == 422


def test_request_validation() -> None:
    assert TestClient(harness_app).post("/run_batch", json={"bad": 1}).status_code == 422

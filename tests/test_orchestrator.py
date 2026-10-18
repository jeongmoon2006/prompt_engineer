"""Orchestrator: random policy shapes, and the full loop in-process (mock LLM) through the HTTP contracts."""

import math
import random
from pathlib import Path

import httpx
import pytest

import env.server as env_srv
import harness.server as harness_srv
from harness.harness import Harness
from harness.llm_client import LLMClient, LLMConfig
from orchestrator.clients import EnvClient, HarnessClient
from orchestrator.policy_random import RandomPerturbationPolicy, perturb_text
from orchestrator.trainer import Trainer, load_config


def test_perturb_text_changes_and_is_seeded() -> None:
    text = "Read the entry. Identify the fields. Respond with a single JSON object."
    a = perturb_text(text, random.Random(1), p_swap=1.0)
    b = perturb_text(text, random.Random(1), p_swap=1.0)
    assert a == b and a != text
    assert len(a.split(".")) == len(text.split("."))  # same sentence count


def test_random_policy_sampling() -> None:
    pol = RandomPerturbationPolicy({"system": "Be careful.", "planner": "Read it. Identify things.", "tool_policy": "Only when stated."}, seed=3)
    bundles = pol.sample(4)
    assert len(bundles) == 4 and len({b.bundle_id for b in bundles}) == 4
    assert all(set(b.slots) == {"system", "planner", "tool_policy"} for b in bundles)


def _mock_clients() -> tuple[HarnessClient, EnvClient]:
    harness_srv._harness = Harness(client=LLMClient(LLMConfig(provider="mock")))
    h = HarnessClient("http://harness", transport=httpx.ASGITransport(app=harness_srv.app))
    e = EnvClient("http://env", transport=httpx.ASGITransport(app=env_srv.app))
    return h, e


async def test_full_loop_offline(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["run"]["experiments_dir"] = str(tmp_path)
    cfg["sampling"]["group_size"] = 4
    h, e = _mock_clients()
    trainer = Trainer(cfg, h, e, run_id="offline-e2e")
    res = await trainer.step(n_tasks=2)
    assert res.step == 0 and res.extra["n_trajectories"] == 8
    assert math.isfinite(res.mean_reward)
    assert set(res.extra["components_mean"]) == {"correctness", "format", "tool_efficiency", "cost"}
    run_dir = tmp_path / "offline-e2e"
    assert (run_dir / "config" / "config.yaml").exists()
    assert (run_dir / "config" / "baseline_prompts.json").exists()
    assert (run_dir / "best_bundle.yaml").exists()
    lines = (run_dir / "traces" / "trajectories.jsonl").read_text().splitlines()
    assert len(lines) == 8
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 1
    res2 = await trainer.step(n_tasks=1)
    assert res2.step == 1
    await h.aclose(); await e.aclose()


async def test_cost_guard_aborts(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["run"]["experiments_dir"] = str(tmp_path)
    cfg["guards"]["max_tokens_per_step"] = 1
    h, e = _mock_clients()
    trainer = Trainer(cfg, h, e, run_id="guard")
    with pytest.raises(RuntimeError, match="cost guard"):
        await trainer.step(n_tasks=1)
    await h.aclose(); await e.aclose()

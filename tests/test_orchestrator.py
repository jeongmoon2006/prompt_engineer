"""Orchestrator: random policy shapes, and the full loop in-process (mock LLM) through the HTTP contracts."""

import json
import math
import random
from pathlib import Path

import httpx
import pytest
import yaml

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
    cfg["policy"]["kind"] = "random"
    cfg["eval"]["every"] = 0
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
    assert (run_dir / "config" / "env_config.json").exists() and (run_dir / "config" / "harness_config.json").exists()
    assert "weights" in json.loads((run_dir / "config" / "env_config.json").read_text())["reward"]
    e8 = res.extra
    assert e8["harness_calls_train"] == 8 and e8["harness_calls_total"] == e8["harness_calls_train"] + e8["harness_calls_eval"]
    assert e8["tokens_train"] == e8["tokens_step"] > 0 and e8["tokens_total"] == e8["tokens_train"] + e8["tokens_eval"]
    res2 = await trainer.step(n_tasks=1)
    assert res2.step == 1
    await h.aclose(); await e.aclose()


async def test_best_bundle_selected_on_slice_not_single_trajectory(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["run"]["experiments_dir"] = str(tmp_path)
    cfg["sampling"]["group_size"] = 3
    cfg["policy"]["kind"] = "random"
    cfg["eval"]["every"] = 0
    cfg["select"] = {"n_tasks": 4, "per_step": 2, "seed": 1}
    h, e = _mock_clients()
    trainer = Trainer(cfg, h, e, run_id="sel")
    res = await trainer.step(n_tasks=2)
    sel = res.extra["selection"]
    assert sel["rule"] == "selection_slice_mean" and sel["n_tasks"] == 4 and len(sel["candidates"]) == 2
    assert all(math.isfinite(c["select_reward"]) for c in sel["candidates"])
    assert res.extra["best_reward"] == max(c["select_reward"] for c in sel["candidates"])
    assert res.extra["harness_calls_eval"] == 2 * 4 and res.extra["harness_calls_train"] == 6
    meta = yaml.safe_load((tmp_path / "sel" / "best_bundle.yaml").read_text())
    assert meta["selected_by"].startswith("selection_slice_mean") and meta["best_train_reward"] == res.extra["best_train_reward"]
    # the selected bundle is one of the re-scored candidates, not necessarily the single-trajectory max
    assert meta["bundle_id"] in {c["bundle_id"] for c in sel["candidates"]}
    # legacy rule
    cfg["select"]["n_tasks"] = 0
    trainer2 = Trainer(cfg, h, e, run_id="sel0")
    res = await trainer2.step(n_tasks=1)
    assert res.extra["selection"]["rule"] == "single_trajectory" and res.extra["best_reward"] == res.extra["best_train_reward"]
    await h.aclose(); await e.aclose()


async def test_restart_with_same_run_id_continues_step_counter(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["run"]["experiments_dir"] = str(tmp_path)
    cfg["sampling"]["group_size"] = 2
    cfg["policy"]["kind"] = "random"
    cfg["eval"]["every"] = 1
    cfg["eval"]["n_tasks"] = 2
    cfg["select"]["n_tasks"] = 0
    h, e = _mock_clients()
    t1 = Trainer(cfg, h, e, run_id="restart")
    await t1.step(n_tasks=1)
    t2 = Trainer(cfg, h, e, run_id="restart")
    assert t2.step_idx == 1 and t2.harness_calls["train"] == 2
    res = await t2.step(n_tasks=1)
    assert res.step == 1
    kinds = [json.loads(l)["kind"] for l in (tmp_path / "restart" / "metrics.jsonl").read_text().splitlines()]
    assert kinds == ["heldout", "train", "heldout", "train", "heldout"]  # no duplicate pre-eval
    await h.aclose(); await e.aclose()


async def test_cost_guard_aborts(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["run"]["experiments_dir"] = str(tmp_path)
    cfg["guards"]["max_tokens_per_step"] = 1
    cfg["policy"]["kind"] = "random"
    cfg["eval"]["every"] = 0
    h, e = _mock_clients()
    trainer = Trainer(cfg, h, e, run_id="guard")
    from orchestrator.trainer import CostGuardExceeded
    with pytest.raises(CostGuardExceeded, match="cost guard"):
        await trainer.step(n_tasks=1)
    await h.aclose(); await e.aclose()


async def test_heldout_eval_and_best_bundle_dropin(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["run"]["experiments_dir"] = str(tmp_path)
    cfg["sampling"]["group_size"] = 2
    cfg["policy"]["kind"] = "random"
    cfg["eval"]["every"] = 1
    cfg["eval"]["n_tasks"] = 3
    h, e = _mock_clients()
    trainer = Trainer(cfg, h, e, run_id="ho")
    res = await trainer.step(n_tasks=1)
    ho = res.extra["heldout"]
    assert ho["n_tasks"] == 3 and math.isfinite(ho["b0"]) and math.isfinite(ho["best_bundle"]) and math.isfinite(ho["policy_sample"])
    kinds = [json.loads(l)["kind"] for l in (tmp_path / "ho" / "metrics.jsonl").read_text().splitlines()]
    assert kinds == ["heldout", "train", "heldout"]  # pre-eval, step, post-eval
    dropin = sorted(p.name for p in (tmp_path / "ho" / "best_bundle").glob("*.yaml"))
    assert dropin == ["planner.yaml", "system.yaml", "tool_policy.yaml"]
    await h.aclose(); await e.aclose()


async def test_gsm8k_domain_loop_offline(tmp_path: Path) -> None:
    """Domain switch exercises /prompts?domain, /tasks?domain and the calculator tool end to end (mock LLM)."""
    import env.server as env_srv
    if not env_srv.STORE.list(domain="gsm8k"):
        pytest.skip("gsm8k tasks not generated (python -m env.tasks.hf_gsm8k)")
    cfg = load_config()
    cfg["run"].update({"experiments_dir": str(tmp_path), "domain": "gsm8k"})
    cfg["sampling"]["group_size"] = 2
    cfg["policy"]["kind"] = "random"
    cfg["eval"]["every"] = 0
    h, e = _mock_clients()
    trainer = Trainer(cfg, h, e, run_id="g")
    res = await trainer.step(n_tasks=2)
    assert res.extra["n_trajectories"] == 4 and set(res.extra["components_mean"]) == {"correctness", "format", "tool_efficiency", "cost"}
    await h.aclose(); await e.aclose()

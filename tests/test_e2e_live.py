"""Live e2e (Phase 1 step 4): 1 task x G=4 -> real harness (Claude Haiku 5.5) -> rewards.

Skipped unless ANTHROPIC_API_KEY is set and RUN_LIVE=1. Costs ~4 short API calls.
"""

import math
import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()
pytestmark = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RUN_LIVE") == "1"),
    reason="set ANTHROPIC_API_KEY and RUN_LIVE=1 to run the live e2e",
)


async def test_one_task_g4_live(tmp_path: Path) -> None:
    import env.server as env_srv
    import harness.server as harness_srv
    from harness.harness import Harness
    from orchestrator.clients import EnvClient, HarnessClient
    from orchestrator.trainer import Trainer, load_config

    os.environ["HARNESS_PROVIDER"] = "anthropic"
    harness_srv._harness = Harness()
    h = HarnessClient("http://harness", transport=httpx.ASGITransport(app=harness_srv.app))
    e = EnvClient("http://env", transport=httpx.ASGITransport(app=env_srv.app))
    cfg = load_config()
    cfg["run"]["experiments_dir"] = str(tmp_path)
    cfg["sampling"]["group_size"] = 4
    cfg["policy"]["kind"] = "random"
    cfg["eval"]["every"] = 0
    trainer = Trainer(cfg, h, e, run_id="live-e2e")
    res = await trainer.step(n_tasks=1)
    assert res.extra["n_trajectories"] == 4
    assert math.isfinite(res.mean_reward)
    lines = (tmp_path / "live-e2e" / "traces" / "trajectories.jsonl").read_text().splitlines()
    assert len(lines) == 4
    import json
    recs = [json.loads(l) for l in lines]
    for r in recs:
        assert r["trace"]["error"] is None, r["trace"]["error"]
        assert r["trace"]["total_tokens"] > 0
        assert set(r["reward"]["components"]) == {"correctness", "format", "tool_efficiency", "cost"}
        assert all(math.isfinite(v) for v in r["reward"]["components"].values())
    print("\nSAMPLE REWARD:", json.dumps(recs[0]["reward"], indent=2))
    print("FINAL OUTPUT:", recs[0]["trace"]["final_output"])
    await h.aclose(); await e.aclose()

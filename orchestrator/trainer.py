"""Outer loop: sample bundles -> harness.run_batch -> env.evaluate -> (update) -> log.

Phase 1: the policy is the random-perturbation placeholder and there is no update.
Phase 2 swaps in the GRPO meta-policy; Phase 4 puts both behind the Optimizer protocol.
"""

from __future__ import annotations

import asyncio
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from orchestrator.clients import EnvClient, HarnessClient
from orchestrator.experiment import ExperimentLogger
from orchestrator.policy_random import RandomPerturbationPolicy
from shared.schemas import PromptBundle, Reward, StepResult, Task, Trace

ROOT = Path(__file__).resolve().parent
_ENV_REF = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def load_config(path: Path = ROOT / "config.yaml") -> dict[str, Any]:
    text = _ENV_REF.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), path.read_text())
    return yaml.safe_load(text)


class Trainer:
    def __init__(self, cfg: dict[str, Any], harness: HarnessClient, env: EnvClient, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.harness = harness
        self.env = env
        self.step_idx = 0
        self.tokens_spent = 0
        self.policy: RandomPerturbationPolicy | None = None
        self.best: tuple[float, PromptBundle] | None = None
        self.logger = ExperimentLogger(
            Path(cfg["run"]["experiments_dir"]), run_id, config_files=[ROOT / "config.yaml"],
        )
        self.logger.write_config("resolved_config.json", cfg)

    async def ensure_policy(self) -> None:
        if self.policy is None:
            baseline = await self.harness.baseline_prompts()
            self.policy = RandomPerturbationPolicy(baseline, seed=int(self.cfg["run"]["seed"]))
            self.logger.write_config("baseline_prompts.json", baseline)

    async def rollout(self, task: Task, bundles: list[PromptBundle]) -> tuple[list[Trace], list[Reward]]:
        traces = await self.harness.run_batch(task, bundles)
        rewards = await self.env.evaluate(task, traces)
        return traces, rewards

    async def step(self, n_tasks: int, split: str = "train") -> StepResult:
        await self.ensure_policy()
        assert self.policy is not None
        g = int(self.cfg["sampling"]["group_size"])
        seed = int(self.cfg["run"]["seed"]) + self.step_idx
        tasks = await self.env.tasks(split=split, n=n_tasks, seed=seed)
        t0 = time.perf_counter()
        groups = [self.policy.sample(g) for _ in tasks]
        results = await asyncio.gather(*(self.rollout(t, b) for t, b in zip(tasks, groups)))

        all_rewards: list[float] = []
        comp_sums: dict[str, float] = {}
        n_err = 0
        step_tokens = 0
        for task, bundles, (traces, rewards) in zip(tasks, groups, results):
            for b, tr, rw in zip(bundles, traces, rewards):
                self.logger.log_trajectory(self.step_idx, task, b, tr, rw)
                all_rewards.append(rw.total)
                step_tokens += tr.total_tokens
                n_err += tr.error is not None
                for k, v in rw.components.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + v
                if self.best is None or rw.total > self.best[0]:
                    self.best = (rw.total, b)
        self.tokens_spent += step_tokens
        cap = int(self.cfg["guards"]["max_tokens_per_step"])
        if step_tokens > cap:
            raise RuntimeError(f"cost guard: step used {step_tokens} tokens > max_tokens_per_step={cap}; aborting run")

        n = max(len(all_rewards), 1)
        res = StepResult(
            step=self.step_idx,
            mean_reward=statistics.fmean(all_rewards) if all_rewards else 0.0,
            std_reward=statistics.pstdev(all_rewards) if len(all_rewards) > 1 else 0.0,
            extra={
                "n_tasks": len(tasks), "group_size": g, "n_trajectories": len(all_rewards),
                "components_mean": {k: v / n for k, v in comp_sums.items()},
                "error_rate": n_err / n, "tokens_step": step_tokens, "tokens_total": self.tokens_spent,
                "best_reward": self.best[0] if self.best else None, "wall_s": round(time.perf_counter() - t0, 2),
                "policy": "random_perturbation",
            },
        )
        self.logger.log_metrics(res.model_dump())
        if self.best:
            (self.logger.dir / "best_bundle.yaml").write_text(
                yaml.safe_dump({"reward": self.best[0], "bundle_id": self.best[1].bundle_id, "slots": self.best[1].slots}, sort_keys=False)
            )
        self.step_idx += 1
        return res

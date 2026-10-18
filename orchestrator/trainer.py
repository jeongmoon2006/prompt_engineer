"""Outer loop: propose bundles -> harness.run_batch -> env.evaluate -> update -> log.

Two interchangeable proposers (Phase 4 formalizes this as the Optimizer protocol):
  - "grpo":   MetaPolicy (LoRA LLM) + GRPOUpdater
  - "random": RandomPerturbationPolicy (Phase 1 placeholder / baseline B1, keep-best, no learning)

Held-out evaluation every `eval.every` steps: (a) baseline hand-written prompts B0 (constant),
(b) the best bundle found so far, (c) a fresh sample from the current policy.
"""

from __future__ import annotations

import argparse
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


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


class Trainer:
    def __init__(self, cfg: dict[str, Any], harness: HarnessClient, env: EnvClient, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.harness = harness
        self.env = env
        self.domain: str = cfg["run"].get("domain", "toy")
        self.kind: str = cfg["policy"].get("kind", "grpo")
        self.step_idx = 0
        self.tokens_spent = 0
        self.harness_calls = 0
        self.baseline: dict[str, str] | None = None
        self.policy: Any = None
        self.updater: Any = None
        self.best: tuple[float, PromptBundle] | None = None   # best single-trajectory reward so far
        self.b0_heldout: float | None = None
        self.logger = ExperimentLogger(Path(cfg["run"]["experiments_dir"]), run_id, config_files=[ROOT / "config.yaml"])
        self.logger.write_config("resolved_config.json", cfg)

    # ------------------------------------------------------------------ setup
    async def ensure_policy(self) -> None:
        if self.policy is not None:
            return
        self.baseline = await self.harness.baseline_prompts(self.domain)
        self.logger.write_config("baseline_prompts.json", self.baseline)
        if self.kind == "random":
            self.policy = RandomPerturbationPolicy(self.baseline, seed=int(self.cfg["run"]["seed"]))
        elif self.kind == "grpo":
            from orchestrator.grpo import GRPOUpdater
            from orchestrator.policy import MetaPolicy
            import torch
            torch.manual_seed(int(self.cfg["run"]["seed"]))
            info = await self.env.domain_info(self.domain)
            self.policy = MetaPolicy(self.cfg["policy"], self.baseline, task_info=info)
            self.updater = GRPOUpdater(self.policy, self.cfg["grpo"])
            self.logger.write_config("meta_prompt.txt", {"meta_prompt": self.policy.meta_prompt,
                                                          "n_trainable": self.policy.n_trainable(),
                                                          "device": str(self.policy.device)})
        else:
            raise ValueError(f"unknown policy.kind {self.kind!r}")

    # ------------------------------------------------------------------ proposing
    def propose(self, n_groups: int, g: int) -> list[list[dict[str, Any]]]:
        """Returns n_groups groups of g samples: {bundle|None, completion_ids|None, reject_reason|None}."""
        if self.kind == "random":
            return [[{"bundle": b, "completion_ids": None, "reject_reason": None} for b in self.policy.sample(g)]
                    for _ in range(n_groups)]
        # The meta-prompt is task-independent, so all n_groups*g completions come from one batched
        # generate call (chunked to bound KV-cache memory), then are split into groups.
        chunk = int(self.cfg["policy"].get("sample_chunk", 32))
        flat: list[dict[str, Any]] = []
        while len(flat) < n_groups * g:
            flat.extend(self.policy.sample(min(chunk, n_groups * g - len(flat))))
        return [flat[i * g:(i + 1) * g] for i in range(n_groups)]

    # ------------------------------------------------------------------ rollouts
    async def rollout(self, task: Task, bundles: list[PromptBundle]) -> tuple[list[Trace], list[Reward]]:
        if not bundles:
            return [], []
        traces = await self.harness.run_batch(task, bundles)
        rewards = await self.env.evaluate(task, traces)
        self.harness_calls += len(bundles)
        return traces, rewards

    async def rollout_group(self, task: Task, samples: list[dict[str, Any]]) -> list[tuple[Trace | None, Reward]]:
        valid = [s["bundle"] for s in samples if s["bundle"] is not None]
        traces, rewards = await self.rollout(task, valid)
        it = iter(zip(traces, rewards))
        out: list[tuple[Trace | None, Reward]] = []
        rej = float(self.cfg["guards"]["rejected_reward"])
        for s in samples:
            if s["bundle"] is None:
                out.append((None, Reward(task_id=task.task_id, bundle_id="rejected", total=rej, components={},
                                         feedback=f"rejected: {s['reject_reason']}")))
            else:
                out.append(next(it))
        return out

    # ------------------------------------------------------------------ one iteration
    async def step(self, n_tasks: int) -> StepResult:
        await self.ensure_policy()
        g = int(self.cfg["sampling"]["group_size"])
        seed = int(self.cfg["run"]["seed"]) + self.step_idx
        t0 = time.perf_counter()
        if self.step_idx == 0 and int(self.cfg["eval"]["every"]) > 0:
            await self.evaluate_heldout(tag="pre")
        tasks = await self.env.tasks(domain=self.domain, split="train", n=n_tasks, seed=seed)
        groups = self.propose(len(tasks), g)
        t_gen = time.perf_counter() - t0
        results = await asyncio.gather(*(self.rollout_group(t, s) for t, s in zip(tasks, groups)))
        t_roll = time.perf_counter() - t0 - t_gen

        all_r: list[float] = []
        valid_r: list[float] = []
        comp_sums: dict[str, float] = {}
        n_err = n_rej = 0
        step_tokens = 0
        completions: list[Any] = []
        advantages: list[float] = []
        reject_reasons: dict[str, int] = {}
        from_torch = None
        for task, samples, res in zip(tasks, groups, results):
            group_r = [rw.total for _, rw in res]
            if self.kind == "grpo":
                import torch
                from orchestrator.grpo import group_advantages
                adv = group_advantages(torch.tensor(group_r), float(self.cfg["grpo"]["adv_eps"])).tolist()
            else:
                adv = [0.0] * len(group_r)
            for s, (tr, rw), a in zip(samples, res, adv):
                all_r.append(rw.total)
                if s["bundle"] is None:
                    n_rej += 1
                    reject_reasons[s["reject_reason"]] = reject_reasons.get(s["reject_reason"], 0) + 1
                    self.logger.log_trajectory(self.step_idx, task, PromptBundle(bundle_id="rejected", slots={"raw": s.get("text", "")}),
                                               Trace(task_id=task.task_id, bundle_id="rejected"), rw)
                else:
                    assert tr is not None
                    valid_r.append(rw.total)
                    step_tokens += tr.total_tokens
                    n_err += tr.error is not None
                    for k, v in rw.components.items():
                        comp_sums[k] = comp_sums.get(k, 0.0) + v
                    self.logger.log_trajectory(self.step_idx, task, s["bundle"], tr, rw)
                    if self.best is None or rw.total > self.best[0]:
                        self.best = (rw.total, s["bundle"])
                if s["completion_ids"] is not None:
                    completions.append(s["completion_ids"])
                    advantages.append(a)
        self.tokens_spent += step_tokens
        cap = int(self.cfg["guards"]["max_tokens_per_step"])
        if step_tokens > cap:
            raise RuntimeError(f"cost guard: step used {step_tokens} tokens > max_tokens_per_step={cap}; aborting run")

        t1 = time.perf_counter()
        upd: dict[str, Any] = {}
        if self.kind == "grpo":
            upd = self.updater.update(completions, advantages)
        t_upd = time.perf_counter() - t1

        nv = max(len(valid_r), 1)
        res_obj = StepResult(
            step=self.step_idx,
            mean_reward=_mean(all_r), std_reward=statistics.pstdev(all_r) if len(all_r) > 1 else 0.0,
            kl=float(upd.get("kl", 0.0)), loss=float(upd.get("loss", 0.0)),
            extra={
                "policy": self.kind, "domain": self.domain, "n_tasks": len(tasks), "group_size": g,
                "n_trajectories": len(all_r), "mean_reward_valid": _mean(valid_r),
                "components_mean": {k: v / nv for k, v in comp_sums.items()},
                "rejection_rate": n_rej / max(len(all_r), 1), "reject_reasons": reject_reasons,
                "error_rate": n_err / nv, "tokens_step": step_tokens, "tokens_total": self.tokens_spent,
                "harness_calls_total": self.harness_calls,
                "best_reward": self.best[0] if self.best else None,
                "update": upd, "timing_s": {"generate": round(t_gen, 1), "rollout": round(t_roll, 1), "update": round(t_upd, 1)},
            },
        )
        self.logger.log_metrics({"kind": "train", **res_obj.model_dump()})
        self._write_best()
        self.step_idx += 1
        ev = int(self.cfg["eval"]["every"])
        if ev > 0 and self.step_idx % ev == 0:
            res_obj.extra["heldout"] = await self.evaluate_heldout(tag="post")
        ck = int(self.cfg["run"]["checkpoint_every"])
        if self.kind == "grpo" and ck > 0 and self.step_idx % ck == 0:
            self.policy.save_adapter(str(self.logger.dir / "checkpoints" / f"step{self.step_idx:04d}"))
        return res_obj

    # ------------------------------------------------------------------ held-out evaluation
    async def evaluate_heldout(self, tag: str) -> dict[str, Any]:
        ecfg = self.cfg["eval"]
        tasks = await self.env.tasks(domain=self.domain, split="heldout", n=int(ecfg["n_tasks"]), seed=int(ecfg["seed"]))
        out: dict[str, Any] = {"step": self.step_idx, "tag": tag, "n_tasks": len(tasks)}
        assert self.baseline is not None
        b0 = PromptBundle(bundle_id="baseline", slots=self.baseline)
        if self.b0_heldout is None:
            self.b0_heldout, b0_comp = await self._eval_bundle(tasks, [b0] * len(tasks))
            out["b0_components"] = b0_comp
        out["b0"] = self.b0_heldout
        if self.best is not None:
            out["best_bundle"], out["best_components"] = await self._eval_bundle(tasks, [self.best[1]] * len(tasks))
        # one fresh sample from the current policy per task
        if self.kind == "grpo":
            samples = self.policy.sample(len(tasks))
        else:
            samples = [{"bundle": b, "reject_reason": None} for b in self.policy.sample(len(tasks))]
        bundles = [s["bundle"] for s in samples]
        n_rej = sum(b is None for b in bundles)
        rej = float(self.cfg["guards"]["rejected_reward"])
        valid_pairs = [(t, b) for t, b in zip(tasks, bundles) if b is not None]
        rewards: list[float] = [rej] * n_rej
        comps: dict[str, float] = {}
        if valid_pairs:
            per = await asyncio.gather(*(self.rollout(t, [b]) for t, b in valid_pairs))
            for _, rws in per:
                rewards.append(rws[0].total)
                for k, v in rws[0].components.items():
                    comps[k] = comps.get(k, 0.0) + v / len(valid_pairs)
        out["policy_sample"] = _mean(rewards)
        out["policy_sample_valid"] = _mean(rewards[n_rej:])
        out["policy_components"] = comps
        out["policy_rejection_rate"] = n_rej / max(len(tasks), 1)
        out["tokens_total"] = self.tokens_spent
        out["harness_calls_total"] = self.harness_calls
        self.logger.log_metrics({"kind": "heldout", **out})
        return out

    async def _eval_bundle(self, tasks: list[Task], bundles: list[PromptBundle]) -> tuple[float, dict[str, float]]:
        per = await asyncio.gather(*(self.rollout(t, [b]) for t, b in zip(tasks, bundles)))
        totals = [rws[0].total for _, rws in per]
        comps: dict[str, float] = {}
        for _, rws in per:
            for k, v in rws[0].components.items():
                comps[k] = comps.get(k, 0.0) + v / len(per)
        return _mean(totals), comps

    def _write_best(self) -> None:
        if not self.best:
            return
        reward, b = self.best
        (self.logger.dir / "best_bundle.yaml").write_text(
            yaml.safe_dump({"reward": reward, "bundle_id": b.bundle_id, "slots": b.slots}, sort_keys=False, allow_unicode=True))
        d = self.logger.dir / "best_bundle"   # drop-in replacement for harness/prompts/<domain>/
        d.mkdir(exist_ok=True)
        for slot, text in b.slots.items():
            (d / f"{slot}.yaml").write_text(yaml.safe_dump({"slot": slot, "text": text}, sort_keys=False, allow_unicode=True))


# ---------------------------------------------------------------------- CLI: make train
async def _main(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.policy:
        cfg["policy"]["kind"] = args.policy
    if args.domain:
        cfg["run"]["domain"] = args.domain
    if args.lr is not None:
        cfg["grpo"]["lr"] = args.lr
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    n_steps = args.steps or int(cfg["run"]["n_steps"])
    harness = HarnessClient(cfg["services"]["harness_url"])
    env = EnvClient(cfg["services"]["env_url"])
    if not (await harness.health() and await env.health()):
        raise SystemExit("harness/env not reachable; start them first (make up or scripts/serve_local.sh)")
    trainer = Trainer(cfg, harness, env, run_id=args.run_id)
    print(f"run {trainer.logger.run_id}: policy={cfg['policy']['kind']} domain={cfg['run']['domain']} steps={n_steps}", flush=True)
    for _ in range(n_steps):
        r = await trainer.step(int(cfg["sampling"]["tasks_per_step"]))
        e = r.extra
        ho = e.get("heldout", {})
        print(f"step {r.step:3d} | R={r.mean_reward:.4f} (valid {e['mean_reward_valid']:.4f}) rej={e['rejection_rate']:.2f} "
              f"| kl={r.kl:.5f} loss={r.loss:.4f} gn={e['update'].get('grad_norm', 0):.3f} vram={e['update'].get('peak_vram_gb', 0)}GB "
              f"| best={e['best_reward']} | heldout b0={ho.get('b0')} best={ho.get('best_bundle')} pol={ho.get('policy_sample')} "
              f"| t={e['timing_s']}", flush=True)
    trainer.logger.close()
    await harness.aclose()
    await env.aclose()


def main() -> None:
    p = argparse.ArgumentParser(description="Run the prompt-optimization training loop.")
    p.add_argument("--run-id", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--policy", choices=["grpo", "random"], default=None)
    p.add_argument("--domain", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()

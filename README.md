# rl-prompt-opt

RL-driven (GRPO) prompt optimization for a frozen agent harness. Three containers:
`orchestrator` (meta-policy + GRPO, GPU), `harness` (agent loop, remote model API),
`env` (deterministic verifiers → scalar reward). See `WORK_INSTRUCTIONS.md` for the plan.

## Run

```bash
cp .env.example .env            # fill in an API key
make smoke                      # docker: build, start, curl /health x3
make smoke GPU=1                # same, orchestrator gets the host GPU
make smoke-local                # no docker: uvicorn from .venv (needs `uv`)
make test                       # pytest (offline; the harness uses HARNESS_PROVIDER=mock in tests)
make e2e                        # live: 1 task x G=4 through Claude Haiku 5.5 -> rewards
make tasks                      # regenerate env/tasks/toy_v2.jsonl (seeded)
```

Training (Phase 2+) — the orchestrator needs harness + env reachable (`make up`, or `make serve-local` without docker):

```bash
make train RUN=toy-grpo STEPS=20                 # GRPO meta-policy (Qwen2.5-0.5B + LoRA), toy domain
make train RUN=toy-random POLICY=random          # B1: random-perturbation keep-best, same budget
make train RUN=gsm8k-grpo DOMAIN=gsm8k           # second domain (python -m env.tasks.hf_gsm8k first)
make analyze RUN=toy-grpo                        # reward / KL / rejection curves -> experiments/analysis/
make analyze COMPARE='{"toy": {"B1": "toy-random", "B2": "toy-grpo"}}'   # B0/B1/B2 table + figure
```

Hardware notes (GTX 1660 Super, 6 GB, no bf16): fp32 LoRA master weights + fp16 autocast with GradScaler,
gradient checkpointing, micro-batched accumulation with an automatic halving above 5.5 GB peak VRAM.
Smoke tests run on CPU; set `CUDA_VISIBLE_DEVICES=` to force CPU.

Then drive one iteration of the loop: `curl -X POST localhost:8000/step -H 'content-type: application/json' -d '{"n_tasks": 2}'`.
Every run writes `experiments/<run_id>/` with frozen configs, `metrics.jsonl`, `best_bundle.yaml`
and `traces/trajectories.jsonl` (one line per task x bundle x trace x reward).

Ports: orchestrator 8000, harness 8001, env 8002.

## Layout
- `shared/schemas.py` — the only module every container imports (pydantic contracts).
- `orchestrator/`, `harness/`, `env/` — one service each; never import across them.
- `experiments/<run_id>/` — per-run configs, metrics, traces (traces gitignored).

## Toy domain (Phase 1)
Structured extraction from synthetic lab-notebook entries (`env/tasks/generate.py`). The agent
may call `lookup(field)` at most 4 times and must emit `{sample_id, temperature_K, concentration_mM, method}`.
Prompt slots: `system` (system prompt), `planner` (user-turn instruction), `tool_policy` (the tool's description).
Reward = 0.4 correctness + 0.3 format + 0.1 tool_efficiency + 0.2 cost (weights in `env/config.yaml`;
rebalanced from the plan's 0.6/0.15/0.15/0.10 because Haiku 5.5 saturates correctness on this domain).
Harness model: `claude-haiku-5-5` via the Anthropic SDK (`HARNESS_MODEL` in `.env`); `HARNESS_PROVIDER=mock` runs offline.

## Domains
A domain = a task file served by the environment (`env/config.yaml: domains`), a correctness verifier
(`env/reward.py` dispatch), baseline prompts under `harness/prompts/<domain>/` and a tool executor in
`harness/tools.py`. Shipped: `toy` (lab-notebook extraction, `lookup` tool) and `gsm8k` (Hugging Face GSM8K,
`calculator` tool). The orchestrator never changes; it reads `/domain_info` to build the meta-prompt.

Adding an optimizer: see Phase 4 (`Optimizer` protocol).

## Results (Phase 2/3)
See `experiments/REPORT_phase2_3.md` for the GRPO training curves, the B0/B1/B2 comparison on both
domains with token cost, the Checkpoint 2 diagnosis, and the list of deviations from `WORK_INSTRUCTIONS.md`.
Figures and the comparison table are in `experiments/analysis/`.

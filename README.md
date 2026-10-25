# rl-prompt-opt

Prompt optimization with reinforcement learning (GRPO) for an agent whose model stays frozen.

Three services, talking only over HTTP:

| Service | Port | Role | Trained? |
|---|---|---|---|
| `orchestrator/` | 8000 | Meta-policy (Qwen2.5-0.5B-Instruct + LoRA) writes prompt bundles; GRPO updates it | yes |
| `harness/` | 8001 | Agent loop on Claude Haiku 5.5 with one tool; prompt slots injected per run | no |
| `env/` | 8002 | Deterministic verifiers turn a trace into a scalar reward | no |

A prompt bundle has three slots: `system`, `planner` (user-turn instruction), `tool_policy` (tool description).
Reward = 0.4 correctness + 0.3 format + 0.1 tool efficiency + 0.2 cost (weights in `env/config.yaml`).
`shared/schemas.py` holds the pydantic contracts; it is the only module every service imports.

## Setup

```bash
cp .env.example .env        # set ANTHROPIC_API_KEY
make test                   # offline tests (mock harness provider); creates .venv with uv
make smoke-local            # start all three services without docker and check /health
```

Docker: `make smoke` (add `GPU=1` for the orchestrator GPU). The compose files are written but untested
on this machine. Training on the host needs the ML extra: `uv sync --extra dev --extra ml`.

## Train and analyze

```bash
make serve-local                                   # harness + env in the background
make train RUN=toy-grpo STEPS=20                   # B2: GRPO meta-policy, toy domain
make train RUN=toy-random POLICY=random            # B1: random perturbation, keep-best
make train RUN=gsm8k-grpo DOMAIN=gsm8k             # second domain (run python -m env.tasks.hf_gsm8k once)
make train RUN=toy-grpo-b RESUME=experiments/toy-grpo/checkpoints/step0020
make analyze RUN=toy-grpo                          # reward / KL / rejection curves
make analyze RUN=toy-grpo COMPARE='{"toy": {"B1": "toy-random", "B2": "toy-grpo"}}'   # + B0/B1/B2 table and figure
make stop-local
```

One step = 8 tasks x 8 sampled bundles = 64 harness runs. Each run writes `experiments/<run_id>/`:

- `config/` frozen orchestrator config, baseline prompts, harness and env `/config` responses
- `metrics.jsonl` one line per step plus held-out evaluations; harness calls and tokens are split into `train` and `eval`
- `best_bundle.yaml` and `best_bundle/*.yaml` the selected prompts, drop-in for `harness/prompts/<domain>/`
- `traces/` and `checkpoints/` (gitignored)

Best bundle selection: each sampled bundle runs on one task, so the top candidate of every step is re-scored on a
fixed 16-task slice of the train split and the best slice mean wins (`select` in `orchestrator/config.yaml`).

Hardware: tuned for a 6 GB GTX 1660 Super (fp32 LoRA weights, fp16 autocast, gradient checkpointing,
micro-batch auto-halving). Set `CUDA_VISIBLE_DEVICES=` to force CPU.

## Domains

| Domain | Tasks | Tool | Correctness |
|---|---|---|---|
| `toy` | 200 synthetic lab-notebook entries (`env/tasks/generate.py`), 160 train / 40 held-out | `lookup(field)`, max 4 | per-field match, partial credit on numbers |
| `gsm8k` | 300 train / 100 held-out from Hugging Face `openai/gsm8k` | `calculator(expression)`, max 6 | exact numeric match |

A domain is a task file plus a block in `env/config.yaml` (fields, numeric fields, verifier), baseline prompts under
`harness/prompts/<domain>/`, and a tool executor in `harness/tools.py`. The orchestrator reads `/domain_info` and
never changes per domain.

## Results

See `experiments/REPORT_phase2_3.md`. Short version: on both domains Haiku solves the task almost regardless of
prompt wording, so neither GRPO nor random search beat the baseline prompt beyond held-out noise, and GRPO mostly
learned to emit well-formed, short prompts. Figures and the comparison table are in `experiments/analysis/`.

## Status

Phases 0 to 3 of `WORK_INSTRUCTIONS.md` are done. Not yet built: the `Optimizer` protocol (Phase 4; the trainer
branches on `policy.kind`) and the conditioning study (Phase 5). The baseline prompts were written by Claude Code,
not hand-tuned by a person.

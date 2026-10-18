# Work Instructions — RL-Driven Prompt Optimization for a Frozen Agent Harness

**Audience:** Claude Code. Read this file fully before touching the repo. Follow the phases in order; do not skip ahead. Stop and report at every checkpoint marked `⏸ CHECKPOINT` and wait for the user.

## 0. Goal and non-goals

**Goal.** Build a three-container system that optimizes the *prompts* of an agentic harness using reinforcement learning (GRPO) while the harness's LLM weights stay frozen. The prompt generator is a small trainable meta-policy; the harness calls a remote model API; a separate environment container scores harness outputs and returns a scalar reward.

**Why this design (keep it, don't "improve" it).** The user is deliberately following a mentor's plan first. Later, DSPy/GEPA will be run as *alternative optimizers* against the same environment, so the architecture must make the optimizer swappable. Do not substitute DSPy, GEPA, TextGrad, MCTS, or evolutionary search for the GRPO meta-policy in this phase. If you think something is wrong with the plan, say so in a note; do not silently change it.

**Non-goals.**
- No fine-tuning of the harness model.
- No gradient-free optimization in the main loop (bandits/MCTS/evolution are out of scope for now).
- No production hardening, auth, or cloud deployment. Local `docker compose` only.

## 1. Architecture (fixed)

```
┌──────────────────────── rl-orchestrator ─────────────────────────┐
│ meta-policy π_φ (small open LLM + LoRA)  ◄── GRPO loss + KL(π_φ‖π_ref) │
│ samples G prompt variants per task; receives rewards; updates φ  │
└──────────────┬────────────────────────────────────▲──────────────┘
      POST /run_batch {task_id, prompts[G]}         │ rewards[G] + traces
               ▼                                    │
┌───── agent-harness ──────┐   traces[G]   ┌───── staging-env ─────────┐
│ frozen model via API     ├──────────────►│ sandbox + verifiers       │
│ dynamic prompt loader    │  POST /evaluate│ composite scalar reward   │
│ trace logger             │               │ (dense, not pass/fail)    │
└──────────────────────────┘               └───────────────────────────┘
```

- **rl-orchestrator** owns the policy, the GRPO update, and the training loop. Only container that needs a GPU (see Phase 2 §6 for the exact hardware constraints).
- **agent-harness** runs the agent loop with a candidate prompt injected. Zero GPUs. All model calls go to a remote API.
- **staging-env** turns an execution trace into a reward. Zero GPUs. Must be deterministic given the same trace.

The three containers communicate only through the HTTP interfaces in §3. Never import code across container boundaries.

## 2. Repository layout

```
rl-prompt-opt/
├── WORK_INSTRUCTIONS.md        # this file
├── docker-compose.yml
├── Makefile                    # make up / down / smoke / train / analyze
├── shared/                     # pydantic schemas only — the one thing all containers import
│   └── schemas.py              # Task, PromptBundle, Trace, Reward, StepResult
├── orchestrator/
│   ├── Dockerfile
│   ├── policy.py               # π_φ: load base model + LoRA, sample G prompts, log-probs
│   ├── grpo.py                 # advantage normalization, clipped surrogate, KL penalty
│   ├── trainer.py              # outer loop: sample → run_batch → evaluate → update
│   ├── conditioning.py         # Fisher / Hessian diagnostics (Phase 5)
│   ├── server.py               # POST /step, GET /status
│   └── config.yaml
├── harness/
│   ├── Dockerfile
│   ├── prompts/                # YAML registry: one file per prompt slot p^(1)..p^(k)
│   ├── harness.py              # agent loop with prompt-slot injection
│   ├── tracer.py               # structured trace capture
│   ├── llm_client.py           # async remote-API client with retries + token caps
│   └── server.py               # POST /run_batch
├── env/
│   ├── Dockerfile
│   ├── tasks/                  # task definitions + ground truth
│   ├── verifiers/              # one module per reward component
│   ├── reward.py               # composite reward, weights from config
│   └── server.py               # POST /evaluate
├── experiments/                # run configs + results (gitignored artifacts)
└── tests/                      # pytest; each container has unit tests, plus one e2e
```

## 3. Interfaces (define these first, in `shared/schemas.py`)

```python
class Task(BaseModel):
    task_id: str
    description: str            # what the agent is asked to do
    context: dict               # anything task-specific the harness needs

class PromptBundle(BaseModel):
    bundle_id: str
    slots: dict[str, str]       # slot_name -> prompt text, e.g. {"planner": "...", "tool_select": "..."}

class Trace(BaseModel):
    task_id: str
    bundle_id: str
    steps: list[dict]           # [{role, content, tool_call?, tool_result?, tokens_in, tokens_out, latency_ms}]
    final_output: str
    total_tokens: int
    total_latency_ms: float
    error: str | None = None

class Reward(BaseModel):
    task_id: str
    bundle_id: str
    total: float
    components: dict[str, float]   # e.g. {"correctness": 0.8, "format": 1.0, "cost": -0.12}
    feedback: str | None = None    # natural-language failure summary (for later GEPA use; unused by GRPO)
```

HTTP contracts:

| Container | Endpoint | Request | Response |
|---|---|---|---|
| agent-harness | `POST /run_batch` | `{task: Task, bundles: list[PromptBundle]}` | `{traces: list[Trace]}` |
| staging-env | `POST /evaluate` | `{task: Task, traces: list[Trace]}` | `{rewards: list[Reward]}` |
| rl-orchestrator | `POST /step` | `{n_tasks: int}` | `{step: int, mean_reward, kl, loss, ...}` |
| all | `GET /health` | — | `{"ok": true}` |

Freeze these schemas at the end of Phase 1. Later phases may *add* optional fields but must not rename or remove any.

## 4. Phases

### Phase 0 — Scaffold (no ML yet)
1. Create the layout above, `docker-compose.yml` with three services on one network, a `Makefile`, and `pyproject.toml` per container (FastAPI + httpx + pydantic; orchestrator additionally torch + transformers + peft).
2. Implement `/health` on all three and a `make smoke` target that brings the stack up and curls all three.
3. Read secrets (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / etc.) from `.env`; never hard-code. Add `.env.example`.

`⏸ CHECKPOINT 0` — show `make smoke` passing.

### Phase 1 — Toy domain, end-to-end with a *random* prompt sampler
The real domain (MD-simulation agent vs. knowledge-graph agent) is intentionally undecided. Build the loop against a **toy domain** so the RL machinery can be debugged independently of the domain. Keep the domain fully behind the `Task` / `verifiers` abstraction so it can be swapped later.

Toy domain spec (use this exactly unless the user says otherwise):
- **Task family:** structured extraction. Input = a short synthetic "lab notebook" paragraph (generated from templates with known ground truth); the agent must call a `lookup(field)` tool at most N times and emit a JSON record with fields `{sample_id, temperature_K, concentration_mM, method}`.
- **Prompt slots:** `system`, `planner`, `tool_policy` (k = 3).
- **Reward components** (all in [0,1] or small negatives, continuous where possible):
  - `correctness`: fraction of fields exactly correct, with partial credit for numeric fields via `exp(-|Δ|/scale)`.
  - `format`: 1 if valid JSON matching schema, 0.5 if parseable but wrong keys, 0 otherwise.
  - `tool_efficiency`: `1 - (tool_calls / N_max)` clipped to [0,1].
  - `cost`: `-λ · total_tokens / token_budget`.
  - `total = w·components` with weights in `env/config.yaml` (start: 0.6, 0.15, 0.15, 0.10).
- Generate ≥200 tasks with a fixed seed; split 160 train / 40 held-out. Store in `env/tasks/`.

Steps:
1. `harness.py`: async agent loop; prompt slots injected from the `PromptBundle`; cap tokens per trajectory and max turns; every API call retried with backoff; `Trace` populated faithfully.
2. `env/verifiers/*.py` + `reward.py`; unit-test each verifier on hand-written traces.
3. In the orchestrator, a **placeholder policy** that returns G random perturbations of the baseline prompts (sentence shuffling / synonym swaps). This is only to exercise the loop.
4. One e2e test: 1 task × G=4 → traces → rewards, assert shapes and that rewards are finite.
5. Log every run to `experiments/<run_id>/` as JSONL (task, bundle, trace, reward).

`⏸ CHECKPOINT 1` — show the e2e test passing and a sample `Reward` with all components populated.

### Phase 2 — Meta-policy π_φ and GRPO
1. `policy.py`: load a small open model (default `Qwen/Qwen2.5-0.5B-Instruct`; configurable) with a LoRA adapter (peft). π_ref is the same model with the adapter disabled (`with model.disable_adapter():`) — never load a second copy. Input = a fixed meta-prompt describing the harness, the slot names, the baseline prompts, and the task family. Output = the k slot prompts in a delimited format; parse into `PromptBundle`. Return per-token log-probs for both π_φ and π_ref.
2. Sampling: for each task, sample **G = 8** bundles at temperature 1.0 (configurable).
3. `grpo.py`:
   - `A_i = (R_i − mean(R)) / (std(R) + ε)` within each group.
   - Clipped surrogate with ratio `π_φ/π_φ_old`, clip ε = 0.2.
   - KL penalty `β · KL(π_φ ‖ π_ref)` estimated per token, β = 0.04 to start.
   - One optimizer step per batch of tasks (batch = 8 tasks → 64 trajectories). AdamW, lr 1e-5, grad-norm clip 1.0.
4. `trainer.py` outer loop; `POST /step` runs one iteration. Checkpoint LoRA weights every N steps. Log: mean/std reward, per-component rewards, KL, loss, ratio stats, fraction clipped, and the **best bundle so far** (also written as plain YAML so it can be dropped back into `harness/prompts/`).
5. Degeneracy guards: reject (reward = −1) any bundle that is empty, exceeds a length cap, is non-UTF8, or contains the delimiter malformed. Report rejection rate.
6. Hardware — the user's GPU is a **GTX 1660 Super (6 GB VRAM, Turing, compute capability 7.5, no bf16)**. Build for it:
   - fp16 autocast with `torch.cuda.amp.GradScaler`; keep LoRA parameters in fp32 (master weights). Never use bf16.
   - `gradient_checkpointing_enable()` on the base model; micro-batch of 2–4 trajectories with gradient accumulation to reach the 64-trajectory batch.
   - Compute π_φ and π_ref log-probs in separate forward passes under `torch.no_grad()` for the reference; free the KV cache between generation and training.
   - Log peak VRAM (`torch.cuda.max_memory_allocated`) every step; if it exceeds 5.5 GB, halve the micro-batch automatically.
   - 1.5B is allowed only via 4-bit QLoRA (bitsandbytes `load_in_4bit`, works on cc 7.5) and must be opt-in via config. Default stays 0.5B.
   - Smoke tests must run on CPU; select device from `CUDA_VISIBLE_DEVICES` at runtime. Do not assume a GPU exists.
   - Expect ~1–2 h for a 20-step run at G=8 × 8 tasks on this card. For the Phase 5 sweeps, the user may move the orchestrator to a Purdue Anvil GPU allocation; keep the orchestrator container portable (no host-specific paths).

`⏸ CHECKPOINT 2` — 20 training steps on the toy domain; show the reward curve (train and held-out) and KL curve. Reward on held-out should rise above the baseline prompt's score; if it doesn't, diagnose before proceeding (reward variance within groups, rejection rate, KL collapse, temperature).

### Phase 3 — Baselines the mentor will ask about
Run and log, under identical budgets (same number of harness calls):
- **B0:** baseline hand-written prompts (no optimization).
- **B1:** random-perturbation sampler from Phase 1, keep-best (no learning).
- **B2:** GRPO meta-policy (Phase 2).
Produce one table and one figure. Report cost (API tokens spent) alongside reward.

**Second domain from Hugging Face (same phase).** To show the loop is not toy-specific, add one public benchmark as a second task family, behind the same `Task`/verifier abstraction:
- Default: a tool-use benchmark such as **BFCL** (function calling) or, if simpler, **GSM8K** with a `calculator` tool. Load with `datasets.load_dataset`; cache under `env/tasks/hf/`; fix the split and seed; use ≤300 train / ≤100 held-out examples to keep API cost bounded.
- Ground truth comes from the dataset labels. The **validator stays programmatic** (exact/numeric match, schema check, tool-call count, token cost) — do not use a learned reward model from Hugging Face. Rewards must remain deterministic and decomposable.
- Re-run B0/B1/B2 on this domain with the same budget and report both domains side by side.

### Phase 4 — Make the optimizer swappable (prep for DSPy/GEPA later; do NOT implement them now)
1. Introduce an `Optimizer` protocol in the orchestrator: `propose(tasks) -> list[list[PromptBundle]]` and `update(tasks, bundles, rewards)`. Move GRPO behind it.
2. Make the harness and environment entirely optimizer-agnostic (they already should be).
3. Confirm `Reward.feedback` is being populated by the environment with a short failure description per trajectory — GRPO ignores it; GEPA will need it.

### Phase 5 — Conditioning study (the mentor's stated interest)
The point of using a continuous meta-policy is that `J(φ) = E_{P∼π_φ}[R(P)]` is smooth in φ even though `R(P)` is discrete in P. Instrument that.
1. `conditioning.py`: on a fixed evaluation batch, estimate the empirical Fisher of the LoRA parameters (diagonal + top-k eigenvalues via power iteration / Lanczos on Fisher-vector products). Report `κ = λ_max/λ_min` over the retained spectrum, and the diagonal distribution.
2. Track κ across training steps and across: temperature τ ∈ {0.7, 1.0, 1.3}, β ∈ {0.0, 0.04, 0.2}, G ∈ {4, 8, 16}.
3. Also report gradient-estimator variance (across repeated sampling at fixed φ) vs. G — this is the practical "how ill-conditioned is this really" number.
4. Write up findings in `experiments/conditioning.md` with plots. Note explicitly which effects are statistically distinguishable at the sample sizes used.

`⏸ CHECKPOINT 5` — present the conditioning write-up.

### Phase 6 (later, user decides) — Swap in the real domain
Only after the toy loop and conditioning study are done. Candidate harnesses: an MD-simulation agent or a scientific-literature knowledge-graph agent. Everything should plug in via new `Task` generators, new verifiers, and a new harness module; the orchestrator must not change.

## 5. Engineering rules
- Python 3.11+, type hints everywhere, pydantic v2 for all cross-container data.
- Async I/O (httpx + asyncio) in the harness; cap concurrency with a semaphore (default 8) and honor 429/5xx with exponential backoff.
- Every run has a `run_id`, a fixed seed, and a frozen copy of all configs in `experiments/<run_id>/config/`.
- Determinism: `staging-env` must give identical rewards for identical traces (test this).
- Cost guard: hard cap on tokens per step in the orchestrator config; abort the run with a clear message if exceeded.
- Tests: `pytest` must pass before every checkpoint. Add a test whenever you fix a bug.
- Commits: small, one concern each, imperative-mood messages. Do not commit `.env`, model weights, or `experiments/*/traces`.
- When something in these instructions is ambiguous or seems wrong, write your question in the checkpoint report rather than guessing silently.

## 6. Definition of done for this engagement
- `make up && make train RUN=toy` runs end to end and produces a reward curve that beats B0 on held-out tasks.
- `make analyze RUN=toy` produces the B0/B1/B2 table and the conditioning plots.
- `Optimizer` protocol in place with GRPO as the only implementation.
- README with: how to run, how to swap the domain, how to add an optimizer.

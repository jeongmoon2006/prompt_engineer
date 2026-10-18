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
make tasks                      # regenerate env/tasks/toy_v1.jsonl (seeded)
```

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
Reward = 0.6 correctness + 0.15 format + 0.15 tool_efficiency + 0.10 cost (weights in `env/config.yaml`).
Harness model: `claude-haiku-5-5` via the Anthropic SDK (`HARNESS_MODEL` in `.env`); `HARNESS_PROVIDER=mock` runs offline.

Swapping the domain and adding an optimizer: documented in Phase 4/6.

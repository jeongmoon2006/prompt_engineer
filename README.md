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
make test                       # pytest
```

Ports: orchestrator 8000, harness 8001, env 8002.

## Layout
- `shared/schemas.py` — the only module every container imports (pydantic contracts).
- `orchestrator/`, `harness/`, `env/` — one service each; never import across them.
- `experiments/<run_id>/` — per-run configs, metrics, traces (traces gitignored).

Swapping the domain and adding an optimizer: documented in Phase 4/6.

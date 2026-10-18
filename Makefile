# make up / down / smoke / train / analyze  (docker)        make smoke-local / test (host)
SHELL := /bin/bash
COMPOSE ?= docker compose
COMPOSE_FILES := -f docker-compose.yml
ifdef GPU
COMPOSE_FILES += -f docker-compose.gpu.yml
endif
RUN ?= toy
UV ?= uv

.PHONY: up down logs smoke smoke-local test e2e tasks train analyze venv clean

up:
	$(COMPOSE) $(COMPOSE_FILES) up -d --build

down:
	$(COMPOSE) $(COMPOSE_FILES) down

logs:
	$(COMPOSE) $(COMPOSE_FILES) logs -f --tail=100

## Bring the stack up in docker and curl /health on all three services.
smoke: up
	./scripts/smoke.sh

## Same checks without docker: run the three servers with uvicorn from .venv.
smoke-local: venv
	./scripts/smoke_local.sh

venv:
	$(UV) sync --extra dev

test: venv
	$(UV) run pytest -q

## Live end-to-end: 1 task x G=4 against the real model API (reads .env).
e2e: venv
	RUN_LIVE=1 $(UV) run pytest -q -s tests/test_e2e_live.py

## Regenerate the toy task set (deterministic; see env/config.yaml).
tasks: venv
	$(UV) run python -m env.tasks.generate

train:
	@echo "make train RUN=$(RUN): implemented in Phase 2 (orchestrator/trainer.py)"; exit 1

analyze:
	@echo "make analyze RUN=$(RUN): implemented in Phase 3/5"; exit 1

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

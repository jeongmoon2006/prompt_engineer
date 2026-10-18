# make up / down / smoke / train / analyze  (docker)        make smoke-local / test (host)
SHELL := /bin/bash
COMPOSE ?= docker compose
COMPOSE_FILES := -f docker-compose.yml
ifdef GPU
COMPOSE_FILES += -f docker-compose.gpu.yml
endif
RUN ?= toy
UV ?= uv

.PHONY: up down logs smoke smoke-local test e2e tasks train serve-local stop-local analyze venv clean

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

## Train: with docker -> inside the orchestrator container; without -> host venv against local harness/env.
STEPS ?=
POLICY ?=
DOMAIN ?=
TRAIN_ARGS := --run-id $(RUN) $(if $(STEPS),--steps $(STEPS)) $(if $(POLICY),--policy $(POLICY)) $(if $(DOMAIN),--domain $(DOMAIN))
train:
	@if command -v docker >/dev/null 2>&1; then \
	  $(COMPOSE) $(COMPOSE_FILES) exec orchestrator python -m orchestrator.trainer $(TRAIN_ARGS); \
	else \
	  ./scripts/serve_local.sh && set -a && . ./.env && set +a && $(UV) run --no-sync python -m orchestrator.trainer $(TRAIN_ARGS); \
	fi

serve-local:
	./scripts/serve_local.sh

stop-local:
	./scripts/stop_local.sh

COMPARE ?=
analyze: venv
	$(UV) run --no-sync python -m orchestrator.analyze $(RUN) $(RUNS) $(if $(COMPARE),--compare '$(COMPARE)')

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

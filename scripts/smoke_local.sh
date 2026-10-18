#!/usr/bin/env bash
# Host-level smoke: start all three servers with uvicorn from .venv, run smoke.sh, tear down.
set -uo pipefail
cd "$(dirname "$0")/.."
RUNDIR=${RUNDIR:-.run}; mkdir -p "$RUNDIR"
export HARNESS_URL=http://localhost:8001 ENV_URL=http://localhost:8002
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT
for spec in orchestrator:8000 harness:8001 env:8002; do
  pkg=${spec%%:*}; port=${spec#*:}
  PORT=$port uv run --no-sync uvicorn "$pkg.server:app" --host 127.0.0.1 --port "$port" \
    >"$RUNDIR/$pkg.log" 2>&1 &
  pids+=($!)
done
ORCH_URL=http://localhost:8000 ./scripts/smoke.sh

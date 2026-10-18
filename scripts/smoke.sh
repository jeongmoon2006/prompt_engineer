#!/usr/bin/env bash
# Curl GET /health on all three services with retries. Exit non-zero on any failure.
# Override hosts with ORCH_URL / HARNESS_URL / ENV_URL.
set -uo pipefail
ORCH_URL=${ORCH_URL:-http://localhost:8000}
HARNESS_URL=${HARNESS_URL:-http://localhost:8001}
ENV_URL=${ENV_URL:-http://localhost:8002}
RETRIES=${RETRIES:-30}
fail=0
for pair in "rl-orchestrator=$ORCH_URL" "agent-harness=$HARNESS_URL" "staging-env=$ENV_URL"; do
  name=${pair%%=*}; url=${pair#*=}
  for ((i=1; i<=RETRIES; i++)); do
    body=$(curl -fsS --max-time 3 "$url/health" 2>/dev/null) && break
    sleep 1
  done
  if [[ "$body" == *'"ok":true'* ]]; then
    printf '  ok    %-16s %s/health -> %s\n' "$name" "$url" "$body"
  else
    printf '  FAIL  %-16s %s/health (no healthy response after %ss)\n' "$name" "$url" "$RETRIES"
    fail=1
  fi
done
[[ $fail -eq 0 ]] && echo "smoke: all services healthy" || { echo "smoke: FAILED"; exit 1; }

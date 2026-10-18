#!/usr/bin/env bash
# Start harness (8001) and env (8002) with uvicorn in the background (no docker). PIDs in .run/.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .run
set -a; [ -f .env ] && . ./.env; set +a
for spec in harness:8001 env:8002; do
  pkg=${spec%%:*}; port=${spec#*:}
  if curl -fsS --max-time 1 "http://localhost:$port/health" >/dev/null 2>&1; then echo "$pkg already up on $port"; continue; fi
  PORT=$port nohup uv run --no-sync uvicorn "$pkg.server:app" --host 127.0.0.1 --port "$port" >".run/$pkg.log" 2>&1 &
  echo $! > ".run/$pkg.pid"; echo "started $pkg on $port (pid $!)"
done
RETRIES=30 HARNESS_URL=http://localhost:8001 ENV_URL=http://localhost:8002 ORCH_URL="" bash -c '
for url in $HARNESS_URL $ENV_URL; do for i in $(seq 1 $RETRIES); do curl -fsS --max-time 2 $url/health >/dev/null 2>&1 && break; sleep 1; done; done'
echo "harness/env ready"

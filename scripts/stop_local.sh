#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for f in .run/*.pid; do [ -f "$f" ] && { kill "$(cat "$f")" 2>/dev/null && echo "stopped $(basename "$f" .pid)"; rm -f "$f"; }; done

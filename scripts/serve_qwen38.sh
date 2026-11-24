#!/usr/bin/env zsh
# STRATA-M5 v2.0 — OpenAI-Compatible Server for OpenCode: Qwen3.8-27B (14.1 GiB Zero-Copy UMA Dense Hybrid)
# Lightweight local server (leaves >110 GiB UMA free for heavy builds/browsers/emulators)
# Endpoint: http://127.0.0.1:8080/v1
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ./build/strata_cli ]]; then
  echo "[strata] Building ./build/strata_cli..."
  make -j8 cli
fi

exec ./build/strata_cli serve \
  --model qwen38 \
  --host 127.0.0.1 \
  --port 8080 \
  --max-seq-len 32768 \
  --max-tokens 16384 \
  "$@"

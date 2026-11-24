#!/usr/bin/env zsh
# STRATA-M5 v2.0 — OpenAI-Compatible Server for OpenCode: Qwen3.8-Flash-Next (180B MoE + PLE + MTP K=3)
# Fastest coding & tool-calling engine (~114-121 tok/s, 256K context, 64K max output, Multi-Checkpoint Prefix Cache + Scratch Slot)
# Endpoint: http://127.0.0.1:8080/v1 (forward any extra flags like `--port 8081 --effort low --compact-mode paged`)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STRATA_BIN="./build/strata-apex"
if [[ ! -x "$STRATA_BIN" ]]; then
  if [[ -x ./build/strata_cli ]]; then STRATA_BIN="./build/strata_cli"
  elif command -v strata-apex >/dev/null 2>&1; then STRATA_BIN="$(command -v strata-apex)"
  else echo "[strata-apex] Building ./build/strata-apex..."; make -j8 cli; fi
fi

STRATA_CACHE="${STRATA_APEX_CACHE_DIR:-${STRATA_CACHE_DIR:-$HOME/.cache/strata-apex}}"
[[ ! -d "$STRATA_CACHE" && -d "$HOME/.cache/strata" ]] && STRATA_CACHE="$HOME/.cache/strata"

BUNDLE_ARGS=()
if [[ -f "$STRATA_CACHE/qwen3.8-flash-next-fast/uma_part0.strata" ]]; then
  BUNDLE_ARGS=(--bundle "$STRATA_CACHE/qwen3.8-flash-next-fast")
elif [[ -f "$HOME/.cache/strata/qwen3.8-flash-next-fast/uma_part0.strata" ]]; then
  BUNDLE_ARGS=(--bundle "$HOME/.cache/strata/qwen3.8-flash-next-fast")
fi

exec caffeinate -dimsu "$STRATA_BIN" serve \
  --model flash \
  "${BUNDLE_ARGS[@]}" \
  --host 127.0.0.1 \
  --port 8080 \
  --effort medium \
  --penalty-window 256 \
  --presence-penalty 0.6 \
  --frequency-penalty 0.08 \
  --max-seq-len 262144 \
  --max-tokens 65536 \
  "$@"

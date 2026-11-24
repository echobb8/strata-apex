#!/usr/bin/env zsh
# STRATA-M5 Repeatable 2K -> 8K -> 32K XHigh Thinking Scaling Benchmark
# Keeps exact same analytical task, thinking=xhigh, deterministic greedy decoding,
# and identical output token cap across 2,048 / 8,192 / 32,768 input tokens,
# reporting TTFT (ms & prefill tok/s) separately from Decode (ms & decode tok/s).

set -euo pipefail

REPO_ROOT="${0:A:h:h}"
cd "$REPO_ROOT"

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

exec caffeinate -dimsu "$STRATA_BIN" bench-flash \
  "${BUNDLE_ARGS[@]}" \
  --xhigh-scaling \
  "$@"

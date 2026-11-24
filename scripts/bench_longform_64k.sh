#!/usr/bin/env zsh
# STRATA-M5 True 64K+ Long-Form Workstream & Subtle Narrative Compaction Benchmark
# - Window C_max = 65,536 (64K) tokens, W_swap = 52,428 tokens, Tier-2 UMA Pool = 32,768 tokens
# - Drives >95,000 cumulative workstream tokens across 7 realistic ~13.5K-token turns + 3 blind probes
# - Idle-gap 180B LLM Semantic Micro-Distiller + Deterministic Rule/Fact Ledger + O(1) RCU Shadow Swap
# - Tests: (1) Math + dormant negative, (2) Subtle implicit narrative nuance buried ~70% into Turn 1
#   with zero regex/keyword crutches, and (3) Casual prompt pressure + zero-echo dormant trigger.

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

exec caffeinate -dimsu "$STRATA_BIN" chat-flash \
  "${BUNDLE_ARGS[@]}" \
  --bench-longform-64k \
  "$@"

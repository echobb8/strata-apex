#!/usr/bin/env zsh
# Strata Apex — OpenAI-Compatible Server for OpenCode: Inkling-2-Mini (26B Looped -fast Q4_A/Q8_0/NVFP4 + 3-Layer Q4_A DSpark Block-8 MTP Adaptive K=0..7 + 32K FR-Spec)
# Endpoint: http://127.0.0.1:8080/v1 (forward any extra flags like `--no-refusal --port 8081 --effort high --kv-cache q8 --max-seq-len 1048576`)
# Add `--no-refusal` to also hot-load the ~80 MiB `inkling-2-mini-norefusal` wo overlay on the same server (routed via OpenCode `/no-refusal`).
# Auto-loads `inkling2_vision.strata` (multimodal `image_url` on both stock & `--no-refusal`), Dynamic W=4 Beam Tree (`--tree-spec`), and 2-Stage Bitplane Head (`--bitplane-head`).
# Add `--max-batch 8` (1..64; e.g. `--max-batch 8 --max-seq-len 16384` or `--max-batch 16 --kv-cache q8`) for opt-in iteration-level continuous batching across concurrent agent swarms (default solo c=1).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STRATA_BIN="./build/strata-apex"
if [[ ! -x "$STRATA_BIN" ]]; then
  if [[ -x ./build/strata_cli ]]; then
    STRATA_BIN="./build/strata_cli"
  elif command -v strata-apex >/dev/null 2>&1; then
    STRATA_BIN="$(command -v strata-apex)"
  else
    echo "[strata-apex] Building ./build/strata-apex..."
    make -j8 cli
  fi
fi

STRATA_CACHE="${STRATA_APEX_CACHE_DIR:-${STRATA_CACHE_DIR:-$HOME/.cache/strata-apex}}"
[[ ! -d "$STRATA_CACHE" && -d "$HOME/.cache/strata" ]] && STRATA_CACHE="$HOME/.cache/strata"

BUNDLE_ARGS=()
if [[ -f "$STRATA_CACHE/inkling-2-mini-fast/inkling2_uma.strata" ]]; then
  BUNDLE_ARGS=(--bundle "$STRATA_CACHE/inkling-2-mini-fast")
elif [[ -f "$HOME/.cache/strata/inkling-2-mini-fast/inkling2_uma.strata" ]]; then
  BUNDLE_ARGS=(--bundle "$HOME/.cache/strata/inkling-2-mini-fast")
fi

exec caffeinate -dimsu "$STRATA_BIN" serve \
  --model inkling2 \
  "${BUNDLE_ARGS[@]}" \
  --host 127.0.0.1 \
  --port 8080 \
  --draft-k 7 \
  --effort medium \
  --penalty-window 256 \
  --presence-penalty 0.6 \
  --frequency-penalty 0.08 \
  --max-seq-len 65536 \
  --max-tokens 32768 \
  "$@"

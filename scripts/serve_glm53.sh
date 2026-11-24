#!/usr/bin/env zsh
# STRATA-M5 v2.0 — OpenAI-Compatible Server for OpenCode: GLM-5.3-Flash (320B / 18B Active)
# Auto-selects 100% RAM-resident E8S-18-V4 (`glm-5.3-flash-fast`, 93.22 GiB UMA, lossless, MTP K=4)
# when present, else falls back to v1 Regime II NVMe-streaming (`glm-5.3-flash`, adaptive, MTP K=2).
# Endpoint: http://127.0.0.1:8080/v1 (forward any extra flags like `--port 8081 --effort medium`)
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

FAST_DIR="$STRATA_CACHE/glm-5.3-flash-fast"
[[ ! -f "$FAST_DIR/spine.strata" && -f "$HOME/.cache/strata/glm-5.3-flash-fast/spine.strata" ]] && FAST_DIR="$HOME/.cache/strata/glm-5.3-flash-fast"
BUNDLE_ARGS=()
MODE_ARGS=(--mode adaptive)
if [[ -f "$FAST_DIR/spine.strata" && -f "$FAST_DIR/experts_uma_part0.strata" ]]; then
  BUNDLE_ARGS=(--bundle "$FAST_DIR")
  MODE_ARGS=(
    --mode lossless
    --draft-k 4
    --penalty-window 256
    --presence-penalty 0.6
    --frequency-penalty 0.08
  )
fi

exec caffeinate -dimsu "$STRATA_BIN" serve \
  --model glm53 \
  "${BUNDLE_ARGS[@]}" \
  --host 127.0.0.1 \
  --port 8080 \
  "${MODE_ARGS[@]}" \
  --effort high \
  --max-seq-len 131072 \
  --max-tokens 32768 \
  "$@"

#!/usr/bin/env zsh
# STRATA-M5 v2.0 — Qwen3.8-Flash-Next (180B) 64K+ Zero-Pause Hybrid Compaction Chat Launcher
# Supports: multi-paragraph Bracketed Paste, /save <file>, /load <file>, /compact-stats, /undo, /undo-compact
# Any extra CLI flags (e.g. `--load my.strata_session`, `--no-think`, `--system "..."`) are forwarded via "$@".
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

exec "$STRATA_BIN" chat-flash \
  "${BUNDLE_ARGS[@]}" \
  --compact-mode hybrid \
  --compact-window 65536 \
  --compact-pool 32768 \
  --effort xhigh \
  "$@"

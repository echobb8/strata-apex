#!/usr/bin/env zsh
# STRATA-M5 v2.0 — Qwen3.8-27B (14.1 GiB Zero-Copy UMA Dense Hybrid) Chat Launcher
# Supports: multi-paragraph Bracketed Paste, /save <file>, /load <file>, /stats, /undo, /reset
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ./build/strata_cli ]]; then
  echo "[strata] Building ./build/strata_cli..."
  make -j8 cli
fi

exec ./build/strata_cli chat-hf "$@"

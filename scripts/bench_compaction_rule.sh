#!/usr/bin/env zsh
# STRATA-M5 Repeatable Multi-Compaction Turn-1 Made-Up Requirement Benchmark
# Injects a made-up behavioral/formatting requirement in Turn 1, drives the session
# through 3+ realtime RCU compactions (with thinking=xhigh and fixed output token cap,
# separating TTFT ms from Decode ms & tok/s on every turn), and tests whether Turn 8
# spontaneously obeys the Turn-1 rule with ZERO reminder in the Turn-8 prompt.
#
# Optional overrides:
#   ./scripts/bench_compaction_rule.sh --bench-rule "Mandatory standing rule: ..." --bench-expect "TOKEN1,TOKEN2"

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
  --bench-compaction-rule \
  "$@"

#!/usr/bin/env zsh
# STRATA // APEX — Clean, Screenshot-Ready Apple Silicon AI Benchmark
# Supports Qwen3.8-Flash-Next (180B), Inkling-2-Mini (26B Looped), and GLM-5.3-Flash (320B).
#
# Usage:
#   ./scripts/apex.sh                          # Qwen3.8-Flash-Next (180B), full 512..32K
#   ./scripts/apex.sh inkling                  # Inkling-2-Mini (26B), full 512..32K + Swarm (1..16x)
#   ./scripts/apex.sh inkling --swarm-only     # Fast Swarm / Concurrent Sessions card only (~15s)
#   ./scripts/apex.sh inkling --swarm-batches 1,4,8,16,32 --kv-cache q8
#   ./scripts/apex.sh inkling --no-swarm       # Solo APEX card only (skip continuous-batch swarm)
#   ./scripts/apex.sh glm53                    # GLM-5.3-Flash (320B), full 512..32K
#   ./scripts/apex.sh glm53 --quick            # Faster: 512..16K + 192-tok workloads
#   ./scripts/apex.sh inkling --runs 3         # True median of 3 repetitions per metric
#   ./scripts/apex.sh inkling --runs 3 --cooldown 60  # 60s idle cooldown between runs
#   ./scripts/apex.sh inkling --int8-prefill   # Opt-in Metal 4.1 INT8 NAX prefill (approx)
#   ./scripts/apex.sh qwen --quick --clear     # Clear screen before printing final card
#   ./scripts/apex.sh --model glm53 --out card.md [--keep-logs] [--allow-battery]
#   ./scripts/bench_showcase.sh --model inkling2 --quick

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
IS_SOURCE_CHECKOUT=0
if [[ "$SCRIPT_DIR" == "$REPO_ROOT/scripts" && -f "$REPO_ROOT/Makefile" && -f "$REPO_ROOT/src/cli/strata_cli.mm" ]]; then
  IS_SOURCE_CHECKOUT=1
  cd "$REPO_ROOT"
fi

STRATA_CACHE="${STRATA_APEX_CACHE_DIR:-${STRATA_CACHE_DIR:-$HOME/.cache/strata-apex}}"
[[ ! -d "$STRATA_CACHE" && -d "$HOME/.cache/strata" ]] && STRATA_CACHE="$HOME/.cache/strata"

resolve_bundle_dir() {
  local fast_rel="$1"
  local fast_marker="$2"
  local v1_rel="$3"
  for root in "${STRATA_APEX_CACHE_DIR:-}" "${STRATA_CACHE_DIR:-}" "$HOME/.cache/strata-apex" "$HOME/.cache/strata"; do
    [[ -z "$root" ]] && continue
    if [[ -f "$root/$fast_rel/$fast_marker" ]]; then
      echo "$root/$fast_rel"
      return 0
    fi
  done
  for root in "${STRATA_APEX_CACHE_DIR:-}" "${STRATA_CACHE_DIR:-}" "$HOME/.cache/strata-apex" "$HOME/.cache/strata"; do
    [[ -z "$root" ]] && continue
    if [[ -d "$root/$v1_rel" ]]; then
      echo "$root/$v1_rel"
      return 0
    fi
  done
  echo "$STRATA_CACHE/$fast_rel"
}

MODEL="qwen"
CONTEXTS="512,2048,4096,16384,32768"
CTX_GEN=128
MAT_GEN=256
DRAFT_K=""
KV_CACHE=""
THINK_FLAG="--no-think"
BUNDLE_DIR=""
OUT_MD=""
COLOR_MODE="auto"
CLEAR_SCREEN=0
RUNS=1
COOLDOWN_SEC=0
KEEP_LOGS=0
INT8_PREFILL=0
ALLOW_BATTERY=0
SWARM_MODE="auto"   # auto (ON for inkling2, OFF for others), on, off
SWARM_ONLY=0
SWARM_BATCHES="1,4,8,16"
SWARM_TOKENS=128
SWARM_CTX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    qwen|flash|qwen3.8|qwen3.8-flash-next)
      MODEL="qwen"; shift
      ;;
    inkling|inkling2|inkling-2-mini)
      MODEL="inkling2"; shift
      ;;
    glm|glm53|glm-5.3|glm-5.3-flash|320b)
      MODEL="glm53"; shift
      ;;
    --model|-m)
      case "$2" in
        qwen|flash|qwen3.8|qwen3.8-flash-next) MODEL="qwen" ;;
        inkling|inkling2|inkling-2-mini)       MODEL="inkling2" ;;
        glm|glm53|glm-5.3|glm-5.3-flash|320b)  MODEL="glm53" ;;
        *)
          echo "[STRATA // APEX] Unknown model '$2' (use 'qwen', 'inkling', or 'glm53')." >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --quick|-q)
      CONTEXTS="512,2048,4096,16384"
      MAT_GEN=192
      shift
      ;;
    --contexts)
      CONTEXTS="$2"; shift 2
      ;;
    --gen-tokens)
      CTX_GEN="$2"; shift 2
      ;;
    --matrix-tokens)
      MAT_GEN="$2"; shift 2
      ;;
    --runs|-r)
      if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "[STRATA // APEX] Invalid --runs '$2' (expected positive integer >= 1)." >&2
        exit 1
      fi
      RUNS="$2"; shift 2
      ;;
    --cooldown)
      if [[ ! "$2" =~ ^[0-9]+$ ]]; then
        echo "[STRATA // APEX] Invalid --cooldown '$2' (expected non-negative integer seconds)." >&2
        exit 1
      fi
      COOLDOWN_SEC="$2"; shift 2
      ;;
    --keep-logs)
      KEEP_LOGS=1; shift
      ;;
    --int8-prefill)
      INT8_PREFILL=1; shift
      ;;
    --fp16-prefill|--no-int8-prefill)
      INT8_PREFILL=0; shift
      ;;
    --allow-battery)
      ALLOW_BATTERY=1; shift
      ;;
    --swarm)
      SWARM_MODE="on"; shift
      ;;
    --no-swarm)
      SWARM_MODE="off"; SWARM_ONLY=0; shift
      ;;
    --swarm-only)
      SWARM_MODE="on"; SWARM_ONLY=1; shift
      ;;
    --swarm-batches)
      if [[ ! "$2" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "[STRATA // APEX] Invalid --swarm-batches '$2' (expected comma-separated integers 1..64, e.g. 1,4,8,16)." >&2
        exit 1
      fi
      SWARM_BATCHES="$2"; SWARM_MODE="on"; shift 2
      ;;
    --swarm-tokens)
      if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "[STRATA // APEX] Invalid --swarm-tokens '$2' (expected positive integer)." >&2
        exit 1
      fi
      SWARM_TOKENS="$2"; shift 2
      ;;
    --swarm-ctx)
      if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "[STRATA // APEX] Invalid --swarm-ctx '$2' (expected positive integer, e.g. 16384)." >&2
        exit 1
      fi
      SWARM_CTX="$2"; shift 2
      ;;
    --draft-k)
      DRAFT_K="$2"; shift 2
      ;;
    --kv-cache)
      case "$2" in
        f16|fp16|q8|q8_0) KV_CACHE="$2" ;;
        *)
          echo "[STRATA // APEX] Invalid --kv-cache '$2' (expected f16 or q8)." >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --think)
      THINK_FLAG="--think"; shift
      ;;
    --no-think)
      THINK_FLAG="--no-think"; shift
      ;;
    --bundle)
      BUNDLE_DIR="$2"; shift 2
      ;;
    --out|-o)
      OUT_MD="$2"; shift 2
      ;;
    --color)
      COLOR_MODE="always"; shift
      ;;
    --no-color)
      COLOR_MODE="never"; shift
      ;;
    --clear)
      CLEAR_SCREEN=1; shift
      ;;
    -h|--help)
      sed -n '2,19p' "$0"
      exit 0
      ;;
    *)
      echo "[STRATA // APEX] Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ── AC-Power & macOS Thermal Pressure Guard ──────────────────────────────────
BATT_OUT="$(pmset -g batt 2>/dev/null || true)"
if [[ "$BATT_OUT" == *"Battery Power"* ]]; then
  if [[ "$ALLOW_BATTERY" -eq 0 ]]; then
    echo "[STRATA // APEX] ERROR: Mac is running on Battery Power (Apple Silicon clocks may throttle)." >&2
    echo "  Plug into AC Power for reproducible results, or pass --allow-battery to override." >&2
    exit 1
  fi
  echo "[STRATA // APEX] WARNING: Running on Battery Power (--allow-battery); clocks/results may throttle." >&2
  POWER_STATUS="Battery (override)"
elif [[ "$BATT_OUT" == *"AC Power"* ]]; then
  POWER_STATUS="AC Power"
else
  POWER_STATUS="AC/Desktop"
fi

THERM_LVL="$(sysctl -n machdep.xcpm.cpu_thermal_level 2>/dev/null || true)"
if [[ -n "$THERM_LVL" ]]; then
  if [[ "$THERM_LVL" == "0" ]]; then
    THERM_STATUS="no OS thermal warning (cpu_thermal_level=0)"
  else
    THERM_STATUS="WARNING (cpu_thermal_level=$THERM_LVL)"
  fi
else
  PM_THERM="$(pmset -g therm 2>/dev/null || true)"
  if [[ "$PM_THERM" == *"No thermal warning level has been recorded"* && "$PM_THERM" != *"CPU_Speed_Limit"* ]]; then
    THERM_STATUS="no OS thermal warning"
  elif [[ -n "$PM_THERM" ]]; then
    THERM_BRIEF="$(printf '%s\n' "$PM_THERM" | grep -v '^Note:' | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//')"
    THERM_STATUS="WARNING (${THERM_BRIEF:-thermal limit active})"
  else
    THERM_STATUS="unknown"
  fi
fi
echo "[STRATA // APEX] Power: $POWER_STATUS | Thermal: $THERM_STATUS" >&2

EXTRA_BENCH_ARGS=()
EXTRA_RUN_ARGS=()

if [[ "$MODEL" == "inkling2" ]]; then
  INSPECT_CMD="inspect-inkling2"
  BENCH_CMD="bench-inkling2"
  RUN_CMD="run-inkling2"
  [[ -z "$DRAFT_K" ]] && DRAFT_K=7
  if [[ -z "$BUNDLE_DIR" ]]; then
    BUNDLE_DIR="$(resolve_bundle_dir inkling-2-mini-fast inkling2_uma.strata inkling-2-mini)"
  fi
  if [[ -n "$KV_CACHE" ]]; then
    EXTRA_BENCH_ARGS=(--kv-cache "$KV_CACHE")
    EXTRA_RUN_ARGS=(--kv-cache "$KV_CACHE")
  fi
  MODEL_LABEL="Inkling-2-Mini"
elif [[ "$MODEL" == "glm53" ]]; then
  INSPECT_CMD="inspect-glm53"
  BENCH_CMD="bench-glm53"
  RUN_CMD="run-glm53"
  if [[ -z "$BUNDLE_DIR" ]]; then
    BUNDLE_DIR="$(resolve_bundle_dir glm-5.3-flash-fast experts_uma_part0.strata glm-5.3-flash)"
  fi
  if [[ -z "$DRAFT_K" ]]; then
    if [[ -f "$BUNDLE_DIR/experts_uma_part0.strata" ]]; then
      DRAFT_K=4
    else
      DRAFT_K=2
    fi
  fi
  EXTRA_BENCH_ARGS=(--mode lossless)
  EXTRA_RUN_ARGS=(--mode lossless)
  MODEL_LABEL="GLM-5.3-Flash"
else
  INSPECT_CMD="inspect-flash"
  BENCH_CMD="bench-flash"
  RUN_CMD="run-flash"
  [[ -z "$DRAFT_K" ]] && DRAFT_K=4
  if [[ -z "$BUNDLE_DIR" ]]; then
    BUNDLE_DIR="$(resolve_bundle_dir qwen3.8-flash-next-fast uma_part0.strata qwen3.8-flash-next)"
  fi
  MODEL_LABEL="Qwen3.8-Flash-Next"
fi

if [[ "$INT8_PREFILL" -eq 1 ]]; then
  EXTRA_BENCH_ARGS+=(--int8-prefill)
  EXTRA_RUN_ARGS+=(--int8-prefill)
fi

SWARM_ENABLED=0
if [[ "$SWARM_MODE" == "auto" ]]; then
  [[ "$MODEL" == "inkling2" ]] && SWARM_ENABLED=1
elif [[ "$SWARM_MODE" == "on" ]]; then
  if [[ "$MODEL" != "inkling2" ]]; then
    echo "[STRATA // APEX] Continuous-batching swarm stats are currently supported for Inkling-2-Mini (--model inkling)." >&2
    exit 1
  fi
  SWARM_ENABLED=1
fi

MAX_SWARM_BATCH=1
if [[ "$SWARM_ENABLED" -eq 1 ]]; then
  for _cb in ${(s:,:)SWARM_BATCHES}; do
    if [[ -z "$_cb" || "$_cb" -lt 1 || "$_cb" -gt 64 ]]; then
      echo "[STRATA // APEX] Invalid batch size '$_cb' in --swarm-batches (expected 1..64)." >&2
      exit 1
    fi
    (( _cb > MAX_SWARM_BATCH )) && MAX_SWARM_BATCH="$_cb"
  done
  if [[ -z "$SWARM_CTX" ]]; then
    if [[ "$MAX_SWARM_BATCH" -gt 32 && "$KV_CACHE" != "q8" && "$KV_CACHE" != "q8_0" ]]; then
      SWARM_CTX=4096
    elif [[ "$MAX_SWARM_BATCH" -gt 16 && "$KV_CACHE" != "q8" && "$KV_CACHE" != "q8_0" ]]; then
      SWARM_CTX=8192
    else
      SWARM_CTX=16384
    fi
  fi
fi

if pgrep -f "(strata-apex|strata_cli) serve" >/dev/null 2>&1; then
  echo "[STRATA // APEX] Stop your running 'strata-apex serve' first (Ctrl+C) for clean memory & numbers." >&2
  exit 1
fi

STRATA_BIN=""
if [[ "$IS_SOURCE_CHECKOUT" -eq 1 ]]; then
  if [[ -x "$REPO_ROOT/build/strata-apex" ]]; then
    STRATA_BIN="./build/strata-apex"
  elif [[ -x "$REPO_ROOT/build/strata_cli" ]]; then
    STRATA_BIN="./build/strata_cli"
  elif [[ -x "$SCRIPT_DIR/strata-apex" ]]; then
    STRATA_BIN="$SCRIPT_DIR/strata-apex"
  elif command -v strata-apex >/dev/null 2>&1; then
    STRATA_BIN="$(command -v strata-apex)"
  else
    echo "[STRATA // APEX] Building ./build/strata-apex..." >&2
    make -C "$REPO_ROOT" -j8 cli >&2
    STRATA_BIN="./build/strata-apex"
  fi
else
  if [[ -x "$SCRIPT_DIR/strata-apex" ]]; then
    STRATA_BIN="$SCRIPT_DIR/strata-apex"
  elif command -v strata-apex >/dev/null 2>&1; then
    STRATA_BIN="$(command -v strata-apex)"
  else
    echo "[STRATA // APEX] ERROR: Could not locate 'strata-apex' binary." >&2
    exit 1
  fi
fi

BIN_STALE=0
if [[ "$IS_SOURCE_CHECKOUT" -eq 1 ]] && [[ "$STRATA_BIN" == "./build/strata-apex" || "$STRATA_BIN" == "./build/strata_cli" ]]; then
  if ! make -C "$REPO_ROOT" -q cli >/dev/null 2>&1; then
    echo "[STRATA // APEX] WARNING: $STRATA_BIN is older than current source files (run 'make -j8 cli' to rebuild)." >&2
    BIN_STALE=1
  fi
fi
BIN_MTIME="$(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$STRATA_BIN" 2>/dev/null || echo 'unknown')"
echo "[STRATA // APEX] Binary: $STRATA_BIN (built $BIN_MTIME)" >&2

TMP_DIR="$(mktemp -d -t strata_apex.XXXXXX)"
SCRIPT_FAILED=0
HAD_GATE_FAIL=0
HAD_MAJFLT_WARN=0
cleanup_tmp() {
  if [[ "$KEEP_LOGS" -eq 1 || "$SCRIPT_FAILED" -ne 0 || "$HAD_GATE_FAIL" -ne 0 || "$HAD_MAJFLT_WARN" -ne 0 ]]; then
    echo "[STRATA // APEX] Raw benchmark logs preserved at: $TMP_DIR" >&2
  else
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup_tmp EXIT

run_ctx_step() {
  local step_desc="$1"
  local json_out="$2"
  local log_out="$3"
  shift 3
  local rc=0
  caffeinate -dimsu "$@" > "$log_out" 2>&1 || rc=$?
  if [[ "$rc" -eq 0 && -s "$json_out" ]]; then
    return 0
  fi
  # Check if JSON was written completely and whether only Steady MajFlt > 0 tripped the exit code
  local check_res=""
  if [[ -s "$json_out" ]]; then
    check_res="$(python3 -I -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    rows = d.get("results", [])
    if not rows:
        print("INCOMPLETE")
    else:
        bad_needle = any(not bool(r.get("needle_pass", False)) for r in rows)
        bad_exact  = any(not bool(r.get("exact_k0_match", True)) for r in rows)
        max_flt    = max(int(r.get("steady_majflt", 0) or 0) for r in rows)
        if not bad_needle and not bad_exact and max_flt > 0:
            print(f"MAJFLT_ONLY:{max_flt}")
        else:
            print("GATE_FAIL")
except Exception:
    print("INCOMPLETE")
' "$json_out" 2>/dev/null || echo "INCOMPLETE")"
  else
    check_res="INCOMPLETE"
  fi

  if [[ "$check_res" == MAJFLT_ONLY:* ]]; then
    local flt_cnt="${check_res#MAJFLT_ONLY:}"
    echo "[STRATA // APEX] WARNING: $step_desc hit transient Steady MajFlt=${flt_cnt}; retrying once..." >&2
    mv "$log_out" "${log_out}.attempt1"
    mv "$json_out" "${json_out}.attempt1"
    rc=0
    caffeinate -dimsu "$@" > "$log_out" 2>&1 || rc=$?
    if [[ "$rc" -eq 0 && -s "$json_out" ]]; then
      rm -f "${log_out}.attempt1" "${json_out}.attempt1"
      return 0
    fi
    if [[ -s "$json_out" ]]; then
      HAD_MAJFLT_WARN=1
      echo "[STRATA // APEX] WARNING: $step_desc still exited with status $rc on retry; continuing to render card." >&2
      return 0
    fi
  elif [[ "$check_res" == "GATE_FAIL" ]]; then
    HAD_GATE_FAIL=1
    echo "[STRATA // APEX] WARNING: $step_desc reported a recall or K=0 exactness failure (exit $rc); continuing so card displays failing cells." >&2
    return 0
  fi

  SCRIPT_FAILED=1
  echo "[STRATA // APEX] ERROR: $step_desc failed (exit $rc) before producing complete results." >&2
  if [[ -f "$log_out" ]]; then
    echo "  ── Last lines of $log_out ──" >&2
    tail -n 25 "$log_out" >&2 || true
  fi
  exit "${rc:-1}"
}

run_gen_step() {
  local step_desc="$1"
  local log_out="$2"
  shift 2
  local rc=0
  caffeinate -dimsu "$@" > "$log_out" 2>&1 || rc=$?
  if [[ "$rc" -eq 0 ]] && grep -q "Decode Throughput" "$log_out" 2>/dev/null; then
    return 0
  fi
  if grep -q "Decode Throughput" "$log_out" 2>/dev/null; then
    if grep -qE "Steady MajFlt=[1-9]" "$log_out" 2>/dev/null && ! grep -q "MISMATCH" "$log_out" 2>/dev/null; then
      echo "[STRATA // APEX] WARNING: $step_desc hit transient Steady MajFlt > 0; retrying once..." >&2
      mv "$log_out" "${log_out}.attempt1"
      rc=0
      caffeinate -dimsu "$@" > "$log_out" 2>&1 || rc=$?
      if [[ "$rc" -eq 0 ]] && grep -q "Decode Throughput" "$log_out" 2>/dev/null; then
        rm -f "${log_out}.attempt1"
        return 0
      fi
      if grep -q "Decode Throughput" "$log_out" 2>/dev/null && ! grep -q "MISMATCH" "$log_out" 2>/dev/null; then
        HAD_MAJFLT_WARN=1
        echo "[STRATA // APEX] WARNING: $step_desc still reported Steady MajFlt > 0 on retry; continuing to render card." >&2
        return 0
      fi
    fi
    if grep -q "Decode Throughput" "$log_out" 2>/dev/null; then
      HAD_GATE_FAIL=1
      echo "[STRATA // APEX] WARNING: $step_desc failed exactness/gate check (exit $rc); continuing so card displays failing cell." >&2
      return 0
    fi
  fi

  SCRIPT_FAILED=1
  echo "[STRATA // APEX] ERROR: $step_desc failed (exit $rc) before completing generation." >&2
  if [[ -f "$log_out" ]]; then
    echo "  ── Last lines of $log_out ──" >&2
    tail -n 25 "$log_out" >&2 || true
  fi
  exit "${rc:-1}"
}

TOTAL_STEPS=4
if [[ "$SWARM_ONLY" -eq 1 ]]; then
  TOTAL_STEPS=2
elif [[ "$SWARM_ENABLED" -eq 1 ]]; then
  TOTAL_STEPS=5
fi

MODE_DESC=""
[[ "$INT8_PREFILL" -eq 1 ]] && MODE_DESC=" [INT8 Prefill]"
[[ "$SWARM_ONLY" -eq 1 ]] && MODE_DESC="${MODE_DESC} [Swarm Only]"
if [[ "$RUNS" -gt 1 ]]; then
  MODE_DESC="${MODE_DESC} (median of $RUNS runs"
  [[ "$COOLDOWN_SEC" -gt 0 ]] && MODE_DESC="${MODE_DESC}, ${COOLDOWN_SEC}s cooldown"
  MODE_DESC="${MODE_DESC})"
else
  MODE_DESC="${MODE_DESC} (single run)"
fi
echo "[STRATA // APEX] Running $MODEL_LABEL benchmark${MODE_DESC}..." >&2
echo "  [1/$TOTAL_STEPS] Inspecting system & model bundle..." >&2
if ! "$STRATA_BIN" "$INSPECT_CMD" --bundle "$BUNDLE_DIR" > "$TMP_DIR/inspect.txt" 2>&1; then
  SCRIPT_FAILED=1
  echo "[STRATA // APEX] ERROR: Step [1/$TOTAL_STEPS] ($INSPECT_CMD --bundle $BUNDLE_DIR) failed." >&2
  tail -n 25 "$TMP_DIR/inspect.txt" >&2 || true
  exit 1
fi

PROSE_PROMPT="Explain in clear, engaging prose how modern unified-memory GPUs differ from discrete GPUs for running large Mixture-of-Experts language models, covering memory bandwidth, cache hierarchy, quantization trade-offs, and speculative decoding. Write complete paragraphs without code blocks."

CODE_PROMPT="Write a clean, well-typed TypeScript module implementing a generic O(1) LRUCache<K, V> backed by an intrusive doubly-linked list and Map, supporting per-entry TTL expiration, hit/miss statistics, peek(), delete(), clear(), and an eviction callback. Output code only."

RUN_COMMON=(
  --bundle "$BUNDLE_DIR"
  "${EXTRA_RUN_ARGS[@]}"
  "$THINK_FLAG"
  --effort medium
  --draft-k "$DRAFT_K"
  --max-tokens "$MAT_GEN"
  --max-seq-len 8192
  --no-eos-stop
)

RUN_COMMON_K0=(
  --bundle "$BUNDLE_DIR"
  "${EXTRA_RUN_ARGS[@]}"
  "$THINK_FLAG"
  --effort medium
  --draft-k 0
  --max-tokens "$MAT_GEN"
  --max-seq-len 8192
  --no-eos-stop
)

SAMPLE_ARGS=(
  --temp 0.6
  --top-k 20
  --top-p 0.95
  --penalty-window 256
  --presence-penalty 0.6
  --frequency-penalty 0.08
  --seed 42
)

if [[ "$SWARM_ONLY" -eq 0 ]]; then
  for (( RUN_IDX = 1; RUN_IDX <= RUNS; ++RUN_IDX )); do
    RUN_TAG=""
    [[ "$RUNS" -gt 1 ]] && RUN_TAG=" [run $RUN_IDX/$RUNS]"

    echo "  [2/$TOTAL_STEPS]${RUN_TAG} Context scaling & recall ($CONTEXTS)..." >&2
    run_ctx_step "Step [2/$TOTAL_STEPS]${RUN_TAG} ($BENCH_CMD)" \
      "$TMP_DIR/ctx_${RUN_IDX}.json" "$TMP_DIR/ctx_${RUN_IDX}.log" \
      "$STRATA_BIN" "$BENCH_CMD" \
      --bundle "$BUNDLE_DIR" \
      "${EXTRA_BENCH_ARGS[@]}" \
      --contexts "$CONTEXTS" \
      --draft-ks "0,$DRAFT_K" \
      --gen-tokens "$CTX_GEN" \
      --no-think \
      --json "$TMP_DIR/ctx_${RUN_IDX}.json"

    echo "  [3/$TOTAL_STEPS]${RUN_TAG} Prose generation (deterministic + creative, $MAT_GEN tokens)..." >&2
    run_gen_step "Step [3/$TOTAL_STEPS]${RUN_TAG} (prose deterministic)" \
      "$TMP_DIR/prose_greedy_${RUN_IDX}.log" \
      "$STRATA_BIN" "$RUN_CMD" "${RUN_COMMON[@]}" \
      --prompt "$PROSE_PROMPT" --greedy --verify-k0

    run_gen_step "Step [3/$TOTAL_STEPS]${RUN_TAG} (prose creative Standard K=0)" \
      "$TMP_DIR/prose_sampled_k0_${RUN_IDX}.log" \
      "$STRATA_BIN" "$RUN_CMD" "${RUN_COMMON_K0[@]}" \
      --prompt "$PROSE_PROMPT" "${SAMPLE_ARGS[@]}"

    run_gen_step "Step [3/$TOTAL_STEPS]${RUN_TAG} (prose creative Turbo)" \
      "$TMP_DIR/prose_sampled_${RUN_IDX}.log" \
      "$STRATA_BIN" "$RUN_CMD" "${RUN_COMMON[@]}" \
      --prompt "$PROSE_PROMPT" "${SAMPLE_ARGS[@]}"

    echo "  [4/$TOTAL_STEPS]${RUN_TAG} Code generation (deterministic + creative, $MAT_GEN tokens)..." >&2
    run_gen_step "Step [4/$TOTAL_STEPS]${RUN_TAG} (code deterministic)" \
      "$TMP_DIR/code_greedy_${RUN_IDX}.log" \
      "$STRATA_BIN" "$RUN_CMD" "${RUN_COMMON[@]}" \
      --prompt "$CODE_PROMPT" --greedy --verify-k0

    run_gen_step "Step [4/$TOTAL_STEPS]${RUN_TAG} (code creative Standard K=0)" \
      "$TMP_DIR/code_sampled_k0_${RUN_IDX}.log" \
      "$STRATA_BIN" "$RUN_CMD" "${RUN_COMMON_K0[@]}" \
      --prompt "$CODE_PROMPT" "${SAMPLE_ARGS[@]}"

    run_gen_step "Step [4/$TOTAL_STEPS]${RUN_TAG} (code creative Turbo)" \
      "$TMP_DIR/code_sampled_${RUN_IDX}.log" \
      "$STRATA_BIN" "$RUN_CMD" "${RUN_COMMON[@]}" \
      --prompt "$CODE_PROMPT" "${SAMPLE_ARGS[@]}"

    if (( RUN_IDX < RUNS && COOLDOWN_SEC > 0 )); then
      echo "  [cooldown] Sleeping ${COOLDOWN_SEC}s before run $((RUN_IDX + 1))/$RUNS..." >&2
      sleep "$COOLDOWN_SEC"
    fi
  done
fi

if [[ "$SWARM_ENABLED" -eq 1 ]]; then
  SWARM_STEP="$TOTAL_STEPS"
  echo "  [$SWARM_STEP/$TOTAL_STEPS] Swarm & concurrent sessions ($SWARM_BATCHES agents, $SWARM_TOKENS tok/agent, ${SWARM_CTX} ctx)..." >&2

  if [[ "$SWARM_ONLY" -eq 1 ]]; then
    # Measure exact solo K=0 baseline (256 toks so GPU boost clocks settle, matching full APEX)
    run_gen_step "Step [$SWARM_STEP/$TOTAL_STEPS] (swarm solo K=0 baseline)" \
      "$TMP_DIR/swarm_k0_ref.log" \
      "$STRATA_BIN" "$RUN_CMD" \
      --bundle "$BUNDLE_DIR" "${EXTRA_RUN_ARGS[@]}" --no-think --effort medium \
      --draft-k 0 --max-tokens 256 --max-seq-len 8192 --no-eos-stop \
      --prompt "$CODE_PROMPT" --greedy
  fi

  SWARM_RC=0
  caffeinate -dimsu python3 -I - "$STRATA_BIN" "$TMP_DIR" "$BUNDLE_DIR" "$SWARM_BATCHES" "$SWARM_TOKENS" "$SWARM_CTX" "$DRAFT_K" "$KV_CACHE" "$INT8_PREFILL" "$RUNS" "$COOLDOWN_SEC" << 'SWARMPY' || SWARM_RC=$?
import concurrent.futures, http.client, json, os, re, socket, statistics, subprocess, sys, threading, time

(strata_bin, tmp_dir, bundle_dir, batches_s, gen_toks_s, swarm_ctx_s,
 draft_k_s, kv_cache, int8_prefill_s, runs_s, cooldown_s) = sys.argv[1:12]
batches      = [int(x) for x in batches_s.split(",") if x.strip()]
gen_toks     = int(gen_toks_s)
swarm_ctx    = int(swarm_ctx_s)
draft_k      = int(draft_k_s)
int8_prefill = (int8_prefill_s == "1")
runs         = max(1, int(runs_s))
cooldown_sec = max(0, int(cooldown_s))
max_batch    = max(max(batches), 2)

PROMPTS = [
    "Write a complete, production-ready TypeScript MinHeap<T> priority queue class with custom comparator, push, pop, peek, updateKey, heapify, iteration, and unit tests. Output code only.",
    "Write a complete, production-ready Python token-bucket rate limiter module with async support, burst capacity, per-key buckets, refill stats, cleanup, and tests. Output code only.",
    "Write a complete, production-ready Rust generic RingBuffer<T> with push, pop, overwrite mode, drain iterator, slice views, Display impl, and unit tests. Output code only.",
    "Write a complete, production-ready Go generic thread-safe TTL cache with sharded locks, Get, Set, Delete, background janitor, metrics, and unit tests. Output code only.",
    "Write a complete, production-ready C++20 intrusive doubly-linked list template with bidirectional iterators, splice, push_front, pop_back, erase, and tests. Output code only.",
    "Write a complete, production-ready TypeScript Trie autocomplete engine supporting weighted suggestions, insert, delete, fuzzy prefix search, serialization, and tests. Output code only.",
    "Write a complete, production-ready Python DisjointSet (Union-Find) class with path compression, union by rank, component sizes, rollback snapshots, and tests. Output code only.",
    "Write a complete, production-ready Rust BloomFilter with optimal bit sizing, double hashing, merge, false-positive rate estimation, serialization, and tests. Output code only.",
    "Write a complete, production-ready Go consistent hash ring with virtual nodes, weighted backends, AddNode, RemoveNode, GetN, thread safety, and unit tests. Output code only.",
    "Write a complete, production-ready C++20 lock-free SPSC ring queue using cache-line padding, std::atomic acquire/release semantics, batch ops, and tests. Output code only.",
    "Write a complete, production-ready TypeScript async Semaphore and RWLock module supporting timeouts, AbortSignal cancellation, FIFO fairness, and tests. Output code only.",
    "Write a complete, production-ready Python in-memory B-Tree map supporting insert, delete, node split/merge, range scan iterator, validation, and tests. Output code only.",
    "Write a complete, production-ready Rust generational Arena<T> allocator returning typed Handle<T>, supporting insert, remove, get_mut, iteration, and unit tests. Output code only.",
    "Write a complete, production-ready Go circuit breaker with Closed/Open/HalfOpen transitions, sliding error window, hooks, metrics snapshot, and unit tests. Output code only.",
    "Write a complete, production-ready C++20 CRC32C slicing-by-8 implementation with constexpr lookup tables, incremental updates, combine, and unit tests. Output code only.",
    "Write a complete, production-ready TypeScript Myers line-diff engine generating unified diff hunks, reverse patches, context lines, and unit tests. Output code only.",
]

def get_swap_used_gib():
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"used\s*=\s*([0-9.]+)([MGT])", out)
        if not m:
            return 0.0
        val, unit = float(m.group(1)), m.group(2)
        if unit == "M":
            return val / 1024.0
        if unit == "T":
            return val * 1024.0
        return val
    except Exception:
        return 0.0

def text_loop_warn(body):
    toks = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", body)
    n = len(toks)
    if n < 48:
        return False
    n8 = n - 7
    dup8 = 1.0 - (len({tuple(toks[i:i + 8]) for i in range(n8)}) / float(n8))
    return dup8 > 0.35

def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def http_get_json(port, path, timeout=5.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        if resp.status != 200:
            return None
        return json.loads(body)
    except Exception:
        return None
    finally:
        conn.close()

def run_sse_request(port, prompt, max_tokens, barrier=None, t_start_box=None):
    payload = json.dumps({
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "enable_thinking": False,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    if barrier is not None:
        idx = barrier.wait()
        if idx == 0 and t_start_box is not None:
            t_start_box[0] = time.perf_counter()
    t_req = time.perf_counter()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    chunks = []
    chunk_times = []
    t_first = None
    t_last = None
    comp_toks = 0
    cached_toks = 0
    ok = False
    err = ""
    try:
        conn.request("POST", "/v1/chat/completions", body=payload, headers={
            "Content-Type": "application/json",
            "Connection": "close",
        })
        resp = conn.getresponse()
        if resp.status != 200:
            err = f"HTTP {resp.status}"
            resp.read()
        else:
            while True:
                raw = resp.fp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_s = line[5:].strip()
                if data_s == "[DONE]":
                    if t_last is None:
                        t_last = time.perf_counter()
                    ok = True
                    break
                try:
                    ev = json.loads(data_s)
                except Exception:
                    continue
                if "usage" in ev and isinstance(ev["usage"], dict):
                    comp_toks = int(ev["usage"].get("completion_tokens", comp_toks) or comp_toks)
                    ptd = ev["usage"].get("prompt_tokens_details") or {}
                    cached_toks = int(ptd.get("cached_tokens", cached_toks) or cached_toks)
                choices = ev.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    txt = delta.get("content") or ""
                    if txt:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        chunks.append(txt)
                        chunk_times.append(now)
            if not ok and chunks:
                ok = True
    except Exception as ex:
        err = str(ex)
    finally:
        conn.close()
    text = "".join(chunks)
    if comp_toks <= 0:
        comp_toks = len(chunks)
    if t_first is None:
        t_first = t_req
    if t_last is None or t_last <= t_first:
        t_last = time.perf_counter()
    return {
        "ok": ok,
        "err": err,
        "text": text,
        "comp_toks": comp_toks,
        "cached_toks": cached_toks,
        "chunk_times": chunk_times,
        "t_req": t_req,
        "t_first": t_first,
        "t_last": t_last,
        "loop_warn": text_loop_warn(text),
    }

port = pick_free_port()
cmd = [
    strata_bin, "serve",
    "--model", "inkling2",
    "--bundle", bundle_dir,
    "--host", "127.0.0.1",
    "--port", str(port),
    "--max-batch", str(max_batch),
    "--max-seq-len", str(swarm_ctx),
    "--draft-k", str(draft_k),
    "--no-think",
]
if kv_cache:
    cmd += ["--kv-cache", kv_cache]
if int8_prefill:
    cmd += ["--int8-prefill"]

srv_log_path = os.path.join(tmp_dir, "swarm_serve.log")
srv_log = open(srv_log_path, "w")
swap_before = get_swap_used_gib()
proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT)

try:
    health = None
    deadline = time.time() + 90.0
    while time.time() < deadline:
        if proc.poll() is not None:
            srv_log.flush()
            tail = open(srv_log_path).read()[-1200:]
            raise RuntimeError(f"strata_cli serve exited early (rc={proc.returncode}):\n{tail}")
        health = http_get_json(port, "/health", timeout=1.0)
        if health and health.get("status") == "ok":
            break
        time.sleep(0.2)
    if not health:
        raise RuntimeError("Timed out waiting for ephemeral strata_cli serve /health")

    # Warm up 1 short request before timed waves
    _ = run_sse_request(port, "Reply with OK.", 8)

    iso_target_c = None
    for c_cand in batches:
        if 2 <= c_cand <= 8:
            iso_target_c = c_cand
            break
    if iso_target_c is None:
        for c_cand in batches:
            if c_cand >= 2:
                iso_target_c = c_cand
                break

    iso_prompts = None
    iso_outputs = None
    samples_by_c = {c_val: [] for c_val in batches}

    for run_idx in range(1, runs + 1):
        for c_val in batches:
            prompts_c = [
                f"[Session R{run_idx}-C{c_val}-A{i}] {PROMPTS[i % len(PROMPTS)]}"
                for i in range(c_val)
            ]
            barrier = threading.Barrier(c_val)
            t_start_box = [0.0]
            stop_poll = threading.Event()
            peak_active = [0]

            def poll_active():
                while not stop_poll.is_set():
                    h = http_get_json(port, "/health", timeout=0.5)
                    if h:
                        ab = int(h.get("active_batch", 0) or 0)
                        if ab > peak_active[0]:
                            peak_active[0] = ab
                    stop_poll.wait(0.035)

            pt = threading.Thread(target=poll_active, daemon=True)
            pt.start()
            t_fallback = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c_val) as ex:
                futs = [
                    ex.submit(run_sse_request, port, prompts_c[i], gen_toks, barrier, t_start_box)
                    for i in range(c_val)
                ]
                res_list = [f.result() for f in futs]
            stop_poll.set()
            pt.join(timeout=1.0)

            h_after = http_get_json(port, "/health", timeout=2.0) or health
            t_wall_start = t_start_box[0] if t_start_box[0] > 0.0 else t_fallback
            t_first_min = min(r["t_first"] for r in res_list)
            t_first_max = max(r["t_first"] for r in res_list)
            t_last_min  = min(r["t_last"] for r in res_list)
            t_last_max  = max(r["t_last"] for r in res_list)
            total_toks  = sum(r["comp_toks"] for r in res_list)
            decode_toks = sum(max(1, r["comp_toks"] - 1) for r in res_list)
            e2e_wall_s  = max(1e-6, t_last_max - t_wall_start)

            # Exact full-concurrency steady-state decode window `[max(t_first), min(t_last)]`
            # (excludes sequential prefill stagger while counting only tokens delivered during full B=c overlap)
            if c_val == 1:
                dec_agg_tps = decode_toks / max(1e-6, t_last_max - t_first_min)
            elif (t_last_min - t_first_max) >= 0.25:
                win_toks = 0.0
                for r in res_list:
                    cts = r["chunk_times"]
                    if len(cts) >= 2:
                        tok_per_ch = float(max(1, r["comp_toks"] - 1)) / float(len(cts) - 1)
                        in_cnt = sum(1 for ts in cts[1:] if t_first_max < ts <= t_last_min)
                        win_toks += in_cnt * tok_per_ch
                dec_agg_tps = win_toks / (t_last_min - t_first_max) if win_toks > 0.0 else (decode_toks / max(1e-6, t_last_max - t_first_min))
            else:
                dec_agg_tps = decode_toks / max(1e-6, t_last_max - t_first_min)
            e2e_agg_tps = total_toks / e2e_wall_s
            ok_cnt      = sum(1 for r in res_list if r["ok"] and r["comp_toks"] > 0)
            lw_any      = any(r["loop_warn"] for r in res_list)
            swap_delta  = max(0.0, get_swap_used_gib() - swap_before)

            samples_by_c[c_val].append({
                "c": c_val,
                "dec_agg_tps": dec_agg_tps,
                "per_agent_tps": dec_agg_tps / float(c_val),
                "e2e_agg_tps": e2e_agg_tps,
                "uma_gib": float(h_after.get("total_uma_gib", 0.0) or 0.0),
                "majflt": int(h_after.get("steady_majflt", 0) or 0),
                "swap_gib": swap_delta,
                "ok_streams": ok_cnt,
                "peak_active": max(peak_active[0], 1),
                "loop_warn": lw_any,
            })

            if run_idx == runs and iso_target_c is not None and c_val == iso_target_c:
                iso_prompts = prompts_c
                iso_outputs = [r["text"] for r in res_list]
        if run_idx < runs and cooldown_sec > 0:
            time.sleep(cooldown_sec)

    # Live bit-exact cross-stream isolation check: replay `iso_target_c` prompts solo at c=1
    iso_exact_cnt = 0
    iso_total_cnt = 0
    if iso_prompts and iso_outputs:
        iso_total_cnt = len(iso_prompts)
        for p_txt, conc_out in zip(iso_prompts, iso_outputs):
            solo_r = run_sse_request(port, p_txt, gen_toks)
            if solo_r["ok"] and solo_r["text"] == conc_out:
                iso_exact_cnt += 1

    agg_rows = []
    for c_val in batches:
        slist = samples_by_c[c_val]
        dec_vals = [x["dec_agg_tps"] for x in slist]
        e2e_vals = [x["e2e_agg_tps"] for x in slist]
        row = {
            "c": c_val,
            "dec_agg_tps": statistics.median(dec_vals),
            "dec_agg_min": min(dec_vals),
            "dec_agg_max": max(dec_vals),
            "per_agent_tps": statistics.median([x["per_agent_tps"] for x in slist]),
            "e2e_agg_tps": statistics.median(e2e_vals),
            "e2e_agg_min": min(e2e_vals),
            "e2e_agg_max": max(e2e_vals),
            "uma_gib": statistics.median([x["uma_gib"] for x in slist]),
            "majflt": max([x["majflt"] for x in slist]),
            "swap_gib": max([x["swap_gib"] for x in slist]),
            "ok_streams": min([x["ok_streams"] for x in slist]),
            "peak_active": max([x["peak_active"] for x in slist]),
            "loop_warn": any(x["loop_warn"] for x in slist),
            "iso_verified": (iso_target_c is not None and c_val == iso_target_c),
            "iso_exact": (iso_exact_cnt == iso_total_cnt and iso_total_cnt > 0) if (iso_target_c == c_val) else None,
            "iso_exact_cnt": iso_exact_cnt if (iso_target_c == c_val) else 0,
            "iso_total_cnt": iso_total_cnt if (iso_target_c == c_val) else 0,
        }
        agg_rows.append(row)

    out_payload = {
        "max_batch": max_batch,
        "swarm_ctx": swarm_ctx,
        "swarm_tokens": gen_toks,
        "kv_cache": health.get("kv_cache", kv_cache or "f16"),
        "iso_target_c": iso_target_c,
        "iso_exact_cnt": iso_exact_cnt,
        "iso_total_cnt": iso_total_cnt,
        "rows": agg_rows,
    }
    with open(os.path.join(tmp_dir, "swarm.json"), "w") as f:
        json.dump(out_payload, f, indent=2)
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3.0)
    srv_log.close()
SWARMPY
  if [[ "$SWARM_RC" -ne 0 ]]; then
    SCRIPT_FAILED=1
    echo "[STRATA // APEX] ERROR: Swarm benchmark step failed (exit $SWARM_RC)." >&2
    if [[ -f "$TMP_DIR/swarm_serve.log" ]]; then
      echo "  ── Last lines of $TMP_DIR/swarm_serve.log ──" >&2
      tail -n 25 "$TMP_DIR/swarm_serve.log" >&2 || true
    fi
    exit "$SWARM_RC"
  fi
fi

USE_COLOR=0
if [[ "$COLOR_MODE" == "always" ]] || { [[ "$COLOR_MODE" == "auto" ]] && [[ -t 1 ]]; }; then
  USE_COLOR=1
fi

if [[ "$CLEAR_SCREEN" -eq 1 && -t 1 ]]; then
  printf '\033[2J\033[H'
fi

RENDER_RC=0
python3 -I - "$TMP_DIR" "$MODEL" "$DRAFT_K" "$THINK_FLAG" "$MAT_GEN" "$USE_COLOR" "$OUT_MD" "$RUNS" "$INT8_PREFILL" "$SWARM_ENABLED" "$SWARM_ONLY" "$COOLDOWN_SEC" "$STRATA_BIN" "$BIN_MTIME" "$BIN_STALE" << 'PYEOF' || RENDER_RC=$?
import json, os, re, statistics, subprocess, sys

(tmp_dir, model, draft_k_s, think_flag, mat_gen_s,
 use_color_s, out_md, runs_s, int8_prefill_s, swarm_en_s, swarm_only_s,
 cooldown_s, strata_bin, bin_mtime, bin_stale_s) = sys.argv[1:16]
draft_k       = int(draft_k_s)
mat_gen       = int(mat_gen_s)
use_color     = (use_color_s == "1")
runs          = max(1, int(runs_s))
int8_prefill  = (int8_prefill_s == "1")
swarm_enabled = (swarm_en_s == "1")
swarm_only    = (swarm_only_s == "1")
cooldown_sec  = max(0, int(cooldown_s))
bin_stale     = (bin_stale_s == "1")
GIB_TO_GB     = (1024.0 ** 3) / 1e9  # Convert binary GiB reported by engine to decimal GB (10^9 B)
ANSI_RE       = re.compile(r"\033\[[0-9;]*m")

# ── ANSI Palette (subtle, high-contrast on dark & light terminals) ────────────
def c(code, text):
    return f"\033[{code}m{text}\033[0m" if use_color else text

BOLD   = "1"
DIM    = "2"
GRAY   = "90"
CYAN   = "38;5;45"
BLUE   = "38;5;39"
ICE    = "38;5;51"
GOLD   = "1;38;5;220"
GREEN  = "1;38;5;42"
RED    = "1;38;5;196"
WHITE  = "1;97"

# ── Hardware Detection ────────────────────────────────────────────────────────
def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "Unknown"

os_name   = sh(["sw_vers", "-productName"])
os_ver    = sh(["sw_vers", "-productVersion"])
chip      = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
mem_gb    = int(sh(["sysctl", "-n", "hw.memsize"]) or "0") // (1024 ** 3)
hw_prof   = sh(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"])

def grep_prof(pattern, default="Unknown"):
    m = re.search(pattern, hw_prof)
    return m.group(1).strip() if m else default

model_name = grep_prof(r"Model Name:\s*(.+)", "Mac")
cpu_total  = grep_prof(r"Total Number of Cores:\s*(\d+)", "?")
gpu_cores  = grep_prof(r"Chipset Model:.*?\n\s*Type:.*?\n\s*Bus:.*?\n\s*Total Number of Cores:\s*(\d+)")
if gpu_cores == "Unknown":
    ms = re.findall(r"Total Number of Cores:\s*(\d+)\s*$", hw_prof, re.M)
    if ms:
        gpu_cores = ms[-1]
metal_ver  = grep_prof(r"Metal Support:\s*(.+)", "Metal 4")

# Modeled Apple Silicon ceilings by chip + GPU core count:
# - M5 sustained NAX compute ceilings from docs/STRATA_M5_ARCHITECTURE_REPORT.md:51
#   (40-core M5 Max sustained real-world: 57-60 TFLOPS FP16, 110-120 TOPS INT8 -> 60.0 / 120.0;
#    scaled linearly by GPU core count across M5 / M5 Pro / M5 Max 32c / M5 Ultra).
# - Unified memory bandwidth ceilings (decimal GB/s) from Apple published chip specifications
#   (e.g., 614 GB/s for 40c M5 Max, 460 GB/s for 32c M5 Max, 307 GB/s for M5 Pro, 153.6 GB/s for M5).
def detect_chip_ceilings(chip_name, gpu_c_str, use_int8):
    cl = chip_name.lower()
    gc = int(gpu_c_str) if gpu_c_str.isdigit() else 0
    if "m5 ultra" in cl:
        scale = (gc / 80.0) if gc > 0 else 1.0
        return ("M5 Ultra", 1228.0, (240.0 if use_int8 else 120.0) * scale)
    if "m5 max" in cl:
        scale = (gc / 40.0) if gc in (32, 40) else 1.0
        bw = 460.0 if gc == 32 else 614.0
        return ("M5 Max", bw, (120.0 if use_int8 else 60.0) * scale)
    if "m5 pro" in cl:
        scale = (gc / 20.0) if gc > 0 else 1.0
        return ("M5 Pro", 307.0, (60.0 if use_int8 else 30.0) * scale)
    if "m5" in cl:
        scale = (gc / 10.0) if gc > 0 else 1.0
        return ("M5", 153.6, (30.0 if use_int8 else 15.0) * scale)
    if "m4 max" in cl:
        bw = 410.0 if gc == 32 else 546.0
        scale = (gc / 40.0) if gc > 0 else 1.0
        return ("M4 Max", bw, 36.0 * scale)
    if "m4 pro" in cl:
        scale = (gc / 20.0) if gc > 0 else 1.0
        return ("M4 Pro", 273.0, 18.0 * scale)
    if "m4" in cl:
        return ("M4", 120.0, 9.0)
    if "m3 ultra" in cl:
        return ("M3 Ultra", 819.0, 54.0)
    if "m3 max" in cl:
        bw = 300.0 if gc == 30 else 400.0
        return ("M3 Max", bw, 27.0)
    if "m2 ultra" in cl:
        return ("M2 Ultra", 800.0, 27.2)
    if "m2 max" in cl:
        return ("M2 Max", 400.0, 13.6)
    if "m1 ultra" in cl:
        return ("M1 Ultra", 800.0, 21.0)
    if "m1 max" in cl:
        return ("M1 Max", 400.0, 10.4)
    short = chip_name.replace("Apple ", "").strip() or "Apple Silicon"
    return (short, 400.0, 32.0)

CHIP_SHORT, M5_MAX_DRAM_GBPS, M5_MAX_PP_TOPS = detect_chip_ceilings(chip, gpu_cores, int8_prefill)

# ── Shingle-Loop Output Sanity Check (Mirrors C++ Token-ID Detector) ─────────
def text_shingle_loop_check(txt):
    # Extract generated text between startup banner (second === line) and telemetry footer (--- line)
    parts = re.split(r"^={20,}\s*$", txt, maxsplit=2, flags=re.M)
    body = parts[2] if len(parts) >= 3 else txt
    body = re.split(r"^-{20,}\s*$", body, maxsplit=1, flags=re.M)[0]
    toks = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", body)
    n = len(toks)
    if n < 48:
        return (0.0, 0.0, False)
    n8 = n - 7
    dup8 = 1.0 - (len({tuple(toks[i:i + 8]) for i in range(n8)}) / float(n8))
    n4 = n - 3
    dup4 = 1.0 - (len({tuple(toks[i:i + 4]) for i in range(n4)}) / float(n4))
    return (dup8, dup4, dup8 > 0.35)

# ── Parse Bundle & Multi-Run Benchmark Outputs ────────────────────────────────
inspect_txt = open(os.path.join(tmp_dir, "inspect.txt")).read()
def grep_insp(pat, default="?"):
    m = re.search(pat, inspect_txt)
    return m.group(1).strip() if m else default

def fmt_gib_to_gb(val_s):
    if not val_s or val_s == "?":
        return "?"
    try:
        return f"{float(val_s) * GIB_TO_GB:.1f}"
    except ValueError:
        return "?"

active_gb_per_tok = 0.0
m_act = re.search(r"Active Traffic\s*:\s*([0-9.]+)\s*GB/tok", inspect_txt)
if not m_act:
    m_act = re.search(r"Active\s+([0-9.]+)\s*GB/tok", inspect_txt)
if m_act:
    active_gb_per_tok = float(m_act.group(1))

def parse_run_log(path):
    txt = open(path).read()
    out = {}
    m = re.search(r"Prefill & TTFT\s*:\s*([0-9.]+)\s*tok/s\s*\|\s*TTFT\s*([0-9.]+)\s*ms\s*\((\d+)\s*prompt", txt)
    if m:
        out["pp_tps"], out["ttft_ms"], out["prompt_toks"] = float(m.group(1)), float(m.group(2)), int(m.group(3))
    m = re.search(r"Decode Throughput\s*:\s*([0-9.]+)\s*tok/s\s*\|\s*(\d+)\s*decode toks in\s*(\d+)\s*steps", txt)
    if m:
        out["tg_tps"], out["dec_toks"], out["dec_steps"] = float(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"(?:Modeled|Physical) DRAM BW\s*:\s*([0-9.]+)\s*GB/s (?:modeled|physical).*?\|\s*([0-9.]+)\s*GB/s", txt)
    if m:
        out["phys_bw"], out["eff_bw"] = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"(?:Modeled|Effective) DRAM BW\s*:\s*([0-9.]+)\s*GB/s eff\s*\(([0-9.]+)\s*GB/s (?:modeled|phys)", txt)
        if m:
            out["eff_bw"], out["phys_bw"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Total UMA\s*([0-9.]+)\s*GiB.*?Swap\s*\+([0-9.]+)\s*GiB\)\s*\|\s*Steady MajFlt=(\d+)", txt)
    if m:
        out["uma_gib"], out["swap_gib"], out["majflt"] = float(m.group(1)), float(m.group(2)), int(m.group(3))
    m = re.search(r"MTP K=\d+ vs K=0 Check\s*:\s*([A-Z0-9% \-]+?)\s*\((\d+) vs (\d+) tokens,\s*Speedup\s*([0-9.]+)x:\s*([0-9.]+) vs ([0-9.]+) tok/s\)", txt)
    if m:
        out["k0_match"] = ("BIT-EXACT" in m.group(1))
        out["speedup"]  = float(m.group(4))
        out["k0_tps"]   = float(m.group(6))
    m = re.search(r"Output Sanity\s*:\s*(OK|LOOP_WARNING)\s*\(dup8=([0-9.]+)%,\s*dup4=([0-9.]+)%", txt)
    c_loop_warn = False
    c_dup8 = 0.0
    if m:
        c_loop_warn = (m.group(1) == "LOOP_WARNING")
        c_dup8 = float(m.group(2)) / 100.0
    t_dup8, _, t_loop_warn = text_shingle_loop_check(txt)
    out["loop_warn"] = c_loop_warn or t_loop_warn
    out["dup8"] = max(c_dup8, t_dup8)
    return out

def calc_spread_pct(vals):
    if len(vals) < 2:
        return (0.0, vals[0] if vals else 0.0, vals[0] if vals else 0.0)
    med = statistics.median(vals)
    mn, mx = min(vals), max(vals)
    return (((mx - mn) / med * 100.0) if med > 0.0 else 0.0, mn, mx)

def aggregate_run_logs(prefix):
    samples = [parse_run_log(os.path.join(tmp_dir, f"{prefix}_{i}.log")) for i in range(1, runs + 1)]
    out = {"samples": samples}
    num_keys = ["pp_tps", "ttft_ms", "prompt_toks", "tg_tps", "dec_toks", "dec_steps",
                "phys_bw", "eff_bw", "uma_gib", "speedup", "k0_tps"]
    for k in num_keys:
        vals = [s[k] for s in samples if k in s]
        if vals:
            out[k] = statistics.median(vals)
            out[f"{k}_vals"] = vals
    swap_vals = [s.get("swap_gib", 0.0) for s in samples]
    out["swap_gib"] = max(swap_vals) if swap_vals else 0.0
    maj_vals = [s.get("majflt", 0) for s in samples]
    out["majflt"] = max(maj_vals) if maj_vals else 0
    k0m_vals = [s["k0_match"] for s in samples if "k0_match" in s]
    if k0m_vals:
        out["k0_match"] = all(k0m_vals)
    out["loop_warn"] = any(s.get("loop_warn", False) for s in samples)
    out["dup8"] = max((s.get("dup8", 0.0) for s in samples), default=0.0)
    return out

if not swarm_only:
    pg    = aggregate_run_logs("prose_greedy")
    ps_k0 = aggregate_run_logs("prose_sampled_k0")
    ps    = aggregate_run_logs("prose_sampled")
    cg    = aggregate_run_logs("code_greedy")
    cs_k0 = aggregate_run_logs("code_sampled_k0")
    cs    = aggregate_run_logs("code_sampled")
else:
    pg, ps_k0, ps, cg, cs_k0, cs = {}, {}, {}, {}, {}, {}

swarm_data = None
swarm_k0_ref = 0.0
if swarm_enabled and os.path.exists(os.path.join(tmp_dir, "swarm.json")):
    swarm_data = json.load(open(os.path.join(tmp_dir, "swarm.json")))
    k0_ref_path = os.path.join(tmp_dir, "swarm_k0_ref.log")
    if os.path.exists(k0_ref_path):
        k0_d = parse_run_log(k0_ref_path)
        swarm_k0_ref = float(k0_d.get("tg_tps", 0.0) or 0.0)

# ── Modeled Active Compute (2 ops/MAC + attention ops; Turbo scales by tok/s) ─
def prefill_tops(tok_s, ctx):
    L = max(1, int(ctx))
    if model == "inkling2":
        # 48 logical visits: attn proj (2.582) + 36x loop ctrl (0.170) + 48x top-8 MoE (6.144) = 8.895 GOPS/tok
        backbone = 8.895
        lm_head  = 0.922
        avg_swa  = (min(L, 1024) ** 2 / 2.0 + max(0, L - 1024) * 1024.0) / L
        swa_ops  = (40 * (4 * 24 * 128 + 2 * 24 * 16) * avg_swa) / 1e9
        glb_ops  = (8 * (4 * 24 * 128 * (L / 2.0) + 2 * 24 * 16 * avg_swa)) / 1e9
        gops_tok = backbone + swa_ops + glb_ops + (lm_head / L)
    elif model == "glm53":
        # GLM-5.3-Flash 45L (320B / 18B active):
        #   45L mHC (0.088) + 34L KDA proj/conv/rec (9.579) + 11L DSA NoPE MLA proj (2.607)
        # + 3L dense L0-2 (0.906) + 42L MoE top-8 + shared + router (19.124) = 32.305 GOPS/tok
        backbone = 32.305
        lm_head  = 1.269
        avg_kv   = (min(L, 2048) ** 2 / 2.0 + max(0, L - 2048) * 2048.0) / L
        mla_ops  = (4 * 11 * 64 * 512 * avg_kv) / 1e9
        sp_frac  = max(0, L - 2048) / L
        idx_ops  = sp_frac * (0.141 + (11 * 32 * 128 * 2 * ((L + 2048) / 2.0) / 4.0) / 1e9)
        gops_tok = backbone + mla_ops + idx_ops + (lm_head / L)
    else:
        # Qwen3.8-Flash-Next 48L: HC (1.279) + 36L DeltaNet (4.340) + 12L QSA proj (1.235) + 48L MoE+sh (5.316) = 12.170 GOPS/tok
        backbone = 12.170
        lm_head  = 1.271
        avg_kv   = (min(L, 2048) ** 2 / 2.0 + max(0, L - 2048) * 2048.0) / L
        qsa_ops  = (4 * 12 * 24 * 256 * avg_kv) / 1e9
        idx_ops  = (max(0, L - 2048) / L) * (12 * 4 * 128 * 2 * ((L + 2048) / 2.0) / 4.0) / 1e9
        gops_tok = backbone + qsa_ops + idx_ops + (lm_head / L)
    return tok_s * gops_tok / 1000.0

def decode_tops(tok_s, prompt_toks, gen_toks):
    avg_pos = max(1.0, float(prompt_toks) + float(gen_toks) / 2.0)
    if model == "inkling2":
        backbone = 8.895 + 0.922
        swa_ops  = (40 * (4 * 24 * 128 + 2 * 24 * 16) * min(avg_pos, 1024.0)) / 1e9
        glb_ops  = (8 * (4 * 24 * 128 * avg_pos + 2 * 24 * 16 * min(avg_pos, 1024.0))) / 1e9
        gops_tok = backbone + swa_ops + glb_ops
    elif model == "glm53":
        backbone = 32.305 + 1.269
        mla_ops  = (4 * 11 * 64 * 512 * min(avg_pos, 2048.0)) / 1e9
        idx_ops  = (0.141 + (11 * 32 * 128 * 2 * (avg_pos / 4.0)) / 1e9) if avg_pos > 2048.0 else 0.0
        gops_tok = backbone + mla_ops + idx_ops
    else:
        backbone = 12.170 + 1.271
        qsa_ops  = (4 * 12 * 24 * 256 * min(avg_pos, 2048.0)) / 1e9
        gops_tok = backbone + qsa_ops
    return tok_s * gops_tok / 1000.0

# ── Model Summary Line (Decimal GB, "?" on missing fields, <= 80 chars) ───────
int8_tag = " · INT8 Prefill" if int8_prefill else ""
if model == "inkling2":
    uma_gib  = grep_insp(r"UMA File\s*:\s*\S+\s*\(([0-9.]+)\s*GiB", "?")
    uma_gb_s = fmt_gib_to_gb(uma_gib)
    model_line = f"Inkling-2-Mini (26B · 4.9B active) · 4-bit/8-bit · {uma_gb_s} GB RAM{int8_tag}"
    md_model_title = "Inkling-2-Mini (26B)" + (" — INT8 Prefill" if int8_prefill else "")
elif model == "glm53":
    ram_total = grep_insp(r"Total Mapped UMA\s*:\s*([0-9.]+)\s*GiB", "")
    if ram_total:
        ram_gb_s = fmt_gib_to_gb(ram_total)
        model_line = f"GLM-5.3-Flash (320B · 18B active) · E8S-18 / Q4 · {ram_gb_s} GB RAM{int8_tag}"
    else:
        sp_gb_s = fmt_gib_to_gb(grep_insp(r"spine\.strata[^:]*:\s*([0-9.]+)\s*GiB", "?"))
        nv_gb_s = fmt_gib_to_gb(grep_insp(r"experts_nvme[^:]*:\s*([0-9.]+)\s*GiB", "?"))
        if int8_prefill:
            model_line = f"GLM-5.3-Flash (320B · 18B) · 4-bit · {sp_gb_s} GB (+{nv_gb_s} GB SSD) · INT8 Prefill"
        else:
            model_line = f"GLM-5.3-Flash (320B · 18B active) · 4-bit · {sp_gb_s} GB RAM (+{nv_gb_s} GB SSD)"
    md_model_title = "GLM-5.3-Flash (320B)" + (" — INT8 Prefill" if int8_prefill else "")
else:
    p0  = grep_insp(r"UMA Part 0[^:]*:\s*([0-9.]+)\s*GiB", "?")
    p1  = grep_insp(r"UMA Part 1[^:]*:\s*([0-9.]+)\s*GiB", "?")
    ple = grep_insp(r"NVMe PLE Store[^:]*:\s*([0-9.]+)\s*GiB", "?")
    ram_gb_s = f"{(float(p0) + float(p1)) * GIB_TO_GB:.1f}" if (p0 != "?" and p1 != "?") else "?"
    ple_gb_s = fmt_gib_to_gb(ple)
    if int8_prefill:
        model_line = f"Qwen3.8-Flash-Next (180B · 6.7B) · Q4/Q8 · {ram_gb_s} GB (+{ple_gb_s} GB SSD) · INT8 Prefill"
    else:
        model_line = f"Qwen3.8-Flash-Next (180B · 6.7B act) · Q4/Q8 · {ram_gb_s} GB RAM (+{ple_gb_s} GB SSD)"
    md_model_title = "Qwen3.8-Flash-Next (180B)" + (" — INT8 Prefill" if int8_prefill else "")

# ── Build Context Rows (True Median Across Runs) & Compute Paired Peaks ──────
ctx_samples = {}
if not swarm_only:
    for i in range(1, runs + 1):
        ctx_data = json.load(open(os.path.join(tmp_dir, f"ctx_{i}.json")))
        for r in ctx_data.get("results", []):
            ctx_samples.setdefault((r["context"], r["draft_k"]), []).append(r)

by_ctx = {}
for (ctx, dk_val), slist in ctx_samples.items():
    pp_vals = [x["prefill_tok_s"] for x in slist]
    tg_vals = [x["decode_tok_s"] for x in slist]
    agg = {
        "context": ctx,
        "draft_k": dk_val,
        "prefill_tok_s": statistics.median(pp_vals),
        "pp_vals": pp_vals,
        "ttft_ms": statistics.median([x["ttft_ms"] for x in slist]),
        "decode_tok_s": statistics.median(tg_vals),
        "tg_vals": tg_vals,
        "phys_dram_gbps": statistics.median([x.get("phys_dram_gbps", 0.0) for x in slist]),
        "eff_dram_gbps": statistics.median([x.get("eff_dram_gbps", 0.0) for x in slist]),
        "total_uma_gib": statistics.median([x["total_uma_gib"] for x in slist]),
        "swap_delta_gib": max([x.get("swap_delta_gib", 0.0) for x in slist]),
        "steady_majflt": max([int(x.get("steady_majflt", 0) or 0) for x in slist]),
        "needle_pass": all(bool(x.get("needle_pass", False)) for x in slist),
        "exact_k0_match": all(bool(x.get("exact_k0_match", True)) for x in slist),
        "loop_warning": any(bool(x.get("loop_warning", False)) for x in slist),
        "dup_8gram_ratio": max([x.get("dup_8gram_ratio", 0.0) for x in slist]),
    }
    by_ctx.setdefault(ctx, {})[dk_val] = agg

ctx_rows = []
peak_pp      = 0.0
peak_pp_tops = 0.0
peak_k0_dec  = 0.0
max_std_bw   = 0.0
max_swap     = 0.0
max_majflt   = 0
all_exact    = True
all_recall   = True
loop_warnings  = []
pp_spread_list = []
tg_spread_list = []
high_spreads   = []

for ctx in sorted(by_ctx.keys()):
    r0  = by_ctx[ctx].get(0)
    rk  = by_ctx[ctx].get(draft_k)
    ref = rk or r0
    pp_src  = (r0 or ref)
    pp_tps  = pp_src["prefill_tok_s"]
    ttft_ms = pp_src["ttft_ms"]
    tops    = prefill_tops(pp_tps, ctx)
    d0      = r0["decode_tok_s"] if r0 else 0.0
    dk      = rk["decode_tok_s"] if rk else 0.0
    d0_bw   = (r0.get("phys_dram_gbps", 0.0) or r0.get("eff_dram_gbps", 0.0)) if r0 else 0.0
    dk_bw   = rk.get("phys_dram_gbps", 0.0) if rk else 0.0
    mem_gib = ref["total_uma_gib"]
    swap    = max(r0.get("swap_delta_gib", 0.0) if r0 else 0.0,
                  rk.get("swap_delta_gib", 0.0) if rk else 0.0)
    flt     = max(r0.get("steady_majflt", 0) if r0 else 0,
                  rk.get("steady_majflt", 0) if rk else 0)
    ok_rec  = bool(ref.get("needle_pass", False))
    ok_ex   = bool(rk.get("exact_k0_match", True)) if rk else True
    row_lw  = (bool(r0.get("loop_warning", False)) if r0 else False) or \
              (bool(rk.get("loop_warning", False)) if rk else False)
    if row_lw:
        loop_warnings.append(f"ctx={ctx}")

    if runs > 1:
        sp_pp, mn_pp, mx_pp = calc_spread_pct(pp_src.get("pp_vals", []))
        pp_spread_list.append((sp_pp, f"ctx {ctx:,}", mn_pp, mx_pp))
        if sp_pp > 10.0:
            high_spreads.append(f"ctx={ctx:,} prefill {sp_pp:.1f}% ({mn_pp:,.0f}..{mx_pp:,.0f} t/s)")
        if r0:
            sp_d0, mn_d0, mx_d0 = calc_spread_pct(r0.get("tg_vals", []))
            tg_spread_list.append((sp_d0, f"ctx {ctx:,} Std", mn_d0, mx_d0))
            if sp_d0 > 10.0:
                high_spreads.append(f"ctx={ctx:,} Standard {sp_d0:.1f}% ({mn_d0:.1f}..{mx_d0:.1f} t/s)")
        if rk:
            sp_dk, mn_dk, mx_dk = calc_spread_pct(rk.get("tg_vals", []))
            tg_spread_list.append((sp_dk, f"ctx {ctx:,} Turbo", mn_dk, mx_dk))
            if sp_dk > 10.0:
                high_spreads.append(f"ctx={ctx:,} Turbo {sp_dk:.1f}% ({mn_dk:.1f}..{mx_dk:.1f} t/s)")

    if d0_bw > max_std_bw:
        max_std_bw = d0_bw
    if pp_tps > peak_pp:
        peak_pp      = pp_tps
        peak_pp_tops = tops
    if d0 > peak_k0_dec:
        peak_k0_dec = d0
    if swap > max_swap:
        max_swap = swap
    if flt > max_majflt:
        max_majflt = flt
    if not ok_rec:
        all_recall = False
    if not ok_ex:
        all_exact = False

    ctx_rows.append({
        "ctx": ctx, "pp": pp_tps, "tops": tops, "ttft_ms": ttft_ms,
        "d0": d0, "dk": dk, "d0_bw": d0_bw, "dk_bw": dk_bw,
        "mem": mem_gib, "mem_gb": mem_gib * GIB_TO_GB,
        "recall": ok_rec, "exact": ok_ex, "loop_warn": row_lw,
    })

mat_specs = [
    ("Prose · Deterministic",  "Prose Det", pg, pg.get("k0_tps"),    pg.get("k0_tps_vals", []),   None,  True),
    ("Prose · Creative (0.6)", "Prose 0.6", ps, ps_k0.get("tg_tps"), ps_k0.get("tg_tps_vals", []), ps_k0, False),
    ("Code  · Deterministic",  "Code Det",  cg, cg.get("k0_tps"),    cg.get("k0_tps_vals", []),   None,  True),
    ("Code  · Creative (0.6)", "Code 0.6",  cs, cs_k0.get("tg_tps"), cs_k0.get("tg_tps_vals", []), cs_k0, False),
]

mat_rows = []
peak_dec      = 0.0
peak_dec_bw   = 0.0
peak_dec_src  = ""
if not swarm_only:
    for label, short_lbl, d, k0_med, k0_vals, k0_log_obj, is_greedy in mat_specs:
        k0  = float(k0_med or 0.0)
        dk  = float(d.get("tg_tps", 0.0) or 0.0)
        # B10: Compute Speedup strictly from the two displayed medians
        spd = (dk / k0) if (k0 > 0.0 and dk > 0.0) else 0.0
        dtp = decode_tops(dk, d.get("prompt_toks", 64), d.get("dec_toks", mat_gen))
        dk_bw = float(d.get("phys_bw", 0.0) or 0.0)
        if k0_log_obj is not None:
            k0_bw = float(k0_log_obj.get("phys_bw", 0.0) or (k0 * active_gb_per_tok if active_gb_per_tok > 0.0 else 0.0))
        else:
            k0_bw = k0 * active_gb_per_tok if active_gb_per_tok > 0.0 else 0.0
        if k0_bw > max_std_bw:
            max_std_bw = k0_bw

        lw = bool(d.get("loop_warn", False)) or (bool(k0_log_obj.get("loop_warn", False)) if k0_log_obj else False)
        if lw:
            loop_warnings.append(f"{label} (dup8={d.get('dup8', 0.0)*100:.1f}%)")

        # B2: Track peak real-world decode and pair it strictly with THAT SAME RUN's modeled bandwidth
        if dk >= k0 and dk > peak_dec:
            peak_dec     = dk
            peak_dec_bw  = dk_bw
            peak_dec_src = short_lbl
        elif k0 > dk and k0 > peak_dec:
            peak_dec     = k0
            peak_dec_bw  = k0_bw
            peak_dec_src = f"{short_lbl} Std"

        if k0 > peak_k0_dec:
            peak_k0_dec = k0
        for src_d in (d, k0_log_obj):
            if src_d:
                if src_d.get("swap_gib", 0.0) > max_swap:
                    max_swap = src_d["swap_gib"]
                if int(src_d.get("majflt", 0) or 0) > max_majflt:
                    max_majflt = int(src_d["majflt"])

        if runs > 1:
            if k0_vals:
                sp_k0, mn_k0, mx_k0 = calc_spread_pct(k0_vals)
                tg_spread_list.append((sp_k0, f"{short_lbl} Std", mn_k0, mx_k0))
                if sp_k0 > 10.0:
                    high_spreads.append(f"{short_lbl} Standard {sp_k0:.1f}% ({mn_k0:.1f}..{mx_k0:.1f} t/s)")
            dk_vals = d.get("tg_tps_vals", [])
            if dk_vals:
                sp_dk, mn_dk, mx_dk = calc_spread_pct(dk_vals)
                tg_spread_list.append((sp_dk, short_lbl, mn_dk, mx_dk))
                if sp_dk > 10.0:
                    high_spreads.append(f"{short_lbl} Turbo {sp_dk:.1f}% ({mn_dk:.1f}..{mx_dk:.1f} t/s)")

        if lw:
            ok  = False
            fid = "LOOP_WARNING"
        elif is_greedy:
            ok = bool(d.get("k0_match", False))
            if not ok:
                all_exact = False
            fid = ("Turbo=Standard" if int8_prefill else "100% Identical") if ok else "Mismatch"
        else:
            # B4: Sampled runs are not K=0-verified; never assert exactness
            ok  = None
            fid = "Not verified"
        mat_rows.append({
            "label": label, "k0": k0, "dk": dk, "spd": spd, "tops": dtp,
            "dk_bw": dk_bw, "k0_bw": k0_bw, "fid": fid, "ok": ok, "loop_warn": lw,
        })

if not swarm_only and peak_dec <= 0.0 and ctx_rows:
    for r in ctx_rows:
        if r["dk"] >= r["d0"] and r["dk"] > peak_dec:
            peak_dec     = r["dk"]
            peak_dec_bw  = r["dk_bw"]
            peak_dec_src = f"ctx {r['ctx']:,}"
        elif r["d0"] > peak_dec:
            peak_dec     = r["d0"]
            peak_dec_bw  = r["d0_bw"]
            peak_dec_src = f"ctx {r['ctx']:,} Std"

# ── Build Swarm / Concurrent Sessions Rows ────────────────────────────────────
swarm_rows = []
swarm_peak_tps   = 0.0
swarm_peak_e2e   = 0.0
swarm_peak_c     = 1
swarm_peak_per   = 0.0
swarm_peak_scale = 0.0
swarm_iso_ok     = True
swarm_all_clean  = True
swarm_k0_base    = cg.get("k0_tps") or pg.get("k0_tps") or peak_k0_dec or swarm_k0_ref
sw_spread_list   = []

if swarm_data:
    for sr in swarm_data.get("rows", []):
        c_val   = int(sr["c"])
        agg_tps = float(sr["dec_agg_tps"])
        per_tps = float(sr["per_agent_tps"])
        e2e_tps = float(sr["e2e_agg_tps"])
        uma_g   = float(sr["uma_gib"])
        swp_g   = float(sr.get("swap_gib", 0.0))
        flt_s   = int(sr.get("majflt", 0) or 0)
        ok_s    = int(sr.get("ok_streams", 0))
        lw_s    = bool(sr.get("loop_warn", False))
        if swp_g > max_swap:
            max_swap = swp_g
        if flt_s > max_majflt:
            max_majflt = flt_s
        if lw_s:
            loop_warnings.append(f"Swarm c={c_val}")
        if runs > 1 and "dec_agg_min" in sr and "dec_agg_max" in sr and agg_tps > 0.0:
            mn_s, mx_s = float(sr["dec_agg_min"]), float(sr["dec_agg_max"])
            sp_s = (mx_s - mn_s) / agg_tps * 100.0
            sw_spread_list.append((sp_s, f"Swarm {c_val}x", mn_s, mx_s))
            if sp_s > 10.0:
                high_spreads.append(f"Swarm {c_val}x Steady {sp_s:.1f}% ({mn_s:.1f}..{mx_s:.1f} t/s)")
        base = swarm_k0_base if swarm_k0_base > 0.0 else (agg_tps if c_val == 1 else 0.0)
        if c_val == 1 and swarm_k0_base <= 0.0:
            swarm_k0_base = agg_tps
            base = agg_tps
        scl = (agg_tps / base) if base > 0.0 else 0.0
        if agg_tps > swarm_peak_tps:
            swarm_peak_tps   = agg_tps
            swarm_peak_e2e   = e2e_tps
            swarm_peak_c     = c_val
            swarm_peak_per   = per_tps
            swarm_peak_scale = scl
        if lw_s:
            iso_str = "LOOP"
            iso_ok  = False
            swarm_all_clean = False
        elif ok_s < c_val:
            iso_str = f"{ok_s}/{c_val} Err"
            iso_ok  = False
            swarm_all_clean = False
        elif sr.get("iso_verified"):
            if sr.get("iso_exact"):
                iso_str = "Turbo=Std" if int8_prefill else "100% Exact"
                iso_ok  = True
            else:
                iso_str = f"{sr.get('iso_exact_cnt', 0)}/{sr.get('iso_total_cnt', c_val)} Diff"
                iso_ok  = False
                swarm_iso_ok = False
                all_exact = False
        elif c_val == 1:
            iso_str = "Solo MTP"
            iso_ok  = True
        else:
            iso_str = f"{ok_s}/{c_val} Clean"
            iso_ok  = True
        lbl = "1 solo (MTP)" if c_val == 1 else f"{c_val} agents"
        swarm_rows.append({
            "c": c_val, "label": lbl, "agg": agg_tps, "per": per_tps,
            "e2e": e2e_tps, "scale": scl, "mem": uma_g, "mem_gb": uma_g * GIB_TO_GB,
            "iso": iso_str, "ok": iso_ok,
        })

if peak_dec_bw <= 0.0 and peak_dec > 0.0 and active_gb_per_tok > 0.0:
    peak_dec_bw = peak_dec * active_gb_per_tok

if loop_warnings:
    print(f"[STRATA // APEX] LOOP_WARNING detected in: {', '.join(loop_warnings)}", file=sys.stderr)
if high_spreads:
    print(f"[STRATA // APEX] WARNING: Run-to-run spread > 10%: {'; '.join(high_spreads)}", file=sys.stderr)
if max_majflt > 0:
    print(f"[STRATA // APEX] WARNING: Steady MajFlt={max_majflt} observed during timed benchmark region.", file=sys.stderr)

pp_ceil_pct  = (100.0 * peak_pp_tops / M5_MAX_PP_TOPS) if M5_MAX_PP_TOPS > 0.0 else 0.0
dec_ceil_pct = (100.0 * peak_dec_bw / M5_MAX_DRAM_GBPS) if M5_MAX_DRAM_GBPS > 0.0 else 0.0

def fmt_ttft(ms):
    if ms < 1000.0:
        return f"{ms:.0f} ms"
    sec = ms / 1000.0
    return f"{sec:.2f} s" if sec < 100.0 else f"{sec:.1f} s"

# ── ASCII Art Banner (87 chars wide, aligned cleanly with tables <= 90 cols) ──
BANNER = [
    "███████╗████████╗██████╗  █████╗ ████████╗ █████╗        █████╗ ██████╗ ███████╗██╗  ██╗",
    "██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██╔══██╗██╔════╝╚██╗██╔╝",
    "███████╗   ██║   ██████╔╝███████║   ██║   ███████║ ───  ███████║██████╔╝█████╗   ╚███╔╝ ",
    "╚════██║   ██║   ██╔══██╗██╔══██║   ██║   ██╔══██║      ██╔══██║██╔═══╝ ██╔══╝   ██╔██╗ ",
    "███████║   ██║   ██║  ██║██║  ██║   ██║   ██║  ██║      ██║  ██║██║     ███████╗██╔╝ ██╗",
    "╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝",
]
BANNER_COLORS = ["1;38;5;51", "1;38;5;45", "1;38;5;39", "1;38;5;39", "1;38;5;33", "1;38;5;27"]

# ── Box-Drawing Table Helper (exact alignment + strict <= 90-col check) ───────
def emit(line=""):
    vis = ANSI_RE.sub("", line)
    if len(vis) > 90:
        raise AssertionError(f"Terminal card line exceeds 90 columns ({len(vis)} cols): {vis!r}")
    print(line)

def hline(widths, left, mid, right):
    return "  " + c(GRAY, left + mid.join("─" * w for w in widths) + right)

def row_line(cells, widths, aligns, styles=None):
    parts = []
    for i, (txt, w, al) in enumerate(zip(cells, widths, aligns)):
        inner = w - 2
        if len(txt) > inner:
            txt = txt[:inner]
        padded = txt.rjust(inner) if al == "r" else (txt.center(inner) if al == "c" else txt.ljust(inner))
        cell_str = f" {padded} "
        if styles and styles[i]:
            cell_str = c(styles[i], cell_str)
        parts.append(cell_str)
    sep = c(GRAY, "│")
    return "  " + sep + sep.join(parts) + sep

def fit_80(s):
    return s if len(s) <= 80 else (s[:79] + "…")

def fit_88(s):
    return s if len(s) <= 88 else (s[:87] + "…")

# ── Render Terminal Screenshot Card (EVERY line strictly <= 90 columns) ───────
emit("")
for line, col in zip(BANNER, BANNER_COLORS):
    emit("  " + c(col, line))

stale_tag = " · STALE BUILD" if bin_stale else ""
if runs > 1:
    cd_tag = f" · {cooldown_sec}s cooldown" if cooldown_sec > 0 else ""
    subtitle = f"Apple Silicon Local AI Benchmark · {metal_ver} · Zero-Copy UMA · median of {runs}{cd_tag}{stale_tag}"
else:
    subtitle = f"Apple Silicon Local AI Benchmark · {metal_ver} · Zero-Copy UMA · single run{stale_tag}"
emit("  " + c(DIM, fit_88(subtitle)))
emit("  " + c(GRAY, "─" * 88))

sys_line    = f"{model_name} · {chip} ({cpu_total}c CPU · {gpu_cores}c GPU · {mem_gb} GiB) · {os_name} {os_ver}"
max_swap_gb = max_swap * GIB_TO_GB
swap_txt    = "Zero Swap" if max_swap_gb < 0.01 else f"+{max_swap_gb:.2f} GB Swap"
swap_short  = "0 Swap" if max_swap_gb < 0.01 else f"+{max_swap_gb:.2f}GB Swap"
if loop_warnings:
    exact_txt = "LOOP_WARNING"
elif not (all_exact and all_recall):
    exact_txt = "Check Results"
elif max_majflt > 0:
    exact_txt = f"MajFlt={max_majflt}"
elif int8_prefill:
    exact_txt = "Turbo=Std"
else:
    exact_txt = "100% Exact"

peak_line = (
    f"{peak_pp:,.0f}/s Prefill ({peak_pp_tops:.1f}T·{pp_ceil_pct:.0f}% mod) · "
    f"{peak_dec:.1f}/s Decode ({dec_ceil_pct:.0f}% mod BW) · {exact_txt} · {swap_txt}"
)
if len(peak_line) > 80:
    peak_line = (
        f"{peak_pp:,.0f}/s PP ({peak_pp_tops:.1f}T·{pp_ceil_pct:.0f}% mod) · "
        f"{peak_dec:.1f}/s TG ({dec_ceil_pct:.0f}% mod BW) · {exact_txt} · {swap_short}"
    )

emit("  " + c(CYAN, "SYSTEM ") + " " + fit_80(sys_line))
emit("  " + c(CYAN, "MODEL  ") + " " + fit_80(model_line))
if not swarm_only:
    emit("  " + c(GOLD, "PEAK   ") + " " + c(WHITE, fit_80(peak_line)))
swarm_line = ""
if swarm_rows:
    sw_iso_txt = "100% Isolated" if (swarm_iso_ok and swarm_all_clean) else "Check Isolation"
    swarm_line = (
        f"{swarm_peak_tps:,.1f}/s Steady ({swarm_peak_e2e:,.1f}/s E2E · "
        f"{swarm_peak_c}x · {swarm_peak_per:.1f}/ag) · {sw_iso_txt} · {swap_short}"
    )
    if len(swarm_line) > 80:
        sw_iso_short = "100% Isol" if (swarm_iso_ok and swarm_all_clean) else "Check Isol"
        swarm_line = (
            f"{swarm_peak_tps:,.1f}/s Steady ({swarm_peak_e2e:,.1f}/s E2E · "
            f"{swarm_peak_c}x) · {sw_iso_short} · {swap_short}"
        )
    emit("  " + c(GOLD, "SWARM  ") + " " + c(WHITE, fit_80(swarm_line)))
emit("")

run_meta_tag = (f"median of {runs}" + (f" · {cooldown_sec}s cd" if cooldown_sec > 0 else "")) if runs > 1 else "single run"

if not swarm_only:
    # Table 1: Prompt & Context Scaling (total width = 2 + 88 = 90 chars, matching banner!)
    t1_w = [9, 12, 11, 9, 10, 10, 10, 8]  # sum(t1_w) + 9 borders = 79 + 9 = 88!
    t1_a = ["r", "r", "r", "r", "r", "r", "r", "c"]
    emit("  " + c(WHITE, "PROMPT & CONTEXT SCALING"))
    emit(hline(t1_w, "┌", "┬", "┐"))
    emit(row_line(
        ["Context", "Prefill", "Mod. TOPS", "TTFT", "Standard", "Turbo", "Memory", "Recall"],
        t1_w, ["c", "c", "c", "c", "c", "c", "c", "c"],
        [CYAN] * 8,
    ))
    emit(hline(t1_w, "├", "┼", "┤"))
    for r in ctx_rows:
        if r["loop_warn"]:
            rec_str = "LOOP"
            rec_col = RED
        else:
            rec_str = "PASS" if r["recall"] else "FAIL"
            rec_col = GREEN if r["recall"] else RED
        is_pk_pp = abs(r["pp"] - peak_pp) < 1e-3
        pp_col   = WHITE if is_pk_pp else ""
        tp_col   = CYAN  if is_pk_pp else ""
        emit(row_line(
            [
                f"{r['ctx']:,}",
                f"{r['pp']:,.0f} t/s",
                f"{r['tops']:.1f} TOPS" if r["tops"] < 100.0 else f"{r['tops']:.0f} TOPS",
                fmt_ttft(r["ttft_ms"]),
                f"{r['d0']:.1f}/s" if r["d0"] > 0 else "—",
                f"{r['dk']:.1f}/s" if r["dk"] > 0 else "—",
                f"{r['mem_gb']:.1f} GB",
                rec_str,
            ],
            t1_w, t1_a,
            ["", pp_col, tp_col, "", "", GOLD, "", rec_col],
        ))
    emit(hline(t1_w, "└", "┴", "┘"))
    emit("")

    # Table 2: Real-World Generation (total width = sum + 7 borders = 81 + 7 = 88, exact match!)
    t2_w = [24, 10, 10, 9, 11, 17]  # sum = 81 + 7 = 88!
    t2_a = ["l", "r", "r", "r", "r", "c"]
    mode_tag = "Thinking" if think_flag == "--think" else "Direct"
    t2_meta  = f"{mat_gen} tokens · {mode_tag} · {run_meta_tag}"
    emit("  " + c(WHITE, "REAL-WORLD GENERATION") + " " + c(DIM, f"({t2_meta})"))
    emit(hline(t2_w, "┌", "┬", "┐"))
    emit(row_line(
        ["Workload", "Standard", "Turbo", "Speedup", "Mod. TOPS", "Fidelity"],
        t2_w, ["l", "c", "c", "c", "c", "c"],
        [CYAN] * 6,
    ))
    emit(hline(t2_w, "├", "┼", "┤"))
    for r in mat_rows:
        fid_col = DIM if (r["ok"] is None) else (GREEN if r["ok"] else RED)
        dt_str  = f"{r['tops']:.2f} TOPS" if r["tops"] < 10.0 else f"{r['tops']:.1f} TOPS"
        emit(row_line(
            [
                r["label"],
                f"{r['k0']:.1f}/s" if r["k0"] > 0 else "—",
                f"{r['dk']:.1f}/s",
                f"{r['spd']:.2f}x" if r["spd"] > 0 else "—",
                dt_str,
                r["fid"],
            ],
            t2_w, t2_a,
            ["", "", GOLD, CYAN, "", fid_col],
        ))
    emit(hline(t2_w, "└", "┴", "┘"))

# Table 3: Swarm & Concurrent Sessions (total width = sum + 8 borders = 80 + 8 = 88, exact match!)
t3_meta = ""
if swarm_rows:
    if not swarm_only:
        emit("")
    t3_w = [14, 12, 11, 11, 9, 9, 14]  # sum = 80 + 8 = 88!
    t3_a = ["l", "r", "r", "r", "r", "r", "c"]
    sw_ctx_k  = swarm_data["swarm_ctx"] // 1024
    sw_kv_lbl = str(swarm_data.get("kv_cache", "f16")).upper()
    t3_meta   = f"{swarm_data['swarm_tokens']} tok/ag · {sw_ctx_k}K ctx · {sw_kv_lbl} KV · SSE · {run_meta_tag}"
    emit("  " + c(WHITE, "SWARM & CONCURRENT SESSIONS") + " " + c(DIM, f"({t3_meta})"))
    emit(hline(t3_w, "┌", "┬", "┐"))
    emit(row_line(
        ["Sessions", "Steady Agg", "Per-Agent", "E2E Wall", "Scaling", "Memory", "Isolation"],
        t3_w, ["l", "c", "c", "c", "c", "c", "c"],
        [CYAN] * 7,
    ))
    emit(hline(t3_w, "├", "┼", "┤"))
    for r in swarm_rows:
        iso_col = GREEN if r["ok"] else RED
        agg_str = f"{r['agg']:,.1f} t/s" if r["agg"] < 1000.0 else f"{r['agg']:,.0f} t/s"
        mem_str = f"{r['mem_gb']:.1f} GB" if r["mem_gb"] < 100.0 else f"{r['mem_gb']:.0f} GB"
        emit(row_line(
            [
                r["label"],
                agg_str,
                f"{r['per']:.1f}/s",
                f"{r['e2e']:.1f}/s",
                f"{r['scale']:.2f}x" if r["scale"] > 0.0 else "—",
                mem_str,
                r["iso"],
            ],
            t3_w, t3_a,
            ["", GOLD, "", "", CYAN, "", iso_col],
        ))
    emit(hline(t3_w, "└", "┴", "┘"))

ceil_lbl = f"{M5_MAX_PP_TOPS:.1f}T {'INT8' if int8_prefill else 'FP16'}"
footer_notes = []
if swarm_only:
    footer = f"Swarm: Steady Agg = full-overlap decode; E2E Wall = start to end (vs {swarm_k0_base:.1f}/s Std)"
else:
    footer = f"Standard=1-tok · Turbo=MTP · TOPS & %BW modeled vs {CHIP_SHORT} ({ceil_lbl}, {M5_MAX_DRAM_GBPS:.0f} GB/s)"
emit("  " + c(DIM, fit_88(footer)))
footer_notes.append(footer)

if swarm_rows and not swarm_only:
    footer2 = "Swarm: Steady Agg = full-overlap decode; E2E Wall = request start to last token"
    emit("  " + c(DIM, fit_88(footer2)))
    footer_notes.append(footer2)

if int8_prefill:
    int8_note = "Note: INT8 prefill is approximate vs default FP16; checks verify Turbo = Standard"
    emit("  " + c(DIM, fit_88(int8_note)))
    footer_notes.append(int8_note)

if runs > 1:
    if not swarm_only and pp_spread_list and tg_spread_list:
        max_pp_sp, max_pp_lbl, _, _ = max(pp_spread_list, key=lambda x: x[0])
        max_tg_sp, max_tg_lbl, _, _ = max(tg_spread_list, key=lambda x: x[0])
        range_note = f"Spread ({runs} runs): max prefill {max_pp_sp:.1f}% ({max_pp_lbl}) · max decode {max_tg_sp:.1f}% ({max_tg_lbl})"
        emit("  " + c(DIM, fit_88(range_note)))
        footer_notes.append(range_note)
    elif swarm_only and sw_spread_list:
        max_sw_sp, max_sw_lbl, _, _ = max(sw_spread_list, key=lambda x: x[0])
        range_note = f"Spread ({runs} runs): max swarm steady decode {max_sw_sp:.1f}% ({max_sw_lbl})"
        emit("  " + c(DIM, fit_88(range_note)))
        footer_notes.append(range_note)

if max_majflt > 0:
    flt_note = f"Warning: Steady MajFlt={max_majflt} observed during timed region (logs preserved)"
    emit("  " + c(RED, fit_88(flt_note)))
    footer_notes.append(flt_note)
emit("")

# ── Optional Clean Markdown Export (--out) ────────────────────────────────────
if out_md:
    md = []
    md.append("```text")
    md.extend(BANNER)
    md.append("```")
    md.append("")
    if runs > 1:
        run_note = f" (median of {runs}" + (f", {cooldown_sec}s cooldown)" if cooldown_sec > 0 else ")")
    else:
        run_note = " (single run)"
    md.append(f"## STRATA // APEX — {md_model_title}{run_note}")
    md.append(f"- **System:** {sys_line}")
    md.append(f"- **Binary:** `{os.path.basename(strata_bin)}` (built {bin_mtime}{' · STALE' if bin_stale else ''})")
    md.append(f"- **Model:** {model_line}")
    if not swarm_only:
        md.append(f"- **Peak:** {peak_line}")
        std_bw_note = f"; max Standard `{max_std_bw:.1f}` GB/s = `{100.0 * max_std_bw / M5_MAX_DRAM_GBPS:.1f}%`" if (max_std_bw > 0.0 and M5_MAX_DRAM_GBPS > 0.0) else ""
        md.append(f"- **{CHIP_SHORT} Modeled Ceiling:** Prefill `{pp_ceil_pct:.1f}%` of `{M5_MAX_PP_TOPS:.1f}` TOPS (sustained) · Peak Decode `{dec_ceil_pct:.1f}%` of `{M5_MAX_DRAM_GBPS:.0f}` GB/s (`{peak_dec_bw:.1f}` GB/s modeled on `{peak_dec_src}`{std_bw_note})")
    if swarm_line:
        md.append(f"- **Swarm:** {swarm_line}")
    md.append("")
    if not swarm_only:
        md.append("### Prompt & Context Scaling")
        md.append("| Context | Prefill | Mod. TOPS | TTFT | Standard | Turbo | Memory | Recall |")
        md.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
        for r in ctx_rows:
            d0_s = f"{r['d0']:.1f} tok/s" if r["d0"] > 0 else "—"
            dk_s = f"**{r['dk']:.1f} tok/s**" if r["dk"] > 0 else "—"
            rec_md = "LOOP" if r["loop_warn"] else ("PASS" if r["recall"] else "FAIL")
            md.append(f"| {r['ctx']:,} | {r['pp']:,.0f} tok/s | {r['tops']:.1f} TOPS | {fmt_ttft(r['ttft_ms'])} | {d0_s} | {dk_s} | {r['mem_gb']:.1f} GB | {rec_md} |")
        md.append("")
        md.append(f"### Real-World Generation ({t2_meta})")
        md.append("| Workload | Standard | Turbo | Speedup | Mod. TOPS | Fidelity |")
        md.append("| :--- | ---: | ---: | ---: | ---: | :---: |")
        for r in mat_rows:
            k0_s  = f"{r['k0']:.1f} tok/s" if r["k0"] > 0 else "—"
            spd_s = f"{r['spd']:.2f}x" if r["spd"] > 0 else "—"
            md.append(f"| {r['label']} | {k0_s} | **{r['dk']:.1f} tok/s** | {spd_s} | {r['tops']:.2f} TOPS | {r['fid']} |")
        md.append("")
    if swarm_rows:
        md.append(f"### Swarm & Concurrent Sessions ({t3_meta})")
        md.append("| Sessions | Steady Agg | Per-Agent | E2E Wall | Scaling | Memory | Isolation |")
        md.append("| :--- | ---: | ---: | ---: | ---: | ---: | :---: |")
        for r in swarm_rows:
            scl_s = f"{r['scale']:.2f}x" if r["scale"] > 0.0 else "—"
            md.append(f"| {r['label']} | **{r['agg']:,.1f} tok/s** | {r['per']:.1f} tok/s | {r['e2e']:.1f} tok/s | {scl_s} | {r['mem_gb']:.1f} GB | {r['iso']} |")
        md.append("")
    for fn in footer_notes:
        md.append(f"*{fn}.*")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  Saved Markdown report to {out_md}", file=sys.stderr)
PYEOF

if [[ "$RENDER_RC" -ne 0 ]]; then
  SCRIPT_FAILED=1
  exit "$RENDER_RC"
fi
if [[ "$HAD_GATE_FAIL" -ne 0 ]]; then
  exit 1
fi

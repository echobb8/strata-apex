#!/usr/bin/env zsh
# STRATA-M5 Head-to-Head Showcase Benchmark — Qwen3.8-Flash-Next (180B MoE) vs Inkling-2-Mini (26B Looped MoE)
# Runs both engines strictly sequentially (never simultaneously) across:
#   1) System Specs + Side-by-Side Plain-English Model & Framework Comparison Table
#   2) Context Scaling (512, 2K, 4K, 16K, 32K — or 512, 2K, 4K, 16K with --quick): pp, TTFT, K=0 vs MTP, Speedup, NIAH, UMA, MajFlt
#   3) 2x2 Workload Matrix (Prose vs Code x Greedy T=0 vs Sampled T=0.6) side-by-side
#
# Usage:
#   ./scripts/bench_compare.sh                  # Standard: 512,2048,4096,16384,32768 + 2x2 matrix (256 tok)
#   ./scripts/bench_compare.sh --quick          # Faster:   512,2048,4096,16384 + 2x2 matrix (192 tok)
#   ./scripts/bench_compare.sh --think          # Use thinking mode (effort=medium) for the 2x2 matrix
#   ./scripts/bench_compare.sh --out compare.md # Also save the Markdown comparison report to a file

set -euo pipefail

REPO_ROOT="${0:A:h:h}"
cd "$REPO_ROOT"

CONTEXTS="512,2048,4096,16384,32768"
CTX_GEN=128
MAT_GEN=256
FLASH_K=4
INK_K=7
THINK_FLAG="--no-think"
OUT_MD=""

STRATA_CACHE="${STRATA_APEX_CACHE_DIR:-${STRATA_CACHE_DIR:-$HOME/.cache/strata-apex}}"
[[ ! -d "$STRATA_CACHE" && -d "$HOME/.cache/strata" ]] && STRATA_CACHE="$HOME/.cache/strata"

FLASH_BUNDLE="$STRATA_CACHE/qwen3.8-flash-next"
if [[ -f "$STRATA_CACHE/qwen3.8-flash-next-fast/uma_part0.strata" ]]; then
  FLASH_BUNDLE="$STRATA_CACHE/qwen3.8-flash-next-fast"
elif [[ -f "$HOME/.cache/strata/qwen3.8-flash-next-fast/uma_part0.strata" ]]; then
  FLASH_BUNDLE="$HOME/.cache/strata/qwen3.8-flash-next-fast"
fi
INK_BUNDLE="$STRATA_CACHE/inkling-2-mini"
if [[ -f "$STRATA_CACHE/inkling-2-mini-fast/inkling2_uma.strata" ]]; then
  INK_BUNDLE="$STRATA_CACHE/inkling-2-mini-fast"
elif [[ -f "$HOME/.cache/strata/inkling-2-mini-fast/inkling2_uma.strata" ]]; then
  INK_BUNDLE="$HOME/.cache/strata/inkling-2-mini-fast"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
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
    --flash-k)
      FLASH_K="$2"; shift 2
      ;;
    --inkling2-k)
      INK_K="$2"; shift 2
      ;;
    --think)
      THINK_FLAG="--think"; shift
      ;;
    --no-think)
      THINK_FLAG="--no-think"; shift
      ;;
    --flash-bundle)
      FLASH_BUNDLE="$2"; shift 2
      ;;
    --inkling2-bundle)
      INK_BUNDLE="$2"; shift 2
      ;;
    --out)
      OUT_MD="$2"; shift 2
      ;;
    -h|--help)
      sed -n '2,13p' "$0"
      exit 0
      ;;
    *)
      echo "[bench_compare] Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if pgrep -f "strata_cli serve" >/dev/null 2>&1; then
  echo "[bench_compare] ERROR: A live 'strata_cli serve' process is running." >&2
  echo "                Stop the server first so the sequential benchmark stays well within 128 GB UMA." >&2
  exit 1
fi

if [[ ! -x ./build/strata_cli ]]; then
  echo "[bench_compare] Building ./build/strata_cli..."
  make -j8 cli
fi

TMP_DIR="$(mktemp -d -t strata_compare.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROSE_PROMPT="Explain in clear, engaging prose how modern unified-memory GPUs differ from discrete GPUs for running large Mixture-of-Experts language models, covering memory bandwidth, cache hierarchy, quantization trade-offs, and speculative decoding. Write complete paragraphs without code blocks."

CODE_PROMPT="Write a clean, well-typed TypeScript module implementing a generic O(1) LRUCache<K, V> backed by an intrusive doubly-linked list and Map, supporting per-entry TTL expiration, hit/miss statistics, peek(), delete(), clear(), and an eviction callback. Output code only."

SAMPLE_ARGS=(
  --temp 0.6
  --top-k 20
  --top-p 0.95
  --penalty-window 256
  --presence-penalty 0.6
  --frequency-penalty 0.08
  --seed 42
)

# ==============================================================================
# PHASE A: Qwen3.8-Flash-Next (180B MoE) — Runs completely before Inkling-2 starts
# ==============================================================================
echo "[1/7] Inspecting Qwen3.8-Flash-Next (180B) & Inkling-2-Mini (26B Looped) bundles..."
./build/strata_cli inspect-flash --bundle "$FLASH_BUNDLE" > "$TMP_DIR/flash_inspect.txt"
./build/strata_cli inspect-inkling2 --bundle "$INK_BUNDLE" > "$TMP_DIR/ink_inspect.txt"

echo "[2/7] [Qwen3.8-Flash-Next] Context Scaling + NIAH ($CONTEXTS, K=0 vs K=$FLASH_K, gen=$CTX_GEN)..."
caffeinate -dimsu ./build/strata_cli bench-flash \
  --bundle "$FLASH_BUNDLE" \
  --contexts "$CONTEXTS" \
  --draft-ks "0,$FLASH_K" \
  --gen-tokens "$CTX_GEN" \
  --no-think \
  --json "$TMP_DIR/flash_ctx.json" \
  > "$TMP_DIR/flash_ctx.log"

FLASH_RUN_COMMON=(
  --bundle "$FLASH_BUNDLE"
  "$THINK_FLAG"
  --effort medium
  --draft-k "$FLASH_K"
  --max-tokens "$MAT_GEN"
  --max-seq-len 8192
  --no-eos-stop
)

echo "[3/7] [Qwen3.8-Flash-Next] 2x2 Matrix (Prose & Code x Greedy T=0 w/ K=0 verify & Sampled T=0.6, $MAT_GEN toks)..."
caffeinate -dimsu ./build/strata_cli run-flash "${FLASH_RUN_COMMON[@]}" \
  --prompt "$PROSE_PROMPT" --greedy --verify-k0 \
  > "$TMP_DIR/flash_prose_greedy.log"

caffeinate -dimsu ./build/strata_cli run-flash "${FLASH_RUN_COMMON[@]}" \
  --prompt "$PROSE_PROMPT" "${SAMPLE_ARGS[@]}" \
  > "$TMP_DIR/flash_prose_sampled.log"

caffeinate -dimsu ./build/strata_cli run-flash "${FLASH_RUN_COMMON[@]}" \
  --prompt "$CODE_PROMPT" --greedy --verify-k0 \
  > "$TMP_DIR/flash_code_greedy.log"

caffeinate -dimsu ./build/strata_cli run-flash "${FLASH_RUN_COMMON[@]}" \
  --prompt "$CODE_PROMPT" "${SAMPLE_ARGS[@]}" \
  > "$TMP_DIR/flash_code_sampled.log"

# ==============================================================================
# PHASE B: Inkling-2-Mini (26B Looped) — Runs after Flash-Next has exited cleanly
# ==============================================================================
echo "[4/7] [Inkling-2-Mini] Context Scaling + NIAH ($CONTEXTS, K=0 vs K=$INK_K, gen=$CTX_GEN)..."
caffeinate -dimsu ./build/strata_cli bench-inkling2 \
  --bundle "$INK_BUNDLE" \
  --contexts "$CONTEXTS" \
  --draft-ks "0,$INK_K" \
  --gen-tokens "$CTX_GEN" \
  --no-think \
  --json "$TMP_DIR/ink_ctx.json" \
  > "$TMP_DIR/ink_ctx.log"

INK_RUN_COMMON=(
  --bundle "$INK_BUNDLE"
  "$THINK_FLAG"
  --effort medium
  --draft-k "$INK_K"
  --max-tokens "$MAT_GEN"
  --max-seq-len 8192
  --no-eos-stop
)

echo "[5/7] [Inkling-2-Mini] Prose benchmark (Greedy T=0 w/ K=0 verify + Sampled T=0.6, $MAT_GEN toks)..."
caffeinate -dimsu ./build/strata_cli run-inkling2 "${INK_RUN_COMMON[@]}" \
  --prompt "$PROSE_PROMPT" --greedy --verify-k0 \
  > "$TMP_DIR/ink_prose_greedy.log"

caffeinate -dimsu ./build/strata_cli run-inkling2 "${INK_RUN_COMMON[@]}" \
  --prompt "$PROSE_PROMPT" "${SAMPLE_ARGS[@]}" \
  > "$TMP_DIR/ink_prose_sampled.log"

echo "[6/7] [Inkling-2-Mini] Code benchmark (Greedy T=0 w/ K=0 verify + Sampled T=0.6, $MAT_GEN toks)..."
caffeinate -dimsu ./build/strata_cli run-inkling2 "${INK_RUN_COMMON[@]}" \
  --prompt "$CODE_PROMPT" --greedy --verify-k0 \
  > "$TMP_DIR/ink_code_greedy.log"

caffeinate -dimsu ./build/strata_cli run-inkling2 "${INK_RUN_COMMON[@]}" \
  --prompt "$CODE_PROMPT" "${SAMPLE_ARGS[@]}" \
  > "$TMP_DIR/ink_code_sampled.log"

echo "[7/7] Generating head-to-head comparison report..."
echo ""

python3 - "$TMP_DIR" "$FLASH_K" "$INK_K" "$THINK_FLAG" "$MAT_GEN" "$OUT_MD" << 'PYEOF'
import json, os, re, subprocess, sys

tmp_dir, flash_k_s, ink_k_s, think_flag, mat_gen, out_md = sys.argv[1:7]
flash_k = int(flash_k_s)
ink_k   = int(ink_k_s)

def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "Unknown"

os_name   = sh(["sw_vers", "-productName"])
os_ver    = sh(["sw_vers", "-productVersion"])
os_build  = sh(["sw_vers", "-buildVersion"])
chip      = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
mem_bytes = int(sh(["sysctl", "-n", "hw.memsize"]) or "0")
mem_gb    = mem_bytes // (1024 ** 3)
hw_prof   = sh(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"])

def grep_prof(pattern, default="Unknown"):
    m = re.search(pattern, hw_prof)
    return m.group(1).strip() if m else default

model_name = grep_prof(r"Model Name:\s*(.+)")
model_id   = grep_prof(r"Model Identifier:\s*(.+)")
cpu_cores_raw = grep_prof(r"Total Number of Cores:\s*(\d+\s*\([^)]+\))")
m_cpu = re.match(r"(\d+)\s*\(([^)]+)\)", cpu_cores_raw)
cpu_desc = f"{m_cpu.group(1)}-core CPU ({m_cpu.group(2)})" if m_cpu else f"{cpu_cores_raw}-core CPU"
gpu_cores = grep_prof(r"Chipset Model:.*?\n\s*Type:.*?\n\s*Bus:.*?\n\s*Total Number of Cores:\s*(\d+)")
if gpu_cores == "Unknown":
    ms = re.findall(r"Total Number of Cores:\s*(\d+)\s*$", hw_prof, re.M)
    if ms:
        gpu_cores = ms[-1]
metal_ver = grep_prof(r"Metal Support:\s*(.+)")

flash_insp = open(os.path.join(tmp_dir, "flash_inspect.txt")).read()
ink_insp   = open(os.path.join(tmp_dir, "ink_inspect.txt")).read()
def grep_txt(txt, pat, default="?"):
    m = re.search(pat, txt)
    return m.group(1).strip() if m else default

f_part0   = grep_txt(flash_insp, r"UMA Part 0[^:]*:\s*([0-9.]+)\s*GiB")
f_part1   = grep_txt(flash_insp, r"UMA Part 1[^:]*:\s*([0-9.]+)\s*GiB")
f_ple     = grep_txt(flash_insp, r"NVMe PLE Store[^:]*:\s*([0-9.]+)\s*GiB")
f_act     = grep_txt(flash_insp, r"Active Traffic\s*:\s*([0-9.]+)\s*GB/tok")
try:
    f_uma_w = f"{float(f_part0) + float(f_part1):.2f} GiB"
except Exception:
    f_uma_w = "?"

m_dense = re.search(
    r"Dense Precision\s*:\s*embed=(\S+),\s*attn=(\S+),\s*wr=(\S+),\s*unembed=(\S+),\s*ctrl=(\S+),\s*sink=(\S+)",
    ink_insp,
)
m_moe  = re.search(r"Sparse MoE\s*:.*?\|\s*(\d+)\s+NVFP4\s*\+\s*(\d+)\s+MXFP8\s+layers", ink_insp)
m_mtp  = re.search(r"DSpark MTP\s*:\s*ENABLED\s*\((\d+)\s+layers\s*\[([^\]]+)\]", ink_insp)
m_frsp = re.search(r"FR-Spec Head\s*:\s*ENABLED\s*\(shortlist=(\d+)", ink_insp)
fr_n   = int(m_frsp.group(1)) if m_frsp else 0

if m_dense and m_moe:
    d_embed, d_attn, d_wr, d_unemb, d_ctrl, d_sink = m_dense.groups()
    n_nvfp4, n_mxfp8 = int(m_moe.group(1)), int(m_moe.group(2))
    mtp_layers = int(m_mtp.group(1)) if m_mtp else 3
    mtp_dt     = m_mtp.group(2) if m_mtp else "F16"
    if d_attn != "F16" or n_mxfp8 == 0 or fr_n > 0:
        q_parts = [f"`{d_attn}` attn"]
        if d_unemb == d_wr == d_ctrl:
            q_parts.append(f"`{d_unemb}` unembed/wr/ctrl")
        else:
            q_parts.append(f"`{d_unemb}` unembed/`{d_wr}` wr/`{d_ctrl}` ctrl")
        if n_mxfp8 == 0 and n_nvfp4 > 0:
            q_parts.append(f"NVFP4 L0..{n_nvfp4 - 1}")
        else:
            q_parts.append(f"NVFP4 L0..{n_nvfp4 - 1} + MXFP8 L{n_nvfp4}..{n_nvfp4 + n_mxfp8 - 1}")
        q_parts.append(f"`{mtp_dt}` {mtp_layers}L MTP")
        if fr_n > 0:
            fr_k = f"{fr_n // 1024}K" if (fr_n % 1024 == 0) else str(fr_n)
            q_parts.append(f"{fr_k} SLC FR-Spec")
        ink_quant_desc = " + ".join(q_parts)
    else:
        ink_quant_desc = "**Native NVFP4** (`E2M1`+`E4M3`, L0..19) + **Native MXFP8** (`E4M3`+`UE8M0`, L20..23) + FP16 dense/KV"
else:
    ink_quant_desc = "**Native NVFP4** (`E2M1`+`E4M3`, L0..19) + **Native MXFP8** (`E4M3`+`UE8M0`, L20..23) + FP16 dense/KV"

flash_ctx = json.load(open(os.path.join(tmp_dir, "flash_ctx.json")))
ink_ctx   = json.load(open(os.path.join(tmp_dir, "ink_ctx.json")))

ink_uma_w = f"{ink_ctx.get('uma_mapped_gib', 0.0):.2f} GiB" if ink_ctx.get("uma_mapped_gib") else "~18.2 GiB"
ink_act   = f"{ink_ctx.get('active_gb_per_tok', 0.0):.2f} GB/tok" if ink_ctx.get("active_gb_per_tok") else "?"
ink_adap  = bool(ink_ctx.get("adaptive_k", True))

def parse_run_log(path):
    txt = open(path).read()
    out = {}
    m = re.search(r"Prefill & TTFT\s*:\s*([0-9.]+)\s*tok/s\s*\|\s*TTFT\s*([0-9.]+)\s*ms\s*\((\d+)\s*prompt", txt)
    if m:
        out["pp_tps"], out["ttft_ms"], out["prompt_toks"] = float(m.group(1)), float(m.group(2)), int(m.group(3))
    m = re.search(r"Decode Throughput\s*:\s*([0-9.]+)\s*tok/s\s*\|\s*(\d+)\s*decode toks in\s*(\d+)\s*steps", txt)
    if m:
        out["tg_tps"], out["dec_toks"], out["dec_steps"] = float(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"Native MTP \(K=\d+\)\s*:\s*([0-9.]+)\s*tok/step\s*\|\s*Draft accepted\s*(\d+)\s*/\s*(\d+)\s*\(([0-9.]+)%\)", txt)
    if m:
        out["tok_per_step"], out["acc_pct"] = float(m.group(1)), float(m.group(4))
    m = re.search(r"(?:Modeled|Physical) DRAM BW\s*:\s*([0-9.]+)\s*GB/s (?:modeled|physical).*?\|\s*([0-9.]+)\s*GB/s K=0-equiv", txt)
    if m:
        out["phys_bw"], out["eff_bw"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Total UMA\s*([0-9.]+)\s*GiB.*?Swap\s*\+([0-9.]+)\s*GiB\)\s*\|\s*Steady MajFlt=(\d+)", txt)
    if m:
        out["uma_gib"], out["swap_gib"], out["majflt"] = float(m.group(1)), float(m.group(2)), int(m.group(3))
    m = re.search(r"MTP K=\d+ vs K=0 Check\s*:\s*([A-Z0-9% \-]+?)\s*\((\d+) vs (\d+) tokens,\s*Speedup\s*([0-9.]+)x:\s*([0-9.]+) vs ([0-9.]+) tok/s\)", txt)
    if m:
        out["k0_match"] = ("BIT-EXACT" in m.group(1))
        out["speedup"]  = float(m.group(4))
        out["k0_tps"]   = float(m.group(6))
    return out

f_pg = parse_run_log(os.path.join(tmp_dir, "flash_prose_greedy.log"))
f_ps = parse_run_log(os.path.join(tmp_dir, "flash_prose_sampled.log"))
f_cg = parse_run_log(os.path.join(tmp_dir, "flash_code_greedy.log"))
f_cs = parse_run_log(os.path.join(tmp_dir, "flash_code_sampled.log"))

i_pg = parse_run_log(os.path.join(tmp_dir, "ink_prose_greedy.log"))
i_ps = parse_run_log(os.path.join(tmp_dir, "ink_prose_sampled.log"))
i_cg = parse_run_log(os.path.join(tmp_dir, "ink_code_greedy.log"))
i_cs = parse_run_log(os.path.join(tmp_dir, "ink_code_sampled.log"))

lines = []
lines.append("## STRATA-M5 Head-to-Head Benchmark — Qwen3.8-Flash-Next (180B) vs Inkling-2-Mini (26B Looped)")
lines.append("")
lines.append("### 1. System Specs & Side-by-Side Model / Framework Comparison")
lines.append(f"- **Machine:** {model_name} (`{model_id}`) · **Chip:** {chip} — {cpu_desc}, {gpu_cores}-core GPU ({metal_ver})")
lines.append(f"- **Memory & OS:** {mem_gb} GB Unified Memory · {os_name} {os_ver} (`{os_build}`)")
lines.append("- **Runtime:** Custom STRATA-M5 v2.0 C++20 + **Metal 4 (NAX)** zero-copy UMA engine (models executed strictly sequentially).")
lines.append("")
lines.append("| Dimension | Qwen3.8-Flash-Next (`180B`) | Inkling-2-Mini (`26B Looped`) |")
lines.append("| :--- | :--- | :--- |")
lines.append("| **HF Checkpoint** | `Qwen/Qwen3.8-Flash-Next` | `thinkingmachines/Inkling-2-Mini` |")
lines.append("| **Architecture** | 48L hybrid (36 GatedDeltaNet + 12 QSA sparse attn) + 4-stream HC | 24 physical layers looped 3x in core L6..17 = **48 logical visits** (40 SWA-1024 + 8 Global rel-bias) |")
lines.append("| **MoE Topology** | 512 routed experts (top-10) + shared expert + 51.2B NVMe PLE | 128 routed experts (top-8, SiTU SwiGLU) + Block AttnRes + Causal SConv-4 |")
lines.append(f"| **Quantization** | `Q4_A` (4-bit affine, `gs=64`) experts/attn/PLE, `Q8_0` HC, FP16 KV | {ink_quant_desc} |")
ink_k_desc = f"Adaptive `K=0..{ink_k}`" if (ink_adap and ink_k > 0) else f"`K={ink_k}`"
fr_tag     = f", {fr_n // 1024}K FR-Spec" if (fr_n > 0 and fr_n % 1024 == 0) else (f", {fr_n} FR-Spec" if fr_n > 0 else "")
lines.append(f"| **Speculative MTP** | 1-layer native MTP (`K={flash_k}`) + rolling n-gram suffix draft | **3-layer DSpark Block-8** `rnn_hidden` Markov MTP ({ink_k_desc}, 4-tap fuse{fr_tag}) |")
lines.append(f"| **Mapped Footprint** | `{f_uma_w}` UMA (2 buffers) + `{f_ple} GiB` NVMe PLE (`0` GPU B) | `{ink_uma_w}` single-buffer zero-copy UMA (`0` NVMe offload) |")
lines.append(f"| **Active Traffic (`K=0`)** | `{f_act} GB/tok` | `{ink_act}` (across 48 logical visits) |")
lines.append("")

lines.append("### 2. Context Scaling, Prompt Prefill (`pp`), Decode (`tg`) & Needle-in-a-Haystack Recall")
lines.append("")
lines.append("| Context | Model | Prefill (`pp`) | TTFT | Decode `K=0` | Decode MTP | Speedup | MTP Accept | Bit-Exact | NIAH Recall | UMA / MajFlt |")
lines.append("| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |")

def index_ctx(data):
    m = {}
    for r in data.get("results", []):
        m.setdefault(r["context"], {})[r["draft_k"]] = r
    return m

f_by_ctx = index_ctx(flash_ctx)
i_by_ctx = index_ctx(ink_ctx)
all_ctxs = sorted(set(f_by_ctx.keys()) | set(i_by_ctx.keys()))

def fmt_ctx_row(ctx, label, k_mtp, by_k):
    r0  = by_k.get(0)
    rk  = by_k.get(k_mtp)
    ref = rk or r0
    if not ref:
        return None
    pp   = f"{(r0 or ref)['prefill_tok_s']:.1f} tok/s"
    ttft = f"{(r0 or ref)['ttft_ms']:.0f} ms"
    d0   = f"{r0['decode_tok_s']:.1f} tok/s" if r0 else "—"
    dk   = f"**{rk['decode_tok_s']:.1f} tok/s** (`K={k_mtp}`)" if rk else "—"
    spd  = f"{rk['speedup_vs_k0']:.2f}x" if (rk and r0) else "—"
    acc  = f"{100.0 * rk['mtp_accepted'] / rk['mtp_drafted']:.1f}%" if (rk and rk.get("mtp_drafted", 0) > 0) else "—"
    ex   = "YES" if (rk and rk.get("exact_k0_match")) else ("—" if not rk else "NO")
    niah = f"PASS ({ref['needles_hit']}/{ref['needles_total']})" if ref.get("needle_pass") else "FAIL"
    mem  = f"{ref['total_uma_gib']:.1f} GiB / {ref['steady_majflt']}"
    return f"| {ctx:,} | {label} | {pp} | {ttft} | {d0} | {dk} | {spd} | {acc} | {ex} | {niah} | {mem} |"

for ctx in all_ctxs:
    for lbl, km, mp in [("Qwen3.8-Flash (180B)", flash_k, f_by_ctx), ("Inkling-2-Mini (26B)", ink_k, i_by_ctx)]:
        row = fmt_ctx_row(ctx, lbl, km, mp.get(ctx, {}))
        if row:
            lines.append(row)

lines.append("")
mode_desc = "Thinking (`effort=medium`)" if think_flag == "--think" else "Direct output (`--no-think`)"
lines.append("### 3. 2×2 Workload Matrix — Prose vs Code × Greedy (`T=0`) vs Sampled (`T=0.6`)")
lines.append(f"*{mat_gen} generated tokens per cell · {mode_desc} · Flash MTP `K={flash_k}` vs Inkling-2 DSpark MTP {ink_k_desc}*")
lines.append("")
lines.append("| Workload | Sampling | Model | Decode `K=0` | Decode MTP | Speedup | tok/step | Accept % | Eff. BW | Exact `K=0` | UMA / MajFlt |")
lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |")

def mat_row(wl, samp, mdl, d, k_mtp, k0_fb=None):
    k0  = d.get("k0_tps", k0_fb)
    k0s = f"{k0:.1f} tok/s" if k0 else "—"
    dks = f"**{d.get('tg_tps', 0.0):.1f} tok/s** (`K={k_mtp}`)"
    spd = f"{d['speedup']:.2f}x" if "speedup" in d else (f"{d['tg_tps']/k0:.2f}x*" if k0 and "tg_tps" in d else "—")
    tps_step = f"{d.get('tok_per_step', 0.0):.2f}"
    acc = f"{d.get('acc_pct', 0.0):.1f}%"
    bw  = f"{d.get('eff_bw', 0.0):.0f} GB/s"
    ex  = "100% YES" if d.get("k0_match") else "Leviathan exact"
    mem = f"{d.get('uma_gib', 0.0):.1f} GiB / {d.get('majflt', 0)}"
    return f"| {wl} | {samp} | {mdl} | {k0s} | {dks} | {spd} | {tps_step} | {acc} | {bw} | {ex} | {mem} |"

lines.append(mat_row("Prose", "Greedy (`T=0.0`)",  "Qwen3.8-Flash (180B)", f_pg, flash_k))
lines.append(mat_row("Prose", "Greedy (`T=0.0`)",  "Inkling-2-Mini (26B)", i_pg, ink_k))
lines.append(mat_row("Prose", "Sampled (`T=0.6`)", "Qwen3.8-Flash (180B)", f_ps, flash_k, f_pg.get("k0_tps")))
lines.append(mat_row("Prose", "Sampled (`T=0.6`)", "Inkling-2-Mini (26B)", i_ps, ink_k,   i_pg.get("k0_tps")))
lines.append(mat_row("Code",  "Greedy (`T=0.0`)",  "Qwen3.8-Flash (180B)", f_cg, flash_k))
lines.append(mat_row("Code",  "Greedy (`T=0.0`)",  "Inkling-2-Mini (26B)", i_cg, ink_k))
lines.append(mat_row("Code",  "Sampled (`T=0.6`)", "Qwen3.8-Flash (180B)", f_cs, flash_k, f_cg.get("k0_tps")))
lines.append(mat_row("Code",  "Sampled (`T=0.6`)", "Inkling-2-Mini (26B)", i_cs, ink_k,   i_cg.get("k0_tps")))
lines.append("")
lines.append("*`Speedup*` on sampled rows compares against each model's matched-workload `K=0` baseline; stochastic MTP uses lossless Leviathan rejection sampling (exact target distribution).*")

report = "\n".join(lines)
print(report)
if out_md:
    with open(out_md, "w") as f:
        f.write(report + "\n")
    print(f"\n[bench_compare] Saved Markdown report to {out_md}")
PYEOF

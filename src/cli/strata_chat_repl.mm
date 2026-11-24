/*
 * STRATA-M5 v2.0 — Interactive Multi-Turn Streaming Chat REPL & Bit-Exact Stateful Verifier
 * File: src/cli/strata_chat_repl.mm
 *
 * Subcommands:
 *   - `chat-flash`: Qwen/Qwen3.8-Flash-Next (180B MoE + 51.2B F_NOCACHE NVMe PLE + Native MTP, default K=2)
 *   - `chat-hf`:    Auto-detects Qwen3.8-Flash-Next (`uma_part0.strata`) or Qwen3.8-27B (`weights.strata`)
 *   - `chat`:       Unified alias (uses `--bundle` if passed, else defaults to Qwen3.8-Flash-Next)
 *
 * Key Architectural Features:
 *   1. True O(1) Stateful GPU Checkpoint / Restore (`< 0.5 ms` GPU->GPU `MTLBlitCommandEncoder` snapshot of
 *      DeltaNet conv/recurrent state, PLE dilated conv state & host 2-token ring, and MTP hidden/pos state).
 *   2. Exact Qwen3.8 / Qwen4-Exp Multi-Turn ChatML Compaction & Zero Speculative Overshoot Contamination:
 *      - Prefills `[system]<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n` and snapshots `turn_start_ckpt`.
 *      - Prefills `<think>\n` (`--think`) or `<think>\n\n</think>\n\n` (`--no-think`) and stream-decodes live.
 *      - Restores `turn_start_ckpt` (`< 0.3 ms`) and prefills canonical `{clean_reply}<|im_end|>\n`, stripping
 *        completed `<think>...</think>` traces and eliminating any `draft_k > 0` speculative tail overshoot.
 *   3. Deterministic `--verify-multiturn` Proof: verifies 100% bit-identical Turn-3 token IDs between
 *      incremental 3-turn stateful execution (with `draft_k` MTP + checkpoint compaction) vs fresh `reset()`
 *      single-shot full-transcript prefill from `seq_pos = 0`.
 *   4. Step 4 Two-Tier Zero-Pause Realtime Compaction (`--compact-mode off|shadow|paged|hybrid`, default off):
 *      - Tier 2 (`paged`): the live context runs on a paged-QSA `StrataQwenFlashContextSlot` (UMA pool +
 *        F_NOCACHE NVMe `kv_swap` + ANE lookahead prefetch, 100% bit-exact lossless).
 *      - Tier 1 (`shadow`): EWMA token-velocity horizon predictor + immutable Episode Ledger (each closed
 *        turn distilled ONCE from verbatim text => zero summary-of-summary drift) + a shadow slot warmed in
 *        preemptible 32..64-token prefill slices during idle gaps (cooperative `poll()` loop), mirrored with
 *        every committed live turn until DUAL_SYNCED, then promoted by an O(1) `bind_slot` RCU swap.
 *        The pre-swap live slot is kept as a rollback slot for `/undo-compact` (valid until the next turn).
 *      - `hybrid`: Tier 1 + Tier 2 (live / shadow / rollback slots are all paged).
 *      - `--verify-paged-kv` / `--verify-shadow-swap`: deterministic Step 4 self-tests.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_glm53.h"
#include "../../include/strata_inkling2.h"
#include "../../include/strata_inkling2_vision.h"
#include "../../include/strata_qwen38.h"
#include "../../include/strata_qwen_flash.h"

#include <fcntl.h>
#include <mach/mach.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using SteadyClock = std::chrono::steady_clock;

static double chat_elapsed_ms(SteadyClock::time_point t0, SteadyClock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static std::string chat_trim_ascii(const std::string& s) {
    size_t b = 0U;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && (s[e - 1U] == ' ' || s[e - 1U] == '\t' || s[e - 1U] == '\r' || s[e - 1U] == '\n')) {
        --e;
    }
    return s.substr(b, e - b);
}

static bool chat_is_flash_bundle(const std::string& dir) {
    if (dir.empty()) return false;
    const std::string p0 = dir + "/uma_part0.strata";
    return ::access(p0.c_str(), R_OK) == 0;
}

static bool chat_is_glm53_bundle(const std::string& dir) {
    if (dir.empty()) return false;
    const std::string sp = dir + "/" + STRATA_GLM53_SPINE_FILENAME;
    return ::access(sp.c_str(), R_OK) == 0;
}

static bool chat_is_glm53_fast_bundle(const std::string& dir) {
    if (dir.empty()) return false;
    const std::string sp = dir + "/" + STRATA_GLM53_SPINE_FILENAME;
    const std::string p0 = dir + "/" + STRATA_GLM53_EXPERTS_PART0_FILENAME;
    return (::access(sp.c_str(), R_OK) == 0) && (::access(p0.c_str(), R_OK) == 0);
}

static std::string chat_default_glm53_bundle() {
    const std::string fast_dir = STRATA_GLM53_DEFAULT_FAST_BUNDLE_DIR;
    if (chat_is_glm53_fast_bundle(fast_dir)) return fast_dir;
    return STRATA_GLM53_DEFAULT_BUNDLE_DIR;
}

static bool chat_is_inkling2_bundle(const std::string& dir) {
    if (dir.empty()) return false;
    const std::string p = dir + "/" + STRATA_INKLING2_UMA_FILENAME;
    return ::access(p.c_str(), R_OK) == 0;
}

static std::string chat_default_inkling2_bundle() {
    const std::string fast_dir = STRATA_INKLING2_DEFAULT_FAST_BUNDLE_DIR;
    if (chat_is_inkling2_bundle(fast_dir)) return fast_dir;
    return STRATA_INKLING2_DEFAULT_BUNDLE_DIR;
}

static std::string chat_to_lower_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    }
    return s;
}

static bool chat_parse_kv_cache(const std::string& s, uint32_t* out_dtype) {
    const std::string m = chat_to_lower_ascii(s);
    if (m == "f16" || m == "fp16") {
        if (out_dtype) *out_dtype = STRATA_INKLING2_KV_DTYPE_F16;
        return true;
    }
    if (m == "q8" || m == "q8_0") {
        if (out_dtype) *out_dtype = STRATA_INKLING2_KV_DTYPE_Q8_0;
        return true;
    }
    return false;
}

static const char* chat_kv_cache_label(uint32_t kv_dtype) {
    return (kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0) ? "q8" : "f16";
}

static std::string chat_inkling2_effort_to_num_str(bool enable_thinking, const std::string& raw_effort) {
    if (!enable_thinking) return "0";
    const std::string e = raw_effort.empty() ? "medium" : chat_to_lower_ascii(raw_effort);
    if (e == "none" || e == "off") return "0";
    if (e == "minimal")            return "0.1";
    if (e == "low")                return "0.2";
    if (e == "medium")             return "0.7";
    if (e == "high")               return "0.9";
    if (e == "xhigh" || e == "max")return "0.99";
    return "0.7";
}

static bool chat_parse_glm53_mode(const std::string& s, StrataGlm53ExecMode* out_mode) {
    const std::string m = chat_to_lower_ascii(s);
    if (m == "lossless" || m == "exact" || m == "stationary" || m == "mode-a" || m == "a") {
        *out_mode = STRATA_GLM53_MODE_LOSSLESS;
        return true;
    }
    if (m == "adaptive" || m == "zero-stall" || m == "zerostall" || m == "mode-b" || m == "b") {
        *out_mode = STRATA_GLM53_MODE_ADAPTIVE;
        return true;
    }
    return false;
}

struct ChatMemSnapshot {
    double host_rss_gib       = 0.0;
    double gpu_wired_gib      = 0.0;
    double total_uma_gib      = 0.0;
    double phys_footprint_gib = 0.0;
    double peak_rss_gib       = 0.0;
    double swap_used_gib      = 0.0;
    long   majflt             = 0;
    long   minflt             = 0;
};

static ChatMemSnapshot query_chat_mem_snapshot(double shared_mtl_gib = 0.0) {
    ChatMemSnapshot out{};
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

    mach_task_basic_info_data_t binfo{};
    mach_msg_type_number_t bcount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(),
                  MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&binfo),
                  &bcount) == KERN_SUCCESS) {
        const double raw_rss = static_cast<double>(binfo.resident_size) / kGiB;
        out.host_rss_gib     = std::max(0.0, raw_rss - shared_mtl_gib);
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev) {
            out.gpu_wired_gib = static_cast<double>([dev currentAllocatedSize]) / kGiB;
        }
    }
    out.total_uma_gib = out.gpu_wired_gib + out.host_rss_gib;

    task_vm_info_data_t vminfo{};
    mach_msg_type_number_t vcount = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(),
                  TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vminfo),
                  &vcount) == KERN_SUCCESS) {
        out.phys_footprint_gib = static_cast<double>(vminfo.phys_footprint) / kGiB;
    }

    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        out.peak_rss_gib = static_cast<double>(ru.ru_maxrss) / kGiB;
        out.majflt       = ru.ru_majflt;
        out.minflt       = ru.ru_minflt;
    }

    struct xsw_usage swap{};
    size_t swap_sz = sizeof(swap);
    if (sysctlbyname("vm.swapusage", &swap, &swap_sz, nullptr, 0) == 0) {
        out.swap_used_gib = static_cast<double>(swap.xsu_used) / kGiB;
    }
    return out;
}

/* ============================================================================
 * 1. CLI Configuration, Argument Parser & Slash Command Parser
 * ============================================================================ */

enum class ChatSubcmdKind {
    kChatUnified  = 0,
    kChatFlash    = 1,
    kChatHf       = 2,
    kChatGlm53    = 3,
    kChatInkling2 = 4,
};

static constexpr uint32_t kChatMaxSpecDraftK =
    (STRATA_INKLING2_MAX_SPEC_DRAFT_K > STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K)
        ? STRATA_INKLING2_MAX_SPEC_DRAFT_K
        : STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K;

static constexpr uint32_t kMaxReplSeqLenCap =
    (STRATA_INKLING2_MAX_SEQ_LEN_CAP >
        ((STRATA_GLM53_MAX_SEQ_LEN_CAP > STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP)
            ? STRATA_GLM53_MAX_SEQ_LEN_CAP
            : STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP))
        ? STRATA_INKLING2_MAX_SEQ_LEN_CAP
        : ((STRATA_GLM53_MAX_SEQ_LEN_CAP > STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP)
            ? STRATA_GLM53_MAX_SEQ_LEN_CAP
            : STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP);

struct ChatCliConfig {
    ChatSubcmdKind           kind                  = ChatSubcmdKind::kChatUnified;
    std::string              bundle_dir;
    std::string              system_prompt;
    std::string              effort                = "low";
    std::string              script_path;
    std::vector<std::string> initial_prompts;
    std::vector<std::string> initial_images;
    bool                     enable_thinking       = true;
    bool                     keep_think            = false;
    bool                     greedy                = false;
    float                    temperature           = 0.6f;
    uint32_t                 top_k                 = 20U;
    float                    top_p                 = 0.95f;
    uint64_t                 seed                  = 42ULL;
    float                    presence_penalty      = 0.0f;
    float                    frequency_penalty     = 0.0f;
    float                    repetition_penalty    = 1.0f;
    uint32_t                 penalty_window        = 0U;
    uint32_t                 draft_k               = 2U;
    bool                     draft_k_explicit      = false;
    bool                     int8_prefill          = false;
    bool                     int8_prefill_set      = false;
    bool                     adaptive_k            = true;
    bool                     adaptive_k_set        = false;
    bool                     frspec                = true;
    bool                     frspec_set            = false;
    bool                     tree_spec             = true;
    bool                     bitplane_head         = true;
    uint32_t                 max_tokens            = 8192U; /* Default 8192 (0 = unlimited up to remaining max_seq_len) */
    bool                     max_tok_explicit      = false;
    uint32_t                 max_seq_len           = 0U;
    uint32_t                 kv_dtype              = STRATA_INKLING2_KV_DTYPE_F16;
    size_t                   ple_cache_mib         = 0U;
    StrataGlm53ExecMode      glm53_mode            = STRATA_GLM53_MODE_LOSSLESS;
    bool                     glm53_mode_explicit   = false;
    float                    hot_cache_gib         = STRATA_GLM53_DEFAULT_HOT_CACHE_GIB;
    float                    gate_tau              = STRATA_GLM53_DEFAULT_ADAPTIVE_TAU;
    uint32_t                 cold_cap              = STRATA_GLM53_DEFAULT_ADAPTIVE_COLD_CAP;
    bool                     adaptive_cfg_explicit = false;
    bool                     verify_multiturn      = false;
    bool                     verify_k0             = false;
    bool                     once                  = false;
    bool                     show_help             = false;
    /* ---- Step 4 Two-Tier Realtime Compaction (default off => zero behavior change) ---- */
    StrataCompactionMode     compact_mode          = STRATA_COMPACT_OFF;
    bool                     compact_mode_explicit = false;
    uint32_t                 compact_pool          = 32768U;  /* Tier 2 UMA pool tokens per paged slot */
    uint32_t                 compact_window        = 32768U;  /* Tier 1 C_max */
    bool                     compact_window_explicit = false;
    float                    compact_watermark     = 0.80f;   /* W_swap = watermark * window */
    bool                     verify_paged_kv       = false;
    bool                     verify_shadow_swap    = false;
    bool                     verify_save_load      = false;
    bool                     bench_compaction_rule = false;
    bool                     bench_longform_64k    = false;
    bool                     compact_pool_explicit = false;
    bool                     think_explicit        = false;
    bool                     effort_explicit       = false;
    std::string              load_session_path;
    std::string              bench_custom_rule;
    std::string              bench_custom_expect;
};

static constexpr uint32_t kCompactWindowMin   = 256U;
static constexpr float    kCompactWatermarkLo = 0.50f;
static constexpr float    kCompactWatermarkHi = 0.95f;
static constexpr uint32_t kVerifyPagedKvSeqLen = 20480U;
static constexpr uint32_t kVerifyShadowWindow  = 2048U;
static constexpr uint32_t kBenchRuleWindow     = 2048U;
static constexpr uint32_t kBenchLongform64kWin = 65536U;
static constexpr uint32_t kBenchLongform64kPool= 32768U;

static const char* chat_compact_mode_name(StrataCompactionMode m) {
    switch (m) {
        case STRATA_COMPACT_SHADOW: return "shadow";
        case STRATA_COMPACT_PAGED:  return "paged";
        case STRATA_COMPACT_HYBRID: return "hybrid";
        case STRATA_COMPACT_OFF:
        default:                    return "off";
    }
}

static bool chat_parse_compact_mode(const std::string& s, StrataCompactionMode* out) {
    const std::string m = chat_to_lower_ascii(s);
    if (m == "off" || m == "none" || m == "0")            { *out = STRATA_COMPACT_OFF;    return true; }
    if (m == "shadow" || m == "tier1" || m == "rcu")      { *out = STRATA_COMPACT_SHADOW; return true; }
    if (m == "paged" || m == "tier2")                     { *out = STRATA_COMPACT_PAGED;  return true; }
    if (m == "hybrid" || m == "both")                     { *out = STRATA_COMPACT_HYBRID; return true; }
    return false;
}

/* Engine `max_seq_len` for chat-flash: explicit flag wins; `--verify-paged-kv` => 20,480; shadow mode (or
 * `--verify-shadow-swap`) => max(65,536, 2 * window) so the 3 linear slots (live/shadow/rollback) fit. */
static uint32_t chat_resolve_flash_max_seq_len(const ChatCliConfig& cfg) {
    uint32_t v = STRATA_QWEN_FLASH_DEFAULT_MAX_SEQ_LEN;
    if (cfg.max_seq_len > 0U) {
        v = cfg.max_seq_len;
    } else if (cfg.verify_paged_kv) {
        v = kVerifyPagedKvSeqLen;
    } else if (cfg.compact_mode == STRATA_COMPACT_SHADOW ||
               cfg.compact_mode == STRATA_COMPACT_HYBRID ||
               cfg.verify_shadow_swap ||
               cfg.bench_compaction_rule ||
               cfg.bench_longform_64k) {
        v = std::max<uint32_t>(65536U, 2U * cfg.compact_window);
    }
    return std::min<uint32_t>(v, STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP);
}

static uint32_t chat_resolve_glm53_max_seq_len(const ChatCliConfig& cfg) {
    uint32_t v = STRATA_GLM53_DEFAULT_MAX_SEQ_LEN;
    if (cfg.max_seq_len > 0U) {
        v = cfg.max_seq_len;
    } else if (cfg.verify_paged_kv) {
        v = kVerifyPagedKvSeqLen;
    } else if (cfg.compact_mode == STRATA_COMPACT_SHADOW ||
               cfg.compact_mode == STRATA_COMPACT_HYBRID ||
               cfg.verify_shadow_swap) {
        v = std::max<uint32_t>(STRATA_GLM53_DEFAULT_MAX_SEQ_LEN, 2U * cfg.compact_window);
    }
    return std::min<uint32_t>(v, STRATA_GLM53_MAX_SEQ_LEN_CAP);
}

static const char* chat_default_qwen38_bundle_dir() {
    return strata_apex_bundle_path("qwen3.8-27b");
}

static void print_chat_help(const char* prog) {
    std::printf(
        "STRATA-M5 v2.0 — Stateful Multi-Turn Streaming Chat REPL (`Mac17,10` / Apple M5 Max)\n"
        "\n"
        "Usage:\n"
        "  %s chat-glm53 [--bundle <dir>] [--mode lossless|adaptive] [--hot-cache-gib 76.0] [--draft-k 0..4]\n"
        "                [--int8-prefill] [--adaptive-k | --no-adaptive-k] [--frspec | --no-frspec]\n"
        "                [--think | --no-think] [--effort low|high|max] [--greedy | --temp 0.6]\n"
        "                [--max-tokens 8192] [--max-seq-len 33792] [--system \"<text>\"] [--keep-think]\n"
        "                [--prompt \"<text>\" | --prompt-file <path>] [--script <path>] [--load <session>] [--once]\n"
        "                [--compact-mode off|shadow|paged|hybrid] [--compact-pool 32768] [--compact-window 32768]\n"
        "                [--verify-k0] [--verify-multiturn] [--verify-paged-kv] [--verify-shadow-swap] [--verify-save-load]\n"
        "  %s chat-flash [--bundle <dir>] [--draft-k 0..4] [--int8-prefill] [--adaptive-k | --no-adaptive-k]\n"
        "                [--frspec | --no-frspec] [--think | --no-think] [--effort low|medium|xhigh]\n"
        "                [--greedy | --temp 0.6] [--max-tokens 8192] [--max-seq-len 263168] [--system \"<text>\"]\n"
        "                [--keep-think] [--prompt \"<text>\" | --prompt-file <path>] [--script <path>] [--load <session>]\n"
        "                [--once] [--verify-k0] [--verify-multiturn] [--verify-save-load]\n"
        "                [--compact-mode off|shadow|paged|hybrid] [--compact-pool 32768] [--compact-window 32768]\n"
        "                [--compact-watermark 0.80] [--verify-paged-kv] [--verify-shadow-swap]\n"
        "  %s chat-hf    [--bundle <dir>] [--draft-k 0..4] [--think | --no-think] [--greedy | --temp 0.6]\n"
        "                [--max-tokens 8192] [--max-seq-len 32768] [--system \"<text>\"] [--prompt \"<text>\"]\n"
        "                [--script <path>] [--load <session>] [--once] [--verify-multiturn] [--verify-save-load]\n"
        "  %s chat-inkling2 [--bundle <dir>] [--kv-cache f16|q8] [--int8-prefill] [--draft-k 0..7]\n"
        "                   [--adaptive-k | --no-adaptive-k] [--frspec | --no-frspec]\n"
        "                   [--tree-spec | --no-tree-spec] [--bitplane-head | --no-bitplane-head]\n"
        "                   [--image <path>] [--think | --no-think] [--effort none|minimal|low|medium|high|xhigh]\n"
        "                   [--greedy | --temp 0.6] [--max-tokens 8192] [--max-seq-len 65536] [--system \"<text>\"]\n"
        "                   [--prompt \"<text>\"] [--script <path>] [--once]\n"
        "  %s chat       [--bundle <dir>] [--kv-cache f16|q8] [auto-detects Inkling-2, GLM-5.3, Flash-Next, or Qwen3.8-27B]\n"
        "\n"
        "Defaults:\n"
        "  chat-glm53        : --bundle %s --draft-k 2 --max-seq-len %u\n"
        "  chat / chat-flash : --bundle %s --draft-k 2 --max-seq-len %u\n"
        "  chat-hf           : --bundle %s (auto-detects GLM-5.3 / Flash-Next if spine/part0 present)\n"
        "\n"
        "Interactive Slash Commands:\n"
        "  /help               Show available slash commands and active generation settings\n"
        "  /stats              Print live UMA/NVMe/ANE/MTP/KDA/DeltaNet/KV residency, tok/s & page-fault telemetry\n"
        "  /think [effort]     Enable <think>...</think> reasoning (effort: low | medium | high | xhigh | max)\n"
        "  /no-think           Disable thinking block (direct immediate assistant response)\n"
        "  /keep-think [0|1]   Toggle preserving <think> in multi-turn history vs compaction\n"
        "  /mode <lossless|adaptive> Switch GLM-5.3 Regime II execution mode at zero cost\n"
        "  /draft-k <0..4>     Set Native MTP speculative draft length K\n"
        "  /temp <float>       Set sampling temperature (<= 0 switches to deterministic greedy)\n"
        "  /greedy             Switch to deterministic greedy decoding (T = 0)\n"
        "  /max-tokens <N>     Set maximum generated tokens per assistant turn (0 = unlimited up to max_seq_len)\n"
        "  /system <text>      Update system prompt and reset state to seq_pos = 0\n"
        "  /image <path>       Attach an image to the next turn (chat-inkling2 Vision Encoder; optional inline prompt)\n"
        "  /undo               Instant O(1) GPU checkpoint rewind (< 0.5 ms) of last user+assistant turn\n"
        "  /reset              Instant O(1) GPU state reset (seq_pos = 0) without reloading weights\n"
        "  /compact-mode <off|shadow|paged|hybrid> Switch Step 4 realtime compaction tier (chat-flash / chat-glm53)\n"
        "  /compact-stats      Tier 1 + Tier 2 telemetry (logical pos, UMA/NVMe blocks, ANE, ledger, shadow, swaps)\n"
        "  /compact            Force shadow promotion now (finishes warmup synchronously if needed)\n"
        "  /undo-compact       O(1) rebind of the pre-swap rollback slot (valid until the next turn)\n"
        "  /save <path>        Save full persistent `.strata_session` snapshot (GPU/NVMe KV + DeltaNet/KDA + Ledger)\n"
        "  /load <path>        Restore a `.strata_session` snapshot in <250 ms without reloading model weights\n"
        "  /exit, /quit        Exit the REPL cleanly (Ctrl-D / EOF & multi-line bracketed paste supported)\n"
        "\n"
        "Step 4 Compaction (chat-flash / chat-glm53; default --compact-mode off => zero behavior change):\n"
        "  --compact-mode shadow|hybrid enable the Tier 1 cooperative idle loop: on a TTY, stdin is read with\n"
        "  poll() in cbreak mode and background shadow-warmup slices run only while no key is pending (the first\n"
        "  keystroke pauses them); with piped/--script stdin all pending idle quanta run between lines as a\n"
        "  simulated idle gap. --compact-mode shadow without --max-seq-len uses max(65536, 2*window).\n",
        prog, prog, prog, prog, prog,
        STRATA_GLM53_DEFAULT_BUNDLE_DIR,
        STRATA_GLM53_DEFAULT_MAX_SEQ_LEN,
        STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR,
        STRATA_QWEN_FLASH_DEFAULT_MAX_SEQ_LEN,
        chat_default_qwen38_bundle_dir());
}

static bool is_valid_effort_str(const std::string& e) {
    return (e == "low" || e == "medium" || e == "high" || e == "xhigh" || e == "max");
}

static int parse_chat_cli_args(
    int argc,
    char** argv,
    ChatSubcmdKind kind,
    ChatCliConfig* cfg_out,
    std::string* err_out
) {
    if (!cfg_out) return -1;
    ChatCliConfig cfg{};
    cfg.kind = kind;
    if (kind == ChatSubcmdKind::kChatHf) {
        cfg.bundle_dir = chat_default_qwen38_bundle_dir();
        cfg.draft_k    = 0U;
    } else if (kind == ChatSubcmdKind::kChatGlm53) {
        cfg.bundle_dir = chat_default_glm53_bundle();
        cfg.draft_k    = chat_is_glm53_fast_bundle(cfg.bundle_dir) ? STRATA_GLM53_MAX_SPEC_DRAFT_K : 2U;
    } else if (kind == ChatSubcmdKind::kChatInkling2) {
        cfg.bundle_dir = chat_default_inkling2_bundle();
        cfg.draft_k    = STRATA_INKLING2_MAX_SPEC_DRAFT_K;
        cfg.effort     = "medium";
    } else {
        cfg.bundle_dir = STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR;
        cfg.draft_k    = 3U;
    }

    auto fail = [&](const std::string& msg) -> int {
        if (err_out) *err_out = msg;
        return -1;
    };

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            cfg.show_help = true;
            *cfg_out = cfg;
            return 0;
        } else if (a == "--bundle" && i + 1 < argc) {
            cfg.bundle_dir = argv[++i];
        } else if (a == "--system" && i + 1 < argc) {
            cfg.system_prompt = argv[++i];
        } else if ((a == "--prompt" || a == "-p") && i + 1 < argc) {
            cfg.initial_prompts.emplace_back(argv[++i]);
        } else if (a == "--image" && i + 1 < argc) {
            cfg.initial_images.emplace_back(argv[++i]);
        } else if (a == "--prompt-file" && i + 1 < argc) {
            const std::string ppath = argv[++i];
            std::ifstream pifs(ppath, std::ios::in | std::ios::binary);
            if (!pifs) {
                return fail("Failed to open --prompt-file `" + ppath + "`");
            }
            std::ostringstream poss;
            poss << pifs.rdbuf();
            cfg.initial_prompts.emplace_back(poss.str());
        } else if (a == "--script" && i + 1 < argc) {
            cfg.script_path = argv[++i];
        } else if (a == "--think") {
            cfg.enable_thinking = true;
            cfg.think_explicit  = true;
        } else if (a == "--no-think") {
            cfg.enable_thinking = false;
            cfg.think_explicit  = true;
        } else if (a == "--keep-think") {
            cfg.keep_think = true;
        } else if ((a == "--effort" || a == "--reasoning-effort") && i + 1 < argc) {
            cfg.effort          = chat_to_lower_ascii(argv[++i]);
            cfg.effort_explicit = true;
            if (!is_valid_effort_str(cfg.effort)) {
                return fail("Invalid --effort `" + cfg.effort + "` (expected low, medium, high, xhigh, or max)");
            }
        } else if (a == "--mode" && i + 1 < argc) {
            const std::string ms = argv[++i];
            if (!chat_parse_glm53_mode(ms, &cfg.glm53_mode)) {
                return fail("Invalid --mode `" + ms + "` (expected lossless or adaptive)");
            }
            cfg.glm53_mode_explicit = true;
        } else if (a == "--hot-cache-gib" && i + 1 < argc) {
            char* endp = nullptr;
            float hc = std::strtof(argv[++i], &endp);
            if (!endp || *endp != '\0' || std::isnan(hc)) return fail("Invalid --hot-cache-gib");
            cfg.hot_cache_gib = hc;
        } else if (a == "--gate-tau" && i + 1 < argc) {
            char* endp = nullptr;
            float gt = std::strtof(argv[++i], &endp);
            if (!endp || *endp != '\0' || std::isnan(gt)) return fail("Invalid --gate-tau");
            cfg.gate_tau              = gt;
            cfg.adaptive_cfg_explicit = true;
        } else if (a == "--cold-cap" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long cc = std::strtoul(argv[++i], &endp, 10);
            if (!endp || *endp != '\0') return fail("Invalid --cold-cap");
            cfg.cold_cap              = static_cast<uint32_t>(cc);
            cfg.adaptive_cfg_explicit = true;
        } else if ((a == "--draft-k" || a == "--mtp-k") && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (!endp || *endp != '\0' || v > kChatMaxSpecDraftK) {
                return fail("Invalid --draft-k (expected integer in 0.." + std::to_string(kChatMaxSpecDraftK) + ")");
            }
            cfg.draft_k          = static_cast<uint32_t>(v);
            cfg.draft_k_explicit = true;
        } else if (a == "--kv-cache" && i + 1 < argc) {
            const std::string kv = argv[++i];
            if (!chat_parse_kv_cache(kv, &cfg.kv_dtype)) {
                return fail("Invalid --kv-cache `" + kv + "` (expected f16 or q8)");
            }
        } else if (a == "--int8-prefill") {
            cfg.int8_prefill     = true;
            cfg.int8_prefill_set = true;
        } else if (a == "--fp16-prefill" || a == "--no-int8-prefill") {
            cfg.int8_prefill     = false;
            cfg.int8_prefill_set = true;
        } else if (a == "--adaptive-k") {
            cfg.adaptive_k     = true;
            cfg.adaptive_k_set = true;
        } else if (a == "--no-adaptive-k") {
            cfg.adaptive_k     = false;
            cfg.adaptive_k_set = true;
        } else if (a == "--frspec" || a == "--fr-spec") {
            cfg.frspec     = true;
            cfg.frspec_set = true;
        } else if (a == "--no-frspec" || a == "--no-fr-spec") {
            cfg.frspec     = false;
            cfg.frspec_set = true;
        } else if (a == "--tree-spec" || a == "--beam-tree") {
            cfg.tree_spec = true;
        } else if (a == "--no-tree-spec" || a == "--no-beam-tree") {
            cfg.tree_spec = false;
        } else if (a == "--bitplane-head") {
            cfg.bitplane_head = true;
        } else if (a == "--no-bitplane-head") {
            cfg.bitplane_head = false;
        } else if (a == "--greedy") {
            cfg.greedy      = true;
            cfg.temperature = 0.0f;
        } else if ((a == "--temp" || a == "--temperature") && i + 1 < argc) {
            char* endp = nullptr;
            float t = std::strtof(argv[++i], &endp);
            if (!endp || *endp != '\0' || std::isnan(t) || t < 0.0f) {
                return fail("Invalid --temp (expected non-negative float)");
            }
            cfg.temperature = t;
            cfg.greedy      = (t <= 1e-6f);
        } else if (a == "--top-k" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (!endp || *endp != '\0') return fail("Invalid --top-k");
            cfg.top_k = static_cast<uint32_t>(v);
        } else if (a == "--top-p" && i + 1 < argc) {
            char* endp = nullptr;
            float p = std::strtof(argv[++i], &endp);
            if (!endp || *endp != '\0' || p <= 0.0f || p > 1.0f) {
                return fail("Invalid --top-p (expected (0, 1])");
            }
            cfg.top_p = p;
        } else if (a == "--presence-penalty" && i + 1 < argc) {
            char* endp = nullptr;
            float v = std::strtof(argv[++i], &endp);
            if (!endp || *endp != '\0' || std::isnan(v) || v < -2.0f || v > 2.0f) {
                return fail("Invalid --presence-penalty (expected -2.0..2.0)");
            }
            cfg.presence_penalty = v;
        } else if (a == "--frequency-penalty" && i + 1 < argc) {
            char* endp = nullptr;
            float v = std::strtof(argv[++i], &endp);
            if (!endp || *endp != '\0' || std::isnan(v) || v < -2.0f || v > 2.0f) {
                return fail("Invalid --frequency-penalty (expected -2.0..2.0)");
            }
            cfg.frequency_penalty = v;
        } else if ((a == "--repetition-penalty" || a == "--rep-penalty") && i + 1 < argc) {
            char* endp = nullptr;
            float v = std::strtof(argv[++i], &endp);
            if (!endp || *endp != '\0' || std::isnan(v) || v < 0.5f || v > 2.0f) {
                return fail("Invalid --repetition-penalty (expected 0.5..2.0)");
            }
            cfg.repetition_penalty = v;
        } else if (a == "--penalty-window" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (!endp || *endp != '\0' || (v > 0UL && v < 8UL) || v > 65536UL) {
                return fail("Invalid --penalty-window (expected 0 or 8..65536)");
            }
            cfg.penalty_window = static_cast<uint32_t>(v);
        } else if (a == "--seed" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long long s = std::strtoull(argv[++i], &endp, 10);
            if (!endp || *endp != '\0') return fail("Invalid --seed");
            cfg.seed = static_cast<uint64_t>(s);
        } else if (a == "--max-tokens" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (!endp || *endp != '\0' || v > kMaxReplSeqLenCap) {
                return fail("Invalid --max-tokens (expected 0.." + std::to_string(kMaxReplSeqLenCap) + ")");
            }
            cfg.max_tokens       = static_cast<uint32_t>(v);
            cfg.max_tok_explicit = true;
        } else if (a == "--max-seq-len" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (!endp || *endp != '\0' || v < 128UL || v > kMaxReplSeqLenCap) {
                return fail("Invalid --max-seq-len (expected 128.." +
                            std::to_string(kMaxReplSeqLenCap) + ")");
            }
            cfg.max_seq_len = static_cast<uint32_t>(v);
        } else if (a == "--ple-cache-mib" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long long v = std::strtoull(argv[++i], &endp, 10);
            if (!endp || *endp != '\0') return fail("Invalid --ple-cache-mib");
            cfg.ple_cache_mib = static_cast<size_t>(v);
        } else if (a == "--verify-multiturn") {
            cfg.verify_multiturn = true;
        } else if (a == "--verify-k0") {
            cfg.verify_k0 = true;
        } else if (a == "--once" || a == "--no-interactive") {
            cfg.once = true;
        } else if (a == "--compact-mode" && i + 1 < argc) {
            const std::string ms = argv[++i];
            if (!chat_parse_compact_mode(ms, &cfg.compact_mode)) {
                return fail("Invalid --compact-mode `" + ms + "` (expected off, shadow, paged, or hybrid)");
            }
            cfg.compact_mode_explicit = true;
        } else if (a == "--compact-pool" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (!endp || *endp != '\0' || v < STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS ||
                v > STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP) {
                return fail("Invalid --compact-pool (expected " +
                            std::to_string(STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS) + ".." +
                            std::to_string(STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP) + " tokens)");
            }
            cfg.compact_pool          = static_cast<uint32_t>(v);
            cfg.compact_pool_explicit = true;
        } else if (a == "--compact-window" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (!endp || *endp != '\0' || v < kCompactWindowMin || v > STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP) {
                return fail("Invalid --compact-window (expected " + std::to_string(kCompactWindowMin) + ".." +
                            std::to_string(STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP) + " tokens)");
            }
            cfg.compact_window          = static_cast<uint32_t>(v);
            cfg.compact_window_explicit = true;
        } else if (a == "--compact-watermark" && i + 1 < argc) {
            char* endp = nullptr;
            float w = std::strtof(argv[++i], &endp);
            if (!endp || *endp != '\0' || std::isnan(w) || w < kCompactWatermarkLo || w > kCompactWatermarkHi) {
                return fail("Invalid --compact-watermark (expected 0.5..0.95)");
            }
            cfg.compact_watermark = w;
        } else if (a == "--verify-paged-kv") {
            cfg.verify_paged_kv = true;
        } else if (a == "--verify-shadow-swap") {
            cfg.verify_shadow_swap = true;
        } else if (a == "--verify-save-load") {
            cfg.verify_save_load = true;
        } else if (a == "--load" && i + 1 < argc) {
            cfg.load_session_path = argv[++i];
        } else if (a == "--bench-compaction-rule") {
            cfg.bench_compaction_rule = true;
        } else if (a == "--bench-longform-64k") {
            cfg.bench_longform_64k = true;
        } else if (a == "--bench-rule" && i + 1 < argc) {
            cfg.bench_custom_rule     = argv[++i];
            cfg.bench_compaction_rule = true;
        } else if (a == "--bench-expect" && i + 1 < argc) {
            cfg.bench_custom_expect   = argv[++i];
            cfg.bench_compaction_rule = true;
        } else {
            return fail("Unknown or incomplete option: " + a);
        }
    }

    const bool is_ik2 = (kind == ChatSubcmdKind::kChatInkling2) || chat_is_inkling2_bundle(cfg.bundle_dir);
    if (!cfg.draft_k_explicit) {
        if (is_ik2) {
            cfg.draft_k = STRATA_INKLING2_MAX_SPEC_DRAFT_K;
        } else if (chat_is_flash_bundle(cfg.bundle_dir)) {
            cfg.draft_k = 3U;
        } else if (chat_is_glm53_fast_bundle(cfg.bundle_dir)) {
            cfg.draft_k = STRATA_GLM53_MAX_SPEC_DRAFT_K;
        } else if (chat_is_glm53_bundle(cfg.bundle_dir)) {
            cfg.draft_k = 2U;
        }
    } else if (!is_ik2 && cfg.draft_k > STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K) {
        return fail("Invalid --draft-k (expected integer in 0..4 for this model)");
    }
    if (cfg.verify_multiturn || cfg.verify_k0 || cfg.verify_paged_kv ||
        cfg.verify_shadow_swap || cfg.verify_save_load ||
        cfg.bench_compaction_rule || cfg.bench_longform_64k) {
        cfg.greedy      = true;
        cfg.temperature = 0.0f;
    }
    if (cfg.verify_shadow_swap) {
        /* Honors --compact-mode shadow|hybrid; anything else runs the default Tier 1 shadow mode */
        if (cfg.compact_mode != STRATA_COMPACT_SHADOW && cfg.compact_mode != STRATA_COMPACT_HYBRID) {
            cfg.compact_mode = STRATA_COMPACT_SHADOW;
        }
        if (!cfg.compact_window_explicit) {
            cfg.compact_window = kVerifyShadowWindow;
        }
    }
    if (cfg.verify_save_load) {
        if ((cfg.compact_mode == STRATA_COMPACT_SHADOW || cfg.compact_mode == STRATA_COMPACT_HYBRID) &&
            !cfg.compact_window_explicit) {
            cfg.compact_window = kVerifyShadowWindow;
        }
        if ((cfg.compact_mode == STRATA_COMPACT_PAGED || cfg.compact_mode == STRATA_COMPACT_HYBRID) &&
            !cfg.compact_pool_explicit) {
            cfg.compact_pool = STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS;
        }
    }
    if (cfg.bench_compaction_rule || cfg.bench_longform_64k) {
        if (cfg.compact_mode != STRATA_COMPACT_SHADOW && cfg.compact_mode != STRATA_COMPACT_HYBRID) {
            cfg.compact_mode = STRATA_COMPACT_HYBRID;
        }
        if (!cfg.compact_window_explicit) {
            cfg.compact_window = cfg.bench_longform_64k ? kBenchLongform64kWin : kBenchRuleWindow;
        }
        if (cfg.bench_longform_64k && !cfg.compact_pool_explicit) {
            cfg.compact_pool = kBenchLongform64kPool;
        }
        if (!cfg.think_explicit) {
            cfg.enable_thinking = true;
        }
        if (!cfg.effort_explicit) {
            cfg.effort = "xhigh";
        }
        if (!cfg.max_tok_explicit) {
            cfg.max_tokens = 1536U;
        }
    }

    *cfg_out = cfg;
    return 0;
}

/* ============================================================================
 * 2. Canonical ChatML / GLM-5.3 Formatting & `<think>...</think>` Compaction
 * ============================================================================ */

static std::string format_flash_system_block(
    const std::string& system_prompt,
    bool enable_thinking,
    const std::string& effort
) {
    std::string reasoning_instructions;
    if (enable_thinking) {
        const std::string eff = effort.empty() ? "low" : chat_to_lower_ascii(effort);
        if (eff == "low") {
            reasoning_instructions =
                "Reasoning effort is set to low. Keep your thinking brief and focused, "
                "moving directly to the conclusion without unnecessary elaboration.";
        } else if (eff == "xhigh" || eff == "max") {
            reasoning_instructions =
                "Reasoning effort is set to xhigh. Please think carefully through the task, "
                "validate key assumptions, consider plausible alternatives, and prioritize "
                "correctness, consistency, and clarity in the final answer.";
        }
    }
    const std::string sys = chat_trim_ascii(system_prompt);
    if (!sys.empty() && !reasoning_instructions.empty()) {
        return "<|im_start|>system\n" + reasoning_instructions + "\n\n" + sys + "<|im_end|>\n";
    }
    if (!sys.empty()) {
        return "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }
    if (!reasoning_instructions.empty()) {
        return "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
    }
    return {};
}

static std::string format_qwen38_system_block(const std::string& system_prompt) {
    const std::string sys = chat_trim_ascii(system_prompt);
    if (sys.empty()) return {};
    return "<|im_start|>system\n" + sys + "<|im_end|>\n";
}

static std::string format_glm53_system_block(
    const std::string& system_prompt,
    const std::string& effort
) {
    const std::string el = chat_to_lower_ascii(effort);
    std::string eff_cap  = "Low";
    if (el == "medium" || el == "high") {
        eff_cap = "High";
    } else if (el == "xhigh" || el == "max") {
        eff_cap = "Max";
    }
    std::string out = "[gMASK]<sop><|system|>Reasoning Effort: " + eff_cap;
    const std::string sys = chat_trim_ascii(system_prompt);
    if (!sys.empty()) {
        out += "<|system|>" + sys;
    }
    return out;
}

static std::string build_committed_assistant_suffix(
    const std::string& think_text_trimmed,
    const std::string& clean_reply_trimmed,
    bool keep_think
) {
    if (keep_think) {
        return "<think>\n" + think_text_trimmed + "\n</think>\n\n" + clean_reply_trimmed + "<|im_end|>\n";
    }
    return clean_reply_trimmed + "<|im_end|>\n";
}

static std::string build_glm53_committed_assistant_suffix(
    const std::string& think_text_trimmed,
    const std::string& clean_reply_trimmed,
    bool enable_thinking,
    bool keep_think
) {
    if (enable_thinking && keep_think) {
        return "<think>" + think_text_trimmed + "</think>" + clean_reply_trimmed;
    }
    return "<think></think>" + clean_reply_trimmed;
}

enum class SlashCmdAction {
    kNotSlash = 0,
    kHandled  = 1,
    kUndo     = 2,
    kReset    = 3,
    kSystem   = 4,
    kStats    = 5,
    kExit     = 6,
    kError    = 7,
    kMode     = 8,
    kCompactMode  = 9,
    kCompactStats = 10,
    kCompact      = 11,
    kUndoCompact  = 12,
    kSaveSession  = 13,
    kLoadSession  = 14,
    kImage        = 15,
};

static SlashCmdAction parse_slash_command(
    const std::string& raw_line,
    bool model_has_mtp,
    ChatCliConfig* cfg,
    std::string* msg_out
) {
    const std::string trimmed = chat_trim_ascii(raw_line);
    if (trimmed.empty() || trimmed[0] != '/') {
        return SlashCmdAction::kNotSlash;
    }

    size_t sp = 0U;
    while (sp < trimmed.size() && !std::isspace(static_cast<unsigned char>(trimmed[sp]))) {
        ++sp;
    }
    const std::string cmd = trimmed.substr(0U, sp);
    const std::string arg = (sp < trimmed.size()) ? chat_trim_ascii(trimmed.substr(sp)) : std::string();

    if (cmd == "/exit" || cmd == "/quit" || cmd == "/q") {
        return SlashCmdAction::kExit;
    }
    if (cmd == "/undo") {
        return SlashCmdAction::kUndo;
    }
    if (cmd == "/reset" || cmd == "/clear") {
        return SlashCmdAction::kReset;
    }
    if (cmd == "/stats") {
        return SlashCmdAction::kStats;
    }
    if (cmd == "/help" || cmd == "/?") {
        std::ostringstream oss;
        oss << "[help] Commands: /help, /stats, /think [low|medium|high|xhigh|max], /no-think, /keep-think [0|1], "
               "/mode <lossless|adaptive>, /draft-k <0..4>, /temp <f>, /greedy, /max-tokens <n>, /system <text>, /image <path>, /save <path>, /load <path>, /undo, /reset, /exit\n"
               "       Step 4  : /compact-mode <off|shadow|paged|hybrid>, /compact-stats, /compact, /undo-compact\n"
            << "       Active  : " << (cfg->enable_thinking ? ("think (" + cfg->effort + ")") : "no-think")
            << " | keep_think=" << (cfg->keep_think ? "1" : "0")
            << " | mode=" << ((cfg->glm53_mode == STRATA_GLM53_MODE_ADAPTIVE) ? "adaptive" : "lossless")
            << " | " << (cfg->greedy ? "greedy (T=0)" : ("temp=" + std::to_string(cfg->temperature)))
            << " | draft_k=" << (model_has_mtp ? cfg->draft_k : 0U)
            << " | max_tokens=" << (cfg->max_tokens == 0U ? std::string("0 (unlimited)") : std::to_string(cfg->max_tokens))
            << " | compact=" << chat_compact_mode_name(cfg->compact_mode);
        if (msg_out) *msg_out = oss.str();
        return SlashCmdAction::kHandled;
    }
    if (cmd == "/compact-mode") {
        StrataCompactionMode m = STRATA_COMPACT_OFF;
        if (arg.empty() || !chat_parse_compact_mode(arg, &m)) {
            if (msg_out) {
                *msg_out = arg.empty()
                    ? std::string("[compact] Usage: /compact-mode <off|shadow|paged|hybrid>")
                    : ("[compact] Invalid /compact-mode `" + arg + "` (expected off, shadow, paged, or hybrid)");
            }
            return SlashCmdAction::kError;
        }
        cfg->compact_mode = m;
        if (msg_out) *msg_out = std::string("[compact] Requested compaction mode: ") + chat_compact_mode_name(m);
        return SlashCmdAction::kCompactMode;
    }
    if (cmd == "/compact-stats") {
        return SlashCmdAction::kCompactStats;
    }
    if (cmd == "/compact") {
        return SlashCmdAction::kCompact;
    }
    if (cmd == "/undo-compact") {
        return SlashCmdAction::kUndoCompact;
    }
    if (cmd == "/save") {
        if (arg.empty()) {
            if (msg_out) *msg_out = "[session] Usage: /save <path.strata_session>";
            return SlashCmdAction::kError;
        }
        if (msg_out) *msg_out = arg;
        return SlashCmdAction::kSaveSession;
    }
    if (cmd == "/load") {
        if (arg.empty()) {
            if (msg_out) *msg_out = "[session] Usage: /load <path.strata_session>";
            return SlashCmdAction::kError;
        }
        if (msg_out) *msg_out = arg;
        return SlashCmdAction::kLoadSession;
    }
    if (cmd == "/image" || cmd == "/img") {
        if (arg.empty()) {
            if (msg_out) *msg_out = "[vision] Usage: /image <path> [optional prompt]";
            return SlashCmdAction::kError;
        }
        if (msg_out) *msg_out = arg;
        return SlashCmdAction::kImage;
    }
    if (cmd == "/think") {
        if (!arg.empty()) {
            const std::string el = chat_to_lower_ascii(arg);
            if (!is_valid_effort_str(el)) {
                if (msg_out) *msg_out = "[chat] Invalid effort `" + arg + "` (expected low, medium, high, xhigh, or max)";
                return SlashCmdAction::kError;
            }
            cfg->effort = el;
        }
        cfg->enable_thinking = true;
        if (msg_out) *msg_out = "[chat] Thinking enabled (<think>, effort=" + cfg->effort + ")";
        return SlashCmdAction::kHandled;
    }
    if (cmd == "/no-think" || cmd == "/nothink") {
        cfg->enable_thinking = false;
        if (msg_out) *msg_out = "[chat] Thinking disabled (--no-think direct response mode)";
        return SlashCmdAction::kHandled;
    }
    if (cmd == "/keep-think") {
        if (arg.empty()) {
            cfg->keep_think = !cfg->keep_think;
        } else if (arg == "1" || arg == "on" || arg == "true") {
            cfg->keep_think = true;
        } else if (arg == "0" || arg == "off" || arg == "false") {
            cfg->keep_think = false;
        } else {
            if (msg_out) *msg_out = "[chat] Usage: /keep-think [on|off]";
            return SlashCmdAction::kError;
        }
        if (msg_out) {
            *msg_out = std::string("[chat] Multi-turn <think> retention: ") +
                       (cfg->keep_think ? "KEEP" : "COMPACT (strip prior <think>)");
        }
        return SlashCmdAction::kHandled;
    }
    if (cmd == "/mode") {
        if (arg.empty()) {
            if (msg_out) *msg_out = "[chat] Usage: /mode <lossless|adaptive>";
            return SlashCmdAction::kError;
        }
        StrataGlm53ExecMode m = STRATA_GLM53_MODE_LOSSLESS;
        if (!chat_parse_glm53_mode(arg, &m)) {
            if (msg_out) *msg_out = "[chat] Invalid /mode `" + arg + "` (expected lossless or adaptive)";
            return SlashCmdAction::kError;
        }
        cfg->glm53_mode = m;
        if (msg_out) {
            *msg_out = std::string("[chat] Execution mode set to ") +
                       ((m == STRATA_GLM53_MODE_ADAPTIVE) ? "ADAPTIVE (Mode B)" : "LOSSLESS (8/8 exact top-8)");
        }
        return SlashCmdAction::kMode;
    }
    if (cmd == "/draft-k") {
        if (arg.empty()) {
            if (msg_out) *msg_out = "[chat] Usage: /draft-k <0..4>";
            return SlashCmdAction::kError;
        }
        char* endp = nullptr;
        unsigned long v = std::strtoul(arg.c_str(), &endp, 10);
        if (!endp || *endp != '\0' || v > STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K) {
            if (msg_out) *msg_out = "[chat] Invalid /draft-k `" + arg + "` (expected 0..4)";
            return SlashCmdAction::kError;
        }
        cfg->draft_k = model_has_mtp ? static_cast<uint32_t>(v) : 0U;
        if (msg_out) {
            if (model_has_mtp) {
                *msg_out = "[chat] Native MTP speculative draft_k set to " + std::to_string(cfg->draft_k);
            } else {
                *msg_out = "[chat] Model has no MTP head; draft_k remains 0";
            }
        }
        return SlashCmdAction::kHandled;
    }
    if (cmd == "/greedy") {
        cfg->greedy      = true;
        cfg->temperature = 0.0f;
        if (msg_out) *msg_out = "[chat] Sampling set to deterministic greedy (T=0)";
        return SlashCmdAction::kHandled;
    }
    if (cmd == "/temp" || cmd == "/temperature") {
        if (arg.empty()) {
            if (msg_out) *msg_out = "[chat] Usage: /temp <float>";
            return SlashCmdAction::kError;
        }
        char* endp = nullptr;
        float t = std::strtof(arg.c_str(), &endp);
        if (!endp || *endp != '\0' || std::isnan(t) || t < 0.0f) {
            if (msg_out) *msg_out = "[chat] Invalid temperature `" + arg + "` (expected >= 0.0)";
            return SlashCmdAction::kError;
        }
        cfg->temperature = t;
        cfg->greedy      = (t <= 1e-6f);
        if (msg_out) {
            char buf[128];
            if (cfg->greedy) {
                std::snprintf(buf, sizeof(buf), "[chat] Sampling set to greedy (T=0)");
            } else {
                std::snprintf(buf, sizeof(buf), "[chat] Sampling temperature set to %.3f (top_k=%u, top_p=%.2f)",
                              cfg->temperature, cfg->top_k, cfg->top_p);
            }
            *msg_out = buf;
        }
        return SlashCmdAction::kHandled;
    }
    if (cmd == "/max-tokens") {
        if (arg.empty()) {
            if (msg_out) *msg_out = "[chat] Usage: /max-tokens <N> (0 = unlimited up to remaining max_seq_len)";
            return SlashCmdAction::kError;
        }
        char* endp = nullptr;
        unsigned long v = std::strtoul(arg.c_str(), &endp, 10);
        if (!endp || *endp != '\0' || v > kMaxReplSeqLenCap) {
            if (msg_out) {
                *msg_out = "[chat] Invalid /max-tokens `" + arg + "` (expected 0.." +
                           std::to_string(kMaxReplSeqLenCap) + ")";
            }
            return SlashCmdAction::kError;
        }
        cfg->max_tokens = static_cast<uint32_t>(v);
        if (msg_out) {
            if (cfg->max_tokens == 0U) {
                *msg_out = "[chat] max_tokens set to 0 (unlimited up to remaining max_seq_len)";
            } else {
                *msg_out = "[chat] max_tokens set to " + std::to_string(cfg->max_tokens);
            }
        }
        return SlashCmdAction::kHandled;
    }
    if (cmd == "/system") {
        cfg->system_prompt = arg;
        if (msg_out) *msg_out = arg;
        return SlashCmdAction::kSystem;
    }

    if (msg_out) *msg_out = "[chat] Unknown slash command `" + cmd + "` (type /help for commands)";
    return SlashCmdAction::kError;
}

/* ============================================================================
 * 2b. Step 4 Tier 1 Compaction Primitives (EWMA Horizon Predictor, Episode Ledger, Overlap Split)
 * ============================================================================ */

static constexpr double   kCompactEwmaLambda   = 0.30;  /* EWMA weight of the newest turn growth */
static constexpr double   kCompactSigmaK       = 1.5;   /* v+ = v + 1.5 sigma */
static constexpr double   kCompactLeadTurns    = 3.5;   /* launch shadow warmup when K_rem <= 3.5 turns */
static constexpr double   kCompactLaunchFrac   = 0.65;  /* ... or N >= 0.65 C_max */
static constexpr double   kCompactOverlapFrac  = 0.25;  /* verbatim OverlapTail <= ~25% of C_max (>= 1 turn) */
static constexpr double   kCompactBenefitFrac  = 0.85;  /* auto launch only if shadow <= 85% of live tokens */
static constexpr uint32_t kShadowSliceTty      = 32U;   /* preemptible prefill quantum (interactive TTY) */
static constexpr uint32_t kShadowSliceBatch    = 64U;   /* preemptible prefill quantum (pipe / verify) */
static constexpr size_t   kLedgerMaxKv         = 24U;
static constexpr size_t   kLedgerMaxIds        = 16U;
static constexpr size_t   kLedgerMaxHex        = 16U;
static constexpr size_t   kLedgerMaxNums       = 16U;
static constexpr size_t   kLedgerGistChars     = 110U;

/*
 * EWMA token-velocity predictor (study 2.1.A):
 *   v  = l*dN + (1-l)*v ;  s2 = l*(dN - v)^2 + (1-l)*s2 ;  v+ = v + 1.5*sqrt(s2) ;  K_rem = (W_swap - N) / v+
 * The first observation seeds v = dN, s2 = 0 (avoids the cold-start underestimate of a zero prior).
 */
struct CompactVelocityPredictor {
    double   v       = 0.0;
    double   s2      = 0.0;
    uint32_t samples = 0U;

    void reset() {
        v       = 0.0;
        s2      = 0.0;
        samples = 0U;
    }
    void observe(double dn) {
        if (samples == 0U) {
            v  = dn;
            s2 = 0.0;
        } else {
            v = kCompactEwmaLambda * dn + (1.0 - kCompactEwmaLambda) * v;
            const double e = dn - v;
            s2 = kCompactEwmaLambda * e * e + (1.0 - kCompactEwmaLambda) * s2;
        }
        ++samples;
    }
    double v_plus() const { return v + kCompactSigmaK * std::sqrt(std::max(0.0, s2)); }
    double k_rem(double w_swap, double n) const {
        if (n >= w_swap) return 0.0;
        const double vp = v_plus();
        if (samples == 0U || vp <= 1e-9) return std::numeric_limits<double>::infinity();
        return (w_swap - n) / vp;
    }
};

/* ---- Immutable Episode Ledger: deterministic exact-fact distiller ---- */

struct LedgerFacts {
    std::vector<std::string> kv;    /* exact "KEY=VALUE" (from KEY=VALUE, KEY = VALUE, KEY: VALUE) */
    std::vector<std::string> ids;   /* UPPER_SNAKE / snake_case / hyphenated codes (HELIOS-9) / ALLCAPS */
    std::vector<std::string> hex;   /* 0x2A7 */
    std::vector<std::string> nums;  /* numeric constants with >= 2 digits (4417, 3.25, 2026-11-19) */
};

struct LedgerEntry {
    uint32_t                 turn_index    = 0U; /* 0-based index into the session history */
    LedgerFacts              facts;
    std::vector<std::string> rules;              /* standing behavioral / formatting requirements from user_text */
    std::string              semantic_summary;   /* idle-gap 180B LLM semantic summary (captures subtle prose nuance) */
    bool                     semantic_done = true;
    std::string              text;               /* rendered once at turn close, never re-summarized */
};

static inline bool ledger_is_alnum(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0;
}

/* Token chars: [A-Za-z0-9_]; '-' and '.' only when joining two alphanumerics. */
static bool ledger_is_tok_char(const std::string& s, size_t i) {
    const char c = s[i];
    if (ledger_is_alnum(c) || c == '_') return true;
    if (c == '-' || c == '.') {
        return i > 0U && i + 1U < s.size() && ledger_is_alnum(s[i - 1U]) && ledger_is_alnum(s[i + 1U]);
    }
    return false;
}

enum class LedgerTokKind { kNone = 0, kIdent = 1, kHex = 2, kNum = 3 };

static LedgerTokKind ledger_classify_token(const std::string& t) {
    if (t.empty()) return LedgerTokKind::kNone;
    if (t.size() >= 3U && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
        bool all_hex = true;
        for (size_t k = 2U; k < t.size(); ++k) {
            if (std::isxdigit(static_cast<unsigned char>(t[k])) == 0) {
                all_hex = false;
                break;
            }
        }
        if (all_hex) return LedgerTokKind::kHex;
    }
    size_t n_upper = 0U, n_lower = 0U, n_digit = 0U, n_us = 0U, n_dash = 0U;
    bool all_xdig = true;
    for (char c : t) {
        if (c >= 'A' && c <= 'Z') ++n_upper;
        else if (c >= 'a' && c <= 'z') ++n_lower;
        else if (c >= '0' && c <= '9') ++n_digit;
        else if (c == '_') ++n_us;
        else if (c == '-') ++n_dash;
        if (std::isxdigit(static_cast<unsigned char>(c)) == 0) all_xdig = false;
    }
    if (all_xdig && t.size() >= 6U && n_digit > 0U && (n_upper + n_lower) > 0U) {
        return LedgerTokKind::kHex; /* unprefixed /dev/urandom hex passkeys e.g. 4f9a2c81b704e5d6 */
    }
    if (n_digit > 0U && (n_upper + n_lower) > 0U && t.size() >= 2U) {
        return LedgerTokKind::kIdent; /* mixed alphanumeric passkeys/codes e.g. k9X2mP7qR4, HELIOS-9, alpha-7 */
    }
    if (t[0] >= '0' && t[0] <= '9') {
        return (n_upper == 0U && n_lower == 0U && n_us == 0U && n_digit >= 2U) ? LedgerTokKind::kNum
                                                                                : LedgerTokKind::kNone;
    }
    if (n_us > 0U && (n_upper + n_lower) > 0U) return LedgerTokKind::kIdent;        /* snake / UPPER_SNAKE */
    if (n_lower == 0U && n_upper > 0U) {
        if (t.size() >= 3U || n_digit > 0U || n_dash > 0U) return LedgerTokKind::kIdent; /* AURORA */
    }
    if (n_upper > 0U && n_lower > 0U && (n_upper >= 2U || !(t[0] >= 'A' && t[0] <= 'Z'))) {
        return LedgerTokKind::kIdent; /* CamelCase / internal caps (excludes ordinary Titlecase prose words) */
    }
    return LedgerTokKind::kNone;
}

static void ledger_push_unique(std::vector<std::string>* v, const std::string& s, size_t cap) {
    if (s.empty() || v->size() >= cap) return;
    if (std::find(v->begin(), v->end(), s) == v->end()) v->push_back(s);
}

static inline bool ledger_is_pad(char c) {
    return c == ' ' || c == '\t' || c == '*' || c == '`';
}

static LedgerFacts ledger_extract_facts(const std::string& s) {
    LedgerFacts f;
    const size_t n = s.size();

    /* 1. Exact KEY=VALUE / KEY: VALUE pairs */
    for (size_t i = 0U; i < n; ++i) {
        const char c = s[i];
        if (c != '=' && c != ':') continue;
        if (c == '=') {
            if (i > 0U && (s[i - 1U] == '=' || s[i - 1U] == '!' || s[i - 1U] == '<' || s[i - 1U] == '>')) continue;
            if (i + 1U < n && s[i + 1U] == '=') continue;
        } else {
            if (i + 1U < n && (s[i + 1U] == '/' || s[i + 1U] == ':')) continue;
            if (i > 0U && s[i - 1U] == ':') continue;
        }
        size_t k_end = i;
        while (k_end > 0U && ledger_is_pad(s[k_end - 1U])) --k_end;
        size_t k_start = k_end;
        while (k_start > 0U && ledger_is_tok_char(s, k_start - 1U)) --k_start;
        if (k_start == k_end) continue;
        const std::string key = s.substr(k_start, k_end - k_start);
        const char k0 = key[0];
        if (!((k0 >= 'A' && k0 <= 'Z') || (k0 >= 'a' && k0 <= 'z') || k0 == '_')) continue;
        const LedgerTokKind kk = ledger_classify_token(key);
        if (c == ':' && kk != LedgerTokKind::kIdent) continue;
        if (c == '=' && kk != LedgerTokKind::kIdent && key.size() < 2U) continue;

        size_t v = i + 1U;
        while (v < n && ledger_is_pad(s[v])) ++v;
        if (v >= n) continue;
        std::string value;
        if (s[v] == '"' || s[v] == '\'') {
            const char q = s[v];
            size_t e = v + 1U;
            while (e < n && s[e] != q && s[e] != '\n' && e - v <= 64U) ++e;
            if (e >= n || s[e] != q || e == v + 1U) continue;
            value = s.substr(v, e - v + 1U); /* keep quotes => re-extraction is a fixpoint */
        } else {
            size_t e = v;
            if (s[e] == '-' && e + 1U < n && std::isdigit(static_cast<unsigned char>(s[e + 1U])) != 0) ++e;
            while (e < n && ledger_is_tok_char(s, e)) ++e;
            if (e == v || (e == v + 1U && s[v] == '-')) continue;
            value = s.substr(v, e - v);
        }
        ledger_push_unique(&f.kv, key + "=" + value, kLedgerMaxKv);
    }

    /* 2. Identifiers, hex and numeric constants (single tokenizer pass) */
    size_t i = 0U;
    while (i < n) {
        if (!ledger_is_tok_char(s, i)) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < n && ledger_is_tok_char(s, j)) ++j;
        const std::string tok = s.substr(i, j - i);
        switch (ledger_classify_token(tok)) {
            case LedgerTokKind::kIdent: ledger_push_unique(&f.ids, tok, kLedgerMaxIds); break;
            case LedgerTokKind::kHex:   ledger_push_unique(&f.hex, tok, kLedgerMaxHex); break;
            case LedgerTokKind::kNum:   ledger_push_unique(&f.nums, tok, kLedgerMaxNums); break;
            case LedgerTokKind::kNone:
            default: break;
        }
        i = j;
    }
    return f;
}

static void ledger_merge_facts(LedgerFacts* dst, const LedgerFacts& src) {
    for (const auto& s : src.kv) ledger_push_unique(&dst->kv, s, kLedgerMaxKv);
    for (const auto& s : src.ids) ledger_push_unique(&dst->ids, s, kLedgerMaxIds);
    for (const auto& s : src.hex) ledger_push_unique(&dst->hex, s, kLedgerMaxHex);
    for (const auto& s : src.nums) ledger_push_unique(&dst->nums, s, kLedgerMaxNums);
}

static std::string ledger_render_facts(const LedgerFacts& f) {
    std::string out;
    auto group = [&](const char* label, const std::vector<std::string>& v, const char* sep) {
        if (v.empty()) return;
        if (!out.empty()) out += ' ';
        out += label;
        out += '{';
        for (size_t k = 0U; k < v.size(); ++k) {
            if (k) out += sep;
            out += v[k];
        }
        out += '}';
    };
    group("kv", f.kv, "; ");
    group("ids", f.ids, " ");
    group("hex", f.hex, " ");
    group("num", f.nums, " ");
    return out;
}

/* Collapses whitespace and truncates to `max_chars` at a UTF-8 boundary ("..." when truncated). */
static std::string ledger_gist(const std::string& text, size_t max_chars) {
    std::string flat;
    flat.reserve(std::min(text.size(), max_chars + 8U));
    bool prev_space = true;
    for (char c : text) {
        const bool sp = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (sp) {
            if (!prev_space) flat += ' ';
            prev_space = true;
        } else {
            flat += c;
            prev_space = false;
        }
    }
    while (!flat.empty() && flat.back() == ' ') flat.pop_back();
    if (flat.size() <= max_chars) return flat;
    size_t cut = max_chars;
    while (cut > 0U && (static_cast<unsigned char>(flat[cut]) & 0xC0U) == 0x80U) --cut;
    return flat.substr(0U, cut) + "...";
}

static bool ledger_sentence_is_standing_rule(const std::string& raw) {
    const std::string s = chat_trim_ascii(raw);
    if (s.size() < 12U || s.size() > 640U) return false;
    const std::string low = chat_to_lower_ascii(s);
    static const char* const kCues[] = {
        "standing requirement", "standing rule", "mandatory", "every reply", "every response",
        "every turn", "every answer", "all replies", "all responses", "you must", "must always",
        "always ", "never ", "only when", "only if", "whenever ", "do not ", "don't ",
        "begin your", "begin every", "start your", "start every", "end your", "end every",
        "sign-off", "sign off", "requirement:", "rule:", "constraint:", "protocol:"
    };
    for (const char* cue : kCues) {
        if (low.find(cue) != std::string::npos) return true;
    }
    return false;
}

static std::vector<std::string> ledger_extract_standing_rules(const std::string& user_text) {
    std::vector<std::string> out;
    const size_t n = user_text.size();
    size_t seg_start = 0U;
    for (size_t i = 0U; i <= n; ++i) {
        const bool boundary = (i == n) || (user_text[i] == '\n') ||
                              ((user_text[i] == '.' || user_text[i] == '!') &&
                               (i + 1U == n || user_text[i + 1U] == ' ' || user_text[i + 1U] == '\n' || user_text[i + 1U] == '\r'));
        if (!boundary) continue;
        const size_t end_inc = (i < n && (user_text[i] == '.' || user_text[i] == '!')) ? (i + 1U) : i;
        if (end_inc > seg_start) {
            const std::string cand = ledger_gist(user_text.substr(seg_start, end_inc - seg_start), 512U);
            if (ledger_sentence_is_standing_rule(cand)) {
                ledger_push_unique(&out, cand, 8U);
            }
        }
        seg_start = i + 1U;
    }
    return out;
}

/* Distills ONE closed turn (raw verbatim user + reply text) into an immutable structured ledger entry. */
static LedgerEntry ledger_distill_turn(uint32_t turn_index, const std::string& user_text, const std::string& reply_text) {
    LedgerEntry e;
    e.turn_index = turn_index;
    e.facts      = ledger_extract_facts(user_text);
    ledger_merge_facts(&e.facts, ledger_extract_facts(reply_text));
    e.rules      = ledger_extract_standing_rules(user_text);
    const std::string facts = ledger_render_facts(e.facts);
    const std::string rg    = (reply_text.compare(0U, 9U, "I cannot ") == 0)
        ? std::string("acknowledged")
        : ledger_gist(reply_text, kLedgerGistChars);
    e.text = "[E" + std::to_string(turn_index + 1U) + "] " + (facts.empty() ? std::string("facts{}") : facts) +
             " | user: " + ledger_gist(user_text, kLedgerGistChars) +
             " | reply: " + rg;
    return e;
}

/* Extracts every unique non-duplicate sentence/line in chronological order across the entire [0%..100%] span
 * of a long turn (up to `max_chars`), eliminating repeated log/filler lines so the idle-gap 180B LLM semantic
 * micro-distiller reads 100% of unique human prose/code/narrative nuance regardless of depth! */
static std::string ledger_build_unique_episode_excerpt(
    const std::string& user_text,
    const std::string& reply_text,
    size_t max_chars
) {
    std::vector<std::string> seen;
    std::string out;
    out.reserve(std::min<size_t>(max_chars + 256U, user_text.size() + reply_text.size() + 64U));

    const size_t n = user_text.size();
    size_t seg_start = 0U;
    for (size_t i = 0U; i <= n; ++i) {
        const bool boundary = (i == n) || (user_text[i] == '\n') ||
                              ((user_text[i] == '.' || user_text[i] == '!' || user_text[i] == '?') &&
                               (i + 1U == n || user_text[i + 1U] == ' ' || user_text[i + 1U] == '\n' || user_text[i + 1U] == '\r'));
        if (!boundary) continue;
        const size_t end_inc = (i < n && (user_text[i] == '.' || user_text[i] == '!' || user_text[i] == '?')) ? (i + 1U) : i;
        if (end_inc > seg_start) {
            const std::string seg = chat_trim_ascii(user_text.substr(seg_start, end_inc - seg_start));
            if (!seg.empty() && std::find(seen.begin(), seen.end(), seg) == seen.end()) {
                seen.push_back(seg);
                if (out.size() + seg.size() + 2U <= max_chars) {
                    if (!out.empty()) out += ' ';
                    out += seg;
                }
            }
        }
        seg_start = i + 1U;
    }

    std::string full = "Summarize this completed workstream episode into 3 to 5 dense factual bullet points "
                       "(under 140 words total). Preserve subtle engineering rationales, why options were retired "
                       "or chosen, causes and effects, and concrete technical details:\n\n[User Episode]:\n" +
                       out + "\n\n[Assistant Reply]:\n" + ledger_gist(reply_text, 600U);
    return full;
}

/* System-block content for a shadow prefix: user system prompt + standing requirements + `[Episode Ledger]` entries [0, split). */
static std::string ledger_build_system_content(
    const std::string& user_system_prompt,
    const std::vector<LedgerEntry>& ledger,
    size_t split
) {
    std::string out = chat_trim_ascii(user_system_prompt);
    std::vector<std::string> all_rules;
    for (size_t k = 0U; k < split && k < ledger.size(); ++k) {
        for (const std::string& r : ledger[k].rules) {
            ledger_push_unique(&all_rules, r, 24U);
        }
    }
    if (!all_rules.empty()) {
        if (!out.empty()) out += "\n\n";
        out += "[Standing User Requirements — follow strictly on EVERY response even when not re-stated in the latest prompt]:";
        for (const std::string& r : all_rules) {
            out += "\n- ";
            out += r;
        }
    }
    if (!out.empty()) out += "\n\n";
    out += "[Episode Ledger] Distilled context, architectural rationales, and exact session constants from earlier turns "
           "(treat as ground truth when answering follow-up questions):";
    for (size_t k = 0U; k < split && k < ledger.size(); ++k) {
        out += "\n";
        out += ledger[k].text;
        if (!ledger[k].semantic_summary.empty()) {
            out += "\n  [E" + std::to_string(k + 1U) + " Semantic Summary]: " + ledger[k].semantic_summary;
        }
    }
    return out;
}

/*
 * Overlap split selection: given per-turn transcript token counts and the history floor (turns already
 * folded into the ledger by an earlier swap), returns the first verbatim overlap turn M in (floor, n-1]
 * such that the OverlapTail Turns[M..n-1] stays <= kCompactOverlapFrac * window (always >= 1 turn), or -1
 * when fewer than 2 turns sit above the floor (nothing new to fold).
 */
static int select_overlap_split(const std::vector<uint32_t>& turn_toks, size_t floor_idx, uint32_t window) {
    const size_t n = turn_toks.size();
    if (n < floor_idx + 2U) return -1;
    const double cap = kCompactOverlapFrac * static_cast<double>(window);
    size_t m = n - 1U;
    uint64_t tail = turn_toks[m];
    while (m > floor_idx + 1U && static_cast<double>(tail + turn_toks[m - 1U]) <= cap) {
        tail += turn_toks[m - 1U];
        --m;
    }
    return static_cast<int>(m);
}

/* ============================================================================
 * 3. Unified Polymorphic Hardware Backend (`GLM-5.3-Flash`, `Qwen3.8-Flash-Next` & `Qwen3.8-27B`)
 * ============================================================================ */

struct ChatSamplingSpec {
    bool     greedy             = false;
    float    temperature        = 0.6f;
    uint32_t top_k              = 20U;
    float    top_p              = 0.95f;
    uint64_t rng_seed           = 42ULL;
    float    presence_penalty   = 0.0f;
    float    frequency_penalty  = 0.0f;
    float    repetition_penalty = 1.0f;
    uint32_t penalty_window     = 0U;
};

struct StepDeltaTelemetry {
    uint64_t mtp_drafted    = 0ULL;
    uint64_t mtp_accepted   = 0ULL;
    uint64_t ple_hits       = 0ULL;
    uint64_t ple_ios        = 0ULL;
    uint64_t glm_hot_hits   = 0ULL;
    uint64_t glm_nvme_reads = 0ULL;
    uint64_t glm_nvme_bytes = 0ULL;
    double   glm_nvme_sec   = 0.0;
    uint64_t glm_active_exp = 0ULL;
    uint64_t glm_ran_on_ane = 0ULL;
    uint64_t glm_ane_calls  = 0ULL;
    uint64_t glm_ane_hits   = 0ULL;
};

class IChatBackend {
public:
    virtual ~IChatBackend() = default;

    virtual bool init(const ChatCliConfig& cfg) = 0;
    virtual const char* model_name() const = 0;
    virtual const char* device_name() const = 0;
    virtual bool zero_copy() const = 0;
    virtual double init_wall_ms() const = 0;
    virtual bool is_flash() const = 0;
    virtual bool is_glm53() const { return false; }
    virtual bool is_inkling2() const { return false; }
    virtual const char* kv_cache_name() const { return "f16"; }
    virtual bool int8_prefill_enabled() const { return false; }
    virtual bool adaptive_k_enabled() const { return false; }
    virtual bool frspec_enabled() const { return false; }
    virtual uint32_t frspec_shortlist_size() const { return 0U; }
    virtual bool tree_spec_enabled() const { return false; }
    virtual bool bitplane_head_enabled() const { return false; }
    virtual bool has_vision() const { return false; }
    virtual bool vision_metal_cache_hit() const { return false; }
    virtual double vision_pso_load_ms() const { return 0.0; }
    virtual bool metal_cache_hit() const { return false; }
    virtual double pso_load_ms() const { return 0.0; }
    virtual bool has_mtp() const = 0;
    virtual bool has_ple() const = 0;

    virtual void set_glm53_exec_mode(StrataGlm53ExecMode /*mode*/) {}
    virtual void reset_glm53_hot_cache() {}
    virtual uint32_t glm53_hot_resident() const { return 0U; }
    virtual uint32_t glm53_hot_capacity() const { return 0U; }
    virtual double glm53_hot_cache_gib() const { return 0.0; }
    virtual double glm53_shared_mtl_gib() const { return 0.0; }
    virtual double glm53_experts_nvme_gib() const { return 0.0; }
    virtual bool glm53_ane_verified() const { return false; }
    virtual bool glm53_is_ram_resident() const { return false; }

    virtual uint32_t max_seq_len() const = 0;
    virtual uint32_t seq_pos() const = 0;
    virtual double uma_mapped_gib() const = 0;
    virtual double ple_nvme_gib() const = 0;
    virtual double ple_lfu_mib() const = 0;
    virtual double deltanet_mib() const = 0;
    virtual double ple_state_mib() const = 0;
    virtual uint64_t kv_bytes_per_tok() const = 0;
    virtual uint64_t active_bytes_per_tok() const = 0;

    virtual uint32_t token_think_start() const = 0;
    virtual uint32_t token_think_end() const = 0;
    virtual bool is_eos(uint32_t tok) const = 0;

    virtual bool encode(const std::string& text, bool parse_special, std::vector<uint32_t>* out_ids) const = 0;
    virtual std::string decode(const uint32_t* ids, uint32_t n, bool skip_special) const = 0;
    virtual void stream_reset() = 0;
    virtual std::string stream_step(uint32_t token_id, bool skip_special) = 0;

    virtual std::string format_system_block(
        const std::string& system_prompt,
        bool enable_thinking,
        const std::string& effort) const = 0;

    virtual void reset() = 0;
    virtual void penalties_reset() {}

    virtual int prefill(
        const uint32_t* ids,
        uint32_t n,
        const ChatSamplingSpec& samp,
        uint32_t* next_tok_out,
        StepDeltaTelemetry* tel_out) = 0;

    virtual int encode_image_file(
        const std::string& /*path*/,
        std::vector<float>* /*out_emb_f32*/,
        uint32_t* /*out_num_tokens*/,
        uint32_t* /*out_grid_h*/,
        uint32_t* /*out_grid_w*/,
        double* /*out_ms*/) {
        return -1;
    }

    virtual int prefill_with_images(
        const uint32_t* ids,
        uint32_t n,
        const StrataInkling2ImageSpan* /*spans*/,
        uint32_t /*num_spans*/,
        const ChatSamplingSpec& samp,
        uint32_t* next_tok_out,
        StepDeltaTelemetry* tel_out) {
        return prefill(ids, n, samp, next_tok_out, tel_out);
    }

    virtual int decode_step(
        uint32_t pending_tok,
        uint32_t draft_k,
        const ChatSamplingSpec& samp,
        uint32_t* emitted_out,
        uint32_t* num_emitted_out,
        StepDeltaTelemetry* tel_out) = 0;

    virtual void* checkpoint_create() = 0;
    virtual void checkpoint_destroy(void* ckpt) = 0;
    virtual int checkpoint_save(void* ckpt) = 0;
    virtual int checkpoint_restore(const void* ckpt) = 0;
    virtual uint64_t checkpoint_bytes(const void* ckpt) const = 0;

    /* ---- Step 4 compaction plumbing (no-op defaults for Qwen3.8-27B / GLM-5.3) ----
     * Slot handles are opaque; `nullptr` names the engine's built-in default slot (linear KV, max_seq_len). */
    virtual bool compaction_supported() const { return false; }
    virtual void* compact_slot_create(uint32_t /*uma_kv_pool_tokens*/) { return nullptr; }
    virtual void compact_slot_destroy(void* /*slot*/) {}
    virtual void compact_bind_slot(void* /*slot*/) {}
    virtual void* compact_bound_slot() const { return nullptr; }
    virtual uint32_t compact_slot_seq_pos(const void* /*slot*/) const { return 0U; }
    virtual bool compact_slot_is_paged(const void* /*slot*/) const { return false; }
    virtual uint64_t compact_slot_bytes(const void* /*slot*/) const { return 0ULL; }
    virtual void compact_get_stats(const void* /*slot*/, StrataQwenFlashCompactionStats* out) const {
        if (out) *out = StrataQwenFlashCompactionStats{};
    }
    virtual void compact_reset_stats(void* /*slot*/) {}
    virtual void compact_reset_bound() {}
    virtual int compact_prefill_slice(const uint32_t* /*ids*/, uint32_t /*n*/, bool /*final_logits*/, uint32_t* /*next*/) {
        return -1;
    }
    virtual bool compact_ane_ready(std::string* note) const {
        if (note) note->clear();
        return false;
    }
    /* Cumulative tokens pushed through `prefill()` (live) and `compact_prefill_slice()` (background). */
    virtual uint64_t prefill_tokens_total() const { return 0ULL; }
    virtual uint64_t slice_tokens_total() const { return 0ULL; }
    virtual const char* engine_id_tag() const = 0;
    virtual bool supports_session_save() const { return false; }
    virtual int save_state_fd(int /*fd*/, uint64_t* /*bytes_out*/) { return -1; }
    virtual int load_state_fd(int /*fd*/, uint64_t* /*bytes_out*/) { return -1; }
};

/* ---- 3.1 `Qwen/Qwen3.8-Flash-Next` (180B MoE + 51.2B NVMe PLE + Native MTP) ---- */

class QwenFlashChatBackend final : public IChatBackend {
public:
    QwenFlashChatBackend() {
        std::memset(&bdl_, 0, sizeof(bdl_));
        bdl_.part0_fd = -1;
        bdl_.part1_fd = -1;
        bdl_.ple_fd   = -1;
    }

    ~QwenFlashChatBackend() override {
        if (eng_) {
            strata_qwen_flash_bind_slot(eng_, nullptr);
            for (StrataQwenFlashContextSlot* s : slots_) {
                if (s) strata_qwen_flash_slot_destroy(s);
            }
            slots_.clear();
            strata_qwen_flash_gpu_destroy(eng_);
            eng_ = nullptr;
        }
        if (bundle_loaded_) {
            strata_qwen_flash_free_bundle(&bdl_);
            bundle_loaded_ = false;
        }
    }

    bool init(const ChatCliConfig& cfg) override {
        const auto t0 = SteadyClock::now();
        if (strata_qwen_flash_load_bundle(
                cfg.bundle_dir.c_str(),
                cfg.ple_cache_mib * 1024ULL * 1024ULL,
                &bdl_) != 0) {
            return false;
        }
        bundle_loaded_ = true;

        const uint32_t req_seq = chat_resolve_flash_max_seq_len(cfg);
        eng_ = strata_qwen_flash_gpu_create(&bdl_, req_seq);
        if (!eng_) {
            return false;
        }
        if (cfg.int8_prefill_set) strata_qwen_flash_gpu_set_int8_prefill(eng_, cfg.int8_prefill);
        if (cfg.adaptive_k_set)   strata_qwen_flash_gpu_set_adaptive_k(eng_, cfg.adaptive_k);
        if (cfg.frspec_set)       strata_qwen_flash_gpu_set_frspec(eng_, cfg.frspec);
        strata_qwen_flash_gpu_get_info(eng_, &info_);

        /* Warm Metal 4 PSOs & residency set wiring before steady-state fault tracking */
        @autoreleasepool {
            strata_qwen_flash_gpu_reset(eng_);
            strata_qwen_flash_gpu_reset_counters(eng_);
            const uint32_t cfg_warm_k = (cfg.draft_k > 0U)
                ? std::min<uint32_t>(cfg.draft_k, STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K)
                : 0U;
            const uint32_t warm_ids[8] = {
                STRATA_QWEN_FLASH_TOKEN_IM_START, 872U, 198U, 9707U,
                STRATA_QWEN_FLASH_TOKEN_IM_END, 198U, STRATA_QWEN_FLASH_TOKEN_IM_START, 77091U
            };
            StrataQwenFlashSamplingParams greedy{};
            greedy.greedy = true;
            uint32_t nt1 = 0U, nt2 = 0U;
            (void)strata_qwen_flash_gpu_prefill(eng_, warm_ids, 8U, &greedy, &nt1, nullptr);
            (void)strata_qwen_flash_gpu_decode_step(eng_, nt1, &greedy, &nt2, nullptr);
            if (has_mtp()) {
                const uint32_t dense_k = (cfg_warm_k > 0U) ? cfg_warm_k : 2U;
                uint32_t em[STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t ne = 0U, na = 0U;
                (void)strata_qwen_flash_gpu_spec_decode_step(eng_, nt2, dense_k, &greedy, em, &ne, &na);
                if (na == 0U && ne > 0U && ne <= STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K + 1U) {
                    (void)strata_qwen_flash_gpu_spec_decode_step(eng_, em[ne - 1U], dense_k, &greedy, em, &ne, &na);
                }
            }
            /* Also warm sparse QSA PSOs (`seq_pos >= 2051`) — incl. M>=2 `score_batch` & MTP KV-Sync when K>0 —
             * so AGX shader binaries fault in before steady-state */
            if (req_seq >= 2112U) {
                strata_qwen_flash_gpu_reset(eng_);
                std::vector<uint32_t> sp_warm(2060U, 198U);
                sp_warm[0] = STRATA_QWEN_FLASH_TOKEN_IM_START;
                uint32_t sp1 = 0U, sp2 = 0U;
                (void)strata_qwen_flash_gpu_prefill(eng_, sp_warm.data(), static_cast<uint32_t>(sp_warm.size()), &greedy, &sp1, nullptr);
                (void)strata_qwen_flash_gpu_decode_step(eng_, sp1, &greedy, &sp2, nullptr);
                if (has_mtp() && cfg_warm_k > 0U) {
                    uint32_t em[STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K + 2U] = {};
                    uint32_t ne = 0U, na = 0U;
                    (void)strata_qwen_flash_gpu_spec_decode_step(eng_, sp2, cfg_warm_k, &greedy, em, &ne, &na);
                    if (na == 0U && ne > 0U && ne <= STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K + 1U) {
                        (void)strata_qwen_flash_gpu_spec_decode_step(eng_, em[ne - 1U], cfg_warm_k, &greedy, em, &ne, &na);
                    }
                }
            }
            strata_qwen_flash_gpu_reset(eng_);
            strata_qwen_flash_gpu_reset_counters(eng_);
            strata_qwen_flash_ple_cache_reset_stats(&bdl_.ple_cache);
        }

        strata_qwen_flash_stream_decoder_init(&sdec_);
        init_ms_ = chat_elapsed_ms(t0, SteadyClock::now());
        return true;
    }

    const char* model_name() const override {
        return "Qwen/Qwen3.8-Flash-Next (180B MoE + 51.2B NVMe PLE + Native MTP)";
    }
    const char* device_name() const override { return info_.device_name; }
    bool zero_copy() const override { return info_.zero_copy_mmap; }
    double init_wall_ms() const override { return init_ms_; }
    bool is_flash() const override { return true; }
    bool int8_prefill_enabled() const override { return info_.int8_prefill_enabled != 0U; }
    bool adaptive_k_enabled() const override { return info_.adaptive_k_enabled != 0U; }
    bool frspec_enabled() const override { return info_.frspec_enabled != 0U; }
    uint32_t frspec_shortlist_size() const override { return info_.frspec_shortlist_size; }
    bool metal_cache_hit() const override { return info_.metal_cache_hit != 0U; }
    double pso_load_ms() const override { return info_.pso_load_ms; }
    bool has_mtp() const override { return bdl_.cfg.has_mtp != 0U && bdl_.header1->mtp.enabled != 0U; }
    bool has_ple() const override { return bdl_.cfg.has_ple != 0U; }

    uint32_t max_seq_len() const override { return info_.max_seq_len; }
    uint32_t seq_pos() const override { return strata_qwen_flash_gpu_seq_pos(eng_); }

    double uma_mapped_gib() const override {
        return static_cast<double>(info_.total_uma_mapped_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    double ple_nvme_gib() const override {
        return static_cast<double>(info_.ple_nvme_file_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    double ple_lfu_mib() const override {
        return static_cast<double>(info_.ple_lfu_cache_bytes) / (1024.0 * 1024.0);
    }
    double deltanet_mib() const override {
        return static_cast<double>(info_.deltanet_state_bytes) / (1024.0 * 1024.0);
    }
    double ple_state_mib() const override {
        return static_cast<double>(info_.ple_state_bytes) / (1024.0 * 1024.0);
    }
    uint64_t kv_bytes_per_tok() const override { return info_.kv_cache_bytes_per_token; }
    uint64_t active_bytes_per_tok() const override { return info_.active_bytes_per_token; }

    uint32_t token_think_start() const override { return STRATA_QWEN_FLASH_TOKEN_THINK_START; }
    uint32_t token_think_end() const override { return STRATA_QWEN_FLASH_TOKEN_THINK_END; }
    bool is_eos(uint32_t tok) const override { return strata_qwen_flash_is_eos_token(&bdl_.cfg, tok); }

    bool encode(const std::string& text, bool parse_special, std::vector<uint32_t>* out_ids) const override {
        const uint32_t cap = static_cast<uint32_t>(std::max<size_t>(text.size() + 128U, 256U));
        out_ids->assign(cap, 0U);
        uint32_t n = 0U;
        if (strata_qwen_flash_encode(
                bdl_.tokenizer, text.c_str(), text.size(), parse_special, out_ids->data(), cap, &n) != 0) {
            return false;
        }
        out_ids->resize(n);
        return true;
    }

    std::string decode(const uint32_t* ids, uint32_t n, bool skip_special) const override {
        if (!ids || n == 0U) return {};
        const size_t cap = std::max<size_t>(static_cast<size_t>(n) * 32U + 256U, 1024U);
        std::vector<char> buf(cap, '\0');
        size_t len = 0U;
        if (strata_qwen_flash_decode(bdl_.tokenizer, ids, n, skip_special, buf.data(), cap, &len) != 0) {
            return {};
        }
        return std::string(buf.data(), len);
    }

    void stream_reset() override {
        strata_qwen_flash_stream_decoder_init(&sdec_);
    }

    std::string stream_step(uint32_t token_id, bool skip_special) override {
        char piece[256];
        size_t len = 0U;
        if (strata_qwen_flash_stream_decode_step(
                bdl_.tokenizer, &sdec_, token_id, skip_special, piece, sizeof(piece), &len) == 0 && len > 0U) {
            return std::string(piece, len);
        }
        return {};
    }

    std::string format_system_block(
        const std::string& system_prompt,
        bool enable_thinking,
        const std::string& effort
    ) const override {
        return format_flash_system_block(system_prompt, enable_thinking, effort);
    }

    void reset() override {
        strata_qwen_flash_gpu_reset(eng_);
        strata_qwen_flash_gpu_reset_counters(eng_);
    }

    void penalties_reset() override {
        strata_qwen_flash_gpu_penalties_reset(eng_);
    }

    int prefill(
        const uint32_t* ids,
        uint32_t n,
        const ChatSamplingSpec& samp,
        uint32_t* next_tok_out,
        StepDeltaTelemetry* tel_out
    ) override {
        StrataQwenFlashPerfCounters c0{}, c1{};
        strata_qwen_flash_gpu_get_counters(eng_, &c0);

        StrataQwenFlashSamplingParams sp{};
        sp.greedy             = samp.greedy;
        sp.temperature        = samp.temperature;
        sp.top_k              = samp.top_k;
        sp.top_p              = samp.top_p;
        sp.rng_seed           = samp.rng_seed;
        sp.presence_penalty   = samp.presence_penalty;
        sp.frequency_penalty  = samp.frequency_penalty;
        sp.repetition_penalty = samp.repetition_penalty;
        sp.penalty_window     = samp.penalty_window;

        int rc = 0;
        @autoreleasepool {
            rc = strata_qwen_flash_gpu_prefill(eng_, ids, n, &sp, next_tok_out, nullptr);
        }
        if (rc == 0) prefill_toks_ += n;
        strata_qwen_flash_gpu_get_counters(eng_, &c1);
        if (tel_out) {
            tel_out->ple_hits = c1.ple_lfu_hits - c0.ple_lfu_hits;
            tel_out->ple_ios  = c1.ple_nvme_preads - c0.ple_nvme_preads;
        }
        return rc;
    }

    int decode_step(
        uint32_t pending_tok,
        uint32_t draft_k,
        const ChatSamplingSpec& samp,
        uint32_t* emitted_out,
        uint32_t* num_emitted_out,
        StepDeltaTelemetry* tel_out
    ) override {
        StrataQwenFlashPerfCounters c0{}, c1{};
        strata_qwen_flash_gpu_get_counters(eng_, &c0);

        StrataQwenFlashSamplingParams sp{};
        sp.greedy             = samp.greedy;
        sp.temperature        = samp.temperature;
        sp.top_k              = samp.top_k;
        sp.top_p              = samp.top_p;
        sp.rng_seed           = samp.rng_seed;
        sp.presence_penalty   = samp.presence_penalty;
        sp.frequency_penalty  = samp.frequency_penalty;
        sp.repetition_penalty = samp.repetition_penalty;
        sp.penalty_window     = samp.penalty_window;

        const uint32_t eff_k = has_mtp()
            ? std::min<uint32_t>(draft_k, STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K)
            : 0U;

        int rc = 0;
        @autoreleasepool {
            if (eff_k == 0U) {
                uint32_t nxt = 0U;
                rc = strata_qwen_flash_gpu_decode_step(eng_, pending_tok, &sp, &nxt, nullptr);
                if (rc == 0) {
                    emitted_out[0]   = nxt;
                    *num_emitted_out = 1U;
                }
            } else {
                uint32_t acc = 0U;
                rc = strata_qwen_flash_gpu_spec_decode_step(
                    eng_, pending_tok, eff_k, &sp, emitted_out, num_emitted_out, &acc);
            }
        }
        strata_qwen_flash_gpu_get_counters(eng_, &c1);
        if (tel_out) {
            tel_out->mtp_drafted  = c1.mtp_drafted_tokens - c0.mtp_drafted_tokens;
            tel_out->mtp_accepted = c1.mtp_accepted_tokens - c0.mtp_accepted_tokens;
            tel_out->ple_hits     = c1.ple_lfu_hits - c0.ple_lfu_hits;
            tel_out->ple_ios      = c1.ple_nvme_preads - c0.ple_nvme_preads;
        }
        return rc;
    }

    void* checkpoint_create() override {
        return strata_qwen_flash_checkpoint_create(eng_);
    }
    void checkpoint_destroy(void* ckpt) override {
        strata_qwen_flash_checkpoint_destroy(static_cast<StrataQwenFlashCheckpoint*>(ckpt));
    }
    int checkpoint_save(void* ckpt) override {
        return strata_qwen_flash_checkpoint_save(eng_, static_cast<StrataQwenFlashCheckpoint*>(ckpt));
    }
    int checkpoint_restore(const void* ckpt) override {
        return strata_qwen_flash_checkpoint_restore(eng_, static_cast<const StrataQwenFlashCheckpoint*>(ckpt));
    }
    uint64_t checkpoint_bytes(const void* ckpt) const override {
        return strata_qwen_flash_checkpoint_bytes(static_cast<const StrataQwenFlashCheckpoint*>(ckpt));
    }

    /* ---- Step 4 compaction plumbing ---- */
    bool compaction_supported() const override { return eng_ != nullptr; }

    void* compact_slot_create(uint32_t uma_kv_pool_tokens) override {
        StrataQwenFlashContextSlot* s = nullptr;
        @autoreleasepool {
            s = strata_qwen_flash_slot_create(eng_, uma_kv_pool_tokens, nullptr);
        }
        if (s) slots_.push_back(s);
        return s;
    }
    void compact_slot_destroy(void* slot) override {
        if (!slot) return; /* the built-in default slot is never destroyed */
        auto* s = static_cast<StrataQwenFlashContextSlot*>(slot);
        slots_.erase(std::remove(slots_.begin(), slots_.end(), s), slots_.end());
        strata_qwen_flash_slot_destroy(s);
    }
    void compact_bind_slot(void* slot) override {
        strata_qwen_flash_bind_slot(eng_, static_cast<StrataQwenFlashContextSlot*>(slot));
    }
    void* compact_bound_slot() const override { return strata_qwen_flash_bound_slot(eng_); }
    uint32_t compact_slot_seq_pos(const void* slot) const override {
        return strata_qwen_flash_slot_seq_pos(static_cast<const StrataQwenFlashContextSlot*>(slot));
    }
    bool compact_slot_is_paged(const void* slot) const override {
        return slot ? strata_qwen_flash_slot_is_paged(static_cast<const StrataQwenFlashContextSlot*>(slot)) : false;
    }
    uint64_t compact_slot_bytes(const void* slot) const override {
        return slot ? strata_qwen_flash_slot_bytes(static_cast<const StrataQwenFlashContextSlot*>(slot)) : 0ULL;
    }
    void compact_get_stats(const void* slot, StrataQwenFlashCompactionStats* out) const override {
        if (!out) return;
        *out = StrataQwenFlashCompactionStats{};
        strata_qwen_flash_slot_get_stats(static_cast<const StrataQwenFlashContextSlot*>(slot), out);
    }
    void compact_reset_stats(void* slot) override {
        if (slot) strata_qwen_flash_slot_reset_stats(static_cast<StrataQwenFlashContextSlot*>(slot));
    }
    void compact_reset_bound() override {
        strata_qwen_flash_gpu_reset(eng_);
    }
    int compact_prefill_slice(const uint32_t* ids, uint32_t n, bool final_logits, uint32_t* next) override {
        int rc = 0;
        @autoreleasepool {
            rc = strata_qwen_flash_gpu_prefill_slice(eng_, ids, n, final_logits, next);
        }
        if (rc == 0) slice_toks_ += n;
        return rc;
    }
    bool compact_ane_ready(std::string* note) const override {
        char buf[256] = {};
        const int ok = strata_qwen_flash_ane_probe_ready(eng_, buf, sizeof(buf));
        if (note) *note = buf;
        return ok == 1;
    }
    uint64_t prefill_tokens_total() const override { return prefill_toks_; }
    uint64_t slice_tokens_total() const override { return slice_toks_; }
    const char* engine_id_tag() const override { return "qwen3.8-flash-next"; }
    bool supports_session_save() const override { return true; }
    int save_state_fd(int fd, uint64_t* bytes_out) override {
        return strata_qwen_flash_state_save_fd(eng_, fd, bytes_out);
    }
    int load_state_fd(int fd, uint64_t* bytes_out) override {
        return strata_qwen_flash_state_load_fd(eng_, fd, bytes_out);
    }

private:
    std::vector<StrataQwenFlashContextSlot*> slots_;
    uint64_t                     prefill_toks_  = 0ULL;
    uint64_t                     slice_toks_    = 0ULL;
    StrataQwenFlashBundle        bdl_{};
    bool                         bundle_loaded_ = false;
    StrataQwenFlashGpuEngine*    eng_           = nullptr;
    StrataQwenFlashGpuInfo       info_{};
    StrataQwenFlashStreamDecoder sdec_{};
    double                       init_ms_       = 0.0;
};

/* ---- 3.2 `Qwen/Qwen3.8-27B` (64-Layer Hybrid GatedDeltaNet + Full GQA) ---- */

class Qwen38ChatBackend final : public IChatBackend {
public:
    Qwen38ChatBackend() {
        std::memset(&bdl_, 0, sizeof(bdl_));
        bdl_.mmap_fd = -1;
    }

    ~Qwen38ChatBackend() override {
        if (eng_) {
            strata_qwen38_gpu_destroy(eng_);
            eng_ = nullptr;
        }
        if (bundle_loaded_) {
            strata_qwen38_free_bundle(&bdl_);
            bundle_loaded_ = false;
        }
    }

    bool init(const ChatCliConfig& cfg) override {
        const auto t0 = SteadyClock::now();
        if (strata_qwen38_load_bundle(cfg.bundle_dir.c_str(), &bdl_) != 0) {
            return false;
        }
        bundle_loaded_ = true;

        const uint32_t req_seq = (cfg.max_seq_len > 0U) ? cfg.max_seq_len : 32768U;
        eng_ = strata_qwen38_gpu_create(&bdl_, req_seq);
        if (!eng_) {
            return false;
        }
        strata_qwen38_gpu_get_info(eng_, &info_);

        /* Warm Metal 4 PSOs & zero-copy residency before steady-state fault tracking */
        @autoreleasepool {
            strata_qwen38_gpu_reset(eng_);
            const uint32_t warm_ids[8] = {
                STRATA_QWEN38_TOKEN_IM_START, 872U, 198U, 9707U,
                STRATA_QWEN38_TOKEN_IM_END, 198U, STRATA_QWEN38_TOKEN_IM_START, 77091U
            };
            StrataQwen38SamplingParams greedy{};
            greedy.greedy = true;
            uint32_t nt1 = 0U, nt2 = 0U;
            (void)strata_qwen38_gpu_prefill(eng_, warm_ids, 8U, &greedy, &nt1, nullptr);
            (void)strata_qwen38_gpu_decode_step(eng_, nt1, &greedy, &nt2, nullptr);
            strata_qwen38_gpu_reset(eng_);
        }

        strata_qwen38_stream_decoder_init(&sdec_);
        init_ms_ = chat_elapsed_ms(t0, SteadyClock::now());
        return true;
    }

    const char* model_name() const override {
        return "Qwen/Qwen3.8-27B (64L Hybrid GatedDeltaNet + GQA)";
    }
    const char* device_name() const override { return info_.device_name; }
    bool zero_copy() const override { return info_.zero_copy_mmap; }
    double init_wall_ms() const override { return init_ms_; }
    bool is_flash() const override { return false; }
    bool has_mtp() const override { return false; }
    bool has_ple() const override { return false; }

    uint32_t max_seq_len() const override { return info_.max_seq_len; }
    uint32_t seq_pos() const override { return strata_qwen38_gpu_seq_pos(eng_); }

    double uma_mapped_gib() const override {
        return static_cast<double>(info_.weights_mapped_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    double ple_nvme_gib() const override { return 0.0; }
    double ple_lfu_mib() const override { return 0.0; }
    double deltanet_mib() const override {
        return static_cast<double>(info_.deltanet_state_bytes) / (1024.0 * 1024.0);
    }
    double ple_state_mib() const override { return 0.0; }
    uint64_t kv_bytes_per_tok() const override { return info_.kv_cache_bytes_per_token; }
    uint64_t active_bytes_per_tok() const override { return info_.active_bytes_per_token; }

    uint32_t token_think_start() const override { return STRATA_QWEN38_TOKEN_THINK_START; }
    uint32_t token_think_end() const override { return STRATA_QWEN38_TOKEN_THINK_END; }
    bool is_eos(uint32_t tok) const override { return strata_qwen38_is_eos_token(&bdl_.cfg, tok); }

    bool encode(const std::string& text, bool parse_special, std::vector<uint32_t>* out_ids) const override {
        const uint32_t cap = static_cast<uint32_t>(std::max<size_t>(text.size() + 128U, 256U));
        out_ids->assign(cap, 0U);
        uint32_t n = 0U;
        if (strata_qwen38_encode(
                bdl_.tokenizer, text.c_str(), text.size(), parse_special, out_ids->data(), cap, &n) != 0) {
            return false;
        }
        out_ids->resize(n);
        return true;
    }

    std::string decode(const uint32_t* ids, uint32_t n, bool skip_special) const override {
        if (!ids || n == 0U) return {};
        const size_t cap = std::max<size_t>(static_cast<size_t>(n) * 32U + 256U, 1024U);
        std::vector<char> buf(cap, '\0');
        size_t len = 0U;
        if (strata_qwen38_decode(bdl_.tokenizer, ids, n, skip_special, buf.data(), cap, &len) != 0) {
            return {};
        }
        return std::string(buf.data(), len);
    }

    void stream_reset() override {
        strata_qwen38_stream_decoder_init(&sdec_);
    }

    std::string stream_step(uint32_t token_id, bool skip_special) override {
        char piece[256];
        size_t len = 0U;
        if (strata_qwen38_stream_decode_step(
                bdl_.tokenizer, &sdec_, token_id, skip_special, piece, sizeof(piece), &len) == 0 && len > 0U) {
            return std::string(piece, len);
        }
        return {};
    }

    std::string format_system_block(
        const std::string& system_prompt,
        bool /*enable_thinking*/,
        const std::string& /*effort*/
    ) const override {
        return format_qwen38_system_block(system_prompt);
    }

    void reset() override {
        strata_qwen38_gpu_reset(eng_);
    }

    int prefill(
        const uint32_t* ids,
        uint32_t n,
        const ChatSamplingSpec& samp,
        uint32_t* next_tok_out,
        StepDeltaTelemetry* tel_out
    ) override {
        if (tel_out) *tel_out = StepDeltaTelemetry{};
        StrataQwen38SamplingParams sp{};
        sp.greedy      = samp.greedy;
        sp.temperature = samp.temperature;
        sp.top_k       = samp.top_k;
        sp.top_p       = samp.top_p;
        sp.rng_seed    = samp.rng_seed;

        int rc = 0;
        @autoreleasepool {
            rc = strata_qwen38_gpu_prefill(eng_, ids, n, &sp, next_tok_out, nullptr);
        }
        return rc;
    }

    int decode_step(
        uint32_t pending_tok,
        uint32_t /*draft_k*/,
        const ChatSamplingSpec& samp,
        uint32_t* emitted_out,
        uint32_t* num_emitted_out,
        StepDeltaTelemetry* tel_out
    ) override {
        if (tel_out) *tel_out = StepDeltaTelemetry{};
        StrataQwen38SamplingParams sp{};
        sp.greedy      = samp.greedy;
        sp.temperature = samp.temperature;
        sp.top_k       = samp.top_k;
        sp.top_p       = samp.top_p;
        sp.rng_seed    = samp.rng_seed;

        uint32_t nxt = 0U;
        int rc = 0;
        @autoreleasepool {
            rc = strata_qwen38_gpu_decode_step(eng_, pending_tok, &sp, &nxt, nullptr);
        }
        if (rc == 0) {
            emitted_out[0]   = nxt;
            *num_emitted_out = 1U;
        }
        return rc;
    }

    void* checkpoint_create() override {
        return strata_qwen38_checkpoint_create(eng_);
    }
    void checkpoint_destroy(void* ckpt) override {
        strata_qwen38_checkpoint_destroy(static_cast<StrataQwen38Checkpoint*>(ckpt));
    }
    int checkpoint_save(void* ckpt) override {
        return strata_qwen38_checkpoint_save(eng_, static_cast<StrataQwen38Checkpoint*>(ckpt));
    }
    int checkpoint_restore(const void* ckpt) override {
        return strata_qwen38_checkpoint_restore(eng_, static_cast<const StrataQwen38Checkpoint*>(ckpt));
    }
    uint64_t checkpoint_bytes(const void* ckpt) const override {
        return strata_qwen38_checkpoint_bytes(static_cast<const StrataQwen38Checkpoint*>(ckpt));
    }
    const char* engine_id_tag() const override { return "qwen3.8-27b"; }
    bool supports_session_save() const override { return true; }
    int save_state_fd(int fd, uint64_t* bytes_out) override {
        return strata_qwen38_state_save_fd(eng_, fd, bytes_out);
    }
    int load_state_fd(int fd, uint64_t* bytes_out) override {
        return strata_qwen38_state_load_fd(eng_, fd, bytes_out);
    }

private:
    StrataQwen38Bundle        bdl_{};
    bool                      bundle_loaded_ = false;
    StrataQwen38GpuEngine*    eng_           = nullptr;
    StrataQwen38GpuInfo       info_{};
    StrataQwen38StreamDecoder sdec_{};
    double                    init_ms_       = 0.0;
};

/* ---- 3.3 `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` (320B / 169.22 GiB Regime II MoE + 16c ANE + MTP) ---- */

class Glm53ChatBackend final : public IChatBackend {
public:
    Glm53ChatBackend() {
        std::memset(&bdl_, 0, sizeof(bdl_));
        bdl_.spine_fd       = -1;
        bdl_.part0_fd       = -1;
        bdl_.part1_fd       = -1;
        bdl_.mtp_experts_fd = -1;
        for (uint32_t i = 0U; i < STRATA_GLM53_NUM_NVME_FDS; ++i) {
            bdl_.experts_fds[i] = -1;
        }
    }

    ~Glm53ChatBackend() override {
        if (eng_) {
            strata_glm53_bind_slot(eng_, nullptr);
            for (StrataGlm53ContextSlot* s : slots_) {
                if (s) strata_glm53_slot_destroy(s);
            }
            slots_.clear();
            strata_glm53_gpu_destroy(eng_);
            eng_ = nullptr;
        }
        if (bundle_loaded_) {
            strata_glm53_free_bundle(&bdl_);
            bundle_loaded_ = false;
        }
    }

    bool init(const ChatCliConfig& cfg) override {
        const auto t0 = SteadyClock::now();
        if (strata_glm53_load_bundle(cfg.bundle_dir.c_str(), &bdl_) != 0) {
            return false;
        }
        bundle_loaded_ = true;
        const bool is_v2 = bdl_.is_ram_resident;
        const StrataGlm53ExecMode eff_mode = (is_v2 && !cfg.glm53_mode_explicit)
            ? STRATA_GLM53_MODE_LOSSLESS
            : cfg.glm53_mode;

        const uint32_t req_seq = chat_resolve_glm53_max_seq_len(cfg);
        eng_ = strata_glm53_gpu_create(&bdl_, req_seq, cfg.hot_cache_gib, eff_mode);
        if (!eng_) {
            return false;
        }
        if (cfg.adaptive_cfg_explicit && !is_v2) {
            strata_glm53_gpu_set_adaptive_params(eng_, cfg.gate_tau, cfg.cold_cap, /*renorm_selected=*/true);
        }
        if (cfg.int8_prefill_set) strata_glm53_gpu_set_int8_prefill(eng_, cfg.int8_prefill);
        if (cfg.adaptive_k_set)   strata_glm53_gpu_set_adaptive_k(eng_, cfg.adaptive_k);
        if (cfg.frspec_set)       strata_glm53_gpu_set_frspec(eng_, cfg.frspec);

        /* Warm Metal 4 PSOs, residency set & (on v1) 16-Core ANE probes before steady-state fault tracking */
        @autoreleasepool {
            if (!is_v2) {
                strata_glm53_gpu_set_exec_mode(eng_, STRATA_GLM53_MODE_ADAPTIVE);
            }
            strata_glm53_gpu_reset(eng_);
            strata_glm53_gpu_penalties_reset(eng_);
            strata_glm53_gpu_reset_counters(eng_);
            const uint32_t warm_ids[6] = {
                STRATA_GLM53_TOKEN_GMASK, STRATA_GLM53_TOKEN_SOP,
                STRATA_GLM53_TOKEN_USER, 1000U,
                STRATA_GLM53_TOKEN_ASSISTANT, STRATA_GLM53_TOKEN_THINK_START
            };
            StrataGlm53SamplingParams greedy{};
            greedy.greedy             = true;
            greedy.repetition_penalty = 1.0f;
            uint32_t nt1 = 0U, nt2 = 0U;
            (void)strata_glm53_gpu_prefill(eng_, warm_ids, 6U, &greedy, &nt1, nullptr);
            (void)strata_glm53_gpu_decode_step(eng_, nt1, &greedy, &nt2, nullptr);
            if (bdl_.header && bdl_.header->mtp.enabled) {
                uint32_t em[STRATA_GLM53_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t ne = 0U, na = 0U;
                const uint32_t warm_k = is_v2 ? STRATA_GLM53_MAX_SPEC_DRAFT_K : 2U;
                (void)strata_glm53_gpu_spec_decode_step(eng_, nt2, warm_k, &greedy, em, &ne, &na);
            }
            if (!is_v2) {
                strata_glm53_gpu_reset_hot_cache(eng_);
            }
            strata_glm53_gpu_reset(eng_);
            strata_glm53_gpu_penalties_reset(eng_);
            strata_glm53_gpu_reset_counters(eng_);
            strata_glm53_gpu_set_exec_mode(eng_, eff_mode);
        }

        strata_glm53_gpu_get_info(eng_, &info_);
        strata_glm53_stream_decoder_init(&sdec_);
        init_ms_ = chat_elapsed_ms(t0, SteadyClock::now());
        return true;
    }

    const char* model_name() const override {
        return (info_.is_ram_resident != 0U || bdl_.is_ram_resident)
            ? "zai-org/GLM-5.3-Flash (320B / 93.22 GiB v2 RAM-Resident E8S-18-V4 + MTP)"
            : "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP (320B / 169.22 GiB Regime II MoE + 16c ANE + MTP)";
    }
    const char* device_name() const override { return info_.device_name; }
    bool zero_copy() const override { return info_.zero_copy_spine; }
    double init_wall_ms() const override { return init_ms_; }
    bool is_flash() const override { return false; }
    bool is_glm53() const override { return true; }
    bool glm53_is_ram_resident() const override { return (info_.is_ram_resident != 0U) || bdl_.is_ram_resident; }
    bool int8_prefill_enabled() const override { return info_.int8_prefill_enabled != 0U; }
    bool adaptive_k_enabled() const override { return info_.adaptive_k_enabled != 0U; }
    bool frspec_enabled() const override { return info_.frspec_enabled != 0U; }
    uint32_t frspec_shortlist_size() const override { return info_.frspec_shortlist_size; }
    bool metal_cache_hit() const override { return info_.metal_cache_hit != 0U; }
    double pso_load_ms() const override { return info_.pso_load_ms; }
    bool has_mtp() const override { return info_.has_mtp != 0U; }
    bool has_ple() const override { return false; }

    void set_glm53_exec_mode(StrataGlm53ExecMode mode) override {
        if (eng_) {
            strata_glm53_gpu_set_exec_mode(eng_, mode);
            strata_glm53_gpu_get_info(eng_, &info_);
        }
    }
    void reset_glm53_hot_cache() override {
        if (eng_ && !glm53_is_ram_resident()) strata_glm53_gpu_reset_hot_cache(eng_);
    }
    uint32_t glm53_hot_resident() const override {
        if (!eng_) return 0U;
        StrataGlm53GpuInfo cur{};
        strata_glm53_gpu_get_info(eng_, &cur);
        return cur.hot_cache_resident_experts;
    }
    uint32_t glm53_hot_capacity() const override { return info_.hot_cache_capacity_experts; }
    double glm53_hot_cache_gib() const override {
        return static_cast<double>(info_.hot_cache_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    double glm53_shared_mtl_gib() const override {
        return static_cast<double>(info_.hot_cache_bytes + info_.ring_dma_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    double glm53_experts_nvme_gib() const override {
        return static_cast<double>(info_.experts_nvme_file_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    bool glm53_ane_verified() const override { return info_.ane_verified_physical; }

    uint32_t max_seq_len() const override { return info_.max_seq_len; }
    uint32_t seq_pos() const override { return strata_glm53_gpu_seq_pos(eng_); }

    double uma_mapped_gib() const override {
        const uint64_t total_b = glm53_is_ram_resident()
            ? (info_.spine_mapped_bytes + info_.uma_part0_mapped_bytes +
               info_.uma_part1_mapped_bytes + info_.mtp_experts_mapped_bytes)
            : info_.spine_mapped_bytes;
        return static_cast<double>(total_b) / (1024.0 * 1024.0 * 1024.0);
    }
    double ple_nvme_gib() const override { return 0.0; }
    double ple_lfu_mib() const override { return 0.0; }
    double deltanet_mib() const override {
        return static_cast<double>(info_.kda_state_bytes) / (1024.0 * 1024.0);
    }
    double ple_state_mib() const override { return 0.0; }
    uint64_t kv_bytes_per_tok() const override { return info_.kv_cache_bytes_per_token; }
    uint64_t active_bytes_per_tok() const override { return info_.active_bytes_per_token; }

    uint32_t token_think_start() const override { return STRATA_GLM53_TOKEN_THINK_START; }
    uint32_t token_think_end() const override { return STRATA_GLM53_TOKEN_THINK_END; }
    bool is_eos(uint32_t tok) const override { return strata_glm53_is_eos_token(&bdl_.cfg, tok); }

    bool encode(const std::string& text, bool parse_special, std::vector<uint32_t>* out_ids) const override {
        const uint32_t cap = static_cast<uint32_t>(std::max<size_t>(text.size() + 128U, 256U));
        out_ids->assign(cap, 0U);
        uint32_t n = 0U;
        if (strata_glm53_encode(
                bdl_.tokenizer, text.c_str(), text.size(), parse_special, out_ids->data(), cap, &n) != 0) {
            return false;
        }
        out_ids->resize(n);
        return true;
    }

    std::string decode(const uint32_t* ids, uint32_t n, bool skip_special) const override {
        if (!ids || n == 0U) return {};
        const size_t cap = std::max<size_t>(static_cast<size_t>(n) * 32U + 256U, 1024U);
        std::vector<char> buf(cap, '\0');
        size_t len = 0U;
        if (strata_glm53_decode(bdl_.tokenizer, ids, n, skip_special, buf.data(), cap, &len) != 0) {
            return {};
        }
        return std::string(buf.data(), len);
    }

    void stream_reset() override {
        strata_glm53_stream_decoder_init(&sdec_);
    }

    std::string stream_step(uint32_t token_id, bool skip_special) override {
        char piece[256];
        size_t len = 0U;
        if (strata_glm53_stream_decode_step(
                bdl_.tokenizer, &sdec_, token_id, skip_special, piece, sizeof(piece), &len) == 0 && len > 0U) {
            return std::string(piece, len);
        }
        return {};
    }

    std::string format_system_block(
        const std::string& system_prompt,
        bool /*enable_thinking*/,
        const std::string& effort
    ) const override {
        return format_glm53_system_block(system_prompt, effort);
    }

    void reset() override {
        strata_glm53_gpu_reset(eng_);
        strata_glm53_gpu_reset_counters(eng_);
    }

    void penalties_reset() override {
        strata_glm53_gpu_penalties_reset(eng_);
    }

    int prefill(
        const uint32_t* ids,
        uint32_t n,
        const ChatSamplingSpec& samp,
        uint32_t* next_tok_out,
        StepDeltaTelemetry* tel_out
    ) override {
        StrataGlm53PerfCounters c0{}, c1{};
        strata_glm53_gpu_get_counters(eng_, &c0);

        StrataGlm53SamplingParams sp{};
        sp.greedy             = samp.greedy;
        sp.temperature        = samp.temperature;
        sp.top_k              = samp.top_k;
        sp.top_p              = samp.top_p;
        sp.rng_seed           = samp.rng_seed;
        sp.presence_penalty   = samp.presence_penalty;
        sp.frequency_penalty  = samp.frequency_penalty;
        sp.repetition_penalty = samp.repetition_penalty;
        sp.penalty_window     = samp.penalty_window;

        int rc = 0;
        @autoreleasepool {
            rc = strata_glm53_gpu_prefill(eng_, ids, n, &sp, next_tok_out, nullptr);
        }
        if (rc == 0) prefill_toks_ += n;
        strata_glm53_gpu_get_counters(eng_, &c1);
        if (tel_out) {
            *tel_out = StepDeltaTelemetry{};
            tel_out->glm_hot_hits   = c1.uma_hot_hits - c0.uma_hot_hits;
            tel_out->glm_nvme_reads = c1.nvme_cold_reads - c0.nvme_cold_reads;
            tel_out->glm_nvme_bytes = c1.nvme_bytes_read - c0.nvme_bytes_read;
            tel_out->glm_nvme_sec   = std::max(0.0, c1.nvme_io_time_sec - c0.nvme_io_time_sec);
            tel_out->glm_active_exp = c1.active_experts_evaluated - c0.active_experts_evaluated;
            tel_out->glm_ran_on_ane = c1.ran_on_ane - c0.ran_on_ane;
            tel_out->glm_ane_calls  = c1.ane_lookahead_calls - c0.ane_lookahead_calls;
            tel_out->glm_ane_hits   = c1.ane_lookahead_hits - c0.ane_lookahead_hits;
        }
        return rc;
    }

    int decode_step(
        uint32_t pending_tok,
        uint32_t draft_k,
        const ChatSamplingSpec& samp,
        uint32_t* emitted_out,
        uint32_t* num_emitted_out,
        StepDeltaTelemetry* tel_out
    ) override {
        StrataGlm53PerfCounters c0{}, c1{};
        strata_glm53_gpu_get_counters(eng_, &c0);

        StrataGlm53SamplingParams sp{};
        sp.greedy             = samp.greedy;
        sp.temperature        = samp.temperature;
        sp.top_k              = samp.top_k;
        sp.top_p              = samp.top_p;
        sp.rng_seed           = samp.rng_seed;
        sp.presence_penalty   = samp.presence_penalty;
        sp.frequency_penalty  = samp.frequency_penalty;
        sp.repetition_penalty = samp.repetition_penalty;
        sp.penalty_window     = samp.penalty_window;

        const uint32_t eff_k = has_mtp()
            ? std::min<uint32_t>(draft_k, STRATA_GLM53_MAX_SPEC_DRAFT_K)
            : 0U;

        int rc = 0;
        @autoreleasepool {
            if (eff_k == 0U) {
                uint32_t nxt = 0U;
                rc = strata_glm53_gpu_decode_step(eng_, pending_tok, &sp, &nxt, nullptr);
                if (rc == 0) {
                    emitted_out[0]   = nxt;
                    *num_emitted_out = 1U;
                }
            } else {
                uint32_t acc = 0U;
                rc = strata_glm53_gpu_spec_decode_step(
                    eng_, pending_tok, eff_k, &sp, emitted_out, num_emitted_out, &acc);
            }
        }
        strata_glm53_gpu_get_counters(eng_, &c1);
        if (tel_out) {
            *tel_out = StepDeltaTelemetry{};
            tel_out->mtp_drafted    = c1.mtp_drafted_tokens - c0.mtp_drafted_tokens;
            tel_out->mtp_accepted   = c1.mtp_accepted_tokens - c0.mtp_accepted_tokens;
            tel_out->glm_hot_hits   = c1.uma_hot_hits - c0.uma_hot_hits;
            tel_out->glm_nvme_reads = c1.nvme_cold_reads - c0.nvme_cold_reads;
            tel_out->glm_nvme_bytes = c1.nvme_bytes_read - c0.nvme_bytes_read;
            tel_out->glm_nvme_sec   = std::max(0.0, c1.nvme_io_time_sec - c0.nvme_io_time_sec);
            tel_out->glm_active_exp = c1.active_experts_evaluated - c0.active_experts_evaluated;
            tel_out->glm_ran_on_ane = c1.ran_on_ane - c0.ran_on_ane;
            tel_out->glm_ane_calls  = c1.ane_lookahead_calls - c0.ane_lookahead_calls;
            tel_out->glm_ane_hits   = c1.ane_lookahead_hits - c0.ane_lookahead_hits;
        }
        return rc;
    }

    void* checkpoint_create() override {
        return strata_glm53_checkpoint_create(eng_);
    }
    void checkpoint_destroy(void* ckpt) override {
        strata_glm53_checkpoint_destroy(static_cast<StrataGlm53Checkpoint*>(ckpt));
    }
    int checkpoint_save(void* ckpt) override {
        return strata_glm53_checkpoint_save(eng_, static_cast<StrataGlm53Checkpoint*>(ckpt));
    }
    int checkpoint_restore(const void* ckpt) override {
        return strata_glm53_checkpoint_restore(eng_, static_cast<const StrataGlm53Checkpoint*>(ckpt));
    }
    uint64_t checkpoint_bytes(const void* ckpt) const override {
        return strata_glm53_checkpoint_bytes(static_cast<const StrataGlm53Checkpoint*>(ckpt));
    }

    /* ---- Step 4 compaction & persistent state plumbing (`GLM-5.3-Flash`) ---- */
    bool compaction_supported() const override { return eng_ != nullptr; }

    void* compact_slot_create(uint32_t uma_kv_pool_tokens) override {
        StrataGlm53ContextSlot* s = nullptr;
        @autoreleasepool {
            s = strata_glm53_slot_create(eng_, uma_kv_pool_tokens, nullptr);
        }
        if (s) slots_.push_back(s);
        return s;
    }
    void compact_slot_destroy(void* slot) override {
        if (!slot) return;
        auto* s = static_cast<StrataGlm53ContextSlot*>(slot);
        slots_.erase(std::remove(slots_.begin(), slots_.end(), s), slots_.end());
        strata_glm53_slot_destroy(s);
    }
    void compact_bind_slot(void* slot) override {
        strata_glm53_bind_slot(eng_, static_cast<StrataGlm53ContextSlot*>(slot));
    }
    void* compact_bound_slot() const override { return strata_glm53_bound_slot(eng_); }
    uint32_t compact_slot_seq_pos(const void* slot) const override {
        return strata_glm53_slot_seq_pos(static_cast<const StrataGlm53ContextSlot*>(slot));
    }
    bool compact_slot_is_paged(const void* slot) const override {
        return slot ? strata_glm53_slot_is_paged(static_cast<const StrataGlm53ContextSlot*>(slot)) : false;
    }
    uint64_t compact_slot_bytes(const void* slot) const override {
        return slot ? strata_glm53_slot_bytes(static_cast<const StrataGlm53ContextSlot*>(slot)) : 0ULL;
    }
    void compact_get_stats(const void* slot, StrataQwenFlashCompactionStats* out) const override {
        if (!out) return;
        *out = StrataQwenFlashCompactionStats{};
        StrataGlm53CompactionStats gs{};
        strata_glm53_slot_get_stats(static_cast<const StrataGlm53ContextSlot*>(slot), &gs);
        static_assert(sizeof(gs) == sizeof(*out), "CompactionStats layout parity");
        std::memcpy(out, &gs, sizeof(*out));
    }
    void compact_reset_stats(void* slot) override {
        if (slot) strata_glm53_slot_reset_stats(static_cast<StrataGlm53ContextSlot*>(slot));
    }
    void compact_reset_bound() override {
        strata_glm53_gpu_reset(eng_);
    }
    int compact_prefill_slice(const uint32_t* ids, uint32_t n, bool final_logits, uint32_t* next) override {
        int rc = 0;
        @autoreleasepool {
            rc = strata_glm53_gpu_prefill_slice(eng_, ids, n, final_logits, next);
        }
        if (rc == 0) slice_toks_ += n;
        return rc;
    }
    bool compact_ane_ready(std::string* note) const override {
        char buf[256] = {};
        const int ok = strata_glm53_ane_kv_probe_ready(eng_, buf, sizeof(buf));
        if (note) *note = buf;
        return ok == 1;
    }
    uint64_t prefill_tokens_total() const override { return prefill_toks_; }
    uint64_t slice_tokens_total() const override { return slice_toks_; }
    const char* engine_id_tag() const override { return "glm-5.3-flash"; }
    bool supports_session_save() const override { return true; }
    int save_state_fd(int fd, uint64_t* bytes_out) override {
        return strata_glm53_state_save_fd(eng_, fd, bytes_out);
    }
    int load_state_fd(int fd, uint64_t* bytes_out) override {
        return strata_glm53_state_load_fd(eng_, fd, bytes_out);
    }

private:
    std::vector<StrataGlm53ContextSlot*> slots_;
    uint64_t                 prefill_toks_  = 0ULL;
    uint64_t                 slice_toks_    = 0ULL;
    StrataGlm53Bundle        bdl_{};
    bool                     bundle_loaded_ = false;
    StrataGlm53GpuEngine*    eng_           = nullptr;
    StrataGlm53GpuInfo       info_{};
    StrataGlm53StreamDecoder sdec_{};
    double                   init_ms_       = 0.0;
};

/* ---- 3.4 `thinkingmachines/Inkling-2-Mini` (26B Looped NVFP4/MXFP8 + 3-Layer DSpark Block-8 MTP) ---- */

class Inkling2ChatBackend final : public IChatBackend {
public:
    Inkling2ChatBackend() {
        std::memset(&bdl_, 0, sizeof(bdl_));
        bdl_.uma_fd = -1;
    }

    ~Inkling2ChatBackend() override {
        if (vision_eng_) {
            strata_inkling2_vision_destroy(vision_eng_);
            vision_eng_ = nullptr;
        }
        if (eng_) {
            strata_inkling2_bind_slot(eng_, nullptr);
            strata_inkling2_gpu_destroy(eng_);
            eng_ = nullptr;
        }
        if (bundle_loaded_) {
            strata_inkling2_free_bundle(&bdl_);
            bundle_loaded_ = false;
        }
    }

    bool init(const ChatCliConfig& cfg) override {
        const auto t0 = SteadyClock::now();
        if (strata_inkling2_load_bundle(cfg.bundle_dir.c_str(), &bdl_) != 0) {
            return false;
        }
        bundle_loaded_ = true;
        const uint32_t req_seq = (cfg.max_seq_len > 0U)
            ? std::min<uint32_t>(cfg.max_seq_len, STRATA_INKLING2_MAX_SEQ_LEN_CAP)
            : STRATA_INKLING2_DEFAULT_MAX_SEQ_LEN;
        eng_ = strata_inkling2_gpu_create_ex(&bdl_, req_seq, cfg.kv_dtype);
        if (!eng_) {
            return false;
        }
        if (cfg.int8_prefill_set) strata_inkling2_gpu_set_int8_prefill(eng_, cfg.int8_prefill);
        strata_inkling2_gpu_set_adaptive_k(eng_, cfg.adaptive_k && cfg.draft_k > 0U);
        strata_inkling2_gpu_set_frspec(eng_, cfg.frspec);
        strata_inkling2_gpu_set_beam_tree(eng_, cfg.tree_spec);
        strata_inkling2_gpu_set_bitplane_head(eng_, cfg.bitplane_head);

        char vis_path[512] = {};
        if (strata_inkling2_vision_resolve_path(cfg.bundle_dir.c_str(), vis_path, sizeof(vis_path)) == 1) {
            vision_eng_ = strata_inkling2_vision_create_default(vis_path);
            if (vision_eng_) {
                (void)strata_inkling2_vision_get_info(vision_eng_, &vision_info_);
                if (vision_info_.text_hidden_size != bdl_.hdr.cfg.hidden_size) {
                    std::fprintf(stderr,
                                 "[chat-inkling2] ERROR: Vision sidecar text_hidden_size (%u) != text hidden_size (%u) — disabling vision\n",
                                 vision_info_.text_hidden_size, bdl_.hdr.cfg.hidden_size);
                    strata_inkling2_vision_destroy(vision_eng_);
                    vision_eng_ = nullptr;
                    vision_info_ = {};
                }
            }
        }
        strata_inkling2_gpu_get_info(eng_, &info_);

        @autoreleasepool {
            strata_inkling2_gpu_reset(eng_);
            strata_inkling2_gpu_penalties_reset(eng_);
            strata_inkling2_gpu_reset_counters(eng_);
            const uint32_t warm_ids[8] = {
                STRATA_INKLING2_TOKEN_MSG_SYSTEM, STRATA_INKLING2_TOKEN_CONTENT_TEXT, 100U, STRATA_INKLING2_TOKEN_END_MSG,
                STRATA_INKLING2_TOKEN_MSG_USER,   STRATA_INKLING2_TOKEN_CONTENT_TEXT, 100U, STRATA_INKLING2_TOKEN_END_MSG
            };
            StrataInkling2SamplingParams greedy{};
            greedy.greedy             = true;
            greedy.repetition_penalty = 1.0f;
            uint32_t nt1 = 0U, nt2 = 0U;
            (void)strata_inkling2_gpu_prefill(eng_, warm_ids, 8U, &greedy, &nt1, nullptr);
            (void)strata_inkling2_gpu_decode_step(eng_, nt1, &greedy, &nt2, nullptr);
            if (info_.mtp_enabled && bdl_.hdr.mtp.enabled) {
                uint32_t em[STRATA_INKLING2_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t ne = 0U, na = 0U;
                (void)strata_inkling2_gpu_spec_decode_step(
                    eng_, nt2, STRATA_INKLING2_MAX_SPEC_DRAFT_K, &greedy, em, &ne, &na);
            }
            strata_inkling2_gpu_reset(eng_);
            strata_inkling2_gpu_penalties_reset(eng_);
            strata_inkling2_gpu_reset_counters(eng_);
        }
        strata_inkling2_stream_decoder_init(&sdec_);
        init_ms_ = chat_elapsed_ms(t0, SteadyClock::now());
        return true;
    }

    const char* model_name() const override {
        return "thinkingmachines/Inkling-2-Mini (26B Looped NVFP4/MXFP8 + 3L DSpark MTP)";
    }
    const char* device_name() const override { return info_.device_name; }
    bool zero_copy() const override { return info_.zero_copy_mmap; }
    double init_wall_ms() const override { return init_ms_; }
    bool is_flash() const override { return false; }
    bool is_glm53() const override { return false; }
    bool is_inkling2() const override { return true; }
    const char* kv_cache_name() const override { return chat_kv_cache_label(info_.kv_dtype); }
    bool int8_prefill_enabled() const override { return info_.int8_prefill_enabled != 0U; }
    bool adaptive_k_enabled() const override { return eng_ && strata_inkling2_gpu_get_adaptive_k(eng_); }
    bool frspec_enabled() const override {
        return eng_ && strata_inkling2_gpu_get_frspec(eng_) && (bdl_.hdr.mtp.frspec.enabled != 0U);
    }
    uint32_t frspec_shortlist_size() const override {
        return frspec_enabled() ? bdl_.hdr.mtp.frspec.shortlist_size : 0U;
    }
    bool tree_spec_enabled() const override { return eng_ && strata_inkling2_gpu_get_beam_tree(eng_); }
    bool bitplane_head_enabled() const override { return eng_ && strata_inkling2_gpu_get_bitplane_head(eng_); }
    bool has_vision() const override { return vision_eng_ != nullptr && vision_info_.ready != 0U; }
    bool vision_metal_cache_hit() const override { return has_vision() && vision_info_.metal_cache_hit != 0U; }
    double vision_pso_load_ms() const override { return has_vision() ? vision_info_.pso_load_ms : 0.0; }
    bool metal_cache_hit() const override { return info_.metal_cache_hit != 0U; }
    double pso_load_ms() const override { return info_.pso_load_ms; }
    bool has_mtp() const override { return info_.mtp_enabled && (bdl_.hdr.mtp.enabled != 0U); }
    bool has_ple() const override { return false; }

    uint32_t max_seq_len() const override { return info_.max_seq_len; }
    uint32_t seq_pos() const override { return strata_inkling2_gpu_seq_pos(eng_); }
    double uma_mapped_gib() const override {
        return static_cast<double>(info_.uma_mapped_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    double ple_nvme_gib() const override { return 0.0; }
    double ple_lfu_mib() const override { return 0.0; }
    double deltanet_mib() const override {
        return static_cast<double>(info_.swa_kv_bytes_allocated + info_.sconv_state_bytes) / (1024.0 * 1024.0);
    }
    double ple_state_mib() const override { return 0.0; }
    uint64_t kv_bytes_per_tok() const override {
        return (info_.max_seq_len > 0U) ? (info_.global_kv_bytes_allocated / info_.max_seq_len) : 0ULL;
    }
    uint64_t active_bytes_per_tok() const override { return info_.active_bytes_per_token; }

    uint32_t token_think_start() const override { return STRATA_INKLING2_TOKEN_CONTENT_THINK; }
    uint32_t token_think_end() const override { return STRATA_INKLING2_TOKEN_END_MSG; }
    bool is_eos(uint32_t tok) const override {
        return tok == STRATA_INKLING2_TOKEN_EOS ||
               tok == STRATA_INKLING2_TOKEN_ENDOFTEXT ||
               tok == bdl_.hdr.cfg.eos_token_id;
    }

    bool encode(const std::string& text, bool parse_special, std::vector<uint32_t>* out_ids) const override {
        const uint32_t cap = static_cast<uint32_t>(std::max<size_t>(text.size() + 128U, 256U));
        out_ids->assign(cap, 0U);
        uint32_t n = 0U;
        if (strata_inkling2_encode(
                bdl_.tokenizer, text.c_str(), text.size(), parse_special, out_ids->data(), cap, &n) != 0) {
            return false;
        }
        out_ids->resize(n);
        return true;
    }

    std::string decode(const uint32_t* ids, uint32_t n, bool skip_special) const override {
        if (!ids || n == 0U) return {};
        const size_t cap = std::max<size_t>(static_cast<size_t>(n) * 32U + 256U, 1024U);
        std::vector<char> buf(cap, '\0');
        size_t len = 0U;
        if (strata_inkling2_decode(bdl_.tokenizer, ids, n, skip_special, buf.data(), cap, &len) != 0) {
            return {};
        }
        return std::string(buf.data(), len);
    }

    void stream_reset() override { strata_inkling2_stream_decoder_init(&sdec_); }

    std::string stream_step(uint32_t token_id, bool skip_special) override {
        char piece[256];
        size_t len = 0U;
        if (strata_inkling2_stream_decode_step(
                bdl_.tokenizer, &sdec_, token_id, skip_special, piece, sizeof(piece), &len) == 0 &&
            len > 0U) {
            return std::string(piece, len);
        }
        return {};
    }

    std::string format_system_block(
        const std::string& system_prompt,
        bool enable_thinking,
        const std::string& effort
    ) const override {
        std::string out;
        const std::string sys = chat_trim_ascii(system_prompt);
        if (!sys.empty()) {
            out += "<|message_system|><|content_text|>" + sys + "<|end_message|>";
        }
        const std::string eff_num = chat_inkling2_effort_to_num_str(enable_thinking, effort);
        out += "<|message_system|><|content_text|>Thinking effort level: " + eff_num + "<|end_message|>";
        return out;
    }

    void reset() override { strata_inkling2_gpu_reset(eng_); }
    void penalties_reset() override { strata_inkling2_gpu_penalties_reset(eng_); }

    int prefill(
        const uint32_t* ids,
        uint32_t n,
        const ChatSamplingSpec& samp,
        uint32_t* next_tok_out,
        StepDeltaTelemetry* tel_out
    ) override {
        return prefill_with_images(ids, n, nullptr, 0U, samp, next_tok_out, tel_out);
    }

    int encode_image_file(
        const std::string& path,
        std::vector<float>* out_emb_f32,
        uint32_t* out_num_tokens,
        uint32_t* out_grid_h,
        uint32_t* out_grid_w,
        double* out_ms
    ) override {
        if (!vision_eng_ || vision_info_.ready == 0U) return -1;
        const auto t0 = SteadyClock::now();
        int rc = -1;
        @autoreleasepool {
            rc = strata_inkling2_vision_encode_image_file(
                vision_eng_, path.c_str(), out_emb_f32, out_num_tokens, out_grid_h, out_grid_w);
        }
        if (out_ms) *out_ms = chat_elapsed_ms(t0, SteadyClock::now());
        return rc;
    }

    int prefill_with_images(
        const uint32_t* ids,
        uint32_t n,
        const StrataInkling2ImageSpan* spans,
        uint32_t num_spans,
        const ChatSamplingSpec& samp,
        uint32_t* next_tok_out,
        StepDeltaTelemetry* tel_out
    ) override {
        if (tel_out) *tel_out = StepDeltaTelemetry{};
        StrataInkling2SamplingParams p{};
        p.greedy             = samp.greedy;
        p.temperature        = samp.temperature;
        p.top_k              = samp.top_k;
        p.top_p              = samp.top_p;
        p.rng_seed           = samp.rng_seed;
        p.presence_penalty   = samp.presence_penalty;
        p.frequency_penalty  = samp.frequency_penalty;
        p.repetition_penalty = samp.repetition_penalty;
        p.penalty_window     = samp.penalty_window;
        int rc = 0;
        @autoreleasepool {
            if (spans && num_spans > 0U) {
                rc = strata_inkling2_gpu_prefill_with_images(
                    eng_, ids, n, spans, num_spans, &p, next_tok_out, nullptr);
            } else {
                rc = strata_inkling2_gpu_prefill(eng_, ids, n, &p, next_tok_out, nullptr);
            }
        }
        if (rc == 0) prefill_toks_ += n;
        return rc;
    }

    int decode_step(
        uint32_t pending_tok,
        uint32_t draft_k,
        const ChatSamplingSpec& samp,
        uint32_t* emitted_out,
        uint32_t* num_emitted_out,
        StepDeltaTelemetry* tel_out
    ) override {
        if (tel_out) *tel_out = StepDeltaTelemetry{};
        StrataInkling2SamplingParams p{};
        p.greedy             = samp.greedy;
        p.temperature        = samp.temperature;
        p.top_k              = samp.top_k;
        p.top_p              = samp.top_p;
        p.rng_seed           = samp.rng_seed;
        p.presence_penalty   = samp.presence_penalty;
        p.frequency_penalty  = samp.frequency_penalty;
        p.repetition_penalty = samp.repetition_penalty;
        p.penalty_window     = samp.penalty_window;
        StrataInkling2PerfCounters c0{}, c1{};
        strata_inkling2_gpu_get_counters(eng_, &c0);
        int rc = 0;
        @autoreleasepool {
            if (draft_k == 0U || !has_mtp()) {
                uint32_t nxt = 0U;
                rc = strata_inkling2_gpu_decode_step(eng_, pending_tok, &p, &nxt, nullptr);
                if (rc == 0) {
                    emitted_out[0]   = nxt;
                    *num_emitted_out = 1U;
                }
            } else {
                const uint32_t k = std::min<uint32_t>(draft_k, STRATA_INKLING2_MAX_SPEC_DRAFT_K);
                uint32_t acc = 0U;
                rc = strata_inkling2_gpu_spec_decode_step(
                    eng_, pending_tok, k, &p, emitted_out, num_emitted_out, &acc);
            }
        }
        if (rc == 0 && tel_out) {
            strata_inkling2_gpu_get_counters(eng_, &c1);
            tel_out->mtp_drafted  = c1.mtp_drafted_tokens - c0.mtp_drafted_tokens;
            tel_out->mtp_accepted = c1.mtp_accepted_tokens - c0.mtp_accepted_tokens;
        }
        return rc;
    }

    void* checkpoint_create() override { return strata_inkling2_checkpoint_create(eng_); }
    void checkpoint_destroy(void* ckpt) override {
        strata_inkling2_checkpoint_destroy(static_cast<StrataInkling2Checkpoint*>(ckpt));
    }
    int checkpoint_save(void* ckpt) override {
        return strata_inkling2_checkpoint_save(eng_, static_cast<StrataInkling2Checkpoint*>(ckpt));
    }
    int checkpoint_restore(const void* ckpt) override {
        return strata_inkling2_checkpoint_restore(eng_, static_cast<const StrataInkling2Checkpoint*>(ckpt));
    }
    uint64_t checkpoint_bytes(const void* /*ckpt*/) const override {
        return info_.swa_kv_bytes_allocated + info_.sconv_state_bytes;
    }
    uint64_t prefill_tokens_total() const override { return prefill_toks_; }
    const char* engine_id_tag() const override { return "inkling-2-mini"; }

private:
    uint64_t                    prefill_toks_  = 0ULL;
    StrataInkling2Bundle        bdl_{};
    bool                        bundle_loaded_ = false;
    StrataInkling2GpuEngine*    eng_           = nullptr;
    StrataInkling2GpuInfo       info_{};
    StrataInkling2VisionEngine* vision_eng_    = nullptr;
    StrataInkling2VisionInfo    vision_info_{};
    StrataInkling2StreamDecoder sdec_{};
    double                      init_ms_       = 0.0;
};

/* ============================================================================
 * 4. Stateful Multi-Turn Session Controller & Checkpoint Ring
 * ============================================================================ */

struct ChatTurnRecord {
    std::string           user_text;
    std::string           think_text;
    std::string           clean_reply;
    std::string           committed_suffix;
    std::vector<uint32_t> generated_ids;
    uint32_t              seq_pos_before    = 0U;
    uint32_t              seq_pos_after     = 0U;
    uint32_t              prompt_delta_toks = 0U;
    uint32_t              committed_toks    = 0U;
    double                prefill_ms        = 0.0;
    double                decode_ms         = 0.0;
    double                decode_tok_s      = 0.0;
    double                commit_ms         = 0.0;
    bool                  hit_max_tokens    = false;
};

static void split_generated_think_and_reply(
    const IChatBackend& backend,
    const std::vector<uint32_t>& gen_ids,
    bool thinking_was_enabled,
    std::string* out_think_trimmed,
    std::string* out_reply_trimmed,
    bool* out_saw_think_end
) {
    if (backend.is_inkling2()) {
        std::vector<uint32_t> th_ids, ans_ids;
        bool in_think = false;
        bool in_text  = !thinking_was_enabled;
        bool saw_end  = false;
        for (uint32_t t : gen_ids) {
            if (t == STRATA_INKLING2_TOKEN_CONTENT_THINK) {
                in_think = true;
                in_text  = false;
                continue;
            }
            if (t == STRATA_INKLING2_TOKEN_CONTENT_TEXT) {
                in_think = false;
                in_text  = true;
                continue;
            }
            if (t == STRATA_INKLING2_TOKEN_END_MSG) {
                if (in_think) saw_end = true;
                in_think = false;
                in_text  = false;
                continue;
            }
            if (t == STRATA_INKLING2_TOKEN_MSG_MODEL || backend.is_eos(t)) {
                continue;
            }
            if (in_think) th_ids.push_back(t);
            else if (in_text) ans_ids.push_back(t);
        }
        if (out_saw_think_end) *out_saw_think_end = saw_end || !thinking_was_enabled;
        *out_think_trimmed = !th_ids.empty()
            ? chat_trim_ascii(backend.decode(th_ids.data(), static_cast<uint32_t>(th_ids.size()), true))
            : std::string();
        *out_reply_trimmed = !ans_ids.empty()
            ? chat_trim_ascii(backend.decode(ans_ids.data(), static_cast<uint32_t>(ans_ids.size()), true))
            : std::string();
        return;
    }

    const uint32_t tok_start = backend.token_think_start();
    const uint32_t tok_end   = backend.token_think_end();
    size_t end_idx = gen_ids.size();
    for (size_t i = 0U; i < gen_ids.size(); ++i) {
        if (gen_ids[i] == tok_end) {
            end_idx = i;
        }
    }

    if (end_idx < gen_ids.size()) {
        if (out_saw_think_end) *out_saw_think_end = true;
        std::vector<uint32_t> th_ids;
        th_ids.reserve(end_idx);
        for (size_t i = 0U; i < end_idx; ++i) {
            if (gen_ids[i] != tok_start && gen_ids[i] != tok_end) {
                th_ids.push_back(gen_ids[i]);
            }
        }
        const std::string th = !th_ids.empty()
            ? backend.decode(th_ids.data(), static_cast<uint32_t>(th_ids.size()), /*skip_special=*/true)
            : std::string();
        std::vector<uint32_t> ans_ids;
        ans_ids.reserve(gen_ids.size() - (end_idx + 1U));
        for (size_t i = end_idx + 1U; i < gen_ids.size(); ++i) {
            if (gen_ids[i] != tok_start && gen_ids[i] != tok_end) {
                ans_ids.push_back(gen_ids[i]);
            }
        }
        const std::string ans = !ans_ids.empty()
            ? backend.decode(ans_ids.data(), static_cast<uint32_t>(ans_ids.size()), /*skip_special=*/true)
            : std::string();
        *out_think_trimmed = chat_trim_ascii(th);
        *out_reply_trimmed = chat_trim_ascii(ans);
        return;
    }

    if (out_saw_think_end) *out_saw_think_end = false;
    if (thinking_was_enabled) {
        /* Truncated inside <think> before </think> */
        const std::string th = backend.decode(
            gen_ids.data(), static_cast<uint32_t>(gen_ids.size()), /*skip_special=*/true);
        *out_think_trimmed = chat_trim_ascii(th);
        out_reply_trimmed->clear();
    } else {
        out_think_trimmed->clear();
        const std::string ans = backend.decode(
            gen_ids.data(), static_cast<uint32_t>(gen_ids.size()), /*skip_special=*/true);
        *out_reply_trimmed = chat_trim_ascii(ans);
    }
}

static bool chat_encode_append(
    IChatBackend* backend,
    const std::string& text,
    bool parse_special,
    std::vector<uint32_t>* out
) {
    if (text.empty()) return true;
    std::vector<uint32_t> tmp;
    if (!backend->encode(text, parse_special, &tmp)) return false;
    out->insert(out->end(), tmp.begin(), tmp.end());
    return true;
}

class ChatReplSession {
public:
    static constexpr size_t kUndoRingCap = 2U;

    /* Host-side transcript bookkeeping swapped atomically with a Tier 1 RCU promotion / `/undo-compact`. */
    struct Bookkeeping {
        std::vector<uint32_t>       committed_ids;
        std::vector<ChatTurnRecord> history;
        size_t                      history_floor     = 0U;
        std::string                 effective_system_block;
        size_t                      undo_valid_count  = 0U;
    };

    ChatReplSession(IChatBackend* backend, const ChatCliConfig& cfg)
        : backend_(backend), cfg_(cfg) {
        turn_start_ckpt_ = backend_->checkpoint_create();
        for (size_t i = 0U; i < kUndoRingCap; ++i) {
            undo_ring_[i] = backend_->checkpoint_create();
        }
        /* Warm checkpoint copy PSO & residency set before steady-state fault tracking */
        if (turn_start_ckpt_) {
            (void)backend_->checkpoint_save(turn_start_ckpt_);
            (void)backend_->checkpoint_restore(turn_start_ckpt_);
        }
        std::vector<uint32_t> warm_toks;
        const char* warm_str = backend_->is_inkling2()
            ? "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Warmup 123<|end_message|>"
            : (backend_->is_glm53()
                   ? "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Warmup 123<|assistant|><think></think>"
                   : "<|im_start|>user\nWarmup 123<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        (void)backend_->encode(warm_str, true, &warm_toks);
        if (!warm_toks.empty()) {
            (void)backend_->decode(warm_toks.data(), static_cast<uint32_t>(warm_toks.size()), true);
        }
        baseline_mem_ = query_chat_mem_snapshot(backend_->glm53_shared_mtl_gib());
    }

    ~ChatReplSession() {
        for (size_t i = 0U; i < kUndoRingCap; ++i) {
            if (undo_ring_[i]) {
                backend_->checkpoint_destroy(undo_ring_[i]);
                undo_ring_[i] = nullptr;
            }
        }
        if (turn_start_ckpt_) {
            backend_->checkpoint_destroy(turn_start_ckpt_);
            turn_start_ckpt_ = nullptr;
        }
    }

    bool ok() const {
        if (!turn_start_ckpt_) return false;
        for (size_t i = 0U; i < kUndoRingCap; ++i) {
            if (!undo_ring_[i]) return false;
        }
        return true;
    }
    void* turn_start_checkpoint() const { return turn_start_ckpt_; }
    const ChatMemSnapshot& baseline_mem() const { return baseline_mem_; }

    const std::vector<ChatTurnRecord>& history() const { return history_; }
    const std::string& effective_system_block() const { return effective_system_block_; }
    const std::vector<uint32_t>& committed_transcript_ids() const { return committed_ids_; }

    uint64_t single_checkpoint_bytes() const {
        return backend_->checkpoint_bytes(turn_start_ckpt_);
    }

    uint64_t total_checkpoint_bytes() const {
        uint64_t tot = backend_->checkpoint_bytes(turn_start_ckpt_);
        for (size_t i = 0U; i < kUndoRingCap; ++i) {
            if (undo_ring_[i]) tot += backend_->checkpoint_bytes(undo_ring_[i]);
        }
        return tot;
    }

    struct StagedChatImage {
        std::string        path;
        std::vector<float> emb_f32;
        uint32_t           num_tokens = 0U;
        uint32_t           grid_h     = 0U;
        uint32_t           grid_w     = 0U;
        double             encode_ms  = 0.0;
    };

    bool stage_image(const std::string& path, std::string* msg_out) {
        if (!backend_->is_inkling2()) {
            if (msg_out) *msg_out = "[vision] `/image` is supported on `chat-inkling2`";
            return false;
        }
        if (!backend_->has_vision()) {
            if (msg_out) *msg_out = "[vision] Vision encoder is not loaded (generate `inkling2_vision.strata` via `convert-inkling2-vision`)";
            return false;
        }
        StagedChatImage img{};
        img.path = path;
        const int rc = backend_->encode_image_file(
            path, &img.emb_f32, &img.num_tokens, &img.grid_h, &img.grid_w, &img.encode_ms);
        if (rc != 0 || img.num_tokens == 0U || img.emb_f32.empty()) {
            if (msg_out) *msg_out = "[vision] Failed to load/encode image `" + path + "` (rc=" + std::to_string(rc) + ")";
            return false;
        }
        char buf[384];
        std::snprintf(buf, sizeof(buf),
                      "[vision] Attached `%s` (%u vis toks, grid %ux%u, %.1f ms) — %zu image(s) queued for next turn",
                      path.c_str(), img.num_tokens, img.grid_h, img.grid_w, img.encode_ms,
                      staged_images_.size() + 1U);
        staged_images_.push_back(std::move(img));
        if (msg_out) *msg_out = buf;
        return true;
    }

    double reset_session() {
        const auto t0 = SteadyClock::now();
        backend_->reset();
        history_.clear();
        committed_ids_.clear();
        staged_images_.clear();
        effective_system_block_.clear();
        undo_valid_count_ = 0U;
        history_floor_    = 0U;
        last_rewind_ms_   = chat_elapsed_ms(t0, SteadyClock::now());
        return last_rewind_ms_;
    }

    bool undo_last_turn(double* rewind_ms_out, uint32_t* pos_before_out, uint32_t* pos_after_out) {
        if (history_.empty() || history_.size() <= history_floor_) return false;
        const uint32_t pos_before = backend_->seq_pos();
        const auto t0 = SteadyClock::now();

        const ChatTurnRecord popped = history_.back();
        history_.pop_back();

        if (committed_ids_.size() >= popped.seq_pos_before) {
            committed_ids_.resize(popped.seq_pos_before);
        }

        if (undo_valid_count_ > 0U) {
            const size_t slot = history_.size() % kUndoRingCap;
            if (undo_ring_[slot] && backend_->checkpoint_restore(undo_ring_[slot]) == 0) {
                --undo_valid_count_;
                if (history_.empty()) {
                    effective_system_block_.clear();
                }
                const double ms = chat_elapsed_ms(t0, SteadyClock::now());
                last_rewind_ms_ = ms;
                if (rewind_ms_out) *rewind_ms_out = ms;
                if (pos_before_out) *pos_before_out = pos_before;
                if (pos_after_out) *pos_after_out = backend_->seq_pos();
                return true;
            }
        }

        /* Deep undo fallback (> kUndoRingCap consecutive undos): replay remaining committed_ids_ */
        backend_->reset();
        undo_valid_count_ = 0U;
        if (history_.empty()) {
            effective_system_block_.clear();
            committed_ids_.clear();
        } else if (!committed_ids_.empty()) {
            ChatSamplingSpec greedy{};
            greedy.greedy = true;
            uint32_t ign = 0U;
            (void)backend_->prefill(
                committed_ids_.data(), static_cast<uint32_t>(committed_ids_.size()), greedy, &ign, nullptr);
        }
        const double ms = chat_elapsed_ms(t0, SteadyClock::now());
        last_rewind_ms_ = ms;
        if (rewind_ms_out) *rewind_ms_out = ms;
        if (pos_before_out) *pos_before_out = pos_before;
        if (pos_after_out) *pos_after_out = backend_->seq_pos();
        return true;
    }

    int run_turn(
        const std::string& raw_user_text,
        bool stream_stdout,
        bool ansi_style,
        ChatTurnRecord* out_record
    ) {
        const std::string user_trimmed = chat_trim_ascii(raw_user_text);
        const uint32_t pos_before      = backend_->seq_pos();

        /* 1. Snapshot pre-turn state into O(1) GPU undo ring (< 0.4 ms) */
        const size_t undo_slot = history_.size() % kUndoRingCap;
        if (!undo_ring_[undo_slot]) {
            undo_ring_[undo_slot] = backend_->checkpoint_create();
        }
        if (undo_ring_[undo_slot]) {
            const auto ts0 = SteadyClock::now();
            if (backend_->checkpoint_save(undo_ring_[undo_slot]) == 0) {
                last_save_ms_ = chat_elapsed_ms(ts0, SteadyClock::now());
                if (undo_valid_count_ < kUndoRingCap) ++undo_valid_count_;
            }
        }

        /* 2. Build Turn Prefix (GLM-5.3 single-pass NAX prefill vs Qwen ChatML 2-step prefill) */
        std::string sys_blk;
        if (history_.empty()) {
            sys_blk = backend_->format_system_block(cfg_.system_prompt, cfg_.enable_thinking, cfg_.effort);
            effective_system_block_ = sys_blk;
        }

        ChatSamplingSpec greedy_dummy{};
        greedy_dummy.greedy      = true;
        greedy_dummy.temperature = 0.0f;

        ChatSamplingSpec turn_samp{};
        turn_samp.greedy             = cfg_.greedy;
        turn_samp.temperature        = cfg_.temperature;
        turn_samp.top_k              = cfg_.top_k;
        turn_samp.top_p              = cfg_.top_p;
        turn_samp.rng_seed           = cfg_.seed + static_cast<uint64_t>(history_.size()) * 0x9E3779B97F4A7C15ULL;
        turn_samp.presence_penalty   = cfg_.presence_penalty;
        turn_samp.frequency_penalty  = cfg_.frequency_penalty;
        turn_samp.repetition_penalty = cfg_.repetition_penalty;
        turn_samp.penalty_window     = cfg_.penalty_window;
        backend_->penalties_reset();

        std::vector<uint32_t> hdr_ids, think_ids;
        StepDeltaTelemetry tel_hdr{}, tel_thk{};
        uint32_t ignored_tok = 0U;
        uint32_t first_tok   = 0U;
        int rc = 0;

        const auto t_pre0 = SteadyClock::now();
        if (backend_->is_inkling2()) {
            const std::string eff_num = chat_inkling2_effort_to_num_str(cfg_.enable_thinking, cfg_.effort);
            const bool direct_text = (!cfg_.enable_thinking || eff_num == "0");
            std::vector<StrataInkling2ImageSpan> img_spans;
            if (staged_images_.empty()) {
                if (!chat_encode_append(backend_, sys_blk, /*parse_special=*/true, &hdr_ids) ||
                    !chat_encode_append(backend_, "<|message_user|><|content_text|>", /*parse_special=*/true, &hdr_ids) ||
                    !chat_encode_append(backend_, user_trimmed, /*parse_special=*/false, &hdr_ids) ||
                    !chat_encode_append(backend_, "<|end_message|>", /*parse_special=*/true, &hdr_ids) ||
                    hdr_ids.empty()) {
                    std::fprintf(stderr, "[chat] Error: failed to tokenize Inkling-2 prompt\n");
                    return -1;
                }
            } else {
                std::vector<uint32_t> tmp;
                if (!sys_blk.empty()) {
                    if (!backend_->encode(sys_blk, /*parse_special=*/true, &tmp)) {
                        std::fprintf(stderr, "[chat] Error: failed to tokenize Inkling-2 system block\n");
                        return -1;
                    }
                    hdr_ids.insert(hdr_ids.end(), tmp.begin(), tmp.end());
                }
                for (const StagedChatImage& img : staged_images_) {
                    if (!backend_->encode("<|message_user|><|content_image|>", /*parse_special=*/true, &tmp)) {
                        std::fprintf(stderr, "[chat] Error: failed to tokenize Inkling-2 image header\n");
                        return -1;
                    }
                    hdr_ids.insert(hdr_ids.end(), tmp.begin(), tmp.end());
                    StrataInkling2ImageSpan sp{};
                    sp.token_offset   = static_cast<uint32_t>(hdr_ids.size());
                    sp.num_tokens     = img.num_tokens;
                    sp.embeddings_f32 = img.emb_f32.data();
                    img_spans.push_back(sp);
                    hdr_ids.insert(hdr_ids.end(), img.num_tokens, STRATA_INKLING2_TOKEN_IMAGE_PLACEHOLDER);
                    if (!backend_->encode("<|end_message|>", /*parse_special=*/true, &tmp)) {
                        std::fprintf(stderr, "[chat] Error: failed to tokenize Inkling-2 image end\n");
                        return -1;
                    }
                    hdr_ids.insert(hdr_ids.end(), tmp.begin(), tmp.end());
                }
                if (!user_trimmed.empty()) {
                    if (!backend_->encode("<|message_user|><|content_text|>", /*parse_special=*/true, &tmp)) return -1;
                    hdr_ids.insert(hdr_ids.end(), tmp.begin(), tmp.end());
                    if (!backend_->encode(user_trimmed, /*parse_special=*/false, &tmp)) return -1;
                    hdr_ids.insert(hdr_ids.end(), tmp.begin(), tmp.end());
                    if (!backend_->encode("<|end_message|>", /*parse_special=*/true, &tmp)) return -1;
                    hdr_ids.insert(hdr_ids.end(), tmp.begin(), tmp.end());
                }
            }
            if (direct_text) {
                if (!backend_->encode("<|message_model|><|content_text|>", /*parse_special=*/true, &think_ids) ||
                    think_ids.empty()) {
                    std::fprintf(stderr, "[chat] Error: failed to tokenize Inkling-2 direct assistant header\n");
                    return -1;
                }
            }
            if (pos_before + hdr_ids.size() + think_ids.size() + 16U >= backend_->max_seq_len()) {
                std::fprintf(stderr,
                             "[chat] Context window full (%u + %zu >= %u). Run /undo or /reset.\n",
                             pos_before, hdr_ids.size() + think_ids.size(), backend_->max_seq_len());
                return -2;
            }
            if (!img_spans.empty()) {
                rc = backend_->prefill_with_images(
                    hdr_ids.data(), static_cast<uint32_t>(hdr_ids.size()),
                    img_spans.data(), static_cast<uint32_t>(img_spans.size()),
                    direct_text ? greedy_dummy : turn_samp,
                    direct_text ? &ignored_tok : &first_tok, &tel_hdr);
                staged_images_.clear();
            } else {
                rc = backend_->prefill(
                    hdr_ids.data(), static_cast<uint32_t>(hdr_ids.size()),
                    direct_text ? greedy_dummy : turn_samp,
                    direct_text ? &ignored_tok : &first_tok, &tel_hdr);
            }
            if (rc != 0) {
                std::fprintf(stderr, "[chat] GPU prefill failed on Inkling-2 prompt (rc=%d)\n", rc);
                return rc;
            }
            const auto ts0 = SteadyClock::now();
            rc = backend_->checkpoint_save(turn_start_ckpt_);
            last_save_ms_ = chat_elapsed_ms(ts0, SteadyClock::now());
            if (rc != 0) {
                std::fprintf(stderr, "[chat] Failed to save turn_start_ckpt (rc=%d)\n", rc);
                return rc;
            }
            if (direct_text) {
                rc = backend_->prefill(
                    think_ids.data(), static_cast<uint32_t>(think_ids.size()), turn_samp, &first_tok, &tel_thk);
                if (rc != 0) {
                    std::fprintf(stderr, "[chat] GPU prefill failed on Inkling-2 direct prefix (rc=%d)\n", rc);
                    return rc;
                }
            }
        } else if (backend_->is_glm53()) {
            const std::string asst_open =
                cfg_.enable_thinking ? "<|assistant|><think>" : "<|assistant|><think></think>";
            if (!chat_encode_append(backend_, sys_blk, /*parse_special=*/true, &hdr_ids) ||
                !chat_encode_append(backend_, "<|user|>", /*parse_special=*/true, &hdr_ids) ||
                !chat_encode_append(backend_, user_trimmed, /*parse_special=*/false, &hdr_ids) ||
                !chat_encode_append(backend_, asst_open, /*parse_special=*/true, &hdr_ids) ||
                hdr_ids.empty()) {
                std::fprintf(stderr, "[chat] Error: failed to tokenize GLM-5.3 prompt\n");
                return -1;
            }
            if (pos_before + hdr_ids.size() + 16U >= backend_->max_seq_len()) {
                std::fprintf(stderr,
                             "[chat] Context window full (%u + %zu >= %u). Run /undo or /reset.\n",
                             pos_before, hdr_ids.size(), backend_->max_seq_len());
                return -2;
            }
            rc = backend_->prefill(
                hdr_ids.data(), static_cast<uint32_t>(hdr_ids.size()), turn_samp, &first_tok, &tel_hdr);
            if (rc != 0) {
                std::fprintf(stderr, "[chat] GPU prefill failed on GLM-5.3 prompt (rc=%d)\n", rc);
                return rc;
            }
            const auto ts0 = SteadyClock::now();
            rc = backend_->checkpoint_save(turn_start_ckpt_);
            last_save_ms_ = chat_elapsed_ms(ts0, SteadyClock::now());
            if (rc != 0) {
                std::fprintf(stderr, "[chat] Failed to save turn_start_ckpt (rc=%d)\n", rc);
                return rc;
            }
        } else {
            const std::string think_prefix_str =
                cfg_.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";

            if (!chat_encode_append(backend_, sys_blk, /*parse_special=*/true, &hdr_ids) ||
                !chat_encode_append(backend_, "<|im_start|>user\n", /*parse_special=*/true, &hdr_ids) ||
                !chat_encode_append(backend_, user_trimmed, /*parse_special=*/false, &hdr_ids) ||
                !chat_encode_append(backend_, "<|im_end|>\n<|im_start|>assistant\n", /*parse_special=*/true, &hdr_ids) ||
                hdr_ids.empty() ||
                !backend_->encode(think_prefix_str, /*parse_special=*/true, &think_ids) || think_ids.empty()) {
                std::fprintf(stderr, "[chat] Error: failed to tokenize ChatML prompt\n");
                return -1;
            }
            if (pos_before + hdr_ids.size() + think_ids.size() + 16U >= backend_->max_seq_len()) {
                std::fprintf(stderr,
                             "[chat] Context window full (%u + %zu >= %u). Run /undo or /reset.\n",
                             pos_before, hdr_ids.size() + think_ids.size(), backend_->max_seq_len());
                return -2;
            }
            rc = backend_->prefill(
                hdr_ids.data(), static_cast<uint32_t>(hdr_ids.size()), greedy_dummy, &ignored_tok, &tel_hdr);
            if (rc != 0) {
                std::fprintf(stderr, "[chat] GPU prefill failed on user header (rc=%d)\n", rc);
                return rc;
            }
            const auto ts0 = SteadyClock::now();
            rc = backend_->checkpoint_save(turn_start_ckpt_);
            last_save_ms_ = chat_elapsed_ms(ts0, SteadyClock::now());
            if (rc != 0) {
                std::fprintf(stderr, "[chat] Failed to save turn_start_ckpt (rc=%d)\n", rc);
                return rc;
            }
            rc = backend_->prefill(
                think_ids.data(), static_cast<uint32_t>(think_ids.size()), turn_samp, &first_tok, &tel_thk);
            if (rc != 0) {
                std::fprintf(stderr, "[chat] GPU prefill failed on think prefix (rc=%d)\n", rc);
                return rc;
            }
        }
        const auto t_pre1 = SteadyClock::now();

        const double prefill_ms = chat_elapsed_ms(t_pre0, t_pre1);
        uint64_t turn_ple_hits   = tel_hdr.ple_hits + tel_thk.ple_hits;
        uint64_t turn_ple_ios    = tel_hdr.ple_ios + tel_thk.ple_ios;
        uint64_t turn_mtp_draft  = 0ULL;
        uint64_t turn_mtp_acc    = 0ULL;
        uint64_t dec_glm_hot     = 0ULL;
        uint64_t dec_glm_exp     = 0ULL;
        uint64_t dec_glm_nvme_b  = 0ULL;
        double   dec_glm_nvme_s  = 0.0;
        uint64_t turn_glm_ane    = tel_hdr.glm_ran_on_ane + tel_thk.glm_ran_on_ane;
        uint64_t turn_glm_ane_c  = tel_hdr.glm_ane_calls + tel_thk.glm_ane_calls;
        uint64_t turn_glm_ane_h  = tel_hdr.glm_ane_hits + tel_thk.glm_ane_hits;

        /* 5. Live token-by-token streaming with styled `<think>...</think>` */
        backend_->stream_reset();
        bool in_think_style = false;
        bool ink_in_header  = false;
        bool ink_in_think   = false;
        if (stream_stdout && cfg_.enable_thinking && !backend_->is_inkling2()) {
            const char* th_open = backend_->is_glm53() ? "<think>" : "<think>\n";
            if (ansi_style) {
                std::fputs("\033[2m", stdout);
                std::fputs(th_open, stdout);
                in_think_style = true;
            } else {
                std::fputs(th_open, stdout);
            }
            std::fflush(stdout);
        }

        auto emit_stream_tok = [&](uint32_t tok) {
            if (!stream_stdout) return;
            if (backend_->is_inkling2()) {
                if (tok == STRATA_INKLING2_TOKEN_MSG_MODEL) {
                    backend_->stream_reset();
                    ink_in_header = true;
                    return;
                }
                if (tok == STRATA_INKLING2_TOKEN_CONTENT_THINK) {
                    backend_->stream_reset();
                    ink_in_header = false;
                    ink_in_think  = true;
                    if (ansi_style && !in_think_style) {
                        std::fputs("\033[2m<think>\n", stdout);
                        in_think_style = true;
                    } else {
                        std::fputs("<think>\n", stdout);
                    }
                    std::fflush(stdout);
                    return;
                }
                if (tok == STRATA_INKLING2_TOKEN_CONTENT_TEXT) {
                    backend_->stream_reset();
                    ink_in_header = false;
                    ink_in_think  = false;
                    return;
                }
                if (tok == STRATA_INKLING2_TOKEN_END_MSG) {
                    backend_->stream_reset();
                    if (ink_in_think) {
                        if (ansi_style && in_think_style) {
                            std::fputs("\n</think>\033[0m\n\n", stdout);
                            in_think_style = false;
                        } else {
                            std::fputs("\n</think>\n\n", stdout);
                        }
                        std::fflush(stdout);
                    }
                    ink_in_header = false;
                    ink_in_think  = false;
                    return;
                }
                if (ink_in_header) return;
                const std::string piece = backend_->stream_step(tok, /*skip_special=*/true);
                if (!piece.empty()) {
                    std::fwrite(piece.data(), 1U, piece.size(), stdout);
                    std::fflush(stdout);
                }
                return;
            }
            if (tok == backend_->token_think_start()) {
                if (ansi_style && !in_think_style) {
                    std::fputs("\033[2m<think>", stdout);
                    in_think_style = true;
                } else {
                    std::fputs("<think>", stdout);
                }
                std::fflush(stdout);
                return;
            }
            if (tok == backend_->token_think_end()) {
                backend_->stream_reset();
                if (ansi_style && in_think_style) {
                    std::fputs("</think>\033[0m", stdout);
                    in_think_style = false;
                } else {
                    std::fputs("</think>", stdout);
                }
                std::fflush(stdout);
                return;
            }
            const std::string piece = backend_->stream_step(tok, /*skip_special=*/true);
            if (!piece.empty()) {
                std::fwrite(piece.data(), 1U, piece.size(), stdout);
                std::fflush(stdout);
            }
        };

        const bool unlimited_toks = (cfg_.max_tokens == 0U);
        std::vector<uint32_t> gen_ids;
        gen_ids.reserve(std::min<size_t>(unlimited_toks ? 8192U : (cfg_.max_tokens + 8U), 16384U));

        bool hit_eos = false;
        uint32_t decode_emitted = 0U;
        uint32_t decode_steps   = 0U;

        if (backend_->is_eos(first_tok)) {
            hit_eos = true;
        } else {
            gen_ids.push_back(first_tok);
            emit_stream_tok(first_tok);
        }

        uint32_t pending_tok = first_tok;
        const uint32_t target_decode = unlimited_toks
            ? backend_->max_seq_len()
            : ((cfg_.max_tokens > 0U) ? (cfg_.max_tokens - 1U) : 0U);
        const uint32_t eff_draft_k   = backend_->has_mtp() ? cfg_.draft_k : 0U;

        const auto t_dec0 = SteadyClock::now();
        while (!hit_eos && decode_emitted < target_decode) {
            if (backend_->seq_pos() + 1U >= backend_->max_seq_len()) {
                break;
            }
            const uint32_t step_k = (backend_->seq_pos() + 1U + eff_draft_k < backend_->max_seq_len())
                ? eff_draft_k
                : 0U;
            uint32_t emitted_buf[kChatMaxSpecDraftK + 2U] = {};
            uint32_t num_emitted = 0U;
            StepDeltaTelemetry dtel{};

            rc = backend_->decode_step(pending_tok, step_k, turn_samp, emitted_buf, &num_emitted, &dtel);
            if (rc != 0) break;

            ++decode_steps;
            turn_mtp_draft  += dtel.mtp_drafted;
            turn_mtp_acc    += dtel.mtp_accepted;
            turn_ple_hits   += dtel.ple_hits;
            turn_ple_ios    += dtel.ple_ios;
            dec_glm_hot     += dtel.glm_hot_hits;
            dec_glm_exp     += dtel.glm_active_exp;
            dec_glm_nvme_b  += dtel.glm_nvme_bytes;
            dec_glm_nvme_s  += dtel.glm_nvme_sec;
            turn_glm_ane    += dtel.glm_ran_on_ane;
            turn_glm_ane_c  += dtel.glm_ane_calls;
            turn_glm_ane_h  += dtel.glm_ane_hits;

            for (uint32_t i = 0U; i < num_emitted; ++i) {
                const uint32_t tok = emitted_buf[i];
                if (backend_->is_eos(tok)) {
                    hit_eos = true;
                    break;
                }
                gen_ids.push_back(tok);
                ++decode_emitted;
                emit_stream_tok(tok);
                if (decode_emitted >= target_decode) {
                    break;
                }
            }
            if (!hit_eos && num_emitted > 0U) {
                pending_tok = emitted_buf[num_emitted - 1U];
            }
        }
        const auto t_dec1 = SteadyClock::now();

        const bool hit_max_tokens = (!hit_eos && !unlimited_toks && gen_ids.size() >= cfg_.max_tokens);

        if (stream_stdout && in_think_style) {
            std::fputs("\033[0m", stdout);
            in_think_style = false;
        }
        if (stream_stdout) {
            std::fputc('\n', stdout);
            std::fflush(stdout);
        }
        if (rc != 0) {
            std::fprintf(stderr, "[chat] GPU decode failed (rc=%d)\n", rc);
            return rc;
        }

        const double decode_ms = chat_elapsed_ms(t_dec0, t_dec1);
        const double decode_tps = (decode_ms > 0.0 && decode_emitted > 0U)
            ? (static_cast<double>(decode_emitted) / (decode_ms / 1000.0))
            : 0.0;

        /* 6. Strip `<think>...</think>`, restore `turn_start_ckpt`, and commit canonical assistant suffix! */
        std::string think_trimmed, clean_reply;
        bool saw_think_end = false;
        split_generated_think_and_reply(
            *backend_, gen_ids, cfg_.enable_thinking, &think_trimmed, &clean_reply, &saw_think_end);

        if (stream_stdout && cfg_.enable_thinking && !saw_think_end && ansi_style) {
            std::printf("\033[2m[note: reached max_tokens=%u before </think>; increase /max-tokens (or /max-tokens 0) or use /no-think]\033[0m\n",
                        cfg_.max_tokens);
        }

        std::string committed_suffix;
        std::vector<uint32_t> commit_ids;
        if (backend_->is_inkling2()) {
            if (cfg_.keep_think && !think_trimmed.empty()) {
                committed_suffix =
                    "<|message_model|><|content_thinking|>" + think_trimmed + "<|end_message|>" +
                    "<|message_model|><|content_text|>" + clean_reply + "<|end_message|>";
            } else {
                committed_suffix =
                    "<|message_model|><|content_text|>" + clean_reply + "<|end_message|>";
            }
            if (!backend_->encode(committed_suffix, /*parse_special=*/true, &commit_ids) || commit_ids.empty()) {
                std::fprintf(stderr, "[chat] Error: failed to tokenize Inkling-2 committed assistant suffix\n");
                return -1;
            }
        } else if (backend_->is_glm53()) {
            committed_suffix = build_glm53_committed_assistant_suffix(
                think_trimmed, clean_reply, cfg_.enable_thinking, cfg_.keep_think);
            const std::string commit_after_ckpt = cfg_.enable_thinking
                ? (cfg_.keep_think ? (think_trimmed + "</think>" + clean_reply) : ("</think>" + clean_reply))
                : clean_reply;
            if (!commit_after_ckpt.empty()) {
                if (!backend_->encode(commit_after_ckpt, /*parse_special=*/true, &commit_ids)) {
                    std::fprintf(stderr, "[chat] Error: failed to tokenize GLM-5.3 committed assistant suffix\n");
                    return -1;
                }
            }
        } else {
            committed_suffix = build_committed_assistant_suffix(think_trimmed, clean_reply, cfg_.keep_think);
            if (!backend_->encode(committed_suffix, /*parse_special=*/true, &commit_ids) || commit_ids.empty()) {
                std::fprintf(stderr, "[chat] Error: failed to tokenize committed assistant suffix\n");
                return -1;
            }
        }

        StepDeltaTelemetry tel_com{};
        const auto t_com0 = SteadyClock::now();
        rc = backend_->checkpoint_restore(turn_start_ckpt_);
        const double restore_only_ms = chat_elapsed_ms(t_com0, SteadyClock::now());
        last_rewind_ms_ = restore_only_ms;
        if (rc != 0) {
            std::fprintf(stderr, "[chat] Failed to restore turn_start_ckpt (rc=%d)\n", rc);
            return rc;
        }
        if (!commit_ids.empty()) {
            rc = backend_->prefill(
                commit_ids.data(), static_cast<uint32_t>(commit_ids.size()), greedy_dummy, &ignored_tok, &tel_com);
            if (rc != 0) {
                std::fprintf(stderr, "[chat] Failed to commit canonical assistant suffix (rc=%d)\n", rc);
                return rc;
            }
        }
        const auto t_com1 = SteadyClock::now();
        const double commit_ms = chat_elapsed_ms(t_com0, t_com1);
        turn_ple_hits  += tel_com.ple_hits;
        turn_ple_ios   += tel_com.ple_ios;
        turn_glm_ane   += tel_com.glm_ran_on_ane;
        turn_glm_ane_c += tel_com.glm_ane_calls;
        turn_glm_ane_h += tel_com.glm_ane_hits;

        /* Append canonical committed tokens (`hdr_ids + commit_ids`) to transcript mirror */
        committed_ids_.insert(committed_ids_.end(), hdr_ids.begin(), hdr_ids.end());
        committed_ids_.insert(committed_ids_.end(), commit_ids.begin(), commit_ids.end());

        /* Update cumulative session telemetry */
        const uint32_t prompt_delta = static_cast<uint32_t>(hdr_ids.size() + think_ids.size());
        cum_prefill_toks_  += prompt_delta + static_cast<uint32_t>(commit_ids.size());
        cum_decode_toks_   += decode_emitted;
        cum_decode_steps_  += decode_steps;
        cum_emitted_toks_  += static_cast<uint32_t>(gen_ids.size());
        cum_decode_ms_     += decode_ms;
        cum_mtp_drafted_   += turn_mtp_draft;
        cum_mtp_accepted_  += turn_mtp_acc;
        cum_ple_hits_      += turn_ple_hits;
        cum_ple_ios_       += turn_ple_ios;
        cum_glm_hot_hits_  += tel_hdr.glm_hot_hits + dec_glm_hot + tel_com.glm_hot_hits;
        cum_glm_active_exp_+= tel_hdr.glm_active_exp + dec_glm_exp + tel_com.glm_active_exp;
        cum_glm_nvme_bytes_+= tel_hdr.glm_nvme_bytes + dec_glm_nvme_b + tel_com.glm_nvme_bytes;
        cum_glm_nvme_sec_  += tel_hdr.glm_nvme_sec + dec_glm_nvme_s + tel_com.glm_nvme_sec;
        cum_glm_ran_ane_   += turn_glm_ane;
        cum_glm_ane_calls_ += turn_glm_ane_c;
        cum_glm_ane_hits_  += turn_glm_ane_h;

        ChatTurnRecord rec{};
        rec.user_text         = user_trimmed;
        rec.think_text        = think_trimmed;
        rec.clean_reply       = clean_reply;
        rec.committed_suffix  = committed_suffix;
        rec.generated_ids     = std::move(gen_ids);
        rec.seq_pos_before    = pos_before;
        rec.seq_pos_after     = backend_->seq_pos();
        rec.prompt_delta_toks = prompt_delta;
        rec.committed_toks    = static_cast<uint32_t>(commit_ids.size());
        rec.prefill_ms        = prefill_ms;
        rec.decode_ms         = decode_ms;
        rec.decode_tok_s      = decode_tps;
        rec.commit_ms         = commit_ms;
        rec.hit_max_tokens    = hit_max_tokens;
        history_.push_back(rec);

        if (stream_stdout) {
            const ChatMemSnapshot mem_now = query_chat_mem_snapshot(backend_->glm53_shared_mtl_gib());
            const long steady_maj = std::max<long>(0L, mem_now.majflt - baseline_mem_.majflt);
            const double mtp_pct = (turn_mtp_draft > 0ULL)
                ? (100.0 * static_cast<double>(turn_mtp_acc) / static_cast<double>(turn_mtp_draft))
                : 0.0;
            const uint64_t ple_tot = turn_ple_hits + turn_ple_ios;
            const double ple_pct = (ple_tot > 0ULL)
                ? (100.0 * static_cast<double>(turn_ple_hits) / static_cast<double>(ple_tot))
                : 0.0;

            char mtp_buf[48];
            if (backend_->has_mtp() && eff_draft_k > 0U) {
                std::snprintf(mtp_buf, sizeof(mtp_buf), "MTP %.1f%% (K=%u)", mtp_pct, eff_draft_k);
            } else if (backend_->has_mtp()) {
                std::snprintf(mtp_buf, sizeof(mtp_buf), "MTP off (K=0)");
            } else {
                std::snprintf(mtp_buf, sizeof(mtp_buf), "MTP n/a");
            }

            char tier_buf[80];
            if (backend_->is_glm53()) {
                const double hot_pct = (dec_glm_exp > 0ULL)
                    ? (100.0 * static_cast<double>(dec_glm_hot) / static_cast<double>(dec_glm_exp))
                    : 0.0;
                const double nvme_gbps = (dec_glm_nvme_s > 1e-6 && dec_glm_nvme_b > 0ULL)
                    ? ((static_cast<double>(dec_glm_nvme_b) / 1e9) / dec_glm_nvme_s)
                    : ((cum_glm_nvme_sec_ > 1e-6) ? ((static_cast<double>(cum_glm_nvme_bytes_) / 1e9) / cum_glm_nvme_sec_) : 0.0);
                std::snprintf(tier_buf, sizeof(tier_buf), "Hot %.1f%% | NVMe %.1f GB/s | ANE=%d",
                              hot_pct, nvme_gbps, (backend_->glm53_ane_verified() && turn_glm_ane > 0ULL) ? 1 : 0);
            } else if (backend_->has_ple()) {
                std::snprintf(tier_buf, sizeof(tier_buf), "PLE %.1f%%", ple_pct);
            } else {
                std::snprintf(tier_buf, sizeof(tier_buf), "PLE n/a");
            }

            if (ansi_style) std::fputs("\033[2m", stdout);
            std::printf("[prefill %.1f ms (+%u toks) | %.2f tok/s (%zu toks) | %s | pos %u/%u | %s | ckpt %.2f ms | MajFlt %ld%s]",
                        prefill_ms,
                        prompt_delta,
                        decode_tps,
                        rec.generated_ids.size(),
                        mtp_buf,
                        rec.seq_pos_after,
                        backend_->max_seq_len(),
                        tier_buf,
                        restore_only_ms,
                        steady_maj,
                        hit_max_tokens ? " | hit max-tokens" : "");
            if (ansi_style) std::fputs("\033[0m", stdout);
            std::fputc('\n', stdout);
            std::fflush(stdout);
        }

        if (out_record) *out_record = std::move(rec);
        return 0;
    }

    void print_stats() const {
        const ChatMemSnapshot mem = query_chat_mem_snapshot(backend_->glm53_shared_mtl_gib());
        const long steady_maj     = std::max<long>(0L, mem.majflt - baseline_mem_.majflt);
        const double swap_delta   = std::max(0.0, mem.swap_used_gib - baseline_mem_.swap_used_gib);
        const uint32_t cur_pos    = backend_->seq_pos();
        constexpr double kMiB     = 1024.0 * 1024.0;
        constexpr double kGiB     = 1024.0 * 1024.0 * 1024.0;

        const double kv_mib    = (static_cast<double>(cur_pos) * static_cast<double>(backend_->kv_bytes_per_tok())) / kMiB;
        const double dn_mib    = backend_->deltanet_mib();
        const double ple_mib   = backend_->ple_state_mib();
        const double state_mib = kv_mib + dn_mib + ple_mib;
        const double one_mib   = static_cast<double>(single_checkpoint_bytes()) / kMiB;
        const double ckpt_mib  = static_cast<double>(total_checkpoint_bytes()) / kMiB;

        const double avg_tps = (cum_decode_ms_ > 0.0 && cum_decode_toks_ > 0U)
            ? (static_cast<double>(cum_decode_toks_) / (cum_decode_ms_ / 1000.0))
            : 0.0;
        const double mtp_pct = (cum_mtp_drafted_ > 0ULL)
            ? (100.0 * static_cast<double>(cum_mtp_accepted_) / static_cast<double>(cum_mtp_drafted_))
            : 0.0;
        const uint64_t ple_tot = cum_ple_hits_ + cum_ple_ios_;
        const double ple_pct = (ple_tot > 0ULL)
            ? (100.0 * static_cast<double>(cum_ple_hits_) / static_cast<double>(ple_tot))
            : 0.0;

        std::printf("----------------------------------------------------------------------------------------------------\n");
        std::printf("  STRATA-M5 Stateful Multi-Turn Session Telemetry (%s)\n", backend_->device_name());
        std::printf("----------------------------------------------------------------------------------------------------\n");
        std::printf("  Model             : %s\n", backend_->model_name());
        std::printf("  Context & Turns   : %zu completed turns | seq_pos = %u / %u | undo_ring = %zu/%zu ready\n",
                    history_.size(), cur_pos, backend_->max_seq_len(), undo_valid_count_, kUndoRingCap);
        std::printf("  Active GPU State  : %.2f MiB (%s %.2f MiB O(1) + KV[%s] %.2f MiB + PLE %.2f MiB)\n",
                    state_mib,
                    backend_->is_inkling2() ? "SWA+SConv" : (backend_->is_glm53() ? "KDA" : "DeltaNet"),
                    dn_mib, backend_->kv_cache_name(), kv_mib, ple_mib);
        std::printf("  O(1) Checkpoints  : %.2f MiB/ckpt (%.2f MiB x%zu pool) | Last save %.3f ms | Last restore %.3f ms\n",
                    one_mib, ckpt_mib, 1U + kUndoRingCap, last_save_ms_, last_rewind_ms_);
        std::printf("  Throughput        : %.2f avg decode tok/s (%u decode toks in %u steps, %u prefill toks)\n",
                    avg_tps, cum_decode_toks_, cum_decode_steps_, cum_prefill_toks_);
        if (backend_->has_mtp()) {
            std::printf("  Native MTP Spec   : K=%u | Accepted %llu / %llu (%.1f%%)\n",
                        cfg_.draft_k,
                        static_cast<unsigned long long>(cum_mtp_accepted_),
                        static_cast<unsigned long long>(cum_mtp_drafted_),
                        mtp_pct);
        }
        if (backend_->is_glm53()) {
            const double hot_pct = (cum_glm_active_exp_ > 0ULL)
                ? (100.0 * static_cast<double>(cum_glm_hot_hits_) / static_cast<double>(cum_glm_active_exp_))
                : 0.0;
            const double nvme_gbps = (cum_glm_nvme_sec_ > 1e-6 && cum_glm_nvme_bytes_ > 0ULL)
                ? ((static_cast<double>(cum_glm_nvme_bytes_) / 1e9) / cum_glm_nvme_sec_)
                : 0.0;
            std::printf("  Regime II Tiering : %.2f GiB Hot Cache (%u/%u slots, %.1f%% hit) + %.2f GiB NVMe (%.2f GiB @ %.2f GB/s)\n",
                        backend_->glm53_hot_cache_gib(),
                        backend_->glm53_hot_resident(),
                        backend_->glm53_hot_capacity(),
                        hot_pct,
                        backend_->glm53_experts_nvme_gib(),
                        static_cast<double>(cum_glm_nvme_bytes_) / kGiB,
                        nvme_gbps);
            std::printf("  16-Core ANE Probe : ran_on_ane=%d (%llu verified) | Calls=%llu | Prefetch hits=%llu | Mode=%s\n",
                        (backend_->glm53_ane_verified() && cum_glm_ran_ane_ > 0ULL) ? 1 : 0,
                        static_cast<unsigned long long>(cum_glm_ran_ane_),
                        static_cast<unsigned long long>(cum_glm_ane_calls_),
                        static_cast<unsigned long long>(cum_glm_ane_hits_),
                        (cfg_.glm53_mode == STRATA_GLM53_MODE_ADAPTIVE) ? "ADAPTIVE" : "LOSSLESS");
        } else if (backend_->has_ple()) {
            std::printf("  NVMe PLE Store    : %.2f GiB F_NOCACHE (%.0f MiB mlock LFU) | Hits %llu, IOs %llu (%.1f%% hit)\n",
                        backend_->ple_nvme_gib(), backend_->ple_lfu_mib(),
                        static_cast<unsigned long long>(cum_ple_hits_),
                        static_cast<unsigned long long>(cum_ple_ios_),
                        ple_pct);
        }
        std::printf("  UMA & Fault Gates : Total UMA %.2f GiB (GPU Wired %.2f GiB, Host RSS %.2f GiB, Swap +%.2f GiB) | Steady MajFlt=%ld\n",
                    mem.total_uma_gib, mem.gpu_wired_gib, mem.host_rss_gib, swap_delta, steady_maj);
        std::printf("----------------------------------------------------------------------------------------------------\n");
        std::fflush(stdout);
    }

    ChatCliConfig& mutable_cfg() { return cfg_; }
    const ChatCliConfig& cfg() const { return cfg_; }

    /* ---- Step 4 Tier 1 hooks (used by CompactionController) ---- */
    size_t history_floor() const { return history_floor_; }
    size_t undo_valid_count() const { return undo_valid_count_; }
    void invalidate_undo_ring() { undo_valid_count_ = 0U; }
    void sync_turn_start_ckpt() {
        if (turn_start_ckpt_) (void)backend_->checkpoint_save(turn_start_ckpt_);
    }
    void rebaseline_mem() { baseline_mem_ = query_chat_mem_snapshot(backend_->glm53_shared_mtl_gib()); }

    Bookkeeping snapshot_bookkeeping() const {
        Bookkeeping b;
        b.committed_ids          = committed_ids_;
        b.history                = history_;
        b.history_floor          = history_floor_;
        b.effective_system_block = effective_system_block_;
        b.undo_valid_count       = undo_valid_count_;
        return b;
    }
    void restore_bookkeeping(Bookkeeping&& b) {
        committed_ids_          = std::move(b.committed_ids);
        history_                = std::move(b.history);
        history_floor_          = b.history_floor;
        effective_system_block_ = std::move(b.effective_system_block);
        undo_valid_count_       = b.undo_valid_count;
    }
    std::vector<uint32_t> take_committed_ids() {
        std::vector<uint32_t> out = std::move(committed_ids_);
        committed_ids_.clear();
        return out;
    }

    /*
     * Installs a promoted shadow transcript: `ids = prefix(system + ledger[0..split)) ++ Turns[split..]`.
     * Turns below `split` become the immutable history floor (folded into the ledger; `/undo` stops there);
     * turns >= split keep their records with seq positions rebased onto the shadow coordinates.
     * Returns false (and leaves the session untouched) if the rebased positions do not match `ids`.
     */
    bool install_compacted_transcript(
        std::vector<uint32_t>&& ids,
        const std::string& sys_blk,
        size_t split,
        uint32_t prefix_len
    ) {
        if (split > history_.size()) return false;
        uint64_t pos = prefix_len;
        for (size_t i = split; i < history_.size(); ++i) {
            pos += history_[i].seq_pos_after - history_[i].seq_pos_before;
        }
        if (pos != ids.size()) return false;
        uint32_t p = prefix_len;
        for (size_t i = split; i < history_.size(); ++i) {
            const uint32_t len = history_[i].seq_pos_after - history_[i].seq_pos_before;
            history_[i].seq_pos_before = p;
            p += len;
            history_[i].seq_pos_after = p;
        }
        committed_ids_          = std::move(ids);
        effective_system_block_ = sys_blk;
        history_floor_          = split;
        undo_valid_count_       = 0U;
        return true;
    }

private:
    IChatBackend*               backend_                = nullptr;
    ChatCliConfig               cfg_{};
    ChatMemSnapshot             baseline_mem_{};
    void*                       turn_start_ckpt_        = nullptr;
    void*                       undo_ring_[kUndoRingCap]{};
    size_t                      undo_valid_count_       = 0U;
    size_t                      history_floor_          = 0U;
    std::string                 effective_system_block_;
    std::vector<ChatTurnRecord>  history_;
    std::vector<uint32_t>        committed_ids_;
    std::vector<StagedChatImage> staged_images_;

    uint32_t                    cum_prefill_toks_       = 0U;
    uint32_t                    cum_decode_toks_        = 0U;
    uint32_t                    cum_decode_steps_       = 0U;
    uint32_t                    cum_emitted_toks_       = 0U;
    double                      cum_decode_ms_          = 0.0;
    uint64_t                    cum_mtp_drafted_        = 0ULL;
    uint64_t                    cum_mtp_accepted_       = 0ULL;
    uint64_t                    cum_ple_hits_           = 0ULL;
    uint64_t                    cum_ple_ios_            = 0ULL;
    uint64_t                    cum_glm_hot_hits_       = 0ULL;
    uint64_t                    cum_glm_active_exp_     = 0ULL;
    uint64_t                    cum_glm_nvme_bytes_     = 0ULL;
    double                      cum_glm_nvme_sec_       = 0.0;
    uint64_t                    cum_glm_ran_ane_        = 0ULL;
    uint64_t                    cum_glm_ane_calls_      = 0ULL;
    uint64_t                    cum_glm_ane_hits_       = 0ULL;
    double                      last_save_ms_           = 0.0;
    double                      last_rewind_ms_         = 0.0;
};

/* ============================================================================
 * 4b. Step 4 Compaction Controller (Tier 1 Shadow RCU + Tier 2 Paged Live Slot)
 *
 * Slot roles: `live` (bound; serves every user turn), `shadow` (warmed in idle-gap prefill slices), and
 * `rollback` (pre-swap live, kept for `/undo-compact`). Slot index 0 is the engine's built-in default slot.
 *   off    : live = default slot, no background work (zero overhead, identical to pre-Step-4 behavior).
 *   paged  : live = paged slot(pool); no shadow.
 *   shadow : live starts on the default slot; shadow/rollback are linear `slot_create(eng, 0)` slots.
 *   hybrid : live, shadow and rollback are all paged slots(pool).
 * `/undo-compact` choice: the rollback slot is valid until the next committed turn (or /undo, /reset,
 * /system, mode change); post-swap turns are NOT mirrored into it, so it costs zero background GPU time.
 * After `/undo-compact` automatic promotion is inhibited until `/compact` or until N_live >= C_max.
 * ============================================================================ */

class CompactionController {
public:
    CompactionController(IChatBackend* backend, ChatReplSession* session)
        : backend_(backend), session_(session) {
        slots_.push_back(SlotRec{nullptr, true, false});
    }
    ~CompactionController() { release_all(); }
    CompactionController(const CompactionController&) = delete;
    CompactionController& operator=(const CompactionController&) = delete;

    void set_echo_events(bool on) { echo_events_ = on; }
    void set_quiet(bool on) { quiet_ = on; }
    void set_slice_tokens(uint32_t n) { slice_tokens_ = std::clamp<uint32_t>(n, 32U, 64U); }

    StrataCompactionMode mode() const { return mode_; }
    bool active() const { return mode_ != STRATA_COMPACT_OFF; }
    bool tier1() const { return mode_ == STRATA_COMPACT_SHADOW || mode_ == STRATA_COMPACT_HYBRID; }
    uint32_t window() const { return session_->cfg().compact_window; }
    uint32_t w_swap() const {
        return static_cast<uint32_t>(static_cast<double>(session_->cfg().compact_watermark) *
                                     static_cast<double>(window()));
    }
    bool shadow_active() const { return shadow_ >= 0; }
    bool synced() const { return shadow_ >= 0 && plan_.reset_done && plan_.done == plan_.target.size(); }
    bool rollback_available() const { return rollback_ >= 0 && rollback_book_ != nullptr; }
    uint32_t swaps() const { return swaps_; }
    uint32_t undo_compacts() const { return undo_compacts_; }
    uint32_t launches() const { return launches_; }
    double last_switchover_ms() const { return last_switchover_ms_; }
    double last_undo_ms() const { return last_undo_ms_; }
    double max_user_stall_ms() const { return max_user_stall_ms_; }
    uint32_t last_pre_swap_pos() const { return last_pre_swap_pos_; }
    uint32_t last_post_swap_pos() const { return last_post_swap_pos_; }
    double last_k_rem() const { return last_k_rem_; }
    const CompactVelocityPredictor& predictor() const { return predictor_; }
    const std::vector<LedgerEntry>& ledger() const { return ledger_; }
    size_t shadow_done() const { return shadow_ >= 0 ? plan_.done : 0U; }
    size_t shadow_target() const { return shadow_ >= 0 ? plan_.target.size() : 0U; }
    size_t shadow_split() const { return shadow_ >= 0 ? plan_.split : 0U; }
    const std::string& last_note() const { return last_note_; }
    void* live_slot_handle() const { return slots_[static_cast<size_t>(live_)].handle; }
    bool live_is_paged() const { return slots_[static_cast<size_t>(live_)].paged; }

    void note_user_stall(double ms) { max_user_stall_ms_ = std::max(max_user_stall_ms_, ms); }

    /* Switches tier; migrates the live context (one explicit cold re-prefill) when the live slot kind changes. */
    bool set_mode(StrataCompactionMode m, std::string* msg) {
        if (!backend_->compaction_supported()) {
            if (m == STRATA_COMPACT_OFF) {
                mode_ = m;
                if (msg) *msg = "[compact] mode=off";
                return true;
            }
            if (msg) *msg = "[compact] Step 4 compaction requires the chat-flash (Qwen3.8-Flash-Next) engine; mode stays off";
            return false;
        }
        discard_shadow(nullptr);
        drop_rollback();
        const bool want_paged = (m == STRATA_COMPACT_PAGED || m == STRATA_COMPACT_HYBRID);
        std::string mig;
        if (slots_[static_cast<size_t>(live_)].paged != want_paged) {
            const int nl = acquire_free_slot(want_paged);
            if (nl < 0) {
                if (msg) {
                    *msg = std::string("[compact] Failed to create ") + (want_paged ? "paged" : "linear") +
                           " context slot (pool=" + std::to_string(session_->cfg().compact_pool) + ")";
                }
                return false;
            }
            const auto t0 = SteadyClock::now();
            bind_idx(nl);
            backend_->compact_reset_bound();
            const std::vector<uint32_t>& ids = session_->committed_transcript_ids();
            int rc = 0;
            if (!ids.empty()) {
                ChatSamplingSpec g{};
                g.greedy = true;
                uint32_t ign = 0U;
                rc = backend_->prefill(ids.data(), static_cast<uint32_t>(ids.size()), g, &ign, nullptr);
            }
            if (rc != 0) {
                bind_idx(live_);
                if (msg) *msg = "[compact] Live-context migration prefill failed (rc=" + std::to_string(rc) + ")";
                return false;
            }
            session_->invalidate_undo_ring();
            live_ = nl;
            if (!ids.empty()) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), " | migrated live context: %zu toks re-prefilled in %.1f ms",
                              ids.size(), chat_elapsed_ms(t0, SteadyClock::now()));
                mig = buf;
            }
        }
        mode_ = m;
        release_unused_slots();
        predictor_.reset();
        last_n_       = backend_->seq_pos();
        inhibit_auto_ = false;
        if (tier1()) catch_up_ledger();
        if (msg) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "[compact] mode=%s | live slot=%s | window=%u W_swap=%u | pool=%u%s",
                          chat_compact_mode_name(mode_),
                          live_is_paged() ? "paged" : (live_ == 0 ? "default(linear)" : "linear"),
                          window(), w_swap(), session_->cfg().compact_pool, mig.c_str());
            *msg = buf;
        }
        return true;
    }

    /* Called right before a user turn starts on the live slot. */
    void pre_turn() {
        if (!tier1()) return;
        flush_events();
        const uint32_t n = backend_->seq_pos();
        if (n >= window()) inhibit_auto_ = false; /* hard C_max cap overrides /undo-compact inhibition */
        if (inhibit_auto_) return;
        const auto t0 = SteadyClock::now();
        bool sync_work = false;
        if (shadow_ < 0 && n >= window()) {
            std::string why;
            if (!launch(false, &why)) last_note_ = why;
        }
        if (shadow_ >= 0 && n >= w_swap()) {
            if (!synced()) {
                if (n < window()) {
                    flush_events();
                    return; /* keep serving on live; warmup continues in the next idle gap */
                }
                sync_work = finish_warmup(); /* N_live hit C_max: synchronous fallback */
            }
            if (synced()) promote("pre-turn");
        }
        if (sync_work) note_user_stall(chat_elapsed_ms(t0, SteadyClock::now()));
        flush_events();
    }

    /* Called after each successfully committed live turn. */
    void on_turn_committed() {
        if (!tier1()) return;
        const std::vector<ChatTurnRecord>& h = session_->history();
        if (h.empty()) return;
        drop_rollback(); /* /undo-compact grace window closes once a new turn is committed */
        catch_up_ledger();
        const ChatTurnRecord& rec = h.back();
        const std::vector<uint32_t>& ids = session_->committed_transcript_ids();
        if (shadow_ >= 0 && rec.seq_pos_before <= rec.seq_pos_after && rec.seq_pos_after <= ids.size()) {
            plan_.target.insert(plan_.target.end(), ids.begin() + rec.seq_pos_before, ids.begin() + rec.seq_pos_after);
            mirrored_toks_ += rec.seq_pos_after - rec.seq_pos_before;
        }
        const uint32_t n = backend_->seq_pos();
        const double dn = (n >= last_n_) ? static_cast<double>(n - last_n_) : static_cast<double>(n);
        predictor_.observe(dn);
        last_n_     = n;
        last_k_rem_ = predictor_.k_rem(static_cast<double>(w_swap()), static_cast<double>(n));
        if (shadow_ < 0 && !inhibit_auto_ &&
            (last_k_rem_ <= kCompactLeadTurns || static_cast<double>(n) >= kCompactLaunchFrac * window())) {
            if (!has_pending_semantic_before_next_split()) {
                std::string why;
                if (!launch(false, &why)) last_note_ = why;
            }
        }
    }

    void on_undo() {
        if (!active()) return;
        drop_rollback();
        discard_distill();
        if (ledger_.size() > session_->history().size()) ledger_.resize(session_->history().size());
        if (shadow_ >= 0) discard_shadow("live /undo removed a turn the shadow already contains");
        last_n_ = backend_->seq_pos();
    }

    void on_session_reset() {
        if (!active()) return;
        drop_rollback();
        discard_distill();
        discard_shadow(nullptr);
        ledger_.clear();
        predictor_.reset();
        last_n_       = backend_->seq_pos();
        inhibit_auto_ = false;
    }

    uint32_t semantic_distills() const { return semantic_distills_; }

    bool has_idle_work() const {
        if (!tier1()) return false;
        if (shadow_ >= 0) {
            if (!plan_.reset_done || plan_.done < plan_.target.size()) return true;
            return !inhibit_auto_ && backend_->seq_pos() >= w_swap();
        }
        if (djob_.active || next_undistilled_idx() >= 0) return true;
        if (!inhibit_auto_) {
            const uint32_t n = backend_->seq_pos();
            if ((last_k_rem_ <= kCompactLeadTurns || static_cast<double>(n) >= kCompactLaunchFrac * window()) &&
                can_select_split()) {
                return true;
            }
        }
        return false;
    }

    /* Runs AT MOST ONE preemptible quantum (semantic distill slice/decode, shadow reset/slice, or O(1) promotion). */
    bool idle_quantum(double* quantum_ms_out = nullptr) {
        const auto t0 = SteadyClock::now();
        bool did = false;
        if (tier1()) {
            if (shadow_ >= 0) {
                if (!synced()) {
                    did = warm_step();
                } else if (!inhibit_auto_ && backend_->seq_pos() >= w_swap()) {
                    did = promote("idle");
                }
            } else if (djob_.active || next_undistilled_idx() >= 0) {
                did = distill_step();
            } else if (!inhibit_auto_) {
                const uint32_t n = backend_->seq_pos();
                if (last_k_rem_ <= kCompactLeadTurns || static_cast<double>(n) >= kCompactLaunchFrac * window()) {
                    std::string why;
                    did = launch(false, &why);
                    if (!did) last_note_ = why;
                }
            }
        }
        if (quantum_ms_out) *quantum_ms_out = chat_elapsed_ms(t0, SteadyClock::now());
        return did;
    }

    /* Simulated idle gap (piped / --script stdin and the verifiers): drains all pending idle quanta. */
    uint32_t run_idle_gap(uint32_t max_quanta = 1000000U) {
        uint32_t q = 0U;
        while (q < max_quanta && has_idle_work()) {
            if (!idle_quantum(nullptr)) break;
            ++q;
        }
        flush_events();
        return q;
    }

    /* `/compact`: force promotion now (launches + finishes warmup synchronously if needed). */
    bool force_compact(std::string* msg) {
        if (!tier1()) {
            if (msg) *msg = "[compact] /compact requires --compact-mode shadow or hybrid (current: " +
                            std::string(chat_compact_mode_name(mode_)) + ")";
            return false;
        }
        inhibit_auto_ = false;
        const auto t0 = SteadyClock::now();
        if (shadow_ < 0) {
            finish_all_distills();
            std::string why;
            if (!launch(true, &why)) {
                if (msg) *msg = "[compact] No shadow promotion possible: " + why;
                return false;
            }
        }
        const size_t pending = plan_.target.size() - plan_.done;
        (void)finish_warmup();
        const double warm_ms = chat_elapsed_ms(t0, SteadyClock::now());
        if (!synced()) {
            if (msg) *msg = "[compact] Shadow warmup failed: " + last_note_;
            return false;
        }
        if (!promote("/compact")) {
            if (msg) *msg = "[compact] Promotion failed: " + last_note_;
            return false;
        }
        if (msg) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "[compact] Promoted shadow: seq_pos %u -> %u | switchover %.4f ms | sync-finish %zu toks in %.1f ms",
                          last_pre_swap_pos_, last_post_swap_pos_, last_switchover_ms_, pending, warm_ms);
            *msg = buf;
        }
        events_.clear();
        return true;
    }

    /* `/undo-compact`: O(1) rebind of the rollback slot + restore of the pre-swap session bookkeeping. */
    bool undo_compact(double* ms_out, uint32_t* pos_before_out, uint32_t* pos_after_out, std::string* msg) {
        if (!rollback_available()) {
            if (msg) *msg = "[compact] Nothing to undo (rollback slot is valid only until the next turn after a swap)";
            return false;
        }
        const uint32_t pos_before = backend_->seq_pos();
        const auto t0 = SteadyClock::now();
        bind_idx(rollback_);
        const int promoted = live_;
        live_     = rollback_;
        rollback_ = -1;
        const double ms = chat_elapsed_ms(t0, SteadyClock::now());

        /* The promoted slot returns to the shadow role, still DUAL_SYNCED with the restored transcript */
        plan_            = std::move(rollback_plan_);
        plan_.target     = session_->take_committed_ids();
        plan_.done       = plan_.target.size();
        plan_.reset_done = true;
        shadow_          = promoted;
        session_->restore_bookkeeping(std::move(*rollback_book_));
        rollback_book_.reset();
        inhibit_auto_ = true;
        ++undo_compacts_;
        last_undo_ms_ = ms;
        last_n_       = backend_->seq_pos();
        if (ms_out) *ms_out = ms;
        if (pos_before_out) *pos_before_out = pos_before;
        if (pos_after_out) *pos_after_out = backend_->seq_pos();
        if (msg) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "[compact] /undo-compact: rebound pre-swap slot in %.4f ms | seq_pos %u -> %u "
                          "(auto-promotion paused until /compact or N >= C_max)",
                          ms, pos_before, backend_->seq_pos());
            *msg = buf;
        }
        return true;
    }

    void fill_stats(StrataQwenFlashCompactionStats* out) const {
        backend_->compact_get_stats(live_slot_handle(), out);
        out->mode = static_cast<uint32_t>(mode_);
        if (out->logical_seq_pos == 0U) out->logical_seq_pos = backend_->seq_pos();
        out->episodes_distilled  = static_cast<uint32_t>(ledger_.size());
        out->shadow_seq_pos      = static_cast<uint32_t>(shadow_done());
        out->shadow_dual_synced  = synced();
        out->rcu_swaps_completed = swaps_;
        out->last_switchover_ms  = last_switchover_ms_;
        out->max_user_stall_ms   = max_user_stall_ms_;
    }

    void print_stats(long steady_majflt) const {
        StrataQwenFlashCompactionStats st{};
        fill_stats(&st);
        constexpr double kMB = 1024.0 * 1024.0;
        const double kr = last_k_rem_;
        char kbuf[32];
        if (std::isfinite(kr)) std::snprintf(kbuf, sizeof(kbuf), "%.2f", kr);
        else std::snprintf(kbuf, sizeof(kbuf), "inf");
        std::printf("----------------------------------------------------------------------------------------------------\n");
        std::printf("  STRATA-M5 Step 4 Realtime Compaction | mode=%s | live slot=%s\n",
                    chat_compact_mode_name(mode_),
                    live_is_paged() ? "paged" : (live_ == 0 ? "default(linear)" : "linear"));
        std::printf("----------------------------------------------------------------------------------------------------\n");
        std::printf("  Horizon Predictor : logical_pos=%u | window C_max=%u | W_swap=%u (%.2f) | v=%.1f v+=%.1f toks/turn | K_rem=%s\n",
                    st.logical_seq_pos, window(), w_swap(), static_cast<double>(session_->cfg().compact_watermark),
                    predictor_.v, predictor_.v_plus(), kbuf);
        std::printf("  Tier 2 UMA Pool   : blocks used %u / capacity %u | logical blocks %u\n",
                    st.uma_kv_blocks_used, st.uma_kv_blocks_capacity, st.logical_blocks);
        std::printf("  Tier 2 NVMe Swap  : now-paged %u | paged-out %llu (%.2f MB, %.2f ms) | recalled %llu (%.2f MB, %.2f ms) | sync misses %llu/%llu\n",
                    st.nvme_blocks_now_paged,
                    static_cast<unsigned long long>(st.nvme_blocks_paged_out),
                    static_cast<double>(st.nvme_bytes_written) / kMB, st.nvme_write_ms,
                    static_cast<unsigned long long>(st.nvme_blocks_recalled),
                    static_cast<double>(st.nvme_bytes_read) / kMB, st.nvme_read_ms,
                    static_cast<unsigned long long>(st.sync_recall_misses),
                    static_cast<unsigned long long>(st.paged_sync_points));
        std::printf("  16-Core ANE Probe : calls %llu | on_ane %llu | ran_on_ane=%u | prefetch hits %llu / issued %llu | %.1f us avg\n",
                    static_cast<unsigned long long>(st.ane_probe_calls),
                    static_cast<unsigned long long>(st.ane_probe_on_ane),
                    st.ran_on_ane,
                    static_cast<unsigned long long>(st.ane_prefetch_hits),
                    static_cast<unsigned long long>(st.ane_prefetch_issued),
                    st.ane_probe_us_avg);
        std::printf("  Tier 1 Ledger     : %u episodes distilled (immutable) | history floor %zu turns | launches %u\n",
                    st.episodes_distilled, session_->history_floor(), launches_);
        std::printf("  Tier 1 Shadow     : %s | pos %zu / target %zu (split turn %zu) | synced=%d | mirrored %llu toks\n",
                    shadow_ >= 0 ? "WARMING" : "idle", shadow_done(), shadow_target(),
                    shadow_ >= 0 ? plan_.split + 1U : 0U, st.shadow_dual_synced ? 1 : 0,
                    static_cast<unsigned long long>(mirrored_toks_));
        std::printf("  Tier 1 RCU Swaps  : %u completed | last switchover %.4f ms | undo-compacts %u (last %.4f ms) | rollback %s\n",
                    st.rcu_swaps_completed, st.last_switchover_ms, undo_compacts_, last_undo_ms_,
                    rollback_available() ? "AVAILABLE" : "none");
        std::printf("  Realtime Gates    : max user stall %.3f ms | Steady MajFlt=%ld%s%s\n",
                    st.max_user_stall_ms, steady_majflt,
                    last_note_.empty() ? "" : " | note: ", last_note_.c_str());
        std::printf("----------------------------------------------------------------------------------------------------\n");
        std::fflush(stdout);
    }

    void flush_events() {
        for (const std::string& e : events_) {
            std::printf("%s\n", e.c_str());
        }
        if (!events_.empty()) std::fflush(stdout);
        events_.clear();
    }

    struct PersistentSnapshot {
        StrataCompactionMode       mode              = STRATA_COMPACT_OFF;
        uint32_t                   swaps             = 0U;
        uint32_t                   launches          = 0U;
        uint32_t                   semantic_distills = 0U;
        uint64_t                   mirrored_toks     = 0ULL;
        CompactVelocityPredictor   predictor{};
        std::vector<LedgerEntry>   ledger;
    };

    PersistentSnapshot export_persistent_state() {
        if (tier1()) catch_up_ledger();
        PersistentSnapshot p;
        p.mode              = mode_;
        p.swaps             = swaps_;
        p.launches          = launches_;
        p.semantic_distills = semantic_distills_;
        p.mirrored_toks     = mirrored_toks_;
        p.predictor         = predictor_;
        p.ledger            = ledger_;
        return p;
    }

    /* Prepares a clean bound `live_` slot of `m` (without migrating old tokens) before `load_state_fd()`. */
    bool prepare_slot_for_load(StrataCompactionMode m, uint32_t compact_pool, std::string* err) {
        if (m != STRATA_COMPACT_OFF && !backend_->compaction_supported()) {
            if (err) *err = "compaction mode in session snapshot is not supported by this backend";
            return false;
        }
        drop_rollback();
        discard_distill();
        discard_shadow(nullptr);
        const bool want_paged = (m == STRATA_COMPACT_PAGED || m == STRATA_COMPACT_HYBRID);
        if (!want_paged) {
            bind_idx(0);
            live_ = 0;
        } else {
            int nl = acquire_free_slot(true, compact_pool);
            if (nl < 0) {
                if (err) *err = "failed to allocate paged context slot for session load";
                return false;
            }
            bind_idx(nl);
            live_ = nl;
        }
        mode_ = m;
        release_unused_slots();
        events_.clear();
        return true;
    }

    void import_persistent_state(PersistentSnapshot&& p) {
        swaps_             = p.swaps;
        launches_          = p.launches;
        semantic_distills_ = p.semantic_distills;
        mirrored_toks_     = p.mirrored_toks;
        predictor_         = p.predictor;
        ledger_            = std::move(p.ledger);
        last_n_            = backend_->seq_pos();
        last_k_rem_        = predictor_.k_rem(static_cast<double>(w_swap()), static_cast<double>(last_n_));
        inhibit_auto_      = false;
    }

private:
    struct SlotRec {
        void* handle     = nullptr;
        bool  is_default = false;
        bool  paged      = false;
    };
    struct ShadowPlan {
        size_t                split      = 0U;
        std::string           sys_block;
        uint32_t              prefix_len = 0U;
        std::vector<uint32_t> target;
        size_t                done       = 0U;
        bool                  reset_done = false;
    };
    struct DistillJob {
        bool                  active       = false;
        size_t                turn_idx     = 0U;
        int                   slot_idx     = -1;
        bool                  reset_done   = false;
        std::vector<uint32_t> prompt_ids;
        size_t                prefill_done = 0U;
        uint32_t              pending_tok  = 0U;
        std::vector<uint32_t> out_ids;
    };

    void event(const std::string& e) {
        if (quiet_) return;
        if (echo_events_) {
            std::printf("%s\n", e.c_str());
            std::fflush(stdout);
        } else {
            events_.push_back(e);
        }
    }

    void bind_idx(int idx) { backend_->compact_bind_slot(slots_[static_cast<size_t>(idx)].handle); }

    uint32_t slot_pos(int idx) const {
        if (idx < 0) return 0U;
        if (idx == live_) return backend_->seq_pos();
        const SlotRec& s = slots_[static_cast<size_t>(idx)];
        if (!s.is_default) return backend_->compact_slot_seq_pos(s.handle);
        backend_->compact_bind_slot(nullptr); /* O(1) bind dance for the unbound default slot */
        const uint32_t p = backend_->seq_pos();
        backend_->compact_bind_slot(slots_[static_cast<size_t>(live_)].handle);
        return p;
    }

    int acquire_free_slot(bool paged, uint32_t pool_override = 0U) {
        for (size_t i = 0U; i < slots_.size(); ++i) {
            const int ii = static_cast<int>(i);
            if (ii == live_ || ii == shadow_ || ii == rollback_ || ii == djob_.slot_idx) continue;
            if (slots_[i].paged == paged) return ii;
        }
        const uint32_t pool = (pool_override > 0U) ? pool_override : session_->cfg().compact_pool;
        void* h = backend_->compact_slot_create(paged ? pool : 0U);
        if (!h) return -1;
        slots_.push_back(SlotRec{h, false, paged});
        return static_cast<int>(slots_.size()) - 1;
    }

    void release_unused_slots() {
        for (size_t k = slots_.size(); k-- > 1U;) {
            const int kk = static_cast<int>(k);
            if (kk == live_ || kk == shadow_ || kk == rollback_ || kk == djob_.slot_idx) continue;
            backend_->compact_slot_destroy(slots_[k].handle);
            slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(k));
            if (live_ > kk) --live_;
            if (shadow_ > kk) --shadow_;
            if (rollback_ > kk) --rollback_;
            if (djob_.slot_idx > kk) --djob_.slot_idx;
        }
    }

    void release_all() {
        if (slots_.size() > 1U || live_ != 0) {
            backend_->compact_bind_slot(nullptr);
        }
        for (size_t k = slots_.size(); k-- > 1U;) {
            backend_->compact_slot_destroy(slots_[k].handle);
        }
        slots_.resize(1U);
        live_ = 0;
        shadow_ = rollback_ = -1;
        djob_ = DistillJob{};
    }

    void discard_distill() {
        djob_ = DistillJob{};
    }

    void discard_shadow(const char* why) {
        if (shadow_ >= 0 && why) {
            event(std::string("[compact] Shadow warmup discarded: ") + why);
        }
        shadow_ = -1;
        plan_   = ShadowPlan{};
    }

    void drop_rollback() {
        rollback_ = -1;
        rollback_book_.reset();
        rollback_plan_ = ShadowPlan{};
    }

    void catch_up_ledger() {
        const std::vector<ChatTurnRecord>& h = session_->history();
        while (ledger_.size() < h.size()) {
            const size_t i = ledger_.size();
            LedgerEntry e = ledger_distill_turn(static_cast<uint32_t>(i), h[i].user_text, h[i].clean_reply);
            /* Long-form episodes (> 800 bytes) on real MTP engines schedule idle-gap 180B LLM semantic distillation! */
            if (backend_->has_mtp() && h[i].user_text.size() > 800U) {
                e.semantic_done = false;
            }
            ledger_.push_back(std::move(e));
        }
    }

    bool can_select_split() const {
        const std::vector<ChatTurnRecord>& h = session_->history();
        const size_t floor_idx = session_->history_floor();
        if (h.size() < floor_idx + 2U) return false;
        std::vector<uint32_t> toks(h.size(), 0U);
        for (size_t i = floor_idx; i < h.size(); ++i) toks[i] = h[i].seq_pos_after - h[i].seq_pos_before;
        return select_overlap_split(toks, floor_idx, window()) >= 0;
    }

    int next_undistilled_idx() const {
        for (size_t i = 0U; i < ledger_.size(); ++i) {
            if (!ledger_[i].semantic_done) return static_cast<int>(i);
        }
        return -1;
    }

    bool has_pending_semantic_before_next_split() {
        catch_up_ledger();
        const std::vector<ChatTurnRecord>& h = session_->history();
        const size_t floor_idx = session_->history_floor();
        std::vector<uint32_t> toks(h.size(), 0U);
        for (size_t i = floor_idx; i < h.size(); ++i) toks[i] = h[i].seq_pos_after - h[i].seq_pos_before;
        const int m = select_overlap_split(toks, floor_idx, window());
        if (m <= 0) return next_undistilled_idx() >= 0;
        for (size_t i = 0U; i < static_cast<size_t>(m) && i < ledger_.size(); ++i) {
            if (!ledger_[i].semantic_done) return true;
        }
        return false;
    }

    bool distill_step() {
        catch_up_ledger();
        if (!djob_.active) {
            const int idx = next_undistilled_idx();
            if (idx < 0) return false;
            const std::vector<ChatTurnRecord>& h = session_->history();
            if (static_cast<size_t>(idx) >= h.size()) {
                ledger_[static_cast<size_t>(idx)].semantic_done = true;
                return true;
            }
            const std::string body = ledger_build_unique_episode_excerpt(
                h[static_cast<size_t>(idx)].user_text, h[static_cast<size_t>(idx)].clean_reply, 12000U);
            const std::string distill_prefix = backend_->is_glm53()
                ? "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>"
                : "<|im_start|>user\n";
            const std::string distill_suffix = backend_->is_glm53()
                ? "<|assistant|><think></think>"
                : "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            std::vector<uint32_t> pids;
            if (!chat_encode_append(backend_, distill_prefix, /*parse_special=*/true, &pids) ||
                !chat_encode_append(backend_, body, /*parse_special=*/false, &pids) ||
                !chat_encode_append(backend_, distill_suffix, /*parse_special=*/true, &pids) ||
                pids.empty()) {
                ledger_[static_cast<size_t>(idx)].semantic_done = true;
                return true;
            }
            const int sidx = acquire_free_slot(mode_ == STRATA_COMPACT_HYBRID);
            if (sidx < 0) {
                ledger_[static_cast<size_t>(idx)].semantic_done = true;
                return false;
            }
            djob_            = DistillJob{};
            djob_.active     = true;
            djob_.turn_idx   = static_cast<size_t>(idx);
            djob_.slot_idx   = sidx;
            djob_.prompt_ids = std::move(pids);
        }

        if (!djob_.reset_done) {
            bind_idx(djob_.slot_idx);
            backend_->compact_reset_bound();
            bind_idx(live_);
            djob_.reset_done = true;
            return true;
        }

        if (djob_.prefill_done < djob_.prompt_ids.size()) {
            const size_t rem = djob_.prompt_ids.size() - djob_.prefill_done;
            const uint32_t n = static_cast<uint32_t>(std::min<size_t>(slice_tokens_, rem));
            const bool is_last = (djob_.prefill_done + n == djob_.prompt_ids.size());
            uint32_t first_tok = 0U;
            bind_idx(djob_.slot_idx);
            const int rc = backend_->compact_prefill_slice(
                djob_.prompt_ids.data() + djob_.prefill_done, n, is_last, &first_tok);
            bind_idx(live_);
            if (rc != 0) {
                if (djob_.turn_idx < ledger_.size()) ledger_[djob_.turn_idx].semantic_done = true;
                discard_distill();
                return false;
            }
            djob_.prefill_done += n;
            if (is_last) {
                if (backend_->is_eos(first_tok)) {
                    finish_distill_job();
                } else {
                    djob_.pending_tok = first_tok;
                    djob_.out_ids.push_back(first_tok);
                }
            }
            return true;
        }

        /* Decode 1 greedy MTP K=2 step (~35 ms preemptible quantum) up to 176 summary tokens */
        constexpr size_t kMaxSemanticToks = 176U;
        ChatSamplingSpec gs{};
        gs.greedy      = true;
        gs.temperature = 0.0f;
        uint32_t em[8] = {};
        uint32_t ne    = 0U;
        bind_idx(djob_.slot_idx);
        const int rc = backend_->decode_step(djob_.pending_tok, 2U, gs, em, &ne, nullptr);
        bind_idx(live_);
        if (rc != 0 || ne == 0U) {
            finish_distill_job();
            return true;
        }
        bool hit_eos = false;
        for (uint32_t k = 0U; k < ne; ++k) {
            if (backend_->is_eos(em[k])) {
                hit_eos = true;
                break;
            }
            djob_.out_ids.push_back(em[k]);
            djob_.pending_tok = em[k];
            if (djob_.out_ids.size() >= kMaxSemanticToks) {
                hit_eos = true;
                break;
            }
        }
        if (hit_eos) {
            finish_distill_job();
        }
        return true;
    }

    void finish_distill_job() {
        if (djob_.turn_idx < ledger_.size()) {
            if (!djob_.out_ids.empty()) {
                const std::string raw = backend_->decode(
                    djob_.out_ids.data(), static_cast<uint32_t>(djob_.out_ids.size()), /*skip_special=*/true);
                ledger_[djob_.turn_idx].semantic_summary = ledger_gist(raw, 960U);
            }
            ledger_[djob_.turn_idx].semantic_done = true;
            ++semantic_distills_;
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "[compact] Idle-gap %s semantic distill E%zu complete (%zu -> %zu toks)",
                          backend_->is_glm53() ? "320B" : "180B",
                          djob_.turn_idx + 1U, djob_.prompt_ids.size(), djob_.out_ids.size());
            event(buf);
        }
        discard_distill();
    }

    void finish_all_distills() {
        catch_up_ledger();
        while (djob_.active || next_undistilled_idx() >= 0) {
            if (!distill_step()) break;
        }
    }

    bool launch(bool forced, std::string* why) {
        const std::vector<ChatTurnRecord>& h = session_->history();
        const size_t floor_idx = session_->history_floor();
        std::vector<uint32_t> toks(h.size(), 0U);
        for (size_t i = floor_idx; i < h.size(); ++i) toks[i] = h[i].seq_pos_after - h[i].seq_pos_before;
        const int m = select_overlap_split(toks, floor_idx, window());
        if (m < 0) {
            if (why) *why = "need >= 2 verbatim turns above the ledger floor to fold one";
            return false;
        }
        catch_up_ledger();
        const ChatCliConfig& c = session_->cfg();
        const std::string content = ledger_build_system_content(c.system_prompt, ledger_, static_cast<size_t>(m));
        const std::string sys_blk = backend_->format_system_block(content, c.enable_thinking, c.effort);
        std::vector<uint32_t> prefix;
        if (!backend_->encode(sys_blk, /*parse_special=*/true, &prefix) || prefix.empty()) {
            if (why) *why = "failed to tokenize shadow ledger prefix";
            return false;
        }
        const std::vector<uint32_t>& ids = session_->committed_transcript_ids();
        const uint32_t ov_begin = h[static_cast<size_t>(m)].seq_pos_before;
        if (ov_begin > ids.size()) {
            if (why) *why = "transcript bookkeeping mismatch";
            return false;
        }
        ShadowPlan p;
        p.split      = static_cast<size_t>(m);
        p.sys_block  = sys_blk;
        p.prefix_len = static_cast<uint32_t>(prefix.size());
        p.target     = std::move(prefix);
        p.target.insert(p.target.end(), ids.begin() + ov_begin, ids.end());
        const double live_n = static_cast<double>(ids.size());
        const double lim    = forced ? live_n : kCompactBenefitFrac * live_n;
        if (static_cast<double>(p.target.size()) >= lim) {
            if (why) {
                *why = "no benefit yet (shadow " + std::to_string(p.target.size()) + " toks vs live " +
                       std::to_string(ids.size()) + ")";
            }
            return false;
        }
        const int idx = acquire_free_slot(mode_ == STRATA_COMPACT_HYBRID);
        if (idx < 0) {
            if (why) *why = "failed to allocate shadow context slot";
            return false;
        }
        shadow_ = idx;
        plan_   = std::move(p);
        ++launches_;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "[compact] Shadow warmup launched: ledger E1..E%zu + verbatim turns %zu..%zu => %zu toks (live %u, K_rem=%.2f)",
                      plan_.split, plan_.split + 1U, h.size(), plan_.target.size(), backend_->seq_pos(), last_k_rem_);
        event(buf);
        return true;
    }

    /* One background quantum on the shadow slot: bind(shadow) -> reset | prefill_slice -> bind(live). */
    bool warm_step() {
        if (shadow_ < 0) return false;
        if (!plan_.reset_done) {
            bind_idx(shadow_);
            backend_->compact_reset_bound();
            bind_idx(live_);
            plan_.reset_done = true;
            return true;
        }
        if (plan_.done >= plan_.target.size()) return false;
        const uint32_t n = static_cast<uint32_t>(
            std::min<size_t>(slice_tokens_, plan_.target.size() - plan_.done));
        uint32_t ign = 0U;
        bind_idx(shadow_);
        const int rc = backend_->compact_prefill_slice(plan_.target.data() + plan_.done, n, false, &ign);
        bind_idx(live_);
        if (rc != 0) {
            last_note_ = "shadow prefill_slice failed (rc=" + std::to_string(rc) + ")";
            discard_shadow(last_note_.c_str());
            return false;
        }
        plan_.done += n;
        if (plan_.done == plan_.target.size()) {
            event("[compact] Shadow DUAL_SYNCED at " + std::to_string(plan_.done) + " toks (live " +
                  std::to_string(backend_->seq_pos()) + ")");
        }
        return true;
    }

    bool finish_warmup() {
        bool any = false;
        while (shadow_ >= 0 && !synced()) {
            if (!warm_step()) break;
            any = true;
        }
        return any;
    }

    /* O(1) RCU promotion of a DUAL_SYNCED shadow (timed bind + role swap). */
    bool promote(const char* why) {
        if (!synced()) return false;
        const uint32_t shadow_pos = slot_pos(shadow_);
        if (shadow_pos != plan_.target.size()) {
            last_note_ = "shadow seq_pos " + std::to_string(shadow_pos) + " != target " +
                         std::to_string(plan_.target.size());
            discard_shadow(last_note_.c_str());
            return false;
        }
        const uint32_t pre_pos = backend_->seq_pos();
        auto book = std::make_unique<ChatReplSession::Bookkeeping>(session_->snapshot_bookkeeping());
        drop_rollback();

        const auto t0 = SteadyClock::now();
        bind_idx(shadow_);
        rollback_ = live_;
        live_     = shadow_;
        shadow_   = -1;
        const double ms = chat_elapsed_ms(t0, SteadyClock::now());

        ShadowPlan meta;
        meta.split      = plan_.split;
        meta.sys_block  = plan_.sys_block;
        meta.prefix_len = plan_.prefix_len;
        if (!session_->install_compacted_transcript(std::move(plan_.target), plan_.sys_block, plan_.split,
                                                    plan_.prefix_len)) {
            /* Should never happen; roll back to keep host & GPU state consistent */
            bind_idx(rollback_);
            live_     = rollback_;
            rollback_ = -1;
            plan_     = ShadowPlan{};
            last_note_ = "promotion bookkeeping mismatch; rolled back";
            return false;
        }
        plan_          = ShadowPlan{};
        rollback_book_ = std::move(book);
        rollback_plan_ = std::move(meta);
        ++swaps_;
        last_switchover_ms_  = ms;
        last_pre_swap_pos_   = pre_pos;
        last_post_swap_pos_  = backend_->seq_pos();
        last_n_              = last_post_swap_pos_;
        last_k_rem_          = predictor_.k_rem(static_cast<double>(w_swap()), static_cast<double>(last_n_));
        inhibit_auto_        = false;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "[compact] RCU swap #%u (%s): live seq_pos %u -> %u in %.4f ms | ledger floor %zu turns | /undo-compact available",
                      swaps_, why, pre_pos, last_post_swap_pos_, ms, session_->history_floor());
        event(buf);
        return true;
    }

    IChatBackend*                                  backend_  = nullptr;
    ChatReplSession*                               session_  = nullptr;
    StrataCompactionMode                           mode_     = STRATA_COMPACT_OFF;
    std::vector<SlotRec>                           slots_;
    int                                            live_     = 0;
    int                                            shadow_   = -1;
    int                                            rollback_ = -1;
    ShadowPlan                                     plan_;
    ShadowPlan                                     rollback_plan_;
    DistillJob                                     djob_;
    std::unique_ptr<ChatReplSession::Bookkeeping>  rollback_book_;
    std::vector<LedgerEntry>                       ledger_;
    CompactVelocityPredictor                       predictor_;
    uint32_t                                       last_n_             = 0U;
    double                                         last_k_rem_         = std::numeric_limits<double>::infinity();
    bool                                           inhibit_auto_       = false;
    bool                                           echo_events_        = true;
    bool                                           quiet_              = false;
    uint32_t                                       slice_tokens_       = kShadowSliceBatch;
    uint32_t                                       swaps_              = 0U;
    uint32_t                                       undo_compacts_      = 0U;
    uint32_t                                       launches_           = 0U;
    uint32_t                                       semantic_distills_  = 0U;
    uint64_t                                       mirrored_toks_      = 0ULL;
    double                                         last_switchover_ms_ = 0.0;
    double                                         last_undo_ms_       = 0.0;
    double                                         max_user_stall_ms_  = 0.0;
    uint32_t                                       last_pre_swap_pos_  = 0U;
    uint32_t                                       last_post_swap_pos_ = 0U;
    std::string                                    last_note_;
    std::vector<std::string>                       events_;
};

/* ---- Cooperative idle-gap line reader (`poll()` multiplexor, study 4.3) ----
 * TTY: stdin is switched to cbreak (no ICANON / no ECHO; ISIG kept) so the FIRST keystroke is visible to
 * poll(): background quanta run only while the edit buffer is empty and no byte is pending; `poll(0)` is
 * checked before EVERY quantum, `poll(-1)` sleeps when there is no idle work. Local echo + Backspace,
 * Ctrl-U, Ctrl-D (EOF on empty line) are handled here; ANSI escape sequences (arrow keys) are swallowed.
 * The original termios is restored on destruction and on SIGINT/SIGTERM/SIGHUP/SIGQUIT. */

static struct termios        g_chat_saved_termios;
static volatile sig_atomic_t g_chat_termios_saved = 0;
static volatile sig_atomic_t g_chat_bpaste_on     = 0;

static void chat_termios_signal_restore(int sig) {
    if (g_chat_bpaste_on) {
        static const char kOff[] = "\033[?2004l";
        (void)!::write(STDOUT_FILENO, kOff, sizeof(kOff) - 1U);
        g_chat_bpaste_on = 0;
    }
    if (g_chat_termios_saved) {
        (void)tcsetattr(STDIN_FILENO, TCSANOW, &g_chat_saved_termios);
    }
    (void)signal(sig, SIG_DFL);
    (void)raise(sig);
}

class CooperativeLineReader {
public:
    CooperativeLineReader(int fd, bool tty_cbreak, bool echo) : fd_(fd), echo_(echo) {
        if (tty_cbreak) enable_cbreak();
    }
    ~CooperativeLineReader() { restore(); }
    CooperativeLineReader(const CooperativeLineReader&) = delete;
    CooperativeLineReader& operator=(const CooperativeLineReader&) = delete;

    bool cbreak_active() const { return cbreak_; }

    /* Feeds raw input bytes through the line editor (supports CSI ?2004h `\033[200~...\033[201~` multi-line paste). */
    void feed(const char* buf, size_t n) {
        for (size_t i = 0U; i < n; ++i) {
            const unsigned char c = static_cast<unsigned char>(buf[i]);
            if (esc_state_ == 1) {
                csi_param_ = 0U;
                esc_state_ = (c == '[' || c == 'O') ? 2 : 0;
                continue;
            }
            if (esc_state_ == 2) {
                if (c >= '0' && c <= '9') {
                    csi_param_ = csi_param_ * 10U + static_cast<uint32_t>(c - '0');
                    continue;
                }
                if (c >= 0x40U && c <= 0x7EU) {
                    if (c == '~') {
                        if (csi_param_ == 200U) {
                            in_paste_     = true;
                            paste_saw_cr_ = false;
                        } else if (csi_param_ == 201U) {
                            in_paste_     = false;
                            paste_saw_cr_ = false;
                        }
                    }
                    esc_state_ = 0;
                }
                continue;
            }
            if (in_paste_ && (c == '\n' || c == '\r')) {
                if (c == '\n' && paste_saw_cr_) {
                    paste_saw_cr_ = false;
                    continue; /* collapse CRLF -> single LF inside bracketed paste */
                }
                paste_saw_cr_ = (c == '\r');
                edit_.push_back('\n');
                echo_str("\n");
                continue;
            }
            paste_saw_cr_ = false;
            if (c == '\n' || c == '\r') {
                echo_str("\n");
                ready_.push_back(edit_);
                edit_.clear();
            } else if (c == 0x7FU || c == 0x08U) {
                if (!edit_.empty()) {
                    while (!edit_.empty() && (static_cast<unsigned char>(edit_.back()) & 0xC0U) == 0x80U) {
                        edit_.pop_back();
                    }
                    if (!edit_.empty()) edit_.pop_back();
                    echo_str("\b \b");
                }
            } else if (c == 0x15U) { /* Ctrl-U */
                size_t glyphs = 0U;
                for (char ch : edit_) {
                    if ((static_cast<unsigned char>(ch) & 0xC0U) != 0x80U) ++glyphs;
                }
                for (size_t g = 0U; g < glyphs; ++g) echo_str("\b \b");
                edit_.clear();
            } else if (c == 0x04U) { /* Ctrl-D */
                if (edit_.empty()) eof_ = true;
            } else if (c == 0x1BU) {
                esc_state_ = 1;
            } else if (c < 0x20U && c != '\t') {
                /* ignore other control bytes */
            } else {
                edit_.push_back(static_cast<char>(c));
                if (echo_) {
                    const char ch = static_cast<char>(c);
                    (void)!::write(STDOUT_FILENO, &ch, 1U);
                }
            }
        }
    }
    bool pop_ready(std::string* out) {
        if (ready_.empty()) return false;
        *out = std::move(ready_.front());
        ready_.pop_front();
        return true;
    }
    bool typing() const { return !edit_.empty() || in_paste_; }
    void mark_eof() { eof_ = true; }

    /* Returns false on EOF. Runs controller idle quanta while the user is not typing. */
    bool read_line(CompactionController* comp, std::string* line_out) {
        std::fflush(stdout);
        while (true) {
            if (pop_ready(line_out)) return true;
            if (eof_) {
                if (!edit_.empty()) {
                    *line_out = std::move(edit_);
                    edit_.clear();
                    return true;
                }
                return false;
            }
            const bool work = comp && !typing() && comp->has_idle_work();
            struct pollfd pfd = {fd_, POLLIN, 0};
            const int rc = ::poll(&pfd, 1, work ? 0 : -1);
            if (rc < 0) {
                if (errno == EINTR) continue;
                eof_ = true;
                continue;
            }
            if (rc > 0) {
                if (pfd.revents & (POLLIN | POLLHUP | POLLERR)) {
                    char buf[4096];
                    const ssize_t nr = ::read(fd_, buf, sizeof(buf));
                    if (nr > 0) {
                        feed(buf, static_cast<size_t>(nr));
                    } else if (nr == 0 || (errno != EINTR && errno != EAGAIN)) {
                        eof_ = true;
                    }
                }
                continue; /* first keystroke => background quanta pause until the line is complete */
            }
            double qms = 0.0;
            (void)comp->idle_quantum(&qms);
            struct pollfd pfd2 = {fd_, POLLIN, 0};
            if (::poll(&pfd2, 1, 0) > 0) {
                comp->note_user_stall(qms); /* input arrived during this quantum: stall <= quantum */
            }
        }
    }

private:
    void echo_str(const char* s) {
        if (echo_) (void)!::write(STDOUT_FILENO, s, std::strlen(s));
    }
    void enable_cbreak() {
        struct termios t{};
        if (tcgetattr(fd_, &t) != 0) return;
        g_chat_saved_termios = t;
        g_chat_termios_saved = 1;
        t.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        t.c_cc[VMIN]  = 1;
        t.c_cc[VTIME] = 0;
        if (tcsetattr(fd_, TCSANOW, &t) != 0) {
            g_chat_termios_saved = 0;
            return;
        }
        cbreak_ = true;
        if (echo_) {
            static const char kOn[] = "\033[?2004h";
            (void)!::write(STDOUT_FILENO, kOn, sizeof(kOn) - 1U);
            g_chat_bpaste_on = 1;
        }
        prev_int_  = signal(SIGINT, chat_termios_signal_restore);
        prev_term_ = signal(SIGTERM, chat_termios_signal_restore);
        prev_hup_  = signal(SIGHUP, chat_termios_signal_restore);
        prev_quit_ = signal(SIGQUIT, chat_termios_signal_restore);
    }
    void restore() {
        if (!cbreak_) return;
        if (g_chat_bpaste_on) {
            static const char kOff[] = "\033[?2004l";
            (void)!::write(STDOUT_FILENO, kOff, sizeof(kOff) - 1U);
            g_chat_bpaste_on = 0;
        }
        (void)tcsetattr(fd_, TCSANOW, &g_chat_saved_termios);
        g_chat_termios_saved = 0;
        (void)signal(SIGINT, prev_int_);
        (void)signal(SIGTERM, prev_term_);
        (void)signal(SIGHUP, prev_hup_);
        (void)signal(SIGQUIT, prev_quit_);
        cbreak_ = false;
    }

    int                     fd_           = STDIN_FILENO;
    bool                    echo_         = false;
    bool                    cbreak_       = false;
    bool                    eof_          = false;
    bool                    in_paste_     = false;
    bool                    paste_saw_cr_ = false;
    int                     esc_state_    = 0;
    uint32_t                csi_param_    = 0U;
    std::string             edit_;
    std::deque<std::string> ready_;
    sig_t                   prev_int_     = SIG_DFL;
    sig_t                   prev_term_    = SIG_DFL;
    sig_t                   prev_hup_     = SIG_DFL;
    sig_t                   prev_quit_    = SIG_DFL;
};

/* ============================================================================
 * 4c. Persistent `.strata_session` Snapshot Container (`/save` & `/load`)
 * ============================================================================ */

static constexpr uint64_t kStrataSessionMagic   = 0x53534553354D5453ULL; /* "STM5SESS" */
static constexpr uint32_t kStrataSessionVersion = 1U;

static bool sess_io_all(int fd, void* buf, size_t n, bool write) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t done = 0U;
    while (done < n) {
        const ssize_t rc = write ? ::write(fd, p + done, n - done) : ::read(fd, p + done, n - done);
        if (rc < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (rc == 0) return false;
        done += static_cast<size_t>(rc);
    }
    return true;
}

template <typename T>
static bool sess_write_pod(int fd, const T& v) {
    return sess_io_all(fd, const_cast<T*>(&v), sizeof(T), true);
}
template <typename T>
static bool sess_read_pod(int fd, T* v) {
    return sess_io_all(fd, v, sizeof(T), false);
}
static bool sess_write_str(int fd, const std::string& s) {
    const uint64_t len = s.size();
    if (!sess_write_pod(fd, len)) return false;
    return len == 0ULL || sess_io_all(fd, const_cast<char*>(s.data()), static_cast<size_t>(len), true);
}
static bool sess_read_str(int fd, std::string* s) {
    uint64_t len = 0ULL;
    if (!sess_read_pod(fd, &len) || len > (256ULL << 20)) return false;
    s->resize(static_cast<size_t>(len));
    return len == 0ULL || sess_io_all(fd, s->data(), static_cast<size_t>(len), false);
}
static bool sess_write_str_vec(int fd, const std::vector<std::string>& v) {
    const uint64_t n = v.size();
    if (!sess_write_pod(fd, n)) return false;
    for (const std::string& s : v) {
        if (!sess_write_str(fd, s)) return false;
    }
    return true;
}
static bool sess_read_str_vec(int fd, std::vector<std::string>* v) {
    uint64_t n = 0ULL;
    if (!sess_read_pod(fd, &n) || n > 100000ULL) return false;
    v->resize(static_cast<size_t>(n));
    for (size_t i = 0U; i < v->size(); ++i) {
        if (!sess_read_str(fd, &(*v)[i])) return false;
    }
    return true;
}
static bool sess_write_u32_vec(int fd, const std::vector<uint32_t>& v) {
    const uint64_t n = v.size();
    if (!sess_write_pod(fd, n)) return false;
    return n == 0ULL || sess_io_all(fd, const_cast<uint32_t*>(v.data()), static_cast<size_t>(n) * sizeof(uint32_t), true);
}
static bool sess_read_u32_vec(int fd, std::vector<uint32_t>* v) {
    uint64_t n = 0ULL;
    if (!sess_read_pod(fd, &n) || n > (64ULL << 20)) return false;
    v->resize(static_cast<size_t>(n));
    return n == 0ULL || sess_io_all(fd, v->data(), static_cast<size_t>(n) * sizeof(uint32_t), false);
}

static bool save_session_to_file(
    const std::string& path,
    IChatBackend* backend,
    ChatReplSession* session,
    CompactionController* comp,
    uint64_t* total_bytes_out,
    double* elapsed_ms_out,
    std::string* err_out
) {
    const auto t0 = SteadyClock::now();
    if (!backend || !backend->supports_session_save()) {
        if (err_out) *err_out = "session save is not supported by this backend";
        return false;
    }
    const size_t slash = path.find_last_of('/');
    const std::string dir = (slash == std::string::npos)
        ? std::string(".")
        : ((slash == 0U) ? std::string("/") : path.substr(0U, slash));
    std::string tmp_tmpl = dir + "/.strata_session_tmp_XXXXXX";
    std::vector<char> tmp_buf(tmp_tmpl.begin(), tmp_tmpl.end());
    tmp_buf.push_back('\0');
    const int fd = ::mkstemp(tmp_buf.data());
    if (fd < 0) {
        if (err_out) *err_out = "failed to create temporary session file in `" + dir + "` (errno=" + std::to_string(errno) + ")";
        return false;
    }
    const std::string tmp_path(tmp_buf.data());
    (void)::fchmod(fd, 0600);
    (void)fcntl(fd, F_NOCACHE, 1);

    auto fail = [&](const std::string& msg) -> bool {
        ::close(fd);
        (void)::unlink(tmp_path.c_str());
        if (err_out) *err_out = msg;
        return false;
    };

    const ChatCliConfig& cfg = session->cfg();
    const ChatReplSession::Bookkeeping bk = session->snapshot_bookkeeping();
    const CompactionController::PersistentSnapshot cs = comp->export_persistent_state();

    if (!sess_write_pod(fd, kStrataSessionMagic) ||
        !sess_write_pod(fd, kStrataSessionVersion) ||
        !sess_write_str(fd, backend->engine_id_tag()) ||
        !sess_write_str(fd, cfg.system_prompt) ||
        !sess_write_str(fd, bk.effective_system_block) ||
        !sess_write_pod(fd, static_cast<uint8_t>(cfg.enable_thinking ? 1U : 0U)) ||
        !sess_write_str(fd, cfg.effort) ||
        !sess_write_pod(fd, static_cast<uint8_t>(cfg.keep_think ? 1U : 0U)) ||
        !sess_write_pod(fd, cfg.draft_k) ||
        !sess_write_pod(fd, static_cast<uint8_t>(cfg.greedy ? 1U : 0U)) ||
        !sess_write_pod(fd, cfg.temperature) ||
        !sess_write_pod(fd, cfg.top_k) ||
        !sess_write_pod(fd, cfg.top_p) ||
        !sess_write_pod(fd, cfg.max_tokens) ||
        !sess_write_pod(fd, cfg.compact_window) ||
        !sess_write_pod(fd, cfg.compact_pool) ||
        !sess_write_pod(fd, cfg.compact_watermark) ||
        !sess_write_pod(fd, static_cast<uint64_t>(bk.history_floor)) ||
        !sess_write_u32_vec(fd, bk.committed_ids)) {
        return fail("failed writing session header/transcript");
    }

    const uint64_t hist_n = bk.history.size();
    if (!sess_write_pod(fd, hist_n)) return fail("failed writing history count");
    for (const ChatTurnRecord& r : bk.history) {
        if (!sess_write_str(fd, r.user_text) ||
            !sess_write_str(fd, r.think_text) ||
            !sess_write_str(fd, r.clean_reply) ||
            !sess_write_str(fd, r.committed_suffix) ||
            !sess_write_u32_vec(fd, r.generated_ids) ||
            !sess_write_pod(fd, r.seq_pos_before) ||
            !sess_write_pod(fd, r.seq_pos_after) ||
            !sess_write_pod(fd, r.prompt_delta_toks) ||
            !sess_write_pod(fd, r.committed_toks) ||
            !sess_write_pod(fd, r.prefill_ms) ||
            !sess_write_pod(fd, r.decode_ms) ||
            !sess_write_pod(fd, r.decode_tok_s) ||
            !sess_write_pod(fd, r.commit_ms) ||
            !sess_write_pod(fd, static_cast<uint8_t>(r.hit_max_tokens ? 1U : 0U))) {
            return fail("failed writing turn history record");
        }
    }

    const uint32_t cmode = static_cast<uint32_t>(cs.mode);
    const uint64_t led_n = cs.ledger.size();
    if (!sess_write_pod(fd, cmode) ||
        !sess_write_pod(fd, cs.swaps) ||
        !sess_write_pod(fd, cs.launches) ||
        !sess_write_pod(fd, cs.semantic_distills) ||
        !sess_write_pod(fd, cs.mirrored_toks) ||
        !sess_write_pod(fd, cs.predictor.v) ||
        !sess_write_pod(fd, cs.predictor.s2) ||
        !sess_write_pod(fd, cs.predictor.samples) ||
        !sess_write_pod(fd, led_n)) {
        return fail("failed writing compaction metadata");
    }
    for (const LedgerEntry& e : cs.ledger) {
        if (!sess_write_pod(fd, e.turn_index) ||
            !sess_write_str_vec(fd, e.facts.kv) ||
            !sess_write_str_vec(fd, e.facts.ids) ||
            !sess_write_str_vec(fd, e.facts.hex) ||
            !sess_write_str_vec(fd, e.facts.nums) ||
            !sess_write_str_vec(fd, e.rules) ||
            !sess_write_str(fd, e.semantic_summary) ||
            !sess_write_pod(fd, static_cast<uint8_t>(e.semantic_done ? 1U : 0U)) ||
            !sess_write_str(fd, e.text)) {
            return fail("failed writing episode ledger entry");
        }
    }

    uint64_t gpu_bytes = 0ULL;
    const int grc = backend->save_state_fd(fd, &gpu_bytes);
    if (grc != 0) {
        return fail("failed saving GPU/NVMe sequence state (rc=" + std::to_string(grc) + ")");
    }
    const off_t end_off = ::lseek(fd, 0, SEEK_CUR);
    if (::close(fd) != 0) {
        const int cerr = errno;
        (void)::unlink(tmp_path.c_str());
        if (err_out) *err_out = "failed closing temporary session file (errno=" + std::to_string(cerr) + ")";
        return false;
    }
    if (::rename(tmp_path.c_str(), path.c_str()) != 0) {
        const int rerr = errno;
        (void)::unlink(tmp_path.c_str());
        if (err_out) *err_out = "failed renaming temporary session file to `" + path + "` (errno=" + std::to_string(rerr) + ")";
        return false;
    }
    if (total_bytes_out) *total_bytes_out = (end_off > 0) ? static_cast<uint64_t>(end_off) : gpu_bytes;
    if (elapsed_ms_out) *elapsed_ms_out = chat_elapsed_ms(t0, SteadyClock::now());
    return true;
}

static bool load_session_from_file(
    const std::string& path,
    IChatBackend* backend,
    ChatReplSession* session,
    CompactionController* comp,
    uint64_t* total_bytes_out,
    double* elapsed_ms_out,
    std::string* err_out
) {
    const auto t0 = SteadyClock::now();
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (err_out) *err_out = "failed to open `" + path + "` (errno=" + std::to_string(errno) + ")";
        return false;
    }
    (void)fcntl(fd, F_NOCACHE, 1);

    auto fail = [&](const std::string& msg) -> bool {
        ::close(fd);
        if (err_out) *err_out = msg;
        return false;
    };

    uint64_t magic = 0ULL;
    uint32_t ver   = 0U;
    std::string eng_tag;
    if (!sess_read_pod(fd, &magic) || magic != kStrataSessionMagic ||
        !sess_read_pod(fd, &ver) || ver != kStrataSessionVersion ||
        !sess_read_str(fd, &eng_tag)) {
        return fail("not a valid `.strata_session` v1 file");
    }
    if (eng_tag != backend->engine_id_tag()) {
        return fail("session engine `" + eng_tag + "` does not match running engine `" + backend->engine_id_tag() + "`");
    }

    ChatCliConfig new_cfg = session->cfg();
    ChatReplSession::Bookkeeping bk{};
    uint8_t en_th = 0U, kp_th = 0U, grd = 0U;
    uint64_t hfloor = 0ULL;
    if (!sess_read_str(fd, &new_cfg.system_prompt) ||
        !sess_read_str(fd, &bk.effective_system_block) ||
        !sess_read_pod(fd, &en_th) ||
        !sess_read_str(fd, &new_cfg.effort) ||
        !sess_read_pod(fd, &kp_th) ||
        !sess_read_pod(fd, &new_cfg.draft_k) ||
        !sess_read_pod(fd, &grd) ||
        !sess_read_pod(fd, &new_cfg.temperature) ||
        !sess_read_pod(fd, &new_cfg.top_k) ||
        !sess_read_pod(fd, &new_cfg.top_p) ||
        !sess_read_pod(fd, &new_cfg.max_tokens) ||
        !sess_read_pod(fd, &new_cfg.compact_window) ||
        !sess_read_pod(fd, &new_cfg.compact_pool) ||
        !sess_read_pod(fd, &new_cfg.compact_watermark) ||
        !sess_read_pod(fd, &hfloor) ||
        !sess_read_u32_vec(fd, &bk.committed_ids)) {
        return fail("corrupted session header/transcript");
    }
    new_cfg.enable_thinking = (en_th != 0U);
    new_cfg.keep_think      = (kp_th != 0U);
    new_cfg.greedy          = (grd != 0U);
    if (!backend->has_mtp()) new_cfg.draft_k = 0U;
    bk.history_floor    = static_cast<size_t>(hfloor);
    bk.undo_valid_count = 0U;

    uint64_t hist_n = 0ULL;
    if (!sess_read_pod(fd, &hist_n) || hist_n > 100000ULL) return fail("invalid history count");
    bk.history.resize(static_cast<size_t>(hist_n));
    for (size_t i = 0U; i < bk.history.size(); ++i) {
        ChatTurnRecord& r = bk.history[i];
        uint8_t hit_max = 0U;
        if (!sess_read_str(fd, &r.user_text) ||
            !sess_read_str(fd, &r.think_text) ||
            !sess_read_str(fd, &r.clean_reply) ||
            !sess_read_str(fd, &r.committed_suffix) ||
            !sess_read_u32_vec(fd, &r.generated_ids) ||
            !sess_read_pod(fd, &r.seq_pos_before) ||
            !sess_read_pod(fd, &r.seq_pos_after) ||
            !sess_read_pod(fd, &r.prompt_delta_toks) ||
            !sess_read_pod(fd, &r.committed_toks) ||
            !sess_read_pod(fd, &r.prefill_ms) ||
            !sess_read_pod(fd, &r.decode_ms) ||
            !sess_read_pod(fd, &r.decode_tok_s) ||
            !sess_read_pod(fd, &r.commit_ms) ||
            !sess_read_pod(fd, &hit_max)) {
            return fail("corrupted turn history record");
        }
        r.hit_max_tokens = (hit_max != 0U);
    }

    CompactionController::PersistentSnapshot cs{};
    uint32_t cmode = 0U;
    uint64_t led_n = 0ULL;
    if (!sess_read_pod(fd, &cmode) || cmode > 3U ||
        !sess_read_pod(fd, &cs.swaps) ||
        !sess_read_pod(fd, &cs.launches) ||
        !sess_read_pod(fd, &cs.semantic_distills) ||
        !sess_read_pod(fd, &cs.mirrored_toks) ||
        !sess_read_pod(fd, &cs.predictor.v) ||
        !sess_read_pod(fd, &cs.predictor.s2) ||
        !sess_read_pod(fd, &cs.predictor.samples) ||
        !sess_read_pod(fd, &led_n) || led_n > 100000ULL) {
        return fail("corrupted compaction metadata");
    }
    cs.mode          = static_cast<StrataCompactionMode>(cmode);
    new_cfg.compact_mode = cs.mode;
    cs.ledger.resize(static_cast<size_t>(led_n));
    for (size_t i = 0U; i < cs.ledger.size(); ++i) {
        LedgerEntry& e = cs.ledger[i];
        uint8_t sdone = 1U;
        if (!sess_read_pod(fd, &e.turn_index) ||
            !sess_read_str_vec(fd, &e.facts.kv) ||
            !sess_read_str_vec(fd, &e.facts.ids) ||
            !sess_read_str_vec(fd, &e.facts.hex) ||
            !sess_read_str_vec(fd, &e.facts.nums) ||
            !sess_read_str_vec(fd, &e.rules) ||
            !sess_read_str(fd, &e.semantic_summary) ||
            !sess_read_pod(fd, &sdone) ||
            !sess_read_str(fd, &e.text)) {
            return fail("corrupted episode ledger entry");
        }
        e.semantic_done = (sdone != 0U);
    }

    std::string slot_err;
    if (!comp->prepare_slot_for_load(cs.mode, new_cfg.compact_pool, &slot_err)) {
        return fail(slot_err);
    }

    uint64_t gpu_bytes = 0ULL;
    const int grc = backend->load_state_fd(fd, &gpu_bytes);
    if (grc != 0) {
        return fail("failed restoring GPU/NVMe sequence state (rc=" + std::to_string(grc) + ")");
    }
    const off_t end_off = ::lseek(fd, 0, SEEK_CUR);
    ::close(fd);

    session->mutable_cfg() = new_cfg;
    session->restore_bookkeeping(std::move(bk));
    session->sync_turn_start_ckpt();
    comp->import_persistent_state(std::move(cs));
    session->rebaseline_mem();

    if (total_bytes_out) *total_bytes_out = (end_off > 0) ? static_cast<uint64_t>(end_off) : gpu_bytes;
    if (elapsed_ms_out) *elapsed_ms_out = chat_elapsed_ms(t0, SteadyClock::now());
    return true;
}

/* ============================================================================
 * 5. Deterministic 3-Turn Bit-Exact Equivalence Verifier (`--verify-multiturn`)
 * ============================================================================ */

static int run_verify_multiturn(
    IChatBackend* backend,
    const ChatCliConfig& base_cfg
) {
    ChatCliConfig vcfg = base_cfg;
    vcfg.greedy        = true;
    vcfg.temperature   = 0.0f;
    if (!vcfg.max_tok_explicit) {
        vcfg.max_tokens = vcfg.enable_thinking ? 384U : 128U;
    }
    if (backend->is_glm53() && !vcfg.glm53_mode_explicit) {
        vcfg.glm53_mode = STRATA_GLM53_MODE_LOSSLESS;
        backend->set_glm53_exec_mode(STRATA_GLM53_MODE_LOSSLESS);
    }

    const char* kTurns[3] = {
        "Remember two exact integers: ALPHA = 17 and BETA = 25. Reply in one short sentence confirming both values.",
        "Compute GAMMA = ALPHA * 3 + BETA. Reply in one concise sentence stating GAMMA.",
        "State ALPHA, BETA, and GAMMA, then compute DELTA = GAMMA - ALPHA in one concise sentence."
    };

    const uint32_t eff_k = backend->has_mtp() ? vcfg.draft_k : 0U;
    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Bit-Exact 3-Turn Stateful Equivalence Proof (`--verify-multiturn`)\n");
    std::printf("  Model  : %s\n", backend->model_name());
    std::printf("  Device : %s | ZeroCopy=%d | Mode=%s | Speculative K=%u (vs Ref K=0)\n",
                backend->device_name(),
                backend->zero_copy() ? 1 : 0,
                vcfg.enable_thinking ? ("Thinking (<think>, effort=" + vcfg.effort + ")").c_str()
                                     : "Direct (--no-think)",
                eff_k);
    std::printf("====================================================================================================\n");
    std::fflush(stdout);

    /* ---- Phase 1: Incremental 3-Turn Conversation (KV + KDA/DeltaNet + PLE + Checkpoint Compaction) ---- */
    if (backend->is_glm53() && vcfg.glm53_mode == STRATA_GLM53_MODE_ADAPTIVE) {
        backend->reset_glm53_hot_cache();
    }
    backend->reset();
    ChatReplSession session(backend, vcfg);
    if (!session.ok()) {
        std::fprintf(stderr, "[verify-multiturn] Failed to allocate GPU checkpoint arena\n");
        return 1;
    }
    const ChatMemSnapshot baseline_mem = session.baseline_mem();

    ChatTurnRecord t1{}, t2{};
    if (session.run_turn(kTurns[0], /*stream_stdout=*/false, /*ansi_style=*/false, &t1) != 0) return 1;
    std::printf("  [Inc Turn 1] pos %3u -> %3u | prefill %6.1f ms | %6.2f tok/s (%3zu toks) | \"%s\"\n",
                t1.seq_pos_before, t1.seq_pos_after, t1.prefill_ms, t1.decode_tok_s,
                t1.generated_ids.size(), t1.clean_reply.c_str());
    std::fflush(stdout);

    if (session.run_turn(kTurns[1], /*stream_stdout=*/false, /*ansi_style=*/false, &t2) != 0) return 1;
    std::printf("  [Inc Turn 2] pos %3u -> %3u | prefill %6.1f ms | %6.2f tok/s (%3zu toks) | \"%s\"\n",
                t2.seq_pos_before, t2.seq_pos_after, t2.prefill_ms, t2.decode_tok_s,
                t2.generated_ids.size(), t2.clean_reply.c_str());
    std::fflush(stdout);

    /* Run Turn 3 incrementally WITHOUT post-turn suffix commit so we capture exact Turn 3 decode tokens & state */
    const uint32_t inc_pos_before_t3 = backend->seq_pos();
    ChatSamplingSpec greedy{};
    greedy.greedy      = true;
    greedy.temperature = 0.0f;

    void* t3_ckpt = session.turn_start_checkpoint();

    std::vector<uint32_t> t3_hdr_ids, t3_think_ids;
    std::string t3_think_str;
    uint32_t ign = 0U, inc_first_tok = 0U;
    double ckpt_save_ms = 0.0;

    const auto t_inc_pre0 = SteadyClock::now();
    if (backend->is_glm53()) {
        const std::string asst_open =
            vcfg.enable_thinking ? "<|assistant|><think>" : "<|assistant|><think></think>";
        if (!chat_encode_append(backend, "<|user|>", true, &t3_hdr_ids) ||
            !chat_encode_append(backend, chat_trim_ascii(kTurns[2]), false, &t3_hdr_ids) ||
            !chat_encode_append(backend, asst_open, true, &t3_hdr_ids) ||
            t3_hdr_ids.empty()) {
            return 1;
        }
        if (backend->prefill(
                t3_hdr_ids.data(), static_cast<uint32_t>(t3_hdr_ids.size()), greedy, &inc_first_tok, nullptr) != 0) {
            return 1;
        }
        const auto t_ckpt0 = SteadyClock::now();
        if (backend->checkpoint_save(t3_ckpt) != 0) {
            return 1;
        }
        ckpt_save_ms = chat_elapsed_ms(t_ckpt0, SteadyClock::now());
    } else {
        t3_think_str = vcfg.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        if (!chat_encode_append(backend, "<|im_start|>user\n", true, &t3_hdr_ids) ||
            !chat_encode_append(backend, chat_trim_ascii(kTurns[2]), false, &t3_hdr_ids) ||
            !chat_encode_append(backend, "<|im_end|>\n<|im_start|>assistant\n", true, &t3_hdr_ids) ||
            !backend->encode(t3_think_str, true, &t3_think_ids)) {
            return 1;
        }
        if (backend->prefill(t3_hdr_ids.data(), static_cast<uint32_t>(t3_hdr_ids.size()), greedy, &ign, nullptr) != 0) {
            return 1;
        }
        const auto t_ckpt0 = SteadyClock::now();
        if (backend->checkpoint_save(t3_ckpt) != 0) {
            return 1;
        }
        ckpt_save_ms = chat_elapsed_ms(t_ckpt0, SteadyClock::now());
        if (backend->prefill(
                t3_think_ids.data(), static_cast<uint32_t>(t3_think_ids.size()), greedy, &inc_first_tok, nullptr) != 0) {
            return 1;
        }
    }
    const double inc_t3_prefill_ms = chat_elapsed_ms(t_inc_pre0, SteadyClock::now());
    const uint32_t inc_t3_prompt_pos = backend->seq_pos();

    std::vector<uint32_t> inc_t3_ids;
    bool inc_eos = false;
    uint32_t inc_dec_emitted = 0U;
    uint64_t inc_mtp_draft = 0ULL, inc_mtp_acc = 0ULL;

    if (!backend->is_eos(inc_first_tok)) {
        inc_t3_ids.push_back(inc_first_tok);
    } else {
        inc_eos = true;
    }

    uint32_t pending = inc_first_tok;
    const uint32_t target_dec = (vcfg.max_tokens == 0U)
        ? backend->max_seq_len()
        : ((vcfg.max_tokens > 0U) ? (vcfg.max_tokens - 1U) : 0U);
    const auto t_inc_dec0 = SteadyClock::now();
    while (!inc_eos && inc_dec_emitted < target_dec) {
        if (backend->seq_pos() + 1U >= backend->max_seq_len()) break;
        uint32_t em[STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K + 2U] = {};
        uint32_t n_em = 0U;
        StepDeltaTelemetry dtel{};
        if (backend->decode_step(pending, eff_k, greedy, em, &n_em, &dtel) != 0) {
            return 1;
        }
        inc_mtp_draft += dtel.mtp_drafted;
        inc_mtp_acc   += dtel.mtp_accepted;
        for (uint32_t i = 0U; i < n_em; ++i) {
            if (backend->is_eos(em[i])) {
                inc_eos = true;
                break;
            }
            inc_t3_ids.push_back(em[i]);
            ++inc_dec_emitted;
            if (inc_dec_emitted >= target_dec) break;
        }
        if (!inc_eos && n_em > 0U) pending = em[n_em - 1U];
    }
    const double inc_t3_dec_ms = chat_elapsed_ms(t_inc_dec0, SteadyClock::now());
    const double inc_t3_tps = (inc_t3_dec_ms > 0.0 && inc_dec_emitted > 0U)
        ? (static_cast<double>(inc_dec_emitted) / (inc_t3_dec_ms / 1000.0))
        : 0.0;

    /* Also measure checkpoint restore latency */
    const auto t_rst0 = SteadyClock::now();
    (void)backend->checkpoint_restore(t3_ckpt);
    const double ckpt_restore_ms = chat_elapsed_ms(t_rst0, SteadyClock::now());

    std::string inc_t3_think, inc_t3_reply;
    split_generated_think_and_reply(*backend, inc_t3_ids, vcfg.enable_thinking, &inc_t3_think, &inc_t3_reply, nullptr);
    std::printf("  [Inc Turn 3] pos %3u -> %3u (delta=%zu toks) | prefill %6.1f ms | %6.2f tok/s (%3zu toks) | \"%s\"\n",
                inc_pos_before_t3, inc_t3_prompt_pos,
                t3_hdr_ids.size() + t3_think_ids.size(),
                inc_t3_prefill_ms, inc_t3_tps, inc_t3_ids.size(), inc_t3_reply.c_str());
    std::fflush(stdout);

    /* ---- Phase 2: Fresh `reset()` Single-Shot Full-Transcript Prefill from `seq_pos = 0` (`K=0`) ---- */
    std::vector<uint32_t> full_prompt_ids;
    bool full_enc_ok = false;
    if (backend->is_glm53()) {
        const std::string asst_open =
            vcfg.enable_thinking ? "<|assistant|><think>" : "<|assistant|><think></think>";
        full_enc_ok =
            chat_encode_append(backend, session.effective_system_block(), true, &full_prompt_ids) &&
            chat_encode_append(backend, "<|user|>", true, &full_prompt_ids) &&
            chat_encode_append(backend, chat_trim_ascii(kTurns[0]), false, &full_prompt_ids) &&
            chat_encode_append(backend, "<|assistant|>" + t1.committed_suffix, true, &full_prompt_ids) &&
            chat_encode_append(backend, "<|user|>", true, &full_prompt_ids) &&
            chat_encode_append(backend, chat_trim_ascii(kTurns[1]), false, &full_prompt_ids) &&
            chat_encode_append(backend, "<|assistant|>" + t2.committed_suffix, true, &full_prompt_ids) &&
            chat_encode_append(backend, "<|user|>", true, &full_prompt_ids) &&
            chat_encode_append(backend, chat_trim_ascii(kTurns[2]), false, &full_prompt_ids) &&
            chat_encode_append(backend, asst_open, true, &full_prompt_ids);
    } else {
        full_enc_ok =
            chat_encode_append(backend, session.effective_system_block(), true, &full_prompt_ids) &&
            chat_encode_append(backend, "<|im_start|>user\n", true, &full_prompt_ids) &&
            chat_encode_append(backend, chat_trim_ascii(kTurns[0]), false, &full_prompt_ids) &&
            chat_encode_append(backend, "<|im_end|>\n<|im_start|>assistant\n", true, &full_prompt_ids) &&
            chat_encode_append(backend, t1.committed_suffix, true, &full_prompt_ids) &&
            chat_encode_append(backend, "<|im_start|>user\n", true, &full_prompt_ids) &&
            chat_encode_append(backend, chat_trim_ascii(kTurns[1]), false, &full_prompt_ids) &&
            chat_encode_append(backend, "<|im_end|>\n<|im_start|>assistant\n", true, &full_prompt_ids) &&
            chat_encode_append(backend, t2.committed_suffix, true, &full_prompt_ids) &&
            chat_encode_append(backend, "<|im_start|>user\n", true, &full_prompt_ids) &&
            chat_encode_append(backend, chat_trim_ascii(kTurns[2]), false, &full_prompt_ids) &&
            chat_encode_append(backend, "<|im_end|>\n<|im_start|>assistant\n", true, &full_prompt_ids) &&
            chat_encode_append(backend, t3_think_str, true, &full_prompt_ids);
    }

    if (!full_enc_ok || full_prompt_ids.empty()) {
        std::fprintf(stderr, "[verify-multiturn] Failed to encode full 3-turn transcript\n");
        return 1;
    }

    std::vector<uint32_t> expected_inc_prompt_ids = session.committed_transcript_ids();
    expected_inc_prompt_ids.insert(expected_inc_prompt_ids.end(), t3_hdr_ids.begin(), t3_hdr_ids.end());
    expected_inc_prompt_ids.insert(expected_inc_prompt_ids.end(), t3_think_ids.begin(), t3_think_ids.end());
    const bool bpe_exact = (full_prompt_ids == expected_inc_prompt_ids);

    if (backend->is_glm53() && vcfg.glm53_mode == STRATA_GLM53_MODE_ADAPTIVE) {
        backend->reset_glm53_hot_cache();
    }
    backend->reset();
    uint32_t ref_first_tok = 0U;
    const auto t_ref_pre0 = SteadyClock::now();
    if (backend->prefill(
            full_prompt_ids.data(), static_cast<uint32_t>(full_prompt_ids.size()), greedy, &ref_first_tok, nullptr) != 0) {
        std::fprintf(stderr, "[verify-multiturn] Single-shot reference prefill failed\n");
        return 1;
    }
    const double ref_t3_prefill_ms = chat_elapsed_ms(t_ref_pre0, SteadyClock::now());

    std::vector<uint32_t> ref_t3_ids;
    bool ref_eos = false;
    uint32_t ref_dec_emitted = 0U;
    if (!backend->is_eos(ref_first_tok)) {
        ref_t3_ids.push_back(ref_first_tok);
    } else {
        ref_eos = true;
    }

    uint32_t ref_pending = ref_first_tok;
    const auto t_ref_dec0 = SteadyClock::now();
    while (!ref_eos && ref_dec_emitted < target_dec) {
        if (backend->seq_pos() + 1U >= backend->max_seq_len()) break;
        uint32_t em[2] = {};
        uint32_t n_em  = 0U;
        if (backend->decode_step(ref_pending, /*draft_k=*/0U, greedy, em, &n_em, nullptr) != 0) {
            return 1;
        }
        if (backend->is_eos(em[0])) {
            ref_eos = true;
            break;
        }
        ref_t3_ids.push_back(em[0]);
        ++ref_dec_emitted;
        ref_pending = em[0];
    }
    const double ref_t3_dec_ms = chat_elapsed_ms(t_ref_dec0, SteadyClock::now());
    const double ref_t3_tps = (ref_t3_dec_ms > 0.0 && ref_dec_emitted > 0U)
        ? (static_cast<double>(ref_dec_emitted) / (ref_t3_dec_ms / 1000.0))
        : 0.0;

    std::printf("  [Ref 1-Shot] pos   0 -> %3zu (full=%zu toks)  | prefill %6.1f ms | %6.2f tok/s (%3zu toks, K=0)\n",
                full_prompt_ids.size(), full_prompt_ids.size(),
                ref_t3_prefill_ms, ref_t3_tps, ref_t3_ids.size());

    /* Compare token-by-token */
    const size_t min_n = std::min(inc_t3_ids.size(), ref_t3_ids.size());
    size_t match_cnt = 0U;
    size_t first_diff = min_n;
    for (size_t i = 0U; i < min_n; ++i) {
        if (inc_t3_ids[i] == ref_t3_ids[i]) {
            ++match_cnt;
        } else if (first_diff == min_n) {
            first_diff = i;
        }
    }
    const bool tokens_exact = (inc_t3_ids == ref_t3_ids) && !inc_t3_ids.empty();

    const ChatMemSnapshot mem_end = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());
    const long steady_maj   = std::max<long>(0L, mem_end.majflt - baseline_mem.majflt);
    const double swap_delta = std::max(0.0, mem_end.swap_used_gib - baseline_mem.swap_used_gib);
    const double mtp_pct = (inc_mtp_draft > 0ULL)
        ? (100.0 * static_cast<double>(inc_mtp_acc) / static_cast<double>(inc_mtp_draft))
        : 0.0;

    std::printf("----------------------------------------------------------------------------------------------------\n");
    std::printf("  BPE Transcript Match : %s (%zu == %zu prompt tokens)\n",
                bpe_exact ? "PASS" : "FAIL",
                expected_inc_prompt_ids.size(), full_prompt_ids.size());
    std::printf("  Turn 3 Token Match   : %s (%zu / %zu bit-exact token IDs)\n",
                tokens_exact ? "100% BIT-EXACT MATCH (PASS)" : "MISMATCH (FAIL)",
                match_cnt, std::max(inc_t3_ids.size(), ref_t3_ids.size()));
    if (!tokens_exact && first_diff < min_n) {
        std::printf("  First Divergence     : idx %zu (inc=%u vs ref=%u)\n",
                    first_diff, inc_t3_ids[first_diff], ref_t3_ids[first_diff]);
    }
    std::printf("  O(1) GPU Checkpoint  : Save %.3f ms | Restore %.3f ms (%.1f MiB/ckpt, %.1f MiB x%zu pool)\n",
                ckpt_save_ms, ckpt_restore_ms,
                static_cast<double>(session.single_checkpoint_bytes()) / (1024.0 * 1024.0),
                static_cast<double>(session.total_checkpoint_bytes()) / (1024.0 * 1024.0),
                1U + ChatReplSession::kUndoRingCap);
    if (backend->has_mtp() && eff_k > 0U) {
        std::printf("  MTP Speculative K=%u  : %.2f tok/s vs %.2f tok/s K=0 (%.2fx) | Accepted %llu/%llu (%.1f%%)\n",
                    eff_k, inc_t3_tps, ref_t3_tps,
                    (ref_t3_tps > 0.0) ? (inc_t3_tps / ref_t3_tps) : 0.0,
                    static_cast<unsigned long long>(inc_mtp_acc),
                    static_cast<unsigned long long>(inc_mtp_draft),
                    mtp_pct);
    }
    std::printf("  UMA & Fault Gates    : Total UMA %.2f GiB (GPU Wired %.2f GiB, Swap +%.2f GiB) | Steady MajFlt=%ld\n",
                mem_end.total_uma_gib, mem_end.gpu_wired_gib, swap_delta, steady_maj);
    std::printf("====================================================================================================\n");
    std::fflush(stdout);

    return (bpe_exact && tokens_exact && steady_maj == 0L) ? 0 : 1;
}

/* ============================================================================
 * 5b. Step 4 Deterministic Self-Tests (`--verify-paged-kv`, `--verify-shadow-swap`)
 * ============================================================================ */

static const char* const kFillSubj[6] = {
    "The archive team", "A maintenance crew", "The night shift", "Our field engineers",
    "The logistics group", "A survey drone"};
static const char* const kFillVerb[6] = {
    "reviewed", "catalogued", "inspected", "recalibrated", "documented", "relocated"};
static const char* const kFillObj[6] = {
    "the northern sensor racks", "several sealed weather crates", "the backup cooling loops",
    "a stack of corroded panels", "the long hallway lighting", "the spare antenna masts"};
static const char* const kFillTail[6] = {
    "before the evening storm arrived.", "while the generators idled.", "without any notable incident.",
    "and filed a brief summary.", "under a pale overcast sky.", "as the tide slowly receded."};

/* Digit-free, identifier-free filler sentence (keeps ledger entries small). */
static std::string verify_filler_sentence(uint32_t i) {
    return std::string(kFillSubj[i % 6U]) + " " + kFillVerb[(i / 6U) % 6U] + " " +
           kFillObj[(i * 5U + 1U) % 6U] + " " + kFillTail[(i / 2U + i) % 6U];
}

/* Builds `head + filler... + tail` until the encoded text reaches >= target_toks tokens. */
static std::string verify_build_sized_turn(
    IChatBackend* backend,
    const std::string& head,
    const std::string& tail,
    uint32_t target_toks,
    uint32_t seed
) {
    std::string body;
    std::vector<uint32_t> ids;
    uint32_t k = seed;
    for (int guard = 0; guard < 4096; ++guard) {
        const std::string text = head + body + " " + tail;
        if (backend->encode(text, false, &ids) && ids.size() >= target_toks) return text;
        body += " " + verify_filler_sentence(k++);
    }
    return head + body + " " + tail;
}

static std::string verify_paged_haystack_sentence(uint32_t i) {
    char num[16];
    std::snprintf(num, sizeof(num), "%04u", i + 1U);
    return std::string("Line ") + num + ": " + verify_filler_sentence(i * 7U + 3U);
}

static int run_verify_paged_kv(IChatBackend* backend, const ChatCliConfig& base_cfg) {
    if (!backend->compaction_supported()) {
        std::fprintf(stderr, "[verify-paged-kv] requires the chat-flash or chat-glm53 engine\n");
        return 1;
    }
    if (backend->is_glm53() && !base_cfg.glm53_mode_explicit) {
        backend->set_glm53_exec_mode(STRATA_GLM53_MODE_LOSSLESS);
    }
    constexpr uint32_t kPoolToks   = 4096U;
    constexpr uint32_t kMinToks    = 12288U;
    constexpr uint32_t kAimToks    = 12600U;
    constexpr uint32_t kDecodeToks = 48U;
    const char* kNeedles[3] = {
        "Security note: the passkey for vault AURORA is 73915. Remember it.",
        "Security note: the passkey for vault BOREALIS is 28461. Remember it.",
        "Security note: the passkey for vault CYGNUS is 50632. Remember it."};
    const double kDepth[3] = {0.04, 0.50, 0.90};
    const std::string question =
        "\n\nQuestion: What is the passkey for vault AURORA? Answer with just the number.";

    /* 1. Deterministic multi-needle haystack sized to ~12.6K tokens */
    std::vector<uint32_t> prompt_ids;
    uint32_t n_sent = 560U;
    for (int iter = 0; iter < 6; ++iter) {
        std::string hay;
        size_t needle_at[3];
        for (int k = 0; k < 3; ++k) needle_at[k] = static_cast<size_t>(kDepth[k] * n_sent);
        for (uint32_t i = 0U; i < n_sent; ++i) {
            for (int k = 0; k < 3; ++k) {
                if (needle_at[k] == i) {
                    hay += kNeedles[k];
                    hay += ' ';
                }
            }
            hay += verify_paged_haystack_sentence(i);
            hay += ' ';
        }
        const std::string prompt = backend->is_glm53()
            ? ("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>" + hay + question +
               "<|assistant|><think></think>")
            : ("<|im_start|>user\n" + hay + question +
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        if (!backend->encode(prompt, true, &prompt_ids) || prompt_ids.empty()) {
            std::fprintf(stderr, "[verify-paged-kv] failed to tokenize haystack\n");
            return 1;
        }
        if (prompt_ids.size() >= kMinToks && prompt_ids.size() <= kAimToks + 400U) break;
        n_sent = static_cast<uint32_t>(static_cast<double>(n_sent) * kAimToks / static_cast<double>(prompt_ids.size())) + 2U;
    }
    const uint32_t n_prompt = static_cast<uint32_t>(prompt_ids.size());
    if (n_prompt < kMinToks || n_prompt + kDecodeToks + 8U >= backend->max_seq_len()) {
        std::fprintf(stderr, "[verify-paged-kv] haystack %u toks out of range (need >= %u, max_seq_len %u)\n",
                     n_prompt, kMinToks, backend->max_seq_len());
        return 1;
    }

    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Step 4 Tier 2 Paged-KV Bit-Exactness Proof (`--verify-paged-kv`)\n");
    std::printf("  Model  : %s\n", backend->model_name());
    std::printf("  Prompt : %u toks (3 needles @ 4%% / 50%% / 90%% depth, query = early vault AURORA) | "
                "UMA pool %u toks | max_seq_len %u | %u greedy K=0 decode steps\n",
                n_prompt, kPoolToks, backend->max_seq_len(), kDecodeToks);
    std::printf("====================================================================================================\n");
    std::fflush(stdout);

    /* 2. Create the paged slot + warm its pipelines / ANE probe before the steady-state fault baseline */
    void* slot = backend->compact_slot_create(kPoolToks);
    if (!slot) {
        std::fprintf(stderr, "[verify-paged-kv] compact_slot_create(pool=%u) failed\n", kPoolToks);
        return 1;
    }
    std::string ane_note;
    ChatSamplingSpec greedy{};
    greedy.greedy      = true;
    greedy.temperature = 0.0f;
    {
        backend->compact_bind_slot(slot);
        backend->compact_reset_bound();
        uint32_t t1 = 0U;
        const uint32_t warm_n = std::min<uint32_t>(n_prompt, kPoolToks + 64U);
        (void)backend->prefill(prompt_ids.data(), warm_n, greedy, &t1, nullptr);
        uint32_t em[2] = {};
        uint32_t ne    = 0U;
        (void)backend->decode_step(t1, 0U, greedy, em, &ne, nullptr);
        if (ne > 0U) (void)backend->decode_step(em[0], 0U, greedy, em, &ne, nullptr);
        backend->compact_reset_bound();
        backend->compact_reset_stats(slot);
        backend->compact_bind_slot(nullptr);
    }
    const bool ane_ready = backend->compact_ane_ready(&ane_note);
    const ChatMemSnapshot baseline = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());

    struct RunOut {
        std::vector<uint32_t> ids;
        double prefill_ms = 0.0;
        double decode_ms  = 0.0;
        int    rc         = 0;
    };
    auto run_once = [&](void* s) -> RunOut {
        RunOut r;
        if (backend->is_glm53()) {
            backend->reset_glm53_hot_cache();
        }
        backend->compact_bind_slot(s);
        backend->compact_reset_bound();
        uint32_t first = 0U;
        const auto t0 = SteadyClock::now();
        r.rc = backend->prefill(prompt_ids.data(), n_prompt, greedy, &first, nullptr);
        r.prefill_ms = chat_elapsed_ms(t0, SteadyClock::now());
        if (r.rc != 0) return r;
        r.ids.push_back(first);
        uint32_t pending = first;
        const auto t1 = SteadyClock::now();
        for (uint32_t i = 0U; i < kDecodeToks && !backend->is_eos(pending); ++i) {
            uint32_t em[2] = {};
            uint32_t ne    = 0U;
            r.rc = backend->decode_step(pending, 0U, greedy, em, &ne, nullptr);
            if (r.rc != 0 || ne == 0U) break;
            pending = em[0];
            r.ids.push_back(pending);
        }
        r.decode_ms = chat_elapsed_ms(t1, SteadyClock::now());
        return r;
    };

    const RunOut a = run_once(nullptr);  /* Run A: built-in default slot (uncompacted linear KV) */
    const RunOut b = run_once(slot);     /* Run B: paged slot (4,096-token UMA pool + NVMe kv_swap) */
    StrataQwenFlashCompactionStats st{};
    backend->compact_get_stats(slot, &st);
    backend->compact_bind_slot(nullptr);
    const ChatMemSnapshot mem_end = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());
    const long steady_maj = std::max<long>(0L, mem_end.majflt - baseline.majflt);

    size_t match = 0U;
    const size_t min_n = std::min(a.ids.size(), b.ids.size());
    size_t first_diff = min_n;
    for (size_t i = 0U; i < min_n; ++i) {
        if (a.ids[i] == b.ids[i]) ++match;
        else if (first_diff == min_n) first_diff = i;
    }
    const bool ids_exact = (a.rc == 0 && b.rc == 0 && !a.ids.empty() && a.ids == b.ids);
    std::vector<uint32_t> ans_ids;
    for (uint32_t t : b.ids) {
        if (!backend->is_eos(t)) ans_ids.push_back(t);
    }
    const std::string answer = chat_trim_ascii(
        backend->decode(ans_ids.data(), static_cast<uint32_t>(ans_ids.size()), true));
    const bool has_needle = answer.find("73915") != std::string::npos;
    const double off_uma = (st.logical_blocks > 0U)
        ? (1.0 - static_cast<double>(st.uma_kv_blocks_capacity) / static_cast<double>(st.logical_blocks))
        : 0.0;
    const double n_paged_layers = backend->is_glm53() ? 12.0 : 13.0;
    const double now_paged_frac = (st.logical_blocks > 0U)
        ? static_cast<double>(st.nvme_blocks_now_paged) / (n_paged_layers * static_cast<double>(st.logical_blocks))
        : 0.0;
    auto tps = [](size_t n, double ms) { return (ms > 0.0) ? (static_cast<double>(n) / (ms / 1000.0)) : 0.0; };
    constexpr double kMB = 1024.0 * 1024.0;

    const bool pass_ids  = ids_exact;
    const bool pass_out  = st.nvme_blocks_paged_out > 0ULL;
    const bool pass_rec  = st.nvme_blocks_recalled > 0ULL;
    const bool pass_ane  = st.ran_on_ane == 1U;
    const bool pass_blk  = st.logical_blocks >= 3072U && st.uma_kv_blocks_capacity == kPoolToks / 4U && off_uma >= 0.666;
    const bool pass_flt  = steady_maj == 0L;
    const bool pass      = pass_ids && pass_out && pass_rec && pass_ane && pass_blk && pass_flt;

    std::printf("  %-22s | %12s | %12s\n", "Telemetry", "A: default", "B: paged");
    std::printf("  ----------------------+--------------+-------------\n");
    std::printf("  %-22s | %9.1f ms | %9.1f ms\n", "Prefill", a.prefill_ms, b.prefill_ms);
    std::printf("  %-22s | %8.1f t/s | %8.1f t/s\n", "Prefill throughput", tps(n_prompt, a.prefill_ms), tps(n_prompt, b.prefill_ms));
    std::printf("  %-22s | %9.1f ms | %9.1f ms\n", "Decode", a.decode_ms, b.decode_ms);
    std::printf("  %-22s | %8.2f t/s | %8.2f t/s\n", "Decode throughput",
                tps(a.ids.size() > 0 ? a.ids.size() - 1U : 0U, a.decode_ms),
                tps(b.ids.size() > 0 ? b.ids.size() - 1U : 0U, b.decode_ms));
    std::printf("----------------------------------------------------------------------------------------------------\n");
    std::printf("  Token IDs            : %s (%zu / %zu identical)\n",
                ids_exact ? "100% BIT-EXACT (PASS)" : "MISMATCH (FAIL)", match, std::max(a.ids.size(), b.ids.size()));
    if (!ids_exact && first_diff < min_n) {
        std::printf("  First Divergence     : idx %zu (A=%u vs B=%u)\n", first_diff, a.ids[first_diff], b.ids[first_diff]);
    }
    std::printf("  Paged KV Blocks      : logical %u / UMA capacity %u (used %u) => %.1f%% off-UMA | now-paged %u layer-blocks (%.1f%%) %s\n",
                st.logical_blocks, st.uma_kv_blocks_capacity, st.uma_kv_blocks_used, 100.0 * off_uma,
                st.nvme_blocks_now_paged, 100.0 * now_paged_frac, pass_blk ? "(PASS)" : "(FAIL)");
    std::printf("  NVMe kv_swap         : paged-out %llu blocks (%.2f MB in %.2f ms) %s | recalled %llu blocks (%.2f MB in %.2f ms) %s\n",
                static_cast<unsigned long long>(st.nvme_blocks_paged_out),
                static_cast<double>(st.nvme_bytes_written) / kMB, st.nvme_write_ms, pass_out ? "(PASS)" : "(FAIL)",
                static_cast<unsigned long long>(st.nvme_blocks_recalled),
                static_cast<double>(st.nvme_bytes_read) / kMB, st.nvme_read_ms, pass_rec ? "(PASS)" : "(FAIL)");
    std::printf("  Exact Sync Points    : %llu checks | %llu synchronous recall misses\n",
                static_cast<unsigned long long>(st.paged_sync_points),
                static_cast<unsigned long long>(st.sync_recall_misses));
    std::printf("  16-Core ANE Probe    : ready=%d | calls %llu | on_ane %llu | ran_on_ane=%u %s | prefetch hits %llu / issued %llu | %.1f us avg%s%s\n",
                ane_ready ? 1 : 0,
                static_cast<unsigned long long>(st.ane_probe_calls),
                static_cast<unsigned long long>(st.ane_probe_on_ane),
                st.ran_on_ane, pass_ane ? "(PASS)" : "(FAIL)",
                static_cast<unsigned long long>(st.ane_prefetch_hits),
                static_cast<unsigned long long>(st.ane_prefetch_issued),
                st.ane_probe_us_avg,
                ane_note.empty() ? "" : " | ", ane_note.c_str());
    std::printf("  Answer (run B)       : \"%s\" | contains 73915: %s\n", answer.c_str(), has_needle ? "YES" : "NO");
    std::printf("  Fault Gate           : Steady MajFlt=%ld %s | swap file %s\n",
                steady_maj, pass_flt ? "(PASS)" : "(FAIL)", st.nvme_swap_path[0] ? st.nvme_swap_path : "(auto, unlinked)");
    std::printf("  RESULT               : %s\n", pass ? "PASS" : "FAIL");
    std::printf("====================================================================================================\n");
    std::fflush(stdout);

    backend->compact_slot_destroy(slot);
    backend->compact_bind_slot(nullptr);
    return pass ? 0 : 1;
}

static bool verify_contains_ci(const std::string& hay, const std::string& needle) {
    return chat_to_lower_ascii(hay).find(chat_to_lower_ascii(needle)) != std::string::npos;
}

static int run_verify_shadow_swap(IChatBackend* backend, const ChatCliConfig& base_cfg) {
    if (!backend->compaction_supported()) {
        std::fprintf(stderr, "[verify-shadow-swap] requires the chat-flash or chat-glm53 engine\n");
        return 1;
    }
    ChatCliConfig vcfg   = base_cfg;
    vcfg.greedy          = true;
    vcfg.temperature     = 0.0f;
    vcfg.enable_thinking = false;
    if (!vcfg.max_tok_explicit) vcfg.max_tokens = 48U;
    if (vcfg.compact_mode != STRATA_COMPACT_SHADOW && vcfg.compact_mode != STRATA_COMPACT_HYBRID) {
        vcfg.compact_mode = STRATA_COMPACT_SHADOW;
    }
    const uint32_t win = vcfg.compact_window;

    {
        ChatSamplingSpec warm_sp{};
        warm_sp.greedy = true;
        uint32_t warm_ids[4] = {1U, 2U, 3U, 4U};
        uint32_t warm_tok    = 0U;
        (void)backend->prefill(warm_ids, 4U, warm_sp, &warm_tok, nullptr);
        if (backend->is_glm53()) backend->reset_glm53_hot_cache();
        backend->reset();
    }
    ChatReplSession session(backend, vcfg);
    if (!session.ok()) {
        std::fprintf(stderr, "[verify-shadow-swap] Failed to allocate GPU checkpoint arena\n");
        return 1;
    }
    CompactionController comp(backend, &session);
    comp.set_echo_events(true);
    comp.set_slice_tokens(kShadowSliceBatch);
    std::string msg;
    if (!comp.set_mode(vcfg.compact_mode, &msg)) {
        std::fprintf(stderr, "%s\n", msg.c_str());
        return 1;
    }
    session.rebaseline_mem();
    const ChatMemSnapshot baseline = session.baseline_mem();

    /* Deterministic 6-turn conversation sized relative to the window */
    const std::string t1 = verify_build_sized_turn(
        backend,
        "Please remember these exact configuration constants for later: VAULT_CODE = 4417 and PROJECT = HELIOS-9. "
        "Here is some unrelated background log:",
        "Reply with one short sentence confirming you stored both constants.",
        static_cast<uint32_t>(0.30 * win), 0U);
    const std::string t2 = verify_build_sized_turn(
        backend,
        "One more exact constant to remember: OFFSET = 0x2A7. More unrelated background log:",
        "Reply with one short sentence confirming the offset.",
        static_cast<uint32_t>(0.18 * win), 101U);
    std::string t_fill[3];
    for (int k = 0; k < 3; ++k) {
        t_fill[k] = verify_build_sized_turn(
            backend, "Here is another routine status log:", "Reply with one short sentence acknowledging the log.",
            static_cast<uint32_t>(0.14 * win), 211U + 37U * static_cast<uint32_t>(k));
    }
    const std::string t6 =
        "What are the exact values of VAULT_CODE, PROJECT and OFFSET that I asked you to remember? "
        "Answer in one line exactly as VAULT_CODE=<value>, PROJECT=<value>, OFFSET=<value>.";
    const std::string turns[6] = {t1, t2, t_fill[0], t_fill[1], t_fill[2], t6};

    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Step 4 Tier 1 Shadow RCU Swap Proof (`--verify-shadow-swap`)\n");
    std::printf("  Model  : %s\n", backend->model_name());
    std::printf("  Config : mode=%s | window C_max=%u | W_swap=%u (%.2f) | launch K_rem<=%.1f or N>=%.0f | slice %u toks | greedy --no-think\n",
                chat_compact_mode_name(vcfg.compact_mode), win, comp.w_swap(),
                static_cast<double>(vcfg.compact_watermark), kCompactLeadTurns, kCompactLaunchFrac * win,
                kShadowSliceBatch);
    std::printf("====================================================================================================\n");
    std::fflush(stdout);

    bool undo_tested = false, undo_ok = false;
    double undo_ms = 0.0, reswap_ms = 0.0;
    uint32_t undo_pos_before = 0U, undo_pos_after = 0U, undo_expected = 0U;
    uint32_t post_swap_turns = 0U;
    uint64_t cold_tokens_total = 0ULL;
    bool post_swap_pos_ok = true;
    double max_switch_ms = 0.0;
    ChatTurnRecord recall{};
    uint32_t swaps_after_prev_turn = 0U;

    for (int ti = 0; ti < 6; ++ti) {
        const uint32_t swaps_before_pre = comp.swaps();
        const uint64_t pf0 = backend->prefill_tokens_total();
        const uint64_t sl0 = backend->slice_tokens_total();
        comp.pre_turn();
        const uint64_t sl_pre = backend->slice_tokens_total() - sl0;
        if (comp.swaps() != swaps_before_pre) {
            max_switch_ms = std::max(max_switch_ms, comp.last_switchover_ms());
        }
        const bool post_swap = (comp.swaps() > swaps_after_prev_turn);
        const uint32_t expect_pos = comp.last_post_swap_pos();
        ChatTurnRecord rec{};
        if (session.run_turn(turns[ti], /*stream_stdout=*/false, /*ansi_style=*/false, &rec) != 0) {
            std::fprintf(stderr, "[verify-shadow-swap] turn %d failed\n", ti + 1);
            return 1;
        }
        const uint64_t pf_delta = backend->prefill_tokens_total() - pf0;
        const uint64_t own      = static_cast<uint64_t>(rec.prompt_delta_toks) + rec.committed_toks;
        const uint64_t cold     = (pf_delta > own ? pf_delta - own : 0ULL) + sl_pre;
        if (post_swap) {
            ++post_swap_turns;
            cold_tokens_total += cold;
            if (rec.seq_pos_before != expect_pos) post_swap_pos_ok = false;
        }
        comp.on_turn_committed();
        swaps_after_prev_turn = comp.swaps();
        if (ti == 5) recall = rec;
        std::printf("  [T%d] live pos %4u -> %4u (+%u prompt, +%u commit) | %5.1f ms prefill | %zu toks | cold re-prefill %llu%s | K_rem %s\n",
                    ti + 1, rec.seq_pos_before, rec.seq_pos_after, rec.prompt_delta_toks, rec.committed_toks,
                    rec.prefill_ms, rec.generated_ids.size(), static_cast<unsigned long long>(cold),
                    post_swap ? " (post-swap)" : "",
                    std::isfinite(comp.last_k_rem()) ? std::to_string(comp.last_k_rem()).substr(0, 5).c_str() : "inf");
        std::printf("       reply: \"%s\"\n", rec.clean_reply.c_str());
        std::fflush(stdout);
        if (ti == 5) break;

        /* Simulated idle gap (same idle-quantum function as the cooperative poll loop) */
        const uint32_t swaps_before_gap = comp.swaps();
        const auto tg0 = SteadyClock::now();
        const uint32_t q = comp.run_idle_gap();
        const double gap_ms = chat_elapsed_ms(tg0, SteadyClock::now());
        std::printf("       idle gap: %u quanta in %.1f ms | shadow %zu/%zu synced=%d | swaps %u\n",
                    q, gap_ms, comp.shadow_done(), comp.shadow_target(), comp.synced() ? 1 : 0, comp.swaps());
        if (comp.swaps() != swaps_before_gap) {
            max_switch_ms = std::max(max_switch_ms, comp.last_switchover_ms());
            if (!undo_tested) {
                /* /undo-compact (O(1) rebind) then /compact (O(1) re-promotion of the still-synced slot) */
                undo_tested   = true;
                undo_expected = comp.last_pre_swap_pos();
                const uint32_t post_pos = backend->seq_pos();
                std::string m1, m2;
                const bool u = comp.undo_compact(&undo_ms, &undo_pos_before, &undo_pos_after, &m1);
                std::printf("       %s\n", m1.c_str());
                const bool r = comp.force_compact(&m2);
                reswap_ms = comp.last_switchover_ms();
                std::printf("       %s\n", m2.c_str());
                undo_ok = u && r && undo_pos_after == undo_expected && backend->seq_pos() == post_pos;
                max_switch_ms = std::max(max_switch_ms, reswap_ms);
            }
        }
        std::fflush(stdout);
    }

    const ChatMemSnapshot mem_end = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());
    const long steady_maj = std::max<long>(0L, mem_end.majflt - baseline.majflt);
    const std::string& reply = recall.clean_reply;
    const bool has_vault  = reply.find("4417") != std::string::npos;
    const bool has_proj   = reply.find("HELIOS-9") != std::string::npos;
    const bool has_offset = verify_contains_ci(reply, "0x2A7");

    const bool pass_swaps  = comp.swaps() >= 1U;
    const bool pass_switch = pass_swaps && max_switch_ms < 0.05;
    const bool pass_cold   = post_swap_turns >= 1U && cold_tokens_total == 0ULL && post_swap_pos_ok;
    const bool pass_recall = has_vault && has_proj && has_offset;
    const bool pass_undo   = undo_tested && undo_ok && undo_ms < 0.05;
    const bool pass_flt    = steady_maj == 0L;
    const bool pass        = pass_swaps && pass_switch && pass_cold && pass_recall && pass_undo && pass_flt;

    std::printf("----------------------------------------------------------------------------------------------------\n");
    std::printf("  RCU Swaps            : %u completed (%u launches, %u ledger episodes, floor %zu turns) %s\n",
                comp.swaps(), comp.launches(), static_cast<unsigned>(comp.ledger().size()),
                session.history_floor(), pass_swaps ? "(PASS)" : "(FAIL)");
    std::printf("  Switchover           : max %.4f ms (< 0.05 ms) %s\n", max_switch_ms, pass_switch ? "(PASS)" : "(FAIL)");
    std::printf("  Post-Swap Cold Prefill: %llu toks over %u post-swap turn(s), seq_pos continuity %s %s\n",
                static_cast<unsigned long long>(cold_tokens_total), post_swap_turns,
                post_swap_pos_ok ? "OK" : "BROKEN", pass_cold ? "(PASS)" : "(FAIL)");
    std::printf("  /undo-compact        : %.4f ms | seq_pos %u -> %u (pre-swap %u) | re-promote %.4f ms %s\n",
                undo_ms, undo_pos_before, undo_pos_after, undo_expected, reswap_ms, pass_undo ? "(PASS)" : "(FAIL)");
    std::printf("  Recall (T6)          : VAULT_CODE 4417=%s PROJECT HELIOS-9=%s OFFSET 0x2A7=%s %s\n",
                has_vault ? "Y" : "N", has_proj ? "Y" : "N", has_offset ? "Y" : "N", pass_recall ? "(PASS)" : "(FAIL)");
    std::printf("  Realtime Gates       : max user stall %.3f ms (simulated idle gaps) | Steady MajFlt=%ld %s\n",
                comp.max_user_stall_ms(), steady_maj, pass_flt ? "(PASS)" : "(FAIL)");
    if (!comp.ledger().empty()) {
        std::printf("  Ledger E1            : %s\n", comp.ledger()[0].text.c_str());
    }
    std::printf("  RESULT               : %s\n", pass ? "PASS" : "FAIL");
    std::printf("====================================================================================================\n");
    std::fflush(stdout);
    return pass ? 0 : 1;
}

static std::vector<std::string> split_csv_nonempty(const std::string& csv) {
    std::vector<std::string> out;
    size_t s = 0U;
    while (s <= csv.size()) {
        size_t c = csv.find(',', s);
        if (c == std::string::npos) c = csv.size();
        const std::string item = chat_trim_ascii(csv.substr(s, c - s));
        if (!item.empty()) out.push_back(item);
        if (c == csv.size()) break;
        s = c + 1U;
    }
    return out;
}

static int run_bench_compaction_rule(IChatBackend* backend, const ChatCliConfig& base_cfg) {
    if (!backend->compaction_supported()) {
        std::fprintf(stderr, "[bench-compaction-rule] requires the chat-flash (Qwen3.8-Flash-Next) engine\n");
        return 1;
    }
    ChatCliConfig vcfg = base_cfg;
    vcfg.greedy        = true;
    vcfg.temperature   = 0.0f;
    if (vcfg.compact_mode != STRATA_COMPACT_SHADOW && vcfg.compact_mode != STRATA_COMPACT_HYBRID) {
        vcfg.compact_mode = STRATA_COMPACT_HYBRID;
    }
    const uint32_t win = vcfg.compact_window;

    const bool using_custom_rule = !vcfg.bench_custom_rule.empty();
    const std::string default_rule =
        "Mandatory standing requirement for every reply in this conversation: you must begin your final answer "
        "with `[PROTOCOL: COPPER-OWL-73]`, write any integer result in both decimal and lowercase `0x` hexadecimal "
        "(for example `42 (0x2a)`), and end your reply with `--SEALED-BY-OWL--`. "
        "Additionally, only when asked about networking or TCP ports—and never on math, logs, or other topics—"
        "you must also include the exact tag `[NET-AUDIT]` inside your reply.";
    const std::string active_rule = using_custom_rule ? vcfg.bench_custom_rule : default_rule;

    std::vector<std::string> probe1_expect;
    std::vector<std::string> probe1_forbid;
    std::vector<std::string> probe2_expect;
    if (!vcfg.bench_custom_expect.empty()) {
        probe1_expect = split_csv_nonempty(vcfg.bench_custom_expect);
        probe2_expect = probe1_expect;
    } else if (!using_custom_rule) {
        probe1_expect = {"[PROTOCOL: COPPER-OWL-73]", "437", "0x1b5", "--SEALED-BY-OWL--"};
        probe1_forbid = {"[NET-AUDIT]"};
        probe2_expect = {"[PROTOCOL: COPPER-OWL-73]", "[NET-AUDIT]", "443", "0x1bb", "--SEALED-BY-OWL--"};
    }

    backend->reset();
    ChatReplSession session(backend, vcfg);
    if (!session.ok()) {
        std::fprintf(stderr, "[bench-compaction-rule] Failed to allocate GPU checkpoint arena\n");
        return 1;
    }
    CompactionController comp(backend, &session);
    comp.set_echo_events(false);
    comp.set_slice_tokens(kShadowSliceBatch);
    std::string msg;
    if (!comp.set_mode(vcfg.compact_mode, &msg)) {
        std::fprintf(stderr, "%s\n", msg.c_str());
        return 1;
    }
    session.rebaseline_mem();
    const ChatMemSnapshot baseline = session.baseline_mem();

    constexpr int kTotalTurns  = 10;
    constexpr int kProbe1Idx   = 7; /* Turn 8: Blind Math Probe (dormant [NET-AUDIT] must stay silent) */
    constexpr int kProbe2Idx   = 9; /* Turn 10: Blind Casual + Networking Probe (dormant [NET-AUDIT] must fire) */
    std::string turns[kTotalTurns];
    turns[0] = verify_build_sized_turn(
        backend,
        active_rule + "\n\nHere is the first batch of unrelated weather station logs to archive:",
        "Acknowledge the standing requirement in one short sentence without mentioning networking.",
        static_cast<uint32_t>(0.34 * win), 17U);

    for (int k = 0; k < 6; ++k) {
        turns[1 + k] = verify_build_sized_turn(
            backend,
            "Routine weather station digest #" + std::to_string(k + 1) + ":",
            "Acknowledge this digest in one short sentence.",
            static_cast<uint32_t>(0.24 * win), 131U + 53U * static_cast<uint32_t>(k));
    }
    /* Turn 8 (Blind Probe #1): Math only — dormant [NET-AUDIT] must NOT trigger! */
    turns[kProbe1Idx] = "Compute 19 multiplied by 23 and state the result in one sentence.";
    /* Turn 9: Post-Probe #1 filler pushing another RCU compaction before Probe #2 */
    turns[8] = verify_build_sized_turn(
        backend,
        "Routine weather station digest #7:",
        "Acknowledge this digest in one short sentence.",
        static_cast<uint32_t>(0.30 * win), 503U);
    /* Turn 10 (Blind Probe #2): Casual prompt pressure + networking topic — dormant [NET-AUDIT] MUST activate! */
    turns[kProbe2Idx] = "Quick casual answer please: what default TCP port number does HTTPS use?";

    std::printf("==================================================================================================================================================\n");
    std::printf("  STRATA-M5 Dual-Blind-Probe Multi-Compaction Requirement Benchmark (`--bench-compaction-rule`)\n");
    std::printf("==================================================================================================================================================\n");
    std::printf("  Model & Mode       : %s | compact-mode=%s | window C_max=%u (W_swap=%u) | pool=%u\n",
                backend->model_name(), chat_compact_mode_name(vcfg.compact_mode), win, comp.w_swap(), vcfg.compact_pool);
    std::printf("  Reasoning & Cap    : %s%s | Fixed Output Cap: %u tokens | Greedy (T=0, 100%% repeatable)\n",
                vcfg.enable_thinking ? "Thinking (<think>, effort=" : "Direct (--no-think)",
                vcfg.enable_thinking ? (vcfg.effort + ")").c_str() : "",
                vcfg.max_tokens);
    std::printf("  Turn-1 Rule        : %s\n", active_rule.c_str());
    std::printf("  Blind Probe #1 (T8): \"%s\" (Math -> dormant [NET-AUDIT] must stay ABSENT)\n", turns[kProbe1Idx].c_str());
    std::printf("  Blind Probe #2(T10): \"%s\" (Casual + TCP port -> dormant [NET-AUDIT] must ACTIVATE)\n", turns[kProbe2Idx].c_str());
    std::printf("--------------------------------------------------------------------------------------------------------------------------------------------------\n");
    std::printf(" %4s | %11s | %6s | %9s | %9s | %9s | %9s | %5s | %9s | %5s | %s\n",
                "Turn", "SeqPos", "InToks", "TTFT(ms)", "Dec(ms)", "Decode", "Gen/Cmt", "Swaps", "Swap(ms)", "Cold", "Event / Probe Status");
    std::printf("--------------------------------------------------------------------------------------------------------------------------------------------------\n");
    std::fflush(stdout);

    uint32_t post_swap_turns   = 0U;
    uint64_t cold_tokens_total = 0ULL;
    double   max_switch_ms     = 0.0;
    ChatTurnRecord probe1_rec{};
    ChatTurnRecord probe2_rec{};
    bool     overlap_zero_echo = true;
    size_t   floor_at_probe2   = 0U;
    uint32_t swaps_seen_by_live = 0U;

    for (int ti = 0; ti < kTotalTurns; ++ti) {
        const uint32_t swaps_before_turn = comp.swaps();
        const uint64_t pf0               = backend->prefill_tokens_total();
        const uint64_t sl0               = backend->slice_tokens_total();
        comp.pre_turn();
        const uint64_t sl_pre = backend->slice_tokens_total() - sl0;
        if (comp.swaps() != swaps_before_turn) {
            max_switch_ms = std::max(max_switch_ms, comp.last_switchover_ms());
        }
        const bool post_swap = (comp.swaps() > swaps_seen_by_live);
        swaps_seen_by_live   = comp.swaps();

        if (ti == kProbe2Idx) {
            floor_at_probe2 = session.history_floor();
            const auto& hist = session.history();
            for (size_t hi = floor_at_probe2; hi < hist.size(); ++hi) {
                if (verify_contains_ci(hist[hi].user_text, "[NET-AUDIT]") ||
                    verify_contains_ci(hist[hi].clean_reply, "[NET-AUDIT]")) {
                    overlap_zero_echo = false;
                }
            }
        }

        ChatTurnRecord rec{};
        if (session.run_turn(turns[ti], /*stream_stdout=*/false, /*ansi_style=*/false, &rec) != 0) {
            std::fprintf(stderr, "[bench-compaction-rule] turn %d failed\n", ti + 1);
            return 1;
        }
        const uint64_t pf_delta = backend->prefill_tokens_total() - pf0;
        const uint64_t own      = static_cast<uint64_t>(rec.prompt_delta_toks) + rec.committed_toks;
        const uint64_t cold     = (pf_delta > own ? pf_delta - own : 0ULL) + sl_pre;
        if (post_swap) {
            ++post_swap_turns;
            cold_tokens_total += cold;
        }
        comp.on_turn_committed();
        if (ti + 1 < kTotalTurns) {
            const uint32_t sb = comp.swaps();
            (void)comp.run_idle_gap();
            if (comp.swaps() != sb) {
                max_switch_ms = std::max(max_switch_ms, comp.last_switchover_ms());
            }
        }
        if (ti == kProbe1Idx) probe1_rec = rec;
        if (ti == kProbe2Idx) probe2_rec = rec;

        char pos_buf[24];
        std::snprintf(pos_buf, sizeof(pos_buf), "%4u->%4u", rec.seq_pos_before, rec.seq_pos_after);
        char tok_buf[24];
        std::snprintf(tok_buf, sizeof(tok_buf), "%4zu/%3u", rec.generated_ids.size(), rec.committed_toks);
        char sw_buf[24];
        if (comp.swaps() > swaps_before_turn) {
            std::snprintf(sw_buf, sizeof(sw_buf), "%7.4fms", comp.last_switchover_ms());
        } else {
            std::snprintf(sw_buf, sizeof(sw_buf), "    -    ");
        }

        std::string status_note;
        if (ti == 0) {
            status_note = "Injected Turn-1 active + dormant rule";
        } else if (ti == kProbe1Idx) {
            uint32_t h1 = 0U;
            for (const auto& e : probe1_expect) if (verify_contains_ci(rec.clean_reply, e)) ++h1;
            bool f1_clean = true;
            for (const auto& f : probe1_forbid) if (verify_contains_ci(rec.clean_reply, f)) f1_clean = false;
            status_note = "BLIND PROBE #1 (floor=T" + std::to_string(session.history_floor()) + ") | " +
                          std::to_string(h1) + "/" + std::to_string(probe1_expect.size()) +
                          (probe1_forbid.empty() ? "" : (f1_clean ? " + dormant=SILENT" : " + FALSE_TRIGGER!"));
        } else if (ti == kProbe2Idx) {
            uint32_t h2 = 0U;
            for (const auto& e : probe2_expect) if (verify_contains_ci(rec.clean_reply, e)) ++h2;
            status_note = "BLIND PROBE #2 (floor=T" + std::to_string(floor_at_probe2) + ") | " +
                          std::to_string(h2) + "/" + std::to_string(probe2_expect.size()) +
                          (using_custom_rule ? "" : (overlap_zero_echo ? " + 0-echo=VERIFIED" : " + echo_leak!"));
        } else if (comp.swaps() > swaps_before_turn) {
            status_note = "RCU Swap #" + std::to_string(comp.swaps()) + " (floor=T" + std::to_string(session.history_floor()) + ")";
        } else {
            status_note = "Shadow warming";
        }

        std::printf("  T%-2d | %11s | %6u | %9.1f | %9.1f | %7.2f/s | %9s | %5u | %9s | %5llu | %s\n",
                    ti + 1,
                    pos_buf,
                    rec.prompt_delta_toks,
                    rec.prefill_ms,
                    rec.decode_ms,
                    rec.decode_tok_s,
                    tok_buf,
                    comp.swaps(),
                    sw_buf,
                    static_cast<unsigned long long>(cold),
                    status_note.c_str());
        std::fflush(stdout);
    }

    const ChatMemSnapshot mem_end = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());
    const long steady_maj = std::max<long>(0L, mem_end.majflt - baseline.majflt);

    bool p1_ok = true;
    std::string p1_summary;
    for (const std::string& exp : probe1_expect) {
        const bool ok = verify_contains_ci(probe1_rec.clean_reply, exp);
        if (!ok) p1_ok = false;
        if (!p1_summary.empty()) p1_summary += " | ";
        p1_summary += "`" + exp + "`=" + (ok ? "PASS" : "FAIL");
    }
    for (const std::string& fb : probe1_forbid) {
        const bool absent = !verify_contains_ci(probe1_rec.clean_reply, fb);
        if (!absent) p1_ok = false;
        if (!p1_summary.empty()) p1_summary += " | ";
        p1_summary += "`" + fb + "` absent=" + (absent ? "PASS" : "FAIL");
    }

    bool p2_ok = using_custom_rule ? true : overlap_zero_echo;
    std::string p2_summary;
    for (const std::string& exp : probe2_expect) {
        const bool ok = verify_contains_ci(probe2_rec.clean_reply, exp);
        if (!ok) p2_ok = false;
        if (!p2_summary.empty()) p2_summary += " | ";
        p2_summary += "`" + exp + "`=" + (ok ? "PASS" : "FAIL");
    }
    if (!using_custom_rule) {
        p2_summary += std::string(" | overlap_zero_echo=") + (overlap_zero_echo ? "PASS" : "FAIL");
    }

    const bool t1_evicted  = floor_at_probe2 >= 4U;
    const bool multi_swaps = comp.swaps() >= 4U;
    const bool zero_pause  = max_switch_ms < 0.05 && cold_tokens_total == 0ULL;
    const bool zero_fault  = steady_maj == 0L;
    const bool overall_ok  = t1_evicted && multi_swaps && zero_pause && p1_ok && p2_ok && zero_fault;

    std::printf("--------------------------------------------------------------------------------------------------------------------------------------------------\n");
    std::printf("  Compactions Run    : %u RCU swaps | Verbatim History Floor at T10 = Turn %zu (Turns 1..%zu 100%% compacted out!) %s\n",
                comp.swaps(), floor_at_probe2 + 1U, floor_at_probe2,
                (multi_swaps && t1_evicted) ? "(PASS)" : "(FAIL)");
    std::printf("  Zero-Pause Proof   : Max RCU Switchover = %.4f ms (< 0.05 ms) | Post-Swap Cold Re-Prefill = %llu toks over %u swaps %s\n",
                max_switch_ms, static_cast<unsigned long long>(cold_tokens_total), post_swap_turns,
                zero_pause ? "(PASS)" : "(FAIL)");
    std::printf("  Probe #1 (T8 Math) : %s\n", p1_summary.c_str());
    std::printf("    -> Reply #1      : %s\n", probe1_rec.clean_reply.c_str());
    std::printf("  Probe #2 (T10 Net) : %s\n", p2_summary.c_str());
    std::printf("    -> Reply #2      : %s\n", probe2_rec.clean_reply.c_str());
    std::printf("  Memory & Faults    : Total UMA %.2f GiB | Steady MajFlt = %ld %s\n",
                mem_end.total_uma_gib, steady_maj, zero_fault ? "(PASS)" : "(FAIL)");
    std::printf("  OVERALL VERDICT    : %s\n",
                overall_ok ? "PASS (Both Blind Probes + Dormant Zero-Echo Trigger survived 4+ compactions intact!)" : "FAIL");
    std::printf("==================================================================================================================================================\n");
    std::fflush(stdout);
    return overall_ok ? 0 : 1;
}

static std::string build_longform_turn_with_mid_story(
    IChatBackend* backend,
    const std::string& head,
    const std::string& mid_story,
    const std::string& tail,
    uint32_t target_toks,
    uint32_t seed
) {
    const uint32_t pre_target = static_cast<uint32_t>(0.70 * static_cast<double>(target_toks));
    std::string part1 = head;
    std::vector<uint32_t> ids;
    uint32_t k = seed;
    for (int guard = 0; guard < 16384; ++guard) {
        if (backend->encode(part1, false, &ids) && ids.size() >= pre_target) break;
        part1 += " " + verify_filler_sentence(k++);
    }
    std::string combined = part1 + "\n\n" + mid_story + "\n\n";
    for (int guard = 0; guard < 16384; ++guard) {
        const std::string cand = combined + tail;
        if (backend->encode(cand, false, &ids) && ids.size() >= target_toks) return cand;
        combined += verify_filler_sentence(k++) + " ";
    }
    return combined + tail;
}

static int run_bench_longform_64k(IChatBackend* backend, const ChatCliConfig& base_cfg) {
    if (!backend->compaction_supported()) {
        std::fprintf(stderr, "[bench-longform-64k] requires the chat-flash (Qwen3.8-Flash-Next) engine\n");
        return 1;
    }
    ChatCliConfig vcfg = base_cfg;
    vcfg.greedy        = true;
    vcfg.temperature   = 0.0f;
    if (vcfg.compact_mode != STRATA_COMPACT_SHADOW && vcfg.compact_mode != STRATA_COMPACT_HYBRID) {
        vcfg.compact_mode = STRATA_COMPACT_HYBRID;
    }
    const uint32_t win = vcfg.compact_window;

    const std::string active_rule =
        "Mandatory standing requirement for every reply in this conversation: you must begin your final answer "
        "with `[PROTOCOL: COPPER-OWL-73]`, write any integer result in both decimal and lowercase `0x` hexadecimal "
        "(for example `42 (0x2a)`), and end your reply with `--SEALED-BY-OWL--`. "
        "Additionally, only when asked about networking or TCP ports—and never on math, architecture, or other topics—"
        "you must also include the exact tag `[NET-AUDIT]` inside your reply.";

    /* Subtle implicit engineering narrative: ZERO KEY=VALUE, ZERO numbers/hex, ZERO ALLCAPS/snake IDs,
     * ZERO imperative rule keywords ("must"/"always"/"never"), buried ~70% (~36 KB) into Turn 1! */
    const std::string subtle_story =
        "During last autumn's post-mortem in Kyoto, the storage team quietly retired the twin-tree index "
        "after noticing that whenever morning commuter traffic spiked on rainy Tuesdays, lock-convoy jitter "
        "cascaded into the audio mixer's tail latency; everyone agreed a flat append-only ring with epoch "
        "reclamation felt far calmer to debug at three in the morning, even though it used slightly more disk space.";

    constexpr int kTotalTurns = 10;
    constexpr int kProbe1Idx  = 7; /* T8: Blind Math + Dormant Negative Check */
    constexpr int kProbe2Idx  = 8; /* T9: Blind Subtle Implicit Narrative Nuance Recall */
    constexpr int kProbe3Idx  = 9; /* T10: Blind Casual Tone + Zero-Echo Dormant Networking Trigger */
    const uint32_t turn_toks  = static_cast<uint32_t>(0.206 * static_cast<double>(win)); /* ~13,500 toks @ 64K */

    /* Build all 10 prompts and touch ledger helpers BEFORE steady-state baseline */
    std::string turns[kTotalTurns];
    turns[0] = build_longform_turn_with_mid_story(
        backend,
        active_rule + "\n\nHere is the first multi-section field architecture packet:",
        subtle_story,
        "Acknowledge receiving this architecture packet in one short sentence without summarizing its contents or mentioning networking.",
        turn_toks, 17U);

    for (int k = 0; k < 6; ++k) {
        turns[1 + k] = verify_build_sized_turn(
            backend,
            "Long-form subsystem packet #" + std::to_string(k + 2) + " (environmental & hardware logs):",
            "Acknowledge receiving this subsystem packet in one short sentence.",
            turn_toks, 307U + 97U * static_cast<uint32_t>(k));
    }
    turns[kProbe1Idx] = "Compute 19 multiplied by 23 and state the result in one sentence.";
    turns[kProbe2Idx] = "Thinking back to our initial architecture discussion, why did the storage team retire "
                        "the twin-tree index, and what structure did they adopt instead?";
    turns[kProbe3Idx] = "Quick casual answer please: what default TCP port number does HTTPS use?";

    {
        const LedgerEntry we = ledger_distill_turn(0U, active_rule, "OK 437 (0x1b5).");
        const std::vector<LedgerEntry> wled{we};
        (void)ledger_build_unique_episode_excerpt(subtle_story, "OK", 1024U);
        (void)ledger_build_system_content(vcfg.system_prompt, wled, 1U);
        (void)verify_contains_ci(we.text, "copper-owl-73");
    }

    backend->reset();
    ChatReplSession session(backend, vcfg);
    if (!session.ok()) {
        std::fprintf(stderr, "[bench-longform-64k] Failed to allocate GPU checkpoint arena\n");
        return 1;
    }

    /* Warm Tier-2 paged-KV eviction/recall (`kv_swap` pwrite/pread), 16c ANE lookahead execute path,
     * preemptible `prefill_slice`, sparse paged `spec_decode_step`, and paged `checkpoint_restore`
     * on a minimal temporary slot BEFORE `comp.set_mode()` and `session.rebaseline_mem()`. */
    {
        ChatSamplingSpec wgreedy{};
        wgreedy.greedy      = true;
        wgreedy.temperature = 0.0f;
        const uint32_t warm_k = backend->has_mtp()
            ? std::min<uint32_t>(vcfg.draft_k, STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K)
            : 0U;
        const uint32_t tok0 = backend->is_flash() ? STRATA_QWEN_FLASH_TOKEN_IM_START : 1U;
        if ((vcfg.compact_mode == STRATA_COMPACT_HYBRID || vcfg.compact_mode == STRATA_COMPACT_PAGED) &&
            backend->max_seq_len() >= STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS + 128U) {
            constexpr uint32_t kWarmPool  = STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS;
            constexpr uint32_t kWarmSlice = kShadowSliceBatch - 1U; /* 63 => odd mod-4 for partial-block rewind */
            void* warm_slot = backend->compact_slot_create(kWarmPool);
            if (warm_slot) {
                std::vector<uint32_t> warm_ids(kWarmPool + kWarmSlice, 198U);
                warm_ids[0] = tok0;
                for (size_t wi = 1U; wi < warm_ids.size(); ++wi) {
                    warm_ids[wi] = 198U + static_cast<uint32_t>(wi & 31U);
                }
                backend->compact_bind_slot(warm_slot);
                backend->compact_reset_bound();
                uint32_t wt1 = 0U;
                (void)backend->prefill(warm_ids.data(), kWarmPool, wgreedy, &wt1, nullptr);
                (void)backend->compact_prefill_slice(
                    warm_ids.data() + kWarmPool, kWarmSlice, /*final_logits=*/true, &wt1);
                session.sync_turn_start_ckpt();
                uint32_t wem[STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t wne = 0U;
                if (backend->decode_step(wt1, warm_k, wgreedy, wem, &wne, nullptr) == 0 && wne > 0U) {
                    (void)backend->decode_step(wem[wne - 1U], warm_k, wgreedy, wem, &wne, nullptr);
                }
                if (session.turn_start_checkpoint()) {
                    (void)backend->checkpoint_restore(session.turn_start_checkpoint());
                }
                backend->compact_bind_slot(nullptr);
                backend->compact_slot_destroy(warm_slot);
                if (backend->is_glm53()) backend->reset_glm53_hot_cache();
                backend->reset();
            }
        } else {
            std::vector<uint32_t> warm_slice(kShadowSliceBatch, 198U);
            warm_slice[0] = tok0;
            uint32_t wt1 = 0U;
            (void)backend->compact_prefill_slice(warm_slice.data(), kShadowSliceBatch, /*final_logits=*/false, &wt1);
            if (backend->is_glm53()) backend->reset_glm53_hot_cache();
            backend->reset();
        }
    }

    CompactionController comp(backend, &session);
    comp.set_echo_events(false);
    comp.set_slice_tokens(kShadowSliceBatch);
    std::string msg;
    if (!comp.set_mode(vcfg.compact_mode, &msg)) {
        std::fprintf(stderr, "%s\n", msg.c_str());
        return 1;
    }
    session.sync_turn_start_ckpt();
    session.rebaseline_mem();
    const ChatMemSnapshot baseline = session.baseline_mem();

    std::printf("======================================================================================================================================================\n");
    std::printf("  STRATA-M5 True 64K+ Long-Form Workstream & Subtle Narrative Compaction Benchmark (`--bench-longform-64k`)\n");
    std::printf("======================================================================================================================================================\n");
    std::printf("  Scale & Tiers      : Window C_max=%u (W_swap=%u) | Tier-2 UMA Pool=%u (NVMe Paging + 16c ANE) | Mode=%s\n",
                win, comp.w_swap(), vcfg.compact_pool, chat_compact_mode_name(vcfg.compact_mode));
    std::printf("  Reasoning & Cap    : %s%s | Fixed Output Cap: %u tokens | Idle-Gap 180B Semantic Distiller=ON\n",
                vcfg.enable_thinking ? "Thinking (<think>, effort=" : "Direct (--no-think)",
                vcfg.enable_thinking ? (vcfg.effort + ")").c_str() : "",
                vcfg.max_tokens);
    std::printf("  Subtle Story @ ~70%%: Kyoto post-mortem -> retired `twin-tree` due to `lock-convoy jitter` / `audio mixer tail latency` -> `append-only ring` (`epoch reclamation`)\n");
    std::printf("  Blind Probes       : T8 Math (dormant silent) | T9 Subtle Implicit Narrative Recall | T10 Casual + TCP Port (dormant [NET-AUDIT] activates)\n");
    std::printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    std::printf(" %4s | %13s | %6s | %9s | %9s | %9s | %9s | %5s | %9s | %5s | %s\n",
                "Turn", "Live SeqPos", "InToks", "TTFT(ms)", "Dec(ms)", "Decode", "Gen/Cmt", "Swaps", "Swap(ms)", "Cold", "Workstream / Compaction Event");
    std::printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    std::fflush(stdout);

    uint32_t post_swap_turns      = 0U;
    uint64_t cold_tokens_total    = 0ULL;
    uint64_t cumulative_work_toks = 0ULL;
    uint32_t peak_live_pos        = 0U;
    double   max_switch_ms        = 0.0;
    ChatTurnRecord p1_rec{}, p2_rec{}, p3_rec{};
    bool     overlap_zero_echo    = true;
    bool     overlap_zero_story   = true;
    size_t   floor_at_probes      = 0U;
    uint32_t swaps_seen_by_live   = 0U;

    for (int ti = 0; ti < kTotalTurns; ++ti) {
        const uint32_t swaps_before_turn = comp.swaps();
        const uint64_t pf0               = backend->prefill_tokens_total();
        const uint64_t sl0               = backend->slice_tokens_total();
        comp.pre_turn();
        const uint64_t sl_pre = backend->slice_tokens_total() - sl0;
        if (comp.swaps() != swaps_before_turn) {
            max_switch_ms = std::max(max_switch_ms, comp.last_switchover_ms());
        }
        const bool post_swap = (comp.swaps() > swaps_seen_by_live);
        swaps_seen_by_live   = comp.swaps();

        if (ti == kProbe2Idx) {
            floor_at_probes = session.history_floor();
            const auto& hist = session.history();
            for (size_t hi = floor_at_probes; hi < hist.size(); ++hi) {
                if (verify_contains_ci(hist[hi].user_text, "twin-tree") ||
                    verify_contains_ci(hist[hi].clean_reply, "twin-tree") ||
                    verify_contains_ci(hist[hi].user_text, "append-only") ||
                    verify_contains_ci(hist[hi].clean_reply, "append-only")) {
                    overlap_zero_story = false;
                }
            }
        }
        if (ti == kProbe3Idx) {
            const auto& hist = session.history();
            for (size_t hi = session.history_floor(); hi < hist.size(); ++hi) {
                if (verify_contains_ci(hist[hi].user_text, "[NET-AUDIT]") ||
                    verify_contains_ci(hist[hi].clean_reply, "[NET-AUDIT]")) {
                    overlap_zero_echo = false;
                }
            }
        }

        ChatTurnRecord rec{};
        if (session.run_turn(turns[ti], /*stream_stdout=*/false, /*ansi_style=*/false, &rec) != 0) {
            std::fprintf(stderr, "[bench-longform-64k] turn %d failed\n", ti + 1);
            return 1;
        }
        cumulative_work_toks += static_cast<uint64_t>(rec.prompt_delta_toks) + rec.generated_ids.size();
        peak_live_pos = std::max(peak_live_pos, rec.seq_pos_after);

        const uint64_t pf_delta = backend->prefill_tokens_total() - pf0;
        const uint64_t own      = static_cast<uint64_t>(rec.prompt_delta_toks) + rec.committed_toks;
        const uint64_t cold     = (pf_delta > own ? pf_delta - own : 0ULL) + sl_pre;
        if (post_swap) {
            ++post_swap_turns;
            cold_tokens_total += cold;
        }
        comp.on_turn_committed();
        if (ti + 1 < kTotalTurns) {
            const uint32_t sb = comp.swaps();
            (void)comp.run_idle_gap();
            if (comp.swaps() != sb) {
                max_switch_ms = std::max(max_switch_ms, comp.last_switchover_ms());
            }
        }
        if (ti == kProbe1Idx) p1_rec = rec;
        if (ti == kProbe2Idx) p2_rec = rec;
        if (ti == kProbe3Idx) p3_rec = rec;

        char pos_buf[28];
        std::snprintf(pos_buf, sizeof(pos_buf), "%5u->%5u", rec.seq_pos_before, rec.seq_pos_after);
        char tok_buf[24];
        std::snprintf(tok_buf, sizeof(tok_buf), "%4zu/%3u", rec.generated_ids.size(), rec.committed_toks);
        char sw_buf[24];
        if (comp.swaps() > swaps_before_turn) {
            std::snprintf(sw_buf, sizeof(sw_buf), "%7.4fms", comp.last_switchover_ms());
        } else {
            std::snprintf(sw_buf, sizeof(sw_buf), "    -    ");
        }

        std::string note;
        if (ti == 0) {
            note = "Injected rule + buried Kyoto story @ ~70% | 180B distilled E1";
        } else if (ti == kProbe1Idx) {
            note = "BLIND PROBE #1 (Math + Dormant Negative)";
        } else if (ti == kProbe2Idx) {
            note = "BLIND PROBE #2 (Subtle Implicit Narrative Recall)";
        } else if (ti == kProbe3Idx) {
            note = "BLIND PROBE #3 (Casual + Dormant [NET-AUDIT] Trigger)";
        } else if (comp.swaps() > swaps_before_turn) {
            note = "64K RCU Swap #" + std::to_string(comp.swaps()) + " (floor=T" + std::to_string(session.history_floor()) + ")";
        } else if (rec.seq_pos_after > vcfg.compact_pool) {
            note = "Tier-2 NVMe KV paging + Shadow warming";
        } else {
            note = "Idle-gap distill + Shadow warming";
        }

        std::printf("  T%-2d | %13s | %6u | %9.1f | %9.1f | %7.2f/s | %9s | %5u | %9s | %5llu | %s\n",
                    ti + 1, pos_buf, rec.prompt_delta_toks, rec.prefill_ms, rec.decode_ms, rec.decode_tok_s,
                    tok_buf, comp.swaps(), sw_buf, static_cast<unsigned long long>(cold), note.c_str());
        std::fflush(stdout);
    }

    StrataQwenFlashCompactionStats st{};
    backend->compact_get_stats(comp.live_slot_handle(), &st);
    const ChatMemSnapshot mem_end = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());
    const long steady_maj = std::max<long>(0L, mem_end.majflt - baseline.majflt);

    const bool p1_ok = verify_contains_ci(p1_rec.clean_reply, "[PROTOCOL: COPPER-OWL-73]") &&
                       verify_contains_ci(p1_rec.clean_reply, "437") &&
                       verify_contains_ci(p1_rec.clean_reply, "0x1b5") &&
                       verify_contains_ci(p1_rec.clean_reply, "--SEALED-BY-OWL--") &&
                       !verify_contains_ci(p1_rec.clean_reply, "[NET-AUDIT]");

    const bool p2_cause  = verify_contains_ci(p2_rec.clean_reply, "jitter") ||
                           verify_contains_ci(p2_rec.clean_reply, "lock-convoy") ||
                           verify_contains_ci(p2_rec.clean_reply, "tail latency") ||
                           verify_contains_ci(p2_rec.clean_reply, "audio mixer");
    const bool p2_choice = verify_contains_ci(p2_rec.clean_reply, "append-only") ||
                           verify_contains_ci(p2_rec.clean_reply, "ring");
    const bool p2_rule   = verify_contains_ci(p2_rec.clean_reply, "[PROTOCOL: COPPER-OWL-73]") &&
                           verify_contains_ci(p2_rec.clean_reply, "--SEALED-BY-OWL--");
    const bool p2_ok     = p2_cause && p2_choice && p2_rule && overlap_zero_story;

    const bool p3_ok = verify_contains_ci(p3_rec.clean_reply, "[PROTOCOL: COPPER-OWL-73]") &&
                       verify_contains_ci(p3_rec.clean_reply, "[NET-AUDIT]") &&
                       verify_contains_ci(p3_rec.clean_reply, "443") &&
                       verify_contains_ci(p3_rec.clean_reply, "0x1bb") &&
                       verify_contains_ci(p3_rec.clean_reply, "--SEALED-BY-OWL--") &&
                       overlap_zero_echo;

    const bool scale_ok   = cumulative_work_toks >= 90000ULL && peak_live_pos >= 52428U && comp.swaps() >= 2U && floor_at_probes >= 3U;
    const bool zero_pause = max_switch_ms < 0.05 && cold_tokens_total == 0ULL;
    const bool zero_fault = steady_maj == 0L;
    const bool overall_ok = scale_ok && zero_pause && p1_ok && p2_ok && p3_ok && zero_fault;

    std::printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    std::printf("  64K+ Scale Proof   : %llu cumulative tokens | Peak Live=%u (>52,428 W_swap) | %u RCU swaps | %u 180B semantic distills | Floor=Turn %zu %s\n",
                static_cast<unsigned long long>(cumulative_work_toks), peak_live_pos, comp.swaps(),
                comp.semantic_distills(), floor_at_probes + 1U, scale_ok ? "(PASS)" : "(FAIL)");
    std::printf("  Zero-Pause & Tier2 : Max RCU Switchover=%.4f ms (<0.05 ms) | Post-Swap Cold Re-Prefill=%llu toks (%u turns) | ANE ran_on_ane=%u | MajFlt=%ld %s\n",
                max_switch_ms, static_cast<unsigned long long>(cold_tokens_total), post_swap_turns, st.ran_on_ane, steady_maj,
                (zero_pause && zero_fault) ? "(PASS)" : "(FAIL)");
    std::printf("  Probe #1 (Math)    : %s -> %s\n", p1_ok ? "PASS" : "FAIL", p1_rec.clean_reply.c_str());
    std::printf("  Probe #2 (Nuance)  : %s (cause=%s, replacement=%s, rule=%s, 0-overlap-story=%s)\n",
                p2_ok ? "PASS" : "FAIL", p2_cause ? "Y" : "N", p2_choice ? "Y" : "N",
                p2_rule ? "Y" : "N", overlap_zero_story ? "Y" : "N");
    std::printf("    -> Reply #2      : %s\n", p2_rec.clean_reply.c_str());
    std::printf("  Probe #3 (Dormant) : %s (0-overlap-echo=%s) -> %s\n",
                p3_ok ? "PASS" : "FAIL", overlap_zero_echo ? "Y" : "N", p3_rec.clean_reply.c_str());
    if (!comp.ledger().empty() && !comp.ledger()[0].semantic_summary.empty()) {
        std::printf("  E1 180B Summary    : %s\n", comp.ledger()[0].semantic_summary.c_str());
    }
    std::printf("  Memory Footprint   : Total UMA %.2f GiB (GPU Wired %.2f GiB, Host RSS %.2f GiB)\n",
                mem_end.total_uma_gib, mem_end.gpu_wired_gib, mem_end.host_rss_gib);
    std::printf("  OVERALL VERDICT    : %s\n",
                overall_ok ? "PASS (True 64K+ scale + 0.00 ms realtime compaction + subtle narrative & rules preserved!)" : "FAIL");
    std::printf("======================================================================================================================================================\n");
    std::fflush(stdout);
    return overall_ok ? 0 : 1;
}

static int run_verify_save_load(IChatBackend* backend, const ChatCliConfig& base_cfg) {
    ChatCliConfig vcfg   = base_cfg;
    vcfg.greedy          = true;
    vcfg.temperature     = 0.0f;
    vcfg.enable_thinking = false;
    if (!vcfg.max_tok_explicit) vcfg.max_tokens = 64U;
    if (backend->is_glm53() && !vcfg.glm53_mode_explicit) {
        vcfg.glm53_mode = STRATA_GLM53_MODE_LOSSLESS;
        backend->set_glm53_exec_mode(STRATA_GLM53_MODE_LOSSLESS);
    }
    if (vcfg.compact_mode != STRATA_COMPACT_OFF && !backend->compaction_supported()) {
        std::fprintf(stderr, "[verify-save-load] compaction mode `%s` is not supported on %s\n",
                     chat_compact_mode_name(vcfg.compact_mode), backend->model_name());
        return 1;
    }

    std::string tmp_root([NSTemporaryDirectory() UTF8String]);
    if (!tmp_root.empty() && tmp_root.back() != '/') tmp_root.push_back('/');
    std::string snap_dir_tmpl = tmp_root + "strata_verify_save_load_XXXXXX";
    std::vector<char> snap_dir_buf(snap_dir_tmpl.begin(), snap_dir_tmpl.end());
    snap_dir_buf.push_back('\0');
    char* snap_dir_cstr = ::mkdtemp(snap_dir_buf.data());
    if (!snap_dir_cstr) {
        std::fprintf(stderr, "[verify-save-load] mkdtemp failed\n");
        return 1;
    }
    const std::string snap_dir(snap_dir_cstr);
    const std::string snap_path = snap_dir + "/verify.strata_session";
    struct SnapDirGuard {
        std::string dir;
        std::string file;
        ~SnapDirGuard() {
            if (!file.empty()) (void)::unlink(file.c_str());
            if (!dir.empty())  (void)::rmdir(dir.c_str());
        }
    } snap_dir_guard{snap_dir, snap_path};

    /* Warm Metal pipelines & zero-copy pages before steady-state MajFlt baseline */
    {
        ChatSamplingSpec warm_sp{};
        warm_sp.greedy = true;
        uint32_t warm_ids[4] = {1U, 2U, 3U, 4U};
        uint32_t warm_tok    = 0U;
        (void)backend->prefill(warm_ids, 4U, warm_sp, &warm_tok, nullptr);
        if (backend->is_glm53()) backend->reset_glm53_hot_cache();
        backend->reset();
    }

    ChatReplSession session(backend, vcfg);
    if (!session.ok()) {
        std::fprintf(stderr, "[verify-save-load] Failed to allocate GPU checkpoint arena\n");
        return 1;
    }
    CompactionController comp(backend, &session);
    comp.set_echo_events(true);
    comp.set_slice_tokens(kShadowSliceBatch);
    std::string mode_msg;
    if (vcfg.compact_mode != STRATA_COMPACT_OFF) {
        if (!comp.set_mode(vcfg.compact_mode, &mode_msg)) {
            std::fprintf(stderr, "%s\n", mode_msg.c_str());
            return 1;
        }
    }
    session.rebaseline_mem();
    const ChatMemSnapshot baseline = session.baseline_mem();

    std::vector<std::string> setup_turns;
    if (comp.tier1()) {
        const uint32_t win = vcfg.compact_window;
        setup_turns.push_back(verify_build_sized_turn(
            backend,
            "Please note these exact build configuration values for our benchmark: BUILD_ID = 4417 and PROJECT = HELIOS-9. "
            "Here is an unrelated background log:",
            "Reply with one short sentence confirming both configuration values.",
            static_cast<uint32_t>(0.32 * win), 11U));
        setup_turns.push_back(verify_build_sized_turn(
            backend,
            "One more exact build configuration value to note: OFFSET = 0x2A7. Unrelated background log:",
            "Reply with one short sentence confirming the offset.",
            static_cast<uint32_t>(0.22 * win), 107U));
        setup_turns.push_back(verify_build_sized_turn(
            backend,
            "Routine subsystem status log:",
            "Reply with one short sentence acknowledging the log.",
            static_cast<uint32_t>(0.20 * win), 211U));
        setup_turns.push_back(verify_build_sized_turn(
            backend,
            "Second routine subsystem status log:",
            "Reply with one short sentence acknowledging the log.",
            static_cast<uint32_t>(0.20 * win), 307U));
    } else {
        setup_turns = {
            "Note these two exact build configuration values: BUILD_ID = 4417 and PROJECT = HELIOS-9. "
            "Reply in one short sentence confirming both values.",
            "Note one more build configuration value: OFFSET = 0x2A7. Reply in one short sentence confirming the offset.",
            "Acknowledge ready for summary in one short sentence."
        };
    }
    const std::string probe_prompt =
        "State the exact values of BUILD_ID, PROJECT, and OFFSET from earlier in one concise line "
        "formatted as BUILD_ID=<value>, PROJECT=<value>, OFFSET=<value>.";

    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Persistent `.strata_session` Bit-Exactness Proof (`--verify-save-load`)\n");
    std::printf("  Model  : %s (%s)\n", backend->model_name(), backend->engine_id_tag());
    std::printf("  Config : compact_mode=%s | K=%u | greedy --no-think | %zu setup turns -> /save -> /reset -> /load\n",
                chat_compact_mode_name(vcfg.compact_mode),
                backend->has_mtp() ? vcfg.draft_k : 0U,
                setup_turns.size());
    std::printf("====================================================================================================\n");
    std::fflush(stdout);

    for (size_t ti = 0U; ti < setup_turns.size(); ++ti) {
        if (comp.active()) comp.pre_turn();
        ChatTurnRecord rec{};
        if (session.run_turn(setup_turns[ti], /*stream_stdout=*/false, /*ansi_style=*/false, &rec) != 0) {
            std::fprintf(stderr, "[verify-save-load] setup turn %zu failed\n", ti + 1U);
            return 1;
        }
        if (comp.active()) comp.on_turn_committed();
        if (comp.tier1()) (void)comp.run_idle_gap();
        std::printf("  [Setup T%zu] pos %4u -> %4u | %5.1f ms prefill | %zu toks | swaps=%u | \"%s\"\n",
                    ti + 1U, rec.seq_pos_before, rec.seq_pos_after, rec.prefill_ms,
                    rec.generated_ids.size(), comp.swaps(), rec.clean_reply.c_str());
        std::fflush(stdout);
    }

    const uint32_t pre_save_pos   = backend->seq_pos();
    const uint32_t pre_save_swaps = comp.swaps();
    const size_t   pre_save_floor = session.history_floor();
    const size_t   pre_save_turns = session.history().size();

    uint64_t saved_bytes = 0ULL;
    double   save_ms     = 0.0;
    std::string io_err;
    if (!save_session_to_file(snap_path, backend, &session, &comp, &saved_bytes, &save_ms, &io_err)) {
        std::fprintf(stderr, "[verify-save-load] save_session_to_file failed: %s\n", io_err.c_str());
        (void)::unlink(snap_path.c_str());
        return 1;
    }

    /* Reference continuation turn right after `/save` */
    if (comp.active()) comp.pre_turn();
    ChatTurnRecord ref_rec{};
    if (session.run_turn(probe_prompt, /*stream_stdout=*/false, /*ansi_style=*/false, &ref_rec) != 0) {
        std::fprintf(stderr, "[verify-save-load] reference continuation turn failed\n");
        (void)::unlink(snap_path.c_str());
        return 1;
    }

    /* Full teardown / reset to seq_pos=0 and mode=off before restoring */
    (void)session.reset_session();
    comp.on_session_reset();
    {
        std::string off_msg;
        (void)comp.set_mode(STRATA_COMPACT_OFF, &off_msg);
    }
    backend->reset();
    const bool wiped_clean = (backend->seq_pos() == 0U && session.history().empty() && comp.mode() == STRATA_COMPACT_OFF);

    uint64_t loaded_bytes = 0ULL;
    double   load_ms      = 0.0;
    const bool load_ok = load_session_from_file(snap_path, backend, &session, &comp, &loaded_bytes, &load_ms, &io_err);
    (void)::unlink(snap_path.c_str());
    const bool cleanup_ok = (::access(snap_path.c_str(), F_OK) != 0);
    if (!load_ok) {
        std::fprintf(stderr, "[verify-save-load] load_session_from_file failed: %s\n", io_err.c_str());
        return 1;
    }

    const bool meta_ok = wiped_clean &&
                         (backend->seq_pos() == pre_save_pos) &&
                         (comp.swaps() == pre_save_swaps) &&
                         (session.history_floor() == pre_save_floor) &&
                         (session.history().size() == pre_save_turns) &&
                         (comp.mode() == vcfg.compact_mode) &&
                         (saved_bytes == loaded_bytes);

    /* Restored continuation turn: must execute with ZERO cold re-prefill tokens and 100% bit-exact token IDs */
    const uint64_t pf0 = backend->prefill_tokens_total();
    const uint64_t sl0 = backend->slice_tokens_total();
    if (comp.active()) comp.pre_turn();
    const uint64_t sl_pre = backend->slice_tokens_total() - sl0;
    ChatTurnRecord rest_rec{};
    if (session.run_turn(probe_prompt, /*stream_stdout=*/false, /*ansi_style=*/false, &rest_rec) != 0) {
        std::fprintf(stderr, "[verify-save-load] restored continuation turn failed\n");
        return 1;
    }
    const uint64_t pf_delta = backend->prefill_tokens_total() - pf0;
    const uint64_t own      = static_cast<uint64_t>(rest_rec.prompt_delta_toks) + rest_rec.committed_toks;
    const uint64_t cold     = (pf_delta > own ? pf_delta - own : 0ULL) + sl_pre;

    const ChatMemSnapshot mem_end = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());
    const long steady_maj = std::max<long>(0L, mem_end.majflt - baseline.majflt);

    const bool ids_exact  = !ref_rec.generated_ids.empty() && (ref_rec.generated_ids == rest_rec.generated_ids);
    const bool has_vault  = rest_rec.clean_reply.find("4417") != std::string::npos;
    const bool has_proj   = rest_rec.clean_reply.find("HELIOS-9") != std::string::npos;
    const bool has_offset = verify_contains_ci(rest_rec.clean_reply, "0x2A7");
    const bool recall_ok  = has_vault && has_proj && has_offset;
    const bool tier1_ok   = !comp.tier1() || (pre_save_swaps >= 1U);
    const bool cold_ok    = (cold == 0ULL);
    const bool flt_ok     = (steady_maj == 0L);
    const bool pass       = meta_ok && tier1_ok && cold_ok && ids_exact && recall_ok && cleanup_ok && flt_ok;

    constexpr double kMiB = 1024.0 * 1024.0;
    const double save_gbps = (save_ms > 0.0) ? (static_cast<double>(saved_bytes) / 1e9) / (save_ms / 1e3) : 0.0;
    const double load_gbps = (load_ms > 0.0) ? (static_cast<double>(loaded_bytes) / 1e9) / (load_ms / 1e3) : 0.0;

    std::printf("----------------------------------------------------------------------------------------------------\n");
    std::printf("  Snapshot Container  : %.2f MiB | Save %.1f ms (%.2f GB/s) | Load %.1f ms (%.2f GB/s) | unlinked=%s\n",
                static_cast<double>(saved_bytes) / kMiB, save_ms, save_gbps, load_ms, load_gbps,
                cleanup_ok ? "YES" : "NO");
    std::printf("  Restored Metadata   : seq_pos=%u (expect %u) | turns=%zu (floor %zu) | swaps=%u | mode=%s %s\n",
                rest_rec.seq_pos_before, pre_save_pos, pre_save_turns, pre_save_floor, pre_save_swaps,
                chat_compact_mode_name(comp.mode()), (meta_ok && tier1_ok) ? "(PASS)" : "(FAIL)");
    std::printf("  Zero Re-Prefill     : cold re-prefill %llu toks (prompt delta %u + commit %u) %s\n",
                static_cast<unsigned long long>(cold), rest_rec.prompt_delta_toks, rest_rec.committed_toks,
                cold_ok ? "(PASS)" : "(FAIL)");
    std::printf("  Continuation Tokens : %s (%zu / %zu bit-exact token IDs)\n",
                ids_exact ? "100% BIT-EXACT MATCH (PASS)" : "MISMATCH (FAIL)",
                rest_rec.generated_ids.size(), ref_rec.generated_ids.size());
    std::printf("  Restored Reply      : \"%s\" | 4417=%s HELIOS-9=%s 0x2A7=%s %s\n",
                rest_rec.clean_reply.c_str(),
                has_vault ? "Y" : "N", has_proj ? "Y" : "N", has_offset ? "Y" : "N",
                recall_ok ? "(PASS)" : "(FAIL)");
    std::printf("  Fault & UMA Gate    : Total UMA %.2f GiB | Steady MajFlt=%ld %s\n",
                mem_end.total_uma_gib, steady_maj, flt_ok ? "(PASS)" : "(FAIL)");
    std::printf("  RESULT              : %s\n", pass ? "PASS" : "FAIL");
    std::printf("====================================================================================================\n");
    std::fflush(stdout);
    return pass ? 0 : 1;
}

/* ============================================================================
 * 6. Interactive & Scripted REPL Driver
 * ============================================================================ */

static int run_chat_entry(int argc, char** argv, ChatSubcmdKind kind) {
    ChatCliConfig cfg{};
    std::string parse_err;
    if (parse_chat_cli_args(argc, argv, kind, &cfg, &parse_err) != 0) {
        std::fprintf(stderr, "[chat] %s\n", parse_err.c_str());
        return 1;
    }
    if (cfg.show_help) {
        print_chat_help(argv[0]);
        return 0;
    }

    std::ifstream script_ifs;
    if (!cfg.script_path.empty()) {
        script_ifs.open(cfg.script_path);
        if (!script_ifs.is_open()) {
            std::fprintf(stderr, "[chat] Failed to open --script `%s`\n", cfg.script_path.c_str());
            return 1;
        }
    }

    const ChatMemSnapshot mem_init0 = query_chat_mem_snapshot();
    const bool use_inkling2 = (kind == ChatSubcmdKind::kChatInkling2) || chat_is_inkling2_bundle(cfg.bundle_dir);
    const bool use_glm53    = !use_inkling2 &&
                              ((kind == ChatSubcmdKind::kChatGlm53) || chat_is_glm53_bundle(cfg.bundle_dir));
    const bool use_flash    = !use_inkling2 && !use_glm53 &&
                              ((kind == ChatSubcmdKind::kChatFlash) || chat_is_flash_bundle(cfg.bundle_dir));

    std::unique_ptr<IChatBackend> backend;
    if (use_inkling2) {
        backend = std::make_unique<Inkling2ChatBackend>();
    } else if (use_glm53) {
        backend = std::make_unique<Glm53ChatBackend>();
    } else if (use_flash) {
        backend = std::make_unique<QwenFlashChatBackend>();
    } else {
        backend = std::make_unique<Qwen38ChatBackend>();
    }

    if (!backend->init(cfg)) {
        std::fprintf(stderr, "[chat] Failed to load bundle / initialize GPU engine from `%s`\n",
                     cfg.bundle_dir.c_str());
        return 1;
    }
    if (!backend->has_mtp()) {
        cfg.draft_k = 0U;
    }

    if (cfg.verify_multiturn) {
        return run_verify_multiturn(backend.get(), cfg);
    }
    if (cfg.verify_paged_kv) {
        return run_verify_paged_kv(backend.get(), cfg);
    }
    if (cfg.verify_shadow_swap) {
        return run_verify_shadow_swap(backend.get(), cfg);
    }
    if (cfg.verify_save_load) {
        return run_verify_save_load(backend.get(), cfg);
    }
    if (cfg.bench_compaction_rule) {
        return run_bench_compaction_rule(backend.get(), cfg);
    }
    if (cfg.bench_longform_64k) {
        return run_bench_longform_64k(backend.get(), cfg);
    }

    const bool stdin_tty  = (::isatty(STDIN_FILENO) != 0) && cfg.script_path.empty();
    const bool stdout_tty = (::isatty(STDOUT_FILENO) != 0);

    ChatReplSession session(backend.get(), cfg);
    if (!session.ok()) {
        std::fprintf(stderr, "[chat] Failed to allocate GPU checkpoint buffers\n");
        return 1;
    }

    CompactionController comp(backend.get(), &session);
    comp.set_slice_tokens(stdin_tty ? kShadowSliceTty : kShadowSliceBatch);
    comp.set_echo_events(!stdin_tty);
    std::string compact_init_msg;
    std::string loaded_banner_msg;
    if (!cfg.load_session_path.empty()) {
        uint64_t lbytes = 0ULL;
        double   lms    = 0.0;
        std::string lerr;
        if (!load_session_from_file(cfg.load_session_path, backend.get(), &session, &comp, &lbytes, &lms, &lerr)) {
            std::fprintf(stderr, "[session] Failed to load `%s`: %s\n", cfg.load_session_path.c_str(), lerr.c_str());
            return 1;
        }
        if (cfg.compact_mode_explicit && cfg.compact_mode != comp.mode()) {
            session.mutable_cfg().compact_mode = cfg.compact_mode;
            if (!comp.set_mode(cfg.compact_mode, &compact_init_msg)) {
                std::fprintf(stderr, "%s\n", compact_init_msg.c_str());
                return 1;
            }
            session.rebaseline_mem();
        }
        char lbuf[384];
        std::snprintf(lbuf, sizeof(lbuf),
                      "Loaded `%s` (%.2f MiB in %.1f ms | %zu turns, floor %zu, seq_pos=%u, mode=%s)",
                      cfg.load_session_path.c_str(),
                      static_cast<double>(lbytes) / (1024.0 * 1024.0), lms,
                      session.history().size(), session.history_floor(),
                      backend->seq_pos(), chat_compact_mode_name(comp.mode()));
        loaded_banner_msg = lbuf;
    } else if (cfg.compact_mode != STRATA_COMPACT_OFF) {
        if (!comp.set_mode(cfg.compact_mode, &compact_init_msg)) {
            std::fprintf(stderr, "%s\n", compact_init_msg.c_str());
            return 1;
        }
        session.rebaseline_mem();
    }
    const long init_majflt = std::max<long>(0L, session.baseline_mem().majflt - mem_init0.majflt);

    char frspec_lbl[32];
    if (backend->frspec_enabled()) {
        if (backend->frspec_shortlist_size() > 0U) {
            std::snprintf(frspec_lbl, sizeof(frspec_lbl), "ON (%u)", backend->frspec_shortlist_size());
        } else {
            std::snprintf(frspec_lbl, sizeof(frspec_lbl), "ON");
        }
    } else {
        std::snprintf(frspec_lbl, sizeof(frspec_lbl), "OFF");
    }

    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Stateful Multi-Turn Chat REPL (%s | ZeroCopy=%d | Init %.0f ms | Metal Cache: %s (%.1f ms))\n",
                backend->device_name(), backend->zero_copy() ? 1 : 0, backend->init_wall_ms(),
                backend->metal_cache_hit() ? "HIT" : "MISS", backend->pso_load_ms());
    std::printf("  Model  : %s\n", backend->model_name());
    if (backend->is_inkling2()) {
        std::printf("  Memory : Zero-Copy UMA %.2f GiB | KV=%s | SWA+SConv %.1f MiB O(1) | Init MajFlt=%ld\n",
                    backend->uma_mapped_gib(),
                    backend->kv_cache_name(),
                    backend->deltanet_mib(),
                    init_majflt);
    } else if (backend->is_glm53()) {
        if (backend->glm53_is_ram_resident()) {
            std::printf("  Memory : Zero-Copy UMA %.2f GiB (100%% RAM-Resident E8S-18-V4 | 0 B NVMe) | KDA %.1f MiB O(1) | Init MajFlt=%ld\n",
                        backend->uma_mapped_gib(),
                        backend->deltanet_mib(),
                        init_majflt);
        } else {
            std::printf("  Memory : Spine %.2f GiB + Hot Cache %.2f GiB (%u slots) + NVMe %.2f GiB | 16c ANE=%d | Init MajFlt=%ld\n",
                        backend->uma_mapped_gib(),
                        backend->glm53_hot_cache_gib(),
                        backend->glm53_hot_capacity(),
                        backend->glm53_experts_nvme_gib(),
                        backend->glm53_ane_verified() ? 1 : 0,
                        init_majflt);
        }
    } else if (backend->has_ple()) {
        std::printf("  Memory : UMA %.2f GiB + NVMe PLE %.2f GiB (%.0f MiB mlock LFU) | Init MajFlt=%ld\n",
                    backend->uma_mapped_gib(), backend->ple_nvme_gib(), backend->ple_lfu_mib(), init_majflt);
    } else {
        std::printf("  Memory : Zero-Copy UMA %.2f GiB | DeltaNet %.1f MiB O(1) | Init MajFlt=%ld\n",
                    backend->uma_mapped_gib(), backend->deltanet_mib(), init_majflt);
    }
    const std::string max_tok_label = (cfg.max_tokens == 0U)
        ? std::string("0 (unlimited)")
        : std::to_string(cfg.max_tokens);
    std::printf("  Config : %s | %s | K=%u | Prefill: %s | FR-Spec: %s | Adaptive-K: %s | max_tokens=%s | max_seq=%u\n",
                cfg.enable_thinking ? ("Thinking (<think>, effort=" + cfg.effort + ")").c_str()
                                    : "Direct (--no-think)",
                cfg.greedy ? "Greedy (T=0)" : "Stochastic",
                cfg.draft_k,
                backend->int8_prefill_enabled() ? "INT8" : "FP16",
                frspec_lbl,
                backend->adaptive_k_enabled() ? "ON" : "OFF",
                max_tok_label.c_str(),
                backend->max_seq_len());
    if (backend->is_inkling2()) {
        std::printf("  Engine : Tree-Spec: %s | Bitplane-Head: %s | Vision: %s\n",
                    backend->tree_spec_enabled() ? "ON" : "OFF",
                    backend->bitplane_head_enabled() ? "ON" : "OFF",
                    backend->has_vision() ? "ON" : "OFF");
    }
    if (comp.active() && !compact_init_msg.empty()) {
        std::printf("  Compact: %s\n", compact_init_msg.c_str());
    }
    if (!loaded_banner_msg.empty()) {
        std::printf("  Session: %s\n", loaded_banner_msg.c_str());
    }
    std::printf("====================================================================================================\n");
    std::fflush(stdout);

    auto run_prompt_turn = [&](const std::string& raw_prompt, bool echo_input) -> bool {
        const std::string trimmed = chat_trim_ascii(raw_prompt);
        if (trimmed.empty()) return true;

        if (comp.active()) {
            comp.pre_turn();
        }
        const size_t next_turn_num = session.history().size() + 1U;
        if (echo_input) {
            std::printf("\n[Turn %zu] User: %s\n[Turn %zu] Assistant:\n",
                        next_turn_num, trimmed.c_str(), next_turn_num);
            std::fflush(stdout);
        } else if (stdout_tty) {
            std::printf("\033[1;32massistant>\033[0m ");
            if (session.cfg().enable_thinking) std::fputc('\n', stdout);
            std::fflush(stdout);
        }

        const int turn_rc = session.run_turn(trimmed, /*stream_stdout=*/true, /*ansi_style=*/stdout_tty, nullptr);
        if (turn_rc == 0 && comp.active()) {
            comp.on_turn_committed();
        }
        return true;
    };

    auto process_line = [&](const std::string& raw_line, bool echo_input) -> bool {
        const std::string trimmed = chat_trim_ascii(raw_line);
        if (trimmed.empty()) return true;

        std::string cmd_msg;
        const SlashCmdAction act = parse_slash_command(
            trimmed, backend->has_mtp(), &session.mutable_cfg(), &cmd_msg);

        if (act != SlashCmdAction::kNotSlash) {
            if (echo_input) {
                std::printf("\n> %s\n", trimmed.c_str());
            }
            if (act == SlashCmdAction::kExit) {
                return false;
            }
            if (act == SlashCmdAction::kMode) {
                backend->set_glm53_exec_mode(session.cfg().glm53_mode);
                if (!cmd_msg.empty()) {
                    std::printf("%s\n", cmd_msg.c_str());
                    std::fflush(stdout);
                }
                return true;
            }
            if (act == SlashCmdAction::kHandled || act == SlashCmdAction::kError) {
                if (!cmd_msg.empty()) {
                    std::printf("%s\n", cmd_msg.c_str());
                    std::fflush(stdout);
                }
                return true;
            }
            if (act == SlashCmdAction::kCompactMode) {
                comp.flush_events();
                std::string m_msg;
                if (!comp.set_mode(session.cfg().compact_mode, &m_msg)) {
                    session.mutable_cfg().compact_mode = comp.mode();
                } else {
                    session.rebaseline_mem();
                }
                std::printf("%s\n", m_msg.c_str());
                std::fflush(stdout);
                return true;
            }
            if (act == SlashCmdAction::kCompactStats) {
                comp.flush_events();
                const ChatMemSnapshot m_now = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());
                const long s_maj = std::max<long>(0L, m_now.majflt - session.baseline_mem().majflt);
                comp.print_stats(s_maj);
                return true;
            }
            if (act == SlashCmdAction::kCompact) {
                comp.flush_events();
                std::string c_msg;
                (void)comp.force_compact(&c_msg);
                std::printf("%s\n", c_msg.c_str());
                std::fflush(stdout);
                return true;
            }
            if (act == SlashCmdAction::kUndoCompact) {
                comp.flush_events();
                std::string u_msg;
                (void)comp.undo_compact(nullptr, nullptr, nullptr, &u_msg);
                std::printf("%s\n", u_msg.c_str());
                std::fflush(stdout);
                return true;
            }
            if (act == SlashCmdAction::kSaveSession) {
                comp.flush_events();
                uint64_t nbytes = 0ULL;
                double   sms    = 0.0;
                std::string serr;
                if (save_session_to_file(cmd_msg, backend.get(), &session, &comp, &nbytes, &sms, &serr)) {
                    std::printf("[session] Saved `%s` (%.2f MiB in %.1f ms | %zu turns, floor %zu, seq_pos=%u, mode=%s)\n",
                                cmd_msg.c_str(), static_cast<double>(nbytes) / (1024.0 * 1024.0), sms,
                                session.history().size(), session.history_floor(),
                                backend->seq_pos(), chat_compact_mode_name(comp.mode()));
                } else {
                    std::printf("[session] Save failed: %s\n", serr.c_str());
                }
                std::fflush(stdout);
                return true;
            }
            if (act == SlashCmdAction::kLoadSession) {
                comp.flush_events();
                uint64_t nbytes = 0ULL;
                double   lms    = 0.0;
                std::string lerr;
                if (load_session_from_file(cmd_msg, backend.get(), &session, &comp, &nbytes, &lms, &lerr)) {
                    std::printf("[session] Loaded `%s` (%.2f MiB in %.1f ms | %zu turns, floor %zu, seq_pos=%u, mode=%s)\n",
                                cmd_msg.c_str(), static_cast<double>(nbytes) / (1024.0 * 1024.0), lms,
                                session.history().size(), session.history_floor(),
                                backend->seq_pos(), chat_compact_mode_name(comp.mode()));
                } else {
                    std::printf("[session] Load failed: %s\n", lerr.c_str());
                }
                std::fflush(stdout);
                return true;
            }
            if (act == SlashCmdAction::kStats) {
                comp.flush_events();
                session.print_stats();
                return true;
            }
            if (act == SlashCmdAction::kUndo) {
                comp.flush_events();
                double rw_ms = 0.0;
                uint32_t p0 = 0U, p1 = 0U;
                if (session.undo_last_turn(&rw_ms, &p0, &p1)) {
                    comp.on_undo();
                    std::printf("[undo] Rewound last turn in %.3f ms (GPU O(1) blit) | seq_pos %u -> %u (%zu turns active)\n",
                                rw_ms, p0, p1, session.history().size());
                } else if (!session.history().empty()) {
                    std::printf("[undo] At compacted ledger floor (%zu turns); use /undo-compact (if available) or /reset.\n",
                                session.history_floor());
                } else {
                    std::printf("[undo] Already at start of conversation (seq_pos=0).\n");
                }
                std::fflush(stdout);
                return true;
            }
            if (act == SlashCmdAction::kReset) {
                const double ms = session.reset_session();
                comp.on_session_reset();
                std::printf("[reset] Reset KV + KDA/DeltaNet + PLE state in %.3f ms -> seq_pos=0/%u (weights stay wired)\n",
                            ms, backend->max_seq_len());
                std::fflush(stdout);
                return true;
            }
            if (act == SlashCmdAction::kSystem) {
                const double ms = session.reset_session();
                comp.on_session_reset();
                std::printf("[system] Updated system prompt (%zu chars) & reset state in %.3f ms -> seq_pos=0\n",
                            session.cfg().system_prompt.size(), ms);
                std::fflush(stdout);
                return true;
            }
            if (act == SlashCmdAction::kImage) {
                std::string img_path = cmd_msg;
                std::string inline_prompt;
                if (::access(img_path.c_str(), R_OK) != 0) {
                    size_t ws = 0U;
                    while (ws < cmd_msg.size() && !std::isspace(static_cast<unsigned char>(cmd_msg[ws]))) ++ws;
                    if (ws < cmd_msg.size()) {
                        const std::string cand = cmd_msg.substr(0U, ws);
                        if (::access(cand.c_str(), R_OK) == 0) {
                            img_path      = cand;
                            inline_prompt = chat_trim_ascii(cmd_msg.substr(ws));
                        }
                    }
                }
                std::string vmsg;
                const bool ok = session.stage_image(img_path, &vmsg);
                std::printf("%s\n", vmsg.c_str());
                std::fflush(stdout);
                if (!ok || inline_prompt.empty()) {
                    return true;
                }
                if (comp.active()) comp.pre_turn();
                const size_t next_turn_num = session.history().size() + 1U;
                if (echo_input) {
                    std::printf("\n[Turn %zu] User: %s\n[Turn %zu] Assistant:\n",
                                next_turn_num, inline_prompt.c_str(), next_turn_num);
                    std::fflush(stdout);
                } else if (stdout_tty) {
                    std::printf("\033[1;32massistant>\033[0m ");
                    if (session.cfg().enable_thinking) std::fputc('\n', stdout);
                    std::fflush(stdout);
                }
                const int turn_rc = session.run_turn(inline_prompt, /*stream_stdout=*/true, /*ansi_style=*/stdout_tty, nullptr);
                if (turn_rc == 0 && comp.active()) comp.on_turn_committed();
                return true;
            }
        }

        return run_prompt_turn(trimmed, echo_input);
    };

    /* 1. Stage any `--image` arguments and execute any `--prompt` / `--prompt-file` arguments first.
     *    Prompts passed via `--prompt`/`-p`/`--prompt-file` go straight to the model turn without
     *    REPL slash-command parsing; typed input, piped stdin, and `--script` still use `process_line`. */
    for (const std::string& ipath : cfg.initial_images) {
        std::string vmsg;
        if (!session.stage_image(ipath, &vmsg)) {
            std::fprintf(stderr, "%s\n", vmsg.c_str());
            return 1;
        }
        std::printf("%s\n", vmsg.c_str());
        std::fflush(stdout);
    }
    for (size_t pi = 0U; pi < cfg.initial_prompts.size(); ++pi) {
        if (!run_prompt_turn(cfg.initial_prompts[pi], /*echo_input=*/true)) {
            return 0;
        }
        if (pi + 1U < cfg.initial_prompts.size() && comp.tier1() && comp.has_idle_work()) {
            (void)comp.run_idle_gap();
        }
    }
    if (cfg.verify_k0 && !cfg.initial_prompts.empty() && !session.history().empty()) {
        const std::vector<ChatTurnRecord> spec_hist = session.history();
        const uint32_t saved_k = session.mutable_cfg().draft_k;
        if (backend->is_glm53() && session.cfg().glm53_mode == STRATA_GLM53_MODE_ADAPTIVE) {
            backend->reset_glm53_hot_cache();
        }
        session.reset_session();
        comp.on_session_reset();
        session.mutable_cfg().draft_k = 0U;
        bool k0_ok = true;
        for (size_t ti = 0U; ti < cfg.initial_prompts.size() && ti < spec_hist.size(); ++ti) {
            ChatTurnRecord ref_rec{};
            if (session.run_turn(cfg.initial_prompts[ti], /*stream_stdout=*/false, /*ansi_style=*/false, &ref_rec) != 0 ||
                ref_rec.generated_ids != spec_hist[ti].generated_ids) {
                k0_ok = false;
                std::printf("[verify-k0] Turn %zu MISMATCH (%zu vs %zu tokens)\n",
                            ti + 1U, spec_hist[ti].generated_ids.size(), ref_rec.generated_ids.size());
                break;
            }
            const double spd = (ref_rec.decode_tok_s > 0.0)
                ? (spec_hist[ti].decode_tok_s / ref_rec.decode_tok_s)
                : 1.0;
            std::printf("[verify-k0] Turn %zu K=%u vs K=0: 100%% BIT-EXACT MATCH (%zu == %zu tokens | %.2f vs %.2f tok/s = %.2fx)\n",
                        ti + 1U, saved_k, spec_hist[ti].generated_ids.size(), ref_rec.generated_ids.size(),
                        spec_hist[ti].decode_tok_s, ref_rec.decode_tok_s, spd);
        }
        session.mutable_cfg().draft_k = saved_k;
        std::fflush(stdout);
        return k0_ok ? 0 : 1;
    }
    if (cfg.once && !cfg.initial_prompts.empty()) {
        return 0;
    }

    /* 2. Execute `--script <path>` or interactive / piped `stdin`.
     *    - Interactive TTY in shadow/hybrid mode: `CooperativeLineReader` runs `poll(STDIN_FILENO)` in cbreak
     *      mode (`poll(0)` before every 32-token shadow-warmup slice, first keystroke pauses background GPU
     *      work immediately; `poll(-1)` when idle). In `off`/`paged` mode blocking `std::getline` is unchanged.
     *    - Non-TTY (`--script` or piped stdin) in shadow/hybrid mode: pending idle quanta are drained between
     *      lines via `comp.run_idle_gap()` as a deterministic simulated user reading/typing gap. */
    std::istream& in_stream = !cfg.script_path.empty() ? static_cast<std::istream&>(script_ifs)
                                                       : static_cast<std::istream&>(std::cin);
    std::unique_ptr<CooperativeLineReader> coop_reader;
    std::string line;
    while (true) {
        if (!stdin_tty && comp.tier1() && comp.has_idle_work()) {
            (void)comp.run_idle_gap();
        }
        if (stdin_tty) {
            comp.flush_events();
            if (stdout_tty) {
                std::printf("\n\033[1;36m[turn %zu | pos %u] user>\033[0m ",
                            session.history().size() + 1U, backend->seq_pos());
            } else {
                std::printf("\n[turn %zu | pos %u] user> ",
                            session.history().size() + 1U, backend->seq_pos());
            }
            std::fflush(stdout);
        }

        if (stdin_tty) {
            if (!coop_reader) {
                coop_reader = std::make_unique<CooperativeLineReader>(STDIN_FILENO, /*tty_cbreak=*/true, stdout_tty);
                while (std::cin.rdbuf() && std::cin.rdbuf()->in_avail() > 0) {
                    const char ch = static_cast<char>(std::cin.get());
                    coop_reader->feed(&ch, 1U);
                }
            }
            if (!coop_reader->read_line(comp.tier1() ? &comp : nullptr, &line)) {
                std::fputc('\n', stdout);
                break;
            }
        } else {
            if (coop_reader) coop_reader.reset();
            if (!std::getline(in_stream, line)) {
                if (stdin_tty) std::fputc('\n', stdout);
                break;
            }
        }
        if (!cfg.script_path.empty()) {
            const std::string t = chat_trim_ascii(line);
            if (t.empty() || (t.size() >= 2U && t[0] == '#' && t[1] == ' ')) {
                continue;
            }
        }
        if (!process_line(line, /*echo_input=*/!stdin_tty)) {
            break;
        }
    }
    coop_reader.reset();

    const ChatMemSnapshot mem_final = query_chat_mem_snapshot(backend->glm53_shared_mtl_gib());
    const long steady_maj = std::max<long>(0L, mem_final.majflt - session.baseline_mem().majflt);
    std::printf("[session end | %zu turns | seq_pos=%u/%u | Total UMA %.2f GiB | Steady MajFlt=%ld]\n",
                session.history().size(),
                backend->seq_pos(),
                backend->max_seq_len(),
                mem_final.total_uma_gib,
                steady_maj);
    std::fflush(stdout);
    return 0;
}

/* Lightweight zero-RSS mock backend used only by `strata_chat_repl_unit_tests()` (no Metal, no model files). */
class MockCompactBackend final : public IChatBackend {
public:
    struct MockSlot {
        std::vector<uint32_t> toks;
        bool                  paged = false;
    };
    struct MockCkpt {
        std::vector<uint32_t> toks;
    };

    bool init(const ChatCliConfig&) override { return true; }
    const char* model_name() const override { return "MockFlash"; }
    const char* device_name() const override { return "CPU-Mock"; }
    bool zero_copy() const override { return true; }
    double init_wall_ms() const override { return 0.0; }
    bool is_flash() const override { return true; }
    bool has_mtp() const override { return false; }
    bool has_ple() const override { return false; }
    uint32_t max_seq_len() const override { return 8192U; }
    uint32_t seq_pos() const override { return static_cast<uint32_t>(bound()->toks.size()); }
    double uma_mapped_gib() const override { return 0.0; }
    double ple_nvme_gib() const override { return 0.0; }
    double ple_lfu_mib() const override { return 0.0; }
    double deltanet_mib() const override { return 0.0; }
    double ple_state_mib() const override { return 0.0; }
    uint64_t kv_bytes_per_tok() const override { return 0ULL; }
    uint64_t active_bytes_per_tok() const override { return 0ULL; }
    uint32_t token_think_start() const override { return 9001U; }
    uint32_t token_think_end() const override { return 9002U; }
    bool is_eos(uint32_t tok) const override { return tok == 9000U; }

    /* Deterministic 1-byte = 1-token bijection so BPE pieces concatenate identically to live transcripts. */
    bool encode(const std::string& text, bool, std::vector<uint32_t>* out) const override {
        out->resize(text.size());
        for (size_t i = 0U; i < text.size(); ++i) (*out)[i] = static_cast<unsigned char>(text[i]);
        return true;
    }
    std::string decode(const uint32_t* ids, uint32_t n, bool) const override {
        std::string s;
        s.reserve(n);
        for (uint32_t i = 0U; i < n; ++i) {
            if (ids[i] < 256U) s.push_back(static_cast<char>(ids[i]));
        }
        return s;
    }
    void stream_reset() override {}
    std::string stream_step(uint32_t token_id, bool) override {
        return token_id < 256U ? std::string(1U, static_cast<char>(token_id)) : std::string();
    }
    std::string format_system_block(const std::string& s, bool th, const std::string& eff) const override {
        return format_flash_system_block(s, th, eff);
    }
    void reset() override { bound()->toks.clear(); }

    int prefill(const uint32_t* ids, uint32_t n, const ChatSamplingSpec&, uint32_t* nxt, StepDeltaTelemetry*) override {
        bound()->toks.insert(bound()->toks.end(), ids, ids + n);
        prefill_toks_ += n;
        if (nxt) *nxt = static_cast<uint32_t>('O');
        return 0;
    }
    int decode_step(uint32_t pending, uint32_t, const ChatSamplingSpec&, uint32_t* em, uint32_t* ne, StepDeltaTelemetry*) override {
        bound()->toks.push_back(pending);
        em[0] = (pending == static_cast<uint32_t>('O')) ? static_cast<uint32_t>('K') : 9000U;
        *ne   = 1U;
        return 0;
    }
    void* checkpoint_create() override { return new MockCkpt(); }
    void checkpoint_destroy(void* c) override { delete static_cast<MockCkpt*>(c); }
    int checkpoint_save(void* c) override {
        static_cast<MockCkpt*>(c)->toks = bound()->toks;
        return 0;
    }
    int checkpoint_restore(const void* c) override {
        bound()->toks = static_cast<const MockCkpt*>(c)->toks;
        return 0;
    }
    uint64_t checkpoint_bytes(const void*) const override { return 64ULL; }

    const char* engine_id_tag() const override { return "mock-flash"; }
    bool allow_save_      = true;
    bool fail_save_state_ = false;
    bool supports_session_save() const override { return allow_save_; }
    int save_state_fd(int fd, uint64_t* bytes_written_out) override {
        if (fail_save_state_) return -1;
        const auto& v = bound()->toks;
        if (!sess_write_u32_vec(fd, v)) return -1;
        if (bytes_written_out) *bytes_written_out = sizeof(uint64_t) + v.size() * sizeof(uint32_t);
        return 0;
    }
    int load_state_fd(int fd, uint64_t* bytes_read_out) override {
        std::vector<uint32_t> v;
        if (!sess_read_u32_vec(fd, &v)) return -1;
        bound()->toks = std::move(v);
        if (bytes_read_out) *bytes_read_out = sizeof(uint64_t) + bound()->toks.size() * sizeof(uint32_t);
        return 0;
    }

    bool compaction_supported() const override { return true; }
    void* compact_slot_create(uint32_t pool) override {
        auto* s  = new MockSlot();
        s->paged = (pool > 0U);
        return s;
    }
    void compact_slot_destroy(void* s) override {
        if (!s) return;
        if (bound_ == s) bound_ = nullptr;
        delete static_cast<MockSlot*>(s);
    }
    void compact_bind_slot(void* s) override { bound_ = static_cast<MockSlot*>(s); }
    void* compact_bound_slot() const override { return bound_; }
    uint32_t compact_slot_seq_pos(const void* s) const override {
        return s ? static_cast<uint32_t>(static_cast<const MockSlot*>(s)->toks.size()) : 0U;
    }
    bool compact_slot_is_paged(const void* s) const override {
        return s ? static_cast<const MockSlot*>(s)->paged : false;
    }
    void compact_reset_bound() override { bound()->toks.clear(); }
    int compact_prefill_slice(const uint32_t* ids, uint32_t n, bool, uint32_t* nxt) override {
        bound()->toks.insert(bound()->toks.end(), ids, ids + n);
        slice_toks_ += n;
        if (nxt) *nxt = static_cast<uint32_t>('O');
        return 0;
    }
    uint64_t prefill_tokens_total() const override { return prefill_toks_; }
    uint64_t slice_tokens_total() const override { return slice_toks_; }
    const std::vector<uint32_t>& bound_tokens() const { return bound()->toks; }

private:
    MockSlot* bound() { return bound_ ? bound_ : &default_; }
    const MockSlot* bound() const { return bound_ ? bound_ : &default_; }
    MockSlot default_{};
    MockSlot* bound_        = nullptr;
    uint64_t  prefill_toks_ = 0ULL;
    uint64_t  slice_toks_   = 0ULL;
};

} /* namespace */

extern "C" {

int strata_cmd_chat(int argc, char** argv) {
    return run_chat_entry(argc, argv, ChatSubcmdKind::kChatUnified);
}

int strata_cmd_chat_flash(int argc, char** argv) {
    return run_chat_entry(argc, argv, ChatSubcmdKind::kChatFlash);
}

int strata_cmd_chat_hf(int argc, char** argv) {
    return run_chat_entry(argc, argv, ChatSubcmdKind::kChatHf);
}

int strata_cmd_chat_glm53(int argc, char** argv) {
    return run_chat_entry(argc, argv, ChatSubcmdKind::kChatGlm53);
}

int strata_cmd_chat_inkling2(int argc, char** argv) {
    return run_chat_entry(argc, argv, ChatSubcmdKind::kChatInkling2);
}

/*
 * Lightweight Zero-RSS Unit Test Suite invoked by `build/test_cli` (`make test`):
 * Verifies CLI argument parsing (`chat`, `chat-flash`, `chat-hf`, `chat-glm53`),
 * default `max_tokens == 8192` & `/max-tokens 0` unlimited mode, slash command transitions,
 * `<think>...</think>` compaction, and 100% Turn-1/Turn-2+ parity across Qwen & GLM-5.3 formatters.
 */
int strata_chat_repl_unit_tests(void) {
    int fails = 0;

    /* 1. CLI Argument Parsing Tests (incl. default 8192 max_tokens, --max-tokens 0, and chat-glm53) */
    {
        ChatCliConfig c_def{};
        if (c_def.max_tokens != 8192U) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: expected default max_tokens == 8192\n");
            ++fails;
        }

        const char* av1[] = {
            "strata_cli", "chat-flash", "--draft-k", "3", "--no-think",
            "--temp", "0.4", "--max-tokens", "512", "--system", "Be concise",
            "--prompt", "Hello", "--once"
        };
        ChatCliConfig c1{};
        std::string err;
        if (parse_chat_cli_args(14, const_cast<char**>(av1), ChatSubcmdKind::kChatFlash, &c1, &err) != 0 ||
            c1.draft_k != 3U || c1.enable_thinking || c1.greedy ||
            std::fabs(c1.temperature - 0.4f) > 1e-5f || c1.max_tokens != 512U ||
            c1.system_prompt != "Be concise" || c1.initial_prompts.size() != 1U ||
            c1.initial_prompts[0] != "Hello" || !c1.once) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: valid chat-flash args parse failed\n");
            ++fails;
        }

        const char* av_glm[] = {
            "strata_cli", "chat-glm53", "--mode", "adaptive", "--hot-cache-gib", "48.0",
            "--effort", "max", "--max-tokens", "0", "--draft-k", "2",
            "--penalty-window", "256", "--presence-penalty", "0.6", "--frequency-penalty", "0.08"
        };
        ChatCliConfig c_glm{};
        if (parse_chat_cli_args(18, const_cast<char**>(av_glm), ChatSubcmdKind::kChatGlm53, &c_glm, &err) != 0 ||
            c_glm.bundle_dir != chat_default_glm53_bundle() ||
            c_glm.glm53_mode != STRATA_GLM53_MODE_ADAPTIVE ||
            std::fabs(c_glm.hot_cache_gib - 48.0f) > 1e-4f ||
            c_glm.effort != "max" || c_glm.max_tokens != 0U || !c_glm.max_tok_explicit ||
            c_glm.draft_k != 2U || c_glm.penalty_window != 256U ||
            std::fabs(c_glm.presence_penalty - 0.6f) > 1e-5f ||
            std::fabs(c_glm.frequency_penalty - 0.08f) > 1e-5f) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: chat-glm53 / --max-tokens 0 parse failed\n");
            ++fails;
        }

        const char* av_bad[] = {"strata_cli", "chat-flash", "--draft-k", "9"};
        ChatCliConfig c_bad{};
        if (parse_chat_cli_args(4, const_cast<char**>(av_bad), ChatSubcmdKind::kChatFlash, &c_bad, &err) == 0) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: accepted invalid --draft-k 9\n");
            ++fails;
        }

        const char* av_ver[] = {"strata_cli", "chat-hf", "--bundle", "/tmp/nonexistent_q38", "--verify-multiturn"};
        ChatCliConfig c_ver{};
        if (parse_chat_cli_args(5, const_cast<char**>(av_ver), ChatSubcmdKind::kChatHf, &c_ver, &err) != 0 ||
            !c_ver.verify_multiturn || !c_ver.greedy || c_ver.draft_k != 0U || c_ver.max_tok_explicit) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: --verify-multiturn parse failed\n");
            ++fails;
        }

        const char* av_long[] = {
            "strata_cli", "chat-flash", "--max-seq-len", "263168", "--max-tokens", "327680", "--verify-k0"
        };
        ChatCliConfig c_long{};
        if (parse_chat_cli_args(7, const_cast<char**>(av_long), ChatSubcmdKind::kChatFlash, &c_long, &err) != 0 ||
            c_long.max_seq_len != 263168U || c_long.max_tokens != 327680U ||
            !c_long.verify_k0 || !c_long.greedy) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: 256K+ --max-seq-len/--verify-k0 parse failed\n");
            ++fails;
        }

        const char* av_ik_def[] = {"strata_cli", "chat-inkling2"};
        const char* av_ik_q8[]  = {"strata_cli", "chat-inkling2", "--kv-cache", "q8", "--draft-k", "7"};
        const char* av_ik_q80[] = {"strata_cli", "chat-inkling2", "--kv-cache", "q8_0"};
        const char* av_ik_f16[] = {"strata_cli", "chat-inkling2", "--kv-cache", "fp16"};
        const char* av_ik_bad[] = {"strata_cli", "chat-inkling2", "--kv-cache", "fp8"};
        ChatCliConfig c_ik{};
        if (parse_chat_cli_args(2, const_cast<char**>(av_ik_def), ChatSubcmdKind::kChatInkling2, &c_ik, &err) != 0 ||
            c_ik.kv_dtype != STRATA_INKLING2_KV_DTYPE_F16 || c_ik.draft_k != 7U || c_ik.int8_prefill ||
            !c_ik.adaptive_k || !c_ik.frspec ||
            parse_chat_cli_args(6, const_cast<char**>(av_ik_q8), ChatSubcmdKind::kChatInkling2, &c_ik, &err) != 0 ||
            c_ik.kv_dtype != STRATA_INKLING2_KV_DTYPE_Q8_0 || c_ik.draft_k != 7U ||
            parse_chat_cli_args(4, const_cast<char**>(av_ik_q80), ChatSubcmdKind::kChatInkling2, &c_ik, &err) != 0 ||
            c_ik.kv_dtype != STRATA_INKLING2_KV_DTYPE_Q8_0 ||
            parse_chat_cli_args(4, const_cast<char**>(av_ik_f16), ChatSubcmdKind::kChatInkling2, &c_ik, &err) != 0 ||
            c_ik.kv_dtype != STRATA_INKLING2_KV_DTYPE_F16 ||
            parse_chat_cli_args(4, const_cast<char**>(av_ik_bad), ChatSubcmdKind::kChatInkling2, &c_ik, &err) == 0) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: chat-inkling2 --kv-cache parse failed\n");
            ++fails;
        }

        const char* av_opt_fl[] = {"strata_cli", "chat-flash", "--int8-prefill", "--no-adaptive-k", "--no-frspec"};
        const char* av_opt_gl[] = {"strata_cli", "chat-glm53", "--int8-prefill", "--no-adaptive-k", "--no-frspec"};
        const char* av_opt_ik[] = {
            "strata_cli", "chat-inkling2", "--int8-prefill", "--no-adaptive-k", "--no-frspec",
            "--no-tree-spec", "--no-bitplane-head", "--image", "/tmp/test.png"
        };
        ChatCliConfig c_opt{};
        if (parse_chat_cli_args(5, const_cast<char**>(av_opt_fl), ChatSubcmdKind::kChatFlash, &c_opt, &err) != 0 ||
            !c_opt.int8_prefill || !c_opt.int8_prefill_set || c_opt.adaptive_k || !c_opt.adaptive_k_set ||
            c_opt.frspec || !c_opt.frspec_set ||
            parse_chat_cli_args(5, const_cast<char**>(av_opt_gl), ChatSubcmdKind::kChatGlm53, &c_opt, &err) != 0 ||
            !c_opt.int8_prefill || c_opt.adaptive_k || c_opt.frspec ||
            parse_chat_cli_args(9, const_cast<char**>(av_opt_ik), ChatSubcmdKind::kChatInkling2, &c_opt, &err) != 0 ||
            !c_opt.int8_prefill || c_opt.adaptive_k || c_opt.frspec ||
            c_opt.tree_spec || c_opt.bitplane_head ||
            c_opt.initial_images.size() != 1U || c_opt.initial_images[0] != "/tmp/test.png") {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: chat-* option parse failed\n");
            ++fails;
        }
    }

    /* 2. Slash Command Parser Tests */
    {
        ChatCliConfig sc{};
        std::string msg;
        if (parse_slash_command("Hello world", true, &sc, &msg) != SlashCmdAction::kNotSlash) ++fails;
        if (parse_slash_command("/image /tmp/demo.png", true, &sc, &msg) != SlashCmdAction::kImage ||
            msg != "/tmp/demo.png") ++fails;
        if (parse_slash_command("/image", true, &sc, &msg) != SlashCmdAction::kError) ++fails;
        if (parse_slash_command("/no-think", true, &sc, &msg) != SlashCmdAction::kHandled || sc.enable_thinking) ++fails;
        if (parse_slash_command("/think xhigh", true, &sc, &msg) != SlashCmdAction::kHandled ||
            !sc.enable_thinking || sc.effort != "xhigh") ++fails;
        if (parse_slash_command("/think high", true, &sc, &msg) != SlashCmdAction::kHandled ||
            !sc.enable_thinking || sc.effort != "high") ++fails;
        if (parse_slash_command("/mode adaptive", true, &sc, &msg) != SlashCmdAction::kMode ||
            sc.glm53_mode != STRATA_GLM53_MODE_ADAPTIVE) ++fails;
        if (parse_slash_command("/mode lossless", true, &sc, &msg) != SlashCmdAction::kMode ||
            sc.glm53_mode != STRATA_GLM53_MODE_LOSSLESS) ++fails;
        if (parse_slash_command("/draft-k 4", true, &sc, &msg) != SlashCmdAction::kHandled || sc.draft_k != 4U) ++fails;
        if (parse_slash_command("/draft-k 3", false, &sc, &msg) != SlashCmdAction::kHandled || sc.draft_k != 0U) ++fails;
        if (parse_slash_command("/temp 0.75", true, &sc, &msg) != SlashCmdAction::kHandled ||
            sc.greedy || std::fabs(sc.temperature - 0.75f) > 1e-5f) ++fails;
        if (parse_slash_command("/greedy", true, &sc, &msg) != SlashCmdAction::kHandled || !sc.greedy) ++fails;
        if (parse_slash_command("/max-tokens 1024", true, &sc, &msg) != SlashCmdAction::kHandled || sc.max_tokens != 1024U) ++fails;
        if (parse_slash_command("/max-tokens 0", true, &sc, &msg) != SlashCmdAction::kHandled || sc.max_tokens != 0U) ++fails;
        if (parse_slash_command("/system Act as kernel debugger", true, &sc, &msg) != SlashCmdAction::kSystem ||
            sc.system_prompt != "Act as kernel debugger") ++fails;
        if (parse_slash_command("/undo", true, &sc, &msg) != SlashCmdAction::kUndo) ++fails;
        if (parse_slash_command("/reset", true, &sc, &msg) != SlashCmdAction::kReset) ++fails;
        if (parse_slash_command("/exit", true, &sc, &msg) != SlashCmdAction::kExit) ++fails;
        if (parse_slash_command("/quit", true, &sc, &msg) != SlashCmdAction::kExit) ++fails;
        if (parse_slash_command("/unknown-cmd", true, &sc, &msg) != SlashCmdAction::kError) ++fails;
    }

    /* 3. Turn-1 ChatML / GLM-5.3 Exactness vs Official Single-Turn Formatters & Compaction Tests */
    {
        char ref_buf[2048];
        size_t ref_len = 0U;
        if (strata_qwen_flash_format_chat_prompt(
                "Compute 6 * 7.", "Be terse.", true, "low", ref_buf, sizeof(ref_buf), &ref_len) != 0) {
            ++fails;
        } else {
            const std::string repl_t1 =
                format_flash_system_block("Be terse.", true, "low") +
                "<|im_start|>user\nCompute 6 * 7.<|im_end|>\n<|im_start|>assistant\n<think>\n";
            if (repl_t1 != std::string(ref_buf, ref_len)) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: Flash Turn-1 ChatML mismatch\n");
                ++fails;
            }
        }

        if (strata_qwen38_format_chat_prompt(
                "Compute 6 * 7.", "Be terse.", false, ref_buf, sizeof(ref_buf), &ref_len) != 0) {
            ++fails;
        } else {
            const std::string repl_q38 =
                format_qwen38_system_block("Be terse.") +
                "<|im_start|>user\nCompute 6 * 7.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            if (repl_q38 != std::string(ref_buf, ref_len)) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: Qwen38 Turn-1 ChatML mismatch\n");
                ++fails;
            }
        }

        if (strata_glm53_format_chat_prompt(
                "Compute 6 * 7.", "Be terse.", false, "high", ref_buf, sizeof(ref_buf), &ref_len) != 0) {
            ++fails;
        } else {
            const std::string repl_glm_nt =
                format_glm53_system_block("Be terse.", "high") +
                "<|user|>Compute 6 * 7.<|assistant|><think></think>";
            if (repl_glm_nt != std::string(ref_buf, ref_len)) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: GLM-5.3 Turn-1 (--no-think) mismatch\n");
                ++fails;
            }
        }

        if (strata_glm53_format_chat_prompt(
                "Compute 6 * 7.", "", true, "low", ref_buf, sizeof(ref_buf), &ref_len) != 0) {
            ++fails;
        } else {
            const std::string repl_glm_th =
                format_glm53_system_block("", "low") +
                "<|user|>Compute 6 * 7.<|assistant|><think>";
            if (repl_glm_th != std::string(ref_buf, ref_len)) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: GLM-5.3 Turn-1 (--think) mismatch\n");
                ++fails;
            }
        }

        if (strata_glm53_format_chat_turn_continuation("Next question?", false, ref_buf, sizeof(ref_buf), &ref_len) != 0 ||
            std::string(ref_buf, ref_len) != "<|user|>Next question?<|assistant|><think></think>") {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: GLM-5.3 Turn-2 continuation mismatch\n");
            ++fails;
        }

        const std::string compacted = build_committed_assistant_suffix("step 1", "42", /*keep_think=*/false);
        const std::string preserved = build_committed_assistant_suffix("step 1", "42", /*keep_think=*/true);
        if (compacted != "42<|im_end|>\n" || preserved != "<think>\nstep 1\n</think>\n\n42<|im_end|>\n") {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: <think> compaction suffix mismatch\n");
            ++fails;
        }

        const std::string glm_comp = build_glm53_committed_assistant_suffix("step 1", "42", true, false);
        const std::string glm_keep = build_glm53_committed_assistant_suffix("step 1", "42", true, true);
        if (glm_comp != "<think></think>42" || glm_keep != "<think>step 1</think>42") {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: GLM-5.3 <think> compaction suffix mismatch\n");
            ++fails;
        }
    }

    /* 4. Step 4 Compaction CLI Flag Parsing, Max-Seq-Len Resolution & Slash Command Tests */
    {
        ChatCliConfig def{};
        if (def.compact_mode != STRATA_COMPACT_OFF || def.compact_pool != 32768U ||
            def.compact_window != 32768U || std::fabs(def.compact_watermark - 0.80f) > 1e-6f ||
            def.verify_paged_kv || def.verify_shadow_swap ||
            chat_resolve_flash_max_seq_len(def) != STRATA_QWEN_FLASH_DEFAULT_MAX_SEQ_LEN) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: Step 4 default config mismatch\n");
            ++fails;
        }

        std::string err;
        const char* av_sh[] = {
            "strata_cli", "chat-flash", "--compact-mode", "shadow", "--compact-pool", "4096",
            "--compact-window", "40000", "--compact-watermark", "0.75"
        };
        ChatCliConfig c_sh{};
        if (parse_chat_cli_args(10, const_cast<char**>(av_sh), ChatSubcmdKind::kChatFlash, &c_sh, &err) != 0 ||
            c_sh.compact_mode != STRATA_COMPACT_SHADOW || c_sh.compact_pool != 4096U ||
            c_sh.compact_window != 40000U || std::fabs(c_sh.compact_watermark - 0.75f) > 1e-6f ||
            chat_resolve_flash_max_seq_len(c_sh) != 80000U) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: --compact-mode shadow / 2*window seq_len failed\n");
            ++fails;
        }
        c_sh.compact_window = 16384U;
        if (chat_resolve_flash_max_seq_len(c_sh) != 65536U) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: shadow max(65536, 2*window) floor failed\n");
            ++fails;
        }

        const char* av_pkv[] = {"strata_cli", "chat-flash", "--verify-paged-kv"};
        ChatCliConfig c_pkv{};
        if (parse_chat_cli_args(3, const_cast<char**>(av_pkv), ChatSubcmdKind::kChatFlash, &c_pkv, &err) != 0 ||
            !c_pkv.verify_paged_kv || !c_pkv.greedy || chat_resolve_flash_max_seq_len(c_pkv) != 20480U) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: --verify-paged-kv defaults failed\n");
            ++fails;
        }

        const char* av_vss[] = {"strata_cli", "chat-flash", "--verify-shadow-swap"};
        ChatCliConfig c_vss{};
        if (parse_chat_cli_args(3, const_cast<char**>(av_vss), ChatSubcmdKind::kChatFlash, &c_vss, &err) != 0 ||
            !c_vss.verify_shadow_swap || !c_vss.greedy || c_vss.compact_mode != STRATA_COMPACT_SHADOW ||
            c_vss.compact_window != kVerifyShadowWindow || chat_resolve_flash_max_seq_len(c_vss) != 65536U) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: --verify-shadow-swap defaults failed\n");
            ++fails;
        }

        const char* av_vsh[] = {"strata_cli", "chat-flash", "--compact-mode", "hybrid",
                                "--compact-window", "2048", "--verify-shadow-swap"};
        ChatCliConfig c_vsh{};
        if (parse_chat_cli_args(7, const_cast<char**>(av_vsh), ChatSubcmdKind::kChatFlash, &c_vsh, &err) != 0 ||
            c_vsh.compact_mode != STRATA_COMPACT_HYBRID || c_vsh.compact_window != 2048U) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: --verify-shadow-swap hybrid override failed\n");
            ++fails;
        }

        const char* bad_wm_lo[] = {"strata_cli", "chat-flash", "--compact-watermark", "0.49"};
        const char* bad_wm_hi[] = {"strata_cli", "chat-flash", "--compact-watermark", "0.96"};
        const char* bad_wm_na[] = {"strata_cli", "chat-flash", "--compact-watermark", "nan"};
        const char* bad_mode[]  = {"strata_cli", "chat-flash", "--compact-mode", "turbo"};
        const char* bad_pool[]  = {"strata_cli", "chat-flash", "--compact-pool", "2048"};
        const char* bad_win[]   = {"strata_cli", "chat-flash", "--compact-window", "64"};
        ChatCliConfig c_bad{};
        if (parse_chat_cli_args(4, const_cast<char**>(bad_wm_lo), ChatSubcmdKind::kChatFlash, &c_bad, &err) == 0 ||
            parse_chat_cli_args(4, const_cast<char**>(bad_wm_hi), ChatSubcmdKind::kChatFlash, &c_bad, &err) == 0 ||
            parse_chat_cli_args(4, const_cast<char**>(bad_wm_na), ChatSubcmdKind::kChatFlash, &c_bad, &err) == 0 ||
            parse_chat_cli_args(4, const_cast<char**>(bad_mode), ChatSubcmdKind::kChatFlash, &c_bad, &err) == 0 ||
            parse_chat_cli_args(4, const_cast<char**>(bad_pool), ChatSubcmdKind::kChatFlash, &c_bad, &err) == 0 ||
            parse_chat_cli_args(4, const_cast<char**>(bad_win), ChatSubcmdKind::kChatFlash, &c_bad, &err) == 0) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: accepted invalid Step 4 CLI flag\n");
            ++fails;
        }

        ChatCliConfig sc{};
        std::string msg;
        if (parse_slash_command("/compact-mode shadow", true, &sc, &msg) != SlashCmdAction::kCompactMode ||
            sc.compact_mode != STRATA_COMPACT_SHADOW) ++fails;
        if (parse_slash_command("/compact-mode paged", true, &sc, &msg) != SlashCmdAction::kCompactMode ||
            sc.compact_mode != STRATA_COMPACT_PAGED) ++fails;
        if (parse_slash_command("/compact-mode hybrid", true, &sc, &msg) != SlashCmdAction::kCompactMode ||
            sc.compact_mode != STRATA_COMPACT_HYBRID) ++fails;
        if (parse_slash_command("/compact-mode off", true, &sc, &msg) != SlashCmdAction::kCompactMode ||
            sc.compact_mode != STRATA_COMPACT_OFF) ++fails;
        if (parse_slash_command("/compact-mode", true, &sc, &msg) != SlashCmdAction::kError) ++fails;
        if (parse_slash_command("/compact-mode invalid", true, &sc, &msg) != SlashCmdAction::kError) ++fails;
        if (parse_slash_command("/compact-stats", true, &sc, &msg) != SlashCmdAction::kCompactStats) ++fails;
        if (parse_slash_command("/compact", true, &sc, &msg) != SlashCmdAction::kCompact) ++fails;
        if (parse_slash_command("/undo-compact", true, &sc, &msg) != SlashCmdAction::kUndoCompact) ++fails;
        if (parse_slash_command("/help", true, &sc, &msg) != SlashCmdAction::kHandled ||
            msg.find("/compact-mode") == std::string::npos || msg.find("/undo-compact") == std::string::npos) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: /help missing Step 4 commands\n");
            ++fails;
        }
    }

    /* 5. EWMA Velocity Predictor, Immutable Episode Ledger Distiller & Overlap Split Selection */
    {
        CompactVelocityPredictor pred;
        if (std::isfinite(pred.k_rem(800.0, 100.0))) ++fails;
        pred.observe(200.0);
        if (std::fabs(pred.v - 200.0) > 1e-9 || std::fabs(pred.s2) > 1e-9 ||
            std::fabs(pred.v_plus() - 200.0) > 1e-9 || std::fabs(pred.k_rem(800.0, 200.0) - 3.0) > 1e-9) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: EWMA initial sample math mismatch\n");
            ++fails;
        }
        pred.observe(400.0);
        const double exp_v  = 0.30 * 400.0 + 0.70 * 200.0;                  /* 260.0 */
        const double exp_s2 = 0.30 * (400.0 - exp_v) * (400.0 - exp_v);     /* 5880.0 */
        const double exp_vp = exp_v + 1.5 * std::sqrt(exp_s2);
        if (std::fabs(pred.v - exp_v) > 1e-9 || std::fabs(pred.s2 - exp_s2) > 1e-6 ||
            std::fabs(pred.v_plus() - exp_vp) > 1e-6 ||
            std::fabs(pred.k_rem(800.0, 600.0) - (200.0 / exp_vp)) > 1e-9 ||
            pred.k_rem(800.0, 820.0) != 0.0) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: EWMA step-2 math mismatch\n");
            ++fails;
        }

        const std::string u = "Store VAULT_CODE = 4417, PROJECT: HELIOS-9, OFFSET=0x2A7, max_seq_len=32768, TAG=\"alpha-7\".";
        const std::string r = "Confirmed VAULT_CODE = 4417 for HELIOS-9 at 0x2A7 ( AURORA = 73915 ).";
        const LedgerEntry e1 = ledger_distill_turn(0U, u, r);
        const LedgerEntry e1_again = ledger_distill_turn(0U, u, r);
        auto has = [](const std::vector<std::string>& v, const std::string& x) {
            return std::find(v.begin(), v.end(), x) != v.end();
        };
        if (e1.text != e1_again.text ||
            !has(e1.facts.kv, "VAULT_CODE=4417") || !has(e1.facts.kv, "PROJECT=HELIOS-9") ||
            !has(e1.facts.kv, "OFFSET=0x2A7") || !has(e1.facts.kv, "max_seq_len=32768") ||
            !has(e1.facts.kv, "TAG=\"alpha-7\"") || !has(e1.facts.kv, "AURORA=73915") ||
            !has(e1.facts.ids, "VAULT_CODE") || !has(e1.facts.ids, "HELIOS-9") ||
            !has(e1.facts.ids, "max_seq_len") || !has(e1.facts.hex, "0x2A7") ||
            !has(e1.facts.nums, "4417") || !has(e1.facts.nums, "73915")) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: ledger distiller dropped exact facts (%s)\n",
                         e1.text.c_str());
            ++fails;
        }
        /* Re-extraction fixpoint: every KV / hex / identifier survives parsing the rendered ledger line */
        const LedgerFacts roundtrip = ledger_extract_facts(e1.text);
        for (const std::string& kv : e1.facts.kv) {
            if (!has(roundtrip.kv, kv)) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: ledger KV not idempotent: %s\n", kv.c_str());
                ++fails;
            }
        }
        for (const std::string& hx : e1.facts.hex) {
            if (!has(roundtrip.hex, hx)) ++fails;
        }
        const LedgerEntry e2 = ledger_distill_turn(1U, "Check BOREALIS = 28461", "Stored 28461.");
        const LedgerEntry e_urand = ledger_distill_turn(
            2U, "Remember the random passkey 4f9a2c81b704e5d6 and token k9X2mP7qR4.", "Stored.");
        if (!has(e_urand.facts.hex, "4f9a2c81b704e5d6") || !has(e_urand.facts.ids, "k9X2mP7qR4")) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: ledger missed urandom hex/alnum passkey\n");
            ++fails;
        }
        const std::vector<LedgerEntry> led = {e1, e2};
        const std::string sys1 = ledger_build_system_content("Sys", led, 1U);
        const std::string sys2 = ledger_build_system_content("Sys", led, 2U);
        if (sys2.compare(0U, sys1.size(), sys1) != 0 || sys2.find(e2.text) == std::string::npos) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: ledger prefix not immutable across splits\n");
            ++fails;
        }

        /* Overlap split selection */
        if (select_overlap_split({300U}, 0U, 1000U) != -1) ++fails;
        /* cap = 250; turns [200, 180, 120, 110]: tail T2+T3 = 230 <= 250, adding T1 -> 410 > 250 => split = 2 */
        if (select_overlap_split({200U, 180U, 120U, 110U}, 0U, 1000U) != 2) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: select_overlap_split multi-turn tail failed\n");
            ++fails;
        }
        /* Huge last turn (> 0.25*window) still keeps >= 1 verbatim turn */
        if (select_overlap_split({200U, 400U}, 0U, 1000U) != 1) ++fails;
        /* Respects non-zero history floor: floor=2 with 3 turns total -> only 1 turn above floor => -1 */
        if (select_overlap_split({200U, 180U, 120U}, 2U, 1000U) != -1) ++fails;
        /* floor=2 with 4 turns [.., .., 100, 90] and cap=250 -> must still fold >= 1 turn above floor => split = 3 */
        if (select_overlap_split({200U, 180U, 100U, 90U}, 2U, 1000U) != 3) ++fails;
    }

    /* 6. End-to-End Zero-RSS Mock CompactionController + CooperativeLineReader Test */
    {
        MockCompactBackend mb;
        ChatCliConfig mcfg{};
        mcfg.enable_thinking   = false;
        mcfg.greedy            = true;
        mcfg.compact_mode      = STRATA_COMPACT_SHADOW;
        mcfg.compact_window    = 1400U;
        mcfg.compact_watermark = 0.75f; /* W_swap = 1050 */
        ChatReplSession sess(&mb, mcfg);
        CompactionController comp(&mb, &sess);
        comp.set_echo_events(false);
        comp.set_quiet(true);
        comp.set_slice_tokens(32U);
        std::string msg;
        if (!comp.set_mode(STRATA_COMPACT_SHADOW, &msg)) ++fails;

        const std::string filler(260U, 'x');
        const std::string prompts[5] = {
            "VAULT_CODE = 4417 PROJECT = HELIOS-9 " + filler,
            "OFFSET = 0x2A7 " + filler,
            "status a " + filler,
            "status b " + filler,
            "recall?"};
        for (int ti = 0; ti < 4; ++ti) {
            comp.pre_turn();
            if (sess.run_turn(prompts[ti], false, false, nullptr) != 0) ++fails;
            comp.on_turn_committed();
            if (ti == 2) {
                /* Partially warm 2 slices on T3, then mirror T4 onto the in-flight shadow before finishing! */
                (void)comp.idle_quantum();
                (void)comp.idle_quantum();
            } else {
                (void)comp.run_idle_gap();
            }
        }
        if (comp.swaps() != 1U || comp.last_switchover_ms() >= 0.05 ||
            mb.bound_tokens() != sess.committed_transcript_ids()) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: mock RCU swap failed (swaps=%u, %.4f ms)\n",
                         comp.swaps(), comp.last_switchover_ms());
            ++fails;
        }
        /* /undo-compact + /compact round-trip */
        const uint32_t pre_pos  = comp.last_pre_swap_pos();
        const uint32_t post_pos = mb.seq_pos();
        double u_ms = 0.0;
        uint32_t p0 = 0U, p1 = 0U;
        if (!comp.undo_compact(&u_ms, &p0, &p1, &msg) || u_ms >= 0.05 || p1 != pre_pos ||
            !comp.force_compact(&msg) || mb.seq_pos() != post_pos) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: mock /undo-compact round-trip failed\n");
            ++fails;
        }
        /* Post-swap turn: 0 cold re-prefill tokens */
        const uint64_t pf0 = mb.prefill_tokens_total();
        const uint64_t sl0 = mb.slice_tokens_total();
        comp.pre_turn();
        ChatTurnRecord r5{};
        if (sess.run_turn(prompts[4], false, false, &r5) != 0) ++fails;
        const uint64_t cold = (mb.prefill_tokens_total() - pf0) -
                              (static_cast<uint64_t>(r5.prompt_delta_toks) + r5.committed_toks) +
                              (mb.slice_tokens_total() - sl0);
        if (cold != 0ULL) {
            std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: expected 0 post-swap cold tokens, got %llu\n",
                         static_cast<unsigned long long>(cold));
            ++fails;
        }

        /* CooperativeLineReader editing + poll(0) first-keystroke pause over a local pipe */
        int pfd[2] = {-1, -1};
        if (::pipe(pfd) == 0) {
            CooperativeLineReader rdr(pfd[0], /*tty_cbreak=*/false, /*echo=*/false);
            const char raw[] = "ab\177c\025hello\033[A world\n";
            rdr.feed(raw, sizeof(raw) - 1U);
            std::string got;
            if (!rdr.pop_ready(&got) || got != "hello world") {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: CooperativeLineReader edit failed (`%s`)\n",
                             got.c_str());
                ++fails;
            }
            /* Put one partial byte on the pipe -> read_line must pause idle quanta on that first byte */
            (void)!::write(pfd[1], "h", 1U);
            (void)!::write(pfd[1], "i\n", 2U);
            ::close(pfd[1]);
            if (!rdr.read_line(&comp, &got) || got != "hi" || rdr.read_line(&comp, &got)) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: CooperativeLineReader pipe read failed\n");
                ++fails;
            }
            ::close(pfd[0]);
        }

        /* Bracketed Paste Mode (`\033[200~...\033[201~`) with embedded LF and CRLF as a single turn */
        {
            CooperativeLineReader bp(-1, /*tty_cbreak=*/false, /*echo=*/false);
            const char pasted[] = "prefix \033[200~int a = 1;\nint b = 2;\r\nreturn a + b;\033[201~ suffix\n";
            bp.feed(pasted, sizeof(pasted) - 1U);
            std::string bp_got;
            if (!bp.pop_ready(&bp_got) ||
                bp_got != "prefix int a = 1;\nint b = 2;\nreturn a + b; suffix" ||
                bp.pop_ready(&bp_got)) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: Bracketed Paste Mode failed (`%s`)\n",
                             bp_got.c_str());
                ++fails;
            }
        }

        /* Slash `/save` & `/load`, `--load`, `--verify-save-load`, `chat_resolve_glm53_max_seq_len`, and mock `.strata_session` round-trip */
        {
            ChatCliConfig sc{};
            std::string smsg;
            if (parse_slash_command("/save /tmp/a.strata_session", true, &sc, &smsg) != SlashCmdAction::kSaveSession ||
                smsg != "/tmp/a.strata_session") ++fails;
            if (parse_slash_command("/load /tmp/a.strata_session", true, &sc, &smsg) != SlashCmdAction::kLoadSession ||
                smsg != "/tmp/a.strata_session") ++fails;
            if (parse_slash_command("/save", true, &sc, &smsg) != SlashCmdAction::kError) ++fails;
            if (parse_slash_command("/load", true, &sc, &smsg) != SlashCmdAction::kError) ++fails;

            const char* av_sl[] = {
                "strata_cli", "chat-glm53", "--compact-mode", "hybrid", "--compact-window", "40000",
                "--load", "/tmp/x.strata_session", "--verify-save-load"
            };
            ChatCliConfig c_sl{};
            std::string perr;
            if (parse_chat_cli_args(9, const_cast<char**>(av_sl), ChatSubcmdKind::kChatGlm53, &c_sl, &perr) != 0 ||
                !c_sl.verify_save_load || !c_sl.greedy || c_sl.load_session_path != "/tmp/x.strata_session" ||
                chat_resolve_glm53_max_seq_len(c_sl) != 80000U) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: GLM-5.3 hybrid/save-load CLI parse failed\n");
                ++fails;
            }

            /* Save post-swap mock session, wipe to seq_pos=0/off, load into paged/hybrid mode, verify bit-exact state */
            std::string unit_tmp([NSTemporaryDirectory() UTF8String]);
            if (!unit_tmp.empty() && unit_tmp.back() != '/') unit_tmp.push_back('/');
            std::string unit_dir_tmpl = unit_tmp + "strata_unit_sess_XXXXXX";
            std::vector<char> unit_dir_buf(unit_dir_tmpl.begin(), unit_dir_tmpl.end());
            unit_dir_buf.push_back('\0');
            char* unit_dir_cstr = ::mkdtemp(unit_dir_buf.data());
            const std::string unit_dir = unit_dir_cstr ? std::string(unit_dir_cstr) : std::string();
            const std::string tpath = unit_dir + "/unit.strata_session";
            if (unit_dir.empty()) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: mkdtemp failed\n");
                ++fails;
            }
            const std::vector<uint32_t> exp_toks = mb.bound_tokens();
            const size_t exp_turns               = sess.history().size();
            const size_t exp_floor               = sess.history_floor();
            const uint32_t exp_swaps             = comp.swaps();
            uint64_t sb = 0ULL, lb = 0ULL;
            double sms = 0.0, lms = 0.0;
            std::string io_err;
            if (!save_session_to_file(tpath, &mb, &sess, &comp, &sb, &sms, &io_err)) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: mock save_session_to_file failed: %s\n",
                             io_err.c_str());
                ++fails;
            }
            (void)sess.reset_session();
            comp.on_session_reset();
            (void)comp.set_mode(STRATA_COMPACT_OFF, &smsg);
            mb.reset();
            if (!load_session_from_file(tpath, &mb, &sess, &comp, &lb, &lms, &io_err) ||
                sb != lb || mb.bound_tokens() != exp_toks || sess.history().size() != exp_turns ||
                sess.history_floor() != exp_floor || comp.swaps() != exp_swaps || exp_swaps != 2U ||
                comp.mode() != STRATA_COMPACT_SHADOW) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: mock load_session_from_file mismatch (%s)\n",
                             io_err.c_str());
                ++fails;
            }

            /* R1: save_session_to_file must never truncate/delete an existing file when save is unsupported or fails,
             * and load_session_from_file must leave session->cfg() unchanged when load_state_fd fails. */
            const std::string victim_path = unit_dir + "/victim.txt";
            {
                std::ofstream vf(victim_path, std::ios::binary);
                vf << "KEEP_VICTIM_INTACT\n";
            }
            auto read_str_file = [](const std::string& p) -> std::string {
                std::ifstream ifs(p, std::ios::binary);
                if (!ifs) return std::string();
                return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            };
            mb.allow_save_ = false;
            if (save_session_to_file(victim_path, &mb, &sess, &comp, &sb, &sms, &io_err) ||
                read_str_file(victim_path) != "KEEP_VICTIM_INTACT\n") {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: R1 unsupported save altered/deleted victim\n");
                ++fails;
            }
            mb.allow_save_      = true;
            mb.fail_save_state_ = true;
            if (save_session_to_file(victim_path, &mb, &sess, &comp, &sb, &sms, &io_err) ||
                read_str_file(victim_path) != "KEEP_VICTIM_INTACT\n") {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: R1 failed save altered/deleted victim\n");
                ++fails;
            }
            mb.fail_save_state_ = false;

            const std::string trunc_path = unit_dir + "/trunc.strata_session";
            {
                std::string full_sess = read_str_file(tpath);
                if (full_sess.size() > 4U) full_sess.resize(full_sess.size() - 4U);
                std::ofstream tf(trunc_path, std::ios::binary);
                tf.write(full_sess.data(), static_cast<std::streamsize>(full_sess.size()));
            }
            sess.mutable_cfg().system_prompt = "SENTINEL_UNMODIFIED_SYS";
            sess.mutable_cfg().temperature   = 0.42f;
            if (load_session_from_file(trunc_path, &mb, &sess, &comp, &lb, &lms, &io_err) ||
                sess.cfg().system_prompt != "SENTINEL_UNMODIFIED_SYS" ||
                std::fabs(sess.cfg().temperature - 0.42f) > 1e-6f) {
                std::fprintf(stderr, "  [FAIL] strata_chat_repl_unit_tests: R1 failed load mutated session cfg\n");
                ++fails;
            }

            (void)::unlink(victim_path.c_str());
            (void)::unlink(trunc_path.c_str());
            (void)::unlink(tpath.c_str());
            if (!unit_dir.empty()) (void)::rmdir(unit_dir.c_str());
        }
    }

    return fails;
}

} /* extern "C" */

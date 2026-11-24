/*
 * STRATA-M5 v2.0 — thinkingmachines/Inkling-2-Mini (26B Looped MoE + Native NVFP4/MXFP8 + 3-Layer DSpark Block-8 MTP) CLI & Benchmark Suite
 * File: src/cli/strata_inkling2_bench.mm
 *
 * Implements:
 *   - `convert-inkling2`: Local HF safetensors -> zero-copy 16 KiB-aligned `inkling2_uma.strata` converter
 *   - `repack-inkling2`:  Fast local v1 -> v2 `-fast` mixed-precision repacker (Q4_A/Q8_0/NVFP4 + 32K FR-Spec)
 *   - `inspect-inkling2`: Zero-copy UMA bundle manifest & memory layout inspector
 *   - `run-inkling2`:     Interactive/file-prompt generation (`--no-think` vs `--think`, `--effort`,
 *                         `K=0` vs Native 3-Layer DSpark MTP `K=1..7`, Adaptive-K, FR-Spec, `--verify-k0`, `--dump-logits`)
 *   - `bench-inkling2`:   512 -> 32K+ long-context throughput, DRAM GB/s, UMA RSS, MajFlt & Passkey Needle suite
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_inkling2.h"
#include "../../include/strata_inkling2_vision.h"

#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using SteadyClock = std::chrono::steady_clock;

static double elapsed_ms(SteadyClock::time_point t0, SteadyClock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template <size_t K>
static double exact_ngram_dup_ratio(const std::vector<uint32_t>& ids) {
    if (ids.size() < K) return 0.0;
    const size_t total = ids.size() - K + 1U;
    std::vector<std::array<uint32_t, K>> grams(total);
    for (size_t i = 0U; i < total; ++i) {
        for (size_t j = 0U; j < K; ++j) {
            grams[i][j] = ids[i + j];
        }
    }
    std::sort(grams.begin(), grams.end());
    size_t uniq = 1U;
    for (size_t i = 1U; i < total; ++i) {
        if (grams[i] != grams[i - 1U]) ++uniq;
    }
    return 1.0 - (static_cast<double>(uniq) / static_cast<double>(total));
}

struct LoopSanityReport {
    double dup_8gram_ratio = 0.0;
    double dup_4gram_ratio = 0.0;
    bool   loop_warning    = false;
    bool   evaluated       = false;
};

static LoopSanityReport analyze_shingle_loops(const std::vector<uint32_t>& ids) {
    LoopSanityReport rep{};
    if (ids.size() < 48U) return rep;
    rep.evaluated       = true;
    rep.dup_8gram_ratio = exact_ngram_dup_ratio<8U>(ids);
    rep.dup_4gram_ratio = exact_ngram_dup_ratio<4U>(ids);
    rep.loop_warning    = (rep.dup_8gram_ratio > 0.35);
    return rep;
}

static uint64_t ik_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

static uint64_t ik_align_up_u64(uint64_t v, uint64_t align) {
    return (align == 0ULL) ? v : (((v + align - 1ULL) / align) * align);
}

static std::string ik_lower_ascii(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

static bool inkling2_is_eos_token(const StrataInkling2Config* cfg, uint32_t tok) {
    if (tok == STRATA_INKLING2_TOKEN_EOS || tok == STRATA_INKLING2_TOKEN_ENDOFTEXT) {
        return true;
    }
    if (cfg && tok == cfg->eos_token_id) {
        return true;
    }
    return false;
}

static bool inkling2_effort_is_valid(const std::string& raw) {
    const std::string e = ik_lower_ascii(raw);
    return (e == "none" || e == "off" || e == "minimal" || e == "low" ||
            e == "medium" || e == "high" || e == "xhigh" || e == "max");
}

static bool parse_inkling2_kv_cache(const std::string& raw, uint32_t* out_dtype) {
    const std::string s = ik_lower_ascii(raw);
    if (s == "f16" || s == "fp16") {
        if (out_dtype) *out_dtype = STRATA_INKLING2_KV_DTYPE_F16;
        return true;
    }
    if (s == "q8" || s == "q8_0") {
        if (out_dtype) *out_dtype = STRATA_INKLING2_KV_DTYPE_Q8_0;
        return true;
    }
    return false;
}

static const char* inkling2_kv_cache_label(uint32_t kv_dtype) {
    return (kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0) ? "q8" : "f16";
}

static std::string inkling2_effort_to_num_str(bool enable_thinking, const std::string& raw_effort) {
    if (!enable_thinking) return "0";
    const std::string e = raw_effort.empty() ? "medium" : ik_lower_ascii(raw_effort);
    if (e == "none" || e == "off") return "0";
    if (e == "minimal")            return "0.1";
    if (e == "low")                return "0.2";
    if (e == "medium")             return "0.7";
    if (e == "high")               return "0.9";
    if (e == "xhigh" || e == "max")return "0.99";
    return "0.7";
}

struct ProcessMemStats {
    double rss_gib            = 0.0; /* Host CPU resident_size */
    double gpu_wired_gib      = 0.0; /* Metal [device currentAllocatedSize] (UMA wired weights + KV + state) */
    double total_uma_rss_gib  = 0.0; /* Combined Process UMA Resident (GPU wired + Host RSS) */
    double phys_footprint_gib = 0.0;
    double peak_rss_gib       = 0.0;
    double swap_used_gib      = 0.0;
    long   majflt             = 0;
    long   minflt             = 0;
};

static ProcessMemStats query_process_mem_stats() {
    ProcessMemStats out{};
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

    mach_task_basic_info_data_t binfo{};
    mach_msg_type_number_t bcount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(),
                  MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&binfo),
                  &bcount) == KERN_SUCCESS) {
        out.rss_gib = static_cast<double>(binfo.resident_size) / kGiB;
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev) {
            out.gpu_wired_gib = static_cast<double>([dev currentAllocatedSize]) / kGiB;
        }
    }
    out.total_uma_rss_gib = out.gpu_wired_gib + out.rss_gib;

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

static bool read_text_file(const std::string& path, std::string* out) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    *out = oss.str();
    return true;
}

static bool encode_string_vec(
    const StrataInkling2Tokenizer* tok,
    const std::string& text,
    bool parse_special,
    std::vector<uint32_t>* out_ids
) {
    const uint32_t cap = static_cast<uint32_t>(std::max<size_t>(text.size() + 128U, 256U));
    out_ids->assign(cap, 0U);
    uint32_t n = 0U;
    if (strata_inkling2_encode(tok, text.c_str(), text.size(), parse_special, out_ids->data(), cap, &n) != 0) {
        return false;
    }
    out_ids->resize(n);
    return true;
}

static bool append_encoded(
    const StrataInkling2Tokenizer* tok,
    const std::string& text,
    bool parse_special,
    std::vector<uint32_t>* dst
) {
    if (text.empty()) return true;
    std::vector<uint32_t> tmp;
    if (!encode_string_vec(tok, text, parse_special, &tmp)) return false;
    dst->insert(dst->end(), tmp.begin(), tmp.end());
    return true;
}

struct EncodedCliImage {
    std::string        path;
    std::vector<float> emb_f32;
    uint32_t           num_tokens = 0U;
    uint32_t           grid_h     = 0U;
    uint32_t           grid_w     = 0U;
    double             encode_ms  = 0.0;
};

static bool encode_inkling2_chat_prompt_with_images(
    const StrataInkling2Tokenizer* tok,
    const std::string& user_prompt,
    const std::string& system_prompt,
    bool enable_thinking,
    const std::string& effort_raw,
    const std::vector<EncodedCliImage>& images,
    std::vector<uint32_t>* out_ids,
    std::vector<StrataInkling2ImageSpan>* out_spans
) {
    out_ids->clear();
    if (out_spans) out_spans->clear();
    if (!system_prompt.empty()) {
        if (!append_encoded(tok, "<|message_system|><|content_text|>", true, out_ids)) return false;
        if (!append_encoded(tok, system_prompt, false, out_ids)) return false;
        if (!append_encoded(tok, "<|end_message|>", true, out_ids)) return false;
    }
    const std::string eff_num = inkling2_effort_to_num_str(enable_thinking, effort_raw);
    if (!append_encoded(tok, "<|message_system|><|content_text|>", true, out_ids)) return false;
    if (!append_encoded(tok, "Thinking effort level: " + eff_num, false, out_ids)) return false;
    if (!append_encoded(tok, "<|end_message|>", true, out_ids)) return false;

    for (const EncodedCliImage& img : images) {
        if (img.num_tokens == 0U || img.emb_f32.empty()) return false;
        if (!append_encoded(tok, "<|message_user|><|content_image|>", true, out_ids)) return false;
        if (out_spans) {
            StrataInkling2ImageSpan sp{};
            sp.token_offset   = static_cast<uint32_t>(out_ids->size());
            sp.num_tokens     = img.num_tokens;
            sp.embeddings_f32 = img.emb_f32.data();
            out_spans->push_back(sp);
        }
        out_ids->insert(out_ids->end(), img.num_tokens, STRATA_INKLING2_TOKEN_IMAGE_PLACEHOLDER);
        if (!append_encoded(tok, "<|end_message|>", true, out_ids)) return false;
    }

    if (!user_prompt.empty() || images.empty()) {
        if (!append_encoded(tok, "<|message_user|><|content_text|>", true, out_ids)) return false;
        if (!append_encoded(tok, user_prompt, false, out_ids)) return false;
        if (!append_encoded(tok, "<|end_message|>", true, out_ids)) return false;
    }
    if (!enable_thinking || eff_num == "0") {
        if (!append_encoded(tok, "<|message_model|><|content_text|>", true, out_ids)) return false;
    }
    return !out_ids->empty();
}

static std::string decode_tokens_str(
    const StrataInkling2Tokenizer* tok,
    const std::vector<uint32_t>& ids,
    bool skip_special
) {
    if (ids.empty()) return {};
    const size_t cap = std::max<size_t>(ids.size() * 32U + 256U, 1024U);
    std::vector<char> buf(cap, '\0');
    size_t len = 0U;
    if (strata_inkling2_decode(
            tok, ids.data(), static_cast<uint32_t>(ids.size()), skip_special, buf.data(), cap, &len) != 0) {
        return {};
    }
    return std::string(buf.data(), len);
}

struct Inkling2CliStreamState {
    StrataInkling2StreamDecoder dec{};
    bool in_header = false;
    bool in_think  = false;
    bool in_tool   = false;
};

static void stream_emit_token(
    const StrataInkling2Tokenizer* tok,
    Inkling2CliStreamState* st,
    uint32_t token_id
) {
    if (token_id == STRATA_INKLING2_TOKEN_MSG_MODEL) {
        strata_inkling2_stream_decoder_init(&st->dec);
        st->in_header = true;
        return;
    }
    if (token_id == STRATA_INKLING2_TOKEN_CONTENT_THINK) {
        strata_inkling2_stream_decoder_init(&st->dec);
        st->in_header = false;
        st->in_think  = true;
        st->in_tool   = false;
        std::fputs("<think>\n", stdout);
        std::fflush(stdout);
        return;
    }
    if (token_id == STRATA_INKLING2_TOKEN_CONTENT_TEXT ||
        token_id == STRATA_INKLING2_TOKEN_TOOL_ERROR ||
        token_id == STRATA_INKLING2_TOKEN_INVOKE_TEXT) {
        strata_inkling2_stream_decoder_init(&st->dec);
        st->in_header = false;
        st->in_think  = false;
        st->in_tool   = false;
        return;
    }
    if (token_id == STRATA_INKLING2_TOKEN_INVOKE_JSON) {
        strata_inkling2_stream_decoder_init(&st->dec);
        st->in_header = false;
        st->in_think  = false;
        st->in_tool   = true;
        std::fputs("\n[tool_call] ", stdout);
        std::fflush(stdout);
        return;
    }
    if (token_id == STRATA_INKLING2_TOKEN_END_MSG) {
        strata_inkling2_stream_decoder_init(&st->dec);
        if (st->in_think) {
            std::fputs("\n</think>\n\n", stdout);
            std::fflush(stdout);
        } else if (st->in_tool) {
            std::fputc('\n', stdout);
            std::fflush(stdout);
        }
        st->in_header = false;
        st->in_think  = false;
        st->in_tool   = false;
        return;
    }
    if (st->in_header) {
        /* Suppress optional model/tool name prefix between `<|message_model|>` and `<|content_*|>` */
        return;
    }
    char piece[256];
    size_t piece_len = 0U;
    if (strata_inkling2_stream_decode_step(
            tok, &st->dec, token_id, /*skip_special=*/true, piece, sizeof(piece), &piece_len) == 0 &&
        piece_len > 0U) {
        std::fwrite(piece, 1U, piece_len, stdout);
        std::fflush(stdout);
    }
}

static std::vector<uint32_t> parse_u32_csv(const std::string& csv) {
    std::vector<uint32_t> out;
    size_t pos = 0U;
    while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        std::string item = (comma == std::string::npos) ? csv.substr(pos) : csv.substr(pos, comma - pos);
        if (!item.empty()) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(item.c_str(), &endp, 10);
            if (endp != item.c_str()) {
                out.push_back(static_cast<uint32_t>(v));
            }
        }
        if (comma == std::string::npos) break;
        pos = comma + 1U;
    }
    return out;
}

static void run_warmup_pass(
    const StrataInkling2Bundle* bundle,
    StrataInkling2GpuEngine* eng
) {
    @autoreleasepool {
        strata_inkling2_gpu_reset(eng);
        strata_inkling2_gpu_penalties_reset(eng);
        strata_inkling2_gpu_reset_counters(eng);

        const uint32_t warmup_ids[8] = {
            STRATA_INKLING2_TOKEN_MSG_SYSTEM,
            STRATA_INKLING2_TOKEN_CONTENT_TEXT,
            100U,
            STRATA_INKLING2_TOKEN_END_MSG,
            STRATA_INKLING2_TOKEN_MSG_USER,
            STRATA_INKLING2_TOKEN_CONTENT_TEXT,
            100U,
            STRATA_INKLING2_TOKEN_END_MSG
        };
        StrataInkling2SamplingParams greedy{};
        greedy.greedy      = true;
        greedy.temperature = 0.0f;
        greedy.rng_seed    = 1ULL;

        uint32_t next_tok = 0U;
        (void)strata_inkling2_gpu_prefill(eng, warmup_ids, 8U, &greedy, &next_tok, nullptr);
        uint32_t next2 = 0U;
        (void)strata_inkling2_gpu_decode_step(eng, next_tok, &greedy, &next2, nullptr);
        if (bundle->hdr.mtp.enabled) {
            uint32_t emitted[STRATA_INKLING2_MAX_SPEC_DRAFT_K + 2U] = {};
            uint32_t n_emit = 0U;
            uint32_t n_acc  = 0U;
            (void)strata_inkling2_gpu_spec_decode_step(
                eng, next2, STRATA_INKLING2_MAX_SPEC_DRAFT_K, &greedy, emitted, &n_emit, &n_acc);
        }

        strata_inkling2_gpu_reset(eng);
        strata_inkling2_gpu_penalties_reset(eng);
        strata_inkling2_gpu_reset_counters(eng);
    }
}

/* Diverse technical corpus paragraphs for realistic Needle-in-a-Haystack construction (matches bench-flash). */
static const char* const kHaystackParagraphs[] = {
    "Modern unified memory architectures integrate central processor clusters, neural accelerators, and wide graphics cores onto a single coherent LPDDR5X memory fabric, eliminating PCIe bounce buffers during autoregressive inference.",
    "By mapping page-aligned immutable weight containers directly into GPU virtual address space through Mach virtual memory descriptors, zero-copy residency sets avoid duplicate host allocations across multi-tens-of-gigabyte models.",
    "Looped middle-core transformer architectures revisit a shared block of twelve physical layers across three recurrent passes with low-rank gated message controllers, tripling effective network depth while keeping DRAM footprint compact.",
    "Interleaving five local sliding-window attention layers with one global full-attention layer preserves exact long-range token retrieval across tens of thousands of tokens while bounding per-layer key-value cache growth.",
    "Relative-position bias attention scores token distances through learned low-dimensional projections and lookup tables without rotary phase aliasing, combined with per-query learned attention sinks and L2 query-key clipping.",
    "Causal depthwise short convolutions with kernel size four capture immediate multi-token n-gram context across key, value, attention output, and feed-forward residual streams using compact four-step rolling buffers.",
    "Mixture-of-experts feed-forward layers route each hidden representation to eight specialized experts out of one hundred twenty-eight candidates using float32 sigmoid scoring with learned routing biases.",
    "Softcapped SiTU gated feed-forward units bound both gate and up activations via smooth hyperbolic tangent ceilings to maintain numerical stability across deep looped recurrence passes without overflow.",
    "Block attention residuals aggregate checkpoints every six logical layer visits through unweighted root-mean-square normalized value mixing, allowing deep layers to selectively attend over earlier representations.",
    "Native four-bit floating-point expert quantization packs two E2M1 nibbles per byte alongside E4M3 block-sixteen micro-scales and float32 per-expert amax factors to cut memory bandwidth nearly fourfold.",
    "Mixed-precision late-layer MXFP8 quantization preserves full eight-bit dynamic range with shared UE8M0 power-of-two scales across the final four decoder layers where output distribution sensitivity peaks.",
    "Native three-layer DSpark block-eight multi-token prediction heads fuse hidden states tapped at four backbone depths through a recurrent Markov hidden head to draft up to seven future tokens per verification step.",
    "During speculative verification, evaluating the committed token alongside seven drafted candidates in a single batched pass amortizes expert router and weight memory traffic across multiple accepted tokens.",
    "Exact Leviathan rejection sampling and deterministic greedy prefix verification guarantee that speculative multi-token decoding follows the exact target distribution of the underlying language model.",
    "When a drafted token diverges from the main model verification pass, constant-time device-side state snapshots restore short-convolution histories, attention residual accumulators, and sliding-window cursors cleanly.",
    "Log-position query and relative-bias scaling smoothly sharpens global attention entropy beyond eight thousand tokens so needle retrieval remains crisp across ultra-long contexts.",
    "Cooperative tensor matrix primitives in Metal 4 schedule SIMD-group tiles across neural hardware execution units during batched prefill, reaching high compute utilization on dense and grouped projections.",
    "Final logit softcapping bounds vocabulary scores inside thirty before softmax, stabilizing sampling temperature dynamics and sliding-window presence and frequency repetition penalties.",
    "Distributed systems consensus protocols rely on monotonic ballot numbers and quorum intersections so safety invariants hold even under asynchronous packet reordering, crash-recovery cycles, and network partitions.",
    "Optimizing compilers lower static single-assignment intermediate representations through dominance frontiers, loop-invariant code motion, polyhedral tile fusion, and register-pressure-aware instruction scheduling.",
    "Log-structured merge trees batch random mutations into sequential immutable sorted string tables on flash storage while Bloom filters prune unnecessary block reads during point lookups.",
    "Elliptic-curve key exchange over prime fields derives shared ephemeral secrets with compact public keys, feeding authenticated encryption modes that protect confidentiality and ciphertext integrity simultaneously.",
    "Numerical Krylov subspace solvers accelerate ill-conditioned sparse linear systems via incomplete Cholesky preconditioners and orthogonal Arnoldi recurrences with bounded memory vectors.",
    "Operating system schedulers balance cache affinity against fairness across heterogeneous performance and efficiency cores by tracking runnable thread load averages and energy-delay heuristics.",
    "High-frequency phased-array transceivers steer constructive electromagnetic wavefronts electronically by calibrating sub-nanosecond phase shifts across aperture elements without mechanical gimbals.",
    "Columnar analytical query engines vectorize predicate evaluation directly over dictionary-encoded bit-packed pages, saturating SIMD lanes while minimizing branch mispredictions per tuple.",
    "Photolithographic extreme-ultraviolet scanners project reflective multilayer mask patterns through high-numerical-aperture vacuum optics to resolve nanoscale transistor fins and gate-all-around nanosheets.",
    "Transactional write-ahead logs guarantee physiological crash recovery by replaying redo records from fuzzy checkpoints before rolling back uncommitted loser transactions in reverse log-sequence order.",
    "Congestion-control algorithms estimate bottleneck bandwidth and minimum propagation round-trip delay to pace outgoing segments near Kleinrock's optimal operating point without inducing bufferbloat.",
    "Formal model checkers exhaustively explore reachable state transitions or encode bounded temporal logic properties into conjunctive normal form clauses solved via conflict-driven clause learning.",
    "Asynchronous double-buffered DMA pipelines overlap storage prefetch latency with arithmetic execution so compute units stay fed without stalling on high-latency memory hierarchies.",
    "Hardware page-aligned memory mappings prevent minor and major page faults during steady-state inference, ensuring deterministic tail latency across long autoregressive generation runs."
};

struct HaystackTokenizerCache {
    std::vector<std::vector<uint32_t>> para_tokens;
    bool                               initialized = false;
};

static bool init_haystack_cache(const StrataInkling2Tokenizer* tok, HaystackTokenizerCache* cache) {
    if (cache->initialized) return true;
    const size_t n_paras = sizeof(kHaystackParagraphs) / sizeof(kHaystackParagraphs[0]);
    cache->para_tokens.resize(n_paras);
    for (size_t i = 0U; i < n_paras; ++i) {
        std::string s = std::string(" [Archive Section ") + std::to_string(i + 1U) + "] " + kHaystackParagraphs[i] + "\n";
        if (!encode_string_vec(tok, s, /*parse_special=*/false, &cache->para_tokens[i]) || cache->para_tokens[i].empty()) {
            return false;
        }
    }
    cache->initialized = true;
    return true;
}

static void append_haystack_exact(
    const HaystackTokenizerCache& cache,
    uint32_t count,
    uint64_t* rng_state,
    std::vector<uint32_t>* dst
) {
    uint32_t rem = count;
    const size_t n_paras = cache.para_tokens.size();
    while (rem > 0U) {
        *rng_state = ik_splitmix64(*rng_state);
        const auto& blk = cache.para_tokens[static_cast<size_t>(*rng_state % n_paras)];
        if (blk.size() <= rem) {
            dst->insert(dst->end(), blk.begin(), blk.end());
            rem -= static_cast<uint32_t>(blk.size());
        } else {
            dst->insert(dst->end(), blk.begin(), blk.begin() + rem);
            rem = 0U;
        }
    }
}

static std::string deterministic_passkey_for_ctx_slot(uint32_t ctx_len, uint32_t slot_idx) {
    uint64_t h = ik_splitmix64(0x5354524154414D35ULL ^
                               (static_cast<uint64_t>(ctx_len) + static_cast<uint64_t>(slot_idx) * 0x9E3779B97F4A7C15ULL));
    uint32_t code = 1000000U + static_cast<uint32_t>(h % 9000000ULL);
    return std::to_string(code);
}

struct PasskeyPromptSpec {
    std::vector<uint32_t>    prompt_ids;
    std::vector<std::string> passkeys;
    std::vector<uint32_t>    needle_offsets;
    std::string              summary_label;
};

static bool build_passkey_prompt_exact(
    const StrataInkling2Tokenizer* tok,
    HaystackTokenizerCache* cache,
    uint32_t target_ctx,
    double custom_needle_depth,
    bool enable_thinking,
    const std::string& effort,
    PasskeyPromptSpec* out_spec
) {
    if (!init_haystack_cache(tok, cache) || !out_spec) return false;
    *out_spec = PasskeyPromptSpec{};

    const std::string eff_num = inkling2_effort_to_num_str(enable_thinking, effort);
    const std::string sys_hdr =
        "<|message_system|><|content_text|>Thinking effort level: " + eff_num + "<|end_message|>";
    const std::string asst_tail =
        (!enable_thinking || eff_num == "0") ? "<|message_model|><|content_text|>" : "";

    if (target_ctx >= 131072U && custom_needle_depth < 0.0) {
        const std::string key_a = deterministic_passkey_for_ctx_slot(target_ctx, 0U);
        const std::string key_b = deterministic_passkey_for_ctx_slot(target_ctx, 1U);
        const std::string key_c = deterministic_passkey_for_ctx_slot(target_ctx, 2U);

        const std::string prefix_str =
            sys_hdr +
            "<|message_user|><|content_text|>"
            "Read the following technical archive carefully. Three 7-digit STRATA benchmark reference numbers "
            "(ALPHA, BETA, and GAMMA) are recorded at early, middle, and late positions inside the archive.\n\n";
        const std::string needle_a =
            "\n[BENCHMARK ARCHIVE RECORD ALPHA] The exact 7-digit STRATA ALPHA reference number is " +
            key_a + ". Reference number ALPHA: " + key_a + ".\n";
        const std::string needle_b =
            "\n[BENCHMARK ARCHIVE RECORD BETA] The exact 7-digit STRATA BETA reference number is " +
            key_b + ". Reference number BETA: " + key_b + ".\n";
        const std::string needle_c =
            "\n[BENCHMARK ARCHIVE RECORD GAMMA] The exact 7-digit STRATA GAMMA reference number is " +
            key_c + ". Reference number GAMMA: " + key_c + ".\n";
        const std::string suffix_str =
            "\nQuestion: On the very first line, state the exact 7-digit STRATA reference numbers for ALPHA, BETA, and GAMMA "
            "(format: ALPHA=<digits>, BETA=<digits>, GAMMA=<digits>). "
            "Then write a detailed multi-paragraph technical analysis of unified memory architecture, looped MoE transformers, "
            "mixed NVFP4/MXFP8 quantization, and native multi-token speculative decoding.<|end_message|>" +
            asst_tail;

        std::vector<uint32_t> prefix_ids, na_ids, nb_ids, nc_ids, suffix_ids;
        if (!encode_string_vec(tok, prefix_str, /*parse_special=*/true, &prefix_ids)) return false;
        if (!encode_string_vec(tok, needle_a,   /*parse_special=*/false, &na_ids)) return false;
        if (!encode_string_vec(tok, needle_b,   /*parse_special=*/false, &nb_ids)) return false;
        if (!encode_string_vec(tok, needle_c,   /*parse_special=*/false, &nc_ids)) return false;
        if (!encode_string_vec(tok, suffix_str, /*parse_special=*/true, &suffix_ids)) return false;

        const uint32_t fixed = static_cast<uint32_t>(
            prefix_ids.size() + na_ids.size() + nb_ids.size() + nc_ids.size() + suffix_ids.size());
        if (target_ctx <= fixed) return false;

        const uint32_t hay_budget = target_ctx - fixed;
        const uint32_t seg0 = static_cast<uint32_t>(0.10 * static_cast<double>(hay_budget));
        const uint32_t seg1 = static_cast<uint32_t>(0.40 * static_cast<double>(hay_budget));
        const uint32_t seg2 = static_cast<uint32_t>(0.35 * static_cast<double>(hay_budget));
        const uint32_t seg3 = hay_budget - (seg0 + seg1 + seg2);

        uint64_t rng = 0xC0FFEE123456789ULL ^ static_cast<uint64_t>(target_ctx);
        auto& dst = out_spec->prompt_ids;
        dst.reserve(target_ctx);
        dst.insert(dst.end(), prefix_ids.begin(), prefix_ids.end());

        append_haystack_exact(*cache, seg0, &rng, &dst);
        const uint32_t off_a = static_cast<uint32_t>(dst.size());
        dst.insert(dst.end(), na_ids.begin(), na_ids.end());

        append_haystack_exact(*cache, seg1, &rng, &dst);
        const uint32_t off_b = static_cast<uint32_t>(dst.size());
        dst.insert(dst.end(), nb_ids.begin(), nb_ids.end());

        append_haystack_exact(*cache, seg2, &rng, &dst);
        const uint32_t off_c = static_cast<uint32_t>(dst.size());
        dst.insert(dst.end(), nc_ids.begin(), nc_ids.end());

        append_haystack_exact(*cache, seg3, &rng, &dst);
        dst.insert(dst.end(), suffix_ids.begin(), suffix_ids.end());

        out_spec->passkeys       = {key_a, key_b, key_c};
        out_spec->needle_offsets = {off_a, off_b, off_c};
        out_spec->summary_label  = key_a + "," + key_b + "," + key_c;
        return dst.size() == target_ctx;
    }

    const std::string passkey = deterministic_passkey_for_ctx_slot(target_ctx, 0U);
    std::string prefix_str;
    std::string needle_str;
    std::string suffix_str;

    if (target_ctx >= 256U) {
        prefix_str =
            sys_hdr +
            "<|message_user|><|content_text|>"
            "Read the following technical archive carefully. A 7-digit STRATA benchmark reference number is recorded inside one entry.\n\n";
        needle_str =
            "\n[BENCHMARK ARCHIVE RECORD] The exact 7-digit STRATA reference number for this document is " +
            passkey + ". Reference number: " + passkey + ".\n";
        suffix_str =
            "\nQuestion: On the very first line, state the exact 7-digit STRATA reference number recorded in the archive above. "
            "Then write a detailed multi-paragraph technical analysis of unified memory architecture, looped MoE transformers, "
            "mixed NVFP4/MXFP8 quantization, and native multi-token speculative decoding.<|end_message|>" +
            asst_tail;
    } else {
        prefix_str = sys_hdr + "<|message_user|><|content_text|>Archive:\n";
        needle_str = "\nReference number: " + passkey + ".\n";
        suffix_str = "\nState the 7-digit reference number first, then explain unified memory in detail.<|end_message|>" +
                     asst_tail;
    }

    std::vector<uint32_t> prefix_ids, needle_ids, suffix_ids;
    if (!encode_string_vec(tok, prefix_str, /*parse_special=*/true, &prefix_ids)) return false;
    if (!encode_string_vec(tok, needle_str, /*parse_special=*/false, &needle_ids)) return false;
    if (!encode_string_vec(tok, suffix_str, /*parse_special=*/true, &suffix_ids)) return false;

    const uint32_t fixed = static_cast<uint32_t>(prefix_ids.size() + needle_ids.size() + suffix_ids.size());
    auto& dst = out_spec->prompt_ids;
    dst.clear();
    dst.reserve(std::max<uint32_t>(target_ctx, fixed));

    uint32_t needle_off = 0U;
    if (target_ctx <= fixed) {
        dst.insert(dst.end(), prefix_ids.begin(), prefix_ids.end());
        needle_off = static_cast<uint32_t>(dst.size());
        dst.insert(dst.end(), needle_ids.begin(), needle_ids.end());
        dst.insert(dst.end(), suffix_ids.begin(), suffix_ids.end());
        out_spec->passkeys       = {passkey};
        out_spec->needle_offsets = {needle_off};
        out_spec->summary_label  = passkey;
        return true;
    }

    const double depth_frac   = (custom_needle_depth >= 0.0) ? std::clamp(custom_needle_depth, 0.01, 0.99) : 0.35;
    const uint32_t hay_budget = target_ctx - fixed;
    const uint32_t before_cnt = static_cast<uint32_t>(depth_frac * static_cast<double>(hay_budget));
    const uint32_t after_cnt  = hay_budget - before_cnt;

    uint64_t rng = 0xC0FFEE123456789ULL ^ static_cast<uint64_t>(target_ctx);
    dst.insert(dst.end(), prefix_ids.begin(), prefix_ids.end());
    append_haystack_exact(*cache, before_cnt, &rng, &dst);
    needle_off = static_cast<uint32_t>(dst.size());
    dst.insert(dst.end(), needle_ids.begin(), needle_ids.end());
    append_haystack_exact(*cache, after_cnt, &rng, &dst);
    dst.insert(dst.end(), suffix_ids.begin(), suffix_ids.end());

    out_spec->passkeys       = {passkey};
    out_spec->needle_offsets = {needle_off};
    out_spec->summary_label  = passkey;
    return dst.size() == target_ctx;
}

struct Inkling2RunResult {
    uint32_t              prompt_tokens          = 0U;
    uint32_t              emitted_total          = 0U;
    uint32_t              decode_emitted         = 0U;
    uint32_t              decode_steps           = 0U;
    uint32_t              mtp_drafted            = 0U;
    uint32_t              mtp_accepted           = 0U;
    double                ttft_ms                = 0.0;
    double                prefill_tok_s          = 0.0;
    double                decode_ms              = 0.0;
    double                decode_tok_s           = 0.0;
    double                phys_dram_gbps         = 0.0;
    double                eff_dram_gbps          = 0.0;
    uint64_t              decode_dram_bytes      = 0ULL;
    uint32_t              final_seq_pos          = 0U;
    double                active_state_mib       = 0.0;
    double                global_kv_active_mib   = 0.0;
    double                swa_kv_mib             = 0.0;
    double                sconv_mib              = 0.0;
    double                rss_gib                = 0.0; /* Total UMA (GPU Wired + Host RSS) */
    double                host_rss_gib           = 0.0; /* Host CPU resident_size */
    double                footprint_gib          = 0.0; /* GPU Wired GiB */
    double                peak_rss_gib           = 0.0;
    double                swap_delta_gib         = 0.0;
    long                  steady_majflt          = 0;
    std::vector<uint32_t> generated_ids;
};

struct PrefillReuseSlot {
    StrataInkling2Checkpoint* ckpt          = nullptr;
    bool                      valid         = false;
    uint32_t                  first_tok     = 0U;
    double                    ttft_ms       = 0.0;
    double                    prefill_tok_s = 0.0;
};

static int execute_inkling2_sequence(
    const StrataInkling2Bundle*         bundle,
    StrataInkling2GpuEngine*            eng,
    const StrataInkling2GpuInfo&        info,
    const std::vector<uint32_t>&        prompt_ids,
    uint32_t                            max_new_tokens,
    bool                                count_decode_only,
    uint32_t                            draft_k,
    const StrataInkling2SamplingParams& base_sampling,
    bool                                stop_on_eos,
    bool                                stream_stdout,
    Inkling2RunResult*                  res_out,
    PrefillReuseSlot*                   reuse_slot = nullptr,
    const std::string&                  dump_logits_path = {},
    const std::vector<StrataInkling2ImageSpan>* image_spans = nullptr
) {
    *res_out = Inkling2RunResult{};
    if (prompt_ids.empty()) return -1;

    const uint32_t eff_k = (bundle->hdr.mtp.enabled && info.mtp_enabled)
        ? std::min<uint32_t>(draft_k, STRATA_INKLING2_MAX_SPEC_DRAFT_K)
        : 0U;

    StrataInkling2SamplingParams step_samp = base_sampling;
    uint32_t first_tok = 0U;
    ProcessMemStats mem0{};
    StrataInkling2PerfCounters cnt_pre{};

    const bool want_logits_dump = !dump_logits_path.empty();
    std::vector<float> dumped_logits;

    if (!want_logits_dump && reuse_slot && reuse_slot->ckpt && reuse_slot->valid) {
        strata_inkling2_gpu_penalties_reset(eng);
        if (strata_inkling2_checkpoint_restore(eng, reuse_slot->ckpt) != 0) {
            return -1;
        }
        mem0 = query_process_mem_stats();
        strata_inkling2_gpu_get_counters(eng, &cnt_pre);
        first_tok              = reuse_slot->first_tok;
        res_out->prompt_tokens = static_cast<uint32_t>(prompt_ids.size());
        res_out->ttft_ms       = reuse_slot->ttft_ms;
        res_out->prefill_tok_s = reuse_slot->prefill_tok_s;
    } else {
        strata_inkling2_gpu_reset(eng);
        strata_inkling2_gpu_penalties_reset(eng);
        strata_inkling2_gpu_reset_counters(eng);

        if (want_logits_dump) {
            const uint32_t v_unpad = bundle->hdr.cfg.vocab_size_unpadded > 0U
                ? bundle->hdr.cfg.vocab_size_unpadded
                : 200058U;
            dumped_logits.assign(v_unpad, 0.0f);
        }

        mem0 = query_process_mem_stats();

        const auto t_pre0 = SteadyClock::now();
        int rc = 0;
        @autoreleasepool {
            if (image_spans && !image_spans->empty()) {
                rc = strata_inkling2_gpu_prefill_with_images(
                    eng,
                    prompt_ids.data(),
                    static_cast<uint32_t>(prompt_ids.size()),
                    image_spans->data(),
                    static_cast<uint32_t>(image_spans->size()),
                    &step_samp,
                    &first_tok,
                    want_logits_dump ? dumped_logits.data() : nullptr);
            } else {
                rc = strata_inkling2_gpu_prefill(
                    eng,
                    prompt_ids.data(),
                    static_cast<uint32_t>(prompt_ids.size()),
                    &step_samp,
                    &first_tok,
                    want_logits_dump ? dumped_logits.data() : nullptr);
            }
        }
        const auto t_pre1 = SteadyClock::now();
        if (rc != 0) return rc;

        if (want_logits_dump) {
            std::ofstream ofs(dump_logits_path, std::ios::out | std::ios::binary | std::ios::trunc);
            if (!ofs) {
                std::fprintf(stderr, "[run-inkling2] Failed to open --dump-logits path `%s`\n",
                             dump_logits_path.c_str());
                return -1;
            }
            ofs.write(reinterpret_cast<const char*>(dumped_logits.data()),
                      static_cast<std::streamsize>(dumped_logits.size() * sizeof(float)));
            if (!ofs) {
                std::fprintf(stderr, "[run-inkling2] Failed to write --dump-logits file `%s`\n",
                             dump_logits_path.c_str());
                return -1;
            }
            std::fprintf(stderr, "[run-inkling2] Wrote %zu F32 prefill logits (%zu bytes) to `%s`\n",
                         dumped_logits.size(), dumped_logits.size() * sizeof(float), dump_logits_path.c_str());
        }

        strata_inkling2_gpu_get_counters(eng, &cnt_pre);

        res_out->prompt_tokens = static_cast<uint32_t>(prompt_ids.size());
        res_out->ttft_ms       = elapsed_ms(t_pre0, t_pre1);
        res_out->prefill_tok_s = (res_out->ttft_ms > 0.0)
            ? (static_cast<double>(res_out->prompt_tokens) / (res_out->ttft_ms / 1000.0))
            : 0.0;

        if (reuse_slot && reuse_slot->ckpt) {
            if (strata_inkling2_checkpoint_save(eng, reuse_slot->ckpt) == 0) {
                reuse_slot->valid         = true;
                reuse_slot->first_tok     = first_tok;
                reuse_slot->ttft_ms       = res_out->ttft_ms;
                reuse_slot->prefill_tok_s = res_out->prefill_tok_s;
            }
        }
    }

    int rc = 0;

    Inkling2CliStreamState stream_st{};
    strata_inkling2_stream_decoder_init(&stream_st.dec);

    const uint32_t total_cap = count_decode_only ? (1U + max_new_tokens) : max_new_tokens;
    res_out->generated_ids.reserve(total_cap + 8U);

    bool hit_eos = false;
    if (total_cap > 0U) {
        if (stop_on_eos && inkling2_is_eos_token(&bundle->hdr.cfg, first_tok)) {
            hit_eos = true;
        } else {
            res_out->generated_ids.push_back(first_tok);
            res_out->emitted_total = 1U;
            if (stream_stdout) {
                stream_emit_token(bundle->tokenizer, &stream_st, first_tok);
            }
        }
    }

    uint32_t pending_tok = first_tok;
    const uint32_t target_decode_tokens = count_decode_only
        ? max_new_tokens
        : ((max_new_tokens > 0U) ? (max_new_tokens - 1U) : 0U);

    const auto t_dec0 = SteadyClock::now();
    while (!hit_eos && res_out->decode_emitted < target_decode_tokens) {
        @autoreleasepool {
            if (eff_k == 0U) {
                uint32_t next_tok = 0U;
                rc = strata_inkling2_gpu_decode_step(eng, pending_tok, &step_samp, &next_tok, nullptr);
                if (rc != 0) break;
                ++res_out->decode_steps;
                if (stop_on_eos && inkling2_is_eos_token(&bundle->hdr.cfg, next_tok)) {
                    hit_eos = true;
                    break;
                }
                res_out->generated_ids.push_back(next_tok);
                ++res_out->emitted_total;
                ++res_out->decode_emitted;
                if (stream_stdout) {
                    stream_emit_token(bundle->tokenizer, &stream_st, next_tok);
                }
                pending_tok = next_tok;
            } else {
                uint32_t emitted_buf[STRATA_INKLING2_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t num_emitted  = 0U;
                uint32_t num_accepted = 0U;
                rc = strata_inkling2_gpu_spec_decode_step(
                    eng, pending_tok, eff_k, &step_samp, emitted_buf, &num_emitted, &num_accepted);
                if (rc != 0) break;
                ++res_out->decode_steps;
                for (uint32_t i = 0U; i < num_emitted; ++i) {
                    const uint32_t tok = emitted_buf[i];
                    if (stop_on_eos && inkling2_is_eos_token(&bundle->hdr.cfg, tok)) {
                        hit_eos = true;
                        break;
                    }
                    res_out->generated_ids.push_back(tok);
                    ++res_out->emitted_total;
                    ++res_out->decode_emitted;
                    if (stream_stdout) {
                        stream_emit_token(bundle->tokenizer, &stream_st, tok);
                    }
                    if (res_out->decode_emitted >= target_decode_tokens) {
                        break;
                    }
                }
                if (!hit_eos && num_emitted > 0U) {
                    pending_tok = emitted_buf[num_emitted - 1U];
                }
            }
        }
    }
    const auto t_dec1 = SteadyClock::now();
    if (rc != 0) return rc;

    StrataInkling2PerfCounters cnt_end{};
    strata_inkling2_gpu_get_counters(eng, &cnt_end);
    const ProcessMemStats mem1 = query_process_mem_stats();

    res_out->decode_ms = elapsed_ms(t_dec0, t_dec1);
    res_out->decode_tok_s = (res_out->decode_ms > 0.0 && res_out->decode_emitted > 0U)
        ? (static_cast<double>(res_out->decode_emitted) / (res_out->decode_ms / 1000.0))
        : 0.0;

    res_out->mtp_drafted  = static_cast<uint32_t>(cnt_end.mtp_drafted_tokens - cnt_pre.mtp_drafted_tokens);
    res_out->mtp_accepted = static_cast<uint32_t>(cnt_end.mtp_accepted_tokens - cnt_pre.mtp_accepted_tokens);
    res_out->decode_dram_bytes = (cnt_end.physical_dram_bytes_read >= cnt_pre.physical_dram_bytes_read)
        ? (cnt_end.physical_dram_bytes_read - cnt_pre.physical_dram_bytes_read)
        : 0ULL;

    res_out->phys_dram_gbps = (res_out->decode_ms > 0.0)
        ? ((static_cast<double>(res_out->decode_dram_bytes) / 1e9) / (res_out->decode_ms / 1000.0))
        : 0.0;
    const uint64_t act_per_tok = (info.active_bytes_per_token > 0ULL)
        ? info.active_bytes_per_token
        : bundle->hdr.active_bytes_per_tok;
    res_out->eff_dram_gbps = (res_out->decode_ms > 0.0)
        ? ((static_cast<double>(res_out->decode_emitted) * static_cast<double>(act_per_tok) / 1e9) /
           (res_out->decode_ms / 1000.0))
        : 0.0;

    res_out->final_seq_pos = strata_inkling2_gpu_seq_pos(eng);
    constexpr double kMiB = 1024.0 * 1024.0;
    const double global_kv_frac = (info.max_seq_len > 0U)
        ? std::min<double>(1.0, static_cast<double>(res_out->final_seq_pos) / static_cast<double>(info.max_seq_len))
        : 0.0;
    res_out->global_kv_active_mib = (static_cast<double>(info.global_kv_bytes_allocated) * global_kv_frac) / kMiB;
    res_out->swa_kv_mib           = static_cast<double>(info.swa_kv_bytes_allocated) / kMiB;
    res_out->sconv_mib            = static_cast<double>(info.sconv_state_bytes) / kMiB;
    res_out->active_state_mib     = res_out->global_kv_active_mib + res_out->swa_kv_mib + res_out->sconv_mib;

    res_out->rss_gib        = mem1.total_uma_rss_gib;
    res_out->host_rss_gib   = mem1.rss_gib;
    res_out->footprint_gib  = mem1.gpu_wired_gib;
    res_out->peak_rss_gib   = std::max(mem1.peak_rss_gib, mem1.total_uma_rss_gib);
    res_out->swap_delta_gib = std::max(0.0, mem1.swap_used_gib - mem0.swap_used_gib);
    res_out->steady_majflt  = std::max<long>(0L, mem1.majflt - mem0.majflt);

    return 0;
}

static std::string resolve_inkling2_default_bundle_dir() {
    const std::string fast_uma =
        std::string(STRATA_INKLING2_DEFAULT_FAST_BUNDLE_DIR) + "/" + STRATA_INKLING2_UMA_FILENAME;
    if (::access(fast_uma.c_str(), R_OK) == 0) {
        return STRATA_INKLING2_DEFAULT_FAST_BUNDLE_DIR;
    }
    return STRATA_INKLING2_DEFAULT_BUNDLE_DIR;
}

static void print_inkling2_convert_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s convert-inkling2 [--hf-dir <dir>] [--bundle <dir>] [--threads <N>]\n"
        "  Defaults:\n"
        "    --hf-dir  %s\n"
        "    --bundle  %s\n"
        "    --threads 8\n",
        prog,
        STRATA_INKLING2_DEFAULT_HF_DIR,
        STRATA_INKLING2_DEFAULT_BUNDLE_DIR);
}

static void print_inkling2_convert_vision_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s convert-inkling2-vision [--hf-dir <dir>] [--out <path> | --bundle <dir>]\n"
        "  Defaults:\n"
        "    --hf-dir  %s\n"
        "    --out     %s\n",
        prog,
        STRATA_INKLING2_DEFAULT_HF_DIR,
        STRATA_INKLING2_VISION_DEFAULT_FAST_PATH);
}

static void print_inkling2_repack_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s repack-inkling2 [--src <dir>] [--dst <dir>] [--q4-unembed] [--q4-mtp | --q8-mtp | --f16-mtp]\n"
        "                     [--lora <peft_lora_dir>] [--lora-scale <float>] [--threads <N>]\n"
        "  Defaults:\n"
        "    --src        %s\n"
        "    --dst        %s\n"
        "    --q4-unembed off (Q8_0 unembed, Cfg B++: ~3.074 GB/tok, cos=0.9978)\n"
        "    --q4-mtp     on  (Q4_A 3L MTP + Q8_0 RNN: ~0.203 GB/blk; pass --q8-mtp for Q8_0 3L MTP)\n"
        "    --lora-scale 1.0 (multiplier on adapter `lora_alpha / r` when `--lora` is specified)\n"
        "    --threads    8\n",
        prog,
        STRATA_INKLING2_DEFAULT_BUNDLE_DIR,
        STRATA_INKLING2_DEFAULT_FAST_BUNDLE_DIR);
}

static void print_inkling2_run_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s run-inkling2 [--bundle <dir>] [--image <path>]... [--vision <inkling2_vision.strata>]\n"
        "                  [--prompt \"<text>\" | --prompt-file <path>] [--system \"<text>\"]\n"
        "                  [--no-think | --think] [--effort none|minimal|low|medium|high|xhigh] [--raw-prompt]\n"
        "                  [--kv-cache f16|q8] [--int8-prefill] [--draft-k 0..7] [--adaptive-k | --no-adaptive-k]\n"
        "                  [--frspec | --no-frspec] [--tree-spec | --no-tree-spec] [--bitplane-head | --no-bitplane-head]\n"
        "                  [--max-tokens 256] [--max-seq-len 65536] [--greedy | --temp 0.6]\n"
        "                  [--top-k 20] [--top-p 0.95] [--presence-penalty -2..2] [--frequency-penalty -2..2]\n"
        "                  [--repetition-penalty 0.5..2] [--penalty-window 0|8..65536] [--seed 42]\n"
        "                  [--no-eos-stop] [--verify-k0] [--dump-logits <path>]\n",
        prog);
}

static void print_inkling2_bench_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s bench-inkling2 [--bundle <dir>] [--contexts 512,2048,4096,16384,32768] [--gen-tokens 128]\n"
        "                    [--kv-cache f16|q8] [--int8-prefill] [--no-think | --think] [--effort low|medium|high|xhigh]\n"
        "                    [--draft-k 7 | --draft-ks 0,7] [--adaptive-k | --no-adaptive-k]\n"
        "                    [--frspec | --no-frspec] [--tree-spec | --no-tree-spec] [--bitplane-head | --no-bitplane-head]\n"
        "                    [--needle-depth 0.01..0.99] [--json <out.json>]\n",
        prog);
}

struct Inkling2RunCliOpts {
    std::string                  bundle_dir;
    std::string                  vision_path;
    std::vector<std::string>     image_paths;
    std::string                  prompt = "Explain how a 26B looped Mixture-of-Experts model with native NVFP4/MXFP8 weights and a 3-layer DSpark block-8 MTP drafter achieves ultra-fast lossless inference on Apple Silicon.";
    std::string                  prompt_file;
    std::string                  system_prompt;
    std::string                  effort = "medium";
    std::string                  dump_logits_path;
    bool                         enable_thinking   = true;
    bool                         raw_prompt        = false;
    bool                         verify_k0         = false;
    bool                         temp_explicit     = false;
    bool                         stop_on_eos       = true;
    bool                         adaptive_k        = true;
    bool                         frspec            = true;
    bool                         tree_spec         = true;
    bool                         bitplane_head     = true;
    bool                         int8_prefill      = false;
    bool                         int8_prefill_set  = false;
    bool                         draft_k_explicit  = false;
    bool                         show_help         = false;
    uint32_t                     max_tokens        = 256U;
    uint32_t                     max_seq_len       = 0U;
    uint32_t                     draft_k           = 7U;
    uint32_t                     kv_dtype          = STRATA_INKLING2_KV_DTYPE_F16;
    StrataInkling2SamplingParams samp{};
};

static int parse_inkling2_run_args(int argc, char** argv, Inkling2RunCliOpts* out, std::string* err_out) {
    if (!out) return -1;
    Inkling2RunCliOpts o{};
    o.samp.greedy             = false;
    o.samp.temperature        = 0.6f;
    o.samp.top_k              = 20U;
    o.samp.top_p              = 0.95f;
    o.samp.rng_seed           = 42ULL;
    o.samp.presence_penalty   = 0.0f;
    o.samp.frequency_penalty  = 0.0f;
    o.samp.repetition_penalty = 1.0f;
    o.samp.penalty_window     = 0U;

    auto fail = [&](const std::string& msg) -> int {
        if (err_out) *err_out = msg;
        return 1;
    };

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            o.bundle_dir = argv[++i];
        } else if (a == "--image" && i + 1 < argc) {
            o.image_paths.emplace_back(argv[++i]);
        } else if ((a == "--vision" || a == "--vision-path") && i + 1 < argc) {
            o.vision_path = argv[++i];
        } else if ((a == "--prompt" || a == "-p") && i + 1 < argc) {
            o.prompt = argv[++i];
        } else if (a == "--prompt-file" && i + 1 < argc) {
            o.prompt_file = argv[++i];
        } else if (a == "--system" && i + 1 < argc) {
            o.system_prompt = argv[++i];
        } else if (a == "--no-think" || a == "--nothink") {
            o.enable_thinking = false;
        } else if (a == "--think") {
            o.enable_thinking = true;
        } else if ((a == "--effort" || a == "--reasoning-effort") && i + 1 < argc) {
            o.effort = ik_lower_ascii(argv[++i]);
            if (!inkling2_effort_is_valid(o.effort)) {
                return fail("Invalid --effort `" + o.effort + "` (expected none, minimal, low, medium, high, or xhigh)");
            }
            if (o.effort == "none" || o.effort == "off") {
                o.enable_thinking = false;
            }
        } else if (a == "--raw" || a == "--raw-prompt") {
            o.raw_prompt = true;
        } else if (a == "--max-tokens" && i + 1 < argc) {
            o.max_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--max-seq-len" && i + 1 < argc) {
            const unsigned long v = std::strtoul(argv[++i], nullptr, 10);
            if (v < 128UL || v > STRATA_INKLING2_MAX_SEQ_LEN_CAP) {
                return fail("Invalid --max-seq-len (expected 128.." + std::to_string(STRATA_INKLING2_MAX_SEQ_LEN_CAP) + ")");
            }
            o.max_seq_len = static_cast<uint32_t>(v);
        } else if ((a == "--draft-k" || a == "--mtp-k") && i + 1 < argc) {
            const unsigned long v = std::strtoul(argv[++i], nullptr, 10);
            if (v > STRATA_INKLING2_MAX_SPEC_DRAFT_K) {
                return fail("Invalid --draft-k (expected 0.." + std::to_string(STRATA_INKLING2_MAX_SPEC_DRAFT_K) + ")");
            }
            o.draft_k          = static_cast<uint32_t>(v);
            o.draft_k_explicit = true;
        } else if (a == "--adaptive-k") {
            o.adaptive_k = true;
        } else if (a == "--no-adaptive-k") {
            o.adaptive_k = false;
        } else if (a == "--frspec" || a == "--fr-spec") {
            o.frspec = true;
        } else if (a == "--no-frspec" || a == "--no-fr-spec") {
            o.frspec = false;
        } else if (a == "--tree-spec" || a == "--beam-tree") {
            o.tree_spec = true;
        } else if (a == "--no-tree-spec" || a == "--no-beam-tree") {
            o.tree_spec = false;
        } else if (a == "--bitplane-head") {
            o.bitplane_head = true;
        } else if (a == "--no-bitplane-head") {
            o.bitplane_head = false;
        } else if (a == "--int8-prefill") {
            o.int8_prefill     = true;
            o.int8_prefill_set = true;
        } else if (a == "--fp16-prefill" || a == "--no-int8-prefill") {
            o.int8_prefill     = false;
            o.int8_prefill_set = true;
        } else if (a == "--kv-cache" && i + 1 < argc) {
            const std::string kv = argv[++i];
            if (!parse_inkling2_kv_cache(kv, &o.kv_dtype)) {
                return fail("Invalid --kv-cache `" + kv + "` (expected f16 or q8)");
            }
        } else if (a == "--greedy") {
            o.samp.greedy      = true;
            o.samp.temperature = 0.0f;
        } else if ((a == "--temp" || a == "--temperature") && i + 1 < argc) {
            o.samp.temperature = std::strtof(argv[++i], nullptr);
            o.samp.greedy      = (o.samp.temperature <= 1e-6f);
            o.temp_explicit    = true;
        } else if (a == "--top-k" && i + 1 < argc) {
            o.samp.top_k = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--top-p" && i + 1 < argc) {
            o.samp.top_p = std::strtof(argv[++i], nullptr);
        } else if (a == "--presence-penalty" && i + 1 < argc) {
            o.samp.presence_penalty = std::clamp(std::strtof(argv[++i], nullptr), -2.0f, 2.0f);
        } else if (a == "--frequency-penalty" && i + 1 < argc) {
            o.samp.frequency_penalty = std::clamp(std::strtof(argv[++i], nullptr), -2.0f, 2.0f);
        } else if ((a == "--repetition-penalty" || a == "--rep-penalty") && i + 1 < argc) {
            o.samp.repetition_penalty = std::clamp(std::strtof(argv[++i], nullptr), 0.5f, 2.0f);
        } else if (a == "--penalty-window" && i + 1 < argc) {
            const unsigned long w = std::strtoul(argv[++i], nullptr, 10);
            o.samp.penalty_window = (w == 0UL) ? 0U : static_cast<uint32_t>(std::clamp<unsigned long>(w, 8UL, 65536UL));
        } else if (a == "--seed" && i + 1 < argc) {
            o.samp.rng_seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (a == "--no-eos-stop") {
            o.stop_on_eos = false;
        } else if (a == "--verify-k0") {
            o.verify_k0 = true;
        } else if (a == "--dump-logits" && i + 1 < argc) {
            o.dump_logits_path = argv[++i];
        } else if (a == "--help" || a == "-h") {
            o.show_help = true;
            *out = std::move(o);
            return 0;
        } else {
            return fail("Unknown or incomplete option: " + a);
        }
    }

    if (o.bundle_dir.empty()) {
        o.bundle_dir = resolve_inkling2_default_bundle_dir();
    }
    if (!o.draft_k_explicit) {
        o.draft_k = o.adaptive_k ? 7U : 3U;
    }
    if (o.verify_k0 && !o.temp_explicit) {
        o.samp.greedy      = true;
        o.samp.temperature = 0.0f;
    }
    *out = std::move(o);
    return 0;
}

struct Inkling2BenchCliOpts {
    std::string bundle_dir;
    std::string contexts_csv = "512,2048,4096,16384,32768";
    std::string draft_ks_csv;
    std::string json_out_path;
    std::string bench_effort        = "medium";
    bool        bench_think         = false;
    bool        adaptive_k          = true;
    bool        frspec              = true;
    bool        tree_spec           = true;
    bool        bitplane_head       = true;
    bool        int8_prefill        = false;
    bool        int8_prefill_set    = false;
    bool        draft_k_explicit    = false;
    bool        show_help           = false;
    uint32_t    gen_tokens          = 128U;
    uint32_t    default_mtp_k       = 7U;
    uint32_t    kv_dtype            = STRATA_INKLING2_KV_DTYPE_F16;
    double      custom_needle_depth = -1.0;
};

static int parse_inkling2_bench_args(int argc, char** argv, Inkling2BenchCliOpts* out, std::string* err_out) {
    if (!out) return -1;
    Inkling2BenchCliOpts o{};
    auto fail = [&](const std::string& msg) -> int {
        if (err_out) *err_out = msg;
        return 1;
    };

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            o.bundle_dir = argv[++i];
        } else if (a == "--contexts" && i + 1 < argc) {
            o.contexts_csv = argv[++i];
        } else if (a == "--gen-tokens" && i + 1 < argc) {
            o.gen_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--think") {
            o.bench_think = true;
        } else if (a == "--no-think" || a == "--nothink") {
            o.bench_think = false;
        } else if ((a == "--effort" || a == "--reasoning-effort") && i + 1 < argc) {
            o.bench_effort = ik_lower_ascii(argv[++i]);
            if (!inkling2_effort_is_valid(o.bench_effort)) {
                return fail("Invalid --effort `" + o.bench_effort + "` (expected none, minimal, low, medium, high, or xhigh)");
            }
            o.bench_think = (o.bench_effort != "none" && o.bench_effort != "off");
        } else if ((a == "--draft-k" || a == "--mtp-k") && i + 1 < argc) {
            const unsigned long v = std::strtoul(argv[++i], nullptr, 10);
            if (v > STRATA_INKLING2_MAX_SPEC_DRAFT_K) {
                return fail("Invalid --draft-k (expected 0.." + std::to_string(STRATA_INKLING2_MAX_SPEC_DRAFT_K) + ")");
            }
            o.default_mtp_k    = static_cast<uint32_t>(v);
            o.draft_k_explicit = true;
        } else if (a == "--draft-ks" && i + 1 < argc) {
            o.draft_ks_csv = argv[++i];
        } else if (a == "--adaptive-k") {
            o.adaptive_k = true;
        } else if (a == "--no-adaptive-k") {
            o.adaptive_k = false;
        } else if (a == "--frspec" || a == "--fr-spec") {
            o.frspec = true;
        } else if (a == "--no-frspec" || a == "--no-fr-spec") {
            o.frspec = false;
        } else if (a == "--tree-spec" || a == "--beam-tree") {
            o.tree_spec = true;
        } else if (a == "--no-tree-spec" || a == "--no-beam-tree") {
            o.tree_spec = false;
        } else if (a == "--bitplane-head") {
            o.bitplane_head = true;
        } else if (a == "--no-bitplane-head") {
            o.bitplane_head = false;
        } else if (a == "--int8-prefill") {
            o.int8_prefill     = true;
            o.int8_prefill_set = true;
        } else if (a == "--fp16-prefill" || a == "--no-int8-prefill") {
            o.int8_prefill     = false;
            o.int8_prefill_set = true;
        } else if (a == "--kv-cache" && i + 1 < argc) {
            const std::string kv = argv[++i];
            if (!parse_inkling2_kv_cache(kv, &o.kv_dtype)) {
                return fail("Invalid --kv-cache `" + kv + "` (expected f16 or q8)");
            }
        } else if (a == "--needle-depth" && i + 1 < argc) {
            o.custom_needle_depth = std::strtod(argv[++i], nullptr);
        } else if (a == "--json" && i + 1 < argc) {
            o.json_out_path = argv[++i];
        } else if (a == "--help" || a == "-h") {
            o.show_help = true;
            *out = std::move(o);
            return 0;
        } else {
            return fail("Unknown or incomplete option: " + a);
        }
    }

    if (o.bundle_dir.empty()) {
        o.bundle_dir = resolve_inkling2_default_bundle_dir();
    }
    if (!o.draft_k_explicit) {
        o.default_mtp_k = o.adaptive_k ? 7U : 3U;
    }
    *out = std::move(o);
    return 0;
}

} /* namespace */

extern "C" {

int strata_inkling2_cmd_convert(int argc, char** argv) {
    std::string hf_dir     = STRATA_INKLING2_DEFAULT_HF_DIR;
    std::string bundle_dir = STRATA_INKLING2_DEFAULT_BUNDLE_DIR;
    uint32_t num_threads   = 8U;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--hf-dir" || a == "--repo" || a == "--in") && i + 1 < argc) {
            hf_dir = argv[++i];
        } else if ((a == "--bundle" || a == "--out") && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if ((a == "--threads" || a == "--convert-threads" || a == "-j") && i + 1 < argc) {
            num_threads = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--help" || a == "-h") {
            print_inkling2_convert_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[convert-inkling2] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    StrataInkling2ConvertConfig cfg{};
    cfg.hf_dir      = hf_dir.c_str();
    cfg.bundle_dir  = bundle_dir.c_str();
    cfg.num_threads = num_threads;

    const auto t0 = SteadyClock::now();
    const int rc = strata_inkling2_convert_hf(&cfg);
    const double sec = elapsed_ms(t0, SteadyClock::now()) / 1000.0;
    if (rc != 0) {
        std::fprintf(stderr, "[convert-inkling2] Conversion failed (rc=%d)\n", rc);
        return 1;
    }

    std::printf("[convert-inkling2] Conversion complete in %.2f s -> `%s`\n", sec, bundle_dir.c_str());
    return strata_inkling2_inspect_bundle(bundle_dir.c_str());
}

int strata_inkling2_cmd_convert_vision(int argc, char** argv) {
    std::string hf_dir   = STRATA_INKLING2_DEFAULT_HF_DIR;
    std::string out_path = STRATA_INKLING2_VISION_DEFAULT_FAST_PATH;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--hf-dir" || a == "--repo" || a == "--in") && i + 1 < argc) {
            hf_dir = argv[++i];
        } else if ((a == "--out" || a == "--vision-out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--bundle" && i + 1 < argc) {
            out_path = std::string(argv[++i]) + "/" + STRATA_INKLING2_VISION_FILENAME;
        } else if (a == "--help" || a == "-h") {
            print_inkling2_convert_vision_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[convert-inkling2-vision] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    struct stat st{};
    if (::stat(out_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        out_path += std::string("/") + STRATA_INKLING2_VISION_FILENAME;
    }

    const auto t0 = SteadyClock::now();
    const int rc = strata_inkling2_convert_vision_from_hf(hf_dir.c_str(), out_path.c_str());
    const double sec = elapsed_ms(t0, SteadyClock::now()) / 1000.0;
    if (rc != 0) {
        std::fprintf(stderr, "[convert-inkling2-vision] Vision conversion failed (rc=%d)\n", rc);
        return 1;
    }

    std::printf("[convert-inkling2-vision] Conversion complete in %.2f s -> `%s`\n", sec, out_path.c_str());
    return 0;
}

int strata_inkling2_cmd_repack(int argc, char** argv) {
    std::string src_dir  = STRATA_INKLING2_DEFAULT_BUNDLE_DIR;
    std::string dst_dir  = STRATA_INKLING2_DEFAULT_FAST_BUNDLE_DIR;
    std::string lora_dir;
    float lora_scale     = 1.0f;
    bool q4_unembed      = false;
    uint32_t mtp_dtype   = STRATA_INKLING2_DTYPE_Q4_A;
    uint32_t num_threads = 8U;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--src" || a == "--in" || a == "--bundle") && i + 1 < argc) {
            src_dir = argv[++i];
        } else if ((a == "--dst" || a == "--out") && i + 1 < argc) {
            dst_dir = argv[++i];
        } else if ((a == "--lora" || a == "--lora-dir") && i + 1 < argc) {
            lora_dir = argv[++i];
        } else if (a == "--lora-scale" && i + 1 < argc) {
            lora_scale = std::strtof(argv[++i], nullptr);
            if (!std::isfinite(lora_scale) || lora_scale <= 0.0f) {
                std::fprintf(stderr, "[repack-inkling2] Invalid --lora-scale (expected > 0.0)\n");
                return 1;
            }
        } else if (a == "--q4-unembed") {
            q4_unembed = true;
        } else if (a == "--q8-unembed" || a == "--no-q4-unembed") {
            q4_unembed = false;
        } else if (a == "--q4-mtp") {
            mtp_dtype = STRATA_INKLING2_DTYPE_Q4_A;
        } else if (a == "--q8-mtp" || a == "--no-q4-mtp") {
            mtp_dtype = STRATA_INKLING2_DTYPE_Q8_0;
        } else if (a == "--f16-mtp" || a == "--fp16-mtp") {
            mtp_dtype = STRATA_INKLING2_DTYPE_F16;
        } else if ((a == "--threads" || a == "-j") && i + 1 < argc) {
            num_threads = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--help" || a == "-h") {
            print_inkling2_repack_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[repack-inkling2] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    StrataInkling2RepackConfig cfg{};
    cfg.src_bundle_dir = src_dir.c_str();
    cfg.dst_bundle_dir = dst_dir.c_str();
    cfg.q4_unembed     = q4_unembed;
    cfg.q4_mtp         = (mtp_dtype == STRATA_INKLING2_DTYPE_Q4_A);
    cfg.num_threads    = num_threads;
    cfg.mtp_dtype      = mtp_dtype;
    cfg.lora_dir       = lora_dir.empty() ? nullptr : lora_dir.c_str();
    cfg.lora_scale     = lora_scale;

    const auto t0 = SteadyClock::now();
    const int rc = strata_inkling2_repack_fast(&cfg);
    const double sec = elapsed_ms(t0, SteadyClock::now()) / 1000.0;
    if (rc != 0) {
        std::fprintf(stderr, "[repack-inkling2] Repack failed (rc=%d)\n", rc);
        return 1;
    }

    const char* mtp_lbl = (mtp_dtype == STRATA_INKLING2_DTYPE_Q4_A) ? "Q4_A"
                        : (mtp_dtype == STRATA_INKLING2_DTYPE_Q8_0) ? "Q8_0" : "F16";
    std::printf("[repack-inkling2] Repack complete in %.2f s (%s -> %s, unembed=%s, mtp=%s)\n",
                sec,
                src_dir.c_str(),
                dst_dir.c_str(),
                q4_unembed ? "Q4_A" : "Q8_0",
                mtp_lbl);
    return strata_inkling2_inspect_bundle(dst_dir.c_str());
}

int strata_inkling2_cmd_inspect(int argc, char** argv) {
    std::string bundle_dir = STRATA_INKLING2_DEFAULT_BUNDLE_DIR;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s inspect-inkling2 [--bundle <dir>]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[inspect-inkling2] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }
    return strata_inkling2_inspect_bundle(bundle_dir.c_str());
}

int strata_inkling2_cmd_run(int argc, char** argv) {
    Inkling2RunCliOpts opts{};
    std::string perr;
    if (parse_inkling2_run_args(argc, argv, &opts, &perr) != 0) {
        std::fprintf(stderr, "[run-inkling2] %s\n", perr.c_str());
        return 1;
    }
    if (opts.show_help) {
        print_inkling2_run_help(argv[0]);
        return 0;
    }

    const std::string& bundle_dir       = opts.bundle_dir;
    std::string        prompt           = opts.prompt;
    const std::string& prompt_file      = opts.prompt_file;
    const std::string& system_prompt    = opts.system_prompt;
    const std::string& effort           = opts.effort;
    const std::string& dump_logits_path = opts.dump_logits_path;
    const bool enable_thinking          = opts.enable_thinking;
    const bool raw_prompt               = opts.raw_prompt;
    const bool verify_k0                = opts.verify_k0;
    const bool stop_on_eos              = opts.stop_on_eos;
    const bool adaptive_k               = opts.adaptive_k;
    const bool frspec                   = opts.frspec;
    const bool tree_spec                = opts.tree_spec;
    const bool bitplane_head            = opts.bitplane_head;
    const bool int8_prefill             = opts.int8_prefill;
    const bool int8_prefill_set         = opts.int8_prefill_set;
    uint32_t max_tokens                 = opts.max_tokens;
    const uint32_t max_seq_len          = opts.max_seq_len;
    const uint32_t draft_k              = opts.draft_k;
    const uint32_t kv_dtype             = opts.kv_dtype;
    StrataInkling2SamplingParams samp   = opts.samp;

    if (!prompt_file.empty()) {
        if (!read_text_file(prompt_file, &prompt)) {
            std::fprintf(stderr, "[run-inkling2] Failed to read `--prompt-file %s`\n", prompt_file.c_str());
            return 1;
        }
    }
    if (raw_prompt && !opts.image_paths.empty()) {
        std::fprintf(stderr, "[run-inkling2] `--image` cannot be combined with `--raw-prompt`\n");
        return 1;
    }

    const ProcessMemStats mem_init0 = query_process_mem_stats();
    const auto t_load0 = SteadyClock::now();
    StrataInkling2Bundle bdl{};
    bdl.uma_fd = -1;
    if (strata_inkling2_load_bundle(bundle_dir.c_str(), &bdl) != 0) {
        std::fprintf(stderr, "[run-inkling2] Failed to load bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    struct VisionEngineGuard {
        StrataInkling2VisionEngine* ptr = nullptr;
        ~VisionEngineGuard() {
            if (ptr) strata_inkling2_vision_destroy(ptr);
        }
    } vision_guard{};
    StrataInkling2VisionInfo vis_info{};
    std::vector<EncodedCliImage> encoded_images;
    double total_vis_encode_ms = 0.0;
    uint32_t total_vis_tokens  = 0U;

    if (!opts.image_paths.empty()) {
        char vis_path[512] = {};
        if (!opts.vision_path.empty()) {
            std::snprintf(vis_path, sizeof(vis_path), "%s", opts.vision_path.c_str());
        } else if (strata_inkling2_vision_resolve_path(bundle_dir.c_str(), vis_path, sizeof(vis_path)) != 1) {
            std::fprintf(stderr,
                         "[run-inkling2] Vision weights (`%s`) not found for bundle `%s`.\n"
                         "               Generate them with: ./build/strata_cli convert-inkling2-vision\n",
                         STRATA_INKLING2_VISION_FILENAME, bundle_dir.c_str());
            strata_inkling2_free_bundle(&bdl);
            return 1;
        }
        vision_guard.ptr = strata_inkling2_vision_create_default(vis_path);
        if (!vision_guard.ptr) {
            std::fprintf(stderr, "[run-inkling2] Failed to initialize Vision encoder from `%s`\n", vis_path);
            strata_inkling2_free_bundle(&bdl);
            return 1;
        }
        (void)strata_inkling2_vision_get_info(vision_guard.ptr, &vis_info);
        if (vis_info.text_hidden_size != bdl.hdr.cfg.hidden_size) {
            std::fprintf(stderr,
                         "[run-inkling2] Vision sidecar text_hidden_size (%u) != text hidden_size (%u)\n",
                         vis_info.text_hidden_size, bdl.hdr.cfg.hidden_size);
            strata_inkling2_free_bundle(&bdl);
            return 1;
        }

        encoded_images.reserve(opts.image_paths.size());
        for (const std::string& img_path : opts.image_paths) {
            EncodedCliImage img{};
            img.path = img_path;
            const auto tv0 = SteadyClock::now();
            const int vrc = strata_inkling2_vision_encode_image_file(
                vision_guard.ptr, img_path.c_str(), &img.emb_f32, &img.num_tokens, &img.grid_h, &img.grid_w);
            img.encode_ms = elapsed_ms(tv0, SteadyClock::now());
            if (vrc != 0 || img.num_tokens == 0U) {
                std::fprintf(stderr, "[run-inkling2] Failed to encode image `%s` (rc=%d)\n", img_path.c_str(), vrc);
                strata_inkling2_free_bundle(&bdl);
                return 1;
            }
            total_vis_encode_ms += img.encode_ms;
            total_vis_tokens    += img.num_tokens;
            encoded_images.push_back(std::move(img));
        }
        (void)strata_inkling2_vision_get_info(vision_guard.ptr, &vis_info);
    }

    std::vector<uint32_t> prompt_ids;
    std::vector<StrataInkling2ImageSpan> image_spans;
    if (raw_prompt) {
        if (!encode_string_vec(bdl.tokenizer, prompt, /*parse_special=*/true, &prompt_ids) || prompt_ids.empty()) {
            std::fprintf(stderr, "[run-inkling2] Failed to tokenize raw prompt\n");
            strata_inkling2_free_bundle(&bdl);
            return 1;
        }
    } else {
        if (!encode_inkling2_chat_prompt_with_images(
                bdl.tokenizer, prompt, system_prompt, enable_thinking, effort,
                encoded_images, &prompt_ids, &image_spans)) {
            std::fprintf(stderr, "[run-inkling2] Failed to format/tokenize Inkling-2 chat prompt\n");
            strata_inkling2_free_bundle(&bdl);
            return 1;
        }
    }

    const bool unlimited_tokens = (max_tokens == 0U);
    const uint32_t default_unlim_seq = 65536U;
    const uint32_t needed_seq = unlimited_tokens
        ? std::max<uint32_t>(default_unlim_seq, static_cast<uint32_t>(ik_align_up_u64(static_cast<uint64_t>(prompt_ids.size()) + 4096ULL, 256ULL)))
        : static_cast<uint32_t>(
              ik_align_up_u64(static_cast<uint64_t>(prompt_ids.size()) + static_cast<uint64_t>(max_tokens) + 64ULL, 256ULL));
    const uint32_t alloc_seq = std::min<uint32_t>(
        STRATA_INKLING2_MAX_SEQ_LEN_CAP,
        std::max<uint32_t>(max_seq_len, std::max<uint32_t>(2048U, needed_seq)));
    if (unlimited_tokens) {
        max_tokens = (alloc_seq > prompt_ids.size() + 64U)
            ? (alloc_seq - static_cast<uint32_t>(prompt_ids.size()) - 64U)
            : 256U;
    }

    StrataInkling2GpuEngine* eng = strata_inkling2_gpu_create_ex(&bdl, alloc_seq, kv_dtype);
    if (!eng) {
        std::fprintf(stderr, "[run-inkling2] Failed to initialize Metal 4 Inkling-2-Mini GPU engine\n");
        strata_inkling2_free_bundle(&bdl);
        return 1;
    }

    if (int8_prefill_set) {
        strata_inkling2_gpu_set_int8_prefill(eng, int8_prefill);
    }

    StrataInkling2GpuInfo info{};
    strata_inkling2_gpu_get_info(eng, &info);

    const uint32_t eff_k = (bdl.hdr.mtp.enabled && info.mtp_enabled)
        ? std::min<uint32_t>(draft_k, STRATA_INKLING2_MAX_SPEC_DRAFT_K)
        : 0U;
    strata_inkling2_gpu_set_adaptive_k(eng, adaptive_k && eff_k > 0U);
    strata_inkling2_gpu_set_frspec(eng, frspec);
    strata_inkling2_gpu_set_beam_tree(eng, tree_spec);
    strata_inkling2_gpu_set_bitplane_head(eng, bitplane_head);
    strata_inkling2_gpu_get_info(eng, &info);

    const bool penalties_active =
        (samp.presence_penalty != 0.0f) ||
        (samp.frequency_penalty != 0.0f) ||
        (samp.repetition_penalty > 0.0f && std::fabs(samp.repetition_penalty - 1.0f) > 1e-6f);

    StrataInkling2Checkpoint* run_ckpt = nullptr;
    if (verify_k0 && eff_k > 0U && samp.greedy && !penalties_active && dump_logits_path.empty()) {
        run_ckpt = strata_inkling2_checkpoint_create(eng);
        if (run_ckpt) {
            (void)strata_inkling2_checkpoint_save(eng, run_ckpt);
            (void)strata_inkling2_checkpoint_restore(eng, run_ckpt);
        }
    }
    run_warmup_pass(&bdl, eng);
    const auto t_load1 = SteadyClock::now();
    const ProcessMemStats mem_init1 = query_process_mem_stats();
    const long init_majflt = std::max<long>(0L, mem_init1.majflt - mem_init0.majflt);

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    const std::string eff_num = inkling2_effort_to_num_str(enable_thinking, effort);

    const bool eff_adaptive_k = strata_inkling2_gpu_get_adaptive_k(eng);
    const bool eff_frspec     = strata_inkling2_gpu_get_frspec(eng) && (bdl.hdr.mtp.frspec.enabled != 0U);
    const bool eff_tree_spec  = strata_inkling2_gpu_get_beam_tree(eng);
    const bool eff_bitplane   = strata_inkling2_gpu_get_bitplane_head(eng);
    const bool eff_vision     = (vision_guard.ptr != nullptr && vis_info.ready != 0U);

    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Live thinkingmachines/Inkling-2-Mini (26B Looped + 3L DSpark Block-8 MTP, v%u)\n", bdl.hdr.version);
    std::printf("  Device : %s | ZeroCopy=%d | Init %.0f ms | Metal Cache: %s (%.1f ms) | MaxSeq=%u | KV=%s\n",
                info.device_name, info.zero_copy_mmap ? 1 : 0, elapsed_ms(t_load0, t_load1),
                info.metal_cache_hit ? "HIT" : "MISS", info.pso_load_ms,
                info.max_seq_len, inkling2_kv_cache_label(info.kv_dtype));
    std::printf("  Memory : UMA %.2f GiB single-buffer | Active %.3f GB/tok (48 visits) + %.3f GB/blk MTP\n",
                static_cast<double>(info.uma_mapped_bytes) / kGiB,
                static_cast<double>(info.active_bytes_per_token) / 1e9,
                static_cast<double>(info.mtp_active_bytes_per_block) / 1e9);
    std::printf("  Config : %s (effort=%s) | %s (K=%u) | Prefill: %s | FR-Spec: %s | Adaptive-K: %s\n",
                raw_prompt ? "Raw Prompt" : (enable_thinking ? "Thinking" : "Direct (--no-think)"),
                raw_prompt ? "raw" : eff_num.c_str(),
                (eff_k > 0U) ? "Native DSpark MTP" : "Standard Autoregressive",
                eff_k,
                info.int8_prefill_enabled ? "INT8" : "FP16",
                eff_frspec ? "ON (32K)" : "OFF",
                eff_adaptive_k ? "ON" : "OFF");
    std::printf("  Engine : Tree-Spec: %s | Bitplane-Head: %s | Vision: %s | %s | Prompt=%zu, MaxNew=%u\n",
                eff_tree_spec ? "ON" : "OFF",
                eff_bitplane ? "ON" : "OFF",
                eff_vision ? "ON" : "OFF",
                samp.greedy ? "Greedy (T=0)" : "Stochastic Leviathan",
                prompt_ids.size(),
                max_tokens);
    if (eff_vision && !encoded_images.empty()) {
        std::printf("  Vision : %zu img (%u vis toks) encoded in %.1f ms | Vision Cache: %s (%.1f ms)\n",
                    encoded_images.size(), total_vis_tokens, total_vis_encode_ms,
                    vis_info.metal_cache_hit ? "HIT" : "MISS", vis_info.pso_load_ms);
    }
    std::printf("====================================================================================================\n");

    PrefillReuseSlot run_reuse{};
    run_reuse.ckpt  = run_ckpt;
    run_reuse.valid = false;

    Inkling2RunResult res{};
    const int rc = execute_inkling2_sequence(
        &bdl, eng, info, prompt_ids, max_tokens, /*count_decode_only=*/false,
        eff_k, samp, stop_on_eos, /*stream_stdout=*/true, &res,
        (run_ckpt != nullptr) ? &run_reuse : nullptr,
        dump_logits_path,
        image_spans.empty() ? nullptr : &image_spans);
    std::fputc('\n', stdout);

    if (rc != 0) {
        std::fprintf(stderr, "[run-inkling2] Generation failed (rc=%d)\n", rc);
        if (run_ckpt) strata_inkling2_checkpoint_destroy(run_ckpt);
        strata_inkling2_gpu_destroy(eng);
        strata_inkling2_free_bundle(&bdl);
        return 1;
    }

    const double accept_pct = (res.mtp_drafted > 0U)
        ? (100.0 * static_cast<double>(res.mtp_accepted) / static_cast<double>(res.mtp_drafted))
        : 0.0;
    const double avg_tok_per_step = (res.decode_steps > 0U)
        ? (static_cast<double>(res.decode_emitted) / static_cast<double>(res.decode_steps))
        : 0.0;

    std::printf("----------------------------------------------------------------------------------------------------\n");
    std::printf("  Inkling-2-Mini Silicon Telemetry (%s)\n", info.device_name);
    std::printf("----------------------------------------------------------------------------------------------------\n");
    if (eff_vision && !encoded_images.empty()) {
        std::printf("  Vision Encode        : %7.2f ms (%zu img, %u vis toks, last grid %ux%u | Cache %s %.1f ms)\n",
                    total_vis_encode_ms, encoded_images.size(), total_vis_tokens,
                    encoded_images.back().grid_h, encoded_images.back().grid_w,
                    vis_info.metal_cache_hit ? "HIT" : "MISS", vis_info.pso_load_ms);
    }
    std::printf("  Prefill & TTFT       : %7.2f tok/s | TTFT %7.2f ms (%u prompt tokens)\n",
                res.prefill_tok_s, res.ttft_ms, res.prompt_tokens);
    std::printf("  Decode Throughput    : %7.2f tok/s | %u decode toks in %u steps (%.2f ms, total=%u)\n",
                res.decode_tok_s, res.decode_emitted, res.decode_steps, res.decode_ms, res.emitted_total);
    if (eff_k > 0U) {
        std::printf("  Native MTP (K=%u)     : %6.2f tok/step | Draft accepted %u / %u (%.1f%%)\n",
                    eff_k, avg_tok_per_step, res.mtp_accepted, res.mtp_drafted, accept_pct);
    }
    std::printf("  Modeled DRAM BW      : %7.2f GB/s modeled (%.2f GB read) | %7.2f GB/s K=0-equiv\n",
                res.phys_dram_gbps, static_cast<double>(res.decode_dram_bytes) / 1e9, res.eff_dram_gbps);
    std::printf("  Active State (pos=%u): %7.2f MiB total (Global KV[%s] %.2f MiB + SWA KV %.2f MiB + SConv %.2f MiB)\n",
                res.final_seq_pos, res.active_state_mib, inkling2_kv_cache_label(info.kv_dtype),
                res.global_kv_active_mib, res.swa_kv_mib, res.sconv_mib);
    std::printf("  UMA & Fault Gates    : Total UMA %.2f GiB (GPU Wired %.2f GiB, Host RSS %.2f GiB, Peak %.2f GiB, Swap +%.2f GiB) | Steady MajFlt=%ld (Init=%ld)\n",
                res.rss_gib, res.footprint_gib, res.host_rss_gib, res.peak_rss_gib, res.swap_delta_gib, res.steady_majflt, init_majflt);
    const LoopSanityReport loop_rep = analyze_shingle_loops(res.generated_ids);
    if (loop_rep.evaluated) {
        std::printf("  Output Sanity        : %s (dup8=%.2f%%, dup4=%.2f%%, %zu toks)\n",
                    loop_rep.loop_warning ? "LOOP_WARNING" : "OK",
                    100.0 * loop_rep.dup_8gram_ratio,
                    100.0 * loop_rep.dup_4gram_ratio,
                    res.generated_ids.size());
    }

    bool verify_ok = true;
    if (verify_k0 && eff_k > 0U && samp.greedy) {
        Inkling2RunResult ref{};
        if (execute_inkling2_sequence(
                &bdl, eng, info, prompt_ids, max_tokens, /*count_decode_only=*/false,
                /*draft_k=*/0U, samp, stop_on_eos, /*stream_stdout=*/false, &ref,
                (run_ckpt != nullptr) ? &run_reuse : nullptr,
                {},
                image_spans.empty() ? nullptr : &image_spans) != 0) {
            verify_ok = false;
        } else {
            verify_ok = (res.generated_ids == ref.generated_ids);
            std::printf("  MTP K=%u vs K=0 Check : %s (%zu vs %zu tokens, Speedup %.2fx: %.2f vs %.2f tok/s)\n",
                        eff_k,
                        verify_ok ? "100% BIT-EXACT MATCH" : "MISMATCH",
                        res.generated_ids.size(),
                        ref.generated_ids.size(),
                        (ref.decode_tok_s > 0.0) ? (res.decode_tok_s / ref.decode_tok_s) : 0.0,
                        res.decode_tok_s,
                        ref.decode_tok_s);
            if (!verify_ok) {
                const size_t min_n = std::min(res.generated_ids.size(), ref.generated_ids.size());
                size_t first_diff = min_n;
                for (size_t di = 0U; di < min_n; ++di) {
                    if (res.generated_ids[di] != ref.generated_ids[di]) {
                        first_diff = di;
                        break;
                    }
                }
                if (first_diff < min_n) {
                    std::printf("  First Diff @ tok[%zu] : K=%u got %u vs K=0 got %u\n",
                                first_diff, eff_k,
                                res.generated_ids[first_diff],
                                ref.generated_ids[first_diff]);
                }
            }
        }
    }
    std::printf("====================================================================================================\n");

    if (run_ckpt) strata_inkling2_checkpoint_destroy(run_ckpt);
    strata_inkling2_gpu_destroy(eng);
    strata_inkling2_free_bundle(&bdl);
    return verify_ok ? 0 : 1;
}

int strata_inkling2_cmd_bench(int argc, char** argv) {
    Inkling2BenchCliOpts opts{};
    std::string perr;
    if (parse_inkling2_bench_args(argc, argv, &opts, &perr) != 0) {
        std::fprintf(stderr, "[bench-inkling2] %s\n", perr.c_str());
        return 1;
    }
    if (opts.show_help) {
        print_inkling2_bench_help(argv[0]);
        return 0;
    }

    const std::string& bundle_dir     = opts.bundle_dir;
    const std::string& contexts_csv   = opts.contexts_csv;
    const std::string& draft_ks_csv   = opts.draft_ks_csv;
    const std::string& json_out_path  = opts.json_out_path;
    const std::string& bench_effort   = opts.bench_effort;
    const bool bench_think            = opts.bench_think;
    const bool adaptive_k             = opts.adaptive_k;
    const bool frspec                 = opts.frspec;
    const bool tree_spec              = opts.tree_spec;
    const bool bitplane_head          = opts.bitplane_head;
    const bool int8_prefill           = opts.int8_prefill;
    const bool int8_prefill_set       = opts.int8_prefill_set;
    const uint32_t gen_tokens         = opts.gen_tokens;
    const uint32_t default_mtp_k      = opts.default_mtp_k;
    const uint32_t kv_dtype           = opts.kv_dtype;
    const double custom_needle_depth  = opts.custom_needle_depth;

    const std::vector<uint32_t> contexts = parse_u32_csv(contexts_csv);
    if (contexts.empty()) {
        std::fprintf(stderr, "[bench-inkling2] `--contexts` parsed empty\n");
        return 1;
    }

    std::vector<uint32_t> draft_ks;
    if (!draft_ks_csv.empty()) {
        draft_ks = parse_u32_csv(draft_ks_csv);
        for (uint32_t& k : draft_ks) {
            k = std::min<uint32_t>(k, STRATA_INKLING2_MAX_SPEC_DRAFT_K);
        }
    } else {
        draft_ks.push_back(0U);
        if (default_mtp_k > 0U) {
            draft_ks.push_back(std::min<uint32_t>(default_mtp_k, STRATA_INKLING2_MAX_SPEC_DRAFT_K));
        }
    }

    uint32_t max_ctx = 0U;
    for (uint32_t c : contexts) max_ctx = std::max(max_ctx, c);
    const uint64_t max_seq_u64 = ik_align_up_u64(
        static_cast<uint64_t>(max_ctx) + static_cast<uint64_t>(gen_tokens) + 64ULL, 256ULL);
    if (max_seq_u64 > STRATA_INKLING2_MAX_SEQ_LEN_CAP) {
        std::fprintf(stderr, "[bench-inkling2] Requested context+gen (%llu) exceeds max_seq_len cap (%u)\n",
                     static_cast<unsigned long long>(max_seq_u64), STRATA_INKLING2_MAX_SEQ_LEN_CAP);
        return 1;
    }
    const uint32_t max_seq_len = static_cast<uint32_t>(max_seq_u64);

    const ProcessMemStats mem_boot0 = query_process_mem_stats();
    const auto t_boot0 = SteadyClock::now();

    StrataInkling2Bundle bdl{};
    bdl.uma_fd = -1;
    if (strata_inkling2_load_bundle(bundle_dir.c_str(), &bdl) != 0) {
        std::fprintf(stderr, "[bench-inkling2] Failed to load bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    StrataInkling2GpuEngine* eng = strata_inkling2_gpu_create_ex(&bdl, max_seq_len, kv_dtype);
    if (!eng) {
        std::fprintf(stderr, "[bench-inkling2] Failed to create GPU engine (max_seq_len=%u, kv_cache=%s)\n",
                     max_seq_len, inkling2_kv_cache_label(kv_dtype));
        strata_inkling2_free_bundle(&bdl);
        return 1;
    }

    if (int8_prefill_set) {
        strata_inkling2_gpu_set_int8_prefill(eng, int8_prefill);
    }

    StrataInkling2GpuInfo info{};
    strata_inkling2_gpu_set_adaptive_k(eng, adaptive_k);
    strata_inkling2_gpu_set_frspec(eng, frspec);
    strata_inkling2_gpu_set_beam_tree(eng, tree_spec);
    strata_inkling2_gpu_set_bitplane_head(eng, bitplane_head);
    strata_inkling2_gpu_get_info(eng, &info);
    const bool eff_adaptive_k = strata_inkling2_gpu_get_adaptive_k(eng);
    const bool eff_frspec     = strata_inkling2_gpu_get_frspec(eng) && (bdl.hdr.mtp.frspec.enabled != 0U);
    const bool eff_tree_spec  = strata_inkling2_gpu_get_beam_tree(eng);
    const bool eff_bitplane   = strata_inkling2_gpu_get_bitplane_head(eng);
    char bench_vis_path[512] = {};
    const bool vis_sidecar_avail =
        (strata_inkling2_vision_resolve_path(bundle_dir.c_str(), bench_vis_path, sizeof(bench_vis_path)) == 1);
    (void)vis_sidecar_avail;

    StrataInkling2Checkpoint* bench_ckpt = nullptr;
    if (draft_ks.size() > 1U) {
        bench_ckpt = strata_inkling2_checkpoint_create(eng);
        if (bench_ckpt) {
            (void)strata_inkling2_checkpoint_save(eng, bench_ckpt);
            (void)strata_inkling2_checkpoint_restore(eng, bench_ckpt);
        }
    }
    run_warmup_pass(&bdl, eng);

    const auto t_boot1 = SteadyClock::now();
    const ProcessMemStats mem_boot1 = query_process_mem_stats();
    const long init_majflt = std::max<long>(0L, mem_boot1.majflt - mem_boot0.majflt);

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;

    std::printf("========================================================================================================================================\n");
    std::printf("  STRATA-M5 v2.0 — thinkingmachines/Inkling-2-Mini (26B Looped NVFP4/MXFP8 + 3-Layer DSpark Block-8 MTP) Context & NIAH Suite\n");
    std::printf("========================================================================================================================================\n");
    std::printf("  Silicon & Metal 4 : %s | ZeroCopy=%d | MaxSeq=%u | Boot+Warmup %.1f ms (Init MajFlt=%ld) | Metal Cache: %s (%.1f ms)\n",
                info.device_name, info.zero_copy_mmap ? 1 : 0, info.max_seq_len,
                elapsed_ms(t_boot0, t_boot1), init_majflt,
                info.metal_cache_hit ? "HIT" : "MISS", info.pso_load_ms);
    std::printf("  Zero-Copy UMA     : %.3f GiB single-buffer | Active %.3f GB/tok (48 visits) + %.3f GB/blk MTP (%s)\n",
                static_cast<double>(info.uma_mapped_bytes) / kGiB,
                static_cast<double>(info.active_bytes_per_token) / 1e9,
                static_cast<double>(info.mtp_active_bytes_per_block) / 1e9,
                info.mtp_enabled ? "ENABLED" : "DISABLED");
    std::printf("  KV & SConv Alloc  : Global KV[%s] %.2f GiB + SWA KV %.1f MiB + SConv %.1f MiB | Fixed Output Cap: %u decode toks\n",
                inkling2_kv_cache_label(info.kv_dtype),
                static_cast<double>(info.global_kv_bytes_allocated) / kGiB,
                static_cast<double>(info.swa_kv_bytes_allocated) / kMiB,
                static_cast<double>(info.sconv_state_bytes) / kMiB,
                gen_tokens);
    std::printf("  Reasoning & Spec  : %s (effort=%s) | Prefill: %s | Adaptive-K: %s | FR-Spec: %s | Tree-Spec: %s | Bitplane-Head: %s | Vision: OFF | Bundle=%s (v%u)\n",
                bench_think ? "Thinking" : "Direct (--no-think)",
                inkling2_effort_to_num_str(bench_think, bench_effort).c_str(),
                info.int8_prefill_enabled ? "INT8" : "FP16",
                eff_adaptive_k ? "ON" : "OFF",
                eff_frspec ? "ON (32K)" : "OFF",
                eff_tree_spec ? "ON" : "OFF",
                eff_bitplane ? "ON" : "OFF",
                bundle_dir.c_str(),
                bdl.hdr.version);
    std::printf("----------------------------------------------------------------------------------------------------------------------------------------\n");
    std::printf(" %6s | %5s | %9s | %9s | %9s | %9s | %7s | %8s | %9s | %19s | %6s | %s\n",
                "Ctx", "Mode", "Prefill", "TTFT(ms)", "Dec(ms)", "Decode", "Accept", "DRAM BW",
                "State MiB", "GPU/Host/TotUMA GiB", "MajFlt", "Needle");
    std::printf("----------------------------------------------------------------------------------------------------------------------------------------\n");

    struct BenchRow {
        uint32_t                 ctx_len        = 0U;
        uint32_t                 needle_offset  = 0U;
        std::vector<uint32_t>    needle_offsets;
        std::vector<std::string> passkeys;
        uint32_t                 needles_hit    = 0U;
        uint32_t                 draft_k        = 0U;
        std::string              passkey_label;
        bool                     needle_pass    = false;
        bool                     exact_k0_match = true;
        double                   speedup_vs_k0  = 1.0;
        std::string              first_line;
        LoopSanityReport         loop_rep;
        Inkling2RunResult        r;
    };

    std::vector<BenchRow> rows;
    HaystackTokenizerCache hay_cache{};
    bool all_ok = true;

    StrataInkling2SamplingParams greedy{};
    greedy.greedy             = true;
    greedy.temperature        = 0.0f;
    greedy.rng_seed           = 42ULL;
    greedy.repetition_penalty = 1.0f;

    for (uint32_t ctx : contexts) {
        PasskeyPromptSpec pspec{};
        if (!build_passkey_prompt_exact(
                bdl.tokenizer, &hay_cache, ctx, custom_needle_depth, bench_think, bench_effort, &pspec)) {
            std::fprintf(stderr, "[bench-inkling2] Failed to construct exact %u-token needle prompt\n", ctx);
            if (bench_ckpt) strata_inkling2_checkpoint_destroy(bench_ckpt);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl);
            return 1;
        }

        PrefillReuseSlot reuse_slot{};
        reuse_slot.ckpt  = bench_ckpt;
        reuse_slot.valid = false;

        std::vector<uint32_t> k0_ids;
        double k0_decode_tok_s = 0.0;
        bool have_k0 = false;

        for (uint32_t k : draft_ks) {
            BenchRow row{};
            row.ctx_len        = ctx;
            row.needle_offset  = pspec.needle_offsets.empty() ? 0U : pspec.needle_offsets[0];
            row.needle_offsets = pspec.needle_offsets;
            row.passkeys       = pspec.passkeys;
            row.draft_k        = k;
            row.passkey_label  = pspec.summary_label;

            const int rc = execute_inkling2_sequence(
                &bdl, eng, info, pspec.prompt_ids, gen_tokens, /*count_decode_only=*/true,
                k, greedy, /*stop_on_eos=*/false, /*stream_stdout=*/false, &row.r,
                (bench_ckpt != nullptr) ? &reuse_slot : nullptr);
            if (rc != 0) {
                std::fprintf(stderr, "[bench-inkling2] Engine failed at ctx=%u K=%u (rc=%d)\n", ctx, k, rc);
                if (bench_ckpt) strata_inkling2_checkpoint_destroy(bench_ckpt);
                strata_inkling2_gpu_destroy(eng);
                strata_inkling2_free_bundle(&bdl);
                return 1;
            }

            const std::string decoded = decode_tokens_str(bdl.tokenizer, row.r.generated_ids, /*skip_special=*/true);
            uint32_t hits = 0U;
            for (const std::string& pk : row.passkeys) {
                if (decoded.find(pk) != std::string::npos) {
                    ++hits;
                }
            }
            row.needles_hit = hits;
            row.needle_pass = (!row.passkeys.empty() && hits == static_cast<uint32_t>(row.passkeys.size()));
            row.loop_rep    = analyze_shingle_loops(row.r.generated_ids);

            const size_t nl = decoded.find('\n');
            row.first_line = (nl == std::string::npos) ? decoded : decoded.substr(0U, nl);
            if (row.first_line.size() > 76U) {
                row.first_line = row.first_line.substr(0U, 76U) + "...";
            }

            if (k == 0U) {
                k0_ids          = row.r.generated_ids;
                k0_decode_tok_s = row.r.decode_tok_s;
                have_k0         = true;
            } else if (have_k0) {
                const size_t cmp_n = std::min(k0_ids.size(), row.r.generated_ids.size());
                row.exact_k0_match = std::equal(k0_ids.begin(), k0_ids.begin() + cmp_n, row.r.generated_ids.begin());
                row.speedup_vs_k0  = (k0_decode_tok_s > 0.0) ? (row.r.decode_tok_s / k0_decode_tok_s) : 1.0;
                if (!row.exact_k0_match) all_ok = false;
            }

            if (!row.needle_pass) all_ok = false;

            char mode_str[16];
            if (k == 0U) {
                std::snprintf(mode_str, sizeof(mode_str), "K=0");
            } else {
                std::snprintf(mode_str, sizeof(mode_str), "MTP%u", k);
            }

            char acc_str[16];
            if (k == 0U || row.r.mtp_drafted == 0U) {
                std::snprintf(acc_str, sizeof(acc_str), "   -   ");
            } else {
                std::snprintf(acc_str, sizeof(acc_str), "%5.1f%%",
                              100.0 * static_cast<double>(row.r.mtp_accepted) / static_cast<double>(row.r.mtp_drafted));
            }

            char uma_str[32];
            std::snprintf(uma_str, sizeof(uma_str), "%5.2f/%4.2f/%5.2f",
                          row.r.footprint_gib, row.r.host_rss_gib, row.r.rss_gib);

            char needle_tag[64];
            if (row.passkeys.size() > 1U) {
                std::snprintf(needle_tag, sizeof(needle_tag), "%s (%u/%zu:%s)",
                              row.needle_pass ? "PASS" : "FAIL",
                              row.needles_hit, row.passkeys.size(), row.passkey_label.c_str());
            } else {
                std::snprintf(needle_tag, sizeof(needle_tag), "%s (%s)",
                              row.needle_pass ? "PASS" : "FAIL", row.passkey_label.c_str());
            }

            std::printf(" %6u | %5s | %7.1f/s | %9.1f | %9.1f | %7.2f/s | %7s | %6.1fGB | %9.1f | %19s | %6ld | %s\n",
                        row.ctx_len,
                        mode_str,
                        row.r.prefill_tok_s,
                        row.r.ttft_ms,
                        row.r.decode_ms,
                        row.r.decode_tok_s,
                        acc_str,
                        row.r.phys_dram_gbps,
                        row.r.active_state_mib,
                        uma_str,
                        row.r.steady_majflt,
                        needle_tag);
            std::fflush(stdout);

            rows.push_back(row);
        }
    }

    std::printf("----------------------------------------------------------------------------------------------------------------------------------------\n");
    std::printf("  Sample Needle Verifications, Swap Delta & MTP Greedy Equivalence:\n");
    for (const auto& row : rows) {
        if (row.draft_k == 0U && draft_ks.size() > 1U) continue;
        std::ostringstream off_oss;
        for (size_t i = 0U; i < row.needle_offsets.size(); ++i) {
            if (i > 0U) off_oss << ",";
            off_oss << row.needle_offsets[i];
        }
        char loop_tag[48] = "";
        if (row.loop_rep.evaluated) {
            std::snprintf(loop_tag, sizeof(loop_tag), " | %s(dup8=%.1f%%)",
                          row.loop_rep.loop_warning ? "LOOP_WARNING" : "Sanity=OK",
                          100.0 * row.loop_rep.dup_8gram_ratio);
        }
        std::printf("    - ctx=%6u (needle@%s, keys=%s, K=%u): %s (%u/%zu) | K=0 bit-exact=%s | Speedup=%.2fx | Swap=+%.2fGiB%s | \"%s\"\n",
                    row.ctx_len,
                    off_oss.str().c_str(),
                    row.passkey_label.c_str(),
                    row.draft_k,
                    row.needle_pass ? "PASS" : "FAIL",
                    row.needles_hit,
                    row.passkeys.size(),
                    row.exact_k0_match ? "YES" : "NO",
                    row.speedup_vs_k0,
                    row.r.swap_delta_gib,
                    loop_tag,
                    row.first_line.c_str());
    }
    std::printf("========================================================================================================================================\n");

    if (!json_out_path.empty()) {
        std::ofstream jfs(json_out_path, std::ios::out | std::ios::trunc);
        if (jfs) {
            jfs << "{\n"
                << "  \"model\": \"thinkingmachines/Inkling-2-Mini\",\n"
                << "  \"device\": \"" << info.device_name << "\",\n"
                << "  \"uma_mapped_gib\": " << (static_cast<double>(info.uma_mapped_bytes) / kGiB) << ",\n"
                << "  \"active_gb_per_tok\": " << (static_cast<double>(info.active_bytes_per_token) / 1e9) << ",\n"
                << "  \"mtp_active_gb_per_block\": " << (static_cast<double>(info.mtp_active_bytes_per_block) / 1e9) << ",\n"
                << "  \"mtp_enabled\": " << (info.mtp_enabled ? "true" : "false") << ",\n"
                << "  \"kv_cache\": \"" << inkling2_kv_cache_label(info.kv_dtype) << "\",\n"
                << "  \"int8_prefill\": " << (info.int8_prefill_enabled ? "true" : "false") << ",\n"
                << "  \"adaptive_k\": " << (eff_adaptive_k ? "true" : "false") << ",\n"
                << "  \"frspec\": " << (eff_frspec ? "true" : "false") << ",\n"
                << "  \"tree_spec\": " << (eff_tree_spec ? "true" : "false") << ",\n"
                << "  \"bitplane_head\": " << (eff_bitplane ? "true" : "false") << ",\n"
                << "  \"vision\": false,\n"
                << "  \"metal_cache_hit\": " << (info.metal_cache_hit ? "true" : "false") << ",\n"
                << "  \"pso_load_ms\": " << info.pso_load_ms << ",\n"
                << "  \"init_majflt\": " << init_majflt << ",\n"
                << "  \"thinking\": " << (bench_think ? "true" : "false") << ",\n"
                << "  \"effort\": \"" << (bench_think ? bench_effort : "none") << "\",\n"
                << "  \"gen_tokens_cap\": " << gen_tokens << ",\n"
                << "  \"results\": [\n";
            for (size_t i = 0U; i < rows.size(); ++i) {
                const auto& r = rows[i];
                jfs << "    {\"context\": " << r.ctx_len
                    << ", \"needle_offset\": " << r.needle_offset
                    << ", \"needles_hit\": " << r.needles_hit
                    << ", \"needles_total\": " << r.passkeys.size()
                    << ", \"draft_k\": " << r.draft_k
                    << ", \"prefill_tok_s\": " << r.r.prefill_tok_s
                    << ", \"ttft_ms\": " << r.r.ttft_ms
                    << ", \"decode_ms\": " << r.r.decode_ms
                    << ", \"decode_emitted\": " << r.r.decode_emitted
                    << ", \"decode_tok_s\": " << r.r.decode_tok_s
                    << ", \"mtp_drafted\": " << r.r.mtp_drafted
                    << ", \"mtp_accepted\": " << r.r.mtp_accepted
                    << ", \"phys_dram_gbps\": " << r.r.phys_dram_gbps
                    << ", \"eff_dram_gbps\": " << r.r.eff_dram_gbps
                    << ", \"active_state_mib\": " << r.r.active_state_mib
                    << ", \"total_uma_gib\": " << r.r.rss_gib
                    << ", \"gpu_wired_gib\": " << r.r.footprint_gib
                    << ", \"host_rss_gib\": " << r.r.host_rss_gib
                    << ", \"rss_gib\": " << r.r.rss_gib
                    << ", \"phys_footprint_gib\": " << r.r.footprint_gib
                    << ", \"peak_rss_gib\": " << r.r.peak_rss_gib
                    << ", \"swap_delta_gib\": " << r.r.swap_delta_gib
                    << ", \"steady_majflt\": " << r.r.steady_majflt
                    << ", \"needle_pass\": " << (r.needle_pass ? "true" : "false")
                    << ", \"exact_k0_match\": " << (r.exact_k0_match ? "true" : "false")
                    << ", \"speedup_vs_k0\": " << r.speedup_vs_k0
                    << ", \"dup_8gram_ratio\": " << r.loop_rep.dup_8gram_ratio
                    << ", \"dup_4gram_ratio\": " << r.loop_rep.dup_4gram_ratio
                    << ", \"loop_warning\": " << (r.loop_rep.loop_warning ? "true" : "false")
                    << "}" << ((i + 1U < rows.size()) ? "," : "") << "\n";
            }
            jfs << "  ]\n}\n";
        }
    }

    if (bench_ckpt) strata_inkling2_checkpoint_destroy(bench_ckpt);
    strata_inkling2_gpu_destroy(eng);
    strata_inkling2_free_bundle(&bdl);
    return all_ok ? 0 : 1;
}

int strata_inkling2_cli_unit_tests(void) {
    int fails = 0;
    std::string err;

    /* 1. run-inkling2 defaults: tree_spec=true, bitplane_head=true, image_paths empty */
    {
        const char* av[] = {"strata_cli", "run-inkling2", "--prompt", "Hello"};
        Inkling2RunCliOpts o{};
        if (parse_inkling2_run_args(4, const_cast<char**>(av), &o, &err) != 0 ||
            !o.tree_spec || !o.bitplane_head || !o.image_paths.empty() || o.draft_k != 7U) {
            std::fprintf(stderr, "  [FAIL] strata_inkling2_cli_unit_tests: run-inkling2 defaults failed (%s)\n", err.c_str());
            ++fails;
        }
    }

    /* 2. run-inkling2 repeatable --image + --no-tree-spec + --no-bitplane-head */
    {
        const char* av[] = {
            "strata_cli", "run-inkling2",
            "--image", "/tmp/a.png",
            "--image", "/tmp/b.jpg",
            "--no-tree-spec",
            "--no-bitplane-head",
            "--vision", "/tmp/custom_vision.strata"
        };
        Inkling2RunCliOpts o{};
        if (parse_inkling2_run_args(10, const_cast<char**>(av), &o, &err) != 0 ||
            o.image_paths.size() != 2U ||
            o.image_paths[0] != "/tmp/a.png" ||
            o.image_paths[1] != "/tmp/b.jpg" ||
            o.tree_spec || o.bitplane_head ||
            o.vision_path != "/tmp/custom_vision.strata") {
            std::fprintf(stderr, "  [FAIL] strata_inkling2_cli_unit_tests: run-inkling2 --image/--no-tree-spec/--no-bitplane-head failed (%s)\n", err.c_str());
            ++fails;
        }
    }

    /* 3. run-inkling2 explicit --tree-spec --bitplane-head re-enable */
    {
        const char* av[] = {
            "strata_cli", "run-inkling2", "--no-tree-spec", "--tree-spec", "--no-bitplane-head", "--bitplane-head"
        };
        Inkling2RunCliOpts o{};
        if (parse_inkling2_run_args(6, const_cast<char**>(av), &o, &err) != 0 ||
            !o.tree_spec || !o.bitplane_head) {
            std::fprintf(stderr, "  [FAIL] strata_inkling2_cli_unit_tests: run-inkling2 toggle re-enable failed (%s)\n", err.c_str());
            ++fails;
        }
    }

    /* 4. bench-inkling2 defaults & --no-tree-spec --no-bitplane-head */
    {
        const char* av_def[] = {"strata_cli", "bench-inkling2"};
        Inkling2BenchCliOpts bdef{};
        if (parse_inkling2_bench_args(2, const_cast<char**>(av_def), &bdef, &err) != 0 ||
            !bdef.tree_spec || !bdef.bitplane_head) {
            std::fprintf(stderr, "  [FAIL] strata_inkling2_cli_unit_tests: bench-inkling2 defaults failed (%s)\n", err.c_str());
            ++fails;
        }
        const char* av_off[] = {"strata_cli", "bench-inkling2", "--no-tree-spec", "--no-bitplane-head"};
        Inkling2BenchCliOpts boff{};
        if (parse_inkling2_bench_args(4, const_cast<char**>(av_off), &boff, &err) != 0 ||
            boff.tree_spec || boff.bitplane_head) {
            std::fprintf(stderr, "  [FAIL] strata_inkling2_cli_unit_tests: bench-inkling2 --no-tree-spec/--no-bitplane-head failed (%s)\n", err.c_str());
            ++fails;
        }
    }

    return fails;
}

} /* extern "C" */

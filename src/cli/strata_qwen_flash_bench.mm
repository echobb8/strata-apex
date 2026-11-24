/*
 * STRATA-M5 v2.0 — Qwen/Qwen3.8-Flash-Next (180B MoE + 51.2B NVMe PLE + Native MTP) CLI & Benchmark Suite
 * File: src/cli/strata_qwen_flash_bench.mm
 *
 * Implements:
 *   - `convert-flash`: Pipelined 131-shard HF downloader + ARM NEON Q4/FP16/FP32 streaming converter
 *   - `inspect-flash`: Zero-copy Two-Part UMA + NVMe PLE bundle manifest & memory layout inspector
 *   - `run-flash`:     Interactive/file-prompt generation (`--no-think` vs `<think>`, `K=0` vs Native `MTP` `K=1..4`)
 *   - `bench-flash`:   512 -> 64K long-context throughput, DRAM GB/s, NVMe PLE I/O, UMA RSS, MajFlt & Passkey Needle suite
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_qwen_flash.h"

#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
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
    const StrataQwenFlashTokenizer* tok,
    const std::string& text,
    bool parse_special,
    std::vector<uint32_t>* out_ids
) {
    const uint32_t cap = static_cast<uint32_t>(std::max<size_t>(text.size() + 128U, 256U));
    out_ids->assign(cap, 0U);
    uint32_t n = 0U;
    if (strata_qwen_flash_encode(tok, text.c_str(), text.size(), parse_special, out_ids->data(), cap, &n) != 0) {
        return false;
    }
    out_ids->resize(n);
    return true;
}

/* Chat-mode prompt tokenization for the one-shot command.
 *
 * Only the TEMPLATE may spell control tokens. The user's own text is encoded as plain text (as the REPL and the
 * server already do), so `--prompt "$(cat untrusted.txt)"` cannot forge a role boundary or an end-of-turn marker.
 *
 * `rendered_a` / `rendered_b` are the chat template rendered around the one-byte probes "A" and "B". The formatter
 * emits `prefix + user_text + suffix`, so the single byte where the two renderings differ is the user slot. */
static bool encode_chat_prompt_plain_user(
    const StrataQwenFlashTokenizer* tok,
    const std::string& rendered_a,
    const std::string& rendered_b,
    const std::string& user_text,
    std::vector<uint32_t>* out_ids
) {
    if (rendered_a.size() != rendered_b.size()) return false;
    size_t slot = 0U;
    while (slot < rendered_a.size() && rendered_a[slot] == rendered_b[slot]) ++slot;
    if (slot == rendered_a.size() ||
        rendered_a.compare(slot + 1U, std::string::npos, rendered_b, slot + 1U, std::string::npos) != 0) {
        return false;
    }
    const std::string parts[3] = {rendered_a.substr(0U, slot), user_text, rendered_a.substr(slot + 1U)};
    out_ids->clear();
    std::vector<uint32_t> piece;
    for (size_t i = 0U; i < 3U; ++i) {
        if (parts[i].empty()) continue;
        if (!encode_string_vec(tok, parts[i], /*parse_special=*/(i != 1U), &piece)) return false;
        out_ids->insert(out_ids->end(), piece.begin(), piece.end());
    }
    return true;
}

static std::string decode_tokens_str(
    const StrataQwenFlashTokenizer* tok,
    const std::vector<uint32_t>& ids,
    bool skip_special
) {
    if (ids.empty()) return {};
    const size_t cap = std::max<size_t>(ids.size() * 32U + 256U, 1024U);
    std::vector<char> buf(cap, '\0');
    size_t len = 0U;
    if (strata_qwen_flash_decode(
            tok, ids.data(), static_cast<uint32_t>(ids.size()), skip_special, buf.data(), cap, &len) != 0) {
        return {};
    }
    return std::string(buf.data(), len);
}

static void stream_emit_token(
    const StrataQwenFlashTokenizer* tok,
    StrataQwenFlashStreamDecoder* dec,
    uint32_t token_id
) {
    if (token_id == STRATA_QWEN_FLASH_TOKEN_THINK_START) {
        std::fputs("<think>", stdout);
        std::fflush(stdout);
        return;
    }
    if (token_id == STRATA_QWEN_FLASH_TOKEN_THINK_END) {
        std::fputs("</think>", stdout);
        std::fflush(stdout);
        return;
    }
    char piece[256];
    size_t piece_len = 0U;
    if (strata_qwen_flash_stream_decode_step(tok, dec, token_id, /*skip_special=*/true, piece, sizeof(piece), &piece_len) == 0 &&
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
    StrataQwenFlashBundle* bundle,
    StrataQwenFlashGpuEngine* eng
) {
    @autoreleasepool {
        strata_qwen_flash_gpu_reset(eng);
        strata_qwen_flash_gpu_reset_counters(eng);

        const uint32_t warmup_ids[8] = {
            STRATA_QWEN_FLASH_TOKEN_IM_START, 872U, 198U, 9707U,
            STRATA_QWEN_FLASH_TOKEN_IM_END, 198U, STRATA_QWEN_FLASH_TOKEN_IM_START, 77091U
        };
        StrataQwenFlashSamplingParams greedy{};
        greedy.greedy      = true;
        greedy.temperature = 0.0f;
        greedy.rng_seed    = 1ULL;

        uint32_t next_tok = 0U;
        (void)strata_qwen_flash_gpu_prefill(eng, warmup_ids, 8U, &greedy, &next_tok, nullptr);
        uint32_t next2 = 0U;
        (void)strata_qwen_flash_gpu_decode_step(eng, next_tok, &greedy, &next2, nullptr);
        if (bundle->cfg.has_mtp) {
            uint32_t emitted[8] = {};
            uint32_t n_emit = 0U;
            uint32_t n_acc  = 0U;
            const uint32_t warm_k = strata_qwen_flash_gpu_get_spec_tree(eng) ? 3U : 2U;
            (void)strata_qwen_flash_gpu_spec_decode_step(eng, next2, warm_k, &greedy, emitted, &n_emit, &n_acc);
        }

        strata_qwen_flash_gpu_reset(eng);
        strata_qwen_flash_gpu_reset_counters(eng);
        strata_qwen_flash_ple_cache_reset_stats(&bundle->ple_cache);
    }
}

/* Diverse technical corpus paragraphs for realistic Needle-in-a-Haystack construction (non-trivial N-grams). */
static const char* const kHaystackParagraphs[] = {
    "Modern unified memory architectures integrate central processor clusters, neural accelerators, and wide graphics cores onto a single coherent LPDDR5X memory fabric, eliminating PCIe bounce buffers during autoregressive inference.",
    "By mapping page-aligned immutable weight containers directly into GPU virtual address space through Mach virtual memory descriptors, zero-copy residency sets avoid duplicate host allocations across multi-tens-of-gigabyte models.",
    "Linear-time gated delta networks maintain fixed-dimensional matrix recurrence states updated via rank-one associative householder-like delta rules, providing bounded memory footprint regardless of sequence length.",
    "Interleaving three gated linear recurrence blocks with one sparse block-indexed full attention layer preserves exact long-range token retrieval while cutting quadratic key-value cache growth by seventy-five percent.",
    "Block-compressed query-sensitive attention indexers average consecutive key projections into compact four-token landmarks, applying rotary position embeddings before scoring candidate blocks with rectified multi-head dot products.",
    "Hardware radix selection over IEEE-754 floating-point block scores identifies the top five hundred twelve memory blocks in a few microseconds on graphics threadgroups without host synchronization barriers.",
    "Mixture-of-experts feed-forward layers route each hidden representation to ten specialized experts out of five hundred twelve candidates, activating under two percent of routed parameters on any individual token step.",
    "Combining fine-grained routed SwiGLU experts with an always-active sigmoid-gated shared expert stabilizes gradient flow during pretraining and captures common syntactic transformations without routing overhead.",
    "Four-stream gated hyper-connections generalize standard residual pathways by maintaining parallel hidden lanes mixed through low-rank bottleneck projections prior to each attention and expert sublayer.",
    "Per-layer n-gram embedding tables decouple lexical and factual surface capacity from active compute FLOPs by hashing causal bigram and trigram histories through SplitMix64 multipliers modulo consecutive prime vocabularies.",
    "Streaming compact four-bit affine n-gram rows from NVMe flash using uncached file descriptors prevents thirty-gigabyte embedding tables from evicting wired transformer weights inside unified DRAM.",
    "A head-partitioned four-way least-frequently-used cache pinned with mlock captures Zipfian natural-language bigram and trigram locality across sixteen independent lookup banks without mutex contention.",
    "Dilated causal depthwise convolutions over gated per-layer embedding values propagate multi-token lexical context across three-step strides using a tiny nine-step rolling state buffer.",
    "Native multi-token prediction heads fuse the four-stream pre-mixer backbone state with next-token embeddings through a single sparse-attention mixture-of-experts block to draft multiple future tokens directly on GPU.",
    "During speculative verification, evaluating the committed token alongside k drafted candidates in a single batched pass amortizes expert router and weight memory traffic across multiple accepted tokens.",
    "Exact Leviathan rejection sampling and deterministic greedy prefix verification guarantee that speculative multi-token decoding follows the exact target distribution of the underlying language model.",
    "When a drafted token diverges from the main model verification pass, constant-time device-side state snapshots restore recurrent delta matrices, causal convolution buffers, and n-gram history pointers cleanly.",
    "Group-wise four-bit affine quantization packs eight unsigned nibbles per thirty-two-bit word alongside half-precision scale and bias pairs every sixty-four input channels to approach memory-bandwidth-optimal GEMV.",
    "Cooperative tensor matrix primitives in Metal 4 schedule SIMD-group tiles across neural hardware execution units during batched prefill, reaching high compute utilization on dense and grouped projections.",
    "NeoX half-split rotary position embeddings rotate only the leading sixty-four dimensions of each two-hundred-fifty-six-dimensional attention head at base frequency ten million to extrapolate cleanly past sixty-four thousand tokens.",
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
    "Formal model checkers exhaustively explore reachable state transitions or encode bounded temporal logic properties into conjunctive normal form clauses solved via conflict-driven clause learning."
};

struct HaystackTokenizerCache {
    std::vector<std::vector<uint32_t>> para_tokens;
    std::vector<uint32_t>              space_tokens;
    bool                               initialized = false;
};

static bool init_haystack_cache(const StrataQwenFlashTokenizer* tok, HaystackTokenizerCache* cache) {
    if (cache->initialized) return true;
    const size_t n_paras = sizeof(kHaystackParagraphs) / sizeof(kHaystackParagraphs[0]);
    cache->para_tokens.resize(n_paras);
    for (size_t i = 0U; i < n_paras; ++i) {
        std::string s = std::string(" [Archive Section ") + std::to_string(i + 1U) + "] " + kHaystackParagraphs[i] + "\n";
        if (!encode_string_vec(tok, s, /*parse_special=*/false, &cache->para_tokens[i]) || cache->para_tokens[i].empty()) {
            return false;
        }
    }
    if (!encode_string_vec(tok, ".", /*parse_special=*/false, &cache->space_tokens) || cache->space_tokens.empty()) {
        return false;
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
        *rng_state = sqf_splitmix64(*rng_state);
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
    uint64_t h = sqf_splitmix64(0x5354524154414D35ULL ^
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

static std::string build_reasoning_system_header(bool enable_thinking, const std::string& effort_raw) {
    if (!enable_thinking) return {};
    std::string e = effort_raw.empty() ? "low" : effort_raw;
    if (e == "high" || e == "max") e = "xhigh";
    if (e == "low") {
        return "<|im_start|>system\n"
               "Reasoning effort is set to low. Keep your thinking brief and focused, "
               "moving directly to the conclusion without unnecessary elaboration."
               "<|im_end|>\n";
    }
    if (e == "xhigh") {
        return "<|im_start|>system\n"
               "Reasoning effort is set to xhigh. Please think carefully through the task, "
               "validate key assumptions, consider plausible alternatives, and prioritize "
               "correctness, consistency, and clarity in the final answer."
               "<|im_end|>\n";
    }
    return {};
}

static bool build_passkey_prompt_exact(
    const StrataQwenFlashTokenizer* tok,
    HaystackTokenizerCache* cache,
    uint32_t target_ctx,
    double custom_needle_depth,
    bool enable_thinking,
    const std::string& effort,
    PasskeyPromptSpec* out_spec
) {
    if (!init_haystack_cache(tok, cache) || !out_spec) return false;
    *out_spec = PasskeyPromptSpec{};

    const std::string sys_hdr   = build_reasoning_system_header(enable_thinking, effort);
    const std::string asst_tail = enable_thinking
        ? "<|im_start|>assistant\n<think>\n"
        : "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    // Ultra-long contexts (>= 128K = 131,072) without custom single depth use 3-depth multi-needle recall
    // (early/deep 10%, mid 50%, late 85%) to test QSA sparse indexing across the entire context span!
    if (target_ctx >= 131072U && custom_needle_depth < 0.0) {
        const std::string key_a = deterministic_passkey_for_ctx_slot(target_ctx, 0U);
        const std::string key_b = deterministic_passkey_for_ctx_slot(target_ctx, 1U);
        const std::string key_c = deterministic_passkey_for_ctx_slot(target_ctx, 2U);

        const std::string prefix_str =
            sys_hdr +
            "<|im_start|>user\n"
            "Read the following technical archive carefully. Three confidential 7-digit STRATA vault passkeys "
            "(ALPHA, BETA, and GAMMA) are recorded at early, middle, and late positions inside the archive.\n\n";
        const std::string needle_a =
            "\n[CONFIDENTIAL VAULT RECORD ALPHA] The exact 7-digit STRATA ALPHA passkey is " +
            key_a + ". Verify ALPHA passkey: " + key_a + ".\n";
        const std::string needle_b =
            "\n[CONFIDENTIAL VAULT RECORD BETA] The exact 7-digit STRATA BETA passkey is " +
            key_b + ". Verify BETA passkey: " + key_b + ".\n";
        const std::string needle_c =
            "\n[CONFIDENTIAL VAULT RECORD GAMMA] The exact 7-digit STRATA GAMMA passkey is " +
            key_c + ". Verify GAMMA passkey: " + key_c + ".\n";
        const std::string suffix_str =
            "\nQuestion: On the very first line, state the exact 7-digit STRATA passkeys for ALPHA, BETA, and GAMMA "
            "(format: ALPHA=<digits>, BETA=<digits>, GAMMA=<digits>). "
            "Then write a detailed multi-paragraph technical analysis of unified memory architecture, hybrid linear and sparse block attention, "
            "NVMe per-layer n-gram embeddings, and native multi-token speculative decoding.<|im_end|>\n" +
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
            "<|im_start|>user\n"
            "Read the following technical archive carefully. A confidential 7-digit STRATA vault passkey is recorded inside one entry.\n\n";
        needle_str =
            "\n[CONFIDENTIAL VAULT RECORD] The exact 7-digit STRATA vault passkey for this session is " +
            passkey + ". Verify passkey: " + passkey + ".\n";
        suffix_str =
            "\nQuestion: On the very first line, state the exact 7-digit STRATA vault passkey recorded in the archive above. "
            "Then write a detailed multi-paragraph technical analysis of unified memory architecture, hybrid linear and sparse block attention, "
            "NVMe per-layer n-gram embeddings, and native multi-token speculative decoding.<|im_end|>\n" +
            asst_tail;
    } else {
        prefix_str = sys_hdr + "<|im_start|>user\nArchive:\n";
        needle_str = "\nPasskey: " + passkey + ".\n";
        suffix_str = "\nState the 7-digit passkey first, then explain unified memory in detail.<|im_end|>\n" +
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

struct FlashRunResult {
    uint32_t              prompt_tokens          = 0U;
    uint32_t              emitted_total          = 0U; /* first_tok + decode_emitted */
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
    uint64_t              prefill_ple_ios        = 0ULL;
    uint64_t              prefill_ple_hits       = 0ULL;
    uint64_t              decode_ple_ios         = 0ULL;
    uint64_t              decode_ple_hits        = 0ULL;
    uint64_t              total_ple_ios          = 0ULL;
    uint64_t              total_ple_hits         = 0ULL;
    uint64_t              total_ple_bytes        = 0ULL;
    uint32_t              final_seq_pos          = 0U;
    double                active_state_mib       = 0.0;
    double                kv_active_mib          = 0.0;
    double                deltanet_mib           = 0.0;
    double                ple_state_mib          = 0.0;
    double                rss_gib                = 0.0; /* Total UMA (GPU Wired + Host RSS) */
    double                host_rss_gib           = 0.0; /* Host CPU resident_size */
    double                footprint_gib          = 0.0; /* GPU Wired GiB */
    double                peak_rss_gib           = 0.0;
    double                swap_delta_gib         = 0.0;
    long                  steady_majflt          = 0;
    std::vector<uint32_t> generated_ids;
};

struct PrefillProgressCtx {
    SteadyClock::time_point t0;
    uint32_t                draft_k   = 0U;
    uint32_t                last_emit = 0U;
    bool                    printed   = false;
};

static void bench_prefill_progress_cb(
    uint32_t done_tokens,
    uint32_t total_tokens,
    uint32_t /*seq_pos*/,
    void*    user_ctx
) {
    auto* ctx = static_cast<PrefillProgressCtx*>(user_ctx);
    if (!ctx || total_tokens < 65536U) return;
    if (done_tokens < total_tokens && (done_tokens - ctx->last_emit) < 16384U) return;
    ctx->last_emit = done_tokens;
    ctx->printed   = true;
    const double ms  = elapsed_ms(ctx->t0, SteadyClock::now());
    const double tps = (ms > 0.0) ? (static_cast<double>(done_tokens) / (ms / 1000.0)) : 0.0;
    const double pct = 100.0 * static_cast<double>(done_tokens) / static_cast<double>(total_tokens);
    std::fprintf(stderr,
                 "\r  [prefill ctx=%u K=%u] %6u / %6u toks (%5.1f%%) | %6.1f tok/s | %.1f s   ",
                 total_tokens, ctx->draft_k, done_tokens, total_tokens, pct, tps, ms / 1000.0);
    std::fflush(stderr);
}

struct PrefillReuseSlot {
    StrataQwenFlashCheckpoint* ckpt             = nullptr;
    bool                       valid            = false;
    uint32_t                   first_tok        = 0U;
    double                     ttft_ms          = 0.0;
    double                     prefill_tok_s    = 0.0;
    uint64_t                   prefill_ple_ios  = 0ULL;
    uint64_t                   prefill_ple_hits = 0ULL;
    uint64_t                   prefill_ple_bytes= 0ULL;
};

static int execute_flash_sequence(
    StrataQwenFlashBundle*               bundle,
    StrataQwenFlashGpuEngine*            eng,
    const StrataQwenFlashGpuInfo&        info,
    const std::vector<uint32_t>&         prompt_ids,
    uint32_t                             max_new_tokens,
    bool                                 count_decode_only, /* true in bench: run exact `max_new_tokens` decode tokens after prefill */
    uint32_t                             draft_k,
    const StrataQwenFlashSamplingParams& base_sampling,
    bool                                 stop_on_eos,
    bool                                 stream_stdout,
    FlashRunResult*                      res_out,
    PrefillReuseSlot*                    reuse_slot = nullptr
) {
    *res_out = FlashRunResult{};
    if (prompt_ids.empty()) return -1;

    const uint32_t eff_k = bundle->cfg.has_mtp
        ? std::min<uint32_t>(draft_k, STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K)
        : 0U;

    StrataQwenFlashSamplingParams step_samp = base_sampling;
    uint32_t first_tok = 0U;
    ProcessMemStats mem0{};
    StrataQwenFlashPerfCounters cnt_pre{};

    if (reuse_slot && reuse_slot->ckpt && reuse_slot->valid) {
        if (strata_qwen_flash_checkpoint_restore(eng, reuse_slot->ckpt) != 0) {
            return -1;
        }
        mem0 = query_process_mem_stats();
        strata_qwen_flash_gpu_get_counters(eng, &cnt_pre);
        first_tok                 = reuse_slot->first_tok;
        res_out->prompt_tokens    = static_cast<uint32_t>(prompt_ids.size());
        res_out->ttft_ms          = reuse_slot->ttft_ms;
        res_out->prefill_tok_s    = reuse_slot->prefill_tok_s;
        res_out->prefill_ple_ios  = reuse_slot->prefill_ple_ios;
        res_out->prefill_ple_hits = reuse_slot->prefill_ple_hits;
    } else {
        strata_qwen_flash_gpu_reset(eng);
        strata_qwen_flash_gpu_reset_counters(eng);

        mem0 = query_process_mem_stats();

        const auto t_pre0 = SteadyClock::now();
        PrefillProgressCtx prog_ctx{};
        prog_ctx.t0      = t_pre0;
        prog_ctx.draft_k = eff_k;
        if (!stream_stdout && prompt_ids.size() >= 65536U) {
            strata_qwen_flash_gpu_set_prefill_progress_cb(eng, bench_prefill_progress_cb, &prog_ctx);
        }

        int rc = 0;
        @autoreleasepool {
            rc = strata_qwen_flash_gpu_prefill(
                eng,
                prompt_ids.data(),
                static_cast<uint32_t>(prompt_ids.size()),
                &step_samp,
                &first_tok,
                nullptr);
        }
        const auto t_pre1 = SteadyClock::now();
        if (!stream_stdout && prompt_ids.size() >= 65536U) {
            strata_qwen_flash_gpu_set_prefill_progress_cb(eng, nullptr, nullptr);
            if (prog_ctx.printed) {
                std::fprintf(stderr, "\r%84s\r", "");
                std::fflush(stderr);
            }
        }
        if (rc != 0) return rc;

        strata_qwen_flash_gpu_get_counters(eng, &cnt_pre);

        res_out->prompt_tokens    = static_cast<uint32_t>(prompt_ids.size());
        res_out->ttft_ms          = elapsed_ms(t_pre0, t_pre1);
        res_out->prefill_tok_s    = (res_out->ttft_ms > 0.0)
            ? (static_cast<double>(res_out->prompt_tokens) / (res_out->ttft_ms / 1000.0))
            : 0.0;
        res_out->prefill_ple_ios  = cnt_pre.ple_nvme_preads;
        res_out->prefill_ple_hits = cnt_pre.ple_lfu_hits;

        if (reuse_slot && reuse_slot->ckpt) {
            if (strata_qwen_flash_checkpoint_save(eng, reuse_slot->ckpt) == 0) {
                reuse_slot->valid             = true;
                reuse_slot->first_tok         = first_tok;
                reuse_slot->ttft_ms           = res_out->ttft_ms;
                reuse_slot->prefill_tok_s     = res_out->prefill_tok_s;
                reuse_slot->prefill_ple_ios   = res_out->prefill_ple_ios;
                reuse_slot->prefill_ple_hits  = res_out->prefill_ple_hits;
                reuse_slot->prefill_ple_bytes = cnt_pre.ple_nvme_bytes_read;
            }
        }
    }

    int rc = 0;

    StrataQwenFlashStreamDecoder dec{};
    strata_qwen_flash_stream_decoder_init(&dec);

    const uint32_t total_cap = count_decode_only ? (1U + max_new_tokens) : max_new_tokens;
    res_out->generated_ids.reserve(total_cap + 8U);

    bool hit_eos = false;
    if (total_cap > 0U) {
        if (stop_on_eos && strata_qwen_flash_is_eos_token(&bundle->cfg, first_tok)) {
            hit_eos = true;
        } else {
            res_out->generated_ids.push_back(first_tok);
            res_out->emitted_total = 1U;
            if (stream_stdout) {
                stream_emit_token(bundle->tokenizer, &dec, first_tok);
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
                rc = strata_qwen_flash_gpu_decode_step(eng, pending_tok, &step_samp, &next_tok, nullptr);
                if (rc != 0) break;
                ++res_out->decode_steps;
                if (stop_on_eos && strata_qwen_flash_is_eos_token(&bundle->cfg, next_tok)) {
                    hit_eos = true;
                    break;
                }
                res_out->generated_ids.push_back(next_tok);
                ++res_out->emitted_total;
                ++res_out->decode_emitted;
                if (stream_stdout) {
                    stream_emit_token(bundle->tokenizer, &dec, next_tok);
                }
                pending_tok = next_tok;
            } else {
                uint32_t emitted_buf[STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t num_emitted  = 0U;
                uint32_t num_accepted = 0U;
                rc = strata_qwen_flash_gpu_spec_decode_step(
                    eng, pending_tok, eff_k, &step_samp, emitted_buf, &num_emitted, &num_accepted);
                if (rc != 0) break;
                ++res_out->decode_steps;
                for (uint32_t i = 0U; i < num_emitted; ++i) {
                    const uint32_t tok = emitted_buf[i];
                    if (stop_on_eos && strata_qwen_flash_is_eos_token(&bundle->cfg, tok)) {
                        hit_eos = true;
                        break;
                    }
                    res_out->generated_ids.push_back(tok);
                    ++res_out->emitted_total;
                    ++res_out->decode_emitted;
                    if (stream_stdout) {
                        stream_emit_token(bundle->tokenizer, &dec, tok);
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

    StrataQwenFlashPerfCounters cnt_end{};
    strata_qwen_flash_gpu_get_counters(eng, &cnt_end);
    const ProcessMemStats mem1 = query_process_mem_stats();

    res_out->decode_ms = elapsed_ms(t_dec0, t_dec1);
    res_out->decode_tok_s = (res_out->decode_ms > 0.0 && res_out->decode_emitted > 0U)
        ? (static_cast<double>(res_out->decode_emitted) / (res_out->decode_ms / 1000.0))
        : 0.0;

    res_out->mtp_drafted       = static_cast<uint32_t>(cnt_end.mtp_drafted_tokens - cnt_pre.mtp_drafted_tokens);
    res_out->mtp_accepted      = static_cast<uint32_t>(cnt_end.mtp_accepted_tokens - cnt_pre.mtp_accepted_tokens);
    res_out->decode_dram_bytes = (cnt_end.physical_dram_bytes_read >= cnt_pre.physical_dram_bytes_read)
        ? (cnt_end.physical_dram_bytes_read - cnt_pre.physical_dram_bytes_read)
        : 0ULL;

    res_out->phys_dram_gbps = (res_out->decode_ms > 0.0)
        ? ((static_cast<double>(res_out->decode_dram_bytes) / 1e9) / (res_out->decode_ms / 1000.0))
        : 0.0;
    res_out->eff_dram_gbps = (res_out->decode_ms > 0.0)
        ? ((static_cast<double>(res_out->decode_emitted) * static_cast<double>(info.active_bytes_per_token) / 1e9) /
           (res_out->decode_ms / 1000.0))
        : 0.0;

    res_out->decode_ple_ios  = cnt_end.ple_nvme_preads - cnt_pre.ple_nvme_preads;
    res_out->decode_ple_hits = cnt_end.ple_lfu_hits - cnt_pre.ple_lfu_hits;
    res_out->total_ple_ios   = res_out->prefill_ple_ios + res_out->decode_ple_ios;
    res_out->total_ple_hits  = res_out->prefill_ple_hits + res_out->decode_ple_hits;
    res_out->total_ple_bytes = (reuse_slot && reuse_slot->valid)
        ? (reuse_slot->prefill_ple_bytes + (cnt_end.ple_nvme_bytes_read - cnt_pre.ple_nvme_bytes_read))
        : cnt_end.ple_nvme_bytes_read;

    res_out->final_seq_pos = strata_qwen_flash_gpu_seq_pos(eng);
    constexpr double kMiB = 1024.0 * 1024.0;
    res_out->kv_active_mib    = (static_cast<double>(res_out->final_seq_pos) * static_cast<double>(info.kv_cache_bytes_per_token)) / kMiB;
    res_out->deltanet_mib     = static_cast<double>(info.deltanet_state_bytes) / kMiB;
    res_out->ple_state_mib    = static_cast<double>(info.ple_state_bytes) / kMiB;
    res_out->active_state_mib = res_out->kv_active_mib + res_out->deltanet_mib + res_out->ple_state_mib;

    res_out->rss_gib        = mem1.total_uma_rss_gib;
    res_out->host_rss_gib   = mem1.rss_gib;
    res_out->footprint_gib  = mem1.gpu_wired_gib;
    res_out->peak_rss_gib   = std::max(mem1.peak_rss_gib, mem1.total_uma_rss_gib);
    res_out->swap_delta_gib = std::max(0.0, mem1.swap_used_gib - mem0.swap_used_gib);
    res_out->steady_majflt  = std::max<long>(0L, mem1.majflt - mem0.majflt);

    return 0;
}

static void print_flash_convert_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s convert-flash [--hf-dir <repo_or_dir>] [--bundle <dir>] [--dl-workers <N>] [--threads <N>]\n"
        "  Defaults:\n"
        "    --hf-dir %s\n"
        "    --bundle %s\n",
        prog,
        STRATA_QWEN_FLASH_DEFAULT_HF_REPO,
        STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR);
}

static void print_flash_run_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s run-flash [--bundle <dir>] [--prompt \"<text>\" | --prompt-file <path>] [--system \"<text>\"]\n"
        "               [--no-think | --think] [--effort low|medium|xhigh] [--raw-prompt]\n"
        "               [--draft-k 0..6] [--adaptive-k | --no-adaptive-k] [--frspec | --no-frspec]\n"
        "               [--int8-prefill] [--tree | --no-tree] [--mtp-sync | --no-mtp-sync]\n"
        "               [--max-tokens 256] [--max-seq-len 263168] [--greedy | --temp 0.6]\n"
        "               [--top-k 20] [--top-p 0.95] [--presence-penalty -2..2] [--frequency-penalty -2..2]\n"
        "               [--repetition-penalty 0.5..2] [--penalty-window 0|8..65536] [--seed 42] [--verify-k0]\n",
        prog);
}

static void print_flash_bench_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s bench-flash [--bundle <dir>] [--contexts 512,4096,16384,32768,65536,131072,262144] [--gen-tokens 128]\n"
        "                 [--no-think | --think] [--effort low|medium|high|xhigh|max] [--xhigh-scaling]\n"
        "                 [--draft-k 2 | --draft-ks 0,2,3] [--adaptive-k | --no-adaptive-k] [--frspec | --no-frspec]\n"
        "                 [--int8-prefill] [--tree | --no-tree] [--mtp-sync | --no-mtp-sync]\n"
        "                 [--needle-depth 0.01..0.99] [--json <out.json>]\n"
        "\n"
        "Preset:\n"
        "  --xhigh-scaling : Runs exact 2K, 8K, and 32K input tokens (--contexts 2048,8192,32768)\n"
        "                    with --think --effort xhigh and fixed output cap (--gen-tokens 512 unless overridden),\n"
        "                    reporting TTFT (ms) separately from Decode (ms & tok/s).\n",
        prog);
}

} /* namespace */

extern "C" {

int strata_qwen_flash_cmd_convert(int argc, char** argv) {
    std::string hf_src     = STRATA_QWEN_FLASH_DEFAULT_HF_REPO;
    std::string bundle_dir = STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR;
    uint32_t dl_workers    = 0U;
    uint32_t conv_threads  = 0U;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--hf-dir" || a == "--repo" || a == "--in") && i + 1 < argc) {
            hf_src = argv[++i];
        } else if ((a == "--bundle" || a == "--out") && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if ((a == "--dl-workers" || a == "--download-concurrency") && i + 1 < argc) {
            dl_workers = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if ((a == "--threads" || a == "--convert-threads") && i + 1 < argc) {
            conv_threads = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--help" || a == "-h") {
            print_flash_convert_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[convert-flash] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    StrataQwenFlashConvertStats st{};
    const int rc = strata_qwen_flash_convert_hf(
        hf_src.c_str(), bundle_dir.c_str(), dl_workers, conv_threads, &st);
    if (rc != 0) {
        std::fprintf(stderr, "[convert-flash] Conversion failed (rc=%d)\n", rc);
        return 1;
    }

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Qwen3.8-Flash-Next (180B) Streaming Conversion Complete\n");
    std::printf("====================================================================================================\n");
    std::printf("  Source -> Bundle     : %s -> %s\n", hf_src.c_str(), bundle_dir.c_str());
    std::printf("  Shards Processed     : %u total (%u converted, %u resumed) | BF16 Payload: %.3f GiB\n",
                st.num_shards_total, st.num_shards_converted, st.num_shards_resumed,
                static_cast<double>(st.hf_safetensors_bytes) / kGiB);
    std::printf("  Tensor Accounting    : %u UMA text | %u PLE shards | %u MTP | %u vision skipped\n",
                st.num_text_tensors_converted, st.num_ple_shards_converted,
                st.num_mtp_tensors_converted, st.num_vision_tensors_skipped);
    std::printf("  uma_part0.strata     : %7.3f GiB (%llu B, 16 KiB aligned)\n",
                static_cast<double>(st.uma_part0_bytes) / kGiB,
                static_cast<unsigned long long>(st.uma_part0_bytes));
    std::printf("  uma_part1.strata     : %7.3f GiB (%llu B, 16 KiB aligned)\n",
                static_cast<double>(st.uma_part1_bytes) / kGiB,
                static_cast<unsigned long long>(st.uma_part1_bytes));
    std::printf("  ple_ngram.strata     : %7.3f GiB (%llu B, F_NOCACHE NVMe)\n",
                static_cast<double>(st.ple_ngram_bytes) / kGiB,
                static_cast<unsigned long long>(st.ple_ngram_bytes));
    std::printf("  Total Bundle / Active: %7.3f GiB total | %.3f GB active/tok (K=0)\n",
                static_cast<double>(st.total_bundle_bytes) / kGiB,
                static_cast<double>(st.active_bytes_per_tok) / 1e9);
    std::printf("  Pipeline Telemetry   : %.2f s elapsed | %.2f MB/s network | Peak RSS %.3f GiB\n",
                st.elapsed_sec, st.network_mbps, st.peak_rss_gib);
    std::printf("  Verification Gates   : 16 KiB Page Align=%s | .staging Clean=%s\n",
                st.page_aligned_verified ? "PASS" : "FAIL",
                st.staging_clean_verified ? "PASS" : "FAIL");
    std::printf("====================================================================================================\n");
    return (st.page_aligned_verified && st.staging_clean_verified) ? 0 : 1;
}

int strata_qwen_flash_cmd_repack(int argc, char** argv) {
    std::string src_dir = STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR;
    std::string dst_dir = std::string(STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR) + "-fast";
    uint32_t threads    = 0U;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--src" || a == "--in" || a == "--bundle") && i + 1 < argc) {
            src_dir = argv[++i];
        } else if ((a == "--dst" || a == "--out") && i + 1 < argc) {
            dst_dir = argv[++i];
        } else if ((a == "--threads" || a == "-j") && i + 1 < argc) {
            threads = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage:\n  %s repack-flash [--src <dir>] [--dst <dir>] [--threads <N>]\n", argv[0]);
            std::printf("  Quantizes local bundle HC mixers + PLE/MTP FC (FP16->Q8_0) & Shared Experts (FP16->Q4_A).\n");
            return 0;
        } else {
            std::fprintf(stderr, "[repack-flash] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    StrataQwenFlashConvertStats st{};
    const int rc = strata_qwen_flash_repack_bundle(src_dir.c_str(), dst_dir.c_str(), threads, &st);
    if (rc != 0) {
        std::fprintf(stderr, "[repack-flash] Repack failed (rc=%d)\n", rc);
        return 1;
    }

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Phase C Local Bundle Repack Complete (HC/PLE/FC=Q8_0, Shared=Q4_A)\n");
    std::printf("====================================================================================================\n");
    std::printf("  Source -> Dest       : %s -> %s\n", src_dir.c_str(), dst_dir.c_str());
    std::printf("  Tensors Processed    : %u (0 network bytes, PLE hardlinked)\n", st.num_text_tensors_converted);
    std::printf("  uma_part0.strata     : %7.3f GiB (%llu B, 16 KiB aligned)\n",
                static_cast<double>(st.uma_part0_bytes) / kGiB,
                static_cast<unsigned long long>(st.uma_part0_bytes));
    std::printf("  uma_part1.strata     : %7.3f GiB (%llu B, 16 KiB aligned)\n",
                static_cast<double>(st.uma_part1_bytes) / kGiB,
                static_cast<unsigned long long>(st.uma_part1_bytes));
    std::printf("  Active Traffic (K=0) : %.3f GB/tok (-19.0%% vs FP16 dense)\n",
                static_cast<double>(st.active_bytes_per_tok) / 1e9);
    std::printf("  Elapsed & Peak RSS   : %.2f s | Peak RSS %.3f GiB | 16 KiB Align=%s\n",
                st.elapsed_sec, st.peak_rss_gib, st.page_aligned_verified ? "PASS" : "FAIL");
    std::printf("====================================================================================================\n");
    return st.page_aligned_verified ? 0 : 1;
}

int strata_qwen_flash_cmd_inspect(int argc, char** argv) {
    std::string bundle_dir = STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s inspect-flash [--bundle <dir>]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[inspect-flash] Unknown option: %s\n", a.c_str());
            return 1;
        }
    }

    StrataQwenFlashBundle bdl{};
    if (strata_qwen_flash_load_bundle(bundle_dir.c_str(), 0U, &bdl) != 0) {
        std::fprintf(stderr, "[inspect-flash] Failed to load bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    const auto& c = bdl.cfg;
    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Qwen3.8-Flash-Next Disaggregated Bundle Inspection\n");
    std::printf("====================================================================================================\n");
    std::printf("  Bundle Directory  : %s\n", bdl.bundle_dir);
    std::printf("  Architecture      : qwen4_exp_text (L=%u: %u GatedDeltaNet + %u QSA w/ Indexer budget=%u)\n",
                c.num_hidden_layers, c.num_linear_layers, c.num_full_attn_layers, c.indexer_budget);
    std::printf("  Backbone & MoE    : d=%u, hc=%ux%u (lowrank=%u) | E=%u, top_k=%u, d_ff=%u + shared=%u | V=%u\n",
                c.hidden_size, c.hc_count, c.hidden_size, c.hc_lowrank,
                c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size,
                c.shared_intermediate_size, c.vocab_size);
    std::printf("  UMA Part 0 (L0-23): %7.3f GiB (%llu B, 16 KiB aligned=%s)\n",
                static_cast<double>(bdl.part0_bytes) / kGiB,
                static_cast<unsigned long long>(bdl.part0_bytes),
                (bdl.part0_bytes % STRATA_QWEN_FLASH_PAGE_BYTES == 0ULL) ? "YES" : "NO");
    std::printf("  UMA Part 1 (L24+) : %7.3f GiB (%llu B, 16 KiB aligned=%s, MTP=%s)\n",
                static_cast<double>(bdl.part1_bytes) / kGiB,
                static_cast<unsigned long long>(bdl.part1_bytes),
                (bdl.part1_bytes % STRATA_QWEN_FLASH_PAGE_BYTES == 0ULL) ? "YES" : "NO",
                bdl.header1->mtp.enabled ? "ENABLED" : "DISABLED");
    std::printf("  NVMe PLE Store    : %7.3f GiB (%u shards, %llu rows x %u B, F_NOCACHE + %.1f MiB mlock LFU)\n",
                static_cast<double>(bdl.ple_file_bytes) / kGiB,
                bdl.ple_header.num_shards,
                static_cast<unsigned long long>(bdl.ple_header.padded_vocab_rows),
                bdl.ple_header.row_bytes,
                static_cast<double>(bdl.ple_cache.arena_bytes) / kMiB);
    const char* hc_dt_str = (bdl.header0->layers[0].attn_hc.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0)
                                ? "Q8_0" : "F16";
    const char* sh_dt_str = (bdl.header0->layers[0].shared_gate_proj.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A)
                                ? "Q4_A" : "F16";
    std::printf("  Active Traffic    : %.3f GB/tok Main (48L) | %.3f GB/draft MTP (1L) | Dense: HC=%s, Shared=%s | Vocab=%u\n",
                static_cast<double>(bdl.header0->active_bytes_per_tok) / 1e9,
                static_cast<double>(bdl.header0->mtp_active_bytes_per_tok) / 1e9,
                hc_dt_str, sh_dt_str,
                strata_qwen_flash_tokenizer_vocab_size(bdl.tokenizer));
    std::printf("====================================================================================================\n");

    strata_qwen_flash_free_bundle(&bdl);
    return 0;
}

int strata_qwen_flash_cmd_run(int argc, char** argv) {
    std::string bundle_dir = STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR;
    std::string prompt     = "Explain why decoupling 51.2B N-gram embeddings onto NVMe via F_NOCACHE while wiring 68.6 GiB of 512-expert MoE weights across two Metal 4 zero-copy UMA buffers enables 180B inference on a 128 GiB laptop.";
    std::string prompt_file;
    std::string system_prompt;
    std::string effort;
    bool enable_thinking  = true;
    bool raw_prompt       = false;
    bool verify_k0        = false;
    bool temp_explicit    = false;
    bool stop_on_eos      = true;
    bool spec_tree        = false;
    bool mtp_kv_sync      = true;
    bool mtp_shortlist_set= false;
    uint32_t mtp_shortlist= 32768U;
    bool ngram_set        = false;
    bool ngram_draft      = false;
    bool int8_prefill     = false;
    bool int8_prefill_set = false;
    bool adaptive_k       = true;
    bool adaptive_k_set   = false;
    bool frspec           = true;
    bool frspec_set       = false;
    uint32_t max_tokens   = 256U;
    uint32_t max_seq_len  = 0U;
    uint32_t draft_k      = 3U;
    size_t ple_cache_mib  = 0U;

    StrataQwenFlashSamplingParams samp{};
    samp.greedy      = false;
    samp.temperature = 0.6f;
    samp.top_k       = 20U;
    samp.top_p       = 0.95f;
    samp.rng_seed    = 42ULL;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if ((a == "--prompt" || a == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (a == "--prompt-file" && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (a == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (a == "--no-think") {
            enable_thinking = false;
        } else if (a == "--think") {
            enable_thinking = true;
        } else if ((a == "--effort" || a == "--reasoning-effort") && i + 1 < argc) {
            effort = argv[++i];
        } else if (a == "--raw" || a == "--raw-prompt") {
            raw_prompt = true;
        } else if (a == "--max-tokens" && i + 1 < argc) {
            max_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--max-seq-len" && i + 1 < argc) {
            unsigned long v = std::strtoul(argv[++i], nullptr, 10);
            if (v < 128UL || v > STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP) {
                std::fprintf(stderr, "[run-flash] Invalid --max-seq-len (expected 128..%u)\n",
                             STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP);
                return 1;
            }
            max_seq_len = static_cast<uint32_t>(v);
        } else if (a == "--draft-k" && i + 1 < argc) {
            draft_k = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--int8-prefill") {
            int8_prefill     = true;
            int8_prefill_set = true;
        } else if (a == "--fp16-prefill" || a == "--no-int8-prefill") {
            int8_prefill     = false;
            int8_prefill_set = true;
        } else if (a == "--adaptive-k") {
            adaptive_k     = true;
            adaptive_k_set = true;
        } else if (a == "--no-adaptive-k") {
            adaptive_k     = false;
            adaptive_k_set = true;
        } else if (a == "--frspec" || a == "--fr-spec") {
            frspec     = true;
            frspec_set = true;
        } else if (a == "--no-frspec" || a == "--no-fr-spec") {
            frspec     = false;
            frspec_set = true;
        } else if (a == "--tree" || a == "--spec-tree") {
            spec_tree = true;
        } else if (a == "--no-tree" || a == "--no-spec-tree") {
            spec_tree = false;
        } else if (a == "--mtp-sync") {
            mtp_kv_sync = true;
        } else if (a == "--no-mtp-sync") {
            mtp_kv_sync = false;
        } else if (a == "--mtp-shortlist" && i + 1 < argc) {
            mtp_shortlist     = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            mtp_shortlist_set = true;
        } else if (a == "--ngram" || a == "--ngram-draft") {
            ngram_draft = true;
            ngram_set   = true;
        } else if (a == "--no-ngram" || a == "--no-ngram-draft") {
            ngram_draft = false;
            ngram_set   = true;
        } else if (a == "--greedy") {
            samp.greedy      = true;
            samp.temperature = 0.0f;
        } else if ((a == "--temp" || a == "--temperature") && i + 1 < argc) {
            samp.temperature = std::strtof(argv[++i], nullptr);
            samp.greedy      = (samp.temperature <= 1e-6f);
            temp_explicit    = true;
        } else if (a == "--top-k" && i + 1 < argc) {
            samp.top_k = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--top-p" && i + 1 < argc) {
            samp.top_p = std::strtof(argv[++i], nullptr);
        } else if (a == "--presence-penalty" && i + 1 < argc) {
            samp.presence_penalty = std::clamp(std::strtof(argv[++i], nullptr), -2.0f, 2.0f);
        } else if (a == "--frequency-penalty" && i + 1 < argc) {
            samp.frequency_penalty = std::clamp(std::strtof(argv[++i], nullptr), -2.0f, 2.0f);
        } else if ((a == "--repetition-penalty" || a == "--rep-penalty") && i + 1 < argc) {
            samp.repetition_penalty = std::clamp(std::strtof(argv[++i], nullptr), 0.5f, 2.0f);
        } else if (a == "--penalty-window" && i + 1 < argc) {
            const unsigned long w = std::strtoul(argv[++i], nullptr, 10);
            samp.penalty_window = (w == 0UL) ? 0U : static_cast<uint32_t>(std::clamp<unsigned long>(w, 8UL, 65536UL));
        } else if (a == "--seed" && i + 1 < argc) {
            samp.rng_seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (a == "--ple-cache-mib" && i + 1 < argc) {
            ple_cache_mib = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (a == "--no-eos-stop") {
            stop_on_eos = false;
        } else if (a == "--verify-k0") {
            verify_k0 = true;
        } else if (a == "--help" || a == "-h") {
            print_flash_run_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[run-flash] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    if (verify_k0 && !temp_explicit) {
        samp.greedy      = true;
        samp.temperature = 0.0f;
    }

    if (!prompt_file.empty()) {
        if (!read_text_file(prompt_file, &prompt)) {
            std::fprintf(stderr, "[run-flash] Failed to read `--prompt-file %s`\n", prompt_file.c_str());
            return 1;
        }
    }

    const ProcessMemStats mem_init0 = query_process_mem_stats();
    const auto t_load0 = SteadyClock::now();
    StrataQwenFlashBundle bdl{};
    if (strata_qwen_flash_load_bundle(bundle_dir.c_str(), ple_cache_mib * 1024ULL * 1024ULL, &bdl) != 0) {
        std::fprintf(stderr, "[run-flash] Failed to load bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    /* `--raw` is the caller's own template, so control tokens in it are honoured. In chat mode the user's text is
     * plain text and only the template may spell control tokens (see `encode_chat_prompt_plain_user`). */
    std::vector<uint32_t> prompt_ids;
    bool tok_ok = false;
    if (raw_prompt) {
        tok_ok = encode_string_vec(bdl.tokenizer, prompt, /*parse_special=*/true, &prompt_ids);
    } else {
        const char* sys_ptr = system_prompt.empty() ? nullptr : system_prompt.c_str();
        const char* eff_ptr = effort.empty() ? nullptr : effort.c_str();
        auto render = [&](const char* user_text, std::string* out) -> bool {
            const size_t cap = std::strlen(user_text) + system_prompt.size() + 4096U;
            std::vector<char> cbuf(cap, '\0');
            size_t flen = 0U;
            if (strata_qwen_flash_format_chat_prompt(
                    user_text, sys_ptr, enable_thinking, eff_ptr, cbuf.data(), cap, &flen) != 0) {
                return false;
            }
            out->assign(cbuf.data(), flen);
            return true;
        };
        std::string rendered_a, rendered_b;
        if (!render("A", &rendered_a) || !render("B", &rendered_b)) {
            std::fprintf(stderr, "[run-flash] Failed to format ChatML prompt\n");
            strata_qwen_flash_free_bundle(&bdl);
            return 1;
        }
        tok_ok = encode_chat_prompt_plain_user(bdl.tokenizer, rendered_a, rendered_b, prompt, &prompt_ids);
    }
    if (!tok_ok || prompt_ids.empty()) {
        std::fprintf(stderr, "[run-flash] Failed to tokenize prompt\n");
        strata_qwen_flash_free_bundle(&bdl);
        return 1;
    }

    const bool unlimited_tokens = (max_tokens == 0U);
    const uint32_t default_unlim_seq = 65536U;
    const uint32_t needed_seq = unlimited_tokens
        ? std::max<uint32_t>(default_unlim_seq, static_cast<uint32_t>(sqf_align_up_u64(static_cast<uint64_t>(prompt_ids.size()) + 4096ULL, 256ULL)))
        : static_cast<uint32_t>(
              sqf_align_up_u64(static_cast<uint64_t>(prompt_ids.size()) + static_cast<uint64_t>(max_tokens) + 64ULL, 256ULL));
    const uint32_t alloc_seq = std::min<uint32_t>(
        STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP,
        std::max<uint32_t>(max_seq_len, std::max<uint32_t>(2048U, needed_seq)));
    if (unlimited_tokens) {
        max_tokens = (alloc_seq > prompt_ids.size() + 64U)
            ? (alloc_seq - static_cast<uint32_t>(prompt_ids.size()) - 64U)
            : 256U;
    }

    StrataQwenFlashGpuEngine* eng = strata_qwen_flash_gpu_create(&bdl, alloc_seq);
    if (!eng) {
        std::fprintf(stderr, "[run-flash] Failed to initialize Metal 4 Qwen3.8-Flash-Next GPU engine\n");
        strata_qwen_flash_free_bundle(&bdl);
        return 1;
    }
    strata_qwen_flash_gpu_set_spec_tree(eng, spec_tree);
    strata_qwen_flash_gpu_set_mtp_kv_sync(eng, mtp_kv_sync);
    if (mtp_shortlist_set) strata_qwen_flash_gpu_set_mtp_shortlist(eng, mtp_shortlist);
    if (ngram_set)         strata_qwen_flash_gpu_set_ngram_draft(eng, ngram_draft);
    if (int8_prefill_set)  strata_qwen_flash_gpu_set_int8_prefill(eng, int8_prefill);
    if (adaptive_k_set)    strata_qwen_flash_gpu_set_adaptive_k(eng, adaptive_k);
    if (frspec_set)        strata_qwen_flash_gpu_set_frspec(eng, frspec);

    const uint32_t eff_k = bdl.cfg.has_mtp
        ? std::min<uint32_t>(draft_k, STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K)
        : 0U;

    StrataQwenFlashGpuInfo info{};
    strata_qwen_flash_gpu_get_info(eng, &info);
    StrataQwenFlashCheckpoint* run_ckpt = nullptr;
    if (verify_k0 && eff_k > 0U && samp.greedy) {
        run_ckpt = strata_qwen_flash_checkpoint_create(eng);
        if (run_ckpt) {
            (void)strata_qwen_flash_checkpoint_save(eng, run_ckpt);
            (void)strata_qwen_flash_checkpoint_restore(eng, run_ckpt);
        }
    }
    run_warmup_pass(&bdl, eng);
    const auto t_load1 = SteadyClock::now();
    const ProcessMemStats mem_init1 = query_process_mem_stats();
    const long init_majflt = std::max<long>(0L, mem_init1.majflt - mem_init0.majflt);

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    char frspec_lbl[32];
    if (info.frspec_enabled) {
        if (info.frspec_shortlist_size > 0U) {
            std::snprintf(frspec_lbl, sizeof(frspec_lbl), "ON (%u)", info.frspec_shortlist_size);
        } else {
            std::snprintf(frspec_lbl, sizeof(frspec_lbl), "ON");
        }
    } else {
        std::snprintf(frspec_lbl, sizeof(frspec_lbl), "OFF");
    }

    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 Live Qwen/Qwen3.8-Flash-Next (180B MoE + 51.2B NVMe PLE + Native MTP)\n");
    std::printf("  Device : %s | ZeroCopy=%d | Init %.0f ms | Metal Cache: %s (%.1f ms)\n",
                info.device_name, info.zero_copy_mmap ? 1 : 0, elapsed_ms(t_load0, t_load1),
                info.metal_cache_hit ? "HIT" : "MISS", info.pso_load_ms);
    std::printf("  Memory : UMA %.2f + %.2f = %.2f GiB | NVMe PLE %.2f GiB (0 GPU B, %.0f MiB mlock LFU)\n",
                static_cast<double>(info.uma_part0_mapped_bytes) / kGiB,
                static_cast<double>(info.uma_part1_mapped_bytes) / kGiB,
                static_cast<double>(info.total_uma_mapped_bytes) / kGiB,
                static_cast<double>(info.ple_nvme_file_bytes) / kGiB,
                static_cast<double>(info.ple_lfu_cache_bytes) / kMiB);
    std::printf("  Config : %s | %s (K=%u%s%s) | Prefill: %s | FR-Spec: %s | Adaptive-K: %s | %s | Prompt=%zu, MaxNew=%u\n",
                enable_thinking ? "Thinking (<think>)" : "Direct (--no-think)",
                (eff_k > 0U) ? "Native MTP Speculative" : "Standard Autoregressive",
                eff_k,
                (eff_k == 3U && spec_tree) ? ", W2xD3 Tree" : "",
                (eff_k > 0U && mtp_kv_sync) ? ", KV-Sync" : "",
                info.int8_prefill_enabled ? "INT8" : "FP16",
                frspec_lbl,
                info.adaptive_k_enabled ? "ON" : "OFF",
                samp.greedy ? "Greedy (T=0)" : "Stochastic Leviathan",
                prompt_ids.size(),
                max_tokens);
    std::printf("====================================================================================================\n");
    if (enable_thinking && !raw_prompt) {
        std::fputs("<think>\n", stdout);
        std::fflush(stdout);
    }

    PrefillReuseSlot run_reuse{};
    run_reuse.ckpt  = run_ckpt;
    run_reuse.valid = false;

    FlashRunResult res{};
    const int rc = execute_flash_sequence(
        &bdl, eng, info, prompt_ids, max_tokens, /*count_decode_only=*/false,
        eff_k, samp, stop_on_eos, /*stream_stdout=*/true, &res,
        (run_ckpt != nullptr) ? &run_reuse : nullptr);
    std::fputc('\n', stdout);

    if (rc != 0) {
        std::fprintf(stderr, "[run-flash] Generation failed (rc=%d)\n", rc);
        if (run_ckpt) strata_qwen_flash_checkpoint_destroy(run_ckpt);
        strata_qwen_flash_gpu_destroy(eng);
        strata_qwen_flash_free_bundle(&bdl);
        return 1;
    }

    const double accept_pct = (res.mtp_drafted > 0U)
        ? (100.0 * static_cast<double>(res.mtp_accepted) / static_cast<double>(res.mtp_drafted))
        : 0.0;
    const double avg_tok_per_step = (res.decode_steps > 0U)
        ? (static_cast<double>(res.decode_emitted) / static_cast<double>(res.decode_steps))
        : 0.0;
    const uint64_t ple_lookups = res.total_ple_ios + res.total_ple_hits;
    const double lfu_hit_pct = (ple_lookups > 0ULL)
        ? (100.0 * static_cast<double>(res.total_ple_hits) / static_cast<double>(ple_lookups))
        : 0.0;

    std::printf("----------------------------------------------------------------------------------------------------\n");
    std::printf("  Qwen3.8-Flash-Next Silicon Telemetry (%s)\n", info.device_name);
    std::printf("----------------------------------------------------------------------------------------------------\n");
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
    std::printf("  NVMe PLE (F_NOCACHE) : %6llu IOs (%.1f KiB) | %6llu LFU hits (%.1f%%) | Decode IOs=%llu\n",
                static_cast<unsigned long long>(res.total_ple_ios),
                static_cast<double>(res.total_ple_bytes) / 1024.0,
                static_cast<unsigned long long>(res.total_ple_hits),
                lfu_hit_pct,
                static_cast<unsigned long long>(res.decode_ple_ios));
    std::printf("  Active State (pos=%u): %7.2f MiB total (KV+Idx %.2f MiB + DeltaNet %.2f MiB + PLE %.2f MiB)\n",
                res.final_seq_pos, res.active_state_mib, res.kv_active_mib, res.deltanet_mib, res.ple_state_mib);
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
        FlashRunResult ref{};
        if (execute_flash_sequence(
                &bdl, eng, info, prompt_ids, max_tokens, /*count_decode_only=*/false,
                /*draft_k=*/0U, samp, stop_on_eos, /*stream_stdout=*/false, &ref,
                (run_ckpt != nullptr) ? &run_reuse : nullptr) != 0) {
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

    if (run_ckpt) strata_qwen_flash_checkpoint_destroy(run_ckpt);
    strata_qwen_flash_gpu_destroy(eng);
    strata_qwen_flash_free_bundle(&bdl);
    return verify_ok ? 0 : 1;
}

int strata_qwen_flash_cmd_bench(int argc, char** argv) {
    std::string bundle_dir   = STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR;
    std::string contexts_csv = "512,4096,16384,32768,65536";
    std::string draft_ks_csv;
    std::string json_out_path;
    std::string bench_effort   = "low";
    bool bench_think           = false;
    bool contexts_explicit     = false;
    bool gen_tokens_explicit   = false;
    bool bench_spec_tree       = false;
    bool bench_mtp_sync        = true;
    bool bench_sl_set          = false;
    uint32_t bench_sl_v        = 65536U;
    bool bench_ng_set          = false;
    bool bench_ng_on           = false;
    bool bench_int8_prefill    = false;
    bool bench_int8_set        = false;
    bool bench_adaptive_k      = true;
    bool bench_adaptive_k_set  = false;
    bool bench_frspec          = true;
    bool bench_frspec_set      = false;
    uint32_t gen_tokens        = 128U;
    uint32_t default_mtp_k     = 2U;
    size_t ple_cache_mib       = 0U;
    double custom_needle_depth = -1.0;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--contexts" && i + 1 < argc) {
            contexts_csv      = argv[++i];
            contexts_explicit = true;
        } else if (a == "--gen-tokens" && i + 1 < argc) {
            gen_tokens          = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            gen_tokens_explicit = true;
        } else if (a == "--think") {
            bench_think = true;
        } else if (a == "--no-think") {
            bench_think = false;
        } else if ((a == "--effort" || a == "--reasoning-effort") && i + 1 < argc) {
            bench_effort = argv[++i];
            bench_think  = true;
            if (bench_effort != "low" && bench_effort != "medium" && bench_effort != "high" &&
                bench_effort != "xhigh" && bench_effort != "max") {
                std::fprintf(stderr, "[bench-flash] Invalid --effort `%s` (expected low, medium, high, xhigh, or max)\n",
                             bench_effort.c_str());
                return 1;
            }
        } else if (a == "--xhigh-scaling" || a == "--preset-xhigh") {
            bench_think  = true;
            bench_effort = "xhigh";
            if (!contexts_explicit) contexts_csv = "2048,8192,32768";
            if (!gen_tokens_explicit) gen_tokens = 512U;
        } else if ((a == "--draft-k" || a == "--mtp-k") && i + 1 < argc) {
            default_mtp_k = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--draft-ks" && i + 1 < argc) {
            draft_ks_csv = argv[++i];
        } else if (a == "--int8-prefill") {
            bench_int8_prefill = true;
            bench_int8_set     = true;
        } else if (a == "--fp16-prefill" || a == "--no-int8-prefill") {
            bench_int8_prefill = false;
            bench_int8_set     = true;
        } else if (a == "--adaptive-k") {
            bench_adaptive_k     = true;
            bench_adaptive_k_set = true;
        } else if (a == "--no-adaptive-k") {
            bench_adaptive_k     = false;
            bench_adaptive_k_set = true;
        } else if (a == "--frspec" || a == "--fr-spec") {
            bench_frspec     = true;
            bench_frspec_set = true;
        } else if (a == "--no-frspec" || a == "--no-fr-spec") {
            bench_frspec     = false;
            bench_frspec_set = true;
        } else if (a == "--tree" || a == "--spec-tree") {
            bench_spec_tree = true;
        } else if (a == "--no-tree" || a == "--no-spec-tree") {
            bench_spec_tree = false;
        } else if (a == "--mtp-sync") {
            bench_mtp_sync = true;
        } else if (a == "--no-mtp-sync") {
            bench_mtp_sync = false;
        } else if (a == "--mtp-shortlist" && i + 1 < argc) {
            bench_sl_v   = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            bench_sl_set = true;
        } else if (a == "--ngram" || a == "--ngram-draft") {
            bench_ng_on  = true;
            bench_ng_set = true;
        } else if (a == "--no-ngram" || a == "--no-ngram-draft") {
            bench_ng_on  = false;
            bench_ng_set = true;
        } else if (a == "--needle-depth" && i + 1 < argc) {
            custom_needle_depth = std::strtod(argv[++i], nullptr);
        } else if (a == "--ple-cache-mib" && i + 1 < argc) {
            ple_cache_mib = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (a == "--json" && i + 1 < argc) {
            json_out_path = argv[++i];
        } else if (a == "--help" || a == "-h") {
            print_flash_bench_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[bench-flash] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    const std::vector<uint32_t> contexts = parse_u32_csv(contexts_csv);
    if (contexts.empty()) {
        std::fprintf(stderr, "[bench-flash] `--contexts` parsed empty\n");
        return 1;
    }

    std::vector<uint32_t> draft_ks;
    if (!draft_ks_csv.empty()) {
        draft_ks = parse_u32_csv(draft_ks_csv);
    } else {
        draft_ks.push_back(0U);
        if (default_mtp_k > 0U) {
            draft_ks.push_back(std::min<uint32_t>(default_mtp_k, STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K));
        }
    }

    uint32_t max_ctx = 0U;
    for (uint32_t c : contexts) max_ctx = std::max(max_ctx, c);
    const uint64_t max_seq_u64 = sqf_align_up_u64(
        static_cast<uint64_t>(max_ctx) + static_cast<uint64_t>(gen_tokens) + 64ULL, 256ULL);
    if (max_seq_u64 > STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP) {
        std::fprintf(stderr, "[bench-flash] Requested context+gen (%llu) exceeds max_seq_len cap (%u)\n",
                     static_cast<unsigned long long>(max_seq_u64), STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP);
        return 1;
    }
    const uint32_t max_seq_len = static_cast<uint32_t>(max_seq_u64);

    const ProcessMemStats mem_boot0 = query_process_mem_stats();
    const auto t_boot0 = SteadyClock::now();

    StrataQwenFlashBundle bdl{};
    if (strata_qwen_flash_load_bundle(bundle_dir.c_str(), ple_cache_mib * 1024ULL * 1024ULL, &bdl) != 0) {
        std::fprintf(stderr, "[bench-flash] Failed to load bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    StrataQwenFlashGpuEngine* eng = strata_qwen_flash_gpu_create(&bdl, max_seq_len);
    if (!eng) {
        std::fprintf(stderr, "[bench-flash] Failed to create GPU engine (max_seq_len=%u)\n", max_seq_len);
        strata_qwen_flash_free_bundle(&bdl);
        return 1;
    }
    strata_qwen_flash_gpu_set_spec_tree(eng, bench_spec_tree);
    strata_qwen_flash_gpu_set_mtp_kv_sync(eng, bench_mtp_sync);
    if (bench_sl_set)         strata_qwen_flash_gpu_set_mtp_shortlist(eng, bench_sl_v);
    if (bench_ng_set)         strata_qwen_flash_gpu_set_ngram_draft(eng, bench_ng_on);
    if (bench_int8_set)       strata_qwen_flash_gpu_set_int8_prefill(eng, bench_int8_prefill);
    if (bench_adaptive_k_set) strata_qwen_flash_gpu_set_adaptive_k(eng, bench_adaptive_k);
    if (bench_frspec_set)     strata_qwen_flash_gpu_set_frspec(eng, bench_frspec);

    StrataQwenFlashGpuInfo info{};
    strata_qwen_flash_gpu_get_info(eng, &info);
    StrataQwenFlashCheckpoint* bench_ckpt = nullptr;
    if (draft_ks.size() > 1U) {
        bench_ckpt = strata_qwen_flash_checkpoint_create(eng);
        if (bench_ckpt) {
            (void)strata_qwen_flash_checkpoint_save(eng, bench_ckpt);
            (void)strata_qwen_flash_checkpoint_restore(eng, bench_ckpt);
        }
    }
    run_warmup_pass(&bdl, eng);

    const auto t_boot1 = SteadyClock::now();
    const ProcessMemStats mem_boot1 = query_process_mem_stats();
    const long init_majflt = std::max<long>(0L, mem_boot1.majflt - mem_boot0.majflt);

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    char frspec_lbl[32];
    if (info.frspec_enabled) {
        if (info.frspec_shortlist_size > 0U) {
            std::snprintf(frspec_lbl, sizeof(frspec_lbl), "ON (%u)", info.frspec_shortlist_size);
        } else {
            std::snprintf(frspec_lbl, sizeof(frspec_lbl), "ON");
        }
    } else {
        std::snprintf(frspec_lbl, sizeof(frspec_lbl), "OFF");
    }

    std::printf("========================================================================================================================================\n");
    std::printf("  STRATA-M5 v2.0 — Qwen/Qwen3.8-Flash-Next (180B / 335.28 GiB BF16) Ultra-Long-Context (up to 256K) & Native MTP Suite\n");
    std::printf("========================================================================================================================================\n");
    std::printf("  Silicon & Metal 4 : %s | ZeroCopy=%d | MaxSeq=%u | Boot+Warmup %.1f ms (Init MajFlt=%ld) | Metal Cache: %s (%.1f ms)\n",
                info.device_name, info.zero_copy_mmap ? 1 : 0, info.max_seq_len,
                elapsed_ms(t_boot0, t_boot1), init_majflt,
                info.metal_cache_hit ? "HIT" : "MISS", info.pso_load_ms);
    std::printf("  Disaggregated Mem : UMA Part0 %.3f GiB + Part1 %.3f GiB = %.3f GiB wired | NVMe PLE %.3f GiB (0 GPU B, %.1f MiB LFU)\n",
                static_cast<double>(info.uma_part0_mapped_bytes) / kGiB,
                static_cast<double>(info.uma_part1_mapped_bytes) / kGiB,
                static_cast<double>(info.total_uma_mapped_bytes) / kGiB,
                static_cast<double>(info.ple_nvme_file_bytes) / kGiB,
                static_cast<double>(info.ple_lfu_cache_bytes) / kMiB);
    std::printf("  State & Bandwidth : Active %.3f GB/tok (K=0) + %.3f GB/draft (MTP) | DeltaNet %.1f MiB + KV/Idx %.1f KiB/tok (Alloc %.2f GiB)\n",
                static_cast<double>(info.active_bytes_per_token) / 1e9,
                static_cast<double>(info.mtp_active_bytes_per_draft) / 1e9,
                static_cast<double>(info.deltanet_state_bytes) / kMiB,
                static_cast<double>(info.kv_cache_bytes_per_token) / 1024.0,
                static_cast<double>(info.kv_cache_bytes_allocated) / kGiB);
    std::printf("  Reasoning & Spec  : %s%s | Prefill: %s | FR-Spec: %s | Adaptive-K: %s | Cap: %u decode toks (+1)\n",
                bench_think ? "Thinking (<think>, effort=" : "Direct (--no-think)",
                bench_think ? (bench_effort + ")").c_str() : "",
                info.int8_prefill_enabled ? "INT8" : "FP16",
                frspec_lbl,
                info.adaptive_k_enabled ? "ON" : "OFF",
                gen_tokens);
    std::printf("--------------------------------------------------------------------------------------------------------------------------------------------------\n");
    std::printf(" %6s | %5s | %9s | %9s | %9s | %9s | %7s | %8s | %15s | %9s | %19s | %6s | %s\n",
                "Ctx", "Mode", "Prefill", "TTFT(ms)", "Dec(ms)", "Decode", "Accept", "DRAM BW",
                "PLE IO / LFU", "State MiB", "GPU/Host/TotUMA GiB", "MajFlt", "Needle");
    std::printf("--------------------------------------------------------------------------------------------------------------------------------------------------\n");

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
        FlashRunResult           r;
    };

    std::vector<BenchRow> rows;
    HaystackTokenizerCache hay_cache{};
    bool all_ok = true;

    StrataQwenFlashSamplingParams greedy{};
    greedy.greedy      = true;
    greedy.temperature = 0.0f;
    greedy.rng_seed    = 42ULL;

    for (uint32_t ctx : contexts) {
        PasskeyPromptSpec pspec{};
        if (!build_passkey_prompt_exact(
                bdl.tokenizer, &hay_cache, ctx, custom_needle_depth, bench_think, bench_effort, &pspec)) {
            std::fprintf(stderr, "[bench-flash] Failed to construct exact %u-token needle prompt\n", ctx);
            if (bench_ckpt) strata_qwen_flash_checkpoint_destroy(bench_ckpt);
            strata_qwen_flash_gpu_destroy(eng);
            strata_qwen_flash_free_bundle(&bdl);
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

            const int rc = execute_flash_sequence(
                &bdl, eng, info, pspec.prompt_ids, gen_tokens, /*count_decode_only=*/true,
                k, greedy, /*stop_on_eos=*/false, /*stream_stdout=*/false, &row.r,
                (bench_ckpt != nullptr) ? &reuse_slot : nullptr);
            if (rc != 0) {
                std::fprintf(stderr, "[bench-flash] Engine failed at ctx=%u K=%u (rc=%d)\n", ctx, k, rc);
                if (bench_ckpt) strata_qwen_flash_checkpoint_destroy(bench_ckpt);
                strata_qwen_flash_gpu_destroy(eng);
                strata_qwen_flash_free_bundle(&bdl);
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

            size_t nl = decoded.find('\n');
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

            if (!row.needle_pass || row.r.steady_majflt != 0L) {
                if (!row.needle_pass) all_ok = false;
            }

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

            char ple_str[32];
            std::snprintf(ple_str, sizeof(ple_str), "%6llu / %6llu",
                          static_cast<unsigned long long>(row.r.total_ple_ios),
                          static_cast<unsigned long long>(row.r.total_ple_hits));

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

            std::printf(" %6u | %5s | %7.1f/s | %9.1f | %9.1f | %7.2f/s | %7s | %6.1fGB | %15s | %9.1f | %19s | %6ld | %s\n",
                        row.ctx_len,
                        mode_str,
                        row.r.prefill_tok_s,
                        row.r.ttft_ms,
                        row.r.decode_ms,
                        row.r.decode_tok_s,
                        acc_str,
                        row.r.phys_dram_gbps,
                        ple_str,
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
                << "  \"model\": \"Qwen/Qwen3.8-Flash-Next\",\n"
                << "  \"device\": \"" << info.device_name << "\",\n"
                << "  \"uma_mapped_gib\": " << (static_cast<double>(info.total_uma_mapped_bytes) / kGiB) << ",\n"
                << "  \"ple_nvme_gib\": " << (static_cast<double>(info.ple_nvme_file_bytes) / kGiB) << ",\n"
                << "  \"int8_prefill\": " << (info.int8_prefill_enabled ? "true" : "false") << ",\n"
                << "  \"adaptive_k\": " << (info.adaptive_k_enabled ? "true" : "false") << ",\n"
                << "  \"frspec\": " << (info.frspec_enabled ? "true" : "false") << ",\n"
                << "  \"frspec_shortlist_size\": " << info.frspec_shortlist_size << ",\n"
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
                    << ", \"ple_nvme_ios\": " << r.r.total_ple_ios
                    << ", \"ple_lfu_hits\": " << r.r.total_ple_hits
                    << ", \"decode_ple_ios\": " << r.r.decode_ple_ios
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

    if (bench_ckpt) strata_qwen_flash_checkpoint_destroy(bench_ckpt);
    strata_qwen_flash_gpu_destroy(eng);
    strata_qwen_flash_free_bundle(&bdl);
    return all_ok ? 0 : 1;
}

} /* extern "C" */

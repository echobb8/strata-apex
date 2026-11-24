/*
 * STRATA-M5 v2.0 — Step 3: Regime II Frontier MoE (`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`) CLI & Benchmark Suite
 * File: src/cli/strata_glm53_bench.mm
 *
 * Target Model: `Glm5NextForConditionalGeneration` (`GLM-5.3-Flash`: `320B` total / `18B` active,
 *               `43` shards = `169.22 GiB` > `128 GiB` physical UMA on `Mac17,10`).
 *
 * Implements:
 *   - `convert-glm53`: Pipelined 43-shard HF downloader + ARM NEON streaming `.strata` disaggregator
 *   - `inspect-glm53`: Zero-copy `spine.strata` (`5.36 GiB`) + `256 KiB`-aligned `experts_nvme.strata` (`163.27 GiB`) inspector
 *   - `run-glm53`:     Live streaming Regime II inference (`lossless` & `adaptive` Mode B, `K=0` & Native Layer-45 `MTP`, `--verify-k0`)
 *   - `bench-glm53`:   `512 .. 32,768` context sweep in BOTH `lossless` & `adaptive` modes, measuring prefill `tok/s`,
 *                      decode `tok/s` (`K=0` & `MTP`), `UMA Hot Hit %`, `NVMe Cold DMA GB/s`, `16c ANE` (`ran_on_ane`),
 *                      `Total UMA < 80 GiB`, `Swap +0.00 GiB`, `Steady MajFlt == 0`, and Passkey Needle recall
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_glm53.h"

#include <mach/mach.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

/* Use Darwin `CLOCK_UPTIME_RAW` (`mach_absolute_time`), which pauses during macOS sleep! */
struct SteadyClock {
    using time_point = uint64_t;
    static time_point now() {
        return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    }
};

static double elapsed_ms(SteadyClock::time_point t0, SteadyClock::time_point t1) {
    return (t1 >= t0) ? (static_cast<double>(t1 - t0) * 1e-6) : 0.0;
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

static uint64_t sg53_cli_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

struct ProcessMemStats {
    double rss_gib            = 0.0; /* Host CPU resident_size */
    double gpu_wired_gib      = 0.0; /* Metal [device currentAllocatedSize] */
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

static std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    }
    return s;
}

static bool parse_exec_mode_str(const std::string& s, StrataGlm53ExecMode* out_mode) {
    const std::string m = to_lower_ascii(s);
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

static bool parse_exec_modes_csv(const std::string& csv, std::vector<StrataGlm53ExecMode>* out_modes) {
    out_modes->clear();
    if (to_lower_ascii(csv) == "both" || to_lower_ascii(csv) == "all") {
        out_modes->push_back(STRATA_GLM53_MODE_LOSSLESS);
        out_modes->push_back(STRATA_GLM53_MODE_ADAPTIVE);
        return true;
    }
    size_t pos = 0U;
    while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        std::string item = (comma == std::string::npos) ? csv.substr(pos) : csv.substr(pos, comma - pos);
        if (!item.empty()) {
            StrataGlm53ExecMode em = STRATA_GLM53_MODE_LOSSLESS;
            if (!parse_exec_mode_str(item, &em)) return false;
            out_modes->push_back(em);
        }
        if (comma == std::string::npos) break;
        pos = comma + 1U;
    }
    return !out_modes->empty();
}

static bool encode_string_vec(
    const StrataGlm53Tokenizer* tok,
    const std::string& text,
    bool parse_special,
    std::vector<uint32_t>* out_ids
) {
    const uint32_t cap = static_cast<uint32_t>(std::max<size_t>(text.size() + 128U, 256U));
    out_ids->assign(cap, 0U);
    uint32_t n = 0U;
    if (strata_glm53_encode(tok, text.c_str(), text.size(), parse_special, out_ids->data(), cap, &n) != 0) {
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
    const StrataGlm53Tokenizer* tok,
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
    const StrataGlm53Tokenizer* tok,
    const std::vector<uint32_t>& ids,
    bool skip_special
) {
    if (ids.empty()) return {};
    const size_t cap = std::max<size_t>(ids.size() * 32U + 256U, 1024U);
    std::vector<char> buf(cap, '\0');
    size_t len = 0U;
    if (strata_glm53_decode(
            tok, ids.data(), static_cast<uint32_t>(ids.size()), skip_special, buf.data(), cap, &len) != 0) {
        return {};
    }
    return std::string(buf.data(), len);
}

static void stream_emit_token(
    const StrataGlm53Tokenizer* tok,
    StrataGlm53StreamDecoder* dec,
    uint32_t token_id
) {
    if (token_id == STRATA_GLM53_TOKEN_THINK_START) {
        std::fputs("<think>", stdout);
        std::fflush(stdout);
        return;
    }
    if (token_id == STRATA_GLM53_TOKEN_THINK_END) {
        strata_glm53_stream_decoder_init(dec);
        std::fputs("</think>", stdout);
        std::fflush(stdout);
        return;
    }
    char piece[256];
    size_t piece_len = 0U;
    if (strata_glm53_stream_decode_step(tok, dec, token_id, /*skip_special=*/true, piece, sizeof(piece), &piece_len) == 0 &&
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

static bool is_glm53_fast_bundle_dir(const std::string& dir) {
    if (dir.empty()) return false;
    const std::string sp = dir + "/" + STRATA_GLM53_SPINE_FILENAME;
    const std::string p0 = dir + "/" + STRATA_GLM53_EXPERTS_PART0_FILENAME;
    return (::access(sp.c_str(), R_OK) == 0) && (::access(p0.c_str(), R_OK) == 0);
}

static std::string resolve_glm53_default_bundle_dir() {
    const std::string fast_dir = STRATA_GLM53_DEFAULT_FAST_BUNDLE_DIR;
    if (is_glm53_fast_bundle_dir(fast_dir)) {
        return fast_dir;
    }
    return STRATA_GLM53_DEFAULT_BUNDLE_DIR;
}

static std::string glm53_effort_title_case(const std::string& effort) {
    const std::string e = to_lower_ascii(effort);
    if (e == "max" || e == "xhigh") return "Max";
    if (e == "high" || e == "medium") return "High";
    return "Low";
}

static void run_warmup_pass(
    StrataGlm53Bundle* bundle,
    StrataGlm53GpuEngine* eng
) {
    @autoreleasepool {
        const bool is_v2 = (bundle && bundle->is_ram_resident);
        const StrataGlm53ExecMode saved_mode = strata_glm53_gpu_get_exec_mode(eng);
        if (!is_v2) {
            strata_glm53_gpu_set_exec_mode(eng, STRATA_GLM53_MODE_ADAPTIVE);
        }
        strata_glm53_gpu_reset(eng);
        strata_glm53_gpu_penalties_reset(eng);
        strata_glm53_gpu_reset_counters(eng);

        StrataGlm53GpuInfo winfo{};
        strata_glm53_gpu_get_info(eng, &winfo);
        const bool can_warm_64 = (winfo.max_seq_len >= 80U);
        uint32_t warmup_ids[64] = {
            STRATA_GLM53_TOKEN_GMASK, STRATA_GLM53_TOKEN_SOP,
            STRATA_GLM53_TOKEN_USER, 1000U,
            STRATA_GLM53_TOKEN_ASSISTANT, STRATA_GLM53_TOKEN_THINK_START
        };
        for (uint32_t i = 6U; i < 64U; ++i) {
            warmup_ids[i] = 1000U + (i % 16U);
        }
        const uint32_t n_warm = can_warm_64 ? 64U : 6U;

        StrataGlm53SamplingParams greedy{};
        greedy.greedy             = true;
        greedy.temperature        = 0.0f;
        greedy.repetition_penalty = 1.0f;
        greedy.rng_seed           = 1ULL;

        uint32_t next_tok = 0U;
        (void)strata_glm53_gpu_prefill(eng, warmup_ids, n_warm, &greedy, &next_tok, nullptr);
        uint32_t next2 = 0U;
        (void)strata_glm53_gpu_decode_step(eng, next_tok, &greedy, &next2, nullptr);
        if (bundle && bundle->header && bundle->header->mtp.enabled) {
            uint32_t emitted[STRATA_GLM53_MAX_SPEC_DRAFT_K + 2U] = {};
            uint32_t n_emit = 0U;
            uint32_t n_acc  = 0U;
            const uint32_t warm_k = is_v2 ? STRATA_GLM53_MAX_SPEC_DRAFT_K : 2U;
            (void)strata_glm53_gpu_spec_decode_step(eng, next2, warm_k, &greedy, emitted, &n_emit, &n_acc);
            if (n_emit > 0U) next2 = emitted[n_emit - 1U];
        }

        // Also warm stochastic sampler & penalty PSOs + tokenizer BPE/stream tables before steady-state fault tracking!
        if (can_warm_64) {
            StrataGlm53SamplingParams samp{};
            samp.greedy             = false;
            samp.temperature        = 0.6f;
            samp.top_k              = 20U;
            samp.top_p              = 0.95f;
            samp.presence_penalty   = 0.6f;
            samp.frequency_penalty  = 0.08f;
            samp.repetition_penalty = 1.05f;
            samp.penalty_window     = 256U;
            samp.rng_seed           = 42ULL;
            uint32_t next3 = 0U;
            (void)strata_glm53_gpu_decode_step(eng, next2, &samp, &next3, nullptr);
            if (bundle && bundle->header && bundle->header->mtp.enabled) {
                uint32_t emitted[STRATA_GLM53_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t n_emit = 0U;
                uint32_t n_acc  = 0U;
                const uint32_t warm_k = is_v2 ? STRATA_GLM53_MAX_SPEC_DRAFT_K : 2U;
                (void)strata_glm53_gpu_spec_decode_step(eng, next3, warm_k, &samp, emitted, &n_emit, &n_acc);
            }
        }
        if (bundle && bundle->tokenizer) {
            std::vector<uint32_t> enc_tmp;
            (void)encode_string_vec(bundle->tokenizer, "Warmup 123", /*parse_special=*/false, &enc_tmp);
            const std::vector<uint32_t> dec_tmp = {next_tok, next2};
            (void)decode_tokens_str(bundle->tokenizer, dec_tmp, /*skip_special=*/true);
            StrataGlm53StreamDecoder sdec{};
            strata_glm53_stream_decoder_init(&sdec);
            char piece[256];
            size_t plen = 0U;
            const uint32_t v_cap = bundle->cfg.vocab_size;
            for (uint32_t tid = 0U; tid < v_cap; tid += 16U) {
                (void)strata_glm53_stream_decode_step(bundle->tokenizer, &sdec, tid, true, piece, sizeof(piece), &plen);
            }
        }
        // Fault & pin tiny host-executed segments (__TEXT/__DATA_CONST ~4 MiB + 56 KiB spine header) AFTER the
        // 94 GiB GPU residency set is wired so macOS VM reclamation never causes a steady-state host MajFlt!
        if (bundle && bundle->header) {
            const auto* hp = reinterpret_cast<const volatile uint8_t*>(bundle->header);
            volatile uint8_t acc = 0U;
            for (size_t off = 0U; off < sizeof(StrataGlm53SpineHeader); off += 4096U) acc ^= hp[off];
            (void)acc;
            (void)::mlock(bundle->header, sizeof(StrataGlm53SpineHeader));
        }
        if (const auto* mh = reinterpret_cast<const struct mach_header_64*>(_dyld_get_image_header(0U))) {
            const char* const segs[2] = {"__TEXT", "__DATA_CONST"};
            volatile uint8_t acc = 0U;
            for (const char* seg_name : segs) {
                unsigned long seg_sz = 0UL;
                const uint8_t* seg_ptr = getsegmentdata(mh, seg_name, &seg_sz);
                if (seg_ptr && seg_sz > 0UL) {
                    for (unsigned long off = 0UL; off < seg_sz; off += 4096UL) acc ^= seg_ptr[off];
                    (void)::mlock(seg_ptr, seg_sz);
                }
            }
            (void)acc;
        }

        if (!is_v2) {
            strata_glm53_gpu_reset_hot_cache(eng);
        }
        strata_glm53_gpu_reset(eng);
        strata_glm53_gpu_penalties_reset(eng);
        strata_glm53_gpu_reset_counters(eng);
        strata_glm53_gpu_set_exec_mode(eng, saved_mode);
    }
}

/* Diverse technical corpus paragraphs for realistic Needle-in-a-Haystack construction (exercises DSA pool indexer & 288E MoE). */
static const char* const kHaystackParagraphs[] = {
    "Modern unified memory architectures integrate central processor clusters, neural accelerators, and wide graphics cores onto a single coherent LPDDR5X memory fabric, eliminating PCIe bounce buffers during autoregressive inference.",
    "By mapping page-aligned immutable spine containers directly into GPU virtual address space through Mach virtual memory descriptors, zero-copy residency sets avoid duplicate host allocations across multi-hundred-billion-parameter models.",
    "Key-driven linear attention blocks maintain fixed-dimensional matrix recurrence states updated via bounded per-channel key decay and rank-one associative delta rules, providing constant memory footprint regardless of sequence length.",
    "Interleaving key-driven linear recurrence layers with dynamic sparse multi-head latent attention preserves exact long-range token retrieval while compressing key-value state down to a single five-hundred-twelve-element latent vector per token.",
    "Pool-compressed dynamic sparse attention indexers aggregate four-token windows through learned per-channel softmax gates and additive positional encodings before scoring candidate memory pools with rectified multi-head dot products.",
    "Hardware radix selection over signed floating-point pool scores identifies the top five hundred twelve memory pools in microseconds on graphics threadgroups without host synchronization barriers.",
    "Fine-grained mixture-of-experts layers route each hidden representation to eight specialized experts out of two hundred eighty-eight candidates using bias-corrected sigmoid scores alongside an un-gated shared expert.",
    "Four-stream manifold-constrained hyper-connections generalize residual pathways by projecting normalized parallel hidden lanes onto doubly-stochastic Sinkhorn-Knopp transport matrices prior to each attention and expert sublayer.",
    "Disaggregating one hundred sixty-three gigabytes of routed four-bit experts onto uncached NVMe stripes while wiring fifty-six gigabytes of hot experts in unified memory breaks the physical RAM ceiling with zero swap activity.",
    "Multi-horizon router lookahead probes executing on the physical sixteen-core neural engine predict upcoming cold expert activations four layers ahead, overlapping parallel eight-descriptor NVMe transfers with graphics compute.",
    "Native single-stream multi-token prediction layers fuse normalized token embeddings and final backbone states through a dedicated sparse latent attention and mixture-of-experts block to draft multiple tokens directly on GPU.",
    "Exact Leviathan rejection sampling and deterministic greedy prefix verification guarantee that speculative multi-token decoding follows the exact target distribution of the underlying frontier model.",
    "When a drafted token diverges from the main model verification pass, constant-time device-side state snapshots roll back recurrent matrices, causal convolution buffers, and partial indexer windows in under half a millisecond.",
    "Group-wise four-bit affine quantization packs eight unsigned nibbles per thirty-two-bit word alongside half-precision scale and bias pairs every sixty-four input channels to saturate memory bandwidth during single-token decode.",
    "Cooperative tensor matrix primitives in Metal four schedule SIMD-group tiles across neural hardware execution units during batched prefill, reaching sixty teraflops of half-precision throughput.",
    "Distributed systems consensus protocols rely on monotonic ballot numbers and quorum intersections so safety invariants hold even under asynchronous packet reordering, crash-recovery cycles, and network partitions.",
    "Optimizing compilers lower static single-assignment intermediate representations through dominance frontiers, loop-invariant code motion, polyhedral tile fusion, and register-pressure-aware instruction scheduling.",
    "Log-structured merge trees batch random mutations into sequential immutable sorted string tables on flash storage while Bloom filters prune unnecessary block reads during point lookups.",
    "Elliptic-curve key exchange over prime fields derives shared ephemeral secrets with compact public keys, feeding authenticated encryption modes that protect confidentiality and ciphertext integrity simultaneously.",
    "Numerical Krylov subspace solvers accelerate ill-conditioned sparse linear systems via incomplete Cholesky preconditioners and orthogonal Arnoldi recurrences with bounded memory vectors.",
    "Operating system schedulers balance cache affinity against fairness across heterogeneous performance and efficiency cores by tracking runnable thread load averages and energy-delay heuristics.",
    "Columnar analytical query engines vectorize predicate evaluation directly over dictionary-encoded bit-packed pages, saturating SIMD lanes while minimizing branch mispredictions per tuple.",
    "Transactional write-ahead logs guarantee physiological crash recovery by replaying redo records from fuzzy checkpoints before rolling back uncommitted loser transactions in reverse log-sequence order.",
    "Formal model checkers exhaustively explore reachable state transitions or encode bounded temporal logic properties into conjunctive normal form clauses solved via conflict-driven clause learning."
};

struct HaystackTokenizerCache {
    std::vector<std::vector<uint32_t>> para_tokens;
    bool                               initialized = false;
};

static bool init_haystack_cache(const StrataGlm53Tokenizer* tok, HaystackTokenizerCache* cache) {
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
        *rng_state = sg53_cli_splitmix64(*rng_state);
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
    uint64_t h = sg53_cli_splitmix64(0x474C4D35334D35ULL ^
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

static bool build_glm53_passkey_prompt_exact(
    const StrataGlm53Tokenizer* tok,
    HaystackTokenizerCache* cache,
    uint32_t target_ctx,
    double custom_needle_depth,
    bool enable_thinking,
    const std::string& effort,
    PasskeyPromptSpec* out_spec
) {
    if (!init_haystack_cache(tok, cache) || !out_spec) return false;
    *out_spec = PasskeyPromptSpec{};

    const std::string passkey = deterministic_passkey_for_ctx_slot(target_ctx, 0U);
    const std::string eff_hdr = glm53_effort_title_case(effort);
    const std::string asst_tail = enable_thinking
        ? "<|assistant|><think>"
        : "<|assistant|><think></think>";
    std::string prefix_str;
    std::string needle_str;
    std::string suffix_str;

    if (target_ctx >= 256U) {
        prefix_str =
            "[gMASK]<sop><|system|>Reasoning Effort: " + eff_hdr + "<|user|>"
            "Read the following technical archive carefully. A confidential 7-digit STRATA vault passkey is recorded inside one entry.\n\n";
        needle_str =
            "\n[CONFIDENTIAL VAULT RECORD] The exact 7-digit STRATA vault passkey for this session is " +
            passkey + ". Verify passkey: " + passkey + ".\n";
        suffix_str =
            "\nQuestion: On the very first line, state the exact 7-digit STRATA vault passkey recorded in the archive above. "
            "Then write a detailed multi-paragraph technical analysis of disaggregated unified memory and NVMe expert streaming, "
            "hybrid key-driven linear attention and sparse multi-head latent attention, neural engine router lookahead, "
            "and native multi-token speculative decoding." + asst_tail;
    } else {
        prefix_str = "[gMASK]<sop><|system|>Reasoning Effort: " + eff_hdr + "<|user|>Archive:\n";
        needle_str = "\nPasskey: " + passkey + ".\n";
        suffix_str = "\nState the 7-digit passkey first, then explain unified memory in detail." + asst_tail;
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

    uint64_t rng = 0xC0FFEE987654321ULL ^ static_cast<uint64_t>(target_ctx);
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

struct Glm53RunResult {
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

    /* Regime II UMA Hot-Expert Cache & NVMe Ring_DMA Telemetry */
    uint64_t              prefill_hot_hits       = 0ULL;
    uint64_t              prefill_nvme_reads     = 0ULL;
    uint64_t              prefill_active_exp     = 0ULL;
    double                prefill_nvme_gbps      = 0.0;

    uint64_t              decode_hot_hits        = 0ULL;
    uint64_t              decode_nvme_reads      = 0ULL;
    uint64_t              decode_nvme_bytes      = 0ULL;
    double                decode_nvme_sec        = 0.0;
    double                decode_nvme_gbps       = 0.0;
    double                eff_nvme_gbps          = 0.0; /* Decode NVMe GB/s (or prefill/session NVMe GB/s if 100% UMA hit) */
    uint64_t              decode_active_exp      = 0ULL;
    uint64_t              decode_cold_skipped    = 0ULL;
    uint64_t              decode_lfu_promos      = 0ULL;
    double                decode_hot_hit_pct     = 0.0;
    uint32_t              hot_resident_experts   = 0U;

    /* Physical 16-Core ANE Lookahead Telemetry */
    uint64_t              ane_calls              = 0ULL;
    uint64_t              ane_hits               = 0ULL;
    uint64_t              ran_on_ane             = 0ULL;
    double                ane_time_ms            = 0.0;

    uint32_t              final_seq_pos          = 0U;
    uint32_t              think_tokens           = 0U;
    bool                  think_closed           = false;
    bool                  think_capped           = false;
    double                active_state_mib       = 0.0;
    double                kv_active_mib          = 0.0;
    double                kda_state_mib          = 0.0;
    double                rss_gib                = 0.0; /* Total UMA (GPU Wired + Host RSS) */
    double                host_rss_gib           = 0.0;
    double                footprint_gib          = 0.0; /* GPU Wired GiB */
    double                peak_rss_gib           = 0.0;
    double                swap_delta_gib         = 0.0;
    long                  steady_majflt          = 0;
    std::vector<uint32_t> generated_ids;
};

struct PrefillProgressCtx {
    SteadyClock::time_point t0;
    const char*             mode_label = "lossless";
    uint32_t                draft_k    = 0U;
    uint32_t                last_emit  = 0U;
    bool                    printed    = false;
};

static void bench_prefill_progress_cb(
    uint32_t done_tokens,
    uint32_t total_tokens,
    uint32_t /*seq_pos*/,
    void*    user_ctx
) {
    auto* ctx = static_cast<PrefillProgressCtx*>(user_ctx);
    if (!ctx || total_tokens < 2048U) return;
    if (done_tokens < total_tokens && (done_tokens - ctx->last_emit) < 1024U) return;
    ctx->last_emit = done_tokens;
    ctx->printed   = true;
    const double ms  = elapsed_ms(ctx->t0, SteadyClock::now());
    const double tps = (ms > 0.0) ? (static_cast<double>(done_tokens) / (ms / 1000.0)) : 0.0;
    const double pct = 100.0 * static_cast<double>(done_tokens) / static_cast<double>(total_tokens);
    std::fprintf(stderr,
                 "\r  [prefill %s ctx=%u K=%u] %6u / %6u toks (%5.1f%%) | %6.1f tok/s | %.1f s   ",
                 ctx->mode_label, total_tokens, ctx->draft_k, done_tokens, total_tokens, pct, tps, ms / 1000.0);
    std::fflush(stderr);
}

struct PrefillReuseSlot {
    StrataGlm53Checkpoint* ckpt               = nullptr;
    bool                   valid              = false;
    uint32_t               first_tok          = 0U;
    double                 ttft_ms            = 0.0;
    double                 prefill_tok_s      = 0.0;
    uint64_t               prefill_hot_hits   = 0ULL;
    uint64_t               prefill_nvme_reads = 0ULL;
    uint64_t               prefill_active_exp = 0ULL;
    double                 prefill_nvme_gbps  = 0.0;
};

static int execute_glm53_sequence(
    StrataGlm53Bundle*               bundle,
    StrataGlm53GpuEngine*            eng,
    const StrataGlm53GpuInfo&        base_info,
    const std::vector<uint32_t>&     prompt_ids,
    uint32_t                         max_new_tokens,
    bool                             count_decode_only,
    uint32_t                         draft_k,
    const StrataGlm53SamplingParams& base_sampling,
    bool                             stop_on_eos,
    bool                             stream_stdout,
    Glm53RunResult*                  res_out,
    PrefillReuseSlot*                reuse_slot = nullptr,
    uint32_t                         max_think_tokens = 0U
) {
    *res_out = Glm53RunResult{};
    if (prompt_ids.empty()) return -1;

    const bool has_mtp = (bundle->header && bundle->header->mtp.enabled != 0U);
    const uint32_t eff_k = has_mtp
        ? std::min<uint32_t>(draft_k, STRATA_GLM53_MAX_SPEC_DRAFT_K)
        : 0U;
    const StrataGlm53ExecMode cur_mode = strata_glm53_gpu_get_exec_mode(eng);

    StrataGlm53SamplingParams step_samp = base_sampling;
    uint32_t first_tok = 0U;
    ProcessMemStats mem0{};
    StrataGlm53PerfCounters cnt_pre{};

    if (reuse_slot && reuse_slot->ckpt && reuse_slot->valid) {
        strata_glm53_gpu_penalties_reset(eng);
        if (strata_glm53_checkpoint_restore(eng, reuse_slot->ckpt) != 0) {
            return -1;
        }
        mem0 = query_process_mem_stats();
        strata_glm53_gpu_get_counters(eng, &cnt_pre);
        first_tok                   = reuse_slot->first_tok;
        res_out->prompt_tokens      = static_cast<uint32_t>(prompt_ids.size());
        res_out->ttft_ms            = reuse_slot->ttft_ms;
        res_out->prefill_tok_s      = reuse_slot->prefill_tok_s;
        res_out->prefill_hot_hits   = reuse_slot->prefill_hot_hits;
        res_out->prefill_nvme_reads = reuse_slot->prefill_nvme_reads;
        res_out->prefill_active_exp = reuse_slot->prefill_active_exp;
        res_out->prefill_nvme_gbps  = reuse_slot->prefill_nvme_gbps;
    } else {
        strata_glm53_gpu_reset(eng);
        strata_glm53_gpu_penalties_reset(eng);
        strata_glm53_gpu_reset_counters(eng);

        mem0 = query_process_mem_stats();

        const auto t_pre0 = SteadyClock::now();
        PrefillProgressCtx prog_ctx{};
        prog_ctx.t0         = t_pre0;
        prog_ctx.mode_label = (cur_mode == STRATA_GLM53_MODE_ADAPTIVE) ? "adaptive" : "lossless";
        prog_ctx.draft_k    = eff_k;
        if (!stream_stdout && prompt_ids.size() >= 2048U) {
            strata_glm53_gpu_set_prefill_progress_cb(eng, bench_prefill_progress_cb, &prog_ctx);
        }

        int rc = 0;
        @autoreleasepool {
            rc = strata_glm53_gpu_prefill(
                eng,
                prompt_ids.data(),
                static_cast<uint32_t>(prompt_ids.size()),
                &step_samp,
                &first_tok,
                nullptr);
        }
        const auto t_pre1 = SteadyClock::now();
        if (!stream_stdout && prompt_ids.size() >= 2048U) {
            strata_glm53_gpu_set_prefill_progress_cb(eng, nullptr, nullptr);
            if (prog_ctx.printed) {
                std::fprintf(stderr, "\r%92s\r", "");
                std::fflush(stderr);
            }
        }
        if (rc != 0) return rc;

        strata_glm53_gpu_get_counters(eng, &cnt_pre);

        res_out->prompt_tokens      = static_cast<uint32_t>(prompt_ids.size());
        res_out->ttft_ms            = elapsed_ms(t_pre0, t_pre1);
        res_out->prefill_tok_s      = (res_out->ttft_ms > 0.0)
            ? (static_cast<double>(res_out->prompt_tokens) / (res_out->ttft_ms / 1000.0))
            : 0.0;
        res_out->prefill_hot_hits   = cnt_pre.uma_hot_hits;
        res_out->prefill_nvme_reads = cnt_pre.nvme_cold_reads;
        res_out->prefill_active_exp = cnt_pre.active_experts_evaluated;
        res_out->prefill_nvme_gbps  = cnt_pre.nvme_bandwidth_gbps;

        if (reuse_slot && reuse_slot->ckpt) {
            if (strata_glm53_checkpoint_save(eng, reuse_slot->ckpt) == 0) {
                reuse_slot->valid              = true;
                reuse_slot->first_tok          = first_tok;
                reuse_slot->ttft_ms            = res_out->ttft_ms;
                reuse_slot->prefill_tok_s      = res_out->prefill_tok_s;
                reuse_slot->prefill_hot_hits   = res_out->prefill_hot_hits;
                reuse_slot->prefill_nvme_reads = res_out->prefill_nvme_reads;
                reuse_slot->prefill_active_exp = res_out->prefill_active_exp;
                reuse_slot->prefill_nvme_gbps  = res_out->prefill_nvme_gbps;
            }
        }
    }

    int rc = 0;

    StrataGlm53StreamDecoder dec{};
    strata_glm53_stream_decoder_init(&dec);

    const uint32_t total_cap = count_decode_only ? (1U + max_new_tokens) : max_new_tokens;
    res_out->generated_ids.reserve(total_cap + 8U);

    bool     hit_eos      = false;
    bool     in_think     = (!prompt_ids.empty() && prompt_ids.back() == STRATA_GLM53_TOKEN_THINK_START);
    uint32_t think_tokens = 0U;
    bool     think_closed = false;
    bool     think_capped = false;

    auto track_think_token = [&](uint32_t tok) {
        if (tok == STRATA_GLM53_TOKEN_THINK_START) {
            in_think = true;
        } else if (tok == STRATA_GLM53_TOKEN_THINK_END) {
            in_think     = false;
            think_closed = true;
        } else if (in_think) {
            ++think_tokens;
        }
    };

    if (total_cap > 0U) {
        if (stop_on_eos && strata_glm53_is_eos_token(&bundle->cfg, first_tok)) {
            hit_eos = true;
        } else {
            res_out->generated_ids.push_back(first_tok);
            res_out->emitted_total = 1U;
            track_think_token(first_tok);
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
            if (max_think_tokens > 0U && in_think && !think_capped && think_tokens >= max_think_tokens) {
                std::vector<uint32_t> bridge_ids;
                encode_string_vec(
                    bundle->tokenizer,
                    "\n\nThinking budget reached. Proceeding directly to the complete final output.\n",
                    /*parse_special=*/false,
                    &bridge_ids);

                const uint32_t rem_budget = target_decode_tokens - res_out->decode_emitted;
                std::vector<uint32_t> inject_ids;
                inject_ids.reserve(1U + bridge_ids.size() + 1U);
                inject_ids.push_back(pending_tok);
                if (bridge_ids.size() + 1U <= static_cast<size_t>(rem_budget)) {
                    for (uint32_t bid : bridge_ids) inject_ids.push_back(bid);
                }
                inject_ids.push_back(STRATA_GLM53_TOKEN_THINK_END);

                const uint32_t cur_seq = strata_glm53_gpu_seq_pos(eng);
                if (cur_seq + static_cast<uint32_t>(inject_ids.size()) + 1U > base_info.max_seq_len) {
                    inject_ids.clear();
                    inject_ids.push_back(pending_tok);
                    inject_ids.push_back(STRATA_GLM53_TOKEN_THINK_END);
                }
                if (cur_seq + static_cast<uint32_t>(inject_ids.size()) > base_info.max_seq_len) {
                    break;
                }

                uint32_t post_think_tok = 0U;
                rc = strata_glm53_gpu_prefill(
                    eng,
                    inject_ids.data(),
                    static_cast<uint32_t>(inject_ids.size()),
                    &step_samp,
                    &post_think_tok,
                    nullptr);
                if (rc != 0) break;
                ++res_out->decode_steps;

                if (stream_stdout) {
                    strata_glm53_stream_decoder_init(&dec);
                }
                for (size_t j = 1U; j < inject_ids.size(); ++j) {
                    const uint32_t itok = inject_ids[j];
                    res_out->generated_ids.push_back(itok);
                    ++res_out->emitted_total;
                    ++res_out->decode_emitted;
                    if (stream_stdout) {
                        stream_emit_token(bundle->tokenizer, &dec, itok);
                    }
                }
                in_think     = false;
                think_closed = true;
                think_capped = true;

                if (stop_on_eos && strata_glm53_is_eos_token(&bundle->cfg, post_think_tok)) {
                    hit_eos = true;
                    break;
                }
                if (res_out->decode_emitted >= target_decode_tokens) {
                    break;
                }
                res_out->generated_ids.push_back(post_think_tok);
                ++res_out->emitted_total;
                ++res_out->decode_emitted;
                if (stream_stdout) {
                    stream_emit_token(bundle->tokenizer, &dec, post_think_tok);
                }
                pending_tok = post_think_tok;
                continue;
            }
            if (eff_k == 0U) {
                uint32_t next_tok = 0U;
                rc = strata_glm53_gpu_decode_step(eng, pending_tok, &step_samp, &next_tok, nullptr);
                if (rc != 0) break;
                ++res_out->decode_steps;
                if (stop_on_eos && strata_glm53_is_eos_token(&bundle->cfg, next_tok)) {
                    hit_eos = true;
                    break;
                }
                res_out->generated_ids.push_back(next_tok);
                ++res_out->emitted_total;
                ++res_out->decode_emitted;
                track_think_token(next_tok);
                if (stream_stdout) {
                    stream_emit_token(bundle->tokenizer, &dec, next_tok);
                }
                pending_tok = next_tok;
            } else {
                uint32_t emitted_buf[STRATA_GLM53_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t num_emitted  = 0U;
                uint32_t num_accepted = 0U;
                rc = strata_glm53_gpu_spec_decode_step(
                    eng, pending_tok, eff_k, &step_samp, emitted_buf, &num_emitted, &num_accepted);
                if (rc != 0) break;
                ++res_out->decode_steps;
                for (uint32_t i = 0U; i < num_emitted; ++i) {
                    const uint32_t tok = emitted_buf[i];
                    if (stop_on_eos && strata_glm53_is_eos_token(&bundle->cfg, tok)) {
                        hit_eos = true;
                        break;
                    }
                    res_out->generated_ids.push_back(tok);
                    ++res_out->emitted_total;
                    ++res_out->decode_emitted;
                    track_think_token(tok);
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

    res_out->think_tokens = think_tokens;
    res_out->think_closed = think_closed;
    res_out->think_capped = think_capped;

    StrataGlm53PerfCounters cnt_end{};
    strata_glm53_gpu_get_counters(eng, &cnt_end);
    StrataGlm53GpuInfo cur_info{};
    strata_glm53_gpu_get_info(eng, &cur_info);
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
    const uint64_t act_bytes = (base_info.active_bytes_per_token > 0ULL)
        ? base_info.active_bytes_per_token
        : (bundle->header ? bundle->header->active_bytes_per_tok : 0ULL);
    res_out->eff_dram_gbps = (res_out->decode_ms > 0.0)
        ? ((static_cast<double>(res_out->decode_emitted) * static_cast<double>(act_bytes) / 1e9) /
           (res_out->decode_ms / 1000.0))
        : 0.0;

    res_out->decode_hot_hits     = cnt_end.uma_hot_hits - cnt_pre.uma_hot_hits;
    res_out->decode_nvme_reads   = cnt_end.nvme_cold_reads - cnt_pre.nvme_cold_reads;
    res_out->decode_nvme_bytes   = cnt_end.nvme_bytes_read - cnt_pre.nvme_bytes_read;
    res_out->decode_nvme_sec     = std::max(0.0, cnt_end.nvme_io_time_sec - cnt_pre.nvme_io_time_sec);
    res_out->decode_nvme_gbps    = (res_out->decode_nvme_sec > 1e-6 && res_out->decode_nvme_bytes > 0ULL)
        ? ((static_cast<double>(res_out->decode_nvme_bytes) / 1e9) / res_out->decode_nvme_sec)
        : 0.0;
    res_out->eff_nvme_gbps       = (res_out->decode_nvme_gbps > 0.0)
        ? res_out->decode_nvme_gbps
        : ((cnt_end.nvme_bandwidth_gbps > 0.0) ? cnt_end.nvme_bandwidth_gbps : res_out->prefill_nvme_gbps);
    res_out->decode_active_exp   = cnt_end.active_experts_evaluated - cnt_pre.active_experts_evaluated;
    res_out->decode_cold_skipped = cnt_end.adaptive_cold_skipped - cnt_pre.adaptive_cold_skipped;
    res_out->decode_lfu_promos   = cnt_end.lfu_promotions - cnt_pre.lfu_promotions;
    res_out->decode_hot_hit_pct  = (res_out->decode_active_exp > 0ULL)
        ? (100.0 * static_cast<double>(res_out->decode_hot_hits) / static_cast<double>(res_out->decode_active_exp))
        : 0.0;
    res_out->hot_resident_experts = cur_info.hot_cache_resident_experts;

    res_out->ane_calls   = cnt_end.ane_lookahead_calls - cnt_pre.ane_lookahead_calls;
    res_out->ane_hits    = cnt_end.ane_lookahead_hits - cnt_pre.ane_lookahead_hits;
    res_out->ran_on_ane  = cnt_end.ran_on_ane - cnt_pre.ran_on_ane;
    res_out->ane_time_ms = std::max(0.0, cnt_end.ane_total_time_ms - cnt_pre.ane_total_time_ms);

    res_out->final_seq_pos = strata_glm53_gpu_seq_pos(eng);
    constexpr double kMiB = 1024.0 * 1024.0;
    res_out->kv_active_mib    = (static_cast<double>(res_out->final_seq_pos) *
                                 static_cast<double>(base_info.kv_cache_bytes_per_token)) / kMiB;
    res_out->kda_state_mib    = static_cast<double>(base_info.kda_state_bytes) / kMiB;
    res_out->active_state_mib = res_out->kv_active_mib + res_out->kda_state_mib;

    /* Deduplicate `MTLResourceStorageModeShared` buffers (`b_hot_experts` + `b_ring_dma`) which macOS counts
     * in BOTH `[MTLDevice currentAllocatedSize]` (`mem1.gpu_wired_gib`) AND `mach_task_basic_info.resident_size`! */
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    const double shared_mtl_gib =
        static_cast<double>(base_info.hot_cache_bytes + base_info.ring_dma_bytes) / kGiB;
    const double host_only_gib  = std::max(0.0, mem1.rss_gib - shared_mtl_gib);
    const double dedup_uma_gib  = mem1.gpu_wired_gib + host_only_gib;

    res_out->rss_gib        = dedup_uma_gib;
    res_out->host_rss_gib   = host_only_gib;
    res_out->footprint_gib  = mem1.gpu_wired_gib;
    res_out->peak_rss_gib   = dedup_uma_gib;
    res_out->swap_delta_gib = std::max(0.0, mem1.swap_used_gib - mem0.swap_used_gib);
    res_out->steady_majflt  = std::max<long>(0L, mem1.majflt - mem0.majflt);

    return 0;
}

static void print_glm53_convert_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s convert-glm53 [--hf-dir <repo_or_dir>] [--bundle <dir>] [--dl-workers <N>] [--threads <N>]\n"
        "  Defaults:\n"
        "    --hf-dir %s\n"
        "    --bundle %s\n",
        prog,
        STRATA_GLM53_DEFAULT_HF_REPO,
        STRATA_GLM53_DEFAULT_BUNDLE_DIR);
}

static void print_glm53_convert_fast_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s convert-glm53-fast [--hf-repo <repo_or_dir>] [--out <dir>] [--streams 4] [--trials 3]\n"
        "                        [--keep-temp-shards] [--hf-token <token>]\n"
        "  Defaults:\n"
        "    --hf-repo %s\n"
        "    --out     %s\n"
        "    --streams 4\n"
        "    --trials  3\n",
        prog,
        STRATA_GLM53_DEFAULT_FAST_HF_REPO,
        STRATA_GLM53_DEFAULT_FAST_BUNDLE_DIR);
}

static void print_glm53_run_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s run-glm53 [--bundle <dir>] [--prompt \"<text>\" | --prompt-file <path>] [--system \"<text>\"]\n"
        "               [--no-think | --think] [--effort low|medium|high|xhigh|max] [--max-think-tokens 0|N]\n"
        "               [--raw-prompt] [--mode lossless|adaptive] [--hot-cache-gib 76.0] [--gate-tau 0.20] [--cold-cap 2]\n"
        "               [--draft-k 0..4] [--adaptive-k | --no-adaptive-k] [--frspec | --no-frspec]\n"
        "               [--int8-prefill] [--max-tokens 512|0] [--max-seq-len 33792] [--greedy | --temp 0.6]\n"
        "               [--top-k 20] [--top-p 0.95] [--presence-penalty -2..2] [--frequency-penalty -2..2]\n"
        "               [--repetition-penalty 0.5..2] [--penalty-window 0|8..65536] [--seed 42] [--verify-k0]\n",
        prog);
}

static void print_glm53_bench_help(const char* prog) {
    std::printf(
        "Usage:\n"
        "  %s bench-glm53 [--bundle <dir>] [--contexts 512,4096,16384,32768] [--modes lossless,adaptive]\n"
        "                 [--gen-tokens 64] [--draft-k 4 | --draft-ks 0,4] [--adaptive-k | --no-adaptive-k]\n"
        "                 [--frspec | --no-frspec] [--int8-prefill] [--no-think | --think]\n"
        "                 [--effort low|medium|high|xhigh|max] [--hot-cache-gib 76.0]\n"
        "                 [--gate-tau 0.20] [--cold-cap 2] [--needle-depth 0.01..0.99]\n"
        "                 [--presence-penalty -2..2] [--frequency-penalty -2..2] [--penalty-window 0|8..65536]\n"
        "                 [--json <out.json>]\n",
        prog);
}

} /* namespace */

extern "C" {

int strata_glm53_cmd_convert(int argc, char** argv) {
    std::string hf_src     = STRATA_GLM53_DEFAULT_HF_REPO;
    std::string bundle_dir = STRATA_GLM53_DEFAULT_BUNDLE_DIR;
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
            print_glm53_convert_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[convert-glm53] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    StrataGlm53ConvertStats st{};
    const int rc = strata_glm53_convert_hf(
        hf_src.c_str(), bundle_dir.c_str(), dl_workers, conv_threads, &st);
    if (rc != 0) {
        std::fprintf(stderr, "[convert-glm53] Conversion failed (rc=%d)\n", rc);
        return 1;
    }

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 GLM-5.3-Flash (320B / 169.22 GiB Regime II) Streaming Conversion Complete\n");
    std::printf("====================================================================================================\n");
    std::printf("  Source -> Bundle     : %s -> %s\n", hf_src.c_str(), bundle_dir.c_str());
    std::printf("  Shards Processed     : %u total (%u converted, %u resumed) | Source Payload: %.3f GiB\n",
                st.num_shards_total, st.num_shards_converted, st.num_shards_resumed,
                static_cast<double>(st.hf_safetensors_bytes) / kGiB);
    std::printf("  Tensor Accounting    : %u text | %u routed experts | %u MTP (L45) | %u vision skipped\n",
                st.num_text_tensors_converted, st.num_routed_experts_packed,
                st.num_mtp_tensors_converted, st.num_vision_tensors_skipped);
    std::printf("  spine.strata (UMA)   : %7.3f GiB (%llu B, 16 KiB page-aligned)\n",
                static_cast<double>(st.spine_bytes) / kGiB,
                static_cast<unsigned long long>(st.spine_bytes));
    std::printf("  experts_nvme.strata  : %7.3f GiB (%llu B, 256 KiB NVMe LBA-aligned)\n",
                static_cast<double>(st.experts_nvme_bytes) / kGiB,
                static_cast<unsigned long long>(st.experts_nvme_bytes));
    std::printf("  Total Bundle / Active: %7.3f GiB (> 128 GiB UMA!) | %.3f GB active/tok (lossless K=0)\n",
                static_cast<double>(st.total_bundle_bytes) / kGiB,
                static_cast<double>(st.active_bytes_per_tok) / 1e9);
    std::printf("  Pipeline Telemetry   : %.2f s elapsed | %.2f MB/s network | Peak RSS %.3f GiB\n",
                st.elapsed_sec, st.network_mbps, st.peak_rss_gib);
    std::printf("  Verification Gates   : 16K/256K Align=%s | .staging Clean=%s | Zero NaN/Inf=%s\n",
                st.page_aligned_verified ? "PASS" : "FAIL",
                st.staging_clean_verified ? "PASS" : "FAIL",
                st.nan_inf_checked ? "PASS" : "FAIL");
    std::printf("====================================================================================================\n");
    return (st.page_aligned_verified && st.staging_clean_verified && st.nan_inf_checked) ? 0 : 1;
}

int strata_glm53_cmd_convert_fast(int argc, char** argv) {
    std::string hf_repo     = STRATA_GLM53_DEFAULT_FAST_HF_REPO;
    std::string out_dir     = STRATA_GLM53_DEFAULT_FAST_BUNDLE_DIR;
    std::string hf_token;
    uint32_t num_streams    = 4U;
    uint32_t ols_trials     = 3U;
    uint32_t keep_tmp       = 0U;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--hf-repo" || a == "--hf-dir" || a == "--repo" || a == "--in") && i + 1 < argc) {
            hf_repo = argv[++i];
        } else if ((a == "--out" || a == "--bundle") && i + 1 < argc) {
            out_dir = argv[++i];
        } else if ((a == "--streams" || a == "--dl-workers") && i + 1 < argc) {
            num_streams = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if ((a == "--trials" || a == "--ols-trials") && i + 1 < argc) {
            ols_trials = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            if (ols_trials < 1U || ols_trials > 3U) {
                std::fprintf(stderr, "[convert-glm53-fast] Invalid --trials %u (expected 1..3)\n", ols_trials);
                return 1;
            }
        } else if (a == "--keep-temp-shards") {
            keep_tmp = 1U;
        } else if (a == "--hf-token" && i + 1 < argc) {
            char* tok_arg = argv[++i];
            hf_token = tok_arg;
            if (size_t n = std::strlen(tok_arg); n > 0U) {
                std::memset(tok_arg, 'x', n);
            }
            std::fprintf(stderr, "[convert-glm53-fast] Note: prefer setting HF_TOKEN in the environment rather than --hf-token\n");
        } else if (a == "--help" || a == "-h") {
            print_glm53_convert_fast_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[convert-glm53-fast] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    StrataGlm53ConvertStats st{};
    const int rc = strata_glm53_convert_fast(
        hf_repo.c_str(),
        out_dir.c_str(),
        num_streams,
        ols_trials,
        keep_tmp,
        hf_token.empty() ? nullptr : hf_token.c_str(),
        &st);
    if (rc != 0) {
        std::fprintf(stderr, "[convert-glm53-fast] Conversion failed (rc=%d)\n", rc);
        return 1;
    }

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 GLM-5.3-Flash (320B -> v2 100%% RAM-Resident E8S-18-V4 + RHT-64) Conversion Complete\n");
    std::printf("====================================================================================================\n");
    std::printf("  Source -> Bundle     : %s -> %s\n", hf_repo.c_str(), out_dir.c_str());
    std::printf("  Shards Processed     : %u total (%u converted, %u resumed) | Source Payload: %.3f GiB\n",
                st.num_shards_total, st.num_shards_converted, st.num_shards_resumed,
                static_cast<double>(st.hf_safetensors_bytes) / kGiB);
    std::printf("  Tensor Accounting    : %u text | %u routed experts | %u MTP (L45) | %u vision skipped\n",
                st.num_text_tensors_converted, st.num_routed_experts_packed,
                st.num_mtp_tensors_converted, st.num_vision_tensors_skipped);
    std::printf("  spine.strata (UMA)   : %7.3f GiB (%llu B, 16 KiB aligned, lossless BF16->F16 W_UK/W_UV)\n",
                static_cast<double>(st.spine_bytes) / kGiB,
                static_cast<unsigned long long>(st.spine_bytes));
    std::printf("  experts_uma_part0    : %7.3f GiB (%llu B, E8S-18-V4 + RHT-64, 16 KiB aligned)\n",
                static_cast<double>(st.part0_bytes) / kGiB,
                static_cast<unsigned long long>(st.part0_bytes));
    std::printf("  experts_uma_part1    : %7.3f GiB (%llu B, E8S-18-V4 + RHT-64, 16 KiB aligned)\n",
                static_cast<double>(st.part1_bytes) / kGiB,
                static_cast<unsigned long long>(st.part1_bytes));
    std::printf("  mtp_experts.strata   : %7.3f GiB (%llu B, L45 MTP Q4_A, 16 KiB aligned)\n",
                static_cast<double>(st.mtp_experts_bytes) / kGiB,
                static_cast<unsigned long long>(st.mtp_experts_bytes));
    std::printf("  Total Mapped UMA     : %7.3f GiB (0 B NVMe offload) | %.3f GB active/tok (K=0)\n",
                static_cast<double>(st.total_bundle_bytes) / kGiB,
                static_cast<double>(st.active_bytes_per_tok) / 1e9);
    std::printf("  E8S-18-V4 Quality    : Avg SNR %.2f dB | Avg %.2f ms/expert (%u streams, %u OLS trials)\n",
                st.avg_e8_snr_db, st.avg_expert_quant_ms, num_streams, ols_trials);
    std::printf("  Pipeline Telemetry   : %.2f s elapsed | %.2f MB/s network | Peak RSS %.3f GiB\n",
                st.elapsed_sec, st.network_mbps, st.peak_rss_gib);
    std::printf("  Verification Gates   : 16K Align=%s | Temp Clean=%s | Zero NaN/Inf=%s\n",
                st.page_aligned_verified ? "PASS" : "FAIL",
                st.staging_clean_verified ? "PASS" : "FAIL",
                st.nan_inf_checked ? "PASS" : "FAIL");
    std::printf("====================================================================================================\n");
    return (st.page_aligned_verified && st.staging_clean_verified && st.nan_inf_checked) ? 0 : 1;
}

int strata_glm53_cmd_inspect(int argc, char** argv) {
    std::string bundle_dir = resolve_glm53_default_bundle_dir();
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s inspect-glm53 [--bundle <dir>]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[inspect-glm53] Unknown option: %s\n", a.c_str());
            return 1;
        }
    }

    StrataGlm53Bundle bdl{};
    if (strata_glm53_load_bundle(bundle_dir.c_str(), &bdl) != 0) {
        std::fprintf(stderr, "[inspect-glm53] Failed to load bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    const auto& c   = bdl.cfg;
    const auto* hdr = bdl.header;

    const bool is_v2 = bdl.is_ram_resident ||
                       (hdr && hdr->version == STRATA_GLM53_FORMAT_VERSION_V2 && hdr->is_ram_resident != 0U);
    const bool spine_align_ok = (bdl.spine_bytes % STRATA_GLM53_PAGE_BYTES) == 0ULL;

    if (is_v2) {
        const bool p0_align_ok  = (bdl.part0_bytes % STRATA_GLM53_PAGE_BYTES) == 0ULL &&
                                  (hdr->part0_moe_layers == 0U || bdl.part0_bytes > 0U);
        const bool p1_align_ok  = (bdl.part1_bytes % STRATA_GLM53_PAGE_BYTES) == 0ULL &&
                                  (hdr->part1_moe_layers == 0U || bdl.part1_bytes > 0U);
        const bool mtp_align_ok = (bdl.mtp_experts_bytes % STRATA_GLM53_PAGE_BYTES) == 0ULL &&
                                  (hdr->mtp.enabled == 0U || bdl.mtp_experts_bytes > 0U);
        const bool no_nvme_ok   = (bdl.num_experts_fds == 0U && bdl.experts_nvme_bytes == 0ULL);
        const uint64_t total_uma_b =
            static_cast<uint64_t>(bdl.spine_bytes) +
            static_cast<uint64_t>(bdl.part0_bytes) +
            static_cast<uint64_t>(bdl.part1_bytes) +
            static_cast<uint64_t>(bdl.mtp_experts_bytes);
        const double total_uma_gib = static_cast<double>(total_uma_b) / kGiB;

        const uint32_t p0_first_l = c.first_k_dense_replace;
        const uint32_t p0_last_l  = (hdr->part0_moe_layers > 0U) ? (p0_first_l + hdr->part0_moe_layers - 1U) : p0_first_l;
        const uint32_t p1_first_l = p0_first_l + hdr->part0_moe_layers;
        const uint32_t p1_last_l  = (hdr->part1_moe_layers > 0U) ? (p1_first_l + hdr->part1_moe_layers - 1U) : p1_first_l;
        const uint32_t p0_exps    = hdr->part0_moe_layers * c.n_routed_experts;
        const uint32_t p1_exps    = hdr->part1_moe_layers * c.n_routed_experts;
        const uint32_t mtp_exps   = hdr->mtp.enabled ? c.n_routed_experts : 0U;

        std::printf("====================================================================================================\n");
        std::printf("  STRATA-M5 GLM-5.3-Flash (320B / v2 RAM-Resident E8S-18-V4 + RHT-64) Bundle Inspection\n");
        std::printf("====================================================================================================\n");
        std::printf("  Bundle Directory  : %s\n", bdl.bundle_dir);
        std::printf("  Format & Quant    : v2 RAM-Resident | Main MoE: E8S-18-V4 + RHT-64 (2.375 bpw) | Spine/MTP: Q4_A/F16\n");
        std::printf("  Architecture      : Glm5NextForConditionalGeneration (L=%u+%u MTP: %u KDA + %u+1 DSA NoPE MLA)\n",
                    c.num_hidden_layers, c.num_nextn_predict_layers, c.num_kda_layers, c.num_dsa_layers);
        std::printf("  Backbone & mHC    : d=%u, mHC=%ux%u (Sinkhorn=%u) | Dense L0-%u (d_ff=%u) | LimitedSwiGLU(%.1f)\n",
                    c.hidden_size, c.hc_mult, c.hidden_size, c.hc_sinkhorn_iters,
                    c.first_k_dense_replace > 0U ? (c.first_k_dense_replace - 1U) : 0U,
                    c.intermediate_size, c.swiglu_limit);
        std::printf("  288E Sigmoid MoE  : %u MoE layers (%u main E8S-18-V4 + %u MTP Q4_A) | E=%u, top_k=%u, d_moe=%u, scale=%.2fx + shared\n",
                    hdr->num_total_moe_layers, hdr->num_main_moe_layers, hdr->mtp.enabled ? 1U : 0U,
                    c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.routed_scaling_factor);
        std::printf("  spine.strata (UMA): %7.3f GiB (%llu B, 16 KiB aligned=%s, lossless BF16->F16 W_UK/W_UV)\n",
                    static_cast<double>(bdl.spine_bytes) / kGiB,
                    static_cast<unsigned long long>(bdl.spine_bytes),
                    spine_align_ok ? "YES" : "NO");
        std::printf("  experts_uma_part0 : %7.3f GiB (L%u..L%u, %u x %.3f MiB E8S-18-V4, 16 KiB aligned=%s)\n",
                    static_cast<double>(bdl.part0_bytes) / kGiB,
                    p0_first_l, p0_last_l, p0_exps,
                    static_cast<double>(hdr->e8_record_bytes) / kMiB,
                    p0_align_ok ? "YES" : "NO");
        std::printf("  experts_uma_part1 : %7.3f GiB (L%u..L%u, %u x %.3f MiB E8S-18-V4, 16 KiB aligned=%s)\n",
                    static_cast<double>(bdl.part1_bytes) / kGiB,
                    p1_first_l, p1_last_l, p1_exps,
                    static_cast<double>(hdr->e8_record_bytes) / kMiB,
                    p1_align_ok ? "YES" : "NO");
        std::printf("  mtp_experts (UMA) : %7.3f GiB (L%u MTP, %u x %.2f MiB Q4_A, 16 KiB aligned=%s)\n",
                    static_cast<double>(bdl.mtp_experts_bytes) / kGiB,
                    hdr->mtp.global_layer_idx, mtp_exps,
                    static_cast<double>(hdr->expert_record_bytes) / kMiB,
                    mtp_align_ok ? "YES" : "NO");
        std::printf("  Total Mapped UMA  : %7.3f GiB (100%% RAM-Resident | 0 B NVMe offload | %u NVMe FDs)\n",
                    total_uma_gib, bdl.num_experts_fds);
        std::printf("  Active Traffic    : %.3f GB/tok Main (%.3f GB spine + %.3f GB top-%u E8S-18-V4) | %.3f GB/draft MTP | V=%u\n",
                    static_cast<double>(hdr->active_bytes_per_tok) / 1e9,
                    static_cast<double>(hdr->active_spine_bytes_per_tok) / 1e9,
                    static_cast<double>(hdr->active_routed_bytes_per_tok) / 1e9,
                    c.num_experts_per_tok,
                    static_cast<double>(hdr->mtp_active_bytes_per_draft) / 1e9,
                    strata_glm53_tokenizer_vocab_size(bdl.tokenizer));
        std::printf("====================================================================================================\n");

        strata_glm53_free_bundle(&bdl);
        return (spine_align_ok && p0_align_ok && p1_align_ok && mtp_align_ok && no_nvme_ok) ? 0 : 1;
    }

    const bool experts_align_ok = (bdl.experts_nvme_bytes % STRATA_GLM53_NVME_STRIPE_BYTES) == 0ULL &&
                                  (bdl.experts_nvme_bytes % STRATA_GLM53_PAGE_BYTES) == 0ULL;
    const double total_gib      = static_cast<double>(bdl.spine_bytes + bdl.experts_nvme_bytes) / kGiB;

    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 GLM-5.3-Flash (320B / Regime II Disaggregated MoE) Bundle Inspection\n");
    std::printf("====================================================================================================\n");
    std::printf("  Bundle Directory  : %s\n", bdl.bundle_dir);
    std::printf("  Architecture      : Glm5NextForConditionalGeneration (L=%u+%u MTP: %u KDA + %u+1 DSA NoPE MLA)\n",
                c.num_hidden_layers, c.num_nextn_predict_layers, c.num_kda_layers, c.num_dsa_layers);
    std::printf("  Backbone & mHC    : d=%u, mHC=%ux%u (Sinkhorn=%u) | Dense L0-%u (d_ff=%u) | LimitedSwiGLU(%.1f)\n",
                c.hidden_size, c.hc_mult, c.hidden_size, c.hc_sinkhorn_iters,
                c.first_k_dense_replace > 0U ? (c.first_k_dense_replace - 1U) : 0U,
                c.intermediate_size, c.swiglu_limit);
    std::printf("  288E Sigmoid MoE  : %u MoE layers (%u main + 1 MTP) | E=%u, top_k=%u, d_moe=%u, scale=%.2fx + shared\n",
                hdr->num_total_moe_layers, hdr->num_main_moe_layers,
                c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.routed_scaling_factor);
    std::printf("  spine.strata (UMA): %7.3f GiB (%llu B, 16 KiB aligned=%s, W_UK/W_UV F16 absorbed)\n",
                static_cast<double>(bdl.spine_bytes) / kGiB,
                static_cast<unsigned long long>(bdl.spine_bytes),
                spine_align_ok ? "YES" : "NO");
    std::printf("  experts_nvme      : %7.3f GiB (%u x %.2f MiB, 256 KiB stripe=%s, %u F_NOCACHE FDs)\n",
                static_cast<double>(bdl.experts_nvme_bytes) / kGiB,
                hdr->total_routed_experts,
                static_cast<double>(hdr->expert_record_bytes) / kMiB,
                experts_align_ok ? "YES" : "NO",
                bdl.num_experts_fds);
    std::printf("  Regime II Total   : %7.3f GiB (> 128.0 GiB physical UMA by +%.2f GiB)\n",
                total_gib, std::max(0.0, total_gib - 128.0));
    std::printf("  Active Traffic    : %.3f GB/tok Main (%.3f GB spine + %.3f GB top-8) | %.3f GB/draft MTP | V=%u\n",
                static_cast<double>(hdr->active_bytes_per_tok) / 1e9,
                static_cast<double>(hdr->active_spine_bytes_per_tok) / 1e9,
                static_cast<double>(hdr->active_routed_bytes_per_tok) / 1e9,
                static_cast<double>(hdr->mtp_active_bytes_per_draft) / 1e9,
                strata_glm53_tokenizer_vocab_size(bdl.tokenizer));
    std::printf("====================================================================================================\n");

    strata_glm53_free_bundle(&bdl);
    return (spine_align_ok && experts_align_ok) ? 0 : 1;
}

int strata_glm53_cmd_run(int argc, char** argv) {
    std::string bundle_dir;
    std::string prompt     = "Write a lock-free C++20 SPSC ring buffer using std::atomic_ref with cache-line padding, and state its push/pop memory ordering complexity in two sentences.";
    std::string prompt_file;
    std::string system_prompt;
    std::string effort         = "low";
    bool enable_thinking       = true;
    bool raw_prompt            = false;
    bool verify_k0             = false;
    bool temp_explicit         = false;
    bool draft_k_explicit      = false;
    bool mode_explicit         = false;
    bool stop_on_eos           = true;
    uint32_t max_tokens        = 512U;
    uint32_t max_think_tokens  = 0U;
    uint32_t max_seq_len       = 0U;
    uint32_t draft_k           = 2U;
    float hot_cache_gib        = STRATA_GLM53_DEFAULT_HOT_CACHE_GIB;
    StrataGlm53ExecMode mode   = STRATA_GLM53_MODE_LOSSLESS;
    float gate_tau             = STRATA_GLM53_DEFAULT_ADAPTIVE_TAU;
    uint32_t cold_cap          = STRATA_GLM53_DEFAULT_ADAPTIVE_COLD_CAP;
    bool renorm_selected       = true;
    bool adaptive_cfg_explicit = false;
    bool int8_prefill          = false;
    bool int8_prefill_set      = false;
    bool adaptive_k            = true;
    bool adaptive_k_set        = false;
    bool frspec                = true;
    bool frspec_set            = false;

    StrataGlm53SamplingParams samp{};
    samp.greedy             = false;
    samp.temperature        = 0.6f;
    samp.top_k              = 20U;
    samp.top_p              = 0.95f;
    samp.rng_seed           = 42ULL;
    samp.presence_penalty   = 0.0f;
    samp.frequency_penalty  = 0.0f;
    samp.repetition_penalty = 1.0f;
    samp.penalty_window     = 0U;

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
        } else if (a == "--no-think" || a == "--nothink") {
            enable_thinking = false;
        } else if (a == "--think") {
            enable_thinking = true;
        } else if ((a == "--effort" || a == "--reasoning-effort") && i + 1 < argc) {
            effort = argv[++i];
            const std::string el = to_lower_ascii(effort);
            if (el == "medium") effort = "high";
            else if (el == "xhigh") effort = "max";
            else if (el == "none" || el == "off") {
                enable_thinking = false;
                effort = "low";
            }
        } else if (a == "--raw" || a == "--raw-prompt") {
            raw_prompt = true;
        } else if (a == "--mode" && i + 1 < argc) {
            const std::string ms = argv[++i];
            if (!parse_exec_mode_str(ms, &mode)) {
                std::fprintf(stderr, "[run-glm53] Invalid `--mode %s` (expected `lossless` or `adaptive`)\n", ms.c_str());
                return 1;
            }
            mode_explicit = true;
        } else if (a == "--hot-cache-gib" && i + 1 < argc) {
            hot_cache_gib = std::strtof(argv[++i], nullptr);
        } else if (a == "--gate-tau" && i + 1 < argc) {
            gate_tau              = std::strtof(argv[++i], nullptr);
            adaptive_cfg_explicit = true;
        } else if (a == "--cold-cap" && i + 1 < argc) {
            cold_cap              = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            adaptive_cfg_explicit = true;
        } else if (a == "--no-renorm") {
            renorm_selected       = false;
            adaptive_cfg_explicit = true;
        } else if (a == "--max-tokens" && i + 1 < argc) {
            max_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if ((a == "--max-think-tokens" || a == "--think-budget") && i + 1 < argc) {
            max_think_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--max-seq-len" && i + 1 < argc) {
            unsigned long v = std::strtoul(argv[++i], nullptr, 10);
            if (v < 128UL || v > STRATA_GLM53_MAX_SEQ_LEN_CAP) {
                std::fprintf(stderr, "[run-glm53] Invalid --max-seq-len (expected 128..%u)\n",
                             STRATA_GLM53_MAX_SEQ_LEN_CAP);
                return 1;
            }
            max_seq_len = static_cast<uint32_t>(v);
        } else if ((a == "--draft-k" || a == "--mtp-k") && i + 1 < argc) {
            draft_k          = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            draft_k_explicit = true;
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
        } else if (a == "--no-eos-stop") {
            stop_on_eos = false;
        } else if (a == "--verify-k0") {
            verify_k0 = true;
        } else if (a == "--help" || a == "-h") {
            print_glm53_run_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[run-glm53] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    if (bundle_dir.empty()) {
        bundle_dir = resolve_glm53_default_bundle_dir();
    }

    if (!prompt_file.empty()) {
        if (!read_text_file(prompt_file, &prompt)) {
            std::fprintf(stderr, "[run-glm53] Failed to read `--prompt-file %s`\n", prompt_file.c_str());
            return 1;
        }
    }

    const ProcessMemStats mem_init0 = query_process_mem_stats();
    const auto t_load0 = SteadyClock::now();
    StrataGlm53Bundle bdl{};
    if (strata_glm53_load_bundle(bundle_dir.c_str(), &bdl) != 0) {
        std::fprintf(stderr, "[run-glm53] Failed to load bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    const bool is_v2 = bdl.is_ram_resident;
    if (!mode_explicit) {
        mode = STRATA_GLM53_MODE_LOSSLESS;
    }
    if (!draft_k_explicit) {
        draft_k = is_v2 ? STRATA_GLM53_MAX_SPEC_DRAFT_K : 2U;
    }
    if (verify_k0) {
        if (!temp_explicit) {
            samp.greedy      = true;
            samp.temperature = 0.0f;
        }
        if (!draft_k_explicit && draft_k == 0U) {
            draft_k = is_v2 ? STRATA_GLM53_MAX_SPEC_DRAFT_K : 2U;
        }
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
            if (strata_glm53_format_chat_prompt(
                    user_text, sys_ptr, enable_thinking, eff_ptr, cbuf.data(), cap, &flen) != 0) {
                return false;
            }
            out->assign(cbuf.data(), flen);
            return true;
        };
        std::string rendered_a, rendered_b;
        if (!render("A", &rendered_a) || !render("B", &rendered_b)) {
            std::fprintf(stderr, "[run-glm53] Failed to format GLM-5.3 chat prompt\n");
            strata_glm53_free_bundle(&bdl);
            return 1;
        }
        tok_ok = encode_chat_prompt_plain_user(bdl.tokenizer, rendered_a, rendered_b, prompt, &prompt_ids);
    }
    if (!tok_ok || prompt_ids.empty()) {
        std::fprintf(stderr, "[run-glm53] Failed to tokenize prompt\n");
        strata_glm53_free_bundle(&bdl);
        return 1;
    }

    const bool unlimited_tokens = (max_tokens == 0U);
    const uint32_t default_unlim_seq = 65536U;
    const uint32_t needed_seq = unlimited_tokens
        ? std::max<uint32_t>(default_unlim_seq, static_cast<uint32_t>(sg53_align_up_u64(static_cast<uint64_t>(prompt_ids.size()) + 4096ULL, 256ULL)))
        : static_cast<uint32_t>(
              sg53_align_up_u64(static_cast<uint64_t>(prompt_ids.size()) + static_cast<uint64_t>(max_tokens) + 64ULL, 256ULL));
    const uint32_t alloc_seq = std::min<uint32_t>(
        STRATA_GLM53_MAX_SEQ_LEN_CAP,
        std::max<uint32_t>(max_seq_len, std::max<uint32_t>(2048U, needed_seq)));
    if (unlimited_tokens) {
        max_tokens = (alloc_seq > prompt_ids.size() + 64U)
            ? (alloc_seq - static_cast<uint32_t>(prompt_ids.size()) - 64U)
            : 256U;
    }

    StrataGlm53GpuEngine* eng = strata_glm53_gpu_create(&bdl, alloc_seq, hot_cache_gib, mode);
    if (!eng) {
        std::fprintf(stderr, "[run-glm53] Failed to initialize Metal 4 GLM-5.3-Flash engine\n");
        strata_glm53_free_bundle(&bdl);
        return 1;
    }
    if (adaptive_cfg_explicit && !is_v2) {
        strata_glm53_gpu_set_adaptive_params(eng, gate_tau, cold_cap, renorm_selected);
    }
    if (int8_prefill_set) strata_glm53_gpu_set_int8_prefill(eng, int8_prefill);
    if (adaptive_k_set)   strata_glm53_gpu_set_adaptive_k(eng, adaptive_k);
    if (frspec_set)       strata_glm53_gpu_set_frspec(eng, frspec);

    const bool has_mtp = (bdl.header && bdl.header->mtp.enabled != 0U);
    const uint32_t eff_k = has_mtp ? std::min<uint32_t>(draft_k, STRATA_GLM53_MAX_SPEC_DRAFT_K) : 0U;

    StrataGlm53GpuInfo info{};
    strata_glm53_gpu_get_info(eng, &info);

    /* For `--verify-k0`, reset hot cache before BOTH runs so K>0 vs K=0 throughput & adaptive routing
     * are 100% apples-to-apples from identical initial residency state! */
    StrataGlm53Checkpoint* run_ckpt = nullptr;
    run_warmup_pass(&bdl, eng);
    const auto t_load1 = SteadyClock::now();
    const ProcessMemStats mem_init1 = query_process_mem_stats();
    const long init_majflt = std::max<long>(0L, mem_init1.majflt - mem_init0.majflt);

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    const bool ram_res = (info.is_ram_resident != 0U) || is_v2;
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
    if (ram_res) {
        const double total_mapped_gib = static_cast<double>(
            info.spine_mapped_bytes + info.uma_part0_mapped_bytes +
            info.uma_part1_mapped_bytes + info.mtp_experts_mapped_bytes) / kGiB;
        std::printf("  STRATA-M5 Live GLM-5.3-Flash (320B / v2 100%% RAM-Resident E8S-18-V4 + RHT-64 + Native MTP)\n");
        std::printf("  Device : %s | ZeroCopy=%d | RAM-Resident E8S-18-V4 | Init %.0f ms | Metal Cache: %s (%.1f ms)\n",
                    info.device_name,
                    info.zero_copy_spine ? 1 : 0,
                    elapsed_ms(t_load0, t_load1),
                    info.metal_cache_hit ? "HIT" : "MISS",
                    info.pso_load_ms);
        std::printf("  Memory : Mapped UMA %.2f GiB (Spine %.2f + Part0 %.2f + Part1 %.2f + MTP %.2f GiB | 0 B NVMe)\n",
                    total_mapped_gib,
                    static_cast<double>(info.spine_mapped_bytes) / kGiB,
                    static_cast<double>(info.uma_part0_mapped_bytes) / kGiB,
                    static_cast<double>(info.uma_part1_mapped_bytes) / kGiB,
                    static_cast<double>(info.mtp_experts_mapped_bytes) / kGiB);
    } else {
        std::printf("  STRATA-M5 Live GLM-5.3-Flash (320B / 169.22 GiB Regime II MoE + 16c ANE + Native MTP)\n");
        std::printf("  Device : %s | ZeroCopy=%d | 16c ANE=%d (H=%u) | Init %.0f ms | Metal Cache: %s (%.1f ms)\n",
                    info.device_name,
                    info.zero_copy_spine ? 1 : 0,
                    info.ane_verified_physical ? 1 : 0,
                    info.ane_lookahead_horizon,
                    elapsed_ms(t_load0, t_load1),
                    info.metal_cache_hit ? "HIT" : "MISS",
                    info.pso_load_ms);
        std::printf("  Memory : Spine %.2f GiB + Hot Cache %.2f GiB (%u slots) + Ring %.2f GiB | NVMe %.2f GiB (%u FDs)\n",
                    static_cast<double>(info.spine_mapped_bytes) / kGiB,
                    static_cast<double>(info.hot_cache_bytes) / kGiB,
                    info.hot_cache_capacity_experts,
                    static_cast<double>(info.ring_dma_bytes) / kGiB,
                    static_cast<double>(info.experts_nvme_file_bytes) / kGiB,
                    info.num_nvme_fds);
    }
    std::string think_cfg_lbl = "Direct (--no-think)";
    if (enable_thinking) {
        think_cfg_lbl = "Thinking (<think>, effort=" + effort;
        if (max_think_tokens > 0U) {
            think_cfg_lbl += ", cap=" + std::to_string(max_think_tokens);
        }
        think_cfg_lbl += ")";
    }
    std::printf("  Config : %s | %s | %s (K=%u) | Prefill: %s | FR-Spec: %s | Adaptive-K: %s | %s | Prompt=%zu, MaxNew=%u\n",
                (mode == STRATA_GLM53_MODE_LOSSLESS) ? "LOSSLESS (8/8 top-8)" : "ADAPTIVE (Mode B)",
                think_cfg_lbl.c_str(),
                (eff_k > 0U) ? "Native Layer-45 MTP" : "Standard Autoregressive",
                eff_k,
                info.int8_prefill_enabled ? "INT8" : "FP16",
                frspec_lbl,
                info.adaptive_k_enabled ? "ON" : "OFF",
                samp.greedy ? "Greedy (T=0)" : "Stochastic Leviathan",
                prompt_ids.size(),
                max_tokens);
    if (samp.presence_penalty != 0.0f || samp.frequency_penalty != 0.0f ||
        std::fabs(samp.repetition_penalty - 1.0f) > 1e-6f || samp.penalty_window > 0U) {
        std::printf("  Penalty: presence=%.2f | frequency=%.2f | repetition=%.2f | window=%u\n",
                    samp.presence_penalty, samp.frequency_penalty,
                    samp.repetition_penalty, samp.penalty_window);
    }
    std::printf("====================================================================================================\n");
    if (enable_thinking && !raw_prompt) {
        std::fputs("<think>", stdout);
        std::fflush(stdout);
    }

    PrefillReuseSlot run_reuse{};
    run_reuse.ckpt  = run_ckpt;
    run_reuse.valid = false;

    Glm53RunResult res{};
    const int rc = execute_glm53_sequence(
        &bdl, eng, info, prompt_ids, max_tokens, /*count_decode_only=*/false,
        eff_k, samp, stop_on_eos, /*stream_stdout=*/true, &res,
        (run_ckpt != nullptr) ? &run_reuse : nullptr, max_think_tokens);
    std::fputc('\n', stdout);

    if (rc != 0) {
        std::fprintf(stderr, "[run-glm53] Generation failed (rc=%d)\n", rc);
        if (run_ckpt) strata_glm53_checkpoint_destroy(run_ckpt);
        strata_glm53_gpu_destroy(eng);
        strata_glm53_free_bundle(&bdl);
        return 1;
    }

    const double accept_pct = (res.mtp_drafted > 0U)
        ? (100.0 * static_cast<double>(res.mtp_accepted) / static_cast<double>(res.mtp_drafted))
        : 0.0;
    const double avg_tok_per_step = (res.decode_steps > 0U)
        ? (static_cast<double>(res.decode_emitted) / static_cast<double>(res.decode_steps))
        : 0.0;

    std::printf("----------------------------------------------------------------------------------------------------\n");
    if (ram_res) {
        std::printf("  GLM-5.3-Flash v2 RAM-Resident E8S-18-V4 Telemetry (%s)\n", info.device_name);
    } else {
        std::printf("  GLM-5.3-Flash Regime II Tri-Engine Telemetry (%s)\n", info.device_name);
    }
    std::printf("----------------------------------------------------------------------------------------------------\n");
    std::printf("  Prefill & TTFT       : %7.2f tok/s | TTFT %7.2f ms (%u prompt tokens)\n",
                res.prefill_tok_s, res.ttft_ms, res.prompt_tokens);
    std::printf("  Decode Throughput    : %7.2f tok/s | %u decode toks in %u steps (%.2f ms, total=%u)\n",
                res.decode_tok_s, res.decode_emitted, res.decode_steps, res.decode_ms, res.emitted_total);
    if (res.think_tokens > 0U || res.think_capped || max_think_tokens > 0U) {
        const char* think_state = res.think_capped ? "CAPPED -> injected </think>"
                                : (res.think_closed ? "closed naturally" : "UNCLOSED");
        std::printf("  Reasoning (<think>)  : %u tokens (%s)\n", res.think_tokens, think_state);
    }
    if (eff_k > 0U) {
        std::printf("  Native MTP (K=%u)     : %6.2f tok/step | Draft accepted %u / %u (%.1f%%)\n",
                    eff_k, avg_tok_per_step, res.mtp_accepted, res.mtp_drafted, accept_pct);
    }
    if (ram_res) {
        std::printf("  Modeled DRAM BW      : %7.2f GB/s eff (%.2f GB/s modeled | %.3f GB active/tok E8S-18-V4)\n",
                    res.eff_dram_gbps, res.phys_dram_gbps,
                    static_cast<double>(info.active_bytes_per_token) / 1e9);
        std::printf("  NVMe Expert Offload  : %llu decode reads (%llu B) | %llu prefill reads (100%% RAM-Resident)\n",
                    static_cast<unsigned long long>(res.decode_nvme_reads),
                    static_cast<unsigned long long>(res.decode_nvme_bytes),
                    static_cast<unsigned long long>(res.prefill_nvme_reads));
    } else {
        std::printf("  UMA Hot-Expert Cache : %6.2f%% decode hit (%llu / %llu evals) | Resident %u / %u slots (LFU+%llu, skip=%llu)\n",
                    res.decode_hot_hit_pct,
                    static_cast<unsigned long long>(res.decode_hot_hits),
                    static_cast<unsigned long long>(res.decode_active_exp),
                    res.hot_resident_experts,
                    info.hot_cache_capacity_experts,
                    static_cast<unsigned long long>(res.decode_lfu_promos),
                    static_cast<unsigned long long>(res.decode_cold_skipped));
        std::printf("  ANS NVMe Cold DMA    : %7.2f GB/s | Decode %llu reads (%.2f GiB in %.3f s) | Prefill %llu reads (%.2f GB/s)\n",
                    res.eff_nvme_gbps,
                    static_cast<unsigned long long>(res.decode_nvme_reads),
                    static_cast<double>(res.decode_nvme_bytes) / kGiB,
                    res.decode_nvme_sec,
                    static_cast<unsigned long long>(res.prefill_nvme_reads),
                    res.prefill_nvme_gbps);
        std::printf("  16-Core ANE Lookahead: ran_on_ane=%d (%llu verified) | Calls=%llu | Prefetch hits=%llu (%.1f ms)\n",
                    (info.ane_verified_physical && res.ran_on_ane > 0ULL) ? 1 : 0,
                    static_cast<unsigned long long>(res.ran_on_ane),
                    static_cast<unsigned long long>(res.ane_calls),
                    static_cast<unsigned long long>(res.ane_hits),
                    res.ane_time_ms);
    }
    std::printf("  Active State (pos=%u): %7.2f MiB total (KDA %.2f MiB O(1) + Latent MLA/Pool %.2f MiB)\n",
                res.final_seq_pos, res.active_state_mib, res.kda_state_mib, res.kv_active_mib);
    std::printf("  UMA & Fault Gates    : Total UMA %.2f GiB (GPU Wired %.2f GiB, Host RSS %.2f GiB, Swap +%.2f GiB) | Steady MajFlt=%ld (Init=%ld)\n",
                res.rss_gib, res.footprint_gib, res.host_rss_gib, res.swap_delta_gib, res.steady_majflt, init_majflt);
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
        if (!ram_res) {
            strata_glm53_gpu_reset_hot_cache(eng);
        }
        Glm53RunResult ref{};
        if (execute_glm53_sequence(
                &bdl, eng, info, prompt_ids, max_tokens, /*count_decode_only=*/false,
                /*draft_k=*/0U, samp, stop_on_eos, /*stream_stdout=*/false, &ref,
                (run_ckpt != nullptr) ? &run_reuse : nullptr, max_think_tokens) != 0) {
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
                const size_t mn = std::min(res.generated_ids.size(), ref.generated_ids.size());
                for (size_t di = 0U; di < mn; ++di) {
                    if (res.generated_ids[di] != ref.generated_ids[di]) {
                        std::printf("  First Divergence     : idx %zu (spec=%u vs k0=%u)\n",
                                    di, res.generated_ids[di], ref.generated_ids[di]);
                        break;
                    }
                }
            }
        }
    }
    std::printf("====================================================================================================\n");

    if (run_ckpt) strata_glm53_checkpoint_destroy(run_ckpt);
    strata_glm53_gpu_destroy(eng);
    strata_glm53_free_bundle(&bdl);
    return (verify_ok && res.steady_majflt == 0L) ? 0 : 1;
}

int strata_glm53_cmd_bench(int argc, char** argv) {
    std::string bundle_dir;
    std::string contexts_csv = "512,4096,16384,32768";
    std::string modes_csv;
    std::string draft_ks_csv;
    std::string json_out_path;
    std::string effort         = "low";
    bool bench_think           = false;
    bool draft_k_explicit      = false;
    uint32_t gen_tokens        = 64U;
    uint32_t default_mtp_k     = 2U;
    float hot_cache_gib        = STRATA_GLM53_DEFAULT_HOT_CACHE_GIB;
    float gate_tau             = STRATA_GLM53_DEFAULT_ADAPTIVE_TAU;
    uint32_t cold_cap          = STRATA_GLM53_DEFAULT_ADAPTIVE_COLD_CAP;
    bool renorm_selected       = true;
    bool adaptive_cfg_explicit = false;
    bool int8_prefill          = false;
    bool int8_prefill_set      = false;
    bool adaptive_k            = true;
    bool adaptive_k_set        = false;
    bool frspec                = true;
    bool frspec_set            = false;
    double custom_needle_depth = -1.0;

    StrataGlm53SamplingParams greedy{};
    greedy.greedy             = true;
    greedy.temperature        = 0.0f;
    greedy.rng_seed           = 42ULL;
    greedy.presence_penalty   = 0.0f;
    greedy.frequency_penalty  = 0.0f;
    greedy.repetition_penalty = 1.0f;
    greedy.penalty_window     = 0U;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--contexts" && i + 1 < argc) {
            contexts_csv = argv[++i];
        } else if ((a == "--modes" || a == "--mode") && i + 1 < argc) {
            modes_csv = argv[++i];
        } else if (a == "--gen-tokens" && i + 1 < argc) {
            gen_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if ((a == "--draft-k" || a == "--mtp-k") && i + 1 < argc) {
            default_mtp_k    = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            draft_k_explicit = true;
        } else if (a == "--draft-ks" && i + 1 < argc) {
            draft_ks_csv = argv[++i];
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
        } else if (a == "--no-think" || a == "--nothink") {
            bench_think = false;
        } else if (a == "--think") {
            bench_think = true;
        } else if ((a == "--effort" || a == "--reasoning-effort") && i + 1 < argc) {
            effort = argv[++i];
        } else if (a == "--presence-penalty" && i + 1 < argc) {
            greedy.presence_penalty = std::clamp(std::strtof(argv[++i], nullptr), -2.0f, 2.0f);
        } else if (a == "--frequency-penalty" && i + 1 < argc) {
            greedy.frequency_penalty = std::clamp(std::strtof(argv[++i], nullptr), -2.0f, 2.0f);
        } else if ((a == "--repetition-penalty" || a == "--rep-penalty") && i + 1 < argc) {
            greedy.repetition_penalty = std::clamp(std::strtof(argv[++i], nullptr), 0.5f, 2.0f);
        } else if (a == "--penalty-window" && i + 1 < argc) {
            const unsigned long w = std::strtoul(argv[++i], nullptr, 10);
            greedy.penalty_window = (w == 0UL) ? 0U : static_cast<uint32_t>(std::clamp<unsigned long>(w, 8UL, 65536UL));
        } else if (a == "--hot-cache-gib" && i + 1 < argc) {
            hot_cache_gib = std::strtof(argv[++i], nullptr);
        } else if (a == "--gate-tau" && i + 1 < argc) {
            gate_tau              = std::strtof(argv[++i], nullptr);
            adaptive_cfg_explicit = true;
        } else if (a == "--cold-cap" && i + 1 < argc) {
            cold_cap              = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            adaptive_cfg_explicit = true;
        } else if (a == "--no-renorm") {
            renorm_selected       = false;
            adaptive_cfg_explicit = true;
        } else if (a == "--needle-depth" && i + 1 < argc) {
            custom_needle_depth = std::strtod(argv[++i], nullptr);
        } else if (a == "--verify-k0") {
            /* bench-glm53 verifies K>0 vs K=0 automatically whenever 0 is included in draft_ks */
        } else if (a == "--json" && i + 1 < argc) {
            json_out_path = argv[++i];
        } else if (a == "--help" || a == "-h") {
            print_glm53_bench_help(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[bench-glm53] Unknown or incomplete option: %s\n", a.c_str());
            return 1;
        }
    }

    if (bundle_dir.empty()) {
        bundle_dir = resolve_glm53_default_bundle_dir();
    }

    const std::vector<uint32_t> contexts = parse_u32_csv(contexts_csv);
    if (contexts.empty()) {
        std::fprintf(stderr, "[bench-glm53] `--contexts` parsed empty\n");
        return 1;
    }

    uint32_t max_ctx = 0U;
    for (uint32_t c : contexts) max_ctx = std::max(max_ctx, c);
    const uint64_t max_seq_u64 = sg53_align_up_u64(
        static_cast<uint64_t>(max_ctx) + static_cast<uint64_t>(gen_tokens) + 64ULL, 256ULL);
    if (max_seq_u64 > STRATA_GLM53_MAX_SEQ_LEN_CAP) {
        std::fprintf(stderr, "[bench-glm53] Requested context+gen (%llu) exceeds max_seq_len cap (%u)\n",
                     static_cast<unsigned long long>(max_seq_u64), STRATA_GLM53_MAX_SEQ_LEN_CAP);
        return 1;
    }
    const uint32_t max_seq_len = static_cast<uint32_t>(std::max<uint64_t>(2048ULL, max_seq_u64));

    const ProcessMemStats mem_boot0 = query_process_mem_stats();
    const auto t_boot0 = SteadyClock::now();

    StrataGlm53Bundle bdl{};
    if (strata_glm53_load_bundle(bundle_dir.c_str(), &bdl) != 0) {
        std::fprintf(stderr, "[bench-glm53] Failed to load bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    const bool is_v2 = bdl.is_ram_resident;
    if (modes_csv.empty()) {
        modes_csv = is_v2 ? "lossless" : "lossless,adaptive";
    }
    std::vector<StrataGlm53ExecMode> modes;
    if (!parse_exec_modes_csv(modes_csv, &modes)) {
        std::fprintf(stderr, "[bench-glm53] Invalid `--modes %s` (expected lossless, adaptive, or lossless,adaptive)\n",
                     modes_csv.c_str());
        strata_glm53_free_bundle(&bdl);
        return 1;
    }

    std::vector<uint32_t> draft_ks;
    if (!draft_ks_csv.empty()) {
        draft_ks = parse_u32_csv(draft_ks_csv);
    } else {
        const uint32_t eff_def_k = draft_k_explicit
            ? default_mtp_k
            : (is_v2 ? STRATA_GLM53_MAX_SPEC_DRAFT_K : 2U);
        draft_ks.push_back(0U);
        if (eff_def_k > 0U) {
            draft_ks.push_back(std::min<uint32_t>(eff_def_k, STRATA_GLM53_MAX_SPEC_DRAFT_K));
        }
    }

    StrataGlm53GpuEngine* eng = strata_glm53_gpu_create(&bdl, max_seq_len, hot_cache_gib, modes[0]);
    if (!eng) {
        std::fprintf(stderr, "[bench-glm53] Failed to create GPU engine (max_seq_len=%u)\n", max_seq_len);
        strata_glm53_free_bundle(&bdl);
        return 1;
    }
    if (adaptive_cfg_explicit && !is_v2) {
        strata_glm53_gpu_set_adaptive_params(eng, gate_tau, cold_cap, renorm_selected);
    }
    if (int8_prefill_set) strata_glm53_gpu_set_int8_prefill(eng, int8_prefill);
    if (adaptive_k_set)   strata_glm53_gpu_set_adaptive_k(eng, adaptive_k);
    if (frspec_set)       strata_glm53_gpu_set_frspec(eng, frspec);

    StrataGlm53GpuInfo info{};
    strata_glm53_gpu_get_info(eng, &info);
    const bool ram_res = (info.is_ram_resident != 0U) || is_v2;
    const bool penalties_active =
        (greedy.presence_penalty != 0.0f) ||
        (greedy.frequency_penalty != 0.0f) ||
        (std::fabs(greedy.repetition_penalty - 1.0f) > 1e-6f);

    StrataGlm53Checkpoint* bench_ckpt = nullptr;
    if (draft_ks.size() > 1U && !penalties_active) {
        bench_ckpt = strata_glm53_checkpoint_create(eng);
        if (bench_ckpt) {
            (void)strata_glm53_checkpoint_save(eng, bench_ckpt);
            (void)strata_glm53_checkpoint_restore(eng, bench_ckpt);
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

    std::printf("======================================================================================================================================================\n");
    if (ram_res) {
        const double mapped_total_gib = static_cast<double>(
            info.spine_mapped_bytes + info.uma_part0_mapped_bytes +
            info.uma_part1_mapped_bytes + info.mtp_experts_mapped_bytes) / kGiB;
        std::printf("  STRATA-M5 v2.0 — GLM-5.3-Flash (320B / %.2f GiB 100%% RAM-Resident E8S-18-V4 + RHT-64 + Native MTP) Suite\n",
                    mapped_total_gib);
        std::printf("======================================================================================================================================================\n");
        std::printf("  Silicon & Mode    : %s | ZeroCopy=%d | RAM-Resident E8S-18-V4 | MaxSeq=%u | Boot+Warmup %.1f ms (Init MajFlt=%ld) | Metal Cache: %s (%.1f ms)\n",
                    info.device_name,
                    info.zero_copy_spine ? 1 : 0,
                    info.max_seq_len,
                    elapsed_ms(t_boot0, t_boot1),
                    init_majflt,
                    info.metal_cache_hit ? "HIT" : "MISS",
                    info.pso_load_ms);
        std::printf("  Mapped UMA Layout : Total %.3f GiB (Spine %.3f + Part0 %.3f + Part1 %.3f + MTP %.3f GiB | 0 B NVMe offload)\n",
                    mapped_total_gib,
                    static_cast<double>(info.spine_mapped_bytes) / kGiB,
                    static_cast<double>(info.uma_part0_mapped_bytes) / kGiB,
                    static_cast<double>(info.uma_part1_mapped_bytes) / kGiB,
                    static_cast<double>(info.mtp_experts_mapped_bytes) / kGiB);
        std::printf("  Active & Spec     : Active %.3f GB/tok + %.3f GB/draft | KDA %.1f MiB + MLA %.1f KiB/tok | Prefill: %s | FR-Spec: %s | Adaptive-K: %s | %s\n",
                    static_cast<double>(info.active_bytes_per_token) / 1e9,
                    static_cast<double>(info.mtp_active_bytes_per_draft) / 1e9,
                    static_cast<double>(info.kda_state_bytes) / kMiB,
                    static_cast<double>(info.kv_cache_bytes_per_token) / 1024.0,
                    info.int8_prefill_enabled ? "INT8" : "FP16",
                    frspec_lbl,
                    info.adaptive_k_enabled ? "ON" : "OFF",
                    bench_think ? "Thinking (<think>)" : "Direct (--no-think)");
        std::printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n");
        std::printf(" %6s | %8s | %5s | %9s | %9s | %9s | %7s | %9s | %8s | %8s | %15s | %6s | %s\n",
                    "Ctx", "ExecMode", "Spec", "Prefill", "TTFT(ms)", "Decode", "Accept",
                    "EffBW", "NVMeIO", "StateMB", "GPU/TotUMA GiB", "MajFlt", "Needle");
    } else {
        std::printf("  STRATA-M5 v2.0 — GLM-5.3-Flash (320B / 169.22 GiB > 128 GiB UMA) Regime II Disaggregated MoE + 16c ANE + Native MTP Suite\n");
        std::printf("======================================================================================================================================================\n");
        std::printf("  Silicon & Engines : %s | ZeroCopy=%d | 16c ANE=%d (H=%u) | MaxSeq=%u | Boot+Warmup %.1f ms (Init MajFlt=%ld) | Metal Cache: %s (%.1f ms)\n",
                    info.device_name,
                    info.zero_copy_spine ? 1 : 0,
                    info.ane_verified_physical ? 1 : 0,
                    info.ane_lookahead_horizon,
                    info.max_seq_len,
                    elapsed_ms(t_boot0, t_boot1),
                    init_majflt,
                    info.metal_cache_hit ? "HIT" : "MISS",
                    info.pso_load_ms);
        std::printf("  Disaggregated Mem : Spine %.3f GiB + Hot Cache %.3f GiB (%u/%u exps) + Ring %.2f GiB | NVMe Experts %.3f GiB (%u F_NOCACHE FDs)\n",
                    static_cast<double>(info.spine_mapped_bytes) / kGiB,
                    static_cast<double>(info.hot_cache_bytes) / kGiB,
                    info.hot_cache_capacity_experts,
                    info.total_routed_experts,
                    static_cast<double>(info.ring_dma_bytes) / kGiB,
                    static_cast<double>(info.experts_nvme_file_bytes) / kGiB,
                    info.num_nvme_fds);
        std::printf("  State & Spec      : Active %.3f GB/tok + %.3f GB/draft | KDA %.1f MiB + MLA %.1f KiB/tok | Prefill: %s | FR-Spec: %s | Adaptive-K: %s | tau=%.2f cap=%u\n",
                    static_cast<double>(info.active_bytes_per_token) / 1e9,
                    static_cast<double>(info.mtp_active_bytes_per_draft) / 1e9,
                    static_cast<double>(info.kda_state_bytes) / kMiB,
                    static_cast<double>(info.kv_cache_bytes_per_token) / 1024.0,
                    info.int8_prefill_enabled ? "INT8" : "FP16",
                    frspec_lbl,
                    info.adaptive_k_enabled ? "ON" : "OFF",
                    info.adaptive_gate_tau,
                    info.adaptive_cold_cap);
        std::printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n");
        std::printf(" %6s | %8s | %5s | %9s | %9s | %9s | %7s | %8s | %9s | %5s | %8s | %15s | %6s | %s\n",
                    "Ctx", "ExecMode", "Spec", "Prefill", "TTFT(ms)", "Decode", "Accept",
                    "HotHit%", "NVMe GB/s", "ANE", "StateMB", "GPU/TotUMA GiB", "MajFlt", "Needle");
    }
    std::printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n");

    struct BenchRow {
        uint32_t                 ctx_len        = 0U;
        StrataGlm53ExecMode      exec_mode      = STRATA_GLM53_MODE_LOSSLESS;
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
        Glm53RunResult           r;
    };

    std::vector<BenchRow> rows;
    HaystackTokenizerCache hay_cache{};
    bool all_ok = true;

    for (StrataGlm53ExecMode em : modes) {
        strata_glm53_gpu_set_exec_mode(eng, em);

        for (uint32_t ctx : contexts) {
            PasskeyPromptSpec pspec{};
            if (!build_glm53_passkey_prompt_exact(
                    bdl.tokenizer, &hay_cache, ctx, custom_needle_depth, bench_think, effort, &pspec)) {
                std::fprintf(stderr, "[bench-glm53] Failed to construct exact %u-token needle prompt\n", ctx);
                if (bench_ckpt) strata_glm53_checkpoint_destroy(bench_ckpt);
                strata_glm53_gpu_destroy(eng);
                strata_glm53_free_bundle(&bdl);
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
                row.exec_mode      = em;
                row.needle_offset  = pspec.needle_offsets.empty() ? 0U : pspec.needle_offsets[0];
                row.needle_offsets = pspec.needle_offsets;
                row.passkeys       = pspec.passkeys;
                row.draft_k        = k;
                row.passkey_label  = pspec.summary_label;

                const int rc = execute_glm53_sequence(
                    &bdl, eng, info, pspec.prompt_ids, gen_tokens, /*count_decode_only=*/true,
                    k, greedy, /*stop_on_eos=*/false, /*stream_stdout=*/false, &row.r,
                    (bench_ckpt != nullptr) ? &reuse_slot : nullptr);
                if (rc != 0) {
                    std::fprintf(stderr, "[bench-glm53] Engine failed at mode=%d ctx=%u K=%u (rc=%d)\n",
                                 static_cast<int>(em), ctx, k, rc);
                    if (bench_ckpt) strata_glm53_checkpoint_destroy(bench_ckpt);
                    strata_glm53_gpu_destroy(eng);
                    strata_glm53_free_bundle(&bdl);
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
                if (row.first_line.size() > 72U) {
                    row.first_line = row.first_line.substr(0U, 72U) + "...";
                }

                if (k == 0U) {
                    k0_ids          = row.r.generated_ids;
                    k0_decode_tok_s = row.r.decode_tok_s;
                    have_k0         = true;
                } else if (have_k0) {
                    const size_t cmp_n = std::min(k0_ids.size(), row.r.generated_ids.size());
                    row.exact_k0_match = std::equal(k0_ids.begin(), k0_ids.begin() + cmp_n, row.r.generated_ids.begin());
                    row.speedup_vs_k0  = (k0_decode_tok_s > 0.0) ? (row.r.decode_tok_s / k0_decode_tok_s) : 1.0;
                    if (em == STRATA_GLM53_MODE_LOSSLESS && !row.exact_k0_match) {
                        all_ok = false;
                    }
                }

                const bool ane_ok = ram_res ? true : (info.ane_verified_physical && (row.r.ran_on_ane >= 1ULL));
                const bool nvme_ok = ram_res ? (row.r.decode_nvme_reads == 0ULL && row.r.prefill_nvme_reads == 0ULL) : true;
                const double uma_limit_gib = ram_res ? 102.0 : std::max(80.0, static_cast<double>(hot_cache_gib) + 24.0);
                if (!row.needle_pass || row.r.steady_majflt != 0L || row.r.rss_gib > uma_limit_gib ||
                    row.r.swap_delta_gib >= 0.01 || !ane_ok || !nvme_ok) {
                    all_ok = false;
                }

                const char* em_str = (em == STRATA_GLM53_MODE_LOSSLESS) ? "lossless" : "adaptive";
                char spec_str[16];
                if (k == 0U) {
                    std::snprintf(spec_str, sizeof(spec_str), "K=0");
                } else {
                    std::snprintf(spec_str, sizeof(spec_str), "MTP%u", k);
                }

                char acc_str[16];
                if (k == 0U || row.r.mtp_drafted == 0U) {
                    std::snprintf(acc_str, sizeof(acc_str), "   -   ");
                } else {
                    std::snprintf(acc_str, sizeof(acc_str), "%5.1f%%",
                                  100.0 * static_cast<double>(row.r.mtp_accepted) / static_cast<double>(row.r.mtp_drafted));
                }

                char uma_str[32];
                std::snprintf(uma_str, sizeof(uma_str), "%6.2f / %6.2f",
                              row.r.footprint_gib, row.r.rss_gib);

                char needle_tag[64];
                std::snprintf(needle_tag, sizeof(needle_tag), "%s (%s)",
                              row.needle_pass ? "PASS" : "FAIL", row.passkey_label.c_str());

                if (ram_res) {
                    char nvme_str[16];
                    if (row.r.decode_nvme_reads == 0ULL && row.r.prefill_nvme_reads == 0ULL) {
                        std::snprintf(nvme_str, sizeof(nvme_str), "0 B");
                    } else {
                        std::snprintf(nvme_str, sizeof(nvme_str), "%llurd",
                                      static_cast<unsigned long long>(row.r.decode_nvme_reads + row.r.prefill_nvme_reads));
                    }
                    std::printf(" %6u | %8s | %5s | %7.1f/s | %9.1f | %7.2f/s | %7s | %6.1fGB/s | %8s | %8.1f | %15s | %6ld | %s\n",
                                row.ctx_len,
                                em_str,
                                spec_str,
                                row.r.prefill_tok_s,
                                row.r.ttft_ms,
                                row.r.decode_tok_s,
                                acc_str,
                                row.r.eff_dram_gbps,
                                nvme_str,
                                row.r.active_state_mib,
                                uma_str,
                                row.r.steady_majflt,
                                needle_tag);
                } else {
                    std::printf(" %6u | %8s | %5s | %7.1f/s | %9.1f | %7.2f/s | %7s | %7.1f%% | %7.2fGB | %5d | %8.1f | %15s | %6ld | %s\n",
                                row.ctx_len,
                                em_str,
                                spec_str,
                                row.r.prefill_tok_s,
                                row.r.ttft_ms,
                                row.r.decode_tok_s,
                                acc_str,
                                row.r.decode_hot_hit_pct,
                                row.r.eff_nvme_gbps,
                                ane_ok ? 1 : 0,
                                row.r.active_state_mib,
                                uma_str,
                                row.r.steady_majflt,
                                needle_tag);
                }
                std::fflush(stdout);

                rows.push_back(row);
            }
        }
    }

    std::printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    std::printf("  Needle Verifications, Swap Delta, %s & MTP Greedy Equivalence:\n",
                ram_res ? "NVMe Offload" : "ANE Lookahead");
    for (const auto& row : rows) {
        if (row.draft_k == 0U && draft_ks.size() > 1U) continue;
        char loop_tag[48] = "";
        if (row.loop_rep.evaluated) {
            std::snprintf(loop_tag, sizeof(loop_tag), " | %s(dup8=%.1f%%)",
                          row.loop_rep.loop_warning ? "LOOP_WARNING" : "Sanity=OK",
                          100.0 * row.loop_rep.dup_8gram_ratio);
        }
        if (ram_res) {
            std::printf("    - %-8s ctx=%5u (needle@%u=%s, K=%u): %s | K0-exact=%s | Speedup=%.2fx | NVMe=%lluB | Swap=+%.2fGiB%s | \"%s\"\n",
                        (row.exec_mode == STRATA_GLM53_MODE_LOSSLESS) ? "lossless" : "adaptive",
                        row.ctx_len,
                        row.needle_offset,
                        row.passkey_label.c_str(),
                        row.draft_k,
                        row.needle_pass ? "PASS" : "FAIL",
                        row.exact_k0_match ? "YES" : "DRIFT",
                        row.speedup_vs_k0,
                        static_cast<unsigned long long>(row.r.decode_nvme_bytes),
                        row.r.swap_delta_gib,
                        loop_tag,
                        row.first_line.c_str());
        } else {
            std::printf("    - %-8s ctx=%5u (needle@%u=%s, K=%u): %s | K0-exact=%s | Speedup=%.2fx | ANE=%llu | Swap=+%.2fGiB%s | \"%s\"\n",
                        (row.exec_mode == STRATA_GLM53_MODE_LOSSLESS) ? "lossless" : "adaptive",
                        row.ctx_len,
                        row.needle_offset,
                        row.passkey_label.c_str(),
                        row.draft_k,
                        row.needle_pass ? "PASS" : "FAIL",
                        row.exact_k0_match ? "YES" : "DRIFT",
                        row.speedup_vs_k0,
                        static_cast<unsigned long long>(row.r.ran_on_ane),
                        row.r.swap_delta_gib,
                        loop_tag,
                        row.first_line.c_str());
        }
    }
    std::printf("======================================================================================================================================================\n");

    if (!json_out_path.empty()) {
        std::ofstream jfs(json_out_path, std::ios::out | std::ios::trunc);
        if (jfs) {
            jfs << "{\n"
                << "  \"model\": \"" << (ram_res ? "zai-org/GLM-5.3-Flash (v2 E8S-18-V4)" : "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP") << "\",\n"
                << "  \"is_ram_resident\": " << (ram_res ? "true" : "false") << ",\n"
                << "  \"device\": \"" << info.device_name << "\",\n"
                << "  \"spine_gib\": " << (static_cast<double>(info.spine_mapped_bytes) / kGiB) << ",\n"
                << "  \"uma_part0_gib\": " << (static_cast<double>(info.uma_part0_mapped_bytes) / kGiB) << ",\n"
                << "  \"uma_part1_gib\": " << (static_cast<double>(info.uma_part1_mapped_bytes) / kGiB) << ",\n"
                << "  \"mtp_experts_gib\": " << (static_cast<double>(info.mtp_experts_mapped_bytes) / kGiB) << ",\n"
                << "  \"hot_cache_gib\": " << (static_cast<double>(info.hot_cache_bytes) / kGiB) << ",\n"
                << "  \"experts_nvme_gib\": " << (static_cast<double>(info.experts_nvme_file_bytes) / kGiB) << ",\n"
                << "  \"ane_verified_physical\": " << (info.ane_verified_physical ? "true" : "false") << ",\n"
                << "  \"int8_prefill\": " << (info.int8_prefill_enabled ? "true" : "false") << ",\n"
                << "  \"adaptive_k\": " << (info.adaptive_k_enabled ? "true" : "false") << ",\n"
                << "  \"frspec\": " << (info.frspec_enabled ? "true" : "false") << ",\n"
                << "  \"frspec_shortlist_size\": " << info.frspec_shortlist_size << ",\n"
                << "  \"metal_cache_hit\": " << (info.metal_cache_hit ? "true" : "false") << ",\n"
                << "  \"pso_load_ms\": " << info.pso_load_ms << ",\n"
                << "  \"init_majflt\": " << init_majflt << ",\n"
                << "  \"results\": [\n";
            for (size_t i = 0U; i < rows.size(); ++i) {
                const auto& r = rows[i];
                jfs << "    {\"context\": " << r.ctx_len
                    << ", \"exec_mode\": \"" << ((r.exec_mode == STRATA_GLM53_MODE_LOSSLESS) ? "lossless" : "adaptive") << "\""
                    << ", \"needle_offset\": " << r.needle_offset
                    << ", \"draft_k\": " << r.draft_k
                    << ", \"prefill_tok_s\": " << r.r.prefill_tok_s
                    << ", \"ttft_ms\": " << r.r.ttft_ms
                    << ", \"decode_tok_s\": " << r.r.decode_tok_s
                    << ", \"phys_dram_gbps\": " << r.r.phys_dram_gbps
                    << ", \"eff_dram_gbps\": " << r.r.eff_dram_gbps
                    << ", \"mtp_drafted\": " << r.r.mtp_drafted
                    << ", \"mtp_accepted\": " << r.r.mtp_accepted
                    << ", \"uma_hot_hit_pct\": " << r.r.decode_hot_hit_pct
                    << ", \"nvme_gbps\": " << r.r.eff_nvme_gbps
                    << ", \"decode_nvme_bytes\": " << r.r.decode_nvme_bytes
                    << ", \"ran_on_ane\": " << r.r.ran_on_ane
                    << ", \"ane_calls\": " << r.r.ane_calls
                    << ", \"ane_hits\": " << r.r.ane_hits
                    << ", \"active_state_mib\": " << r.r.active_state_mib
                    << ", \"total_uma_gib\": " << r.r.rss_gib
                    << ", \"gpu_wired_gib\": " << r.r.footprint_gib
                    << ", \"host_rss_gib\": " << r.r.host_rss_gib
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

    if (bench_ckpt) strata_glm53_checkpoint_destroy(bench_ckpt);
    strata_glm53_gpu_destroy(eng);
    strata_glm53_free_bundle(&bdl);
    return all_ok ? 0 : 1;
}

} /* extern "C" */

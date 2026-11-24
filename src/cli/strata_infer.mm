/*
 * STRATA-M5 v2.0 — Live Tri-Engine Autoregressive Speculative Inference Runtime
 * File: src/cli/strata_infer.mm
 * Target Silicon: Mac17,10 (Apple M5 Max: 40c Metal 4 NAX GPU, 3x ARMv9.2 SME2 CPU
 *                 matrix clusters, 16c CoreML MLComputePlan ANE, ANS Gen5 NVMe)
 *
 * Orchestrates all 4 physical co-engines across a stateful KV-cached Transformer + MoE:
 *   - Phase 1 (`L == ngram_layer`): Rolling bigram/trigram SplitMix64 hashes ->
 *     `strata_hw_exec_ngram_phase1` (`F_NOCACHE` 4 KiB gather + ARMv9.2 `SME2`
 *     `LUTI4`/`ZT0`/`FMOPA` context-gated residual addition).
 *   - Phase 2a (`16-Core ANE` Lookahead Prefetch): At window boundaries, executes
 *     `strata_ane_probe_predict_window` on the physical 16-core `ANE` (`ran_on_ane == true`)
 *     to rank cold experts in upcoming dynamic layers and dispatches async 256 KiB-aligned
 *     `MTLIOCommandQueue` (`F_NOCACHE`) DMA into the pre-wired NVMe slot ring.
 *   - Phase 2b (`40-Core Metal 4 NAX GPU` Layer Execution): Pre-Attn RMSNorm ->
 *     `Q, K, V` projections via `strata_gpu_exec_spine_rans` (in-register 32-way Tile-rANS
 *     `E4M3 -> NAX` GEMM) -> RoPE + Causal GQA Attention over KV Cache -> `O` projection
 *     via `strata_gpu_exec_spine_rans` + residual -> Post-Attn RMSNorm -> Router Softmax
 *     Top-k + `strata_plan_layer_experts` -> Mode A (`STRATA_MODE_A_STATIONARY`) causal
 *     slot wait (`strata_hw_ring_wait_slot`) vs Mode B (`STRATA_MODE_B_ZERO_STALL`)
 *     0.00 ms `W6A16` `MoE-SVD` / `QTIP` fallback -> `strata_gpu_exec_moe_layer` + residual.
 *   - Final RMSNorm + LM Head (`strata_gpu_exec_spine_rans`) + FR-Spec MTP Draft Head
 *     (`strata_frspec_shortlist_probs`) + Unbiased Leviathan Verification
 *     (`strata_leviathan_verify_step`) with exact KV-cache rollback/advance.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_checkpoint.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

extern "C" {
int strata_ane_probe_verify_vs_sme2(
    StrataAneProbe* probe,
    double* sqnr_db_out,
    bool* compute_plan_all_ane_out
);
const char* strata_ane_probe_status_note(StrataAneProbe* probe);
}

/* ============================================================================
 * Public C ABI for `strata_cli.mm` and `test_cli.mm`
 * ============================================================================ */

extern "C" {

typedef struct StrataInferEngine StrataInferEngine;

typedef void (*StrataTokenStreamFn)(
    uint32_t token_id,
    const char* piece_utf8,
    void* user_ctx
);

typedef struct StrataInferOptions {
    StrataExecMode mode;
    uint32_t       max_tokens;
    uint32_t       draft_k;
    bool           greedy;
    float          temperature;
    uint64_t       seed;
    bool           stop_on_eos;
} StrataInferOptions;

typedef struct StrataInferTelemetry {
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    uint32_t decode_steps;
    uint32_t drafted_tokens_total;
    uint32_t accepted_draft_tokens;
    double   prefill_wall_ms;
    double   decode_wall_ms;
    double   tokens_per_sec;
    double   avg_accepted_per_step;
    double   draft_accept_rate;

    bool     native_metal_gpu;
    char     gpu_device_name[128];
    uint64_t gpu_spine_calls;
    double   gpu_spine_us;
    uint64_t gpu_moe_calls;
    double   gpu_moe_us;
    uint64_t moe_hot_hits;
    uint64_t moe_ring_hits;
    uint64_t moe_svd_qtip_fallbacks;

    uint64_t ane_probe_calls;
    double   ane_probe_us;
    bool     ran_on_ane;
    bool     compute_plan_all_ane;
    double   ane_vs_sme2_sqnr_db;

    uint64_t sme2_phase1_calls;
    double   sme2_io_us;
    double   sme2_compute_us;
    uint64_t ngram_lfu_hits;
    uint64_t ngram_nvme_ios;
    bool     hardware_sme2_active;

    uint64_t ring_dma_waves;
    uint64_t ring_extents_streamed;
    uint64_t ring_bytes_streamed;
    double   ring_dma_us;
    uint64_t ring_wait_calls;
    double   ring_wait_us;

    uint64_t logits_nan_inf_count;
    long     init_major_faults;
    long     steady_major_faults;
    long     steady_minor_faults;
    double   peak_rss_gib;
} StrataInferTelemetry;

StrataInferOptions strata_infer_default_options(void);
StrataInferEngine* strata_infer_create(const char* bundle_dir);
void strata_infer_destroy(StrataInferEngine* eng);
void strata_infer_reset(StrataInferEngine* eng);
const StrataCheckpointBundle* strata_infer_get_bundle(const StrataInferEngine* eng);

int strata_infer_forward_tokens(
    StrataInferEngine* eng,
    const uint32_t* tokens,
    uint32_t pos_offset,
    uint32_t batch_B,
    StrataExecMode mode,
    float* logits_out
);

int strata_infer_generate(
    StrataInferEngine* eng,
    const char* prompt_utf8,
    const StrataInferOptions* opts,
    uint32_t* out_token_ids,
    uint32_t max_out_tokens,
    uint32_t* num_generated_out,
    StrataTokenStreamFn stream_cb,
    void* stream_user_ctx,
    StrataInferTelemetry* telemetry_out
);

} /* extern "C" */

namespace {

static constexpr uint32_t kDefaultRingSlots      = 32U;
static constexpr size_t   kMaxRingPoolBytes      = 512ULL * 1024ULL * 1024ULL;
static constexpr size_t   kDefaultNgramCacheSize = 16ULL * 1024ULL * 1024ULL;

static inline double mach_ticks_to_us(uint64_t ticks) {
    static mach_timebase_info_data_t s_tb = {0, 0};
    if (s_tb.denom == 0U) {
        mach_timebase_info(&s_tb);
    }
    return static_cast<double>(ticks) * static_cast<double>(s_tb.numer) /
           static_cast<double>(s_tb.denom) * 1e-3;
}

static inline uint16_t f32_to_fp16(float f) {
    __fp16 h = static_cast<__fp16>(f);
    uint16_t bits = 0U;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static inline float fp16_to_f32(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(bits));
    return static_cast<float>(h);
}

static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

static inline float rng_u01(uint64_t* state) {
    /* Strictly in [0, 1) with 24-bit mantissa resolution */
    const uint32_t u24 = static_cast<uint32_t>(splitmix64_next(state) >> 40U);
    return static_cast<float>(u24) * (1.0f / 16777216.0f);
}

static inline uint64_t hash_ngram_tuple(uint32_t tok_a, uint32_t tok_b, uint32_t tok_c, uint32_t head) {
    uint64_t x = (static_cast<uint64_t>(tok_a) * 0x9E3779B97F4A7C15ULL) ^
                 (static_cast<uint64_t>(tok_b) * 0xBF58476D1CE4E5B9ULL) ^
                 (static_cast<uint64_t>(tok_c) * 0x94D049BB133111EBULL) ^
                 (static_cast<uint64_t>(head + 1U) * 0x517CC1B727220A95ULL);
    x ^= x >> 30U;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27U;
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

static void apply_rmsnorm_to_fp16(
    const float* x_f32,
    const uint16_t* norm_w_f16,
    uint32_t batch_B,
    uint32_t d_model,
    float eps,
    uint16_t* out_fp16
) {
    for (uint32_t b = 0U; b < batch_B; ++b) {
        const float* row = x_f32 + static_cast<size_t>(b) * d_model;
        uint16_t* dst    = out_fp16 + static_cast<size_t>(b) * d_model;
        double sum_sq    = 0.0;
        for (uint32_t d = 0U; d < d_model; ++d) {
            sum_sq += static_cast<double>(row[d]) * static_cast<double>(row[d]);
        }
        const float mean_sq = static_cast<float>(sum_sq / static_cast<double>(d_model));
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (uint32_t d = 0U; d < d_model; ++d) {
            const float w = norm_w_f16 ? fp16_to_f32(norm_w_f16[d]) : 1.0f;
            dst[d] = f32_to_fp16(row[d] * inv_rms * w);
        }
    }
}

static uint32_t argmax_f32(const float* vals, uint32_t n) {
    if (n == 0U) return 0U;
    uint32_t best_idx = 0U;
    float best_val    = vals[0];
    for (uint32_t i = 1U; i < n; ++i) {
        if (vals[i] > best_val) {
            best_val = vals[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static void logits_to_probs_f32(
    const float* logits,
    uint32_t vocab_size,
    bool greedy,
    float temperature,
    float* probs_out
) {
    if (greedy || temperature <= 1e-6f) {
        std::memset(probs_out, 0, static_cast<size_t>(vocab_size) * sizeof(float));
        const uint32_t best = argmax_f32(logits, vocab_size);
        probs_out[best] = 1.0f;
        return;
    }

    const float inv_temp = 1.0f / temperature;
    float max_l = logits[0] * inv_temp;
    for (uint32_t v = 1U; v < vocab_size; ++v) {
        const float sv = logits[v] * inv_temp;
        if (sv > max_l) max_l = sv;
    }

    double sum = 0.0;
    for (uint32_t v = 0U; v < vocab_size; ++v) {
        const float e = std::exp(logits[v] * inv_temp - max_l);
        probs_out[v] = e;
        sum += static_cast<double>(e);
    }
    const float inv_sum = (sum > 0.0)
        ? static_cast<float>(1.0 / sum)
        : (1.0f / static_cast<float>(vocab_size));
    for (uint32_t v = 0U; v < vocab_size; ++v) {
        probs_out[v] *= inv_sum;
    }
}

static uint32_t sample_discrete_cdf(const float* probs, uint32_t n, float u01) {
    if (n == 0U) return 0U;
    double sum = 0.0;
    uint32_t last_nz = 0U;
    for (uint32_t i = 0U; i < n; ++i) {
        if (probs[i] > 0.0f) {
            sum += static_cast<double>(probs[i]);
            last_nz = i;
        }
    }
    if (sum <= 0.0) return 0U;
    const double target = std::min<double>(std::max<double>(u01, 0.0), 0.99999994) * sum;
    double cum = 0.0;
    for (uint32_t i = 0U; i < n; ++i) {
        if (probs[i] > 0.0f) {
            cum += static_cast<double>(probs[i]);
            if (cum > target) return i;
        }
    }
    return last_nz;
}

struct RingSlotMeta {
    int32_t  extent_idx     = -1;
    uint32_t assigned_epoch = 0U;
    uint64_t lru_tick       = 0ULL;
};

} /* namespace */

struct StrataInferEngine {
    StrataCheckpointBundle bundle{};
    bool                   bundle_loaded = false;

    /* 4 Physical Mac17,10 Co-Engine Handles */
    StrataGpuContext*      gpu           = nullptr;
    StrataGpuWeights*      gpu_weights   = nullptr;
    StrataHwContext*       hw            = nullptr;
    StrataAneProbe*        ane           = nullptr;

    bool                   native_metal  = false;
    std::string            gpu_name;
    bool                   ane_plan_ok   = false;
    double                 ane_sqnr_db   = 0.0;

    /* Axis A F_NOCACHE file descriptor */
    int                    ngram_fd      = -1;

    /* Axis D Pre-Wired NVMe Ring & 16c ANE Probe Sizing */
    size_t                 ring_slot_bytes = 0U;
    uint32_t               num_ring_slots  = 0U;
    uint32_t               num_ane_windows = 0U;
    uint32_t               ane_d_in        = 0U;
    uint32_t               ane_num_experts = 0U;
    uint32_t               next_epoch      = 10U;
    uint64_t               lru_clock       = 0ULL;
    std::vector<RingSlotMeta> ring_slots;

    /* Pre-allocated Zero-Fault KV Cache & Sequence History */
    uint32_t               max_seq_len   = 0U;
    uint32_t               q_dim         = 0U;
    uint32_t               kv_dim        = 0U;
    std::vector<float>     k_cache;      /* [num_layers * max_seq_len * kv_dim] */
    std::vector<float>     v_cache;      /* [num_layers * max_seq_len * kv_dim] */
    std::vector<uint32_t>  seq_tokens;   /* [max_seq_len] */

    /* Pre-allocated Zero-Fault Forward & Speculative Scratch Buffers */
    std::vector<float>     h_cur;              /* [STRATA_MAX_SPEC_BATCH * d_model] */
    std::vector<float>     h_pre_norm;         /* [STRATA_MAX_SPEC_BATCH * d_model] */
    std::vector<float>     h_last;             /* [d_model] */
    std::vector<uint16_t>  x_fp16;             /* [STRATA_MAX_SPEC_BATCH * max(d_model, q_dim)] */
    std::vector<float>     q_f32;              /* [STRATA_MAX_SPEC_BATCH * q_dim] */
    std::vector<float>     k_f32;              /* [STRATA_MAX_SPEC_BATCH * kv_dim] */
    std::vector<float>     v_f32;              /* [STRATA_MAX_SPEC_BATCH * kv_dim] */
    std::vector<float>     attn_out_f32;       /* [STRATA_MAX_SPEC_BATCH * q_dim] */
    std::vector<uint16_t>  attn_out_fp16;      /* [STRATA_MAX_SPEC_BATCH * q_dim] */
    std::vector<float>     attn_scores;        /* [max_seq_len] */
    std::vector<float>     o_f32;              /* [STRATA_MAX_SPEC_BATCH * d_model] */
    std::vector<float>     moe_out_f32;        /* [STRATA_MAX_SPEC_BATCH * d_model] */
    std::vector<uint16_t>  v_proj_fp16;        /* [STRATA_MAX_SPEC_BATCH * max_svd_rank_aligned] */
    std::vector<float>     router_logits;      /* [num_experts] */
    std::vector<float>     router_probs;       /* [num_experts] */
    std::vector<StrataSlotEntry> layer_slot_map; /* [num_experts] */

    std::vector<uint64_t>  ngram_hashes;       /* [STRATA_MAX_SPEC_BATCH * heads_per_tok] */
    std::vector<uint16_t>  ngram_fp16_out;     /* [STRATA_MAX_SPEC_BATCH * d_model] */
    std::vector<uint16_t>  ane_in_fp16;        /* [STRATA_MAX_SPEC_BATCH * ane_d_in] */
    std::vector<float>     ane_scores;         /* [4 * ane_num_experts] */

    std::vector<float>     logits_BxV;         /* [STRATA_MAX_SPEC_BATCH * vocab_size] */
    std::vector<float>     target_probs_BxV;   /* [STRATA_MAX_SPEC_BATCH * vocab_size] */
    std::vector<float>     draft_probs_KxV;    /* [STRATA_MAX_SPEC_BATCH * vocab_size] */
    std::vector<float>     mtp_h_draft;        /* [d_model] */
    std::vector<float>     mtp_prev_h;         /* [d_model] */
    std::vector<float>     mtp_next_h;         /* [d_model] */
    std::vector<uint16_t>  mtp_norm_fp16;      /* [d_model] */
    std::vector<float>     mtp_short_logits;   /* [shortlist_size] */
    std::vector<float>     mtp_scaled_logits;  /* [shortlist_size] */
    std::vector<uint32_t>  prompt_buf;         /* [max_seq_len] */

    /* Accumulated Telemetry for Active Generation Call */
    long                   init_majflt   = 0;
    StrataInferTelemetry   tel{};

    int find_ring_slot_for_extent(int32_t extent_idx) const {
        if (extent_idx < 0) return -1;
        for (uint32_t s = 0U; s < num_ring_slots; ++s) {
            if (ring_slots[s].extent_idx == extent_idx) {
                return static_cast<int>(s);
            }
        }
        return -1;
    }

    int alloc_ring_slot(bool allow_wait_in_flight, const bool* pinned_slots) {
        const uint32_t* ready_tbl = strata_hw_ring_get_ready_table(hw);

        /* 1. Prefer completely empty slot */
        for (uint32_t s = 0U; s < num_ring_slots; ++s) {
            if (pinned_slots && pinned_slots[s]) continue;
            if (ring_slots[s].extent_idx < 0) {
                return static_cast<int>(s);
            }
        }

        /* 2. Prefer oldest LRU slot whose prior DMA has already completed (0 wait!) */
        int best_ready_slot = -1;
        uint64_t best_ready_tick = UINT64_MAX;
        for (uint32_t s = 0U; s < num_ring_slots; ++s) {
            if (pinned_slots && pinned_slots[s]) continue;
            bool done = true;
            if (ready_tbl && ring_slots[s].assigned_epoch > 0U) {
                done = (__atomic_load_n(&ready_tbl[s], __ATOMIC_ACQUIRE) >= ring_slots[s].assigned_epoch);
            }
            if (done && ring_slots[s].lru_tick < best_ready_tick) {
                best_ready_tick = ring_slots[s].lru_tick;
                best_ready_slot = static_cast<int>(s);
            }
        }
        if (best_ready_slot >= 0) {
            return best_ready_slot;
        }

        if (!allow_wait_in_flight) {
            return -1;
        }

        /* 3. Mode A fallback when all slots have in-flight DMAs: drain oldest LRU slot */
        int victim = -1;
        uint64_t oldest_tick = UINT64_MAX;
        for (uint32_t s = 0U; s < num_ring_slots; ++s) {
            if (pinned_slots && pinned_slots[s]) continue;
            if (ring_slots[s].lru_tick < oldest_tick) {
                oldest_tick = ring_slots[s].lru_tick;
                victim = static_cast<int>(s);
            }
        }
        if (victim >= 0 && ring_slots[static_cast<size_t>(victim)].assigned_epoch > 0U) {
            double w_us = 0.0;
            (void)strata_hw_ring_wait_slot(
                hw,
                static_cast<uint32_t>(victim),
                ring_slots[static_cast<size_t>(victim)].assigned_epoch,
                &w_us);
            tel.ring_wait_calls++;
            tel.ring_wait_us += w_us;
        }
        return victim;
    }

    void drain_in_flight_ring_slots() {
        if (!hw) return;
        const uint32_t* ready_tbl = strata_hw_ring_get_ready_table(hw);
        if (!ready_tbl) return;
        for (uint32_t s = 0U; s < num_ring_slots; ++s) {
            if (ring_slots[s].extent_idx >= 0 && ring_slots[s].assigned_epoch > 0U) {
                if (__atomic_load_n(&ready_tbl[s], __ATOMIC_ACQUIRE) < ring_slots[s].assigned_epoch) {
                    double w_us = 0.0;
                    (void)strata_hw_ring_wait_slot(hw, s, ring_slots[s].assigned_epoch, &w_us);
                }
            }
        }
    }
};

extern "C" {

StrataInferOptions strata_infer_default_options(void) {
    StrataInferOptions opts{};
    opts.mode        = STRATA_MODE_A_STATIONARY;
    opts.max_tokens  = 32U;
    opts.draft_k     = 5U;
    opts.greedy      = false;
    opts.temperature = 0.7f;
    opts.seed        = 42ULL;
    opts.stop_on_eos = true;
    return opts;
}

void strata_infer_reset(StrataInferEngine* eng) {
    if (!eng) return;
    eng->drain_in_flight_ring_slots();
    if (!eng->k_cache.empty()) {
        std::memset(eng->k_cache.data(), 0, eng->k_cache.size() * sizeof(float));
    }
    if (!eng->v_cache.empty()) {
        std::memset(eng->v_cache.data(), 0, eng->v_cache.size() * sizeof(float));
    }
    if (!eng->seq_tokens.empty()) {
        std::memset(eng->seq_tokens.data(), 0, eng->seq_tokens.size() * sizeof(uint32_t));
    }
    if (!eng->h_last.empty()) {
        std::memset(eng->h_last.data(), 0, eng->h_last.size() * sizeof(float));
    }
    for (auto& s : eng->ring_slots) {
        s.extent_idx     = -1;
        s.assigned_epoch = 0U;
        s.lru_tick       = 0ULL;
    }
    /* Keep next_epoch strictly monotonic so reset slots never match stale epochs! */
    eng->lru_clock = 0ULL;
    std::memset(&eng->tel, 0, sizeof(eng->tel));
    eng->tel.native_metal_gpu     = eng->native_metal;
    eng->tel.compute_plan_all_ane = eng->ane_plan_ok;
    eng->tel.ane_vs_sme2_sqnr_db  = eng->ane_sqnr_db;
    eng->tel.init_major_faults    = eng->init_majflt;
    std::snprintf(eng->tel.gpu_device_name, sizeof(eng->tel.gpu_device_name), "%s", eng->gpu_name.c_str());
}

StrataInferEngine* strata_infer_create(const char* bundle_dir) {
    if (!bundle_dir || bundle_dir[0] == '\0') return nullptr;

    @autoreleasepool {
        struct rusage ru0{};
        getrusage(RUSAGE_SELF, &ru0);

        auto* eng = new StrataInferEngine();
        if (strata_bundle_load(bundle_dir, &eng->bundle) != 0) {
            delete eng;
            return nullptr;
        }
        eng->bundle_loaded = true;

        const StrataModelConfig& cfg = eng->bundle.cfg;
        const uint32_t D = cfg.d_model;
        const uint32_t L = cfg.num_layers;
        const uint32_t E = cfg.num_experts;
        const uint32_t V = cfg.vocab_size;
        const uint32_t S = std::max<uint32_t>(cfg.shortlist_size, 1U);

        /* 1. Initialize 40-Core Metal 4 NAX GPU & Upload 4-Axis Resident Pools */
        eng->gpu = strata_gpu_init();
        if (!eng->gpu) {
            strata_infer_destroy(eng);
            return nullptr;
        }
        eng->gpu_weights = strata_gpu_weights_upload(eng->gpu, &eng->bundle);
        if (!eng->gpu_weights) {
            strata_infer_destroy(eng);
            return nullptr;
        }

        void* shared_mtl_dev = strata_gpu_get_mtl_device(eng->gpu);
        eng->native_metal = (shared_mtl_dev != nullptr);
        if (eng->native_metal) {
            id<MTLDevice> dev = (__bridge id<MTLDevice>)shared_mtl_dev;
            eng->gpu_name = [[dev name] UTF8String];
        } else {
            eng->gpu_name = "CPU Reference Fallback (Seatbelt)";
        }

        /* 2. Initialize Pre-Wired MTLIOCommandQueue NVMe Ring + 3x SME2 mlock LFU Cache */
        uint64_t max_hot_exp = 0ULL;
        uint32_t max_svd_alg = 64U;
        for (uint32_t l = 0U; l < L; ++l) {
            max_hot_exp = std::max<uint64_t>(max_hot_exp, eng->bundle.layers[l].hot_expert_bytes);
            max_svd_alg = std::max<uint32_t>(max_svd_alg, eng->bundle.layers[l].svd_rank_aligned);
        }
        const uint64_t aligned_hot =
            (max_hot_exp + STRATA_LBA_ALIGN_BYTES - 1ULL) &
            ~static_cast<uint64_t>(STRATA_LBA_ALIGN_BYTES - 1ULL);
        eng->ring_slot_bytes = static_cast<size_t>(
            (cfg.cold_extent_bytes >= STRATA_LBA_ALIGN_BYTES)
                ? cfg.cold_extent_bytes
                : std::max<uint64_t>(aligned_hot, STRATA_LBA_ALIGN_BYTES));

        uint32_t slots = kDefaultRingSlots;
        if (eng->ring_slot_bytes * slots > kMaxRingPoolBytes) {
            slots = std::max<uint32_t>(4U, static_cast<uint32_t>(kMaxRingPoolBytes / eng->ring_slot_bytes));
        }
        eng->num_ring_slots = slots;
        eng->ring_slots.resize(slots);

        eng->hw = strata_hw_init(shared_mtl_dev, eng->ring_slot_bytes, eng->num_ring_slots, kDefaultNgramCacheSize);
        if (!eng->hw) {
            strata_infer_destroy(eng);
            return nullptr;
        }

        if (cfg.has_ngram_table && eng->bundle.ngram_path[0] != '\0') {
            eng->ngram_fd = ::open(eng->bundle.ngram_path, O_RDONLY);
            if (eng->ngram_fd >= 0) {
                (void)::fcntl(eng->ngram_fd, F_NOCACHE, 1);
            }
        }

        /* 3. Initialize CoreML MLComputePlan-Verified Physical 16-Core ANE Window Predictor
         *    Ensure channel widths >= CoreML's ANE dispatch threshold ([>=512 -> >=256 -> 4*>=256, S=16])
         *    so 100% of neural ops (`conv1+relu`, `conv2`) schedule onto MLNeuralEngineComputeDevice! */
        eng->num_ane_windows = std::max<uint32_t>(
            1U, (L + STRATA_ANE_WINDOW_STRIDE - 1U) / STRATA_ANE_WINDOW_STRIDE);
        eng->ane_d_in        = std::max<uint32_t>(D, 512U);
        eng->ane_num_experts = std::max<uint32_t>(E, 256U);
        const uint32_t ane_rank = std::max<uint32_t>(eng->bundle.ane_probe_rank, 256U);
        eng->ane = strata_ane_probe_init(eng->ane_d_in, ane_rank, eng->ane_num_experts, eng->num_ane_windows);
        if (!eng->ane) {
            strata_infer_destroy(eng);
            return nullptr;
        }
        (void)strata_ane_probe_verify_vs_sme2(eng->ane, &eng->ane_sqnr_db, &eng->ane_plan_ok);

        /* 4. Pre-allocate & Pre-fault All KV Caches and Scratch Buffers (Zero Runtime Allocations!) */
        eng->max_seq_len = std::max<uint32_t>(eng->bundle.max_seq_len, 512U);
        eng->q_dim       = eng->bundle.n_heads * eng->bundle.head_dim;
        eng->kv_dim      = eng->bundle.n_kv_heads * eng->bundle.head_dim;

        const size_t kv_total = static_cast<size_t>(L) * eng->max_seq_len * eng->kv_dim;
        eng->k_cache.assign(kv_total, 0.0f);
        eng->v_cache.assign(kv_total, 0.0f);
        eng->seq_tokens.assign(eng->max_seq_len, 0U);

        const uint32_t max_act_dim = std::max<uint32_t>(D, eng->q_dim);
        eng->h_cur.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(D), 0.0f);
        eng->h_pre_norm.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(D), 0.0f);
        eng->h_last.assign(D, 0.0f);
        eng->x_fp16.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(max_act_dim), 0U);
        eng->q_f32.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(eng->q_dim), 0.0f);
        eng->k_f32.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(eng->kv_dim), 0.0f);
        eng->v_f32.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(eng->kv_dim), 0.0f);
        eng->attn_out_f32.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(eng->q_dim), 0.0f);
        eng->attn_out_fp16.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(eng->q_dim), 0U);
        eng->attn_scores.assign(eng->max_seq_len, 0.0f);
        eng->o_f32.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(D), 0.0f);
        eng->moe_out_f32.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(D), 0.0f);
        eng->v_proj_fp16.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(max_svd_alg), 0U);
        eng->router_logits.assign(E, 0.0f);
        eng->router_probs.assign(E, 0.0f);
        eng->layer_slot_map.resize(E);

        const uint32_t n_heads_ng = std::max<uint32_t>(cfg.ngram_heads_per_tok, 1U);
        eng->ngram_hashes.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(n_heads_ng), 0ULL);
        eng->ngram_fp16_out.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(D), 0U);
        eng->ane_in_fp16.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(eng->ane_d_in), 0U);
        eng->ane_scores.assign(4ULL * eng->ane_num_experts, 0.0f);

        eng->logits_BxV.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(V), 0.0f);
        eng->target_probs_BxV.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(V), 0.0f);
        eng->draft_probs_KxV.assign(STRATA_MAX_SPEC_BATCH * static_cast<size_t>(V), 0.0f);
        eng->mtp_h_draft.assign(D, 0.0f);
        eng->mtp_prev_h.assign(D, 0.0f);
        eng->mtp_next_h.assign(D, 0.0f);
        eng->mtp_norm_fp16.assign(D, 0U);
        eng->mtp_short_logits.assign(S, 0.0f);
        eng->mtp_scaled_logits.assign(S, 0.0f);
        eng->prompt_buf.assign(eng->max_seq_len, 0U);

        /* 5. One-Time Pipeline Warmup (opens MTLIOFileHandle, faults Metal/GCD/ANE dylib pages -> Steady MajFlt == 0) */
        if (eng->bundle.num_cold_extents > 0U && eng->bundle.cold_strata_path[0] != '\0') {
            uint64_t off0 = eng->bundle.cold_extents[0].offset_bytes;
            size_t   sz0  = static_cast<size_t>(eng->bundle.cold_extents[0].aligned_bytes);
            uint32_t sl0  = 0U;
            uint32_t ep0  = ++eng->next_epoch;
            (void)strata_hw_ring_stream_extents(
                eng->hw, eng->bundle.cold_strata_path, &off0, &sz0, &sl0, &ep0, 1U, true, nullptr);
        }
        {
            uint32_t warm_tok = eng->bundle.bos_token_id;
            (void)strata_infer_forward_tokens(eng, &warm_tok, 0U, 1U, STRATA_MODE_A_STATIONARY, nullptr);
        }

        struct rusage ru1{};
        getrusage(RUSAGE_SELF, &ru1);
        eng->init_majflt = ru1.ru_majflt - ru0.ru_majflt;

        strata_infer_reset(eng);
        return eng;
    }
}

void strata_infer_destroy(StrataInferEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        eng->drain_in_flight_ring_slots();
        if (eng->ngram_fd >= 0) {
            ::close(eng->ngram_fd);
            eng->ngram_fd = -1;
        }
        if (eng->ane) {
            strata_ane_probe_destroy(eng->ane);
            eng->ane = nullptr;
        }
        if (eng->hw) {
            strata_hw_destroy(eng->hw);
            eng->hw = nullptr;
        }
        if (eng->gpu_weights) {
            strata_gpu_weights_free(eng->gpu_weights);
            eng->gpu_weights = nullptr;
        }
        if (eng->gpu) {
            strata_gpu_destroy(eng->gpu);
            eng->gpu = nullptr;
        }
        if (eng->bundle_loaded) {
            strata_bundle_free(&eng->bundle);
            eng->bundle_loaded = false;
        }
        delete eng;
    }
}

const StrataCheckpointBundle* strata_infer_get_bundle(const StrataInferEngine* eng) {
    return (eng && eng->bundle_loaded) ? &eng->bundle : nullptr;
}

int strata_infer_forward_tokens(
    StrataInferEngine* eng,
    const uint32_t* tokens,
    uint32_t pos_offset,
    uint32_t batch_B,
    StrataExecMode mode,
    float* logits_out
) {
    if (!eng || !eng->bundle_loaded || !tokens || batch_B == 0U || batch_B > STRATA_MAX_SPEC_BATCH) {
        return -1;
    }
    if (pos_offset + batch_B > eng->max_seq_len) {
        return -2;
    }

    const StrataCheckpointBundle& bdl = eng->bundle;
    const StrataModelConfig& cfg      = bdl.cfg;
    const uint32_t D  = cfg.d_model;
    const uint32_t L  = cfg.num_layers;
    const uint32_t E  = cfg.num_experts;
    const uint32_t K  = std::min<uint32_t>(std::max<uint32_t>(cfg.top_k, 1U), STRATA_MAX_TOP_K);
    const uint32_t V  = cfg.vocab_size;

    void* ring_mtl_buf           = strata_hw_ring_get_mtl_buffer(eng->hw);
    const uint32_t* ring_ready   = strata_hw_ring_get_ready_table(eng->hw);
    const bool has_cold_container = (bdl.num_cold_extents > 0U && bdl.cold_strata_path[0] != '\0');

    /* 0. Record causal token history & lookup Token Embeddings [B, d_model] */
    for (uint32_t t = 0U; t < batch_B; ++t) {
        const uint32_t tid = tokens[t] % V;
        eng->seq_tokens[pos_offset + t] = tid;
        const uint16_t* emb_row = bdl.tok_embed_f16 + static_cast<size_t>(tid) * D;
        float* dst_row          = eng->h_cur.data() + static_cast<size_t>(t) * D;
        for (uint32_t d = 0U; d < D; ++d) {
            dst_row[d] = fp16_to_f32(emb_row[d]);
        }
    }

    /* ========================================================================
     * Layer-by-Layer Heterogeneous Tri-Engine Execution
     * ======================================================================== */
    for (uint32_t l = 0U; l < L; ++l) {
        const StrataLayerWeights& lw = bdl.layers[l];
        const uint32_t n_heads       = lw.n_heads;
        const uint32_t n_kv_heads    = lw.n_kv_heads;
        const uint32_t head_dim      = lw.head_dim;
        const uint32_t q_dim         = n_heads * head_dim;
        const uint32_t kv_dim        = n_kv_heads * head_dim;
        const uint32_t kv_group      = std::max<uint32_t>(1U, n_heads / std::max<uint32_t>(1U, n_kv_heads));

        /* --------------------------------------------------------------------
         * Phase 1 (`L == ngram_layer_idx`): Isolated 4 KiB F_NOCACHE Gather +
         * 3x ARMv9.2 SME2 (`LUTI4`/`ZT0`/`FMOPA`) Context-Gated Residual
         * -------------------------------------------------------------------- */
        if (cfg.has_ngram_table && cfg.ngram_heads_per_tok > 0U && l == bdl.ngram_layer_idx) {
            const uint32_t H = cfg.ngram_heads_per_tok;
            for (uint32_t t = 0U; t < batch_B; ++t) {
                const uint32_t abs_p = pos_offset + t;
                const uint32_t tok0  = eng->seq_tokens[abs_p];
                const uint32_t tok1  = (abs_p >= 1U) ? eng->seq_tokens[abs_p - 1U] : bdl.bos_token_id;
                const uint32_t tok2  = (abs_p >= 2U) ? eng->seq_tokens[abs_p - 2U] : bdl.bos_token_id;
                for (uint32_t h = 0U; h < H; ++h) {
                    /* Even heads: rolling bigram hash; Odd heads: rolling trigram hash */
                    const uint32_t tri = ((h & 1U) != 0U) ? tok2 : 0x5A5A5A5AU;
                    eng->ngram_hashes[static_cast<size_t>(t) * H + h] =
                        hash_ngram_tuple(tok0, tok1, tri, h);
                }
            }

            StrataSme2IoMetrics sme2_m{};
            int ng_rc = strata_hw_exec_ngram_phase1(
                eng->hw,
                eng->ngram_fd,
                bdl.ngram_file_bytes,
                eng->ngram_hashes.data(),
                batch_B,
                H,
                D,
                eng->ngram_fp16_out.data(),
                &sme2_m);
            if (ng_rc != 0) return -3;

            const size_t total_elem = static_cast<size_t>(batch_B) * D;
            for (size_t i = 0U; i < total_elem; ++i) {
                eng->h_cur[i] += fp16_to_f32(eng->ngram_fp16_out[i]);
            }

            eng->tel.sme2_phase1_calls++;
            eng->tel.sme2_io_us           += sme2_m.phase1_io_us;
            eng->tel.sme2_compute_us      += sme2_m.sme2_compute_us;
            eng->tel.ngram_lfu_hits       += sme2_m.lfu_cache_hits;
            eng->tel.ngram_nvme_ios       += sme2_m.nvme_pread_ios;
            eng->tel.hardware_sme2_active  = sme2_m.hardware_sme2_active;
        }

        /* --------------------------------------------------------------------
         * Phase 2a (`16-Core ANE` Lookahead Prefetch): At window boundaries
         * (strictly AFTER Phase 1 so 4K random IOs and 256K ring DMAs never collide!),
         * predict upcoming dynamic layers on physical 16c ANE & stream async DMA!
         * -------------------------------------------------------------------- */
        const bool is_ane_window = cfg.has_ngram_table
            ? ((l == bdl.ngram_layer_idx) ||
               (l > bdl.ngram_layer_idx && ((l - bdl.ngram_layer_idx) % STRATA_ANE_WINDOW_STRIDE) == 0U))
            : ((l % STRATA_ANE_WINDOW_STRIDE) == 0U);

        if (is_ane_window && eng->ane != nullptr) {
            for (uint32_t t = 0U; t < batch_B; ++t) {
                const float* src_h  = eng->h_cur.data() + static_cast<size_t>(t) * D;
                uint16_t* dst_ane   = eng->ane_in_fp16.data() + static_cast<size_t>(t) * eng->ane_d_in;
                for (uint32_t c = 0U; c < eng->ane_d_in; ++c) {
                    dst_ane[c] = f32_to_fp16(src_h[c % D]);
                }
            }

            const uint32_t win_idx = (l / STRATA_ANE_WINDOW_STRIDE) % eng->num_ane_windows;
            double ane_us   = 0.0;
            bool   used_ane = false;
            int ane_rc = strata_ane_probe_predict_window(
                eng->ane,
                win_idx,
                eng->ane_in_fp16.data(),
                batch_B,
                eng->ane_scores.data(),
                &ane_us,
                &used_ane);
            if (ane_rc == 0) {
                if (eng->tel.ane_probe_calls == 0ULL) {
                    eng->tel.ran_on_ane = used_ane;
                } else {
                    eng->tel.ran_on_ane = eng->tel.ran_on_ane && used_ane;
                }
                eng->tel.ane_probe_calls++;
                eng->tel.ane_probe_us += ane_us;

                /* Rank cold experts across upcoming dynamic layers & issue async 256 KiB DMA */
                if (has_cold_container) {
                    uint64_t dma_offsets[16];
                    size_t   dma_sizes[16];
                    uint32_t dma_slots[16];
                    uint32_t dma_epochs[16];
                    uint32_t num_dma = 0U;

                    const uint32_t dyn_start = std::max<uint32_t>(l + 1U, cfg.pinned_shallow_layers);
                    const uint32_t per_layer_cap = (mode == STRATA_MODE_B_ZERO_STALL)
                        ? std::max<uint32_t>(cfg.mode_b_cold_cap, 2U)
                        : 2U;

                    for (uint32_t h = 0U; h < 4U && num_dma < 16U; ++h) {
                        const uint32_t tl = dyn_start + h;
                        if (tl >= L) break;
                        const StrataLayerWeights& tlw = bdl.layers[tl];
                        const float* h_scores =
                            eng->ane_scores.data() + static_cast<size_t>(h) * eng->ane_num_experts;

                        struct Cand {
                            uint16_t e;
                            float    sc;
                        };
                        Cand cands[1024];
                        uint32_t nc = 0U;
                        const uint32_t e_lim = std::min<uint32_t>(E, 1024U);
                        for (uint32_t e = 0U; e < e_lim; ++e) {
                            if (!tlw.is_uma_hot[e] && tlw.cold_extent_idx[e] >= 0) {
                                cands[nc++] = Cand{static_cast<uint16_t>(e), h_scores[e]};
                            }
                        }
                        std::sort(cands, cands + nc, [](const Cand& a, const Cand& b) {
                            if (a.sc != b.sc) return a.sc > b.sc;
                            return a.e < b.e;
                        });

                        const uint32_t take = std::min<uint32_t>(nc, per_layer_cap);
                        for (uint32_t c = 0U; c < take && num_dma < 16U; ++c) {
                            const int32_t ext_idx = tlw.cold_extent_idx[cands[c].e];
                            if (ext_idx < 0 || static_cast<uint32_t>(ext_idx) >= bdl.num_cold_extents) {
                                continue;
                            }
                            int existing = eng->find_ring_slot_for_extent(ext_idx);
                            if (existing >= 0) {
                                eng->ring_slots[static_cast<size_t>(existing)].lru_tick = ++eng->lru_clock;
                                continue;
                            }
                            /* Never wait during async ANE lookahead prefetch! */
                            int slot = eng->alloc_ring_slot(/*allow_wait_in_flight=*/false, nullptr);
                            if (slot < 0) break;

                            const StrataExtentRecord& rec = bdl.cold_extents[ext_idx];
                            const uint32_t ep = ++eng->next_epoch;
                            eng->ring_slots[static_cast<size_t>(slot)].extent_idx     = ext_idx;
                            eng->ring_slots[static_cast<size_t>(slot)].assigned_epoch = ep;
                            eng->ring_slots[static_cast<size_t>(slot)].lru_tick       = ++eng->lru_clock;

                            dma_offsets[num_dma] = rec.offset_bytes;
                            dma_sizes[num_dma]   = static_cast<size_t>(rec.aligned_bytes);
                            dma_slots[num_dma]   = static_cast<uint32_t>(slot);
                            dma_epochs[num_dma]  = ep;
                            ++num_dma;
                        }
                    }

                    if (num_dma > 0U) {
                        StrataRingIoMetrics ring_m{};
                        int d_rc = strata_hw_ring_stream_extents(
                            eng->hw,
                            bdl.cold_strata_path,
                            dma_offsets,
                            dma_sizes,
                            dma_slots,
                            dma_epochs,
                            num_dma,
                            /*wait_completion=*/false,
                            &ring_m);
                        if (d_rc == 0) {
                            eng->tel.ring_dma_waves++;
                            eng->tel.ring_extents_streamed += ring_m.extents_streamed;
                            eng->tel.ring_bytes_streamed   += ring_m.bytes_streamed;
                            eng->tel.ring_dma_us           += ring_m.dma_wall_us;
                        }
                    }
                }
            }
        }

        /* --------------------------------------------------------------------
         * Phase 2b (`40-Core Metal 4 NAX GPU` Layer Execution):
         *   1) Pre-Attn RMSNorm -> Q, K, V Spine Tile-rANS GEMMs
         *   2) RoPE + Causal GQA Attention against KV Cache
         *   3) O Projection Spine Tile-rANS GEMM + Residual
         *   4) Post-Attn RMSNorm -> Router Softmax Top-k + Expert Planner
         *   5) Mode A Stationary Wait vs Mode B Zero-Stall -> Slot-Indexed MoE + Residual
         * -------------------------------------------------------------------- */
        apply_rmsnorm_to_fp16(
            eng->h_cur.data(), lw.attn_norm_f16, batch_B, D, bdl.rms_norm_eps, eng->x_fp16.data());

        {
            const uint64_t t_sp0 = mach_absolute_time();
            if (strata_gpu_exec_spine_rans(
                    eng->gpu, lw.q_rans_blob, lw.q_rans_bytes,
                    eng->x_fp16.data(), batch_B, D, q_dim, eng->q_f32.data()) != 0) return -4;
            if (strata_gpu_exec_spine_rans(
                    eng->gpu, lw.k_rans_blob, lw.k_rans_bytes,
                    eng->x_fp16.data(), batch_B, D, kv_dim, eng->k_f32.data()) != 0) return -4;
            if (strata_gpu_exec_spine_rans(
                    eng->gpu, lw.v_rans_blob, lw.v_rans_bytes,
                    eng->x_fp16.data(), batch_B, D, kv_dim, eng->v_f32.data()) != 0) return -4;
            const uint64_t t_sp1 = mach_absolute_time();
            eng->tel.gpu_spine_calls += 3ULL;
            eng->tel.gpu_spine_us    += mach_ticks_to_us(t_sp1 - t_sp0);
        }

        /* RoPE + KV Cache Write + Strictly Causal GQA Attention */
        float* k_layer_cache = eng->k_cache.data() + static_cast<size_t>(l) * eng->max_seq_len * kv_dim;
        float* v_layer_cache = eng->v_cache.data() + static_cast<size_t>(l) * eng->max_seq_len * kv_dim;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const float rope_theta = (bdl.rope_theta > 0.0f) ? bdl.rope_theta : 10000.0f;

        for (uint32_t t = 0U; t < batch_B; ++t) {
            const uint32_t pos = pos_offset + t;
            float* q_tok = eng->q_f32.data() + static_cast<size_t>(t) * q_dim;
            float* k_tok = eng->k_f32.data() + static_cast<size_t>(t) * kv_dim;
            const float* v_tok = eng->v_f32.data() + static_cast<size_t>(t) * kv_dim;

            /* Apply Rotary Position Embedding (RoPE) to Q and K */
            for (uint32_t p = 0U; p + 1U < head_dim; p += 2U) {
                const float freq = 1.0f / std::pow(rope_theta, static_cast<float>(p) / static_cast<float>(head_dim));
                const float ang  = static_cast<float>(pos) * freq;
                const float cs   = std::cos(ang);
                const float sn   = std::sin(ang);

                for (uint32_t h = 0U; h < n_heads; ++h) {
                    float* qh = q_tok + static_cast<size_t>(h) * head_dim;
                    const float q0 = qh[p];
                    const float q1 = qh[p + 1U];
                    qh[p]          = q0 * cs - q1 * sn;
                    qh[p + 1U]     = q0 * sn + q1 * cs;
                }
                for (uint32_t h = 0U; h < n_kv_heads; ++h) {
                    float* kh = k_tok + static_cast<size_t>(h) * head_dim;
                    const float k0 = kh[p];
                    const float k1 = kh[p + 1U];
                    kh[p]          = k0 * cs - k1 * sn;
                    kh[p + 1U]     = k0 * sn + k1 * cs;
                }
            }

            /* Commit rotated K and V to stateful KV Cache at exact sequence position `pos` */
            std::memcpy(k_layer_cache + static_cast<size_t>(pos) * kv_dim, k_tok, kv_dim * sizeof(float));
            std::memcpy(v_layer_cache + static_cast<size_t>(pos) * kv_dim, v_tok, kv_dim * sizeof(float));

            /* Causal GQA over s = 0 .. pos */
            float* out_tok = eng->attn_out_f32.data() + static_cast<size_t>(t) * q_dim;
            const uint32_t seq_len_t = pos + 1U;

            for (uint32_t qh = 0U; qh < n_heads; ++qh) {
                const uint32_t kvh   = qh / kv_group;
                const float* q_vec   = q_tok + static_cast<size_t>(qh) * head_dim;
                float*       out_vec = out_tok + static_cast<size_t>(qh) * head_dim;

                float max_sc = -1e30f;
                for (uint32_t s = 0U; s < seq_len_t; ++s) {
                    const float* k_vec = k_layer_cache + static_cast<size_t>(s) * kv_dim +
                                         static_cast<size_t>(kvh) * head_dim;
                    double dot = 0.0;
                    for (uint32_t d = 0U; d < head_dim; ++d) {
                        dot += static_cast<double>(q_vec[d]) * static_cast<double>(k_vec[d]);
                    }
                    const float sc = static_cast<float>(dot) * attn_scale;
                    eng->attn_scores[s] = sc;
                    if (sc > max_sc) max_sc = sc;
                }

                double sum_w = 0.0;
                for (uint32_t s = 0U; s < seq_len_t; ++s) {
                    const float w = std::exp(eng->attn_scores[s] - max_sc);
                    eng->attn_scores[s] = w;
                    sum_w += static_cast<double>(w);
                }
                const float inv_w = (sum_w > 0.0) ? static_cast<float>(1.0 / sum_w) : 0.0f;

                for (uint32_t d = 0U; d < head_dim; ++d) out_vec[d] = 0.0f;
                for (uint32_t s = 0U; s < seq_len_t; ++s) {
                    const float w = eng->attn_scores[s] * inv_w;
                    const float* v_vec = v_layer_cache + static_cast<size_t>(s) * kv_dim +
                                         static_cast<size_t>(kvh) * head_dim;
                    for (uint32_t d = 0U; d < head_dim; ++d) {
                        out_vec[d] += w * v_vec[d];
                    }
                }
            }
        }

        const size_t q_total = static_cast<size_t>(batch_B) * q_dim;
        for (size_t i = 0U; i < q_total; ++i) {
            eng->attn_out_fp16[i] = f32_to_fp16(eng->attn_out_f32[i]);
        }

        /* O Projection via Spine Tile-rANS + Residual */
        {
            const uint64_t t_sp0 = mach_absolute_time();
            if (strata_gpu_exec_spine_rans(
                    eng->gpu, lw.o_rans_blob, lw.o_rans_bytes,
                    eng->attn_out_fp16.data(), batch_B, q_dim, D, eng->o_f32.data()) != 0) return -5;
            const uint64_t t_sp1 = mach_absolute_time();
            eng->tel.gpu_spine_calls += 1ULL;
            eng->tel.gpu_spine_us    += mach_ticks_to_us(t_sp1 - t_sp0);
        }

        const size_t bd_total = static_cast<size_t>(batch_B) * D;
        for (size_t i = 0U; i < bd_total; ++i) {
            eng->h_cur[i] += eng->o_f32[i];
        }

        /* Post-Attention RMSNorm -> MoE Router Softmax Top-k */
        apply_rmsnorm_to_fp16(
            eng->h_cur.data(), lw.ffn_norm_f16, batch_B, D, bdl.rms_norm_eps, eng->x_fp16.data());

        uint16_t topk_ids[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K]   = {0U};
        float    topk_gates[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K] = {0.0f};

        for (uint32_t t = 0U; t < batch_B; ++t) {
            const uint16_t* xt = eng->x_fp16.data() + static_cast<size_t>(t) * D;
            float max_rl = -1e30f;
            for (uint32_t e = 0U; e < E; ++e) {
                const uint16_t* rw = lw.router_w_f16 + static_cast<size_t>(e) * D;
                double dot = 0.0;
                for (uint32_t d = 0U; d < D; ++d) {
                    dot += static_cast<double>(fp16_to_f32(xt[d])) * static_cast<double>(fp16_to_f32(rw[d]));
                }
                const float rl = static_cast<float>(dot);
                eng->router_logits[e] = rl;
                if (rl > max_rl) max_rl = rl;
            }

            double sum_rp = 0.0;
            for (uint32_t e = 0U; e < E; ++e) {
                const float p = std::exp(eng->router_logits[e] - max_rl);
                eng->router_probs[e] = p;
                sum_rp += static_cast<double>(p);
            }
            const float inv_rp = (sum_rp > 0.0) ? static_cast<float>(1.0 / sum_rp) : (1.0f / static_cast<float>(E));
            for (uint32_t e = 0U; e < E; ++e) {
                eng->router_probs[e] *= inv_rp;
            }

            /* Select Top-k deterministically (ties broken by lower expert index) & renormalize */
            double topk_sum = 0.0;
            bool picked[1024] = {false};
            for (uint32_t j = 0U; j < K; ++j) {
                uint32_t best_e = 0U;
                float    best_p = -1.0f;
                for (uint32_t e = 0U; e < E; ++e) {
                    if (e < 1024U && picked[e]) continue;
                    if (eng->router_probs[e] > best_p) {
                        best_p = eng->router_probs[e];
                        best_e = e;
                    }
                }
                if (best_e < 1024U) picked[best_e] = true;
                topk_ids[t * K + j]   = static_cast<uint16_t>(best_e);
                topk_gates[t * K + j] = best_p;
                topk_sum += static_cast<double>(best_p);
            }
            const float inv_tk = (topk_sum > 0.0) ? static_cast<float>(1.0 / topk_sum) : (1.0f / static_cast<float>(K));
            for (uint32_t j = 0U; j < K; ++j) {
                topk_gates[t * K + j] *= inv_tk;
            }
        }

        StrataExpertPlan plan{};
        if (strata_plan_layer_experts(
                mode,
                batch_B,
                K,
                topk_ids,
                topk_gates,
                lw.is_uma_hot,
                cfg.mode_a_tau_tok,
                cfg.mode_b_cold_cap,
                &plan) != 0) {
            return -6;
        }
        eng->tel.moe_hot_hits += plan.num_uma_hot_hits;

        /* Build current layer SLC slot map from default_slot_map + active ring cache */
        std::memcpy(eng->layer_slot_map.data(), lw.default_slot_map, static_cast<size_t>(E) * sizeof(StrataSlotEntry));
        bool pinned_for_layer[64] = {false};
        for (uint32_t e = 0U; e < E; ++e) {
            if (!lw.is_uma_hot[e] && lw.cold_extent_idx[e] >= 0) {
                int s = eng->find_ring_slot_for_extent(lw.cold_extent_idx[e]);
                if (s >= 0) {
                    eng->layer_slot_map[e].ring_slot_idx  = static_cast<uint16_t>(s);
                    eng->layer_slot_map[e].expected_epoch = eng->ring_slots[static_cast<size_t>(s)].assigned_epoch;
                }
            }
        }
        for (uint32_t c = 0U; c < plan.num_cold_fetches; ++c) {
            const uint16_t ce = plan.cold_fetch_experts[c];
            if (ce < E && eng->layer_slot_map[ce].ring_slot_idx != STRATA_SLOT_INVALID &&
                eng->layer_slot_map[ce].ring_slot_idx < 64U) {
                pinned_for_layer[eng->layer_slot_map[ce].ring_slot_idx] = true;
            }
        }

        if (mode == STRATA_MODE_A_STATIONARY && has_cold_container && plan.num_cold_fetches > 0U) {
            /* Mode A: Guarantee 100% prefix-stationary exactness by streaming any ANE-missed
             * cold experts and waiting via strata_hw_ring_wait_slot until ready! */
            uint64_t miss_offsets[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K];
            size_t   miss_sizes[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K];
            uint32_t miss_slots[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K];
            uint32_t miss_epochs[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K];
            uint32_t num_miss = 0U;

            for (uint32_t c = 0U; c < plan.num_cold_fetches; ++c) {
                const uint16_t ce = plan.cold_fetch_experts[c];
                if (ce >= E || lw.is_uma_hot[ce] || lw.cold_extent_idx[ce] < 0) continue;
                const int32_t ext_idx = lw.cold_extent_idx[ce];
                if (static_cast<uint32_t>(ext_idx) >= bdl.num_cold_extents) continue;

                int s = eng->find_ring_slot_for_extent(ext_idx);
                if (s < 0) {
                    s = eng->alloc_ring_slot(/*allow_wait_in_flight=*/true, pinned_for_layer);
                    if (s >= 0) {
                        const StrataExtentRecord& rec = bdl.cold_extents[ext_idx];
                        const uint32_t ep = ++eng->next_epoch;
                        eng->ring_slots[static_cast<size_t>(s)].extent_idx     = ext_idx;
                        eng->ring_slots[static_cast<size_t>(s)].assigned_epoch = ep;
                        eng->ring_slots[static_cast<size_t>(s)].lru_tick       = ++eng->lru_clock;
                        if (s < 64) pinned_for_layer[s] = true;

                        eng->layer_slot_map[ce].ring_slot_idx  = static_cast<uint16_t>(s);
                        eng->layer_slot_map[ce].expected_epoch = ep;

                        miss_offsets[num_miss] = rec.offset_bytes;
                        miss_sizes[num_miss]   = static_cast<size_t>(rec.aligned_bytes);
                        miss_slots[num_miss]   = static_cast<uint32_t>(s);
                        miss_epochs[num_miss]  = ep;
                        ++num_miss;
                    }
                } else {
                    eng->ring_slots[static_cast<size_t>(s)].lru_tick = ++eng->lru_clock;
                    eng->layer_slot_map[ce].ring_slot_idx  = static_cast<uint16_t>(s);
                    eng->layer_slot_map[ce].expected_epoch = eng->ring_slots[static_cast<size_t>(s)].assigned_epoch;
                }
            }

            if (num_miss > 0U) {
                StrataRingIoMetrics ring_m{};
                if (strata_hw_ring_stream_extents(
                        eng->hw,
                        bdl.cold_strata_path,
                        miss_offsets,
                        miss_sizes,
                        miss_slots,
                        miss_epochs,
                        num_miss,
                        /*wait_completion=*/false,
                        &ring_m) == 0) {
                    eng->tel.ring_dma_waves++;
                    eng->tel.ring_extents_streamed += ring_m.extents_streamed;
                    eng->tel.ring_bytes_streamed   += ring_m.bytes_streamed;
                    eng->tel.ring_dma_us           += ring_m.dma_wall_us;
                }
            }

            /* Wait for all Mode A planned cold experts so target distribution is 100% invariant! */
            for (uint32_t c = 0U; c < plan.num_cold_fetches; ++c) {
                const uint16_t ce = plan.cold_fetch_experts[c];
                if (ce >= E) continue;
                const uint16_t rslot = eng->layer_slot_map[ce].ring_slot_idx;
                const uint32_t rep   = eng->layer_slot_map[ce].expected_epoch;
                if (rslot != STRATA_SLOT_INVALID && rep > 0U) {
                    double w_us = 0.0;
                    (void)strata_hw_ring_wait_slot(eng->hw, rslot, rep, &w_us);
                    eng->tel.ring_wait_calls++;
                    eng->tel.ring_wait_us += w_us;
                }
            }
        }

        /* Mode B (or Regime II fallback): Compute W6A16 shared subspace projection `x_fp16 @ V_shared^T` */
        const uint16_t* v_proj_arg = nullptr;
        if (cfg.svd_rank > 0U && lw.svd_rank_aligned > 0U && lw.svd_v_shared_f16 != nullptr) {
            const uint32_t r_alg = lw.svd_rank_aligned;
            const uint32_t r_eff = std::min<uint32_t>(cfg.svd_rank, r_alg);
            for (uint32_t b = 0U; b < batch_B; ++b) {
                const uint16_t* xb = eng->x_fp16.data() + static_cast<size_t>(b) * D;
                uint16_t* vp_dst   = eng->v_proj_fp16.data() + static_cast<size_t>(b) * r_alg;
                for (uint32_t k = 0U; k < r_eff; ++k) {
                    const uint16_t* vr = lw.svd_v_shared_f16 + static_cast<size_t>(k) * D;
                    float acc = 0.0f;
                    for (uint32_t d = 0U; d < D; ++d) {
                        acc += fp16_to_f32(xb[d]) * fp16_to_f32(vr[d]);
                    }
                    vp_dst[k] = f32_to_fp16(acc);
                }
                for (uint32_t k = r_eff; k < r_alg; ++k) {
                    vp_dst[k] = 0U;
                }
            }
            v_proj_arg = eng->v_proj_fp16.data();
        }

        uint32_t used_ring = 0U;
        uint32_t fb_cnt    = 0U;
        const uint64_t t_moe0 = mach_absolute_time();
        if (strata_gpu_exec_moe_layer(
                eng->gpu,
                eng->gpu_weights,
                l,
                mode,
                batch_B,
                eng->x_fp16.data(),
                eng->layer_slot_map.data(),
                topk_ids,
                topk_gates,
                v_proj_arg,
                ring_mtl_buf,
                ring_ready,
                eng->moe_out_f32.data(),
                &used_ring,
                &fb_cnt) != 0) {
            return -7;
        }
        const uint64_t t_moe1 = mach_absolute_time();

        eng->tel.gpu_moe_calls++;
        eng->tel.gpu_moe_us             += mach_ticks_to_us(t_moe1 - t_moe0);
        eng->tel.moe_ring_hits          += used_ring;
        eng->tel.moe_svd_qtip_fallbacks += fb_cnt;

        for (size_t i = 0U; i < bd_total; ++i) {
            eng->h_cur[i] += eng->moe_out_f32[i];
        }
    }

    /* ========================================================================
     * Final RMSNorm + LM Head Logits [batch_B, vocab_size]
     * ======================================================================== */
    std::memcpy(eng->h_pre_norm.data(), eng->h_cur.data(), static_cast<size_t>(batch_B) * D * sizeof(float));

    apply_rmsnorm_to_fp16(
        eng->h_cur.data(), bdl.final_norm_f16, batch_B, D, bdl.rms_norm_eps, eng->x_fp16.data());

    if (bdl.lm_head_rans_blob != nullptr && (V % 32U) == 0U && (D % 32U) == 0U) {
        const uint64_t t_lm0 = mach_absolute_time();
        if (strata_gpu_exec_spine_rans(
                eng->gpu, bdl.lm_head_rans_blob, bdl.lm_head_rans_bytes,
                eng->x_fp16.data(), batch_B, D, V, eng->logits_BxV.data()) != 0) {
            return -8;
        }
        const uint64_t t_lm1 = mach_absolute_time();
        eng->tel.gpu_spine_calls += 1ULL;
        eng->tel.gpu_spine_us    += mach_ticks_to_us(t_lm1 - t_lm0);
    } else {
        for (uint32_t b = 0U; b < batch_B; ++b) {
            const uint16_t* xb = eng->x_fp16.data() + static_cast<size_t>(b) * D;
            float* out_l       = eng->logits_BxV.data() + static_cast<size_t>(b) * V;
            for (uint32_t v = 0U; v < V; ++v) {
                const uint16_t* wr = bdl.lm_head_f16 + static_cast<size_t>(v) * D;
                double acc = 0.0;
                for (uint32_t d = 0U; d < D; ++d) {
                    acc += static_cast<double>(fp16_to_f32(xb[d])) * static_cast<double>(fp16_to_f32(wr[d]));
                }
                out_l[v] = static_cast<float>(acc);
            }
        }
    }

    const size_t total_logits = static_cast<size_t>(batch_B) * V;
    for (size_t i = 0U; i < total_logits; ++i) {
        if (std::isnan(eng->logits_BxV[i]) || std::isinf(eng->logits_BxV[i])) {
            eng->tel.logits_nan_inf_count++;
        }
    }

    if (logits_out != nullptr) {
        std::memcpy(logits_out, eng->logits_BxV.data(), total_logits * sizeof(float));
    }
    return 0;
}

static void exec_mtp_draft_step(
    StrataInferEngine* eng,
    const float* h_prev_f32,
    uint32_t cur_tok,
    bool greedy,
    float temperature,
    uint64_t* rng,
    uint32_t* draft_tok_out,
    float* full_draft_probs_out,
    float* h_draft_next_out
) {
    const StrataCheckpointBundle& bdl = eng->bundle;
    const uint32_t D = bdl.cfg.d_model;
    const uint32_t S = bdl.cfg.shortlist_size;
    const uint32_t V = bdl.cfg.vocab_size;

    const uint16_t* emb = bdl.tok_embed_f16 + static_cast<size_t>(cur_tok % V) * D;

    /* 1. State fusion `h_draft [D] = mtp_proj [D, 2*D] @ [h_prev; tok_embed[cur_tok]]` */
    for (uint32_t r = 0U; r < D; ++r) {
        const uint16_t* pr = bdl.mtp_proj_f16 + static_cast<size_t>(r) * (2U * D);
        double acc = 0.0;
        for (uint32_t c = 0U; c < D; ++c) {
            acc += static_cast<double>(h_prev_f32[c]) * static_cast<double>(fp16_to_f32(pr[c]));
        }
        for (uint32_t c = 0U; c < D; ++c) {
            acc += static_cast<double>(fp16_to_f32(emb[c])) * static_cast<double>(fp16_to_f32(pr[D + c]));
        }
        eng->mtp_h_draft[r] = static_cast<float>(acc);
    }
    std::memcpy(h_draft_next_out, eng->mtp_h_draft.data(), D * sizeof(float));

    /* 2. MTP RMSNorm -> Shortlist Head (`strata_gpu_exec_spine_rans` or FP16) */
    apply_rmsnorm_to_fp16(
        eng->mtp_h_draft.data(), bdl.mtp_norm_f16, 1U, D, bdl.rms_norm_eps, eng->mtp_norm_fp16.data());

    if (bdl.mtp_head_rans_blob != nullptr && (S % 32U) == 0U && (D % 32U) == 0U) {
        const uint64_t t0 = mach_absolute_time();
        (void)strata_gpu_exec_spine_rans(
            eng->gpu, bdl.mtp_head_rans_blob, bdl.mtp_head_rans_bytes,
            eng->mtp_norm_fp16.data(), 1U, D, S, eng->mtp_short_logits.data());
        const uint64_t t1 = mach_absolute_time();
        eng->tel.gpu_spine_calls += 1ULL;
        eng->tel.gpu_spine_us    += mach_ticks_to_us(t1 - t0);
    } else {
        for (uint32_t s = 0U; s < S; ++s) {
            const uint16_t* wr = bdl.mtp_head_f16 + static_cast<size_t>(s) * D;
            double acc = 0.0;
            for (uint32_t d = 0U; d < D; ++d) {
                acc += static_cast<double>(fp16_to_f32(eng->mtp_norm_fp16[d])) *
                       static_cast<double>(fp16_to_f32(wr[d]));
            }
            eng->mtp_short_logits[s] = static_cast<float>(acc);
        }
    }

    /* 3. Convert shortlist logits -> full-vocab FR-Spec draft probabilities (`strata_frspec_shortlist_probs`) */
    if (greedy || temperature <= 1e-6f) {
        const uint32_t best_s = argmax_f32(eng->mtp_short_logits.data(), S);
        for (uint32_t s = 0U; s < S; ++s) {
            eng->mtp_scaled_logits[s] = (s == best_s) ? 30.0f : -30.0f;
        }
        strata_frspec_shortlist_probs(
            eng->mtp_scaled_logits.data(),
            bdl.shortlist_vocab_ids,
            S,
            V,
            full_draft_probs_out);
        *draft_tok_out = bdl.shortlist_vocab_ids[best_s] % V;
    } else {
        const float inv_temp = 1.0f / temperature;
        for (uint32_t s = 0U; s < S; ++s) {
            eng->mtp_scaled_logits[s] = eng->mtp_short_logits[s] * inv_temp;
        }
        strata_frspec_shortlist_probs(
            eng->mtp_scaled_logits.data(),
            bdl.shortlist_vocab_ids,
            S,
            V,
            full_draft_probs_out);
        *draft_tok_out = sample_discrete_cdf(full_draft_probs_out, V, rng_u01(rng));
    }
}

int strata_infer_generate(
    StrataInferEngine* eng,
    const char* prompt_utf8,
    const StrataInferOptions* opts,
    uint32_t* out_token_ids,
    uint32_t max_out_tokens,
    uint32_t* num_generated_out,
    StrataTokenStreamFn stream_cb,
    void* stream_user_ctx,
    StrataInferTelemetry* telemetry_out
) {
    if (num_generated_out) *num_generated_out = 0U;
    if (!eng || !eng->bundle_loaded || !prompt_utf8) return -1;

    @autoreleasepool {
        StrataInferOptions cfg_opts = opts ? *opts : strata_infer_default_options();
        const uint32_t target_max_tokens = (out_token_ids != nullptr && max_out_tokens > 0U)
            ? std::min<uint32_t>(cfg_opts.max_tokens, max_out_tokens)
            : cfg_opts.max_tokens;
        const uint32_t draft_K = std::min<uint32_t>(cfg_opts.draft_k, STRATA_MAX_SPEC_BATCH - 1U);

        strata_infer_reset(eng);

        struct rusage ru_steady_start{};
        getrusage(RUSAGE_SELF, &ru_steady_start);

        const StrataCheckpointBundle& bdl = eng->bundle;
        const uint32_t D = bdl.cfg.d_model;
        const uint32_t V = bdl.cfg.vocab_size;

        /* 1. Tokenize Prompt */
        uint32_t num_prompt = 0U;
        const uint32_t max_prompt = eng->max_seq_len / 2U;
        if (strata_tokenize_prompt(&bdl, prompt_utf8, /*add_bos=*/true,
                                   eng->prompt_buf.data(), max_prompt, &num_prompt) != 0 ||
            num_prompt == 0U) {
            eng->prompt_buf[0] = bdl.bos_token_id;
            num_prompt = 1U;
        }
        eng->tel.prompt_tokens = num_prompt;

        /* 2. Prefill `prompt_buf[0 .. num_prompt - 2]` identically regardless of draft_K */
        const uint64_t t_prefill0 = mach_absolute_time();
        uint32_t cur_pos = 0U;
        if (num_prompt > 1U) {
            const uint32_t prefill_n = num_prompt - 1U;
            while (cur_pos < prefill_n) {
                const uint32_t chunk = std::min<uint32_t>(prefill_n - cur_pos, STRATA_MAX_SPEC_BATCH);
                int rc = strata_infer_forward_tokens(
                    eng, eng->prompt_buf.data() + cur_pos, cur_pos, chunk, cfg_opts.mode, nullptr);
                if (rc != 0) return rc;
                std::memcpy(eng->h_last.data(),
                            eng->h_pre_norm.data() + static_cast<size_t>(chunk - 1U) * D,
                            D * sizeof(float));
                cur_pos += chunk;
            }
        } else {
            const uint16_t* emb0 = bdl.tok_embed_f16 + static_cast<size_t>(eng->prompt_buf[0] % V) * D;
            for (uint32_t d = 0U; d < D; ++d) {
                eng->h_last[d] = fp16_to_f32(emb0[d]);
            }
        }
        const uint64_t t_prefill1 = mach_absolute_time();
        eng->tel.prefill_wall_ms = mach_ticks_to_us(t_prefill1 - t_prefill0) * 1e-3;

        /* 3. Autoregressive Decode Loop (`K = 0` single-token vs `K > 0` Speculative `B = K + 1`) */
        uint64_t rng = (cfg_opts.seed != 0ULL) ? cfg_opts.seed : 42ULL;
        rng ^= 0xA5A55A5A9E3779B9ULL;

        uint32_t cur_tok   = eng->prompt_buf[num_prompt - 1U];
        uint32_t generated = 0U;

        const uint64_t t_dec0 = mach_absolute_time();

        while (generated < target_max_tokens) {
            const uint32_t avail_ctx = (cur_pos + 1U < eng->max_seq_len)
                ? (eng->max_seq_len - 1U - cur_pos)
                : 0U;
            if (avail_ctx == 0U) break;

            const uint32_t eff_K = std::min<uint32_t>(draft_K, avail_ctx);
            const uint32_t B     = eff_K + 1U;

            uint32_t draft_tokens[STRATA_MAX_SPEC_BATCH] = {0U};
            uint32_t batch_tokens[STRATA_MAX_SPEC_BATCH] = {0U};
            batch_tokens[0] = cur_tok;

            if (eff_K > 0U) {
                std::memcpy(eng->mtp_prev_h.data(), eng->h_last.data(), D * sizeof(float));
                uint32_t d_tok = cur_tok;

                for (uint32_t m = 0U; m < eff_K; ++m) {
                    uint32_t proposed = 0U;
                    exec_mtp_draft_step(
                        eng,
                        eng->mtp_prev_h.data(),
                        d_tok,
                        cfg_opts.greedy,
                        cfg_opts.temperature,
                        &rng,
                        &proposed,
                        eng->draft_probs_KxV.data() + static_cast<size_t>(m) * V,
                        eng->mtp_next_h.data());
                    draft_tokens[m]      = proposed;
                    batch_tokens[1U + m] = proposed;
                    d_tok                = proposed;
                    std::memcpy(eng->mtp_prev_h.data(), eng->mtp_next_h.data(), D * sizeof(float));
                }
            }

            /* Execute parallel target forward pass over [cur_tok, draft_0 .. draft_{eff_K-1}] */
            int fwd_rc = strata_infer_forward_tokens(eng, batch_tokens, cur_pos, B, cfg_opts.mode, nullptr);
            if (fwd_rc != 0) return fwd_rc;

            eng->tel.decode_steps++;
            eng->tel.drafted_tokens_total += eff_K;

            for (uint32_t b = 0U; b < B; ++b) {
                logits_to_probs_f32(
                    eng->logits_BxV.data() + static_cast<size_t>(b) * V,
                    V,
                    cfg_opts.greedy,
                    cfg_opts.temperature,
                    eng->target_probs_BxV.data() + static_cast<size_t>(b) * V);
            }

            float rands[STRATA_MAX_SPEC_BATCH] = {0.0f};
            for (uint32_t r = 0U; r <= eff_K; ++r) {
                rands[r] = cfg_opts.greedy ? 0.5f : rng_u01(&rng);
            }

            StrataVerifyResult vr{};
            int ver_rc = strata_leviathan_verify_step(
                eff_K,
                V,
                (eff_K > 0U) ? draft_tokens : nullptr,
                (eff_K > 0U) ? eng->draft_probs_KxV.data() : nullptr,
                eng->target_probs_BxV.data(),
                rands,
                &vr);
            if (ver_rc != 0 || vr.accepted_count == 0U) return -9;

            const uint32_t A = vr.accepted_count;
            /* Count how many draft tokens matched before resample/bonus */
            uint32_t matched_drafts = 0U;
            for (uint32_t m = 0U; m < eff_K && m < A; ++m) {
                if (vr.emitted_tokens[m] == draft_tokens[m]) {
                    ++matched_drafts;
                } else {
                    break;
                }
            }
            eng->tel.accepted_draft_tokens += matched_drafts;

            bool hit_eos = false;
            uint32_t emitted_this_step = 0U;
            for (uint32_t i = 0U; i < A && generated < target_max_tokens; ++i) {
                const uint32_t tok = vr.emitted_tokens[i];
                if (cfg_opts.stop_on_eos && tok == bdl.eos_token_id) {
                    hit_eos = true;
                    break;
                }
                if (out_token_ids != nullptr) {
                    out_token_ids[generated] = tok;
                }
                if (stream_cb != nullptr) {
                    stream_cb(tok, strata_detokenize_token(&bdl, tok), stream_user_ctx);
                }
                ++generated;
                ++emitted_this_step;
            }

            if (hit_eos || generated >= target_max_tokens || emitted_this_step == 0U) {
                break;
            }

            /* Advance KV-cache cursor by A (implicitly rolling back rejected draft slots >= cur_pos + A!) */
            std::memcpy(eng->h_last.data(),
                        eng->h_pre_norm.data() + static_cast<size_t>(A - 1U) * D,
                        D * sizeof(float));
            cur_pos += A;
            cur_tok  = vr.emitted_tokens[A - 1U];
        }

        const uint64_t t_dec1 = mach_absolute_time();
        eng->drain_in_flight_ring_slots();

        struct rusage ru_end{};
        getrusage(RUSAGE_SELF, &ru_end);

        eng->tel.generated_tokens      = generated;
        eng->tel.decode_wall_ms        = mach_ticks_to_us(t_dec1 - t_dec0) * 1e-3;
        eng->tel.tokens_per_sec        = (eng->tel.decode_wall_ms > 0.0)
            ? (static_cast<double>(generated) / (eng->tel.decode_wall_ms * 1e-3))
            : 0.0;
        eng->tel.avg_accepted_per_step = (eng->tel.decode_steps > 0U)
            ? (static_cast<double>(generated) / static_cast<double>(eng->tel.decode_steps))
            : 0.0;
        eng->tel.draft_accept_rate     = (eng->tel.drafted_tokens_total > 0U)
            ? (static_cast<double>(eng->tel.accepted_draft_tokens) /
               static_cast<double>(eng->tel.drafted_tokens_total))
            : 0.0;
        eng->tel.steady_major_faults   = ru_end.ru_majflt - ru_steady_start.ru_majflt;
        eng->tel.steady_minor_faults   = ru_end.ru_minflt - ru_steady_start.ru_minflt;
        eng->tel.peak_rss_gib          = static_cast<double>(ru_end.ru_maxrss) /
                                         (1024.0 * 1024.0 * 1024.0);

        if (num_generated_out) *num_generated_out = generated;
        if (telemetry_out) *telemetry_out = eng->tel;
        return 0;
    }
}

} /* extern "C" */

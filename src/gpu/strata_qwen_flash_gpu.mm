#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
#include <dispatch/dispatch.h>

#include "../../include/strata_qwen_flash.h"
#include "../../include/strata_embedded_shaders.h"
#import "../../include/strata_metal_cache.h"

#include <fcntl.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMaxPrefillChunk = 1024U;
constexpr uint32_t kMaxSpecBatch     = 8U; // Supports M = 1 .. 8 (chains K=0..6 & width-2 tree M=7)
constexpr uint32_t kMaxSpecSnapSlots = 6U; // Direct zero-copy ring snapshot slots (0..5)
constexpr uint32_t kMaxDecodeSplits = 16U;
constexpr uint32_t kDefaultMtpShortlistV  = 32768U;  // 32K SLC-pinned FR-Spec shortlist (45.0 MiB < 48 MiB SLC!)
constexpr uint32_t kFrSpecShortlistSize   = 32768U;
constexpr uint32_t kFrSpecDynSlots        = 1024U;   // Dynamic GPU-patched context slots (prompt + online self-healing!)
constexpr uint32_t kFrSpecStaticRows      = kFrSpecShortlistSize - kFrSpecDynSlots; // 31,744 immutable rows
constexpr uint32_t kMtpShortlistTailStart = 248044U;
constexpr uint32_t kNgramHashSize         = 65536U;  // 256 KiB direct-mapped rolling 4-gram latest-occurrence table

constexpr uint32_t kOutF16    = 0U;
constexpr uint32_t kOutF32    = 1U;

struct SqfPadCastParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t pad;
};

struct SqfEmbedParamsHost {
    uint32_t M;
    uint32_t cols;
    uint32_t vocab_size;
    uint32_t hc_tile_count;
};

struct SqfGemvBatchParamsHost {
    uint32_t M;
    uint32_t rows;
    uint32_t cols;
    uint32_t out_mode;
};

struct SqfLinQzGemvParamsHost {
    uint32_t M;
    uint32_t qkv_rows;
    uint32_t z_rows;
    uint32_t cols;
};

struct SqfAttnQkvGemvParamsHost {
    uint32_t M;
    uint32_t q_rows;
    uint32_t kv_rows;
    uint32_t cols;
};

struct SqfNaxGemmParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t N_logical;
    uint32_t N_padded;
    uint32_t K;
    uint32_t out_mode;
    uint32_t pad0;
    uint32_t pad1;
};

struct SqfHcDownNaxParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t lowrank;
    uint32_t cols;
    float    inv_hc;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SqfHcUpBlendNaxParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t lowrank;
    uint32_t hc_count;
    float    inv_hc;
    uint32_t write_f32;
    uint32_t pad0;
};

struct SqfHcInjectPrefillParamsHost {
    uint32_t M_valid;
    uint32_t hc_count;
    uint32_t cols;
    float    inv_hc;
};

struct SqfSharedGuNaxParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t d_ff;
    uint32_t K;
};

struct SqfGroupedNormParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t groups;
    uint32_t group_dim;
    float    eps;
    uint32_t write_f32;
    uint32_t pad0;
    uint32_t pad1;
};

struct SqfHcDownActParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t lowrank;
    uint32_t hc_count;
    float    inv_hc;
    uint32_t has_inject;
    uint32_t pad0;
    uint32_t pad1;
};

struct SqfHcDownInjParamsHost {
    uint32_t M_valid;
    uint32_t lowrank;
    uint32_t hc_count;
    uint32_t cols;
    float    inv_hc;
    uint32_t has_inject;
    uint32_t pad0;
    uint32_t pad1;
};

struct SqfLinAbParamsHost {
    uint32_t M;
    uint32_t a_rows;
    uint32_t b_rows;
    uint32_t cols;
};

struct SqfHcBlendParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t hc_count;
    float    inv_hc;
    uint32_t write_f32;
    uint32_t pad0;
    uint32_t pad1;
};

struct SqfHcResAddParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t hc_count;
    uint32_t pad;
};

struct SqfPleGateParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t hc_count;
    float    eps;
};

struct SqfPleConvParamsHost {
    uint32_t M;
    uint32_t channels;
    uint32_t dilation;
    uint32_t save_snap_mask;
    uint32_t tree_branch_t;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SqfDeltaConvParamsHost {
    uint32_t M;
    uint32_t channels;
    uint32_t save_snap_mask;
    uint32_t snap_layer_idx;
    uint32_t num_lin_layers;
    uint32_t tree_branch_t;
    uint32_t pad1;
    uint32_t pad2;
};

struct SqfDeltaNetParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t n_k;
    uint32_t n_v;
    float    eps;
    uint32_t gate_sigmoid;
    uint32_t save_snap_mask;
    uint32_t snap_layer_idx;
    uint32_t num_lin_layers;
    uint32_t tree_branch_t;
    uint32_t pad1;
    uint32_t pad2;
};

struct SqfDeltaWyParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t num_chunks;
    uint32_t n_k;
    uint32_t n_v;
    float    eps;
    uint32_t gate_sigmoid;
    uint32_t pad0;
};

struct SqfQsaPrepParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t seq_pos_start;
    uint32_t max_seq_alloc;
    float    eps;
    uint32_t kv_zero_end;
    uint32_t paged;
    uint32_t tree_mode;
};

struct SqfIndexerParamsHost {
    uint32_t M_valid;
    uint32_t idx_n_heads;
    uint32_t idx_head_dim;
    uint32_t rotary_dim;
    uint32_t compress_ratio;
    uint32_t block_topk;
    uint32_t seq_pos_start;
    uint32_t first_block;
    uint32_t num_new_blocks;
    float    eps;
    uint32_t max_blocks_stride;
    uint32_t tree_mode;
    uint32_t M_padded;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SqfQsaDecodeParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t q_per_tg;
    uint32_t seq_pos_start;
    uint32_t max_seq_alloc;
    uint32_t max_splits;
    uint32_t compress_ratio;
    uint32_t block_topk;
    float    attn_scale;
    uint32_t m_base;
    uint32_t paged;
    uint32_t tree_mode;
    uint32_t m_end;
};

struct SqfTreeCompactParamsHost {
    uint32_t dst_pos;
    uint32_t src_pos;
    uint32_t count;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t idx_head_dim;
    uint32_t max_seq_alloc;
    uint32_t pad0;
};

struct SqfQsaPrefillNaxParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t seq_pos_start;
    uint32_t total_kv;
    uint32_t max_seq_alloc;
    uint32_t paged;
};

struct SqfMoeRouterParamsHost {
    uint32_t M_valid;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t norm_topk_prob;
};

struct SqfSwigluParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t d_ff;
    uint32_t pad;
};

struct SqfSharedGateParamsHost {
    uint32_t M_valid;
    uint32_t D;
};

struct SqfQ8HcUpBlendParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t lowrank;
    float    inv_hc;
    uint32_t write_f32;
    uint32_t pad0;
};

struct SqfMoeTopkSharedGateParamsHost {
    uint32_t M_valid;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t D;
};

struct SqfMoeGatherParamsHost {
    uint32_t M_valid;
    uint32_t top_k;
    uint32_t D;
    uint32_t moe_d_ff;
    uint32_t num_experts;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SqfMoeCombineParamsHost {
    uint32_t M_valid;
    uint32_t top_k;
    uint32_t D;
    uint32_t hc_count;
};

struct SqfMtpNormHiddenParamsHost {
    uint32_t M_valid;
    uint32_t rows_padded;
    uint32_t D;
    uint32_t hc_count;
    float    eps;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SqfMtpAddParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t hc_count;
    uint32_t pad;
};

struct SqfShortlistLmHeadParamsHost {
    uint32_t M;
    uint32_t cols;
    uint32_t v_short;
    uint32_t tail_start;
    uint32_t v_total;
    uint32_t v_out;
    uint32_t pad0;
    uint32_t pad1;
};

struct SqfPackFrspecParamsHost {
    uint32_t num_rows;
    uint32_t cols;
    uint32_t src_v_total;
    uint32_t pad0;
};

struct SqfI8QuantParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t N_logical;
    uint32_t N_padded;
    uint32_t K;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SqfSampleParamsHost {
    uint32_t M;
    uint32_t vocab_size;
    uint32_t greedy;
    float    inv_temp;
    uint32_t top_k;
    float    top_p;
    uint32_t rng_lo;
    uint32_t rng_hi;
    uint32_t use_token_map;
    uint32_t compute_pmax;
    uint32_t pad0;
    uint32_t pad1;
};

struct SqfLogitPenaltyParamsHost {
    uint32_t M;
    uint32_t vocab_size;
    uint32_t logits_stride;
    uint32_t use_shortlist;
    uint32_t pad_sl;
    uint32_t num_hist_ids;
    uint32_t extra_mode;
    uint32_t extra_count;
    float    presence_penalty;
    float    frequency_penalty;
    float    repetition_penalty;
    uint32_t pad0;
    uint32_t drop_toks[8];
};
static_assert(sizeof(SqfLogitPenaltyParamsHost) == 80U, "SqfLogitPenaltyParamsHost must match Metal layout");

enum SqfSpecCtlState : uint32_t {
    kSpecActiveProbe   = 0U,
    kSpecCooldownK0    = 1U,
    kSpecHighConfFused = 2U
};

struct SqfSpecCtl {
    // True concave Qwen3.8-Flash-Next (180B MoE) verify latency curve on M5 Max (M=1..8):
    // marginal cost per extra token decreases as Top-10/512 routed experts deduplicate across batch!
    static constexpr double kSeedVerMs[kMaxSpecBatch] = {12.2, 16.2, 19.0, 21.0, 22.6, 24.3, 25.9, 27.4};

    SqfSpecCtlState state = kSpecActiveProbe;
    double          ema_ver_ms[kMaxSpecBatch] = {12.2, 16.2, 19.0, 21.0, 22.6, 24.3, 25.9, 27.4};
    uint32_t        ver_obs[kMaxSpecBatch]    = {};
    double          ver_scale                 = 1.0;  // Coherent context-length scaling for unvisited k slots
    double          ema_draft_ms              = 0.43; // per single 32K SLC MTP draft token
    double          ema_succ                  = 4.20; // Wald-unbiased numerator (Beta prior p = 0.70, N = 6)
    double          ema_trials                = 6.00; // Wald-unbiased denominator
    double          ema_geom_p                = 0.70;
    double          ema_pmax                  = 0.70;
    uint32_t        cooldown_rem              = 0U;
    uint32_t        consec_cooldowns          = 0U;
    uint32_t        consec_zero_acc           = 0U;
    uint32_t        consec_high_acc           = 0U;
    uint32_t        obs_steps                 = 0U;
    bool            probe_after_cd            = false;

    double est_ver_ms(uint32_t k) const {
        if (k >= kMaxSpecBatch) k = kMaxSpecBatch - 1U;
        return (ver_obs[k] >= 2U) ? ema_ver_ms[k] : (kSeedVerMs[k] * ver_scale);
    }

    void record_ver_ms(uint32_t k, double ms) {
        if (k >= kMaxSpecBatch || ms <= 1.0 || ms >= 500.0) return;
        const double ratio = std::clamp(ms / kSeedVerMs[k], 0.65, 3.0);
        ver_scale = (obs_steps == 0U && ver_obs[k] == 0U) ? ratio : (0.80 * ver_scale + 0.20 * ratio);
        if (ver_obs[k] == 0U) {
            ema_ver_ms[k] = ms;
        } else {
            ema_ver_ms[k] = 0.80 * ema_ver_ms[k] + 0.20 * ms;
        }
        ++ver_obs[k];
    }

    void reset() {
        state = kSpecActiveProbe;
        for (uint32_t i = 0U; i < kMaxSpecBatch; ++i) {
            ema_ver_ms[i] = kSeedVerMs[i];
            ver_obs[i]    = 0U;
        }
        ver_scale        = 1.0;
        ema_draft_ms     = 0.43;
        ema_succ         = 4.20;
        ema_trials       = 6.00;
        ema_geom_p       = 0.70;
        ema_pmax         = 0.70;
        cooldown_rem     = 0U;
        consec_cooldowns = 0U;
        consec_zero_acc  = 0U;
        consec_high_acc  = 0U;
        obs_steps        = 0U;
        probe_after_cd   = false;
    }
};

std::string sqf_read_text_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

std::string sqf_resolve_sibling_path(const char* rel_from_src_gpu) {
    std::string self_file = __FILE__;
    size_t slash = self_file.find_last_of('/');
    if (slash != std::string::npos) {
        std::string candidate = self_file.substr(0, slash + 1) + rel_from_src_gpu;
        std::ifstream test(candidate);
        if (test.good()) return candidate;
    }
    return std::string("src/gpu/") + rel_from_src_gpu;
}

id<MTLBuffer> sqf_alloc_shared_prefaulted(id<MTLDevice> dev, size_t bytes) {
    const size_t aligned = std::max<size_t>(4096U, (bytes + 4095U) & ~static_cast<size_t>(4095U));
    id<MTLBuffer> buf = [dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
    if (buf) {
        std::memset([buf contents], 0, aligned);
    }
    return buf;
}

id<MTLBuffer> sqf_alloc_private_aligned(id<MTLDevice> dev, size_t bytes) {
    const size_t aligned = std::max<size_t>(4096U, (bytes + 4095U) & ~static_cast<size_t>(4095U));
    return [dev newBufferWithLength:aligned options:MTLResourceStorageModePrivate];
}

void sqf_encode_zero_buffer(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pso_zero,
    id<MTLBuffer> buf,
    size_t byte_len
) {
    const uint32_t count_u4 = static_cast<uint32_t>((byte_len + 15ULL) / 16ULL);
    if (count_u4 == 0U || !buf) return;
    [enc setComputePipelineState:pso_zero];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&count_u4 length:sizeof(uint32_t) atIndex:1];
    const uint32_t tg = 256U;
    [enc dispatchThreadgroups:MTLSizeMake((count_u4 + tg - 1U) / tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void sqf_encode_copy_buffer(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pso_copy,
    id<MTLBuffer> src,
    NSUInteger src_off,
    id<MTLBuffer> dst,
    NSUInteger dst_off,
    size_t byte_len
) {
    const uint32_t count_u4 = static_cast<uint32_t>((byte_len + 15ULL) / 16ULL);
    if (count_u4 == 0U || !src || !dst) return;
    [enc setComputePipelineState:pso_copy];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:dst_off atIndex:1];
    [enc setBytes:&count_u4 length:sizeof(uint32_t) atIndex:2];
    const uint32_t tg = 256U;
    [enc dispatchThreadgroups:MTLSizeMake((count_u4 + tg - 1U) / tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

id<MTLResidencySet> sqf_create_residency_set(
    id<MTLDevice> dev,
    id<MTLCommandQueue> queue,
    NSArray<id<MTLBuffer>>* buffers
) {
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
        desc.initialCapacity = std::max<NSUInteger>(buffers.count, 1U);
        NSError* err = nil;
        id<MTLResidencySet> rset = [dev newResidencySetWithDescriptor:desc error:&err];
        if (rset && !err) {
            for (id<MTLBuffer> b in buffers) {
                if (b) [rset addAllocation:b];
            }
            [rset commit];
            [rset requestResidency];
            if (queue) [queue addResidencySet:rset];
            return rset;
        }
    }
    return nil;
}

void sqf_detach_residency_set(id<MTLCommandQueue> queue, id<MTLResidencySet> rset) {
    if (!rset) return;
    if (@available(macOS 15.0, *)) {
        if (queue) [queue removeResidencySet:rset];
        [rset endResidency];
    }
}

uint32_t sqf_choose_q_per_tg(uint32_t n_q, uint32_t n_kv) {
    const uint32_t gqa = std::max<uint32_t>(1U, n_q / std::max<uint32_t>(1U, n_kv));
    for (uint32_t g = std::min<uint32_t>(6U, gqa); g >= 1U; --g) {
        if ((gqa % g) == 0U) return g;
    }
    return 1U;
}

float sqf_host_rand_u01(uint64_t seed, uint64_t step) {
    uint64_t z = sqf_splitmix64(seed + step * 0x9E3779B97F4A7C15ULL);
    uint32_t u24 = static_cast<uint32_t>(z >> 40ULL) & 0xFFFFFFU;
    return (static_cast<float>(u24) + 0.5f) * (1.0f / 16777216.0f);
}

float sqf_lookup_sparse_prob(
    uint32_t tok_id,
    const uint32_t* ids,
    const float* probs,
    uint32_t keep_k
) {
    for (uint32_t i = 0U; i < keep_k; ++i) {
        if (ids[i] == tok_id) return probs[i];
    }
    return 0.0f;
}

} // namespace

struct SqfAneQsaProbe;

struct StrataQwenFlashGpuEngine {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLResidencySet>         residency_set;
    bool                        has_nax;

    // 49 Metal 4 Compute Pipeline States
    id<MTLComputePipelineState> pso_zero_u32;
    id<MTLComputePipelineState> pso_copy_u32;
    id<MTLComputePipelineState> pso_f32_to_f16_pad;
    id<MTLComputePipelineState> pso_embed_q4;
    id<MTLComputePipelineState> pso_q4_gemv_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q4_gemv_8lane_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q4_lm_head_shortlist_m[2];
    id<MTLComputePipelineState> pso_q4_gemv_lin_qz_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q4_gemv_attn_qkv_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_f16_gemv_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_f16_gemv_row_tg128_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_f16_small_n_prefill;
    id<MTLComputePipelineState> pso_f16_gemv_8lane;
    id<MTLComputePipelineState> pso_hc_down_inj_act;
    id<MTLComputePipelineState> pso_f16_lin_ab_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_f16_shared_gu_silu;
    id<MTLComputePipelineState> pso_q8_gemv_batch;
    id<MTLComputePipelineState> pso_q8_gemv_8lane;
    id<MTLComputePipelineState> pso_q8_hc_down_inj_act_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q8_hc_up_blend_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q8_gemm_nax_64x64_k128_sg4;
    id<MTLComputePipelineState> pso_q8_gemm_nax_32x64_k128_sg2;
    id<MTLComputePipelineState> pso_q8_gemm_nax_16x64_k128_sg2;
    id<MTLComputePipelineState> pso_q8_gemm_nax_m16;
    id<MTLComputePipelineState> pso_q8_gemm_nax_m64;
    id<MTLComputePipelineState> pso_q4_shared_gu_silu_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q4_shared_gu_silu_nax_m64;
    id<MTLComputePipelineState> pso_q4_shared_gu_silu_nax_m32;
    id<MTLComputePipelineState> pso_q4_gemm_nax_64x64_k128_sg4;
    id<MTLComputePipelineState> pso_q4_gemm_nax_32x64_k128_sg2;
    id<MTLComputePipelineState> pso_q4_gemm_nax_16x64_k128_sg2;
    id<MTLComputePipelineState> pso_q4_gemm_nax_m16;
    id<MTLComputePipelineState> pso_q4_gemm_nax_m64;
    id<MTLComputePipelineState> pso_f16_gemm_nax_64x128_sg4;
    id<MTLComputePipelineState> pso_f16_gemm_nax_64x64_sg4;
    id<MTLComputePipelineState> pso_f16_gemm_nax_32x64_sg2;
    id<MTLComputePipelineState> pso_f16_gemm_nax_16x32_sg2;
    id<MTLComputePipelineState> pso_q8_hc_down_act_nax_m64;
    id<MTLComputePipelineState> pso_q8_hc_down_act_nax_m32;
    id<MTLComputePipelineState> pso_f16_hc_down_act_nax_m64;
    id<MTLComputePipelineState> pso_f16_hc_down_act_nax_m32;
    id<MTLComputePipelineState> pso_q8_hc_up_blend_nax_m64;
    id<MTLComputePipelineState> pso_q8_hc_up_blend_nax_m32;
    id<MTLComputePipelineState> pso_f16_hc_up_blend_nax_m64;
    id<MTLComputePipelineState> pso_f16_hc_up_blend_nax_m32;
    id<MTLComputePipelineState> pso_f16_hc_inject_prefill;
    id<MTLComputePipelineState> pso_f16_lin_ab_prefill;
    id<MTLComputePipelineState> pso_grouped_rmsnorm;
    id<MTLComputePipelineState> pso_hc_down_act;
    id<MTLComputePipelineState> pso_hc_blend;
    id<MTLComputePipelineState> pso_hc_residual_add;
    id<MTLComputePipelineState> pso_hc_res_add_rmsnorm;
    id<MTLComputePipelineState> pso_ple_gate_gv;
    id<MTLComputePipelineState> pso_ple_conv1d_add;
    id<MTLComputePipelineState> pso_deltanet_conv1d;
    id<MTLComputePipelineState> pso_deltanet_conv1d_prefill;
    id<MTLComputePipelineState> pso_deltanet_conv1d_save_tail;
    id<MTLComputePipelineState> pso_deltanet_rec;
    id<MTLComputePipelineState> pso_deltanet_prep_batch;
    id<MTLComputePipelineState> pso_deltanet_rec_batch;
    id<MTLComputePipelineState> pso_deltanet_post_batch;
    id<MTLComputePipelineState> pso_deltanet_wy_qk;
    id<MTLComputePipelineState> pso_deltanet_wy_prep_hv;
    id<MTLComputePipelineState> pso_deltanet_wy_recur;
    id<MTLComputePipelineState> pso_qsa_qkv_prep;
    id<MTLComputePipelineState> pso_qsa_idx_prep;
    id<MTLComputePipelineState> pso_qsa_idx_prep_prefill;
    id<MTLComputePipelineState> pso_qsa_idx_blocks;
    id<MTLComputePipelineState> pso_qsa_idx_branch_b_blk;
    id<MTLComputePipelineState> pso_qsa_idx_score;
    id<MTLComputePipelineState> pso_qsa_idx_score_batch;
    id<MTLComputePipelineState> pso_qsa_idx_score_tiled;
    id<MTLComputePipelineState> pso_qsa_idx_score_nax;
    id<MTLComputePipelineState> pso_qsa_idx_topk;
    id<MTLComputePipelineState> pso_qsa_decode_split;
    id<MTLComputePipelineState> pso_qsa_decode_split_gqa12;
    id<MTLComputePipelineState> pso_qsa_decode_reduce;
    id<MTLComputePipelineState> pso_tree_compact_b;
    id<MTLComputePipelineState> pso_qsa_prefill_nax;
    id<MTLComputePipelineState> pso_qsa_sparse_prefill_nax;
    id<MTLComputePipelineState> pso_moe_topk;
    id<MTLComputePipelineState> pso_swiglu;
    id<MTLComputePipelineState> pso_shared_gate_dot;
    id<MTLComputePipelineState> pso_moe_topk_shared_gate;
    id<MTLComputePipelineState> pso_moe_gather_gu;
    id<MTLComputePipelineState> pso_moe_gather_dn;
    id<MTLComputePipelineState> pso_moe_bin_tokens;
    id<MTLComputePipelineState> pso_moe_grouped_gu_b[4]; // max_b in {4, 5, 6, 8}
    id<MTLComputePipelineState> pso_moe_grouped_dn_b[4]; // max_b in {4, 5, 6, 8}
    id<MTLComputePipelineState> pso_moe_prefill_bin_tokens;
    id<MTLComputePipelineState> pso_moe_prefill_gu_nax;
    id<MTLComputePipelineState> pso_moe_prefill_dn_nax;
    id<MTLComputePipelineState> pso_moe_bin_tokens_tri;
    id<MTLComputePipelineState> pso_moe_gather_x_f16;
    id<MTLComputePipelineState> pso_moe_prefill_gu_nax_m16;
    id<MTLComputePipelineState> pso_moe_prefill_gu_nax_m32;
    id<MTLComputePipelineState> pso_moe_prefill_gu_nax_m64;
    id<MTLComputePipelineState> pso_moe_prefill_gu_nax_m128;
    id<MTLComputePipelineState> pso_moe_prefill_dn_nax_m16;
    id<MTLComputePipelineState> pso_moe_prefill_dn_nax_m32;
    id<MTLComputePipelineState> pso_moe_prefill_dn_nax_m64;
    id<MTLComputePipelineState> pso_moe_prefill_dn_nax_m128;
    id<MTLComputePipelineState> pso_moe_combine;
    id<MTLComputePipelineState> pso_moe_combine_f16;
    id<MTLComputePipelineState> pso_mtp_norm_split;
    id<MTLComputePipelineState> pso_mtp_add_bcast;
    id<MTLComputePipelineState> pso_sample_vocab;
    id<MTLComputePipelineState> pso_sample_stage1;
    id<MTLComputePipelineState> pso_sample_stage2;
    id<MTLComputePipelineState> pso_apply_logit_penalties;
    id<MTLComputePipelineState> pso_pack_frspec_q4;
    id<MTLComputePipelineState> pso_patch_frspec_q4;
    id<MTLComputePipelineState> pso_quant_f16_to_i8_row;
    id<MTLComputePipelineState> pso_q8_requant_row_i8;
    id<MTLComputePipelineState> pso_q8_requant_row_i8_w32;
    id<MTLComputePipelineState> pso_q4_requant_row_i8;
    id<MTLComputePipelineState> pso_i8_gemm_nax_64x128_sg4;
    id<MTLComputePipelineState> pso_i8_gemm_nax_64x64_sg4;
    id<MTLComputePipelineState> pso_i8_gemm_nax_32x64_sg2;
    id<MTLComputePipelineState> pso_i8_hc_down_act_nax_m64;
    id<MTLComputePipelineState> pso_i8_hc_down_act_nax_m32;
    id<MTLComputePipelineState> pso_i8_hc_up_blend_nax_m64;
    id<MTLComputePipelineState> pso_i8_hc_up_blend_nax_m32;
    id<MTLComputePipelineState> pso_i8_shared_gu_silu_nax_m64;
    id<MTLComputePipelineState> pso_i8_shared_gu_silu_nax_m32;

    // Bundle & Headers
    StrataQwenFlashBundle*      bundle;
    StrataQwenFlashPart0Header  hdr0;
    StrataQwenFlashPart1Header  hdr1;
    StrataQwenFlashConfig       cfg;

    // Sequence &PLE/MTP Tracking
    uint32_t                    max_seq_len;
    uint32_t                    max_seq_alloc;
    uint32_t                    max_blocks_alloc;
    uint32_t                    seq_pos;
    uint32_t                    mtp_seq_pos;
    uint32_t                    ple_prev_tokens[2];
    uint64_t                    sample_step;
    bool                        spec_tree_enabled;
    bool                        mtp_kv_sync_enabled;
    StrataQwenFlashPrefillProgressFn progress_cb;
    void*                       progress_user_ctx;

    // Telemetry & Byte Accounting
    char                        device_name[128];
    bool                        zero_copy_mmap;
    double                      load_time_ms;
    uint64_t                    part0_mapped_bytes;
    uint64_t                    part1_mapped_bytes;
    uint64_t                    conv_state_bytes;
    uint64_t                    rec_state_bytes;
    uint64_t                    ple_state_bytes;
    uint64_t                    kv_cache_bytes_allocated;
    uint64_t                    kv_cache_bytes_per_token;
    uint64_t                    workspace_bytes;
    StrataQwenFlashPerfCounters counters;

    // Stride Geometry
    uint64_t                    conv_layer_stride;
    uint64_t                    rec_layer_stride;
    uint64_t                    kv_layer_stride;
    uint64_t                    idx_raw_layer_stride;
    uint64_t                    idx_blk_layer_stride;

    // Two Zero-Copy UMA Weight Buffers (`< 40 GiB` each << `80.64 GiB` Metal limit!)
    id<MTLBuffer>               b_part0;
    id<MTLBuffer>               b_part1;

    // Persistent States (DeltaNet + PLE + 12-Layer & 1-MTP KV + QSA Indexer + MTP HC)
    id<MTLBuffer>               b_conv_state;
    id<MTLBuffer>               b_rec_state;
    id<MTLBuffer>               b_ple_conv_state;
    id<MTLBuffer>               b_k_cache;
    id<MTLBuffer>               b_v_cache;
    id<MTLBuffer>               b_idx_raw_k;
    id<MTLBuffer>               b_idx_block_k;
    id<MTLBuffer>               b_mtp_hc_prev;

    // Speculative Rollback Zero-Copy Ring Snapshot Slots (`kMaxSpecSnapSlots = 6` boundary states)
    // Each slot carries its own `MTLResidencySet` so O(1) pointer swaps work safely across ALL context slots!
    id<MTLBuffer>               b_rec_slots[kMaxSpecSnapSlots];
    id<MTLBuffer>               b_conv_slots[kMaxSpecSnapSlots];
    id<MTLBuffer>               b_ple_conv_slots[kMaxSpecSnapSlots];
    id<MTLResidencySet>         ring_rec_rsets[kMaxSpecSnapSlots];

    // Private GPU Scratch Activations (`kMaxPrefillChunk = 256`, `0` runtime allocations!)
    id<MTLBuffer>               b_hc;
    id<MTLBuffer>               b_mtp_hc;
    id<MTLBuffer>               b_hc_norm_f32;
    id<MTLBuffer>               b_hc_norm_f16;
    id<MTLBuffer>               b_hc_down_f32;
    id<MTLBuffer>               b_hc_u_f16;
    id<MTLBuffer>               b_hc_up_f32;
    id<MTLBuffer>               b_hc_inj_f32;
    id<MTLBuffer>               b_x_in_f16;
    id<MTLBuffer>               b_x_in_f32;
    id<MTLBuffer>               b_block_y_f32;
    id<MTLBuffer>               b_ple_emb_f16;
    id<MTLBuffer>               b_ple_gvn_f32;
    id<MTLBuffer>               b_qkv;
    id<MTLBuffer>               b_qkv_conv;
    id<MTLBuffer>               b_z;
    id<MTLBuffer>               b_a;
    id<MTLBuffer>               b_b;
    id<MTLBuffer>               b_dn_qk_hat;
    id<MTLBuffer>               b_dn_gates;
    id<MTLBuffer>               b_dn_raw_out;
    id<MTLBuffer>               b_k_raw;
    id<MTLBuffer>               b_v_raw;
    id<MTLBuffer>               b_q_rope;
    id<MTLBuffer>               b_attn_out;
    id<MTLBuffer>               b_idx_qk_f32;
    id<MTLBuffer>               b_idx_q_rot;
    id<MTLBuffer>               b_idx_scores;
    id<MTLBuffer>               b_idx_sel_blocks;
    id<MTLBuffer>               b_idx_blk_branch_b;
    id<MTLBuffer>               b_part_m;
    id<MTLBuffer>               b_part_l;
    id<MTLBuffer>               b_part_o;
    id<MTLBuffer>               b_router_logits;
    id<MTLBuffer>               b_topk_idx;
    id<MTLBuffer>               b_topk_w;
    id<MTLBuffer>               b_moe_mid_f16;
    id<MTLBuffer>               b_moe_down_part;
    id<MTLBuffer>               b_moe_sorted_pairs;
    id<MTLBuffer>               b_moe_work_tiles;
    id<MTLBuffer>               b_moe_num_tiles;
    id<MTLBuffer>               b_moe_indirect_args;
    id<MTLBuffer>               b_shared_gate_f32;
    id<MTLBuffer>               b_shared_up_f32;
    id<MTLBuffer>               b_shared_mid_f16;
    id<MTLBuffer>               b_shared_out_f32;
    id<MTLBuffer>               b_shared_scalar;
    id<MTLBuffer>               b_sample_stage_v;
    id<MTLBuffer>               b_sample_stage_i;
    id<MTLBuffer>               b_sample_stage_lse;
    // 32K SLC-Pinned FR-Spec Draft `lm_head` (`45.0 MiB < 48 MiB` SLC) + Bijective Vocab Maps
    id<MTLBuffer>               b_frspec_lm_head;
    id<MTLBuffer>               b_frspec_ids;
    id<MTLBuffer>               b_frspec_full_to_short;
    StrataQwenFlashQuantMatrix  frspec_lm_head_desc = {};

    // Shared Host <-> GPU IO Buffers
    id<MTLBuffer>               b_ple_emb_f32;
    id<MTLBuffer>               b_ple_emb_f32_alt; // Double-buffered async prefill PLE staging (`640 KiB`)
    id<MTLBuffer>               b_inv_freq;
    id<MTLBuffer>               b_token_ids;
    id<MTLBuffer>               b_mtp_token_ids;
    id<MTLBuffer>               b_logits;
    id<MTLBuffer>               b_next_tokens;
    id<MTLBuffer>               b_sample_top_ids;
    id<MTLBuffer>               b_sample_top_probs;
    id<MTLBuffer>               b_sample_keep_k;
    id<MTLBuffer>               b_draft_top_ids;
    id<MTLBuffer>               b_draft_top_probs;
    id<MTLBuffer>               b_draft_keep_k;
    id<MTLBuffer>               b_draft_pmax;
    id<MTLBuffer>               b_pen_ids;         // Shared [V] uint32_t (unique active-window token IDs)
    id<MTLBuffer>               b_pen_counts;      // Shared [V] uint16_t (active-window per-token counts)
    uint32_t                    pen_num_ids       = 0U;
    std::vector<uint32_t>       pen_hist;          // Full completion token history for exact sliding-window eviction
    std::vector<uint32_t>       pen_pos;           // [V] index of token in `b_pen_ids` (or `0xFFFFFFFFU` if inactive)
    uint32_t                    pen_window_start  = 0U;
    uint32_t                    pen_window_end    = 0U;

    // ---- Step 4: O(1) Context-Slot Binding (Tier 1) + Global-pos Paged QSA KV (Tier 2) ----
    StrataQwenFlashContextSlot* default_slot      = nullptr; // Wraps the engine-created linear KV/state buffers
    StrataQwenFlashContextSlot* bound_slot        = nullptr; // Never NULL after create
    id<MTLBuffer>               b_block_loc;                  // Bound slot block table (4 KiB dummy when unpaged)
    id<MTLBuffer>               b_block_loc_dummy;
    uint32_t                    kv_stride_tokens  = 0U;      // `max_seq_alloc` (linear) or UMA pool tokens (paged)
    bool                        kv_paged          = false;
    bool                        paged_error       = false;
    id<MTLBuffer>               b_probe_x;                    // Shared [num_full, D] F16 QSA attention inputs (ANE lookahead input)
    SqfAneQsaProbe*             ane               = nullptr;
    bool                        ane_init_tried    = false;
    id<MTLCommandBuffer>        pending_fix_cb         = nil;      // Async spec_decode rollback/commit CB overlapped with next step
    dispatch_queue_t            ple_async_queue        = nil;
    dispatch_group_t            ple_async_group        = nil;
    std::vector<uint64_t>       ple_all_rows; // Heap scratch [kMaxPrefillChunk * MAX_NGRAM_HEADS] (off GCD stack!)
    // Phase D3: Zero-Copy Adaptive Shortlisted-Vocab MTP Draft Head
    uint32_t                    mtp_shortlist_v        = kDefaultMtpShortlistV;
    uint32_t                    mtp_shortlist_tail     = kMtpShortlistTailStart;
    uint32_t                    shortlist_miss_mask    = 0U;
    uint32_t                    shortlist_obs_count    = 0U;
    uint32_t                    shortlist_bypass_steps = 0U;
    // Phase E1: Rolling Prompt/History N-Gram Suffix Drafting (0.00 ms GPU draft on verbatim suffix hits, 100% exact!)
    bool                        ngram_draft_enabled    = true;
    // Upgrade 5-Rec Controls & Telemetry (`MTLBinaryArchive` cache, 32K SLC FR-Spec, 3-State SpecCtl, INT8 Prefill)
    bool                        int8_prefill_enabled   = false;
    bool                        adaptive_k_enabled     = true;
    bool                        frspec_enabled         = true;
    bool                        frspec_widened         = false;
    uint32_t                    frspec_shortlist_size  = 0U;
    uint32_t                    frspec_dyn_cursor      = 0U;
    bool                        metal_cache_hit        = false;
    double                      pso_load_ms            = 0.0;
    SqfSpecCtl                  spec_ctl;
    // Optional env-gated (`STRATA_PROFILE_FLASH=1`) hardware GPU stage profiler (0 overhead when unset)
    bool                        prof_enabled           = false;
    bool                        prof_active            = false;
    bool                        force_serial_dn        = false;
    uint32_t                    prof_decode_step_cnt   = 0U;
    uint32_t                    prof_prefill_chunk_cnt = 0U;
    double                      prof_ms[12]            = {};
};

/* ============================================================================
 * Step 4 — Tier 1 Context Slots + Tier 2 Global-pos Paged QSA KV + F_NOCACHE NVMe + 16-Core ANE Lookahead
 * ============================================================================ */

@interface SqfAneQsaFeatureProvider : NSObject <MLFeatureProvider>
@property(nonatomic, strong) MLMultiArray* x;
- (instancetype)initWithArray:(MLMultiArray*)x;
@end

@implementation SqfAneQsaFeatureProvider
- (instancetype)initWithArray:(MLMultiArray*)x {
    if ((self = [super init])) {
        _x = x;
    }
    return self;
}
- (NSSet<NSString*>*)featureNames {
    return [NSSet setWithObject:@"x"];
}
- (MLFeatureValue*)featureValueForName:(NSString*)featureName {
    if ([featureName isEqualToString:@"x"]) {
        return [MLFeatureValue featureValueWithMultiArray:_x];
    }
    return nil;
}
@end

namespace {

constexpr uint8_t  kBlkUnalloc            = 0U;
constexpr uint8_t  kBlkResident           = 1U;
constexpr uint8_t  kBlkPaged              = 2U;
constexpr uint32_t kPhysFree              = 0xFFFFFFFFU;
constexpr uint32_t kAneSpatial            = 16U;   /* Orion `[1, C, 1, S=16]` ANE layout */
constexpr uint32_t kAneMaxPrefetchPerLyr  = 24U;
constexpr float    kAnePrefetchMargin     = 0.96f;
constexpr uint32_t kMaxIoBatchBlocks      = 1024U; /* 8 MiB 16 KiB-aligned coalesced NVMe DMA window */
constexpr uint64_t kCoreMLSpecV7          = 7ULL;
constexpr uint64_t kCoreMLArrayFloat16    = 65552ULL;

inline double sqf_ticks_ms(uint64_t ticks) {
    static mach_timebase_info_data_t tb = [] {
        mach_timebase_info_data_t t{};
        mach_timebase_info(&t);
        return t;
    }();
    return static_cast<double>(ticks) * static_cast<double>(tb.numer) / static_cast<double>(tb.denom) / 1e6;
}

} // namespace

struct SqfPagedCounters {
    uint64_t paged_out  = 0ULL;
    uint64_t recalled   = 0ULL;
    uint64_t wbytes     = 0ULL;
    uint64_t rbytes     = 0ULL;
    uint64_t misses     = 0ULL;
    uint64_t pf_issued  = 0ULL;
    uint64_t pf_hits    = 0ULL;
    double   wms        = 0.0;
    double   rms        = 0.0;
    bool     err        = false;
};

struct SqfMGroup {
    uint32_t              beg = 0U;
    uint32_t              end = 0U;
    std::vector<uint32_t> need;
};

struct SqfPagedIoRun {
    uint32_t idx0 = 0U;
    uint32_t len  = 0U;
    off_t    foff = 0;
};

struct SqfPagedLayerState {
    std::vector<uint8_t>  st;          /* [max_blocks] kBlkUnalloc / kBlkResident / kBlkPaged */
    std::vector<uint8_t>  on_nvme;     /* [max_blocks] 1 => bit-exact copy valid in kv_swap */
    std::vector<uint32_t> phys_owner;  /* [pool_blocks] logical block or kPhysFree */
    std::vector<uint32_t> free_list;   /* LIFO of physical pool blocks (lazy-deleted) */
    std::vector<float>    u;           /* [max_blocks] EWMA QSAIndexer importance */
    std::vector<uint32_t> pin_epoch;   /* [max_blocks] pinned while == pin_epoch_ctr */
    std::vector<uint32_t> pf_epoch;    /* [max_blocks] prefetched by the ANE lookahead in step `pf_epoch` */
    uint32_t              pin_epoch_ctr = 1U;
    uint32_t              resident      = 0U;
    uint32_t              paged_count   = 0U; /* exact count of blocks with st[b] == kBlkPaged */
    float                 last_thresh   = 0.0f;
    bool                  has_thresh    = false;

    /* Pre-allocated per-layer scratch buffers (0 per-step heap allocs; disjoint across dispatch_apply) */
    std::vector<uint32_t>                      scratch_need;
    std::vector<uint32_t>                      scratch_missing;
    std::vector<uint32_t>                      scratch_cand;
    std::vector<std::pair<uint32_t, uint32_t>> scratch_io_pairs;
    std::vector<SqfPagedIoRun>                 scratch_io_runs;
    std::vector<std::pair<float, uint32_t>>    scratch_ane_cand;
};

struct StrataQwenFlashContextSlot {
    StrataQwenFlashGpuEngine* eng        = nullptr;
    bool                      is_default = false;

    id<MTLBuffer>       b_conv_state;
    id<MTLBuffer>       b_rec_state;
    id<MTLBuffer>       b_ple_conv_state;
    id<MTLResidencySet> rec_rset;       // Paired residency set for `{b_conv_state, b_rec_state, b_ple_conv_state}` (travels on O(1) swap!)
    id<MTLBuffer>       b_k_cache;
    id<MTLBuffer>       b_v_cache;
    id<MTLBuffer>       b_mtp_k_cache;  // Optional unpaged 1-layer MTP KV cache in paged mode (0 CB splits in cb_draft + unlocks MTP KV-Sync!)
    id<MTLBuffer>       b_mtp_v_cache;
    id<MTLBuffer>       b_idx_raw_k;
    id<MTLBuffer>       b_idx_block_k;
    id<MTLBuffer>       b_mtp_hc_prev;
    id<MTLBuffer>       b_block_loc;
    id<MTLResidencySet> rset;

    uint32_t seq_pos            = 0U;
    uint32_t mtp_seq_pos        = 0U;
    uint32_t ple_prev_tokens[2] = {0U, 0U};
    uint64_t sample_step        = 0ULL;

    uint32_t kv_stride_tokens = 0U;
    uint64_t kv_layer_stride  = 0ULL;

    bool     paged        = false;
    bool     mtp_unpaged  = false;
    uint32_t pool_blocks  = 0U;
    uint32_t pool_tokens  = 0U;
    uint32_t n_layers     = 0U; // Total full slots (main + MTP)
    uint32_t paged_layers = 0U; // Number of paged layers (`num_full` when `mtp_unpaged`, else `n_layers`)
    uint32_t max_blocks  = 0U;
    int      swap_fd     = -1;
    char     swap_path[256] = {};
    uint8_t* io_stage    = nullptr; /* 16 KiB-aligned [kMaxIoBatchBlocks * block_bytes] DMA buffer */
    uint8_t* ane_stage   = nullptr; /* 16 KiB-aligned [n_layers * kAneMaxPrefetchPerLyr * block_bytes] */

    std::vector<SqfPagedLayerState> layers;
    uint32_t step_epoch    = 1U;
    bool     x_valid       = false;
    uint64_t bytes         = 0ULL;

    /* Reusable host-thread scratch buffers (zero per-step heap allocs) */
    std::vector<SqfMGroup>         scratch_mgroups;
    std::vector<uint32_t>          scratch_mine;
    std::vector<SqfPagedCounters>  scratch_ane_lc;

    StrataQwenFlashCompactionStats stats{};
    double                         ane_us_total       = 0.0;
    dispatch_group_t               probe_grp          = nil;
    bool                           probe_pending      = false;
    SqfPagedCounters               probe_out_counters{};
    bool                           probe_out_on_ane   = false;
    double                         probe_out_us       = 0.0;

    /* Phase E1: Per-slot rolling 4-gram suffix index (< 1 MiB, survives O(1) slot swaps cleanly) */
    std::vector<uint32_t>          ngram_hist;
    std::vector<uint32_t>          ngram_ht;
    uint32_t                       ngram_cooldown     = 0U;
};

namespace {

inline uint32_t sqf_ngram5_hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
    uint64_t h = (static_cast<uint64_t>(a) * 0x9E3779B97F4A7C15ULL) ^
                 (static_cast<uint64_t>(b) * 0xBF58476D1CE4E5B9ULL) ^
                 (static_cast<uint64_t>(c) * 0x94D049BB133111EBULL) ^
                 (static_cast<uint64_t>(d) * 0xD6E8FEB86659FD93ULL) ^
                 static_cast<uint64_t>(e);
    h ^= (h >> 33ULL);
    h *= 0xFF51AFD7ED558CCDULL;
    return static_cast<uint32_t>((h >> 32ULL) & (kNgramHashSize - 1U));
}

inline void sqf_ngram_reset(StrataQwenFlashContextSlot* s) {
    if (!s) return;
    s->ngram_hist.clear();
    if (!s->ngram_ht.empty()) {
        std::fill(s->ngram_ht.begin(), s->ngram_ht.end(), 0xFFFFFFFFU);
    }
    s->ngram_cooldown = 0U;
}

inline void sqf_ngram_append(StrataQwenFlashContextSlot* s, const uint32_t* toks, uint32_t n) {
    if (!s || !toks || n == 0U) return;
    if (s->ngram_ht.empty()) {
        s->ngram_ht.assign(kNgramHashSize, 0xFFFFFFFFU);
    }
    const size_t base = s->ngram_hist.size();
    s->ngram_hist.insert(s->ngram_hist.end(), toks, toks + n);
    const uint32_t* h   = s->ngram_hist.data();
    const size_t    end = s->ngram_hist.size();
    size_t          i   = (base >= 4U) ? base : 4U;
    for (; i < end; ++i) {
        const uint32_t bucket = sqf_ngram5_hash(h[i - 4U], h[i - 3U], h[i - 2U], h[i - 1U], h[i]);
        s->ngram_ht[bucket]   = static_cast<uint32_t>(i);
    }
}

inline void sqf_ngram_rewind(StrataQwenFlashContextSlot* s, uint32_t new_len) {
    if (!s) return;
    if (new_len < s->ngram_hist.size()) {
        s->ngram_hist.resize(new_len);
    }
    if (!s->ngram_ht.empty()) {
        std::fill(s->ngram_ht.begin(), s->ngram_ht.end(), 0xFFFFFFFFU);
        const uint32_t* h   = s->ngram_hist.data();
        const size_t    end = s->ngram_hist.size();
        for (size_t i = 4U; i < end; ++i) {
            const uint32_t bucket = sqf_ngram5_hash(h[i - 4U], h[i - 3U], h[i - 2U], h[i - 1U], h[i]);
            s->ngram_ht[bucket]   = static_cast<uint32_t>(i);
        }
    }
    s->ngram_cooldown = 0U;
}

inline uint32_t sqf_ngram_propose(
    const StrataQwenFlashContextSlot* s,
    uint32_t                          pending_tok,
    uint32_t                          max_k,
    uint32_t*                         out_drafts
) {
    if (!s || !out_drafts || max_k == 0U || s->ngram_cooldown > 0U || s->ngram_ht.empty()) return 0U;
    const size_t sz = s->ngram_hist.size();
    if (sz < 5U) return 0U;
    const uint32_t* h = s->ngram_hist.data();
    const uint32_t  a = h[sz - 4U];
    const uint32_t  b = h[sz - 3U];
    const uint32_t  c = h[sz - 2U];
    const uint32_t  d = h[sz - 1U];
    const uint32_t  e = pending_tok;
    const uint32_t  pos = s->ngram_ht[sqf_ngram5_hash(a, b, c, d, e)];
    if (pos == 0xFFFFFFFFU || pos < 4U || static_cast<size_t>(pos) + 1U >= sz) return 0U;
    if (h[pos - 4U] != a || h[pos - 3U] != b || h[pos - 2U] != c || h[pos - 1U] != d || h[pos] != e) return 0U;

    const uint32_t avail = static_cast<uint32_t>(sz - 1U - static_cast<size_t>(pos));
    const uint32_t min_k = std::min<uint32_t>(max_k, 3U);
    if (avail < min_k) return 0U;
    const uint32_t eff_k = std::min<uint32_t>(max_k, avail);
    for (uint32_t k = 0U; k < eff_k; ++k) {
        out_drafts[k] = h[static_cast<size_t>(pos) + 1U + k];
    }
    return eff_k;
}

} // namespace

struct SqfAneModelBundle {
    MLModel*                  model     = nil;
    MLMultiArray*             in_arr    = nil;
    MLMultiArray*             out_arr   = nil;
    SqfAneQsaFeatureProvider* fp        = nil;
    MLPredictionOptions*      opts      = nil;
    uint16_t*                 in_buf    = nullptr;
    uint16_t*                 out_buf   = nullptr;
    uint32_t                  in_ch     = 0U;
    uint32_t                  out_ch    = 0U;
    uint32_t                  layer_base= 0U;
    uint32_t                  n_layers  = 0U;
};

struct SqfAneQsaProbe {
    bool                           ready     = false;
    std::string                    note;
    std::vector<SqfAneModelBundle> bundles;
    uint32_t                       n_layers  = 0U;
    uint32_t                       D         = 0U;
    uint32_t                       n_heads   = 0U;
    uint32_t                       head_dim  = 0U;
    uint32_t                       rot_dim   = 64U;
    uint32_t                       ane_ops   = 0U;
    uint32_t                       total_ops = 0U;
    long                           ane_cores = 0;
    float                          eps       = 1e-6f;
    std::vector<float>             q_norm;   /* [n_layers][head_dim] */
    std::vector<float>             inv_freq; /* [rot_dim / 2] */
    std::vector<uint16_t>          x_snap;   /* [n_layers][D] */
    std::vector<float>             q_hat;    /* [n_layers][n_heads * head_dim] */
    dispatch_queue_t               queue     = nil;
    double                         init_ms   = 0.0;

    ~SqfAneQsaProbe() {
        for (auto& b : bundles) {
            b.opts = nil; b.fp = nil; b.in_arr = nil; b.out_arr = nil; b.model = nil;
            if (b.in_buf) { free(b.in_buf); b.in_buf = nullptr; }
            if (b.out_buf) { free(b.out_buf); b.out_buf = nullptr; }
        }
    }
};

namespace {

/* ---- Minimal CoreML spec-v7 protobuf writer (same proven builder pattern as src/hw/strata_ane_probe.mm) ---- */
struct SqfPb {
    std::vector<uint8_t> buf;
    void varint(uint64_t v) {
        while (v >= 0x80ULL) { buf.push_back(static_cast<uint8_t>((v & 0x7FULL) | 0x80ULL)); v >>= 7U; }
        buf.push_back(static_cast<uint8_t>(v));
    }
    void tag(uint32_t f, uint32_t wt) { varint((static_cast<uint64_t>(f) << 3U) | wt); }
    void u64(uint32_t f, uint64_t v) { tag(f, 0U); varint(v); }
    void bytes(uint32_t f, const void* d, size_t n) {
        tag(f, 2U); varint(n);
        const auto* p = static_cast<const uint8_t*>(d);
        buf.insert(buf.end(), p, p + n);
    }
    void str(uint32_t f, const char* s) { bytes(f, s, std::strlen(s)); }
    void msg(uint32_t f, const SqfPb& m) { bytes(f, m.buf.data(), m.buf.size()); }
};

SqfPb sqf_pb_shape4(uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    SqfPb t; t.u64(1U, 4ULL); t.u64(2U, n); t.u64(2U, c); t.u64(2U, h); t.u64(2U, w); return t;
}

SqfPb sqf_pb_feature_fp16(const char* name, uint32_t c) {
    SqfPb aft; aft.u64(1U, 1U); aft.u64(1U, c); aft.u64(1U, 1U); aft.u64(1U, kAneSpatial); aft.u64(2U, kCoreMLArrayFloat16);
    SqfPb ft; ft.msg(5U, aft);
    SqfPb fd; fd.str(1U, name); fd.msg(3U, ft);
    return fd;
}

/* `y = GroupedConv1x1(x)` with `groups` independent `[out_ch/groups, in_ch/groups]` FP16 projections. */
std::vector<uint8_t> sqf_serialize_qsa_probe_model(uint32_t in_ch, uint32_t out_ch, uint32_t groups, const uint16_t* w) {
    SqfPb wp; wp.bytes(2U, w, static_cast<size_t>(out_ch) * (in_ch / groups) * sizeof(uint16_t));
    SqfPb border; SqfPb valid; valid.msg(1U, border);
    SqfPb conv;
    conv.u64(1U, out_ch);
    conv.u64(2U, in_ch / groups);
    conv.u64(10U, groups);
    conv.u64(20U, 1ULL); conv.u64(20U, 1ULL);
    conv.u64(30U, 1ULL); conv.u64(30U, 1ULL);
    conv.u64(40U, 1ULL); conv.u64(40U, 1ULL);
    conv.msg(50U, valid);
    conv.u64(60U, 0ULL);
    conv.u64(70U, 0ULL);
    conv.msg(90U, wp);
    SqfPb layer;
    layer.str(1U, "qsa_qidx_lookahead");
    layer.str(2U, "x");
    layer.str(3U, "y");
    layer.msg(4U, sqf_pb_shape4(1U, in_ch, 1U, kAneSpatial));
    layer.msg(5U, sqf_pb_shape4(1U, out_ch, 1U, kAneSpatial));
    layer.msg(100U, conv);
    SqfPb desc; desc.msg(1U, sqf_pb_feature_fp16("x", in_ch)); desc.msg(10U, sqf_pb_feature_fp16("y", out_ch));
    SqfPb nn; nn.msg(1U, layer); nn.u64(5U, 1ULL);
    SqfPb model; model.u64(1U, kCoreMLSpecV7); model.msg(2U, desc); model.msg(500U, nn);
    return model.buf;
}

struct SqfScopedUrl {
    NSURL* url = nil;
    ~SqfScopedUrl() { reset(); }
    void reset() { if (url) { [[NSFileManager defaultManager] removeItemAtURL:url error:nil]; url = nil; } }
};

/* Compiles + MLComputePlan-verifies + loads one probe model; returns true iff 100% of ops prefer the ANE. */
bool sqf_ane_build_bundle(uint32_t in_ch, uint32_t out_ch, uint32_t groups, const uint16_t* w,
                          SqfAneModelBundle* out, uint32_t* ane_ops, uint32_t* total_ops, std::string* err) {
    @autoreleasepool {
        static std::atomic<uint64_t> s_nonce{1ULL};
        std::vector<uint8_t> pb = sqf_serialize_qsa_probe_model(in_ch, out_ch, groups, w);
        std::string tmp_root([NSTemporaryDirectory() UTF8String]);
        if (!tmp_root.empty() && tmp_root.back() != '/') tmp_root.push_back('/');
        std::string dir_tmpl = tmp_root + "strata_qsa_probe_XXXXXX";
        std::vector<char> dir_buf(dir_tmpl.begin(), dir_tmpl.end());
        dir_buf.push_back('\0');
        char* staged_dir = ::mkdtemp(dir_buf.data());
        if (!staged_dir) { *err = "mkdtemp failed"; return false; }
        NSURL* dir_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:staged_dir]];
        SqfScopedUrl src_guard{dir_url};
        NSString* fname = [NSString stringWithFormat:@"probe_%llu.mlmodel",
                           (unsigned long long)s_nonce.fetch_add(1ULL)];
        NSURL* src = [dir_url URLByAppendingPathComponent:fname];
        NSData* data = [NSData dataWithBytesNoCopy:pb.data() length:pb.size() freeWhenDone:NO];
        if (![data writeToURL:src options:NSDataWritingAtomic error:nil]) { *err = "stage .mlmodel failed"; return false; }

        __block NSURL* compiled = nil;
        __block NSError* cerr = nil;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [MLModel compileModelAtURL:src completionHandler:^(NSURL* _Nullable u, NSError* _Nullable e) {
            compiled = u; cerr = e; dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        src_guard.reset();
        if (!compiled) {
            *err = std::string("CoreML compile failed: ") + (cerr ? [[cerr localizedDescription] UTF8String] : "nil");
            return false;
        }
        SqfScopedUrl comp_guard{compiled};

        MLModelConfiguration* mcfg = [[MLModelConfiguration alloc] init];
        mcfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        __block MLComputePlan* plan = nil;
        __block NSError* perr = nil;
        dispatch_semaphore_t psem = dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:compiled configuration:mcfg completionHandler:^(MLComputePlan* _Nullable cp, NSError* _Nullable e) {
            plan = cp; perr = e; dispatch_semaphore_signal(psem);
        }];
        dispatch_semaphore_wait(psem, DISPATCH_TIME_FOREVER);
        if (!plan || !plan.modelStructure.neuralNetwork) {
            *err = std::string("MLComputePlan failed: ") + (perr ? [[perr localizedDescription] UTF8String] : "no NN");
            return false;
        }
        uint32_t ane = 0U, tot = 0U;
        for (MLModelStructureNeuralNetworkLayer* layer in plan.modelStructure.neuralNetwork.layers) {
            ++tot;
            MLComputePlanDeviceUsage* usage = [plan computeDeviceUsageForNeuralNetworkLayer:layer];
            if (usage && [usage.preferredComputeDevice isKindOfClass:[MLNeuralEngineComputeDevice class]]) ++ane;
        }
        *ane_ops += ane;
        *total_ops += tot;
        if (tot == 0U || ane != tot) {
            *err = "MLComputePlan preferred non-ANE device (" + std::to_string(ane) + "/" + std::to_string(tot) + " ops on ANE)";
            return false;
        }

        NSError* lerr = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:compiled configuration:mcfg error:&lerr];
        comp_guard.reset();
        if (!model) {
            *err = std::string("MLModel load failed: ") + (lerr ? [[lerr localizedDescription] UTF8String] : "nil");
            return false;
        }

        const size_t in_bytes  = sqf_align_up_u64(static_cast<uint64_t>(in_ch) * kAneSpatial * 2ULL, STRATA_QWEN_FLASH_PAGE_BYTES);
        const size_t out_bytes = sqf_align_up_u64(static_cast<uint64_t>(out_ch) * kAneSpatial * 2ULL, STRATA_QWEN_FLASH_PAGE_BYTES);
        void* ib = nullptr;
        void* ob = nullptr;
        if (posix_memalign(&ib, STRATA_QWEN_FLASH_PAGE_BYTES, in_bytes) != 0 ||
            posix_memalign(&ob, STRATA_QWEN_FLASH_PAGE_BYTES, out_bytes) != 0) {
            if (ib) free(ib);
            if (ob) free(ob);
            *err = "posix_memalign failed";
            return false;
        }
        std::memset(ib, 0, in_bytes);
        std::memset(ob, 0, out_bytes);
        out->in_buf  = static_cast<uint16_t*>(ib);
        out->out_buf = static_cast<uint16_t*>(ob);
        NSError* aerr = nil;
        out->in_arr = [[MLMultiArray alloc] initWithDataPointer:ib
                                                          shape:@[@1, @(in_ch), @1, @(kAneSpatial)]
                                                       dataType:MLMultiArrayDataTypeFloat16
                                                        strides:@[@(in_ch * kAneSpatial), @(kAneSpatial), @(kAneSpatial), @1]
                                                    deallocator:nil
                                                          error:&aerr];
        out->out_arr = [[MLMultiArray alloc] initWithDataPointer:ob
                                                           shape:@[@1, @(out_ch), @1, @(kAneSpatial)]
                                                        dataType:MLMultiArrayDataTypeFloat16
                                                         strides:@[@(out_ch * kAneSpatial), @(kAneSpatial), @(kAneSpatial), @1]
                                                     deallocator:nil
                                                           error:&aerr];
        if (!out->in_arr || !out->out_arr) { *err = "MLMultiArray initWithDataPointer failed"; return false; }
        out->fp   = [[SqfAneQsaFeatureProvider alloc] initWithArray:out->in_arr];
        out->opts = [[MLPredictionOptions alloc] init];
        out->opts.outputBackings = @{@"y": out->out_arr};
        out->model  = model;
        out->in_ch  = in_ch;
        out->out_ch = out_ch;

        NSError* werr = nil;
        id<MLFeatureProvider> warm = [model predictionFromFeatures:out->fp options:out->opts error:&werr];
        if (!warm || werr) {
            *err = std::string("ANE warmup prediction failed: ") + (werr ? [[werr localizedDescription] UTF8String] : "nil");
            out->model = nil;
            return false;
        }
        return true;
    }
}

/* Builds the engine's 16-core ANE query-lookahead probe: `q_idx_hat[l] = W_idx_q[l] @ x_attn[l]` for all 12 QSA
 * layers (the EXACT `indexer.index_qk_proj` q-rows, FP16) in ONE grouped 1x1 conv (12 groups x 2560 -> 512). */
void sqf_ane_probe_init(StrataQwenFlashGpuEngine* eng) {
    if (eng->ane_init_tried) return;
    eng->ane_init_tried = true;
    const auto t0 = std::chrono::steady_clock::now();
    auto* pr = new SqfAneQsaProbe();
    pr->n_layers = eng->cfg.num_full_attn_layers;
    pr->D        = eng->cfg.hidden_size;
    pr->n_heads  = eng->cfg.indexer_n_heads;
    pr->head_dim = eng->cfg.indexer_head_dim;
    pr->rot_dim  = eng->cfg.rotary_dim;
    pr->eps      = eng->cfg.rms_norm_eps;
    pr->queue    = dispatch_queue_create("strata.qsa.ane_lookahead", DISPATCH_QUEUE_SERIAL);
    const uint32_t q_rows = pr->n_heads * pr->head_dim;
    pr->q_norm.assign(static_cast<size_t>(pr->n_layers) * pr->head_dim, 1.0f);
    pr->x_snap.assign(static_cast<size_t>(pr->n_layers) * pr->D, 0U);
    pr->q_hat.assign(static_cast<size_t>(pr->n_layers) * q_rows, 0.0f);
    const float* inv_f = reinterpret_cast<const float*>([eng->b_inv_freq contents]);
    pr->inv_freq.assign(inv_f, inv_f + pr->rot_dim / 2U);

    @autoreleasepool {
        for (id<MLComputeDeviceProtocol> dev in MLAllComputeDevices()) {
            if ([dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                pr->ane_cores = static_cast<MLNeuralEngineComputeDevice*>(dev).totalCoreCount;
            }
        }
    }

    std::vector<uint16_t> w(static_cast<size_t>(pr->n_layers) * q_rows * pr->D, 0U);
    uint32_t fl = 0U;
    for (uint32_t l = 0U; l < eng->cfg.num_hidden_layers && fl < pr->n_layers; ++l) {
        const StrataQwenFlashLayerDesc& ld = (l < eng->cfg.split_layer_idx) ? eng->hdr0.layers[l]
                                                                            : eng->hdr1.layers[l - eng->cfg.split_layer_idx];
        if (ld.layer_type != STRATA_QWEN_FLASH_LAYER_FULL_ATTN) continue;
        const uint8_t* base = strata_qwen_flash_part_base(eng->bundle, ld.part_id);
        std::memcpy(&w[static_cast<size_t>(fl) * q_rows * pr->D], base + ld.indexer_qk_proj.offset,
                    static_cast<size_t>(q_rows) * pr->D * sizeof(uint16_t));
        std::memcpy(&pr->q_norm[static_cast<size_t>(fl) * pr->head_dim], base + ld.indexer_q_norm.offset,
                    static_cast<size_t>(pr->head_dim) * sizeof(float));
        ++fl;
    }

    std::string err;
    SqfAneModelBundle grouped{};
    grouped.layer_base = 0U;
    grouped.n_layers   = pr->n_layers;
    if (pr->ane_cores > 0 &&
        sqf_ane_build_bundle(pr->n_layers * pr->D, pr->n_layers * q_rows, pr->n_layers, w.data(),
                             &grouped, &pr->ane_ops, &pr->total_ops, &err)) {
        pr->bundles.push_back(grouped);
        pr->ready = true;
    } else {
        /* Fallback: one verified 2560 -> 512 conv model per QSA layer */
        if (grouped.in_buf) { free(grouped.in_buf); grouped.in_buf = nullptr; }
        if (grouped.out_buf) { free(grouped.out_buf); grouped.out_buf = nullptr; }
        const std::string grouped_err = err;
        pr->ane_ops = 0U;
        pr->total_ops = 0U;
        bool all_ok = (pr->ane_cores > 0);
        for (uint32_t l = 0U; l < pr->n_layers && all_ok; ++l) {
            SqfAneModelBundle b{};
            b.layer_base = l;
            b.n_layers   = 1U;
            if (!sqf_ane_build_bundle(pr->D, q_rows, 1U, &w[static_cast<size_t>(l) * q_rows * pr->D],
                                      &b, &pr->ane_ops, &pr->total_ops, &err)) {
                if (b.in_buf) free(b.in_buf);
                if (b.out_buf) free(b.out_buf);
                all_ok = false;
                break;
            }
            pr->bundles.push_back(b);
        }
        pr->ready = all_ok;
        if (!all_ok) {
            pr->note = "ANE probe unavailable (grouped: " + grouped_err + "; per-layer: " + err + ")";
        }
    }
    if (pr->ready) {
        /* Warm the ANE model asynchronously on a real `pr->queue` GCD worker thread AND exercise
         * `[outF featureValueForName:@"y"].multiArrayValue` + `QOS_CLASS_USER_INTERACTIVE` worker pool
         * with zeroed `in_buf` (without touching any slot stats) so CoreML/ANE/XPC/GCD pages are 100%
         * resident before steady-state fault tracking. */
        dispatch_group_t wgrp = dispatch_group_create();
        dispatch_group_async(wgrp, pr->queue, ^{
            @autoreleasepool {
                for (SqfAneModelBundle& bdl : pr->bundles) {
                    if (bdl.in_buf) {
                        std::memset(bdl.in_buf, 0, static_cast<size_t>(bdl.in_ch) * kAneSpatial * sizeof(uint16_t));
                    }
                    NSError* werr = nil;
                    id<MLFeatureProvider> outF = [bdl.model predictionFromFeatures:bdl.fp options:bdl.opts error:&werr];
                    if (outF && !werr) {
                        MLMultiArray* oa = [outF featureValueForName:@"y"].multiArrayValue;
                        if (oa && oa != bdl.out_arr) {
                            [oa getBytesWithHandler:^(const void* bytes, NSInteger size) {
                                volatile uint8_t touch = (bytes && size > 0) ? static_cast<const uint8_t*>(bytes)[0] : 0U;
                                (void)touch;
                            }];
                        } else if (bdl.out_buf) {
                            volatile uint16_t touch = bdl.out_buf[0];
                            (void)touch;
                        }
                    }
                }
                dispatch_apply(pr->n_layers, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t li) {
                    (void)li;
                });
            }
        });
        dispatch_group_wait(wgrp, DISPATCH_TIME_FOREVER);
    }
    pr->init_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (pr->ready) {
        char nb[384];
        std::snprintf(nb, sizeof(nb),
                      "CoreML spec-v7 MLComputePlan-verified %ld-core ANE (MLNeuralEngineComputeDevice %u/%u ops, %zu model%s, "
                      "%u QSA layers x %u->%u FP16 index_qk_proj q-rows, init %.0f ms)",
                      pr->ane_cores, pr->ane_ops, pr->total_ops, pr->bundles.size(), pr->bundles.size() == 1U ? "" : "s",
                      pr->n_layers, pr->D, q_rows, pr->init_ms);
        pr->note = nb;
    }
    eng->ane = pr;
}

/* ---- Physical pool addressing: K/V pools are `[13][n_kv][pool_tokens][head_dim]` FP16 (exact linear layout) ---- */
inline size_t sqf_pool_chunk_off(const StrataQwenFlashContextSlot* s, uint32_t ls, uint32_t h, uint32_t phys) {
    const StrataQwenFlashGpuEngine* eng = s->eng;
    return static_cast<size_t>(ls) * s->kv_layer_stride +
           (static_cast<size_t>(h) * s->pool_tokens + static_cast<size_t>(phys) * STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) *
               eng->cfg.head_dim * sizeof(uint16_t);
}

inline size_t sqf_paged_block_bytes(const StrataQwenFlashContextSlot* s) {
    const StrataQwenFlashGpuEngine* eng = s->eng;
    return static_cast<size_t>(2U * eng->cfg.num_key_value_heads * STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) *
           eng->cfg.head_dim * sizeof(uint16_t);
}

inline void sqf_paged_pack_block(const StrataQwenFlashContextSlot* s, uint32_t ls, uint32_t phys, uint8_t* dst) {
    const StrataQwenFlashGpuEngine* eng = s->eng;
    const uint32_t n_kv  = eng->cfg.num_key_value_heads;
    const size_t   chunk = static_cast<size_t>(STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) * eng->cfg.head_dim * sizeof(uint16_t);
    const uint8_t* kb = static_cast<const uint8_t*>([s->b_k_cache contents]);
    const uint8_t* vb = static_cast<const uint8_t*>([s->b_v_cache contents]);
    for (uint32_t h = 0U; h < n_kv; ++h) {
        const size_t off = sqf_pool_chunk_off(s, ls, h, phys);
        std::memcpy(dst + static_cast<size_t>(h) * chunk, kb + off, chunk);
        std::memcpy(dst + static_cast<size_t>(n_kv + h) * chunk, vb + off, chunk);
    }
}

inline void sqf_paged_unpack_block(StrataQwenFlashContextSlot* s, uint32_t ls, uint32_t phys, const uint8_t* src) {
    const StrataQwenFlashGpuEngine* eng = s->eng;
    const uint32_t n_kv  = eng->cfg.num_key_value_heads;
    const size_t   chunk = static_cast<size_t>(STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) * eng->cfg.head_dim * sizeof(uint16_t);
    uint8_t* kb = static_cast<uint8_t*>([s->b_k_cache contents]);
    uint8_t* vb = static_cast<uint8_t*>([s->b_v_cache contents]);
    for (uint32_t h = 0U; h < n_kv; ++h) {
        const size_t off = sqf_pool_chunk_off(s, ls, h, phys);
        std::memcpy(kb + off, src + static_cast<size_t>(h) * chunk, chunk);
        std::memcpy(vb + off, src + static_cast<size_t>(n_kv + h) * chunk, chunk);
    }
}

/*
 * Coalesced + multi-queue `F_NOCACHE` NVMe batch mover:
 *   - Packs/unpacks 4x2 KiB K/V head slices into 8 KiB-aligned `stage_buf` (avoiding sub-4 KiB APFS F_NOCACHE RMW).
 *   - Sorts `(logical_b, phys_b)` ascending by `logical_b`, merges adjacent logical blocks into contiguous
 *     multi-page extents, and dispatches disjoint `pwrite` / `pread` syscalls in parallel across ANS Gen5 NVMe.
 */
bool sqf_paged_io_batch(StrataQwenFlashContextSlot* s, uint32_t ls,
                        std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                        bool write, uint8_t* stage_buf, uint32_t stage_cap_blocks, SqfPagedCounters& c) {
    if (pairs.empty()) return true;
    if (!stage_buf || stage_cap_blocks == 0U) { c.err = true; return false; }
    if (pairs.size() > 1U) {
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    const size_t blk_bytes = sqf_paged_block_bytes(s);
    std::vector<SqfPagedIoRun>& runs = s->layers[ls].scratch_io_runs;

    for (size_t base = 0U; base < pairs.size(); base += stage_cap_blocks) {
        const uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(stage_cap_blocks, pairs.size() - base));
        if (write) {
            for (uint32_t i = 0U; i < cnt; ++i) {
                sqf_paged_pack_block(s, ls, pairs[base + i].second, stage_buf + static_cast<size_t>(i) * blk_bytes);
            }
        }
        runs.clear();
        uint32_t r0 = 0U;
        while (r0 < cnt) {
            uint32_t rlen = 1U;
            while (r0 + rlen < cnt && pairs[base + r0 + rlen].first == pairs[base + r0].first + rlen) {
                ++rlen;
            }
            const off_t foff = static_cast<off_t>(
                (static_cast<uint64_t>(ls) * s->max_blocks + pairs[base + r0].first) * blk_bytes);
            runs.push_back(SqfPagedIoRun{r0, rlen, foff});
            r0 += rlen;
        }

        std::atomic<bool> io_fail{false};
        std::atomic<bool>* fail_ptr = &io_fail;
        const SqfPagedIoRun* run_ptr = runs.data();
        const int fd = s->swap_fd;
        const uint64_t t0 = mach_absolute_time();
        if (runs.size() == 1U) {
            const size_t nbytes = static_cast<size_t>(runs[0].len) * blk_bytes;
            const ssize_t got = write ? ::pwrite(fd, stage_buf, nbytes, runs[0].foff)
                                      : ::pread(fd, stage_buf, nbytes, runs[0].foff);
            if (got != static_cast<ssize_t>(nbytes)) io_fail.store(true, std::memory_order_relaxed);
        } else {
            dispatch_apply(runs.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t ri) {
                const size_t nbytes = static_cast<size_t>(run_ptr[ri].len) * blk_bytes;
                uint8_t* ptr = stage_buf + static_cast<size_t>(run_ptr[ri].idx0) * blk_bytes;
                const ssize_t got = write ? ::pwrite(fd, ptr, nbytes, run_ptr[ri].foff)
                                          : ::pread(fd, ptr, nbytes, run_ptr[ri].foff);
                if (got != static_cast<ssize_t>(nbytes)) fail_ptr->store(true, std::memory_order_relaxed);
            });
        }
        const double ms = sqf_ticks_ms(mach_absolute_time() - t0);
        if (io_fail.load(std::memory_order_relaxed)) {
            c.err = true;
            return false;
        }
        if (!write) {
            for (uint32_t i = 0U; i < cnt; ++i) {
                sqf_paged_unpack_block(s, ls, pairs[base + i].second, stage_buf + static_cast<size_t>(i) * blk_bytes);
            }
        }
        const uint64_t bytes_done = static_cast<uint64_t>(cnt) * blk_bytes;
        if (write) { c.paged_out += cnt; c.wbytes += bytes_done; c.wms += ms; }
        else       { c.recalled  += cnt; c.rbytes += bytes_done; c.rms += ms; }
    }
    return true;
}

void sqf_paged_zero_phys(StrataQwenFlashContextSlot* s, uint32_t ls, uint32_t phys) {
    const StrataQwenFlashGpuEngine* eng = s->eng;
    const size_t chunk = static_cast<size_t>(STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) * eng->cfg.head_dim * sizeof(uint16_t);
    uint8_t* kb = static_cast<uint8_t*>([s->b_k_cache contents]);
    uint8_t* vb = static_cast<uint8_t*>([s->b_v_cache contents]);
    for (uint32_t h = 0U; h < eng->cfg.num_key_value_heads; ++h) {
        const size_t off = sqf_pool_chunk_off(s, ls, h, phys);
        std::memset(kb + off, 0, chunk);
        std::memset(vb + off, 0, chunk);
    }
}

inline uint32_t* sqf_loc_row(StrataQwenFlashContextSlot* s, uint32_t ls) {
    return static_cast<uint32_t*>([s->b_block_loc contents]) + static_cast<size_t>(ls) * s->max_blocks;
}

inline uint32_t sqf_next_pin_epoch(SqfPagedLayerState& L) {
    if (++L.pin_epoch_ctr == 0U) {
        std::fill(L.pin_epoch.begin(), L.pin_epoch.end(), 0U);
        L.pin_epoch_ctr = 1U;
    }
    return L.pin_epoch_ctr;
}

uint32_t sqf_paged_take_phys(StrataQwenFlashContextSlot* s, SqfPagedLayerState& L, uint32_t b) {
    if (b < s->pool_blocks && L.phys_owner[b] == kPhysFree) return b; /* identity preference (dense NAX path) */
    while (!L.free_list.empty()) {
        const uint32_t p = L.free_list.back();
        L.free_list.pop_back();
        if (L.phys_owner[p] == kPhysFree) return p;
    }
    return kPhysFree;
}

void sqf_paged_release(StrataQwenFlashContextSlot* s, SqfPagedLayerState& L, uint32_t ls, uint32_t b, uint8_t new_state) {
    uint32_t* loc = sqf_loc_row(s, ls);
    const uint8_t old_state = L.st[b];
    if (old_state == kBlkResident) {
        const uint32_t phys = loc[b] & STRATA_QWEN_FLASH_KV_LOC_IDX_MASK;
        if (phys < s->pool_blocks) {
            L.phys_owner[phys] = kPhysFree;
            L.free_list.push_back(phys);
        }
        if (L.resident > 0U) --L.resident;
    } else if (old_state == kBlkPaged) {
        if (L.paged_count > 0U) --L.paged_count;
    }
    if (new_state == kBlkPaged) {
        ++L.paged_count;
    }
    L.st[b] = new_state;
    loc[b]  = (new_state == kBlkPaged) ? (b & STRATA_QWEN_FLASH_KV_LOC_IDX_MASK) : 0U;
}

/* Evicts up to `count` lowest-importance resident blocks (never pinned; never sinks / local window unless
 * `allow_protected`). Dirty blocks are written to `kv_swap` first (clean re-evictions are a 0 B flip). */
uint32_t sqf_paged_evict_staged(StrataQwenFlashContextSlot* s, uint32_t ls, uint32_t count, uint32_t newest_blk,
                                bool allow_protected, uint8_t* stage_buf, uint32_t stage_cap, SqfPagedCounters& c) {
    if (count == 0U) return 0U;
    SqfPagedLayerState& L = s->layers[ls];
    const uint32_t sink   = STRATA_QWEN_FLASH_COMPACT_SINK_BLOCKS;
    const uint32_t local  = STRATA_QWEN_FLASH_COMPACT_LOCAL_BLOCKS;
    const uint32_t lim    = std::min<uint32_t>(s->max_blocks, newest_blk + 1U);
    uint32_t* loc         = sqf_loc_row(s, ls);

    if (count == 1U) {
        uint32_t best_b = kPhysFree;
        float    best_u = 0.0f;
        for (uint32_t p = 0U; p < s->pool_blocks; ++p) {
            const uint32_t b = L.phys_owner[p];
            if (b == kPhysFree || L.pin_epoch[b] == L.pin_epoch_ctr) continue;
            if (!allow_protected && (b < sink || (b < lim && b + local > newest_blk))) continue;
            const float ub = L.u[b];
            if (best_b == kPhysFree || ub < best_u || (ub == best_u && b < best_b)) {
                best_b = b;
                best_u = ub;
            }
        }
        if (best_b == kPhysFree) return 0U;
        if (!L.on_nvme[best_b]) {
            std::vector<std::pair<uint32_t, uint32_t>>& dirty = L.scratch_io_pairs;
            dirty.clear();
            dirty.emplace_back(best_b, loc[best_b] & STRATA_QWEN_FLASH_KV_LOC_IDX_MASK);
            if (!sqf_paged_io_batch(s, ls, dirty, /*write=*/true, stage_buf, stage_cap, c)) return 0U;
            L.on_nvme[best_b] = 1U;
        }
        sqf_paged_release(s, L, ls, best_b, kBlkPaged);
        return 1U;
    }

    std::vector<uint32_t>& cand = L.scratch_cand;
    cand.clear();
    for (uint32_t p = 0U; p < s->pool_blocks; ++p) {
        const uint32_t b = L.phys_owner[p];
        if (b == kPhysFree || L.pin_epoch[b] == L.pin_epoch_ctr) continue;
        if (!allow_protected && (b < sink || (b < lim && b + local > newest_blk))) continue;
        cand.push_back(b);
    }
    const size_t k = std::min<size_t>(count, cand.size());
    if (k == 0U) return 0U;
    auto cmp = [&](uint32_t a, uint32_t b2) { return (L.u[a] < L.u[b2]) || (L.u[a] == L.u[b2] && a < b2); };
    if (k < cand.size()) std::nth_element(cand.begin(), cand.begin() + static_cast<ptrdiff_t>(k), cand.end(), cmp);

    std::vector<std::pair<uint32_t, uint32_t>>& dirty = L.scratch_io_pairs;
    dirty.clear();
    for (size_t i = 0U; i < k; ++i) {
        const uint32_t b = cand[i];
        if (!L.on_nvme[b]) {
            dirty.emplace_back(b, loc[b] & STRATA_QWEN_FLASH_KV_LOC_IDX_MASK);
        }
    }
    if (!dirty.empty()) {
        if (!sqf_paged_io_batch(s, ls, dirty, /*write=*/true, stage_buf, stage_cap, c)) return 0U;
        for (const auto& pr : dirty) L.on_nvme[pr.first] = 1U;
    }
    for (size_t i = 0U; i < k; ++i) {
        sqf_paged_release(s, L, ls, cand[i], kBlkPaged);
    }
    return static_cast<uint32_t>(k);
}

/* Makes every block in `need` (already stamped with the current pin epoch) UMA-resident. */
bool sqf_paged_ensure_staged(StrataQwenFlashContextSlot* s, uint32_t ls, const std::vector<uint32_t>& need,
                             uint32_t newest_blk, bool exact_check,
                             uint8_t* stage_buf, uint32_t stage_cap, SqfPagedCounters& c) {
    SqfPagedLayerState& L = s->layers[ls];
    uint32_t* loc = sqf_loc_row(s, ls);
    std::vector<uint32_t>& missing = L.scratch_missing;
    missing.clear();
    for (uint32_t b : need) {
        if (b >= s->max_blocks) continue;
        if (L.st[b] == kBlkResident) {
            if (exact_check && L.pf_epoch[b] == s->step_epoch) {
                c.pf_hits += 1ULL;
                L.pf_epoch[b] = 0U;
            }
        } else {
            missing.push_back(b);
        }
    }
    if (missing.empty()) return true;
    const uint32_t free_cnt = s->pool_blocks - L.resident;
    if (missing.size() > free_cnt) {
        uint32_t want = static_cast<uint32_t>(missing.size()) - free_cnt;
        /* For large UMA pools (>= 8K tokens), keep 32-block eviction headroom to amortize scans & coalesce pwrites */
        const uint32_t headroom = (exact_check && s->pool_blocks >= 2048U) ? 32U : 0U;
        const uint32_t got0 = sqf_paged_evict_staged(s, ls, want + headroom, newest_blk, false, stage_buf, stage_cap, c);
        want = (got0 >= want) ? 0U : (want - got0);
        if (want > 0U) want -= sqf_paged_evict_staged(s, ls, want, newest_blk, true, stage_buf, stage_cap, c);
        if (want > 0U || c.err) { c.err = true; return false; }
    }
    std::vector<std::pair<uint32_t, uint32_t>>& to_read = L.scratch_io_pairs;
    to_read.clear();
    for (uint32_t b : missing) {
        const uint32_t phys = sqf_paged_take_phys(s, L, b);
        if (phys == kPhysFree) { c.err = true; return false; }
        L.phys_owner[phys] = b;
        if (L.st[b] == kBlkPaged) {
            if (L.paged_count > 0U) --L.paged_count;
            if (L.on_nvme[b]) {
                to_read.emplace_back(b, phys);
            } else {
                sqf_paged_zero_phys(s, ls, phys);
                L.on_nvme[b] = 0U;
            }
        } else {
            sqf_paged_zero_phys(s, ls, phys);
            L.on_nvme[b] = 0U;
        }
        L.st[b] = kBlkResident;
        loc[b]  = STRATA_QWEN_FLASH_KV_LOC_UMA_BIT | phys;
        ++L.resident;
    }
    if (!to_read.empty()) {
        if (!sqf_paged_io_batch(s, ls, to_read, /*write=*/false, stage_buf, stage_cap, c)) return false;
        if (exact_check) c.misses += to_read.size();
    }
    return true;
}

bool sqf_paged_ensure(StrataQwenFlashContextSlot* s, uint32_t ls, const std::vector<uint32_t>& need,
                      uint32_t newest_blk, bool exact_check, SqfPagedCounters& c) {
    return sqf_paged_ensure_staged(s, ls, need, newest_blk, exact_check, s->io_stage, kMaxIoBatchBlocks, c);
}

void sqf_paged_merge(StrataQwenFlashContextSlot* s, const SqfPagedCounters& c) {
    s->stats.nvme_blocks_paged_out += c.paged_out;
    s->stats.nvme_blocks_recalled  += c.recalled;
    s->stats.nvme_bytes_written    += c.wbytes;
    s->stats.nvme_bytes_read       += c.rbytes;
    s->stats.nvme_write_ms         += c.wms;
    s->stats.nvme_read_ms          += c.rms;
    s->stats.sync_recall_misses    += c.misses;
    s->stats.ane_prefetch_issued   += c.pf_issued;
    s->stats.ane_prefetch_hits     += c.pf_hits;
    if (c.err) s->eng->paged_error = true;
}

void sqf_paged_join_probe(StrataQwenFlashContextSlot* s) {
    if (s->probe_pending && s->probe_grp) {
        dispatch_group_wait(s->probe_grp, DISPATCH_TIME_FOREVER);
        s->probe_pending = false;
        sqf_paged_merge(s, s->probe_out_counters);
        s->stats.ane_probe_calls += 1ULL;
        if (s->probe_out_on_ane) {
            s->stats.ane_probe_on_ane += 1ULL;
            s->stats.ran_on_ane = 1U;
        }
        s->ane_us_total += s->probe_out_us;
        s->probe_out_counters = SqfPagedCounters{};
        s->probe_out_on_ane   = false;
        s->probe_out_us       = 0.0;
    }
}

void sqf_paged_reset_tables(StrataQwenFlashContextSlot* s) {
    if (!s->paged) return;
    sqf_paged_join_probe(s);
    const uint32_t pl = s->paged_layers;
    for (uint32_t ls = 0U; ls < pl; ++ls) {
        SqfPagedLayerState& L = s->layers[ls];
        std::fill(L.st.begin(), L.st.end(), kBlkUnalloc);
        std::fill(L.on_nvme.begin(), L.on_nvme.end(), 0U);
        std::fill(L.phys_owner.begin(), L.phys_owner.end(), kPhysFree);
        std::fill(L.u.begin(), L.u.end(), 0.0f);
        std::fill(L.pin_epoch.begin(), L.pin_epoch.end(), 0U);
        std::fill(L.pf_epoch.begin(), L.pf_epoch.end(), 0U);
        L.free_list.resize(s->pool_blocks);
        for (uint32_t i = 0U; i < s->pool_blocks; ++i) L.free_list[i] = s->pool_blocks - 1U - i;
        L.pin_epoch_ctr = 1U;
        L.resident      = 0U;
        L.paged_count   = 0U;
        L.has_thresh    = false;
        L.last_thresh   = 0.0f;
    }
    std::memset([s->b_block_loc contents], 0, static_cast<size_t>(pl) * s->max_blocks * sizeof(uint32_t));
    s->x_valid = false;
}

/* After any `seq_pos` rewind (checkpoint restore): blocks holding only rewound positions are freed (their NVMe
 * copies invalidated), and the partially valid boundary block is recalled into UMA so new writes append to it. */
void sqf_paged_on_rewind(StrataQwenFlashContextSlot* s, uint32_t seq_pos, uint32_t mtp_seq_pos) {
    if (!s->paged) return;
    sqf_paged_join_probe(s);
    SqfPagedCounters c;
    const uint32_t pl = s->paged_layers;
    for (uint32_t ls = 0U; ls < pl; ++ls) {
        SqfPagedLayerState& L = s->layers[ls];
        const uint32_t valid_pos = (!s->mtp_unpaged && ls + 1U == s->n_layers && s->eng->cfg.has_mtp) ? mtp_seq_pos : seq_pos;
        const uint32_t first_dead = (valid_pos + STRATA_QWEN_FLASH_KV_BLOCK_TOKENS - 1U) / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
        for (uint32_t b = first_dead; b < s->max_blocks; ++b) {
            if (L.st[b] != kBlkUnalloc) sqf_paged_release(s, L, ls, b, kBlkUnalloc);
            L.on_nvme[b] = 0U;
            L.u[b]       = 0.0f;
        }
        if ((valid_pos % STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) != 0U) {
            const uint32_t bb = valid_pos / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
            L.pin_epoch[bb] = sqf_next_pin_epoch(L);
            L.scratch_need.clear();
            L.scratch_need.push_back(bb);
            (void)sqf_paged_ensure(s, ls, L.scratch_need, bb, false, c);
            L.on_nvme[bb] = 0U;
        }
        L.has_thresh = false;
    }
    s->x_valid = false;
    sqf_paged_merge(s, c);
}

/*
 * Zero-Sync Fast-Path predicate for sparse QSA layers in paged/hybrid mode:
 * returns false iff layer `ls` has ZERO blocks paged out on NVMe (`L.paged_count == 0U`),
 * no in-flight ANE probe touching `ls`, and every write-target block `[b0 .. b1]` fits in
 * free UMA pool slots with ZERO evictions -> caller skips `sqf_paged_sync_point` completely!
 */
bool sqf_paged_layer_needs_sync(const StrataQwenFlashGpuEngine* eng, uint32_t ls, uint32_t pos_start,
                                uint32_t M, uint32_t comp_ratio, bool any_sparse, uint32_t tree_mode) {
    if (!any_sparse || !eng || !eng->kv_paged || !eng->bound_slot) return false;
    const StrataQwenFlashContextSlot* s = eng->bound_slot;
    if (s->mtp_unpaged && ls >= s->paged_layers) return false;
    if (tree_mode != 0U) return true;
    if (ls >= s->paged_layers || s->max_blocks == 0U) return true;
    if (s->probe_pending && ls < eng->cfg.num_full_attn_layers) return true;
    const SqfPagedLayerState& L = s->layers[ls];
    if (L.paged_count > 0U) return true;
    const uint32_t b0 = pos_start / comp_ratio;
    const uint32_t b1 = (pos_start + M - 1U) / comp_ratio;
    if (b1 >= s->max_blocks) return true;
    uint32_t need_new = 0U;
    for (uint32_t b = b0; b <= b1; ++b) {
        if (L.st[b] != kBlkResident) ++need_new;
    }
    return (L.resident + need_new) > s->pool_blocks;
}

/* Host-side block allocation for non-sparse launches AND zero-sync resident sparse launches (0 GPU stalls!). */
void sqf_paged_alloc_range(StrataQwenFlashGpuEngine* eng, uint32_t ls, uint32_t pos_start, uint32_t M) {
    StrataQwenFlashContextSlot* s = eng->bound_slot;
    if (s->max_blocks == 0U) return;
    if (ls < eng->cfg.num_full_attn_layers) sqf_paged_join_probe(s);
    SqfPagedLayerState& L = s->layers[ls];
    const uint32_t b0 = pos_start / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
    const uint32_t b1 = std::min<uint32_t>(s->max_blocks - 1U, (pos_start + M - 1U) / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS);
    if (L.paged_count == 0U) {
        uint32_t* loc = sqf_loc_row(s, ls);
        uint32_t need_new = 0U;
        for (uint32_t b = b0; b <= b1; ++b) {
            if (L.st[b] != kBlkResident) ++need_new;
        }
        if (need_new == 0U) {
            for (uint32_t b = b0; b <= b1; ++b) L.on_nvme[b] = 0U;
            return;
        }
        if (L.resident + need_new <= s->pool_blocks) {
            for (uint32_t b = b0; b <= b1; ++b) {
                if (L.st[b] != kBlkResident) {
                    const uint32_t phys = sqf_paged_take_phys(s, L, b);
                    if (phys == kPhysFree) { eng->paged_error = true; return; }
                    L.phys_owner[phys] = b;
                    sqf_paged_zero_phys(s, ls, phys);
                    L.st[b] = kBlkResident;
                    loc[b]  = STRATA_QWEN_FLASH_KV_LOC_UMA_BIT | phys;
                    ++L.resident;
                }
                L.on_nvme[b] = 0U;
            }
            return;
        }
    }
    const uint32_t pin_ep = sqf_next_pin_epoch(L);
    std::vector<uint32_t>& need = L.scratch_need;
    need.clear();
    const uint32_t req_start = (L.paged_count > 0U && b1 < s->pool_blocks) ? 0U : b0;
    for (uint32_t b = req_start; b <= b1; ++b) {
        L.pin_epoch[b] = pin_ep;
        need.push_back(b);
    }
    SqfPagedCounters c;
    (void)sqf_paged_ensure(s, ls, need, b1, false, c);
    for (uint32_t b = b0; b <= b1; ++b) L.on_nvme[b] = 0U;
    sqf_paged_merge(s, c);
}

/* Dense NAX tiles slice 32 contiguous KV rows: valid iff every live block is identity-mapped. */
bool sqf_paged_identity_ok(StrataQwenFlashGpuEngine* eng, uint32_t ls, uint32_t total_kv) {
    StrataQwenFlashContextSlot* s = eng->bound_slot;
    const uint32_t nb  = (total_kv + STRATA_QWEN_FLASH_KV_BLOCK_TOKENS - 1U) / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
    const uint32_t* loc = sqf_loc_row(s, ls);
    const uint32_t tiles_end = std::min<uint32_t>(s->pool_blocks, ((nb + 7U) / 8U) * 8U);
    if (tiles_end < nb) return false;
    for (uint32_t b = 0U; b < nb; ++b) {
        if (loc[b] != (STRATA_QWEN_FLASH_KV_LOC_UMA_BIT | b)) return false;
    }
    return true;
}

void sqf_paged_commit_wait(StrataQwenFlashGpuEngine* eng,
                           id<MTLComputeCommandEncoder> __strong& enc,
                           id<MTLCommandBuffer> __strong& cb) {
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError) {
        NSError* err = [cb error];
        std::fprintf(stderr, "[strata_qwen_flash_gpu] paged sync-point command buffer error: %s\n",
                     err ? [[err localizedDescription] UTF8String] : "unknown");
        eng->paged_error = true;
    }
    cb  = [eng->queue commandBuffer];
    enc = [cb computeCommandEncoder];
}

void sqf_paged_make_group_resident(StrataQwenFlashGpuEngine* eng, uint32_t ls, uint32_t newest_blk,
                                   const SqfMGroup& g, const SqfMGroup* next_g) {
    StrataQwenFlashContextSlot* s = eng->bound_slot;
    SqfPagedLayerState& L = s->layers[ls];
    const uint32_t pin_ep = sqf_next_pin_epoch(L);
    for (uint32_t b : g.need) L.pin_epoch[b] = pin_ep;
    if (next_g) {
        for (uint32_t b : next_g->need) if (b < s->max_blocks) L.u[b] += 1000.0f;
    }
    SqfPagedCounters c;
    (void)sqf_paged_ensure(s, ls, g.need, newest_blk, /*exact_check=*/true, c);
    if (next_g) {
        for (uint32_t b : next_g->need) if (b < s->max_blocks) L.u[b] -= 1000.0f;
    }
    sqf_paged_merge(s, c);
}

/*
 * EXACT selected-block check (sparse QSA layer `ls`): commits + waits for `QSAIndexer` top-512 selection
 * (overlapping GPU work with any in-flight ANE lookahead on layers 0..11!), updates per-block importance
 * `u = 0.98 u + score`, splits the `M` queries into groups whose union fits the UMA pool using pre-allocated
 * scratch buffers (0 per-step heap allocs), and makes group 0 resident BEFORE `sqf_qsa_qkv_prep` / attention.
 */
void sqf_paged_sync_point(StrataQwenFlashGpuEngine* eng,
                          id<MTLComputeCommandEncoder> __strong& enc,
                          id<MTLCommandBuffer> __strong& cb,
                          uint32_t ls, uint32_t pos_start, uint32_t M,
                          uint32_t block_topk, uint32_t comp_ratio,
                          size_t* num_groups_out) {
    StrataQwenFlashContextSlot* s = eng->bound_slot;
    /* Commit & wait GPU first so Layers 0..3 Indexer (and cb_draft MTP) overlap with ANE prefetch on layers 0..11 */
    sqf_paged_commit_wait(eng, enc, cb);
    if (ls < eng->cfg.num_full_attn_layers) {
        sqf_paged_join_probe(s);
        s->x_valid = true;
    }
    s->stats.paged_sync_points += 1ULL;

    SqfPagedLayerState& L = s->layers[ls];
    const uint32_t newest_blk = (pos_start + M - 1U) / comp_ratio;
    const uint32_t* sel = static_cast<const uint32_t*>([eng->b_idx_sel_blocks contents]);

    /* 1. Importance EWMA + exact 512th-score threshold (O(block_topk) min over GPU top-512 -> 0 alloc, 0 nth_element) */
    {
        const uint32_t q_last = pos_start + M - 1U;
        const uint32_t nc     = (q_last + 1U) / comp_ratio;
        if (nc > block_topk) {
            const float* sc = static_cast<const float*>([eng->b_idx_scores contents]) +
                              static_cast<size_t>(M - 1U) * eng->max_blocks_alloc;
            float* u_ptr = L.u.data();
            for (uint32_t b = 0U; b < nc; ++b) u_ptr[b] = 0.98f * u_ptr[b] + sc[b];
            if (ls < eng->cfg.num_full_attn_layers) {
                const uint32_t* last_sel = sel + static_cast<size_t>(M - 1U) * block_topk;
                float min_sc = (last_sel[0] < nc) ? sc[last_sel[0]] : 0.0f;
                for (uint32_t i = 1U; i < block_topk; ++i) {
                    const uint32_t b = last_sel[i];
                    if (b < nc && sc[b] < min_sc) min_sc = sc[b];
                }
                L.last_thresh = min_sc;
                L.has_thresh  = true;
            }
        }
    }

    /* 2. Group the M queries so each group's needed set fits the physical pool (zero heap allocations) */
    const uint32_t b0 = pos_start / comp_ratio;
    std::vector<SqfMGroup>& groups = s->scratch_mgroups;
    size_t n_groups = 0U;
    auto open_group = [&](uint32_t beg) -> SqfMGroup& {
        if (n_groups >= groups.size()) groups.emplace_back();
        SqfMGroup& g = groups[n_groups++];
        g.beg = beg;
        g.end = beg;
        g.need.clear();
        const uint32_t pin_ep = sqf_next_pin_epoch(L);
        for (uint32_t b = b0; b <= newest_blk && b < s->max_blocks; ++b) {
            L.pin_epoch[b] = pin_ep;
            g.need.push_back(b);
        }
        return g;
    };

    const uint32_t max_write_blks = (newest_blk >= b0) ? (newest_blk - b0 + 1U) : 1U;
    if (static_cast<uint64_t>(M) * block_topk + max_write_blks <= s->pool_blocks) {
        SqfMGroup& g = open_group(0U);
        const uint32_t pin_ep = L.pin_epoch_ctr;
        for (uint32_t m = 0U; m < M; ++m) {
            const uint32_t q_pos = pos_start + m;
            const uint32_t nc    = (q_pos + 1U) / comp_ratio;
            if (nc > block_topk) {
                const uint32_t* row = sel + static_cast<size_t>(m) * block_topk;
                for (uint32_t i = 0U; i < block_topk; ++i) {
                    const uint32_t b = row[i];
                    if (b < s->max_blocks && L.pin_epoch[b] != pin_ep) {
                        L.pin_epoch[b] = pin_ep;
                        g.need.push_back(b);
                    }
                }
            } else {
                for (uint32_t b = 0U; b < nc && b < s->max_blocks; ++b) {
                    if (L.pin_epoch[b] != pin_ep) {
                        L.pin_epoch[b] = pin_ep;
                        g.need.push_back(b);
                    }
                }
            }
        }
        g.end = M;
    } else {
        open_group(0U);
        std::vector<uint32_t>& mine = s->scratch_mine;
        for (uint32_t m = 0U; m < M; ++m) {
            const uint32_t q_pos = pos_start + m;
            const uint32_t nc    = (q_pos + 1U) / comp_ratio;
            mine.clear();
            if (nc > block_topk) {
                const uint32_t* row = sel + static_cast<size_t>(m) * block_topk;
                for (uint32_t i = 0U; i < block_topk; ++i) mine.push_back(row[i]);
            } else {
                for (uint32_t b = 0U; b < nc; ++b) mine.push_back(b);
            }
            uint32_t fresh = 0U;
            for (uint32_t b : mine) if (b < s->max_blocks && L.pin_epoch[b] != L.pin_epoch_ctr) ++fresh;
            if (groups[n_groups - 1U].need.size() + fresh > s->pool_blocks && m > groups[n_groups - 1U].beg) {
                groups[n_groups - 1U].end = m;
                open_group(m);
            }
            SqfMGroup& cur_g = groups[n_groups - 1U];
            const uint32_t pin_ep = L.pin_epoch_ctr;
            for (uint32_t b : mine) {
                if (b < s->max_blocks && L.pin_epoch[b] != pin_ep) {
                    L.pin_epoch[b] = pin_ep;
                    cur_g.need.push_back(b);
                }
            }
        }
        groups[n_groups - 1U].end = M;
    }

    /* 3. Group 0 residency (allocates the chunk's write blocks too; soft-keeps Group 1 blocks if split) */
    SqfPagedCounters c;
    if (n_groups > 1U) {
        const uint32_t pin_ep = sqf_next_pin_epoch(L);
        for (uint32_t b : groups[0].need) L.pin_epoch[b] = pin_ep;
        const SqfMGroup* g1 = &groups[1];
        for (uint32_t b : g1->need) if (b < s->max_blocks) L.u[b] += 1000.0f;
        (void)sqf_paged_ensure(s, ls, groups[0].need, newest_blk, /*exact_check=*/true, c);
        for (uint32_t b : g1->need) if (b < s->max_blocks) L.u[b] -= 1000.0f;
    } else {
        /* Group 0 is already stamped with L.pin_epoch_ctr */
        (void)sqf_paged_ensure(s, ls, groups[0].need, newest_blk, /*exact_check=*/true, c);
    }
    for (uint32_t b = b0; b <= newest_blk && b < s->max_blocks; ++b) L.on_nvme[b] = 0U;
    sqf_paged_merge(s, c);
    *num_groups_out = n_groups;
}

/* ANE lookahead: predicts each QSA layer's index query from the previous step's attention inputs on the 16-core ANE,
 * scores ONLY the NVMe-paged blocks against the resident `block_k`, and preads predicted top-512 entrants ahead of
 * the GPU (the exact sync-point check above still guarantees bit-exactness on any misprediction). */
void sqf_ane_probe_execute(StrataQwenFlashGpuEngine* eng, StrataQwenFlashContextSlot* s, uint32_t q_pos) {
    SqfAneQsaProbe* pr = eng->ane;
    const uint64_t t0 = mach_absolute_time();
    const uint32_t q_rows = pr->n_heads * pr->head_dim;
    bool on_ane = pr->ready;
    @autoreleasepool {
        for (SqfAneModelBundle& bdl : pr->bundles) {
            const uint32_t S = kAneSpatial;
            for (uint32_t li = 0U; li < bdl.n_layers; ++li) {
                const uint16_t* xs = &pr->x_snap[static_cast<size_t>(bdl.layer_base + li) * pr->D];
                uint16_t* dst = bdl.in_buf + static_cast<size_t>(li) * pr->D * S;
                for (uint32_t c = 0U; c < pr->D; ++c) dst[static_cast<size_t>(c) * S] = xs[c];
            }
            NSError* err = nil;
            id<MLFeatureProvider> outF = [bdl.model predictionFromFeatures:bdl.fp options:bdl.opts error:&err];
            if (!outF || err) { on_ane = false; continue; }
            MLMultiArray* oa = [outF featureValueForName:@"y"].multiArrayValue;
            const uint32_t oc_n = bdl.out_ch;
            float* qh = &pr->q_hat[static_cast<size_t>(bdl.layer_base) * q_rows];
            if (oa == bdl.out_arr || !oa) {
                const __fp16* ob = reinterpret_cast<const __fp16*>(bdl.out_buf);
                for (uint32_t oc = 0U; oc < oc_n; ++oc) qh[oc] = static_cast<float>(ob[static_cast<size_t>(oc) * S]);
            } else {
                [oa getBytesWithHandler:^(const void* bytes, NSInteger size) {
                    if (size >= static_cast<NSInteger>(static_cast<size_t>(oc_n) * S * 2U)) {
                        const __fp16* ob = static_cast<const __fp16*>(bytes);
                        for (uint32_t oc = 0U; oc < oc_n; ++oc) qh[oc] = static_cast<float>(ob[static_cast<size_t>(oc) * S]);
                    }
                }];
            }
        }
    }

    /* q_layernorm (+1.0 folded) + NeoX RoPE_64 at the TRUE global q_pos (matches `sqf_qsa_indexer_prep`) */
    const uint32_t half_rot = std::min<uint32_t>(pr->rot_dim / 2U, 32U);
    float cs_tab[32];
    float sn_tab[32];
    for (uint32_t i = 0U; i < half_rot; ++i) {
        const float ang = static_cast<float>(q_pos) * pr->inv_freq[i];
        cs_tab[i] = std::cos(ang);
        sn_tab[i] = std::sin(ang);
    }
    for (uint32_t l = 0U; l < pr->n_layers; ++l) {
        for (uint32_t h = 0U; h < pr->n_heads; ++h) {
            float* v = &pr->q_hat[static_cast<size_t>(l) * q_rows + static_cast<size_t>(h) * pr->head_dim];
            float ss = 0.0f;
            for (uint32_t d = 0U; d < pr->head_dim; ++d) ss += v[d] * v[d];
            const float inv = 1.0f / std::sqrt(ss / static_cast<float>(pr->head_dim) + pr->eps);
            float tmp[64];
            for (uint32_t d = 0U; d < pr->head_dim; ++d) v[d] = v[d] * inv * pr->q_norm[static_cast<size_t>(l) * pr->head_dim + d];
            for (uint32_t d = 0U; d < 2U * half_rot; ++d) tmp[d] = v[d];
            for (uint32_t i = 0U; i < half_rot; ++i) {
                v[i]            = tmp[i] * cs_tab[i] - tmp[i + half_rot] * sn_tab[i];
                v[i + half_rot] = tmp[i + half_rot] * cs_tab[i] + tmp[i] * sn_tab[i];
            }
        }
    }

    const uint64_t t_ane_done = mach_absolute_time();

    /* Score paged blocks per layer in parallel (each layer owns disjoint pool rows / tables / ane_stage) and prefetch */
    const uint32_t nc        = (q_pos + 1U) / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
    const uint32_t n_lyr     = std::min<uint32_t>(pr->n_layers, s->paged_layers);
    const uint8_t* bk_base   = static_cast<const uint8_t*>([s->b_idx_block_k contents]);
    const float    inv_sqrt  = 1.0f / std::sqrt(static_cast<float>(pr->head_dim));
    const size_t   blk_bytes = sqf_paged_block_bytes(s);
    if (s->scratch_ane_lc.size() < n_lyr) s->scratch_ane_lc.resize(n_lyr);
    for (uint32_t li = 0U; li < n_lyr; ++li) s->scratch_ane_lc[li] = SqfPagedCounters{};
    SqfPagedCounters* lcp = s->scratch_ane_lc.data();
    dispatch_apply(n_lyr, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t li) {
        const uint32_t ls = static_cast<uint32_t>(li);
        SqfPagedLayerState& L = s->layers[ls];
        if (!L.has_thresh || !s->ane_stage || L.paged_count == 0U) return;
        uint8_t* l_stage = s->ane_stage + li * static_cast<size_t>(kAneMaxPrefetchPerLyr) * blk_bytes;
        const __fp16* bk = reinterpret_cast<const __fp16*>(bk_base + static_cast<size_t>(ls) * eng->idx_blk_layer_stride);
        const float* qh  = &pr->q_hat[static_cast<size_t>(ls) * q_rows];
        const float thr  = L.last_thresh * kAnePrefetchMargin;
        std::vector<std::pair<float, uint32_t>>& cand = L.scratch_ane_cand;
        cand.clear();
        for (uint32_t b = 0U; b < nc && b < s->max_blocks; ++b) {
            if (L.st[b] != kBlkPaged) continue;
            const __fp16* row = bk + static_cast<size_t>(b) * pr->head_dim;
            float tot = 0.0f;
            for (uint32_t h = 0U; h < pr->n_heads; ++h) {
                const float* q = qh + static_cast<size_t>(h) * pr->head_dim;
                float d = 0.0f;
                for (uint32_t k = 0U; k < pr->head_dim; ++k) d += q[k] * static_cast<float>(row[k]);
                tot += std::max(0.0f, d);
            }
            tot *= inv_sqrt;
            if (tot >= thr) cand.emplace_back(tot, b);
        }
        if (cand.empty()) return;
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b2) { return a.first > b2.first; });
        if (cand.size() > kAneMaxPrefetchPerLyr) cand.resize(kAneMaxPrefetchPerLyr);
        const uint32_t pin_epoch = sqf_next_pin_epoch(L);
        std::vector<uint32_t>& need = L.scratch_need;
        need.clear();
        for (const auto& pc : cand) { L.pin_epoch[pc.second] = pin_epoch; need.push_back(pc.second); }
        const uint32_t newest = (nc > 0U) ? nc - 1U : 0U;
        /* Prefetch only into free slots or by evicting cold middle blocks (never sinks / local window) */
        const uint32_t free_cnt = s->pool_blocks - L.resident;
        if (need.size() > free_cnt) {
            const uint32_t got = sqf_paged_evict_staged(
                s, ls, static_cast<uint32_t>(need.size()) - free_cnt, newest, false, l_stage, kAneMaxPrefetchPerLyr, lcp[li]);
            const uint32_t cap = free_cnt + got;
            if (need.size() > cap) need.resize(cap);
        }
        if (need.empty()) return;
        (void)sqf_paged_ensure_staged(s, ls, need, newest, false, l_stage, kAneMaxPrefetchPerLyr, lcp[li]);
        for (uint32_t b : need) {
            if (L.st[b] == kBlkResident) { L.pf_epoch[b] = s->step_epoch; lcp[li].pf_issued += 1ULL; }
        }
    });
    SqfPagedCounters tot;
    for (uint32_t li = 0U; li < n_lyr; ++li) {
        const auto& c = s->scratch_ane_lc[li];
        tot.paged_out += c.paged_out; tot.recalled += c.recalled; tot.wbytes += c.wbytes; tot.rbytes += c.rbytes;
        tot.wms += c.wms; tot.rms += c.rms; tot.pf_issued += c.pf_issued; tot.err = tot.err || c.err;
    }
    s->probe_out_counters = tot;
    s->probe_out_on_ane   = on_ane;
    s->probe_out_us       = sqf_ticks_ms(t_ane_done - t0) * 1000.0;
}

/* Called at the start of every decode / spec / prefill-chunk step on the bound slot. */
void sqf_paged_step_begin(StrataQwenFlashGpuEngine* eng, uint32_t q_pos_last) {
    StrataQwenFlashContextSlot* s = eng->bound_slot;
    if (!s || !s->paged) return;
    sqf_paged_join_probe(s);
    s->step_epoch += 1U;
    const uint32_t topk = std::max<uint32_t>(1U, eng->cfg.indexer_budget / std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio));
    const uint32_t nc   = (q_pos_last + 1U) / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
    if (!eng->ane || eng->ane->bundles.empty() || !s->x_valid || nc <= topk) return;
    bool any_paged = false;
    const uint32_t n_probe_lyrs = std::min<uint32_t>(eng->ane->n_layers, s->paged_layers);
    for (uint32_t l = 0U; l < n_probe_lyrs; ++l) {
        if (s->layers[l].paged_count > 0U) { any_paged = true; break; }
    }
    if (!any_paged) return;
    SqfAneQsaProbe* pr = eng->ane;
    std::memcpy(pr->x_snap.data(), [eng->b_probe_x contents], pr->x_snap.size() * sizeof(uint16_t));
    s->probe_pending = true;
    dispatch_group_async(s->probe_grp, pr->queue, ^{
        sqf_ane_probe_execute(eng, s, q_pos_last);
    });
}

bool sqf_take_paged_error(StrataQwenFlashGpuEngine* eng) {
    if (eng->bound_slot) sqf_paged_join_probe(eng->bound_slot);
    const bool e = eng->paged_error;
    eng->paged_error = false;
    return e;
}

} // namespace

namespace {

inline id<MTLBuffer> sqf_part_buf(StrataQwenFlashGpuEngine* eng, uint32_t part_id) {
    return (part_id == 0U) ? eng->b_part0 : eng->b_part1;
}

inline const StrataQwenFlashLayerDesc& sqf_layer_desc(const StrataQwenFlashGpuEngine* eng, uint32_t l) {
    if (l < eng->cfg.split_layer_idx) {
        return eng->hdr0.layers[l];
    }
    return eng->hdr1.layers[l - eng->cfg.split_layer_idx];
}

// Parallel lock-free 16-bank NVMe `F_NOCACHE` `pread` + `mlock` LFU PLE lookup
// Executes on CPU concurrently while GPU runs `Embed + Layer 0`!
bool sqf_drain_pending_fix(StrataQwenFlashGpuEngine* eng) {
    if (!eng || !eng->pending_fix_cb) return true;
    id<MTLCommandBuffer> cb = eng->pending_fix_cb;
    eng->pending_fix_cb = nil;
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError) {
        NSError* err = [cb error];
        std::fprintf(stderr, "[strata_qwen_flash_gpu] async spec rollback CB error: %s\n",
                     err ? [[err localizedDescription] UTF8String] : "unknown");
        return false;
    }
    return true;
}

bool sqf_ple_all_cached(
    const StrataQwenFlashGpuEngine* eng,
    const uint32_t*                 token_ids,
    uint32_t                        num_tokens,
    uint32_t                        tree_branch_t = 0U
) {
    if (!eng || !eng->cfg.has_ple) return true;
    if (!eng->bundle || eng->bundle->ple_fd < 0) return true;
    if (num_tokens == 0U || num_tokens > kMaxSpecBatch) return false;
    const StrataQwenFlashPleHeader* ph    = &eng->bundle->ple_header;
    const StrataQwenFlashPleCache*  cache = &eng->bundle->ple_cache;
    const uint32_t                  n_heads = ph->ngram_heads;
    if (!cache->arena_base || n_heads > cache->ngram_heads) return false;

    uint32_t p2 = eng->ple_prev_tokens[0];
    uint32_t p1 = eng->ple_prev_tokens[1];
    uint32_t root_p2 = p2, root_p1 = p1;
    uint64_t rows[STRATA_QWEN_FLASH_MAX_NGRAM_HEADS];
    for (uint32_t t = 0U; t < num_tokens; ++t) {
        if (tree_branch_t != 0U && t == tree_branch_t) {
            p2 = root_p2;
            p1 = root_p1;
        }
        const uint32_t tok0 = token_ids[t];
        strata_qwen_flash_ple_hash_token(ph, tok0, p1, p2, rows);
        p2 = p1;
        p1 = tok0;
        if (t == 0U) { root_p2 = p2; root_p1 = p1; }
        for (uint32_t h = 0U; h < n_heads; ++h) {
            const uint64_t row       = rows[h];
            const uint32_t set_idx   = static_cast<uint32_t>((row * 0x9E3779B97F4A7C15ULL) >> 32U) & cache->set_mask;
            const size_t   base_slot = (static_cast<size_t>(h) * cache->sets_per_head + set_idx) *
                                       STRATA_QWEN_FLASH_PLE_LFU_WAYS;
            bool found = false;
            for (uint32_t w = 0U; w < STRATA_QWEN_FLASH_PLE_LFU_WAYS; ++w) {
                if (cache->tags[base_slot + w] == row) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    return true;
}

int sqf_execute_ple_lookup_host_buf(
    StrataQwenFlashGpuEngine* eng,
    id<MTLBuffer>             dst_buf,
    const uint32_t*           token_ids,
    uint32_t                  num_tokens,
    uint32_t                  prev_in[2],
    uint32_t                  prev_out[2],
    uint32_t                  prev_after_out[kMaxSpecBatch][2],
    uint32_t                  tree_branch_t = 0U
) {
    if (!eng->cfg.has_ple) return 0;
    float* dst_base = reinterpret_cast<float*>([dst_buf contents]);
    const uint32_t emb_dim = eng->cfg.ple_embed_dim;

    if (!eng->bundle || eng->bundle->ple_fd < 0) {
        std::memset(dst_base, 0, static_cast<size_t>(num_tokens) * emb_dim * sizeof(float));
        uint32_t p2 = prev_in[0];
        uint32_t p1 = prev_in[1];
        uint32_t root_p2 = p2, root_p1 = p1;
        for (uint32_t t = 0U; t < num_tokens; ++t) {
            if (tree_branch_t != 0U && t == tree_branch_t) {
                p2 = root_p2;
                p1 = root_p1;
            }
            p2 = p1;
            p1 = token_ids[t];
            if (t == 0U) { root_p2 = p2; root_p1 = p1; }
            if (prev_after_out && t < kMaxSpecBatch) {
                prev_after_out[t][0] = p2;
                prev_after_out[t][1] = p1;
            }
        }
        prev_out[0] = p2;
        prev_out[1] = p1;
        return 0;
    }

    StrataQwenFlashBundle*          bundle   = eng->bundle;
    const StrataQwenFlashPleHeader* ph       = &bundle->ple_header;
    StrataQwenFlashPleCache*        cache    = &bundle->ple_cache;
    const uint32_t                  n_heads  = ph->ngram_heads;
    const uint32_t                  head_dim = ph->head_dim_per_ngram;

    const size_t rows_need = static_cast<size_t>(num_tokens) * STRATA_QWEN_FLASH_MAX_NGRAM_HEADS;
    if (eng->ple_all_rows.size() < rows_need) eng->ple_all_rows.resize(rows_need);
    uint64_t* all_rows_flat = eng->ple_all_rows.data();
    uint32_t p2 = prev_in[0];
    uint32_t p1 = prev_in[1];
    uint32_t root_p2 = p2, root_p1 = p1;
    for (uint32_t t = 0U; t < num_tokens; ++t) {
        if (tree_branch_t != 0U && t == tree_branch_t) {
            p2 = root_p2;
            p1 = root_p1;
        }
        const uint32_t tok0 = token_ids[t];
        strata_qwen_flash_ple_hash_token(
            ph, tok0, p1, p2, all_rows_flat + static_cast<size_t>(t) * STRATA_QWEN_FLASH_MAX_NGRAM_HEADS);
        p2 = p1;
        p1 = tok0;
        if (t == 0U) { root_p2 = p2; root_p1 = p1; }
        if (prev_after_out && t < kMaxSpecBatch) {
            prev_after_out[t][0] = p2;
            prev_after_out[t][1] = p1;
        }
    }
    prev_out[0] = p2;
    prev_out[1] = p1;

    // Fast inline check for M <= 8 when all (M * n_heads) rows hit LFU cache (zero GCD wakeup overhead)
    bool all_cached = (num_tokens <= kMaxSpecBatch) && cache->arena_base && (n_heads <= cache->ngram_heads);
    if (all_cached) {
        for (uint32_t t = 0U; t < num_tokens && all_cached; ++t) {
            for (uint32_t h = 0U; h < n_heads; ++h) {
                const uint64_t row       = all_rows_flat[static_cast<size_t>(t) * STRATA_QWEN_FLASH_MAX_NGRAM_HEADS + h];
                const uint32_t set_idx   = static_cast<uint32_t>((row * 0x9E3779B97F4A7C15ULL) >> 32U) & cache->set_mask;
                const size_t   base_slot = (static_cast<size_t>(h) * cache->sets_per_head + set_idx) *
                                           STRATA_QWEN_FLASH_PLE_LFU_WAYS;
                bool found = false;
                for (uint32_t w = 0U; w < STRATA_QWEN_FLASH_PLE_LFU_WAYS; ++w) {
                    if (cache->tags[base_slot + w] == row) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    all_cached = false;
                    break;
                }
            }
        }
    }

    uint64_t head_hits[STRATA_QWEN_FLASH_MAX_NGRAM_HEADS]   = {};
    uint64_t head_misses[STRATA_QWEN_FLASH_MAX_NGRAM_HEADS] = {};
    int      head_err[STRATA_QWEN_FLASH_MAX_NGRAM_HEADS]    = {};

    if (all_cached) {
        for (uint32_t t = 0U; t < num_tokens; ++t) {
            for (uint32_t h = 0U; h < n_heads; ++h) {
                uint64_t hd = 0ULL, md = 0ULL;
                if (strata_qwen_flash_ple_lookup_head_f32(
                        bundle->ple_fd, ph, cache, h,
                        all_rows_flat[static_cast<size_t>(t) * STRATA_QWEN_FLASH_MAX_NGRAM_HEADS + h],
                        dst_base + static_cast<size_t>(t) * emb_dim + static_cast<size_t>(h) * head_dim,
                        &hd, &md) != 0) {
                    return -1;
                }
                head_hits[h]   += hd;
                head_misses[h] += md;
            }
        }
    } else {
        const uint64_t* rows_ptr = all_rows_flat;
        uint64_t* hits_ptr   = head_hits;
        uint64_t* misses_ptr = head_misses;
        int*      err_ptr    = head_err;
        const int ple_fd     = bundle->ple_fd;

        dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
        dispatch_apply(n_heads, q, ^(size_t h_sz) {
            const uint32_t h = static_cast<uint32_t>(h_sz);
            uint64_t h_hit = 0ULL, h_miss = 0ULL;
            for (uint32_t t = 0U; t < num_tokens; ++t) {
                uint64_t hd = 0ULL, md = 0ULL;
                float* dst = dst_base + static_cast<size_t>(t) * emb_dim + static_cast<size_t>(h) * head_dim;
                if (strata_qwen_flash_ple_lookup_head_f32(
                        ple_fd, ph, cache, h,
                        rows_ptr[static_cast<size_t>(t) * STRATA_QWEN_FLASH_MAX_NGRAM_HEADS + h], dst, &hd, &md) != 0) {
                    err_ptr[h] = -1;
                    break;
                }
                h_hit  += hd;
                h_miss += md;
            }
            hits_ptr[h]   = h_hit;
            misses_ptr[h] = h_miss;
        });

        for (uint32_t h = 0U; h < n_heads; ++h) {
            if (head_err[h] != 0) return -1;
        }
    }

    uint64_t tot_hits = 0ULL, tot_misses = 0ULL;
    for (uint32_t h = 0U; h < n_heads; ++h) {
        tot_hits   += head_hits[h];
        tot_misses += head_misses[h];
    }
    const uint64_t nvme_bytes = tot_misses * static_cast<uint64_t>(ph->row_bytes);

    cache->lfu_hits        += tot_hits;
    cache->lfu_misses      += tot_misses;
    cache->nvme_preads     += tot_misses;
    cache->nvme_bytes_read += nvme_bytes;

    eng->counters.ple_lfu_hits        += tot_hits;
    eng->counters.ple_lfu_misses      += tot_misses;
    eng->counters.ple_nvme_preads     += tot_misses;
    eng->counters.ple_nvme_bytes_read += nvme_bytes;
    return 0;
}

int sqf_execute_ple_lookup_host(
    StrataQwenFlashGpuEngine* eng,
    const uint32_t*           token_ids,
    uint32_t                  num_tokens,
    uint32_t                  prev_after_out[kMaxSpecBatch][2],
    uint32_t                  tree_branch_t = 0U
) {
    return sqf_execute_ple_lookup_host_buf(
        eng, eng->b_ple_emb_f32, token_ids, num_tokens,
        eng->ple_prev_tokens, eng->ple_prev_tokens, prev_after_out, tree_branch_t);
}

inline id<MTLComputePipelineState> sqf_get_pso_m(
    StrataQwenFlashGpuEngine*             eng,
    id<MTLComputePipelineState> __strong& slot,
    const char*                           fn_name,
    uint32_t                              m_const
) {
    if (__builtin_expect(slot != nil, 1)) return slot;
    MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
    [fcv setConstantValue:&m_const type:MTLDataTypeUInt atIndex:0];
    NSError* fn_err = nil;
    NSString* ns_name = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [eng->library newFunctionWithName:ns_name constantValues:fcv error:&fn_err];
    if (!fn) return nil;
    NSError* p_err = nil;
    slot = [eng->device newComputePipelineStateWithFunction:fn error:&p_err];
    return slot;
}

inline uint64_t sqf_align_256(uint64_t x) {
    return (x + 255ULL) & ~255ULL;
}

inline bool sqf_can_use_q4_i8(
    const StrataQwenFlashGpuEngine*   eng,
    const StrataQwenFlashQuantMatrix& mat,
    uint32_t                          M_valid,
    uint32_t                          M_padded,
    bool                              force_gemv,
    uint64_t*                         off_x_sc_out = nullptr,
    uint64_t*                         off_w_i8_out = nullptr,
    uint64_t*                         off_w_sc_out = nullptr
) {
    if (!eng || !eng->int8_prefill_enabled || force_gemv || !eng->has_nax) return false;
    if (M_valid < 32U || M_padded < 32U) return false;
    if ((mat.cols % 64U) != 0U || (mat.rows_padded % 64U) != 0U) return false;
    if (!eng->b_moe_down_part || !eng->pso_quant_f16_to_i8_row || !eng->pso_q4_requant_row_i8) return false;
    const uint64_t x_i8_b   = static_cast<uint64_t>(M_padded) * mat.cols;
    const uint64_t off_x_sc = sqf_align_256(x_i8_b);
    const uint64_t off_w_i8 = sqf_align_256(off_x_sc + static_cast<uint64_t>(M_padded) * sizeof(float));
    const uint64_t off_w_sc = sqf_align_256(off_w_i8 + static_cast<uint64_t>(mat.rows_padded) * mat.cols);
    const uint64_t need     = off_w_sc + static_cast<uint64_t>(mat.rows_padded) * sizeof(float);
    if ([eng->b_moe_down_part length] < need) return false;
    if (off_x_sc_out) *off_x_sc_out = off_x_sc;
    if (off_w_i8_out) *off_w_i8_out = off_w_i8;
    if (off_w_sc_out) *off_w_sc_out = off_w_sc;
    return true;
}

void sqf_encode_quant_x_i8(
    id<MTLComputeCommandEncoder> enc,
    StrataQwenFlashGpuEngine*    eng,
    id<MTLBuffer>                x_in_f16,
    NSUInteger                   x_off,
    id<MTLBuffer>                scratch,
    NSUInteger                   off_x_i8,
    NSUInteger                   off_x_sc,
    uint32_t                     M_valid,
    uint32_t                     M_padded,
    uint32_t                     K
) {
    SqfI8QuantParamsHost qp{M_valid, M_padded, 0U, 0U, K, 0U, 0U, 0U};
    [enc setComputePipelineState:eng->pso_quant_f16_to_i8_row];
    [enc setBuffer:x_in_f16 offset:x_off atIndex:0];
    [enc setBuffer:scratch offset:off_x_i8 atIndex:1];
    [enc setBuffer:scratch offset:off_x_sc atIndex:2];
    [enc setBytes:&qp length:sizeof(qp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(M_padded, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sqf_encode_q4_linear(
    id<MTLComputeCommandEncoder> enc,
    StrataQwenFlashGpuEngine* eng,
    id<MTLBuffer> w_buf,
    const StrataQwenFlashQuantMatrix& mat,
    id<MTLBuffer> x_in_f16,
    NSUInteger x_off,
    id<MTLBuffer> y_f16,
    NSUInteger y_f16_off,
    id<MTLBuffer> y_f32,
    NSUInteger y_f32_off,
    uint32_t M_valid,
    uint32_t M_padded,
    uint32_t out_mode,
    bool force_gemv,
    bool skip_x_quant = false
) {
    uint64_t off_x_sc = 0ULL, off_w_i8 = 0ULL, off_w_sc = 0ULL;
    if (sqf_can_use_q4_i8(eng, mat, M_valid, M_padded, force_gemv, &off_x_sc, &off_w_i8, &off_w_sc)) {
        if (!skip_x_quant) {
            sqf_encode_quant_x_i8(enc, eng, x_in_f16, x_off, eng->b_moe_down_part, 0U,
                                  static_cast<NSUInteger>(off_x_sc), M_valid, M_padded, mat.cols);
        }
        SqfI8QuantParamsHost wp{M_valid, M_padded, mat.rows_logical, mat.rows_padded, mat.cols, 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_q4_requant_row_i8];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.biases_off) atIndex:2];
        [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_i8) atIndex:3];
        [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_sc) atIndex:4];
        [enc setBytes:&wp length:sizeof(wp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(mat.rows_padded, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        const bool use_64x128 = (M_padded >= 64U) && ((mat.rows_padded % 128U) == 0U) &&
                                ((mat.rows_padded / 128U) * (M_padded / 64U) >= 40U);
        const bool use_64x64  = !use_64x128 && (M_padded >= 64U);
        id<MTLComputePipelineState> pso = use_64x128 ? eng->pso_i8_gemm_nax_64x128_sg4
                                        : (use_64x64 ? eng->pso_i8_gemm_nax_64x64_sg4
                                                     : eng->pso_i8_gemm_nax_32x64_sg2);
        const uint32_t m_tile = (use_64x128 || use_64x64) ? 64U : 32U;
        const uint32_t n_tile = use_64x128 ? 128U : 64U;
        const uint32_t n_th   = (use_64x128 || use_64x64) ? 128U : 64U;

        SqfNaxGemmParamsHost p{
            M_valid, M_padded, mat.rows_logical, mat.rows_padded, mat.cols, out_mode, 0U, 0U
        };
        [enc setComputePipelineState:pso];
        [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_i8) atIndex:0];
        [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_sc) atIndex:1];
        [enc setBuffer:eng->b_moe_down_part offset:0U atIndex:2];
        [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_x_sc) atIndex:3];
        if (y_f16) [enc setBuffer:y_f16 offset:y_f16_off atIndex:4];
        else       [enc setBuffer:x_in_f16 offset:0 atIndex:4];
        if (y_f32) [enc setBuffer:y_f32 offset:y_f32_off atIndex:5];
        else       [enc setBuffer:x_in_f16 offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(mat.rows_padded / n_tile, (M_padded + m_tile - 1U) / m_tile, 1)
            threadsPerThreadgroup:MTLSizeMake(n_th, 1, 1)];
        return;
    }

    const bool use_nax = !force_gemv && eng->has_nax && ((mat.cols % 64U) == 0U);
    if (use_nax) {
        const uint32_t n_tiles   = mat.rows_padded / 64U;
        const uint32_t tiles_m64 = n_tiles * ((M_padded + 63U) / 64U);
        const uint32_t tiles_m32 = n_tiles * ((M_padded + 31U) / 32U);
        uint32_t m_tile = 64U;
        uint32_t n_th   = 128U;
        id<MTLComputePipelineState> pso = nil;
        if ((mat.cols % 128U) == 0U) {
            if (M_padded >= 64U && tiles_m64 >= 80U) {
                pso = eng->pso_q4_gemm_nax_64x64_k128_sg4; m_tile = 64U; n_th = 128U;
            } else if (M_padded >= 32U && tiles_m32 >= 64U) {
                pso = eng->pso_q4_gemm_nax_32x64_k128_sg2; m_tile = 32U; n_th = 64U;
            } else {
                pso = eng->pso_q4_gemm_nax_16x64_k128_sg2; m_tile = 16U; n_th = 64U;
            }
        } else {
            const bool use_m16 = (M_padded < 64U) || (tiles_m64 < 80U);
            pso    = use_m16 ? eng->pso_q4_gemm_nax_m16 : eng->pso_q4_gemm_nax_m64;
            m_tile = use_m16 ? 16U : 64U;
            n_th   = use_m16 ? 64U : 128U;
        }
        SqfNaxGemmParamsHost p{
            M_valid, M_padded, mat.rows_logical, mat.rows_padded, mat.cols, out_mode, 0U, 0U
        };
        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.biases_off) atIndex:2];
        [enc setBuffer:x_in_f16 offset:x_off atIndex:3];
        if (y_f16) [enc setBuffer:y_f16 offset:y_f16_off atIndex:4];
        else       [enc setBuffer:x_in_f16 offset:0 atIndex:4];
        if (y_f32) [enc setBuffer:y_f32 offset:y_f32_off atIndex:5];
        else       [enc setBuffer:x_in_f16 offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tiles, (M_padded + m_tile - 1U) / m_tile, 1)
            threadsPerThreadgroup:MTLSizeMake(n_th, 1, 1)];
        return;
    }

    const bool use_8lane = (mat.cols <= 1024U) && ((mat.cols % 64U) == 0U) && (mat.rows_logical >= 1024U);
    const uint32_t grid_x = use_8lane ? ((mat.rows_logical + 15U) / 16U) : ((mat.rows_logical + 7U) / 8U);

    for (uint32_t m0 = 0U; m0 < M_valid; m0 += kMaxSpecBatch) {
        const uint32_t mb = std::min<uint32_t>(kMaxSpecBatch, M_valid - m0);
        SqfGemvBatchParamsHost p{mb, mat.rows_logical, mat.cols, out_mode};
        id<MTLComputePipelineState> pso = use_8lane
            ? sqf_get_pso_m(eng, eng->pso_q4_gemv_8lane_m[mb - 1U], "sqf_q4_gemv_8lane", mb)
            : sqf_get_pso_m(eng, eng->pso_q4_gemv_m[mb - 1U], "sqf_q4_gemv_batch", mb);
        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.biases_off) atIndex:2];
        [enc setBuffer:x_in_f16 offset:(x_off + static_cast<NSUInteger>(m0) * mat.cols * sizeof(uint16_t)) atIndex:3];
        if (y_f16) {
            [enc setBuffer:y_f16
                    offset:(y_f16_off + static_cast<NSUInteger>(m0) * mat.rows_logical * sizeof(uint16_t))
                   atIndex:4];
        } else {
            [enc setBuffer:x_in_f16 offset:0 atIndex:4];
        }
        if (y_f32) {
            [enc setBuffer:y_f32
                    offset:(y_f32_off + static_cast<NSUInteger>(m0) * mat.rows_logical * sizeof(float))
                   atIndex:5];
        } else {
            [enc setBuffer:x_in_f16 offset:0 atIndex:5];
        }
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
}

void sqf_encode_f16_linear(
    id<MTLComputeCommandEncoder> enc,
    StrataQwenFlashGpuEngine* eng,
    id<MTLBuffer> w_buf,
    const StrataQwenFlashTensorSlice& sl,
    id<MTLBuffer> x_in_f16,
    NSUInteger x_off,
    id<MTLBuffer> y_f16,
    NSUInteger y_f16_off,
    id<MTLBuffer> y_f32,
    NSUInteger y_f32_off,
    uint32_t M_valid,
    uint32_t M_padded,
    uint32_t out_mode,
    bool force_gemv
) {
    const uint32_t rows    = sl.shape0;
    const uint32_t cols    = sl.shape1;

    if (sl.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A) {
        const StrataQwenFlashQuantMatrix qm = sqf_q4_slice_as_qmat(&sl);
        sqf_encode_q4_linear(enc, eng, w_buf, qm, x_in_f16, x_off, y_f16, y_f16_off, y_f32, y_f32_off,
                             M_valid, M_padded, out_mode, force_gemv);
        return;
    }

    if (sl.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) {
        uint64_t wb = 0ULL;
        sqf_q8_slice_layout(rows, cols, &wb, nullptr);
        const NSUInteger w_off = static_cast<NSUInteger>(sl.offset);
        const NSUInteger s_off = static_cast<NSUInteger>(sl.offset + wb);
        const bool use_nax = !force_gemv && eng->has_nax && ((cols % 64U) == 0U);
        if (use_nax) {
            const uint32_t n_padded  = (rows + 63U) & ~63U;
            const uint32_t n_tiles   = n_padded / 64U;
            const uint32_t tiles_m64 = n_tiles * ((M_padded + 63U) / 64U);
            const uint32_t tiles_m32 = n_tiles * ((M_padded + 31U) / 32U);
            uint32_t m_tile = 64U;
            uint32_t n_th   = 128U;
            id<MTLComputePipelineState> pso = nil;
            if ((cols % 128U) == 0U) {
                if (M_padded >= 64U && tiles_m64 >= 80U) {
                    pso = eng->pso_q8_gemm_nax_64x64_k128_sg4; m_tile = 64U; n_th = 128U;
                } else if (M_padded >= 32U && tiles_m32 >= 64U) {
                    pso = eng->pso_q8_gemm_nax_32x64_k128_sg2; m_tile = 32U; n_th = 64U;
                } else {
                    pso = eng->pso_q8_gemm_nax_16x64_k128_sg2; m_tile = 16U; n_th = 64U;
                }
            } else {
                const bool use_m16 = (M_padded < 64U) || (tiles_m64 < 80U);
                pso    = use_m16 ? eng->pso_q8_gemm_nax_m16 : eng->pso_q8_gemm_nax_m64;
                m_tile = use_m16 ? 16U : 64U;
                n_th   = use_m16 ? 64U : 128U;
            }
            SqfNaxGemmParamsHost p{
                M_valid, M_padded, rows, n_padded, cols, out_mode, 0U, 0U
            };
            [enc setComputePipelineState:pso];
            [enc setBuffer:w_buf offset:w_off atIndex:0];
            [enc setBuffer:w_buf offset:s_off atIndex:1];
            [enc setBuffer:x_in_f16 offset:x_off atIndex:2];
            if (y_f16) [enc setBuffer:y_f16 offset:y_f16_off atIndex:3];
            else       [enc setBuffer:x_in_f16 offset:0 atIndex:3];
            if (y_f32) [enc setBuffer:y_f32 offset:y_f32_off atIndex:4];
            else       [enc setBuffer:x_in_f16 offset:0 atIndex:4];
            [enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(n_tiles, (M_padded + m_tile - 1U) / m_tile, 1)
                threadsPerThreadgroup:MTLSizeMake(n_th, 1, 1)];
            return;
        }

        const bool use_8lane = (rows > 1024U) && (cols <= 1024U) && ((cols % 64U) == 0U);
        id<MTLComputePipelineState> pso = use_8lane ? eng->pso_q8_gemv_8lane : eng->pso_q8_gemv_batch;
        const uint32_t grid_x = use_8lane ? ((rows + 31U) / 32U) : ((rows + 7U) / 8U);

        for (uint32_t m0 = 0U; m0 < M_valid; m0 += kMaxSpecBatch) {
            const uint32_t mb = std::min<uint32_t>(kMaxSpecBatch, M_valid - m0);
            SqfGemvBatchParamsHost p{mb, rows, cols, out_mode};
            [enc setComputePipelineState:pso];
            [enc setBuffer:w_buf offset:w_off atIndex:0];
            [enc setBuffer:w_buf offset:s_off atIndex:1];
            [enc setBuffer:x_in_f16 offset:(x_off + static_cast<NSUInteger>(m0) * cols * sizeof(uint16_t)) atIndex:2];
            if (y_f16) {
                [enc setBuffer:y_f16
                        offset:(y_f16_off + static_cast<NSUInteger>(m0) * rows * sizeof(uint16_t))
                       atIndex:3];
            } else {
                [enc setBuffer:x_in_f16 offset:0 atIndex:3];
            }
            if (y_f32) {
                [enc setBuffer:y_f32
                        offset:(y_f32_off + static_cast<NSUInteger>(m0) * rows * sizeof(float))
                       atIndex:4];
            } else {
                [enc setBuffer:x_in_f16 offset:0 atIndex:4];
            }
            [enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        return;
    }

    // High-occupancy 2D-tiled FP16 prefill for small-N matrices (`rows <= 64`: `block_inject [4,10240]`, `lin_a/b [48,2560]`)
    if (!force_gemv && M_valid > kMaxSpecBatch && rows <= 64U && ((cols % 8U) == 0U)) {
        SqfGemvBatchParamsHost p{M_valid, rows, cols, out_mode};
        [enc setComputePipelineState:eng->pso_f16_small_n_prefill];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(sl.offset) atIndex:0];
        [enc setBuffer:x_in_f16 offset:x_off atIndex:1];
        if (y_f16) [enc setBuffer:y_f16 offset:y_f16_off atIndex:2];
        else       [enc setBuffer:x_in_f16 offset:0 atIndex:2];
        if (y_f32) [enc setBuffer:y_f32 offset:y_f32_off atIndex:3];
        else       [enc setBuffer:x_in_f16 offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(rows, (M_valid + 7U) / 8U, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }

    const bool     use_nax = !force_gemv && eng->has_nax && ((cols % 64U) == 0U);
    if (use_nax) {
        const uint32_t m_tiles_64 = (M_padded + 63U) / 64U;
        const uint32_t m_tiles_32 = (M_padded + 31U) / 32U;
        const uint32_t tiles_64x128 = ((rows + 127U) / 128U) * m_tiles_64;
        const uint32_t tiles_64x64  = ((rows + 63U)  / 64U)  * m_tiles_64;
        const uint32_t tiles_32x64  = ((rows + 63U)  / 64U)  * m_tiles_32;
        uint32_t m_tile = 64U;
        uint32_t n_tile = 64U;
        uint32_t n_th   = 128U;
        id<MTLComputePipelineState> pso = nil;
        if (M_padded >= 64U && rows >= 128U && tiles_64x128 >= 80U) {
            pso = eng->pso_f16_gemm_nax_64x128_sg4; m_tile = 64U; n_tile = 128U; n_th = 128U;
        } else if (M_padded >= 64U && tiles_64x64 >= 80U) {
            pso = eng->pso_f16_gemm_nax_64x64_sg4;  m_tile = 64U; n_tile = 64U;  n_th = 128U;
        } else if (M_padded >= 32U && tiles_32x64 >= 64U) {
            pso = eng->pso_f16_gemm_nax_32x64_sg2;  m_tile = 32U; n_tile = 64U;  n_th = 64U;
        } else {
            pso = eng->pso_f16_gemm_nax_16x32_sg2;  m_tile = 16U; n_tile = 32U;  n_th = 64U;
        }
        /* The direct-device kernel takes fixed `n_tile`-row static slices of `W`, and Metal does not clamp a static
         * slice to the tensor extents: a ragged last tile would read up to `n_tile - 1` rows past the weight slice
         * (past the weight buffer when the slice is last in its file). Only dispatch whole tiles: fall back to the
         * widest row tile that divides `rows`, otherwise to the exact GEMV paths below. */
        if ((rows % n_tile) != 0U) {
            if ((rows % 64U) == 0U && M_padded >= 64U) {
                pso = eng->pso_f16_gemm_nax_64x64_sg4;  m_tile = 64U; n_tile = 64U;  n_th = 128U;
            } else if ((rows % 64U) == 0U && M_padded >= 32U) {
                pso = eng->pso_f16_gemm_nax_32x64_sg2;  m_tile = 32U; n_tile = 64U;  n_th = 64U;
            } else if ((rows % 32U) == 0U) {
                pso = eng->pso_f16_gemm_nax_16x32_sg2;  m_tile = 16U; n_tile = 32U;  n_th = 64U;
            }
        }
        if ((rows % n_tile) == 0U) {
            SqfNaxGemmParamsHost p{
                M_valid, M_padded, rows, rows, cols, out_mode, 0U, 0U
            };
            [enc setComputePipelineState:pso];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(sl.offset) atIndex:0];
            [enc setBuffer:x_in_f16 offset:x_off atIndex:1];
            if (y_f16) [enc setBuffer:y_f16 offset:y_f16_off atIndex:2];
            else       [enc setBuffer:x_in_f16 offset:0 atIndex:2];
            if (y_f32) [enc setBuffer:y_f32 offset:y_f32_off atIndex:3];
            else       [enc setBuffer:x_in_f16 offset:0 atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(rows / n_tile, (M_padded + m_tile - 1U) / m_tile, 1)
                threadsPerThreadgroup:MTLSizeMake(n_th, 1, 1)];
            return;
        }
    }

    const bool use_row_tg128 = (rows <= 1024U) && (cols >= 512U) && ((cols % 8U) == 0U);
    const bool use_8lane     = !use_row_tg128 && (rows > 1024U) && (cols <= 1024U) && ((cols % 64U) == 0U);
    const uint32_t grid_x = use_row_tg128
        ? rows
        : (use_8lane ? ((rows + 15U) / 16U) : ((rows + 7U) / 8U));

    for (uint32_t m0 = 0U; m0 < M_valid; m0 += kMaxSpecBatch) {
        const uint32_t mb = std::min<uint32_t>(kMaxSpecBatch, M_valid - m0);
        SqfGemvBatchParamsHost p{mb, rows, cols, out_mode};
        id<MTLComputePipelineState> pso = use_row_tg128
            ? sqf_get_pso_m(eng, eng->pso_f16_gemv_row_tg128_m[mb - 1U], "sqf_f16_gemv_row_tg128", mb)
            : (use_8lane
                ? eng->pso_f16_gemv_8lane
                : sqf_get_pso_m(eng, eng->pso_f16_gemv_m[mb - 1U], "sqf_f16_gemv_batch", mb));
        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(sl.offset) atIndex:0];
        [enc setBuffer:x_in_f16 offset:(x_off + static_cast<NSUInteger>(m0) * cols * sizeof(uint16_t)) atIndex:1];
        if (y_f16) {
            [enc setBuffer:y_f16
                    offset:(y_f16_off + static_cast<NSUInteger>(m0) * rows * sizeof(uint16_t))
                   atIndex:2];
        } else {
            [enc setBuffer:x_in_f16 offset:0 atIndex:2];
        }
        if (y_f32) {
            [enc setBuffer:y_f32
                    offset:(y_f32_off + static_cast<NSUInteger>(m0) * rows * sizeof(float))
                   atIndex:3];
        } else {
            [enc setBuffer:x_in_f16 offset:0 atIndex:3];
        }
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
}

// Encodes Pre-Attn / Pre-MLP 4-Stream Hyper-Connection (`use_combine = true`) -> `b_x_in_f16`, `b_x_in_f32`, `b_hc_inj_f32`
void sqf_encode_hc_pre(
    id<MTLComputeCommandEncoder> enc,
    StrataQwenFlashGpuEngine* eng,
    id<MTLBuffer> w_buf,
    const StrataQwenFlashHCDesc& hc,
    id<MTLBuffer> hc_buf,
    uint32_t M_valid,
    uint32_t M_padded,
    bool write_x_f32,
    bool force_gemv,
    bool skip_rmsnorm = false
) {
    const uint32_t D      = eng->cfg.hidden_size;
    const uint32_t H      = eng->cfg.hc_count;
    const uint32_t R      = eng->cfg.hc_lowrank;
    const float    inv_hc = 1.0f / static_cast<float>(H);

    const bool use_up_blend_nax =
        !force_gemv && eng->has_nax && M_valid > kMaxSpecBatch &&
        (hc.input_mix_weight_up.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0 ||
         hc.input_mix_weight_up.dtype == STRATA_QWEN_FLASH_DTYPE_F16) &&
        ((D % 64U) == 0U) && ((R % 64U) == 0U) &&
        hc.input_mix_weight_up.shape0 == H * D &&
        hc.input_mix_weight_up.shape1 == R;

    // 1. Grouped RMSNorm (`[M_valid, H * D]` -> `b_hc_norm_f16`, skip `40 MiB` `b_hc_norm_f32` write on NAX prefill!)
    if (!skip_rmsnorm) {
        SqfGroupedNormParamsHost np{M_valid, M_padded, H, D, eng->cfg.rms_norm_eps, use_up_blend_nax ? 0U : 1U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_grouped_rmsnorm];
        [enc setBuffer:hc_buf offset:0 atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.hc_norm.offset) atIndex:1];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:3];
        [enc setBytes:&np length:sizeof(np) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(H, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 2 + 3. Fused `down` + `block_inject` + `silu/2*sigmoid` (1 dispatch of `R+H` TGs saturating all 40 cores!)
    if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch &&
        hc.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0 &&
        hc.block_inject_weight.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
        ((hc.input_mix_weight_down.shape1 % 64U) == 0U) &&
        hc.input_mix_weight_down.shape0 == R &&
        hc.block_inject_weight.shape0 == H &&
        hc.block_inject_weight.shape1 == hc.input_mix_weight_down.shape1) {
        uint64_t wb = 0ULL;
        sqf_q8_slice_layout(R, hc.input_mix_weight_down.shape1, &wb, nullptr);
        SqfHcDownInjParamsHost hp{
            M_valid, R, H, hc.input_mix_weight_down.shape1, inv_hc, 1U, 0U, 0U
        };
        [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_q8_hc_down_inj_act_m[M_valid - 1U], "sqf_q8_hc_down_inj_act_batch", M_valid)];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_down.offset) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_down.offset + wb) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.block_inject_weight.offset) atIndex:2];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:4];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:5];
        [enc setBytes:&hp length:sizeof(hp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(R + H, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch &&
        hc.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
        hc.block_inject_weight.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
        ((hc.input_mix_weight_down.shape1 % 8U) == 0U) &&
        hc.input_mix_weight_down.shape0 == R &&
        hc.block_inject_weight.shape0 == H &&
        hc.block_inject_weight.shape1 == hc.input_mix_weight_down.shape1) {
        SqfHcDownInjParamsHost hp{
            M_valid, R, H, hc.input_mix_weight_down.shape1, inv_hc, 1U, 0U, 0U
        };
        [enc setComputePipelineState:eng->pso_hc_down_inj_act];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_down.offset) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.block_inject_weight.offset) atIndex:1];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:4];
        [enc setBytes:&hp length:sizeof(hp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(R + H, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else if (!force_gemv && eng->has_nax && M_valid > kMaxSpecBatch &&
        (hc.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0 ||
         hc.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_F16) &&
        hc.block_inject_weight.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
        ((hc.input_mix_weight_down.shape1 % 128U) == 0U) &&
        ((R % 64U) == 0U) &&
        hc.input_mix_weight_down.shape0 == R &&
        hc.block_inject_weight.shape0 == H &&
        hc.block_inject_weight.shape1 == hc.input_mix_weight_down.shape1) {
        const uint32_t cols    = hc.input_mix_weight_down.shape1;
        const bool     use_m64 = (M_padded >= 64U) && ((R / 64U) * (M_padded / 64U) >= 40U);
        const uint32_t m_tile  = use_m64 ? 64U : 32U;
        const uint32_t n_th    = use_m64 ? 128U : 64U;
        SqfHcDownNaxParamsHost dp{M_valid, M_padded, R, cols, inv_hc, 0U, 0U, 0U};
        if (hc.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) {
            uint64_t wb = 0ULL;
            sqf_q8_slice_layout(R, cols, &wb, nullptr);
            const uint64_t off_x_sc = sqf_align_256(static_cast<uint64_t>(M_padded) * cols);
            const uint64_t off_w_i8 = sqf_align_256(off_x_sc + static_cast<uint64_t>(M_padded) * sizeof(float));
            const uint64_t off_w_sc = sqf_align_256(off_w_i8 + static_cast<uint64_t>(R) * cols);
            const uint64_t need_i8  = off_w_sc + static_cast<uint64_t>(R) * sizeof(float);
            if (eng->int8_prefill_enabled && M_valid >= 32U && M_padded >= 32U &&
                [eng->b_moe_down_part length] >= need_i8) {
                sqf_encode_quant_x_i8(enc, eng, eng->b_hc_norm_f16, 0U, eng->b_moe_down_part, 0U,
                                      static_cast<NSUInteger>(off_x_sc), M_valid, M_padded, cols);
                SqfI8QuantParamsHost wp{M_valid, M_padded, R, R, cols, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_q8_requant_row_i8];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_down.offset) atIndex:0];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_down.offset + wb) atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_i8) atIndex:2];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_sc) atIndex:3];
                [enc setBytes:&wp length:sizeof(wp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:(use_m64 ? eng->pso_i8_hc_down_act_nax_m64 : eng->pso_i8_hc_down_act_nax_m32)];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_i8) atIndex:0];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_sc) atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:0U atIndex:2];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_x_sc) atIndex:3];
                [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:4];
                [enc setBytes:&dp length:sizeof(dp) atIndex:5];
            } else {
                [enc setComputePipelineState:(use_m64 ? eng->pso_q8_hc_down_act_nax_m64 : eng->pso_q8_hc_down_act_nax_m32)];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_down.offset) atIndex:0];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_down.offset + wb) atIndex:1];
                [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:2];
                [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:3];
                [enc setBytes:&dp length:sizeof(dp) atIndex:4];
            }
        } else {
            [enc setComputePipelineState:(use_m64 ? eng->pso_f16_hc_down_act_nax_m64 : eng->pso_f16_hc_down_act_nax_m32)];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_down.offset) atIndex:0];
            [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:1];
            [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:2];
            [enc setBytes:&dp length:sizeof(dp) atIndex:3];
        }
        [enc dispatchThreadgroups:MTLSizeMake(R / 64U, (M_padded + m_tile - 1U) / m_tile, 1)
            threadsPerThreadgroup:MTLSizeMake(n_th, 1, 1)];

        SqfHcInjectPrefillParamsHost ip{M_valid, H, cols, inv_hc};
        [enc setComputePipelineState:eng->pso_f16_hc_inject_prefill];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.block_inject_weight.offset) atIndex:0];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:1];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:2];
        [enc setBytes:&ip length:sizeof(ip) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, (M_valid + 3U) / 4U, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        sqf_encode_f16_linear(enc, eng, w_buf, hc.input_mix_weight_down,
                              eng->b_hc_norm_f16, 0, nil, 0, eng->b_hc_down_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        sqf_encode_f16_linear(enc, eng, w_buf, hc.block_inject_weight,
                              eng->b_hc_norm_f16, 0, nil, 0, eng->b_hc_inj_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);

        SqfHcDownActParamsHost dp{M_valid, M_padded, R, H, inv_hc, 1U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_hc_down_act];
        [enc setBuffer:eng->b_hc_down_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:1];
        [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:3];
        [enc setBytes:&dp length:sizeof(dp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((R + 255U) / 256U, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 4. `up` projection + 4-stream weighted mean blend (fused single-dispatch for BOTH Q8_0/F16 prefill NAX & decode!)
    if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch && H == 4U &&
        hc.input_mix_weight_up.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0 &&
        ((R % 64U) == 0U) &&
        hc.input_mix_weight_up.shape0 == H * D &&
        hc.input_mix_weight_up.shape1 == R) {
        uint64_t wb = 0ULL;
        sqf_q8_slice_layout(H * D, R, &wb, nullptr);
        SqfQ8HcUpBlendParamsHost bp{M_valid, D, R, inv_hc, write_x_f32 ? 1U : 0U, 0U};
        [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_q8_hc_up_blend_m[M_valid - 1U], "sqf_q8_hc_up_blend", M_valid)];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_up.offset) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_up.offset + wb) atIndex:1];
        [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:3];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:4];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:5];
        [enc setBytes:&bp length:sizeof(bp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (use_up_blend_nax) {
        const bool     use_m64 = (M_padded >= 64U) && ((D / 64U) * (M_padded / 64U) >= 80U);
        const uint32_t m_tile  = use_m64 ? 64U : 32U;
        const uint32_t n_th    = use_m64 ? 128U : 64U;
        SqfHcUpBlendNaxParamsHost bp{M_valid, M_padded, D, R, H, inv_hc, write_x_f32 ? 1U : 0U, 0U};
        if (hc.input_mix_weight_up.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) {
            uint64_t wb = 0ULL;
            sqf_q8_slice_layout(H * D, R, &wb, nullptr);
            const uint32_t up_rows  = H * D;
            const uint64_t off_u_sc = sqf_align_256(static_cast<uint64_t>(M_padded) * R);
            const uint64_t off_w_i8 = sqf_align_256(off_u_sc + static_cast<uint64_t>(M_padded) * sizeof(float));
            const uint64_t off_w_sc = sqf_align_256(off_w_i8 + static_cast<uint64_t>(up_rows) * R);
            const uint64_t need_i8  = off_w_sc + static_cast<uint64_t>(up_rows) * sizeof(float);
            if (eng->int8_prefill_enabled && M_valid >= 32U && M_padded >= 32U &&
                [eng->b_moe_down_part length] >= need_i8) {
                sqf_encode_quant_x_i8(enc, eng, eng->b_hc_u_f16, 0U, eng->b_moe_down_part, 0U,
                                      static_cast<NSUInteger>(off_u_sc), M_valid, M_padded, R);
                SqfI8QuantParamsHost wp{M_valid, M_padded, up_rows, up_rows, R, 0U, 0U, 0U};
                const bool small_k = (R <= 512U);
                [enc setComputePipelineState:(small_k ? eng->pso_q8_requant_row_i8_w32 : eng->pso_q8_requant_row_i8)];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_up.offset) atIndex:0];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_up.offset + wb) atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_i8) atIndex:2];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_sc) atIndex:3];
                [enc setBytes:&wp length:sizeof(wp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(up_rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(small_k ? 32 : 128, 1, 1)];

                [enc setComputePipelineState:(use_m64 ? eng->pso_i8_hc_up_blend_nax_m64 : eng->pso_i8_hc_up_blend_nax_m32)];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_i8) atIndex:0];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_w_sc) atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:0U atIndex:2];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_u_sc) atIndex:3];
                [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:4];
                [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:5];
                [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:6];
                [enc setBytes:&bp length:sizeof(bp) atIndex:7];
            } else {
                [enc setComputePipelineState:(use_m64 ? eng->pso_q8_hc_up_blend_nax_m64 : eng->pso_q8_hc_up_blend_nax_m32)];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_up.offset) atIndex:0];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_up.offset + wb) atIndex:1];
                [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:2];
                [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:4];
                [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:5];
                [enc setBytes:&bp length:sizeof(bp) atIndex:6];
            }
        } else {
            [enc setComputePipelineState:(use_m64 ? eng->pso_f16_hc_up_blend_nax_m64 : eng->pso_f16_hc_up_blend_nax_m32)];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(hc.input_mix_weight_up.offset) atIndex:0];
            [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:1];
            [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:4];
            [enc setBytes:&bp length:sizeof(bp) atIndex:5];
        }
        [enc dispatchThreadgroups:MTLSizeMake(D / 64U, (M_padded + m_tile - 1U) / m_tile, 1)
            threadsPerThreadgroup:MTLSizeMake(n_th, 1, 1)];
    } else {
        sqf_encode_f16_linear(enc, eng, w_buf, hc.input_mix_weight_up,
                              eng->b_hc_u_f16, 0, nil, 0, eng->b_hc_up_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        SqfHcBlendParamsHost bp{M_valid, M_padded, D, H, inv_hc, write_x_f32 ? 1U : 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_hc_blend];
        [enc setBuffer:eng->b_hc_up_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:1];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:3];
        [enc setBytes:&bp length:sizeof(bp) atIndex:4];
        const uint32_t d4 = D / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}

// Encodes Final 4-Stream -> 1-Stream Mixer (`use_combine = false`) -> `b_x_in_f16`
void sqf_encode_mixer(
    id<MTLComputeCommandEncoder> enc,
    StrataQwenFlashGpuEngine* eng,
    id<MTLBuffer> w_buf,
    const StrataQwenFlashMixerDesc& mx,
    id<MTLBuffer> hc_buf,
    NSUInteger hc_byte_off,
    uint32_t M_valid,
    uint32_t M_padded,
    bool force_gemv
) {
    const uint32_t D      = eng->cfg.hidden_size;
    const uint32_t H      = eng->cfg.hc_count;
    const uint32_t R      = eng->cfg.hc_lowrank;
    const float    inv_hc = 1.0f / static_cast<float>(H);

    {
        SqfGroupedNormParamsHost np{M_valid, M_padded, H, D, eng->cfg.rms_norm_eps, 1U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_grouped_rmsnorm];
        [enc setBuffer:hc_buf offset:hc_byte_off atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mx.hc_norm.offset) atIndex:1];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:3];
        [enc setBytes:&np length:sizeof(np) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(H, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch &&
        mx.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0 &&
        ((mx.input_mix_weight_down.shape1 % 64U) == 0U) &&
        mx.input_mix_weight_down.shape0 == R) {
        uint64_t wb = 0ULL;
        sqf_q8_slice_layout(R, mx.input_mix_weight_down.shape1, &wb, nullptr);
        SqfHcDownInjParamsHost hp{
            M_valid, R, H, mx.input_mix_weight_down.shape1, inv_hc, 0U, 0U, 0U
        };
        [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_q8_hc_down_inj_act_m[M_valid - 1U], "sqf_q8_hc_down_inj_act_batch", M_valid)];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mx.input_mix_weight_down.offset) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mx.input_mix_weight_down.offset + wb) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mx.input_mix_weight_down.offset) atIndex:2];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:4];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:5];
        [enc setBytes:&hp length:sizeof(hp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch &&
        mx.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
        ((mx.input_mix_weight_down.shape1 % 8U) == 0U) &&
        mx.input_mix_weight_down.shape0 == R) {
        SqfHcDownInjParamsHost hp{
            M_valid, R, H, mx.input_mix_weight_down.shape1, inv_hc, 0U, 0U, 0U
        };
        [enc setComputePipelineState:eng->pso_hc_down_inj_act];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mx.input_mix_weight_down.offset) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mx.input_mix_weight_down.offset) atIndex:1];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:4];
        [enc setBytes:&hp length:sizeof(hp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        sqf_encode_f16_linear(enc, eng, w_buf, mx.input_mix_weight_down,
                              eng->b_hc_norm_f16, 0, nil, 0, eng->b_hc_down_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        SqfHcDownActParamsHost dp{M_valid, M_padded, R, H, inv_hc, 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_hc_down_act];
        [enc setBuffer:eng->b_hc_down_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_hc_down_f32 offset:0 atIndex:1];
        [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:3];
        [enc setBytes:&dp length:sizeof(dp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((R + 255U) / 256U, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch && H == 4U &&
        mx.input_mix_weight_up.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0 &&
        ((R % 64U) == 0U) &&
        mx.input_mix_weight_up.shape0 == H * D &&
        mx.input_mix_weight_up.shape1 == R) {
        uint64_t wb = 0ULL;
        sqf_q8_slice_layout(H * D, R, &wb, nullptr);
        SqfQ8HcUpBlendParamsHost bp{M_valid, D, R, inv_hc, 0U, 0U};
        [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_q8_hc_up_blend_m[M_valid - 1U], "sqf_q8_hc_up_blend", M_valid)];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mx.input_mix_weight_up.offset) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mx.input_mix_weight_up.offset + wb) atIndex:1];
        [enc setBuffer:eng->b_hc_u_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:3];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:4];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:5];
        [enc setBytes:&bp length:sizeof(bp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        sqf_encode_f16_linear(enc, eng, w_buf, mx.input_mix_weight_up,
                              eng->b_hc_u_f16, 0, nil, 0, eng->b_hc_up_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        SqfHcBlendParamsHost bp{M_valid, M_padded, D, H, inv_hc, 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_hc_blend];
        [enc setBuffer:eng->b_hc_up_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:1];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:3];
        [enc setBytes:&bp length:sizeof(bp) atIndex:4];
        const uint32_t d4 = D / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}

inline void sqf_prof_mark(
    StrataQwenFlashGpuEngine* eng,
    id<MTLComputeCommandEncoder> __strong& enc,
    id<MTLCommandBuffer> __strong& cb,
    uint32_t bucket
) {
    if (__builtin_expect(!eng->prof_active, 1)) return;
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const double gs = [cb GPUStartTime];
    const double ge = [cb GPUEndTime];
    if (ge > gs && bucket < 12U) eng->prof_ms[bucket] += (ge - gs) * 1000.0;
    cb  = [eng->queue commandBuffer];
    enc = [cb computeCommandEncoder];
}

// Encodes one complete Hybrid Decoder Layer (`GatedDeltaNet` OR `QSA + Indexer`, plus `PLE` if `has_ple`, plus `512`-Expert MoE)
void sqf_encode_decoder_layer(
    id<MTLComputeCommandEncoder> __strong& enc,
    id<MTLCommandBuffer> __strong& cb,
    StrataQwenFlashGpuEngine* eng,
    const StrataQwenFlashLayerDesc& ld,
    id<MTLBuffer> hc_buf,
    uint32_t lin_slot,
    uint32_t full_slot,
    uint32_t seq_pos_start,
    uint32_t M_valid,
    uint32_t M_padded,
    uint32_t save_snap_mask,
    bool force_gemv,
    uint32_t tree_mode = 0U,
    bool kv_only = false
) {
    id<MTLBuffer>  w_buf         = sqf_part_buf(eng, ld.part_id);
    const uint32_t D             = eng->cfg.hidden_size;
    const uint32_t H             = eng->cfg.hc_count;
    const uint32_t hc_dim        = H * D;
    const float    eps           = eng->cfg.rms_norm_eps;
    const uint32_t tree_branch_t = (tree_mode == 1U) ? 4U : 0U;

    // 0. Optional Per-Layer Embedding (`PLE` at `layer_idx == 1`, BEFORE `attn_hc`!)
    if (ld.has_ple != 0U) {
        const uint32_t ple_dim = eng->cfg.ple_embed_dim;
        {
            SqfPadCastParamsHost cp{M_valid, M_padded, ple_dim, 0U};
            [enc setComputePipelineState:eng->pso_f32_to_f16_pad];
            [enc setBuffer:eng->b_ple_emb_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_ple_emb_f16 offset:0 atIndex:1];
            [enc setBytes:&cp length:sizeof(cp) atIndex:2];
            const uint32_t d4 = ple_dim / 4U;
            [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // `ple.key_proj` -> `b_hc_up_f32 [M_valid, hc_dim]`, `ple.value_proj` -> `b_block_y_f32 [M_valid, D]`
        sqf_encode_f16_linear(enc, eng, w_buf, ld.ple.key_proj,
                              eng->b_ple_emb_f16, 0, nil, 0, eng->b_hc_up_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        sqf_encode_f16_linear(enc, eng, w_buf, ld.ple.value_proj,
                              eng->b_ple_emb_f16, 0, nil, 0, eng->b_block_y_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);

        {
            SqfPleGateParamsHost gp{M_valid, D, H, eps};
            [enc setComputePipelineState:eng->pso_ple_gate_gv];
            [enc setBuffer:eng->b_hc_up_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_block_y_f32 offset:0 atIndex:1];
            [enc setBuffer:hc_buf offset:0 atIndex:2];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.ple.norm_key.offset) atIndex:3];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.ple.norm_query.offset) atIndex:4];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.ple.norm_conv.offset) atIndex:5];
            [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:6]; // `gv_out`
            [enc setBuffer:eng->b_ple_gvn_f32 offset:0 atIndex:7]; // `gv_norm_out`
            [enc setBytes:&gp length:sizeof(gp) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(H, M_valid, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        {
            SqfPleConvParamsHost pc{M_valid, hc_dim, eng->cfg.ngram_size, save_snap_mask, tree_branch_t, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_ple_conv1d_add];
            [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_ple_gvn_f32 offset:0 atIndex:1];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.ple.conv1d_weight.offset) atIndex:2];
            [enc setBuffer:eng->b_ple_conv_state offset:0 atIndex:3];
            [enc setBuffer:hc_buf offset:0 atIndex:4];
            for (uint32_t s = 0U; s < kMaxSpecSnapSlots; ++s) {
                id<MTLBuffer> sb = eng->b_ple_conv_slots[s] ? eng->b_ple_conv_slots[s] : eng->b_ple_conv_state;
                [enc setBuffer:sb offset:0 atIndex:(5U + s)];
            }
            [enc setBytes:&pc length:sizeof(pc) atIndex:11];
            [enc dispatchThreadgroups:MTLSizeMake((hc_dim + 255U) / 256U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }

    // 1. Pre-Attention 4-Stream Hyper-Connection
    sqf_encode_hc_pre(enc, eng, w_buf, ld.attn_hc, hc_buf, M_valid, M_padded, false, force_gemv);
    sqf_prof_mark(eng, enc, cb, 0U); // [0] HC_pre_attn + PLE

    // 2. Hybrid Attention Block (`GatedDeltaNet` OR `QSA + QSAIndexer`)
    if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) {
        const uint32_t n_k      = eng->cfg.linear_num_key_heads;
        const uint32_t n_v      = eng->cfg.linear_num_value_heads;
        const uint32_t k_dim    = n_k * eng->cfg.linear_key_head_dim;
        const uint32_t v_dim    = n_v * eng->cfg.linear_value_head_dim;
        const uint32_t conv_dim = 2U * k_dim + v_dim;

        if ((force_gemv || !eng->has_nax) && M_valid <= kMaxSpecBatch) {
            SqfLinQzGemvParamsHost qp{M_valid, conv_dim, v_dim, D};
            [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_q4_gemv_lin_qz_m[M_valid - 1U], "sqf_q4_gemv_lin_qz_batch", M_valid)];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_qkv.weight_off) atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_qkv.scales_off) atIndex:1];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_qkv.biases_off) atIndex:2];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_z.weight_off) atIndex:3];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_z.scales_off) atIndex:4];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_z.biases_off) atIndex:5];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:6];
            [enc setBuffer:eng->b_qkv offset:0 atIndex:7];
            [enc setBuffer:eng->b_z offset:0 atIndex:8];
            [enc setBytes:&qp length:sizeof(qp) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake((conv_dim + v_dim + 7U) / 8U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            const bool reuse_x = sqf_can_use_q4_i8(eng, ld.lin_in_proj_qkv, M_valid, M_padded, force_gemv) &&
                                 (ld.lin_in_proj_qkv.cols == ld.lin_in_proj_z.cols);
            sqf_encode_q4_linear(enc, eng, w_buf, ld.lin_in_proj_qkv,
                                 eng->b_x_in_f16, 0, eng->b_qkv, 0, nil, 0,
                                 M_valid, M_padded, kOutF16, force_gemv, false);
            sqf_encode_q4_linear(enc, eng, w_buf, ld.lin_in_proj_z,
                                 eng->b_x_in_f16, 0, eng->b_z, 0, nil, 0,
                                 M_valid, M_padded, kOutF16, force_gemv, reuse_x);
        }

        if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch &&
            ((ld.lin_in_proj_a.shape1 % 8U) == 0U) &&
            ld.lin_in_proj_a.shape1 == ld.lin_in_proj_b.shape1) {
            SqfLinAbParamsHost abp{
                M_valid, ld.lin_in_proj_a.shape0, ld.lin_in_proj_b.shape0, ld.lin_in_proj_a.shape1
            };
            [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_f16_lin_ab_m[M_valid - 1U], "sqf_f16_lin_ab_batch", M_valid)];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_a.offset) atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_b.offset) atIndex:1];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_a offset:0 atIndex:3];
            [enc setBuffer:eng->b_b offset:0 atIndex:4];
            [enc setBytes:&abp length:sizeof(abp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(abp.a_rows + abp.b_rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (!force_gemv && M_valid > kMaxSpecBatch &&
            ld.lin_in_proj_a.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
            ld.lin_in_proj_b.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
            ((ld.lin_in_proj_a.shape1 % 8U) == 0U) &&
            ld.lin_in_proj_a.shape1 == ld.lin_in_proj_b.shape1) {
            SqfLinAbParamsHost abp{
                M_valid, ld.lin_in_proj_a.shape0, ld.lin_in_proj_b.shape0, ld.lin_in_proj_a.shape1
            };
            [enc setComputePipelineState:eng->pso_f16_lin_ab_prefill];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_a.offset) atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_in_proj_b.offset) atIndex:1];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_a offset:0 atIndex:3];
            [enc setBuffer:eng->b_b offset:0 atIndex:4];
            [enc setBytes:&abp length:sizeof(abp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(abp.a_rows + abp.b_rows, (M_valid + 7U) / 8U, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            sqf_encode_f16_linear(enc, eng, w_buf, ld.lin_in_proj_a,
                                  eng->b_x_in_f16, 0, nil, 0, eng->b_a, 0,
                                  M_valid, M_padded, kOutF32, force_gemv);
            sqf_encode_f16_linear(enc, eng, w_buf, ld.lin_in_proj_b,
                                  eng->b_x_in_f16, 0, nil, 0, eng->b_b, 0,
                                  M_valid, M_padded, kOutF32, force_gemv);
        }

        {
            const NSUInteger conv_off = static_cast<NSUInteger>(static_cast<uint64_t>(lin_slot) * eng->conv_layer_stride);
            SqfDeltaConvParamsHost cp{
                M_valid, conv_dim, save_snap_mask, lin_slot, eng->cfg.num_linear_layers, tree_branch_t, 0U, 0U
            };
            if (!force_gemv && M_valid > kMaxSpecBatch && save_snap_mask == 0U && tree_branch_t == 0U) {
                [enc setComputePipelineState:eng->pso_deltanet_conv1d_prefill];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:0];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_conv1d_weight.offset) atIndex:1];
                [enc setBuffer:eng->b_conv_state offset:conv_off atIndex:2];
                [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:3];
                [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((conv_dim + 255U) / 256U, (M_valid + 15U) / 16U, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:eng->pso_deltanet_conv1d_save_tail];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:0];
                [enc setBuffer:eng->b_conv_state offset:conv_off atIndex:1];
                [enc setBytes:&cp length:sizeof(cp) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((conv_dim + 255U) / 256U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [enc setComputePipelineState:eng->pso_deltanet_conv1d];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:0];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_conv1d_weight.offset) atIndex:1];
                [enc setBuffer:eng->b_conv_state offset:conv_off atIndex:2];
                [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:3];
                for (uint32_t s = 0U; s < kMaxSpecSnapSlots; ++s) {
                    id<MTLBuffer> sb = eng->b_conv_slots[s] ? eng->b_conv_slots[s] : eng->b_conv_state;
                    [enc setBuffer:sb offset:conv_off atIndex:(4U + s)];
                }
                [enc setBytes:&cp length:sizeof(cp) atIndex:10];
                [enc dispatchThreadgroups:MTLSizeMake((conv_dim + 255U) / 256U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
        sqf_prof_mark(eng, enc, cb, 1U); // [1] Lin_QKVZ_AB_Conv

        SqfDeltaNetParamsHost dp{
            M_valid, M_padded, n_k, n_v, eps,
            eng->cfg.linear_output_gate_sigmoid,
            save_snap_mask, lin_slot, eng->cfg.num_linear_layers,
            tree_branch_t, 0U, 0U
        };

        // Metal 4.1 SLC-Resident 3-Kernel Chunk-32 WY NAX GatedDeltaNet for Prefill (`M > 8`, zero extra VRAM!)
        // vs 192-TG 100%-Coalesced Serial Recurrence for Decode/Spec/Tree (`M <= 8`, 100% bit-exact K>0 == K=0!)
        {
            const NSUInteger rec_off    = static_cast<NSUInteger>(static_cast<uint64_t>(lin_slot) * eng->rec_layer_stride);
            const uint32_t   num_chunks = M_padded / 32U;
            const uint64_t   qk_bytes   = static_cast<uint64_t>(n_k) * M_padded * 128ULL * sizeof(uint16_t);
            const uint64_t   dot_bytes  = static_cast<uint64_t>(n_k) * num_chunks * 1024ULL * sizeof(float);
            const uint64_t   wu_bytes   = static_cast<uint64_t>(n_v) * M_padded * 128ULL * sizeof(uint16_t);
            const uint64_t   p_bytes    = static_cast<uint64_t>(n_v) * num_chunks * 1024ULL * sizeof(uint16_t);
            const uint64_t   wy_need    = 2ULL * qk_bytes + 2ULL * dot_bytes + 2ULL * wu_bytes + p_bytes;
            const uint64_t   g_bytes    = static_cast<uint64_t>(n_v) * M_padded * sizeof(float);
            const uint64_t   raw_bytes  = static_cast<uint64_t>(M_padded) * n_v * 128ULL * sizeof(float);
            const bool use_wy_nax =
                !force_gemv && eng->has_nax && (M_valid > kMaxSpecBatch) &&
                (save_snap_mask == 0U) && (tree_branch_t == 0U) &&
                ((M_padded % 32U) == 0U) && (num_chunks > 0U) &&
                (eng->cfg.linear_key_head_dim == 128U) && (eng->cfg.linear_value_head_dim == 128U) &&
                ([eng->b_moe_down_part length] >= wy_need) &&
                ([eng->b_dn_gates length] >= g_bytes) &&
                ([eng->b_dn_raw_out length] >= raw_bytes) &&
                !eng->force_serial_dn;

            if (use_wy_nax) {
                const NSUInteger off_q_norm = 0U;
                const NSUInteger off_k_hat  = static_cast<NSUInteger>(qk_bytes);
                const NSUInteger off_kk_dot = static_cast<NSUInteger>(2ULL * qk_bytes);
                const NSUInteger off_qk_dot = static_cast<NSUInteger>(2ULL * qk_bytes + dot_bytes);
                const NSUInteger off_W      = static_cast<NSUInteger>(2ULL * qk_bytes + 2ULL * dot_bytes);
                const NSUInteger off_U      = static_cast<NSUInteger>(2ULL * qk_bytes + 2ULL * dot_bytes + wu_bytes);
                const NSUInteger off_P      = static_cast<NSUInteger>(2ULL * qk_bytes + 2ULL * dot_bytes + 2ULL * wu_bytes);

                SqfDeltaWyParamsHost wp{
                    M_valid, M_padded, num_chunks, n_k, n_v, eps, eng->cfg.linear_output_gate_sigmoid, 0U
                };

                [enc setComputePipelineState:eng->pso_deltanet_wy_qk];
                [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:0];
                [enc setBuffer:eng->b_moe_down_part offset:off_q_norm atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:off_k_hat atIndex:2];
                [enc setBuffer:eng->b_moe_down_part offset:off_kk_dot atIndex:3];
                [enc setBuffer:eng->b_moe_down_part offset:off_qk_dot atIndex:4];
                [enc setBytes:&wp length:sizeof(wp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(num_chunks, n_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                [enc setComputePipelineState:eng->pso_deltanet_wy_prep_hv];
                [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:0];
                [enc setBuffer:eng->b_a offset:0 atIndex:1];
                [enc setBuffer:eng->b_b offset:0 atIndex:2];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_A_log.offset) atIndex:3];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_dt_bias.offset) atIndex:4];
                [enc setBuffer:eng->b_moe_down_part offset:off_k_hat atIndex:5];
                [enc setBuffer:eng->b_moe_down_part offset:off_kk_dot atIndex:6];
                [enc setBuffer:eng->b_moe_down_part offset:off_qk_dot atIndex:7];
                [enc setBuffer:eng->b_dn_gates offset:0 atIndex:8];
                [enc setBuffer:eng->b_moe_down_part offset:off_W atIndex:9];
                [enc setBuffer:eng->b_moe_down_part offset:off_U atIndex:10];
                [enc setBuffer:eng->b_moe_down_part offset:off_P atIndex:11];
                [enc setBytes:&wp length:sizeof(wp) atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake(num_chunks, n_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                [enc setComputePipelineState:eng->pso_deltanet_wy_recur];
                [enc setBuffer:eng->b_dn_gates offset:0 atIndex:0];
                [enc setBuffer:eng->b_moe_down_part offset:off_q_norm atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:off_k_hat atIndex:2];
                [enc setBuffer:eng->b_moe_down_part offset:off_W atIndex:3];
                [enc setBuffer:eng->b_moe_down_part offset:off_U atIndex:4];
                [enc setBuffer:eng->b_moe_down_part offset:off_P atIndex:5];
                [enc setBuffer:eng->b_rec_state offset:rec_off atIndex:6];
                [enc setBuffer:eng->b_dn_raw_out offset:0 atIndex:7];
                [enc setBytes:&wp length:sizeof(wp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(4, n_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                [enc setComputePipelineState:eng->pso_deltanet_post_batch];
                [enc setBuffer:eng->b_dn_raw_out offset:0 atIndex:0];
                [enc setBuffer:eng->b_z offset:0 atIndex:1];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_norm.offset) atIndex:2];
                [enc setBuffer:eng->b_attn_out offset:0 atIndex:3];
                [enc setBytes:&dp length:sizeof(dp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(n_v, M_padded, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc setComputePipelineState:eng->pso_deltanet_prep_batch];
                [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:0];
                [enc setBuffer:eng->b_a offset:0 atIndex:1];
                [enc setBuffer:eng->b_b offset:0 atIndex:2];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_A_log.offset) atIndex:3];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_dt_bias.offset) atIndex:4];
                [enc setBuffer:eng->b_dn_qk_hat offset:0 atIndex:5];
                [enc setBuffer:eng->b_dn_gates offset:0 atIndex:6];
                [enc setBytes:&dp length:sizeof(dp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(2U * n_k + 1U, M_valid, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_deltanet_rec_batch];
                [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:0];
                [enc setBuffer:eng->b_dn_qk_hat offset:0 atIndex:1];
                [enc setBuffer:eng->b_dn_gates offset:0 atIndex:2];
                [enc setBuffer:eng->b_rec_state offset:rec_off atIndex:3];
                [enc setBuffer:eng->b_dn_raw_out offset:0 atIndex:4];
                [enc setBytes:&dp length:sizeof(dp) atIndex:5];
                for (uint32_t s = 0U; s < kMaxSpecSnapSlots; ++s) {
                    id<MTLBuffer> sb = eng->b_rec_slots[s] ? eng->b_rec_slots[s] : eng->b_rec_state;
                    [enc setBuffer:sb offset:rec_off atIndex:(6U + s)];
                }
                [enc dispatchThreadgroups:MTLSizeMake(4, n_v, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_deltanet_post_batch];
                [enc setBuffer:eng->b_dn_raw_out offset:0 atIndex:0];
                [enc setBuffer:eng->b_z offset:0 atIndex:1];
                [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.lin_norm.offset) atIndex:2];
                [enc setBuffer:eng->b_attn_out offset:0 atIndex:3];
                [enc setBytes:&dp length:sizeof(dp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(n_v, M_padded, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        }
        sqf_prof_mark(eng, enc, cb, 2U); // [2] DeltaNet_Rec

        sqf_encode_q4_linear(enc, eng, w_buf, ld.lin_out_proj,
                             eng->b_attn_out, 0, nil, 0, eng->b_block_y_f32, 0,
                             M_valid, M_padded, kOutF32, force_gemv);
        sqf_prof_mark(eng, enc, cb, 3U); // [3] Lin_Out
    } else {
        const uint32_t n_q        = eng->cfg.num_attention_heads;
        const uint32_t n_kv       = eng->cfg.num_key_value_heads;
        const uint32_t head_dim   = eng->cfg.head_dim;
        const uint32_t rot_dim    = eng->cfg.rotary_dim;
        const uint32_t q_rows     = ld.attn_q_proj.rows_logical;
        const uint32_t kv_rows    = ld.attn_k_proj.rows_logical;
        const uint32_t max_rope_p = (tree_mode == 1U) ? (seq_pos_start + 3U)
                                  : ((tree_mode == 2U || tree_mode == 3U) ? seq_pos_start
                                                                          : (seq_pos_start + M_valid - 1U));
        const uint32_t total_kv   = max_rope_p + 1U;
        const uint32_t kv_zero    = (tree_mode == 0U) ? ((total_kv + 63U) & ~63U) : 0U;

        const NSUInteger kv_off      = static_cast<NSUInteger>(static_cast<uint64_t>(full_slot) * eng->kv_layer_stride);
        const NSUInteger idx_raw_off = static_cast<NSUInteger>(static_cast<uint64_t>(full_slot) * eng->idx_raw_layer_stride);
        const NSUInteger idx_blk_off = static_cast<NSUInteger>(static_cast<uint64_t>(full_slot) * eng->idx_blk_layer_stride);

        if (kv_only) {
            // MTP KV-Sync / Prefill Warmup fast-path (`run_head == false`): skip `attn_q_proj`, `idx_score/topk`, `QSA_Attn`, `attn_o_proj`, and entire `MoE`!
            sqf_encode_q4_linear(enc, eng, w_buf, ld.attn_k_proj,
                                 eng->b_x_in_f16, 0, eng->b_k_raw, 0, nil, 0,
                                 M_valid, M_padded, kOutF16, force_gemv);
            sqf_encode_q4_linear(enc, eng, w_buf, ld.attn_v_proj,
                                 eng->b_x_in_f16, 0, eng->b_v_raw, 0, nil, 0,
                                 M_valid, M_padded, kOutF16, force_gemv);
        } else if ((force_gemv || !eng->has_nax) && M_valid <= kMaxSpecBatch) {
            SqfAttnQkvGemvParamsHost qp{M_valid, q_rows, kv_rows, D};
            [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_q4_gemv_attn_qkv_m[M_valid - 1U], "sqf_q4_gemv_attn_qkv_batch", M_valid)];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_q_proj.weight_off) atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_q_proj.scales_off) atIndex:1];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_q_proj.biases_off) atIndex:2];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_k_proj.weight_off) atIndex:3];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_k_proj.scales_off) atIndex:4];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_k_proj.biases_off) atIndex:5];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_v_proj.weight_off) atIndex:6];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_v_proj.scales_off) atIndex:7];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_v_proj.biases_off) atIndex:8];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:9];
            [enc setBuffer:eng->b_qkv offset:0 atIndex:10];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:11];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:12];
            [enc setBytes:&qp length:sizeof(qp) atIndex:13];
            [enc dispatchThreadgroups:MTLSizeMake((q_rows + 2U * kv_rows + 7U) / 8U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            const bool reuse_q_x = sqf_can_use_q4_i8(eng, ld.attn_q_proj, M_valid, M_padded, force_gemv);
            sqf_encode_q4_linear(enc, eng, w_buf, ld.attn_q_proj,
                                 eng->b_x_in_f16, 0, eng->b_qkv, 0, nil, 0,
                                 M_valid, M_padded, kOutF16, force_gemv, false);
            sqf_encode_q4_linear(enc, eng, w_buf, ld.attn_k_proj,
                                 eng->b_x_in_f16, 0, eng->b_k_raw, 0, nil, 0,
                                 M_valid, M_padded, kOutF16, force_gemv,
                                 reuse_q_x && (ld.attn_k_proj.cols == ld.attn_q_proj.cols));
            sqf_encode_q4_linear(enc, eng, w_buf, ld.attn_v_proj,
                                 eng->b_x_in_f16, 0, eng->b_v_raw, 0, nil, 0,
                                 M_valid, M_padded, kOutF16, force_gemv,
                                 reuse_q_x && (ld.attn_v_proj.cols == ld.attn_q_proj.cols));
        }

        // QSA Indexer projections + cache update
        const uint32_t comp_ratio    = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
        const uint32_t block_topk    = std::max<uint32_t>(1U, eng->cfg.indexer_budget / comp_ratio);
        const uint32_t first_blk     = seq_pos_start / comp_ratio;
        const uint32_t end_complete  = total_kv / comp_ratio;
        const uint32_t num_new_blks  = (end_complete > first_blk) ? (end_complete - first_blk) : 0U;
        const uint32_t max_q_complete= total_kv / comp_ratio;
        const bool     any_sparse    = (max_q_complete > block_topk);
        const bool     d2_prev_blk   = (tree_mode == 3U) && (num_new_blks == 0U) &&
                                       (seq_pos_start > 0U) && ((seq_pos_start % comp_ratio) == 0U);
        const uint32_t b_patch_blk   = d2_prev_blk ? ((seq_pos_start - 1U) / comp_ratio) : first_blk;

        sqf_encode_f16_linear(enc, eng, w_buf, ld.indexer_qk_proj,
                              eng->b_x_in_f16, 0, nil, 0, eng->b_idx_qk_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);

        SqfIndexerParamsHost ip{
            M_valid,
            eng->cfg.indexer_n_heads,
            eng->cfg.indexer_head_dim,
            rot_dim,
            comp_ratio,
            block_topk,
            seq_pos_start,
            b_patch_blk,
            num_new_blks,
            eps,
            eng->max_blocks_alloc,
            tree_mode,
            M_padded,
            0U, 0U, 0U
        };

        const bool use_nax_idx_score =
            !kv_only && any_sparse && !force_gemv && eng->has_nax &&
            (tree_mode == 0U) && (M_valid > kMaxSpecBatch) &&
            (ip.idx_n_heads == 4U) && (ip.idx_head_dim == 128U) &&
            ((M_padded % 64U) == 0U);

        if (use_nax_idx_score) {
            [enc setComputePipelineState:eng->pso_qsa_idx_prep_prefill];
            [enc setBuffer:eng->b_idx_qk_f32 offset:0 atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.indexer_q_norm.offset) atIndex:1];
            [enc setBuffer:eng->b_inv_freq offset:0 atIndex:2];
            [enc setBuffer:eng->b_idx_raw_k offset:idx_raw_off atIndex:3];
            [enc setBuffer:eng->b_idx_q_rot offset:0 atIndex:4];
            [enc setBytes:&ip length:sizeof(ip) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(eng->cfg.indexer_n_heads + 1U, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(eng->cfg.indexer_head_dim, 1, 1)];
        } else {
            [enc setComputePipelineState:eng->pso_qsa_idx_prep];
            [enc setBuffer:eng->b_idx_qk_f32 offset:0 atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.indexer_q_norm.offset) atIndex:1];
            [enc setBuffer:eng->b_inv_freq offset:0 atIndex:2];
            [enc setBuffer:eng->b_idx_raw_k offset:idx_raw_off atIndex:3];
            [enc setBuffer:eng->b_idx_q_rot offset:0 atIndex:4];
            [enc setBytes:&ip length:sizeof(ip) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(eng->cfg.indexer_n_heads + 1U, M_valid, 1)
                threadsPerThreadgroup:MTLSizeMake(eng->cfg.indexer_head_dim, 1, 1)];
        }

        if (num_new_blks > 0U) {
            [enc setComputePipelineState:eng->pso_qsa_idx_blocks];
            [enc setBuffer:eng->b_idx_raw_k offset:idx_raw_off atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.indexer_k_norm.offset) atIndex:1];
            [enc setBuffer:eng->b_inv_freq offset:0 atIndex:2];
            [enc setBuffer:eng->b_idx_block_k offset:idx_blk_off atIndex:3];
            [enc setBytes:&ip length:sizeof(ip) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(num_new_blks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(eng->cfg.indexer_head_dim, 1, 1)];
        }

        if (tree_mode != 0U && (num_new_blks > 0U || d2_prev_blk)) {
            [enc setComputePipelineState:eng->pso_qsa_idx_branch_b_blk];
            [enc setBuffer:eng->b_idx_raw_k offset:idx_raw_off atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.indexer_k_norm.offset) atIndex:1];
            [enc setBuffer:eng->b_inv_freq offset:0 atIndex:2];
            [enc setBuffer:eng->b_idx_blk_branch_b offset:0 atIndex:3];
            [enc setBytes:&ip length:sizeof(ip) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(eng->cfg.indexer_head_dim, 1, 1)];
        }

        if (!kv_only && any_sparse) {
            const bool use_tiled_score =
                (!force_gemv) && (tree_mode == 0U) && (M_valid > 8U) &&
                (ip.idx_n_heads == 4U) && (ip.idx_head_dim == 128U);
            if (use_nax_idx_score) {
                [enc setComputePipelineState:eng->pso_qsa_idx_score_nax];
                [enc setBuffer:eng->b_idx_q_rot offset:0 atIndex:0];
                [enc setBuffer:eng->b_idx_block_k offset:idx_blk_off atIndex:1];
                [enc setBuffer:eng->b_idx_scores offset:0 atIndex:2];
                [enc setBytes:&ip length:sizeof(ip) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((max_q_complete + 63U) / 64U, M_padded / 64U, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else if (use_tiled_score) {
                [enc setComputePipelineState:eng->pso_qsa_idx_score_tiled];
                [enc setBuffer:eng->b_idx_q_rot offset:0 atIndex:0];
                [enc setBuffer:eng->b_idx_block_k offset:idx_blk_off atIndex:1];
                [enc setBuffer:eng->b_idx_scores offset:0 atIndex:2];
                [enc setBytes:&ip length:sizeof(ip) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((max_q_complete + 63U) / 64U, (M_valid + 31U) / 32U, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else if ((M_valid >= 2U && M_valid <= kMaxSpecBatch) || tree_mode != 0U) {
                [enc setComputePipelineState:eng->pso_qsa_idx_score_batch];
                [enc setBuffer:eng->b_idx_q_rot offset:0 atIndex:0];
                [enc setBuffer:eng->b_idx_block_k offset:idx_blk_off atIndex:1];
                [enc setBuffer:eng->b_idx_scores offset:0 atIndex:2];
                [enc setBytes:&ip length:sizeof(ip) atIndex:3];
                [enc setBuffer:eng->b_idx_blk_branch_b offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((max_q_complete + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [enc setComputePipelineState:eng->pso_qsa_idx_score];
                [enc setBuffer:eng->b_idx_q_rot offset:0 atIndex:0];
                [enc setBuffer:eng->b_idx_block_k offset:idx_blk_off atIndex:1];
                [enc setBuffer:eng->b_idx_scores offset:0 atIndex:2];
                [enc setBytes:&ip length:sizeof(ip) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((max_q_complete + 7U) / 8U, M_valid, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            [enc setComputePipelineState:eng->pso_qsa_idx_topk];
            [enc setBuffer:eng->b_idx_scores offset:0 atIndex:0];
            [enc setBuffer:eng->b_idx_sel_blocks offset:0 atIndex:1];
            [enc setBytes:&ip length:sizeof(ip) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        sqf_prof_mark(eng, enc, cb, 4U); // [4] QSA_QKV + Idx

        // Step 4 Tier 2: paged slots run the EXACT selected-block residency check ONLY when needed
        // (Zero-Sync Fast-Path: when layer `full_slot` has 0 blocks on NVMe and write targets fit in UMA,
        // skip `sqf_paged_sync_point` completely -> 0 GPU stalls, 0 command buffer splits, single-pass attention!)
        // In paged slots with `mtp_unpaged`, the tiny 1-layer MTP KV (`full_slot == num_full_attn_layers`) stays
        // linear in UMA so `cb_draft` has 0 sync stalls and async True-State MTP KV-Sync runs at full speed!
        const bool is_mtp_unpaged =
            eng->kv_paged && eng->bound_slot && eng->bound_slot->mtp_unpaged &&
            (full_slot == eng->cfg.num_full_attn_layers);
        const bool       layer_paged     = eng->kv_paged && !is_mtp_unpaged;
        const uint32_t   paged_u         = layer_paged ? 1U : 0U;
        id<MTLBuffer>    k_buf           = is_mtp_unpaged ? eng->bound_slot->b_mtp_k_cache : eng->b_k_cache;
        id<MTLBuffer>    v_buf           = is_mtp_unpaged ? eng->bound_slot->b_mtp_v_cache : eng->b_v_cache;
        const NSUInteger layer_kv_off    = is_mtp_unpaged ? 0U : kv_off;
        const uint32_t   layer_kv_stride = is_mtp_unpaged ? eng->max_seq_alloc : eng->kv_stride_tokens;
        id<MTLBuffer>    loc_buf         = layer_paged ? eng->b_block_loc : eng->b_block_loc_dummy;
        const NSUInteger loc_off         = layer_paged
            ? static_cast<NSUInteger>(static_cast<uint64_t>(full_slot) * eng->max_blocks_alloc * sizeof(uint32_t))
            : 0U;
        size_t n_mgroups = 0U;
        if (layer_paged) {
            if (!kv_only && any_sparse &&
                sqf_paged_layer_needs_sync(eng, full_slot, seq_pos_start, M_valid, comp_ratio, true, tree_mode)) {
                if (full_slot < eng->cfg.num_full_attn_layers) {
                    sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                           eng->b_x_in_f16, static_cast<NSUInteger>(M_valid - 1U) * D * sizeof(uint16_t),
                                           eng->b_probe_x, static_cast<NSUInteger>(full_slot) * D * sizeof(uint16_t),
                                           static_cast<size_t>(D) * sizeof(uint16_t));
                }
                sqf_paged_sync_point(eng, enc, cb, full_slot, seq_pos_start, M_valid, block_topk, comp_ratio, &n_mgroups);
            } else {
                sqf_paged_alloc_range(eng, full_slot, seq_pos_start, M_valid);
            }
        }

        // Main Q/K RMSNorm + Partial NeoX RoPE + KV Cache write
        {
            SqfQsaPrepParamsHost pp{
                M_valid, M_padded, n_q, n_kv, head_dim, rot_dim,
                seq_pos_start, layer_kv_stride, eps, kv_zero, paged_u, tree_mode
            };
            [enc setComputePipelineState:eng->pso_qsa_qkv_prep];
            [enc setBuffer:eng->b_qkv offset:0 atIndex:0];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:1];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:2];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_q_norm.offset) atIndex:3];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.attn_k_norm.offset) atIndex:4];
            [enc setBuffer:eng->b_inv_freq offset:0 atIndex:5];
            [enc setBuffer:eng->b_q_rope offset:0 atIndex:6];
            [enc setBuffer:k_buf offset:layer_kv_off atIndex:7];
            [enc setBuffer:v_buf offset:layer_kv_off atIndex:8];
            [enc setBytes:&pp length:sizeof(pp) atIndex:9];
            [enc setBuffer:loc_buf offset:loc_off atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake(n_q + 2U * n_kv, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
        }
        if (kv_only) {
            return;
        }

        bool use_dense_nax = !force_gemv && (tree_mode == 0U) && eng->has_nax && !any_sparse && (head_dim == 256U);
        if (use_dense_nax && layer_paged && !sqf_paged_identity_ok(eng, full_slot, total_kv)) {
            use_dense_nax = false; // scrambled pool after a deep rewind: exact split-KV gather instead of 32-row tiles
        }
        if (use_dense_nax) {
            SqfQsaPrefillNaxParamsHost fp{
                M_valid, M_padded, n_q, n_kv, seq_pos_start, total_kv, layer_kv_stride, paged_u
            };
            [enc setComputePipelineState:eng->pso_qsa_prefill_nax];
            [enc setBuffer:eng->b_q_rope offset:0 atIndex:0];
            [enc setBuffer:k_buf offset:layer_kv_off atIndex:1];
            [enc setBuffer:v_buf offset:layer_kv_off atIndex:2];
            [enc setBuffer:eng->b_qkv offset:0 atIndex:3];
            [enc setBuffer:eng->b_attn_out offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc setBuffer:loc_buf offset:loc_off atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((M_padded + 31U) / 32U, n_q, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            const uint32_t q_per_tg     = sqf_choose_q_per_tg(n_q, n_kv);
            const uint32_t num_q_groups = (n_q + q_per_tg - 1U) / q_per_tg;
            const uint32_t max_eff_kv   = any_sparse ? (block_topk * comp_ratio + comp_ratio) : total_kv;
            const uint32_t launch_splits= std::clamp<uint32_t>((max_eff_kv + 63U) / 64U, 1U, kMaxDecodeSplits);
            const float    attn_scale   = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const uint32_t newest_blk   = (total_kv - 1U) / comp_ratio;

            SqfQsaDecodeParamsHost dp{
                M_valid, M_padded, n_q, n_kv, head_dim, q_per_tg,
                seq_pos_start, layer_kv_stride, kMaxDecodeSplits,
                comp_ratio, block_topk, attn_scale, 0U, paged_u, tree_mode, 0U
            };
            const bool use_sparse_prefill_nax =
                !force_gemv && (tree_mode == 0U) && eng->has_nax && (M_valid > kMaxSpecBatch) &&
                (head_dim == 256U) && (n_kv > 0U) && (n_q == 12U * n_kv) && (comp_ratio == 4U);
            const SqfMGroup* mgroups = (n_mgroups > 0U) ? eng->bound_slot->scratch_mgroups.data() : nullptr;
            const size_t n_groups    = (n_mgroups == 0U) ? 1U : n_mgroups;
            if (use_sparse_prefill_nax) {
                for (size_t g = 0U; g < n_groups; ++g) {
                    const uint32_t g_beg = (n_mgroups == 0U) ? 0U : mgroups[g].beg;
                    const uint32_t g_end = (n_mgroups == 0U) ? M_valid : mgroups[g].end;
                    if (g > 0U) {
                        sqf_paged_commit_wait(eng, enc, cb);
                        const SqfMGroup* ng = (g + 1U < n_groups) ? &mgroups[g + 1U] : nullptr;
                        sqf_paged_make_group_resident(eng, full_slot, newest_blk, mgroups[g], ng);
                    }
                    dp.m_base = g_beg;
                    dp.m_end  = g_end;
                    const uint32_t g_rows = (g + 1U == n_groups) ? (M_padded - g_beg) : (g_end - g_beg);
                    [enc setComputePipelineState:eng->pso_qsa_sparse_prefill_nax];
                    [enc setBuffer:eng->b_q_rope offset:0 atIndex:0];
                    [enc setBuffer:k_buf offset:layer_kv_off atIndex:1];
                    [enc setBuffer:v_buf offset:layer_kv_off atIndex:2];
                    [enc setBuffer:eng->b_idx_sel_blocks offset:0 atIndex:3];
                    [enc setBuffer:eng->b_qkv offset:0 atIndex:4];
                    [enc setBuffer:eng->b_attn_out offset:0 atIndex:5];
                    [enc setBytes:&dp length:sizeof(dp) atIndex:6];
                    [enc setBuffer:loc_buf offset:loc_off atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((g_rows + 15U) / 16U, n_q, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                }
            } else {
                for (size_t g = 0U; g < n_groups; ++g) {
                    const uint32_t g_beg = (n_mgroups == 0U) ? 0U : mgroups[g].beg;
                    const uint32_t g_end = (n_mgroups == 0U) ? M_valid : mgroups[g].end;
                    if (g > 0U) {
                        // Pool cannot hold every query's top-512 at once: drain, recall the next group's blocks, continue
                        sqf_paged_commit_wait(eng, enc, cb);
                        const SqfMGroup* ng = (g + 1U < n_groups) ? &mgroups[g + 1U] : nullptr;
                        sqf_paged_make_group_resident(eng, full_slot, newest_blk, mgroups[g], ng);
                    }
                    dp.m_base = g_beg;
                    const bool use_fast_gqa12 =
                        (head_dim == 256U) && (n_kv > 0U) && (n_q == 12U * n_kv) && (comp_ratio == 4U);
                    if (use_fast_gqa12) {
                        [enc setComputePipelineState:eng->pso_qsa_decode_split_gqa12];
                        [enc setBuffer:eng->b_q_rope offset:0 atIndex:0];
                        [enc setBuffer:k_buf offset:layer_kv_off atIndex:1];
                        [enc setBuffer:v_buf offset:layer_kv_off atIndex:2];
                        [enc setBuffer:eng->b_idx_sel_blocks offset:0 atIndex:3];
                        [enc setBuffer:eng->b_part_m offset:0 atIndex:4];
                        [enc setBuffer:eng->b_part_l offset:0 atIndex:5];
                        [enc setBuffer:eng->b_part_o offset:0 atIndex:6];
                        [enc setBytes:&dp length:sizeof(dp) atIndex:7];
                        [enc setBuffer:loc_buf offset:loc_off atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(launch_splits, n_kv, g_end - g_beg)
                            threadsPerThreadgroup:MTLSizeMake(192, 1, 1)];
                    } else {
                        [enc setComputePipelineState:eng->pso_qsa_decode_split];
                        [enc setBuffer:eng->b_q_rope offset:0 atIndex:0];
                        [enc setBuffer:k_buf offset:layer_kv_off atIndex:1];
                        [enc setBuffer:v_buf offset:layer_kv_off atIndex:2];
                        [enc setBuffer:eng->b_idx_sel_blocks offset:0 atIndex:3];
                        [enc setBuffer:eng->b_part_m offset:0 atIndex:4];
                        [enc setBuffer:eng->b_part_l offset:0 atIndex:5];
                        [enc setBuffer:eng->b_part_o offset:0 atIndex:6];
                        [enc setBytes:&dp length:sizeof(dp) atIndex:7];
                        [enc setBuffer:loc_buf offset:loc_off atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(launch_splits, num_q_groups, g_end - g_beg)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                }
                dp.m_base = 0U;

                [enc setComputePipelineState:eng->pso_qsa_decode_reduce];
                [enc setBuffer:eng->b_part_m offset:0 atIndex:0];
                [enc setBuffer:eng->b_part_l offset:0 atIndex:1];
                [enc setBuffer:eng->b_part_o offset:0 atIndex:2];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:3];
                [enc setBuffer:eng->b_attn_out offset:0 atIndex:4];
                [enc setBytes:&dp length:sizeof(dp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(n_q, M_padded, 1)
                    threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
            }
        }

        sqf_encode_q4_linear(enc, eng, w_buf, ld.attn_o_proj,
                             eng->b_attn_out, 0, nil, 0, eng->b_block_y_f32, 0,
                             M_valid, M_padded, kOutF32, force_gemv);
        sqf_prof_mark(eng, enc, cb, 5U); // [5] QSA_Attn + O
    }

    // 3 + 4. Fused Post-Attention Residual Writeback (`hc_buf += inj * y`) + Pre-MLP Grouped RMSNorm (-48 dispatches!)
    {
        const uint32_t R_mlp = eng->cfg.hc_lowrank;
        const bool skip_norm_f32 =
            !force_gemv && eng->has_nax && M_valid > kMaxSpecBatch &&
            (ld.mlp_hc.input_mix_weight_up.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0 ||
             ld.mlp_hc.input_mix_weight_up.dtype == STRATA_QWEN_FLASH_DTYPE_F16) &&
            ((D % 64U) == 0U) && ((R_mlp % 64U) == 0U) &&
            ld.mlp_hc.input_mix_weight_up.shape0 == H * D &&
            ld.mlp_hc.input_mix_weight_up.shape1 == R_mlp;
        SqfGroupedNormParamsHost np{M_valid, M_padded, H, D, eng->cfg.rms_norm_eps, skip_norm_f32 ? 0U : 1U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_hc_res_add_rmsnorm];
        [enc setBuffer:eng->b_block_y_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:1];
        [enc setBuffer:hc_buf offset:0 atIndex:2];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.mlp_hc.hc_norm.offset) atIndex:3];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:4];
        [enc setBuffer:eng->b_hc_norm_f32 offset:0 atIndex:5];
        [enc setBytes:&np length:sizeof(np) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(H, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 4b. Pre-MLP 4-Stream Hyper-Connection (`write_x_f32 = true` for `shared_expert_gate`, `skip_rmsnorm = true`!)
    sqf_encode_hc_pre(enc, eng, w_buf, ld.mlp_hc, hc_buf, M_valid, M_padded, true, force_gemv, true);
    sqf_prof_mark(eng, enc, cb, 6U); // [6] HC_res + HC_pre_mlp

    // 5. 512-Expert Top-10 SwiGLU MoE + Gated Shared Expert
    const uint32_t E          = eng->cfg.num_experts;
    const uint32_t top_k      = eng->cfg.num_experts_per_tok;
    const uint32_t moe_dff    = eng->cfg.moe_intermediate_size;
    const uint32_t shared_dff = eng->cfg.shared_intermediate_size;

    // 5a. Router logits (`[M_valid, E]` FP16 -> FP32) + Fused Zero-Barrier Top-K Softmax & `shared_gate_dot`!
    sqf_encode_f16_linear(enc, eng, w_buf, ld.moe_router_gate,
                          eng->b_x_in_f16, 0, nil, 0, eng->b_router_logits, 0,
                          M_valid, M_padded, kOutF32, force_gemv);
    const bool fused_topk_gate = (E <= 512U) && (top_k <= 16U) && (eng->cfg.norm_topk_prob != 0U);
    if (fused_topk_gate) {
        SqfMoeTopkSharedGateParamsHost tgp{M_valid, E, top_k, D};
        [enc setComputePipelineState:eng->pso_moe_topk_shared_gate];
        [enc setBuffer:eng->b_router_logits offset:0 atIndex:0];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:1];
        [enc setBuffer:eng->b_topk_w offset:0 atIndex:2];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:3];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.shared_expert_gate.offset) atIndex:4];
        [enc setBuffer:eng->b_shared_scalar offset:0 atIndex:5];
        [enc setBytes:&tgp length:sizeof(tgp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
        SqfMoeRouterParamsHost rp{M_valid, E, top_k, eng->cfg.norm_topk_prob};
        [enc setComputePipelineState:eng->pso_moe_topk];
        [enc setBuffer:eng->b_router_logits offset:0 atIndex:0];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:1];
        [enc setBuffer:eng->b_topk_w offset:0 atIndex:2];
        [enc setBytes:&rp length:sizeof(rp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 5b. Shared Expert (+ fallback `shared_gate_dot` only if `E > 512` or `norm_topk_prob == 0`)
    if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch &&
        ld.shared_gate_proj.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A &&
        ld.shared_up_proj.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A &&
        ((ld.shared_gate_proj.shape1 % 64U) == 0U) &&
        ld.shared_gate_proj.shape0 == shared_dff &&
        ld.shared_up_proj.shape0 == shared_dff &&
        ld.shared_up_proj.shape1 == ld.shared_gate_proj.shape1) {
        const StrataQwenFlashQuantMatrix qg = sqf_q4_slice_as_qmat(&ld.shared_gate_proj);
        const StrataQwenFlashQuantMatrix qu = sqf_q4_slice_as_qmat(&ld.shared_up_proj);
        SqfGemvBatchParamsHost shp{M_valid, shared_dff, ld.shared_gate_proj.shape1, kOutF16};
        [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_q4_shared_gu_silu_m[M_valid - 1U], "sqf_q4_shared_gate_up_silu_batch", M_valid)];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.biases_off) atIndex:2];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.weight_off) atIndex:3];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.scales_off) atIndex:4];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.biases_off) atIndex:5];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:6];
        [enc setBuffer:eng->b_shared_mid_f16 offset:0 atIndex:7];
        [enc setBytes:&shp length:sizeof(shp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake((shared_dff + 3U) / 4U, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (force_gemv && M_valid >= 1U && M_valid <= kMaxSpecBatch &&
        ld.shared_gate_proj.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
        ld.shared_up_proj.dtype == STRATA_QWEN_FLASH_DTYPE_F16 &&
        ((ld.shared_gate_proj.shape1 % 8U) == 0U) &&
        ld.shared_gate_proj.shape0 == shared_dff &&
        ld.shared_up_proj.shape0 == shared_dff &&
        ld.shared_up_proj.shape1 == ld.shared_gate_proj.shape1) {
        SqfGemvBatchParamsHost shp{M_valid, shared_dff, ld.shared_gate_proj.shape1, kOutF16};
        [enc setComputePipelineState:eng->pso_f16_shared_gu_silu];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.shared_gate_proj.offset) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.shared_up_proj.offset) atIndex:1];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_shared_mid_f16 offset:0 atIndex:3];
        [enc setBytes:&shp length:sizeof(shp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(shared_dff, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (!force_gemv && eng->has_nax && M_valid > kMaxSpecBatch &&
        ld.shared_gate_proj.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A &&
        ld.shared_up_proj.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A &&
        ((ld.shared_gate_proj.shape1 % 128U) == 0U) &&
        ((shared_dff % 64U) == 0U) &&
        ld.shared_gate_proj.shape0 == shared_dff &&
        ld.shared_up_proj.shape0 == shared_dff &&
        ld.shared_up_proj.shape1 == ld.shared_gate_proj.shape1) {
        const StrataQwenFlashQuantMatrix qg = sqf_q4_slice_as_qmat(&ld.shared_gate_proj);
        const StrataQwenFlashQuantMatrix qu = sqf_q4_slice_as_qmat(&ld.shared_up_proj);
        const uint32_t sh_cols = ld.shared_gate_proj.shape1;
        const bool     use_m64 = (M_padded >= 64U) && ((shared_dff / 64U) * (M_padded / 64U) >= 80U);
        const uint32_t m_tile  = use_m64 ? 64U : 32U;
        const uint32_t n_th    = use_m64 ? 128U : 64U;
        SqfSharedGuNaxParamsHost shp{M_valid, M_padded, shared_dff, sh_cols};
        const uint64_t off_x_sc  = sqf_align_256(static_cast<uint64_t>(M_padded) * sh_cols);
        const uint64_t off_wg_i8 = sqf_align_256(off_x_sc + static_cast<uint64_t>(M_padded) * sizeof(float));
        const uint64_t off_wg_sc = sqf_align_256(off_wg_i8 + static_cast<uint64_t>(shared_dff) * sh_cols);
        const uint64_t off_wu_i8 = sqf_align_256(off_wg_sc + static_cast<uint64_t>(shared_dff) * sizeof(float));
        const uint64_t off_wu_sc = sqf_align_256(off_wu_i8 + static_cast<uint64_t>(shared_dff) * sh_cols);
        const uint64_t need_sh   = off_wu_sc + static_cast<uint64_t>(shared_dff) * sizeof(float);
        if (eng->int8_prefill_enabled && M_valid >= 32U && M_padded >= 32U &&
            [eng->b_moe_down_part length] >= need_sh) {
            sqf_encode_quant_x_i8(enc, eng, eng->b_x_in_f16, 0U, eng->b_moe_down_part, 0U,
                                  static_cast<NSUInteger>(off_x_sc), M_valid, M_padded, sh_cols);
            SqfI8QuantParamsHost wp{M_valid, M_padded, shared_dff, shared_dff, sh_cols, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_q4_requant_row_i8];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.weight_off) atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.scales_off) atIndex:1];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.biases_off) atIndex:2];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_wg_i8) atIndex:3];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_wg_sc) atIndex:4];
            [enc setBytes:&wp length:sizeof(wp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(shared_dff, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.weight_off) atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.scales_off) atIndex:1];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.biases_off) atIndex:2];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_wu_i8) atIndex:3];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_wu_sc) atIndex:4];
            [enc setBytes:&wp length:sizeof(wp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(shared_dff, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:(use_m64 ? eng->pso_i8_shared_gu_silu_nax_m64 : eng->pso_i8_shared_gu_silu_nax_m32)];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_wg_i8) atIndex:0];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_wg_sc) atIndex:1];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_wu_i8) atIndex:2];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_wu_sc) atIndex:3];
            [enc setBuffer:eng->b_moe_down_part offset:0U atIndex:4];
            [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(off_x_sc) atIndex:5];
            [enc setBuffer:eng->b_shared_mid_f16 offset:0 atIndex:6];
            [enc setBytes:&shp length:sizeof(shp) atIndex:7];
        } else {
            [enc setComputePipelineState:(use_m64 ? eng->pso_q4_shared_gu_silu_nax_m64 : eng->pso_q4_shared_gu_silu_nax_m32)];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.weight_off) atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.scales_off) atIndex:1];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qg.biases_off) atIndex:2];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.weight_off) atIndex:3];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.scales_off) atIndex:4];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(qu.biases_off) atIndex:5];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:6];
            [enc setBuffer:eng->b_shared_mid_f16 offset:0 atIndex:7];
            [enc setBytes:&shp length:sizeof(shp) atIndex:8];
        }
        [enc dispatchThreadgroups:MTLSizeMake(shared_dff / 64U, (M_padded + m_tile - 1U) / m_tile, 1)
            threadsPerThreadgroup:MTLSizeMake(n_th, 1, 1)];
    } else {
        sqf_encode_f16_linear(enc, eng, w_buf, ld.shared_gate_proj,
                              eng->b_x_in_f16, 0, nil, 0, eng->b_shared_gate_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        sqf_encode_f16_linear(enc, eng, w_buf, ld.shared_up_proj,
                              eng->b_x_in_f16, 0, nil, 0, eng->b_shared_up_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        SqfSwigluParamsHost sp{M_valid, M_padded, shared_dff, 0U};
        [enc setComputePipelineState:eng->pso_swiglu];
        [enc setBuffer:eng->b_shared_gate_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_shared_up_f32 offset:0 atIndex:1];
        [enc setBuffer:eng->b_shared_mid_f16 offset:0 atIndex:2];
        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
        const uint32_t d4 = shared_dff / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    sqf_encode_f16_linear(enc, eng, w_buf, ld.shared_down_proj,
                          eng->b_shared_mid_f16, 0, nil, 0, eng->b_shared_out_f32, 0,
                          M_valid, M_padded, kOutF32, force_gemv);
    if (!fused_topk_gate) {
        SqfSharedGateParamsHost gp{M_valid, D};
        [enc setComputePipelineState:eng->pso_shared_gate_dot];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.shared_expert_gate.offset) atIndex:1];
        [enc setBuffer:eng->b_shared_scalar offset:0 atIndex:2];
        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    sqf_prof_mark(eng, enc, cb, 7U); // [7] Router + Shared

    // 5c. Active Routed Experts (`M == 1` Gather-GEMV vs `M >= 2` Expert-Grouped Register-Batched Deduplication!)
    bool moe_down_is_f16 = false;
    SqfMoeGatherParamsHost mp{M_valid, top_k, D, moe_dff, E, 0U, 0U, 0U};
    if (M_valid == 1U || E > 512U) {
        [enc setComputePipelineState:eng->pso_moe_gather_gu];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.biases_off) atIndex:2];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:4];
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((moe_dff + 3U) / 4U, top_k, M_valid)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:eng->pso_moe_gather_dn];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.biases_off) atIndex:2];
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:4];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:5];
        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, top_k, M_valid)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (!force_gemv && eng->has_nax && M_valid > kMaxSpecBatch &&
               ((D % 128U) == 0U) && ((moe_dff % 128U) == 0U) && E <= 512U) {
        // Metal 4.1 Quad-Bin (`BM = 16 / 32 / 64 / 128`, `BK = 128`, `uint4` 16B loads) Direct-tX Concurrent NAX MoE!
        // `x_sorted` aliases `eng->b_moe_down_part` (`50.6 MiB <= 100 MiB`, disjoint lifecycle -> 0 extra VRAM!).
        moe_down_is_f16 = true;
        const uint32_t total_pairs = M_valid * top_k;

        [enc setComputePipelineState:eng->pso_moe_bin_tokens_tri];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
        [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:3];
        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

        [enc setComputePipelineState:eng->pso_moe_gather_x_f16];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:0];
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:2]; // aliased as `x_sorted`!
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];   // zeroes `+128` tail!
        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(total_pairs + 128U, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // Switch to Concurrent Compute Encoder so ALL 4 GU bins (`16/32/64/128`) execute simultaneously (0 tail drain!)
        [enc endEncoding];
        enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.biases_off) atIndex:2];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:3]; // `x_sorted`
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];   // `mid_sorted`
        [enc setBytes:&mp length:sizeof(mp) atIndex:6];

        [enc setComputePipelineState:eng->pso_moe_prefill_gu_nax_m128];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                               indirectBufferOffset:48
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:eng->pso_moe_prefill_gu_nax_m64];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                               indirectBufferOffset:32
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:eng->pso_moe_prefill_gu_nax_m32];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                               indirectBufferOffset:16
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:eng->pso_moe_prefill_gu_nax_m16];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                               indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        if (eng->prof_active) {
            sqf_prof_mark(eng, enc, cb, 8U); // [8] MoE_Bin + Gather + MoE_GU (Quad-Bin Concurrent NAX)
            [enc endEncoding];
            enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        } else {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // DN Quad-Bin Concurrent Indirect Dispatches (`BM=128/64/32/16` longest-first!)
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.biases_off) atIndex:2];
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];   // `mid_sorted`
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:6]; // `down_part` (`x_sorted` is dead!)
        [enc setBytes:&mp length:sizeof(mp) atIndex:7];

        [enc setComputePipelineState:eng->pso_moe_prefill_dn_nax_m128];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                               indirectBufferOffset:112
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:eng->pso_moe_prefill_dn_nax_m64];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                               indirectBufferOffset:96
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:eng->pso_moe_prefill_dn_nax_m32];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                               indirectBufferOffset:80
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:eng->pso_moe_prefill_dn_nax_m16];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                               indirectBufferOffset:64
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc endEncoding];
        enc = [cb computeCommandEncoder];
        sqf_prof_mark(eng, enc, cb, 9U); // [9] MoE_DN (Quad-Bin Concurrent NAX)
    } else if (!force_gemv && eng->has_nax && M_valid > kMaxSpecBatch &&
               ((D % 64U) == 0U) && ((moe_dff % 64U) == 0U)) {
        // Prefill-Only Metal 4 NAX TensorOps Expert-Grouped MoE (`tile_cap = 16`, 0 impact on decode PSO registers!)
        const uint32_t total_pairs = M_valid * top_k;
        const uint32_t max_tiles   = std::min<uint32_t>(total_pairs, (total_pairs / 16U) + E);

        [enc setComputePipelineState:eng->pso_moe_prefill_bin_tokens];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:3];
        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

        [enc setComputePipelineState:eng->pso_moe_prefill_gu_nax];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.biases_off) atIndex:2];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:7];
        [enc setBytes:&mp length:sizeof(mp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(moe_dff / 64U, max_tiles, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        sqf_prof_mark(eng, enc, cb, 8U); // [8] MoE_Bin + MoE_GU (NAX)

        [enc setComputePipelineState:eng->pso_moe_prefill_dn_nax];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.biases_off) atIndex:2];
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:7];
        [enc setBytes:&mp length:sizeof(mp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(D / 64U, max_tiles, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        sqf_prof_mark(eng, enc, cb, 9U); // [9] MoE_DN (NAX)
    } else {
        const uint32_t total_pairs = M_valid * top_k;
        const uint32_t max_tiles   = std::min<uint32_t>(total_pairs, (total_pairs / 8U) + E);
        const uint32_t b_slot      = (M_valid <= 4U) ? 0U : ((M_valid == 5U) ? 1U : ((M_valid == 6U) ? 2U : 3U));
        constexpr uint32_t kMoeMaxB[4] = {4U, 5U, 6U, 8U};
        const uint32_t b_cap       = kMoeMaxB[b_slot];

        [enc setComputePipelineState:eng->pso_moe_bin_tokens];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:3];
        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

        [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_moe_grouped_gu_b[b_slot], "sqf_moe_grouped_gate_up_silu", b_cap)];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_gate_up.biases_off) atIndex:2];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:7];
        [enc setBytes:&mp length:sizeof(mp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake((moe_dff + 3U) / 4U, max_tiles, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        sqf_prof_mark(eng, enc, cb, 8U); // [8] MoE_Bin + MoE_GU

        [enc setComputePipelineState:sqf_get_pso_m(eng, eng->pso_moe_grouped_dn_b[b_slot], "sqf_moe_grouped_down", b_cap)];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(ld.moe_experts_down.biases_off) atIndex:2];
        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:7];
        [enc setBytes:&mp length:sizeof(mp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, max_tiles, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        sqf_prof_mark(eng, enc, cb, 9U); // [9] MoE_DN
    }

    // 5d. Combine Weighted Routed Experts + Shared Expert + 4-Stream HC Residual Writeback
    {
        SqfMoeCombineParamsHost cp{M_valid, top_k, D, H};
        [enc setComputePipelineState:(moe_down_is_f16 ? eng->pso_moe_combine_f16 : eng->pso_moe_combine)];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:0];
        [enc setBuffer:eng->b_topk_w offset:0 atIndex:1];
        [enc setBuffer:eng->b_shared_out_f32 offset:0 atIndex:2];
        [enc setBuffer:eng->b_shared_scalar offset:0 atIndex:3];
        [enc setBuffer:eng->b_hc_inj_f32 offset:0 atIndex:4];
        [enc setBuffer:hc_buf offset:0 atIndex:5];
        [enc setBytes:&cp length:sizeof(cp) atIndex:6];
        const uint32_t d4 = D / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_valid, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    sqf_prof_mark(eng, enc, cb, 10U); // [10] MoE_Combine
}

// Returns true if 32K SLC-Pinned FR-Spec Draft `lm_head` (`45.0 MiB < 48 MiB` SLC!) should be used for this draft pass.
inline bool sqf_frspec_active(const StrataQwenFlashGpuEngine* eng, uint32_t* v_out_out = nullptr) {
    if (!eng || !eng->frspec_enabled || eng->frspec_widened || eng->frspec_shortlist_size == 0U ||
        !eng->b_frspec_lm_head || !eng->b_frspec_ids) {
        return false;
    }
    if (v_out_out) *v_out_out = eng->frspec_shortlist_size;
    return true;
}

// Self-healing dynamic GPU-patched context slots (`31744..32767`) for the 32K SLC-pinned FR-Spec `lm_head`:
//   - Records misses BEFORE patching (when `record_misses == true`) and auto-widens on sustained >= 5/32 misses.
//   - Patches EVERY out-of-shortlist token into the 1,024-slot dynamic ring via `< 1 us/row` `sqf_patch_frspec_q4`
//     (using inline `setBytes` chunks of <= 512 pairs: 0 extra buffers, 0 async races in `cb_fix`!).
inline void sqf_frspec_stage_and_encode_patches(
    id<MTLComputeCommandEncoder> enc,
    StrataQwenFlashGpuEngine*    eng,
    const uint32_t*              tokens,
    uint32_t                     n_tokens,
    bool                         record_misses
) {
    if (!eng || !tokens || n_tokens == 0U || eng->frspec_shortlist_size != kFrSpecShortlistSize ||
        !eng->b_frspec_lm_head || !eng->b_frspec_ids || !eng->b_frspec_full_to_short || !eng->pso_patch_frspec_q4) {
        return;
    }
    const uint32_t V   = eng->cfg.vocab_size;
    uint32_t*      ids = reinterpret_cast<uint32_t*>([eng->b_frspec_ids contents]);
    uint32_t*      f2s = reinterpret_cast<uint32_t*>([eng->b_frspec_full_to_short contents]);

    uint32_t patch_dst[512];
    uint32_t patch_src[512];
    uint32_t n_patch = 0U;

    auto flush_patches = [&]() {
        if (n_patch == 0U || !enc) {
            n_patch = 0U;
            return;
        }
        SqfPackFrspecParamsHost pp{n_patch, eng->frspec_lm_head_desc.cols, V, kFrSpecShortlistSize};
        [enc setComputePipelineState:eng->pso_patch_frspec_q4];
        [enc setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(eng->hdr1.lm_head.weight_off) atIndex:0];
        [enc setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(eng->hdr1.lm_head.scales_off) atIndex:1];
        [enc setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(eng->hdr1.lm_head.biases_off) atIndex:2];
        [enc setBytes:patch_dst length:static_cast<NSUInteger>(n_patch) * sizeof(uint32_t) atIndex:3];
        [enc setBytes:patch_src length:static_cast<NSUInteger>(n_patch) * sizeof(uint32_t) atIndex:4];
        [enc setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(eng->frspec_lm_head_desc.weight_off) atIndex:5];
        [enc setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(eng->frspec_lm_head_desc.scales_off) atIndex:6];
        [enc setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(eng->frspec_lm_head_desc.biases_off) atIndex:7];
        [enc setBytes:&pp length:sizeof(pp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(n_patch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        n_patch = 0U;
    };

    for (uint32_t i = 0U; i < n_tokens; ++i) {
        const uint32_t t = tokens[i];
        const bool out_of_shortlist = (t >= V) || (f2s[t] == 0xFFFFFFFFU);
        if (record_misses && eng->frspec_enabled) {
            eng->shortlist_miss_mask = (eng->shortlist_miss_mask << 1U) | (out_of_shortlist ? 1U : 0U);
            if (eng->shortlist_obs_count < 32U) ++eng->shortlist_obs_count;
        }
        if (t < V && f2s[t] == 0xFFFFFFFFU) {
            const uint32_t slot  = kFrSpecStaticRows + (eng->frspec_dyn_cursor % kFrSpecDynSlots);
            eng->frspec_dyn_cursor = (eng->frspec_dyn_cursor + 1U) % kFrSpecDynSlots;
            const uint32_t old_t = ids[slot];
            if (old_t < V && f2s[old_t] == slot) {
                f2s[old_t] = 0xFFFFFFFFU;
            }
            ids[slot] = t;
            f2s[t]    = slot;
            patch_dst[n_patch] = slot;
            patch_src[n_patch] = t;
            ++n_patch;
            if (n_patch == 512U) flush_patches();
        }
    }
    flush_patches();

    if (record_misses && eng->frspec_enabled) {
        const int misses = __builtin_popcount(eng->shortlist_miss_mask);
        // Auto-widen when >= 5/32 committed tokens still miss AFTER self-healing (e.g. non-repeating CJK/rare vocab);
        // automatically re-narrow once misses drop back <= 2/32!
        if (!eng->frspec_widened && eng->shortlist_obs_count >= 16U && misses >= 5) {
            eng->frspec_widened = true;
        } else if (eng->frspec_widened && eng->shortlist_obs_count >= 32U && misses <= 2) {
            eng->frspec_widened = false;
        }
    }
}

// 3-State SpecCtl Adaptive-K Helpers (matching Inkling-2 #193, tuned for Qwen3.8-Flash-Next 180B M5 Max!)
inline uint32_t sqf_spec_ctl_choose_k(
    const StrataQwenFlashGpuEngine* eng,
    uint32_t                        max_k,
    bool*                           high_conf_out
) {
    if (high_conf_out) *high_conf_out = false;
    if (!eng || !eng->adaptive_k_enabled || max_k == 0U) return max_k;

    const SqfSpecCtl& ctl = eng->spec_ctl;
    if (ctl.state == kSpecCooldownK0 && ctl.cooldown_rem > 0U) {
        return 0U;
    }
    if (ctl.state == kSpecHighConfFused) {
        if (high_conf_out) *high_conf_out = true;
        return max_k;
    }
    if (ctl.probe_after_cd) {
        return std::min<uint32_t>(max_k, 2U);
    }
    if (ctl.obs_steps < 3U) {
        return max_k;
    }

    // Optimize E[emitted(k)] / (k * ema_draft_ms + est_ver_ms(k)) over k in 0..max_k
    // with coherent concave verify curve AND positive-autocorrelation uplift on structured code/prose!
    const double p      = std::clamp(ctl.ema_geom_p, 0.02, 0.98);
    const double k0_tps = 1.0 / std::max(1.0, ctl.est_ver_ms(0U));
    uint32_t     best_spec_k   = std::min<uint32_t>(max_k, 1U);
    double       best_spec_tps = -1.0;

    for (uint32_t k = 1U; k <= max_k; ++k) {
        // Real LLM code/structured text has positive run-length autocorrelation (bursts of 4/4 accepts):
        // gently uplift tail acceptance for k >= 3 when p >= 0.56 so SpecCtl doesn't prematurely trim K=4 -> K=2/3!
        const double pk     = (k >= 3U && p >= 0.56) ? std::min(0.98, p + 0.08) : p;
        const double e_emit = (pk < 0.999)
            ? ((1.0 - std::pow(pk, static_cast<double>(k + 1U))) / (1.0 - pk))
            : static_cast<double>(k + 1U);
        const double t_ms   = static_cast<double>(k) * ctl.ema_draft_ms + ctl.est_ver_ms(k);
        const double tps    = e_emit / std::max(1.0, t_ms);
        if (tps > best_spec_tps) {
            best_spec_tps = tps;
            best_spec_k   = k;
        }
    }
    if (best_spec_tps < k0_tps * 0.98) {
        return 0U;
    }
    return best_spec_k;
}

inline void sqf_spec_ctl_update_spec(
    StrataQwenFlashGpuEngine* eng,
    uint32_t                  eff_k,
    uint32_t                  max_k,
    uint32_t                  accepted,
    double                    draft_ms,
    double                    ver_ms,
    float                     mean_pmax,
    bool                      used_ngram
) {
    if (!eng || !eng->adaptive_k_enabled || eff_k == 0U || eff_k >= kMaxSpecBatch) return;
    SqfSpecCtl& ctl = eng->spec_ctl;
    ctl.probe_after_cd = false;

    ctl.record_ver_ms(eff_k, ver_ms);
    if (!used_ngram && draft_ms > 0.05 && draft_ms < 100.0) {
        const double per_tok = draft_ms / static_cast<double>(eff_k);
        ctl.ema_draft_ms = 0.80 * ctl.ema_draft_ms + 0.20 * per_tok;
    }
    if (!used_ngram && mean_pmax > 0.0f && mean_pmax <= 1.0f) {
        ctl.ema_pmax = 0.80 * ctl.ema_pmax + 0.20 * static_cast<double>(mean_pmax);
    }

    // Wald-unbiased geometric acceptance estimator (exact zero right-censoring bias!)
    if (!used_ngram) {
        const double trials = (accepted < eff_k) ? static_cast<double>(accepted + 1U) : static_cast<double>(eff_k);
        ctl.ema_succ   = 0.85 * ctl.ema_succ + static_cast<double>(accepted);
        ctl.ema_trials = 0.85 * ctl.ema_trials + trials;
        ctl.ema_geom_p = std::clamp(ctl.ema_succ / std::max(1e-6, ctl.ema_trials), 0.02, 0.98);
        ++ctl.obs_steps;
    }

    if (accepted == 0U) {
        ++ctl.consec_zero_acc;
        ctl.consec_high_acc = 0U;
    } else {
        ctl.consec_zero_acc = 0U;
        if (accepted >= std::max<uint32_t>(2U, (eff_k + 1U) / 2U)) {
            ++ctl.consec_high_acc;
        } else {
            ctl.consec_high_acc = 0U;
        }
    }

    // State transitions
    if (ctl.state == kSpecHighConfFused) {
        if (ctl.consec_zero_acc >= 2U || ctl.ema_geom_p < 0.56) {
            ctl.state = kSpecActiveProbe;
            ctl.consec_high_acc = 0U;
        }
    } else {
        // Check if K=0 beats best spec OR 3 consecutive 0-accept steps -> enter CooldownK0!
        const double p      = ctl.ema_geom_p;
        const double k0_tps = 1.0 / std::max(1.0, ctl.est_ver_ms(0U));
        double best_spec_tps = 0.0;
        for (uint32_t k = 1U; k <= max_k; ++k) {
            const double pk     = (k >= 3U && p >= 0.56) ? std::min(0.98, p + 0.08) : p;
            const double e_emit = (pk < 0.999)
                ? ((1.0 - std::pow(pk, static_cast<double>(k + 1U))) / (1.0 - pk))
                : static_cast<double>(k + 1U);
            const double t_ms   = static_cast<double>(k) * ctl.ema_draft_ms + ctl.est_ver_ms(k);
            best_spec_tps = std::max(best_spec_tps, e_emit / std::max(1.0, t_ms));
        }

        if ((ctl.obs_steps >= 3U && best_spec_tps < k0_tps * 0.98) || ctl.consec_zero_acc >= 3U) {
            ctl.state        = kSpecCooldownK0;
            ctl.cooldown_rem = std::min<uint32_t>(16U, 6U + 2U * ctl.consec_cooldowns);
            ++ctl.consec_cooldowns;
            ctl.consec_zero_acc = 0U;
            ctl.consec_high_acc = 0U;
        } else if (max_k >= 2U && ctl.ema_geom_p >= 0.65 && ctl.consec_high_acc >= 2U) {
            ctl.state            = kSpecHighConfFused;
            ctl.consec_cooldowns = 0U;
        } else if (accepted >= 1U) {
            ctl.consec_cooldowns = 0U;
        }
    }
}

// Encodes Native 1-Layer `MTP` (`mtp.*`) over `M_valid` tokens starting at `mtp_pos_start`:
//   - Reads `hc_prev_buf` (`[M_valid, hc_count * D]`) + `tok_ids_buf` (`[M_valid]` at `tok_byte_off`)
//   - Updates MTP KV & Indexer cache (`full_slot = cfg.num_full_attn_layers`) at `mtp_pos_start .. mtp_pos_start + M_valid - 1`
//   - Writes post-layer0 4-stream state into `eng->b_mtp_hc`
//   - If `run_head == true`: mixes via `mtp.mixer` -> `lm_head` (or shortlisted `lm_head`) -> `b_logits`!
void sqf_encode_mtp_pass(
    id<MTLComputeCommandEncoder> __strong& enc,
    id<MTLCommandBuffer> __strong& cb,
    StrataQwenFlashGpuEngine* eng,
    id<MTLBuffer> hc_prev_buf,
    NSUInteger hc_prev_off,
    id<MTLBuffer> tok_ids_buf,
    NSUInteger tok_byte_off,
    uint32_t mtp_pos_start,
    uint32_t M_valid,
    bool run_head,
    bool force_gemv,
    uint32_t tree_mode = 0U,
    bool use_shortlist = false
) {
    if (!eng->cfg.has_mtp || !eng->hdr1.mtp.enabled || M_valid == 0U) return;

    const StrataQwenFlashMtpDesc& mtp = eng->hdr1.mtp;
    const uint32_t D        = eng->cfg.hidden_size;
    const uint32_t H        = eng->cfg.hc_count;
    const uint32_t V        = eng->cfg.vocab_size;
    const float    eps      = eng->cfg.rms_norm_eps;
    const uint32_t M_padded = (!force_gemv && eng->has_nax) ? ((M_valid + 63U) & ~63U) : M_valid;

    // 1. `embed_tokens(tok)` (`b_part0`!) -> `b_x_in_f32 [M_valid, D]` -> `pre_fc_norm_embedding` -> `fc_embedding` -> `b_block_y_f32`
    {
        SqfEmbedParamsHost ep{M_valid, D, V, 1U};
        [enc setComputePipelineState:eng->pso_embed_q4];
        [enc setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.weight_off) atIndex:0];
        [enc setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.scales_off) atIndex:1];
        [enc setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.biases_off) atIndex:2];
        [enc setBuffer:tok_ids_buf offset:tok_byte_off atIndex:3];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:4];
        [enc setBytes:&ep length:sizeof(ep) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
    }

    {
        SqfGroupedNormParamsHost np{M_valid, M_padded, 1U, D, eps, 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_grouped_rmsnorm];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(mtp.pre_fc_norm_embedding.offset) atIndex:1];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:3];
        [enc setBytes:&np length:sizeof(np) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    sqf_encode_f16_linear(enc, eng, eng->b_part1, mtp.fc_embedding,
                          eng->b_x_in_f16, 0, nil, 0, eng->b_block_y_f32, 0,
                          M_valid, M_padded, kOutF32, force_gemv);

    // 2. UNGROUPED `10240`-dim `pre_fc_norm_hidden(hc_prev)` -> `b_hc_norm_f16 [M_valid * H, D]` -> `fc_hidden` -> `b_hc_up_f32`
    const uint32_t mh_valid  = M_valid * H;
    const uint32_t mh_padded = (!force_gemv && eng->has_nax) ? ((mh_valid + 63U) & ~63U) : mh_valid;
    {
        SqfMtpNormHiddenParamsHost hp{M_valid, mh_padded, D, H, eps, 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_mtp_norm_split];
        [enc setBuffer:hc_prev_buf offset:hc_prev_off atIndex:0];
        [enc setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(mtp.pre_fc_norm_hidden.offset) atIndex:1];
        [enc setBuffer:eng->b_hc_norm_f16 offset:0 atIndex:2];
        [enc setBytes:&hp length:sizeof(hp) atIndex:3];
        const uint32_t grid_m = std::max<uint32_t>(M_valid, (mh_padded + H - 1U) / H);
        [enc dispatchThreadgroups:MTLSizeMake(1, grid_m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    sqf_encode_f16_linear(enc, eng, eng->b_part1, mtp.fc_hidden,
                          eng->b_hc_norm_f16, 0, nil, 0, eng->b_hc_up_f32, 0,
                          mh_valid, mh_padded, kOutF32, force_gemv);

    // 3. `b_mtp_hc[m, s, d] = e_proj[m, d] + h_fc[m * H + s, d]`
    {
        SqfMtpAddParamsHost ap{M_valid, D, H, 0U};
        [enc setComputePipelineState:eng->pso_mtp_add_bcast];
        [enc setBuffer:eng->b_block_y_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_hc_up_f32 offset:0 atIndex:1];
        [enc setBuffer:eng->b_mtp_hc offset:0 atIndex:2];
        [enc setBytes:&ap length:sizeof(ap) atIndex:3];
        const uint32_t d4 = D / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_valid, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 4. `mtp.layer0` (`FULL_ATTN` QSA + Indexer + 512-Expert MoE, using KV slot `num_full_attn_layers`)
    // When `run_head == false` (async `cb_fix` KV-sync & prefill MTP warmup), `kv_only = true` skips ~85% dead compute!
    sqf_encode_decoder_layer(
        enc, cb, eng, mtp.layer0, eng->b_mtp_hc,
        0U, eng->cfg.num_full_attn_layers,
        mtp_pos_start, M_valid, M_padded, 0U, force_gemv, tree_mode,
        /*kv_only=*/!run_head
    );

    // 5. Optional `mtp.mixer` + `lm_head` (32K SLC-pinned FR-Spec shortlist during speculative drafting!)
    if (run_head) {
        sqf_encode_mixer(enc, eng, eng->b_part1, mtp.mixer, eng->b_mtp_hc, 0, M_valid, M_padded, true);
        if (use_shortlist && sqf_frspec_active(eng)) {
            sqf_encode_q4_linear(enc, eng, eng->b_frspec_lm_head, eng->frspec_lm_head_desc,
                                 eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                                 M_valid, M_padded, kOutF32, true);
        } else {
            sqf_encode_q4_linear(enc, eng, eng->b_part1, eng->hdr1.lm_head,
                                 eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                                 M_valid, M_padded, kOutF32, true);
        }
    }
}

inline bool sqf_penalties_active(const StrataQwenFlashSamplingParams* s) {
    if (!s) return false;
    return (s->presence_penalty != 0.0f) ||
           (s->frequency_penalty != 0.0f) ||
           (s->repetition_penalty > 0.0f && s->repetition_penalty != 1.0f);
}

inline void sqf_penalties_inc(
    StrataQwenFlashGpuEngine* eng,
    uint32_t*                 ids,
    uint16_t*                 cnt,
    uint32_t                  t
) {
    const uint32_t V = eng->cfg.vocab_size;
    if (t >= V) return;
    if (cnt[t] == 0U && eng->pen_num_ids < V) {
        const uint32_t idx = eng->pen_num_ids++;
        ids[idx]        = t;
        eng->pen_pos[t] = idx;
    }
    if (cnt[t] < 65535U) {
        ++cnt[t];
    }
}

inline void sqf_penalties_dec(
    StrataQwenFlashGpuEngine* eng,
    uint32_t*                 ids,
    uint16_t*                 cnt,
    uint32_t                  t
) {
    const uint32_t V = eng->cfg.vocab_size;
    if (t >= V || cnt[t] == 0U) return;
    if (--cnt[t] == 0U) {
        const uint32_t idx = eng->pen_pos[t];
        if (idx < eng->pen_num_ids && eng->pen_num_ids > 0U) {
            const uint32_t last = --eng->pen_num_ids;
            if (idx != last) {
                const uint32_t moved = ids[last];
                ids[idx]            = moved;
                eng->pen_pos[moved] = idx;
            }
        }
        eng->pen_pos[t] = 0xFFFFFFFFU;
    }
}

inline void sqf_penalties_clear_active(StrataQwenFlashGpuEngine* eng) {
    if (!eng || !eng->b_pen_ids || !eng->b_pen_counts) return;
    const uint32_t* ids = reinterpret_cast<const uint32_t*>([eng->b_pen_ids contents]);
    uint16_t*       cnt = reinterpret_cast<uint16_t*>([eng->b_pen_counts contents]);
    const uint32_t  V   = eng->cfg.vocab_size;
    const uint32_t  n   = eng->pen_num_ids;
    for (uint32_t i = 0U; i < n; ++i) {
        const uint32_t t = ids[i];
        if (t < V) {
            cnt[t]          = 0U;
            eng->pen_pos[t] = 0xFFFFFFFFU;
        }
    }
    eng->pen_num_ids = 0U;
}

void sqf_penalties_sync_window(StrataQwenFlashGpuEngine* eng, uint32_t eff_window) {
    if (!eng || !eng->b_pen_ids || !eng->b_pen_counts) return;
    const uint32_t L = static_cast<uint32_t>(eng->pen_hist.size());
    const uint32_t target_start = (eff_window > 0U && L > eff_window) ? (L - eff_window) : 0U;
    if (eng->pen_window_start == target_start && eng->pen_window_end == L) return;

    uint32_t* ids = reinterpret_cast<uint32_t*>([eng->b_pen_ids contents]);
    uint16_t* cnt = reinterpret_cast<uint16_t*>([eng->b_pen_counts contents]);

    if (target_start >= eng->pen_window_end || eng->pen_window_end > L) {
        sqf_penalties_clear_active(eng);
        eng->pen_window_start = target_start;
        eng->pen_window_end   = target_start;
    }
    if (target_start < eng->pen_window_start) {
        for (uint32_t i = target_start; i < eng->pen_window_start; ++i) {
            sqf_penalties_inc(eng, ids, cnt, eng->pen_hist[i]);
        }
        eng->pen_window_start = target_start;
    }
    if (eng->pen_window_end < L) {
        for (uint32_t i = eng->pen_window_end; i < L; ++i) {
            sqf_penalties_inc(eng, ids, cnt, eng->pen_hist[i]);
        }
        eng->pen_window_end = L;
    }
    if (eng->pen_window_start < target_start) {
        for (uint32_t i = eng->pen_window_start; i < target_start; ++i) {
            sqf_penalties_dec(eng, ids, cnt, eng->pen_hist[i]);
        }
        eng->pen_window_start = target_start;
    }
}

void sqf_encode_logit_penalties(
    id<MTLComputeCommandEncoder>         enc,
    StrataQwenFlashGpuEngine*            eng,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t                             M,
    id<MTLBuffer>                        extra_toks_buf,
    uint32_t                             extra_mode,
    uint32_t                             extra_count,
    uint32_t                             shortlist_v_out = 0U
) {
    if (!sqf_penalties_active(sampling) || !eng->pso_apply_logit_penalties ||
        !eng->b_pen_ids || !eng->b_pen_counts) {
        return;
    }
    const uint32_t eff_window = (sampling->penalty_window == 0U)
        ? 0U
        : std::max<uint32_t>(sampling->penalty_window, 8U);
    sqf_penalties_sync_window(eng, eff_window);

    const uint32_t total_items = eng->pen_num_ids + extra_count;
    if (total_items == 0U) return;

    const bool use_sl = (shortlist_v_out > 0U) && (eng->b_frspec_full_to_short != nil);
    SqfLogitPenaltyParamsHost pp{};
    pp.M                  = M;
    pp.vocab_size         = eng->cfg.vocab_size;
    pp.logits_stride      = use_sl ? shortlist_v_out : eng->cfg.vocab_size;
    pp.use_shortlist      = use_sl ? 1U : 0U;
    pp.pad_sl             = 0U;
    pp.num_hist_ids       = eng->pen_num_ids;
    pp.extra_mode         = extra_mode;
    pp.extra_count        = extra_count;
    pp.presence_penalty   = sampling->presence_penalty;
    pp.frequency_penalty  = sampling->frequency_penalty;
    pp.repetition_penalty = sampling->repetition_penalty;
    pp.pad0               = 0U;
    for (uint32_t j = 0U; j < 8U; ++j) pp.drop_toks[j] = 0xFFFFFFFFU;
    if (eff_window > 0U && extra_count > 0U) {
        const uint32_t L = static_cast<uint32_t>(eng->pen_hist.size());
        const uint32_t max_depth = (extra_mode == 2U) ? 3U : std::min<uint32_t>(extra_count, 8U);
        for (uint32_t j = 0U; j < max_depth; ++j) {
            if (L + j >= eff_window) {
                const uint32_t evict_idx = L + j - eff_window;
                if (evict_idx < L) {
                    pp.drop_toks[j] = eng->pen_hist[evict_idx];
                }
            }
        }
    }

    [enc setComputePipelineState:eng->pso_apply_logit_penalties];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_pen_ids offset:0 atIndex:1];
    [enc setBuffer:eng->b_pen_counts offset:0 atIndex:2];
    [enc setBuffer:(extra_toks_buf ? extra_toks_buf : eng->b_token_ids) offset:0 atIndex:3];
    [enc setBuffer:(use_sl ? eng->b_frspec_full_to_short : eng->b_pen_ids) offset:0 atIndex:4];
    [enc setBytes:&pp length:sizeof(pp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((total_items + 255U) / 256U, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sqf_encode_sample_batch(
    id<MTLComputeCommandEncoder> enc,
    StrataQwenFlashGpuEngine* eng,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t M,
    id<MTLBuffer> out_tok_buf,
    NSUInteger out_tok_off,
    id<MTLBuffer> top_ids_buf,
    NSUInteger top_ids_off,
    id<MTLBuffer> top_probs_buf,
    NSUInteger top_probs_off,
    id<MTLBuffer> keep_k_buf,
    NSUInteger keep_k_off,
    bool want_top2 = false,
    uint32_t shortlist_v_out = 0U,
    id<MTLBuffer> out_pmax_buf = nil,
    NSUInteger out_pmax_off = 0U
) {
    const bool use_map  = (shortlist_v_out > 0U) && (eng->b_frspec_ids != nil);
    const bool do_pmax  = (out_pmax_buf != nil) && (eng->b_sample_stage_lse != nil);

    SqfSampleParamsHost sp{};
    sp.M             = M;
    sp.vocab_size    = use_map ? shortlist_v_out : eng->cfg.vocab_size;
    sp.use_token_map = use_map ? 1U : 0U;
    sp.compute_pmax  = do_pmax ? 1U : 0U;

    const bool is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
    if (is_greedy) {
        sp.greedy   = 1U | (want_top2 ? 2U : 0U);
        sp.inv_temp = 1.0f;
        sp.top_k    = want_top2 ? 2U : 1U;
        sp.top_p    = 1.0f;
        sp.rng_lo   = 0U;
        sp.rng_hi   = 0U;
    } else {
        sp.greedy   = want_top2 ? 2U : 0U;
        sp.inv_temp = 1.0f / std::max<float>(sampling->temperature, 1e-5f);
        sp.top_k    = (sampling->top_k > 0U && sampling->top_k <= 64U) ? sampling->top_k : 20U;
        sp.top_p    = sampling->top_p;
        const uint64_t seed = sampling->rng_seed ^ (eng->sample_step * 0xD1342543DE82EF95ULL);
        sp.rng_lo = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
        sp.rng_hi = static_cast<uint32_t>(seed >> 32ULL);
        eng->sample_step += M;
    }

    [enc setComputePipelineState:eng->pso_sample_stage1];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:1];
    [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:2];
    [enc setBuffer:(eng->b_sample_stage_lse ? eng->b_sample_stage_lse : eng->b_sample_stage_v) offset:0 atIndex:3];
    [enc setBytes:&sp length:sizeof(sp) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(64, M, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:eng->pso_sample_stage2];
    [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
    [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
    [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:2];
    [enc setBuffer:(top_ids_buf ? top_ids_buf : out_tok_buf) offset:top_ids_off atIndex:3];
    [enc setBuffer:(top_probs_buf ? top_probs_buf : eng->b_logits) offset:top_probs_off atIndex:4];
    [enc setBuffer:(keep_k_buf ? keep_k_buf : out_tok_buf) offset:keep_k_off atIndex:5];
    [enc setBuffer:(eng->b_sample_stage_lse ? eng->b_sample_stage_lse : eng->b_sample_stage_v) offset:0 atIndex:6];
    [enc setBuffer:(use_map ? eng->b_frspec_ids : out_tok_buf) offset:0 atIndex:7];
    [enc setBuffer:(do_pmax ? out_pmax_buf : eng->b_sample_stage_v) offset:(do_pmax ? out_pmax_off : 0U) atIndex:8];
    [enc setBytes:&sp length:sizeof(sp) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

} // namespace

extern "C" {

StrataQwenFlashGpuEngine* strata_qwen_flash_gpu_create(
    StrataQwenFlashBundle* bundle,
    uint32_t               max_seq_len
) {
    if (!bundle || !bundle->part0_base || !bundle->part1_base || !bundle->header0 || !bundle->header1) {
        return nullptr;
    }

    /* Engine geometry validation (Round 6 rules F2–F9, Round 7 rules F10–F11) before shader compile / buffer allocation.
     *   F10: `indexer_budget` is whole KV blocks and at most `STRATA_QWEN_FLASH_MAX_INDEXER_BUDGET`. While the context
     *        is within the budget, attention is dense and reads every block through `block_loc` with no residency
     *        sync; a budget larger than a paged pool would make it read evicted blocks at their logical index,
     *        i.e. outside the pool.
     *   F11: the router kernels keep `sel_ids[16]` and clamp `top_k` to 16. */
    static_assert(STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS >= STRATA_QWEN_FLASH_MAX_INDEXER_BUDGET + 512U,
                  "paged KV pool minimum must cover the dense attention range");
    {
        const StrataQwenFlashConfig& gc = bundle->cfg;
        if (gc.indexer_budget < STRATA_QWEN_FLASH_KV_BLOCK_TOKENS ||
            (gc.indexer_budget % STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) != 0U ||
            gc.indexer_budget > STRATA_QWEN_FLASH_MAX_INDEXER_BUDGET ||
            gc.num_experts_per_tok == 0U || gc.num_experts_per_tok > 16U ||
            (gc.rotary_dim & 1U) != 0U || gc.rotary_dim > 64U ||
            gc.rotary_dim > gc.head_dim || gc.rotary_dim > gc.indexer_head_dim ||
            gc.num_key_value_heads == 0U || (gc.num_attention_heads % gc.num_key_value_heads) != 0U ||
            gc.head_dim == 0U || gc.head_dim > 256U ||
            gc.indexer_head_dim != 128U ||
            gc.linear_key_head_dim != 128U || gc.linear_value_head_dim != 128U ||
            gc.linear_num_key_heads == 0U || (gc.linear_num_value_heads % gc.linear_num_key_heads) != 0U ||
            gc.linear_conv_kernel_dim != 4U ||
            (gc.has_ple != 0U &&
             (gc.ple_conv_kernel_size != 4U || gc.ngram_size < 2U || gc.ngram_size > 4U ||
              gc.ngram_size != bundle->ple_header.ngram_size)) ||
            gc.indexer_compress_ratio != STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] Unsupported Qwen-Flash geometry in bundle header\n");
            return nullptr;
        }
    }

    @autoreleasepool {
        const auto t_start = std::chrono::steady_clock::now();

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] No Metal GPU device available\n");
            return nullptr;
        }

        std::string hdr_src = strata_load_shader_asset("strata_qwen_flash.h", "");
        if (hdr_src.empty()) hdr_src = sqf_read_text_file(sqf_resolve_sibling_path("../../include/strata_qwen_flash.h"));
        std::string msl_src = strata_load_shader_asset("strata_qwen_flash_kernels.metal", "");
        if (msl_src.empty()) msl_src = sqf_read_text_file(sqf_resolve_sibling_path("strata_qwen_flash_kernels.metal"));
        if (hdr_src.empty() || msl_src.empty()) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] Failed to read strata_qwen_flash.h or strata_qwen_flash_kernels.metal\n");
            return nullptr;
        }

        const std::string marker = "// @@STRATA_QWEN_FLASH_HEADER@@";
        size_t pos = msl_src.find(marker);
        if (pos != std::string::npos) {
            msl_src.replace(pos, marker.size(), hdr_src);
        }

        NSString* msl_ns = [NSString stringWithUTF8String:msl_src.c_str()];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        bool compiled_v4 = false;
        if (@available(macOS 27.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_1;
            compiled_v4 = true;
        } else if (@available(macOS 26.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_0;
            compiled_v4 = true;
        } else {
            opts.languageVersion = MTLLanguageVersion3_1;
        }
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
        }

        auto* eng    = new StrataQwenFlashGpuEngine();
        eng->device  = dev;
        eng->queue   = [dev newCommandQueue];
        eng->has_nax = compiled_v4;
        eng->bundle  = bundle;
        eng->hdr0    = *bundle->header0;
        eng->hdr1    = *bundle->header1;
        eng->cfg     = bundle->cfg;

        const uint32_t req_seq  = (max_seq_len > 0U) ? max_seq_len : STRATA_QWEN_FLASH_DEFAULT_MAX_SEQ_LEN;
        if (req_seq > STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP) {
            std::fprintf(stderr,
                         "[strata_qwen_flash_gpu] max_seq_len=%u exceeds indexer sel_bits[4096] cap (%u)\n",
                         req_seq, STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP);
            delete eng;
            return nullptr;
        }
        const uint32_t comp_rat = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
        eng->max_seq_len        = req_seq;
        eng->max_seq_alloc      = (std::max<uint32_t>(req_seq, 64U) + 63U) & ~63U;
        eng->max_blocks_alloc   = std::max<uint32_t>(64U, ((eng->max_seq_alloc / comp_rat) + 63U) & ~63U);
        eng->seq_pos            = 0U;
        eng->mtp_seq_pos        = 0U;
        eng->ple_prev_tokens[0] = eng->cfg.ple_eos_token_id;
        eng->ple_prev_tokens[1] = eng->cfg.ple_eos_token_id;
        eng->sample_step        = 0ULL;
        eng->spec_tree_enabled  = false;
        eng->mtp_kv_sync_enabled= true;
        eng->prof_enabled       = (std::getenv("STRATA_PROFILE_FLASH") != nullptr);
        eng->force_serial_dn    = (std::getenv("STRATA_SQF_SERIAL_DN") != nullptr);
        {
            const char* env_i8 = std::getenv("STRATA_QWEN_FLASH_INT8_PREFILL");
            eng->int8_prefill_enabled = (env_i8 != nullptr && std::atoi(env_i8) != 0);
            const char* env_no_ak = std::getenv("STRATA_QWEN_FLASH_NO_ADAPTIVE_K");
            eng->adaptive_k_enabled = !(env_no_ak != nullptr && std::atoi(env_no_ak) != 0);
            const char* env_no_fr = std::getenv("STRATA_QWEN_FLASH_NO_FRSPEC");
            eng->frspec_enabled = !(env_no_fr != nullptr && std::atoi(env_no_fr) != 0);
            eng->frspec_widened = false;
        }
        eng->progress_cb        = nullptr;
        eng->progress_user_ctx  = nullptr;
        std::memset(&eng->counters, 0, sizeof(eng->counters));

        std::snprintf(eng->device_name, sizeof(eng->device_name), "%s", [[dev name] UTF8String]);

        // Overlap Metal 4 library + persistent `MTLBinaryArchive` PSO loading concurrently with zero-copy UMA wiring!
        __block bool shader_ok = true;
        const char*  msl_ptr   = msl_src.data();
        const size_t msl_len   = msl_src.size();
        dispatch_group_t compile_grp = dispatch_group_create();
        dispatch_queue_t bg_q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        dispatch_group_async(compile_grp, bg_q, ^{
            @autoreleasepool {
                NSError* err = nil;
                id<MTLLibrary> lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
                if (!lib && compiled_v4) {
                    if (@available(macOS 26.0, *)) {
                        std::fprintf(stderr, "[strata_qwen_flash_gpu] Metal 4.1 compile failed (%s), trying Metal 4.0...\n",
                                     err ? [[err localizedDescription] UTF8String] : "unknown");
                        err = nil;
                        opts.languageVersion = MTLLanguageVersion4_0;
                        lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
                    }
                }
                if (!lib) {
                    std::fprintf(stderr, "[strata_qwen_flash_gpu] Metal 4 compile error: %s\n",
                                 err ? [[err localizedDescription] UTF8String] : "unknown");
                    shader_ok = false;
                    return;
                }
                if (err) {
                    std::fprintf(stderr, "[strata_qwen_flash_gpu] Metal shader warnings:\n%s\n",
                                 [[err localizedDescription] UTF8String]);
                }
                eng->library = lib;

                StrataMetalPsoCache* pso_cache = strata_metal_cache_open_with_opts(
                    dev, lib, "qwen_flash", msl_ptr, msl_len, opts);

                auto make_pso_const = [&](const char* fn_name, uint32_t m_const) -> id<MTLComputePipelineState> {
                    NSError* p_err = nil;
                    id<MTLComputePipelineState> pso = nil;
                    if (pso_cache) {
                        pso = (m_const > 0U)
                            ? strata_metal_cache_make_pso_const_u32(pso_cache, fn_name, 0U, m_const, &p_err)
                            : strata_metal_cache_make_pso(pso_cache, fn_name, nil, &p_err);
                    } else {
                        NSString* name = [NSString stringWithUTF8String:fn_name];
                        id<MTLFunction> fn = nil;
                        if (m_const > 0U) {
                            MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
                            [fcv setConstantValue:&m_const type:MTLDataTypeUInt atIndex:0];
                            fn = [lib newFunctionWithName:name constantValues:fcv error:&p_err];
                        } else {
                            fn = [lib newFunctionWithName:name];
                        }
                        if (fn) pso = [dev newComputePipelineStateWithFunction:fn error:&p_err];
                    }
                    if (!pso) {
                        std::fprintf(stderr, "[strata_qwen_flash_gpu] PSO compile error (%s, M=%u): %s\n",
                                     fn_name, m_const,
                                     p_err ? [[p_err localizedDescription] UTF8String] : "unknown");
                    }
                    return pso;
                };

                struct PsoSlot {
                    const char*                           fn_name;
                    uint32_t                              m_const; // 0 => unspecialized, >0 => SQF_CONST_M
                    id<MTLComputePipelineState> __strong* dst;
                };
                std::vector<PsoSlot> pso_vec = {
                    {"sqf_zero_u32",                      0U, &eng->pso_zero_u32},
                    {"sqf_copy_u32",                      0U, &eng->pso_copy_u32},
                    {"sqf_f32_to_f16_pad",                0U, &eng->pso_f32_to_f16_pad},
                    {"sqf_embed_lookup_q4",               0U, &eng->pso_embed_q4},
                    {"sqf_f16_small_n_prefill",           0U, &eng->pso_f16_small_n_prefill},
                    {"sqf_f16_gemv_8lane",                0U, &eng->pso_f16_gemv_8lane},
                    {"sqf_hc_down_inj_act_batch",         0U, &eng->pso_hc_down_inj_act},
                    {"sqf_f16_shared_gate_up_silu_batch", 0U, &eng->pso_f16_shared_gu_silu},
                    {"sqf_q8_gemv_batch",                 0U, &eng->pso_q8_gemv_batch},
                    {"sqf_q8_gemv_8lane",                 0U, &eng->pso_q8_gemv_8lane},
                    {"sqf_q8_gemm_nax_64x64_k128_sg4",    0U, &eng->pso_q8_gemm_nax_64x64_k128_sg4},
                    {"sqf_q8_gemm_nax_32x64_k128_sg2",    0U, &eng->pso_q8_gemm_nax_32x64_k128_sg2},
                    {"sqf_q8_gemm_nax_16x64_k128_sg2",    0U, &eng->pso_q8_gemm_nax_16x64_k128_sg2},
                    {"sqf_q8_gemm_nax_m16",               0U, &eng->pso_q8_gemm_nax_m16},
                    {"sqf_q8_gemm_nax_m64",               0U, &eng->pso_q8_gemm_nax_m64},
                    {"sqf_q4_shared_gu_silu_nax_m64",     0U, &eng->pso_q4_shared_gu_silu_nax_m64},
                    {"sqf_q4_shared_gu_silu_nax_m32",     0U, &eng->pso_q4_shared_gu_silu_nax_m32},
                    {"sqf_q4_gemm_nax_64x64_k128_sg4",    0U, &eng->pso_q4_gemm_nax_64x64_k128_sg4},
                    {"sqf_q4_gemm_nax_32x64_k128_sg2",    0U, &eng->pso_q4_gemm_nax_32x64_k128_sg2},
                    {"sqf_q4_gemm_nax_16x64_k128_sg2",    0U, &eng->pso_q4_gemm_nax_16x64_k128_sg2},
                    {"sqf_q4_gemm_nax_m16",               0U, &eng->pso_q4_gemm_nax_m16},
                    {"sqf_q4_gemm_nax_m64",               0U, &eng->pso_q4_gemm_nax_m64},
                    {"sqf_f16_gemm_nax_64x128_sg4",       0U, &eng->pso_f16_gemm_nax_64x128_sg4},
                    {"sqf_f16_gemm_nax_64x64_sg4",        0U, &eng->pso_f16_gemm_nax_64x64_sg4},
                    {"sqf_f16_gemm_nax_32x64_sg2",        0U, &eng->pso_f16_gemm_nax_32x64_sg2},
                    {"sqf_f16_gemm_nax_16x32_sg2",        0U, &eng->pso_f16_gemm_nax_16x32_sg2},
                    {"sqf_q8_hc_down_act_nax_m64",        0U, &eng->pso_q8_hc_down_act_nax_m64},
                    {"sqf_q8_hc_down_act_nax_m32",        0U, &eng->pso_q8_hc_down_act_nax_m32},
                    {"sqf_f16_hc_down_act_nax_m64",       0U, &eng->pso_f16_hc_down_act_nax_m64},
                    {"sqf_f16_hc_down_act_nax_m32",       0U, &eng->pso_f16_hc_down_act_nax_m32},
                    {"sqf_q8_hc_up_blend_nax_m64",        0U, &eng->pso_q8_hc_up_blend_nax_m64},
                    {"sqf_q8_hc_up_blend_nax_m32",        0U, &eng->pso_q8_hc_up_blend_nax_m32},
                    {"sqf_f16_hc_up_blend_nax_m64",       0U, &eng->pso_f16_hc_up_blend_nax_m64},
                    {"sqf_f16_hc_up_blend_nax_m32",       0U, &eng->pso_f16_hc_up_blend_nax_m32},
                    {"sqf_f16_hc_inject_prefill",         0U, &eng->pso_f16_hc_inject_prefill},
                    {"sqf_f16_lin_ab_prefill",            0U, &eng->pso_f16_lin_ab_prefill},
                    {"sqf_grouped_rmsnorm",               0U, &eng->pso_grouped_rmsnorm},
                    {"sqf_hc_down_act",                   0U, &eng->pso_hc_down_act},
                    {"sqf_hc_blend",                      0U, &eng->pso_hc_blend},
                    {"sqf_hc_residual_add",               0U, &eng->pso_hc_residual_add},
                    {"sqf_hc_res_add_grouped_rmsnorm",    0U, &eng->pso_hc_res_add_rmsnorm},
                    {"sqf_ple_gate_gv",                   0U, &eng->pso_ple_gate_gv},
                    {"sqf_ple_dilated_conv1d_add",        0U, &eng->pso_ple_conv1d_add},
                    {"sqf_deltanet_conv1d_silu",          0U, &eng->pso_deltanet_conv1d},
                    {"sqf_deltanet_conv1d_silu_prefill",  0U, &eng->pso_deltanet_conv1d_prefill},
                    {"sqf_deltanet_conv1d_save_tail",     0U, &eng->pso_deltanet_conv1d_save_tail},
                    {"sqf_deltanet_recurrent_gated_norm", 0U, &eng->pso_deltanet_rec},
                    {"sqf_deltanet_prep_batch",           0U, &eng->pso_deltanet_prep_batch},
                    {"sqf_deltanet_rec_batch",            0U, &eng->pso_deltanet_rec_batch},
                    {"sqf_deltanet_post_batch",           0U, &eng->pso_deltanet_post_batch},
                    {"sqf_deltanet_wy_qk_chunk32",        0U, &eng->pso_deltanet_wy_qk},
                    {"sqf_deltanet_wy_prep_hv_chunk32",   0U, &eng->pso_deltanet_wy_prep_hv},
                    {"sqf_deltanet_wy_recur_chunk32",     0U, &eng->pso_deltanet_wy_recur},
                    {"sqf_qsa_qkv_prep",                  0U, &eng->pso_qsa_qkv_prep},
                    {"sqf_qsa_indexer_prep",              0U, &eng->pso_qsa_idx_prep},
                    {"sqf_qsa_indexer_prep_prefill",      0U, &eng->pso_qsa_idx_prep_prefill},
                    {"sqf_qsa_indexer_build_blocks",      0U, &eng->pso_qsa_idx_blocks},
                    {"sqf_qsa_indexer_build_branch_b_block", 0U, &eng->pso_qsa_idx_branch_b_blk},
                    {"sqf_qsa_indexer_score",             0U, &eng->pso_qsa_idx_score},
                    {"sqf_qsa_indexer_score_batch",       0U, &eng->pso_qsa_idx_score_batch},
                    {"sqf_qsa_indexer_score_tiled",       0U, &eng->pso_qsa_idx_score_tiled},
                    {"sqf_qsa_indexer_score_nax",         0U, &eng->pso_qsa_idx_score_nax},
                    {"sqf_qsa_indexer_topk",              0U, &eng->pso_qsa_idx_topk},
                    {"sqf_qsa_decode_splitkv",            0U, &eng->pso_qsa_decode_split},
                    {"sqf_qsa_decode_splitkv_gqa12_hd256",0U, &eng->pso_qsa_decode_split_gqa12},
                    {"sqf_qsa_decode_reduce",             0U, &eng->pso_qsa_decode_reduce},
                    {"sqf_tree_compact_branch_b",         0U, &eng->pso_tree_compact_b},
                    {"sqf_qsa_prefill_dense_nax",         0U, &eng->pso_qsa_prefill_nax},
                    {"sqf_qsa_sparse_prefill_nax",        0U, &eng->pso_qsa_sparse_prefill_nax},
                    {"sqf_moe_topk_softmax",              0U, &eng->pso_moe_topk},
                    {"sqf_swiglu_f32_to_f16",             0U, &eng->pso_swiglu},
                    {"sqf_shared_gate_dot",               0U, &eng->pso_shared_gate_dot},
                    {"sqf_moe_topk_and_shared_gate",      0U, &eng->pso_moe_topk_shared_gate},
                    {"sqf_moe_gather_gate_up_silu",       0U, &eng->pso_moe_gather_gu},
                    {"sqf_moe_gather_down",               0U, &eng->pso_moe_gather_dn},
                    {"sqf_moe_bin_tokens",                0U, &eng->pso_moe_bin_tokens},
                    {"sqf_moe_prefill_bin_tokens",        0U, &eng->pso_moe_prefill_bin_tokens},
                    {"sqf_moe_prefill_gu_nax",            0U, &eng->pso_moe_prefill_gu_nax},
                    {"sqf_moe_prefill_dn_nax",            0U, &eng->pso_moe_prefill_dn_nax},
                    {"sqf_moe_bin_tokens_tri",            0U, &eng->pso_moe_bin_tokens_tri},
                    {"sqf_moe_gather_x_f16",              0U, &eng->pso_moe_gather_x_f16},
                    {"sqf_moe_prefill_gu_nax_m16",        0U, &eng->pso_moe_prefill_gu_nax_m16},
                    {"sqf_moe_prefill_gu_nax_m32",        0U, &eng->pso_moe_prefill_gu_nax_m32},
                    {"sqf_moe_prefill_gu_nax_m64",        0U, &eng->pso_moe_prefill_gu_nax_m64},
                    {"sqf_moe_prefill_gu_nax_m128",       0U, &eng->pso_moe_prefill_gu_nax_m128},
                    {"sqf_moe_prefill_dn_nax_m16",        0U, &eng->pso_moe_prefill_dn_nax_m16},
                    {"sqf_moe_prefill_dn_nax_m32",        0U, &eng->pso_moe_prefill_dn_nax_m32},
                    {"sqf_moe_prefill_dn_nax_m64",        0U, &eng->pso_moe_prefill_dn_nax_m64},
                    {"sqf_moe_prefill_dn_nax_m128",       0U, &eng->pso_moe_prefill_dn_nax_m128},
                    {"sqf_moe_combine_hc_res",            0U, &eng->pso_moe_combine},
                    {"sqf_moe_combine_hc_res_f16",        0U, &eng->pso_moe_combine_f16},
                    {"sqf_mtp_ungrouped_norm_split",      0U, &eng->pso_mtp_norm_split},
                    {"sqf_mtp_add_broadcast",             0U, &eng->pso_mtp_add_bcast},
                    {"sqf_sample_vocab_batch",            0U, &eng->pso_sample_vocab},
                    {"sqf_sample_vocab_stage1",           0U, &eng->pso_sample_stage1},
                    {"sqf_sample_vocab_stage2",           0U, &eng->pso_sample_stage2},
                    {"sqf_apply_logit_penalties",         0U, &eng->pso_apply_logit_penalties},
                    {"sqf_pack_frspec_q4",                0U, &eng->pso_pack_frspec_q4},
                    {"sqf_patch_frspec_q4",               0U, &eng->pso_patch_frspec_q4},
                    {"sqf_quant_f16_to_i8_row",           0U, &eng->pso_quant_f16_to_i8_row},
                    {"sqf_q8_requant_row_i8",             0U, &eng->pso_q8_requant_row_i8},
                    {"sqf_q8_requant_row_i8_w32",         0U, &eng->pso_q8_requant_row_i8_w32},
                    {"sqf_q4_requant_row_i8",             0U, &eng->pso_q4_requant_row_i8},
                    {"sqf_i8_gemm_nax_64x128_sg4",        0U, &eng->pso_i8_gemm_nax_64x128_sg4},
                    {"sqf_i8_gemm_nax_64x64_sg4",         0U, &eng->pso_i8_gemm_nax_64x64_sg4},
                    {"sqf_i8_gemm_nax_32x64_sg2",         0U, &eng->pso_i8_gemm_nax_32x64_sg2},
                    {"sqf_i8_hc_down_act_nax_m64",        0U, &eng->pso_i8_hc_down_act_nax_m64},
                    {"sqf_i8_hc_down_act_nax_m32",        0U, &eng->pso_i8_hc_down_act_nax_m32},
                    {"sqf_i8_hc_up_blend_nax_m64",        0U, &eng->pso_i8_hc_up_blend_nax_m64},
                    {"sqf_i8_hc_up_blend_nax_m32",        0U, &eng->pso_i8_hc_up_blend_nax_m32},
                    {"sqf_i8_shared_gu_silu_nax_m64",     0U, &eng->pso_i8_shared_gu_silu_nax_m64},
                    {"sqf_i8_shared_gu_silu_nax_m32",     0U, &eng->pso_i8_shared_gu_silu_nax_m32},
                };
                // Precompile ALL `SQF_CONST_M` shapes (`M=1..8`, `max_b=4,5,6,8`, `shortlist=1,2`) in parallel during weight page-in
                // so EVERY decode/prefill/N-gram/tree/`cb_fix` shape has 0 mid-session shader compiles and `Steady MajFlt == 0`!
                for (uint32_t m = 1U; m <= kMaxSpecBatch; ++m) {
                    pso_vec.push_back({"sqf_q4_gemv_batch",                 m, &eng->pso_q4_gemv_m[m - 1U]});
                    pso_vec.push_back({"sqf_q4_gemv_8lane",                 m, &eng->pso_q4_gemv_8lane_m[m - 1U]});
                    pso_vec.push_back({"sqf_q4_gemv_lin_qz_batch",          m, &eng->pso_q4_gemv_lin_qz_m[m - 1U]});
                    pso_vec.push_back({"sqf_q4_gemv_attn_qkv_batch",        m, &eng->pso_q4_gemv_attn_qkv_m[m - 1U]});
                    pso_vec.push_back({"sqf_f16_gemv_batch",                m, &eng->pso_f16_gemv_m[m - 1U]});
                    pso_vec.push_back({"sqf_f16_gemv_row_tg128",            m, &eng->pso_f16_gemv_row_tg128_m[m - 1U]});
                    pso_vec.push_back({"sqf_f16_lin_ab_batch",              m, &eng->pso_f16_lin_ab_m[m - 1U]});
                    pso_vec.push_back({"sqf_q8_hc_down_inj_act_batch",      m, &eng->pso_q8_hc_down_inj_act_m[m - 1U]});
                    pso_vec.push_back({"sqf_q8_hc_up_blend",                m, &eng->pso_q8_hc_up_blend_m[m - 1U]});
                    pso_vec.push_back({"sqf_q4_shared_gate_up_silu_batch",  m, &eng->pso_q4_shared_gu_silu_m[m - 1U]});
                }
                pso_vec.push_back({"sqf_q4_lm_head_shortlist",     1U, &eng->pso_q4_lm_head_shortlist_m[0]});
                pso_vec.push_back({"sqf_q4_lm_head_shortlist",     2U, &eng->pso_q4_lm_head_shortlist_m[1]});
                constexpr uint32_t kMoeCaps[4] = {4U, 5U, 6U, 8U};
                for (uint32_t bi = 0U; bi < 4U; ++bi) {
                    pso_vec.push_back({"sqf_moe_grouped_gate_up_silu", kMoeCaps[bi], &eng->pso_moe_grouped_gu_b[bi]});
                    pso_vec.push_back({"sqf_moe_grouped_down",         kMoeCaps[bi], &eng->pso_moe_grouped_dn_b[bi]});
                }

                const size_t n_psos = pso_vec.size();
                PsoSlot* pso_ptr    = pso_vec.data();

                dispatch_apply(n_psos, bg_q, ^(size_t idx) {
                    @autoreleasepool {
                        *pso_ptr[idx].dst = make_pso_const(pso_ptr[idx].fn_name, pso_ptr[idx].m_const);
                    }
                });

                if (pso_cache) {
                    (void)strata_metal_cache_finish(pso_cache);
                    eng->metal_cache_hit = pso_cache->cache_hit;
                    eng->pso_load_ms     = pso_cache->elapsed_ms;
                    strata_metal_cache_destroy(pso_cache);
                }

                for (size_t i = 0; i < n_psos; ++i) {
                    if (!(*pso_vec[i].dst)) {
                        shader_ok = false;
                        break;
                    }
                }
            }
        });

        // 1. Two-Buffer Zero-Copy 16 KiB-Aligned UMA Mapping (`uma_part0.strata` + `uma_part1.strata`)
        (void)posix_madvise(const_cast<uint8_t*>(bundle->part0_base), bundle->part0_bytes, POSIX_MADV_WILLNEED);
        (void)posix_madvise(const_cast<uint8_t*>(bundle->part1_base), bundle->part1_bytes, POSIX_MADV_WILLNEED);

        bool zc0 = false, zc1 = false;
        if ((reinterpret_cast<uintptr_t>(bundle->part0_base) % STRATA_QWEN_FLASH_PAGE_BYTES) == 0U &&
            (bundle->part0_bytes % STRATA_QWEN_FLASH_PAGE_BYTES) == 0U) {
            eng->b_part0 = [dev newBufferWithBytesNoCopy:const_cast<uint8_t*>(bundle->part0_base)
                                                  length:bundle->part0_bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
            if (eng->b_part0) zc0 = true;
        }
        if (!eng->b_part0) {
            eng->b_part0 = [dev newBufferWithBytes:bundle->part0_base
                                            length:bundle->part0_bytes
                                           options:MTLResourceStorageModeShared];
        }

        if ((reinterpret_cast<uintptr_t>(bundle->part1_base) % STRATA_QWEN_FLASH_PAGE_BYTES) == 0U &&
            (bundle->part1_bytes % STRATA_QWEN_FLASH_PAGE_BYTES) == 0U) {
            eng->b_part1 = [dev newBufferWithBytesNoCopy:const_cast<uint8_t*>(bundle->part1_base)
                                                  length:bundle->part1_bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
            if (eng->b_part1) zc1 = true;
        }
        if (!eng->b_part1) {
            eng->b_part1 = [dev newBufferWithBytes:bundle->part1_base
                                            length:bundle->part1_bytes
                                           options:MTLResourceStorageModeShared];
        }

        if (!eng->b_part0 || !eng->b_part1) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] Failed to bind UMA part0/part1 buffers\n");
            dispatch_group_wait(compile_grp, DISPATCH_TIME_FOREVER);
            strata_qwen_flash_gpu_destroy(eng);
            return nullptr;
        }

        eng->zero_copy_mmap     = (zc0 && zc1);
        eng->part0_mapped_bytes = static_cast<uint64_t>(bundle->part0_bytes);
        eng->part1_mapped_bytes = static_cast<uint64_t>(bundle->part1_bytes);

        // 2. Persistent State & KV/Indexer Caches
        const uint32_t D          = eng->cfg.hidden_size;
        const uint32_t H          = eng->cfg.hc_count;
        const uint32_t hc_dim     = H * D;
        const uint32_t R          = eng->cfg.hc_lowrank;
        const uint32_t V          = eng->cfg.vocab_size;
        const uint32_t num_lin    = std::max<uint32_t>(1U, eng->cfg.num_linear_layers);
        const uint32_t num_full   = eng->cfg.num_full_attn_layers;
        const uint32_t full_slots = std::max<uint32_t>(1U, num_full + (eng->cfg.has_mtp ? 1U : 0U));
        const uint32_t n_k        = std::max<uint32_t>(1U, eng->cfg.linear_num_key_heads);
        const uint32_t n_v        = std::max<uint32_t>(1U, eng->cfg.linear_num_value_heads);
        const uint32_t k_dim      = n_k * std::max<uint32_t>(128U, eng->cfg.linear_key_head_dim);
        const uint32_t v_dim      = n_v * std::max<uint32_t>(128U, eng->cfg.linear_value_head_dim);
        const uint32_t conv_dim   = 2U * k_dim + v_dim;
        const uint32_t n_q        = std::max<uint32_t>(1U, eng->cfg.num_attention_heads);
        const uint32_t n_kv       = std::max<uint32_t>(1U, eng->cfg.num_key_value_heads);
        const uint32_t head_dim   = std::max<uint32_t>(64U, eng->cfg.head_dim);
        const uint32_t q_out      = n_q * 2U * head_dim;
        const uint32_t kv_out     = n_kv * head_dim;
        const uint32_t o_in       = n_q * head_dim;
        const uint32_t idx_nh     = std::max<uint32_t>(1U, eng->cfg.indexer_n_heads);
        const uint32_t idx_hd     = std::max<uint32_t>(64U, eng->cfg.indexer_head_dim);
        const uint32_t idx_out    = (idx_nh + std::max<uint32_t>(1U, eng->cfg.indexer_kv_heads)) * idx_hd;
        const uint32_t blk_topk   = std::max<uint32_t>(1U, eng->cfg.indexer_budget / comp_rat);
        const uint32_t E          = std::max<uint32_t>(1U, eng->cfg.num_experts);
        const uint32_t top_k      = std::max<uint32_t>(1U, eng->cfg.num_experts_per_tok);
        const uint32_t moe_dff    = std::max<uint32_t>(64U, eng->cfg.moe_intermediate_size);
        const uint32_t sh_dff     = std::max<uint32_t>(64U, eng->cfg.shared_intermediate_size);
        const uint32_t ple_dim    = std::max<uint32_t>(64U, eng->cfg.ple_embed_dim);
        const uint32_t ple_slen   = 3U * std::max<uint32_t>(1U, eng->cfg.ngram_size);

        eng->conv_layer_stride    = 3ULL * conv_dim * sizeof(uint16_t);
        eng->rec_layer_stride     = static_cast<uint64_t>(n_v) * 128ULL * 128ULL * sizeof(float);
        eng->kv_layer_stride      = static_cast<uint64_t>(n_kv) * eng->max_seq_alloc * head_dim * sizeof(uint16_t);
        eng->idx_raw_layer_stride = static_cast<uint64_t>(eng->max_seq_alloc) * idx_hd * sizeof(uint16_t);
        eng->idx_blk_layer_stride = static_cast<uint64_t>(eng->max_blocks_alloc) * idx_hd * sizeof(uint16_t);

        eng->conv_state_bytes = static_cast<uint64_t>(num_lin) * eng->conv_layer_stride;
        eng->rec_state_bytes  = static_cast<uint64_t>(num_lin) * eng->rec_layer_stride;
        eng->ple_state_bytes  = eng->cfg.has_ple ? (static_cast<uint64_t>(ple_slen) * hc_dim * sizeof(float)) : 0ULL;

        const uint64_t kv_single = static_cast<uint64_t>(full_slots) * eng->kv_layer_stride;
        const uint64_t idx_raw_b = static_cast<uint64_t>(full_slots) * eng->idx_raw_layer_stride;
        const uint64_t idx_blk_b = static_cast<uint64_t>(full_slots) * eng->idx_blk_layer_stride;

        eng->kv_cache_bytes_allocated = 2ULL * kv_single + idx_raw_b + idx_blk_b;
        eng->kv_cache_bytes_per_token = static_cast<uint64_t>(num_full) *
            (2ULL * n_kv * head_dim * sizeof(uint16_t) +
             idx_hd * sizeof(uint16_t) +
             (idx_hd * sizeof(uint16_t)) / comp_rat);

        eng->b_conv_state     = sqf_alloc_private_aligned(dev, eng->conv_state_bytes);
        eng->b_rec_state      = sqf_alloc_private_aligned(dev, eng->rec_state_bytes);
        eng->b_ple_conv_state = sqf_alloc_private_aligned(dev, std::max<size_t>(4096U, eng->ple_state_bytes));
        eng->b_k_cache        = sqf_alloc_private_aligned(dev, kv_single);
        eng->b_v_cache        = sqf_alloc_private_aligned(dev, kv_single);
        eng->b_idx_raw_k      = sqf_alloc_private_aligned(dev, idx_raw_b);
        eng->b_idx_block_k    = sqf_alloc_private_aligned(dev, idx_blk_b);
        eng->b_mtp_hc_prev    = sqf_alloc_private_aligned(dev, static_cast<size_t>(hc_dim) * sizeof(float));

        // 3. Speculative Rollback Zero-Copy Ring Snapshot Slots (`kMaxSpecSnapSlots = 6`)
        size_t rollback_bytes = 0U;
        bool   snap_slots_ok  = true;
        if (eng->cfg.has_mtp) {
            const size_t p_sz = std::max<size_t>(4096U, eng->ple_state_bytes);
            for (uint32_t s = 0U; s < kMaxSpecSnapSlots; ++s) {
                eng->b_rec_slots[s]      = sqf_alloc_private_aligned(dev, eng->rec_state_bytes);
                eng->b_conv_slots[s]     = sqf_alloc_private_aligned(dev, eng->conv_state_bytes);
                eng->b_ple_conv_slots[s] = sqf_alloc_private_aligned(dev, p_sz);
                if (!eng->b_rec_slots[s] || !eng->b_conv_slots[s] || !eng->b_ple_conv_slots[s]) {
                    snap_slots_ok = false;
                }
            }
            rollback_bytes = static_cast<size_t>(kMaxSpecSnapSlots) *
                             (eng->rec_state_bytes + eng->conv_state_bytes + p_sz);
        }

        // 4. Scratch Workspaces (`kMaxPrefillChunk = 256`)
        auto track_priv = [&](size_t sz, size_t* acc) -> id<MTLBuffer> {
            *acc += sz;
            return sqf_alloc_private_aligned(dev, sz);
        };
        size_t ws = rollback_bytes;

        eng->b_hc          = track_priv(static_cast<size_t>(kMaxPrefillChunk) * hc_dim * sizeof(float), &ws);
        eng->b_mtp_hc      = track_priv(static_cast<size_t>(kMaxPrefillChunk) * hc_dim * sizeof(float), &ws);
        eng->b_hc_norm_f32 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * hc_dim * sizeof(float), &ws);
        eng->b_hc_norm_f16 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * hc_dim * sizeof(uint16_t), &ws);
        eng->b_hc_down_f32 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * R * sizeof(float), &ws);
        /* `+ 128`: the Q8 up-blend NAX kernel walks `R` in 128-column static slices (`kt_cnt = ceil(R / 128)`), so for
         * `R % 128 != 0` (stock `R = 320`) the last tile of the last row reads up to 127 halves past `[chunk, R]`.
         * Those columns meet zero weights; the zero-filled tail keeps the read inside this buffer. */
        eng->b_hc_u_f16    = track_priv((static_cast<size_t>(kMaxPrefillChunk) * R + 128ULL) * sizeof(uint16_t), &ws);
        eng->b_hc_up_f32   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * hc_dim * sizeof(float), &ws);
        eng->b_hc_inj_f32  = track_priv(static_cast<size_t>(kMaxPrefillChunk) * H * sizeof(float), &ws);
        eng->b_x_in_f16    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(uint16_t), &ws);
        eng->b_x_in_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float), &ws);
        eng->b_block_y_f32 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float), &ws);
        eng->b_ple_emb_f16 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * ple_dim * sizeof(uint16_t), &ws);
        eng->b_ple_gvn_f32 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * hc_dim * sizeof(float), &ws);

        eng->b_qkv         = track_priv(static_cast<size_t>(kMaxPrefillChunk) * std::max(conv_dim, q_out) * sizeof(uint16_t), &ws);
        eng->b_qkv_conv    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * conv_dim * sizeof(uint16_t), &ws);
        eng->b_z           = track_priv(static_cast<size_t>(kMaxPrefillChunk) * v_dim * sizeof(uint16_t), &ws);
        eng->b_a           = track_priv(static_cast<size_t>(kMaxPrefillChunk) * n_v * sizeof(float), &ws);
        eng->b_b           = track_priv(static_cast<size_t>(kMaxPrefillChunk) * n_v * sizeof(float), &ws);
        eng->b_dn_qk_hat   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 2ULL * k_dim * sizeof(float), &ws);
        eng->b_dn_gates    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 2ULL * n_v * sizeof(float), &ws);
        eng->b_dn_raw_out  = track_priv(static_cast<size_t>(kMaxPrefillChunk) * v_dim * sizeof(float), &ws);
        eng->b_k_raw       = track_priv(static_cast<size_t>(kMaxPrefillChunk) * kv_out * sizeof(uint16_t), &ws);
        eng->b_v_raw       = track_priv(static_cast<size_t>(kMaxPrefillChunk) * kv_out * sizeof(uint16_t), &ws);
        eng->b_q_rope      = track_priv(static_cast<size_t>(n_q) * kMaxPrefillChunk * head_dim * sizeof(uint16_t), &ws);
        eng->b_attn_out    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * std::max(v_dim, o_in) * sizeof(uint16_t), &ws);
        eng->b_idx_qk_f32       = track_priv(static_cast<size_t>(kMaxPrefillChunk) * idx_out * sizeof(float), &ws);
        eng->b_idx_q_rot        = track_priv(static_cast<size_t>(kMaxPrefillChunk) * idx_nh * idx_hd * sizeof(float), &ws);
        eng->b_idx_blk_branch_b = track_priv(4096U, &ws);
        // Shared (UMA host-visible) so the Step 4 paged-KV exact selected-block check can read top-512 + scores
        eng->b_idx_scores  = sqf_alloc_shared_prefaulted(dev, static_cast<size_t>(kMaxPrefillChunk) * eng->max_blocks_alloc * sizeof(float));
        eng->b_idx_sel_blocks = sqf_alloc_shared_prefaulted(dev, static_cast<size_t>(kMaxPrefillChunk) * blk_topk * sizeof(uint32_t));
        ws += static_cast<size_t>(kMaxPrefillChunk) * (eng->max_blocks_alloc * sizeof(float) + blk_topk * sizeof(uint32_t));
        const bool has_sparse_nax_shape =
            eng->has_nax && (head_dim == 256U) && (n_kv > 0U) && (n_q == 12U * n_kv) && (comp_rat == 4U);
        const uint32_t part_rows = has_sparse_nax_shape ? kMaxSpecBatch : kMaxPrefillChunk;
        eng->b_part_m      = track_priv(static_cast<size_t>(part_rows) * kMaxDecodeSplits * n_q * sizeof(float), &ws);
        eng->b_part_l      = track_priv(static_cast<size_t>(part_rows) * kMaxDecodeSplits * n_q * sizeof(float), &ws);
        eng->b_part_o      = track_priv(static_cast<size_t>(part_rows) * kMaxDecodeSplits * n_q * head_dim * sizeof(float), &ws);

        eng->b_router_logits    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * E * sizeof(float), &ws);
        eng->b_topk_idx         = track_priv(static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(uint32_t), &ws);
        eng->b_topk_w           = track_priv(static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(float), &ws);
        eng->b_moe_mid_f16      = track_priv((static_cast<size_t>(kMaxPrefillChunk) * top_k + 128ULL) * moe_dff * sizeof(uint16_t), &ws);
        const size_t moe_dn_sz  = std::max<size_t>(
            static_cast<size_t>(kMaxPrefillChunk) * top_k * D * sizeof(float),
            (static_cast<size_t>(kMaxPrefillChunk) * top_k + 128ULL) * D * sizeof(uint16_t));
        eng->b_moe_down_part    = track_priv(moe_dn_sz, &ws);
        eng->b_moe_sorted_pairs = track_priv(static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(uint32_t), &ws);
        const size_t moe_tiles_cap = std::max<size_t>(2048ULL, static_cast<size_t>(kMaxPrefillChunk) * top_k + E);
        eng->b_moe_work_tiles   = track_priv(moe_tiles_cap * 16ULL, &ws);
        eng->b_moe_num_tiles    = track_priv(64U, &ws);
        eng->b_moe_indirect_args= track_priv(256U, &ws);
        eng->ple_all_rows.resize(static_cast<size_t>(kMaxPrefillChunk) * STRATA_QWEN_FLASH_MAX_NGRAM_HEADS, 0ULL);
        eng->b_shared_gate_f32  = track_priv(static_cast<size_t>(kMaxPrefillChunk) * sh_dff * sizeof(float), &ws);
        eng->b_shared_up_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * sh_dff * sizeof(float), &ws);
        eng->b_shared_mid_f16   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * sh_dff * sizeof(uint16_t), &ws);
        eng->b_shared_out_f32   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float), &ws);
        eng->b_shared_scalar    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * sizeof(float), &ws);
        eng->b_sample_stage_v   = track_priv(static_cast<size_t>(kMaxSpecBatch) * 64ULL * 64ULL * sizeof(float), &ws);
        eng->b_sample_stage_i   = track_priv(static_cast<size_t>(kMaxSpecBatch) * 64ULL * 64ULL * sizeof(uint32_t), &ws);
        eng->b_sample_stage_lse = track_priv(static_cast<size_t>(kMaxSpecBatch) * 64ULL * sizeof(float), &ws);

        // 32K SLC-Pinned FR-Spec Draft `lm_head` (`45.0 MiB < 48 MiB` M5 Max SLC!)
        const bool build_frspec = (eng->cfg.has_mtp != 0U) && (eng->hdr1.mtp.enabled != 0U) &&
                                  (V > kFrSpecShortlistSize) && ((eng->hdr1.lm_head.cols % 64U) == 0U);
        if (build_frspec) {
            const uint32_t sl_cols = eng->hdr1.lm_head.cols;
            const uint64_t w_b     = sqf_align_up_u64(static_cast<uint64_t>(kFrSpecShortlistSize) * (sl_cols / 2U), 4096ULL);
            const uint64_t s_b     = sqf_align_up_u64(static_cast<uint64_t>(kFrSpecShortlistSize) * (sl_cols / 64U) * sizeof(uint16_t), 4096ULL);
            const uint64_t tot_b   = w_b + 2ULL * s_b;
            eng->b_frspec_lm_head  = track_priv(static_cast<size_t>(tot_b), &ws);
            eng->b_frspec_ids      = sqf_alloc_shared_prefaulted(dev, static_cast<size_t>(kFrSpecShortlistSize) * sizeof(uint32_t));
            eng->b_frspec_full_to_short = sqf_alloc_shared_prefaulted(dev, static_cast<size_t>(V) * sizeof(uint32_t));
            eng->frspec_lm_head_desc.weight_off   = 0ULL;
            eng->frspec_lm_head_desc.scales_off   = w_b;
            eng->frspec_lm_head_desc.biases_off   = w_b + s_b;
            eng->frspec_lm_head_desc.rows_logical = kFrSpecShortlistSize;
            eng->frspec_lm_head_desc.rows_padded  = kFrSpecShortlistSize;
            eng->frspec_lm_head_desc.cols         = sl_cols;
            eng->frspec_lm_head_desc.group_size   = 64U;
            eng->frspec_shortlist_size            = kFrSpecShortlistSize;

            if (eng->b_frspec_ids && eng->b_frspec_full_to_short) {
                uint32_t* ids = reinterpret_cast<uint32_t*>([eng->b_frspec_ids contents]);
                uint32_t* f2s = reinterpret_cast<uint32_t*>([eng->b_frspec_full_to_short contents]);
                for (uint32_t i = 0U; i < V; ++i) f2s[i] = 0xFFFFFFFFU;
                const uint32_t tail_s = (V > kMtpShortlistTailStart) ? kMtpShortlistTailStart : V;
                const uint32_t n_tail = (V > tail_s) ? std::min<uint32_t>(V - tail_s, kFrSpecStaticRows / 4U) : 0U;
                const uint32_t n_static_head = kFrSpecStaticRows - n_tail; // 31,468 immutable top-BPE rows
                // Immutable static head [0 .. n_static_head - 1]
                for (uint32_t i = 0U; i < n_static_head; ++i) {
                    ids[i] = i;
                    f2s[i] = i;
                }
                // Immutable special tail [n_static_head .. kFrSpecStaticRows - 1] (NEVER evicted!)
                for (uint32_t i = 0U; i < n_tail; ++i) {
                    const uint32_t tid = tail_s + i;
                    ids[n_static_head + i] = tid;
                    f2s[tid]               = n_static_head + i;
                }
                // Dynamic context-token ring [kFrSpecStaticRows .. kFrSpecShortlistSize - 1] (1,024 slots):
                // seeded initially with the next 1,024 BPE frequency tokens so all 32,768 rows start valid & unique!
                for (uint32_t i = 0U; i < kFrSpecDynSlots; ++i) {
                    const uint32_t tid  = n_static_head + i;
                    const uint32_t slot = kFrSpecStaticRows + i;
                    ids[slot] = tid;
                    f2s[tid]  = slot;
                }
                eng->frspec_dyn_cursor = 0U;
            }
        } else {
            eng->frspec_shortlist_size  = 0U;
            eng->b_frspec_lm_head       = nil;
            eng->b_frspec_ids           = sqf_alloc_shared_prefaulted(dev, 4096U);
            eng->b_frspec_full_to_short = sqf_alloc_shared_prefaulted(dev, 4096U);
        }
        eng->ple_async_queue    = dispatch_queue_create("strata.sqf.ple_async", DISPATCH_QUEUE_SERIAL);
        eng->ple_async_group    = dispatch_group_create();

        const size_t sz_logits = static_cast<size_t>(kMaxSpecBatch) * V * sizeof(float);
        const size_t sz_ple_sh = static_cast<size_t>(kMaxPrefillChunk) * ple_dim * sizeof(float);
        eng->b_ple_emb_f32      = sqf_alloc_shared_prefaulted(dev, sz_ple_sh);
        eng->b_ple_emb_f32_alt  = sqf_alloc_shared_prefaulted(dev, sz_ple_sh);
        eng->b_inv_freq         = sqf_alloc_shared_prefaulted(dev, 64ULL * sizeof(float));
        eng->b_token_ids        = sqf_alloc_shared_prefaulted(dev, kMaxPrefillChunk * sizeof(uint32_t));
        eng->b_mtp_token_ids    = sqf_alloc_shared_prefaulted(dev, kMaxPrefillChunk * sizeof(uint32_t));
        eng->b_logits           = sqf_alloc_shared_prefaulted(dev, sz_logits);
        eng->b_next_tokens      = sqf_alloc_shared_prefaulted(dev, kMaxSpecBatch * sizeof(uint32_t));
        eng->b_sample_top_ids   = sqf_alloc_shared_prefaulted(dev, kMaxSpecBatch * 64ULL * sizeof(uint32_t));
        eng->b_sample_top_probs = sqf_alloc_shared_prefaulted(dev, kMaxSpecBatch * 64ULL * sizeof(float));
        eng->b_sample_keep_k    = sqf_alloc_shared_prefaulted(dev, kMaxSpecBatch * sizeof(uint32_t));
        eng->b_draft_top_ids    = sqf_alloc_shared_prefaulted(dev, kMaxSpecBatch * 64ULL * sizeof(uint32_t));
        eng->b_draft_top_probs  = sqf_alloc_shared_prefaulted(dev, kMaxSpecBatch * 64ULL * sizeof(float));
        eng->b_draft_keep_k     = sqf_alloc_shared_prefaulted(dev, kMaxSpecBatch * sizeof(uint32_t));
        eng->b_draft_pmax       = sqf_alloc_shared_prefaulted(dev, kMaxSpecBatch * sizeof(float));

        eng->b_block_loc_dummy  = sqf_alloc_shared_prefaulted(dev, 4096U);
        eng->b_probe_x          = sqf_alloc_shared_prefaulted(dev, static_cast<size_t>(full_slots) * D * sizeof(uint16_t));
        eng->b_pen_ids          = sqf_alloc_shared_prefaulted(dev, static_cast<size_t>(V) * sizeof(uint32_t));
        eng->b_pen_counts       = sqf_alloc_shared_prefaulted(dev, static_cast<size_t>(V) * sizeof(uint16_t));
        eng->pen_num_ids        = 0U;
        eng->pen_pos.assign(V, 0xFFFFFFFFU);
        eng->pen_hist.reserve(4096U);
        eng->pen_window_start   = 0U;
        eng->pen_window_end     = 0U;
        eng->b_block_loc        = eng->b_block_loc_dummy;
        eng->kv_stride_tokens   = eng->max_seq_alloc;
        eng->kv_paged           = false;

        eng->workspace_bytes = ws + sz_logits + 2ULL * sz_ple_sh + 16384ULL;

        if (!eng->b_conv_state || !eng->b_rec_state || !eng->b_ple_conv_state ||
            !eng->b_k_cache || !eng->b_v_cache || !eng->b_idx_raw_k || !eng->b_idx_block_k ||
            !eng->b_mtp_hc_prev || !eng->b_hc || !eng->b_mtp_hc || !eng->b_hc_norm_f32 ||
            !eng->b_hc_norm_f16 || !eng->b_hc_down_f32 || !eng->b_hc_u_f16 || !eng->b_hc_up_f32 ||
            !eng->b_hc_inj_f32 || !eng->b_x_in_f16 || !eng->b_x_in_f32 || !eng->b_block_y_f32 ||
            !eng->b_ple_emb_f16 || !eng->b_ple_gvn_f32 || !eng->b_qkv || !eng->b_qkv_conv ||
            !eng->b_z || !eng->b_a || !eng->b_b || !eng->b_dn_qk_hat || !eng->b_dn_gates ||
            !eng->b_dn_raw_out || !eng->b_k_raw || !eng->b_v_raw || !eng->b_q_rope ||
            !eng->b_attn_out || !eng->b_idx_qk_f32 || !eng->b_idx_q_rot || !eng->b_idx_blk_branch_b || !eng->b_idx_scores ||
            !eng->b_idx_sel_blocks || !eng->b_part_m || !eng->b_part_l || !eng->b_part_o ||
            !eng->b_router_logits || !eng->b_topk_idx || !eng->b_topk_w || !eng->b_moe_mid_f16 ||
            !eng->b_moe_down_part || !eng->b_moe_sorted_pairs || !eng->b_moe_work_tiles ||
            !eng->b_moe_num_tiles || !eng->b_moe_indirect_args || !eng->b_shared_gate_f32 || !eng->b_shared_up_f32 ||
            !eng->b_shared_mid_f16 || !eng->b_shared_out_f32 || !eng->b_shared_scalar ||
            !eng->b_sample_stage_v || !eng->b_sample_stage_i || !eng->b_sample_stage_lse ||
            !eng->b_frspec_ids || !eng->b_frspec_full_to_short || (build_frspec && !eng->b_frspec_lm_head) ||
            !eng->b_ple_emb_f32 || !eng->b_ple_emb_f32_alt || !eng->b_inv_freq || !eng->b_token_ids || !eng->b_mtp_token_ids ||
            !eng->b_logits || !eng->b_next_tokens || !eng->b_sample_top_ids ||
            !eng->b_sample_top_probs || !eng->b_sample_keep_k || !eng->b_draft_top_ids ||
            !eng->b_draft_top_probs || !eng->b_draft_keep_k || !eng->b_draft_pmax || !eng->b_block_loc_dummy || !eng->b_probe_x ||
            !eng->b_pen_ids || !eng->b_pen_counts ||
            (eng->cfg.has_mtp && !snap_slots_ok)) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] Failed to allocate GPU state or workspace buffers\n");
            dispatch_group_wait(compile_grp, DISPATCH_TIME_FOREVER);
            strata_qwen_flash_gpu_destroy(eng);
            return nullptr;
        }

        // Precompute Partial NeoX RoPE inverse frequencies (`rotary_dim = 64`, `theta = 10,000,000.0`)
        {
            float* inv_f = reinterpret_cast<float*>([eng->b_inv_freq contents]);
            const double theta = (eng->cfg.rope_theta > 0.0f) ? static_cast<double>(eng->cfg.rope_theta) : 10000000.0;
            const uint32_t rot = (eng->cfg.rotary_dim > 0U) ? eng->cfg.rotary_dim : 64U;
            const uint32_t h_r = rot / 2U;
            for (uint32_t i = 0U; i < h_r && i < 64U; ++i) {
                inv_f[i] = static_cast<float>(std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(rot)));
            }
        }

        // 5. Wire ALL Weight, KV/Indexer & Workspace Allocations into `MTLResidencySet`.
        //    Each swappable `{b_conv_state, b_rec_state, b_ple_conv_state}` triple carries its own paired
        //    `MTLResidencySet` (`rec_rset` / `ring_rec_rsets[s]`) so O(1) pointer swaps work safely across ALL slots!
        NSMutableArray<id<MTLBuffer>>* all_bufs = [NSMutableArray arrayWithObjects:
            eng->b_part0, eng->b_part1,
            eng->b_k_cache, eng->b_v_cache, eng->b_idx_raw_k, eng->b_idx_block_k, eng->b_mtp_hc_prev,
            eng->b_hc, eng->b_mtp_hc, eng->b_hc_norm_f32, eng->b_hc_norm_f16,
            eng->b_hc_down_f32, eng->b_hc_u_f16, eng->b_hc_up_f32, eng->b_hc_inj_f32,
            eng->b_x_in_f16, eng->b_x_in_f32, eng->b_block_y_f32,
            eng->b_ple_emb_f16, eng->b_ple_gvn_f32,
            eng->b_qkv, eng->b_qkv_conv, eng->b_z, eng->b_a, eng->b_b,
            eng->b_dn_qk_hat, eng->b_dn_gates, eng->b_dn_raw_out,
            eng->b_k_raw, eng->b_v_raw, eng->b_q_rope, eng->b_attn_out,
            eng->b_idx_qk_f32, eng->b_idx_q_rot, eng->b_idx_blk_branch_b, eng->b_idx_scores, eng->b_idx_sel_blocks,
            eng->b_part_m, eng->b_part_l, eng->b_part_o,
            eng->b_router_logits, eng->b_topk_idx, eng->b_topk_w,
            eng->b_moe_mid_f16, eng->b_moe_down_part, eng->b_moe_sorted_pairs,
            eng->b_moe_work_tiles, eng->b_moe_num_tiles, eng->b_moe_indirect_args,
            eng->b_shared_gate_f32, eng->b_shared_up_f32, eng->b_shared_mid_f16,
            eng->b_shared_out_f32, eng->b_shared_scalar,
            eng->b_sample_stage_v, eng->b_sample_stage_i, eng->b_sample_stage_lse,
            eng->b_frspec_ids, eng->b_frspec_full_to_short,
            eng->b_ple_emb_f32, eng->b_ple_emb_f32_alt, eng->b_inv_freq, eng->b_token_ids, eng->b_mtp_token_ids,
            eng->b_logits, eng->b_next_tokens, eng->b_sample_top_ids,
            eng->b_sample_top_probs, eng->b_sample_keep_k,
            eng->b_draft_top_ids, eng->b_draft_top_probs, eng->b_draft_keep_k, eng->b_draft_pmax,
            eng->b_block_loc_dummy, eng->b_probe_x, eng->b_pen_ids, eng->b_pen_counts,
            nil];
        if (eng->b_frspec_lm_head) [all_bufs addObject:eng->b_frspec_lm_head];

        eng->residency_set = sqf_create_residency_set(dev, eng->queue, all_bufs);
        id<MTLResidencySet> def_rec_rset = sqf_create_residency_set(dev, eng->queue, @[
            eng->b_conv_state, eng->b_rec_state, eng->b_ple_conv_state
        ]);
        for (uint32_t s = 0U; s < kMaxSpecSnapSlots; ++s) {
            if (eng->b_rec_slots[s] && eng->b_conv_slots[s] && eng->b_ple_conv_slots[s]) {
                eng->ring_rec_rsets[s] = sqf_create_residency_set(dev, eng->queue, @[
                    eng->b_conv_slots[s], eng->b_rec_slots[s], eng->b_ple_conv_slots[s]
                ]);
            }
        }

        // 6. Join concurrent shader compilation & zero-init states on GPU
        dispatch_group_wait(compile_grp, DISPATCH_TIME_FOREVER);
        if (!shader_ok) {
            sqf_detach_residency_set(eng->queue, def_rec_rset);
            strata_qwen_flash_gpu_destroy(eng);
            return nullptr;
        }

        // Built-in default context slot wraps the engine-created linear KV / state buffers (Tier 1 binding)
        {
            auto* ds = new StrataQwenFlashContextSlot();
            ds->eng              = eng;
            ds->is_default       = true;
            ds->b_conv_state     = eng->b_conv_state;
            ds->b_rec_state      = eng->b_rec_state;
            ds->b_ple_conv_state = eng->b_ple_conv_state;
            ds->rec_rset         = def_rec_rset;
            ds->b_k_cache        = eng->b_k_cache;
            ds->b_v_cache        = eng->b_v_cache;
            ds->b_idx_raw_k      = eng->b_idx_raw_k;
            ds->b_idx_block_k    = eng->b_idx_block_k;
            ds->b_mtp_hc_prev    = eng->b_mtp_hc_prev;
            ds->b_block_loc      = eng->b_block_loc_dummy;
            ds->kv_stride_tokens = eng->max_seq_alloc;
            ds->kv_layer_stride  = eng->kv_layer_stride;
            ds->n_layers         = full_slots;
            ds->paged_layers     = full_slots;
            ds->max_blocks       = eng->max_blocks_alloc;
            ds->bytes            = eng->conv_state_bytes + eng->rec_state_bytes + eng->ple_state_bytes + eng->kv_cache_bytes_allocated;
            eng->default_slot    = ds;
            eng->bound_slot      = ds;
        }

        // Compact the 32K SLC-pinned FR-Spec `lm_head` once on GPU (< 0.5 ms!) before initial reset
        if (eng->b_frspec_lm_head && eng->frspec_shortlist_size > 0U && eng->pso_pack_frspec_q4) {
            id<MTLCommandBuffer> cb_pk          = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_pk = [cb_pk computeCommandEncoder];
            SqfPackFrspecParamsHost pp{eng->frspec_shortlist_size, eng->frspec_lm_head_desc.cols, V, 0U};
            [enc_pk setComputePipelineState:eng->pso_pack_frspec_q4];
            [enc_pk setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(eng->hdr1.lm_head.weight_off) atIndex:0];
            [enc_pk setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(eng->hdr1.lm_head.scales_off) atIndex:1];
            [enc_pk setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(eng->hdr1.lm_head.biases_off) atIndex:2];
            [enc_pk setBuffer:eng->b_frspec_ids offset:0 atIndex:3];
            [enc_pk setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(eng->frspec_lm_head_desc.weight_off) atIndex:4];
            [enc_pk setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(eng->frspec_lm_head_desc.scales_off) atIndex:5];
            [enc_pk setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(eng->frspec_lm_head_desc.biases_off) atIndex:6];
            [enc_pk setBytes:&pp length:sizeof(pp) atIndex:7];
            [enc_pk dispatchThreadgroups:MTLSizeMake(eng->frspec_shortlist_size, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc_pk endEncoding];
            [cb_pk commit];
            [cb_pk waitUntilCompleted];
        }

        strata_qwen_flash_gpu_reset(eng);

        const auto t_end  = std::chrono::steady_clock::now();
        eng->load_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        return eng;
    }
}

void strata_qwen_flash_gpu_destroy(StrataQwenFlashGpuEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        (void)sqf_drain_pending_fix(eng);
        if (eng->bound_slot) sqf_paged_join_probe(eng->bound_slot);
        if (eng->default_slot) {
            sqf_detach_residency_set(eng->queue, eng->default_slot->rec_rset);
            eng->default_slot->rec_rset = nil;
            delete eng->default_slot;
            eng->default_slot = nullptr;
            eng->bound_slot   = nullptr;
        }
        for (uint32_t s = 0U; s < kMaxSpecSnapSlots; ++s) {
            sqf_detach_residency_set(eng->queue, eng->ring_rec_rsets[s]);
            eng->ring_rec_rsets[s] = nil;
        }
        if (eng->ane) {
            delete eng->ane;
            eng->ane = nullptr;
        }
        sqf_detach_residency_set(eng->queue, eng->residency_set);
        eng->residency_set = nil;
        delete eng;
    }
}

void strata_qwen_flash_gpu_penalties_reset(StrataQwenFlashGpuEngine* eng) {
    if (!eng || !eng->b_pen_ids || !eng->b_pen_counts) return;
    sqf_penalties_clear_active(eng);
    eng->pen_hist.clear();
    eng->pen_window_start = 0U;
    eng->pen_window_end   = 0U;
}

void strata_qwen_flash_gpu_penalties_record(
    StrataQwenFlashGpuEngine* eng,
    const uint32_t*           tokens,
    uint32_t                  num_tokens
) {
    if (!eng || !tokens || num_tokens == 0U || !eng->b_pen_ids || !eng->b_pen_counts) return;
    const uint32_t V = eng->cfg.vocab_size;
    for (uint32_t i = 0U; i < num_tokens; ++i) {
        const uint32_t t = tokens[i];
        if (t >= V) continue;
        eng->pen_hist.push_back(t);
    }
}

void strata_qwen_flash_gpu_reset(StrataQwenFlashGpuEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        (void)sqf_drain_pending_fix(eng);
        eng->seq_pos                = 0U;
        eng->mtp_seq_pos            = 0U;
        eng->ple_prev_tokens[0]     = eng->cfg.ple_eos_token_id;
        eng->ple_prev_tokens[1]     = eng->cfg.ple_eos_token_id;
        eng->sample_step            = 0ULL;
        eng->shortlist_miss_mask    = 0U;
        eng->shortlist_obs_count    = 0U;
        eng->shortlist_bypass_steps = 0U;
        eng->frspec_widened         = false;
        eng->frspec_dyn_cursor      = 0U;
        eng->spec_ctl.reset();
        strata_qwen_flash_gpu_penalties_reset(eng);
        sqf_ngram_reset(eng->bound_slot);
        if (eng->bound_slot && eng->bound_slot->paged) {
            sqf_paged_reset_tables(eng->bound_slot);
        }

        id<MTLCommandBuffer> cb          = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        sqf_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_conv_state, [eng->b_conv_state length]);
        sqf_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_rec_state, [eng->b_rec_state length]);
        sqf_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_ple_conv_state, [eng->b_ple_conv_state length]);
        sqf_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_mtp_hc_prev, [eng->b_mtp_hc_prev length]);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

uint32_t strata_qwen_flash_gpu_seq_pos(const StrataQwenFlashGpuEngine* eng) {
    return eng ? eng->seq_pos : 0U;
}

void strata_qwen_flash_gpu_get_info(
    const StrataQwenFlashGpuEngine* eng,
    StrataQwenFlashGpuInfo*         info_out
) {
    if (!info_out) return;
    std::memset(info_out, 0, sizeof(*info_out));
    if (!eng) return;

    std::memcpy(info_out->device_name, eng->device_name, sizeof(info_out->device_name));
    info_out->zero_copy_mmap           = eng->zero_copy_mmap;
    info_out->load_time_ms             = eng->load_time_ms;
    info_out->max_seq_len              = eng->max_seq_len;
    info_out->uma_part0_mapped_bytes   = eng->part0_mapped_bytes;
    info_out->uma_part1_mapped_bytes   = eng->part1_mapped_bytes;
    info_out->total_uma_mapped_bytes   = eng->part0_mapped_bytes + eng->part1_mapped_bytes;
    info_out->ple_nvme_file_bytes      = eng->bundle ? eng->bundle->ple_file_bytes : 0ULL;
    info_out->ple_lfu_cache_bytes      = eng->bundle ? static_cast<uint64_t>(eng->bundle->ple_cache.arena_bytes) : 0ULL;
    info_out->active_bytes_per_token   = eng->hdr0.active_bytes_per_tok;
    info_out->mtp_active_bytes_per_draft = eng->hdr0.mtp_active_bytes_per_tok;
    info_out->deltanet_state_bytes     = eng->conv_state_bytes + eng->rec_state_bytes;
    info_out->ple_state_bytes          = eng->ple_state_bytes;
    info_out->kv_cache_bytes_allocated = eng->kv_cache_bytes_allocated;
    info_out->kv_cache_bytes_per_token = eng->kv_cache_bytes_per_token;
    info_out->workspace_bytes          = eng->workspace_bytes;
    info_out->int8_prefill_enabled     = eng->int8_prefill_enabled ? 1U : 0U;
    info_out->adaptive_k_enabled       = eng->adaptive_k_enabled ? 1U : 0U;
    info_out->frspec_enabled           = eng->frspec_enabled ? 1U : 0U;
    info_out->frspec_shortlist_size    = eng->frspec_shortlist_size;
    info_out->metal_cache_hit          = eng->metal_cache_hit ? 1U : 0U;
    info_out->pso_load_ms              = eng->pso_load_ms;
}

void strata_qwen_flash_gpu_get_counters(
    const StrataQwenFlashGpuEngine* eng,
    StrataQwenFlashPerfCounters*    counters_out
) {
    if (!counters_out) return;
    std::memset(counters_out, 0, sizeof(*counters_out));
    if (!eng) return;
    *counters_out = eng->counters;
}

void strata_qwen_flash_gpu_reset_counters(StrataQwenFlashGpuEngine* eng) {
    if (!eng) return;
    std::memset(&eng->counters, 0, sizeof(eng->counters));
    if (eng->bundle) {
        strata_qwen_flash_ple_cache_reset_stats(&eng->bundle->ple_cache);
    }
}

void strata_qwen_flash_gpu_set_prefill_progress_cb(
    StrataQwenFlashGpuEngine*        eng,
    StrataQwenFlashPrefillProgressFn cb,
    void*                            user_ctx
) {
    if (!eng) return;
    eng->progress_cb       = cb;
    eng->progress_user_ctx = user_ctx;
}

void strata_qwen_flash_gpu_set_spec_tree(StrataQwenFlashGpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->spec_tree_enabled = enable;
}

bool strata_qwen_flash_gpu_get_spec_tree(const StrataQwenFlashGpuEngine* eng) {
    return eng ? eng->spec_tree_enabled : false;
}

void strata_qwen_flash_gpu_set_mtp_kv_sync(StrataQwenFlashGpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->mtp_kv_sync_enabled = enable;
}

bool strata_qwen_flash_gpu_get_mtp_kv_sync(const StrataQwenFlashGpuEngine* eng) {
    return eng ? eng->mtp_kv_sync_enabled : false;
}

void strata_qwen_flash_gpu_set_mtp_shortlist(StrataQwenFlashGpuEngine* eng, uint32_t v_short) {
    if (!eng) return;
    eng->mtp_shortlist_v        = v_short;
    eng->frspec_enabled         = (v_short > 0U);
    eng->frspec_widened         = false;
    eng->shortlist_miss_mask    = 0U;
    eng->shortlist_obs_count    = 0U;
    eng->shortlist_bypass_steps = 0U;
}

uint32_t strata_qwen_flash_gpu_get_mtp_shortlist(const StrataQwenFlashGpuEngine* eng) {
    return eng ? (eng->frspec_enabled ? eng->frspec_shortlist_size : 0U) : 0U;
}

void strata_qwen_flash_gpu_set_ngram_draft(StrataQwenFlashGpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->ngram_draft_enabled = enable;
}

bool strata_qwen_flash_gpu_get_ngram_draft(const StrataQwenFlashGpuEngine* eng) {
    return eng ? eng->ngram_draft_enabled : false;
}

void strata_qwen_flash_gpu_set_int8_prefill(StrataQwenFlashGpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->int8_prefill_enabled = enable;
}

void strata_qwen_flash_gpu_set_adaptive_k(StrataQwenFlashGpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->adaptive_k_enabled = enable;
    eng->spec_ctl.reset();
}

void strata_qwen_flash_gpu_set_frspec(StrataQwenFlashGpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->frspec_enabled = enable;
    if (enable) {
        eng->frspec_widened         = false;
        eng->shortlist_miss_mask    = 0U;
        eng->shortlist_obs_count    = 0U;
        eng->shortlist_bypass_steps = 0U;
    }
}

int strata_qwen_flash_gpu_decode_step(
    StrataQwenFlashGpuEngine*            eng,
    uint32_t                             token_id,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t*                            next_token_out,
    float*                               logits_out
) {
    if (!eng) return -1;
    if (eng->seq_pos >= eng->max_seq_len) return -2;

    @autoreleasepool {
        if (!sqf_drain_pending_fix(eng)) return -4;
        const uint32_t D          = eng->cfg.hidden_size;
        const uint32_t H          = eng->cfg.hc_count;
        const uint32_t hc_dim     = H * D;
        const uint32_t V          = eng->cfg.vocab_size;
        const uint32_t num_layers = eng->cfg.num_hidden_layers;
        const uint32_t cur_pos    = eng->seq_pos;
        uint32_t       split_ple  = (eng->cfg.has_ple && eng->cfg.ple_layer_idx > 0U && eng->cfg.ple_layer_idx < num_layers)
                                        ? eng->cfg.ple_layer_idx : 0U;

        *reinterpret_cast<uint32_t*>([eng->b_token_ids contents]) = token_id;
        if (eng->kv_paged) sqf_paged_step_begin(eng, cur_pos);

        // Single-CB fast-path when all 16 PLE rows hit LFU cache (< 2 us memcpy, saves 1 MTLCommandBuffer boundary!)
        __block int ple_async_rc = 0;
        bool ple_in_flight = false;
        if (split_ple > 0U && sqf_ple_all_cached(eng, &token_id, 1U)) {
            if (sqf_execute_ple_lookup_host(eng, &token_id, 1U, nullptr) != 0) return -3;
            split_ple = 0U;
        } else if (split_ple > 0U) {
            // Overlap NVMe PLE lookup with cb0 encode + GPU Layer 0 + cb_final encode!
            const uint32_t tok_copy = token_id;
            ple_in_flight = true;
            dispatch_group_async(eng->ple_async_group, eng->ple_async_queue, ^{
                ple_async_rc = sqf_execute_ple_lookup_host(eng, &tok_copy, 1U, nullptr);
            });
        } else if (eng->cfg.has_ple) {
            if (sqf_execute_ple_lookup_host(eng, &token_id, 1U, nullptr) != 0) return -3;
        }

        uint32_t lin_slot  = 0U;
        uint32_t full_slot = 0U;

        id<MTLCommandBuffer> cb0          = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc0 = [cb0 computeCommandEncoder];

        // Embed + tile 4x -> `b_hc [1, 4, D]`
        {
            SqfEmbedParamsHost ep{1U, D, V, H};
            [enc0 setComputePipelineState:eng->pso_embed_q4];
            [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.weight_off) atIndex:0];
            [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.scales_off) atIndex:1];
            [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.biases_off) atIndex:2];
            [enc0 setBuffer:eng->b_token_ids offset:0 atIndex:3];
            [enc0 setBuffer:eng->b_hc offset:0 atIndex:4];
            [enc0 setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc0 dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
        }

        const uint32_t phase0_end = (split_ple > 0U) ? split_ple : num_layers;
        for (uint32_t l = 0U; l < phase0_end; ++l) {
            const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
            sqf_encode_decoder_layer(enc0, cb0, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, 1U, 1U, 0U, true);
            if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
            else ++full_slot;
        }

        id<MTLCommandBuffer> cb_final          = cb0;
        id<MTLComputeCommandEncoder> enc_final = enc0;

        // Overlap NVMe `F_NOCACHE` `pread` PLE lookup + `cb_final` encoding while GPU executes `Embed + Layer 0`!
        if (split_ple > 0U) {
            [enc0 endEncoding];
            [cb0 commit];

            cb_final  = [eng->queue commandBuffer];
            enc_final = [cb_final computeCommandEncoder];

            for (uint32_t l = split_ple; l < num_layers; ++l) {
                const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                if (ple_in_flight && eng->kv_paged && ld.layer_type == STRATA_QWEN_FLASH_LAYER_FULL_ATTN) {
                    const uint32_t cr = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
                    const uint32_t bt = std::max<uint32_t>(1U, eng->cfg.indexer_budget / cr);
                    const bool     sp = ((cur_pos + 1U) / cr) > bt;
                    if (sqf_paged_layer_needs_sync(eng, full_slot, cur_pos, 1U, cr, sp, 0U)) {
                        dispatch_group_wait(eng->ple_async_group, DISPATCH_TIME_FOREVER);
                        ple_in_flight = false;
                        if (ple_async_rc != 0) {
                            [enc_final endEncoding];
                            [cb0 waitUntilCompleted];
                            sqf_paged_join_probe(eng->bound_slot);
                            return -3;
                        }
                    }
                }
                sqf_encode_decoder_layer(enc_final, cb_final, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, 1U, 1U, 0U, true);
                if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                else ++full_slot;
            }
        }

        // Save pre-mixer 4-stream state `b_mtp_hc_prev = b_hc[0]` for MTP!
        if (eng->cfg.has_mtp) {
            sqf_encode_copy_buffer(enc_final, eng->pso_copy_u32,
                                   eng->b_hc, 0, eng->b_mtp_hc_prev, 0,
                                   static_cast<size_t>(hc_dim) * sizeof(float));
        }

        // Final Mixer + Untied LM Head + Optional Logit Penalties + GPU Sampler
        sqf_encode_mixer(enc_final, eng, eng->b_part1, eng->hdr1.final_mixer, eng->b_hc, 0, 1U, 1U, true);
        sqf_encode_q4_linear(enc_final, eng, eng->b_part1, eng->hdr1.lm_head,
                             eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                             1U, 1U, kOutF32, true);
        sqf_encode_logit_penalties(enc_final, eng, sampling, 1U, eng->b_token_ids, 0U, 0U);
        eng->sample_step = static_cast<uint64_t>(cur_pos + 1U) << 8U;
        sqf_encode_sample_batch(enc_final, eng, sampling, 1U, eng->b_next_tokens, 0, nil, 0, nil, 0, nil, 0);

        [enc_final endEncoding];
        if (ple_in_flight) {
            dispatch_group_wait(eng->ple_async_group, DISPATCH_TIME_FOREVER);
            ple_in_flight = false;
            if (ple_async_rc != 0) {
                [cb0 waitUntilCompleted];
                return -3;
            }
        }
        [cb_final commit];
        [cb_final waitUntilCompleted];
        if ([cb0 status] == MTLCommandBufferStatusError || [cb_final status] == MTLCommandBufferStatusError) {
            NSError* err = ([cb0 status] == MTLCommandBufferStatusError) ? [cb0 error] : [cb_final error];
            std::fprintf(stderr, "[strata_qwen_flash_gpu] GPU decode command buffer error: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return -4;
        }
        if (eng->kv_paged && sqf_take_paged_error(eng)) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] paged KV recall/evict failure during decode at pos=%u\n", cur_pos);
            return -5;
        }

        eng->seq_pos += 1U;
        sqf_ngram_append(eng->bound_slot, &token_id, 1U);
        eng->counters.decode_steps += 1ULL;
        const uint32_t comp_rat  = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
        const uint64_t num_comp  = static_cast<uint64_t>((cur_pos + 1U) / comp_rat);
        const uint64_t blk_topk  = std::max<uint64_t>(1ULL, eng->cfg.indexer_budget / comp_rat);
        const uint64_t idx_reads = (num_comp > blk_topk)
            ? (num_comp * eng->cfg.indexer_head_dim * sizeof(uint16_t) * eng->cfg.num_full_attn_layers)
            : 0ULL;
        const uint64_t eff_kv = std::min<uint64_t>(cur_pos + 1U, eng->cfg.indexer_budget + 4U);
        eng->counters.physical_dram_bytes_read +=
            eng->hdr0.active_bytes_per_tok +
            2ULL * eng->rec_state_bytes +
            eff_kv * eng->kv_cache_bytes_per_token +
            idx_reads;

        const uint32_t sampled_tok = *reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents]);
        strata_qwen_flash_gpu_penalties_record(eng, &sampled_tok, 1U);
        if (next_token_out) {
            *next_token_out = sampled_tok;
        }
        if (logits_out) {
            std::memcpy(logits_out, [eng->b_logits contents], static_cast<size_t>(V) * sizeof(float));
        }
        return 0;
    }
}

} // extern "C"

namespace {

int sqf_prefill_impl(
    StrataQwenFlashGpuEngine*            eng,
    const uint32_t*                      token_ids,
    uint32_t                             num_tokens,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t*                            next_token_out,
    float*                               last_logits_out,
    bool                                 want_head
) {
    if (!eng || !token_ids || num_tokens == 0U) return -1;
    if (eng->seq_pos + num_tokens > eng->max_seq_len) return -2;

    @autoreleasepool {
        if (!sqf_drain_pending_fix(eng)) return -4;
        if (want_head) {
            strata_qwen_flash_gpu_penalties_reset(eng);
        }
        const uint32_t D          = eng->cfg.hidden_size;
        const uint32_t H          = eng->cfg.hc_count;
        const uint32_t hc_dim     = H * D;
        const uint32_t V          = eng->cfg.vocab_size;
        const uint32_t num_layers = eng->cfg.num_hidden_layers;
        const uint32_t split_ple  = (eng->cfg.has_ple && eng->cfg.ple_layer_idx > 0U && eng->cfg.ple_layer_idx < num_layers)
                                        ? eng->cfg.ple_layer_idx : 0U;

        // Select prefill chunk size (default `kMaxPrefillChunk = 1024`, optional `STRATA_SQF_CHUNK` A/B override,
        // and automatic tiny-pool paged-KV safety cap so `block_topk + chunk_write_blocks + 16 <= pool_blocks`!)
        uint32_t chunk_cap = kMaxPrefillChunk;
        if (const char* env_ch = std::getenv("STRATA_SQF_CHUNK")) {
            const int v = std::atoi(env_ch);
            if (v >= 64 && v <= static_cast<int>(kMaxPrefillChunk)) {
                chunk_cap = static_cast<uint32_t>(v) & ~63U;
            }
        }
        if (eng->kv_paged && eng->bound_slot) {
            const uint32_t comp_ratio = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
            const uint32_t block_topk = std::max<uint32_t>(1U, eng->cfg.indexer_budget / comp_ratio);
            const uint32_t pool_blks  = eng->bound_slot->pool_blocks;
            if (pool_blks > block_topk + 16U) {
                const uint32_t safe_toks = ((pool_blks - block_topk - 16U) * comp_ratio) & ~63U;
                if (safe_toks >= 64U && safe_toks < chunk_cap) chunk_cap = safe_toks;
            } else {
                chunk_cap = std::min<uint32_t>(chunk_cap, 256U);
            }
        }

        // Phase D2: Double-Buffered Async Prefill PLE Pipeline (hides 100% of NVMe PLE reads for chunks >= 1
        // behind the preceding chunk's GPU compute window, eliminating cb0/cb_final splits!)
        const bool can_async_ple_pipe =
            eng->cfg.has_ple && (eng->b_ple_emb_f32_alt != nil) && (num_tokens > chunk_cap);
        bool        next_ple_in_flight = false;
        int         next_ple_rc        = 0;
        uint32_t    next_ple_prev_out[2] = {eng->ple_prev_tokens[0], eng->ple_prev_tokens[1]};
        auto join_next_ple = [&]() -> bool {
            if (!next_ple_in_flight) return true;
            dispatch_group_wait(eng->ple_async_group, DISPATCH_TIME_FOREVER);
            next_ple_in_flight = false;
            return next_ple_rc == 0;
        };

        for (uint32_t chunk_off = 0U; chunk_off < num_tokens; chunk_off += chunk_cap) {
            const uint32_t M             = std::min<uint32_t>(chunk_cap, num_tokens - chunk_off);
            const bool     is_last_chunk = (chunk_off + M == num_tokens);
            const uint32_t cur_pos       = eng->seq_pos;
            const bool     use_gemv      = (M <= kMaxSpecBatch);
            const uint32_t M_padded      = (!use_gemv && eng->has_nax) ? ((M + 63U) & ~63U) : M;

            // If chunk > 0 had its PLE pre-fetched asynchronously during the previous chunk, swap it in now!
            uint32_t prof_idx = 0U;
            if (eng->prof_enabled && (M == chunk_cap)) prof_idx = ++eng->prof_prefill_chunk_cnt;
            const bool do_prof_chunk = (prof_idx == 2U || prof_idx == 3U);
            if (do_prof_chunk) {
                std::memset(eng->prof_ms, 0, sizeof(eng->prof_ms));
                eng->prof_active = true;
            }
            const bool ple_preloaded = (chunk_off > 0U) && can_async_ple_pipe;
            if (ple_preloaded) {
                if (!join_next_ple()) return -3;
                std::swap(eng->b_ple_emb_f32, eng->b_ple_emb_f32_alt);
                eng->ple_prev_tokens[0] = next_ple_prev_out[0];
                eng->ple_prev_tokens[1] = next_ple_prev_out[1];
            }

            std::memcpy([eng->b_token_ids contents], token_ids + chunk_off, static_cast<size_t>(M) * sizeof(uint32_t));
            if (eng->kv_paged) sqf_paged_step_begin(eng, cur_pos + M - 1U);

            const uint32_t eff_split_ple = ple_preloaded ? 0U : split_ple;
            if (!ple_preloaded && eff_split_ple == 0U && eng->cfg.has_ple) {
                if (sqf_execute_ple_lookup_host(eng, token_ids + chunk_off, M, nullptr) != 0) return -3;
            }

            auto launch_next_chunk_ple = [&]() {
                if (!can_async_ple_pipe || is_last_chunk || next_ple_in_flight) return;
                const uint32_t nxt_off = chunk_off + M;
                const uint32_t nxt_M   = std::min<uint32_t>(chunk_cap, num_tokens - nxt_off);
                uint32_t in_p0 = eng->ple_prev_tokens[0];
                uint32_t in_p1 = eng->ple_prev_tokens[1];
                uint32_t* out_prev_ptr = next_ple_prev_out;
                const uint32_t* nxt_toks = token_ids + nxt_off;
                id<MTLBuffer> alt_buf = eng->b_ple_emb_f32_alt;
                int* rc_ptr = &next_ple_rc;
                next_ple_rc        = 0;
                next_ple_in_flight = true;
                dispatch_group_async(eng->ple_async_group, eng->ple_async_queue, ^{
                    uint32_t pin[2] = {in_p0, in_p1};
                    *rc_ptr = sqf_execute_ple_lookup_host_buf(
                        eng, alt_buf, nxt_toks, nxt_M, pin, out_prev_ptr, nullptr);
                });
            };

            uint32_t lin_slot  = 0U;
            uint32_t full_slot = 0U;

            id<MTLCommandBuffer> cb0          = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc0 = [cb0 computeCommandEncoder];

            // If `cur_pos > 0` and MTP has a pending cross-chunk `(b_mtp_hc_prev, token_ids[chunk_off])`, prime MTP slot `cur_pos - 1`!
            if (eng->cfg.has_mtp && eng->hdr1.mtp.enabled && cur_pos > 0U && eng->mtp_seq_pos == cur_pos - 1U) {
                sqf_encode_mtp_pass(enc0, cb0, eng, eng->b_mtp_hc_prev, 0,
                                    eng->b_token_ids, 0, cur_pos - 1U, 1U, false, true);
                eng->mtp_seq_pos = cur_pos;
            }

            // 1. Batched Token Embedding Lookup + 4-Stream Tiling -> `b_hc [M, 4, D]`
            {
                SqfEmbedParamsHost ep{M, D, V, H};
                [enc0 setComputePipelineState:eng->pso_embed_q4];
                [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.weight_off) atIndex:0];
                [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.scales_off) atIndex:1];
                [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.biases_off) atIndex:2];
                [enc0 setBuffer:eng->b_token_ids offset:0 atIndex:3];
                [enc0 setBuffer:eng->b_hc offset:0 atIndex:4];
                [enc0 setBytes:&ep length:sizeof(ep) atIndex:5];
                [enc0 dispatchThreadgroups:MTLSizeMake(M, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
            }

            // Split Part 0 vs Part 1 across two back-to-back CBs so no single CB exceeds ~0.5 s at 256K!
            const uint32_t part0_end = (eng->cfg.split_layer_idx > eff_split_ple && eng->cfg.split_layer_idx < num_layers)
                                           ? eng->cfg.split_layer_idx
                                           : num_layers;
            const uint32_t phase0_end = (eff_split_ple > 0U) ? eff_split_ple : part0_end;
            for (uint32_t l = 0U; l < phase0_end; ++l) {
                const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                sqf_encode_decoder_layer(enc0, cb0, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, M, M_padded, 0U, use_gemv);
                if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                else ++full_slot;
            }

            id<MTLCommandBuffer> cb_mid            = nil;
            id<MTLCommandBuffer> cb_final          = cb0;
            id<MTLComputeCommandEncoder> enc_final = enc0;

            if (eff_split_ple > 0U) {
                [enc0 endEncoding];
                [cb0 commit];

                if (sqf_execute_ple_lookup_host(eng, token_ids + chunk_off, M, nullptr) != 0) {
                    [cb0 waitUntilCompleted];
                    return -3;
                }
                // Current chunk's `ple_prev_tokens` is now updated; immediately prefetch next chunk's PLE async!
                launch_next_chunk_ple();

                cb_final  = [eng->queue commandBuffer];
                enc_final = [cb_final computeCommandEncoder];

                for (uint32_t l = eff_split_ple; l < part0_end; ++l) {
                    const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                    sqf_encode_decoder_layer(enc_final, cb_final, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, M, M_padded, 0U, use_gemv);
                    if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                    else ++full_slot;
                }

                if (part0_end < num_layers) {
                    [enc_final endEncoding];
                    [cb_final commit];
                    cb_mid    = cb_final;
                    cb_final  = [eng->queue commandBuffer];
                    enc_final = [cb_final computeCommandEncoder];

                    for (uint32_t l = part0_end; l < num_layers; ++l) {
                        const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                        sqf_encode_decoder_layer(enc_final, cb_final, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, M, M_padded, 0U, use_gemv);
                        if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                        else ++full_slot;
                    }
                }
            } else {
                // Chunk PLE already resident (`ple_preloaded` or `split_ple == 0`): launch next chunk's PLE NOW
                // so it overlaps with ALL 48 layers of GPU compute for the current chunk!
                launch_next_chunk_ple();

                if (part0_end < num_layers) {
                    [enc_final endEncoding];
                    [cb_final commit];
                    cb_mid    = cb_final;
                    cb_final  = [eng->queue commandBuffer];
                    enc_final = [cb_final computeCommandEncoder];

                    for (uint32_t l = part0_end; l < num_layers; ++l) {
                        const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                        sqf_encode_decoder_layer(enc_final, cb_final, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, M, M_padded, 0U, use_gemv);
                        if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                        else ++full_slot;
                    }
                }
            }

            // Synchronize MTP KV & Indexer cache over intra-chunk positions `cur_pos .. cur_pos + M - 2`
            // and save `b_mtp_hc_prev = b_hc[M - 1]`!
            if (eng->cfg.has_mtp && eng->hdr1.mtp.enabled) {
                if (M > 1U && eng->mtp_seq_pos == cur_pos) {
                    std::memcpy([eng->b_mtp_token_ids contents], token_ids + chunk_off + 1U,
                                static_cast<size_t>(M - 1U) * sizeof(uint32_t));
                    sqf_encode_mtp_pass(enc_final, cb_final, eng, eng->b_hc, 0,
                                        eng->b_mtp_token_ids, 0, cur_pos, M - 1U, false, (M - 1U <= kMaxSpecBatch));
                    eng->mtp_seq_pos = cur_pos + M - 1U;
                }
                const NSUInteger last_hc_off = static_cast<NSUInteger>(M - 1U) * hc_dim * sizeof(float);
                sqf_encode_copy_buffer(enc_final, eng->pso_copy_u32,
                                       eng->b_hc, last_hc_off, eng->b_mtp_hc_prev, 0,
                                       static_cast<size_t>(hc_dim) * sizeof(float));
            }

            // Self-heal FR-Spec 32K SLC shortlist dynamic slots (`31744..32767`) with all out-of-shortlist prompt tokens!
            if (eng->frspec_shortlist_size == kFrSpecShortlistSize) {
                sqf_frspec_stage_and_encode_patches(enc_final, eng, token_ids + chunk_off, M, /*record_misses=*/false);
            }

            // Final chunk: mix `b_hc[M - 1]` -> `lm_head` -> Sampler!
            if (is_last_chunk && want_head) {
                const NSUInteger last_hc_off = static_cast<NSUInteger>(M - 1U) * hc_dim * sizeof(float);
                sqf_encode_mixer(enc_final, eng, eng->b_part1, eng->hdr1.final_mixer,
                                 eng->b_hc, last_hc_off, 1U, 1U, true);
                sqf_encode_q4_linear(enc_final, eng, eng->b_part1, eng->hdr1.lm_head,
                                     eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                                     1U, 1U, kOutF32, true);
                eng->sample_step = static_cast<uint64_t>(cur_pos + M) << 8U;
                sqf_encode_sample_batch(enc_final, eng, sampling, 1U, eng->b_next_tokens, 0, nil, 0, nil, 0, nil, 0);
            }

            [enc_final endEncoding];
            [cb_final commit];
            [cb_final waitUntilCompleted];
            if (do_prof_chunk) {
                eng->prof_active = false;
                double tot_gpu = 0.0;
                for (int b = 0; b < 11; ++b) tot_gpu += eng->prof_ms[b];
                std::fprintf(stderr,
                    "\n[PROF PREFILL M=%u pos=%u] TotalGPU=%.2f ms (%.1f tok/s eq)\n"
                    "  HC_attn+PLE=%.2f | LinQKVZ=%.2f | DeltaRec=%.2f | LinOut=%.2f\n"
                    "  QSA_QKV+Idx=%.2f | QSA_Attn+O=%.2f | HC_mlp=%.2f | Router+Sh=%.2f\n"
                    "  MoE_GU=%.2f | MoE_DN=%.2f | MoE_Comb=%.2f\n",
                    M, cur_pos, tot_gpu, M / (tot_gpu / 1000.0),
                    eng->prof_ms[0], eng->prof_ms[1], eng->prof_ms[2], eng->prof_ms[3],
                    eng->prof_ms[4], eng->prof_ms[5], eng->prof_ms[6], eng->prof_ms[7],
                    eng->prof_ms[8], eng->prof_ms[9], eng->prof_ms[10]);
            }
            if ([cb0 status] == MTLCommandBufferStatusError ||
                (cb_mid && [cb_mid status] == MTLCommandBufferStatusError) ||
                [cb_final status] == MTLCommandBufferStatusError) {
                (void)join_next_ple();
                NSError* err = ([cb0 status] == MTLCommandBufferStatusError) ? [cb0 error]
                             : ((cb_mid && [cb_mid status] == MTLCommandBufferStatusError) ? [cb_mid error] : [cb_final error]);
                std::fprintf(stderr, "[strata_qwen_flash_gpu] GPU prefill command buffer error at pos=%u: %s\n",
                             cur_pos, err ? [[err localizedDescription] UTF8String] : "unknown");
                return -4;
            }
            if (eng->kv_paged && sqf_take_paged_error(eng)) {
                (void)join_next_ple();
                std::fprintf(stderr, "[strata_qwen_flash_gpu] paged KV recall/evict failure during prefill at pos=%u\n", cur_pos);
                return -5;
            }

            eng->seq_pos += M;
            sqf_ngram_append(eng->bound_slot, token_ids + chunk_off, M);
            if (eng->progress_cb) {
                eng->progress_cb(chunk_off + M, num_tokens, eng->seq_pos, eng->progress_user_ctx);
            }
        }

        if (want_head) {
            const uint32_t first_tok = *reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents]);
            strata_qwen_flash_gpu_penalties_record(eng, &first_tok, 1U);
            if (next_token_out) {
                *next_token_out = first_tok;
            }
        }
        if (want_head && last_logits_out) {
            std::memcpy(last_logits_out, [eng->b_logits contents], static_cast<size_t>(V) * sizeof(float));
        }
        return 0;
    }
}

// Phase B: Width-2 x Depth-3 Speculative Tree Decode (`M = 7`: root + Branch A [a0,a1,a2] + Branch B [b0,b1,b2])
//   - Drafting (`cb_draft`): 3 batched `lm_head` passes (`M=1` top-2 -> `M=2` depth-1 -> `M=2` depth-2), zero extra DRAM reads!
//   - Verification (`cb_ver`): Single `M=7` 48-layer forward with tree attention masks, grouped MoE expert dedup,
//     100%-coalesced `rec_batch` reloading `snap0` before Branch B (`t=4`), and 6 zero-copy ring snapshot slots (`0..5`).
//   - Exact greedy (`Tree == K=0` bit-identical) AND mathematically lossless 2-stage SpecInfer/Leviathan at `T > 0`.
//   - Async `cb_fix`: O(1) zero-copy ring swap + Branch-B KV/Indexer compaction + optional True-State MTP KV Sync replay.
int sqf_spec_tree_decode_step_impl(
    StrataQwenFlashGpuEngine*            eng,
    uint32_t                             pending_token_id,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t*                            emitted_tokens_out,
    uint32_t*                            num_emitted_out,
    uint32_t*                            num_draft_accepted_out
) {
    @autoreleasepool {
        constexpr uint32_t kTreeDraftK = 3U;
        constexpr uint32_t kTreeM      = 7U;
        constexpr uint32_t kBranchT    = 4U;

        const uint32_t D          = eng->cfg.hidden_size;
        const uint32_t H          = eng->cfg.hc_count;
        const uint32_t hc_dim     = H * D;
        const uint32_t num_layers = eng->cfg.num_hidden_layers;
        const uint32_t cur_pos    = eng->seq_pos; // P >= 1
        const bool     is_greedy  = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
        uint32_t       split_ple  = (eng->cfg.has_ple && eng->cfg.ple_layer_idx > 0U && eng->cfg.ple_layer_idx < num_layers)
                                        ? eng->cfg.ple_layer_idx : 0U;

        uint32_t* mtp_toks_host = reinterpret_cast<uint32_t*>([eng->b_mtp_token_ids contents]);
        mtp_toks_host[0] = pending_token_id;
        const uint32_t mtp_base_pos = (eng->mtp_seq_pos < cur_pos) ? eng->mtp_seq_pos : (cur_pos - 1U);

        // 1. Batched Width-2 x Depth-3 MTP Tree Drafting (1 CB, 3 `lm_head` passes: M=1 -> M=2 -> M=2)
        //    Layout in `b_mtp_token_ids`: [0]=root, [1]=a0, [2]=b0, [3]=a1, [4]=b1, [5]=a2, [6]=b2
        //    Layout in `b_draft_top_*`:   [0]=q0(root), [1]=q(a1|a0), [2]=q(b1|b0), [3]=q(a2|a1), [4]=q(b2|b1)
        {
            uint32_t sl_v_out = 0U;
            const bool sl_on = sqf_frspec_active(eng, &sl_v_out);

            id<MTLCommandBuffer> cb_draft          = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_draft = [cb_draft computeCommandEncoder];
            eng->sample_step = (static_cast<uint64_t>(cur_pos + 1U) << 8U) | 0x80ULL;

            // Depth 0 (`M=1`, `tree_mode=0`): emits `(a0, b0)` -> `b_mtp_token_ids[1..2]`
            sqf_encode_mtp_pass(
                enc_draft, cb_draft, eng,
                eng->b_mtp_hc_prev, 0,
                eng->b_mtp_token_ids, 0,
                mtp_base_pos, 1U, true, true, 0U, sl_on
            );
            sqf_encode_sample_batch(
                enc_draft, eng, sampling, 1U,
                eng->b_mtp_token_ids, 1U * sizeof(uint32_t),
                is_greedy ? nil : eng->b_draft_top_ids,   0U,
                is_greedy ? nil : eng->b_draft_top_probs, 0U,
                is_greedy ? nil : eng->b_draft_keep_k,    0U,
                /*want_top2=*/true,
                sl_on ? sl_v_out : 0U
            );

            // Duplicate `b_mtp_hc[0]` -> `b_mtp_hc[1]` so both `[a0, b0]` share `root`'s post-layer0 MTP state
            sqf_encode_copy_buffer(
                enc_draft, eng->pso_copy_u32,
                eng->b_mtp_hc, 0,
                eng->b_mtp_hc, static_cast<NSUInteger>(hc_dim) * sizeof(float),
                static_cast<size_t>(hc_dim) * sizeof(float)
            );

            // Depth 1 (`M=2`, `tree_mode=2`, `mtp_pos = Q`): emits `(a1, b1)` -> `b_mtp_token_ids[3..4]`
            sqf_encode_mtp_pass(
                enc_draft, cb_draft, eng,
                eng->b_mtp_hc, 0,
                eng->b_mtp_token_ids, 1U * sizeof(uint32_t),
                mtp_base_pos + 1U, 2U, true, true, 2U, sl_on
            );
            sqf_encode_sample_batch(
                enc_draft, eng, sampling, 2U,
                eng->b_mtp_token_ids, 3U * sizeof(uint32_t),
                is_greedy ? nil : eng->b_draft_top_ids,   1U * 64ULL * sizeof(uint32_t),
                is_greedy ? nil : eng->b_draft_top_probs, 1U * 64ULL * sizeof(float),
                is_greedy ? nil : eng->b_draft_keep_k,    1U * sizeof(uint32_t),
                /*want_top2=*/false,
                sl_on ? sl_v_out : 0U
            );

            // Depth 2 (`M=2`, `tree_mode=3`, `mtp_pos = Q+1`): emits `(a2, b2)` -> `b_mtp_token_ids[5..6]`
            sqf_encode_mtp_pass(
                enc_draft, cb_draft, eng,
                eng->b_mtp_hc, 0,
                eng->b_mtp_token_ids, 3U * sizeof(uint32_t),
                mtp_base_pos + 2U, 2U, true, true, 3U, sl_on
            );
            sqf_encode_sample_batch(
                enc_draft, eng, sampling, 2U,
                eng->b_mtp_token_ids, 5U * sizeof(uint32_t),
                is_greedy ? nil : eng->b_draft_top_ids,   3U * 64ULL * sizeof(uint32_t),
                is_greedy ? nil : eng->b_draft_top_probs, 3U * 64ULL * sizeof(float),
                is_greedy ? nil : eng->b_draft_keep_k,    3U * sizeof(uint32_t),
                /*want_top2=*/false,
                sl_on ? sl_v_out : 0U
            );

            [enc_draft endEncoding];
            [cb_draft commit];
            [cb_draft waitUntilCompleted];
            if (eng->pending_fix_cb) {
                if (!sqf_drain_pending_fix(eng)) return -4;
            }
            if ([cb_draft status] == MTLCommandBufferStatusError) {
                NSError* err = [cb_draft error];
                std::fprintf(stderr, "[strata_qwen_flash_gpu] GPU spec_tree_draft CB error at pos=%u: %s\n",
                             cur_pos, err ? [[err localizedDescription] UTF8String] : "unknown");
                return -4;
            }
        }

        // Arrange `verify_tokens[7] = {root, a0, a1, a2, b0, b1, b2}` (contiguous per branch!)
        uint32_t verify_tokens[kMaxSpecBatch] = {};
        verify_tokens[0] = mtp_toks_host[0]; // root
        verify_tokens[1] = mtp_toks_host[1]; // a0
        verify_tokens[2] = mtp_toks_host[3]; // a1
        verify_tokens[3] = mtp_toks_host[5]; // a2
        verify_tokens[4] = mtp_toks_host[2]; // b0
        verify_tokens[5] = mtp_toks_host[4]; // b1
        verify_tokens[6] = mtp_toks_host[6]; // b2
        std::memcpy([eng->b_token_ids contents], verify_tokens, static_cast<size_t>(kTreeM) * sizeof(uint32_t));

        // 2. Single Batched 48-Layer Tree Verification (`M = 7`, `snap_mask = 0x3F`, `tree_mode = 1`, `force_gemv = true`)
        constexpr uint32_t kSnapMask = 0x3FU; // bits 0..5 -> direct snapshots for nodes 0..5 (node 6 stays in live state)
        uint32_t ple_prev_after[kMaxSpecBatch][2] = {};
        __block int ple_async_rc = 0;
        bool ple_in_flight = false;

        if (split_ple > 0U && sqf_ple_all_cached(eng, verify_tokens, kTreeM, kBranchT)) {
            if (sqf_execute_ple_lookup_host(eng, verify_tokens, kTreeM, ple_prev_after, kBranchT) != 0) return -3;
            split_ple = 0U;
        } else if (split_ple > 0U) {
            const uint32_t* toks_ptr = verify_tokens;
            uint32_t (*prev_ptr)[2]  = ple_prev_after;
            ple_in_flight = true;
            dispatch_group_async(eng->ple_async_group, eng->ple_async_queue, ^{
                ple_async_rc = sqf_execute_ple_lookup_host(eng, toks_ptr, kTreeM, prev_ptr, kBranchT);
            });
        } else if (eng->cfg.has_ple) {
            if (sqf_execute_ple_lookup_host(eng, verify_tokens, kTreeM, ple_prev_after, kBranchT) != 0) return -3;
        }

        uint32_t lin_slot  = 0U;
        uint32_t full_slot = 0U;

        id<MTLCommandBuffer> cb0          = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc0 = [cb0 computeCommandEncoder];

        {
            SqfEmbedParamsHost ep{kTreeM, D, eng->cfg.vocab_size, H};
            [enc0 setComputePipelineState:eng->pso_embed_q4];
            [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.weight_off) atIndex:0];
            [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.scales_off) atIndex:1];
            [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.biases_off) atIndex:2];
            [enc0 setBuffer:eng->b_token_ids offset:0 atIndex:3];
            [enc0 setBuffer:eng->b_hc offset:0 atIndex:4];
            [enc0 setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc0 dispatchThreadgroups:MTLSizeMake(kTreeM, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
        }

        const uint32_t phase0_end = (split_ple > 0U) ? split_ple : num_layers;
        for (uint32_t l = 0U; l < phase0_end; ++l) {
            const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
            sqf_encode_decoder_layer(enc0, cb0, eng, ld, eng->b_hc, lin_slot, full_slot,
                                     cur_pos, kTreeM, kTreeM, kSnapMask, true, 1U);
            if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
            else ++full_slot;
        }

        id<MTLCommandBuffer> cb_ver          = cb0;
        id<MTLComputeCommandEncoder> enc_ver = enc0;

        if (split_ple > 0U) {
            [enc0 endEncoding];
            [cb0 commit];

            cb_ver  = [eng->queue commandBuffer];
            enc_ver = [cb_ver computeCommandEncoder];

            for (uint32_t l = split_ple; l < num_layers; ++l) {
                const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                sqf_encode_decoder_layer(enc_ver, cb_ver, eng, ld, eng->b_hc, lin_slot, full_slot,
                                         cur_pos, kTreeM, kTreeM, kSnapMask, true, 1U);
                if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                else ++full_slot;
            }
        }

        // Final Mixer + Untied LM Head (1x DRAM read for all 7 nodes!) + Optional Logit Penalties + Batched Sampler (`M = 7`)
        sqf_encode_mixer(enc_ver, eng, eng->b_part1, eng->hdr1.final_mixer, eng->b_hc, 0, kTreeM, kTreeM, true);
        sqf_encode_q4_linear(enc_ver, eng, eng->b_part1, eng->hdr1.lm_head,
                             eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                             kTreeM, kTreeM, kOutF32, true);
        sqf_encode_logit_penalties(enc_ver, eng, sampling, kTreeM, eng->b_token_ids, 2U, kTreeM - 1U);
        eng->sample_step = static_cast<uint64_t>(cur_pos + 1U) << 8U;
        sqf_encode_sample_batch(
            enc_ver, eng, sampling, kTreeM,
            eng->b_next_tokens, 0,
            is_greedy ? nil : eng->b_sample_top_ids, 0,
            is_greedy ? nil : eng->b_sample_top_probs, 0,
            is_greedy ? nil : eng->b_sample_keep_k, 0,
            /*want_top2=*/false
        );

        [enc_ver endEncoding];
        if (ple_in_flight) {
            dispatch_group_wait(eng->ple_async_group, DISPATCH_TIME_FOREVER);
            ple_in_flight = false;
            if (ple_async_rc != 0) {
                [cb0 waitUntilCompleted];
                return -3;
            }
        }
        [cb_ver commit];
        [cb_ver waitUntilCompleted];
        if ([cb0 status] == MTLCommandBufferStatusError ||
            [cb_ver status] == MTLCommandBufferStatusError) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] GPU spec_tree_ver CB error at pos=%u\n", cur_pos);
            return -4;
        }

        // 3. Exact Greedy (`Tree == K=0`) OR Mathematically Exact 2-Stage SpecInfer/Leviathan Tree Verification
        const uint32_t* target_toks = reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents]);
        uint32_t accepted      = 0U;
        uint32_t winner_branch = 0U; // 0 = Branch A (or rejected at root), 1 = Branch B
        uint32_t bonus_tok     = target_toks[0];

        if (is_greedy) {
            if (target_toks[0] == verify_tokens[1]) {
                // Branch A: nodes 1, 2, 3
                winner_branch = 0U;
                emitted_tokens_out[accepted++] = verify_tokens[1];
                if (target_toks[1] == verify_tokens[2]) {
                    emitted_tokens_out[accepted++] = verify_tokens[2];
                    if (target_toks[2] == verify_tokens[3]) {
                        emitted_tokens_out[accepted++] = verify_tokens[3];
                    }
                }
                bonus_tok = target_toks[accepted];
            } else if (target_toks[0] == verify_tokens[4]) {
                // Branch B: nodes 4, 5, 6
                winner_branch = 1U;
                emitted_tokens_out[accepted++] = verify_tokens[4];
                if (target_toks[4] == verify_tokens[5]) {
                    emitted_tokens_out[accepted++] = verify_tokens[5];
                    if (target_toks[5] == verify_tokens[6]) {
                        emitted_tokens_out[accepted++] = verify_tokens[6];
                    }
                }
                bonus_tok = target_toks[3U + accepted];
            } else {
                winner_branch = 0U;
                accepted      = 0U;
                bonus_tok     = target_toks[0];
            }
            emitted_tokens_out[accepted] = bonus_tok;
        } else {
            const uint32_t* d_ids   = reinterpret_cast<const uint32_t*>([eng->b_draft_top_ids contents]);
            const float*    d_probs = reinterpret_cast<const float*>([eng->b_draft_top_probs contents]);
            const uint32_t* d_keep  = reinterpret_cast<const uint32_t*>([eng->b_draft_keep_k contents]);

            const uint32_t* p_ids   = reinterpret_cast<const uint32_t*>([eng->b_sample_top_ids contents]);
            const float*    p_probs = reinterpret_cast<const float*>([eng->b_sample_top_probs contents]);
            const uint32_t* p_keep  = reinterpret_cast<const uint32_t*>([eng->b_sample_keep_k contents]);

            const uint64_t rng_base = sampling ? sampling->rng_seed : 12345ULL;
            eng->sample_step = (static_cast<uint64_t>(cur_pos + 1U) << 8U) | 0x40ULL;

            auto sample_discrete = [&](const uint32_t* ids, const float* w, uint32_t n, float sum_w, uint32_t fallback) -> uint32_t {
                if (sum_w <= 1e-12f || n == 0U) return fallback;
                const float u = sqf_host_rand_u01(rng_base, eng->sample_step++) * sum_w;
                float cdf = 0.0f;
                for (uint32_t i = 0U; i < n; ++i) {
                    cdf += w[i];
                    if (u <= cdf || i + 1U == n) return ids[i];
                }
                return ids[0];
            };

            // Stage 1: Depth-0 root (`node = 0`, draft slot `0`) test `a0` then `b0` (without-replacement residual!)
            const uint32_t a0    = verify_tokens[1];
            const uint32_t b0    = verify_tokens[4];
            const float    q_a0  = std::max<float>(1e-12f, sqf_lookup_sparse_prob(a0, d_ids, d_probs, d_keep[0]));
            const float    p_a0  = sqf_lookup_sparse_prob(a0, p_ids, p_probs, p_keep[0]);
            const float    u0    = sqf_host_rand_u01(rng_base, eng->sample_step++);
            bool           d0_ok = false;

            if (p_a0 >= q_a0 || u0 < (p_a0 / q_a0)) {
                winner_branch = 0U;
                emitted_tokens_out[accepted++] = a0;
                d0_ok = true;
            } else {
                // Exact residual `p0'(x) = norm(max(0, p0(x) - q0(x)))`
                const uint32_t kp0 = p_keep[0];
                float r0[64];
                float sum_r0 = 0.0f;
                for (uint32_t i = 0U; i < kp0; ++i) {
                    const uint32_t cand = p_ids[i];
                    const float    pv   = p_probs[i];
                    const float    qv   = sqf_lookup_sparse_prob(cand, d_ids, d_probs, d_keep[0]);
                    const float    rv   = std::max<float>(0.0f, pv - qv);
                    r0[i]   = rv;
                    sum_r0 += rv;
                }

                if (sum_r0 > 1e-12f && b0 != a0 && q_a0 < 1.0f - 1e-7f) {
                    const float inv_rem_q = 1.0f / (1.0f - q_a0);
                    const float q_b0_raw  = sqf_lookup_sparse_prob(b0, d_ids, d_probs, d_keep[0]);
                    const float q_b0_p    = std::max<float>(1e-12f, q_b0_raw * inv_rem_q);
                    const float p_b0_raw  = sqf_lookup_sparse_prob(b0, p_ids, p_probs, kp0);
                    const float p_b0_p    = std::max<float>(0.0f, p_b0_raw - q_b0_raw) / sum_r0;
                    const float u1        = sqf_host_rand_u01(rng_base, eng->sample_step++);

                    if (p_b0_p >= q_b0_p || u1 < (p_b0_p / q_b0_p)) {
                        winner_branch = 1U;
                        emitted_tokens_out[accepted++] = b0;
                        d0_ok = true;
                    } else {
                        // Second-stage residual `r1(x) = max(0, p0'(x) - q0'(x))`
                        float r1[64];
                        float sum_r1 = 0.0f;
                        const float inv_r0 = 1.0f / sum_r0;
                        for (uint32_t i = 0U; i < kp0; ++i) {
                            const uint32_t cand = p_ids[i];
                            const float    pp   = r0[i] * inv_r0;
                            const float    qp   = (cand == a0)
                                ? 0.0f
                                : (sqf_lookup_sparse_prob(cand, d_ids, d_probs, d_keep[0]) * inv_rem_q);
                            const float    rv   = std::max<float>(0.0f, pp - qp);
                            r1[i]   = rv;
                            sum_r1 += rv;
                        }
                        bonus_tok = (sum_r1 > 1e-12f)
                            ? sample_discrete(p_ids, r1, kp0, sum_r1, target_toks[0])
                            : sample_discrete(p_ids, r0, kp0, sum_r0, target_toks[0]);
                        emitted_tokens_out[0] = bonus_tok;
                    }
                } else {
                    bonus_tok = sample_discrete(p_ids, r0, kp0, sum_r0, target_toks[0]);
                    emitted_tokens_out[0] = bonus_tok;
                }
            }

            // Stage 2: Standard exact Leviathan walk along winning branch for depths 1 and 2
            if (d0_ok) {
                const uint32_t tok_nodes[2] = {
                    (winner_branch == 0U) ? 2U : 5U,
                    (winner_branch == 0U) ? 3U : 6U
                };
                const uint32_t p_nodes[2] = {
                    (winner_branch == 0U) ? 1U : 4U,
                    (winner_branch == 0U) ? 2U : 5U
                };
                const uint32_t d_slots[2] = {
                    (winner_branch == 0U) ? 1U : 2U,
                    (winner_branch == 0U) ? 3U : 4U
                };
                const uint32_t final_node = (winner_branch == 0U) ? 3U : 6U;

                bool rejected = false;
                for (uint32_t step = 0U; step < 2U; ++step) {
                    const uint32_t drafted = verify_tokens[tok_nodes[step]];
                    const uint32_t pn      = p_nodes[step];
                    const uint32_t ds      = d_slots[step];
                    const float    qv      = std::max<float>(
                        1e-12f, sqf_lookup_sparse_prob(drafted, d_ids + ds * 64U, d_probs + ds * 64U, d_keep[ds]));
                    const float    pv      = sqf_lookup_sparse_prob(
                        drafted, p_ids + pn * 64U, p_probs + pn * 64U, p_keep[pn]);
                    const float    ua      = sqf_host_rand_u01(rng_base, eng->sample_step++);

                    if (pv >= qv || ua < (pv / qv)) {
                        emitted_tokens_out[accepted++] = drafted;
                    } else {
                        const uint32_t kp = p_keep[pn];
                        float res_p[64];
                        float res_sum = 0.0f;
                        for (uint32_t i = 0U; i < kp; ++i) {
                            const uint32_t cand = p_ids[pn * 64U + i];
                            const float    p_c  = p_probs[pn * 64U + i];
                            const float    q_c  = sqf_lookup_sparse_prob(
                                cand, d_ids + ds * 64U, d_probs + ds * 64U, d_keep[ds]);
                            const float    rv   = std::max<float>(0.0f, p_c - q_c);
                            res_p[i]  = rv;
                            res_sum  += rv;
                        }
                        bonus_tok = sample_discrete(p_ids + pn * 64U, res_p, kp, res_sum, target_toks[pn]);
                        emitted_tokens_out[accepted] = bonus_tok;
                        rejected = true;
                        break;
                    }
                }
                if (!rejected) {
                    bonus_tok = target_toks[final_node];
                    emitted_tokens_out[accepted] = bonus_tok;
                }
            }
        }

        // 4. Commit Winner State + O(1) Ring Rollback + Branch-B Compaction + True-State MTP KV Sync Replay (`cb_fix`)
        const uint32_t winner_node = (accepted == 0U)
            ? 0U
            : ((winner_branch == 0U) ? accepted : (3U + accepted));
        {
            const bool need_rollback = (winner_node < 6U);
            const bool can_zero_copy_swap =
                need_rollback && eng->bound_slot != nullptr && eng->ring_rec_rsets[winner_node] != nil;
            if (can_zero_copy_swap) {
                std::swap(eng->b_rec_state, eng->b_rec_slots[winner_node]);
                eng->bound_slot->b_rec_state = eng->b_rec_state;
                std::swap(eng->b_conv_state, eng->b_conv_slots[winner_node]);
                eng->bound_slot->b_conv_state = eng->b_conv_state;
                if (eng->cfg.has_ple && eng->ple_state_bytes > 0ULL) {
                    std::swap(eng->b_ple_conv_state, eng->b_ple_conv_slots[winner_node]);
                    eng->bound_slot->b_ple_conv_state = eng->b_ple_conv_state;
                }
                std::swap(eng->bound_slot->rec_rset, eng->ring_rec_rsets[winner_node]);
            }

            id<MTLCommandBuffer> cb_fix          = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_fix = [cb_fix computeCommandEncoder];

            // 4a. Save true pre-mixer `b_mtp_hc_prev = b_hc[winner_node]` FIRST (before any compaction of `b_hc`)
            const NSUInteger hc_win_off = static_cast<NSUInteger>(winner_node) * hc_dim * sizeof(float);
            sqf_encode_copy_buffer(enc_fix, eng->pso_copy_u32,
                                   eng->b_hc, hc_win_off, eng->b_mtp_hc_prev, 0,
                                   static_cast<size_t>(hc_dim) * sizeof(float));

            // 4b. Non-default slot fallback rollback copy
            if (need_rollback && !can_zero_copy_swap) {
                sqf_encode_copy_buffer(enc_fix, eng->pso_copy_u32,
                                       eng->b_rec_slots[winner_node], 0,
                                       eng->b_rec_state, 0, eng->rec_state_bytes);
                sqf_encode_copy_buffer(enc_fix, eng->pso_copy_u32,
                                       eng->b_conv_slots[winner_node], 0,
                                       eng->b_conv_state, 0, eng->conv_state_bytes);
                if (eng->cfg.has_ple && eng->ple_state_bytes > 0ULL) {
                    sqf_encode_copy_buffer(enc_fix, eng->pso_copy_u32,
                                           eng->b_ple_conv_slots[winner_node], 0,
                                           eng->b_ple_conv_state, 0, eng->ple_state_bytes);
                }
            }

            const uint32_t comp_ratio = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
            const uint32_t block_topk = std::max<uint32_t>(1U, eng->cfg.indexer_budget / comp_ratio);
            const uint32_t n_kv       = eng->cfg.num_key_value_heads;
            const uint32_t head_dim   = eng->cfg.head_dim;
            const uint32_t idx_hd     = eng->cfg.indexer_head_dim;
            const uint32_t tg_compact = std::max<uint32_t>(head_dim, idx_hd);

            // 4c. If Branch B won (`accepted >= 1`): compact `[P+4..P+4+accepted-1] -> [P+1..P+accepted]`
            //     across all 12 main full-attention layers AND rebuild `block_k_cache[P / comp_ratio]` if completed!
            if (winner_branch == 1U && accepted >= 1U) {
                const bool rebuild_blk =
                    ((cur_pos + 1U + accepted) / comp_ratio) > ((cur_pos + 1U) / comp_ratio);
                const SqfTreeCompactParamsHost cp{
                    cur_pos + 1U, cur_pos + 4U, accepted, n_kv, head_dim, idx_hd, eng->kv_stride_tokens, 0U
                };
                SqfIndexerParamsHost ip{
                    1U, eng->cfg.indexer_n_heads, idx_hd, eng->cfg.rotary_dim, comp_ratio, block_topk,
                    cur_pos, cur_pos / comp_ratio, 1U, eng->cfg.rms_norm_eps, eng->max_blocks_alloc, 0U,
                    1U, 0U, 0U, 0U
                };

                uint32_t fs = 0U;
                for (uint32_t l = 0U; l < num_layers; ++l) {
                    const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                    if (ld.layer_type != STRATA_QWEN_FLASH_LAYER_FULL_ATTN) continue;
                    const NSUInteger kv_off      = static_cast<NSUInteger>(static_cast<uint64_t>(fs) * eng->kv_layer_stride);
                    const NSUInteger idx_raw_off = static_cast<NSUInteger>(static_cast<uint64_t>(fs) * eng->idx_raw_layer_stride);
                    const NSUInteger idx_blk_off = static_cast<NSUInteger>(static_cast<uint64_t>(fs) * eng->idx_blk_layer_stride);

                    [enc_fix setComputePipelineState:eng->pso_tree_compact_b];
                    [enc_fix setBuffer:eng->b_k_cache offset:kv_off atIndex:0];
                    [enc_fix setBuffer:eng->b_v_cache offset:kv_off atIndex:1];
                    [enc_fix setBuffer:eng->b_idx_raw_k offset:idx_raw_off atIndex:2];
                    [enc_fix setBytes:&cp length:sizeof(cp) atIndex:3];
                    [enc_fix dispatchThreadgroups:MTLSizeMake(2U * n_kv + 1U, accepted, 1)
                            threadsPerThreadgroup:MTLSizeMake(tg_compact, 1, 1)];

                    if (rebuild_blk) {
                        id<MTLBuffer> w_buf = sqf_part_buf(eng, ld.part_id);
                        [enc_fix setComputePipelineState:eng->pso_qsa_idx_blocks];
                        [enc_fix setBuffer:eng->b_idx_raw_k offset:idx_raw_off atIndex:0];
                        [enc_fix setBuffer:w_buf offset:static_cast<NSUInteger>(ld.indexer_k_norm.offset) atIndex:1];
                        [enc_fix setBuffer:eng->b_inv_freq offset:0 atIndex:2];
                        [enc_fix setBuffer:eng->b_idx_block_k offset:idx_blk_off atIndex:3];
                        [enc_fix setBytes:&ip length:sizeof(ip) atIndex:4];
                        [enc_fix dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(idx_hd, 1, 1)];
                    }
                    ++fs;
                }
            }

            // 4d. MTP KV Sync (true main-model `b_hc` replay for `accepted >= 1`, or fast MTP KV compaction if disabled)
            if (accepted >= 1U) {
                if (eng->mtp_kv_sync_enabled) {
                    if (winner_branch == 1U && accepted >= 2U) {
                        sqf_encode_copy_buffer(
                            enc_fix, eng->pso_copy_u32,
                            eng->b_hc, static_cast<NSUInteger>(4U) * hc_dim * sizeof(float),
                            eng->b_hc, static_cast<NSUInteger>(1U) * hc_dim * sizeof(float),
                            static_cast<size_t>(accepted - 1U) * hc_dim * sizeof(float)
                        );
                    }
                    const NSUInteger tok_off = static_cast<NSUInteger>(winner_branch == 0U ? 1U : 4U) * sizeof(uint32_t);
                    sqf_encode_mtp_pass(
                        enc_fix, cb_fix, eng,
                        eng->b_hc, 0,
                        eng->b_token_ids, tok_off,
                        mtp_base_pos + 1U, accepted, /*run_head=*/false, /*force_gemv=*/true, /*tree_mode=*/0U
                    );
                } else {
                    const uint32_t   mtp_fs      = eng->cfg.num_full_attn_layers;
                    const uint32_t   mtp_q       = mtp_base_pos + 1U;
                    const NSUInteger kv_off      = static_cast<NSUInteger>(static_cast<uint64_t>(mtp_fs) * eng->kv_layer_stride);
                    const NSUInteger idx_raw_off = static_cast<NSUInteger>(static_cast<uint64_t>(mtp_fs) * eng->idx_raw_layer_stride);
                    const NSUInteger idx_blk_off = static_cast<NSUInteger>(static_cast<uint64_t>(mtp_fs) * eng->idx_blk_layer_stride);
                    auto compact_one = [&](uint32_t dst_p, uint32_t src_p) {
                        const SqfTreeCompactParamsHost cp{
                            dst_p, src_p, 1U, n_kv, head_dim, idx_hd, eng->kv_stride_tokens, 0U
                        };
                        [enc_fix setComputePipelineState:eng->pso_tree_compact_b];
                        [enc_fix setBuffer:eng->b_k_cache offset:kv_off atIndex:0];
                        [enc_fix setBuffer:eng->b_v_cache offset:kv_off atIndex:1];
                        [enc_fix setBuffer:eng->b_idx_raw_k offset:idx_raw_off atIndex:2];
                        [enc_fix setBytes:&cp length:sizeof(cp) atIndex:3];
                        [enc_fix dispatchThreadgroups:MTLSizeMake(2U * n_kv + 1U, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(tg_compact, 1, 1)];
                    };
                    if (winner_branch == 0U) {
                        if (accepted >= 2U) compact_one(mtp_q + 1U, mtp_q + 2U);
                    } else {
                        compact_one(mtp_q, mtp_q + 1U);
                        if (accepted >= 2U) compact_one(mtp_q + 1U, mtp_q + 3U);
                    }
                    const uint32_t synced = std::min<uint32_t>(accepted, 2U);
                    if (((mtp_q + synced) / comp_ratio) > (mtp_q / comp_ratio)) {
                        SqfIndexerParamsHost ip{
                            1U, eng->cfg.indexer_n_heads, idx_hd, eng->cfg.rotary_dim, comp_ratio, block_topk,
                            mtp_q, mtp_q / comp_ratio, 1U, eng->cfg.rms_norm_eps, eng->max_blocks_alloc, 0U,
                            1U, 0U, 0U, 0U
                        };
                        [enc_fix setComputePipelineState:eng->pso_qsa_idx_blocks];
                        [enc_fix setBuffer:eng->b_idx_raw_k offset:idx_raw_off atIndex:0];
                        [enc_fix setBuffer:eng->b_part1 offset:static_cast<NSUInteger>(eng->hdr1.mtp.layer0.indexer_k_norm.offset) atIndex:1];
                        [enc_fix setBuffer:eng->b_inv_freq offset:0 atIndex:2];
                        [enc_fix setBuffer:eng->b_idx_block_k offset:idx_blk_off atIndex:3];
                        [enc_fix setBytes:&ip length:sizeof(ip) atIndex:4];
                        [enc_fix dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(idx_hd, 1, 1)];
                    }
                }
            }

            sqf_frspec_stage_and_encode_patches(enc_fix, eng, emitted_tokens_out, 1U + accepted, /*record_misses=*/true);

            [enc_fix endEncoding];
            [cb_fix commit];
            eng->pending_fix_cb = cb_fix;
        }

        if (eng->cfg.has_ple) {
            eng->ple_prev_tokens[0] = ple_prev_after[winner_node][0];
            eng->ple_prev_tokens[1] = ple_prev_after[winner_node][1];
        }

        eng->seq_pos     = cur_pos + 1U + accepted;
        eng->mtp_seq_pos = mtp_base_pos + 1U + accepted;
        strata_qwen_flash_gpu_penalties_record(eng, emitted_tokens_out, 1U + accepted);
        if (eng->bound_slot) {
            uint32_t committed_ng[4];
            committed_ng[0] = pending_token_id;
            for (uint32_t k = 0U; k < accepted; ++k) committed_ng[1U + k] = emitted_tokens_out[k];
            sqf_ngram_append(eng->bound_slot, committed_ng, 1U + accepted);
        }

        *num_emitted_out = 1U + accepted;
        if (num_draft_accepted_out) *num_draft_accepted_out = accepted;

        // Telemetry accounting (depth = 3, width = 2 tree -> 3 drafted slots along winning path)
        eng->counters.spec_verify_steps   += 1ULL;
        eng->counters.mtp_drafted_tokens  += kTreeDraftK;
        eng->counters.mtp_accepted_tokens += accepted;

        const uint32_t comp_rat  = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
        const uint64_t num_comp  = static_cast<uint64_t>((cur_pos + 4U) / comp_rat);
        const uint64_t blk_topk  = std::max<uint64_t>(1ULL, eng->cfg.indexer_budget / comp_rat);
        const uint64_t idx_reads = (num_comp > blk_topk)
            ? (num_comp * eng->cfg.indexer_head_dim * sizeof(uint16_t) *
               (static_cast<uint64_t>(eng->cfg.num_full_attn_layers) + kTreeDraftK))
            : 0ULL;
        const uint64_t eff_kv = std::min<uint64_t>(cur_pos + 4U, eng->cfg.indexer_budget + 4U);
        const uint64_t one_moe_bytes =
            ((2ULL * eng->cfg.moe_intermediate_size * D) / 2ULL +
             2ULL * eng->cfg.moe_intermediate_size * (D / 64ULL) * 4ULL +
             (static_cast<uint64_t>(D) * eng->cfg.moe_intermediate_size) / 2ULL +
             static_cast<uint64_t>(D) * (eng->cfg.moe_intermediate_size / 64ULL) * 4ULL) *
            eng->cfg.num_experts_per_tok * num_layers;
        const uint64_t extra_moe = 4ULL * (one_moe_bytes * 6ULL / 10ULL);
        eng->counters.physical_dram_bytes_read +=
            static_cast<uint64_t>(kTreeDraftK) * eng->hdr0.mtp_active_bytes_per_tok +
            eng->hdr0.active_bytes_per_tok + extra_moe +
            2ULL * eng->rec_state_bytes +
            eff_kv * eng->kv_cache_bytes_per_token +
            idx_reads;

        return 0;
    }
}

} // namespace

extern "C" {

int strata_qwen_flash_gpu_prefill(
    StrataQwenFlashGpuEngine*            eng,
    const uint32_t*                      token_ids,
    uint32_t                             num_tokens,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t*                            next_token_out,
    float*                               last_logits_out
) {
    return sqf_prefill_impl(eng, token_ids, num_tokens, sampling, next_token_out, last_logits_out, true);
}

int strata_qwen_flash_gpu_prefill_slice(
    StrataQwenFlashGpuEngine* eng,
    const uint32_t*           token_ids,
    uint32_t                  slice_tokens,
    bool                      compute_final_logits,
    uint32_t*                 next_token_out
) {
    if (!eng || !token_ids || slice_tokens == 0U || slice_tokens > kMaxPrefillChunk) return -1;
    StrataQwenFlashSamplingParams greedy{};
    greedy.greedy = true;
    return sqf_prefill_impl(eng, token_ids, slice_tokens, &greedy, next_token_out, nullptr, compute_final_logits);
}

int strata_qwen_flash_gpu_spec_decode_step(
    StrataQwenFlashGpuEngine*            eng,
    uint32_t                             pending_token_id,
    uint32_t                             draft_k,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t*                            emitted_tokens_out,
    uint32_t*                            num_emitted_out,
    uint32_t*                            num_draft_accepted_out
) {
    if (!eng || !emitted_tokens_out || !num_emitted_out) return -1;

    if (draft_k == 0U || !eng->cfg.has_mtp || !eng->hdr1.mtp.enabled || eng->seq_pos == 0U) {
        uint32_t next_tok = 0U;
        int rc = strata_qwen_flash_gpu_decode_step(eng, pending_token_id, sampling, &next_tok, nullptr);
        if (rc != 0) return rc;
        emitted_tokens_out[0] = next_tok;
        *num_emitted_out      = 1U;
        if (num_draft_accepted_out) *num_draft_accepted_out = 0U;
        return 0;
    }

    if (draft_k > STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K) {
        draft_k = STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K;
    }
    if (eng->spec_tree_enabled && draft_k == 3U && !eng->kv_paged &&
        (eng->seq_pos + 7U <= eng->max_seq_len)) {
        return sqf_spec_tree_decode_step_impl(
            eng, pending_token_id, sampling,
            emitted_tokens_out, num_emitted_out, num_draft_accepted_out
        );
    }
    if (eng->seq_pos + 1U + draft_k > eng->max_seq_len) {
        uint32_t next_tok = 0U;
        int rc = strata_qwen_flash_gpu_decode_step(eng, pending_token_id, sampling, &next_tok, nullptr);
        if (rc != 0) return rc;
        emitted_tokens_out[0] = next_tok;
        *num_emitted_out      = 1U;
        if (num_draft_accepted_out) *num_draft_accepted_out = 0U;
        return 0;
    }

    @autoreleasepool {
        const uint32_t D          = eng->cfg.hidden_size;
        const uint32_t H          = eng->cfg.hc_count;
        const uint32_t hc_dim     = H * D;
        const uint32_t V          = eng->cfg.vocab_size;
        const uint32_t num_layers = eng->cfg.num_hidden_layers;
        const uint32_t cur_pos    = eng->seq_pos; // P >= 1
        const bool     is_greedy  = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
        uint32_t       split_ple  = (eng->cfg.has_ple && eng->cfg.ple_layer_idx > 0U && eng->cfg.ple_layer_idx < num_layers)
                                        ? eng->cfg.ple_layer_idx : 0U;

        // 1. Phase E1 Zero-Cost Rolling 4-Gram Suffix Drafter + 3-State SpecCtl Adaptive-K (`CooldownK0` / `ActiveProbe` / `HighConfFused`)
        uint32_t ng_drafts[kMaxSpecBatch] = {};
        uint32_t eff_k      = draft_k;
        bool     used_ngram = false;
        bool     high_conf  = false;
        if (eng->ngram_draft_enabled && eng->bound_slot) {
            const uint32_t ng_k = sqf_ngram_propose(eng->bound_slot, pending_token_id, draft_k, ng_drafts);
            if (ng_k > 0U) {
                used_ngram = true;
                eff_k      = ng_k;
            } else if (eng->bound_slot->ngram_cooldown > 0U) {
                --eng->bound_slot->ngram_cooldown;
            }
        }
        if (!used_ngram && eng->adaptive_k_enabled) {
            eff_k = sqf_spec_ctl_choose_k(eng, draft_k, &high_conf);
            if (eff_k > draft_k) eff_k = draft_k;
        }
        const bool     is_cooldown_k0 = (!used_ngram && eff_k == 0U);
        const uint32_t M              = 1U + eff_k;

        uint32_t* mtp_toks_host = reinterpret_cast<uint32_t*>([eng->b_mtp_token_ids contents]);
        if (eng->kv_paged) sqf_paged_step_begin(eng, cur_pos + M - 1U);

        const uint32_t mtp_base_pos = (eng->mtp_seq_pos < cur_pos) ? eng->mtp_seq_pos : (cur_pos - 1U);
        double prof_draft_gpu_ms = 0.0;
        double draft_gpu_ms      = 0.0;
        float  draft_mean_pmax   = 0.0f;
        uint32_t verify_tokens[kMaxSpecBatch] = {};
        const uint32_t snap_mask = (eff_k > 0U) ? ((1U << eff_k) - 1U) : 0U;

        const bool will_prof_this_step = eng->prof_enabled && ((eng->prof_decode_step_cnt + 1U) == 15U);
        // Zero-bubble `cb_draft -> cb0(Embed + Layer 0)` pipeline on ALL non-paged MTP steps (both ActiveProbe & HighConfFused)!
        const bool use_fused_l0_pipe =
            !used_ngram && !is_cooldown_k0 && !eng->kv_paged && !will_prof_this_step && (split_ple > 0U);
        (void)high_conf;

        uint32_t lin_slot  = 0U;
        uint32_t full_slot = 0U;
        id<MTLCommandBuffer> cb0          = nil;
        id<MTLComputeCommandEncoder> enc0 = nil;

        if (is_cooldown_k0) {
            // State A (`CooldownK0`): skip MTP drafting completely (`0.00 ms` draft, pure `M=1` forward, async `cb_fix` keeps MTP synced!)
            if (eng->pending_fix_cb) {
                if (!sqf_drain_pending_fix(eng)) return -4;
            }
            const bool do_prof_spec = eng->prof_enabled && (++eng->prof_decode_step_cnt == 15U);
            if (do_prof_spec) {
                prof_draft_gpu_ms = 0.0;
                std::memset(eng->prof_ms, 0, sizeof(eng->prof_ms));
                eng->prof_active = true;
            }
            verify_tokens[0] = pending_token_id;
        } else if (used_ngram) {
            // Drain any async `cb_fix` BEFORE mutating host-visible `b_token_ids`!
            if (eng->pending_fix_cb) {
                if (!sqf_drain_pending_fix(eng)) return -4;
            }
            const bool do_prof_spec = eng->prof_enabled && (++eng->prof_decode_step_cnt == 15U);
            if (do_prof_spec) {
                prof_draft_gpu_ms = 0.0;
                std::memset(eng->prof_ms, 0, sizeof(eng->prof_ms));
                eng->prof_active = true;
            }
            verify_tokens[0] = pending_token_id;
            for (uint32_t k = 0U; k < eff_k; ++k) verify_tokens[1U + k] = ng_drafts[k];
        } else {
            // Draft `eff_k` candidate tokens using Native 1-Layer `MTP` + 32K SLC-Pinned FR-Spec Shortlist!
            mtp_toks_host[0] = pending_token_id;
            uint32_t sl_v_out = 0U;
            const bool sl_on = sqf_frspec_active(eng, &sl_v_out);

            id<MTLCommandBuffer> cb_draft          = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_draft = [cb_draft computeCommandEncoder];
            eng->sample_step = (static_cast<uint64_t>(cur_pos + 1U) << 8U) | 0x80ULL;

            for (uint32_t k = 0U; k < eff_k; ++k) {
                id<MTLBuffer> hc_in = (k == 0U) ? eng->b_mtp_hc_prev : eng->b_mtp_hc;
                sqf_encode_mtp_pass(
                    enc_draft, cb_draft, eng,
                    hc_in, 0,
                    eng->b_mtp_token_ids, static_cast<NSUInteger>(k) * sizeof(uint32_t),
                    mtp_base_pos + k, 1U, true, true, 0U, sl_on
                );

                // Apply matching completion-scoped penalties to draft logits so MTP acceptance stays high!
                sqf_encode_logit_penalties(
                    enc_draft, eng, sampling, 1U, eng->b_mtp_token_ids, 1U, k,
                    sl_on ? sl_v_out : 0U
                );

                // Write drafted token directly into `b_mtp_token_ids[k + 1]` + `p_max` into `b_draft_pmax[k]`!
                sqf_encode_sample_batch(
                    enc_draft, eng, sampling, 1U,
                    eng->b_mtp_token_ids, static_cast<NSUInteger>(k + 1U) * sizeof(uint32_t),
                    is_greedy ? nil : eng->b_draft_top_ids,
                    static_cast<NSUInteger>(k) * 64ULL * sizeof(uint32_t),
                    is_greedy ? nil : eng->b_draft_top_probs,
                    static_cast<NSUInteger>(k) * 64ULL * sizeof(float),
                    is_greedy ? nil : eng->b_draft_keep_k,
                    static_cast<NSUInteger>(k) * sizeof(uint32_t),
                    /*want_top2=*/false,
                    sl_on ? sl_v_out : 0U,
                    eng->adaptive_k_enabled ? eng->b_draft_pmax : nil,
                    static_cast<NSUInteger>(k) * sizeof(float)
                );
            }

            [enc_draft endEncoding];
            [cb_draft commit];

            // State C (`HighConfFused`) Zero-Bubble Overlap: immediately submit `Embed(b_mtp_token_ids) + Layers 0..split_ple-1`
            // on the serial queue BEFORE waiting on `cb_draft`, so GPU starts Layer 0 with 0.00 ms bubble while CPU does PLE lookup!
            if (use_fused_l0_pipe) {
                cb0  = [eng->queue commandBuffer];
                enc0 = [cb0 computeCommandEncoder];
                {
                    SqfEmbedParamsHost ep{M, D, V, H};
                    [enc0 setComputePipelineState:eng->pso_embed_q4];
                    [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.weight_off) atIndex:0];
                    [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.scales_off) atIndex:1];
                    [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.biases_off) atIndex:2];
                    [enc0 setBuffer:eng->b_mtp_token_ids offset:0 atIndex:3];
                    [enc0 setBuffer:eng->b_hc offset:0 atIndex:4];
                    [enc0 setBytes:&ep length:sizeof(ep) atIndex:5];
                    [enc0 dispatchThreadgroups:MTLSizeMake(M, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
                }
                for (uint32_t l = 0U; l < split_ple; ++l) {
                    const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                    sqf_encode_decoder_layer(enc0, cb0, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, M, M, snap_mask, true);
                    if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                    else ++full_slot;
                }
                [enc0 endEncoding];
                [cb0 commit];
            }

            [cb_draft waitUntilCompleted];
            draft_gpu_ms = ([cb_draft GPUEndTime] - [cb_draft GPUStartTime]) * 1000.0;
            const bool do_prof_spec = eng->prof_enabled && (++eng->prof_decode_step_cnt == 15U);
            if (do_prof_spec) {
                prof_draft_gpu_ms = draft_gpu_ms;
                std::memset(eng->prof_ms, 0, sizeof(eng->prof_ms));
                eng->prof_active = true;
            }
            if (eng->pending_fix_cb) {
                if (!sqf_drain_pending_fix(eng)) return -4;
            }
            if ([cb_draft status] == MTLCommandBufferStatusError) {
                NSError* err = [cb_draft error];
                std::fprintf(stderr, "[strata_qwen_flash_gpu] GPU spec_draft command buffer error at pos=%u: %s\n",
                             cur_pos, err ? [[err localizedDescription] UTF8String] : "unknown");
                if (cb0) [cb0 waitUntilCompleted];
                return -4;
            }

            for (uint32_t m = 0U; m < M; ++m) {
                verify_tokens[m] = mtp_toks_host[m];
            }
            if (eng->adaptive_k_enabled && eng->b_draft_pmax) {
                const float* pm = reinterpret_cast<const float*>([eng->b_draft_pmax contents]);
                float sum_pm = 0.0f;
                for (uint32_t k = 0U; k < eff_k; ++k) sum_pm += pm[k];
                draft_mean_pmax = sum_pm / static_cast<float>(eff_k);
            }
        }
        std::memcpy([eng->b_token_ids contents], verify_tokens, static_cast<size_t>(M) * sizeof(uint32_t));

        // 2. Single Batched 48-Layer Main Model Verification (`M = 1 + eff_k`, `force_gemv = true`
        //    so every floating-point op for `m = 0 .. M-1` is 100.0000% bit-identical to `K=0`!)
        uint32_t ple_prev_after[kMaxSpecBatch][2] = {};
        __block int ple_async_rc = 0;
        bool ple_in_flight = false;

        if (use_fused_l0_pipe) {
            // `cb0` (Embed + Layer 0) is ALREADY executing on GPU! Run PLE lookup concurrently while encoding `cb_ver`!
            if (sqf_ple_all_cached(eng, verify_tokens, M)) {
                if (sqf_execute_ple_lookup_host(eng, verify_tokens, M, ple_prev_after) != 0) {
                    [cb0 waitUntilCompleted];
                    return -3;
                }
            } else {
                const uint32_t* toks_ptr = verify_tokens;
                uint32_t (*prev_ptr)[2]  = ple_prev_after;
                ple_in_flight = true;
                dispatch_group_async(eng->ple_async_group, eng->ple_async_queue, ^{
                    ple_async_rc = sqf_execute_ple_lookup_host(eng, toks_ptr, M, prev_ptr);
                });
            }
        } else if (split_ple > 0U && sqf_ple_all_cached(eng, verify_tokens, M)) {
            if (sqf_execute_ple_lookup_host(eng, verify_tokens, M, ple_prev_after) != 0) return -3;
            split_ple = 0U;
        } else if (split_ple > 0U) {
            const uint32_t* toks_ptr = verify_tokens;
            uint32_t (*prev_ptr)[2]  = ple_prev_after;
            ple_in_flight = true;
            dispatch_group_async(eng->ple_async_group, eng->ple_async_queue, ^{
                ple_async_rc = sqf_execute_ple_lookup_host(eng, toks_ptr, M, prev_ptr);
            });
        } else if (eng->cfg.has_ple) {
            if (sqf_execute_ple_lookup_host(eng, verify_tokens, M, ple_prev_after) != 0) return -3;
        }

        id<MTLCommandBuffer> cb_ver          = nil;
        id<MTLComputeCommandEncoder> enc_ver = nil;

        if (use_fused_l0_pipe) {
            cb_ver  = [eng->queue commandBuffer];
            enc_ver = [cb_ver computeCommandEncoder];
            for (uint32_t l = split_ple; l < num_layers; ++l) {
                const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                sqf_encode_decoder_layer(enc_ver, cb_ver, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, M, M, snap_mask, true);
                if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                else ++full_slot;
            }
        } else {
            cb0  = [eng->queue commandBuffer];
            enc0 = [cb0 computeCommandEncoder];

            {
                SqfEmbedParamsHost ep{M, D, V, H};
                [enc0 setComputePipelineState:eng->pso_embed_q4];
                [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.weight_off) atIndex:0];
                [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.scales_off) atIndex:1];
                [enc0 setBuffer:eng->b_part0 offset:static_cast<NSUInteger>(eng->hdr0.embed_tokens.biases_off) atIndex:2];
                [enc0 setBuffer:eng->b_token_ids offset:0 atIndex:3];
                [enc0 setBuffer:eng->b_hc offset:0 atIndex:4];
                [enc0 setBytes:&ep length:sizeof(ep) atIndex:5];
                [enc0 dispatchThreadgroups:MTLSizeMake(M, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
            }

            const uint32_t phase0_end = (split_ple > 0U) ? split_ple : num_layers;
            for (uint32_t l = 0U; l < phase0_end; ++l) {
                const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                sqf_encode_decoder_layer(enc0, cb0, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, M, M, snap_mask, true);
                if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                else ++full_slot;
            }

            cb_ver  = cb0;
            enc_ver = enc0;

            if (split_ple > 0U) {
                [enc0 endEncoding];
                [cb0 commit];

                cb_ver  = [eng->queue commandBuffer];
                enc_ver = [cb_ver computeCommandEncoder];

                for (uint32_t l = split_ple; l < num_layers; ++l) {
                    const StrataQwenFlashLayerDesc& ld = sqf_layer_desc(eng, l);
                    if (ple_in_flight && eng->kv_paged && ld.layer_type == STRATA_QWEN_FLASH_LAYER_FULL_ATTN) {
                        const uint32_t cr = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
                        const uint32_t bt = std::max<uint32_t>(1U, eng->cfg.indexer_budget / cr);
                        const bool     sp = ((cur_pos + M) / cr) > bt;
                        if (sqf_paged_layer_needs_sync(eng, full_slot, cur_pos, M, cr, sp, 0U)) {
                            dispatch_group_wait(eng->ple_async_group, DISPATCH_TIME_FOREVER);
                            ple_in_flight = false;
                            if (ple_async_rc != 0) {
                                [enc_ver endEncoding];
                                [cb0 waitUntilCompleted];
                                sqf_paged_join_probe(eng->bound_slot);
                                return -3;
                            }
                        }
                    }
                    sqf_encode_decoder_layer(enc_ver, cb_ver, eng, ld, eng->b_hc, lin_slot, full_slot, cur_pos, M, M, snap_mask, true);
                    if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++lin_slot;
                    else ++full_slot;
                }
            }
        }

        // Mix all `M` tokens -> `lm_head` (`1x` DRAM read!) -> Optional Logit Penalties -> Batched Sampler (`M` tokens)!
        sqf_encode_mixer(enc_ver, eng, eng->b_part1, eng->hdr1.final_mixer, eng->b_hc, 0, M, M, true);
        sqf_encode_q4_linear(enc_ver, eng, eng->b_part1, eng->hdr1.lm_head,
                             eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                             M, M, kOutF32, true);
        sqf_encode_logit_penalties(enc_ver, eng, sampling, M, eng->b_token_ids, 0U, eff_k);
        eng->sample_step = static_cast<uint64_t>(cur_pos + 1U) << 8U;
        sqf_encode_sample_batch(
            enc_ver, eng, sampling, M,
            eng->b_next_tokens, 0,
            is_greedy ? nil : eng->b_sample_top_ids, 0,
            is_greedy ? nil : eng->b_sample_top_probs, 0,
            is_greedy ? nil : eng->b_sample_keep_k, 0
        );
        sqf_prof_mark(eng, enc_ver, cb_ver, 11U); // [11] Final_Mixer + lm_head + Sample

        [enc_ver endEncoding];
        if (ple_in_flight) {
            dispatch_group_wait(eng->ple_async_group, DISPATCH_TIME_FOREVER);
            ple_in_flight = false;
            if (ple_async_rc != 0) {
                [cb0 waitUntilCompleted];
                return -3;
            }
        }
        [cb_ver commit];
        [cb_ver waitUntilCompleted];
        const double ver_gpu_ms = ([cb_ver GPUEndTime] - [cb0 GPUStartTime]) * 1000.0;
        if (eng->prof_active) {
            eng->prof_active = false;
            double tot_ver = 0.0;
            for (int b = 0; b < 12; ++b) tot_ver += eng->prof_ms[b];
            std::fprintf(stderr,
                "\n[PROF SPEC M=%u (K=%u eff_k=%u st=%u) pos=%u] DraftGPU=%.2f ms | VerifyGPU=%.2f ms | Sum=%.2f ms\n"
                "  HC_attn+PLE=%.2f | LinQKVZ=%.2f | DeltaRec=%.2f | LinOut=%.2f\n"
                "  QSA_QKV+Idx=%.2f | QSA_Attn+O=%.2f | HC_mlp=%.2f | Router+Sh=%.2f\n"
                "  MoE_GU=%.2f | MoE_DN=%.2f | MoE_Comb=%.2f | Head+Samp=%.2f\n",
                M, draft_k, eff_k, static_cast<uint32_t>(eng->spec_ctl.state), cur_pos, prof_draft_gpu_ms, tot_ver, prof_draft_gpu_ms + tot_ver,
                eng->prof_ms[0], eng->prof_ms[1], eng->prof_ms[2], eng->prof_ms[3],
                eng->prof_ms[4], eng->prof_ms[5], eng->prof_ms[6], eng->prof_ms[7],
                eng->prof_ms[8], eng->prof_ms[9], eng->prof_ms[10], eng->prof_ms[11]);
        }
        if ([cb0 status] == MTLCommandBufferStatusError ||
            [cb_ver status] == MTLCommandBufferStatusError) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] GPU spec_decode command buffer error at pos=%u\n", cur_pos);
            return -4;
        }
        if (eng->kv_paged && sqf_take_paged_error(eng)) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] paged KV recall/evict failure during spec decode at pos=%u\n", cur_pos);
            return -5;
        }

        // 3. Exact Leviathan Greedy / Stochastic Verification on CPU (`< 200 ns`!)
        const uint32_t* target_toks = reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents]);
        uint32_t accepted = 0U;
        uint32_t bonus_tok = target_toks[0];

        if (is_cooldown_k0) {
            emitted_tokens_out[0] = bonus_tok;
        } else if (is_greedy) {
            for (uint32_t k = 0U; k < eff_k; ++k) {
                const uint32_t drafted = verify_tokens[k + 1U];
                if (target_toks[k] == drafted) {
                    emitted_tokens_out[accepted++] = drafted;
                } else {
                    break;
                }
            }
            bonus_tok = target_toks[accepted];
            emitted_tokens_out[accepted] = bonus_tok;
        } else {
            const uint32_t* d_ids   = reinterpret_cast<const uint32_t*>([eng->b_draft_top_ids contents]);
            const float*    d_probs = reinterpret_cast<const float*>([eng->b_draft_top_probs contents]);
            const uint32_t* d_keep  = reinterpret_cast<const uint32_t*>([eng->b_draft_keep_k contents]);

            const uint32_t* p_ids   = reinterpret_cast<const uint32_t*>([eng->b_sample_top_ids contents]);
            const float*    p_probs = reinterpret_cast<const float*>([eng->b_sample_top_probs contents]);
            const uint32_t* p_keep  = reinterpret_cast<const uint32_t*>([eng->b_sample_keep_k contents]);

            const uint64_t rng_base = sampling ? sampling->rng_seed : 12345ULL;
            eng->sample_step = (static_cast<uint64_t>(cur_pos + 1U) << 8U) | 0x40ULL;
            bool rejected = false;

            for (uint32_t k = 0U; k < eff_k; ++k) {
                const uint32_t drafted = verify_tokens[k + 1U];
                // Point-mass N-gram draft has exact q(drafted) = 1.0, q(x != drafted) = 0.0 -> mathematically lossless Leviathan!
                const float q_val = used_ngram
                    ? 1.0f
                    : std::max<float>(1e-12f, sqf_lookup_sparse_prob(drafted, d_ids + k * 64U, d_probs + k * 64U, d_keep[k]));
                const float p_val = sqf_lookup_sparse_prob(
                    drafted, p_ids + k * 64U, p_probs + k * 64U, p_keep[k]);

                const float u_acc = sqf_host_rand_u01(rng_base, eng->sample_step++);
                if (p_val >= q_val || u_acc < (p_val / q_val)) {
                    emitted_tokens_out[accepted++] = drafted;
                } else {
                    // Sample from exact residual `normalize(max(0, p_k - q_k))` over `p_k`'s top-64 support!
                    const uint32_t kp = p_keep[k];
                    float res_p[64];
                    float res_sum = 0.0f;
                    for (uint32_t i = 0U; i < kp; ++i) {
                        const uint32_t cand = p_ids[k * 64U + i];
                        const float    pv   = p_probs[k * 64U + i];
                        const float    qv   = used_ngram
                            ? ((cand == drafted) ? 1.0f : 0.0f)
                            : sqf_lookup_sparse_prob(cand, d_ids + k * 64U, d_probs + k * 64U, d_keep[k]);
                        const float    rv   = std::max<float>(0.0f, pv - qv);
                        res_p[i] = rv;
                        res_sum += rv;
                    }
                    if (res_sum > 0.0f) {
                        const float u_res = sqf_host_rand_u01(rng_base, eng->sample_step++) * res_sum;
                        float cdf = 0.0f;
                        bonus_tok = p_ids[k * 64U];
                        for (uint32_t i = 0U; i < kp; ++i) {
                            cdf += res_p[i];
                            if (u_res <= cdf || i + 1U == kp) {
                                bonus_tok = p_ids[k * 64U + i];
                                break;
                            }
                        }
                    } else {
                        bonus_tok = target_toks[k];
                    }
                    emitted_tokens_out[accepted] = bonus_tok;
                    rejected = true;
                    break;
                }
            }

            if (!rejected) {
                bonus_tok = target_toks[eff_k];
                emitted_tokens_out[eff_k] = bonus_tok;
            }
        }

        // 4. Commit `b_mtp_hc_prev = b_hc[accepted]` AND Zero-Copy O(1) Ring-Slot Rollback (`accepted < eff_k`)!
        {
            const bool can_zero_copy_swap =
                (accepted < eff_k) && eng->bound_slot != nullptr && eng->ring_rec_rsets[accepted] != nil;
            if (can_zero_copy_swap) {
                std::swap(eng->b_rec_state, eng->b_rec_slots[accepted]);
                eng->bound_slot->b_rec_state = eng->b_rec_state;
                std::swap(eng->b_conv_state, eng->b_conv_slots[accepted]);
                eng->bound_slot->b_conv_state = eng->b_conv_state;
                if (eng->cfg.has_ple && eng->ple_state_bytes > 0ULL) {
                    std::swap(eng->b_ple_conv_state, eng->b_ple_conv_slots[accepted]);
                    eng->bound_slot->b_ple_conv_state = eng->b_ple_conv_state;
                }
                std::swap(eng->bound_slot->rec_rset, eng->ring_rec_rsets[accepted]);
            }

            id<MTLCommandBuffer> cb_fix          = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_fix = [cb_fix computeCommandEncoder];

            const bool mtp_sync_safe =
                !eng->kv_paged || (eng->bound_slot && eng->bound_slot->mtp_unpaged);

            // If N-gram or `CooldownK0` skipped `cb_draft`, replay `(old b_mtp_hc_prev, pending_token_id)` at `mtp_base_pos`
            // BEFORE overwriting `b_mtp_hc_prev` so MTP KV stays 100% synced!
            if ((used_ngram || is_cooldown_k0) && eng->mtp_kv_sync_enabled && mtp_sync_safe) {
                sqf_encode_mtp_pass(
                    enc_fix, cb_fix, eng,
                    eng->b_mtp_hc_prev, 0,
                    eng->b_token_ids, 0,
                    mtp_base_pos, 1U, /*run_head=*/false, /*force_gemv=*/true, /*tree_mode=*/0U
                );
            }

            const NSUInteger hc_acc_off = static_cast<NSUInteger>(accepted) * hc_dim * sizeof(float);
            sqf_encode_copy_buffer(enc_fix, eng->pso_copy_u32,
                                   eng->b_hc, hc_acc_off, eng->b_mtp_hc_prev, 0,
                                   static_cast<size_t>(hc_dim) * sizeof(float));

            if (accepted < eff_k && !can_zero_copy_swap) {
                sqf_encode_copy_buffer(
                    enc_fix, eng->pso_copy_u32,
                    eng->b_rec_slots[accepted], 0,
                    eng->b_rec_state, 0, eng->rec_state_bytes);
                sqf_encode_copy_buffer(
                    enc_fix, eng->pso_copy_u32,
                    eng->b_conv_slots[accepted], 0,
                    eng->b_conv_state, 0, eng->conv_state_bytes);
                if (eng->cfg.has_ple && eng->ple_state_bytes > 0ULL) {
                    sqf_encode_copy_buffer(
                        enc_fix, eng->pso_copy_u32,
                        eng->b_ple_conv_slots[accepted], 0,
                        eng->b_ple_conv_state, 0, eng->ple_state_bytes);
                }
            }

            if (eng->mtp_kv_sync_enabled && mtp_sync_safe && accepted >= 1U) {
                sqf_encode_mtp_pass(
                    enc_fix, cb_fix, eng,
                    eng->b_hc, 0,
                    eng->b_token_ids, 1U * sizeof(uint32_t),
                    mtp_base_pos + 1U, accepted, /*run_head=*/false, /*force_gemv=*/true, /*tree_mode=*/0U
                );
            }

            // Online self-healing: patch pending_token_id (e.g. prefill bonus token) + all newly committed
            // out-of-shortlist tokens into dynamic slots `31744..32767` inside async `cb_fix` (< 2 us)!
            sqf_frspec_stage_and_encode_patches(enc_fix, eng, &pending_token_id, 1U, /*record_misses=*/false);
            sqf_frspec_stage_and_encode_patches(enc_fix, eng, emitted_tokens_out, 1U + accepted, /*record_misses=*/true);

            [enc_fix endEncoding];
            [cb_fix commit];
            // Overlap tiny 40 KB `b_mtp_hc_prev` commit + MTP KV sync + FR-Spec patch asynchronously with CPU output!
            eng->pending_fix_cb = cb_fix;
        }

        if (eng->cfg.has_ple && accepted < eff_k) {
            eng->ple_prev_tokens[0] = ple_prev_after[accepted][0];
            eng->ple_prev_tokens[1] = ple_prev_after[accepted][1];
        }

        // Update 3-State SpecCtl Adaptive-K live timings & state machine!
        // On stochastic runs pass 0.0 ms so SpecCtl uses deterministic analytical timings across runs.
        if (eng->adaptive_k_enabled) {
            const double ctl_ver_ms   = is_greedy ? ver_gpu_ms   : 0.0;
            const double ctl_draft_ms = is_greedy ? draft_gpu_ms : 0.0;
            if (is_cooldown_k0) {
                eng->spec_ctl.record_ver_ms(0U, ctl_ver_ms);
                if (eng->spec_ctl.cooldown_rem > 0U) --eng->spec_ctl.cooldown_rem;
                if (eng->spec_ctl.cooldown_rem == 0U) {
                    eng->spec_ctl.state          = kSpecActiveProbe;
                    eng->spec_ctl.probe_after_cd = true;
                }
            } else {
                sqf_spec_ctl_update_spec(eng, eff_k, draft_k, accepted, ctl_draft_ms, ctl_ver_ms, draft_mean_pmax, used_ngram);
            }
        }

        eng->seq_pos     = cur_pos + 1U + accepted;
        eng->mtp_seq_pos = mtp_base_pos + 1U + accepted;
        strata_qwen_flash_gpu_penalties_record(eng, emitted_tokens_out, 1U + accepted);
        if (eng->bound_slot) {
            if (used_ngram) {
                eng->bound_slot->ngram_cooldown = (accepted < 2U) ? 2U : 0U;
            }
            sqf_ngram_append(eng->bound_slot, verify_tokens, 1U + accepted);
        }

        *num_emitted_out = 1U + accepted;
        if (num_draft_accepted_out) *num_draft_accepted_out = accepted;

        // Telemetry accounting
        eng->counters.spec_verify_steps  += 1ULL;
        eng->counters.mtp_drafted_tokens += eff_k;
        eng->counters.mtp_accepted_tokens+= accepted;

        const uint32_t comp_rat  = std::max<uint32_t>(1U, eng->cfg.indexer_compress_ratio);
        const uint64_t num_comp  = static_cast<uint64_t>((cur_pos + M) / comp_rat);
        const uint64_t blk_topk  = std::max<uint64_t>(1ULL, eng->cfg.indexer_budget / comp_rat);
        const uint64_t idx_reads = (num_comp > blk_topk)
            ? (num_comp * eng->cfg.indexer_head_dim * sizeof(uint16_t) *
               (static_cast<uint64_t>(M) * eng->cfg.num_full_attn_layers + (used_ngram ? 0ULL : eff_k)))
            : 0ULL;
        const uint64_t eff_kv = std::min<uint64_t>(cur_pos + M, eng->cfg.indexer_budget + 4U);
        const uint64_t one_moe_bytes =
            ((2ULL * eng->cfg.moe_intermediate_size * D) / 2ULL +
             2ULL * eng->cfg.moe_intermediate_size * (D / 64ULL) * 4ULL +
             (static_cast<uint64_t>(D) * eng->cfg.moe_intermediate_size) / 2ULL +
             static_cast<uint64_t>(D) * (eng->cfg.moe_intermediate_size / 64ULL) * 4ULL) *
            eng->cfg.num_experts_per_tok * num_layers;
        const uint64_t extra_moe = static_cast<uint64_t>(eff_k) * (one_moe_bytes * 7ULL / 10ULL);
        eng->counters.physical_dram_bytes_read +=
            (used_ngram ? 0ULL : static_cast<uint64_t>(eff_k) * eng->hdr0.mtp_active_bytes_per_tok) +
            eng->hdr0.active_bytes_per_tok + extra_moe +
            2ULL * eng->rec_state_bytes +
            eff_kv * eng->kv_cache_bytes_per_token +
            idx_reads;

        return 0;
    }
}

struct StrataQwenFlashCheckpoint {
    id<MTLResidencySet> residency_set;
    id<MTLBuffer>       b_arena;
    uint64_t            conv_off;
    uint64_t            conv_len;
    uint64_t            rec_off;
    uint64_t            rec_len;
    uint64_t            ple_off;
    uint64_t            ple_len;
    uint64_t            mtp_off;
    uint64_t            mtp_len;
    uint64_t            total_bytes;
    uint32_t            seq_pos;
    uint32_t            mtp_seq_pos;
    uint32_t            ple_prev_tokens[2];
    uint64_t            sample_step;
    bool                valid;
};

StrataQwenFlashCheckpoint* strata_qwen_flash_checkpoint_create(const StrataQwenFlashGpuEngine* eng) {
    if (!eng || !eng->device) return nullptr;
    @autoreleasepool {
        auto* ckpt = new StrataQwenFlashCheckpoint();
        ckpt->residency_set      = eng->residency_set;
        ckpt->seq_pos            = 0U;
        ckpt->mtp_seq_pos        = 0U;
        ckpt->ple_prev_tokens[0] = eng->cfg.ple_eos_token_id;
        ckpt->ple_prev_tokens[1] = eng->cfg.ple_eos_token_id;
        ckpt->sample_step        = 0ULL;
        ckpt->valid              = false;

        uint64_t cur = 0ULL;
        ckpt->conv_off = cur;
        ckpt->conv_len = eng->b_conv_state ? static_cast<uint64_t>([eng->b_conv_state length]) : 0ULL;
        cur = sqf_align_up_u64(cur + ckpt->conv_len, 256ULL);

        ckpt->rec_off = cur;
        ckpt->rec_len = eng->b_rec_state ? static_cast<uint64_t>([eng->b_rec_state length]) : 0ULL;
        cur = sqf_align_up_u64(cur + ckpt->rec_len, 256ULL);

        ckpt->ple_off = cur;
        ckpt->ple_len = eng->b_ple_conv_state ? static_cast<uint64_t>([eng->b_ple_conv_state length]) : 0ULL;
        cur = sqf_align_up_u64(cur + ckpt->ple_len, 256ULL);

        ckpt->mtp_off = cur;
        ckpt->mtp_len = eng->b_mtp_hc_prev ? static_cast<uint64_t>([eng->b_mtp_hc_prev length]) : 0ULL;
        cur = sqf_align_up_u64(cur + ckpt->mtp_len, 4096ULL);

        ckpt->total_bytes = std::max<uint64_t>(4096ULL, cur);
        ckpt->b_arena     = sqf_alloc_private_aligned(eng->device, static_cast<size_t>(ckpt->total_bytes));
        if (!ckpt->b_arena) {
            delete ckpt;
            return nullptr;
        }

        if (@available(macOS 15.0, *)) {
            if (ckpt->residency_set) {
                [ckpt->residency_set addAllocation:ckpt->b_arena];
                [ckpt->residency_set commit];
            }
        }
        return ckpt;
    }
}

void strata_qwen_flash_checkpoint_destroy(StrataQwenFlashCheckpoint* ckpt) {
    if (!ckpt) return;
    @autoreleasepool {
        if (@available(macOS 15.0, *)) {
            if (ckpt->residency_set && ckpt->b_arena) {
                [ckpt->residency_set removeAllocation:ckpt->b_arena];
                [ckpt->residency_set commit];
            }
        }
        ckpt->b_arena       = nil;
        ckpt->residency_set = nil;
        delete ckpt;
    }
}

int strata_qwen_flash_checkpoint_save(StrataQwenFlashGpuEngine* eng, StrataQwenFlashCheckpoint* ckpt) {
    if (!eng || !ckpt || !ckpt->b_arena) return -1;
    @autoreleasepool {
        if (!sqf_drain_pending_fix(eng)) return -1;
        ckpt->seq_pos            = eng->seq_pos;
        ckpt->mtp_seq_pos        = eng->mtp_seq_pos;
        ckpt->ple_prev_tokens[0] = eng->ple_prev_tokens[0];
        ckpt->ple_prev_tokens[1] = eng->ple_prev_tokens[1];
        ckpt->sample_step        = eng->sample_step;

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (ckpt->conv_len > 0ULL && eng->b_conv_state) {
            sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   eng->b_conv_state, 0,
                                   ckpt->b_arena, static_cast<NSUInteger>(ckpt->conv_off),
                                   ckpt->conv_len);
        }
        if (ckpt->rec_len > 0ULL && eng->b_rec_state) {
            sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   eng->b_rec_state, 0,
                                   ckpt->b_arena, static_cast<NSUInteger>(ckpt->rec_off),
                                   ckpt->rec_len);
        }
        if (ckpt->ple_len > 0ULL && eng->b_ple_conv_state) {
            sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   eng->b_ple_conv_state, 0,
                                   ckpt->b_arena, static_cast<NSUInteger>(ckpt->ple_off),
                                   ckpt->ple_len);
        }
        if (ckpt->mtp_len > 0ULL && eng->b_mtp_hc_prev) {
            sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   eng->b_mtp_hc_prev, 0,
                                   ckpt->b_arena, static_cast<NSUInteger>(ckpt->mtp_off),
                                   ckpt->mtp_len);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        ckpt->valid = true;
        return 0;
    }
}

int strata_qwen_flash_checkpoint_restore(StrataQwenFlashGpuEngine* eng, const StrataQwenFlashCheckpoint* ckpt) {
    if (!eng || !ckpt || !ckpt->b_arena || !ckpt->valid) return -1;
    @autoreleasepool {
        if (!sqf_drain_pending_fix(eng)) return -1;
        eng->seq_pos            = ckpt->seq_pos;
        eng->mtp_seq_pos        = ckpt->mtp_seq_pos;
        eng->ple_prev_tokens[0] = ckpt->ple_prev_tokens[0];
        eng->ple_prev_tokens[1] = ckpt->ple_prev_tokens[1];
        eng->sample_step        = ckpt->sample_step;
        eng->frspec_widened     = false;
        eng->shortlist_miss_mask = 0U;
        eng->shortlist_obs_count = 0U;
        eng->spec_ctl.reset();
        strata_qwen_flash_gpu_penalties_reset(eng);
        sqf_ngram_rewind(eng->bound_slot, eng->seq_pos);
        if (eng->kv_paged && eng->bound_slot) {
            sqf_paged_on_rewind(eng->bound_slot, eng->seq_pos, eng->mtp_seq_pos);
        }

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (ckpt->conv_len > 0ULL && eng->b_conv_state) {
            sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   ckpt->b_arena, static_cast<NSUInteger>(ckpt->conv_off),
                                   eng->b_conv_state, 0,
                                   ckpt->conv_len);
        }
        if (ckpt->rec_len > 0ULL && eng->b_rec_state) {
            sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   ckpt->b_arena, static_cast<NSUInteger>(ckpt->rec_off),
                                   eng->b_rec_state, 0,
                                   ckpt->rec_len);
        }
        if (ckpt->ple_len > 0ULL && eng->b_ple_conv_state) {
            sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   ckpt->b_arena, static_cast<NSUInteger>(ckpt->ple_off),
                                   eng->b_ple_conv_state, 0,
                                   ckpt->ple_len);
        }
        if (ckpt->mtp_len > 0ULL && eng->b_mtp_hc_prev) {
            sqf_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   ckpt->b_arena, static_cast<NSUInteger>(ckpt->mtp_off),
                                   eng->b_mtp_hc_prev, 0,
                                   ckpt->mtp_len);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return 0;
    }
}

uint64_t strata_qwen_flash_checkpoint_bytes(const StrataQwenFlashCheckpoint* ckpt) {
    return ckpt ? ckpt->total_bytes : 0ULL;
}

/* ============================================================================
 * Step 4 Public Slot ABI (Tier 1 O(1) RCU binding + Tier 2 paged KV)
 * ============================================================================ */

StrataQwenFlashContextSlot* strata_qwen_flash_slot_create(
    StrataQwenFlashGpuEngine* eng,
    uint32_t                  uma_kv_pool_tokens,
    const char*               nvme_swap_path
) {
    if (!eng || !eng->default_slot) return nullptr;
    if (uma_kv_pool_tokens != 0U && uma_kv_pool_tokens < STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS) {
        std::fprintf(stderr, "[strata_qwen_flash_gpu] slot pool %u < minimum %u tokens\n",
                     uma_kv_pool_tokens, STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS);
        return nullptr;
    }
    @autoreleasepool {
        id<MTLDevice> dev = eng->device;
        const StrataQwenFlashContextSlot* ds = eng->default_slot;
        const uint32_t full_slots = std::max<uint32_t>(1U, eng->cfg.num_full_attn_layers + (eng->cfg.has_mtp ? 1U : 0U));
        const uint32_t n_kv       = std::max<uint32_t>(1U, eng->cfg.num_key_value_heads);
        const uint32_t head_dim   = std::max<uint32_t>(64U, eng->cfg.head_dim);

        auto* s = new StrataQwenFlashContextSlot();
        s->eng        = eng;
        s->n_layers   = full_slots;
        s->max_blocks = eng->max_blocks_alloc;

        s->b_conv_state     = sqf_alloc_private_aligned(dev, [ds->b_conv_state length]);
        s->b_rec_state      = sqf_alloc_private_aligned(dev, [ds->b_rec_state length]);
        s->b_ple_conv_state = sqf_alloc_private_aligned(dev, [ds->b_ple_conv_state length]);
        s->b_mtp_hc_prev    = sqf_alloc_private_aligned(dev, [ds->b_mtp_hc_prev length]);
        s->b_idx_raw_k      = sqf_alloc_private_aligned(dev, [ds->b_idx_raw_k length]);
        s->bytes = [ds->b_conv_state length] + [ds->b_rec_state length] + [ds->b_ple_conv_state length] +
                   [ds->b_mtp_hc_prev length] + [ds->b_idx_raw_k length];

        if (uma_kv_pool_tokens == 0U) {
            s->paged            = false;
            s->mtp_unpaged      = false;
            s->paged_layers     = full_slots;
            s->kv_stride_tokens = eng->max_seq_alloc;
            s->kv_layer_stride  = ds->kv_layer_stride;
            s->b_k_cache        = sqf_alloc_private_aligned(dev, [ds->b_k_cache length]);
            s->b_v_cache        = sqf_alloc_private_aligned(dev, [ds->b_v_cache length]);
            s->b_idx_block_k    = sqf_alloc_private_aligned(dev, [ds->b_idx_block_k length]);
            s->b_block_loc      = eng->b_block_loc_dummy;
            s->bytes += [ds->b_k_cache length] * 2U + [ds->b_idx_block_k length];
        } else {
            uint32_t pool_tok = (uma_kv_pool_tokens / 64U) * 64U;
            pool_tok          = std::min<uint32_t>(pool_tok, eng->max_seq_alloc);
            s->paged            = true;
            s->mtp_unpaged      = (eng->cfg.has_mtp != 0U) && (eng->cfg.num_full_attn_layers > 0U);
            s->paged_layers     = s->mtp_unpaged ? eng->cfg.num_full_attn_layers : full_slots;
            s->pool_tokens      = pool_tok;
            s->pool_blocks      = pool_tok / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
            s->kv_stride_tokens = pool_tok;
            s->kv_layer_stride  = static_cast<uint64_t>(n_kv) * pool_tok * head_dim * sizeof(uint16_t);
            const size_t kv_bytes  = static_cast<size_t>(s->paged_layers) * s->kv_layer_stride;
            const size_t loc_bytes = static_cast<size_t>(s->paged_layers) * s->max_blocks * sizeof(uint32_t);
            s->b_k_cache     = sqf_alloc_shared_prefaulted(dev, kv_bytes);
            s->b_v_cache     = sqf_alloc_shared_prefaulted(dev, kv_bytes);
            s->b_idx_block_k = sqf_alloc_shared_prefaulted(dev, [ds->b_idx_block_k length]); /* 100% linear + UMA-resident */
            s->b_block_loc   = sqf_alloc_shared_prefaulted(dev, loc_bytes);
            s->bytes += kv_bytes * 2U + [ds->b_idx_block_k length] + loc_bytes;
            if (s->mtp_unpaged) {
                const size_t mtp_kv_b = static_cast<size_t>(n_kv) * eng->max_seq_alloc * head_dim * sizeof(uint16_t);
                s->b_mtp_k_cache = sqf_alloc_private_aligned(dev, mtp_kv_b);
                s->b_mtp_v_cache = sqf_alloc_private_aligned(dev, mtp_kv_b);
                s->bytes += mtp_kv_b * 2U;
            }

            s->layers.resize(s->paged_layers);
            for (auto& L : s->layers) {
                L.st.assign(s->max_blocks, kBlkUnalloc);
                L.on_nvme.assign(s->max_blocks, 0U);
                L.phys_owner.assign(s->pool_blocks, kPhysFree);
                L.u.assign(s->max_blocks, 0.0f);
                L.pin_epoch.assign(s->max_blocks, 0U);
                L.pf_epoch.assign(s->max_blocks, 0U);
                L.free_list.reserve(static_cast<size_t>(s->pool_blocks) * 2U);
                L.scratch_need.reserve(std::min<size_t>(s->pool_blocks, 1024U));
                L.scratch_missing.reserve(s->pool_blocks);
                L.scratch_cand.reserve(s->pool_blocks);
                L.scratch_io_pairs.reserve(s->pool_blocks);
                L.scratch_io_runs.reserve(kMaxIoBatchBlocks);
                L.scratch_ane_cand.reserve(1024U);
            }
            s->scratch_mgroups.resize(2U);
            for (auto& g : s->scratch_mgroups) g.need.reserve(s->pool_blocks);
            s->scratch_mine.reserve(1024U);
            s->scratch_ane_lc.resize(s->paged_layers);

            std::string path;
            if (nvme_swap_path && nvme_swap_path[0] != '\0') {
                path = nvme_swap_path;
                s->swap_fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
            } else {
                std::string tmpl = std::string([NSTemporaryDirectory() UTF8String]) + "strata_kv_swap_XXXXXX";
                std::vector<char> buf(tmpl.begin(), tmpl.end());
                buf.push_back('\0');
                s->swap_fd = ::mkstemp(buf.data());
                if (s->swap_fd >= 0) path.assign(buf.data());
            }
            std::snprintf(s->swap_path, sizeof(s->swap_path), "%s", path.c_str());
            if (s->swap_fd < 0) {
                std::fprintf(stderr, "[strata_qwen_flash_gpu] failed to open kv_swap `%s` (errno=%d)\n", path.c_str(), errno);
                delete s;
                return nullptr;
            }
            (void)fcntl(s->swap_fd, F_NOCACHE, 1);   /* bypass the unified buffer cache: true NVMe tier, 0 B RAM */
            (void)::unlink(path.c_str());            /* anonymous inode: auto-reclaimed on close / crash (0 leftovers) */
            s->probe_grp = dispatch_group_create();
            std::snprintf(s->stats.nvme_swap_path, sizeof(s->stats.nvme_swap_path), "%s (F_NOCACHE, unlinked)", s->swap_path);

            const size_t blk_bytes  = static_cast<size_t>(2U * n_kv * STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) * head_dim * sizeof(uint16_t);
            const size_t io_bytes   = sqf_align_up_u64(static_cast<uint64_t>(kMaxIoBatchBlocks) * blk_bytes, STRATA_QWEN_FLASH_PAGE_BYTES);
            const size_t ane_bytes  = sqf_align_up_u64(static_cast<uint64_t>(s->paged_layers) * kAneMaxPrefetchPerLyr * blk_bytes, STRATA_QWEN_FLASH_PAGE_BYTES);
            void* raw_io  = nullptr;
            void* raw_ane = nullptr;
            if (posix_memalign(&raw_io, STRATA_QWEN_FLASH_PAGE_BYTES, io_bytes) != 0 ||
                posix_memalign(&raw_ane, STRATA_QWEN_FLASH_PAGE_BYTES, ane_bytes) != 0) {
                if (raw_io) free(raw_io);
                if (raw_ane) free(raw_ane);
                ::close(s->swap_fd);
                delete s;
                return nullptr;
            }
            std::memset(raw_io, 0, io_bytes);
            std::memset(raw_ane, 0, ane_bytes);
            s->io_stage  = static_cast<uint8_t*>(raw_io);
            s->ane_stage = static_cast<uint8_t*>(raw_ane);
        }

        if (!s->b_conv_state || !s->b_rec_state || !s->b_ple_conv_state || !s->b_mtp_hc_prev || !s->b_idx_raw_k ||
            !s->b_k_cache || !s->b_v_cache || !s->b_idx_block_k || !s->b_block_loc ||
            (s->mtp_unpaged && (!s->b_mtp_k_cache || !s->b_mtp_v_cache))) {
            std::fprintf(stderr, "[strata_qwen_flash_gpu] failed to allocate context slot buffers\n");
            if (s->io_stage) free(s->io_stage);
            if (s->ane_stage) free(s->ane_stage);
            if (s->swap_fd >= 0) ::close(s->swap_fd);
            delete s;
            return nullptr;
        }

        NSMutableArray<id<MTLBuffer>>* bufs = [NSMutableArray arrayWithObjects:
            s->b_mtp_hc_prev, s->b_idx_raw_k, s->b_k_cache, s->b_v_cache, s->b_idx_block_k, nil];
        if (s->paged) [bufs addObject:s->b_block_loc];
        if (s->mtp_unpaged) {
            [bufs addObject:s->b_mtp_k_cache];
            [bufs addObject:s->b_mtp_v_cache];
        }
        s->rset     = sqf_create_residency_set(dev, eng->queue, bufs);
        s->rec_rset = sqf_create_residency_set(dev, eng->queue, @[
            s->b_conv_state, s->b_rec_state, s->b_ple_conv_state
        ]);

        /* Zero DeltaNet / PLE / MTP state on GPU (slot starts at seq_pos = 0) */
        id<MTLCommandBuffer> cb          = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        sqf_encode_zero_buffer(enc, eng->pso_zero_u32, s->b_conv_state, [s->b_conv_state length]);
        sqf_encode_zero_buffer(enc, eng->pso_zero_u32, s->b_rec_state, [s->b_rec_state length]);
        sqf_encode_zero_buffer(enc, eng->pso_zero_u32, s->b_ple_conv_state, [s->b_ple_conv_state length]);
        sqf_encode_zero_buffer(enc, eng->pso_zero_u32, s->b_mtp_hc_prev, [s->b_mtp_hc_prev length]);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        s->seq_pos            = 0U;
        s->mtp_seq_pos        = 0U;
        s->ple_prev_tokens[0] = eng->cfg.ple_eos_token_id;
        s->ple_prev_tokens[1] = eng->cfg.ple_eos_token_id;
        s->sample_step        = 0ULL;
        if (s->paged) {
            /* Warm F_NOCACHE kv_swap pack/pwrite/pread/unpack path once before steady-state fault tracking */
            static std::atomic<bool> s_swap_io_warmed{false};
            if (!s_swap_io_warmed.exchange(true, std::memory_order_relaxed)) {
                std::vector<std::pair<uint32_t, uint32_t>> wp = {{0U, 0U}};
                SqfPagedCounters wc{};
                (void)sqf_paged_io_batch(s, 0U, wp, /*write=*/true, s->io_stage, kMaxIoBatchBlocks, wc);
                (void)sqf_paged_io_batch(s, 0U, wp, /*write=*/false, s->io_stage, kMaxIoBatchBlocks, wc);
                (void)::ftruncate(s->swap_fd, 0);
            }
            sqf_paged_reset_tables(s);
            sqf_ane_probe_init(eng);
        }
        s->stats.mode                   = s->paged ? STRATA_COMPACT_PAGED : STRATA_COMPACT_OFF;
        s->stats.uma_kv_blocks_capacity = s->paged ? s->pool_blocks : (eng->max_seq_alloc / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS);
        return s;
    }
}

void strata_qwen_flash_bind_slot(StrataQwenFlashGpuEngine* eng, StrataQwenFlashContextSlot* slot) {
    if (!eng || !eng->default_slot) return;
    (void)sqf_drain_pending_fix(eng);
    StrataQwenFlashContextSlot* tgt = slot ? slot : eng->default_slot;
    if (tgt->eng != eng) return;
    StrataQwenFlashContextSlot* cur = eng->bound_slot;
    if (cur == tgt) return;
    if (cur) {
        if (cur->probe_pending) sqf_paged_join_probe(cur);
        cur->seq_pos            = eng->seq_pos;
        cur->mtp_seq_pos        = eng->mtp_seq_pos;
        cur->ple_prev_tokens[0] = eng->ple_prev_tokens[0];
        cur->ple_prev_tokens[1] = eng->ple_prev_tokens[1];
        cur->sample_step        = eng->sample_step;
    }
    /* O(1) RCU pointer/binding swap: 9 buffer bindings + 5 scalars (0 B copied, 0 B weights duplicated) */
    eng->b_conv_state       = tgt->b_conv_state;
    eng->b_rec_state        = tgt->b_rec_state;
    eng->b_ple_conv_state   = tgt->b_ple_conv_state;
    eng->b_k_cache          = tgt->b_k_cache;
    eng->b_v_cache          = tgt->b_v_cache;
    eng->b_idx_raw_k        = tgt->b_idx_raw_k;
    eng->b_idx_block_k      = tgt->b_idx_block_k;
    eng->b_mtp_hc_prev      = tgt->b_mtp_hc_prev;
    eng->b_block_loc        = tgt->b_block_loc;
    eng->kv_stride_tokens   = tgt->kv_stride_tokens;
    eng->kv_layer_stride    = tgt->kv_layer_stride;
    eng->kv_paged           = tgt->paged;
    eng->seq_pos            = tgt->seq_pos;
    eng->mtp_seq_pos        = tgt->mtp_seq_pos;
    eng->ple_prev_tokens[0] = tgt->ple_prev_tokens[0];
    eng->ple_prev_tokens[1] = tgt->ple_prev_tokens[1];
    eng->sample_step        = tgt->sample_step;
    eng->bound_slot         = tgt;
}

StrataQwenFlashContextSlot* strata_qwen_flash_bound_slot(const StrataQwenFlashGpuEngine* eng) {
    if (!eng || !eng->bound_slot || eng->bound_slot->is_default) return nullptr;
    return eng->bound_slot;
}

void strata_qwen_flash_slot_destroy(StrataQwenFlashContextSlot* slot) {
    if (!slot || slot->is_default) return;
    @autoreleasepool {
        StrataQwenFlashGpuEngine* eng = slot->eng;
        if (eng && eng->bound_slot == slot) {
            strata_qwen_flash_bind_slot(eng, nullptr);
        }
        sqf_paged_join_probe(slot);
        if (slot->io_stage) { free(slot->io_stage); slot->io_stage = nullptr; }
        if (slot->ane_stage) { free(slot->ane_stage); slot->ane_stage = nullptr; }
        if (slot->swap_fd >= 0) {
            ::close(slot->swap_fd);
            slot->swap_fd = -1;
        }
        if (eng) {
            sqf_detach_residency_set(eng->queue, slot->rset);
            sqf_detach_residency_set(eng->queue, slot->rec_rset);
        }
        slot->rset     = nil;
        slot->rec_rset = nil;
        delete slot;
    }
}

uint32_t strata_qwen_flash_slot_seq_pos(const StrataQwenFlashContextSlot* slot) {
    if (!slot) return 0U;
    return (slot->eng && slot->eng->bound_slot == slot) ? slot->eng->seq_pos : slot->seq_pos;
}

bool strata_qwen_flash_slot_is_paged(const StrataQwenFlashContextSlot* slot) {
    return slot ? slot->paged : false;
}

uint64_t strata_qwen_flash_slot_bytes(const StrataQwenFlashContextSlot* slot) {
    return slot ? slot->bytes : 0ULL;
}

void strata_qwen_flash_slot_get_stats(const StrataQwenFlashContextSlot* slot, StrataQwenFlashCompactionStats* out) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    if (!slot) return;
    auto* s = const_cast<StrataQwenFlashContextSlot*>(slot);
    sqf_paged_join_probe(s);
    *out = s->stats;
    out->episodes_distilled  = 0U;
    out->shadow_seq_pos      = 0U;
    out->shadow_dual_synced  = false;
    out->rcu_swaps_completed = 0U;
    out->last_switchover_ms  = 0.0;
    out->max_user_stall_ms   = 0.0;
    const uint32_t pos       = strata_qwen_flash_slot_seq_pos(slot);
    out->logical_seq_pos     = pos;
    out->logical_blocks      = (pos + STRATA_QWEN_FLASH_KV_BLOCK_TOKENS - 1U) / STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
    if (s->paged) {
        uint32_t max_res = 0U;
        uint32_t paged_now = 0U;
        for (const auto& L : s->layers) {
            max_res    = std::max<uint32_t>(max_res, L.resident);
            paged_now += L.paged_count;
        }
        out->uma_kv_blocks_used    = max_res;
        out->nvme_blocks_now_paged = paged_now;
    } else {
        out->uma_kv_blocks_used = out->logical_blocks;
    }
    out->ane_probe_us_avg = (s->stats.ane_probe_calls > 0ULL)
        ? (s->ane_us_total / static_cast<double>(s->stats.ane_probe_calls)) : 0.0;
}

void strata_qwen_flash_slot_reset_stats(StrataQwenFlashContextSlot* slot) {
    if (!slot) return;
    sqf_paged_join_probe(slot);
    const uint32_t mode = slot->stats.mode;
    const uint32_t cap  = slot->stats.uma_kv_blocks_capacity;
    char path[256];
    std::memcpy(path, slot->stats.nvme_swap_path, sizeof(path));
    std::memset(&slot->stats, 0, sizeof(slot->stats));
    slot->stats.mode                   = mode;
    slot->stats.uma_kv_blocks_capacity = cap;
    std::memcpy(slot->stats.nvme_swap_path, path, sizeof(path));
    slot->ane_us_total = 0.0;
}

int strata_qwen_flash_ane_probe_ready(const StrataQwenFlashGpuEngine* eng, char* note_out, size_t note_cap) {
    if (!eng) return 0;
    const char* note = eng->ane ? eng->ane->note.c_str() : "ANE lookahead probe not built yet (created with the first paged slot)";
    if (note_out && note_cap > 0U) std::snprintf(note_out, note_cap, "%s", note);
    return (eng->ane && eng->ane->ready) ? 1 : 0;
}

namespace {

bool sqf_io_all(int fd, void* buf, size_t n, bool write) {
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

bool sqf_blit_copy(StrataQwenFlashGpuEngine* eng, id<MTLBuffer> src, NSUInteger src_off,
                   id<MTLBuffer> dst, NSUInteger dst_off, size_t len) {
    if (len == 0U) return true;
    id<MTLCommandBuffer> cb = [eng->queue commandBuffer];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:src sourceOffset:src_off toBuffer:dst destinationOffset:dst_off size:len];
    [bl endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return [cb status] != MTLCommandBufferStatusError;
}

bool sqf_blit_buf_io(StrataQwenFlashGpuEngine* eng, id<MTLBuffer> gpu_buf, NSUInteger off, size_t len,
                     id<MTLBuffer> stage, int fd, bool save) {
    if (len == 0U) return true;
    const size_t cap = [stage length];
    uint8_t* sp = static_cast<uint8_t*>([stage contents]);
    size_t done = 0U;
    while (done < len) {
        const size_t n = std::min<size_t>(cap, len - done);
        if (!save && !sqf_io_all(fd, sp, n, false)) return false;
        if (save) {
            if (!sqf_blit_copy(eng, gpu_buf, off + done, stage, 0U, n)) return false;
            if (!sqf_io_all(fd, sp, n, true)) return false;
        } else {
            if (!sqf_blit_copy(eng, stage, 0U, gpu_buf, off + done, n)) return false;
        }
        done += n;
    }
    return true;
}

struct SqfStateHdr {
    uint64_t magic;
    uint32_t seq_pos;
    uint32_t mtp_seq_pos;
    uint32_t ple_prev0;
    uint32_t ple_prev1;
    uint32_t full_slots;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t idx_hd;
    uint64_t sample_step;
    uint64_t conv_bytes;
    uint64_t rec_bytes;
    uint64_t ple_bytes;
    uint64_t mtp_hc_bytes;
};
constexpr uint64_t kSqfStateMagic = 0x31565346514D5453ULL; /* "STMQFSV1" */

} // namespace

int strata_qwen_flash_state_save_fd(StrataQwenFlashGpuEngine* eng, int fd, uint64_t* bytes_written_out) {
    if (!eng || !eng->bound_slot || fd < 0) return -1;
    @autoreleasepool {
        if (!sqf_drain_pending_fix(eng)) return -1;
        StrataQwenFlashContextSlot* s = eng->bound_slot;
        sqf_paged_join_probe(s);

        const uint32_t full_slots = std::max<uint32_t>(1U, eng->cfg.num_full_attn_layers + (eng->cfg.has_mtp ? 1U : 0U));
        const uint32_t n_kv       = std::max<uint32_t>(1U, eng->cfg.num_key_value_heads);
        const uint32_t head_dim   = std::max<uint32_t>(64U, eng->cfg.head_dim);
        const uint32_t idx_hd     = std::max<uint32_t>(64U, eng->cfg.indexer_head_dim);
        const size_t   blk_toks   = STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
        const size_t   chunk_b    = blk_toks * head_dim * sizeof(uint16_t);
        const size_t   blk_bytes  = 2U * n_kv * chunk_b;

        constexpr size_t kStageBytes = 16ULL << 20;
        if (blk_bytes == 0U || blk_bytes > kStageBytes) return -2;
        const uint32_t kStageBatchBlks = std::max<uint32_t>(
            1U, std::min<uint32_t>(256U, static_cast<uint32_t>(kStageBytes / blk_bytes)));
        id<MTLBuffer> stage = sqf_alloc_shared_prefaulted(eng->device, kStageBytes);
        if (!stage) return -2;

        SqfStateHdr sh{
            kSqfStateMagic,
            eng->seq_pos,
            eng->mtp_seq_pos,
            eng->ple_prev_tokens[0],
            eng->ple_prev_tokens[1],
            full_slots,
            n_kv,
            head_dim,
            idx_hd,
            eng->sample_step,
            static_cast<uint64_t>([eng->b_conv_state length]),
            static_cast<uint64_t>([eng->b_rec_state length]),
            static_cast<uint64_t>([eng->b_ple_conv_state length]),
            static_cast<uint64_t>([eng->b_mtp_hc_prev length])
        };
        if (!sqf_io_all(fd, &sh, sizeof(sh), true)) return -3;
        uint64_t tot = sizeof(sh);

        if (!sqf_blit_buf_io(eng, eng->b_conv_state, 0U, sh.conv_bytes, stage, fd, true) ||
            !sqf_blit_buf_io(eng, eng->b_rec_state, 0U, sh.rec_bytes, stage, fd, true) ||
            !sqf_blit_buf_io(eng, eng->b_ple_conv_state, 0U, sh.ple_bytes, stage, fd, true) ||
            !sqf_blit_buf_io(eng, eng->b_mtp_hc_prev, 0U, sh.mtp_hc_bytes, stage, fd, true)) {
            return -4;
        }
        tot += sh.conv_bytes + sh.rec_bytes + sh.ple_bytes + sh.mtp_hc_bytes;

        std::vector<uint8_t> pack_buf(
            std::min<size_t>(kStageBatchBlks, std::max<size_t>(1U, (eng->max_seq_alloc + 3U) / 4U)) * blk_bytes);
        for (uint32_t ls = 0U; ls < full_slots; ++ls) {
            const uint32_t valid_pos = (ls + 1U == full_slots && eng->cfg.has_mtp) ? eng->mtp_seq_pos : eng->seq_pos;
            const uint32_t num_blks  = (valid_pos + 3U) / 4U;
            const uint32_t comp_blks = valid_pos / 4U;

            const size_t raw_bytes = static_cast<size_t>(valid_pos) * idx_hd * sizeof(uint16_t);
            const size_t blk_k_b   = static_cast<size_t>(comp_blks) * idx_hd * sizeof(uint16_t);
            if (!sqf_blit_buf_io(eng, eng->b_idx_raw_k, static_cast<NSUInteger>(ls * eng->idx_raw_layer_stride),
                                 raw_bytes, stage, fd, true) ||
                !sqf_blit_buf_io(eng, eng->b_idx_block_k, static_cast<NSUInteger>(ls * eng->idx_blk_layer_stride),
                                 blk_k_b, stage, fd, true)) {
                return -5;
            }
            tot += raw_bytes + blk_k_b;
            if (num_blks == 0U) continue;

            const bool ls_unpaged = (!s->paged) || (s->mtp_unpaged && ls >= s->paged_layers);
            if (ls_unpaged) {
                /* Batch-blit up to `kStageBatchBlks` blocks at a time per head into canonical block order */
                const bool     use_mtp_buf = (s->paged && s->mtp_unpaged && ls >= s->paged_layers);
                id<MTLBuffer>  k_src       = use_mtp_buf ? s->b_mtp_k_cache : eng->b_k_cache;
                id<MTLBuffer>  v_src       = use_mtp_buf ? s->b_mtp_v_cache : eng->b_v_cache;
                const uint64_t lyr_off     = use_mtp_buf ? 0ULL : (static_cast<uint64_t>(ls) * eng->kv_layer_stride);
                const uint32_t kv_str      = use_mtp_buf ? eng->max_seq_alloc : eng->kv_stride_tokens;
                const uint32_t kBatchBlks  = kStageBatchBlks;
                uint8_t* sp = static_cast<uint8_t*>([stage contents]);
                for (uint32_t b0 = 0U; b0 < num_blks; b0 += kBatchBlks) {
                    const uint32_t cnt     = std::min<uint32_t>(kBatchBlks, num_blks - b0);
                    const size_t   tok_off = static_cast<size_t>(b0) * blk_toks * head_dim * sizeof(uint16_t);
                    const size_t   h_bytes = static_cast<size_t>(cnt) * chunk_b;
                    for (uint32_t h = 0U; h < n_kv; ++h) {
                        const NSUInteger src_off = static_cast<NSUInteger>(
                            lyr_off + static_cast<size_t>(h) * kv_str * head_dim * sizeof(uint16_t) + tok_off);
                        if (!sqf_blit_copy(eng, k_src, src_off, stage, h * h_bytes, h_bytes) ||
                            !sqf_blit_copy(eng, v_src, src_off, stage, (n_kv + h) * h_bytes, h_bytes)) {
                            return -6;
                        }
                    }
                    for (uint32_t bi = 0U; bi < cnt; ++bi) {
                        uint8_t* dst = pack_buf.data() + static_cast<size_t>(bi) * blk_bytes;
                        for (uint32_t h = 0U; h < 2U * n_kv; ++h) {
                            std::memcpy(dst + h * chunk_b, sp + h * h_bytes + static_cast<size_t>(bi) * chunk_b, chunk_b);
                        }
                    }
                    const size_t out_b = static_cast<size_t>(cnt) * blk_bytes;
                    if (!sqf_io_all(fd, pack_buf.data(), out_b, true)) return -6;
                    tot += out_b;
                }
            } else {
                const SqfPagedLayerState& L = s->layers[ls];
                const uint32_t* loc = sqf_loc_row(s, ls);
                const uint32_t kBatchBlks = kStageBatchBlks;
                for (uint32_t b0 = 0U; b0 < num_blks; b0 += kBatchBlks) {
                    const uint32_t cnt = std::min<uint32_t>(kBatchBlks, num_blks - b0);
                    for (uint32_t bi = 0U; bi < cnt; ++bi) {
                        const uint32_t b   = b0 + bi;
                        uint8_t*       dst = pack_buf.data() + static_cast<size_t>(bi) * blk_bytes;
                        if (L.st[b] == kBlkResident) {
                            const uint32_t phys = loc[b] & STRATA_QWEN_FLASH_KV_LOC_IDX_MASK;
                            sqf_paged_pack_block(s, ls, phys, dst);
                        } else if (L.st[b] == kBlkPaged && L.on_nvme[b]) {
                            const off_t foff = static_cast<off_t>(
                                (static_cast<uint64_t>(ls) * s->max_blocks + b) * blk_bytes);
                            if (::pread(s->swap_fd, dst, blk_bytes, foff) != static_cast<ssize_t>(blk_bytes)) return -7;
                        } else {
                            std::memset(dst, 0, blk_bytes);
                        }
                    }
                    const size_t out_b = static_cast<size_t>(cnt) * blk_bytes;
                    if (!sqf_io_all(fd, pack_buf.data(), out_b, true)) return -7;
                    tot += out_b;
                }
            }
        }
        if (bytes_written_out) *bytes_written_out = tot;
        return 0;
    }
}

int strata_qwen_flash_state_load_fd(StrataQwenFlashGpuEngine* eng, int fd, uint64_t* bytes_read_out) {
    if (!eng || !eng->bound_slot || fd < 0) return -1;
    @autoreleasepool {
        (void)sqf_drain_pending_fix(eng);
        SqfStateHdr sh{};
        if (!sqf_io_all(fd, &sh, sizeof(sh), false)) return -2;

        const uint32_t full_slots = std::max<uint32_t>(1U, eng->cfg.num_full_attn_layers + (eng->cfg.has_mtp ? 1U : 0U));
        const uint32_t n_kv       = std::max<uint32_t>(1U, eng->cfg.num_key_value_heads);
        const uint32_t head_dim   = std::max<uint32_t>(64U, eng->cfg.head_dim);
        const uint32_t idx_hd     = std::max<uint32_t>(64U, eng->cfg.indexer_head_dim);
        if (sh.magic != kSqfStateMagic || sh.full_slots != full_slots || sh.n_kv != n_kv ||
            sh.head_dim != head_dim || sh.idx_hd != idx_hd ||
            sh.seq_pos > eng->max_seq_len || sh.mtp_seq_pos > eng->max_seq_len ||
            sh.conv_bytes != [eng->b_conv_state length] || sh.rec_bytes != [eng->b_rec_state length] ||
            sh.ple_bytes != [eng->b_ple_conv_state length] || sh.mtp_hc_bytes != [eng->b_mtp_hc_prev length]) {
            return -3;
        }

        constexpr size_t kStageBytes = 16ULL << 20;
        id<MTLBuffer> stage = sqf_alloc_shared_prefaulted(eng->device, kStageBytes);
        if (!stage) return -4;

        strata_qwen_flash_gpu_reset(eng);
        StrataQwenFlashContextSlot* s = eng->bound_slot;
        uint64_t tot = sizeof(sh);

        if (!sqf_blit_buf_io(eng, eng->b_conv_state, 0U, sh.conv_bytes, stage, fd, false) ||
            !sqf_blit_buf_io(eng, eng->b_rec_state, 0U, sh.rec_bytes, stage, fd, false) ||
            !sqf_blit_buf_io(eng, eng->b_ple_conv_state, 0U, sh.ple_bytes, stage, fd, false) ||
            !sqf_blit_buf_io(eng, eng->b_mtp_hc_prev, 0U, sh.mtp_hc_bytes, stage, fd, false)) {
            return -5;
        }
        tot += sh.conv_bytes + sh.rec_bytes + sh.ple_bytes + sh.mtp_hc_bytes;

        const size_t blk_toks  = STRATA_QWEN_FLASH_KV_BLOCK_TOKENS;
        const size_t chunk_b   = blk_toks * head_dim * sizeof(uint16_t);
        const size_t blk_bytes = 2U * n_kv * chunk_b;
        if (blk_bytes == 0U || blk_bytes > kStageBytes) return -4;
        const uint32_t kStageBatchBlks = std::max<uint32_t>(
            1U, std::min<uint32_t>(256U, static_cast<uint32_t>(kStageBytes / blk_bytes)));
        std::vector<uint8_t> pack_buf(static_cast<size_t>(kStageBatchBlks) * blk_bytes);

        for (uint32_t ls = 0U; ls < full_slots; ++ls) {
            const uint32_t valid_pos = (ls + 1U == full_slots && eng->cfg.has_mtp) ? sh.mtp_seq_pos : sh.seq_pos;
            const uint32_t num_blks  = (valid_pos + 3U) / 4U;
            const uint32_t comp_blks = valid_pos / 4U;

            const size_t raw_bytes = static_cast<size_t>(valid_pos) * idx_hd * sizeof(uint16_t);
            const size_t blk_k_b   = static_cast<size_t>(comp_blks) * idx_hd * sizeof(uint16_t);
            if (!sqf_blit_buf_io(eng, eng->b_idx_raw_k, static_cast<NSUInteger>(ls * eng->idx_raw_layer_stride),
                                 raw_bytes, stage, fd, false) ||
                !sqf_blit_buf_io(eng, eng->b_idx_block_k, static_cast<NSUInteger>(ls * eng->idx_blk_layer_stride),
                                 blk_k_b, stage, fd, false)) {
                return -6;
            }
            tot += raw_bytes + blk_k_b;
            if (num_blks == 0U) continue;

            const bool ls_unpaged = (!s->paged) || (s->mtp_unpaged && ls >= s->paged_layers);
            if (ls_unpaged) {
                const bool     use_mtp_buf = (s->paged && s->mtp_unpaged && ls >= s->paged_layers);
                id<MTLBuffer>  k_dst       = use_mtp_buf ? s->b_mtp_k_cache : eng->b_k_cache;
                id<MTLBuffer>  v_dst       = use_mtp_buf ? s->b_mtp_v_cache : eng->b_v_cache;
                const uint64_t lyr_off     = use_mtp_buf ? 0ULL : (static_cast<uint64_t>(ls) * eng->kv_layer_stride);
                const uint32_t kv_str      = use_mtp_buf ? eng->max_seq_alloc : eng->kv_stride_tokens;
                const uint32_t kBatchBlks  = kStageBatchBlks;
                uint8_t* sp = static_cast<uint8_t*>([stage contents]);
                for (uint32_t b0 = 0U; b0 < num_blks; b0 += kBatchBlks) {
                    const uint32_t cnt     = std::min<uint32_t>(kBatchBlks, num_blks - b0);
                    const size_t   in_b    = static_cast<size_t>(cnt) * blk_bytes;
                    const size_t   tok_off = static_cast<size_t>(b0) * blk_toks * head_dim * sizeof(uint16_t);
                    const size_t   h_bytes = static_cast<size_t>(cnt) * chunk_b;
                    if (!sqf_io_all(fd, pack_buf.data(), in_b, false)) return -7;
                    tot += in_b;
                    for (uint32_t bi = 0U; bi < cnt; ++bi) {
                        const uint8_t* src = pack_buf.data() + static_cast<size_t>(bi) * blk_bytes;
                        for (uint32_t h = 0U; h < 2U * n_kv; ++h) {
                            std::memcpy(sp + h * h_bytes + static_cast<size_t>(bi) * chunk_b, src + h * chunk_b, chunk_b);
                        }
                    }
                    for (uint32_t h = 0U; h < n_kv; ++h) {
                        const NSUInteger dst_off = static_cast<NSUInteger>(
                            lyr_off + static_cast<size_t>(h) * kv_str * head_dim * sizeof(uint16_t) + tok_off);
                        if (!sqf_blit_copy(eng, stage, h * h_bytes, k_dst, dst_off, h_bytes) ||
                            !sqf_blit_copy(eng, stage, (n_kv + h) * h_bytes, v_dst, dst_off, h_bytes)) {
                            return -7;
                        }
                    }
                }
            } else {
                SqfPagedLayerState& L = s->layers[ls];
                uint32_t* loc = sqf_loc_row(s, ls);
                const uint32_t sink_keep = (num_blks <= s->pool_blocks)
                    ? num_blks
                    : std::min<uint32_t>(STRATA_QWEN_FLASH_COMPACT_SINK_BLOCKS, s->pool_blocks / 4U);
                const uint32_t tail_keep = (num_blks <= s->pool_blocks) ? 0U : (s->pool_blocks - sink_keep);
                uint32_t phys_cur = 0U;

                const uint32_t kBatchBlks = kStageBatchBlks;
                for (uint32_t b0 = 0U; b0 < num_blks; b0 += kBatchBlks) {
                    const uint32_t cnt  = std::min<uint32_t>(kBatchBlks, num_blks - b0);
                    const size_t   in_b = static_cast<size_t>(cnt) * blk_bytes;
                    if (!sqf_io_all(fd, pack_buf.data(), in_b, false)) return -8;
                    tot += in_b;
                    for (uint32_t bi = 0U; bi < cnt; ++bi) {
                        const uint32_t b   = b0 + bi;
                        const uint8_t* src = pack_buf.data() + static_cast<size_t>(bi) * blk_bytes;
                        const bool in_uma  = (num_blks <= s->pool_blocks) || (b < sink_keep) || (b >= num_blks - tail_keep);
                        if (in_uma && phys_cur < s->pool_blocks) {
                            const uint32_t phys = phys_cur++;
                            sqf_paged_unpack_block(s, ls, phys, src);
                            L.st[b]          = kBlkResident;
                            L.on_nvme[b]     = 0U;
                            L.phys_owner[phys] = b;
                            loc[b]           = STRATA_QWEN_FLASH_KV_LOC_UMA_BIT | phys;
                            ++L.resident;
                        } else {
                            const off_t foff = static_cast<off_t>(
                                (static_cast<uint64_t>(ls) * s->max_blocks + b) * blk_bytes);
                            if (::pwrite(s->swap_fd, src, blk_bytes, foff) != static_cast<ssize_t>(blk_bytes)) return -8;
                            L.st[b]      = kBlkPaged;
                            L.on_nvme[b] = 1U;
                            loc[b]       = b & STRATA_QWEN_FLASH_KV_LOC_IDX_MASK;
                            ++L.paged_count;
                        }
                    }
                }
                L.free_list.clear();
                for (uint32_t p = s->pool_blocks; p-- > phys_cur;) {
                    L.free_list.push_back(p);
                }
            }
        }

        eng->seq_pos            = sh.seq_pos;
        eng->mtp_seq_pos        = sh.mtp_seq_pos;
        eng->ple_prev_tokens[0] = sh.ple_prev0;
        eng->ple_prev_tokens[1] = sh.ple_prev1;
        eng->sample_step        = sh.sample_step;
        s->seq_pos              = sh.seq_pos;
        s->mtp_seq_pos          = sh.mtp_seq_pos;
        s->ple_prev_tokens[0]   = sh.ple_prev0;
        s->ple_prev_tokens[1]   = sh.ple_prev1;
        s->sample_step          = sh.sample_step;
        if (bytes_read_out) *bytes_read_out = tot;
        return 0;
    }
}

} // extern "C"


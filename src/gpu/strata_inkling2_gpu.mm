#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include "../../include/strata_inkling2.h"
#include "../../include/strata_embedded_shaders.h"
#import  "../../include/strata_metal_cache.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kMaxPrefillChunk   = 2048U; // A/B 1024 vs 2048: halves MoE DRAM reads & fills BM=128 tiles!
constexpr uint32_t kMaxSpecBatch      = 8U;    // M = 1..8 (pending + up to 7 drafted tokens)
constexpr uint32_t kMaxSpecSnapSlots  = 7U;    // SConv rollback slots 0..6 (for accepted = 0..6)
constexpr uint32_t kMaxAttnResBlocks  = 8U;
constexpr uint32_t kMaxBankBlocks     = 4U;
constexpr uint32_t kSwaRingLen        = 4096U; // 4096 > 1024 + 2048 => zero SWA KV overlap/rollback!
constexpr uint32_t kSwaCkptLen        = 1024U; // Active window saved per checkpoint
constexpr uint32_t kMaxDecodeSplits   = 36U;   // SWA (B=32 => <=33 splits) & Global (<=32 splits)
constexpr uint32_t kMtpMaxSplits      = 32U;
constexpr uint32_t kNgramHashSize     = 65536U;

constexpr uint32_t kOutF16            = 0U;
constexpr uint32_t kOutF32            = 1U;
constexpr uint32_t kOutF32Softcap     = 2U;

/* ---- Host Mirrors of Metal Shader POD Parameter Structs ---- */

struct SikPadCastParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t pad;
};

struct SikCaptureTapParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t tap_idx;
    uint32_t num_taps;
};

struct SikSwaCkptParamsHost {
    uint32_t num_layers;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t start_pos;
    uint32_t count;
    uint32_t ring_len;
    uint32_t ckpt_stride;
    uint32_t is_save;
};

struct SikEmbedParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t vocab_size;
    float    eps;
};

struct SikRmsNormParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    float    eps;
    uint32_t has_weight;
    uint32_t write_f32;
    uint32_t tap_capture;
    uint32_t tap_idx;
};

struct SikAttnResParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t num_blocks;
    uint32_t block_stride;
    uint32_t num_queries;
    float    eps;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikAttnResNormParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t num_blocks;
    uint32_t block_stride;
    float    eps;
    uint32_t write_f32;
    uint32_t tap_capture;
    uint32_t tap_idx;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SikLoopCtrlPrepParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    float    eps;
};

struct SikPassCopyParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t bank_count;
    uint32_t src_start;
    uint32_t block_stride;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SikLoopCtrlApplyParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t gate_rank;
};

struct SikLoopCtrl3OffsetsHost {
    uint64_t w_off[3];
    uint64_t s_off[3];
    uint64_t f_off[3];
    uint64_t fs_off[3];
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t gate_rank;
    uint32_t num_ctrls;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SikLoopBatch12OffsetsHost {
    uint64_t w_off[12][4];
    uint64_t s_off[12][4];
    uint32_t M_valid;
    uint32_t D;
    uint32_t gate_rank;
    uint32_t num_layers;
};

struct SikGemvBatchParamsHost {
    uint32_t M;
    uint32_t rows;
    uint32_t cols;
    uint32_t out_mode;
    float    softcap;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SikAttnQgrsMixedParamsHost {
    uint64_t w_off[8];
    uint64_t s_off[8];
    uint64_t b_off[8];
    uint32_t dtype[8];
    uint32_t rows[8];
    uint32_t M;
    uint32_t cols;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikLoopPreAttnParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t bank_count;
    uint32_t num_blocks;
    uint32_t block_stride;
    uint32_t num_attn_q;
    float    eps;
    uint32_t pad0;
};

struct SikMtpQkvMixedParamsHost {
    uint64_t w_off[3];
    uint64_t s_off[3];
    uint64_t b_off[3];
    uint32_t dtype[3];
    uint32_t rows[3];
    uint32_t cols;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikBlkSumParamsHost {
    uint32_t M;
    uint32_t cols;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikNaxGemmParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t N_logical;
    uint32_t N_padded;
    uint32_t K;
    uint32_t out_mode;
    float    softcap;
    uint32_t pad0;
};

struct SikQuantI8ParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t K;
    uint32_t emit_xsum;
    uint32_t num_slices;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct SikSconvParamsHost {
    uint32_t M;
    uint32_t channels;
    uint32_t save_snap_mask;
    uint32_t res_mode;
    uint32_t head_dim;
    uint32_t n_kv;
    uint32_t v_state_off_f4;
    uint32_t top_k;
};

struct SikWoSconvParamsHost {
    uint32_t M;
    uint32_t rows;
    uint32_t cols;
    uint32_t save_snap_mask;
    uint32_t res_mode;
    uint32_t snap_stride_f4;
    uint32_t pad1;
    uint32_t pad2;
};

struct SikAttnPrepParamsHost {
    uint32_t M_valid;
    uint32_t M_pad16;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t d_rel;
    uint32_t seq_pos_start;
    uint32_t kv_stride;
    uint32_t is_local;
    uint32_t save_snap_mask;
    uint32_t v_state_off_f4;
    uint32_t kv_zero_tail;
    float    qk_clip;
    float    rel_clip;
    float    eps;
    float    log_alpha;
    float    log_n_floor;
    float    attn_scale;
    uint32_t kv_dtype;
    uint32_t snap_stride_f4;
};

struct SikAttnParamsHost {
    uint32_t M_valid;
    uint32_t M_pad16;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t d_rel;
    uint32_t rel_extent;
    uint32_t swa_win;
    uint32_t is_local;
    uint32_t seq_pos_start;
    uint32_t m_offset;
    uint32_t kv_stride;
    uint32_t max_splits;
    uint32_t chunk_size;
    uint32_t c_batch_first;
    float    attn_scale;
};

struct SikMoeRouterParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t num_experts;
    uint32_t top_k;
    float    route_scale;
    uint32_t inter_size;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikMoeRouterRmsParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t num_experts;
    uint32_t top_k;
    float    eps;
    uint32_t n_sq_tiles;
    uint32_t emit_m1_pairs;
    uint32_t pad0;
};

struct SikMoeW13M1ParamsHost {
    uint32_t D;
    uint32_t inter;
    uint32_t num_experts;
    uint32_t top_k;
    float    route_scale;
    float    gate_softcap;
    float    up_softcap;
    uint32_t pad0;
};

struct SikSconvAttnM1ParamsHost {
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t d_rel;
    uint32_t rel_extent;
    uint32_t swa_win;
    uint32_t seq_pos;
    uint32_t kv_stride;
    uint32_t is_local;
    uint32_t v_state_off_f4;
    float    qk_clip;
    float    rel_clip;
    float    eps;
    float    log_alpha;
    float    log_n_floor;
    float    attn_scale;
    uint32_t chunk_size;
    uint32_t c_batch_first;
    uint32_t max_splits;
    uint32_t pad0;
};

struct SikMoeExecParamsHost {
    uint32_t M_valid;
    uint32_t top_k;
    uint32_t D;
    uint32_t intermediate_size;
    uint32_t num_experts;
    uint32_t tile_cap;
    float    gate_softcap;
    float    up_softcap;
};

struct SikMtpFoldParamsHost {
    uint32_t V;
    uint32_t D;
    uint32_t R;
    uint32_t unembed_dtype;
};

struct SikMtpFuseParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t alpha_row;
    float    eps;
    uint32_t write_f32;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikMtpCtxKvParamsHost {
    uint32_t M_valid;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t pos_start;
    uint32_t kv_stride;
    uint32_t is_local;
    float    eps;
    uint32_t pad0;
};

struct SikMtpInitBlockParamsHost {
    uint32_t block_size;
    uint32_t D;
    uint32_t vocab_size;
    float    eps;
};

struct SikMtpBlockPrepParamsHost {
    uint32_t block_size;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t pos_start;
    float    eps;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikMtpAttnParamsHost {
    uint32_t block_size;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t ctx_start;
    uint32_t ctx_end;
    uint32_t kv_stride;
    uint32_t is_local;
    uint32_t max_splits;
    float    attn_scale;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikMtpAddNormParamsHost {
    uint32_t M_valid;
    uint32_t D;
    float    eps;
    uint32_t write_f32;
};

struct SikMtpSwigluMixedParamsHost {
    uint64_t wg_off;
    uint64_t wg_s_off;
    uint64_t wg_b_off;
    uint64_t wu_off;
    uint64_t wu_s_off;
    uint64_t wu_b_off;
    uint32_t dtype;
    uint32_t M;
    uint32_t rows;
    uint32_t cols;
};

struct SikMtpRnnPrepParamsHost {
    uint32_t rank;
    uint32_t D;
    uint32_t step_k;
    uint32_t vocab_pad;
};

struct SikMtpFoldAddParamsHost {
    uint32_t V;
    uint32_t rank;
    float    softcap;
    uint32_t compute_pmax;
};

struct SikLogitPenaltyParamsHost {
    uint32_t M;
    uint32_t vocab_size;
    uint32_t logits_stride;
    uint32_t num_hist_ids;
    uint32_t extra_mode;
    uint32_t extra_count;
    float    presence_penalty;
    float    frequency_penalty;
    float    repetition_penalty;
    uint32_t use_shortlist;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t drop_toks[8];
};

struct SikSampleParamsHost {
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

struct SikBpMetaHost {
    uint32_t cand_count;
    uint32_t overflow;
    float    T_raw;
    float    T_pen;
    uint32_t ind_stage2[4];
    uint32_t ind_cand_s1[4];
    uint32_t ind_cand_s2[4];
    uint32_t ind_fb_gemv[4];
    uint32_t ind_fb_pen[4];
    uint32_t ind_fb_s1[4];
    uint32_t ind_fb_s2[4];
};

struct SikBpParamsHost {
    uint32_t V;
    uint32_t D;
    float    softcap;
    uint32_t top_k;
    uint32_t num_seeds;
    uint32_t has_pen;
    float    pres_pen;
    float    freq_pen;
    float    rep_pen;
    uint32_t pen_tgs;
    uint32_t is_greedy;
    float    inv_temp;
    float    top_p;
    uint32_t rng_lo;
    uint32_t rng_hi;
    uint32_t pad0;
};

struct SikTreeMetaHost {
    uint32_t M;
    uint32_t max_nodes;
    uint32_t is_linear;
    uint32_t active_parent;
    uint32_t parent[8];
    uint32_t depth[8];
    uint32_t anc_mask[8];
    uint32_t path_packed[8];
    uint32_t p2[8];
    uint32_t p3[8];
    uint32_t tok_id[8];
    float    node_prob[8];
    float    cum_prob[8];
    uint32_t sampled_tok[8];
    uint32_t pool_count;
    uint32_t beam_width;
    uint32_t pool_parent[32];
    uint32_t pool_tok[32];
    float    pool_cond_p[32];
    float    pool_cum_p[32];
    uint32_t pool_used[32];
};

struct SikTreeInitParamsHost {
    uint32_t max_nodes;
    uint32_t beam_width;
    uint32_t rank;
    uint32_t pad0;
};

struct SikTreeCompactParamsHost {
    uint32_t num_local_visits;
    uint32_t num_global_visits;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t old_pos;
    uint32_t swa_stride;
    uint32_t glob_stride;
    uint32_t kv_dtype;
    uint64_t glob_layer_stride_b;
    uint64_t glob_q8_data_b;
    uint32_t path_len;
    uint32_t D;
    uint32_t path[8];
};

struct SikChunkImageCopy {
    uint32_t   dst_row;
    uint32_t   num_rows;
    NSUInteger src_byte_off;
};

static_assert(sizeof(SikLogitPenaltyParamsHost) == 80U, "SikLogitPenaltyParamsHost size mismatch");
static_assert(sizeof(SikSampleParamsHost) == 48U,       "SikSampleParamsHost size mismatch");
static_assert(sizeof(SikAttnPrepParamsHost) == 80U,     "SikAttnPrepParamsHost size mismatch");
static_assert(sizeof(SikAttnParamsHost) == 64U,         "SikAttnParamsHost size mismatch");
static_assert(sizeof(SikBpMetaHost) == 128U,            "SikBpMetaHost size mismatch");
static_assert(sizeof(SikBpParamsHost) == 64U,           "SikBpParamsHost size mismatch");
static_assert(sizeof(SikTreeMetaHost) == 984U,          "SikTreeMetaHost size mismatch");
static_assert(sizeof(SikTreeInitParamsHost) == 16U,     "SikTreeInitParamsHost size mismatch");
static_assert(sizeof(SikTreeCompactParamsHost) == 88U,  "SikTreeCompactParamsHost size mismatch");

struct SikBatchSlotMetaHost {
    uint64_t swa_k_addr;
    uint64_t swa_v_addr;
    uint64_t glob_k_addr;
    uint64_t glob_v_addr;
    uint64_t sconv_addr;
    uint32_t seq_pos;
    uint32_t pad0;
};

struct SikBatchAttnPrepParamsHost {
    uint32_t B;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t d_rel;
    uint32_t kv_stride;
    uint32_t is_local;
    uint32_t kv_dtype;
    uint32_t sconv_k_off_f4;
    uint32_t v_state_off_f4;
    uint32_t pad0;
    uint32_t pad1;
    uint64_t kv_lyr_off;
    uint64_t kv_q8_data_b;
    float    qk_clip;
    float    rel_clip;
    float    eps;
    float    log_alpha;
    float    log_n_floor;
    float    pad2;
};

struct SikBatchAttnParamsHost {
    uint32_t B;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t d_rel;
    uint32_t rel_extent;
    uint32_t swa_win;
    uint32_t is_local;
    uint32_t kv_stride;
    uint32_t max_splits;
    uint32_t skip_under_256;
    uint32_t pad0;
    uint64_t kv_lyr_off;
    uint64_t kv_q8_data_b;
    float    attn_scale;
    uint32_t pad1;
};

struct SikBatchSconvParamsHost {
    uint32_t B;
    uint32_t channels;
    uint32_t res_mode;
    uint32_t sconv_off_f4;
    uint32_t top_k;
    uint32_t cols;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikBatchPenItemHost {
    uint32_t pen_off;
    uint32_t pen_len;
    float    pres_pen;
    float    freq_pen;
    float    rep_pen;
    uint32_t pad0;
};

struct SikBatchPenParamsHost {
    uint32_t B_chunk;
    uint32_t vocab_size;
    uint32_t item_offset;
    uint32_t pad0;
};

struct SikBatchSampleItemHost {
    uint32_t greedy;
    float    inv_temp;
    uint32_t top_k;
    float    top_p;
    uint32_t rng_lo;
    uint32_t rng_hi;
    uint32_t pad0;
    uint32_t pad1;
};

struct SikBatchSampleParamsHost {
    uint32_t B_chunk;
    uint32_t vocab_size;
    uint32_t item_offset;
    uint32_t pad0;
};

static_assert(sizeof(SikBatchSlotMetaHost) == 48U,       "SikBatchSlotMetaHost size mismatch");
static_assert(sizeof(SikBatchAttnPrepParamsHost) == 88U, "SikBatchAttnPrepParamsHost size mismatch");
static_assert(sizeof(SikBatchAttnParamsHost) == 72U,     "SikBatchAttnParamsHost size mismatch");
static_assert(sizeof(SikBatchSconvParamsHost) == 32U,    "SikBatchSconvParamsHost size mismatch");
static_assert(sizeof(SikBatchPenItemHost) == 24U,        "SikBatchPenItemHost size mismatch");
static_assert(sizeof(SikBatchPenParamsHost) == 16U,      "SikBatchPenParamsHost size mismatch");
static_assert(sizeof(SikBatchSampleItemHost) == 32U,     "SikBatchSampleItemHost size mismatch");
static_assert(sizeof(SikBatchSampleParamsHost) == 16U,   "SikBatchSampleParamsHost size mismatch");

inline uint64_t sik_align_up_u64(uint64_t v, uint64_t a) {
    return (v + a - 1ULL) & ~(a - 1ULL);
}

inline void sik_slice_sub_offsets(
    const StrataInkling2TensorSlice& sl,
    uint64_t&                        w_off,
    uint64_t&                        s_off,
    uint64_t&                        b_off
) {
    w_off = sl.offset;
    s_off = sl.offset;
    b_off = sl.offset;
    if (sl.dtype == STRATA_INKLING2_DTYPE_Q8_0) {
        uint64_t wb = 0ULL, sb = 0ULL;
        strata_inkling2_q8_layout(sl.shape0, sl.shape1, &wb, &sb);
        s_off = sl.offset + wb;
    } else if (sl.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
        uint64_t wb = 0ULL, sb = 0ULL, bb = 0ULL;
        strata_inkling2_q4a_layout(sl.shape0, sl.shape1, &wb, &sb, &bb);
        s_off = sl.offset + wb;
        b_off = sl.offset + wb + sb;
    }
}

inline uint32_t sik_host_attn_chunk_size(uint32_t q_pos, uint32_t is_local) {
    if (q_pos < 160U)    return 16U;
    if (q_pos < 512U)    return 32U;
    if (is_local != 0U)  return 64U;
    if (q_pos < 2048U)   return 64U;
    if (q_pos < 4096U)   return 128U;
    if (q_pos < 8192U)   return 256U;
    if (q_pos < 16384U)  return 512U;
    if (q_pos < 32768U)  return 1024U;
    if (q_pos < 65536U)  return 2048U;
    if (q_pos < 262144U) return 8192U;
    if (q_pos < 524288U) return 16384U;
    return 32768U;
}

inline uint32_t sik_compute_mtp_splits(uint32_t total_kv, uint32_t max_splits) {
    const uint32_t target = std::clamp<uint32_t>((total_kv + 31U) >> 5U, 1U, max_splits);
    const uint32_t kps    = ((total_kv + target - 1U) / target + 15U) & ~15U;
    return std::max<uint32_t>(1U, (total_kv + kps - 1U) / kps);
}

inline uint64_t sik_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30ULL)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27ULL)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31ULL);
}

inline float sik_host_rand_u01(uint64_t seed, uint64_t step) {
    const uint64_t z   = sik_splitmix64(seed + step * 0x9E3779B97F4A7C15ULL);
    const uint32_t u24 = static_cast<uint32_t>(z >> 40ULL) & 0xFFFFFFU;
    return (static_cast<float>(u24) + 0.5f) * (1.0f / 16777216.0f);
}

inline float sik_lookup_sparse_prob(
    uint32_t        tok_id,
    const uint32_t* ids,
    const float*    probs,
    uint32_t        keep_k
) {
    for (uint32_t i = 0U; i < keep_k; ++i) {
        if (ids[i] == tok_id) return probs[i];
    }
    return 0.0f;
}

std::string sik_read_text_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

std::string sik_resolve_sibling_path(const char* rel_from_src_gpu) {
    std::string self_file = __FILE__;
    const size_t slash = self_file.find_last_of('/');
    if (slash != std::string::npos) {
        std::string candidate = self_file.substr(0, slash + 1) + rel_from_src_gpu;
        std::ifstream test(candidate);
        if (test.good()) return candidate;
    }
    return std::string("src/gpu/") + rel_from_src_gpu;
}

id<MTLBuffer> sik_alloc_shared_prefaulted(id<MTLDevice> dev, size_t bytes) {
    const size_t aligned = std::max<size_t>(4096U, (bytes + 4095U) & ~static_cast<size_t>(4095U));
    id<MTLBuffer> buf = [dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
    if (buf) std::memset([buf contents], 0, aligned);
    return buf;
}

id<MTLBuffer> sik_alloc_private_aligned(id<MTLDevice> dev, size_t bytes) {
    const size_t aligned = std::max<size_t>(4096U, (bytes + 4095U) & ~static_cast<size_t>(4095U));
    const size_t max_len = static_cast<size_t>([dev maxBufferLength]);
    if (max_len > 0U && aligned > max_len) {
        std::fprintf(stderr,
                     "[strata_inkling2_gpu] Buffer request %.2f GiB exceeds device maxBufferLength %.2f GiB "
                     "(try `--kv-cache q8` or smaller `--max-seq-len`)\n",
                     static_cast<double>(aligned) / (1024.0 * 1024.0 * 1024.0),
                     static_cast<double>(max_len) / (1024.0 * 1024.0 * 1024.0));
        return nil;
    }
    return [dev newBufferWithLength:aligned options:MTLResourceStorageModePrivate];
}

void sik_encode_zero_buffer(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState>  pso_zero,
    id<MTLBuffer>                buf,
    size_t                       byte_len
) {
    const uint32_t count_u4 = static_cast<uint32_t>((byte_len + 15ULL) / 16ULL);
    if (count_u4 == 0U || !buf) return;
    [enc setComputePipelineState:pso_zero];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&count_u4 length:sizeof(uint32_t) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((count_u4 + 255U) / 256U, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sik_encode_copy_buffer(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState>  pso_copy,
    id<MTLBuffer>                src,
    NSUInteger                   src_off,
    id<MTLBuffer>                dst,
    NSUInteger                   dst_off,
    size_t                       byte_len
) {
    const uint32_t count_u4 = static_cast<uint32_t>((byte_len + 15ULL) / 16ULL);
    if (count_u4 == 0U || !src || !dst) return;
    [enc setComputePipelineState:pso_copy];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:dst_off atIndex:1];
    [enc setBytes:&count_u4 length:sizeof(uint32_t) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((count_u4 + 255U) / 256U, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

id<MTLResidencySet> sik_create_residency_set(
    id<MTLDevice>            dev,
    id<MTLCommandQueue>      queue,
    NSArray<id<MTLBuffer>>*  buffers
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

void sik_detach_residency_set(id<MTLCommandQueue> queue, id<MTLResidencySet> rset) {
    if (!rset) return;
    if (@available(macOS 15.0, *)) {
        if (queue) [queue removeResidencySet:rset];
        [rset endResidency];
    }
}

} // namespace

/* ============================================================================
 * Context Slot & Engine Structs
 * ============================================================================ */

struct StrataInkling2ContextSlot {
    StrataInkling2GpuEngine* eng        = nullptr;
    bool                     is_default = false;

    id<MTLBuffer>       b_swa_k_cache;
    id<MTLBuffer>       b_swa_v_cache;
    id<MTLBuffer>       b_glob_k_cache;
    id<MTLBuffer>       b_glob_v_cache;
    id<MTLBuffer>       b_sconv_state;     // [48, visit_stride] float4 per channel!
    id<MTLBuffer>       b_mtp_swa_k_cache;
    id<MTLBuffer>       b_mtp_swa_v_cache;
    id<MTLBuffer>       b_mtp_glob_k_cache;
    id<MTLBuffer>       b_mtp_glob_v_cache;
    id<MTLBuffer>       b_mtp_taps_last;

    id<MTLResidencySet> rset;
    id<MTLResidencySet> sconv_rset;

    uint32_t seq_pos        = 0U;
    uint32_t mtp_ctx_pos    = 0U;
    uint64_t sample_step    = 0ULL;
    bool     mtp_synced     = true;

    // Per-Slot Sliding-Window Logit Penalty History (strictly isolated across concurrent streams!)
    std::vector<uint32_t> pen_hist;

    // Longest-match-first Rolling 5-Gram -> 4-Gram Suffix Index (0.00 ms GPU draft, 100% lossless!)
    std::vector<uint32_t> ngram_hist;
    std::vector<uint32_t> ngram5_ht;
    std::vector<uint32_t> ngram4_ht;
    uint32_t              ngram_cooldown = 0U;
};

struct StrataInkling2GpuEngine {
    id<MTLDevice>        device;
    id<MTLCommandQueue>  queue;
    id<MTLLibrary>       library;
    id<MTLResidencySet>  residency_set;
    bool                 has_nax = false;

    // Metal 4.1 Compute Pipeline States
    id<MTLComputePipelineState> pso_zero_u32;
    id<MTLComputePipelineState> pso_copy_u32;
    id<MTLComputePipelineState> pso_f32_to_f16_pad;
    id<MTLComputePipelineState> pso_capture_tap;
    id<MTLComputePipelineState> pso_swa_ring_copy;

    id<MTLComputePipelineState> pso_embed_rmsnorm;
    id<MTLComputePipelineState> pso_embed_rmsnorm_q4a;
    id<MTLComputePipelineState> pso_rmsnorm;
    id<MTLComputePipelineState> pso_rmsnorm3;
    id<MTLComputePipelineState> pso_rmsnorm1_fast;
    id<MTLComputePipelineState> pso_rmsnorm3_fast;
    id<MTLComputePipelineState> pso_attn_res;
    id<MTLComputePipelineState> pso_attn_res_rmsnorm;
    id<MTLComputePipelineState> pso_attn_res_rmsnorm_576;
    id<MTLComputePipelineState> pso_attn_res3_rmsnorm3;
    id<MTLComputePipelineState> pso_attn_res3_rmsnorm3_576;
    id<MTLComputePipelineState> pso_cross_pass_attn_res4;
    id<MTLComputePipelineState> pso_cross_pass_batch12_576;
    id<MTLComputePipelineState> pso_attn_res_prep_left_576;
    id<MTLComputePipelineState> pso_pass_boundary_copy;
    id<MTLComputePipelineState> pso_loop_pre_attn_prep3;
    id<MTLComputePipelineState> pso_attn_res_loop_prep1;

    id<MTLComputePipelineState> pso_loop_ctrl_prep;
    id<MTLComputePipelineState> pso_loop_ctrl_prep3;
    id<MTLComputePipelineState> pso_loop_ctrl_prep_i8;
    id<MTLComputePipelineState> pso_loop_ctrl_apply;
    id<MTLComputePipelineState> pso_loop_ctrl_apply_q8_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_apply3;
    id<MTLComputePipelineState> pso_loop_ctrl_apply3_q8_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_down3_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_down3_q8_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_down_right_batch12_q8_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_down3_left_q8_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_down1_left_q8_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_down_nax_f16;
    id<MTLComputePipelineState> pso_loop_ctrl_down_nax_f16_m32;
    id<MTLComputePipelineState> pso_loop_ctrl_down_nax_q8;
    id<MTLComputePipelineState> pso_loop_ctrl_down_nax_q8_m32;
    id<MTLComputePipelineState> pso_loop_ctrl_apply_nax_f16_m64;
    id<MTLComputePipelineState> pso_loop_ctrl_apply_nax_f16_m16;
    id<MTLComputePipelineState> pso_loop_ctrl_apply_nax_q8_m64;
    id<MTLComputePipelineState> pso_loop_ctrl_apply_nax_q8_m16;
    id<MTLComputePipelineState> pso_loop_ctrl_down_nax_q8_i8_m64;
    id<MTLComputePipelineState> pso_loop_ctrl_down_nax_q8_i8_m32;
    id<MTLComputePipelineState> pso_loop_ctrl_apply_nax_q8_i8_m64;
    id<MTLComputePipelineState> pso_loop_ctrl_apply_nax_q8_i8_m32;

    id<MTLComputePipelineState> pso_f16_gemv_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q8_gemv_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q4a_gemv_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q4a_gemv_xsum_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q4a_wo_sconv_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q4a_wo_sconv_m1;
    id<MTLComputePipelineState> pso_f16_row_tg128_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_q8_row_tg128_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_attn_qgrs_mixed_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_mtp_qkv_mixed;
    id<MTLComputePipelineState> pso_mtp_qkv_q4a;
    id<MTLComputePipelineState> pso_mtp_kv_ctx_mixed_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_f16_blk64_sum;
    id<MTLComputePipelineState> pso_f16_small_n_prefill;
    id<MTLComputePipelineState> pso_quant_f16_to_i8_row;
    id<MTLComputePipelineState> pso_quant_f16_to_i8_row3;
    id<MTLComputePipelineState> pso_f16_gemm_nax_64x128_sg4;
    id<MTLComputePipelineState> pso_f16_gemm_nax_64x64_sg4;
    id<MTLComputePipelineState> pso_f16_gemm_nax_32x64_sg2;
    id<MTLComputePipelineState> pso_f16_gemm_nax_16x32_sg2;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_m128;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_m64;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_m32;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_m16;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_i8i4_m64;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_i8i4_m32;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_i8_m128;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_i8_m64;
    id<MTLComputePipelineState> pso_q4a_gemm_nax_i8_m32;
    id<MTLComputePipelineState> pso_q8_gemm_nax_m128;
    id<MTLComputePipelineState> pso_q8_gemm_nax_m64;
    id<MTLComputePipelineState> pso_q8_gemm_nax_m32_n32;
    id<MTLComputePipelineState> pso_q8_gemm_nax_m16_n32;
    id<MTLComputePipelineState> pso_q8_gemm_nax_i8_m64;
    id<MTLComputePipelineState> pso_q8_gemm_nax_i8_m32;

    id<MTLComputePipelineState> pso_sconv_static;
    id<MTLComputePipelineState> pso_sconv_static_prefill;
    id<MTLComputePipelineState> pso_sconv_save_tail;
    id<MTLComputePipelineState> pso_sconv_static_kv_prefill;
    id<MTLComputePipelineState> pso_sconv_dynamic_kv_prefill;
    id<MTLComputePipelineState> pso_sconv_save_tail_kv;
    id<MTLComputePipelineState> pso_moe_combine_sconv;

    id<MTLComputePipelineState> pso_sconv_kv_qkvr_prep;
    id<MTLComputePipelineState> pso_attn_qkvr_prep_prefill;
    id<MTLComputePipelineState> pso_attn_decode_splitkv_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_attn_decode_single_gqa3_sg8;
    id<MTLComputePipelineState> pso_sconv_attn_single_gqa3_m1;
    id<MTLComputePipelineState> pso_sconv_attn_split_gqa3_m1;
    id<MTLComputePipelineState> pso_attn_decode_split_gqa3_reg;
    id<MTLComputePipelineState> pso_attn_decode_splitkv_q8_m[2];
    id<MTLComputePipelineState> pso_attn_decode_q8_reg_m[2];
    id<MTLComputePipelineState> pso_attn_decode_nax;
    id<MTLComputePipelineState> pso_attn_decode_nax_q8;
    id<MTLComputePipelineState> pso_attn_decode_reduce;
    id<MTLComputePipelineState> pso_attn_prefill_nax;
    id<MTLComputePipelineState> pso_attn_prefill_nax_q8;

    id<MTLComputePipelineState> pso_moe_router_f32_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_moe_router_f32_m1;
    id<MTLComputePipelineState> pso_moe_router_f32_rms_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_moe_router_f32_prefill;
    id<MTLComputePipelineState> pso_moe_topk_sigmoid;
    id<MTLComputePipelineState> pso_moe_bin_tokens_quad;
    id<MTLComputePipelineState> pso_moe_bin_tokens_grouped;
    id<MTLComputePipelineState> pso_moe_gather_x_f16;
    id<MTLComputePipelineState> pso_moe_gather_x_i8;
    id<MTLComputePipelineState> pso_moe_topk_bin_small;
    id<MTLComputePipelineState> pso_moe_topk_m1;
    id<MTLComputePipelineState> pso_moe_nvfp4_w13_b[4]; // max_b in {1, 2, 4, 8}
    id<MTLComputePipelineState> pso_moe_w13_nvfp4_m1_topk;
    id<MTLComputePipelineState> pso_moe_nvfp4_w2_b[4];
    id<MTLComputePipelineState> pso_moe_w2_nvfp4_m1_fused_sconv;
    id<MTLComputePipelineState> pso_moe_mxfp8_w13_b[4];
    id<MTLComputePipelineState> pso_moe_mxfp8_w2_b[4];
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w13_m16_sg2[2]; // [0]=gate, [1]=up+SiTU
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w13_m32[2];
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w13_m64[2];
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w13_m128[2];
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w2_m16_sg2;
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w2_m32;
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w2_m64;
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w2_m128;
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w13_m16_sg2[2];
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w13_m32[2];
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w13_m64[2];
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w13_m128[2];
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w2_m16_sg2;
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w2_m32;
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w2_m64;
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w2_m128;
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w13_i8_m16[2];
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w13_i8_m32[2];
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w13_i8_m64[2];
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w13_i8_m128[2];
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w2_i8_m16;
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w2_i8_m32;
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w2_i8_m64;
    id<MTLComputePipelineState> pso_moe_prefill_nvfp4_w2_i8_m128;
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w13_i8_m16[2];
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w13_i8_m32[2];
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w13_i8_m64[2];
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w13_i8_m128[2];
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w2_i8_m16;
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w2_i8_m32;
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w2_i8_m64;
    id<MTLComputePipelineState> pso_moe_prefill_mxfp8_w2_i8_m128;
    id<MTLComputePipelineState> pso_moe_combine;
    id<MTLComputePipelineState> pso_moe_combine_f16;

    id<MTLComputePipelineState> pso_mtp_fold_head_nax;
    id<MTLComputePipelineState> pso_mtp_fuse_taps_norm;
    id<MTLComputePipelineState> pso_mtp_ctx_kv_prep;
    id<MTLComputePipelineState> pso_mtp_init_block;
    id<MTLComputePipelineState> pso_mtp_init_block_q4a;
    id<MTLComputePipelineState> pso_mtp_block_qk_prep;
    id<MTLComputePipelineState> pso_mtp_block_attn_splitkv;
    id<MTLComputePipelineState> pso_mtp_block_attn_reduce;
    id<MTLComputePipelineState> pso_mtp_add_rmsnorm;
    id<MTLComputePipelineState> pso_mtp_gate_up_silu_b8_mixed;
    id<MTLComputePipelineState> pso_mtp_gate_up_silu_b8_q4a;
    id<MTLComputePipelineState> pso_mtp_rnn_prep;
    id<MTLComputePipelineState> pso_mtp_rnn_gate;
    id<MTLComputePipelineState> pso_mtp_fold_add_softcap;
    id<MTLComputePipelineState> pso_mtp_fold_add_softcap_q8;
    id<MTLComputePipelineState> pso_mtp_fold_sample_greedy_q8;

    id<MTLComputePipelineState> pso_apply_logit_penalties;
    id<MTLComputePipelineState> pso_sample_greedy_stage1;
    id<MTLComputePipelineState> pso_sample_greedy_stage2;
    id<MTLComputePipelineState> pso_sample_stage1;
    id<MTLComputePipelineState> pso_sample_stage2;

    // 2-Stage Bitplane Exact Q8_0 LM Head PSOs
    id<MTLComputePipelineState> pso_bp_repack_q8_to_hi4;
    id<MTLComputePipelineState> pso_bp_prep_x;
    id<MTLComputePipelineState> pso_bp_stage1_hi4;
    id<MTLComputePipelineState> pso_bp_stage15_seed;
    id<MTLComputePipelineState> pso_bp_filter_compact;
    id<MTLComputePipelineState> pso_bp_prep_indirect;
    id<MTLComputePipelineState> pso_bp_stage2_exact_cand;
    id<MTLComputePipelineState> pso_bp_cand_sample_greedy;
    id<MTLComputePipelineState> pso_bp_cand_sample_stoch_s1;
    id<MTLComputePipelineState> pso_bp_cand_sample_stoch_s2;

    // Dynamic W=4 Multi-Branch On-GPU Markov Beam Tree PSOs
    id<MTLComputePipelineState> pso_tree_init;
    id<MTLComputePipelineState> pso_mtp_rnn_prep_tree;
    id<MTLComputePipelineState> pso_mtp_rnn_gate_tree;
    id<MTLComputePipelineState> pso_mtp_fold_top4_q8;
    id<MTLComputePipelineState> pso_mtp_fold_add_softcap_tree_q8;
    id<MTLComputePipelineState> pso_mtp_fold_add_softcap_tree_f16;
    id<MTLComputePipelineState> pso_sample_top4_stage1_tree;
    id<MTLComputePipelineState> pso_tree_expand_top4;
    id<MTLComputePipelineState> pso_sample_vocab_stage1_tree;
    id<MTLComputePipelineState> pso_sample_vocab_stage2_tree;
    id<MTLComputePipelineState> pso_apply_logit_penalties_tree;
    id<MTLComputePipelineState> pso_sconv_kv_qkvr_prep_tree;
    id<MTLComputePipelineState> pso_attn_decode_single_gqa3_sg8_tree;
    id<MTLComputePipelineState> pso_attn_decode_split_gqa3_reg_tree;
    id<MTLComputePipelineState> pso_attn_decode_q8_reg_tree;
    id<MTLComputePipelineState> pso_attn_decode_reduce_tree;
    id<MTLComputePipelineState> pso_q4a_wo_sconv_tree_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_sconv_static_tree;
    id<MTLComputePipelineState> pso_moe_combine_sconv_tree;
    id<MTLComputePipelineState> pso_compact_kv_tree;
    id<MTLComputePipelineState> pso_compact_taps_tree;

    // True GPU-Batched Continuous-Batching Multi-Stream Decode PSOs (`B = 1..64`)
    id<MTLComputePipelineState> pso_sconv_kv_qkvr_prep_batched;
    id<MTLComputePipelineState> pso_attn_decode_single_gqa3_batched;
    id<MTLComputePipelineState> pso_attn_decode_split_gqa3_batched;
    id<MTLComputePipelineState> pso_attn_decode_q8_reg_batched;
    id<MTLComputePipelineState> pso_attn_decode_reduce_batched;
    id<MTLComputePipelineState> pso_q4a_wo_sconv_batched_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_sconv_static_batched;
    id<MTLComputePipelineState> pso_moe_combine_sconv_batched;
    id<MTLComputePipelineState> pso_apply_logit_penalties_batched;
    id<MTLComputePipelineState> pso_sample_batched_stage1;
    id<MTLComputePipelineState> pso_sample_batched_stage2;
    id<MTLComputePipelineState> pso_attn_qgrs_mixed_2d;
    id<MTLComputePipelineState> pso_q4a_wo_sconv_batched_2d;
    id<MTLComputePipelineState> pso_q8_gemv_batch_2d;
    id<MTLComputePipelineState> pso_moe_router_f32_2d;
    id<MTLComputePipelineState> pso_loop_ctrl_down_right_batch12_q8_2d_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_down3_left_q8_2d_m[kMaxSpecBatch];
    id<MTLComputePipelineState> pso_loop_ctrl_down1_left_q8_2d;

    // Bundle & Config
    const StrataInkling2Bundle* bundle = nullptr;
    StrataInkling2Header        hdr{};
    StrataInkling2Config        cfg{};

    // Optional Alt-`wo` Hot-Swap Overlay (e.g. `--no-refusal`, ~80 MiB for 20 Q4_A layers[4..23].wo)
    id<MTLBuffer>               b_alt_wo = nil;
    StrataInkling2TensorSlice   alt_wo_slice[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
    bool                        alt_wo_has[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
    bool                        alt_wo_active = false;
    uint32_t                    alt_wo_layers = 0U;
    uint64_t                    alt_wo_bytes  = 0ULL;

    // Geometry, Controls & Adaptive-K / FR-Spec State
    uint32_t max_seq_len         = 0U;
    uint32_t max_seq_alloc       = 0U;
    uint32_t kv_dtype            = STRATA_INKLING2_KV_DTYPE_F16;
    uint32_t seq_pos             = 0U;
    uint32_t mtp_ctx_pos         = 0U;
    uint64_t sample_step         = 0ULL;
    bool     prof_enabled        = false;
    bool     prof_detail         = false;
    bool     prof_active         = false;
    bool     prof_use_csb        = false;
    uint32_t prof_pass_count     = 0U;
    double   prof_ms[28]         = {};
    uint32_t prof_cnt[28]        = {};
    uint8_t  prof_pass_bkt[2048] = {};
    id<MTLCounterSampleBuffer> prof_csb = nil;
    enum SikSpecState : uint32_t {
        kSpecActiveProbe   = 0U,
        kSpecCooldownK0    = 1U,
        kSpecHighConfFused = 2U,
    };

    bool         ngram_draft_enabled   = true;
    bool         adaptive_k_enabled    = true;
    bool         frspec_enabled        = true;
    bool         bitplane_head_enabled = false;
    bool         bp_ready              = false;
    bool         beam_tree_enabled     = false;
    uint32_t     beam_tree_width       = 4U;
    bool         frspec_widened        = false;
    bool         full_fold_ready     = false;
    uint32_t     shortlist_miss_mask = 0U;
    double       ema_draft_ms        = 1.05;
    double       ema_cal_num         = 4.80;
    double       ema_cal_den         = 5.00;
    double       ema_acc_cal         = 0.96;
    double       ema_geom_num        = 2.00;
    double       ema_geom_den        = 4.00;
    double       ema_geom_p          = 0.50;
    double       ema_ver_ms[kMaxSpecBatch] = {};
    SikSpecState spec_state          = kSpecActiveProbe;
    uint32_t     spec_cooldown_rem   = 0U;
    uint32_t     spec_high_streak    = 0U;
    uint32_t     spec_fused_rem      = 0U;
    uint32_t     spec_locked_k       = 0U;
    uint32_t     spec_low_streak     = 0U;
    uint32_t     spec_cooldown_level = 0U;

    // Strides (bytes)
    uint64_t swa_kv_layer_stride      = 0ULL;
    uint64_t glob_kv_layer_stride     = 0ULL;
    uint64_t glob_kv_q8_data_bytes    = 0ULL;
    uint64_t glob_kv_q8_scale_bytes   = 0ULL;
    uint64_t mtp_glob_kv_layer_stride = 0ULL;
    uint64_t sconv_k_bytes         = 0ULL;
    uint64_t sconv_v_bytes         = 0ULL;
    uint64_t sconv_attn_bytes      = 0ULL;
    uint64_t sconv_mlp_bytes       = 0ULL;
    uint64_t sconv_visit_stride    = 0ULL;
    uint64_t sconv_total_bytes     = 0ULL;
    uint64_t block_slot_elems      = 0ULL;
    uint64_t block_slot_bytes      = 0ULL;

    // Telemetry
    char     device_name[128]          = {};
    bool     zero_copy_mmap            = false;
    double   load_time_ms              = 0.0;
    uint32_t int8_prefill_enabled      = 0U;
    uint32_t metal_cache_hit           = 0U;
    double   pso_load_ms               = 0.0;
    uint64_t uma_mapped_bytes          = 0ULL;
    uint64_t swa_kv_bytes_allocated    = 0ULL;
    uint64_t global_kv_bytes_allocated = 0ULL;
    uint64_t workspace_bytes           = 0ULL;
    size_t   slot_memory_bytes         = 0U;
    StrataInkling2PerfCounters counters{};

    // Single Zero-Copy UMA Weight Buffer
    id<MTLBuffer> b_uma;

    // Currently Bound Slot Buffers
    id<MTLBuffer> b_swa_k_cache;
    id<MTLBuffer> b_swa_v_cache;
    id<MTLBuffer> b_glob_k_cache;
    id<MTLBuffer> b_glob_v_cache;
    id<MTLBuffer> b_sconv_state;
    id<MTLBuffer> b_mtp_swa_k_cache;
    id<MTLBuffer> b_mtp_swa_v_cache;
    id<MTLBuffer> b_mtp_glob_k_cache;
    id<MTLBuffer> b_mtp_glob_v_cache;
    id<MTLBuffer> b_mtp_taps_last;

    // SConv Rollback Zero-Copy Snapshot Slots (0..6) + Paired Residency Sets
    id<MTLBuffer>       b_sconv_snaps; // Single contiguous `[kMaxSpecSnapSlots, sconv_total_bytes]` snapshot buffer!

    StrataInkling2ContextSlot* default_slot = nullptr;
    StrataInkling2ContextSlot* bound_slot   = nullptr;
    id<MTLCommandBuffer>       pending_fix_cb = nil;

    // Private GPU Scratch Workspaces (`kMaxPrefillChunk = 512`)
    id<MTLBuffer> b_cur;
    id<MTLBuffer> b_prelude_cur;
    id<MTLBuffer> b_bank_last;
    id<MTLBuffer> b_blocks;
    id<MTLBuffer> b_bank;
    id<MTLBuffer> b_inputs[3];
    id<MTLBuffer> b_messages[4];
    id<MTLBuffer> b_ctrl_cat_f16;      // [3, 2048, 2*D] F16
    id<MTLBuffer> b_ctrl_h_f32;        // [3, 2048, 64] F32 (or F16 during prefill NAX)
    id<MTLBuffer> b_ctrl_cat_sc;       // [3, 2048] F16 row scales for INT8 prefill (`cat_i8` aliases `b_moe_down_part`)
    id<MTLBuffer> b_ctrl_h_i8;         // [3, 2048, 64] INT8 for INT8 prefill
    id<MTLBuffer> b_ctrl_h_sc;         // [3, 2048] F16 row scales for INT8 prefill
    id<MTLBuffer> b_in_f16[3];
    id<MTLBuffer> b_in_i8[3];          // [0]: [2048, max(D, q_dim)] INT8; [1..2]: [2048, D] INT8
    id<MTLBuffer> b_in_xsc[3];         // [2048] F16 per-row scales
    id<MTLBuffer> b_in_xsum_f16[3];    // [2048, max(D, q_dim)/64] F16 per-group sums for Q4_A INT8xINT4
    id<MTLBuffer> b_x_blk_sum[3];
    id<MTLBuffer> b_wo_blk_sum;
    id<MTLBuffer> b_rms_sq;
    id<MTLBuffer> b_msg_bank12;
    id<MTLBuffer> b_msg_norm12_f16;
    id<MTLBuffer> b_ctrl_h_msg12;
    id<MTLBuffer> b_x_f32;
    id<MTLBuffer> b_q_raw;
    id<MTLBuffer> b_q_f32;
    id<MTLBuffer> b_q_hm_f16;          // [24, 512, 128] head-major FP16 for NAX FlashAttention
    id<MTLBuffer> b_r_hm_f32;          // [24, 512, 16]  head-major FP32 for NAX FlashAttention
    id<MTLBuffer> b_gate_f32;
    id<MTLBuffer> b_r_raw;
    id<MTLBuffer> b_r_f32;
    id<MTLBuffer> b_sink_f32;
    id<MTLBuffer> b_k_raw;
    id<MTLBuffer> b_v_raw;
    id<MTLBuffer> b_k_conv;
    id<MTLBuffer> b_v_conv;
    id<MTLBuffer> b_k_dyn;
    id<MTLBuffer> b_v_dyn;
    id<MTLBuffer> b_attn_gated_f16;
    id<MTLBuffer> b_wo_out_f32;
    id<MTLBuffer> b_part_m;
    id<MTLBuffer> b_part_l;
    id<MTLBuffer> b_part_o;
    id<MTLBuffer> b_router_logits;
    id<MTLBuffer> b_topk_idx;
    id<MTLBuffer> b_topk_w;
    id<MTLBuffer> b_moe_mid_f16;
    id<MTLBuffer> b_moe_mid_i8;        // [2048*top_k + 128, inter] INT8 for INT8 MoE prefill
    id<MTLBuffer> b_moe_mid_sc;        // [2048*top_k + 128] F16 scales for INT8 MoE prefill
    id<MTLBuffer> b_moe_x_sc;          // [2048*top_k + 128] F16 sorted input scales (`x_sorted_i8` aliases `b_moe_down_part`)
    id<MTLBuffer> b_moe_down_part;     // Also aliased as `x_sorted` during MoE prefill w13 (saves 36-76 MiB UMA!)
    id<MTLBuffer> b_moe_out_f32;
    id<MTLBuffer> b_moe_sorted_pairs;
    id<MTLBuffer> b_moe_work_tiles;    // Decode tiles & Prefill quad-bin tiles (4 bins x 512 = 2048 tiles)
    id<MTLBuffer> b_moe_num_tiles;
    id<MTLBuffer> b_moe_indirect_args; // 8 x uint4 indirect dispatch args (128B)
    id<MTLBuffer> b_mtp_taps_chunk;
    id<MTLBuffer> b_sample_stage_v;
    id<MTLBuffer> b_sample_stage_i;
    id<MTLBuffer> b_sample_stage_lse;

    // MTP Drafter Private Scratch & Optional/Lazy Full-Vocab Folded Head
    id<MTLBuffer> b_mtp_fold_weight;   // Lazy when frspec.enabled!
    id<MTLBuffer> b_mtp_fold_bias;
    id<MTLBuffer> b_mtp_base_logits;
    id<MTLBuffer> b_mtp_res_f32;
    id<MTLBuffer> b_mtp_norm_f16;
    id<MTLBuffer> b_mtp_norm_f32;
    id<MTLBuffer> b_mtp_draft_k;
    id<MTLBuffer> b_mtp_draft_v;
    id<MTLBuffer> b_mtp_ffn_mid_f16;
    id<MTLBuffer> b_mtp_f_rnn;
    id<MTLBuffer> b_mtp_rnn_state;
    id<MTLBuffer> b_mtp_rnn_in_f16;
    id<MTLBuffer> b_mtp_rnn_raw_f32;
    id<MTLBuffer> b_mtp_latent_f16;

    // Shared Host <-> GPU IO Buffers
    id<MTLBuffer> b_mtp_inv_freq;
    id<MTLBuffer> b_token_ids;
    id<MTLBuffer> b_mtp_token_ids;
    id<MTLBuffer> b_logits;
    id<MTLBuffer> b_next_tokens;
    id<MTLBuffer> b_draft_pmax;        // [8] F32 for Bayes Adaptive-K
    id<MTLBuffer> b_sample_top_ids;
    id<MTLBuffer> b_sample_top_probs;
    id<MTLBuffer> b_sample_keep_k;
    id<MTLBuffer> b_draft_top_ids;
    id<MTLBuffer> b_draft_top_probs;
    id<MTLBuffer> b_draft_keep_k;
    id<MTLBuffer> b_frspec_full_to_short; // [V] U32 mapping full vocab -> shortlist index or 0xFFFFFFFF

    // 2-Stage Bitplane Exact Q8_0 LM Head Buffers (allocated only when unembed.dtype == Q8_0)
    id<MTLBuffer> b_bp_hi4;
    id<MTLBuffer> b_bp_rnorm4;
    id<MTLBuffer> b_bp_ub;
    id<MTLBuffer> b_bp_xprep;
    id<MTLBuffer> b_bp_stage_v;
    id<MTLBuffer> b_bp_stage_i;
    id<MTLBuffer> b_bp_cand_ids;
    id<MTLBuffer> b_bp_cand_logits;
    id<MTLBuffer> b_bp_meta;

    // Dynamic W=4 Markov Beam Tree Buffers (allocated when mtp.enabled != 0)
    id<MTLBuffer> b_tree_meta;
    id<MTLBuffer> b_mtp_tree_states;

    // Lazy Multimodal Vision Embedding Staging Buffer (nil unless prefill_with_images called with num_spans > 0)
    id<MTLBuffer> b_vis_stage;
    size_t        vis_stage_bytes = 0U;

    // Sparse O(U + M) Sliding-Window Logit Penalty State (per-slot `pen_hist`, cached on GPU for solo decode)
    id<MTLBuffer>              b_pen_ids;
    id<MTLBuffer>              b_pen_counts;
    uint32_t                   pen_num_ids      = 0U;
    StrataInkling2ContextSlot* pen_cache_slot   = nullptr;
    std::vector<uint32_t>      pen_pos;
    uint32_t                   pen_window_start = 0U;
    uint32_t                   pen_window_end   = 0U;

    // Lazy Continuous-Batching Workspaces (`0.00 MiB` overhead when only solo `c=1` is used!)
    id<MTLBuffer>         b_batch_slot_meta;    // [64] SikBatchSlotMetaHost (shared)
    id<MTLBuffer>         b_batch_next_tokens;  // [64] uint32_t (shared)
    id<MTLBuffer>         b_batch_pen_items;    // [64] SikBatchPenItemHost (shared)
    id<MTLBuffer>         b_batch_sample_items; // [64] SikBatchSampleItemHost (shared)
    id<MTLBuffer>         b_batch_pen_toks;     // dynamic [batch_pen_cap] uint32_t (shared)
    id<MTLBuffer>         b_batch_pen_cnts;     // dynamic [batch_pen_cap] uint16_t (shared)
    uint32_t              batch_pen_cap     = 0U;
    std::vector<uint16_t> batch_pen_scratch;    // lazy [V] host histogram scratch for O(U_b) packing
    id<MTLBuffer>         b_batch_part_m;       // lazy (B > 8): [64, n_q, 36] F32 (private)
    id<MTLBuffer>         b_batch_part_l;       // lazy (B > 8): [64, n_q, 36] F32 (private)
    id<MTLBuffer>         b_batch_part_o;       // lazy (B > 8): [64, n_q, 36, hd] F32 (private)
    id<MTLBuffer>         b_batch_logits;       // lazy (B > 8): [16 or 32, V] F32 (private)
    uint32_t              batch_logits_rows = 0U;
    id<MTLBuffer>         b_batch_sample_v;       // lazy (B > 8): [32, 4096] F32 (private)
    id<MTLBuffer>         b_batch_sample_i;       // lazy (B > 8): [32, 4096] U32 (private)
    id<MTLBuffer>         b_batch_msg_bank12;     // lazy (B > 8): [12, 4, 64, D] F32 (private)
    id<MTLBuffer>         b_batch_msg_norm12_f16; // lazy (B > 8): [12, 4, 64, D] F16 (private)
    id<MTLBuffer>         b_batch_ctrl_h_msg12;   // lazy (B > 8): [12, 4, 64, R] F32 (private)
};

namespace {

/* ============================================================================
 * Upgraded Longest-Match-First 5-Gram -> 4-Gram Suffix Drafter (0.00 ms GPU)
 * ============================================================================ */

inline uint32_t sik_ngram5_hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
    uint64_t h = (static_cast<uint64_t>(a) * 0x9E3779B97F4A7C15ULL) ^
                 (static_cast<uint64_t>(b) * 0xBF58476D1CE4E5B9ULL) ^
                 (static_cast<uint64_t>(c) * 0x94D049BB133111EBULL) ^
                 (static_cast<uint64_t>(d) * 0xD6E8FEB86659FD93ULL) ^
                 static_cast<uint64_t>(e);
    h ^= (h >> 33ULL);
    h *= 0xFF51AFD7ED558CCDULL;
    return static_cast<uint32_t>((h >> 32ULL) & (kNgramHashSize - 1U));
}

inline uint32_t sik_ngram4_hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint64_t h = (static_cast<uint64_t>(a) * 0xBF58476D1CE4E5B9ULL) ^
                 (static_cast<uint64_t>(b) * 0x94D049BB133111EBULL) ^
                 (static_cast<uint64_t>(c) * 0xD6E8FEB86659FD93ULL) ^
                 static_cast<uint64_t>(d);
    h ^= (h >> 33ULL);
    h *= 0xC4CEB9FE1A85EC53ULL;
    return static_cast<uint32_t>((h >> 32ULL) & (kNgramHashSize - 1U));
}

inline void sik_ngram_reset(StrataInkling2ContextSlot* s) {
    if (!s) return;
    s->ngram_hist.clear();
    if (!s->ngram5_ht.empty()) std::fill(s->ngram5_ht.begin(), s->ngram5_ht.end(), 0xFFFFFFFFU);
    if (!s->ngram4_ht.empty()) std::fill(s->ngram4_ht.begin(), s->ngram4_ht.end(), 0xFFFFFFFFU);
    s->ngram_cooldown = 0U;
}

inline void sik_ngram_append(StrataInkling2ContextSlot* s, const uint32_t* toks, uint32_t n) {
    if (!s || !toks || n == 0U) return;
    if (s->ngram5_ht.empty()) {
        s->ngram5_ht.assign(kNgramHashSize, 0xFFFFFFFFU);
        s->ngram4_ht.assign(kNgramHashSize, 0xFFFFFFFFU);
    }
    const size_t base = s->ngram_hist.size();
    s->ngram_hist.insert(s->ngram_hist.end(), toks, toks + n);
    const uint32_t* h   = s->ngram_hist.data();
    const size_t    end = s->ngram_hist.size();
    for (size_t i = (base >= 3U) ? base : 3U; i < end; ++i) {
        s->ngram4_ht[sik_ngram4_hash(h[i - 3U], h[i - 2U], h[i - 1U], h[i])] = static_cast<uint32_t>(i);
        if (i >= 4U) {
            s->ngram5_ht[sik_ngram5_hash(h[i - 4U], h[i - 3U], h[i - 2U], h[i - 1U], h[i])] = static_cast<uint32_t>(i);
        }
    }
}

inline void sik_ngram_rewind(StrataInkling2ContextSlot* s, uint32_t new_len) {
    if (!s) return;
    if (new_len < s->ngram_hist.size()) s->ngram_hist.resize(new_len);
    if (!s->ngram5_ht.empty()) {
        std::fill(s->ngram5_ht.begin(), s->ngram5_ht.end(), 0xFFFFFFFFU);
        std::fill(s->ngram4_ht.begin(), s->ngram4_ht.end(), 0xFFFFFFFFU);
        const uint32_t* h   = s->ngram_hist.data();
        const size_t    end = s->ngram_hist.size();
        for (size_t i = 3U; i < end; ++i) {
            s->ngram4_ht[sik_ngram4_hash(h[i - 3U], h[i - 2U], h[i - 1U], h[i])] = static_cast<uint32_t>(i);
            if (i >= 4U) {
                s->ngram5_ht[sik_ngram5_hash(h[i - 4U], h[i - 3U], h[i - 2U], h[i - 1U], h[i])] = static_cast<uint32_t>(i);
            }
        }
    }
    s->ngram_cooldown = 0U;
}

inline uint32_t sik_ngram_propose(
    const StrataInkling2ContextSlot* s,
    uint32_t                         pending_tok,
    uint32_t                         max_k,
    uint32_t*                        out_drafts,
    bool                             in_cooldown = false
) {
    if (!s || !out_drafts || max_k == 0U || s->ngram_cooldown > 0U || s->ngram5_ht.empty()) return 0U;
    const size_t sz = s->ngram_hist.size();
    if (sz < 4U) return 0U;
    const uint32_t* h = s->ngram_hist.data();

    // 1. Strict 10-gram verbatim-copy drafter (`0.00 ms` ONLY on exact long verbatim loops >= 10 tokens
    //    with full `avail >= max_k` so 4/5/8-gram boilerplate collisions NEVER steal ~1.05 ms Q4_A DSpark
    //    steps on code, refactors, or prose!).
    if (sz >= 10U) {
        const uint32_t a = h[sz - 4U], b = h[sz - 3U], c = h[sz - 2U], d = h[sz - 1U], e = pending_tok;
        const uint32_t pos = s->ngram5_ht[sik_ngram5_hash(a, b, c, d, e)];
        if (pos != 0xFFFFFFFFU && pos >= 9U && static_cast<size_t>(pos) + 1U < sz &&
            h[pos - 4U] == a && h[pos - 3U] == b && h[pos - 2U] == c && h[pos - 1U] == d && h[pos] == e &&
            h[pos - 5U] == h[sz - 5U] && h[pos - 6U] == h[sz - 6U] && h[pos - 7U] == h[sz - 7U] &&
            h[pos - 8U] == h[sz - 8U] && h[pos - 9U] == h[sz - 9U]) {
            const uint32_t avail = static_cast<uint32_t>(sz - 1U - static_cast<size_t>(pos));
            const uint32_t min_k = std::min<uint32_t>(max_k, in_cooldown ? 4U : max_k);
            if (avail >= min_k) {
                const uint32_t eff_k = std::min<uint32_t>(std::min<uint32_t>(max_k, 7U), avail);
                for (uint32_t k = 0U; k < eff_k; ++k) out_drafts[k] = h[static_cast<size_t>(pos) + 1U + k];
                return eff_k;
            }
        }
    }
    return 0U;
}

bool sik_drain_pending_fix(StrataInkling2GpuEngine* eng) {
    if (!eng || !eng->pending_fix_cb) return true;
    id<MTLCommandBuffer> cb = eng->pending_fix_cb;
    eng->pending_fix_cb = nil;
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError) {
        NSError* err = [cb error];
        std::fprintf(stderr, "[strata_inkling2_gpu] Async cb_fix command buffer error: %s\n",
                     err ? [[err localizedDescription] UTF8String] : "unknown");
        return false;
    }
    return true;
}

/* ============================================================================
 * Unified Mixed-Precision (`FP16`, `Q8_0`, `Q4_A`) Dense Linear Dispatcher
 * ============================================================================ */

void sik_encode_dense_linear(
    id<MTLComputeCommandEncoder>     enc,
    StrataInkling2GpuEngine*         eng,
    const StrataInkling2TensorSlice& sl,
    id<MTLBuffer>                    x_in_f16,
    NSUInteger                       x_off,
    id<MTLBuffer>                    y_f16,
    NSUInteger                       y_f16_off,
    id<MTLBuffer>                    y_f32,
    NSUInteger                       y_f32_off,
    uint32_t                         M_valid,
    uint32_t                         M_padded,
    uint32_t                         out_mode,
    bool                             force_gemv,
    float                            softcap = 0.0f,
    id<MTLBuffer>                    w_buf   = nil
) {
    const uint32_t rows = sl.shape0;
    const uint32_t cols = sl.shape1;
    if (rows == 0U || cols == 0U || M_valid == 0U) return;

    id<MTLBuffer> weight_buf = w_buf ? w_buf : eng->b_uma;
    uint64_t w_off = 0ULL, s_off = 0ULL, b_off = 0ULL;
    sik_slice_sub_offsets(sl, w_off, s_off, b_off);

    // 1. Metal 4.1 Direct / Quantized (`BK=128`) NAX Prefill Path (`M > 8`, `rows >= 64`)
    if (!force_gemv && eng->has_nax && M_valid > kMaxSpecBatch && (cols % 64U) == 0U && rows >= 64U) {
        if (sl.dtype == STRATA_INKLING2_DTYPE_Q4_A && (cols % 128U) == 0U) {
            const uint32_t N_padded = (rows + 63U) & ~63U;
            uint32_t bm = 16U, th = 64U;
            id<MTLComputePipelineState> pso = eng->pso_q4a_gemm_nax_m16;
            if (M_padded >= 256U && (M_padded % 128U) == 0U && rows >= 1024U) {
                bm = 128U; th = 128U; pso = eng->pso_q4a_gemm_nax_m128;
            } else if (M_padded >= 64U && rows >= 1024U) {
                bm = 64U; th = 128U; pso = eng->pso_q4a_gemm_nax_m64;
            } else if (M_padded >= 32U) {
                bm = 32U; th = 64U; pso = eng->pso_q4a_gemm_nax_m32;
            }
            SikNaxGemmParamsHost np{M_valid, M_padded, rows, N_padded, cols, out_mode, softcap, 0U};
            [enc setComputePipelineState:pso];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(s_off) atIndex:1];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(b_off) atIndex:2];
            [enc setBuffer:x_in_f16 offset:x_off atIndex:3];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:(y_f16 ? y_f16_off : 0U) atIndex:4];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:(y_f32 ? y_f32_off : 0U) atIndex:5];
            [enc setBytes:&np length:sizeof(np) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(N_padded / 64U, (M_padded + bm - 1U) / bm, 1)
                threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
            return;
        }
        if (sl.dtype == STRATA_INKLING2_DTYPE_Q8_0 && (cols % 128U) == 0U) {
            uint32_t bm = 16U, bn = 32U, th = 64U;
            id<MTLComputePipelineState> pso = eng->pso_q8_gemm_nax_m16_n32;
            if (M_padded >= 256U && (M_padded % 128U) == 0U && rows >= 1024U) {
                bm = 128U; bn = 64U; th = 128U; pso = eng->pso_q8_gemm_nax_m128;
            } else if (M_padded >= 64U && rows >= 1024U) {
                bm = 64U; bn = 64U; th = 128U; pso = eng->pso_q8_gemm_nax_m64;
            } else if (M_padded >= 32U) {
                bm = 32U; bn = 32U; th = 64U; pso = eng->pso_q8_gemm_nax_m32_n32;
            }
            const uint32_t N_padded = (rows + bn - 1U) & ~(bn - 1U);
            SikNaxGemmParamsHost np{M_valid, M_padded, rows, N_padded, cols, out_mode, softcap, 0U};
            [enc setComputePipelineState:pso];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(s_off) atIndex:1];
            [enc setBuffer:x_in_f16 offset:x_off atIndex:2];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:(y_f16 ? y_f16_off : 0U) atIndex:3];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:(y_f32 ? y_f32_off : 0U) atIndex:4];
            [enc setBytes:&np length:sizeof(np) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(N_padded / bn, (M_padded + bm - 1U) / bm, 1)
                threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
            return;
        }
        if (sl.dtype == STRATA_INKLING2_DTYPE_F16) {
            // Direct-Device `static_slice` `K=dynamic_extent` FP16 NAX (`58.6 TF`!)
            uint32_t bm = 16U, bn = 32U, th = 64U;
            id<MTLComputePipelineState> pso = eng->pso_f16_gemm_nax_16x32_sg2;
            if (M_padded >= 256U && rows >= 2048U) {
                bm = 64U; bn = 128U; th = 128U; pso = eng->pso_f16_gemm_nax_64x128_sg4;
            } else if (M_padded >= 64U && rows >= 1024U) {
                bm = 64U; bn = 64U; th = 128U; pso = eng->pso_f16_gemm_nax_64x64_sg4;
            } else if (M_padded >= 32U && rows >= 512U) {
                bm = 32U; bn = 64U; th = 64U; pso = eng->pso_f16_gemm_nax_32x64_sg2;
            }
            const uint32_t N_padded = (rows + bn - 1U) & ~(bn - 1U);
            SikNaxGemmParamsHost np{M_valid, M_padded, rows, N_padded, cols, out_mode, softcap, 0U};
            [enc setComputePipelineState:pso];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
            [enc setBuffer:x_in_f16 offset:x_off atIndex:1];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:(y_f16 ? y_f16_off : 0U) atIndex:2];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:(y_f32 ? y_f32_off : 0U) atIndex:3];
            [enc setBytes:&np length:sizeof(np) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(N_padded / bn, (M_padded + bm - 1U) / bm, 1)
                threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
            return;
        }
    }

    // 2. Prefill small-N FP16 path (`rows < 64`, `M > 8`)
    if (!force_gemv && M_valid > kMaxSpecBatch && rows < 64U && sl.dtype == STRATA_INKLING2_DTYPE_F16) {
        SikGemvBatchParamsHost gp{M_valid, rows, cols, out_mode, softcap, 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_f16_small_n_prefill];
        [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
        [enc setBuffer:x_in_f16 offset:x_off atIndex:1];
        [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:(y_f16 ? y_f16_off : 0U) atIndex:2];
        [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:(y_f32 ? y_f32_off : 0U) atIndex:3];
        [enc setBytes:&gp length:sizeof(gp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(rows, (M_valid + 7U) / 8U, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }

    // 3. Coalesced 16-Lane Exact-M (`1..8`) SIMT GEMV (100.0000% bit-identical `K>0 == K=0`!)
    //    Uses 4 rows/TG (`64` th) for medium matrices like `wo [2304, 3072]` (-> 576 TGs!),
    //    8 rows/TG (`128` th) for large matrices (`>= 3072`), 1 row/TG (`128` th) for tiny `rows <= 384`.
    const bool use_row_tg128 = (rows <= 384U) && (sl.dtype != STRATA_INKLING2_DTYPE_Q4_A);
    const bool use_4rows_tg  = (!use_row_tg128) && (sl.dtype != STRATA_INKLING2_DTYPE_Q4_A) && (rows <= 2560U);
    const uint32_t tg_th     = use_4rows_tg ? 64U : 128U;
    const uint32_t rows_tg   = use_row_tg128 ? 1U : (use_4rows_tg ? 4U : 8U);
    const uint32_t grid_x    = (rows + rows_tg - 1U) / rows_tg;

    for (uint32_t m0 = 0U; m0 < M_valid; m0 += kMaxSpecBatch) {
        const uint32_t mb = std::min<uint32_t>(kMaxSpecBatch, M_valid - m0);
        SikGemvBatchParamsHost gp{mb, rows, cols, out_mode, softcap, 0U, 0U, 0U};
        const NSUInteger xo = x_off + static_cast<NSUInteger>(m0) * cols * sizeof(uint16_t);
        const NSUInteger y16o = y_f16 ? (y_f16_off + static_cast<NSUInteger>(m0) * rows * sizeof(uint16_t)) : 0U;
        const NSUInteger y32o = y_f32 ? (y_f32_off + static_cast<NSUInteger>(m0) * rows * sizeof(float)) : 0U;

        if (sl.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
            [enc setComputePipelineState:eng->pso_q4a_gemv_m[mb - 1U]];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(s_off) atIndex:1];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(b_off) atIndex:2];
            [enc setBuffer:x_in_f16 offset:xo atIndex:3];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:y16o atIndex:4];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:y32o atIndex:5];
            [enc setBytes:&gp length:sizeof(gp) atIndex:6];
        } else if (sl.dtype == STRATA_INKLING2_DTYPE_Q8_0) {
            id<MTLComputePipelineState> pso = use_row_tg128 ? eng->pso_q8_row_tg128_m[mb - 1U] : eng->pso_q8_gemv_m[mb - 1U];
            [enc setComputePipelineState:pso];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(s_off) atIndex:1];
            [enc setBuffer:x_in_f16 offset:xo atIndex:2];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:y16o atIndex:3];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:y32o atIndex:4];
            [enc setBytes:&gp length:sizeof(gp) atIndex:5];
        } else {
            id<MTLComputePipelineState> pso = use_row_tg128 ? eng->pso_f16_row_tg128_m[mb - 1U] : eng->pso_f16_gemv_m[mb - 1U];
            [enc setComputePipelineState:pso];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
            [enc setBuffer:x_in_f16 offset:xo atIndex:1];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:y16o atIndex:2];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:y32o atIndex:3];
            [enc setBytes:&gp length:sizeof(gp) atIndex:4];
        }
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_th, 1, 1)];
    }
}

void sik_encode_dense_linear_i8(
    id<MTLComputeCommandEncoder>     enc,
    StrataInkling2GpuEngine*         eng,
    const StrataInkling2TensorSlice& sl,
    id<MTLBuffer>                    x_in_f16,
    id<MTLBuffer>                    x_i8,
    id<MTLBuffer>                    x_sc,
    id<MTLBuffer>                    x_sum_f16,
    id<MTLBuffer>                    y_f16,
    NSUInteger                       y_f16_off,
    id<MTLBuffer>                    y_f32,
    NSUInteger                       y_f32_off,
    uint32_t                         M_valid,
    uint32_t                         M_padded,
    uint32_t                         out_mode,
    bool                             force_gemv,
    float                            softcap = 0.0f,
    id<MTLBuffer>                    w_buf   = nil
) {
    const uint32_t rows = sl.shape0;
    const uint32_t cols = sl.shape1;
    id<MTLBuffer> weight_buf = w_buf ? w_buf : eng->b_uma;
    uint64_t w_off = 0ULL, s_off = 0ULL, b_off = 0ULL;
    sik_slice_sub_offsets(sl, w_off, s_off, b_off);

    if (!force_gemv && eng->has_nax && M_valid >= 32U && (M_padded % 32U) == 0U && rows >= 64U) {
        (void)x_sum_f16;
        const uint32_t N_padded = (rows + 63U) & ~63U;
        SikNaxGemmParamsHost np{M_valid, M_padded, rows, N_padded, cols, out_mode, softcap, 0U};

        if (sl.dtype == STRATA_INKLING2_DTYPE_Q4_A && (cols % 128U) == 0U) {
            const bool     use_m128 = (M_padded >= 128U && (M_padded % 128U) == 0U);
            const bool     use_m64  = (!use_m128 && (M_padded % 64U) == 0U);
            const uint32_t bm       = use_m128 ? 128U : (use_m64 ? 64U : 32U);
            const uint32_t th       = (use_m128 || use_m64) ? 128U : 64U;
            id<MTLComputePipelineState> pso = use_m128 ? eng->pso_q4a_gemm_nax_i8_m128
                                            : (use_m64 ? eng->pso_q4a_gemm_nax_i8_m64 : eng->pso_q4a_gemm_nax_i8_m32);
            [enc setComputePipelineState:pso];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(s_off) atIndex:1];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(b_off) atIndex:2];
            [enc setBuffer:x_i8 offset:0 atIndex:3];
            [enc setBuffer:x_sc offset:0 atIndex:4];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:(y_f16 ? y_f16_off : 0U) atIndex:5];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:(y_f32 ? y_f32_off : 0U) atIndex:6];
            [enc setBytes:&np length:sizeof(np) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(N_padded / 64U, M_padded / bm, 1)
                threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
            return;
        }
        if (sl.dtype == STRATA_INKLING2_DTYPE_Q8_0 && (cols % 64U) == 0U) {
            const bool     use_m64 = ((M_padded % 64U) == 0U);
            const uint32_t bm      = use_m64 ? 64U : 32U;
            const uint32_t th      = use_m64 ? 128U : 64U;
            [enc setComputePipelineState:(use_m64 ? eng->pso_q8_gemm_nax_i8_m64 : eng->pso_q8_gemm_nax_i8_m32)];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(w_off) atIndex:0];
            [enc setBuffer:weight_buf offset:static_cast<NSUInteger>(s_off) atIndex:1];
            [enc setBuffer:x_i8 offset:0 atIndex:2];
            [enc setBuffer:x_sc offset:0 atIndex:3];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:(y_f16 ? y_f16_off : 0U) atIndex:4];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:(y_f32 ? y_f32_off : 0U) atIndex:5];
            [enc setBytes:&np length:sizeof(np) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(N_padded / 64U, M_padded / bm, 1)
                threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
            return;
        }
    }
    sik_encode_dense_linear(enc, eng, sl, x_in_f16, 0, y_f16, y_f16_off, y_f32, y_f32_off,
                            M_valid, M_padded, out_mode, force_gemv, softcap, w_buf);
}

void sik_encode_rmsnorm(
    id<MTLComputeCommandEncoder> enc,
    StrataInkling2GpuEngine*     eng,
    id<MTLBuffer>                x_in,
    NSUInteger                   x_off,
    uint64_t                     norm_w_off,
    bool                         has_weight,
    id<MTLBuffer>                out_f16,
    id<MTLBuffer>                out_f32,
    bool                         write_f32,
    uint32_t                     M_valid,
    uint32_t                     M_padded,
    bool                         tap_capture = false,
    uint32_t                     tap_idx     = 0U
) {
    SikRmsNormParamsHost np{
        M_valid, M_padded, eng->cfg.hidden_size, eng->cfg.rms_norm_eps,
        has_weight ? 1U : 0U, write_f32 ? 1U : 0U,
        tap_capture ? 1U : 0U, tap_idx
    };
    [enc setComputePipelineState:eng->pso_rmsnorm];
    [enc setBuffer:x_in offset:x_off atIndex:0];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(norm_w_off) atIndex:1];
    [enc setBuffer:out_f16 offset:0 atIndex:2];
    [enc setBuffer:(out_f32 ? out_f32 : eng->b_x_f32) offset:0 atIndex:3];
    [enc setBuffer:eng->b_mtp_taps_chunk offset:0 atIndex:4];
    [enc setBytes:&np length:sizeof(np) atIndex:5];
    [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(M_padded, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sik_encode_attn_res(
    id<MTLComputeCommandEncoder> enc,
    StrataInkling2GpuEngine*     eng,
    id<MTLBuffer>                blocks_buf,
    uint32_t                     num_blocks,
    id<MTLBuffer>                partial_buf,
    uint64_t                     query_off,
    uint32_t                     num_queries,
    id<MTLBuffer>                out0,
    id<MTLBuffer>                out1,
    id<MTLBuffer>                out2,
    uint32_t                     M_valid
) {
    SikAttnResParamsHost ap{
        M_valid, eng->cfg.hidden_size, num_blocks,
        static_cast<uint32_t>(eng->block_slot_elems),
        num_queries, eng->cfg.rms_norm_eps, 0U, 0U
    };
    [enc setComputePipelineState:eng->pso_attn_res];
    [enc setBuffer:blocks_buf offset:0 atIndex:0];
    [enc setBuffer:partial_buf offset:0 atIndex:1];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(query_off) atIndex:2];
    [enc setBuffer:out0 offset:0 atIndex:3];
    [enc setBuffer:(out1 ? out1 : out0) offset:0 atIndex:4];
    [enc setBuffer:(out2 ? out2 : out0) offset:0 atIndex:5];
    [enc setBytes:&ap length:sizeof(ap) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sik_encode_attn_res_rmsnorm(
    id<MTLComputeCommandEncoder> enc,
    StrataInkling2GpuEngine*     eng,
    id<MTLBuffer>                blocks_buf,
    uint32_t                     num_blocks,
    id<MTLBuffer>                partial_buf,
    uint64_t                     query_off,
    uint64_t                     norm_w_off,
    id<MTLBuffer>                out_f16,
    id<MTLBuffer>                out_f32,
    bool                         write_f32,
    uint32_t                     M_valid,
    uint32_t                     M_padded,
    bool                         tap_capture = false,
    uint32_t                     tap_idx     = 0U
) {
    SikAttnResNormParamsHost ap{
        M_valid, M_padded, eng->cfg.hidden_size, num_blocks,
        static_cast<uint32_t>(eng->block_slot_elems),
        eng->cfg.rms_norm_eps, write_f32 ? 1U : 0U,
        tap_capture ? 1U : 0U, tap_idx, 0U, 0U, 0U
    };
    const bool use_576 = (eng->cfg.hidden_size == 2304U);
    [enc setComputePipelineState:(use_576 ? eng->pso_attn_res_rmsnorm_576 : eng->pso_attn_res_rmsnorm)];
    [enc setBuffer:blocks_buf offset:0 atIndex:0];
    [enc setBuffer:partial_buf offset:0 atIndex:1];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(query_off) atIndex:2];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(norm_w_off) atIndex:3];
    [enc setBuffer:out_f16 offset:0 atIndex:4];
    [enc setBuffer:(out_f32 ? out_f32 : eng->b_x_f32) offset:0 atIndex:5];
    [enc setBuffer:eng->b_mtp_taps_chunk offset:0 atIndex:6];
    [enc setBytes:&ap length:sizeof(ap) atIndex:7];
    [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(M_padded, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(use_576 ? 576 : 256, 1, 1)];
}

static void sik_encode_loop_ctrl_batch_impl(
    id<MTLCommandBuffer> __strong&         cb,
    id<MTLComputeCommandEncoder> __strong& enc,
    StrataInkling2GpuEngine*          eng,
    const StrataInkling2LoopCtrlDesc* ctrls,     // [num_ctrls] (1 or 3)
    uint32_t                          num_ctrls, // 1 or 3
    id<MTLBuffer>                     c0,
    id<MTLBuffer>                     c1,
    id<MTLBuffer>                     c2,
    id<MTLBuffer>                     m0,
    id<MTLBuffer>                     m1,
    id<MTLBuffer>                     m2,
    id<MTLBuffer>                     o0,
    id<MTLBuffer>                     o1,
    id<MTLBuffer>                     o2,
    uint32_t                          M_valid,
    uint32_t                          M_padded,
    bool                              force_gemv,
    bool                              skip_prep,
    NSUInteger                        c0_off,
    NSUInteger                        c1_off,
    NSUInteger                        c2_off,
    NSUInteger                        m0_off,
    NSUInteger                        m1_off,
    NSUInteger                        m2_off,
    NSUInteger                        o0_off,
    NSUInteger                        o1_off,
    NSUInteger                        o2_off,
    id<MTLBuffer>                     h_msg_buf,
    NSUInteger                        h_msg_off
) {
    const uint32_t D    = eng->cfg.hidden_size;
    const uint32_t R    = eng->cfg.loop_gate_rank;
    const bool     is_q8 = (ctrls[0].down_weight.dtype == STRATA_INKLING2_DTYPE_Q8_0);
    const bool     use_nax = (!force_gemv && eng->has_nax && M_valid > kMaxSpecBatch && (D % 64U) == 0U && R == 64U);
    /* `cat_i8 [num_ctrls, M_padded, 2 * D]` lives in the aliased `b_moe_down_part`: take the INT8 path only if it fits. */
    const uint64_t cat_i8_bytes = static_cast<uint64_t>(num_ctrls) * M_padded * 2ULL * D;
    const bool     use_i8_ctrl = (use_nax && is_q8 && eng->int8_prefill_enabled != 0U && M_valid >= 32U && (M_padded % 32U) == 0U &&
                                  cat_i8_bytes <= static_cast<uint64_t>([eng->b_moe_down_part length]));

    // 1. Prep: `[unweighted_rms(cur); unweighted_rms(msg)]` -> `b_ctrl_cat_f16` (or direct `cat_i8` when `--int8-prefill`!)
    if (!skip_prep) {
        if (use_i8_ctrl) {
            SikLoopCtrlPrepParamsHost pp{M_valid, M_padded, D, eng->cfg.rms_norm_eps};
            [enc setComputePipelineState:eng->pso_loop_ctrl_prep_i8];
            [enc setBuffer:c0 offset:c0_off atIndex:0];
            [enc setBuffer:(c1 ? c1 : c0) offset:(c1 ? c1_off : c0_off) atIndex:1];
            [enc setBuffer:(c2 ? c2 : c0) offset:(c2 ? c2_off : c0_off) atIndex:2];
            [enc setBuffer:m0 offset:m0_off atIndex:3];
            [enc setBuffer:(m1 ? m1 : m0) offset:(m1 ? m1_off : m0_off) atIndex:4];
            [enc setBuffer:(m2 ? m2 : m0) offset:(m2 ? m2_off : m0_off) atIndex:5];
            [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:6]; // aliased `cat_i8` (27 MiB <= 72.6 MiB)
            [enc setBuffer:eng->b_ctrl_cat_sc offset:0 atIndex:7];
            [enc setBytes:&pp length:sizeof(pp) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(M_padded, num_ctrls, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else if (num_ctrls == 3U) {
            SikLoopCtrlPrepParamsHost pp{M_valid, use_nax ? M_padded : M_valid, D, eng->cfg.rms_norm_eps};
            [enc setComputePipelineState:eng->pso_loop_ctrl_prep3];
            [enc setBuffer:c0 offset:c0_off atIndex:0];
            [enc setBuffer:c1 offset:c1_off atIndex:1];
            [enc setBuffer:c2 offset:c2_off atIndex:2];
            [enc setBuffer:m0 offset:m0_off atIndex:3];
            [enc setBuffer:m1 offset:m1_off atIndex:4];
            [enc setBuffer:m2 offset:m2_off atIndex:5];
            [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:6];
            [enc setBytes:&pp length:sizeof(pp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(2, use_nax ? M_padded : M_valid, 3)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            SikLoopCtrlPrepParamsHost pp{M_valid, use_nax ? M_padded : M_valid, D, eng->cfg.rms_norm_eps};
            [enc setComputePipelineState:eng->pso_loop_ctrl_prep];
            [enc setBuffer:c0 offset:c0_off atIndex:0];
            [enc setBuffer:m0 offset:m0_off atIndex:1];
            [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:2];
            [enc setBytes:&pp length:sizeof(pp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(2, use_nax ? M_padded : M_valid, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }

    // Build shared offset struct for down & apply
    SikLoopCtrl3OffsetsHost down_off{};
    SikLoopCtrl3OffsetsHost apply_off{};
    down_off.M_valid = apply_off.M_valid = M_valid;
    down_off.M_padded = apply_off.M_padded = use_nax ? M_padded : M_valid;
    down_off.D = apply_off.D = D;
    down_off.gate_rank = apply_off.gate_rank = R;
    down_off.num_ctrls = apply_off.num_ctrls = num_ctrls;
    for (uint32_t j = 0U; j < num_ctrls; ++j) {
        uint64_t dw = 0ULL, ds = 0ULL, db = 0ULL;
        uint64_t gw = 0ULL, gs = 0ULL, gb = 0ULL;
        uint64_t fw = 0ULL, fs = 0ULL, fb = 0ULL;
        sik_slice_sub_offsets(ctrls[j].down_weight,   dw, ds, db);
        sik_slice_sub_offsets(ctrls[j].gain_weight,   gw, gs, gb);
        sik_slice_sub_offsets(ctrls[j].forget_weight, fw, fs, fb);
        down_off.w_off[j]   = dw; down_off.s_off[j]  = ds;
        apply_off.w_off[j]  = gw; apply_off.s_off[j] = gs;
        apply_off.f_off[j]  = fw; apply_off.fs_off[j]= fs;
    }

    if (use_nax) {
        if (use_i8_ctrl) {
            // Opt-In `--int8-prefill` Direct-Device INT8xINT8->INT32 Loop Controller (`86.8 TOPS`, 0B TG, 0 FP16 cat roundtrip!)
            const bool     use_m64 = ((M_padded % 64U) == 0U);
            const uint32_t bm      = use_m64 ? 64U : 32U;
            const uint32_t th      = use_m64 ? 128U : 64U;

            [enc setComputePipelineState:(use_m64 ? eng->pso_loop_ctrl_down_nax_q8_i8_m64 : eng->pso_loop_ctrl_down_nax_q8_i8_m32)];
            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
            [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:1];
            [enc setBuffer:eng->b_ctrl_cat_sc offset:0 atIndex:2];
            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:3];
            [enc setBytes:&down_off length:sizeof(down_off) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(R / 64U, M_padded / bm, num_ctrls)
                threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];

            SikQuantI8ParamsHost qp_h{M_valid, M_padded, R, 0U, num_ctrls, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_quant_f16_to_i8_row];
            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_ctrl_h_i8 offset:0 atIndex:1];
            [enc setBuffer:eng->b_ctrl_h_sc offset:0 atIndex:2];
            [enc setBuffer:eng->b_ctrl_h_sc offset:0 atIndex:3];
            [enc setBytes:&qp_h length:sizeof(qp_h) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((M_padded + 3U) / 4U, num_ctrls, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:(use_m64 ? eng->pso_loop_ctrl_apply_nax_q8_i8_m64 : eng->pso_loop_ctrl_apply_nax_q8_i8_m32)];
            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
            [enc setBuffer:eng->b_ctrl_h_i8 offset:0 atIndex:1];
            [enc setBuffer:eng->b_ctrl_h_sc offset:0 atIndex:2];
            [enc setBuffer:c0 offset:c0_off atIndex:3];
            [enc setBuffer:(c1 ? c1 : c0) offset:(c1 ? c1_off : c0_off) atIndex:4];
            [enc setBuffer:(c2 ? c2 : c0) offset:(c2 ? c2_off : c0_off) atIndex:5];
            [enc setBuffer:m0 offset:m0_off atIndex:6];
            [enc setBuffer:(m1 ? m1 : m0) offset:(m1 ? m1_off : m0_off) atIndex:7];
            [enc setBuffer:(m2 ? m2 : m0) offset:(m2 ? m2_off : m0_off) atIndex:8];
            [enc setBuffer:o0 offset:o0_off atIndex:9];
            [enc setBuffer:(o1 ? o1 : o0) offset:(o1 ? o1_off : o0_off) atIndex:10];
            [enc setBuffer:(o2 ? o2 : o0) offset:(o2 ? o2_off : o0_off) atIndex:11];
            [enc setBytes:&apply_off length:sizeof(apply_off) atIndex:12];
            [enc dispatchThreadgroups:MTLSizeMake(D / 64U, M_padded / bm, num_ctrls)
                threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
            return;
        }

        // 2. High-Occupancy `BK=128` Prefill NAX `down` (`BM=32` for `M>=256`, `BM=16` for smaller chunks)
        const bool     down_m32 = (M_padded >= 256U && (M_padded % 32U) == 0U);
        const uint32_t down_bm  = down_m32 ? 32U : 16U;
        id<MTLComputePipelineState> pso_dn = is_q8
            ? (down_m32 ? eng->pso_loop_ctrl_down_nax_q8_m32  : eng->pso_loop_ctrl_down_nax_q8)
            : (down_m32 ? eng->pso_loop_ctrl_down_nax_f16_m32 : eng->pso_loop_ctrl_down_nax_f16);
        [enc setComputePipelineState:pso_dn];
        [enc setBuffer:eng->b_uma offset:0 atIndex:0];
        [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:1];
        [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:2];
        [enc setBytes:&down_off length:sizeof(down_off) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(R / 32U, M_padded / down_bm, num_ctrls)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        // 3. Single-Tile K=64 Prefill NAX `apply` (`gain + forget + tanh blend`, ZERO K-loop!)
        const bool use_m64 = (M_padded >= 64U && (M_padded % 64U) == 0U);
        const uint32_t bm  = use_m64 ? 64U : 16U;
        const uint32_t th  = use_m64 ? 128U : 64U;
        id<MTLComputePipelineState> pso_ap = is_q8
            ? (use_m64 ? eng->pso_loop_ctrl_apply_nax_q8_m64  : eng->pso_loop_ctrl_apply_nax_q8_m16)
            : (use_m64 ? eng->pso_loop_ctrl_apply_nax_f16_m64 : eng->pso_loop_ctrl_apply_nax_f16_m16);
        [enc setComputePipelineState:pso_ap];
        [enc setBuffer:eng->b_uma offset:0 atIndex:0];
        [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:1];
        [enc setBuffer:c0 offset:c0_off atIndex:2];
        [enc setBuffer:(c1 ? c1 : c0) offset:(c1 ? c1_off : c0_off) atIndex:3];
        [enc setBuffer:(c2 ? c2 : c0) offset:(c2 ? c2_off : c0_off) atIndex:4];
        [enc setBuffer:m0 offset:m0_off atIndex:5];
        [enc setBuffer:(m1 ? m1 : m0) offset:(m1 ? m1_off : m0_off) atIndex:6];
        [enc setBuffer:(m2 ? m2 : m0) offset:(m2 ? m2_off : m0_off) atIndex:7];
        [enc setBuffer:o0 offset:o0_off atIndex:8];
        [enc setBuffer:(o1 ? o1 : o0) offset:(o1 ? o1_off : o0_off) atIndex:9];
        [enc setBuffer:(o2 ? o2 : o0) offset:(o2 ? o2_off : o0_off) atIndex:10];
        [enc setBytes:&apply_off length:sizeof(apply_off) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(D / 64U, M_padded / bm, num_ctrls)
            threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
        return;
    }

    // Decode (`M <= 8`) path: exact FP32 intermediate `h`
    if (num_ctrls == 3U) {
        if (is_q8) {
            if (h_msg_buf != nil) {
                [enc setComputePipelineState:eng->pso_loop_ctrl_down3_left_q8_m[M_valid - 1U]];
                [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:1];
                [enc setBuffer:h_msg_buf offset:h_msg_off atIndex:2];
                [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:3];
                [enc setBytes:&down_off length:sizeof(down_off) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(R, 3, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            } else {
                [enc setComputePipelineState:eng->pso_loop_ctrl_down3_q8_m[M_valid - 1U]];
                [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:1];
                [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:2];
                [enc setBytes:&down_off length:sizeof(down_off) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(R, 3, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            if (__builtin_expect(eng->prof_active && eng->prof_detail, 0)) {
                [enc endEncoding];
                if (eng->prof_pass_count < 2040U) eng->prof_pass_bkt[eng->prof_pass_count++] = 15U;
                MTLComputePassDescriptor* cpd = [MTLComputePassDescriptor computePassDescriptor];
                cpd.sampleBufferAttachments[0].sampleBuffer = eng->prof_csb;
                cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex = eng->prof_pass_count * 2U;
                cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex   = eng->prof_pass_count * 2U + 1U;
                enc = [cb computeCommandEncoderWithDescriptor:cpd];
            }

            [enc setComputePipelineState:eng->pso_loop_ctrl_apply3_q8_m[M_valid - 1U]];
            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:1];
            [enc setBuffer:c0 offset:c0_off atIndex:2];
            [enc setBuffer:c1 offset:c1_off atIndex:3];
            [enc setBuffer:c2 offset:c2_off atIndex:4];
            [enc setBuffer:m0 offset:m0_off atIndex:5];
            [enc setBuffer:m1 offset:m1_off atIndex:6];
            [enc setBuffer:m2 offset:m2_off atIndex:7];
            [enc setBuffer:o0 offset:o0_off atIndex:8];
            [enc setBuffer:o1 offset:o1_off atIndex:9];
            [enc setBuffer:o2 offset:o2_off atIndex:10];
            [enc setBytes:&apply_off length:sizeof(apply_off) atIndex:11];
            [enc setBuffer:eng->b_rms_sq offset:0 atIndex:12];
            [enc dispatchThreadgroups:MTLSizeMake((D + 31U) / 32U, (M_valid + 7U) / 8U, 3)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            if (__builtin_expect(eng->prof_active && eng->prof_detail, 0)) {
                [enc endEncoding];
                if (eng->prof_pass_count < 2040U) eng->prof_pass_bkt[eng->prof_pass_count++] = 16U;
                MTLComputePassDescriptor* cpd = [MTLComputePassDescriptor computePassDescriptor];
                cpd.sampleBufferAttachments[0].sampleBuffer = eng->prof_csb;
                cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex = eng->prof_pass_count * 2U;
                cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex   = eng->prof_pass_count * 2U + 1U;
                enc = [cb computeCommandEncoderWithDescriptor:cpd];
            }
        } else {
            SikGemvBatchParamsHost gp{M_valid, R, 2U * D, kOutF32, 0.0f, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_loop_ctrl_down3_m[M_valid - 1U]];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[0].down_weight.offset) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[1].down_weight.offset) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[2].down_weight.offset) atIndex:2];
            [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:4];
            [enc setBytes:&gp length:sizeof(gp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(R, 3, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            SikLoopCtrlApplyParamsHost ap{M_valid, M_valid, D, R};
            [enc setComputePipelineState:eng->pso_loop_ctrl_apply3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[0].gain_weight.offset) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[1].gain_weight.offset) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[2].gain_weight.offset) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[0].forget_weight.offset) atIndex:3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[1].forget_weight.offset) atIndex:4];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ctrls[2].forget_weight.offset) atIndex:5];
            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:6];
            [enc setBuffer:c0 offset:c0_off atIndex:7];
            [enc setBuffer:c1 offset:c1_off atIndex:8];
            [enc setBuffer:c2 offset:c2_off atIndex:9];
            [enc setBuffer:m0 offset:m0_off atIndex:10];
            [enc setBuffer:m1 offset:m1_off atIndex:11];
            [enc setBuffer:m2 offset:m2_off atIndex:12];
            [enc setBuffer:o0 offset:o0_off atIndex:13];
            [enc setBuffer:o1 offset:o1_off atIndex:14];
            [enc setBuffer:o2 offset:o2_off atIndex:15];
            [enc setBytes:&ap length:sizeof(ap) atIndex:16];
            [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, (M_valid + 7U) / 8U, 3)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    } else {
        if (is_q8 && M_valid <= kMaxSpecBatch) {
            if (h_msg_buf != nil) {
                [enc setComputePipelineState:eng->pso_loop_ctrl_down1_left_q8_m[M_valid - 1U]];
                [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:1];
                [enc setBuffer:h_msg_buf offset:h_msg_off atIndex:2];
                [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:3];
                [enc setBytes:&down_off length:sizeof(down_off) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                // High-MLP 128-th/TG (`4` SGs/TG -> 256 SIMDs across 40 cores!) beats 32-th by 19% on single-ctrl Down1!
                SikGemvBatchParamsHost dp{M_valid, R, 2U * D, kOutF32, 0.0f, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_q8_row_tg128_m[M_valid - 1U]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(down_off.w_off[0]) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(down_off.s_off[0]) atIndex:1];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:2];
                [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:3];
                [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:4];
                [enc setBytes:&dp length:sizeof(dp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            if (__builtin_expect(eng->prof_active && eng->prof_detail, 0)) {
                [enc endEncoding];
                if (eng->prof_pass_count < 2040U) eng->prof_pass_bkt[eng->prof_pass_count++] = 20U;
                MTLComputePassDescriptor* cpd = [MTLComputePassDescriptor computePassDescriptor];
                cpd.sampleBufferAttachments[0].sampleBuffer = eng->prof_csb;
                cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex = eng->prof_pass_count * 2U;
                cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex   = eng->prof_pass_count * 2U + 1U;
                enc = [cb computeCommandEncoderWithDescriptor:cpd];
            }

            SikLoopCtrlApplyParamsHost ap{M_valid, M_valid, D, R};
            [enc setComputePipelineState:eng->pso_loop_ctrl_apply_q8_m[M_valid - 1U]];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.w_off[0]) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.s_off[0]) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.f_off[0]) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.fs_off[0]) atIndex:3];
            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:4];
            [enc setBuffer:c0 offset:c0_off atIndex:5];
            [enc setBuffer:m0 offset:m0_off atIndex:6];
            [enc setBuffer:o0 offset:o0_off atIndex:7];
            [enc setBytes:&ap length:sizeof(ap) atIndex:8];
            [enc setBuffer:eng->b_rms_sq offset:0 atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake((D + 31U) / 32U, (M_valid + 7U) / 8U, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            if (__builtin_expect(eng->prof_active && eng->prof_detail, 0)) {
                [enc endEncoding];
                if (eng->prof_pass_count < 2040U) eng->prof_pass_bkt[eng->prof_pass_count++] = 21U;
                MTLComputePassDescriptor* cpd = [MTLComputePassDescriptor computePassDescriptor];
                cpd.sampleBufferAttachments[0].sampleBuffer = eng->prof_csb;
                cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex = eng->prof_pass_count * 2U;
                cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex   = eng->prof_pass_count * 2U + 1U;
                enc = [cb computeCommandEncoderWithDescriptor:cpd];
            }
        } else {
            sik_encode_dense_linear(enc, eng, ctrls[0].down_weight, eng->b_ctrl_cat_f16, 0,
                                    nil, 0, eng->b_ctrl_h_f32, 0, M_valid, M_valid, kOutF32, true);
            SikLoopCtrlApplyParamsHost ap{M_valid, M_valid, D, R};
            [enc setComputePipelineState:eng->pso_loop_ctrl_apply];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.w_off[0]) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.f_off[0]) atIndex:1];
            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:2];
            [enc setBuffer:c0 offset:c0_off atIndex:3];
            [enc setBuffer:m0 offset:m0_off atIndex:4];
            [enc setBuffer:o0 offset:o0_off atIndex:5];
            [enc setBytes:&ap length:sizeof(ap) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, (M_valid + 7U) / 8U, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }
}

void sik_encode_loop_ctrl_batch(
    id<MTLCommandBuffer> __strong&         cb,
    id<MTLComputeCommandEncoder> __strong& enc,
    StrataInkling2GpuEngine*          eng,
    const StrataInkling2LoopCtrlDesc* ctrls,     // [num_ctrls] (1 or 3)
    uint32_t                          num_ctrls, // 1 or 3
    id<MTLBuffer>                     c0,
    id<MTLBuffer>                     c1,
    id<MTLBuffer>                     c2,
    id<MTLBuffer>                     m0,
    id<MTLBuffer>                     m1,
    id<MTLBuffer>                     m2,
    id<MTLBuffer>                     o0,
    id<MTLBuffer>                     o1,
    id<MTLBuffer>                     o2,
    uint32_t                          M_valid,
    uint32_t                          M_padded,
    bool                              force_gemv,
    bool                              skip_prep = false,
    NSUInteger                        m0_off    = 0U,
    NSUInteger                        m1_off    = 0U,
    NSUInteger                        m2_off    = 0U,
    id<MTLBuffer>                     h_msg_buf = nil,
    NSUInteger                        h_msg_off = 0U
) {
    if (M_valid == 0U) return;
    const uint32_t D       = eng->cfg.hidden_size;
    const uint32_t R       = eng->cfg.loop_gate_rank;
    const bool     use_nax = (!force_gemv && eng->has_nax && M_valid > kMaxSpecBatch && (D % 64U) == 0U && R == 64U);
    if (!use_nax && M_valid > kMaxSpecBatch) {
        const NSUInteger row_stride = static_cast<NSUInteger>(D) * sizeof(float);
        for (uint32_t row0 = 0U; row0 < M_valid; row0 += kMaxSpecBatch) {
            const uint32_t   mb  = std::min<uint32_t>(kMaxSpecBatch, M_valid - row0);
            const NSUInteger off = static_cast<NSUInteger>(row0) * row_stride;
            sik_encode_loop_ctrl_batch_impl(
                cb, enc, eng, ctrls, num_ctrls,
                c0, c1, c2, m0, m1, m2, o0, o1, o2,
                mb, mb, /*force_gemv=*/true, /*skip_prep=*/false,
                off, off, off,
                m0_off + off, m1_off + off, m2_off + off,
                off, off, off,
                nil, 0U
            );
        }
        return;
    }
    sik_encode_loop_ctrl_batch_impl(
        cb, enc, eng, ctrls, num_ctrls,
        c0, c1, c2, m0, m1, m2, o0, o1, o2,
        M_valid, M_padded, force_gemv, skip_prep,
        0U, 0U, 0U,
        m0_off, m1_off, m2_off,
        0U, 0U, 0U,
        h_msg_buf, h_msg_off
    );
}

void sik_encode_sconv_static(
    id<MTLComputeCommandEncoder> enc,
    StrataInkling2GpuEngine*     eng,
    id<MTLBuffer>                x_in,
    uint64_t                     weight_off,
    uint64_t                     state_byte_off,
    id<MTLBuffer>                out_or_cur,
    id<MTLBuffer>                block_dst,
    NSUInteger                   block_dst_off,
    uint32_t                     M_valid,
    uint32_t                     channels,
    uint32_t                     save_snap_mask,
    uint32_t                     res_mode,
    bool                         use_tree = false
) {
    const uint32_t snap_stride_f4 = static_cast<uint32_t>(eng->sconv_total_bytes / 16ULL);
    SikSconvParamsHost sp{
        M_valid, channels, save_snap_mask, res_mode,
        eng->cfg.head_dim, eng->cfg.num_key_value_heads, snap_stride_f4, 0U
    };
    if (M_valid > kMaxSpecBatch && save_snap_mask == 0U) {
        // 2D `(c, m)` Parallel FIR-4 Prefill SConv (saturates all 40 cores!) + tiny `<1 us` tail state save!
        [enc setComputePipelineState:eng->pso_sconv_static_prefill];
        [enc setBuffer:x_in offset:0 atIndex:0];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(weight_off) atIndex:1];
        [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(state_byte_off) atIndex:2];
        [enc setBuffer:out_or_cur offset:0 atIndex:3];
        [enc setBuffer:(block_dst ? block_dst : out_or_cur) offset:block_dst_off atIndex:4];
        [enc setBytes:&sp length:sizeof(sp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((channels + 255U) / 256U, (M_valid + 15U) / 16U, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:eng->pso_sconv_save_tail];
        [enc setBuffer:x_in offset:0 atIndex:0];
        [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(state_byte_off) atIndex:1];
        [enc setBytes:&sp length:sizeof(sp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((channels + 255U) / 256U, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return;
    }

    if (use_tree && eng->b_tree_meta != nil) {
        [enc setComputePipelineState:eng->pso_sconv_static_tree];
    } else {
        [enc setComputePipelineState:eng->pso_sconv_static];
    }
    [enc setBuffer:x_in offset:0 atIndex:0];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(weight_off) atIndex:1];
    [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(state_byte_off) atIndex:2];
    [enc setBuffer:out_or_cur offset:0 atIndex:3];
    [enc setBuffer:(block_dst ? block_dst : out_or_cur) offset:block_dst_off atIndex:4];
    [enc setBuffer:(eng->b_sconv_snaps ? eng->b_sconv_snaps : eng->b_sconv_state)
            offset:static_cast<NSUInteger>(state_byte_off) atIndex:5];
    [enc setBytes:&sp length:sizeof(sp) atIndex:6];
    if (use_tree && eng->b_tree_meta != nil) {
        [enc setBuffer:eng->b_tree_meta offset:0 atIndex:7];
    }
    [enc dispatchThreadgroups:MTLSizeMake((channels + 31U) / 32U, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

inline id<MTLComputeCommandEncoder> sik_prof_new_enc(
    StrataInkling2GpuEngine* eng,
    id<MTLCommandBuffer>     cb
) {
    if (eng->prof_use_csb && eng->prof_csb && eng->prof_pass_count < 2040U) {
        MTLComputePassDescriptor* cpd = [MTLComputePassDescriptor computePassDescriptor];
        cpd.sampleBufferAttachments[0].sampleBuffer              = eng->prof_csb;
        cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex = eng->prof_pass_count * 2U;
        cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex   = eng->prof_pass_count * 2U + 1U;
        return [cb computeCommandEncoderWithDescriptor:cpd];
    }
    return [cb computeCommandEncoder];
}

inline void sik_prof_mark(
    StrataInkling2GpuEngine*               eng,
    id<MTLCommandBuffer> __strong&         cb,
    id<MTLComputeCommandEncoder> __strong& enc,
    uint32_t                               bucket
) {
    if (__builtin_expect(!eng->prof_active, 1)) return;
    [enc endEncoding];
    if (eng->prof_use_csb && eng->prof_csb) {
        if (eng->prof_pass_count < 2040U) {
            eng->prof_pass_bkt[eng->prof_pass_count] = static_cast<uint8_t>(bucket);
            eng->prof_pass_count += 1U;
        }
        enc = sik_prof_new_enc(eng, cb);
        return;
    }
    [cb commit];
    [cb waitUntilCompleted];
    const double gs = [cb GPUStartTime];
    const double ge = [cb GPUEndTime];
    if (ge > gs && bucket < 28U) { eng->prof_ms[bucket] += (ge - gs) * 1000.0; eng->prof_cnt[bucket] += 1U; }
    cb  = [eng->queue commandBuffer];
    enc = [cb computeCommandEncoder];
}

inline void sik_prof_mark_detail(
    StrataInkling2GpuEngine*               eng,
    id<MTLCommandBuffer> __strong&         cb,
    id<MTLComputeCommandEncoder> __strong& enc,
    uint32_t                               bucket
) {
    if (__builtin_expect(!(eng->prof_active && eng->prof_detail), 1)) return;
    sik_prof_mark(eng, cb, enc, bucket);
}

inline void sik_prof_resolve_csb(StrataInkling2GpuEngine* eng, id<MTLCommandBuffer> cb, const char* tag, uint32_t M_valid, uint32_t pos) {
    if (!eng || !eng->prof_use_csb || !eng->prof_csb || eng->prof_pass_count == 0U) return;
    const uint32_t n = eng->prof_pass_count;
    NSData* data = [eng->prof_csb resolveCounterRange:NSMakeRange(0U, static_cast<NSUInteger>(n) * 2U)];
    double gap_ms = 0.0;
    if (data && [data length] >= static_cast<NSUInteger>(n) * 2U * sizeof(uint64_t)) {
        const uint64_t* ts = reinterpret_cast<const uint64_t*>([data bytes]);
        for (uint32_t i = 0U; i < n; ++i) {
            const uint64_t t0 = ts[2U * i + 0U];
            const uint64_t t1 = ts[2U * i + 1U];
            const uint32_t b  = eng->prof_pass_bkt[i];
            if (t1 >= t0 && b < 28U) { eng->prof_ms[b] += static_cast<double>(t1 - t0) * 1e-6; eng->prof_cnt[b] += 1U; }
            if (i > 0U && t0 >= ts[2U * i - 1U]) gap_ms += static_cast<double>(t0 - ts[2U * i - 1U]) * 1e-6;
        }
    }
    const double cb_gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
    static const char* kNames[28] = {
        "Embed", "PreAttn/Loop", "AttnProj", "SConvKV+QKVR", "Attention", "Wo+AttnSConv",
        "PreMoE/Loop", "MoE_Rtr/Bin", "MoE_W13", "MoE_W2", "MoE_Comb+SConv", "Head+MTP",
        "PassCopy+Tap", "PreAttn_ARNorm", "PreAttn_Prep3", "PreAttn_Down3", "PreAttn_Apply3", "PreAttn_RMS3",
        "PreMoE_ARNorm", "PreMoE_Prep1", "PreMoE_Down1", "PreMoE_Apply1", "PreMoE_RMS1",
        "MoE_Router", "MoE_TopK_Bin", "Head_FinalNorm", "Head_Unembed", "Head_Sample"
    };
    double tot = 0.0;
    for (uint32_t i = 0U; i < 28U; ++i) tot += eng->prof_ms[i];
    std::fprintf(stderr, "[STRATA_PROFILE_INKLING2] %s (M=%u, pos=%u, kernels=%.2f ms, cb_gpu=%.2f ms, enc_gaps=%.2f ms, passes=%u, %.1f tok/s):\n",
                 tag, M_valid, pos, tot, cb_gpu_ms, gap_ms, n, (cb_gpu_ms > 0.0) ? (1000.0 * M_valid / cb_gpu_ms) : 0.0);
    for (uint32_t i = 0U; i < 28U; ++i) {
        if (eng->prof_ms[i] > 0.0005) {
            std::fprintf(stderr, "  [%2u] %-15s : %7.3f ms (%5.1f%%) [%3ux, %.1f us/call]\n",
                         i, kNames[i], eng->prof_ms[i], (tot > 0.0) ? (100.0 * eng->prof_ms[i] / tot) : 0.0,
                         eng->prof_cnt[i], eng->prof_cnt[i] ? (eng->prof_ms[i] * 1000.0 / eng->prof_cnt[i]) : 0.0);
        }
    }
    std::fflush(stderr);
}

inline void sik_prof_dump(const StrataInkling2GpuEngine* eng, const char* tag, uint32_t M_valid, uint32_t pos) {
    if (!eng || !eng->prof_active) return;
    static const char* kNames[12] = {
        "Embed", "PreAttn/Loop", "AttnProj", "SConvKV+QKVR", "Attention",
        "Wo+AttnSConv", "PreMoE/Loop", "MoE_Rtr/Bin", "MoE_W13", "MoE_W2",
        "MoE_Comb+SConv", "Head+MTP"
    };
    double tot = 0.0;
    for (uint32_t i = 0U; i < 12U; ++i) tot += eng->prof_ms[i];
    std::fprintf(stderr, "[STRATA_PROFILE_INKLING2] %s (M=%u, pos=%u, total=%.2f ms, %.1f tok/s):\n",
                 tag, M_valid, pos, tot, (tot > 0.0) ? (1000.0 * M_valid / tot) : 0.0);
    for (uint32_t i = 0U; i < 12U; ++i) {
        if (eng->prof_ms[i] > 0.001) {
            std::fprintf(stderr, "  [%2u] %-15s : %7.2f ms (%5.1f%%)\n",
                         i, kNames[i], eng->prof_ms[i], (tot > 0.0) ? (100.0 * eng->prof_ms[i] / tot) : 0.0);
        }
    }
}

/* ============================================================================
 * 48-Visit Looped Backbone Forward Encoder
 * ============================================================================ */

void sik_encode_backbone_forward(
    id<MTLCommandBuffer> __strong&         cb,
    id<MTLComputeCommandEncoder> __strong& enc,
    StrataInkling2GpuEngine*               eng,
    id<MTLBuffer>                          tok_ids_buf,
    uint32_t                               seq_pos_start,
    uint32_t                               M_valid,
    uint32_t                               M_padded,
    uint32_t                               save_snap_mask,
    bool                                   force_gemv,
    bool                                   capture_mtp_taps,
    bool                                   run_unembed_all,
    bool                                   run_unembed_last,
    bool                                   run_final_norm = false,
    bool                                   use_tree       = false,
    const SikChunkImageCopy*               vis_copies     = nullptr,
    uint32_t                               num_vis_copies = 0U
) {
    const StrataInkling2Config& cfg = eng->cfg;
    const uint32_t D         = cfg.hidden_size;
    const uint32_t n_q       = cfg.num_attention_heads;
    const uint32_t n_kv      = cfg.num_key_value_heads;
    const uint32_t hd        = cfg.head_dim;
    const uint32_t kv_dim    = n_kv * hd;
    const uint32_t inter     = cfg.intermediate_size;
    const uint32_t E         = cfg.num_experts;
    const uint32_t top_k     = cfg.num_experts_per_tok;
    const uint32_t core_beg  = cfg.loop_core_start;
    const uint32_t core_end  = cfg.loop_core_end;
    const uint32_t n_visits  = cfg.num_logical_layers;
    const bool     mtp_on    = (eng->hdr.mtp.enabled != 0U);
    const size_t   cur_bytes = static_cast<size_t>(M_valid) * D * sizeof(float);
    const bool     is_prefill_nax   = (!force_gemv && eng->has_nax && M_valid > kMaxSpecBatch);
    const bool     use_int8_prefill = (eng->int8_prefill_enabled != 0U && is_prefill_nax && M_valid >= 32U && (M_padded % 32U) == 0U);
    const uint32_t M_pad16          = (M_valid + 15U) & ~15U;
    const bool     tree_active      = (use_tree && !is_prefill_nax && M_valid >= 2U && M_valid <= kMaxSpecBatch && eng->b_tree_meta != nil);

    // 1. Embedding (FP16 or Q4_A) + embed_norm -> `b_cur [M_valid, D]` F32
    {
        SikEmbedParamsHost ep{M_valid, D, cfg.vocab_size_padded, cfg.rms_norm_eps};
        if (eng->hdr.embed_tokens.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
            uint64_t ew = 0ULL, es = 0ULL, eb = 0ULL;
            sik_slice_sub_offsets(eng->hdr.embed_tokens, ew, es, eb);
            [enc setComputePipelineState:eng->pso_embed_rmsnorm_q4a];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ew) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(es) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eb) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_norm.offset) atIndex:3];
            [enc setBuffer:tok_ids_buf offset:0 atIndex:4];
            [enc setBuffer:eng->b_cur offset:0 atIndex:5];
            [enc setBytes:&ep length:sizeof(ep) atIndex:6];
        } else {
            [enc setComputePipelineState:eng->pso_embed_rmsnorm];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.offset) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_norm.offset) atIndex:1];
            [enc setBuffer:tok_ids_buf offset:0 atIndex:2];
            [enc setBuffer:eng->b_cur offset:0 atIndex:3];
            [enc setBytes:&ep length:sizeof(ep) atIndex:4];
        }
        [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    // 1b. Multimodal Vision Embedding Injection (post-`embed_norm`, zero overhead when `num_vis_copies == 0`!)
    if (num_vis_copies > 0U && vis_copies != nullptr && eng->b_vis_stage != nil) {
        for (uint32_t ci = 0U; ci < num_vis_copies; ++ci) {
            const SikChunkImageCopy& vc = vis_copies[ci];
            const NSUInteger dst_off = static_cast<NSUInteger>(vc.dst_row) * D * sizeof(float);
            const size_t     n_bytes = static_cast<size_t>(vc.num_rows) * D * sizeof(float);
            sik_encode_copy_buffer(enc, eng->pso_copy_u32,
                                   eng->b_vis_stage, vc.src_byte_off,
                                   eng->b_cur, dst_off, n_bytes);
        }
    }
    sik_prof_mark(eng, cb, enc, 0U);

    uint32_t num_blocks         = 0U;
    uint32_t prelude_num_blocks = 0U;
    uint32_t bank_count         = 0U;
    const bool capture_taps     = mtp_on && capture_mtp_taps;
    const uint32_t core_len     = (core_end > core_beg) ? (core_end - core_beg) : 0U;
    bool use_batch12_loop       = (!is_prefill_nax && M_valid <= kMaxSpecBatch &&
                                   D == 2304U && core_len > 0U && core_len <= 12U && cfg.loop_num_passes > 1U);
    if (use_batch12_loop) {
        for (uint32_t li = 0U; li < core_len; ++li) {
            const StrataInkling2LayerDesc& cld = eng->hdr.layers[core_beg + li];
            for (uint32_t j = 0U; j < 4U; ++j) {
                if (cld.controllers[j].down_weight.dtype != STRATA_INKLING2_DTYPE_Q8_0) {
                    use_batch12_loop = false;
                    break;
                }
            }
            if (!use_batch12_loop) break;
        }
    }

    // 2. 48 Logical Layer Visits (`il = 0..47`)
    for (uint32_t il = 0U; il < n_visits; ++il) {
        const uint32_t ip       = cfg.logical_to_phys[il];
        const uint32_t pass     = cfg.logical_pass_idx[il];
        const bool     is_local = (cfg.logical_is_local[il] != 0U);
        const uint32_t kv_slot  = cfg.logical_kv_slot[il];
        const StrataInkling2LayerDesc& ld = eng->hdr.layers[ip];

        // 2a. Loop Core Pass Boundary (`ip == core_start`)
        if (cfg.loop_num_passes > 1U && ip >= core_beg && ip < core_end) {
            if (ip == core_beg) {
                if (pass == 0U) {
                    prelude_num_blocks = num_blocks;
                    sik_encode_copy_buffer(enc, eng->pso_copy_u32,
                                           eng->b_cur, 0, eng->b_prelude_cur, 0, cur_bytes);
                } else {
                    const uint32_t src_start = prelude_num_blocks + 1U;
                    bank_count = (num_blocks > src_start) ? std::min<uint32_t>(num_blocks - src_start, kMaxBankBlocks) : 0U;
                    // Single-Dispatch Fused Pass-Boundary Copy (replaces 6 serial copy dispatches!)
                    SikPassCopyParamsHost cp{
                        M_valid, D, bank_count, src_start,
                        static_cast<uint32_t>(eng->block_slot_elems), 0U, 0U, 0U
                    };
                    [enc setComputePipelineState:eng->pso_pass_boundary_copy];
                    [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                    [enc setBuffer:eng->b_prelude_cur offset:0 atIndex:1];
                    [enc setBuffer:eng->b_cur offset:0 atIndex:2];
                    [enc setBuffer:eng->b_bank offset:0 atIndex:3];
                    [enc setBuffer:eng->b_bank_last offset:0 atIndex:4];
                    [enc setBytes:&cp length:sizeof(cp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(((D / 4U) + 255U) / 256U, M_valid, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    num_blocks = prelude_num_blocks;

                    if (use_batch12_loop) {
                        // Batch-12 Precompute: ALL 12 core layers' `msg[0..3]` + `rms(msg)` + `down_w_right @ rms(msg)`!
                        // Replaces 24 serial Side-0 runs, 24 `msg3` norms, and halves per-layer `Down3`/`Down1` width!
                        SikAttnResParamsHost bp{
                            M_valid, D, bank_count, static_cast<uint32_t>(eng->block_slot_elems),
                            core_len, cfg.rms_norm_eps, 0U, 0U
                        };
                        [enc setComputePipelineState:eng->pso_cross_pass_batch12_576];
                        [enc setBuffer:eng->b_bank offset:0 atIndex:0];
                        [enc setBuffer:eng->b_bank_last offset:0 atIndex:1];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_queries.offset) atIndex:2];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_kv_queries.offset) atIndex:3];
                        [enc setBuffer:eng->b_msg_bank12 offset:0 atIndex:4];
                        [enc setBuffer:eng->b_msg_norm12_f16 offset:0 atIndex:5];
                        [enc setBytes:&bp length:sizeof(bp) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(M_valid, core_len, 1)
                            threadsPerThreadgroup:MTLSizeMake(576, 1, 1)];

                        SikLoopBatch12OffsetsHost bo{};
                        bo.M_valid    = M_valid;
                        bo.D          = D;
                        bo.gate_rank  = cfg.loop_gate_rank;
                        bo.num_layers = core_len;
                        for (uint32_t li = 0U; li < core_len; ++li) {
                            const StrataInkling2LayerDesc& cld = eng->hdr.layers[core_beg + li];
                            for (uint32_t j = 0U; j < 4U; ++j) {
                                uint64_t dummy_b = 0ULL;
                                sik_slice_sub_offsets(cld.controllers[j].down_weight, bo.w_off[li][j], bo.s_off[li][j], dummy_b);
                            }
                        }
                        [enc setComputePipelineState:eng->pso_loop_ctrl_down_right_batch12_q8_m[M_valid - 1U]];
                        [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                        [enc setBuffer:eng->b_msg_norm12_f16 offset:0 atIndex:1];
                        [enc setBuffer:eng->b_ctrl_h_msg12 offset:0 atIndex:2];
                        [enc setBytes:&bo length:sizeof(bo) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(cfg.loop_gate_rank, 4, core_len)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }
                }
                sik_prof_mark_detail(eng, cb, enc, 12U);
            }
        }

        // 2b. MTP Target Feature Tap Capture (only when MTP context/draft taps are needed!)
        if (capture_taps) {
            for (uint32_t t = 0U; t < STRATA_INKLING2_MAX_MTP_TAPS - 1U; ++t) {
                if (il == eng->hdr.mtp.target_layer_ids[t]) {
                    SikCaptureTapParamsHost tp{M_valid, D, t, STRATA_INKLING2_MAX_MTP_TAPS};
                    [enc setComputePipelineState:eng->pso_capture_tap];
                    [enc setBuffer:eng->b_cur offset:0 atIndex:0];
                    [enc setBuffer:eng->b_mtp_taps_chunk offset:0 atIndex:1];
                    [enc setBytes:&tp length:sizeof(tp) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(((D / 4U) + 255U) / 256U, M_valid, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }
        }

        // 2c-2f. Cross-Pass Messages + Pre-Attention Block AttnRes + Loop Controllers 0..2 + Input RMSNorms!
        const bool has_msgs       = (pass > 0U && ip >= core_beg && ip < core_end);
        const bool has_attn_res_q = (ip > 0U && ld.attn_res_query.elem_count > 0U);
        const bool fused_pre_attn = (has_msgs && has_attn_res_q && !is_prefill_nax && M_valid <= kMaxSpecBatch);
        if (has_msgs && !fused_pre_attn) {
            const uint64_t row_off = static_cast<uint64_t>(2U * (ip - core_beg)) * D * sizeof(float);
            SikAttnResParamsHost ap{
                M_valid, D, bank_count, static_cast<uint32_t>(eng->block_slot_elems),
                4U, cfg.rms_norm_eps, 0U, 0U
            };
            [enc setComputePipelineState:eng->pso_cross_pass_attn_res4];
            [enc setBuffer:eng->b_bank offset:0 atIndex:0];
            [enc setBuffer:eng->b_bank_last offset:0 atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_queries.offset + row_off) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_kv_queries.offset + row_off) atIndex:3];
            [enc setBuffer:eng->b_messages[0] offset:0 atIndex:4];
            [enc setBuffer:eng->b_messages[1] offset:0 atIndex:5];
            [enc setBuffer:eng->b_messages[2] offset:0 atIndex:6];
            [enc setBuffer:eng->b_messages[3] offset:0 atIndex:7];
            [enc setBytes:&ap length:sizeof(ap) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        id<MTLBuffer> in_q = eng->b_in_f16[0];
        id<MTLBuffer> in_k = eng->b_in_f16[1];
        id<MTLBuffer> in_v = eng->b_in_f16[2];
        id<MTLBuffer> xs_q = eng->b_x_blk_sum[0];
        id<MTLBuffer> xs_k = eng->b_x_blk_sum[1];
        id<MTLBuffer> xs_v = eng->b_x_blk_sum[2];

        if (has_attn_res_q && !has_msgs) {
            // FUSED 1-DISPATCH `AttnRes + RMSNorm` (eliminates intermediate DRAM roundtrip & 1 dispatch!)
            if (is_local) {
                sik_encode_attn_res_rmsnorm(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                            ld.attn_res_query.offset, ld.attn_norm.offset,
                                            eng->b_in_f16[0], nil, false, M_valid, M_padded);
                in_k = in_q; in_v = in_q;
                xs_k = xs_q; xs_v = xs_q;
            } else {
                SikAttnResNormParamsHost ap{
                    M_valid, M_padded, D, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    cfg.rms_norm_eps, 0U, 0U, 0U, 0U, 0U, 0U
                };
                const bool use_576 = (D == 2304U);
                [enc setComputePipelineState:(use_576 ? eng->pso_attn_res3_rmsnorm3_576 : eng->pso_attn_res3_rmsnorm3)];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_res_query.offset) atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.q_input_norm.offset) atIndex:3];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.k_input_norm.offset) atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.v_input_norm.offset) atIndex:5];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:6];
                [enc setBuffer:eng->b_in_f16[1] offset:0 atIndex:7];
                [enc setBuffer:eng->b_in_f16[2] offset:0 atIndex:8];
                [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:10];
                [enc setBuffer:eng->b_x_blk_sum[1] offset:0 atIndex:11];
                [enc setBuffer:eng->b_x_blk_sum[2] offset:0 atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake(M_padded, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(use_576 ? 576 : 256, 1, 1)];
            }
        } else {
            id<MTLBuffer> inp_src[3] = {eng->b_cur, eng->b_cur, eng->b_cur};
            const bool batch12_here = (fused_pre_attn && use_batch12_loop);
            if (batch12_here) {
                // 576-Thread 1-Pass Block AttnRes + Left-Half RMSNorm (CrossPass & DownRight already precomputed!)
                SikAttnResParamsHost lp{
                    M_valid, D, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    is_local ? 1U : 3U, cfg.rms_norm_eps, 3U, 0U
                };
                id<MTLBuffer> i0 = is_local ? eng->b_x_f32 : eng->b_inputs[0];
                id<MTLBuffer> i1 = is_local ? eng->b_x_f32 : eng->b_inputs[1];
                id<MTLBuffer> i2 = is_local ? eng->b_x_f32 : eng->b_inputs[2];
                [enc setComputePipelineState:eng->pso_attn_res_prep_left_576];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_res_query.offset) atIndex:2];
                [enc setBuffer:i0 offset:0 atIndex:3];
                [enc setBuffer:i1 offset:0 atIndex:4];
                [enc setBuffer:i2 offset:0 atIndex:5];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:6];
                [enc setBytes:&lp length:sizeof(lp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(576, 1, 1)];
                sik_prof_mark_detail(eng, cb, enc, 14U);
                inp_src[0] = i0; inp_src[1] = i1; inp_src[2] = i2;
            } else if (fused_pre_attn) {
                // FUSED 1-DISPATCH PARALLEL `cross_pass_attn_res4 + attn_res + loop_ctrl_prep3` (L2-hot!)
                const uint64_t row_off = static_cast<uint64_t>(2U * (ip - core_beg)) * D * sizeof(float);
                SikLoopPreAttnParamsHost lp{
                    M_valid, D, bank_count, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    is_local ? 1U : 3U, cfg.rms_norm_eps, 0U
                };
                id<MTLBuffer> i0 = is_local ? eng->b_x_f32 : eng->b_inputs[0];
                id<MTLBuffer> i1 = is_local ? eng->b_x_f32 : eng->b_inputs[1];
                id<MTLBuffer> i2 = is_local ? eng->b_x_f32 : eng->b_inputs[2];
                [enc setComputePipelineState:eng->pso_loop_pre_attn_prep3];
                [enc setBuffer:eng->b_bank offset:0 atIndex:0];
                [enc setBuffer:eng->b_bank_last offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_queries.offset + row_off) atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_kv_queries.offset + row_off) atIndex:3];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:4];
                [enc setBuffer:eng->b_cur offset:0 atIndex:5];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_res_query.offset) atIndex:6];
                [enc setBuffer:eng->b_messages[0] offset:0 atIndex:7];
                [enc setBuffer:eng->b_messages[1] offset:0 atIndex:8];
                [enc setBuffer:eng->b_messages[2] offset:0 atIndex:9];
                [enc setBuffer:eng->b_messages[3] offset:0 atIndex:10];
                [enc setBuffer:i0 offset:0 atIndex:11];
                [enc setBuffer:i1 offset:0 atIndex:12];
                [enc setBuffer:i2 offset:0 atIndex:13];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:14];
                [enc setBytes:&lp length:sizeof(lp) atIndex:15];
                [enc dispatchThreadgroups:MTLSizeMake(M_valid, 2, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                sik_prof_mark_detail(eng, cb, enc, 14U);
                inp_src[0] = i0; inp_src[1] = i1; inp_src[2] = i2;
            } else if (has_attn_res_q) {
                if (is_local) {
                    sik_encode_attn_res(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                        ld.attn_res_query.offset, 1U, eng->b_x_f32, nil, nil, M_valid);
                    inp_src[0] = inp_src[1] = inp_src[2] = eng->b_x_f32;
                } else {
                    sik_encode_attn_res(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                        ld.attn_res_query.offset, 3U,
                                        eng->b_inputs[0], eng->b_inputs[1], eng->b_inputs[2], M_valid);
                    inp_src[0] = eng->b_inputs[0];
                    inp_src[1] = eng->b_inputs[1];
                    inp_src[2] = eng->b_inputs[2];
                }
            }
            if (has_msgs) {
                if (batch12_here) {
                    const uint32_t li = ip - core_beg;
                    const NSUInteger mslot = static_cast<NSUInteger>(M_valid) * D * sizeof(float);
                    const NSUInteger hslot = static_cast<NSUInteger>(M_valid) * cfg.loop_gate_rank * sizeof(float);
                    sik_encode_loop_ctrl_batch(
                        cb, enc, eng, &ld.controllers[0], 3U,
                        inp_src[0], inp_src[1], inp_src[2],
                        eng->b_msg_bank12, eng->b_msg_bank12, eng->b_msg_bank12,
                        eng->b_inputs[0], eng->b_inputs[1], eng->b_inputs[2],
                        M_valid, M_padded, force_gemv, /*skip_prep=*/true,
                        static_cast<NSUInteger>(li * 4U + 0U) * mslot,
                        static_cast<NSUInteger>(li * 4U + 1U) * mslot,
                        static_cast<NSUInteger>(li * 4U + 2U) * mslot,
                        eng->b_ctrl_h_msg12,
                        static_cast<NSUInteger>(li * 4U + 0U) * hslot
                    );
                } else {
                    sik_encode_loop_ctrl_batch(
                        cb, enc, eng, &ld.controllers[0], 3U,
                        inp_src[0], inp_src[1], inp_src[2],
                        eng->b_messages[0], eng->b_messages[1], eng->b_messages[2],
                        eng->b_inputs[0], eng->b_inputs[1], eng->b_inputs[2],
                        M_valid, M_padded, force_gemv, /*skip_prep=*/fused_pre_attn
                    );
                }
                inp_src[0] = eng->b_inputs[0];
                inp_src[1] = eng->b_inputs[1];
                inp_src[2] = eng->b_inputs[2];
            }
            if (is_local && !has_msgs) {
                sik_encode_rmsnorm(enc, eng, inp_src[0], 0, ld.attn_norm.offset, true,
                                   eng->b_in_f16[0], nil, false, M_valid, M_padded);
                in_k = in_q; in_v = in_q;
                xs_k = xs_q; xs_v = xs_q;
            } else if (has_msgs && !is_prefill_nax && M_valid <= kMaxSpecBatch &&
                       ld.controllers[0].down_weight.dtype == STRATA_INKLING2_DTYPE_Q8_0) {
                // Zero-Barrier 108-TG (`1` SG/block) 1-Pass RMSNorm3 + `xsum` consuming `b_rms_sq` from `apply3_q8`!
                const uint64_t nw0 = is_local ? ld.attn_norm.offset : ld.q_input_norm.offset;
                const uint64_t nw1 = is_local ? ld.attn_norm.offset : ld.k_input_norm.offset;
                const uint64_t nw2 = is_local ? ld.attn_norm.offset : ld.v_input_norm.offset;
                SikRmsNormParamsHost np{M_valid, M_padded, D, cfg.rms_norm_eps, (D + 31U) / 32U, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_rmsnorm3_fast];
                [enc setBuffer:inp_src[0] offset:0 atIndex:0];
                [enc setBuffer:inp_src[1] offset:0 atIndex:1];
                [enc setBuffer:inp_src[2] offset:0 atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw0) atIndex:3];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw1) atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw2) atIndex:5];
                [enc setBuffer:eng->b_rms_sq offset:0 atIndex:6];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:7];
                [enc setBuffer:eng->b_in_f16[1] offset:0 atIndex:8];
                [enc setBuffer:eng->b_in_f16[2] offset:0 atIndex:9];
                [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:10];
                [enc setBuffer:eng->b_x_blk_sum[1] offset:0 atIndex:11];
                [enc setBuffer:eng->b_x_blk_sum[2] offset:0 atIndex:12];
                [enc setBytes:&np length:sizeof(np) atIndex:13];
                [enc dispatchThreadgroups:MTLSizeMake(D / 64U, 3, M_valid)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                const uint64_t nw0 = is_local ? ld.attn_norm.offset : ld.q_input_norm.offset;
                const uint64_t nw1 = is_local ? ld.attn_norm.offset : ld.k_input_norm.offset;
                const uint64_t nw2 = is_local ? ld.attn_norm.offset : ld.v_input_norm.offset;
                SikRmsNormParamsHost np{M_valid, M_padded, D, cfg.rms_norm_eps, 1U, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_rmsnorm3];
                [enc setBuffer:inp_src[0] offset:0 atIndex:0];
                [enc setBuffer:inp_src[1] offset:0 atIndex:1];
                [enc setBuffer:inp_src[2] offset:0 atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw0) atIndex:3];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw1) atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw2) atIndex:5];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:6];
                [enc setBuffer:eng->b_in_f16[1] offset:0 atIndex:7];
                [enc setBuffer:eng->b_in_f16[2] offset:0 atIndex:8];
                [enc setBytes:&np length:sizeof(np) atIndex:9];
                [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:10];
                [enc setBuffer:eng->b_x_blk_sum[1] offset:0 atIndex:11];
                [enc setBuffer:eng->b_x_blk_sum[2] offset:0 atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake(M_padded, 3, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        // 2g. Self-Attention Projections:
        //     - Decode/Verify (`M <= 8`): SINGLE-DISPATCH mixed-precision (`Q4_A w/ xsum`, `Q8_0`, `FP16`) `attn_qgrs_mixed`!
        //     - Prefill (`M > 8`): Metal 4.1 Direct/Quantized NAX GEMMs!
        sik_prof_mark(eng, cb, enc, (eng->prof_detail) ? (has_msgs ? 17U : 13U) : 1U);
        if (!is_prefill_nax && M_valid <= kMaxSpecBatch) {
            const StrataInkling2TensorSlice slices[8] = {
                ld.wq, ld.gate_attn, ld.wr, ld.attn_sink_proj,
                ld.wk, ld.wv,
                is_local ? StrataInkling2TensorSlice{} : ld.k_sconv_proj,
                is_local ? StrataInkling2TensorSlice{} : ld.v_sconv_proj
            };
            SikAttnQgrsMixedParamsHost qp{};
            qp.M    = M_valid;
            qp.cols = D;
            uint32_t tot_rows = 0U;
            for (uint32_t j = 0U; j < 8U; ++j) {
                qp.rows[j]  = slices[j].shape0;
                qp.dtype[j] = slices[j].dtype;
                if (slices[j].shape0 > 0U) {
                    sik_slice_sub_offsets(slices[j], qp.w_off[j], qp.s_off[j], qp.b_off[j]);
                    tot_rows += slices[j].shape0;
                }
            }
            [enc setComputePipelineState:eng->pso_attn_qgrs_mixed_m[M_valid - 1U]];
            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
            [enc setBuffer:in_q offset:0 atIndex:1];
            [enc setBuffer:in_k offset:0 atIndex:2];
            [enc setBuffer:in_v offset:0 atIndex:3];
            [enc setBuffer:eng->b_q_raw offset:0 atIndex:4];
            [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:5];
            [enc setBuffer:eng->b_r_raw offset:0 atIndex:6];
            [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:7];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:8];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:9];
            [enc setBuffer:eng->b_k_dyn offset:0 atIndex:10];
            [enc setBuffer:eng->b_v_dyn offset:0 atIndex:11];
            [enc setBytes:&qp length:sizeof(qp) atIndex:12];
            [enc setBuffer:xs_q offset:0 atIndex:13];
            [enc setBuffer:xs_k offset:0 atIndex:14];
            [enc setBuffer:xs_v offset:0 atIndex:15];
            [enc dispatchThreadgroups:MTLSizeMake((tot_rows + 7U) / 8U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            // Concurrent dispatch across all 6-8 independent Attention Projections (eliminates 5-7 tail barriers/visit!)
            [enc endEncoding];
            id<MTLComputeCommandEncoder> cenc_ap = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            const bool ap_i8 = use_int8_prefill &&
                (ld.wq.dtype == STRATA_INKLING2_DTYPE_Q4_A ||
                 ld.wq.dtype == STRATA_INKLING2_DTYPE_Q8_0 ||
                 ld.wr.dtype == STRATA_INKLING2_DTYPE_Q8_0);
            if (ap_i8) {
                const bool same_qkv = (in_k == in_q && in_v == in_q);
                const uint32_t need_xsum = 0U; // Whole-row 8KiB Q4_A INT8xINT8 needs ZERO x_sum!
                if (same_qkv) {
                    SikQuantI8ParamsHost qp_in{M_valid, M_padded, D, need_xsum, 1U, 0U, 0U, 0U};
                    [cenc_ap setComputePipelineState:eng->pso_quant_f16_to_i8_row];
                    [cenc_ap setBuffer:in_q offset:0 atIndex:0];
                    [cenc_ap setBuffer:eng->b_in_i8[0] offset:0 atIndex:1];
                    [cenc_ap setBuffer:eng->b_in_xsc[0] offset:0 atIndex:2];
                    [cenc_ap setBuffer:eng->b_in_xsum_f16[0] offset:0 atIndex:3];
                    [cenc_ap setBytes:&qp_in length:sizeof(qp_in) atIndex:4];
                    [cenc_ap dispatchThreadgroups:MTLSizeMake((M_padded + 3U) / 4U, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    SikQuantI8ParamsHost qp_in{M_valid, M_padded, D, need_xsum, 3U, 0U, 0U, 0U};
                    [cenc_ap setComputePipelineState:eng->pso_quant_f16_to_i8_row3];
                    [cenc_ap setBuffer:in_q offset:0 atIndex:0];
                    [cenc_ap setBuffer:in_k offset:0 atIndex:1];
                    [cenc_ap setBuffer:in_v offset:0 atIndex:2];
                    [cenc_ap setBuffer:eng->b_in_i8[0] offset:0 atIndex:3];
                    [cenc_ap setBuffer:eng->b_in_i8[1] offset:0 atIndex:4];
                    [cenc_ap setBuffer:eng->b_in_i8[2] offset:0 atIndex:5];
                    [cenc_ap setBuffer:eng->b_in_xsc[0] offset:0 atIndex:6];
                    [cenc_ap setBuffer:eng->b_in_xsc[1] offset:0 atIndex:7];
                    [cenc_ap setBuffer:eng->b_in_xsc[2] offset:0 atIndex:8];
                    [cenc_ap setBuffer:eng->b_in_xsum_f16[0] offset:0 atIndex:9];
                    [cenc_ap setBuffer:eng->b_in_xsum_f16[1] offset:0 atIndex:10];
                    [cenc_ap setBuffer:eng->b_in_xsum_f16[2] offset:0 atIndex:11];
                    [cenc_ap setBytes:&qp_in length:sizeof(qp_in) atIndex:12];
                    [cenc_ap dispatchThreadgroups:MTLSizeMake((M_padded + 3U) / 4U, 3, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [cenc_ap memoryBarrierWithScope:MTLBarrierScopeBuffers];
                id<MTLBuffer> i8_q  = eng->b_in_i8[0];
                id<MTLBuffer> i8_k  = same_qkv ? eng->b_in_i8[0] : eng->b_in_i8[1];
                id<MTLBuffer> i8_v  = same_qkv ? eng->b_in_i8[0] : eng->b_in_i8[2];
                id<MTLBuffer> sc_q  = eng->b_in_xsc[0];
                id<MTLBuffer> sc_k  = same_qkv ? eng->b_in_xsc[0] : eng->b_in_xsc[1];
                id<MTLBuffer> sc_v  = same_qkv ? eng->b_in_xsc[0] : eng->b_in_xsc[2];
                id<MTLBuffer> xs_q16 = eng->b_in_xsum_f16[0];
                id<MTLBuffer> xs_k16 = same_qkv ? eng->b_in_xsum_f16[0] : eng->b_in_xsum_f16[1];
                id<MTLBuffer> xs_v16 = same_qkv ? eng->b_in_xsum_f16[0] : eng->b_in_xsum_f16[2];

                sik_encode_dense_linear_i8(cenc_ap, eng, ld.wq, in_q, i8_q, sc_q, xs_q16, nil, 0, eng->b_q_raw, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear_i8(cenc_ap, eng, ld.gate_attn, in_q, i8_q, sc_q, xs_q16, nil, 0, eng->b_gate_f32, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear_i8(cenc_ap, eng, ld.wk, in_k, i8_k, sc_k, xs_k16, nil, 0, eng->b_k_raw, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear_i8(cenc_ap, eng, ld.wv, in_v, i8_v, sc_v, xs_v16, nil, 0, eng->b_v_raw, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear_i8(cenc_ap, eng, ld.wr, in_q, i8_q, sc_q, xs_q16, nil, 0, eng->b_r_raw, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.attn_sink_proj, in_q, 0, nil, 0, eng->b_sink_f32, 0, M_valid, M_padded, kOutF32, force_gemv);
                if (!is_local) {
                    sik_encode_dense_linear(cenc_ap, eng, ld.k_sconv_proj, in_k, 0, nil, 0, eng->b_k_dyn, 0, M_valid, M_padded, kOutF32, force_gemv);
                    sik_encode_dense_linear(cenc_ap, eng, ld.v_sconv_proj, in_v, 0, nil, 0, eng->b_v_dyn, 0, M_valid, M_padded, kOutF32, force_gemv);
                }
            } else {
                sik_encode_dense_linear(cenc_ap, eng, ld.wq, in_q, 0, nil, 0, eng->b_q_raw, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.gate_attn, in_q, 0, nil, 0, eng->b_gate_f32, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.wk, in_k, 0, nil, 0, eng->b_k_raw, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.wv, in_v, 0, nil, 0, eng->b_v_raw, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.wr, in_q, 0, nil, 0, eng->b_r_raw, 0, M_valid, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.attn_sink_proj, in_q, 0, nil, 0, eng->b_sink_f32, 0, M_valid, M_padded, kOutF32, force_gemv);
                if (!is_local) {
                    sik_encode_dense_linear(cenc_ap, eng, ld.k_sconv_proj, in_k, 0, nil, 0, eng->b_k_dyn, 0, M_valid, M_padded, kOutF32, force_gemv);
                    sik_encode_dense_linear(cenc_ap, eng, ld.v_sconv_proj, in_v, 0, nil, 0, eng->b_v_dyn, 0, M_valid, M_padded, kOutF32, force_gemv);
                }
            }
            if (__builtin_expect(eng->prof_active, 0)) {
                [cenc_ap endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double gs = [cb GPUStartTime], ge = [cb GPUEndTime];
                if (ge > gs) eng->prof_ms[2U] += (ge - gs) * 1000.0;
                cb  = [eng->queue commandBuffer];
                enc = [cb computeCommandEncoder];
            } else {
                [cenc_ap endEncoding];
                enc = [cb computeCommandEncoder];
            }
        }

        // 2h-2i. K/V Short Conv + Q/K/V/R L2-Clip, V-RMSNorm, Tau & FP16/Q8_0 KV Cache Store
        if (!is_prefill_nax && M_valid <= kMaxSpecBatch) sik_prof_mark(eng, cb, enc, 2U);
        const uint64_t visit_sconv_base = static_cast<uint64_t>(il) * eng->sconv_visit_stride;
        const uint64_t k_sconv_off      = visit_sconv_base;
        const uint64_t v_sconv_off      = k_sconv_off + eng->sconv_k_bytes;
        const uint64_t attn_sconv_off   = v_sconv_off + eng->sconv_v_bytes;
        const uint64_t mlp_sconv_off    = attn_sconv_off + eng->sconv_attn_bytes;
        const uint32_t v_state_off_f4   = static_cast<uint32_t>(eng->sconv_k_bytes / 16ULL);

        id<MTLBuffer>    k_cache_buf = is_local ? eng->b_swa_k_cache : eng->b_glob_k_cache;
        id<MTLBuffer>    v_cache_buf = is_local ? eng->b_swa_v_cache : eng->b_glob_v_cache;
        const uint32_t   kv_stride   = is_local ? kSwaRingLen : eng->max_seq_alloc;
        const NSUInteger kv_lyr_off  = static_cast<NSUInteger>(
            static_cast<uint64_t>(kv_slot) * (is_local ? eng->swa_kv_layer_stride : eng->glob_kv_layer_stride));
        const bool       use_q8_kv   = (!is_local && eng->kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0);
        const NSUInteger kv_sc_off   = use_q8_kv ? static_cast<NSUInteger>(kv_lyr_off + eng->glob_kv_q8_data_bytes) : kv_lyr_off;
        const float      attn_scale  = 1.0f / std::sqrt(static_cast<float>(hd));

        const uint32_t snap_stride_f4 = static_cast<uint32_t>(eng->sconv_total_bytes / 16ULL);
        const bool can_fuse_sconv_m1 =
            (!is_prefill_nax && M_valid == 1U && !use_q8_kv && save_snap_mask == 0U &&
             n_kv == 8U && n_q == 24U && hd == 128U &&
             (!is_local || cfg.sliding_window_size <= kv_stride));
        const bool use_fused_sconv_single_m1 = (can_fuse_sconv_m1 && seq_pos_start < 256U);
        const bool use_fused_sconv_split_m1  = (can_fuse_sconv_m1 && seq_pos_start >= 256U);
        const bool use_fused_sconv_attn_m1   = (use_fused_sconv_single_m1 || use_fused_sconv_split_m1);

        if (use_fused_sconv_single_m1) {
            // FUSED 1-DISPATCH `SConvKV + QKVR-Prep + Single-Pass GQA-3 Attention + Sink + Gate + wo_xsum` (`pos < 256`)!
            SikSconvAttnM1ParamsHost sap{
                n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                seq_pos_start, kv_stride, is_local ? 1U : 0U, v_state_off_f4,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps,
                cfg.log_scaling_alpha, static_cast<float>(cfg.log_scaling_n_floor), attn_scale,
                0U, 0U, kMaxDecodeSplits, 0U
            };
            [enc setComputePipelineState:eng->pso_sconv_attn_single_gqa3_m1];
            [enc setBuffer:eng->b_q_raw offset:0 atIndex:0];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:1];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:2];
            [enc setBuffer:eng->b_r_raw offset:0 atIndex:3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.k_sconv_w.offset) atIndex:4];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.v_sconv_w.offset) atIndex:5];
            [enc setBuffer:eng->b_k_dyn offset:0 atIndex:6];
            [enc setBuffer:eng->b_v_dyn offset:0 atIndex:7];
            [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(k_sconv_off) atIndex:8];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.q_l2_clip_scale.offset) atIndex:9];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_l2_clip_scale.offset) atIndex:10];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:11];
            [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:12];
            [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:13];
            [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:14];
            [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:15];
            [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:16];
            [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:17];
            [enc setBytes:&sap length:sizeof(sap) atIndex:18];
            [enc dispatchThreadgroups:MTLSizeMake(n_kv, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            sik_prof_mark(eng, cb, enc, 4U);
        } else if (use_fused_sconv_split_m1) {
            // FUSED `SConvKV + QKVR-Prep + Split-KV GQA-3 Attention` (`pos >= 256` -> ALL contexts up to 131K!)
            const uint32_t B_cur    = sik_host_attn_chunk_size(seq_pos_start, is_local ? 1U : 0U);
            const uint32_t ws_first = (is_local && seq_pos_start + 1U > cfg.sliding_window_size)
                ? (seq_pos_start + 1U - cfg.sliding_window_size) : 0U;
            const uint32_t c_first  = ws_first / B_cur;
            const uint32_t c_last   = seq_pos_start / B_cur;
            const uint32_t n_chunks = c_last - c_first + 1U;
            SikSconvAttnM1ParamsHost sap{
                n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                seq_pos_start, kv_stride, is_local ? 1U : 0U, v_state_off_f4,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps,
                cfg.log_scaling_alpha, static_cast<float>(cfg.log_scaling_n_floor), attn_scale,
                B_cur, c_first, kMaxDecodeSplits, 0U
            };
            [enc setComputePipelineState:eng->pso_sconv_attn_split_gqa3_m1];
            [enc setBuffer:eng->b_q_raw offset:0 atIndex:0];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:1];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:2];
            [enc setBuffer:eng->b_r_raw offset:0 atIndex:3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.k_sconv_w.offset) atIndex:4];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.v_sconv_w.offset) atIndex:5];
            [enc setBuffer:eng->b_k_dyn offset:0 atIndex:6];
            [enc setBuffer:eng->b_v_dyn offset:0 atIndex:7];
            [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(k_sconv_off) atIndex:8];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.q_l2_clip_scale.offset) atIndex:9];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_l2_clip_scale.offset) atIndex:10];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:11];
            [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:12];
            [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:13];
            [enc setBuffer:eng->b_part_m offset:0 atIndex:14];
            [enc setBuffer:eng->b_part_l offset:0 atIndex:15];
            [enc setBuffer:eng->b_part_o offset:0 atIndex:16];
            [enc setBytes:&sap length:sizeof(sap) atIndex:17];
            [enc dispatchThreadgroups:MTLSizeMake(n_chunks, n_kv, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            SikAttnParamsHost rp{
                1U, M_pad16, n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                is_local ? 1U : 0U, seq_pos_start, 0U, kv_stride, kMaxDecodeSplits,
                32U, 0U, attn_scale
            };
            [enc setComputePipelineState:eng->pso_attn_decode_reduce];
            [enc setBuffer:eng->b_part_m offset:0 atIndex:0];
            [enc setBuffer:eng->b_part_l offset:0 atIndex:1];
            [enc setBuffer:eng->b_part_o offset:0 atIndex:2];
            [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:3];
            [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:4];
            [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:5];
            [enc setBytes:&rp length:sizeof(rp) atIndex:6];
            [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(n_q, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            sik_prof_mark(eng, cb, enc, 4U);
        } else if (!is_prefill_nax) {
            // FUSED 1-DISPATCH `sconv_kv + qkvr_prep` (linear or tree-indexed!)
            SikAttnPrepParamsHost pp{
                M_valid, M_pad16, n_q, n_kv, hd, cfg.d_rel, seq_pos_start, kv_stride,
                is_local ? 1U : 0U, save_snap_mask, v_state_off_f4, 0U,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps,
                cfg.log_scaling_alpha, static_cast<float>(cfg.log_scaling_n_floor),
                attn_scale, use_q8_kv ? STRATA_INKLING2_KV_DTYPE_Q8_0 : STRATA_INKLING2_KV_DTYPE_F16, snap_stride_f4
            };
            [enc setComputePipelineState:(tree_active ? eng->pso_sconv_kv_qkvr_prep_tree : eng->pso_sconv_kv_qkvr_prep)];
            [enc setBuffer:eng->b_q_raw offset:0 atIndex:0];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:1];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:2];
            [enc setBuffer:eng->b_r_raw offset:0 atIndex:3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.k_sconv_w.offset) atIndex:4];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.v_sconv_w.offset) atIndex:5];
            [enc setBuffer:eng->b_k_dyn offset:0 atIndex:6];
            [enc setBuffer:eng->b_v_dyn offset:0 atIndex:7];
            [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(k_sconv_off) atIndex:8];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.q_l2_clip_scale.offset) atIndex:9];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_l2_clip_scale.offset) atIndex:10];
            [enc setBuffer:eng->b_q_f32 offset:0 atIndex:11];
            [enc setBuffer:eng->b_r_f32 offset:0 atIndex:12];
            [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:13];
            [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:14];
            [enc setBuffer:(eng->b_sconv_snaps ? eng->b_sconv_snaps : eng->b_sconv_state)
                    offset:static_cast<NSUInteger>(k_sconv_off) atIndex:15];
            [enc setBytes:&pp length:sizeof(pp) atIndex:16];
            [enc setBuffer:k_cache_buf offset:kv_sc_off atIndex:17];
            [enc setBuffer:v_cache_buf offset:kv_sc_off atIndex:18];
            if (tree_active) {
                [enc setBuffer:eng->b_tree_meta offset:0 atIndex:19];
            }
            [enc dispatchThreadgroups:MTLSizeMake(n_q * M_valid + n_kv + n_kv + n_q * M_valid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            // Prefill: 2D Parallel K+V SConv -> tiny tail save -> Head-Major Q/R + zero-tailed KV prep!
            SikSconvParamsHost sp{M_valid, kv_dim, 0U, 0U, hd, n_kv, v_state_off_f4, top_k};
            if (is_local) {
                [enc setComputePipelineState:eng->pso_sconv_static_kv_prefill];
                [enc setBuffer:eng->b_k_raw offset:0 atIndex:0];
                [enc setBuffer:eng->b_v_raw offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.k_sconv_w.offset) atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.v_sconv_w.offset) atIndex:3];
                [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(k_sconv_off) atIndex:4];
                [enc setBuffer:eng->b_k_conv offset:0 atIndex:5];
                [enc setBuffer:eng->b_v_conv offset:0 atIndex:6];
                [enc setBytes:&sp length:sizeof(sp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 255U) / 256U, (M_valid + 15U) / 16U, 2)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [enc setComputePipelineState:eng->pso_sconv_dynamic_kv_prefill];
                [enc setBuffer:eng->b_k_raw offset:0 atIndex:0];
                [enc setBuffer:eng->b_v_raw offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.k_sconv_w.offset) atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.v_sconv_w.offset) atIndex:3];
                [enc setBuffer:eng->b_k_dyn offset:0 atIndex:4];
                [enc setBuffer:eng->b_v_dyn offset:0 atIndex:5];
                [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(k_sconv_off) atIndex:6];
                [enc setBuffer:eng->b_k_conv offset:0 atIndex:7];
                [enc setBuffer:eng->b_v_conv offset:0 atIndex:8];
                [enc setBytes:&sp length:sizeof(sp) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 255U) / 256U, (M_valid + 15U) / 16U, 2)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc setComputePipelineState:eng->pso_sconv_save_tail_kv];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:0];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:1];
            [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(k_sconv_off) atIndex:2];
            [enc setBytes:&sp length:sizeof(sp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 255U) / 256U, 2, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            const uint32_t pos_end      = seq_pos_start + M_valid;
            const uint32_t kv_zero_tail = (64U - (pos_end & 63U)) & 63U;
            const uint32_t grid_y       = std::max<uint32_t>(M_pad16, M_valid + kv_zero_tail);
            SikAttnPrepParamsHost pp{
                M_valid, M_pad16, n_q, n_kv, hd, cfg.d_rel, seq_pos_start, kv_stride,
                is_local ? 1U : 0U, 0U, v_state_off_f4, kv_zero_tail,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps,
                cfg.log_scaling_alpha, static_cast<float>(cfg.log_scaling_n_floor),
                attn_scale, use_q8_kv ? STRATA_INKLING2_KV_DTYPE_Q8_0 : STRATA_INKLING2_KV_DTYPE_F16, 0U
            };
            [enc setComputePipelineState:eng->pso_attn_qkvr_prep_prefill];
            [enc setBuffer:eng->b_q_raw offset:0 atIndex:0];
            [enc setBuffer:eng->b_k_conv offset:0 atIndex:1];
            [enc setBuffer:eng->b_v_conv offset:0 atIndex:2];
            [enc setBuffer:eng->b_r_raw offset:0 atIndex:3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.q_l2_clip_scale.offset) atIndex:4];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_l2_clip_scale.offset) atIndex:5];
            [enc setBuffer:eng->b_q_hm_f16 offset:0 atIndex:6];
            [enc setBuffer:eng->b_r_hm_f32 offset:0 atIndex:7];
            [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:8];
            [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:9];
            [enc setBytes:&pp length:sizeof(pp) atIndex:10];
            [enc setBuffer:k_cache_buf offset:kv_sc_off atIndex:11];
            [enc setBuffer:v_cache_buf offset:kv_sc_off atIndex:12];
            [enc dispatchThreadgroups:MTLSizeMake(n_q + n_kv + n_kv + n_q, grid_y, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        if (!use_fused_sconv_attn_m1) {
        sik_prof_mark(eng, cb, enc, 3U);

        // 2j. Relative-Bias + Virtual-Sink FlashAttention + Sigmoid Output Gate -> `b_attn_gated_f16`
        if (!is_prefill_nax && tree_active) {
            // Tree-Aware Direct-SIMT GQA-3 Attention (`100.0000%` BIT-EXACT w/ sequential M=1 along every branch!)
            const uint32_t q_max = seq_pos_start + M_valid - 1U;
            uint32_t q_cur       = seq_pos_start;
            bool     need_reduce = false;
            while (q_cur <= q_max) {
                const uint32_t B_cur      = sik_host_attn_chunk_size(q_cur, is_local ? 1U : 0U);
                const bool     use_single = (!use_q8_kv && q_cur < 256U);
                uint32_t q_end = q_cur + 1U;
                while (q_end <= q_max) {
                    if (sik_host_attn_chunk_size(q_end, is_local ? 1U : 0U) != B_cur) break;
                    if ((!use_q8_kv && q_end < 256U) != use_single) break;
                    ++q_end;
                }
                const uint32_t q_first  = q_cur;
                const uint32_t q_last   = q_end - 1U;
                const uint32_t ws_first = (is_local && q_first + 1U > cfg.sliding_window_size)
                    ? (q_first + 1U - cfg.sliding_window_size) : 0U;
                const uint32_t c_first  = ws_first / B_cur;
                const uint32_t c_last   = q_last / B_cur;
                const uint32_t n_chunks = c_last - c_first + 1U;

                SikAttnParamsHost ap{
                    M_valid, M_pad16, n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                    is_local ? 1U : 0U, seq_pos_start, 0U, kv_stride, kMaxDecodeSplits,
                    B_cur, c_first, attn_scale
                };
                if (use_single) {
                    [enc setComputePipelineState:eng->pso_attn_decode_single_gqa3_sg8_tree];
                    [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                    [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                    [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:3];
                    [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:4];
                    [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:5];
                    [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:6];
                    [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:7];
                    [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:8];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                    [enc setBuffer:eng->b_tree_meta offset:0 atIndex:10];
                    [enc dispatchThreadgroups:MTLSizeMake(n_kv, M_valid, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    need_reduce = true;
                    if (use_q8_kv) {
                        [enc setComputePipelineState:eng->pso_attn_decode_q8_reg_tree];
                        [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                        [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                        [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:3];
                        [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:4];
                        [enc setBuffer:eng->b_part_m offset:0 atIndex:5];
                        [enc setBuffer:eng->b_part_l offset:0 atIndex:6];
                        [enc setBuffer:eng->b_part_o offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc setBuffer:k_cache_buf offset:kv_sc_off atIndex:9];
                        [enc setBuffer:v_cache_buf offset:kv_sc_off atIndex:10];
                        [enc setBuffer:eng->b_tree_meta offset:0 atIndex:11];
                        [enc dispatchThreadgroups:MTLSizeMake(n_chunks, n_kv, M_valid)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    } else {
                        [enc setComputePipelineState:eng->pso_attn_decode_split_gqa3_reg_tree];
                        [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                        [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                        [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:3];
                        [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:4];
                        [enc setBuffer:eng->b_part_m offset:0 atIndex:5];
                        [enc setBuffer:eng->b_part_l offset:0 atIndex:6];
                        [enc setBuffer:eng->b_part_o offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc setBuffer:eng->b_tree_meta offset:0 atIndex:9];
                        [enc dispatchThreadgroups:MTLSizeMake(n_chunks, n_kv, M_valid)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                }
                q_cur = q_end;
            }
            if (need_reduce) {
                SikAttnParamsHost rp{
                    M_valid, M_pad16, n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                    is_local ? 1U : 0U, seq_pos_start, 0U, kv_stride, kMaxDecodeSplits,
                    32U, use_q8_kv ? 0U : 1U, attn_scale
                };
                [enc setComputePipelineState:eng->pso_attn_decode_reduce_tree];
                [enc setBuffer:eng->b_part_m offset:0 atIndex:0];
                [enc setBuffer:eng->b_part_l offset:0 atIndex:1];
                [enc setBuffer:eng->b_part_o offset:0 atIndex:2];
                [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:3];
                [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:4];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:5];
                [enc setBytes:&rp length:sizeof(rp) atIndex:6];
                [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:7];
                [enc setBuffer:eng->b_tree_meta offset:0 atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(n_q, M_valid, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
        } else if (!is_prefill_nax) {
            // Direct-SIMT GQA-3 Decode Attention:
            //   - F16 KV, `q_pos < 160`  -> 0-Split Single-Pass `sik_attn_decode_single_gqa3_sg8` (fuses sink+gate+xsum, skips reduce!)
            //   - F16 KV, `q_pos >= 160` -> Register-Direct Split-KV `sik_attn_decode_split_gqa3_reg` (0 TG KV staging, 0 KV barriers!)
            //   - Q8_0 KV                -> `sik_attn_decode_q8_reg`
            // 100.0000% bit-exact `K>0 == K=0` across every bucket & boundary (including straddling `pos=160`!)
            uint32_t m_cur        = 0U;
            bool     need_reduce  = false;
            uint32_t red_m_start  = M_valid;
            uint32_t red_m_end    = 0U;
            while (m_cur < M_valid) {
                const uint32_t qp0        = seq_pos_start + m_cur;
                const uint32_t B_cur      = sik_host_attn_chunk_size(qp0, is_local ? 1U : 0U);
                const bool     use_single = (!use_q8_kv && qp0 < 256U);
                uint32_t m_end = m_cur + 1U;
                while (m_end < M_valid && (m_end - m_cur) < kMaxSpecBatch) {
                    const uint32_t qpe = seq_pos_start + m_end;
                    if (sik_host_attn_chunk_size(qpe, is_local ? 1U : 0U) != B_cur) break;
                    if ((!use_q8_kv && qpe < 256U) != use_single) break;
                    ++m_end;
                }
                const uint32_t mb       = m_end - m_cur;
                const uint32_t q_first  = seq_pos_start + m_cur;
                const uint32_t q_last   = seq_pos_start + m_end - 1U;
                const uint32_t ws_first = (is_local && q_first + 1U > cfg.sliding_window_size)
                    ? (q_first + 1U - cfg.sliding_window_size) : 0U;
                const uint32_t c_first  = ws_first / B_cur;
                const uint32_t c_last   = q_last / B_cur;
                const uint32_t n_chunks = c_last - c_first + 1U;

                SikAttnParamsHost ap{
                    mb, M_pad16, n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                    is_local ? 1U : 0U, q_first, m_cur, kv_stride, kMaxDecodeSplits,
                    B_cur, c_first, attn_scale
                };
                if (use_single) {
                    [enc setComputePipelineState:eng->pso_attn_decode_single_gqa3_sg8];
                    [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                    [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                    [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:3];
                    [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:4];
                    [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:5];
                    [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:6];
                    [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:7];
                    [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:8];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake(n_kv, mb, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    need_reduce = true;
                    if (m_cur < red_m_start) red_m_start = m_cur;
                    if (m_end > red_m_end)   red_m_end   = m_end;
                    const uint32_t mz = (mb == 1U) ? 1U : ((mb + 1U) / 2U);
                    if (use_q8_kv) {
                        [enc setComputePipelineState:eng->pso_attn_decode_q8_reg_m[(mb == 1U) ? 0U : 1U]];
                        [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                        [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                        [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:3];
                        [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:4];
                        [enc setBuffer:eng->b_part_m offset:0 atIndex:5];
                        [enc setBuffer:eng->b_part_l offset:0 atIndex:6];
                        [enc setBuffer:eng->b_part_o offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc setBuffer:k_cache_buf offset:kv_sc_off atIndex:9];
                        [enc setBuffer:v_cache_buf offset:kv_sc_off atIndex:10];
                        [enc dispatchThreadgroups:MTLSizeMake(n_chunks, n_kv, mz)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    } else {
                        [enc setComputePipelineState:eng->pso_attn_decode_split_gqa3_reg];
                        [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                        [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                        [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:3];
                        [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:4];
                        [enc setBuffer:eng->b_part_m offset:0 atIndex:5];
                        [enc setBuffer:eng->b_part_l offset:0 atIndex:6];
                        [enc setBuffer:eng->b_part_o offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(n_chunks, n_kv, mb)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                }
                m_cur = m_end;
            }

            if (need_reduce) {
                const uint32_t red_mb = red_m_end - red_m_start;
                SikAttnParamsHost rp{
                    red_mb, M_pad16, n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                    is_local ? 1U : 0U, seq_pos_start + red_m_start, red_m_start, kv_stride, kMaxDecodeSplits,
                    32U, 0U, attn_scale
                };
                [enc setComputePipelineState:eng->pso_attn_decode_reduce];
                [enc setBuffer:eng->b_part_m offset:0 atIndex:0];
                [enc setBuffer:eng->b_part_l offset:0 atIndex:1];
                [enc setBuffer:eng->b_part_o offset:0 atIndex:2];
                [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:3];
                [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:4];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:5];
                [enc setBytes:&rp length:sizeof(rp) atIndex:6];
                [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(n_q, red_mb, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
        } else {
            // Metal 4.1 NAX FlashAttention (`3.3-5.8x` faster than scalar; supports FP16 & Q8_0 global KV!)
            SikAttnParamsHost ap{
                M_valid, M_pad16, n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                is_local ? 1U : 0U, seq_pos_start, 0U, kv_stride, kMaxDecodeSplits,
                32U, 0U, attn_scale
            };
            [enc setComputePipelineState:(use_q8_kv ? eng->pso_attn_prefill_nax_q8 : eng->pso_attn_prefill_nax)];
            [enc setBuffer:eng->b_q_hm_f16 offset:0 atIndex:0];
            [enc setBuffer:eng->b_r_hm_f32 offset:0 atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
            [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:3];
            [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:4];
            [enc setBuffer:k_cache_buf offset:kv_lyr_off atIndex:5];
            [enc setBuffer:v_cache_buf offset:kv_lyr_off atIndex:6];
            [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:7];
            [enc setBytes:&ap length:sizeof(ap) atIndex:8];
            if (use_q8_kv) {
                [enc setBuffer:k_cache_buf offset:kv_sc_off atIndex:9];
                [enc setBuffer:v_cache_buf offset:kv_sc_off atIndex:10];
                [enc dispatchThreadgroups:MTLSizeMake((M_valid + 63U) / 64U, n_q, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc dispatchThreadgroups:MTLSizeMake((M_valid + 15U) / 16U, n_kv, 1)
                    threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];
            }
        }
        sik_prof_mark(eng, cb, enc, 4U);
        }

        // 2k. Output Projection (`wo`) + `attn_sconv` + Post-Attention Block Boundary!
        const bool is_block_boundary = ((ip % cfg.attn_res_block_size) == 0U);
        const bool use_alt_wo = (eng->alt_wo_active && eng->b_alt_wo != nil && eng->alt_wo_has[ip]);
        id<MTLBuffer> wo_buf = use_alt_wo ? eng->b_alt_wo : eng->b_uma;
        const StrataInkling2TensorSlice& wo_sl = use_alt_wo ? eng->alt_wo_slice[ip] : ld.wo;
        if (!is_prefill_nax && wo_sl.dtype == STRATA_INKLING2_DTYPE_Q4_A && M_valid <= kMaxSpecBatch) {
            uint64_t ww = 0ULL, ws = 0ULL, wb = 0ULL;
            sik_slice_sub_offsets(wo_sl, ww, ws, wb);
            const uint32_t   res_mode = is_block_boundary ? 2U : 1U;
            const NSUInteger blk_off  = is_block_boundary
                ? static_cast<NSUInteger>(num_blocks * eng->block_slot_bytes) : 0U;
            id<MTLBuffer>    blk_buf  = is_block_boundary ? eng->b_blocks : eng->b_cur;
            SikWoSconvParamsHost wp{M_valid, wo_sl.shape0, wo_sl.shape1, save_snap_mask, res_mode, snap_stride_f4, 0U, 0U};
            if (M_valid == 1U && save_snap_mask == 0U) {
                [enc setComputePipelineState:eng->pso_q4a_wo_sconv_m1];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ww) atIndex:0];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ws) atIndex:1];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(wb) atIndex:2];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:3];
                [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_sconv_w.offset) atIndex:5];
                [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(attn_sconv_off) atIndex:6];
                [enc setBuffer:eng->b_cur offset:0 atIndex:7];
                [enc setBuffer:blk_buf offset:blk_off atIndex:8];
                [enc setBytes:&wp length:sizeof(wp) atIndex:9];
            } else if (tree_active) {
                [enc setComputePipelineState:eng->pso_q4a_wo_sconv_tree_m[M_valid - 1U]];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ww) atIndex:0];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ws) atIndex:1];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(wb) atIndex:2];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:3];
                [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_sconv_w.offset) atIndex:5];
                [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(attn_sconv_off) atIndex:6];
                [enc setBuffer:eng->b_cur offset:0 atIndex:7];
                [enc setBuffer:blk_buf offset:blk_off atIndex:8];
                [enc setBuffer:(eng->b_sconv_snaps ? eng->b_sconv_snaps : eng->b_sconv_state)
                        offset:static_cast<NSUInteger>(attn_sconv_off) atIndex:9];
                [enc setBytes:&wp length:sizeof(wp) atIndex:10];
                [enc setBuffer:eng->b_tree_meta offset:0 atIndex:11];
            } else {
                [enc setComputePipelineState:eng->pso_q4a_wo_sconv_m[M_valid - 1U]];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ww) atIndex:0];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ws) atIndex:1];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(wb) atIndex:2];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:3];
                [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_sconv_w.offset) atIndex:5];
                [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(attn_sconv_off) atIndex:6];
                [enc setBuffer:eng->b_cur offset:0 atIndex:7];
                [enc setBuffer:blk_buf offset:blk_off atIndex:8];
                [enc setBuffer:(eng->b_sconv_snaps ? eng->b_sconv_snaps : eng->b_sconv_state)
                        offset:static_cast<NSUInteger>(attn_sconv_off) atIndex:9];
                [enc setBytes:&wp length:sizeof(wp) atIndex:10];
            }
            [enc dispatchThreadgroups:MTLSizeMake((wo_sl.shape0 + 7U) / 8U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            if (is_block_boundary && num_blocks < kMaxAttnResBlocks) ++num_blocks;
        } else {
            if (use_int8_prefill && (wo_sl.dtype == STRATA_INKLING2_DTYPE_Q4_A || wo_sl.dtype == STRATA_INKLING2_DTYPE_Q8_0)) {
                SikQuantI8ParamsHost qp_wo{M_valid, M_padded, wo_sl.shape1, 0U, 1U, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_quant_f16_to_i8_row];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_in_i8[0] offset:0 atIndex:1];
                [enc setBuffer:eng->b_in_xsc[0] offset:0 atIndex:2];
                [enc setBuffer:eng->b_in_xsum_f16[0] offset:0 atIndex:3];
                [enc setBytes:&qp_wo length:sizeof(qp_wo) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((M_padded + 3U) / 4U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                sik_encode_dense_linear_i8(enc, eng, wo_sl, eng->b_attn_gated_f16,
                                           eng->b_in_i8[0], eng->b_in_xsc[0], eng->b_in_xsum_f16[0],
                                           nil, 0, eng->b_wo_out_f32, 0, M_valid, M_padded, kOutF32, force_gemv,
                                           0.0f, wo_buf);
            } else {
                sik_encode_dense_linear(enc, eng, wo_sl, eng->b_attn_gated_f16, 0,
                                        nil, 0, eng->b_wo_out_f32, 0, M_valid, M_padded, kOutF32, force_gemv,
                                        0.0f, wo_buf);
            }

            if (is_block_boundary) {
                const NSUInteger blk_off = static_cast<NSUInteger>(num_blocks * eng->block_slot_bytes);
                sik_encode_sconv_static(enc, eng, eng->b_wo_out_f32, ld.attn_sconv_w.offset, attn_sconv_off,
                                        eng->b_cur, eng->b_blocks, blk_off, M_valid, D, save_snap_mask, 2U, tree_active);
                if (num_blocks < kMaxAttnResBlocks) ++num_blocks;
            } else {
                sik_encode_sconv_static(enc, eng, eng->b_wo_out_f32, ld.attn_sconv_w.offset, attn_sconv_off,
                                        eng->b_cur, nil, 0, M_valid, D, save_snap_mask, 1U, tree_active);
            }
        }
        sik_prof_mark(eng, cb, enc, 5U);

        // 2l. Pre-MoE Block Attention Residual + Optional Loop Control `3` (`mlp`) + `mlp_norm`
        const bool is_q8_mlp_ctrl = (has_msgs && ld.controllers[3].down_weight.dtype == STRATA_INKLING2_DTYPE_Q8_0);
        const bool fuse_moe_rms   = (M_valid == 1U && is_q8_mlp_ctrl && !is_prefill_nax);
        const bool fast_moe_rms   = (M_valid > 1U && M_valid <= kMaxSpecBatch && is_q8_mlp_ctrl && !is_prefill_nax);
        if (!has_msgs) {
            sik_encode_attn_res_rmsnorm(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                        ld.mlp_res_query.offset, ld.mlp_norm.offset,
                                        eng->b_in_f16[0], eng->b_x_f32, true, M_valid, M_padded);
        } else {
            const bool fused_mlp_prep  = (!is_prefill_nax && M_valid <= kMaxSpecBatch);
            const bool batch12_mlp     = (fused_mlp_prep && use_batch12_loop);
            if (batch12_mlp) {
                // 576-Thread 1-Pass Block AttnRes + Left-Half RMSNorm (msg3 & Down1_right already precomputed!)
                SikAttnResParamsHost ap{
                    M_valid, D, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    1U, cfg.rms_norm_eps, 1U, 0U
                };
                [enc setComputePipelineState:eng->pso_attn_res_prep_left_576];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_res_query.offset) atIndex:2];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:3];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:4];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:5];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:6];
                [enc setBytes:&ap length:sizeof(ap) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(576, 1, 1)];
                sik_prof_mark_detail(eng, cb, enc, 19U);
            } else if (fused_mlp_prep) {
                SikAttnResParamsHost ap{
                    M_valid, D, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    1U, cfg.rms_norm_eps, 0U, 0U
                };
                [enc setComputePipelineState:eng->pso_attn_res_loop_prep1];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_res_query.offset) atIndex:2];
                [enc setBuffer:eng->b_messages[3] offset:0 atIndex:3];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:4];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:5];
                [enc setBytes:&ap length:sizeof(ap) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                sik_prof_mark_detail(eng, cb, enc, 19U);
            } else {
                sik_encode_attn_res(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                    ld.mlp_res_query.offset, 1U, eng->b_inputs[0], nil, nil, M_valid);
            }
            if (batch12_mlp) {
                const uint32_t li = ip - core_beg;
                const NSUInteger mslot = static_cast<NSUInteger>(M_valid) * D * sizeof(float);
                const NSUInteger hslot = static_cast<NSUInteger>(M_valid) * cfg.loop_gate_rank * sizeof(float);
                sik_encode_loop_ctrl_batch(
                    cb, enc, eng, &ld.controllers[3], 1U,
                    eng->b_inputs[0], nil, nil, eng->b_msg_bank12, nil, nil,
                    eng->b_inputs[0], nil, nil, M_valid, M_padded, force_gemv, /*skip_prep=*/true,
                    static_cast<NSUInteger>(li * 4U + 3U) * mslot, 0U, 0U,
                    eng->b_ctrl_h_msg12, static_cast<NSUInteger>(li * 4U + 3U) * hslot
                );
            } else {
                sik_encode_loop_ctrl_batch(
                    cb, enc, eng, &ld.controllers[3], 1U,
                    eng->b_inputs[0], nil, nil, eng->b_messages[3], nil, nil,
                    eng->b_inputs[0], nil, nil, M_valid, M_padded, force_gemv, /*skip_prep=*/fused_mlp_prep
                );
            }
            if (fast_moe_rms) {
                // Zero-Barrier 1-SG 1-Pass RMSNorm1 consuming `b_rms_sq` from `Apply1` (-0.65 ms @ M=8 vs fused router!)
                SikRmsNormParamsHost np{M_valid, M_padded, D, cfg.rms_norm_eps, (D + 31U) / 32U, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_rmsnorm1_fast];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_norm.offset) atIndex:1];
                [enc setBuffer:eng->b_rms_sq offset:0 atIndex:2];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_f32 offset:0 atIndex:4];
                [enc setBytes:&np length:sizeof(np) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(D / 64U, M_valid, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else if (!fuse_moe_rms) {
                sik_encode_rmsnorm(enc, eng, eng->b_inputs[0], 0, ld.mlp_norm.offset, true,
                                   eng->b_in_f16[0], eng->b_x_f32, true, M_valid, M_padded);
            }
        }
        if (!fuse_moe_rms || !eng->prof_detail) {
            sik_prof_mark(eng, cb, enc, (eng->prof_detail) ? (has_msgs ? 22U : 18U) : 6U);
        }

        // 2m-2o. MoE Router + TopK/Bin + Indirect Experts + Combine+SConv!
        {
            const bool is_nvfp4 = (ld.expert_dtype == STRATA_INKLING2_DTYPE_NVFP4);
            SikMoeRouterParamsHost rp{M_valid, D, E, top_k, cfg.route_scale, inter, 4U, 0U};
            SikMoeExecParamsHost   mp{
                M_valid, top_k, D, inter, E, is_prefill_nax ? 64U : 8U,
                cfg.mlp_gate_softcap, cfg.mlp_up_softcap
            };

            // Helper: Single-Dispatch High-Occupancy Grouped-4 Indirect MoE (`b[1]` 2x occupancy, 0 extra barriers!)
            auto encode_grouped_single = [&](uint32_t b_slot) {
                if (is_nvfp4) {
                    [enc setComputePipelineState:eng->pso_moe_nvfp4_w13_b[b_slot]];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.expert_scale_off) atIndex:2];
                    [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                    [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                    [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                    [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:7];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                    [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                           indirectBufferOffset:0U
                                          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    sik_prof_mark(eng, cb, enc, 8U);

                    [enc setComputePipelineState:eng->pso_moe_nvfp4_w2_b[b_slot]];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.expert_scale_off) atIndex:2];
                    [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                    [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                    [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                    [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:7];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                    [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                           indirectBufferOffset:16U
                                          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    sik_prof_mark(eng, cb, enc, 9U);
                } else {
                    [enc setComputePipelineState:eng->pso_moe_mxfp8_w13_b[b_slot]];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                    [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:2];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                    [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                    [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:5];
                    [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                           indirectBufferOffset:0U
                                          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    sik_prof_mark(eng, cb, enc, 8U);

                    [enc setComputePipelineState:eng->pso_moe_mxfp8_w2_b[b_slot]];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                    [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                    [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                    [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:5];
                    [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                           indirectBufferOffset:16U
                                          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    sik_prof_mark(eng, cb, enc, 9U);
                }
            };

            const bool use_short_grouped_moe =
                is_prefill_nax &&
                (M_valid <= 64U || (D % 128U) != 0U || (inter % 128U) != 0U);

            if (is_prefill_nax) {
                // Pure FP32 Tiled `simdgroup_float8x8` Router (`12.2 -> ~0.8 ms`!)
                [enc setComputePipelineState:eng->pso_moe_router_f32_prefill];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_weight.offset) atIndex:0];
                [enc setBuffer:eng->b_x_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_router_logits offset:0 atIndex:2];
                [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(E / 32U, (M_valid + 15U) / 16U, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_moe_topk_sigmoid];
                [enc setBuffer:eng->b_router_logits offset:0 atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_bias.offset) atIndex:1];
                [enc setBuffer:eng->b_topk_idx offset:0 atIndex:2];
                [enc setBuffer:eng->b_topk_w offset:0 atIndex:3];
                [enc setBytes:&rp length:sizeof(rp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                if (use_short_grouped_moe) {
                    // Short-Prompt High-Occupancy Grouped-4 Fast Path (`M = 9..64`, single indirect dispatch!)
                    mp.tile_cap = 4U;
                    [enc setComputePipelineState:eng->pso_moe_bin_tokens_grouped];
                    [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
                    [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
                    [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:3];
                    [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:4];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    sik_prof_mark(eng, cb, enc, 7U);
                    encode_grouped_single(2U);
                } else if (use_int8_prefill) {
                    // Opt-In `--int8-prefill` 8KiB Whole-Row Zero-Acc Quad-Bin (`BM = 16/32/64/128`) INT8xINT8->INT32 MoE (`84.5 TOPS`!)
                    SikQuantI8ParamsHost qp_mx{M_valid, M_valid, D, 0U, 1U, 0U, 0U, 0U};
                    [enc setComputePipelineState:eng->pso_quant_f16_to_i8_row];
                    [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:0];
                    [enc setBuffer:eng->b_in_i8[0] offset:0 atIndex:1];
                    [enc setBuffer:eng->b_in_xsc[0] offset:0 atIndex:2];
                    [enc setBuffer:eng->b_in_xsc[0] offset:0 atIndex:3];
                    [enc setBytes:&qp_mx length:sizeof(qp_mx) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((M_valid + 3U) / 4U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    mp.tile_cap = 128U;
                    [enc setComputePipelineState:eng->pso_moe_bin_tokens_quad];
                    [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
                    [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
                    [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:3];
                    [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:4];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:eng->pso_moe_gather_x_i8];
                    [enc setBuffer:eng->b_in_i8[0] offset:0 atIndex:0];
                    [enc setBuffer:eng->b_in_xsc[0] offset:0 atIndex:1];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:2];
                    [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:3]; // aliased `x_sorted_i8`
                    [enc setBuffer:eng->b_moe_x_sc offset:0 atIndex:4];
                    [enc setBuffer:eng->b_moe_mid_i8 offset:0 atIndex:5];
                    [enc setBuffer:eng->b_moe_mid_sc offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(M_valid * top_k + 128U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    sik_prof_mark(eng, cb, enc, 7U);

                    [enc endEncoding];
                    id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

                    constexpr NSUInteger kBinStrideBytes = 512U * 16U;
                    auto dispatch_i8_w13 = [&](id<MTLComputePipelineState> p13, NSUInteger bin_idx, NSUInteger th) {
                        const NSUInteger tiles_off = bin_idx * kBinStrideBytes;
                        const NSUInteger ind_off   = bin_idx * 32U;
                        if (is_nvfp4) {
                            [cenc setComputePipelineState:p13];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.expert_scale_off) atIndex:2];
                            [cenc setBuffer:eng->b_moe_down_part offset:0 atIndex:3];
                            [cenc setBuffer:eng->b_moe_x_sc offset:0 atIndex:4];
                            [cenc setBuffer:eng->b_moe_work_tiles offset:tiles_off atIndex:5];
                            [cenc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:6];
                            [cenc setBytes:&mp length:sizeof(mp) atIndex:7];
                            [cenc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                                    indirectBufferOffset:ind_off
                                                   threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
                        } else {
                            [cenc setComputePipelineState:p13];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                            [cenc setBuffer:eng->b_moe_down_part offset:0 atIndex:2];
                            [cenc setBuffer:eng->b_moe_x_sc offset:0 atIndex:3];
                            [cenc setBuffer:eng->b_moe_work_tiles offset:tiles_off atIndex:4];
                            [cenc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
                            [cenc setBytes:&mp length:sizeof(mp) atIndex:6];
                            [cenc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                                    indirectBufferOffset:ind_off
                                                   threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
                        }
                    };
                    auto dispatch_i8_w2 = [&](id<MTLComputePipelineState> p2, NSUInteger bin_idx, NSUInteger th) {
                        const NSUInteger tiles_off = bin_idx * kBinStrideBytes;
                        const NSUInteger ind_off   = bin_idx * 32U + 16U;
                        if (is_nvfp4) {
                            [cenc setComputePipelineState:p2];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.expert_scale_off) atIndex:2];
                            [cenc setBuffer:eng->b_moe_mid_i8 offset:0 atIndex:3];
                            [cenc setBuffer:eng->b_moe_mid_sc offset:0 atIndex:4];
                            [cenc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:5];
                            [cenc setBuffer:eng->b_moe_work_tiles offset:tiles_off atIndex:6];
                            [cenc setBuffer:eng->b_moe_down_part offset:0 atIndex:7];
                            [cenc setBytes:&mp length:sizeof(mp) atIndex:8];
                            [cenc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                                    indirectBufferOffset:ind_off
                                                   threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
                        } else {
                            [cenc setComputePipelineState:p2];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                            [cenc setBuffer:eng->b_moe_mid_i8 offset:0 atIndex:2];
                            [cenc setBuffer:eng->b_moe_mid_sc offset:0 atIndex:3];
                            [cenc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                            [cenc setBuffer:eng->b_moe_work_tiles offset:tiles_off atIndex:5];
                            [cenc setBuffer:eng->b_moe_down_part offset:0 atIndex:6];
                            [cenc setBytes:&mp length:sizeof(mp) atIndex:7];
                            [cenc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                                    indirectBufferOffset:ind_off
                                                   threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
                        }
                    };

                    if (is_nvfp4) {
                        dispatch_i8_w13(eng->pso_moe_prefill_nvfp4_w13_i8_m128[0], 3U, 128U);
                        dispatch_i8_w13(eng->pso_moe_prefill_nvfp4_w13_i8_m64[0],  2U, 128U);
                        dispatch_i8_w13(eng->pso_moe_prefill_nvfp4_w13_i8_m32[0],  1U,  64U);
                        dispatch_i8_w13(eng->pso_moe_prefill_nvfp4_w13_i8_m16[0],  0U,  64U);
                        [cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        dispatch_i8_w13(eng->pso_moe_prefill_nvfp4_w13_i8_m128[1], 3U, 128U);
                        dispatch_i8_w13(eng->pso_moe_prefill_nvfp4_w13_i8_m64[1],  2U, 128U);
                        dispatch_i8_w13(eng->pso_moe_prefill_nvfp4_w13_i8_m32[1],  1U,  64U);
                        dispatch_i8_w13(eng->pso_moe_prefill_nvfp4_w13_i8_m16[1],  0U,  64U);
                    } else {
                        dispatch_i8_w13(eng->pso_moe_prefill_mxfp8_w13_i8_m128[0], 3U, 128U);
                        dispatch_i8_w13(eng->pso_moe_prefill_mxfp8_w13_i8_m64[0],  2U, 128U);
                        dispatch_i8_w13(eng->pso_moe_prefill_mxfp8_w13_i8_m32[0],  1U,  64U);
                        dispatch_i8_w13(eng->pso_moe_prefill_mxfp8_w13_i8_m16[0],  0U,  64U);
                        [cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        dispatch_i8_w13(eng->pso_moe_prefill_mxfp8_w13_i8_m128[1], 3U, 128U);
                        dispatch_i8_w13(eng->pso_moe_prefill_mxfp8_w13_i8_m64[1],  2U, 128U);
                        dispatch_i8_w13(eng->pso_moe_prefill_mxfp8_w13_i8_m32[1],  1U,  64U);
                        dispatch_i8_w13(eng->pso_moe_prefill_mxfp8_w13_i8_m16[1],  0U,  64U);
                    }

                    if (__builtin_expect(eng->prof_active, 0)) {
                        [cenc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        const double gs = [cb GPUStartTime], ge = [cb GPUEndTime];
                        if (ge > gs) eng->prof_ms[8U] += (ge - gs) * 1000.0;
                        cb   = [eng->queue commandBuffer];
                        cenc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    } else {
                        [cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }

                    // Quantize `mid_sorted_f16` -> `b_moe_mid_i8` + `b_moe_mid_sc` (`+128` tail already zeroed!)
                    const uint32_t total_pairs = M_valid * top_k;
                    SikQuantI8ParamsHost qp_mid{total_pairs, total_pairs, inter, 0U, 1U, 0U, 0U, 0U};
                    [cenc setComputePipelineState:eng->pso_quant_f16_to_i8_row];
                    [cenc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:0];
                    [cenc setBuffer:eng->b_moe_mid_i8 offset:0 atIndex:1];
                    [cenc setBuffer:eng->b_moe_mid_sc offset:0 atIndex:2];
                    [cenc setBuffer:eng->b_moe_mid_sc offset:0 atIndex:3];
                    [cenc setBytes:&qp_mid length:sizeof(qp_mid) atIndex:4];
                    [cenc dispatchThreadgroups:MTLSizeMake((total_pairs + 3U) / 4U, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    if (is_nvfp4) {
                        dispatch_i8_w2(eng->pso_moe_prefill_nvfp4_w2_i8_m128, 3U, 128U);
                        dispatch_i8_w2(eng->pso_moe_prefill_nvfp4_w2_i8_m64,  2U, 128U);
                        dispatch_i8_w2(eng->pso_moe_prefill_nvfp4_w2_i8_m32,  1U,  64U);
                        dispatch_i8_w2(eng->pso_moe_prefill_nvfp4_w2_i8_m16,  0U,  64U);
                    } else {
                        dispatch_i8_w2(eng->pso_moe_prefill_mxfp8_w2_i8_m128, 3U, 128U);
                        dispatch_i8_w2(eng->pso_moe_prefill_mxfp8_w2_i8_m64,  2U, 128U);
                        dispatch_i8_w2(eng->pso_moe_prefill_mxfp8_w2_i8_m32,  1U,  64U);
                        dispatch_i8_w2(eng->pso_moe_prefill_mxfp8_w2_i8_m16,  0U,  64U);
                    }

                    if (__builtin_expect(eng->prof_active, 0)) {
                        [cenc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        const double gs = [cb GPUStartTime], ge = [cb GPUEndTime];
                        if (ge > gs) eng->prof_ms[9U] += (ge - gs) * 1000.0;
                        cb  = [eng->queue commandBuffer];
                        enc = [cb computeCommandEncoder];
                    } else {
                        [cenc endEncoding];
                        enc = [cb computeCommandEncoder];
                    }
                } else {
                    // Quad-Bin (`BM = 16/32/64/128`) `BK = 128` Concurrent Indirect NAX Prefill!
                    mp.tile_cap = 128U;
                    [enc setComputePipelineState:eng->pso_moe_bin_tokens_quad];
                    [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
                    [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
                    [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:3];
                    [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:4];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    // Pre-gather sorted `x_sorted` into aliased `b_moe_down_part` + zero `+128` tail on BOTH `x_sorted` & `mid_sorted`!
                    [enc setComputePipelineState:eng->pso_moe_gather_x_f16];
                    [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:0];
                    [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
                    [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:2];
                    [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(M_valid * top_k + 128U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    sik_prof_mark(eng, cb, enc, 7U);

                    // Switch to `MTLDispatchTypeConcurrent` so all 4 bins (`128/64/32/16`) overlap across all 40 cores!
                    [enc endEncoding];
                    id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

                    constexpr NSUInteger kBinStrideBytes = 512U * 16U;
                    auto dispatch_quad_w13 = [&](id<MTLComputePipelineState> p13, NSUInteger bin_idx, NSUInteger th) {
                        const NSUInteger tiles_off = bin_idx * kBinStrideBytes;
                        const NSUInteger ind_off   = bin_idx * 32U;
                        if (is_nvfp4) {
                            [cenc setComputePipelineState:p13];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.expert_scale_off) atIndex:2];
                            [cenc setBuffer:eng->b_moe_down_part offset:0 atIndex:3];
                            [cenc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                            [cenc setBuffer:eng->b_moe_work_tiles offset:tiles_off atIndex:5];
                            [cenc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:6];
                            [cenc setBytes:&mp length:sizeof(mp) atIndex:7];
                            [cenc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                                    indirectBufferOffset:ind_off
                                                   threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
                        } else {
                            [cenc setComputePipelineState:p13];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                            [cenc setBuffer:eng->b_moe_down_part offset:0 atIndex:2];
                            [cenc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                            [cenc setBuffer:eng->b_moe_work_tiles offset:tiles_off atIndex:4];
                            [cenc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
                            [cenc setBytes:&mp length:sizeof(mp) atIndex:6];
                            [cenc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                                    indirectBufferOffset:ind_off
                                                   threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
                        }
                    };
                    auto dispatch_quad_w2 = [&](id<MTLComputePipelineState> p2, NSUInteger bin_idx, NSUInteger th) {
                        const NSUInteger tiles_off = bin_idx * kBinStrideBytes;
                        const NSUInteger ind_off   = bin_idx * 32U + 16U;
                        if (is_nvfp4) {
                            [cenc setComputePipelineState:p2];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.expert_scale_off) atIndex:2];
                            [cenc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
                            [cenc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                            [cenc setBuffer:eng->b_moe_work_tiles offset:tiles_off atIndex:5];
                            [cenc setBuffer:eng->b_moe_down_part offset:0 atIndex:6];
                            [cenc setBytes:&mp length:sizeof(mp) atIndex:7];
                            [cenc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                                    indirectBufferOffset:ind_off
                                                   threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
                        } else {
                            [cenc setComputePipelineState:p2];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                            [cenc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                            [cenc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
                            [cenc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                            [cenc setBuffer:eng->b_moe_work_tiles offset:tiles_off atIndex:4];
                            [cenc setBuffer:eng->b_moe_down_part offset:0 atIndex:5];
                            [cenc setBytes:&mp length:sizeof(mp) atIndex:6];
                            [cenc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                                    indirectBufferOffset:ind_off
                                                   threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
                        }
                    };

                    // 2-Pass 1-Loop `BN=64, BK=128` Concurrent Dispatch: Pass 0 (`gate` -> SLC `mid_sorted`), Pass 1 (`up+SiTU`)!
                    if (is_nvfp4) {
                        dispatch_quad_w13(eng->pso_moe_prefill_nvfp4_w13_m128[0],    3U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_nvfp4_w13_m64[0],     2U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_nvfp4_w13_m32[0],     1U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_nvfp4_w13_m16_sg2[0], 0U,  64U);
                        [cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        dispatch_quad_w13(eng->pso_moe_prefill_nvfp4_w13_m128[1],    3U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_nvfp4_w13_m64[1],     2U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_nvfp4_w13_m32[1],     1U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_nvfp4_w13_m16_sg2[1], 0U,  64U);
                    } else {
                        dispatch_quad_w13(eng->pso_moe_prefill_mxfp8_w13_m128[0],    3U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_mxfp8_w13_m64[0],     2U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_mxfp8_w13_m32[0],     1U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_mxfp8_w13_m16_sg2[0], 0U,  64U);
                        [cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        dispatch_quad_w13(eng->pso_moe_prefill_mxfp8_w13_m128[1],    3U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_mxfp8_w13_m64[1],     2U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_mxfp8_w13_m32[1],     1U, 128U);
                        dispatch_quad_w13(eng->pso_moe_prefill_mxfp8_w13_m16_sg2[1], 0U,  64U);
                    }

                    if (__builtin_expect(eng->prof_active, 0)) {
                        [cenc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        const double gs = [cb GPUStartTime], ge = [cb GPUEndTime];
                        if (ge > gs) eng->prof_ms[8U] += (ge - gs) * 1000.0;
                        cb   = [eng->queue commandBuffer];
                        cenc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    } else {
                        [cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }

                    if (is_nvfp4) {
                        dispatch_quad_w2(eng->pso_moe_prefill_nvfp4_w2_m128,    3U, 128U);
                        dispatch_quad_w2(eng->pso_moe_prefill_nvfp4_w2_m64,     2U, 128U);
                        dispatch_quad_w2(eng->pso_moe_prefill_nvfp4_w2_m32,     1U, 128U);
                        dispatch_quad_w2(eng->pso_moe_prefill_nvfp4_w2_m16_sg2, 0U,  64U);
                    } else {
                        dispatch_quad_w2(eng->pso_moe_prefill_mxfp8_w2_m128,    3U, 128U);
                        dispatch_quad_w2(eng->pso_moe_prefill_mxfp8_w2_m64,     2U, 128U);
                        dispatch_quad_w2(eng->pso_moe_prefill_mxfp8_w2_m32,     1U, 128U);
                        dispatch_quad_w2(eng->pso_moe_prefill_mxfp8_w2_m16_sg2, 0U,  64U);
                    }

                    if (__builtin_expect(eng->prof_active, 0)) {
                        [cenc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        const double gs = [cb GPUStartTime], ge = [cb GPUEndTime];
                        if (ge > gs) eng->prof_ms[9U] += (ge - gs) * 1000.0;
                        cb  = [eng->queue commandBuffer];
                        enc = [cb computeCommandEncoder];
                    } else {
                        [cenc endEncoding];
                        enc = [cb computeCommandEncoder];
                    }
                }

                // 2D Parallel `moe_combine` (FP16->FP32 for NAX prefill, FP32 for short grouped-8) -> `sconv_static_prefill`!
                [enc setComputePipelineState:(use_short_grouped_moe ? eng->pso_moe_combine : eng->pso_moe_combine_f16)];
                [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:0];
                [enc setBuffer:eng->b_topk_w offset:0 atIndex:1];
                [enc setBuffer:eng->b_moe_out_f32 offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(((D / 4U) + 255U) / 256U, M_valid, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                sik_encode_sconv_static(enc, eng, eng->b_moe_out_f32, ld.mlp_sconv_w.offset, mlp_sconv_off,
                                        eng->b_cur, nil, 0, M_valid, D, 0U, 1U);
                sik_prof_mark(eng, cb, enc, 10U);
            } else {
                // Decode (`M <= 8`): Exact-M Router (fuses `PreMoE_RMS1` on loop visits + `sigmoid+bias` on M=1 NVFP4!)
                const bool use_fused_m1_topk =
                    (M_valid == 1U && is_nvfp4 && save_snap_mask == 0U && E <= 128U && top_k <= 8U);
                if (fuse_moe_rms) {
                    SikMoeRouterRmsParamsHost rmp{
                        M_valid, D, E, top_k, cfg.rms_norm_eps,
                        (D + 31U) / 32U, use_fused_m1_topk ? 1U : 0U, 0U
                    };
                    [enc setComputePipelineState:eng->pso_moe_router_f32_rms_m[M_valid - 1U]];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_weight.offset) atIndex:0];
                    [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_norm.offset) atIndex:2];
                    [enc setBuffer:eng->b_rms_sq offset:0 atIndex:3];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_bias.offset) atIndex:4];
                    [enc setBuffer:eng->b_router_logits offset:0 atIndex:5];
                    [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:6];
                    [enc setBytes:&rmp length:sizeof(rmp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(E, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else if (use_fused_m1_topk) {
                    [enc setComputePipelineState:eng->pso_moe_router_f32_m1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_weight.offset) atIndex:0];
                    [enc setBuffer:eng->b_x_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_bias.offset) atIndex:2];
                    [enc setBuffer:eng->b_router_logits offset:0 atIndex:3];
                    [enc setBytes:&rp length:sizeof(rp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(E, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    [enc setComputePipelineState:eng->pso_moe_router_f32_m[M_valid - 1U]];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_weight.offset) atIndex:0];
                    [enc setBuffer:eng->b_x_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_router_logits offset:0 atIndex:2];
                    [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(E, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }

                if (use_fused_m1_topk) {
                    sik_prof_mark(eng, cb, enc, eng->prof_detail ? 23U : 7U);
                } else {
                    sik_prof_mark_detail(eng, cb, enc, 23U);
                    if (M_valid == 1U) {
                        [enc setComputePipelineState:eng->pso_moe_topk_m1];
                        [enc setBuffer:eng->b_router_logits offset:0 atIndex:0];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_bias.offset) atIndex:1];
                        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:2];
                        [enc setBuffer:eng->b_topk_w offset:0 atIndex:3];
                        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                        [enc setBytes:&rp length:sizeof(rp) atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } else {
                        [enc setComputePipelineState:eng->pso_moe_topk_bin_small];
                        [enc setBuffer:eng->b_router_logits offset:0 atIndex:0];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_bias.offset) atIndex:1];
                        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:2];
                        [enc setBuffer:eng->b_topk_w offset:0 atIndex:3];
                        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                        [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:7];
                        [enc setBytes:&rp length:sizeof(rp) atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    sik_prof_mark(eng, cb, enc, eng->prof_detail ? 24U : 7U);
                }

                bool did_fused_m1_sconv = false;
                if (M_valid == 1U) {
                    // Direct Zero-Indirect `M = 1` Fast Path (`0` encoder switch overhead!)
                    if (is_nvfp4) {
                        if (use_fused_m1_topk) {
                            SikMoeW13M1ParamsHost w13p{
                                D, inter, E, top_k, cfg.route_scale,
                                cfg.mlp_gate_softcap, cfg.mlp_up_softcap, 0U
                            };
                            [enc setComputePipelineState:eng->pso_moe_w13_nvfp4_m1_topk];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.expert_scale_off) atIndex:2];
                            [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                            [enc setBuffer:eng->b_router_logits offset:0 atIndex:4];
                            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:5];
                            [enc setBuffer:eng->b_topk_w offset:0 atIndex:6];
                            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:7];
                            [enc setBytes:&w13p length:sizeof(w13p) atIndex:8];
                            [enc dispatchThreadgroups:MTLSizeMake((inter + 7U) / 8U, top_k, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        } else {
                            [enc setComputePipelineState:eng->pso_moe_nvfp4_w13_b[0]];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.expert_scale_off) atIndex:2];
                            [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                            [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:7];
                            [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                            [enc dispatchThreadgroups:MTLSizeMake((inter + 7U) / 8U, top_k, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }
                        sik_prof_mark(eng, cb, enc, 8U);

                        if (save_snap_mask == 0U) {
                            [enc setComputePipelineState:eng->pso_moe_w2_nvfp4_m1_fused_sconv];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.expert_scale_off) atIndex:2];
                            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
                            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:4];
                            [enc setBuffer:eng->b_topk_w offset:0 atIndex:5];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_sconv_w.offset) atIndex:6];
                            [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(mlp_sconv_off) atIndex:7];
                            [enc setBuffer:eng->b_cur offset:0 atIndex:8];
                            [enc setBytes:&mp length:sizeof(mp) atIndex:9];
                            [enc dispatchThreadgroups:MTLSizeMake((D + 3U) / 4U, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                            sik_prof_mark(eng, cb, enc, 9U);
                            did_fused_m1_sconv = true;
                        } else {
                            [enc setComputePipelineState:eng->pso_moe_nvfp4_w2_b[0]];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.expert_scale_off) atIndex:2];
                            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
                            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                            [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                            [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:7];
                            [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                            [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, top_k, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            sik_prof_mark(eng, cb, enc, 9U);
                        }
                    } else {
                        [enc setComputePipelineState:eng->pso_moe_mxfp8_w13_b[0]];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                        [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:2];
                        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:5];
                        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:6];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake((inter + 7U) / 8U, top_k, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        sik_prof_mark(eng, cb, enc, 8U);

                        [enc setComputePipelineState:eng->pso_moe_mxfp8_w2_b[0]];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                        [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
                        [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                        [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                        [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:5];
                        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:6];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, top_k, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        sik_prof_mark(eng, cb, enc, 9U);
                    }
                } else {
                    // Single-Dispatch High-Occupancy Grouped-4 Spec-Verify MoE (`b[1]=max2`, `b[2]=max4` -> 2x occupancy!)
                    const uint32_t b_slot = (M_valid <= 2U) ? 1U : 2U;
                    encode_grouped_single(b_slot);
                }

                if (!did_fused_m1_sconv) {
                    // Fused Decode `moe_combine + mlp_sconv + residual` (linear or tree-indexed!)
                    SikSconvParamsHost csp{M_valid, D, save_snap_mask, 1U, hd, n_kv, snap_stride_f4, top_k};
                    [enc setComputePipelineState:(tree_active ? eng->pso_moe_combine_sconv_tree : eng->pso_moe_combine_sconv)];
                    [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:0];
                    [enc setBuffer:eng->b_topk_w offset:0 atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_sconv_w.offset) atIndex:2];
                    [enc setBuffer:eng->b_sconv_state offset:static_cast<NSUInteger>(mlp_sconv_off) atIndex:3];
                    [enc setBuffer:eng->b_cur offset:0 atIndex:4];
                    [enc setBuffer:(eng->b_sconv_snaps ? eng->b_sconv_snaps : eng->b_sconv_state)
                            offset:static_cast<NSUInteger>(mlp_sconv_off) atIndex:5];
                    [enc setBytes:&csp length:sizeof(csp) atIndex:6];
                    if (tree_active) {
                        [enc setBuffer:eng->b_tree_meta offset:0 atIndex:7];
                    }
                    [enc dispatchThreadgroups:MTLSizeMake((D + 63U) / 64U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    sik_prof_mark(eng, cb, enc, 10U);
                }
            }
        }
        // 2-CB Pipelined Encode on unprofiled Decode/Verify (`M <= 8`): commit visits 0..5 (`~0.02 ms` CPU,
        // `~1.0 ms` GPU) so GPU wakes & runs while CPU encodes visits 6..47 + Head + Sample (`~0.15 ms` hidden!)
        if (!eng->prof_active && !is_prefill_nax && M_valid <= kMaxSpecBatch && il == 5U && n_visits >= 16U) {
            [enc endEncoding];
            [cb commit];
            cb  = [eng->queue commandBufferWithUnretainedReferences];
            enc = [cb computeCommandEncoder];
        }
    }

    // 3. Fused Final Head: `attn_res(output_res_query) + rms_norm(final_norm)` (+ Tap 3 capture!)
    if (capture_mtp_taps || run_final_norm || run_unembed_all || run_unembed_last) {
        sik_encode_attn_res_rmsnorm(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                    eng->hdr.output_res_query.offset, eng->hdr.final_norm.offset,
                                    eng->b_in_f16[0], nil, false, M_valid, M_valid,
                                    capture_taps, 3U);
        sik_prof_mark_detail(eng, cb, enc, 25U);

        // 4. Unembed (`FP16`/`Q8_0`/`Q4_A` w/ precomputed `x_blk_sum`) + Final Logit Softcapping (`30.0f`)
        if (run_unembed_all) {
            if (eng->hdr.unembed.dtype == STRATA_INKLING2_DTYPE_Q4_A && M_valid <= kMaxSpecBatch) {
                uint64_t uw = 0ULL, us = 0ULL, ub = 0ULL;
                sik_slice_sub_offsets(eng->hdr.unembed, uw, us, ub);
                SikGemvBatchParamsHost gp{M_valid, eng->hdr.unembed.shape0, eng->hdr.unembed.shape1,
                                          kOutF32Softcap, cfg.final_logit_softcapping, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_q4a_gemv_xsum_m[M_valid - 1U]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ub) atIndex:2];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:4];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:5];
                [enc setBuffer:eng->b_logits offset:0 atIndex:6];
                [enc setBytes:&gp length:sizeof(gp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((eng->hdr.unembed.shape0 + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                sik_encode_dense_linear(enc, eng, eng->hdr.unembed,
                                        eng->b_in_f16[0], 0, nil, 0, eng->b_logits, 0,
                                        M_valid, M_valid, kOutF32Softcap, true, cfg.final_logit_softcapping);
            }
        } else if (run_unembed_last) {
            const NSUInteger last_x_off = static_cast<NSUInteger>(M_valid - 1U) * D * sizeof(uint16_t);
            if (eng->hdr.unembed.dtype == STRATA_INKLING2_DTYPE_Q4_A && M_valid <= kMaxSpecBatch) {
                uint64_t uw = 0ULL, us = 0ULL, ub = 0ULL;
                sik_slice_sub_offsets(eng->hdr.unembed, uw, us, ub);
                const NSUInteger last_xs_off = static_cast<NSUInteger>(M_valid - 1U) * (D >> 6U) * sizeof(float);
                SikGemvBatchParamsHost gp{1U, eng->hdr.unembed.shape0, eng->hdr.unembed.shape1,
                                          kOutF32Softcap, cfg.final_logit_softcapping, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_q4a_gemv_xsum_m[0]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ub) atIndex:2];
                [enc setBuffer:eng->b_in_f16[0] offset:last_x_off atIndex:3];
                [enc setBuffer:eng->b_x_blk_sum[0] offset:last_xs_off atIndex:4];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:5];
                [enc setBuffer:eng->b_logits offset:0 atIndex:6];
                [enc setBytes:&gp length:sizeof(gp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((eng->hdr.unembed.shape0 + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                sik_encode_dense_linear(enc, eng, eng->hdr.unembed,
                                        eng->b_in_f16[0], last_x_off, nil, 0, eng->b_logits, 0,
                                        1U, 1U, kOutF32Softcap, true, cfg.final_logit_softcapping);
            }
        }
    }
}

/* ============================================================================
 * Native 3-Layer DSpark Block-8 MTP + 32K SLC FR-Spec & Lazy Full-Vocab Fold
 * ============================================================================ */

bool sik_ensure_full_mtp_fold(StrataInkling2GpuEngine* eng) {
    if (!eng || eng->hdr.mtp.enabled == 0U) return true;
    if (eng->full_fold_ready) return true;

    const uint32_t V        = eng->cfg.vocab_size_unpadded;
    const uint32_t D        = eng->cfg.hidden_size;
    const uint32_t mtp_rank = (eng->hdr.mtp.markov_rank > 0U) ? eng->hdr.mtp.markov_rank : 256U;

    if (!eng->b_mtp_fold_weight) {
        eng->b_mtp_fold_weight = sik_alloc_private_aligned(eng->device, static_cast<size_t>(V) * mtp_rank * sizeof(uint16_t));
        eng->b_mtp_fold_bias   = sik_alloc_private_aligned(eng->device, static_cast<size_t>(V) * sizeof(float));
        if (!eng->b_mtp_fold_weight || !eng->b_mtp_fold_bias) return false;
        if (@available(macOS 15.0, *)) {
            if (eng->residency_set) {
                [eng->residency_set addAllocation:eng->b_mtp_fold_weight];
                [eng->residency_set addAllocation:eng->b_mtp_fold_bias];
                [eng->residency_set commit];
            }
        }
    }

    uint64_t uw = 0ULL, us = 0ULL, ub = 0ULL;
    sik_slice_sub_offsets(eng->hdr.unembed, uw, us, ub);

    id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    SikMtpFoldParamsHost fp{V, D, mtp_rank, eng->hdr.unembed.dtype};
    [enc setComputePipelineState:eng->pso_mtp_fold_head_nax];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ub) atIndex:2];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.mtp.markov_hidden_out.offset) atIndex:3];
    [enc setBuffer:eng->b_mtp_fold_weight offset:0 atIndex:4];
    [enc setBytes:&fp length:sizeof(fp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(mtp_rank / 64U, (V + 63U) / 64U, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    // `fold_bias = unembed @ half(markov_hidden_bias)`
    {
        SikPadCastParamsHost cp{1U, 1U, D, 0U};
        [enc setComputePipelineState:eng->pso_f32_to_f16_pad];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.mtp.markov_hidden_bias.offset) atIndex:0];
        [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:1];
        [enc setBytes:&cp length:sizeof(cp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(((D / 4U) + 255U) / 256U, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    sik_encode_dense_linear(enc, eng, eng->hdr.unembed, eng->b_in_f16[0], 0,
                            nil, 0, eng->b_mtp_fold_bias, 0, 1U, 1U, kOutF32, true);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError) return false;

    eng->full_fold_ready = true;
    return true;
}

void sik_encode_mtp_ctx_update(
    id<MTLComputeCommandEncoder> enc,
    StrataInkling2GpuEngine*     eng,
    id<MTLBuffer>                taps_buf,
    uint32_t                     pos_start,
    uint32_t                     n_new,
    bool                         force_gemv
) {
    if (eng->hdr.mtp.enabled == 0U || n_new == 0U) return;
    const StrataInkling2MtpDesc& mtp = eng->hdr.mtp;
    const uint32_t D        = eng->cfg.hidden_size;
    const uint32_t n_kv     = eng->cfg.num_key_value_heads;
    const uint32_t hd       = eng->cfg.head_dim;
    const uint32_t M_padded = (!force_gemv && eng->has_nax && n_new > kMaxSpecBatch)
        ? ((n_new + 63U) & ~63U) : n_new;

    uint32_t swa_slot  = 0U;
    uint32_t glob_slot = 0U;
    for (uint32_t l = 0U; l < mtp.num_layers; ++l) {
        const StrataInkling2MtpLayerDesc& ml = mtp.layers[l];
        const bool is_local = (ml.is_local_swa != 0U);

        {
            SikMtpFuseParamsHost fp{n_new, M_padded, D, l, mtp.rms_norm_eps, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_mtp_fuse_taps_norm];
            [enc setBuffer:taps_buf offset:0 atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.fuse_alpha.offset) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ml.dflare_norm.offset) atIndex:2];
            [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
            [enc setBuffer:eng->b_x_f32 offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(M_padded, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        if (n_new <= kMaxSpecBatch) {
            // Single-Dispatch Fused `k_proj_ctx + v_proj_ctx` (FP16/Q8_0/Q4_A w/ precomputed `x_blk_sum`!)
            const StrataInkling2TensorSlice kv[2] = {ml.k_proj_ctx, ml.v_proj_ctx};
            SikMtpQkvMixedParamsHost qp{};
            qp.cols = D;
            uint32_t tot_rows = 0U;
            for (uint32_t j = 0U; j < 2U; ++j) {
                qp.rows[j]  = kv[j].shape0;
                qp.dtype[j] = kv[j].dtype;
                sik_slice_sub_offsets(kv[j], qp.w_off[j], qp.s_off[j], qp.b_off[j]);
                tot_rows += kv[j].shape0;
            }
            [enc setComputePipelineState:eng->pso_mtp_kv_ctx_mixed_m[n_new - 1U]];
            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
            [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:1];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:2];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:3];
            [enc setBytes:&qp length:sizeof(qp) atIndex:4];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((tot_rows + 7U) / 8U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            sik_encode_dense_linear(enc, eng, ml.k_proj_ctx, eng->b_in_f16[0], 0,
                                    nil, 0, eng->b_k_raw, 0, n_new, M_padded, kOutF32, force_gemv);
            sik_encode_dense_linear(enc, eng, ml.v_proj_ctx, eng->b_in_f16[0], 0,
                                    nil, 0, eng->b_v_raw, 0, n_new, M_padded, kOutF32, force_gemv);
        }

        id<MTLBuffer>    k_buf = is_local ? eng->b_mtp_swa_k_cache : eng->b_mtp_glob_k_cache;
        id<MTLBuffer>    v_buf = is_local ? eng->b_mtp_swa_v_cache : eng->b_mtp_glob_v_cache;
        const uint32_t   kstr  = is_local ? kSwaRingLen : eng->max_seq_alloc;
        const NSUInteger loff  = static_cast<NSUInteger>(
            is_local ? (static_cast<uint64_t>(swa_slot) * eng->swa_kv_layer_stride)
                     : (static_cast<uint64_t>(glob_slot) * eng->mtp_glob_kv_layer_stride));

        {
            SikMtpCtxKvParamsHost cp{n_new, n_kv, hd, pos_start, kstr, is_local ? 1U : 0U, mtp.rms_norm_eps, 0U};
            [enc setComputePipelineState:eng->pso_mtp_ctx_kv_prep];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:0];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ml.k_norm_ctx.offset) atIndex:2];
            [enc setBuffer:eng->b_mtp_inv_freq offset:0 atIndex:3];
            [enc setBuffer:k_buf offset:loff atIndex:4];
            [enc setBuffer:v_buf offset:loff atIndex:5];
            [enc setBytes:&cp length:sizeof(cp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(n_kv, n_new, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        if (is_local) ++swa_slot;
        else          ++glob_slot;
    }
}

/* ============================================================================
 * Sparse Sliding-Window Logit Penalties (Full & FR-Spec) & Parallel Sampler
 * ============================================================================ */

inline bool sik_penalties_active(const StrataInkling2SamplingParams* s) {
    if (!s) return false;
    return (s->presence_penalty != 0.0f) ||
           (s->frequency_penalty != 0.0f) ||
           (s->repetition_penalty > 0.0f && s->repetition_penalty != 1.0f);
}

inline void sik_penalties_inc(StrataInkling2GpuEngine* eng, uint32_t* ids, uint16_t* cnt, uint32_t t) {
    const uint32_t V = eng->cfg.vocab_size_unpadded;
    if (t >= V) return;
    if (cnt[t] == 0U && eng->pen_num_ids < V) {
        const uint32_t idx = eng->pen_num_ids++;
        ids[idx]        = t;
        eng->pen_pos[t] = idx;
    }
    if (cnt[t] < 65535U) ++cnt[t];
}

inline void sik_penalties_dec(StrataInkling2GpuEngine* eng, uint32_t* ids, uint16_t* cnt, uint32_t t) {
    const uint32_t V = eng->cfg.vocab_size_unpadded;
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

inline void sik_penalties_clear_active(StrataInkling2GpuEngine* eng) {
    if (!eng || !eng->b_pen_ids || !eng->b_pen_counts) return;
    const uint32_t* ids = reinterpret_cast<const uint32_t*>([eng->b_pen_ids contents]);
    uint16_t*       cnt = reinterpret_cast<uint16_t*>([eng->b_pen_counts contents]);
    const uint32_t  V   = eng->cfg.vocab_size_unpadded;
    const uint32_t  n   = eng->pen_num_ids;
    for (uint32_t i = 0U; i < n; ++i) {
        const uint32_t t = ids[i];
        if (t < V) { cnt[t] = 0U; eng->pen_pos[t] = 0xFFFFFFFFU; }
    }
    eng->pen_num_ids = 0U;
}

void sik_penalties_sync_window(StrataInkling2GpuEngine* eng, uint32_t eff_window) {
    if (!eng || !eng->b_pen_ids || !eng->b_pen_counts || !eng->bound_slot) return;
    const std::vector<uint32_t>& hist = eng->bound_slot->pen_hist;
    const uint32_t L = static_cast<uint32_t>(hist.size());
    const uint32_t target_start = (eff_window > 0U && L > eff_window) ? (L - eff_window) : 0U;
    if (eng->pen_cache_slot == eng->bound_slot &&
        eng->pen_window_start == target_start && eng->pen_window_end == L) {
        return;
    }

    uint32_t* ids = reinterpret_cast<uint32_t*>([eng->b_pen_ids contents]);
    uint16_t* cnt = reinterpret_cast<uint16_t*>([eng->b_pen_counts contents]);

    if (eng->pen_cache_slot != eng->bound_slot ||
        target_start >= eng->pen_window_end || eng->pen_window_end > L) {
        sik_penalties_clear_active(eng);
        eng->pen_cache_slot   = eng->bound_slot;
        eng->pen_window_start = target_start;
        eng->pen_window_end   = target_start;
    }
    if (target_start < eng->pen_window_start) {
        for (uint32_t i = target_start; i < eng->pen_window_start; ++i) sik_penalties_inc(eng, ids, cnt, hist[i]);
        eng->pen_window_start = target_start;
    }
    if (eng->pen_window_end < L) {
        for (uint32_t i = eng->pen_window_end; i < L; ++i) sik_penalties_inc(eng, ids, cnt, hist[i]);
        eng->pen_window_end = L;
    }
    if (eng->pen_window_start < target_start) {
        for (uint32_t i = eng->pen_window_start; i < target_start; ++i) sik_penalties_dec(eng, ids, cnt, hist[i]);
        eng->pen_window_start = target_start;
    }
}

void sik_encode_logit_penalties(
    id<MTLComputeCommandEncoder>        enc,
    StrataInkling2GpuEngine*            eng,
    const StrataInkling2SamplingParams* sampling,
    uint32_t                            M,
    id<MTLBuffer>                       extra_toks_buf,
    uint32_t                            extra_mode,
    uint32_t                            extra_count,
    bool                                use_shortlist = false,
    uint32_t                            shortlist_V   = 0U
) {
    if (!sik_penalties_active(sampling) || !eng->pso_apply_logit_penalties ||
        !eng->b_pen_ids || !eng->b_pen_counts || !eng->bound_slot) {
        return;
    }
    const uint32_t eff_window = (sampling->penalty_window == 0U)
        ? 0U : std::max<uint32_t>(sampling->penalty_window, 8U);
    sik_penalties_sync_window(eng, eff_window);

    const uint32_t total_items = eng->pen_num_ids + extra_count;
    if (total_items == 0U) return;

    const uint32_t V = eng->cfg.vocab_size_unpadded;
    SikLogitPenaltyParamsHost pp{};
    pp.M                  = M;
    pp.vocab_size         = V;
    pp.logits_stride      = use_shortlist ? shortlist_V : V;
    pp.num_hist_ids       = eng->pen_num_ids;
    pp.extra_mode         = extra_mode;
    pp.extra_count        = extra_count;
    pp.presence_penalty   = sampling->presence_penalty;
    pp.frequency_penalty  = sampling->frequency_penalty;
    pp.repetition_penalty = sampling->repetition_penalty;
    pp.use_shortlist      = use_shortlist ? 1U : 0U;
    for (uint32_t j = 0U; j < 8U; ++j) pp.drop_toks[j] = 0xFFFFFFFFU;
    if (eff_window > 0U && extra_count > 0U) {
        const std::vector<uint32_t>& hist = eng->bound_slot->pen_hist;
        const uint32_t L = static_cast<uint32_t>(hist.size());
        const uint32_t max_d = std::min<uint32_t>(extra_count, 8U);
        for (uint32_t j = 0U; j < max_d; ++j) {
            if (L + j >= eff_window) {
                const uint32_t evict_idx = L + j - eff_window;
                if (evict_idx < L) pp.drop_toks[j] = hist[evict_idx];
            }
        }
    }

    [enc setComputePipelineState:eng->pso_apply_logit_penalties];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_pen_ids offset:0 atIndex:1];
    [enc setBuffer:eng->b_pen_counts offset:0 atIndex:2];
    [enc setBuffer:(extra_toks_buf ? extra_toks_buf : eng->b_token_ids) offset:0 atIndex:3];
    [enc setBuffer:eng->b_frspec_full_to_short offset:0 atIndex:4];
    [enc setBytes:&pp length:sizeof(pp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((total_items + 255U) / 256U, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sik_encode_logit_penalties_tree(
    id<MTLComputeCommandEncoder>        enc,
    StrataInkling2GpuEngine*            eng,
    const StrataInkling2SamplingParams* sampling,
    uint32_t                            M,
    uint32_t                            extra_mode,
    uint32_t                            extra_count,
    bool                                use_shortlist = false,
    uint32_t                            shortlist_V   = 0U
) {
    if (!sik_penalties_active(sampling) || !eng->pso_apply_logit_penalties_tree ||
        !eng->b_pen_ids || !eng->b_pen_counts || !eng->b_tree_meta || !eng->bound_slot) {
        return;
    }
    const uint32_t eff_window = (sampling->penalty_window == 0U)
        ? 0U : std::max<uint32_t>(sampling->penalty_window, 8U);
    sik_penalties_sync_window(eng, eff_window);

    const uint32_t total_items = eng->pen_num_ids + extra_count;
    if (total_items == 0U) return;

    const uint32_t V = eng->cfg.vocab_size_unpadded;
    SikLogitPenaltyParamsHost pp{};
    pp.M                  = M;
    pp.vocab_size         = V;
    pp.logits_stride      = use_shortlist ? shortlist_V : V;
    pp.num_hist_ids       = eng->pen_num_ids;
    pp.extra_mode         = extra_mode;
    pp.extra_count        = extra_count;
    pp.presence_penalty   = sampling->presence_penalty;
    pp.frequency_penalty  = sampling->frequency_penalty;
    pp.repetition_penalty = sampling->repetition_penalty;
    pp.use_shortlist      = use_shortlist ? 1U : 0U;
    for (uint32_t j = 0U; j < 8U; ++j) pp.drop_toks[j] = 0xFFFFFFFFU;
    if (eff_window > 0U && extra_count > 0U) {
        const std::vector<uint32_t>& hist = eng->bound_slot->pen_hist;
        const uint32_t L = static_cast<uint32_t>(hist.size());
        const uint32_t max_d = std::min<uint32_t>(extra_count, 8U);
        for (uint32_t j = 0U; j < max_d; ++j) {
            if (L + j >= eff_window) {
                const uint32_t evict_idx = L + j - eff_window;
                if (evict_idx < L) pp.drop_toks[j] = hist[evict_idx];
            }
        }
    }

    [enc setComputePipelineState:eng->pso_apply_logit_penalties_tree];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_pen_ids offset:0 atIndex:1];
    [enc setBuffer:eng->b_pen_counts offset:0 atIndex:2];
    [enc setBuffer:eng->b_tree_meta offset:0 atIndex:3];
    [enc setBuffer:eng->b_frspec_full_to_short offset:0 atIndex:4];
    [enc setBytes:&pp length:sizeof(pp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((total_items + 255U) / 256U, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sik_encode_sample_batch(
    id<MTLComputeCommandEncoder>        enc,
    StrataInkling2GpuEngine*            eng,
    const StrataInkling2SamplingParams* sampling,
    uint32_t                            M,
    id<MTLBuffer>                       out_tok_buf,
    NSUInteger                          out_tok_off,
    id<MTLBuffer>                       out_pmax_buf,
    NSUInteger                          out_pmax_off,
    id<MTLBuffer>                       top_ids_buf,
    NSUInteger                          top_ids_off,
    id<MTLBuffer>                       top_probs_buf,
    NSUInteger                          top_probs_off,
    id<MTLBuffer>                       keep_k_buf,
    NSUInteger                          keep_k_off,
    bool                                use_shortlist  = false,
    uint32_t                            shortlist_V    = 0U,
    uint64_t                            token_map_off  = 0ULL
) {
    SikSampleParamsHost sp{};
    sp.M             = M;
    sp.vocab_size    = use_shortlist ? shortlist_V : eng->cfg.vocab_size_unpadded;
    sp.use_token_map = use_shortlist ? 1U : 0U;
    sp.compute_pmax  = (out_pmax_buf != nil) ? 1U : 0U;

    const bool is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
    if (is_greedy) {
        sp.greedy   = 1U;
        sp.inv_temp = 1.0f;
        sp.top_k    = 1U;
        sp.top_p    = 1.0f;
    } else {
        sp.greedy   = 0U;
        sp.inv_temp = 1.0f / std::max<float>(sampling->temperature, 1e-5f);
        sp.top_k    = (sampling->top_k > 0U && sampling->top_k <= 64U) ? sampling->top_k : 20U;
        sp.top_p    = sampling->top_p;
        const uint64_t seed = sampling->rng_seed ^ (eng->sample_step * 0xD1342543DE82EF95ULL);
        sp.rng_lo = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
        sp.rng_hi = static_cast<uint32_t>(seed >> 32ULL);
        eng->sample_step += M;
    }

    if (is_greedy) {
        [enc setComputePipelineState:eng->pso_sample_greedy_stage1];
        [enc setBuffer:eng->b_logits offset:0 atIndex:0];
        [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:1];
        [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:2];
        [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:3];
        [enc setBytes:&sp length:sizeof(sp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(128, M, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:eng->pso_sample_greedy_stage2];
        [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
        [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
        [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:2];
        [enc setBuffer:(use_shortlist ? eng->b_uma : eng->b_token_ids)
                offset:(use_shortlist ? static_cast<NSUInteger>(token_map_off) : 0U)
               atIndex:3];
        [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:4];
        [enc setBuffer:(out_pmax_buf ? out_pmax_buf : eng->b_draft_pmax)
                offset:(out_pmax_buf ? out_pmax_off : 0U)
               atIndex:5];
        [enc setBytes:&sp length:sizeof(sp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }

    [enc setComputePipelineState:eng->pso_sample_stage1];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:1];
    [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:2];
    [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:3];
    [enc setBytes:&sp length:sizeof(sp) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(64, M, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:eng->pso_sample_stage2];
    [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
    [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
    [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:2];
    [enc setBuffer:(use_shortlist ? eng->b_uma : eng->b_token_ids)
            offset:(use_shortlist ? static_cast<NSUInteger>(token_map_off) : 0U)
           atIndex:3];
    [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:4];
    [enc setBuffer:(out_pmax_buf ? out_pmax_buf : eng->b_draft_pmax)
            offset:(out_pmax_buf ? out_pmax_off : 0U)
           atIndex:5];
    [enc setBuffer:(top_ids_buf   ? top_ids_buf   : eng->b_sample_top_ids)   offset:(top_ids_buf   ? top_ids_off   : 0U) atIndex:6];
    [enc setBuffer:(top_probs_buf ? top_probs_buf : eng->b_sample_top_probs) offset:(top_probs_buf ? top_probs_off : 0U) atIndex:7];
    [enc setBuffer:(keep_k_buf    ? keep_k_buf    : eng->b_sample_keep_k)    offset:(keep_k_buf    ? keep_k_off    : 0U) atIndex:8];
    [enc setBytes:&sp length:sizeof(sp) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

/* ============================================================================
 * 2-Stage Bitplane Exact Q8_0 LM Head (`vhead_bitplane`, 100% Bit-Exact!)
 * ============================================================================ */

inline bool sik_can_use_bitplane(
    const StrataInkling2GpuEngine*      eng,
    const StrataInkling2SamplingParams* sampling,
    const uint32_t*                     next_token_out,
    const float*                        logits_out
) {
    if (!eng || !eng->bitplane_head_enabled || !eng->bp_ready) return false;
    if (logits_out != nullptr || next_token_out == nullptr) return false;
    if (eng->hdr.unembed.dtype != STRATA_INKLING2_DTYPE_Q8_0) return false;
    if ((eng->cfg.hidden_size & 63U) != 0U || eng->cfg.hidden_size > 2304U) return false;
    // Monotonicity certificate `pen(softcap(exact)) <= softcap(ub)` requires non-negative penalties.
    if (sampling) {
        if (sampling->presence_penalty < 0.0f || sampling->frequency_penalty < 0.0f) return false;
        if (sampling->repetition_penalty > 0.0f && sampling->repetition_penalty < 1.0f) return false;
    }
    // For wide stochastic top-K (e.g. K=20), 20th-best threshold admits ~15-25K candidates,
    // where Plane 0 + sparse Stage 2 ~= full Q8_0; keep full Q8_0 for K > 4 so there is 0% slowdown!
    const bool is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
    if (!is_greedy) {
        const uint32_t eff_k = (sampling->top_k > 0U && sampling->top_k <= 64U) ? sampling->top_k : 20U;
        if (eff_k > 4U) return false;
    }
    return true;
}

void sik_encode_bitplane_unembed_sample(
    id<MTLCommandBuffer> __strong&         cb,
    id<MTLComputeCommandEncoder> __strong& enc,
    StrataInkling2GpuEngine*               eng,
    id<MTLBuffer>                          x_in_f16,
    NSUInteger                             x_off,
    const StrataInkling2SamplingParams*    sampling,
    id<MTLBuffer>                          out_tok_buf,
    NSUInteger                             out_tok_off
) {
    const uint32_t V = eng->cfg.vocab_size_unpadded;
    const uint32_t D = eng->cfg.hidden_size;
    const bool is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
    const uint32_t eff_top_k = is_greedy ? 1U
        : ((sampling->top_k > 0U && sampling->top_k <= 64U) ? sampling->top_k : 20U);

    uint32_t rng_lo = 0U, rng_hi = 0U;
    float inv_temp = 1.0f, top_p = 1.0f;
    if (!is_greedy) {
        inv_temp = 1.0f / std::max<float>(sampling->temperature, 1e-5f);
        top_p    = sampling->top_p;
        const uint64_t seed = sampling->rng_seed ^ (eng->sample_step * 0xD1342543DE82EF95ULL);
        rng_lo = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
        rng_hi = static_cast<uint32_t>(seed >> 32ULL);
        eng->sample_step += 1ULL;
    }

    const bool pen_on = sik_penalties_active(sampling);
    uint32_t pen_tgs = 0U;
    if (pen_on && eng->b_pen_ids && eng->b_pen_counts) {
        const uint32_t eff_window = (sampling->penalty_window == 0U)
            ? 0U : std::max<uint32_t>(sampling->penalty_window, 8U);
        sik_penalties_sync_window(eng, eff_window);
        if (eng->pen_num_ids > 0U) {
            pen_tgs = (eng->pen_num_ids + 255U) / 256U;
        }
    }

    uint64_t uw = 0ULL, us = 0ULL, ub = 0ULL;
    sik_slice_sub_offsets(eng->hdr.unembed, uw, us, ub);

    SikBpParamsHost bp{};
    bp.V         = V;
    bp.D         = D;
    bp.softcap   = eng->cfg.final_logit_softcapping;
    bp.top_k     = eff_top_k;
    bp.num_seeds = is_greedy ? 16U : std::min<uint32_t>(64U, std::max<uint32_t>(eff_top_k + 12U, 32U));
    bp.has_pen   = (pen_tgs > 0U) ? 1U : 0U;
    bp.pres_pen  = pen_on ? sampling->presence_penalty   : 0.0f;
    bp.freq_pen  = pen_on ? sampling->frequency_penalty  : 0.0f;
    bp.rep_pen   = pen_on ? sampling->repetition_penalty : 1.0f;
    bp.pen_tgs   = pen_tgs;
    bp.is_greedy = is_greedy ? 1U : 0U;
    bp.inv_temp  = inv_temp;
    bp.top_p     = top_p;
    bp.rng_lo    = rng_lo;
    bp.rng_hi    = rng_hi;

    // 1. Prep x quarter L2 norms + `-120.5 * sum_g` + reset meta
    [enc setComputePipelineState:eng->pso_bp_prep_x];
    [enc setBuffer:x_in_f16 offset:x_off atIndex:0];
    [enc setBuffer:eng->b_bp_xprep offset:0 atIndex:1];
    [enc setBuffer:eng->b_bp_meta offset:0 atIndex:2];
    [enc setBytes:&bp length:sizeof(bp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // 2. Stage 1: Plane 0 (`hi4` + `w_s` + `rnorm4`, 51.9% DRAM!) -> `ub[V]` + 512 block seeds
    [enc setComputePipelineState:eng->pso_bp_stage1_hi4];
    [enc setBuffer:eng->b_bp_hi4 offset:0 atIndex:0];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
    [enc setBuffer:eng->b_bp_rnorm4 offset:0 atIndex:2];
    [enc setBuffer:x_in_f16 offset:x_off atIndex:3];
    [enc setBuffer:eng->b_bp_xprep offset:0 atIndex:4];
    [enc setBuffer:eng->b_bp_ub offset:0 atIndex:5];
    [enc setBuffer:eng->b_bp_stage_v offset:0 atIndex:6];
    [enc setBuffer:eng->b_bp_stage_i offset:0 atIndex:7];
    [enc setBytes:&bp length:sizeof(bp) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(512, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    // 3. Stage 1.5: Exact `Q8_0` threshold on top `num_seeds` rows (`< 3 us`, halves pruning radius!)
    [enc setComputePipelineState:eng->pso_bp_stage15_seed];
    [enc setBuffer:eng->b_bp_stage_v offset:0 atIndex:0];
    [enc setBuffer:eng->b_bp_stage_i offset:0 atIndex:1];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:2];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:3];
    [enc setBuffer:x_in_f16 offset:x_off atIndex:4];
    [enc setBuffer:eng->b_pen_counts offset:0 atIndex:5];
    [enc setBuffer:eng->b_bp_meta offset:0 atIndex:6];
    [enc setBytes:&bp length:sizeof(bp) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // 4. Filter & compact ALL rows with `ub >= T` -> `cand_ids` (`< 5 us`)
    [enc setComputePipelineState:eng->pso_bp_filter_compact];
    [enc setBuffer:eng->b_bp_ub offset:0 atIndex:0];
    [enc setBuffer:eng->b_bp_cand_ids offset:0 atIndex:1];
    [enc setBuffer:eng->b_bp_meta offset:0 atIndex:2];
    [enc setBytes:&bp length:sizeof(bp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(128, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // 5. Prep GPU indirect dispatch args (sparse Stage 2 vs full Q8_0 fallback, 0 CPU syncs!)
    [enc setComputePipelineState:eng->pso_bp_prep_indirect];
    [enc setBuffer:eng->b_bp_meta offset:0 atIndex:0];
    [enc setBytes:&bp length:sizeof(bp) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    // 6. Stage 2: Exact `Q8_0` dot product ONLY on surviving candidates (bit-identical to `sik_q8_gemv_l16_fixed`!)
    [enc setComputePipelineState:eng->pso_bp_stage2_exact_cand];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
    [enc setBuffer:x_in_f16 offset:x_off atIndex:2];
    [enc setBuffer:eng->b_bp_cand_ids offset:0 atIndex:3];
    [enc setBuffer:eng->b_pen_counts offset:0 atIndex:4];
    [enc setBuffer:eng->b_bp_cand_logits offset:0 atIndex:5];
    [enc setBuffer:eng->b_bp_meta offset:0 atIndex:6];
    [enc setBytes:&bp length:sizeof(bp) atIndex:7];
    [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                           indirectBufferOffset:16U
                          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    sik_prof_mark_detail(eng, cb, enc, 26U);

    // 7. Sparse candidate sampler (indirect; 0 TGs if fallback active)
    if (is_greedy) {
        [enc setComputePipelineState:eng->pso_bp_cand_sample_greedy];
        [enc setBuffer:eng->b_bp_cand_ids offset:0 atIndex:0];
        [enc setBuffer:eng->b_bp_cand_logits offset:0 atIndex:1];
        [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:2];
        [enc setBuffer:eng->b_bp_meta offset:0 atIndex:3];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                               indirectBufferOffset:32U
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc setComputePipelineState:eng->pso_bp_cand_sample_stoch_s1];
        [enc setBuffer:eng->b_bp_cand_ids offset:0 atIndex:0];
        [enc setBuffer:eng->b_bp_cand_logits offset:0 atIndex:1];
        [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:2];
        [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:3];
        [enc setBuffer:eng->b_bp_meta offset:0 atIndex:4];
        [enc setBytes:&bp length:sizeof(bp) atIndex:5];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                               indirectBufferOffset:32U
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:eng->pso_bp_cand_sample_stoch_s2];
        [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
        [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
        [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:2];
        [enc setBytes:&bp length:sizeof(bp) atIndex:3];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                               indirectBufferOffset:48U
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 8. Sound Fallback Path (0-TG indirect no-op unless candidate set exceeded 32,768!)
    {
        SikGemvBatchParamsHost gp{1U, eng->hdr.unembed.shape0, eng->hdr.unembed.shape1,
                                  kOutF32Softcap, eng->cfg.final_logit_softcapping, 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_q8_gemv_m[0]];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
        [enc setBuffer:x_in_f16 offset:x_off atIndex:2];
        [enc setBuffer:x_in_f16 offset:0 atIndex:3];
        [enc setBuffer:eng->b_logits offset:0 atIndex:4];
        [enc setBytes:&gp length:sizeof(gp) atIndex:5];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                               indirectBufferOffset:64U
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (pen_tgs > 0U) {
        SikLogitPenaltyParamsHost lpp{};
        lpp.M                  = 1U;
        lpp.vocab_size         = V;
        lpp.logits_stride      = V;
        lpp.num_hist_ids       = eng->pen_num_ids;
        lpp.extra_mode         = 0U;
        lpp.extra_count        = 0U;
        lpp.presence_penalty   = sampling->presence_penalty;
        lpp.frequency_penalty  = sampling->frequency_penalty;
        lpp.repetition_penalty = sampling->repetition_penalty;
        lpp.use_shortlist      = 0U;
        for (uint32_t j = 0U; j < 8U; ++j) lpp.drop_toks[j] = 0xFFFFFFFFU;
        [enc setComputePipelineState:eng->pso_apply_logit_penalties];
        [enc setBuffer:eng->b_logits offset:0 atIndex:0];
        [enc setBuffer:eng->b_pen_ids offset:0 atIndex:1];
        [enc setBuffer:eng->b_pen_counts offset:0 atIndex:2];
        [enc setBuffer:eng->b_token_ids offset:0 atIndex:3];
        [enc setBuffer:eng->b_frspec_full_to_short offset:0 atIndex:4];
        [enc setBytes:&lpp length:sizeof(lpp) atIndex:5];
        [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                               indirectBufferOffset:80U
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    {
        SikSampleParamsHost sp{};
        sp.M             = 1U;
        sp.vocab_size    = V;
        sp.greedy        = is_greedy ? 1U : 0U;
        sp.inv_temp      = inv_temp;
        sp.top_k         = eff_top_k;
        sp.top_p         = top_p;
        sp.rng_lo        = rng_lo;
        sp.rng_hi        = rng_hi;
        sp.use_token_map = 0U;
        sp.compute_pmax  = 0U;
        if (is_greedy) {
            [enc setComputePipelineState:eng->pso_sample_greedy_stage1];
            [enc setBuffer:eng->b_logits offset:0 atIndex:0];
            [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:1];
            [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:2];
            [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:3];
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
            [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                                   indirectBufferOffset:96U
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:eng->pso_sample_greedy_stage2];
            [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
            [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
            [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:2];
            [enc setBuffer:eng->b_token_ids offset:0 atIndex:3];
            [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:4];
            [enc setBuffer:eng->b_draft_pmax offset:0 atIndex:5];
            [enc setBytes:&sp length:sizeof(sp) atIndex:6];
            [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                                   indirectBufferOffset:112U
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setComputePipelineState:eng->pso_sample_stage1];
            [enc setBuffer:eng->b_logits offset:0 atIndex:0];
            [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:1];
            [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:2];
            [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:3];
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
            [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                                   indirectBufferOffset:96U
                                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:eng->pso_sample_stage2];
            [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
            [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
            [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:2];
            [enc setBuffer:eng->b_token_ids offset:0 atIndex:3];
            [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:4];
            [enc setBuffer:eng->b_draft_pmax offset:0 atIndex:5];
            [enc setBuffer:eng->b_sample_top_ids offset:0 atIndex:6];
            [enc setBuffer:eng->b_sample_top_probs offset:0 atIndex:7];
            [enc setBuffer:eng->b_sample_keep_k offset:0 atIndex:8];
            [enc setBytes:&sp length:sizeof(sp) atIndex:9];
            [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_bp_meta
                                   indirectBufferOffset:112U
                                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }
}

// Shared Steps 1-4 of 3-Layer DSpark Block-8 MTP Draft (produces `norm_f32[0..7]`, `f_rnn`, `base_logits[0..eff_k-1]`)
void sik_encode_mtp_dspark_backbone(
    id<MTLComputeCommandEncoder> enc,
    StrataInkling2GpuEngine*     eng,
    uint32_t                     cur_pos,
    uint32_t                     eff_k,
    bool                         use_frspec
) {
    const StrataInkling2MtpDesc& mtp = eng->hdr.mtp;
    const uint32_t B     = mtp.block_size; // 8
    const uint32_t D     = eng->cfg.hidden_size;
    const uint32_t n_q   = eng->cfg.num_attention_heads;
    const uint32_t n_kv  = eng->cfg.num_key_value_heads;
    const uint32_t hd    = eng->cfg.head_dim;

    // 1. Init Block-8 (supports FP16 and Q4_A `embed_tokens`, emits `b_x_blk_sum[0]`!)
    {
        SikMtpInitBlockParamsHost ip{B, D, eng->cfg.vocab_size_padded, mtp.rms_norm_eps};
        if (eng->hdr.embed_tokens.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
            uint64_t ew = 0ULL, es = 0ULL, eb = 0ULL;
            sik_slice_sub_offsets(eng->hdr.embed_tokens, ew, es, eb);
            [enc setComputePipelineState:eng->pso_mtp_init_block_q4a];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ew) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(es) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eb) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_norm.offset) atIndex:3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.mask_embed.offset) atIndex:4];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.layers[0].input_layernorm.offset) atIndex:5];
            [enc setBuffer:eng->b_mtp_token_ids offset:0 atIndex:6];
            [enc setBuffer:eng->b_mtp_res_f32 offset:0 atIndex:7];
            [enc setBuffer:eng->b_mtp_norm_f16 offset:0 atIndex:8];
            [enc setBytes:&ip length:sizeof(ip) atIndex:9];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:10];
        } else {
            [enc setComputePipelineState:eng->pso_mtp_init_block];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.offset) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_norm.offset) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.mask_embed.offset) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.layers[0].input_layernorm.offset) atIndex:3];
            [enc setBuffer:eng->b_mtp_token_ids offset:0 atIndex:4];
            [enc setBuffer:eng->b_mtp_res_f32 offset:0 atIndex:5];
            [enc setBuffer:eng->b_mtp_norm_f16 offset:0 atIndex:6];
            [enc setBytes:&ip length:sizeof(ip) atIndex:7];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:8];
        }
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 2. 3 DSpark Transformer Layers (`B = 8`, mixed-precision FP16/Q8_0/Q4_A w/ `x_blk_sum` + Shared-KV Attention!)
    uint32_t swa_slot  = 0U;
    uint32_t glob_slot = 0U;
    for (uint32_t l = 0U; l < mtp.num_layers; ++l) {
        const StrataInkling2MtpLayerDesc& ml = mtp.layers[l];
        const bool is_local = (ml.is_local_swa != 0U);

        // Single-Dispatch Fused Q/K/V (`FP16`/`Q8_0`/`Q4_A` w/ precomputed `b_x_blk_sum[0]`)
        if (B == 8U) {
            const StrataInkling2TensorSlice qkv[3] = {ml.q_proj, ml.k_proj, ml.v_proj};
            SikMtpQkvMixedParamsHost qp{};
            qp.cols = D;
            uint32_t tot_rows = 0U;
            for (uint32_t j = 0U; j < 3U; ++j) {
                qp.rows[j]  = qkv[j].shape0;
                qp.dtype[j] = qkv[j].dtype;
                sik_slice_sub_offsets(qkv[j], qp.w_off[j], qp.s_off[j], qp.b_off[j]);
                tot_rows += qkv[j].shape0;
            }
            const bool pure_q4a_mtp_qkv =
                (ml.q_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A) &&
                (ml.k_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A) &&
                (ml.v_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A);
            [enc setComputePipelineState:(pure_q4a_mtp_qkv ? eng->pso_mtp_qkv_q4a : eng->pso_mtp_qkv_mixed)];
            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
            [enc setBuffer:eng->b_mtp_norm_f16 offset:0 atIndex:1];
            [enc setBuffer:eng->b_q_raw offset:0 atIndex:2];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:3];
            [enc setBuffer:eng->b_mtp_draft_v offset:0 atIndex:4];
            [enc setBytes:&qp length:sizeof(qp) atIndex:5];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(pure_q4a_mtp_qkv ? ((tot_rows + 3U) / 4U) : ((tot_rows + 7U) / 8U), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            sik_encode_dense_linear(enc, eng, ml.q_proj, eng->b_mtp_norm_f16, 0, nil, 0, eng->b_q_raw, 0, B, B, kOutF32, true);
            sik_encode_dense_linear(enc, eng, ml.k_proj, eng->b_mtp_norm_f16, 0, nil, 0, eng->b_k_raw, 0, B, B, kOutF32, true);
            sik_encode_dense_linear(enc, eng, ml.v_proj, eng->b_mtp_norm_f16, 0, eng->b_mtp_draft_v, 0, nil, 0, B, B, kOutF16, true);
        }

        // Per-head q_norm/k_norm + NeoX RoPE
        {
            SikMtpBlockPrepParamsHost bp{B, n_q, n_kv, hd, cur_pos, mtp.rms_norm_eps, 0U, 0U};
            [enc setComputePipelineState:eng->pso_mtp_block_qk_prep];
            [enc setBuffer:eng->b_q_raw offset:0 atIndex:0];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ml.q_norm.offset) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ml.k_norm.offset) atIndex:3];
            [enc setBuffer:eng->b_mtp_inv_freq offset:0 atIndex:4];
            [enc setBuffer:eng->b_q_f32 offset:0 atIndex:5];
            [enc setBuffer:eng->b_mtp_draft_k offset:0 atIndex:6];
            [enc setBytes:&bp length:sizeof(bp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(n_q + n_kv, B, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        // Shared-KV Block-16 Bidirectional Block-8 Attention (reads ctx_k/v ONCE for all 8 queries!)
        {
            const uint32_t   ctx_start = (is_local && cur_pos > mtp.sliding_window) ? (cur_pos - mtp.sliding_window) : 0U;
            const uint32_t   total_kv  = (cur_pos - ctx_start) + B;
            const uint32_t   max_sp    = kMtpMaxSplits;
            const uint32_t   l_splits  = sik_compute_mtp_splits(total_kv, max_sp);
            id<MTLBuffer>    ctx_k     = is_local ? eng->b_mtp_swa_k_cache : eng->b_mtp_glob_k_cache;
            id<MTLBuffer>    ctx_v     = is_local ? eng->b_mtp_swa_v_cache : eng->b_mtp_glob_v_cache;
            const uint32_t   kstr      = is_local ? kSwaRingLen : eng->max_seq_alloc;
            const NSUInteger loff      = static_cast<NSUInteger>(
                is_local ? (static_cast<uint64_t>(swa_slot) * eng->swa_kv_layer_stride)
                         : (static_cast<uint64_t>(glob_slot) * eng->mtp_glob_kv_layer_stride));

            SikMtpAttnParamsHost ap{
                B, n_q, n_kv, hd, ctx_start, cur_pos, kstr,
                is_local ? 1U : 0U, max_sp, 1.0f / std::sqrt(static_cast<float>(hd)), 0U, 0U
            };
            [enc setComputePipelineState:eng->pso_mtp_block_attn_splitkv];
            [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
            [enc setBuffer:ctx_k offset:loff atIndex:1];
            [enc setBuffer:ctx_v offset:loff atIndex:2];
            [enc setBuffer:eng->b_mtp_draft_k offset:0 atIndex:3];
            [enc setBuffer:eng->b_mtp_draft_v offset:0 atIndex:4];
            [enc setBuffer:eng->b_part_m offset:0 atIndex:5];
            [enc setBuffer:eng->b_part_l offset:0 atIndex:6];
            [enc setBuffer:eng->b_part_o offset:0 atIndex:7];
            [enc setBytes:&ap length:sizeof(ap) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(l_splits, n_kv, 1)
                threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];

            [enc setComputePipelineState:eng->pso_mtp_block_attn_reduce];
            [enc setBuffer:eng->b_part_m offset:0 atIndex:0];
            [enc setBuffer:eng->b_part_l offset:0 atIndex:1];
            [enc setBuffer:eng->b_part_o offset:0 atIndex:2];
            [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(n_q, B, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        // `o_proj` (`Q4_A` uses precomputed `b_wo_blk_sum`!) + `add_rmsnorm` (emits `b_x_blk_sum[0]`!)
        if (ml.o_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A && B <= kMaxSpecBatch) {
            uint64_t ow = 0ULL, os = 0ULL, ob = 0ULL;
            sik_slice_sub_offsets(ml.o_proj, ow, os, ob);
            SikGemvBatchParamsHost gp{B, ml.o_proj.shape0, ml.o_proj.shape1, kOutF32, 0.0f, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_q4a_gemv_xsum_m[B - 1U]];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ow) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(os) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ob) atIndex:2];
            [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:4];
            [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:5];
            [enc setBuffer:eng->b_wo_out_f32 offset:0 atIndex:6];
            [enc setBytes:&gp length:sizeof(gp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((ml.o_proj.shape0 + 7U) / 8U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            sik_encode_dense_linear(enc, eng, ml.o_proj, eng->b_attn_gated_f16, 0,
                                    nil, 0, eng->b_wo_out_f32, 0, B, B, kOutF32, true);
        }
        {
            SikMtpAddNormParamsHost anp{B, D, mtp.rms_norm_eps, 0U};
            [enc setComputePipelineState:eng->pso_mtp_add_rmsnorm];
            [enc setBuffer:eng->b_wo_out_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_mtp_res_f32 offset:0 atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ml.post_attn_norm.offset) atIndex:2];
            [enc setBuffer:eng->b_mtp_norm_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_mtp_norm_f32 offset:0 atIndex:4];
            [enc setBytes:&anp length:sizeof(anp) atIndex:5];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Fused Mixed-Precision SwiGLU FFN (`Q4_A` uses `uint2` + `b_x_blk_sum[0]`!) + `down_proj` (`Q4_A` w/ `b_wo_blk_sum`!)
        {
            SikMtpSwigluMixedParamsHost sp{};
            sik_slice_sub_offsets(ml.gate_proj, sp.wg_off, sp.wg_s_off, sp.wg_b_off);
            sik_slice_sub_offsets(ml.up_proj,   sp.wu_off, sp.wu_s_off, sp.wu_b_off);
            sp.dtype = ml.gate_proj.dtype;
            sp.M     = B;
            sp.rows  = mtp.intermediate_size;
            sp.cols  = D;
            const bool pure_q4a_ffn =
                (ml.gate_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A) &&
                (ml.up_proj.dtype   == STRATA_INKLING2_DTYPE_Q4_A);
            [enc setComputePipelineState:(pure_q4a_ffn ? eng->pso_mtp_gate_up_silu_b8_q4a : eng->pso_mtp_gate_up_silu_b8_mixed)];
            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
            [enc setBuffer:eng->b_mtp_norm_f16 offset:0 atIndex:1];
            [enc setBuffer:eng->b_mtp_ffn_mid_f16 offset:0 atIndex:2];
            [enc setBytes:&sp length:sizeof(sp) atIndex:3];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((mtp.intermediate_size + 3U) / 4U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        if (ml.down_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A && B <= kMaxSpecBatch) {
            SikBlkSumParamsHost bsp{B, mtp.intermediate_size, 0U, 0U};
            [enc setComputePipelineState:eng->pso_f16_blk64_sum];
            [enc setBuffer:eng->b_mtp_ffn_mid_f16 offset:0 atIndex:0];
            [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:1];
            [enc setBytes:&bsp length:sizeof(bsp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            uint64_t dw = 0ULL, ds = 0ULL, db = 0ULL;
            sik_slice_sub_offsets(ml.down_proj, dw, ds, db);
            SikGemvBatchParamsHost gp{B, ml.down_proj.shape0, ml.down_proj.shape1, kOutF32, 0.0f, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_q4a_gemv_xsum_m[B - 1U]];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(dw) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ds) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(db) atIndex:2];
            [enc setBuffer:eng->b_mtp_ffn_mid_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:4];
            [enc setBuffer:eng->b_mtp_ffn_mid_f16 offset:0 atIndex:5];
            [enc setBuffer:eng->b_wo_out_f32 offset:0 atIndex:6];
            [enc setBytes:&gp length:sizeof(gp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((ml.down_proj.shape0 + 7U) / 8U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            sik_encode_dense_linear(enc, eng, ml.down_proj, eng->b_mtp_ffn_mid_f16, 0,
                                    nil, 0, eng->b_wo_out_f32, 0, B, B, kOutF32, true);
        }

        const bool     is_last   = (l + 1U == mtp.num_layers);
        const uint64_t next_norm = is_last ? mtp.final_norm.offset : mtp.layers[l + 1U].input_layernorm.offset;
        {
            SikMtpAddNormParamsHost anp{B, D, mtp.rms_norm_eps, is_last ? 1U : 0U};
            [enc setComputePipelineState:eng->pso_mtp_add_rmsnorm];
            [enc setBuffer:eng->b_wo_out_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_mtp_res_f32 offset:0 atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(next_norm) atIndex:2];
            [enc setBuffer:eng->b_mtp_norm_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_mtp_norm_f32 offset:0 atIndex:4];
            [enc setBytes:&anp length:sizeof(anp) atIndex:5];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        if (is_local) ++swa_slot;
        else          ++glob_slot;
    }

    // 3. `f_rnn = mix(b_mtp_taps_last, fuse_alpha[num_layers], markov_dflare_norm)` -> `b_mtp_f_rnn [D]` F32
    //    (`fuse_alpha` is validated as `[num_layers + 1, 4]`: the final row index is `num_layers`, `3` for the stock drafter)
    {
        SikMtpFuseParamsHost fp{1U, 1U, D, mtp.num_layers, mtp.rms_norm_eps, 1U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_mtp_fuse_taps_norm];
        [enc setBuffer:eng->b_mtp_taps_last offset:0 atIndex:0];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.fuse_alpha.offset) atIndex:1];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.markov_dflare_norm.offset) atIndex:2];
        [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
        [enc setBuffer:eng->b_mtp_f_rnn offset:0 atIndex:4];
        [enc setBytes:&fp length:sizeof(fp) atIndex:5];
        [enc setBuffer:eng->b_x_blk_sum[1] offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 4. Batched `base_logits[0..eff_k-1]`:
    //    - FR-Spec (32K Q4_A w/ precomputed `b_x_blk_sum[0][1..eff_k]`!) OR Full-Vocab fallback!
    const StrataInkling2TensorSlice& unemb_sl = use_frspec ? mtp.frspec.unembed_q4a : eng->hdr.unembed;
    if (unemb_sl.dtype == STRATA_INKLING2_DTYPE_Q4_A && eff_k > 0U && eff_k <= kMaxSpecBatch) {
        uint64_t uw = 0ULL, us = 0ULL, ub = 0ULL;
        sik_slice_sub_offsets(unemb_sl, uw, us, ub);
        SikGemvBatchParamsHost gp{eff_k, unemb_sl.shape0, unemb_sl.shape1, kOutF32, 0.0f, 0U, 0U, 0U};
        [enc setComputePipelineState:eng->pso_q4a_gemv_xsum_m[eff_k - 1U]];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ub) atIndex:2];
        [enc setBuffer:eng->b_mtp_norm_f16 offset:static_cast<NSUInteger>(D * sizeof(uint16_t)) atIndex:3];
        [enc setBuffer:eng->b_x_blk_sum[0] offset:static_cast<NSUInteger>((D >> 6U) * sizeof(float)) atIndex:4];
        [enc setBuffer:eng->b_mtp_norm_f16 offset:0 atIndex:5];
        [enc setBuffer:eng->b_mtp_base_logits offset:0 atIndex:6];
        [enc setBytes:&gp length:sizeof(gp) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((unemb_sl.shape0 + 7U) / 8U, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        sik_encode_dense_linear(enc, eng, unemb_sl,
                                eng->b_mtp_norm_f16, static_cast<NSUInteger>(D * sizeof(uint16_t)),
                                nil, 0, eng->b_mtp_base_logits, 0, eff_k, eff_k, kOutF32, true);
    }
}

// Encodes 3-Layer DSpark Block-8 (`K = 1..7`) Linear Chain MTP Draft
void sik_encode_mtp_draft_block(
    id<MTLComputeCommandEncoder>        enc,
    StrataInkling2GpuEngine*            eng,
    uint32_t                            cur_pos,
    uint32_t                            eff_k,
    bool                                use_frspec,
    bool                                want_pmax,
    const StrataInkling2SamplingParams* sampling
) {
    const StrataInkling2MtpDesc& mtp = eng->hdr.mtp;
    const uint32_t D     = eng->cfg.hidden_size;
    const uint32_t V     = eng->cfg.vocab_size_unpadded;
    const uint32_t rank  = mtp.markov_rank;
    const bool is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
    const uint32_t V_head = use_frspec ? mtp.frspec.shortlist_size : V;

    sik_encode_mtp_dspark_backbone(enc, eng, cur_pos, eff_k, use_frspec);

    // 5. Zero `b_mtp_rnn_state[256]`, then run `eff_k` sequential rank-256 Markov RNN steps!
    sik_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_mtp_rnn_state, static_cast<size_t>(rank) * sizeof(float));

    uint64_t fw_q_off = 0ULL, fw_s_off = 0ULL, fw_b_off = 0ULL;
    if (use_frspec) sik_slice_sub_offsets(mtp.frspec.fold_weight_q8, fw_q_off, fw_s_off, fw_b_off);

    for (uint32_t k = 0U; k < eff_k; ++k) {
        {
            SikMtpRnnPrepParamsHost rp{rank, D, k, eng->cfg.vocab_size_padded};
            [enc setComputePipelineState:eng->pso_mtp_rnn_prep];
            [enc setBuffer:eng->b_mtp_rnn_state offset:0 atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.markov_w1.offset) atIndex:1];
            [enc setBuffer:eng->b_mtp_norm_f32 offset:0 atIndex:2];
            [enc setBuffer:eng->b_mtp_f_rnn offset:0 atIndex:3];
            [enc setBuffer:eng->b_mtp_token_ids offset:0 atIndex:4];
            [enc setBuffer:eng->b_mtp_rnn_in_f16 offset:0 atIndex:5];
            [enc setBytes:&rp length:sizeof(rp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((2U * rank + 2U * D + 255U) / 256U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        sik_encode_dense_linear(enc, eng, mtp.markov_rnn_weight, eng->b_mtp_rnn_in_f16, 0,
                                nil, 0, eng->b_mtp_rnn_raw_f32, 0, 1U, 1U, kOutF32, true);

        {
            [enc setComputePipelineState:eng->pso_mtp_rnn_gate];
            [enc setBuffer:eng->b_mtp_rnn_raw_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.markov_rnn_bias.offset) atIndex:1];
            [enc setBuffer:eng->b_mtp_rnn_state offset:0 atIndex:2];
            [enc setBuffer:eng->b_mtp_latent_f16 offset:0 atIndex:3];
            [enc setBytes:&rank length:sizeof(uint32_t) atIndex:4];
            /* One thread per rank element (`r >= rank` early-outs): fixed 256-wide groups keep any validated
             * `markov_rank` (1..4096) inside Metal's threads-per-threadgroup limit; rank 256 is still 1 x 256. */
            [enc dispatchThreadgroups:MTLSizeMake((rank + 255U) / 256U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // 5d-5f. Folded Markov Head + Softcap + Penalties + Sample!
        // When FR-Spec + Greedy + no penalties (BOTH `want_pmax == false` AND `want_pmax == true` State C Prose/Probe!):
        // FUSE Fold+Softcap+Argmax+OnlineLSE into ONE 128-TG pass (0 `b_logits` roundtrip, 7 fewer dispatches!)
        const bool fuse_frspec_greedy_fold =
            (use_frspec && is_greedy && !sik_penalties_active(sampling));
        if (fuse_frspec_greedy_fold) {
            SikMtpFoldAddParamsHost fp{V_head, rank, eng->cfg.final_logit_softcapping, want_pmax ? 1U : 0U};
            const NSUInteger base_off = static_cast<NSUInteger>(static_cast<uint64_t>(k) * V_head * sizeof(float));
            [enc setComputePipelineState:eng->pso_mtp_fold_sample_greedy_q8];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw_q_off) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw_s_off) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.frspec.fold_bias_f32.offset) atIndex:2];
            [enc setBuffer:eng->b_mtp_base_logits offset:base_off atIndex:3];
            [enc setBuffer:eng->b_mtp_latent_f16 offset:0 atIndex:4];
            [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:5];
            [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:6];
            [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:7];
            [enc setBytes:&fp length:sizeof(fp) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(128, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            SikSampleParamsHost sp{};
            sp.M             = 1U;
            sp.vocab_size    = V_head;
            sp.greedy        = 1U;
            sp.inv_temp      = 1.0f;
            sp.top_k         = 1U;
            sp.top_p         = 1.0f;
            sp.use_token_map = 1U;
            sp.compute_pmax  = want_pmax ? 1U : 0U;
            [enc setComputePipelineState:eng->pso_sample_greedy_stage2];
            [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
            [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
            [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.frspec.token_ids.offset) atIndex:3];
            [enc setBuffer:eng->b_mtp_token_ids offset:static_cast<NSUInteger>(k + 1U) * sizeof(uint32_t) atIndex:4];
            [enc setBuffer:eng->b_draft_pmax offset:(want_pmax ? (static_cast<NSUInteger>(k) * sizeof(float)) : 0U) atIndex:5];
            [enc setBytes:&sp length:sizeof(sp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            {
                SikMtpFoldAddParamsHost fp{V_head, rank, eng->cfg.final_logit_softcapping, 0U};
                const NSUInteger base_off = static_cast<NSUInteger>(static_cast<uint64_t>(k) * V_head * sizeof(float));
                if (use_frspec) {
                    [enc setComputePipelineState:eng->pso_mtp_fold_add_softcap_q8];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw_q_off) atIndex:0];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw_s_off) atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.frspec.fold_bias_f32.offset) atIndex:2];
                    [enc setBuffer:eng->b_mtp_base_logits offset:base_off atIndex:3];
                    [enc setBuffer:eng->b_mtp_latent_f16 offset:0 atIndex:4];
                    [enc setBuffer:eng->b_logits offset:0 atIndex:5];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                } else {
                    [enc setComputePipelineState:eng->pso_mtp_fold_add_softcap];
                    [enc setBuffer:eng->b_mtp_fold_weight offset:0 atIndex:0];
                    [enc setBuffer:eng->b_mtp_fold_bias offset:0 atIndex:1];
                    [enc setBuffer:eng->b_mtp_base_logits offset:base_off atIndex:2];
                    [enc setBuffer:eng->b_mtp_latent_f16 offset:0 atIndex:3];
                    [enc setBuffer:eng->b_logits offset:0 atIndex:4];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:5];
                }
                [enc dispatchThreadgroups:MTLSizeMake((V_head + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            // 5e. Exact Sliding-Window Logit Penalties (maps full tok -> short_idx when FR-Spec!)
            sik_encode_logit_penalties(enc, eng, sampling, 1U, eng->b_mtp_token_ids, 1U, k, use_frspec, V_head);

            // 5f. Sample draft token `k` -> full vocab ID in `b_mtp_token_ids[k + 1]` + `p_max` in `b_draft_pmax[k]`!
            sik_encode_sample_batch(
                enc, eng, sampling, 1U,
                eng->b_mtp_token_ids, static_cast<NSUInteger>(k + 1U) * sizeof(uint32_t),
                want_pmax ? eng->b_draft_pmax : nil, static_cast<NSUInteger>(k) * sizeof(float),
                is_greedy ? nil : eng->b_draft_top_ids,   static_cast<NSUInteger>(k) * 64ULL * sizeof(uint32_t),
                is_greedy ? nil : eng->b_draft_top_probs, static_cast<NSUInteger>(k) * 64ULL * sizeof(float),
                is_greedy ? nil : eng->b_draft_keep_k,    static_cast<NSUInteger>(k) * sizeof(uint32_t),
                use_frspec, V_head, use_frspec ? mtp.frspec.token_ids.offset : 0ULL
            );
        }
    }
}

// Encodes 3-Layer DSpark Block-8 (`N <= 8, W <= 4`) Dynamic Confidence-Adaptive On-GPU Markov Beam Tree Draft!
void sik_encode_mtp_draft_tree(
    id<MTLComputeCommandEncoder>        enc,
    StrataInkling2GpuEngine*            eng,
    uint32_t                            cur_pos,
    uint32_t                            eff_k,
    bool                                use_frspec,
    const StrataInkling2SamplingParams* sampling
) {
    const StrataInkling2MtpDesc& mtp = eng->hdr.mtp;
    const uint32_t D      = eng->cfg.hidden_size;
    const uint32_t V      = eng->cfg.vocab_size_unpadded;
    const uint32_t rank   = mtp.markov_rank;
    const bool is_greedy  = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
    const uint32_t V_head = use_frspec ? mtp.frspec.shortlist_size : V;

    sik_encode_mtp_dspark_backbone(enc, eng, cur_pos, eff_k, use_frspec);

    // 5. Initialize root node 0 + zero 8 x rank tree Markov RNN states
    {
        SikTreeInitParamsHost tip{1U + eff_k, eng->beam_tree_width, rank, 0U};
        [enc setComputePipelineState:eng->pso_tree_init];
        [enc setBuffer:eng->b_mtp_token_ids offset:0 atIndex:0];
        [enc setBuffer:eng->b_tree_meta offset:0 atIndex:1];
        [enc setBuffer:eng->b_mtp_tree_states offset:0 atIndex:2];
        [enc setBytes:&tip length:sizeof(tip) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    uint64_t fw_q_off = 0ULL, fw_s_off = 0ULL, fw_b_off = 0ULL;
    if (use_frspec) sik_slice_sub_offsets(mtp.frspec.fold_weight_q8, fw_q_off, fw_s_off, fw_b_off);

    const bool fuse_frspec_greedy_tree = (use_frspec && is_greedy && !sik_penalties_active(sampling));

    // 6. Run `eff_k` on-GPU best-first Markov beam tree expansion iterations (`0` CPU syncs!)
    for (uint32_t step = 0U; step < eff_k; ++step) {
        {
            SikMtpRnnPrepParamsHost rp{rank, D, step, eng->cfg.vocab_size_padded};
            [enc setComputePipelineState:eng->pso_mtp_rnn_prep_tree];
            [enc setBuffer:eng->b_mtp_tree_states offset:0 atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.markov_w1.offset) atIndex:1];
            [enc setBuffer:eng->b_mtp_norm_f32 offset:0 atIndex:2];
            [enc setBuffer:eng->b_mtp_f_rnn offset:0 atIndex:3];
            [enc setBuffer:eng->b_tree_meta offset:0 atIndex:4];
            [enc setBuffer:eng->b_mtp_rnn_in_f16 offset:0 atIndex:5];
            [enc setBytes:&rp length:sizeof(rp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((2U * rank + 2U * D + 255U) / 256U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        sik_encode_dense_linear(enc, eng, mtp.markov_rnn_weight, eng->b_mtp_rnn_in_f16, 0,
                                nil, 0, eng->b_mtp_rnn_raw_f32, 0, 1U, 1U, kOutF32, true);

        {
            [enc setComputePipelineState:eng->pso_mtp_rnn_gate_tree];
            [enc setBuffer:eng->b_mtp_rnn_raw_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.markov_rnn_bias.offset) atIndex:1];
            [enc setBuffer:eng->b_mtp_tree_states offset:0 atIndex:2];
            [enc setBuffer:eng->b_mtp_latent_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_tree_meta offset:0 atIndex:4];
            [enc setBytes:&rank length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((rank + 255U) / 256U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        if (fuse_frspec_greedy_tree) {
            SikMtpFoldAddParamsHost fp{V_head, rank, eng->cfg.final_logit_softcapping, 1U};
            [enc setComputePipelineState:eng->pso_mtp_fold_top4_q8];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw_q_off) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw_s_off) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.frspec.fold_bias_f32.offset) atIndex:2];
            [enc setBuffer:eng->b_mtp_base_logits offset:0 atIndex:3];
            [enc setBuffer:eng->b_mtp_latent_f16 offset:0 atIndex:4];
            [enc setBuffer:eng->b_tree_meta offset:0 atIndex:5];
            [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:6];
            [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:7];
            [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:8];
            [enc setBytes:&fp length:sizeof(fp) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(128, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            SikSampleParamsHost sp{};
            sp.M             = 1U;
            sp.vocab_size    = V_head;
            sp.greedy        = 1U;
            sp.inv_temp      = 1.0f;
            sp.top_k         = 4U;
            sp.top_p         = 1.0f;
            sp.use_token_map = 1U;
            sp.compute_pmax  = 1U;
            [enc setComputePipelineState:eng->pso_tree_expand_top4];
            [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
            [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
            [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.frspec.token_ids.offset) atIndex:3];
            [enc setBuffer:eng->b_tree_meta offset:0 atIndex:4];
            [enc setBuffer:eng->b_mtp_token_ids offset:0 atIndex:5];
            [enc setBytes:&sp length:sizeof(sp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            {
                SikMtpFoldAddParamsHost fp{V_head, rank, eng->cfg.final_logit_softcapping, 0U};
                if (use_frspec) {
                    [enc setComputePipelineState:eng->pso_mtp_fold_add_softcap_tree_q8];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw_q_off) atIndex:0];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw_s_off) atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(mtp.frspec.fold_bias_f32.offset) atIndex:2];
                    [enc setBuffer:eng->b_mtp_base_logits offset:0 atIndex:3];
                    [enc setBuffer:eng->b_mtp_latent_f16 offset:0 atIndex:4];
                    [enc setBuffer:eng->b_tree_meta offset:0 atIndex:5];
                    [enc setBuffer:eng->b_logits offset:0 atIndex:6];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:7];
                } else {
                    [enc setComputePipelineState:eng->pso_mtp_fold_add_softcap_tree_f16];
                    [enc setBuffer:eng->b_mtp_fold_weight offset:0 atIndex:0];
                    [enc setBuffer:eng->b_mtp_fold_bias offset:0 atIndex:1];
                    [enc setBuffer:eng->b_mtp_base_logits offset:0 atIndex:2];
                    [enc setBuffer:eng->b_mtp_latent_f16 offset:0 atIndex:3];
                    [enc setBuffer:eng->b_tree_meta offset:0 atIndex:4];
                    [enc setBuffer:eng->b_logits offset:0 atIndex:5];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                }
                [enc dispatchThreadgroups:MTLSizeMake((V_head + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            sik_encode_logit_penalties_tree(enc, eng, sampling, 1U, /*extra_mode=*/1U, eff_k, use_frspec, V_head);

            SikSampleParamsHost sp{};
            sp.M             = 1U;
            sp.vocab_size    = V_head;
            sp.use_token_map = use_frspec ? 1U : 0U;
            sp.compute_pmax  = 1U;
            if (is_greedy) {
                sp.greedy   = 1U;
                sp.inv_temp = 1.0f;
                sp.top_k    = 4U;
                sp.top_p    = 1.0f;
                [enc setComputePipelineState:eng->pso_sample_top4_stage1_tree];
                [enc setBuffer:eng->b_logits offset:0 atIndex:0];
                [enc setBuffer:eng->b_tree_meta offset:0 atIndex:1];
                [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:2];
                [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:3];
                [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:4];
                [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(128, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_tree_expand_top4];
                [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
                [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
                [enc setBuffer:eng->b_sample_stage_lse offset:0 atIndex:2];
                [enc setBuffer:(use_frspec ? eng->b_uma : eng->b_token_ids)
                        offset:(use_frspec ? static_cast<NSUInteger>(mtp.frspec.token_ids.offset) : 0U)
                       atIndex:3];
                [enc setBuffer:eng->b_tree_meta offset:0 atIndex:4];
                [enc setBuffer:eng->b_mtp_token_ids offset:0 atIndex:5];
                [enc setBytes:&sp length:sizeof(sp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                sp.greedy   = 0U;
                sp.inv_temp = 1.0f / std::max<float>(sampling->temperature, 1e-5f);
                sp.top_k    = (sampling->top_k > 0U && sampling->top_k <= 64U) ? sampling->top_k : 20U;
                sp.top_p    = sampling->top_p;
                const uint64_t seed = sampling->rng_seed ^ (eng->sample_step * 0xD1342543DE82EF95ULL);
                sp.rng_lo = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
                sp.rng_hi = static_cast<uint32_t>(seed >> 32ULL);
                eng->sample_step += 1ULL;

                [enc setComputePipelineState:eng->pso_sample_vocab_stage1_tree];
                [enc setBuffer:eng->b_logits offset:0 atIndex:0];
                [enc setBuffer:eng->b_tree_meta offset:0 atIndex:1];
                [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:2];
                [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:3];
                [enc setBytes:&sp length:sizeof(sp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:eng->pso_sample_vocab_stage2_tree];
                [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
                [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
                [enc setBuffer:(use_frspec ? eng->b_uma : eng->b_token_ids)
                        offset:(use_frspec ? static_cast<NSUInteger>(mtp.frspec.token_ids.offset) : 0U)
                       atIndex:2];
                [enc setBuffer:eng->b_tree_meta offset:0 atIndex:3];
                [enc setBuffer:eng->b_mtp_token_ids offset:0 atIndex:4];
                [enc setBuffer:eng->b_draft_top_ids offset:0 atIndex:5];
                [enc setBuffer:eng->b_draft_top_probs offset:0 atIndex:6];
                [enc setBuffer:eng->b_draft_keep_k offset:0 atIndex:7];
                [enc setBytes:&sp length:sizeof(sp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
    }
}

inline void sik_reset_spec_ctl(StrataInkling2GpuEngine* eng) {
    if (!eng) return;
    eng->spec_state        = StrataInkling2GpuEngine::kSpecActiveProbe;
    eng->spec_cooldown_rem = 0U;
    eng->spec_high_streak  = 0U;
    eng->spec_fused_rem    = 0U;
    eng->spec_locked_k       = 0U;
    eng->spec_low_streak     = 0U;
    eng->spec_cooldown_level = 0U;
    eng->ema_geom_num        = 2.00;
    eng->ema_geom_den        = 4.00;
    eng->ema_geom_p          = 0.50;
}

void sik_seed_adaptive_k_model(StrataInkling2GpuEngine* eng) {
    if (!eng) return;
    // Calibrate initial M=1..8 verify curve (ms) from live Apple M5 Max silicon measurements!
    const double base_gb = static_cast<double>(eng->hdr.active_bytes_per_tok) / 1e9;
    const bool is_fast   = (base_gb < 4.2);
    // Exact live M5 Max Stage-10+Bitplane verify-only times (M=1..8) for -fast (3.07 GB/tok) vs v1 FP16 (5.53 GB/tok).
    // Note: M=1 with Bitplane Exact Q8_0 LM Head runs at ~8.20 ms (vs ~8.60 ms full Q8_0), so M=1 -> M=2 has a +3.5 ms step!
    const double k0_fast_ms = (eng->bitplane_head_enabled && eng->bp_ready) ? 8.20 : 8.60;
    const double fast_curve[8] = {k0_fast_ms, 11.70, 13.40, 15.15, 16.95, 18.70, 20.65, 22.85};
    const double fp16_curve[8] = {15.5, 18.5, 21.2, 23.9, 26.6, 29.3, 32.0, 34.7};
    const double* curve = is_fast ? fast_curve : fp16_curve;
    for (uint32_t m = 0U; m < kMaxSpecBatch; ++m) {
        eng->ema_ver_ms[m] = curve[m];
    }
    const bool is_q4_mtp = (eng->hdr.mtp.enabled != 0U &&
                            eng->hdr.mtp.layers[0].q_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A);
    eng->ema_draft_ms = (eng->hdr.mtp.frspec.enabled != 0U) ? (is_q4_mtp ? 0.90 : 1.45) : 3.50;
    eng->ema_cal_num  = 4.80;
    eng->ema_cal_den  = 5.00;
    eng->ema_acc_cal  = 0.96;
    sik_reset_spec_ctl(eng);
}

} // namespace


/* ============================================================================
 * O(1) GPU Checkpoint Struct & Per-Slot Context Allocation Helpers
 * ============================================================================ */

struct StrataInkling2Checkpoint {
    StrataInkling2GpuEngine* eng                   = nullptr;
    uint32_t                 seq_pos               = 0U;
    uint32_t                 mtp_ctx_pos           = 0U;
    uint64_t                 sample_step           = 0ULL;
    uint32_t                 glob_stride_alloc     = 0U;
    uint32_t                 mtp_glob_stride_alloc = 0U;
    uint32_t                 kv_dtype              = STRATA_INKLING2_KV_DTYPE_F16;
    uint64_t                 glob_ckpt_layer_bytes = 0ULL;
    uint64_t                 glob_ckpt_q8_data     = 0ULL;

    id<MTLBuffer> b_swa_k_ckpt;
    id<MTLBuffer> b_swa_v_ckpt;
    id<MTLBuffer> b_glob_k_ckpt;
    id<MTLBuffer> b_glob_v_ckpt;
    id<MTLBuffer> b_sconv_ckpt;
    id<MTLBuffer> b_mtp_swa_k_ckpt;
    id<MTLBuffer> b_mtp_swa_v_ckpt;
    id<MTLBuffer> b_mtp_glob_k_ckpt;
    id<MTLBuffer> b_mtp_glob_v_ckpt;
    id<MTLBuffer> b_mtp_taps_last_ckpt;

    std::vector<uint32_t> ngram_hist;
    std::vector<uint32_t> pen_hist;
    bool                  mtp_synced = true;
};

namespace {

bool sik_alloc_slot_buffers(StrataInkling2GpuEngine* eng, StrataInkling2ContextSlot* slot, size_t* out_bytes = nullptr) {
    const StrataInkling2Config& cfg = eng->cfg;
    const uint32_t D      = cfg.hidden_size;
    const uint32_t n_loc  = cfg.num_local_visits;
    const uint32_t n_glob = cfg.num_global_visits;
    const bool     mtp_on = (eng->hdr.mtp.enabled != 0U);

    slot->b_swa_k_cache  = sik_alloc_private_aligned(eng->device, static_cast<size_t>(n_loc)  * eng->swa_kv_layer_stride);
    slot->b_swa_v_cache  = sik_alloc_private_aligned(eng->device, static_cast<size_t>(n_loc)  * eng->swa_kv_layer_stride);
    slot->b_glob_k_cache = sik_alloc_private_aligned(eng->device, static_cast<size_t>(n_glob) * eng->glob_kv_layer_stride);
    slot->b_glob_v_cache = sik_alloc_private_aligned(eng->device, static_cast<size_t>(n_glob) * eng->glob_kv_layer_stride);
    slot->b_sconv_state  = sik_alloc_private_aligned(eng->device, static_cast<size_t>(eng->sconv_total_bytes));
    if (!slot->b_swa_k_cache || !slot->b_swa_v_cache || !slot->b_glob_k_cache ||
        !slot->b_glob_v_cache || !slot->b_sconv_state) {
        return false;
    }

    NSMutableArray<id<MTLBuffer>>* arr = [NSMutableArray arrayWithObjects:
        slot->b_swa_k_cache, slot->b_swa_v_cache,
        slot->b_glob_k_cache, slot->b_glob_v_cache, nil];

    if (mtp_on) {
        uint32_t mtp_loc = 0U, mtp_glob = 0U;
        for (uint32_t l = 0U; l < eng->hdr.mtp.num_layers; ++l) {
            if (eng->hdr.mtp.layers[l].is_local_swa != 0U) ++mtp_loc;
            else                                            ++mtp_glob;
        }
        if (mtp_loc > 0U) {
            slot->b_mtp_swa_k_cache = sik_alloc_private_aligned(eng->device, static_cast<size_t>(mtp_loc) * eng->swa_kv_layer_stride);
            slot->b_mtp_swa_v_cache = sik_alloc_private_aligned(eng->device, static_cast<size_t>(mtp_loc) * eng->swa_kv_layer_stride);
            if (!slot->b_mtp_swa_k_cache || !slot->b_mtp_swa_v_cache) return false;
            [arr addObject:slot->b_mtp_swa_k_cache];
            [arr addObject:slot->b_mtp_swa_v_cache];
        }
        if (mtp_glob > 0U) {
            slot->b_mtp_glob_k_cache = sik_alloc_private_aligned(eng->device, static_cast<size_t>(mtp_glob) * eng->mtp_glob_kv_layer_stride);
            slot->b_mtp_glob_v_cache = sik_alloc_private_aligned(eng->device, static_cast<size_t>(mtp_glob) * eng->mtp_glob_kv_layer_stride);
            if (!slot->b_mtp_glob_k_cache || !slot->b_mtp_glob_v_cache) return false;
            [arr addObject:slot->b_mtp_glob_k_cache];
            [arr addObject:slot->b_mtp_glob_v_cache];
        }
        slot->b_mtp_taps_last = sik_alloc_private_aligned(
            eng->device, static_cast<size_t>(STRATA_INKLING2_MAX_MTP_TAPS) * D * sizeof(float));
        if (!slot->b_mtp_taps_last) return false;
        [arr addObject:slot->b_mtp_taps_last];
    }

    // Add all slot buffers into `eng->residency_set` (avoids Apple Metal's 32-residency-set-per-queue limit
    // when serving 16..64 concurrent context slots!).
    [arr addObject:slot->b_sconv_state];
    if (eng->residency_set) {
        if (@available(macOS 15.0, *)) {
            for (id<MTLBuffer> b in arr) [eng->residency_set addAllocation:b];
            [eng->residency_set commit];
        }
    }
    if (out_bytes) {
        size_t total = 0U;
        for (id<MTLBuffer> b in arr) total += [b length];
        *out_bytes = total;
    }
    return true;
}

void sik_sync_slot_to_engine(StrataInkling2GpuEngine* eng, StrataInkling2ContextSlot* slot) {
    eng->bound_slot         = slot;
    eng->b_swa_k_cache      = slot->b_swa_k_cache;
    eng->b_swa_v_cache      = slot->b_swa_v_cache;
    eng->b_glob_k_cache     = slot->b_glob_k_cache;
    eng->b_glob_v_cache     = slot->b_glob_v_cache;
    eng->b_sconv_state      = slot->b_sconv_state;
    eng->b_mtp_swa_k_cache  = slot->b_mtp_swa_k_cache;
    eng->b_mtp_swa_v_cache  = slot->b_mtp_swa_v_cache;
    eng->b_mtp_glob_k_cache = slot->b_mtp_glob_k_cache;
    eng->b_mtp_glob_v_cache = slot->b_mtp_glob_v_cache;
    eng->b_mtp_taps_last    = slot->b_mtp_taps_last;
    eng->seq_pos            = slot->seq_pos;
    eng->mtp_ctx_pos        = slot->mtp_ctx_pos;
    eng->sample_step        = slot->sample_step;
}

void sik_sync_engine_to_slot(StrataInkling2GpuEngine* eng) {
    if (!eng || !eng->bound_slot) return;
    StrataInkling2ContextSlot* slot = eng->bound_slot;
    slot->b_sconv_state = eng->b_sconv_state;
    slot->seq_pos       = eng->seq_pos;
    slot->mtp_ctx_pos   = eng->mtp_ctx_pos;
    slot->sample_step   = eng->sample_step;
}

void sik_free_slot_contents(StrataInkling2GpuEngine* eng, StrataInkling2ContextSlot* slot) {
    if (!slot) return;
    if (eng && eng->residency_set) {
        if (@available(macOS 15.0, *)) {
            id<MTLBuffer> bufs[10] = {
                slot->b_swa_k_cache, slot->b_swa_v_cache,
                slot->b_glob_k_cache, slot->b_glob_v_cache,
                slot->b_sconv_state,
                slot->b_mtp_swa_k_cache, slot->b_mtp_swa_v_cache,
                slot->b_mtp_glob_k_cache, slot->b_mtp_glob_v_cache,
                slot->b_mtp_taps_last
            };
            bool any = false;
            for (int i = 0; i < 10; ++i) {
                if (bufs[i]) { [eng->residency_set removeAllocation:bufs[i]]; any = true; }
            }
            if (any) [eng->residency_set commit];
        }
    }
    slot->rset               = nil;
    slot->sconv_rset         = nil;
    slot->b_swa_k_cache      = nil;
    slot->b_swa_v_cache      = nil;
    slot->b_glob_k_cache     = nil;
    slot->b_glob_v_cache     = nil;
    slot->b_sconv_state      = nil;
    slot->b_mtp_swa_k_cache  = nil;
    slot->b_mtp_swa_v_cache  = nil;
    slot->b_mtp_glob_k_cache = nil;
    slot->b_mtp_glob_v_cache = nil;
    slot->b_mtp_taps_last    = nil;
}

struct SikPsoSpec {
    __strong id<MTLComputePipelineState>* dst;
    const char*                           name;
    bool                                  use_const_m;
    uint32_t                              const_m_val;
};

} // namespace

/* ============================================================================
 * Public C API (`include/strata_inkling2.h`)
 * ============================================================================ */

extern "C" {

StrataInkling2GpuEngine* strata_inkling2_gpu_create(
    StrataInkling2Bundle* bundle,
    uint32_t              max_seq_len
) {
    return strata_inkling2_gpu_create_ex(bundle, max_seq_len, STRATA_INKLING2_KV_DTYPE_F16);
}

StrataInkling2GpuEngine* strata_inkling2_gpu_create_ex(
    const StrataInkling2Bundle* bundle,
    uint32_t                    max_seq_len,
    uint32_t                    kv_dtype
) {
    if (!bundle || !bundle->uma_base || bundle->hdr.magic != STRATA_INKLING2_MAGIC_U64) {
        std::fprintf(stderr, "[strata_inkling2_gpu] Invalid Inkling-2 bundle passed to gpu_create\n");
        return nullptr;
    }
    if (kv_dtype != STRATA_INKLING2_KV_DTYPE_F16 && kv_dtype != STRATA_INKLING2_KV_DTYPE_Q8_0) {
        std::fprintf(stderr, "[strata_inkling2_gpu] Unsupported kv_dtype=%u (expected F16=0 or Q8_0=1)\n", kv_dtype);
        return nullptr;
    }

    /* Engine geometry validation (Round 6 rules I2–I8, Round 7 rules I11–I12): enforce kernel invariants before
     * shader compile/alloc.
     *   I11: checkpoints save `kSwaCkptLen` window positions per local layer, so a larger sliding window (main
     *        stack or MTP drafter) would restore an incomplete window.
     *   I12: the prefill router computes whole 32-expert groups (`E / 32` threadgroups); a ragged expert count
     *        would leave the tail logits stale. */
    {
        const StrataInkling2Config& gc = bundle->hdr.cfg;
        if (gc.sliding_window_size == 0U || gc.sliding_window_size > kSwaCkptLen ||
            gc.num_key_value_heads == 0U || gc.num_attention_heads != 3U * gc.num_key_value_heads ||
            gc.head_dim != 128U || gc.d_rel != 16U || gc.sconv_kernel_size != 4U ||
            gc.hidden_size == 0U || (gc.hidden_size % 128U) != 0U ||
            gc.intermediate_size == 0U || (gc.intermediate_size % 128U) != 0U ||
            (gc.loop_num_passes > 1U && gc.loop_gate_rank != 64U) ||
            gc.num_experts == 0U || gc.num_experts > 128U || (gc.num_experts % 32U) != 0U ||
            gc.num_experts_per_tok == 0U || gc.num_experts_per_tok > 8U || gc.num_experts_per_tok > gc.num_experts ||
            (bundle->hdr.mtp.enabled != 0U &&
             (bundle->hdr.mtp.block_size != 8U ||
              bundle->hdr.mtp.sliding_window == 0U || bundle->hdr.mtp.sliding_window > kSwaCkptLen))) {
            std::fprintf(stderr, "[strata_inkling2_gpu] Unsupported Inkling-2 geometry in bundle header\n");
            return nullptr;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            std::fprintf(stderr, "[strata_inkling2_gpu] No Metal GPU device available\n");
            return nullptr;
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) {
            std::fprintf(stderr, "[strata_inkling2_gpu] Failed to create MTLCommandQueue\n");
            return nullptr;
        }

        // Load and stitch `strata_inkling2_kernels.metal` + `include/strata_inkling2.h` (embedded first for portability)
        const std::string msl_path = sik_resolve_sibling_path("strata_inkling2_kernels.metal");
        const std::string hdr_path = sik_resolve_sibling_path("../../include/strata_inkling2.h");
        std::string msl_src = strata_load_shader_asset("strata_inkling2_kernels.metal", "");
        if (msl_src.empty()) msl_src = sik_read_text_file(msl_path);
        std::string hdr_src = strata_load_shader_asset("strata_inkling2.h", "");
        if (hdr_src.empty()) hdr_src = sik_read_text_file(hdr_path);
        if (msl_src.empty() || hdr_src.empty()) {
            std::fprintf(stderr, "[strata_inkling2_gpu] Failed to read MSL (`%s`) or header (`%s`)\n",
                         msl_path.c_str(), hdr_path.c_str());
            return nullptr;
        }
        const std::string marker = "// @@STRATA_INKLING2_HEADER@@";
        const size_t pos = msl_src.find(marker);
        if (pos != std::string::npos) {
            msl_src.replace(pos, marker.size(), hdr_src);
        }

        NSString* msl_ns = [NSString stringWithUTF8String:msl_src.c_str()];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        bool compiled_m4 = false;
        if (@available(macOS 27.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_1;
            compiled_m4 = true;
        } else if (@available(macOS 26.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_0;
            compiled_m4 = true;
        } else {
            opts.languageVersion = MTLLanguageVersion3_1;
        }
        if (@available(macOS 15.0, *)) {
            // Safe math prevents M=1 vs M=2..8 FMA/add reassociation -> 0/200058 bit_diffs (100.000% bit-exact K>0==K=0!)
            // Fast FP functions keeps single-cycle HW exp/tanh/rsqrt with zero throughput penalty!
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
        }

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
        if (!lib && compiled_m4) {
            if (@available(macOS 26.0, *)) {
                std::fprintf(stderr, "[strata_inkling2_gpu] Metal 4.1 compile failed (%s), trying Metal 4.0...\n",
                             err ? [[err localizedDescription] UTF8String] : "unknown");
                err = nil;
                opts.languageVersion = MTLLanguageVersion4_0;
                lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
            }
        }
        if (!lib) {
            std::fprintf(stderr, "[strata_inkling2_gpu] Metal shader compilation failed:\n%s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown error");
            return nullptr;
        }
        if (err) {
            std::fprintf(stderr, "[strata_inkling2_gpu] Metal shader warnings:\n%s\n",
                         [[err localizedDescription] UTF8String]);
        }

        StrataInkling2GpuEngine* eng = new StrataInkling2GpuEngine();
        eng->device  = dev;
        eng->queue   = queue;
        eng->library = lib;
        eng->has_nax = compiled_m4;
        eng->bundle   = bundle;
        eng->hdr      = bundle->hdr;
        eng->cfg      = bundle->hdr.cfg;
        eng->kv_dtype = kv_dtype;

        const char* dev_utf8 = [[dev name] UTF8String];
        std::snprintf(eng->device_name, sizeof(eng->device_name), "%s", dev_utf8 ? dev_utf8 : "Apple Silicon");

        // Fast FR-Spec 3L DSpark (~1.05 ms/blk) beats suffix N-gram across code, refactors, & prose (zero boilerplate
        // mispredictions!); keep N-gram default-on only for slow full-vocab bundles, with env overrides in both directions.
        eng->ngram_draft_enabled = !(eng->hdr.mtp.enabled != 0U && eng->hdr.mtp.frspec.enabled != 0U);
        const char* force_ngram = std::getenv("STRATA_INKLING2_NGRAM");
        if (force_ngram && force_ngram[0] != '\0' && force_ngram[0] != '0') {
            eng->ngram_draft_enabled = true;
        }
        const char* no_ngram = std::getenv("STRATA_INKLING2_NO_NGRAM");
        if (no_ngram && no_ngram[0] != '\0' && no_ngram[0] != '0') {
            eng->ngram_draft_enabled = false;
        }
        const char* prof_env = std::getenv("STRATA_PROFILE_INKLING2");
        if (prof_env && prof_env[0] != '\0' && prof_env[0] != '0') {
            eng->prof_enabled = true;
            eng->prof_detail  = (prof_env[0] == '2');
            for (id<MTLCounterSet> cs in [dev counterSets]) {
                if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) {
                    MTLCounterSampleBufferDescriptor* cd = [MTLCounterSampleBufferDescriptor new];
                    cd.counterSet  = cs;
                    cd.sampleCount = 4096U;
                    cd.storageMode = MTLStorageModeShared;
                    NSError* cerr = nil;
                    eng->prof_csb = [dev newCounterSampleBufferWithDescriptor:cd error:&cerr];
                    break;
                }
            }
        }
        const char* int8_env = std::getenv("STRATA_INKLING2_INT8_PREFILL");
        if (int8_env && int8_env[0] != '\0' && int8_env[0] != '0') {
            eng->int8_prefill_enabled = 1U;
        }
        eng->bitplane_head_enabled = (eng->hdr.unembed.dtype == STRATA_INKLING2_DTYPE_Q8_0);
        const char* no_bp_env = std::getenv("STRATA_INKLING2_NO_BITPLANE");
        if (no_bp_env && no_bp_env[0] != '\0' && no_bp_env[0] != '0') {
            eng->bitplane_head_enabled = false;
        }
        eng->beam_tree_enabled = (eng->hdr.mtp.enabled != 0U);
        const char* no_tree_env = std::getenv("STRATA_INKLING2_NO_TREE");
        if (no_tree_env && no_tree_env[0] != '\0' && no_tree_env[0] != '0') {
            eng->beam_tree_enabled = false;
        }
        eng->beam_tree_width = 4U;

        const uint32_t req_seq = (max_seq_len > 0U)
            ? std::min<uint32_t>(max_seq_len, STRATA_INKLING2_MAX_SEQ_LEN_CAP)
            : STRATA_INKLING2_DEFAULT_MAX_SEQ_LEN;
        eng->max_seq_len   = std::max<uint32_t>(req_seq, 256U);
        // Pad global KV stride by +64 (multiple of 64) so KV_TILE=64 NAX FlashAttention tail-zeroing is always in-bounds!
        eng->max_seq_alloc = (eng->max_seq_len + 127U) & ~63U;

        // Build all Metal 4.1 Compute Pipelines via Persistent `MTLBinaryArchive` Cache (`dispatch_apply` parallel!)
        std::vector<SikPsoSpec> pso_specs;
        pso_specs.reserve(420U);
        auto req_pso = [&](__strong id<MTLComputePipelineState>& dst, NSString* nm) {
            pso_specs.push_back(SikPsoSpec{&dst, [nm UTF8String], false, 1U});
        };
        auto req_pso_m = [&](__strong id<MTLComputePipelineState>& dst, NSString* nm, uint32_t m_val) {
            pso_specs.push_back(SikPsoSpec{&dst, [nm UTF8String], true, m_val});
        };

        req_pso(eng->pso_zero_u32,         @"sik_zero_u32");
        req_pso(eng->pso_copy_u32,         @"sik_copy_u32");
        req_pso(eng->pso_f32_to_f16_pad,   @"sik_f32_to_f16_pad");
        req_pso(eng->pso_capture_tap,      @"sik_capture_tap");
        req_pso(eng->pso_swa_ring_copy,    @"sik_swa_ring_copy");

        req_pso(eng->pso_embed_rmsnorm,        @"sik_embed_rmsnorm");
        req_pso(eng->pso_embed_rmsnorm_q4a,    @"sik_embed_rmsnorm_q4a");
        req_pso(eng->pso_rmsnorm,              @"sik_rmsnorm");
        req_pso(eng->pso_rmsnorm3,             @"sik_rmsnorm3");
        req_pso(eng->pso_rmsnorm1_fast,        @"sik_rmsnorm1_fast");
        req_pso(eng->pso_rmsnorm3_fast,        @"sik_rmsnorm3_fast");
        req_pso(eng->pso_attn_res,             @"sik_attn_res");
        req_pso(eng->pso_attn_res_rmsnorm,     @"sik_attn_res_rmsnorm");
        req_pso(eng->pso_attn_res_rmsnorm_576, @"sik_attn_res_rmsnorm_576");
        req_pso(eng->pso_attn_res3_rmsnorm3,   @"sik_attn_res3_rmsnorm3");
        req_pso(eng->pso_attn_res3_rmsnorm3_576, @"sik_attn_res3_rmsnorm3_576");
        req_pso(eng->pso_cross_pass_attn_res4, @"sik_cross_pass_attn_res4");
        req_pso(eng->pso_cross_pass_batch12_576, @"sik_cross_pass_batch12_576");
        req_pso(eng->pso_attn_res_prep_left_576, @"sik_attn_res_prep_left_576");
        req_pso(eng->pso_pass_boundary_copy,   @"sik_pass_boundary_copy");
        req_pso(eng->pso_loop_pre_attn_prep3,  @"sik_loop_pre_attn_prep3");
        req_pso(eng->pso_attn_res_loop_prep1,  @"sik_attn_res_loop_prep1");

        req_pso(eng->pso_loop_ctrl_prep,      @"sik_loop_ctrl_prep");
        req_pso(eng->pso_loop_ctrl_prep3,     @"sik_loop_ctrl_prep3");
        req_pso(eng->pso_loop_ctrl_prep_i8,   @"sik_loop_ctrl_prep_i8");
        req_pso(eng->pso_loop_ctrl_apply,     @"sik_loop_ctrl_apply");
        req_pso(eng->pso_loop_ctrl_apply3,    @"sik_loop_ctrl_apply3");
        if (eng->has_nax) {
            req_pso(eng->pso_loop_ctrl_down_nax_f16,      @"sik_loop_ctrl_down_nax_f16");
            req_pso(eng->pso_loop_ctrl_down_nax_f16_m32,  @"sik_loop_ctrl_down_nax_f16_m32");
            req_pso(eng->pso_loop_ctrl_down_nax_q8,       @"sik_loop_ctrl_down_nax_q8");
            req_pso(eng->pso_loop_ctrl_down_nax_q8_m32,   @"sik_loop_ctrl_down_nax_q8_m32");
            req_pso(eng->pso_loop_ctrl_apply_nax_f16_m64, @"sik_loop_ctrl_apply_nax_f16_m64");
            req_pso(eng->pso_loop_ctrl_apply_nax_f16_m16, @"sik_loop_ctrl_apply_nax_f16_m16");
            req_pso(eng->pso_loop_ctrl_apply_nax_q8_m64,  @"sik_loop_ctrl_apply_nax_q8_m64");
            req_pso(eng->pso_loop_ctrl_apply_nax_q8_m16,  @"sik_loop_ctrl_apply_nax_q8_m16");
            req_pso(eng->pso_loop_ctrl_down_nax_q8_i8_m64,  @"sik_loop_ctrl_down_nax_q8_i8_m64");
            req_pso(eng->pso_loop_ctrl_down_nax_q8_i8_m32,  @"sik_loop_ctrl_down_nax_q8_i8_m32");
            req_pso(eng->pso_loop_ctrl_apply_nax_q8_i8_m64, @"sik_loop_ctrl_apply_nax_q8_i8_m64");
            req_pso(eng->pso_loop_ctrl_apply_nax_q8_i8_m32, @"sik_loop_ctrl_apply_nax_q8_i8_m32");
        }

        for (uint32_t m = 1U; m <= kMaxSpecBatch; ++m) {
            req_pso_m(eng->pso_loop_ctrl_down3_m[m - 1U],                 @"sik_loop_ctrl_down3",                 m);
            req_pso_m(eng->pso_loop_ctrl_down3_q8_m[m - 1U],              @"sik_loop_ctrl_down3_q8",              m);
            req_pso_m(eng->pso_loop_ctrl_down_right_batch12_q8_m[m - 1U], @"sik_loop_ctrl_down_right_batch12_q8", m);
            req_pso_m(eng->pso_loop_ctrl_down3_left_q8_m[m - 1U],         @"sik_loop_ctrl_down3_left_q8",         m);
            req_pso_m(eng->pso_loop_ctrl_down1_left_q8_m[m - 1U],         @"sik_loop_ctrl_down1_left_q8",         m);
            req_pso_m(eng->pso_loop_ctrl_apply_q8_m[m - 1U],              @"sik_loop_ctrl_apply_q8",              m);
            req_pso_m(eng->pso_loop_ctrl_apply3_q8_m[m - 1U],             @"sik_loop_ctrl_apply3_q8",             m);
            req_pso_m(eng->pso_f16_gemv_m[m - 1U],           @"sik_f16_gemv_batch",        m);
            req_pso_m(eng->pso_q8_gemv_m[m - 1U],            @"sik_q8_gemv_batch",         m);
            req_pso_m(eng->pso_q4a_gemv_m[m - 1U],           @"sik_q4a_gemv_batch",        m);
            req_pso_m(eng->pso_q4a_gemv_xsum_m[m - 1U],      @"sik_q4a_gemv_batch_xsum",   m);
            req_pso_m(eng->pso_q4a_wo_sconv_m[m - 1U],       @"sik_q4a_gemv_wo_sconv_xsum",m);
            req_pso_m(eng->pso_f16_row_tg128_m[m - 1U],      @"sik_f16_gemv_row_tg128",    m);
            req_pso_m(eng->pso_q8_row_tg128_m[m - 1U],       @"sik_q8_gemv_row_tg128",     m);
            req_pso_m(eng->pso_attn_qgrs_mixed_m[m - 1U],    @"sik_gemv_attn_qgrs_mixed",  m);
            req_pso_m(eng->pso_mtp_kv_ctx_mixed_m[m - 1U],   @"sik_gemv_mtp_kv_ctx_mixed", m);
            if (m <= 2U) {
                req_pso_m(eng->pso_attn_decode_splitkv_m[m - 1U],    @"sik_attn_decode_splitkv_shared",    m);
                req_pso_m(eng->pso_attn_decode_splitkv_q8_m[m - 1U], @"sik_attn_decode_splitkv_shared_q8", m);
            }
            req_pso_m(eng->pso_moe_router_f32_m[m - 1U],     @"sik_moe_router_f32",        m);
            req_pso_m(eng->pso_moe_router_f32_rms_m[m - 1U], @"sik_moe_router_f32_rms",    m);
        }
        req_pso(eng->pso_q4a_wo_sconv_m1,     @"sik_q4a_gemv_wo_sconv_m1");
        req_pso(eng->pso_mtp_qkv_mixed,       @"sik_gemv_mtp_qkv_mixed");
        req_pso(eng->pso_mtp_qkv_q4a,         @"sik_gemv_mtp_qkv_q4a");
        req_pso(eng->pso_f16_blk64_sum,       @"sik_f16_blk64_sum");
        req_pso(eng->pso_f16_small_n_prefill, @"sik_f16_small_n_prefill");
        req_pso(eng->pso_quant_f16_to_i8_row, @"sik_quant_f16_to_i8_row");
        req_pso(eng->pso_quant_f16_to_i8_row3,@"sik_quant_f16_to_i8_row3");

        if (eng->has_nax) {
            req_pso(eng->pso_f16_gemm_nax_64x128_sg4,@"sik_f16_gemm_nax_64x128_sg4");
            req_pso(eng->pso_f16_gemm_nax_64x64_sg4, @"sik_f16_gemm_nax_64x64_sg4");
            req_pso(eng->pso_f16_gemm_nax_32x64_sg2, @"sik_f16_gemm_nax_32x64_sg2");
            req_pso(eng->pso_f16_gemm_nax_16x32_sg2, @"sik_f16_gemm_nax_16x32_sg2");
            req_pso(eng->pso_q4a_gemm_nax_m128,      @"sik_q4a_gemm_nax_m128");
            req_pso(eng->pso_q4a_gemm_nax_m64,       @"sik_q4a_gemm_nax_m64");
            req_pso(eng->pso_q4a_gemm_nax_m32,       @"sik_q4a_gemm_nax_m32");
            req_pso(eng->pso_q4a_gemm_nax_m16,       @"sik_q4a_gemm_nax_m16");
            req_pso(eng->pso_q4a_gemm_nax_i8i4_m64,  @"sik_q4a_gemm_nax_i8i4_m64");
            req_pso(eng->pso_q4a_gemm_nax_i8i4_m32,  @"sik_q4a_gemm_nax_i8i4_m32");
            req_pso(eng->pso_q4a_gemm_nax_i8_m128,   @"sik_q4a_gemm_nax_i8_m128");
            req_pso(eng->pso_q4a_gemm_nax_i8_m64,    @"sik_q4a_gemm_nax_i8_m64");
            req_pso(eng->pso_q4a_gemm_nax_i8_m32,    @"sik_q4a_gemm_nax_i8_m32");
            req_pso(eng->pso_q8_gemm_nax_m128,       @"sik_q8_gemm_nax_m128");
            req_pso(eng->pso_q8_gemm_nax_m64,        @"sik_q8_gemm_nax_m64");
            req_pso(eng->pso_q8_gemm_nax_m32_n32,    @"sik_q8_gemm_nax_m32_n32");
            req_pso(eng->pso_q8_gemm_nax_m16_n32,    @"sik_q8_gemm_nax_m16_n32");
            req_pso(eng->pso_q8_gemm_nax_i8_m64,     @"sik_q8_gemm_nax_i8_m64");
            req_pso(eng->pso_q8_gemm_nax_i8_m32,     @"sik_q8_gemm_nax_i8_m32");
            req_pso(eng->pso_attn_prefill_nax,       @"sik_attn_prefill_nax");
            req_pso(eng->pso_attn_prefill_nax_q8,    @"sik_attn_prefill_nax_q8");
            req_pso(eng->pso_attn_decode_nax,        @"sik_attn_decode_nax");
            req_pso(eng->pso_attn_decode_nax_q8,     @"sik_attn_decode_nax_q8");
        }

        req_pso(eng->pso_sconv_static,             @"sik_sconv_static");
        req_pso(eng->pso_sconv_static_prefill,     @"sik_sconv_static_prefill");
        req_pso(eng->pso_sconv_save_tail,          @"sik_sconv_save_tail");
        req_pso(eng->pso_sconv_static_kv_prefill,  @"sik_sconv_static_kv_prefill");
        req_pso(eng->pso_sconv_dynamic_kv_prefill, @"sik_sconv_dynamic_kv_prefill");
        req_pso(eng->pso_sconv_save_tail_kv,       @"sik_sconv_save_tail_kv");
        req_pso(eng->pso_moe_combine_sconv,        @"sik_moe_combine_sconv");
        req_pso(eng->pso_sconv_kv_qkvr_prep,       @"sik_sconv_kv_qkvr_prep");
        req_pso(eng->pso_attn_qkvr_prep_prefill,   @"sik_attn_qkvr_prep_prefill");
        req_pso(eng->pso_attn_decode_reduce,          @"sik_attn_decode_reduce");
        req_pso(eng->pso_attn_decode_single_gqa3_sg8, @"sik_attn_decode_single_gqa3_sg8");
        req_pso(eng->pso_sconv_attn_single_gqa3_m1,   @"sik_sconv_attn_single_gqa3_m1");
        req_pso(eng->pso_sconv_attn_split_gqa3_m1,    @"sik_sconv_attn_split_gqa3_m1");
        req_pso(eng->pso_attn_decode_split_gqa3_reg,  @"sik_attn_decode_split_gqa3_reg");
        req_pso_m(eng->pso_attn_decode_q8_reg_m[0], @"sik_attn_decode_q8_reg", 1U);
        req_pso_m(eng->pso_attn_decode_q8_reg_m[1], @"sik_attn_decode_q8_reg", 2U);

        req_pso(eng->pso_moe_router_f32_m1,        @"sik_moe_router_f32_m1");
        req_pso(eng->pso_moe_router_f32_prefill,   @"sik_moe_router_f32_prefill");
        req_pso(eng->pso_moe_topk_sigmoid,         @"sik_moe_topk_sigmoid");
        req_pso(eng->pso_moe_bin_tokens_quad,      @"sik_moe_bin_tokens_quad");
        req_pso(eng->pso_moe_bin_tokens_grouped,   @"sik_moe_bin_tokens_grouped");
        req_pso(eng->pso_moe_gather_x_f16,         @"sik_moe_gather_x_f16");
        req_pso(eng->pso_moe_gather_x_i8,          @"sik_moe_gather_x_i8");
        req_pso(eng->pso_moe_topk_bin_small,          @"sik_moe_topk_bin_small");
        req_pso(eng->pso_moe_topk_m1,                 @"sik_moe_topk_m1");
        req_pso(eng->pso_moe_w13_nvfp4_m1_topk,       @"sik_moe_w13_nvfp4_m1_topk");
        req_pso(eng->pso_moe_w2_nvfp4_m1_fused_sconv, @"sik_moe_w2_nvfp4_m1_fused_sconv");
        const uint32_t moe_b_vals[4] = {1U, 2U, 4U, 8U};
        for (uint32_t bi = 0U; bi < 4U; ++bi) {
            req_pso_m(eng->pso_moe_nvfp4_w13_b[bi], @"sik_moe_grouped_w13_nvfp4", moe_b_vals[bi]);
            req_pso_m(eng->pso_moe_nvfp4_w2_b[bi],  @"sik_moe_grouped_w2_nvfp4",  moe_b_vals[bi]);
            req_pso_m(eng->pso_moe_mxfp8_w13_b[bi], @"sik_moe_grouped_w13_mxfp8", moe_b_vals[bi]);
            req_pso_m(eng->pso_moe_mxfp8_w2_b[bi],  @"sik_moe_grouped_w2_mxfp8",  moe_b_vals[bi]);
        }
        if (eng->has_nax) {
            for (uint32_t pass_i = 0U; pass_i < 2U; ++pass_i) {
                req_pso_m(eng->pso_moe_prefill_nvfp4_w13_m16_sg2[pass_i], @"sik_moe_prefill_w13_nvfp4_nax_m16_sg2", pass_i);
                req_pso_m(eng->pso_moe_prefill_nvfp4_w13_m32[pass_i],     @"sik_moe_prefill_w13_nvfp4_nax_m32",     pass_i);
                req_pso_m(eng->pso_moe_prefill_nvfp4_w13_m64[pass_i],     @"sik_moe_prefill_w13_nvfp4_nax_m64",     pass_i);
                req_pso_m(eng->pso_moe_prefill_nvfp4_w13_m128[pass_i],    @"sik_moe_prefill_w13_nvfp4_nax_m128",    pass_i);
                req_pso_m(eng->pso_moe_prefill_mxfp8_w13_m16_sg2[pass_i], @"sik_moe_prefill_w13_mxfp8_nax_m16_sg2", pass_i);
                req_pso_m(eng->pso_moe_prefill_mxfp8_w13_m32[pass_i],     @"sik_moe_prefill_w13_mxfp8_nax_m32",     pass_i);
                req_pso_m(eng->pso_moe_prefill_mxfp8_w13_m64[pass_i],     @"sik_moe_prefill_w13_mxfp8_nax_m64",     pass_i);
                req_pso_m(eng->pso_moe_prefill_mxfp8_w13_m128[pass_i],    @"sik_moe_prefill_w13_mxfp8_nax_m128",    pass_i);
                req_pso_m(eng->pso_moe_prefill_nvfp4_w13_i8_m16[pass_i],  @"sik_moe_prefill_w13_nvfp4_i8_m16",      pass_i);
                req_pso_m(eng->pso_moe_prefill_nvfp4_w13_i8_m32[pass_i],  @"sik_moe_prefill_w13_nvfp4_i8_m32",      pass_i);
                req_pso_m(eng->pso_moe_prefill_nvfp4_w13_i8_m64[pass_i],  @"sik_moe_prefill_w13_nvfp4_i8_m64",      pass_i);
                req_pso_m(eng->pso_moe_prefill_nvfp4_w13_i8_m128[pass_i], @"sik_moe_prefill_w13_nvfp4_i8_m128",     pass_i);
                req_pso_m(eng->pso_moe_prefill_mxfp8_w13_i8_m16[pass_i],  @"sik_moe_prefill_w13_mxfp8_i8_m16",      pass_i);
                req_pso_m(eng->pso_moe_prefill_mxfp8_w13_i8_m32[pass_i],  @"sik_moe_prefill_w13_mxfp8_i8_m32",      pass_i);
                req_pso_m(eng->pso_moe_prefill_mxfp8_w13_i8_m64[pass_i],  @"sik_moe_prefill_w13_mxfp8_i8_m64",      pass_i);
                req_pso_m(eng->pso_moe_prefill_mxfp8_w13_i8_m128[pass_i], @"sik_moe_prefill_w13_mxfp8_i8_m128",     pass_i);
            }
            req_pso(eng->pso_moe_prefill_nvfp4_w2_m16_sg2,  @"sik_moe_prefill_w2_nvfp4_nax_m16_sg2");
            req_pso(eng->pso_moe_prefill_nvfp4_w2_m32,      @"sik_moe_prefill_w2_nvfp4_nax_m32");
            req_pso(eng->pso_moe_prefill_nvfp4_w2_m64,      @"sik_moe_prefill_w2_nvfp4_nax_m64");
            req_pso(eng->pso_moe_prefill_nvfp4_w2_m128,     @"sik_moe_prefill_w2_nvfp4_nax_m128");
            req_pso(eng->pso_moe_prefill_mxfp8_w2_m16_sg2,  @"sik_moe_prefill_w2_mxfp8_nax_m16_sg2");
            req_pso(eng->pso_moe_prefill_mxfp8_w2_m32,      @"sik_moe_prefill_w2_mxfp8_nax_m32");
            req_pso(eng->pso_moe_prefill_mxfp8_w2_m64,      @"sik_moe_prefill_w2_mxfp8_nax_m64");
            req_pso(eng->pso_moe_prefill_mxfp8_w2_m128,     @"sik_moe_prefill_w2_mxfp8_nax_m128");
            req_pso(eng->pso_moe_prefill_nvfp4_w2_i8_m16,   @"sik_moe_prefill_w2_nvfp4_i8_m16");
            req_pso(eng->pso_moe_prefill_nvfp4_w2_i8_m32,   @"sik_moe_prefill_w2_nvfp4_i8_m32");
            req_pso(eng->pso_moe_prefill_nvfp4_w2_i8_m64,   @"sik_moe_prefill_w2_nvfp4_i8_m64");
            req_pso(eng->pso_moe_prefill_nvfp4_w2_i8_m128,  @"sik_moe_prefill_w2_nvfp4_i8_m128");
            req_pso(eng->pso_moe_prefill_mxfp8_w2_i8_m16,   @"sik_moe_prefill_w2_mxfp8_i8_m16");
            req_pso(eng->pso_moe_prefill_mxfp8_w2_i8_m32,   @"sik_moe_prefill_w2_mxfp8_i8_m32");
            req_pso(eng->pso_moe_prefill_mxfp8_w2_i8_m64,   @"sik_moe_prefill_w2_mxfp8_i8_m64");
            req_pso(eng->pso_moe_prefill_mxfp8_w2_i8_m128,  @"sik_moe_prefill_w2_mxfp8_i8_m128");
            req_pso(eng->pso_mtp_fold_head_nax,             @"sik_mtp_fold_head_nax");
        }
        req_pso(eng->pso_moe_combine,              @"sik_moe_combine");
        req_pso(eng->pso_moe_combine_f16,          @"sik_moe_combine_f16");

        req_pso(eng->pso_mtp_fuse_taps_norm,       @"sik_mtp_fuse_taps_norm");
        req_pso(eng->pso_mtp_ctx_kv_prep,          @"sik_mtp_ctx_kv_prep");
        req_pso(eng->pso_mtp_init_block,           @"sik_mtp_init_block");
        req_pso(eng->pso_mtp_init_block_q4a,       @"sik_mtp_init_block_q4a");
        req_pso(eng->pso_mtp_block_qk_prep,        @"sik_mtp_block_qk_prep");
        req_pso(eng->pso_mtp_block_attn_splitkv,   @"sik_mtp_block_attn_splitkv");
        req_pso(eng->pso_mtp_block_attn_reduce,    @"sik_mtp_block_attn_reduce");
        req_pso(eng->pso_mtp_add_rmsnorm,          @"sik_mtp_add_rmsnorm");
        req_pso(eng->pso_mtp_gate_up_silu_b8_mixed,@"sik_mtp_gate_up_silu_b8_mixed");
        req_pso(eng->pso_mtp_gate_up_silu_b8_q4a,  @"sik_mtp_gate_up_silu_b8_q4a");
        req_pso(eng->pso_mtp_rnn_prep,             @"sik_mtp_rnn_prep");
        req_pso(eng->pso_mtp_rnn_gate,             @"sik_mtp_rnn_gate");
        req_pso(eng->pso_mtp_fold_add_softcap,     @"sik_mtp_fold_add_softcap");
        req_pso(eng->pso_mtp_fold_add_softcap_q8,    @"sik_mtp_fold_add_softcap_q8");
        req_pso(eng->pso_mtp_fold_sample_greedy_q8,  @"sik_mtp_fold_sample_greedy_q8");

        req_pso(eng->pso_apply_logit_penalties,    @"sik_apply_logit_penalties");
        req_pso(eng->pso_sample_greedy_stage1,     @"sik_sample_greedy_stage1");
        req_pso(eng->pso_sample_greedy_stage2,     @"sik_sample_greedy_stage2");
        req_pso(eng->pso_sample_stage1,            @"sik_sample_vocab_stage1");
        req_pso(eng->pso_sample_stage2,            @"sik_sample_vocab_stage2");

        // 2-Stage Bitplane Exact Q8_0 LM Head PSOs
        req_pso(eng->pso_bp_repack_q8_to_hi4,      @"sik_bp_repack_q8_to_hi4");
        req_pso(eng->pso_bp_prep_x,                @"sik_bp_prep_x");
        req_pso(eng->pso_bp_stage1_hi4,            @"sik_bp_stage1_hi4");
        req_pso(eng->pso_bp_stage15_seed,          @"sik_bp_stage15_seed");
        req_pso(eng->pso_bp_filter_compact,        @"sik_bp_filter_compact");
        req_pso(eng->pso_bp_prep_indirect,         @"sik_bp_prep_indirect");
        req_pso(eng->pso_bp_stage2_exact_cand,     @"sik_bp_stage2_exact_cand");
        req_pso(eng->pso_bp_cand_sample_greedy,    @"sik_bp_cand_sample_greedy");
        req_pso(eng->pso_bp_cand_sample_stoch_s1,  @"sik_bp_cand_sample_stoch_s1");
        req_pso(eng->pso_bp_cand_sample_stoch_s2,  @"sik_bp_cand_sample_stoch_s2");

        // Dynamic W=4 Multi-Branch On-GPU Markov Beam Tree PSOs
        req_pso(eng->pso_tree_init,                        @"sik_tree_init");
        req_pso(eng->pso_mtp_rnn_prep_tree,                @"sik_mtp_rnn_prep_tree");
        req_pso(eng->pso_mtp_rnn_gate_tree,                @"sik_mtp_rnn_gate_tree");
        req_pso(eng->pso_mtp_fold_top4_q8,                 @"sik_mtp_fold_top4_q8");
        req_pso(eng->pso_mtp_fold_add_softcap_tree_q8,     @"sik_mtp_fold_add_softcap_tree_q8");
        req_pso(eng->pso_mtp_fold_add_softcap_tree_f16,    @"sik_mtp_fold_add_softcap_tree_f16");
        req_pso(eng->pso_sample_top4_stage1_tree,          @"sik_sample_top4_stage1_tree");
        req_pso(eng->pso_tree_expand_top4,                 @"sik_tree_expand_top4");
        req_pso(eng->pso_sample_vocab_stage1_tree,         @"sik_sample_vocab_stage1_tree");
        req_pso(eng->pso_sample_vocab_stage2_tree,         @"sik_sample_vocab_stage2_tree");
        req_pso(eng->pso_apply_logit_penalties_tree,       @"sik_apply_logit_penalties_tree");
        req_pso(eng->pso_sconv_kv_qkvr_prep_tree,          @"sik_sconv_kv_qkvr_prep_tree");
        req_pso(eng->pso_attn_decode_single_gqa3_sg8_tree, @"sik_attn_decode_single_gqa3_sg8_tree");
        req_pso(eng->pso_attn_decode_split_gqa3_reg_tree,  @"sik_attn_decode_split_gqa3_reg_tree");
        req_pso(eng->pso_attn_decode_q8_reg_tree,          @"sik_attn_decode_q8_reg_tree");
        req_pso(eng->pso_attn_decode_reduce_tree,          @"sik_attn_decode_reduce_tree");
        for (uint32_t m = 1U; m <= kMaxSpecBatch; ++m) {
            req_pso_m(eng->pso_q4a_wo_sconv_tree_m[m - 1U], @"sik_q4a_gemv_wo_sconv_tree", m);
        }
        req_pso(eng->pso_sconv_static_tree,                @"sik_sconv_static_tree");
        req_pso(eng->pso_moe_combine_sconv_tree,           @"sik_moe_combine_sconv_tree");
        req_pso(eng->pso_compact_kv_tree,                  @"sik_compact_kv_tree");
        req_pso(eng->pso_compact_taps_tree,                @"sik_compact_taps_tree");

        // True GPU-Batched Continuous-Batching Multi-Stream Decode PSOs (`B = 1..64`)
        req_pso(eng->pso_sconv_kv_qkvr_prep_batched,        @"sik_sconv_kv_qkvr_prep_batched");
        req_pso(eng->pso_attn_decode_single_gqa3_batched,   @"sik_attn_decode_single_gqa3_batched");
        req_pso(eng->pso_attn_decode_split_gqa3_batched,    @"sik_attn_decode_split_gqa3_batched");
        req_pso(eng->pso_attn_decode_q8_reg_batched,        @"sik_attn_decode_q8_reg_batched");
        req_pso(eng->pso_attn_decode_reduce_batched,        @"sik_attn_decode_reduce_batched");
        for (uint32_t m = 1U; m <= kMaxSpecBatch; ++m) {
            req_pso_m(eng->pso_q4a_wo_sconv_batched_m[m - 1U], @"sik_q4a_gemv_wo_sconv_batched", m);
        }
        req_pso(eng->pso_sconv_static_batched,              @"sik_sconv_static_batched");
        req_pso(eng->pso_moe_combine_sconv_batched,         @"sik_moe_combine_sconv_batched");
        req_pso(eng->pso_apply_logit_penalties_batched,     @"sik_apply_logit_penalties_batched");
        req_pso(eng->pso_sample_batched_stage1,             @"sik_sample_batched_stage1");
        req_pso(eng->pso_sample_batched_stage2,             @"sik_sample_batched_stage2");
        req_pso(eng->pso_attn_qgrs_mixed_2d,                @"sik_gemv_attn_qgrs_mixed_2d");
        req_pso(eng->pso_q4a_wo_sconv_batched_2d,           @"sik_q4a_gemv_wo_sconv_batched_2d");
        req_pso(eng->pso_q8_gemv_batch_2d,                  @"sik_q8_gemv_batch_2d");
        req_pso(eng->pso_moe_router_f32_2d,                 @"sik_moe_router_f32_2d");
        for (uint32_t m = 1U; m <= kMaxSpecBatch; ++m) {
            req_pso_m(eng->pso_loop_ctrl_down_right_batch12_q8_2d_m[m - 1U], @"sik_loop_ctrl_down_right_batch12_q8_2d", m);
            req_pso_m(eng->pso_loop_ctrl_down3_left_q8_2d_m[m - 1U],         @"sik_loop_ctrl_down3_left_q8_2d",         m);
        }
        req_pso(eng->pso_loop_ctrl_down1_left_q8_2d,        @"sik_loop_ctrl_down1_left_q8_2d");

        StrataMetalPsoCache* pso_cache = strata_metal_cache_open_with_opts(
            dev, lib, "inkling2", msl_src.data(), msl_src.size(), opts);
        if (!pso_cache) {
            std::fprintf(stderr, "[strata_inkling2_gpu] Failed to initialize StrataMetalPsoCache\n");
            strata_inkling2_gpu_destroy(eng);
            return nullptr;
        }
        std::atomic<bool> pso_ok{true};
        std::atomic<bool>* pso_ok_ptr = &pso_ok;
        const SikPsoSpec* specs_ptr = pso_specs.data();
        const size_t      num_specs = pso_specs.size();
        dispatch_apply(num_specs, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t idx) {
            @autoreleasepool {
                const SikPsoSpec& sp = specs_ptr[idx];
                NSError* p_err = nil;
                id<MTLComputePipelineState> pso = sp.use_const_m
                    ? strata_metal_cache_make_pso_const_u32(pso_cache, sp.name, 0U, sp.const_m_val, &p_err)
                    : strata_metal_cache_make_pso(pso_cache, sp.name, nil, &p_err);
                if (!pso) {
                    std::fprintf(stderr, "[strata_inkling2_gpu] PSO compile failed for `%s` (const_m=%u): %s\n",
                                 sp.name, sp.use_const_m ? sp.const_m_val : 0U,
                                 p_err ? [[p_err localizedDescription] UTF8String] : "unknown");
                    pso_ok_ptr->store(false, std::memory_order_relaxed);
                } else {
                    *(sp.dst) = pso;
                }
            }
        });
        strata_metal_cache_finish(pso_cache);
        eng->metal_cache_hit = pso_cache->cache_hit ? 1U : 0U;
        eng->pso_load_ms     = pso_cache->elapsed_ms;
        strata_metal_cache_destroy(pso_cache);

        if (!pso_ok.load(std::memory_order_relaxed)) {
            strata_inkling2_gpu_destroy(eng);
            return nullptr;
        }

        // Zero-Copy UMA Weight Buffer (`newBufferWithBytesNoCopy`)
        const size_t uma_page_aligned = sik_align_up_u64(bundle->uma_size, STRATA_INKLING2_PAGE_BYTES);
        eng->b_uma = [dev newBufferWithBytesNoCopy:const_cast<void*>(bundle->uma_base)
                                            length:uma_page_aligned
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
        if (!eng->b_uma) {
            std::fprintf(stderr, "[strata_inkling2_gpu] Failed to wrap zero-copy UMA buffer (%zu B)\n", uma_page_aligned);
            strata_inkling2_gpu_destroy(eng);
            return nullptr;
        }
        eng->zero_copy_mmap   = true;
        eng->uma_mapped_bytes = bundle->uma_size;

        // Compute Exact Strides (`float4` = 16 B/ch for SConv states!)
        const StrataInkling2Config& cfg = eng->cfg;
        const uint32_t D      = cfg.hidden_size;
        const uint32_t n_q    = cfg.num_attention_heads;
        const uint32_t n_kv   = cfg.num_key_value_heads;
        const uint32_t hd     = cfg.head_dim;
        const uint32_t q_dim  = n_q * hd;
        const uint32_t kv_dim = n_kv * hd;
        const uint32_t r_dim  = n_q * cfg.d_rel;
        const uint32_t inter  = cfg.intermediate_size;
        const uint32_t E      = cfg.num_experts;
        const uint32_t top_k  = cfg.num_experts_per_tok;
        const uint32_t V      = cfg.vocab_size_unpadded;

        eng->swa_kv_layer_stride      = static_cast<uint64_t>(n_kv) * kSwaRingLen * hd * sizeof(uint16_t);
        eng->mtp_glob_kv_layer_stride = static_cast<uint64_t>(n_kv) * eng->max_seq_alloc * hd * sizeof(uint16_t);
        if (eng->kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0) {
            strata_inkling2_kv_q8_layer_layout(n_kv, eng->max_seq_alloc, hd,
                                               &eng->glob_kv_q8_data_bytes, &eng->glob_kv_q8_scale_bytes);
            eng->glob_kv_layer_stride = eng->glob_kv_q8_data_bytes + eng->glob_kv_q8_scale_bytes;
        } else {
            eng->glob_kv_q8_data_bytes  = 0ULL;
            eng->glob_kv_q8_scale_bytes = 0ULL;
            eng->glob_kv_layer_stride   = eng->mtp_glob_kv_layer_stride;
        }

        eng->sconv_k_bytes      = sik_align_up_u64(static_cast<uint64_t>(kv_dim) * 16ULL, 256ULL);
        eng->sconv_v_bytes      = sik_align_up_u64(static_cast<uint64_t>(kv_dim) * 16ULL, 256ULL);
        eng->sconv_attn_bytes   = sik_align_up_u64(static_cast<uint64_t>(D)      * 16ULL, 256ULL);
        eng->sconv_mlp_bytes    = sik_align_up_u64(static_cast<uint64_t>(D)      * 16ULL, 256ULL);
        eng->sconv_visit_stride = eng->sconv_k_bytes + eng->sconv_v_bytes + eng->sconv_attn_bytes + eng->sconv_mlp_bytes;
        eng->sconv_total_bytes  = static_cast<uint64_t>(cfg.num_logical_layers) * eng->sconv_visit_stride;

        eng->block_slot_elems = static_cast<uint64_t>(kMaxPrefillChunk) * D;
        eng->block_slot_bytes = eng->block_slot_elems * sizeof(float);

        eng->swa_kv_bytes_allocated    = static_cast<uint64_t>(cfg.num_local_visits)  * eng->swa_kv_layer_stride * 2ULL;
        eng->global_kv_bytes_allocated = static_cast<uint64_t>(cfg.num_global_visits) * eng->glob_kv_layer_stride * 2ULL;

        // Allocate Default Context Slot & 7 SConv Rollback Snapshot Slots
        eng->default_slot = new StrataInkling2ContextSlot();
        eng->default_slot->eng = eng;
        eng->default_slot->is_default = true;
        if (!sik_alloc_slot_buffers(eng, eng->default_slot, &eng->slot_memory_bytes)) {
            strata_inkling2_gpu_destroy(eng);
            return nullptr;
        }
        sik_sync_slot_to_engine(eng, eng->default_slot);

        // Allocate Private GPU Scratch Workspaces (`kMaxPrefillChunk = 512`)
        NSMutableArray<id<MTLBuffer>>* res_bufs = [NSMutableArray arrayWithObject:eng->b_uma];
        uint64_t ws_bytes = 0ULL;
        auto alloc_priv = [&](__strong id<MTLBuffer>& dst, size_t sz) -> bool {
            dst = sik_alloc_private_aligned(dev, sz);
            if (!dst) return false;
            ws_bytes += [dst length];
            [res_bufs addObject:dst];
            return true;
        };
        auto alloc_sh = [&](__strong id<MTLBuffer>& dst, size_t sz) -> bool {
            dst = sik_alloc_shared_prefaulted(dev, sz);
            if (!dst) return false;
            ws_bytes += [dst length];
            [res_bufs addObject:dst];
            return true;
        };

        bool alloc_ok = true;
        alloc_ok &= alloc_priv(eng->b_sconv_snaps,
                               static_cast<size_t>(kMaxSpecSnapSlots) * static_cast<size_t>(eng->sconv_total_bytes));
        const size_t chunk_d_f32 = static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float);
        const size_t chunk_d_f16 = static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(uint16_t);
        alloc_ok &= alloc_priv(eng->b_cur,         chunk_d_f32);
        alloc_ok &= alloc_priv(eng->b_prelude_cur, chunk_d_f32);
        alloc_ok &= alloc_priv(eng->b_bank_last,   chunk_d_f32);
        alloc_ok &= alloc_priv(eng->b_blocks,      static_cast<size_t>(kMaxAttnResBlocks) * eng->block_slot_bytes);
        alloc_ok &= alloc_priv(eng->b_bank,        static_cast<size_t>(kMaxBankBlocks)    * eng->block_slot_bytes);
        for (uint32_t j = 0U; j < 3U; ++j) alloc_ok &= alloc_priv(eng->b_inputs[j],   chunk_d_f32);
        for (uint32_t j = 0U; j < 4U; ++j) alloc_ok &= alloc_priv(eng->b_messages[j], chunk_d_f32);
        alloc_ok &= alloc_priv(eng->b_ctrl_cat_f16, 3ULL * kMaxPrefillChunk * 2ULL * D * sizeof(uint16_t));
        alloc_ok &= alloc_priv(eng->b_ctrl_h_f32,   3ULL * kMaxPrefillChunk * cfg.loop_gate_rank * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_ctrl_cat_sc,  3ULL * kMaxPrefillChunk * sizeof(uint16_t));
        alloc_ok &= alloc_priv(eng->b_ctrl_h_i8,    3ULL * kMaxPrefillChunk * cfg.loop_gate_rank * sizeof(int8_t));
        alloc_ok &= alloc_priv(eng->b_ctrl_h_sc,    3ULL * kMaxPrefillChunk * sizeof(uint16_t));
        const uint32_t max_in_k = std::max<uint32_t>(D, q_dim);
        for (uint32_t j = 0U; j < 3U; ++j) {
            const uint32_t kj = (j == 0U) ? max_in_k : D;
            alloc_ok &= alloc_priv(eng->b_in_f16[j],      chunk_d_f16);
            alloc_ok &= alloc_priv(eng->b_in_i8[j],       static_cast<size_t>(kMaxPrefillChunk) * kj * sizeof(int8_t));
            alloc_ok &= alloc_priv(eng->b_in_xsc[j],      static_cast<size_t>(kMaxPrefillChunk) * sizeof(uint16_t));
            alloc_ok &= alloc_priv(eng->b_in_xsum_f16[j], static_cast<size_t>(kMaxPrefillChunk) * ((kj + 63U) / 64U) * sizeof(uint16_t));
        }
        for (uint32_t j = 0U; j < 3U; ++j) alloc_ok &= alloc_priv(eng->b_x_blk_sum[j], STRATA_INKLING2_MAX_BATCH_SIZE * 256ULL * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_wo_blk_sum,     STRATA_INKLING2_MAX_BATCH_SIZE * 256ULL * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_rms_sq,         STRATA_INKLING2_MAX_BATCH_SIZE * 3ULL * 256ULL * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_msg_bank12,     12ULL * 4ULL * kMaxSpecBatch * D * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_msg_norm12_f16, 12ULL * 4ULL * kMaxSpecBatch * D * sizeof(uint16_t));
        alloc_ok &= alloc_priv(eng->b_ctrl_h_msg12,   12ULL * 4ULL * kMaxSpecBatch * cfg.loop_gate_rank * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_x_f32,        chunk_d_f32);

        alloc_ok &= alloc_priv(eng->b_q_raw,          static_cast<size_t>(kMaxPrefillChunk) * q_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_q_f32,          static_cast<size_t>(kMaxPrefillChunk) * q_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_q_hm_f16,       static_cast<size_t>(n_q) * kMaxPrefillChunk * hd * sizeof(uint16_t));
        alloc_ok &= alloc_priv(eng->b_r_hm_f32,       static_cast<size_t>(n_q) * kMaxPrefillChunk * cfg.d_rel * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_gate_f32,       static_cast<size_t>(kMaxPrefillChunk) * q_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_r_raw,          static_cast<size_t>(kMaxPrefillChunk) * r_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_r_f32,          static_cast<size_t>(kMaxPrefillChunk) * r_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_sink_f32,       static_cast<size_t>(kMaxPrefillChunk) * n_q * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_k_raw,          static_cast<size_t>(kMaxPrefillChunk) * kv_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_v_raw,          static_cast<size_t>(kMaxPrefillChunk) * kv_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_k_conv,         static_cast<size_t>(kMaxPrefillChunk) * kv_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_v_conv,         static_cast<size_t>(kMaxPrefillChunk) * kv_dim * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_k_dyn,          static_cast<size_t>(kMaxPrefillChunk) * n_kv * cfg.sconv_kernel_size * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_v_dyn,          static_cast<size_t>(kMaxPrefillChunk) * n_kv * cfg.sconv_kernel_size * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_attn_gated_f16, static_cast<size_t>(kMaxPrefillChunk) * q_dim * sizeof(uint16_t));
        alloc_ok &= alloc_priv(eng->b_wo_out_f32,     chunk_d_f32);
        alloc_ok &= alloc_priv(eng->b_part_m,         static_cast<size_t>(kMaxSpecBatch) * n_q * kMaxDecodeSplits * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_part_l,         static_cast<size_t>(kMaxSpecBatch) * n_q * kMaxDecodeSplits * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_part_o,         static_cast<size_t>(kMaxSpecBatch) * n_q * kMaxDecodeSplits * hd * sizeof(float));

        alloc_ok &= alloc_priv(eng->b_router_logits,    static_cast<size_t>(kMaxPrefillChunk) * E * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_topk_idx,         static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(uint32_t));
        alloc_ok &= alloc_priv(eng->b_topk_w,           static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_moe_mid_f16,      (static_cast<size_t>(kMaxPrefillChunk) * top_k + 128ULL) * inter * sizeof(uint16_t));
        alloc_ok &= alloc_priv(eng->b_moe_mid_i8,       (static_cast<size_t>(kMaxPrefillChunk) * top_k + 128ULL) * inter * sizeof(int8_t));
        alloc_ok &= alloc_priv(eng->b_moe_mid_sc,       (static_cast<size_t>(kMaxPrefillChunk) * top_k + 128ULL) * sizeof(uint16_t));
        alloc_ok &= alloc_priv(eng->b_moe_x_sc,         (static_cast<size_t>(kMaxPrefillChunk) * top_k + 128ULL) * sizeof(uint16_t));
        /* `b_moe_down_part` is also aliased as the INT8 loop-controller `cat_i8 [3, M_padded, 2 * D]` scratch
         * (`sik_encode_loop_controllers`). That use does NOT scale with `top_k`, so size for it explicitly:
         * with `top_k <= 2` the two MoE terms alone are smaller than `3 * kMaxPrefillChunk * 2 * D` bytes. */
        const size_t moe_dp_bytes = std::max<size_t>({
            (static_cast<size_t>(kMaxPrefillChunk) * top_k + 128ULL) * D * sizeof(uint16_t),
            64ULL * top_k * D * sizeof(float),
            3ULL * kMaxPrefillChunk * 2ULL * D * sizeof(int8_t)});
        alloc_ok &= alloc_priv(eng->b_moe_down_part,    moe_dp_bytes);
        alloc_ok &= alloc_priv(eng->b_moe_out_f32,      chunk_d_f32);
        alloc_ok &= alloc_priv(eng->b_moe_sorted_pairs, static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(uint32_t));
        alloc_ok &= alloc_priv(eng->b_moe_work_tiles,   2048ULL * 16ULL);
        alloc_ok &= alloc_priv(eng->b_moe_num_tiles,    64ULL);
        alloc_ok &= alloc_priv(eng->b_moe_indirect_args,128ULL);

        alloc_ok &= alloc_priv(eng->b_mtp_taps_chunk,
                               static_cast<size_t>(kMaxPrefillChunk) * STRATA_INKLING2_MAX_MTP_TAPS * D * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_sample_stage_v,   static_cast<size_t>(kMaxSpecBatch) * 64ULL * 64ULL * sizeof(float));
        alloc_ok &= alloc_priv(eng->b_sample_stage_i,   static_cast<size_t>(kMaxSpecBatch) * 64ULL * 64ULL * sizeof(uint32_t));
        alloc_ok &= alloc_priv(eng->b_sample_stage_lse, static_cast<size_t>(kMaxSpecBatch) * 128ULL * sizeof(float));

        // MTP Drafter Scratch Buffers
        if (eng->hdr.mtp.enabled != 0U) {
            const StrataInkling2MtpDesc& mtp = eng->hdr.mtp;
            const uint32_t B     = mtp.block_size;
            const uint32_t rank  = (mtp.markov_rank > 0U) ? mtp.markov_rank : 256U;
            const uint32_t m_int = mtp.intermediate_size;

            alloc_ok &= alloc_priv(eng->b_mtp_base_logits,
                                   static_cast<size_t>(STRATA_INKLING2_MAX_SPEC_DRAFT_K) * V * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_mtp_res_f32,     static_cast<size_t>(B) * D * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_mtp_norm_f16,    static_cast<size_t>(B) * D * sizeof(uint16_t));
            alloc_ok &= alloc_priv(eng->b_mtp_norm_f32,    static_cast<size_t>(B) * D * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_mtp_draft_k,     static_cast<size_t>(B) * n_kv * hd * sizeof(uint16_t));
            alloc_ok &= alloc_priv(eng->b_mtp_draft_v,     static_cast<size_t>(B) * n_kv * hd * sizeof(uint16_t));
            alloc_ok &= alloc_priv(eng->b_mtp_ffn_mid_f16, static_cast<size_t>(B) * m_int * sizeof(uint16_t));
            alloc_ok &= alloc_priv(eng->b_mtp_f_rnn,       static_cast<size_t>(D) * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_mtp_rnn_state,   static_cast<size_t>(rank) * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_mtp_rnn_in_f16,  static_cast<size_t>(2U * rank + 2U * D) * sizeof(uint16_t));
            alloc_ok &= alloc_priv(eng->b_mtp_rnn_raw_f32, static_cast<size_t>(3U * rank) * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_mtp_latent_f16,  static_cast<size_t>(rank) * sizeof(uint16_t));
            alloc_ok &= alloc_sh(eng->b_tree_meta,         sizeof(SikTreeMetaHost));
            alloc_ok &= alloc_priv(eng->b_mtp_tree_states, static_cast<size_t>(8U) * rank * sizeof(float));

            alloc_ok &= alloc_sh(eng->b_mtp_inv_freq, static_cast<size_t>(hd / 2U) * sizeof(float));
            if (eng->b_mtp_inv_freq) {
                float* inv_f = reinterpret_cast<float*>([eng->b_mtp_inv_freq contents]);
                const double theta = (mtp.rope_theta > 0.0f) ? static_cast<double>(mtp.rope_theta) : 1000000.0;
                for (uint32_t i = 0U; i < hd / 2U; ++i) {
                    inv_f[i] = static_cast<float>(1.0 / std::pow(theta, static_cast<double>(2U * i) / static_cast<double>(hd)));
                }
            }
        }

        // Shared Host <-> GPU IO Buffers
        alloc_ok &= alloc_sh(eng->b_token_ids,        static_cast<size_t>(kMaxPrefillChunk) * sizeof(uint32_t));
        alloc_ok &= alloc_sh(eng->b_mtp_token_ids,    static_cast<size_t>(kMaxSpecBatch + 2U) * sizeof(uint32_t));
        alloc_ok &= alloc_sh(eng->b_logits,           static_cast<size_t>(kMaxSpecBatch) * V * sizeof(float));
        alloc_ok &= alloc_sh(eng->b_next_tokens,      static_cast<size_t>(kMaxSpecBatch) * sizeof(uint32_t));
        alloc_ok &= alloc_sh(eng->b_draft_pmax,       static_cast<size_t>(kMaxSpecBatch) * sizeof(float));
        alloc_ok &= alloc_sh(eng->b_sample_top_ids,   static_cast<size_t>(kMaxSpecBatch) * 64ULL * sizeof(uint32_t));
        alloc_ok &= alloc_sh(eng->b_sample_top_probs, static_cast<size_t>(kMaxSpecBatch) * 64ULL * sizeof(float));
        alloc_ok &= alloc_sh(eng->b_sample_keep_k,    static_cast<size_t>(kMaxSpecBatch) * sizeof(uint32_t));
        alloc_ok &= alloc_sh(eng->b_draft_top_ids,    static_cast<size_t>(kMaxSpecBatch) * 64ULL * sizeof(uint32_t));
        alloc_ok &= alloc_sh(eng->b_draft_top_probs,  static_cast<size_t>(kMaxSpecBatch) * 64ULL * sizeof(float));
        alloc_ok &= alloc_sh(eng->b_draft_keep_k,     static_cast<size_t>(kMaxSpecBatch) * sizeof(uint32_t));
        alloc_ok &= alloc_sh(eng->b_pen_ids,          static_cast<size_t>(V) * sizeof(uint32_t));
        alloc_ok &= alloc_sh(eng->b_pen_counts,       static_cast<size_t>(V) * sizeof(uint16_t));
        alloc_ok &= alloc_sh(eng->b_frspec_full_to_short, static_cast<size_t>(V) * sizeof(uint32_t));

        // 2-Stage Bitplane Exact Q8_0 LM Head Buffers (allocated only when unembed is Q8_0!)
        const bool can_init_bp = (eng->hdr.unembed.dtype == STRATA_INKLING2_DTYPE_Q8_0 &&
                                  (D & 63U) == 0U && D <= 2304U && eng->hdr.unembed.shape0 == V);
        if (can_init_bp) {
            alloc_ok &= alloc_priv(eng->b_bp_hi4,         static_cast<size_t>(V) * (D >> 1U));
            alloc_ok &= alloc_priv(eng->b_bp_rnorm4,      static_cast<size_t>(V) * 4U * sizeof(uint16_t));
            alloc_ok &= alloc_priv(eng->b_bp_ub,          static_cast<size_t>(V) * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_bp_xprep,       256ULL * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_bp_stage_v,     512ULL * sizeof(float));
            alloc_ok &= alloc_priv(eng->b_bp_stage_i,     512ULL * sizeof(uint32_t));
            alloc_ok &= alloc_priv(eng->b_bp_cand_ids,    32768ULL * sizeof(uint32_t));
            alloc_ok &= alloc_priv(eng->b_bp_cand_logits, 32768ULL * sizeof(float));
            alloc_ok &= alloc_sh(eng->b_bp_meta,          sizeof(SikBpMetaHost));
        }

        if (!alloc_ok) {
            std::fprintf(stderr, "[strata_inkling2_gpu] GPU buffer allocation failed\n");
            strata_inkling2_gpu_destroy(eng);
            return nullptr;
        }

        // Populate `full_to_short[V]` lookup table (0xFFFFFFFF for non-shortlisted tokens)
        {
            uint32_t* f2s = reinterpret_cast<uint32_t*>([eng->b_frspec_full_to_short contents]);
            std::fill(f2s, f2s + V, 0xFFFFFFFFU);
            if (eng->hdr.mtp.enabled != 0U && eng->hdr.mtp.frspec.enabled != 0U && eng->hdr.mtp.frspec.shortlist_size > 0U) {
                const uint32_t sl_n = eng->hdr.mtp.frspec.shortlist_size;
                const uint32_t* tok_ids = reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<const uint8_t*>(bundle->uma_base) + eng->hdr.mtp.frspec.token_ids.offset);
                for (uint32_t i = 0U; i < sl_n; ++i) {
                    const uint32_t t = tok_ids[i];
                    if (t < V) f2s[t] = i;
                }
            }
        }

        eng->pen_pos.assign(V, 0xFFFFFFFFU);
        eng->workspace_bytes = ws_bytes;
        if (eng->default_slot) {
            id<MTLBuffer> ds_bufs[10] = {
                eng->default_slot->b_swa_k_cache, eng->default_slot->b_swa_v_cache,
                eng->default_slot->b_glob_k_cache, eng->default_slot->b_glob_v_cache,
                eng->default_slot->b_sconv_state,
                eng->default_slot->b_mtp_swa_k_cache, eng->default_slot->b_mtp_swa_v_cache,
                eng->default_slot->b_mtp_glob_k_cache, eng->default_slot->b_mtp_glob_v_cache,
                eng->default_slot->b_mtp_taps_last
            };
            for (int i = 0; i < 10; ++i) {
                if (ds_bufs[i]) [res_bufs addObject:ds_bufs[i]];
            }
        }
        eng->residency_set   = sik_create_residency_set(dev, queue, res_bufs);

        // One-time GPU repack of Q8_0 unembed -> Plane 0 (`hi4` + exact grouped-4 residual norms, ~2 ms!)
        if (can_init_bp) {
            uint64_t uw = 0ULL, us = 0ULL, ub = 0ULL;
            sik_slice_sub_offsets(eng->hdr.unembed, uw, us, ub);
            SikBpParamsHost rp{};
            rp.V = V;
            rp.D = D;
            id<MTLCommandBuffer>         cb_bp  = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_bp = [cb_bp computeCommandEncoder];
            [enc_bp setComputePipelineState:eng->pso_bp_repack_q8_to_hi4];
            [enc_bp setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
            [enc_bp setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
            [enc_bp setBuffer:eng->b_bp_hi4 offset:0 atIndex:2];
            [enc_bp setBuffer:eng->b_bp_rnorm4 offset:0 atIndex:3];
            [enc_bp setBytes:&rp length:sizeof(rp) atIndex:4];
            [enc_bp dispatchThreadgroups:MTLSizeMake((V + 3U) / 4U, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc_bp endEncoding];
            [cb_bp commit];
            [cb_bp waitUntilCompleted];
            if ([cb_bp status] == MTLCommandBufferStatusCompleted) {
                eng->bp_ready = true;
            } else {
                eng->bp_ready = false;
                eng->bitplane_head_enabled = false;
            }
        }

        // Seed Bayes Adaptive-K Verify/Draft Timing Model
        sik_seed_adaptive_k_model(eng);

        // Skip startup 200K x 256 full-vocab `fold_weight` compute when 32K SLC FR-Spec is present!
        // (Lazily built only if user passes `--no-frspec` or auto-widen triggers.)
        if (eng->hdr.mtp.enabled != 0U && eng->hdr.mtp.frspec.enabled == 0U) {
            if (!sik_ensure_full_mtp_fold(eng)) {
                std::fprintf(stderr, "[strata_inkling2_gpu] Failed to compute full MTP folded head\n");
                strata_inkling2_gpu_destroy(eng);
                return nullptr;
            }
        }

        // Zero initial SConv state cleanly
        strata_inkling2_gpu_reset(eng);

        const auto t1 = std::chrono::steady_clock::now();
        eng->load_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return eng;
    }
}

void strata_inkling2_gpu_destroy(StrataInkling2GpuEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        sik_drain_pending_fix(eng);
        sik_detach_residency_set(eng->queue, eng->residency_set);
        eng->residency_set     = nil;
        eng->b_alt_wo          = nil;
        eng->b_sconv_snaps     = nil;
        eng->b_vis_stage       = nil;
        eng->vis_stage_bytes   = 0U;
        eng->b_bp_hi4          = nil;
        eng->b_bp_rnorm4       = nil;
        eng->b_bp_ub           = nil;
        eng->b_bp_xprep        = nil;
        eng->b_bp_stage_v      = nil;
        eng->b_bp_stage_i      = nil;
        eng->b_bp_cand_ids     = nil;
        eng->b_bp_cand_logits  = nil;
        eng->b_bp_meta         = nil;
        eng->b_tree_meta           = nil;
        eng->b_mtp_tree_states     = nil;
        eng->b_batch_slot_meta     = nil;
        eng->b_batch_next_tokens   = nil;
        eng->b_batch_pen_items     = nil;
        eng->b_batch_sample_items  = nil;
        eng->b_batch_pen_toks      = nil;
        eng->b_batch_pen_cnts      = nil;
        eng->b_batch_part_m        = nil;
        eng->b_batch_part_l        = nil;
        eng->b_batch_part_o        = nil;
        eng->b_batch_logits        = nil;
        eng->b_batch_sample_v        = nil;
        eng->b_batch_sample_i        = nil;
        eng->b_batch_msg_bank12      = nil;
        eng->b_batch_msg_norm12_f16  = nil;
        eng->b_batch_ctrl_h_msg12    = nil;
        eng->batch_pen_cap           = 0U;
        eng->batch_logits_rows     = 0U;
        eng->pen_cache_slot        = nullptr;
        if (eng->default_slot) {
            sik_free_slot_contents(eng, eng->default_slot);
            delete eng->default_slot;
            eng->default_slot = nullptr;
        }
        eng->bound_slot = nullptr;
        delete eng;
    }
}

void strata_inkling2_gpu_reset(StrataInkling2GpuEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        sik_drain_pending_fix(eng);
        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        sik_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_sconv_state, static_cast<size_t>(eng->sconv_total_bytes));
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        eng->seq_pos             = 0U;
        eng->mtp_ctx_pos         = 0U;
        eng->sample_step         = 0ULL;
        eng->frspec_widened      = false;
        eng->shortlist_miss_mask = 0U;
        sik_seed_adaptive_k_model(eng);
        if (eng->bound_slot) {
            eng->bound_slot->mtp_synced = true;
            sik_ngram_reset(eng->bound_slot);
            sik_sync_engine_to_slot(eng);
        }
        strata_inkling2_gpu_penalties_reset(eng);
    }
}

uint32_t strata_inkling2_gpu_seq_pos(const StrataInkling2GpuEngine* eng) {
    return eng ? eng->seq_pos : 0U;
}

void strata_inkling2_gpu_get_info(const StrataInkling2GpuEngine* eng, StrataInkling2GpuInfo* info_out) {
    if (!eng || !info_out) return;
    std::memset(info_out, 0, sizeof(*info_out));
    std::snprintf(info_out->device_name, sizeof(info_out->device_name), "%s", eng->device_name);
    info_out->zero_copy_mmap              = eng->zero_copy_mmap;
    info_out->load_time_ms                = eng->load_time_ms;
    info_out->max_seq_len                 = eng->max_seq_len;
    info_out->uma_mapped_bytes            = eng->uma_mapped_bytes;
    info_out->active_bytes_per_token      = eng->hdr.active_bytes_per_tok;
    info_out->mtp_active_bytes_per_block  = eng->hdr.mtp_active_bytes_per_blk;
    info_out->swa_kv_bytes_allocated      = eng->swa_kv_bytes_allocated;
    info_out->global_kv_bytes_allocated   = eng->global_kv_bytes_allocated;
    info_out->sconv_state_bytes           = eng->sconv_total_bytes;
    info_out->workspace_bytes             = eng->workspace_bytes;
    info_out->mtp_enabled                 = (eng->hdr.mtp.enabled != 0U);
    info_out->kv_dtype                    = eng->kv_dtype;
    info_out->int8_prefill_enabled        = eng->int8_prefill_enabled;
    info_out->metal_cache_hit             = eng->metal_cache_hit;
    info_out->pso_load_ms                 = eng->pso_load_ms;
    info_out->bitplane_head_enabled       = (eng->bitplane_head_enabled && eng->bp_ready) ? 1U : 0U;
    info_out->beam_tree_enabled           = (eng->beam_tree_enabled && eng->hdr.mtp.enabled != 0U) ? 1U : 0U;
    info_out->beam_tree_width             = (info_out->beam_tree_enabled != 0U) ? eng->beam_tree_width : 1U;
}

void strata_inkling2_gpu_get_counters(const StrataInkling2GpuEngine* eng, StrataInkling2PerfCounters* out) {
    if (!eng || !out) return;
    *out = eng->counters;
}

void strata_inkling2_gpu_reset_counters(StrataInkling2GpuEngine* eng) {
    if (!eng) return;
    std::memset(&eng->counters, 0, sizeof(eng->counters));
}

void strata_inkling2_gpu_set_adaptive_k(StrataInkling2GpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->adaptive_k_enabled = enable;
    sik_reset_spec_ctl(eng);
}

bool strata_inkling2_gpu_get_adaptive_k(const StrataInkling2GpuEngine* eng) {
    return eng ? eng->adaptive_k_enabled : false;
}

void strata_inkling2_gpu_set_frspec(StrataInkling2GpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->frspec_enabled = enable;
    if (enable) {
        eng->frspec_widened      = false;
        eng->shortlist_miss_mask = 0U;
    }
}

bool strata_inkling2_gpu_get_frspec(const StrataInkling2GpuEngine* eng) {
    return eng ? eng->frspec_enabled : false;
}

void strata_inkling2_gpu_set_bitplane_head(StrataInkling2GpuEngine* eng, bool enable) {
    if (!eng) return;
    const bool next = enable && eng->bp_ready;
    if (eng->bitplane_head_enabled != next) {
        eng->bitplane_head_enabled = next;
        sik_seed_adaptive_k_model(eng);
    }
}

bool strata_inkling2_gpu_get_bitplane_head(const StrataInkling2GpuEngine* eng) {
    return eng ? (eng->bitplane_head_enabled && eng->bp_ready) : false;
}

void strata_inkling2_gpu_set_beam_tree(StrataInkling2GpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->beam_tree_enabled = enable && (eng->hdr.mtp.enabled != 0U);
    sik_reset_spec_ctl(eng);
}

bool strata_inkling2_gpu_get_beam_tree(const StrataInkling2GpuEngine* eng) {
    return eng ? (eng->beam_tree_enabled && eng->hdr.mtp.enabled != 0U) : false;
}

void strata_inkling2_gpu_set_int8_prefill(StrataInkling2GpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->int8_prefill_enabled = enable ? 1U : 0U;
}

int strata_inkling2_gpu_load_alt_wo_bundle(StrataInkling2GpuEngine* eng, const char* alt_bundle_dir) {
    if (!eng || !alt_bundle_dir || alt_bundle_dir[0] == '\0') return -1;
    @autoreleasepool {
        sik_drain_pending_fix(eng);
        const std::string uma_path = std::string(alt_bundle_dir) + "/" STRATA_INKLING2_UMA_FILENAME;
        const int fd = ::open(uma_path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[strata_inkling2_gpu] ERROR: cannot open alt-wo bundle `%s` (%s)\n",
                         uma_path.c_str(), std::strerror(errno));
            return -1;
        }
#if defined(F_NOCACHE)
        (void)::fcntl(fd, F_NOCACHE, 1);
#endif
        struct stat st{};
        if (::fstat(fd, &st) != 0 || static_cast<uint64_t>(st.st_size) < sizeof(StrataInkling2Header)) {
            std::fprintf(stderr, "[strata_inkling2_gpu] ERROR: invalid alt-wo file `%s`\n", uma_path.c_str());
            ::close(fd);
            return -1;
        }
        const uint64_t file_bytes = static_cast<uint64_t>(st.st_size);

        auto pread_all = [](int f, void* dst, size_t n, uint64_t off) -> bool {
            uint8_t* p = static_cast<uint8_t*>(dst);
            size_t done = 0U;
            while (done < n) {
                const ssize_t r = ::pread(f, p + done, n - done, static_cast<off_t>(off + done));
                if (r <= 0) {
                    if (r < 0 && errno == EINTR) continue;
                    return false;
                }
                done += static_cast<size_t>(r);
            }
            return true;
        };

        auto alt_hdr_ptr = std::make_unique<StrataInkling2Header>();
        if (!pread_all(fd, alt_hdr_ptr.get(), sizeof(StrataInkling2Header), 0ULL)) {
            std::fprintf(stderr, "[strata_inkling2_gpu] ERROR: failed reading alt-wo header from `%s`\n",
                         uma_path.c_str());
            ::close(fd);
            return -1;
        }
        const StrataInkling2Header& ah = *alt_hdr_ptr;
        if (ah.magic != STRATA_INKLING2_MAGIC_U64 ||
            ah.cfg.num_phys_layers != eng->cfg.num_phys_layers ||
            ah.cfg.hidden_size != eng->cfg.hidden_size) {
            std::fprintf(stderr, "[strata_inkling2_gpu] ERROR: incompatible alt-wo header in `%s`\n",
                         uma_path.c_str());
            ::close(fd);
            return -1;
        }

        const uint8_t* stock_uma = static_cast<const uint8_t*>([eng->b_uma contents]);
        std::vector<uint8_t> tmp;
        bool     diff_layer[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
        uint64_t dst_off[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
        uint64_t total_packed = 0ULL;
        uint32_t diff_cnt     = 0U;

        for (uint32_t l = 0U; l < eng->cfg.num_phys_layers; ++l) {
            const StrataInkling2TensorSlice& swo = eng->hdr.layers[l].wo;
            const StrataInkling2TensorSlice& awo = ah.layers[l].wo;
            if (awo.dtype != swo.dtype || awo.shape0 != swo.shape0 ||
                awo.shape1 != swo.shape1 || awo.bytes != swo.bytes ||
                awo.offset + awo.bytes > file_bytes) {
                std::fprintf(stderr,
                             "[strata_inkling2_gpu] ERROR: layer %u wo layout mismatch between stock and `%s` "
                             "(stock dtype=%u %ux%u %lluB vs alt dtype=%u %ux%u %lluB)\n",
                             l, uma_path.c_str(),
                             swo.dtype, swo.shape0, swo.shape1, static_cast<unsigned long long>(swo.bytes),
                             awo.dtype, awo.shape0, awo.shape1, static_cast<unsigned long long>(awo.bytes));
                ::close(fd);
                return -1;
            }
            if (tmp.size() < awo.bytes) tmp.resize(static_cast<size_t>(awo.bytes));
            if (!pread_all(fd, tmp.data(), static_cast<size_t>(awo.bytes), awo.offset)) {
                std::fprintf(stderr, "[strata_inkling2_gpu] ERROR: failed reading layer %u wo from `%s`\n",
                             l, uma_path.c_str());
                ::close(fd);
                return -1;
            }
            if (!stock_uma || std::memcmp(tmp.data(), stock_uma + swo.offset, static_cast<size_t>(awo.bytes)) != 0) {
                diff_layer[l] = true;
                dst_off[l]    = total_packed;
                total_packed  = strata_inkling2_align256(total_packed + awo.bytes);
                ++diff_cnt;
            }
        }

        if (diff_cnt == 0U || total_packed == 0ULL) {
            std::fprintf(stderr, "[strata_inkling2_gpu] ERROR: alt-wo bundle `%s` has identical wo weights to stock\n",
                         uma_path.c_str());
            ::close(fd);
            return -1;
        }

        const uint64_t alloc_bytes =
            (total_packed + (STRATA_INKLING2_PAGE_BYTES - 1ULL)) & ~(STRATA_INKLING2_PAGE_BYTES - 1ULL);
        id<MTLBuffer> new_buf = [eng->device newBufferWithLength:static_cast<NSUInteger>(alloc_bytes)
                                                         options:MTLResourceStorageModeShared];
        if (!new_buf) {
            std::fprintf(stderr, "[strata_inkling2_gpu] ERROR: failed allocating %.2f MiB Metal buffer for alt-wo\n",
                         static_cast<double>(alloc_bytes) / (1024.0 * 1024.0));
            ::close(fd);
            return -1;
        }
        uint8_t* dst_base = static_cast<uint8_t*>([new_buf contents]);
        std::memset(dst_base, 0, static_cast<size_t>(alloc_bytes));

        StrataInkling2TensorSlice new_slices[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
        for (uint32_t l = 0U; l < eng->cfg.num_phys_layers; ++l) {
            if (!diff_layer[l]) continue;
            const StrataInkling2TensorSlice& awo = ah.layers[l].wo;
            if (!pread_all(fd, dst_base + dst_off[l], static_cast<size_t>(awo.bytes), awo.offset)) {
                ::close(fd);
                return -1;
            }
            new_slices[l]        = awo;
            new_slices[l].offset = dst_off[l];
        }
        ::close(fd);

        if (eng->b_alt_wo && eng->residency_set) {
            [eng->residency_set removeAllocation:eng->b_alt_wo];
        }
        eng->b_alt_wo      = new_buf;
        eng->alt_wo_bytes  = alloc_bytes;
        eng->alt_wo_layers = diff_cnt;
        eng->alt_wo_active = false;
        for (uint32_t l = 0U; l < STRATA_INKLING2_MAX_PHYS_LAYERS; ++l) {
            eng->alt_wo_has[l]   = diff_layer[l];
            eng->alt_wo_slice[l] = new_slices[l];
        }
        if (eng->residency_set) {
            [eng->residency_set addAllocation:eng->b_alt_wo];
            [eng->residency_set commit];
        }

        std::printf("[strata_inkling2_gpu] Loaded alt-wo overlay: %u/%u layers (%.2f MiB UMA) from `%s`\n",
                    diff_cnt, eng->cfg.num_phys_layers,
                    static_cast<double>(alloc_bytes) / (1024.0 * 1024.0),
                    alt_bundle_dir);
        return 0;
    }
}

void strata_inkling2_gpu_set_alt_wo(StrataInkling2GpuEngine* eng, bool enable) {
    if (!eng) return;
    const bool target = enable && (eng->b_alt_wo != nil) && (eng->alt_wo_layers > 0U);
    if (eng->alt_wo_active != target) {
        sik_drain_pending_fix(eng);
        eng->alt_wo_active = target;
        sik_reset_spec_ctl(eng);
    }
}

bool strata_inkling2_gpu_has_alt_wo(const StrataInkling2GpuEngine* eng) {
    return eng && (eng->b_alt_wo != nil) && (eng->alt_wo_layers > 0U);
}

uint64_t strata_inkling2_gpu_alt_wo_bytes(const StrataInkling2GpuEngine* eng) {
    return (eng && eng->b_alt_wo != nil) ? eng->alt_wo_bytes : 0ULL;
}

void strata_inkling2_gpu_penalties_reset(StrataInkling2GpuEngine* eng) {
    if (!eng) return;
    if (eng->bound_slot) eng->bound_slot->pen_hist.clear();
    if (eng->pen_cache_slot == eng->bound_slot || eng->bound_slot == nullptr) {
        sik_penalties_clear_active(eng);
        eng->pen_cache_slot   = eng->bound_slot;
        eng->pen_window_start = 0U;
        eng->pen_window_end   = 0U;
    }
}

void strata_inkling2_gpu_penalties_record(
    StrataInkling2GpuEngine* eng,
    const uint32_t*          tokens,
    uint32_t                 num_tokens
) {
    if (!eng || !eng->bound_slot || !tokens || num_tokens == 0U) return;
    eng->bound_slot->pen_hist.insert(eng->bound_slot->pen_hist.end(), tokens, tokens + num_tokens);
}

int strata_inkling2_gpu_prefill(
    StrataInkling2GpuEngine*            eng,
    const uint32_t*                     token_ids,
    uint32_t                            num_tokens,
    const StrataInkling2SamplingParams* sampling,
    uint32_t*                           next_token_out,
    float*                              last_logits_out
) {
    return strata_inkling2_gpu_prefill_with_images(
        eng, token_ids, num_tokens, nullptr, 0U, sampling, next_token_out, last_logits_out);
}

int strata_inkling2_gpu_prefill_with_images(
    StrataInkling2GpuEngine*            eng,
    const uint32_t*                     token_ids,
    uint32_t                            num_tokens,
    const StrataInkling2ImageSpan*      spans,
    uint32_t                            num_spans,
    const StrataInkling2SamplingParams* sampling,
    uint32_t*                           next_token_out,
    float*                              last_logits_out
) {
    if (!eng || !token_ids || num_tokens == 0U) return -1;
    if (num_spans > 0U && !spans) return -1;
    if (eng->seq_pos + num_tokens > eng->max_seq_len) {
        std::fprintf(stderr, "[strata_inkling2_gpu] Prefill overflow: pos=%u + n=%u > max=%u\n",
                     eng->seq_pos, num_tokens, eng->max_seq_len);
        return -1;
    }
    for (uint32_t s = 0U; s < num_spans; ++s) {
        if (spans[s].num_tokens == 0U) continue;
        if (!spans[s].embeddings_f32 ||
            spans[s].token_offset > num_tokens ||
            spans[s].num_tokens > num_tokens - spans[s].token_offset) {
            std::fprintf(stderr, "[strata_inkling2_gpu] Invalid image span %u (offset=%u, len=%u, total=%u)\n",
                         s, spans[s].token_offset, spans[s].num_tokens, num_tokens);
            return -1;
        }
    }

    @autoreleasepool {
        if (!sik_drain_pending_fix(eng)) return -1;

        const bool     from_zero = (eng->seq_pos == 0U);
        const bool     mtp_on    = (eng->hdr.mtp.enabled != 0U);
        const uint32_t D      = eng->cfg.hidden_size;
        const uint32_t V      = eng->cfg.vocab_size_unpadded;
        const bool     use_bp = sik_can_use_bitplane(eng, sampling, next_token_out, last_logits_out);

        std::vector<SikChunkImageCopy> chunk_vis;
        struct PendingSpanSlice {
            uint32_t span_idx;
            uint32_t src_row;
            uint32_t dst_row;
            uint32_t num_rows;
        };
        std::vector<PendingSpanSlice> slices;

        const uint32_t max_chunk = eng->has_nax ? kMaxPrefillChunk : kMaxSpecBatch;
        uint32_t offset = 0U;
        while (offset < num_tokens) {
            const uint32_t chunk    = std::min<uint32_t>(max_chunk, num_tokens - offset);
            const bool     is_last  = (offset + chunk == num_tokens);
            const uint32_t M_padded = (eng->has_nax && chunk > kMaxSpecBatch) ? ((chunk + 63U) & ~63U) : chunk;

            std::memcpy([eng->b_token_ids contents], token_ids + offset, static_cast<size_t>(chunk) * sizeof(uint32_t));

            // Prepare chunk-local visual embedding staging (zero work & zero allocation when `num_spans == 0`!)
            chunk_vis.clear();
            if (num_spans > 0U) {
                slices.clear();
                uint32_t stage_rows = 0U;
                const uint32_t chunk_end = offset + chunk;
                for (uint32_t s = 0U; s < num_spans; ++s) {
                    if (spans[s].num_tokens == 0U) continue;
                    const uint32_t s_beg = spans[s].token_offset;
                    const uint32_t s_end = s_beg + spans[s].num_tokens;
                    const uint32_t ov_beg = std::max<uint32_t>(offset, s_beg);
                    const uint32_t ov_end = std::min<uint32_t>(chunk_end, s_end);
                    if (ov_beg < ov_end) {
                        const uint32_t nr = ov_end - ov_beg;
                        slices.push_back(PendingSpanSlice{s, ov_beg - s_beg, ov_beg - offset, nr});
                        stage_rows += nr;
                    }
                }
                if (stage_rows > 0U) {
                    const size_t need_bytes = static_cast<size_t>(stage_rows) * D * sizeof(float);
                    if (!eng->b_vis_stage || eng->vis_stage_bytes < need_bytes) {
                        id<MTLBuffer> old_buf = eng->b_vis_stage;
                        id<MTLBuffer> new_buf = sik_alloc_shared_prefaulted(eng->device, need_bytes);
                        if (!new_buf) {
                            std::fprintf(stderr, "[strata_inkling2_gpu] Failed allocating %.2f MiB vision staging buffer\n",
                                         static_cast<double>(need_bytes) / (1024.0 * 1024.0));
                            return -1;
                        }
                        if (@available(macOS 15.0, *)) {
                            if (eng->residency_set) {
                                if (old_buf) [eng->residency_set removeAllocation:old_buf];
                                [eng->residency_set addAllocation:new_buf];
                                [eng->residency_set commit];
                            }
                        }
                        eng->b_vis_stage     = new_buf;
                        eng->vis_stage_bytes = [new_buf length];
                    }
                    uint8_t* dst_base = static_cast<uint8_t*>([eng->b_vis_stage contents]);
                    NSUInteger cur_off = 0U;
                    for (const PendingSpanSlice& sl : slices) {
                        const size_t nbytes = static_cast<size_t>(sl.num_rows) * D * sizeof(float);
                        const float* src_f  = spans[sl.span_idx].embeddings_f32 + static_cast<size_t>(sl.src_row) * D;
                        std::memcpy(dst_base + cur_off, src_f, nbytes);
                        chunk_vis.push_back(SikChunkImageCopy{sl.dst_row, sl.num_rows, cur_off});
                        cur_off += static_cast<NSUInteger>(nbytes);
                    }
                }
            }

            id<MTLCommandBuffer>         cb  = [eng->queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            eng->prof_active = eng->prof_enabled && (offset == 0U || is_last);
            if (eng->prof_active) std::memset(eng->prof_ms, 0, sizeof(eng->prof_ms));

            const bool use_bp_last = (is_last && use_bp);
            sik_encode_backbone_forward(
                cb, enc, eng, eng->b_token_ids, eng->seq_pos, chunk, M_padded,
                /*save_snap_mask=*/0U, /*force_gemv=*/false,
                /*capture_mtp_taps=*/mtp_on,
                /*run_unembed_all=*/false,
                /*run_unembed_last=*/is_last && !use_bp_last,
                /*run_final_norm=*/is_last,
                /*use_tree=*/false,
                chunk_vis.empty() ? nullptr : chunk_vis.data(),
                static_cast<uint32_t>(chunk_vis.size())
            );

            if (is_last && next_token_out) {
                eng->sample_step = static_cast<uint64_t>(eng->seq_pos + chunk) << 8U;
            }
            if (is_last && use_bp_last) {
                // Run bitplane BEFORE `sik_encode_mtp_ctx_update` (which reuses `b_in_f16[0]`)!
                const NSUInteger last_x_off = static_cast<NSUInteger>(chunk - 1U) * D * sizeof(uint16_t);
                sik_encode_bitplane_unembed_sample(
                    cb, enc, eng, eng->b_in_f16[0], last_x_off, sampling, eng->b_next_tokens, 0U);
            }

            if (mtp_on) {
                sik_encode_mtp_ctx_update(enc, eng, eng->b_mtp_taps_chunk, eng->seq_pos, chunk, false);
                if (is_last) {
                    const size_t tap_bytes = static_cast<size_t>(STRATA_INKLING2_MAX_MTP_TAPS) * D * sizeof(float);
                    const NSUInteger src_off = static_cast<NSUInteger>(chunk - 1U) * tap_bytes;
                    sik_encode_copy_buffer(enc, eng->pso_copy_u32,
                                           eng->b_mtp_taps_chunk, src_off,
                                           eng->b_mtp_taps_last, 0, tap_bytes);
                }
            }

            if (is_last && !use_bp_last) {
                if (next_token_out && !last_logits_out) {
                    sik_encode_logit_penalties(enc, eng, sampling, 1U, nil, 0U, 0U, false, 0U);
                }
                if (next_token_out) {
                    sik_encode_sample_batch(
                        enc, eng, sampling, 1U,
                        eng->b_next_tokens, 0,
                        nil, 0, nil, 0, nil, 0, nil, 0,
                        false, 0U, 0ULL
                    );
                }
            }

            if (eng->prof_active) {
                sik_prof_mark(eng, cb, enc, 11U);
                sik_prof_dump(eng, is_last ? "PREFILL_LAST" : "PREFILL_CHUNK0", chunk, eng->seq_pos);
                eng->prof_active = false;
            }

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb status] == MTLCommandBufferStatusError) {
                NSError* err = [cb error];
                std::fprintf(stderr, "[strata_inkling2_gpu] Prefill GPU error: %s\n",
                             err ? [[err localizedDescription] UTF8String] : "unknown");
                return -1;
            }

            eng->seq_pos += chunk;
            if (mtp_on) eng->mtp_ctx_pos = eng->seq_pos;
            offset += chunk;
        }

        if (eng->bound_slot) {
            if (from_zero) eng->bound_slot->mtp_synced = true;
            sik_ngram_append(eng->bound_slot, token_ids, num_tokens);
            sik_sync_engine_to_slot(eng);
        }

        if (last_logits_out) {
            std::memcpy(last_logits_out, [eng->b_logits contents], static_cast<size_t>(V) * sizeof(float));
        }
        if (next_token_out) {
            *next_token_out = reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents])[0];
        }
        return 0;
    }
}

int strata_inkling2_gpu_decode_step(
    StrataInkling2GpuEngine*            eng,
    uint32_t                            token_id,
    const StrataInkling2SamplingParams* sampling,
    uint32_t*                           next_token_out,
    float*                              logits_out
) {
    if (!eng) return -1;
    if (eng->seq_pos + 1U > eng->max_seq_len) return -1;

    @autoreleasepool {
        if (!sik_drain_pending_fix(eng)) return -1;

        // Record incoming completion token BEFORE applying penalties (100% bit-exact K>0 == K=0!)
        strata_inkling2_gpu_penalties_record(eng, &token_id, 1U);

        const bool     mtp_on = (eng->hdr.mtp.enabled != 0U);
        const uint32_t D      = eng->cfg.hidden_size;
        const uint32_t V      = eng->cfg.vocab_size_unpadded;

        reinterpret_cast<uint32_t*>([eng->b_token_ids contents])[0] = token_id;

        const bool measure_unprof = eng->prof_enabled && (eng->counters.decode_steps == 3ULL);
        const auto t_step0        = std::chrono::steady_clock::now();

        eng->prof_active     = eng->prof_enabled && (eng->counters.decode_steps == 4ULL);
        eng->prof_use_csb    = eng->prof_active && (eng->prof_csb != nil);
        eng->prof_pass_count = 0U;
        if (eng->prof_active) {
            std::memset(eng->prof_ms, 0, sizeof(eng->prof_ms));
            std::memset(eng->prof_cnt, 0, sizeof(eng->prof_cnt));
        }

        id<MTLCommandBuffer>         cb       = [eng->queue commandBufferWithUnretainedReferences];
        id<MTLCommandBuffer>         cb_first = cb;
        id<MTLComputeCommandEncoder> enc      = eng->prof_use_csb ? sik_prof_new_enc(eng, cb) : [cb computeCommandEncoder];

        // Update MTP context on the 1st decode step after prefill/reset (supports prefill->1x decode->spec),
        // then skip on continuous K=0 streaming (saves 14 dispatches & ~0.35 ms/tok!).
        const bool slot_mtp_ok = (!eng->bound_slot || eng->bound_slot->mtp_synced);
        const bool update_mtp  = mtp_on && slot_mtp_ok && (eng->counters.decode_steps == 0ULL);
        const bool use_bp      = sik_can_use_bitplane(eng, sampling, next_token_out, logits_out);

        sik_encode_backbone_forward(
            cb, enc, eng, eng->b_token_ids, eng->seq_pos, 1U, 1U,
            /*save_snap_mask=*/0U, /*force_gemv=*/true,
            /*capture_mtp_taps=*/update_mtp,
            /*run_unembed_all=*/false,
            /*run_unembed_last=*/!use_bp,
            /*run_final_norm=*/true
        );

        if (next_token_out) {
            eng->sample_step = static_cast<uint64_t>(eng->seq_pos + 1U) << 8U;
        }
        if (use_bp) {
            // 2-Stage Bitplane Exact Q8_0 LM Head + Sample (runs BEFORE `update_mtp` which reuses `b_in_f16[0]`!)
            sik_encode_bitplane_unembed_sample(
                cb, enc, eng, eng->b_in_f16[0], 0U, sampling, eng->b_next_tokens, 0U);
            if (update_mtp) {
                sik_encode_mtp_ctx_update(enc, eng, eng->b_mtp_taps_chunk, eng->seq_pos, 1U, true);
                const size_t tap_bytes = static_cast<size_t>(STRATA_INKLING2_MAX_MTP_TAPS) * D * sizeof(float);
                sik_encode_copy_buffer(enc, eng->pso_copy_u32,
                                       eng->b_mtp_taps_chunk, 0,
                                       eng->b_mtp_taps_last, 0, tap_bytes);
            }
        } else {
            sik_prof_mark_detail(eng, cb, enc, 26U);
            if (update_mtp) {
                sik_encode_mtp_ctx_update(enc, eng, eng->b_mtp_taps_chunk, eng->seq_pos, 1U, true);
                const size_t tap_bytes = static_cast<size_t>(STRATA_INKLING2_MAX_MTP_TAPS) * D * sizeof(float);
                sik_encode_copy_buffer(enc, eng->pso_copy_u32,
                                       eng->b_mtp_taps_chunk, 0,
                                       eng->b_mtp_taps_last, 0, tap_bytes);
            }
            if (!logits_out) {
                sik_encode_logit_penalties(enc, eng, sampling, 1U, nil, 0U, 0U, false, 0U);
            }
            if (next_token_out) {
                sik_encode_sample_batch(
                    enc, eng, sampling, 1U,
                    eng->b_next_tokens, 0,
                    nil, 0, nil, 0, nil, 0, nil, 0,
                    false, 0U, 0ULL
                );
            }
        }

        if (eng->prof_active && !eng->prof_use_csb) {
            sik_prof_mark(eng, cb, enc, 11U);
            sik_prof_dump(eng, "DECODE_K0", 1U, eng->seq_pos);
            eng->prof_active = false;
        } else if (eng->prof_active && eng->prof_use_csb) {
            if (eng->prof_pass_count < 2040U) {
                eng->prof_pass_bkt[eng->prof_pass_count++] = eng->prof_detail ? 27U : 11U;
            }
        }

        [enc endEncoding];
        const auto t_enc_done = std::chrono::steady_clock::now();
        [cb commit];
        [cb waitUntilCompleted];
        const auto t_wait_done = std::chrono::steady_clock::now();
        if (measure_unprof) {
            const double enc_ms  = std::chrono::duration<double, std::milli>(t_enc_done - t_step0).count();
            const double wall_ms = std::chrono::duration<double, std::milli>(t_wait_done - t_step0).count();
            const double gpu_ms  = ([cb GPUEndTime] - [cb_first GPUStartTime]) * 1000.0;
            std::fprintf(stderr, "[STRATA_PROFILE_INKLING2] UNPROFILED_1CB (M=1, pos=%u): wall=%.2f ms | gpu=%.2f ms (%.1f tok/s) | enc=%.2f ms | submit+wait=%.2f ms\n",
                         eng->seq_pos, wall_ms, gpu_ms, (gpu_ms > 0.0) ? (1000.0 / gpu_ms) : 0.0, enc_ms, wall_ms - gpu_ms);
            std::fflush(stderr);
        }
        if (eng->prof_active && eng->prof_use_csb) {
            sik_prof_resolve_csb(eng, cb, "DECODE_K0", 1U, eng->seq_pos);
            eng->prof_active  = false;
            eng->prof_use_csb = false;
        }
        if ([cb_first status] == MTLCommandBufferStatusError || [cb status] == MTLCommandBufferStatusError) {
            NSError* err = ([cb status] == MTLCommandBufferStatusError) ? [cb error] : [cb_first error];
            std::fprintf(stderr, "[strata_inkling2_gpu] Decode GPU error: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return -1;
        }

        eng->seq_pos += 1U;
        if (mtp_on && slot_mtp_ok) eng->mtp_ctx_pos = eng->seq_pos;
        if (eng->bound_slot) {
            sik_ngram_append(eng->bound_slot, &token_id, 1U);
            sik_sync_engine_to_slot(eng);
        }

        eng->counters.decode_steps            += 1ULL;
        const uint64_t tok_dram = use_bp
            ? (eng->hdr.active_bytes_per_tok - (eng->hdr.unembed.bytes / 2ULL))
            : eng->hdr.active_bytes_per_tok;
        eng->counters.physical_dram_bytes_read += tok_dram;

        if (logits_out) {
            std::memcpy(logits_out, [eng->b_logits contents], static_cast<size_t>(V) * sizeof(float));
        }
        if (next_token_out) {
            *next_token_out = reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents])[0];
        }
        return 0;
    }
}

int strata_inkling2_gpu_spec_decode_step(
    StrataInkling2GpuEngine*            eng,
    uint32_t                            pending_token_id,
    uint32_t                            draft_k,
    const StrataInkling2SamplingParams* sampling,
    uint32_t*                           emitted_tokens_out,
    uint32_t*                           num_emitted_out,
    uint32_t*                           num_accepted_out
) {
    if (!eng || !emitted_tokens_out || !num_emitted_out || !num_accepted_out) return -1;

    const uint32_t max_rem = (eng->seq_pos < eng->max_seq_len) ? (eng->max_seq_len - eng->seq_pos) : 0U;
    if (max_rem == 0U) return -1;

    const uint32_t eff_max_k = std::min<uint32_t>(
        std::min<uint32_t>(draft_k, STRATA_INKLING2_MAX_SPEC_DRAFT_K),
        (max_rem > 1U) ? (max_rem - 1U) : 0U
    );

    if (eff_max_k == 0U || eng->hdr.mtp.enabled == 0U ||
        (eng->bound_slot && !eng->bound_slot->mtp_synced)) {
        uint32_t next_tok = 0U;
        const int rc = strata_inkling2_gpu_decode_step(eng, pending_token_id, sampling, &next_tok, nullptr);
        if (rc != 0) return rc;
        emitted_tokens_out[0] = next_tok;
        *num_emitted_out      = 1U;
        *num_accepted_out     = 0U;
        return 0;
    }

    @autoreleasepool {
        // Record incoming pending token BEFORE applying penalties (100% bit-exact K>0 == K=0!)
        strata_inkling2_gpu_penalties_record(eng, &pending_token_id, 1U);

        const bool     is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
        const uint32_t D         = eng->cfg.hidden_size;
        const uint32_t V         = eng->cfg.vocab_size_unpadded;

        // 1. Check 0.00 ms Longest-Match-First 5-Gram -> 4-Gram Suffix Drafter (greedy only)
        const bool in_cooldown = (eng->adaptive_k_enabled &&
                                  eng->spec_state == StrataInkling2GpuEngine::kSpecCooldownK0 &&
                                  eng->spec_cooldown_rem > 0U);
        uint32_t ngram_drafts[STRATA_INKLING2_MAX_SPEC_DRAFT_K] = {};
        bool     used_ngram = false;
        uint32_t eff_k      = eff_max_k;
        if (is_greedy && eng->ngram_draft_enabled && eng->bound_slot) {
            const uint32_t nk = sik_ngram_propose(eng->bound_slot, pending_token_id, eff_max_k, ngram_drafts, in_cooldown);
            if (nk > 0U) {
                used_ngram = true;
                eff_k      = nk;
            }
        }

        const bool used_cooldown_k0 = (!used_ngram && in_cooldown);

        // 2. Select 48 MiB SLC FR-Spec vs Full-Vocab Folded Head (lazy full fold if needed)
        const bool use_frspec = (!used_ngram && !used_cooldown_k0) &&
                                (eng->hdr.mtp.frspec.enabled != 0U) &&
                                eng->frspec_enabled &&
                                !eng->frspec_widened;
        if (!used_ngram && !used_cooldown_k0 && !use_frspec) {
            if (!sik_drain_pending_fix(eng)) return -1;
            if (!sik_ensure_full_mtp_fold(eng)) return -1;
        }

        uint32_t verify_tokens[kMaxSpecBatch] = {};
        verify_tokens[0] = pending_token_id;

        id<MTLBuffer> verify_tok_buf    = eng->b_mtp_token_ids;
        uint32_t      M                 = 1U + eff_k;
        bool          used_fused_cb     = false;
        bool          used_adaptive_cb  = false;
        bool          used_high_fused   = false;
        bool          drafted_tree      = false;
        bool          verify_tree       = false;
        double        measured_draft_ms = 0.0;
        double        step_draft_est    = 0.0;
        double        probe_best_tps    = 0.0;
        double        probe_k0_tps      = 0.0;
        double        draft_pmax_saved[STRATA_INKLING2_MAX_SPEC_DRAFT_K]{};
        const bool    want_tree         = eng->beam_tree_enabled && !used_ngram && !used_cooldown_k0 &&
                                          (eng->b_tree_meta != nil) && (is_greedy || !eng->adaptive_k_enabled);
        // Accurate M=1 K=0 reference time for this sampling mode: Bitplane Head speeds up greedy/narrow top-K
        // by ~0.35 ms, while wide stochastic sampling (`top_k > 4`) runs full Q8_0 unembed (~+0.35 ms).
        uint32_t      k0_bp_probe       = 0U;
        const bool    k0_uses_bp        = sik_can_use_bitplane(eng, sampling, &k0_bp_probe, nullptr);
        const double  eff_k0_ms         = std::max(0.5, eng->ema_ver_ms[0] +
            ((eng->bitplane_head_enabled && eng->bp_ready && !k0_uses_bp) ? 0.35 : 0.0));

        eng->sample_step = (static_cast<uint64_t>(eng->seq_pos + 1U) << 8U) | 0x80ULL;
        if (used_ngram) {
            // 0.00 ms 10-gram verbatim copy -> copy directly to `b_token_ids` and run single `cb_ver`!
            uint32_t* t_ptr = reinterpret_cast<uint32_t*>([eng->b_token_ids contents]);
            t_ptr[0] = pending_token_id;
            for (uint32_t k = 0U; k < eff_k; ++k) {
                t_ptr[k + 1U]        = ngram_drafts[k];
                verify_tokens[k + 1U] = ngram_drafts[k];
            }
            verify_tok_buf = eng->b_token_ids;
        } else if (used_cooldown_k0) {
            // State A (Low-Acceptance SpecCtl Cooldown): skip `cb_draft` completely (`0` draft cost + `0` CPU sync!),
            // run pure `K = 0` (`M = 1`) verify WITH MTP tap capture + async `cb_fix` so DSpark state stays 100% exact!
            reinterpret_cast<uint32_t*>([eng->b_token_ids contents])[0] = pending_token_id;
            verify_tok_buf = eng->b_token_ids;
            eff_k          = 0U;
            M              = 1U;
        } else if (eng->adaptive_k_enabled &&
                   eng->spec_state == StrataInkling2GpuEngine::kSpecHighConfFused) {
            // State B (High-Confidence Zero-Sync Geometric-MLE Fused CB):
            // Runs `cb_draft + cb_ver` in ONE single CB (0 CPU-GPU sync stalls!) while choosing optimal `K in 2..eff_max_k`
            // directly from unbiased observed acceptance `ema_geom_p` & live `ema_ver_ms[0..7]`!
            const double p_acc = std::clamp<double>(eng->ema_geom_p, 0.0, 0.99);
            uint32_t best_k    = std::min<uint32_t>(eff_max_k, 2U);
            double   best_tps  = 0.0;
            double   cum_p     = 1.0;
            double   exp_toks  = 1.0;
            for (uint32_t k = 1U; k <= eff_max_k; ++k) {
                cum_p    *= p_acc;
                exp_toks += cum_p;
                if (k >= 2U) {
                    const double d_ms = eng->ema_draft_ms * (0.45 + 0.55 * static_cast<double>(k) / 7.0);
                    const double tps  = exp_toks / std::max(d_ms + eng->ema_ver_ms[k], 0.1);
                    if (tps > best_tps) {
                        best_tps = tps;
                        best_k   = k;
                    }
                }
            }
            if (eng->spec_locked_k > best_k) {
                best_k = std::min<uint32_t>(eff_max_k, eng->spec_locked_k);
            }
            eff_k          = best_k;
            M              = 1U + eff_k;
            step_draft_est = eng->ema_draft_ms * (0.45 + 0.55 * static_cast<double>(eff_k) / 7.0);
            reinterpret_cast<uint32_t*>([eng->b_mtp_token_ids contents])[0] = pending_token_id;
            verify_tok_buf  = eng->b_mtp_token_ids;
            used_fused_cb   = true;
            used_high_fused = true;
            // When acceptance is already high (greedy >= 0.82 or locked at full `eff_max_k`, OR stochastic
            // >= 0.52 / locked >= 4 where linear Leviathan beats multi-branch tree overhead on Code Creative!),
            // bypass tree draft/verify so high-conf streams run zero-overhead linear depth-`eff_k`!
            const bool skip_tree_high_conf = is_greedy
                ? (eng->ema_geom_p >= 0.82 || eng->spec_locked_k >= eff_max_k)
                : (eng->ema_geom_p >= 0.52 || eng->spec_locked_k >= 4U);
            if (want_tree && eff_k >= 2U && !skip_tree_high_conf) {
                drafted_tree = true;
                verify_tree  = true;
            }
        } else if (eng->adaptive_k_enabled) {
            // State C (Active Probe/Adapt w/ Per-Node Probabilities + Confidence-Adaptive Beam Tree!)
            reinterpret_cast<uint32_t*>([eng->b_mtp_token_ids contents])[0] = pending_token_id;
            // On high-acceptance stochastic code (`ema_geom_p >= 0.52`), linear draft+verify avoids stochastic
            // stage1/stage2 tree overhead and sibling depth loss; keep tree on greedy & lower-conf stochastic!
            const bool stoch_prefer_linear = (!is_greedy && eng->ema_geom_p >= 0.52);
            drafted_tree = (want_tree && eff_max_k >= 2U && !stoch_prefer_linear);

            id<MTLCommandBuffer>         cb_draft  = [eng->queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc_draft = [cb_draft computeCommandEncoder];
            if (drafted_tree) {
                sik_encode_mtp_draft_tree(enc_draft, eng, eng->seq_pos, eff_max_k, use_frspec, sampling);
            } else {
                sik_encode_mtp_draft_block(enc_draft, eng, eng->seq_pos, eff_max_k, use_frspec, /*want_pmax=*/true, sampling);
            }
            [enc_draft endEncoding];
            [cb_draft commit];
            [cb_draft waitUntilCompleted];
            if ([cb_draft status] == MTLCommandBufferStatusError) {
                NSError* err = [cb_draft error];
                std::fprintf(stderr, "[strata_inkling2_gpu] MTP draft GPU error: %s\n",
                             err ? [[err localizedDescription] UTF8String] : "unknown");
                return -1;
            }
            if (eng->pending_fix_cb) {
                eng->pending_fix_cb = nil; // Guaranteed completed before cb_draft on same serial queue
            }

            const double g_dt = ([cb_draft GPUEndTime] - [cb_draft GPUStartTime]) * 1e3;
            if (is_greedy && g_dt > 0.05 && g_dt < 100.0) {
                measured_draft_ms = g_dt;
                eng->ema_draft_ms = 0.85 * eng->ema_draft_ms + 0.15 * g_dt;
            } else {
                measured_draft_ms = eng->ema_draft_ms;
            }

            const uint32_t* mtoks = reinterpret_cast<const uint32_t*>([eng->b_mtp_token_ids contents]);
            const double    cal   = eng->ema_acc_cal;

            uint32_t best_k    = 0U;
            double   best_tps  = 1.0 / std::max(measured_draft_ms + eff_k0_ms, 0.1);
            double   best_etok = 1.0;

            if (drafted_tree) {
                const SikTreeMetaHost* tr = reinterpret_cast<const SikTreeMetaHost*>([eng->b_tree_meta contents]);
                const uint32_t max_draft  = (tr->M > 1U) ? std::min<uint32_t>(eff_max_k, tr->M - 1U) : 0U;
                double cal_cp[kMaxSpecBatch] = {};
                cal_cp[0] = 1.0;
                double exp_toks = 1.0;
                bool prefix_linear = true;
                for (uint32_t i = 1U; i <= max_draft; ++i) {
                    const uint32_t par = tr->parent[i];
                    if (par != i - 1U || (!is_greedy && tr->tok_id[i] != tr->sampled_tok[i - 1U])) {
                        prefix_linear = false;
                    }
                    const double   pm  = std::clamp<double>(static_cast<double>(tr->node_prob[i]), 0.0, 1.0);
                    const double   pa  = std::min<double>(0.985, cal * pm);
                    cal_cp[i] = (par < i) ? (cal_cp[par] * pa) : 0.0;
                    exp_toks += cal_cp[i];
                    const double tree_ovh = prefix_linear ? 0.0 : 0.35;
                    const double tps = exp_toks / std::max(measured_draft_ms + eng->ema_ver_ms[i] + tree_ovh, 0.1);
                    if (tps > best_tps) {
                        best_tps  = tps;
                        best_k    = i;
                        best_etok = exp_toks;
                    }
                }
                bool subtree_linear = true;
                for (uint32_t i = 1U; i <= best_k; ++i) {
                    if (tr->parent[i] != i - 1U || (!is_greedy && tr->tok_id[i] != tr->sampled_tok[i - 1U])) {
                        subtree_linear = false;
                        break;
                    }
                }
                // Use zero-overhead linear verify whenever the selected subtree is a verified linear chain
                // (works identically for both greedy and stochastic Leviathan since `d_ids[u*64]` aligns 1:1)!
                verify_tree = (best_k >= 1U) && !subtree_linear;
            } else {
                const float* pmax = reinterpret_cast<const float*>([eng->b_draft_pmax contents]);
                double cum_prob = 1.0;
                double exp_toks = 1.0;
                for (uint32_t k = 1U; k <= eff_max_k; ++k) {
                    const double pm = std::clamp<double>(static_cast<double>(pmax[k - 1U]), 0.0, 1.0);
                    draft_pmax_saved[k - 1U] = pm;
                    const double pa = std::min<double>(0.985, cal * pm);
                    cum_prob *= pa;
                    exp_toks += cum_prob;
                    const double tps = exp_toks / std::max(measured_draft_ms + eng->ema_ver_ms[k], 0.1);
                    if (tps > best_tps) {
                        best_tps  = tps;
                        best_k    = k;
                        best_etok = exp_toks;
                    }
                }
            }
            // Compare steady-state fused spec throughput vs real K=0 throughput!
            // On greedy, account for the ~0.25 ms State C sync + 0.35 ms branching tree verify overhead so
            // low-acc Prose drops cleanly to Bitplane K=0 (`~122 tok/s`) instead of over-drafting!
            const double probe_tree_ovh = verify_tree ? 0.35 : 0.0;
            const double probe_sync_ovh = is_greedy ? 0.25 : 0.0;
            const double probe_d_est    = (best_k > 0U)
                ? (eng->ema_draft_ms * (0.45 + 0.55 * static_cast<double>(best_k) / 7.0))
                : 0.0;
            probe_best_tps   = (best_k > 0U)
                ? (best_etok / std::max(probe_d_est + probe_sync_ovh + eng->ema_ver_ms[best_k] + probe_tree_ovh, 0.1))
                : 0.0;
            probe_k0_tps     = 1.0 / eff_k0_ms;
            eff_k            = best_k;
            M                = 1U + eff_k;
            used_adaptive_cb = true;
            for (uint32_t k = 0U; k < eff_k; ++k) {
                verify_tokens[k + 1U] = mtoks[k + 1U];
            }
            verify_tok_buf = eng->b_mtp_token_ids;
        } else {
            // Fixed-K Zero-Sync Single-Command-Buffer Mode (`cb_draft + cb_ver` fused with 0 CPU sync!)
            step_draft_est = eng->ema_draft_ms * (0.45 + 0.55 * static_cast<double>(eff_k) / 7.0);
            reinterpret_cast<uint32_t*>([eng->b_mtp_token_ids contents])[0] = pending_token_id;
            verify_tok_buf = eng->b_mtp_token_ids;
            used_fused_cb  = true;
            if (want_tree && eff_k >= 2U) {
                drafted_tree = true;
                verify_tree  = true;
            }
        }

        // 3. Encode & Execute Verification Command Buffer `cb_ver` (`M = 1 + eff_k`)
        const uint32_t snap_mask = (eff_k > 0U) ? ((1U << eff_k) - 1U) : 0U;
        uint32_t       bp_dummy  = 0U;
        const bool     use_bp_m1 = (M == 1U && !verify_tree && sik_can_use_bitplane(eng, sampling, &bp_dummy, nullptr));
        eng->prof_active     = eng->prof_enabled && (eng->counters.spec_verify_steps == 4ULL);
        eng->prof_use_csb    = eng->prof_active && (eng->prof_csb != nil);
        eng->prof_pass_count = 0U;
        if (eng->prof_active) {
            std::memset(eng->prof_ms, 0, sizeof(eng->prof_ms));
            std::memset(eng->prof_cnt, 0, sizeof(eng->prof_cnt));
        }
        id<MTLCommandBuffer>         cb_ver       = [eng->queue commandBufferWithUnretainedReferences];
        id<MTLCommandBuffer>         cb_ver_first = cb_ver;
        id<MTLComputeCommandEncoder> enc_ver      = eng->prof_use_csb ? sik_prof_new_enc(eng, cb_ver) : [cb_ver computeCommandEncoder];

        if (used_fused_cb) {
            if (drafted_tree) {
                sik_encode_mtp_draft_tree(enc_ver, eng, eng->seq_pos, eff_k, use_frspec, sampling);
            } else {
                sik_encode_mtp_draft_block(enc_ver, eng, eng->seq_pos, eff_k, use_frspec, /*want_pmax=*/false, sampling);
            }
            sik_prof_mark(eng, cb_ver, enc_ver, 11U);
        }

        sik_encode_backbone_forward(
            cb_ver, enc_ver, eng, verify_tok_buf, eng->seq_pos, M, M,
            snap_mask, /*force_gemv=*/true,
            /*capture_mtp_taps=*/true,
            /*run_unembed_all=*/!use_bp_m1,
            /*run_unembed_last=*/false,
            /*run_final_norm=*/true,
            /*use_tree=*/verify_tree
        );

        eng->sample_step = static_cast<uint64_t>(eng->seq_pos + 1U) << 8U;
        if (use_bp_m1) {
            sik_encode_bitplane_unembed_sample(
                cb_ver, enc_ver, eng, eng->b_in_f16[0], 0U, sampling, eng->b_next_tokens, 0U);
        } else {
            sik_prof_mark_detail(eng, cb_ver, enc_ver, 26U);

            if (verify_tree) {
                sik_encode_logit_penalties_tree(
                    enc_ver, eng, sampling, M, (eff_k > 0U) ? 2U : 0U, eff_k, false, 0U
                );
            } else {
                sik_encode_logit_penalties(
                    enc_ver, eng, sampling, M, verify_tok_buf,
                    (eff_k > 0U) ? 2U : 0U, eff_k, false, 0U
                );
            }

            sik_encode_sample_batch(
                enc_ver, eng, sampling, M,
                eng->b_next_tokens, 0,
                nil, 0,
                is_greedy ? nil : eng->b_sample_top_ids,   0,
                is_greedy ? nil : eng->b_sample_top_probs, 0,
                is_greedy ? nil : eng->b_sample_keep_k,    0,
                false, 0U, 0ULL
            );
        }

        if (eng->prof_active && !eng->prof_use_csb) {
            sik_prof_mark(eng, cb_ver, enc_ver, 11U);
            sik_prof_dump(eng, "SPEC_VERIFY", M, eng->seq_pos);
            eng->prof_active = false;
        } else if (eng->prof_active && eng->prof_use_csb) {
            if (eng->prof_pass_count < 2040U) {
                eng->prof_pass_bkt[eng->prof_pass_count++] = eng->prof_detail ? 27U : 11U;
            }
        }

        [enc_ver endEncoding];
        [cb_ver commit];
        [cb_ver waitUntilCompleted];
        if (eng->prof_active && eng->prof_use_csb) {
            sik_prof_resolve_csb(eng, cb_ver, "SPEC_VERIFY", M, eng->seq_pos);
            eng->prof_active  = false;
            eng->prof_use_csb = false;
        }
        if ([cb_ver_first status] == MTLCommandBufferStatusError || [cb_ver status] == MTLCommandBufferStatusError) {
            NSError* err = ([cb_ver status] == MTLCommandBufferStatusError) ? [cb_ver error] : [cb_ver_first error];
            std::fprintf(stderr, "[strata_inkling2_gpu] Spec verify GPU error: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return -1;
        }
        if (eng->pending_fix_cb) {
            eng->pending_fix_cb = nil; // Guaranteed completed before cb_ver on same serial queue
        }

        // Update Adaptive-K verify timing EMA (stays calibrated at every ctx len across fused & K=0 steps;
        // on stochastic runs keep analytical `ema_ver_ms` so `eff_k` is 100% seed-deterministic across runs!)
        double step_raw_dt_ms = step_draft_est + eng->ema_ver_ms[std::min<uint32_t>(M - 1U, kMaxSpecBatch - 1U)];
        double step_ver_dt_ms = eng->ema_ver_ms[std::min<uint32_t>(M - 1U, kMaxSpecBatch - 1U)];
        if (is_greedy) {
            const double raw_dt = ([cb_ver GPUEndTime] - [cb_ver_first GPUStartTime]) * 1e3;
            const double v_dt   = used_fused_cb ? (raw_dt - step_draft_est) : raw_dt;
            step_raw_dt_ms = raw_dt;
            step_ver_dt_ms = v_dt;
            if (v_dt > 0.2 && v_dt < 200.0 && (M - 1U) < kMaxSpecBatch) {
                const double old_m = eng->ema_ver_ms[M - 1U];
                const double ratio = std::clamp<double>(v_dt / std::max(old_m, 0.5), 0.7, 1.4);
                const double global_nudge = 1.0 + 0.06 * (ratio - 1.0);
                // Scale ALL M=1..8 entries together so GPU thermal/background-load drift never skews M=1 vs M>=2!
                for (uint32_t j = 0U; j < kMaxSpecBatch; ++j) {
                    eng->ema_ver_ms[j] *= global_nudge;
                }
                const double alpha = (M == 1U) ? 0.30 : 0.20;
                const double obs_v = (M == 1U && eng->bitplane_head_enabled && eng->bp_ready && !use_bp_m1)
                    ? std::max(0.5, v_dt - 0.35)
                    : v_dt;
                eng->ema_ver_ms[M - 1U] = (1.0 - alpha) * eng->ema_ver_ms[M - 1U] + alpha * obs_v;
                for (uint32_t j = 1U; j < kMaxSpecBatch; ++j) {
                    if (eng->ema_ver_ms[j] < eng->ema_ver_ms[j - 1U] + 0.3) {
                        eng->ema_ver_ms[j] = eng->ema_ver_ms[j - 1U] + 0.3;
                    }
                }
            }
        }

        if (used_fused_cb && !verify_tree) {
            const uint32_t* mtoks = reinterpret_cast<const uint32_t*>([eng->b_mtp_token_ids contents]);
            for (uint32_t k = 0U; k < eff_k; ++k) {
                verify_tokens[k + 1U] = mtoks[k + 1U];
            }
        }

        // 4. Exact Greedy / Lossless Multi-Branch Leviathan Tree Verification (`0.00000` KL!)
        const uint32_t* target_toks = reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents]);
        uint32_t path[kMaxSpecBatch] = {};
        path[0] = 0U;
        uint32_t accepted = 0U;

        const SikTreeMetaHost* tr_meta = (drafted_tree && eng->b_tree_meta != nil)
            ? reinterpret_cast<const SikTreeMetaHost*>([eng->b_tree_meta contents]) : nullptr;

        if (verify_tree && tr_meta != nullptr && eff_k > 0U) {
            const uint32_t M_tree = std::min<uint32_t>(M, tr_meta->M);
            if (is_greedy) {
                uint32_t u = 0U;
                while (true) {
                    const uint32_t y = target_toks[u];
                    uint32_t matched_c = 0xFFFFFFFFU;
                    for (uint32_t c = u + 1U; c < M_tree; ++c) {
                        if (tr_meta->parent[c] == u && tr_meta->tok_id[c] == y) {
                            matched_c = c;
                            break;
                        }
                    }
                    if (matched_c != 0xFFFFFFFFU && accepted + 1U < M_tree) {
                        emitted_tokens_out[accepted++] = y;
                        path[accepted] = matched_c;
                        u = matched_c;
                    } else {
                        emitted_tokens_out[accepted] = y;
                        break;
                    }
                }
            } else {
                const uint32_t* t_ids   = reinterpret_cast<const uint32_t*>([eng->b_sample_top_ids contents]);
                const float*    t_probs = reinterpret_cast<const float*>([eng->b_sample_top_probs contents]);
                const uint32_t* t_keep  = reinterpret_cast<const uint32_t*>([eng->b_sample_keep_k contents]);
                const uint32_t* d_ids   = reinterpret_cast<const uint32_t*>([eng->b_draft_top_ids contents]);
                const float*    d_probs = reinterpret_cast<const float*>([eng->b_draft_top_probs contents]);
                const uint32_t* d_keep  = reinterpret_cast<const uint32_t*>([eng->b_draft_keep_k contents]);

                uint32_t u = 0U;
                while (true) {
                    const uint32_t samp_tok = tr_meta->sampled_tok[u];
                    uint32_t primary_c = 0xFFFFFFFFU;
                    if (samp_tok != 0xFFFFFFFFU) {
                        for (uint32_t c = u + 1U; c < M_tree; ++c) {
                            if (tr_meta->parent[c] == u && tr_meta->tok_id[c] == samp_tok) {
                                primary_c = c;
                                break;
                            }
                        }
                    }

                    if (primary_c != 0xFFFFFFFFU) {
                        const uint32_t x   = samp_tok;
                        const uint32_t tk  = std::min<uint32_t>(t_keep[u], 64U);
                        const uint32_t dk  = std::min<uint32_t>(d_keep[u], 64U);
                        const float    p_x = sik_lookup_sparse_prob(x, t_ids + u * 64U, t_probs + u * 64U, tk);
                        const float    q_x = sik_lookup_sparse_prob(x, d_ids + u * 64U, d_probs + u * 64U, dk);
                        const float    acc = (q_x > 1e-12f) ? std::min<float>(1.0f, p_x / q_x) : ((p_x > 0.0f) ? 1.0f : 0.0f);
                        const uint64_t lev_pos = (static_cast<uint64_t>(eng->seq_pos + 1U) << 8U) | 0x40ULL;
                        const float    ru  = sik_host_rand_u01(sampling->rng_seed ^ 0xA5A5A5A5ULL, lev_pos + accepted);

                        if (ru < acc && accepted + 1U < M_tree) {
                            emitted_tokens_out[accepted++] = x;
                            path[accepted] = primary_c;
                            u = primary_c;
                            continue;
                        }
                        // Exact residual distribution `norm(max(0, p_u - q_u))` -> mathematically exact `p_u`!
                        float res_p[64];
                        float res_sum = 0.0f;
                        for (uint32_t i = 0U; i < tk; ++i) {
                            const uint32_t tid = t_ids[u * 64U + i];
                            const float    pv  = t_probs[u * 64U + i];
                            const float    qv  = sik_lookup_sparse_prob(tid, d_ids + u * 64U, d_probs + u * 64U, dk);
                            const float    rv  = std::max<float>(0.0f, pv - qv);
                            res_p[i] = rv;
                            res_sum += rv;
                        }
                        uint32_t res_tok = target_toks[u];
                        if (res_sum > 1e-12f) {
                            const float u2 = sik_host_rand_u01(sampling->rng_seed ^ 0x5A5A5A5AULL, lev_pos + accepted + 17ULL) * res_sum;
                            float cdf = 0.0f;
                            res_tok = t_ids[u * 64U];
                            for (uint32_t i = 0U; i < tk; ++i) {
                                cdf += res_p[i];
                                if (u2 <= cdf) {
                                    res_tok = t_ids[u * 64U + i];
                                    break;
                                }
                            }
                        }
                        // Multi-branch lossless rescue: if `res_tok` is already materialized as another child of `u`, keep walking!
                        uint32_t rescue_c = 0xFFFFFFFFU;
                        for (uint32_t c = u + 1U; c < M_tree; ++c) {
                            if (c != primary_c && tr_meta->parent[c] == u && tr_meta->tok_id[c] == res_tok) {
                                rescue_c = c;
                                break;
                            }
                        }
                        if (rescue_c != 0xFFFFFFFFU && accepted + 1U < M_tree) {
                            emitted_tokens_out[accepted++] = res_tok;
                            path[accepted] = rescue_c;
                            u = rescue_c;
                            continue;
                        }
                        emitted_tokens_out[accepted] = res_tok;
                        break;
                    } else {
                        const uint32_t y = target_toks[u];
                        uint32_t matched_c = 0xFFFFFFFFU;
                        for (uint32_t c = u + 1U; c < M_tree; ++c) {
                            if (tr_meta->parent[c] == u && tr_meta->tok_id[c] == y) {
                                matched_c = c;
                                break;
                            }
                        }
                        if (matched_c != 0xFFFFFFFFU && accepted + 1U < M_tree) {
                            emitted_tokens_out[accepted++] = y;
                            path[accepted] = matched_c;
                            u = matched_c;
                            continue;
                        }
                        emitted_tokens_out[accepted] = y;
                        break;
                    }
                }
            }
            for (uint32_t k = 0U; k < accepted; ++k) {
                verify_tokens[k + 1U] = emitted_tokens_out[k];
            }
        } else if (is_greedy || eff_k == 0U) {
            for (uint32_t k = 0U; k < eff_k; ++k) {
                if (verify_tokens[k + 1U] == target_toks[k]) {
                    emitted_tokens_out[accepted++] = verify_tokens[k + 1U];
                    path[accepted] = accepted;
                } else {
                    break;
                }
            }
            emitted_tokens_out[accepted] = target_toks[accepted];
        } else {
            const uint32_t* t_ids   = reinterpret_cast<const uint32_t*>([eng->b_sample_top_ids contents]);
            const float*    t_probs = reinterpret_cast<const float*>([eng->b_sample_top_probs contents]);
            const uint32_t* t_keep  = reinterpret_cast<const uint32_t*>([eng->b_sample_keep_k contents]);
            const uint32_t* d_ids   = reinterpret_cast<const uint32_t*>([eng->b_draft_top_ids contents]);
            const float*    d_probs = reinterpret_cast<const float*>([eng->b_draft_top_probs contents]);
            const uint32_t* d_keep  = reinterpret_cast<const uint32_t*>([eng->b_draft_keep_k contents]);

            bool rejected = false;
            for (uint32_t k = 0U; k < eff_k; ++k) {
                const uint32_t x    = verify_tokens[k + 1U];
                const uint32_t tk   = std::min<uint32_t>(t_keep[k], 64U);
                const uint32_t dk   = std::min<uint32_t>(d_keep[k], 64U);
                const float    p_x  = sik_lookup_sparse_prob(x, t_ids + k * 64U, t_probs + k * 64U, tk);
                const float    q_x  = sik_lookup_sparse_prob(x, d_ids + k * 64U, d_probs + k * 64U, dk);
                const float    acc  = (q_x > 1e-12f) ? std::min<float>(1.0f, p_x / q_x) : ((p_x > 0.0f) ? 1.0f : 0.0f);
                const uint64_t lev_pos = (static_cast<uint64_t>(eng->seq_pos + 1U) << 8U) | 0x40ULL;
                const float    u    = sik_host_rand_u01(sampling->rng_seed ^ 0xA5A5A5A5ULL, lev_pos + k);

                if (u < acc) {
                    emitted_tokens_out[accepted++] = x;
                    path[accepted] = accepted;
                } else {
                    // Exact residual distribution `norm(max(0, p - q))`
                    float res_p[64];
                    float res_sum = 0.0f;
                    for (uint32_t i = 0U; i < tk; ++i) {
                        const uint32_t tid = t_ids[k * 64U + i];
                        const float    pv  = t_probs[k * 64U + i];
                        const float    qv  = sik_lookup_sparse_prob(tid, d_ids + k * 64U, d_probs + k * 64U, dk);
                        const float    rv  = std::max<float>(0.0f, pv - qv);
                        res_p[i] = rv;
                        res_sum += rv;
                    }
                    uint32_t res_tok = target_toks[k];
                    if (res_sum > 1e-12f) {
                        const float u2 = sik_host_rand_u01(sampling->rng_seed ^ 0x5A5A5A5AULL, lev_pos + k + 17ULL) * res_sum;
                        float cdf = 0.0f;
                        res_tok = t_ids[k * 64U];
                        for (uint32_t i = 0U; i < tk; ++i) {
                            cdf += res_p[i];
                            if (u2 <= cdf) {
                                res_tok = t_ids[k * 64U + i];
                                break;
                            }
                        }
                    }
                    emitted_tokens_out[accepted] = res_tok;
                    rejected = true;
                    break;
                }
            }
            if (!rejected) {
                emitted_tokens_out[accepted] = target_toks[accepted];
            }
        }

        const uint32_t n_commit = 1U + accepted;

        // Compute exact tree expected-prob sum along winning path (exact Wald martingale!)
        // and detect full leaf acceptance (`hit_tree_leaf`: winning node had 0 drafted children -> NO rejection!).
        double tree_tested_sum_p = 0.0;
        bool   hit_tree_leaf     = false;
        if (drafted_tree && tr_meta != nullptr && eff_k > 0U) {
            const uint32_t M_tree = std::min<uint32_t>(M, tr_meta->M);
            for (uint32_t d = 0U; d <= accepted; ++d) {
                const uint32_t u = path[d];
                double cp_sum = 0.0;
                uint32_t n_ch = 0U;
                for (uint32_t c = u + 1U; c < M_tree; ++c) {
                    if (tr_meta->parent[c] == u) {
                        cp_sum += std::clamp<double>(static_cast<double>(tr_meta->node_prob[c]), 0.0, 1.0);
                        ++n_ch;
                    }
                }
                if (n_ch > 0U) {
                    tree_tested_sum_p += std::min<double>(1.0, cp_sum);
                } else if (d == accepted && accepted > 0U) {
                    hit_tree_leaf = true;
                }
            }
        }

        // Exact Wald-martingale calibration of MTP probabilities -> P(accept) (State C)
        if (used_adaptive_cb && eff_k > 0U) {
            double sum_p = 0.0;
            if (drafted_tree && tr_meta != nullptr) {
                sum_p = tree_tested_sum_p;
            } else {
                const uint32_t n_tested = std::min<uint32_t>(accepted + 1U, eff_k);
                for (uint32_t j = 0U; j < n_tested; ++j) {
                    sum_p += draft_pmax_saved[j];
                }
            }
            eng->ema_cal_num = 0.88 * eng->ema_cal_num + static_cast<double>(accepted);
            eng->ema_cal_den = 0.88 * eng->ema_cal_den + sum_p;
            const double cal_hi = is_greedy ? 1.02 : 1.25;
            eng->ema_acc_cal = std::clamp<double>(eng->ema_cal_num / std::max(eng->ema_cal_den, 1e-3), 0.65, cal_hi);
        }

        // Unbiased Observed Geometric Acceptance Estimator (`ema_geom_p`, updated on ALL drafted steps!)
        // When a branching tree reaches a leaf with zero rejections (`hit_tree_leaf`), score `accepted/accepted`
        // (no phantom miss!); on rejection, score `accepted/min(accepted+1, eff_k)` so shallow trees don't inflate `eff_k`.
        if (eng->adaptive_k_enabled && !used_ngram && eff_k > 0U) {
            const uint32_t n_tested = hit_tree_leaf ? accepted : std::min<uint32_t>(accepted + 1U, eff_k);
            eng->ema_geom_num = 0.78 * eng->ema_geom_num + static_cast<double>(accepted);
            eng->ema_geom_den = 0.78 * eng->ema_geom_den + static_cast<double>(n_tested);
            eng->ema_geom_p   = std::clamp<double>(eng->ema_geom_num / std::max(eng->ema_geom_den, 1e-3), 0.0, 0.99);
        }

        // Hybrid 3-State Zero-Loss SpecCtl (`CooldownK0` / Zero-Sync Geometric-MLE `HighConfFused` / `ActiveProbe`)
        if (eng->adaptive_k_enabled && !used_ngram) {
            const double step_real_ms = used_adaptive_cb
                ? ((step_ver_dt_ms > 0.2 ? step_ver_dt_ms : eng->ema_ver_ms[eff_k]) + measured_draft_ms + 0.35)
                : (step_raw_dt_ms > 0.2 ? step_raw_dt_ms : (step_draft_est + eng->ema_ver_ms[eff_k]));
            const double step_real_tps = static_cast<double>(n_commit) / std::max(step_real_ms, 0.5);
            const double ref_k0_tps    = 1.0 / eff_k0_ms;

            if (used_cooldown_k0) {
                if (eng->spec_cooldown_rem > 0U) --eng->spec_cooldown_rem;
                if (eng->spec_cooldown_rem == 0U) {
                    eng->spec_state = StrataInkling2GpuEngine::kSpecActiveProbe;
                }
            } else if (used_high_fused) {
                eng->spec_cooldown_level = 0U;
                const bool full_hit_fused =
                    (accepted == eff_k && eff_k >= 3U) ||
                    (verify_tree && hit_tree_leaf && accepted >= 3U && accepted + 2U >= eff_k);
                if (full_hit_fused) {
                    ++eng->spec_high_streak;
                    if (eng->spec_high_streak >= 2U) {
                        eng->spec_locked_k = std::min<uint32_t>(eff_max_k, eff_k + 2U);
                    }
                } else {
                    eng->spec_high_streak = 0U;
                    eng->spec_locked_k    = 0U;
                }
                if (eng->ema_geom_p < 0.56 || (accepted <= 1U && step_real_tps < ref_k0_tps * 0.95)) {
                    eng->spec_state       = StrataInkling2GpuEngine::kSpecActiveProbe;
                    eng->spec_high_streak = 0U;
                    eng->spec_locked_k    = 0U;
                }
            } else if (used_adaptive_cb) {
                // Zero-loss SpecCtl guard gated by smoothed acceptance `ema_geom_p < 0.48`:
                // since initial prior is `0.50`, `low_acc_regime` ONLY activates after genuine low-acc steps
                // (Prose ~0.32..0.43), guaranteeing zero false cooldowns on Code (`ema_geom_p >= 0.55`)!
                const bool low_acc_regime   = (eng->ema_geom_p < (is_greedy ? 0.50 : 0.47));
                const bool spec_loses_to_k0 =
                    (eff_k == 0U) ||
                    (probe_best_tps < probe_k0_tps && (accepted == 0U || (low_acc_regime && step_real_tps < ref_k0_tps * 0.99))) ||
                    (eng->ema_geom_p < (is_greedy ? 0.47 : 0.44) && step_real_tps < ref_k0_tps * 0.96);
                if (spec_loses_to_k0) {
                    ++eng->spec_low_streak;
                    eng->spec_high_streak = 0U;
                    const uint32_t low_thresh = (eng->spec_cooldown_level > 0U && low_acc_regime) ? 1U : 3U;
                    if (eng->spec_low_streak >= low_thresh && (low_acc_regime || eff_k == 0U)) {
                        eng->spec_state = StrataInkling2GpuEngine::kSpecCooldownK0;
                        const uint32_t cd_steps = (eng->spec_cooldown_level == 0U) ? 10U
                                                : (eng->spec_cooldown_level == 1U) ? 16U : 22U;
                        eng->spec_cooldown_rem = cd_steps;
                        if (eng->spec_cooldown_level < 3U) ++eng->spec_cooldown_level;
                    }
                } else {
                    eng->spec_low_streak     = 0U;
                    eng->spec_cooldown_level = 0U;
                    const bool full_hit_c =
                        (accepted == eff_k) ||
                        (drafted_tree && hit_tree_leaf && accepted >= 3U && accepted + 1U >= eff_k);
                    const bool high_lin =
                        (eff_k >= 4U && accepted * 100U >= eff_k * 65U) ||
                        (drafted_tree && hit_tree_leaf && accepted >= 4U) ||
                        (!is_greedy && eff_k >= 4U && accepted >= 3U);
                    if (high_lin) {
                        ++eng->spec_high_streak;
                        if (eng->spec_high_streak >= 2U || (full_hit_c && eng->ema_geom_p >= 0.72)) {
                            eng->spec_state       = StrataInkling2GpuEngine::kSpecHighConfFused;
                            eng->spec_high_streak = full_hit_c ? 1U : 0U;
                            eng->spec_locked_k    = full_hit_c ? std::min<uint32_t>(eff_max_k, eff_k + 1U) : eff_k;
                        }
                    } else {
                        eng->spec_high_streak = 0U;
                    }
                }
            }
        }

        // 5. Record ONLY the `accepted` draft tokens into `pen_hist` (bonus token is next step's `pending`!)
        if (accepted > 0U) {
            strata_inkling2_gpu_penalties_record(eng, emitted_tokens_out, accepted);
        }

        // 7. FR-Spec Auto-Widen Tracker (sliding 32-token miss window; widen if >= 8/32 emitted tokens miss shortlist!)
        if (use_frspec) {
            const uint32_t* f2s = reinterpret_cast<const uint32_t*>([eng->b_frspec_full_to_short contents]);
            for (uint32_t i = 0U; i < n_commit; ++i) {
                const uint32_t tok  = emitted_tokens_out[i];
                const uint32_t miss = (tok >= V || f2s[tok] == 0xFFFFFFFFU) ? 1U : 0U;
                eng->shortlist_miss_mask = (eng->shortlist_miss_mask << 1U) | miss;
            }
            if (__builtin_popcount(eng->shortlist_miss_mask) >= 8) {
                eng->frspec_widened = true;
            }
        }

        // 8. N-Gram Cooldown & History Update
        if (eng->bound_slot) {
            if (used_ngram) {
                if (accepted <= 1U) eng->bound_slot->ngram_cooldown = 16U;
                else if (accepted * 2U < eff_k) eng->bound_slot->ngram_cooldown = 8U;
            } else if (eng->bound_slot->ngram_cooldown > 0U) {
                --eng->bound_slot->ngram_cooldown;
            }
            sik_ngram_append(eng->bound_slot, verify_tokens, n_commit);
        }

        // 9. Async `cb_fix`: Tree KV/Tap Compaction (iff `!path_is_identity`) + SConv Commit (`8.5 us`!) + MTP Ctx KV Update!
        const uint32_t old_pos = eng->seq_pos;
        const uint32_t leaf    = verify_tree ? path[accepted] : accepted;
        bool path_is_identity  = true;
        if (verify_tree) {
            for (uint32_t k = 1U; k <= accepted; ++k) {
                if (path[k] != k) { path_is_identity = false; break; }
            }
        }
        id<MTLCommandBuffer>         cb_fix  = [eng->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc_fix = [cb_fix computeCommandEncoder];
        if (!path_is_identity) {
            SikTreeCompactParamsHost cp{};
            cp.num_local_visits    = eng->cfg.num_local_visits;
            cp.num_global_visits   = eng->cfg.num_global_visits;
            cp.n_kv                = eng->cfg.num_key_value_heads;
            cp.head_dim            = eng->cfg.head_dim;
            cp.old_pos             = old_pos;
            cp.swa_stride          = kSwaRingLen;
            cp.glob_stride         = eng->max_seq_alloc;
            cp.kv_dtype            = eng->kv_dtype;
            cp.glob_layer_stride_b = eng->glob_kv_layer_stride;
            cp.glob_q8_data_b      = eng->glob_kv_q8_data_bytes;
            cp.path_len            = n_commit;
            cp.D                   = D;
            for (uint32_t k = 0U; k < n_commit; ++k) cp.path[k] = path[k];

            [enc_fix setComputePipelineState:eng->pso_compact_kv_tree];
            [enc_fix setBuffer:eng->b_swa_k_cache offset:0 atIndex:0];
            [enc_fix setBuffer:eng->b_swa_v_cache offset:0 atIndex:1];
            [enc_fix setBuffer:eng->b_glob_k_cache offset:0 atIndex:2];
            [enc_fix setBuffer:eng->b_glob_v_cache offset:0 atIndex:3];
            [enc_fix setBytes:&cp length:sizeof(cp) atIndex:4];
            [enc_fix dispatchThreadgroups:MTLSizeMake(eng->cfg.num_local_visits + eng->cfg.num_global_visits,
                                                     eng->cfg.num_key_value_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc_fix setComputePipelineState:eng->pso_compact_taps_tree];
            [enc_fix setBuffer:eng->b_mtp_taps_chunk offset:0 atIndex:0];
            [enc_fix setBytes:&cp length:sizeof(cp) atIndex:1];
            [enc_fix dispatchThreadgroups:MTLSizeMake(((D / 4U) + 255U) / 256U, STRATA_INKLING2_MAX_MTP_TAPS, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        if (leaf < eff_k && eng->b_sconv_snaps) {
            const NSUInteger snap_off = static_cast<NSUInteger>(leaf) * static_cast<NSUInteger>(eng->sconv_total_bytes);
            sik_encode_copy_buffer(enc_fix, eng->pso_copy_u32,
                                   eng->b_sconv_snaps, snap_off,
                                   eng->b_sconv_state, 0,
                                   static_cast<size_t>(eng->sconv_total_bytes));
        }
        sik_encode_mtp_ctx_update(enc_fix, eng, eng->b_mtp_taps_chunk, old_pos, n_commit, true);
        {
            const size_t tap_bytes = static_cast<size_t>(STRATA_INKLING2_MAX_MTP_TAPS) * D * sizeof(float);
            const NSUInteger src_off = static_cast<NSUInteger>(accepted) * tap_bytes;
            sik_encode_copy_buffer(enc_fix, eng->pso_copy_u32,
                                   eng->b_mtp_taps_chunk, src_off,
                                   eng->b_mtp_taps_last, 0, tap_bytes);
        }
        [enc_fix endEncoding];
        [cb_fix commit];
        eng->pending_fix_cb = cb_fix;

        eng->seq_pos    += n_commit;
        eng->mtp_ctx_pos = eng->seq_pos;
        if (eng->bound_slot) sik_sync_engine_to_slot(eng);

        // 10. Telemetry Counters (exact DRAM accounting for FR-Spec vs Full-Vocab!)
        eng->counters.spec_verify_steps  += 1ULL;
        eng->counters.mtp_drafted_tokens += static_cast<uint64_t>(eff_k);
        eng->counters.mtp_accepted_tokens += static_cast<uint64_t>(accepted);
        uint64_t step_dram = use_bp_m1
            ? (eng->hdr.active_bytes_per_tok - (eng->hdr.unembed.bytes / 2ULL))
            : eng->hdr.active_bytes_per_tok;
        if (!used_ngram && !used_cooldown_k0 && (used_adaptive_cb || eff_k > 0U)) {
            const uint32_t d_steps = used_adaptive_cb ? eff_max_k : eff_k;
            const uint64_t head_read = use_frspec
                ? (eng->hdr.mtp.frspec.unembed_q4a.bytes + static_cast<uint64_t>(d_steps) * eng->hdr.mtp.frspec.fold_weight_q8.bytes)
                : (eng->hdr.unembed.bytes + static_cast<uint64_t>(d_steps) * static_cast<uint64_t>(V) * eng->hdr.mtp.markov_rank * 2ULL);
            step_dram += eng->hdr.mtp_active_bytes_per_blk + head_read;
        }
        eng->counters.physical_dram_bytes_read += step_dram;

        *num_emitted_out  = n_commit;
        *num_accepted_out = accepted;
        return 0;
    }
}

/* ============================================================================
 * O(1) GPU Checkpoint & Multi-Slot Prefix Isolation API
 * ============================================================================ */

StrataInkling2Checkpoint* strata_inkling2_checkpoint_create(const StrataInkling2GpuEngine* eng) {
    if (!eng) return nullptr;
    @autoreleasepool {
        StrataInkling2Checkpoint* ckpt = new StrataInkling2Checkpoint();
        ckpt->eng = const_cast<StrataInkling2GpuEngine*>(eng);
        const StrataInkling2Config& cfg = eng->cfg;
        const uint32_t n_kv  = cfg.num_key_value_heads;
        const uint32_t hd    = cfg.head_dim;
        const uint32_t D     = cfg.hidden_size;

        const size_t swa_bytes = static_cast<size_t>(cfg.num_local_visits) * n_kv * kSwaCkptLen * hd * sizeof(uint16_t);
        ckpt->b_swa_k_ckpt = sik_alloc_private_aligned(eng->device, swa_bytes);
        ckpt->b_swa_v_ckpt = sik_alloc_private_aligned(eng->device, swa_bytes);
        ckpt->b_sconv_ckpt = sik_alloc_private_aligned(eng->device, static_cast<size_t>(eng->sconv_total_bytes));
        if (!ckpt->b_swa_k_ckpt || !ckpt->b_swa_v_ckpt || !ckpt->b_sconv_ckpt) {
            strata_inkling2_checkpoint_destroy(ckpt);
            return nullptr;
        }

        if (eng->hdr.mtp.enabled != 0U) {
            uint32_t mtp_loc = 0U;
            for (uint32_t l = 0U; l < eng->hdr.mtp.num_layers; ++l) {
                if (eng->hdr.mtp.layers[l].is_local_swa != 0U) ++mtp_loc;
            }
            if (mtp_loc > 0U) {
                const size_t mswa_bytes = static_cast<size_t>(mtp_loc) * n_kv * kSwaCkptLen * hd * sizeof(uint16_t);
                ckpt->b_mtp_swa_k_ckpt = sik_alloc_private_aligned(eng->device, mswa_bytes);
                ckpt->b_mtp_swa_v_ckpt = sik_alloc_private_aligned(eng->device, mswa_bytes);
                if (!ckpt->b_mtp_swa_k_ckpt || !ckpt->b_mtp_swa_v_ckpt) {
                    strata_inkling2_checkpoint_destroy(ckpt);
                    return nullptr;
                }
            }
            ckpt->b_mtp_taps_last_ckpt = sik_alloc_private_aligned(
                eng->device, static_cast<size_t>(STRATA_INKLING2_MAX_MTP_TAPS) * D * sizeof(float));
            if (!ckpt->b_mtp_taps_last_ckpt) {
                strata_inkling2_checkpoint_destroy(ckpt);
                return nullptr;
            }
        }
        return ckpt;
    }
}

void strata_inkling2_checkpoint_destroy(StrataInkling2Checkpoint* ckpt) {
    if (!ckpt) return;
    @autoreleasepool {
        ckpt->b_swa_k_ckpt         = nil;
        ckpt->b_swa_v_ckpt         = nil;
        ckpt->b_glob_k_ckpt        = nil;
        ckpt->b_glob_v_ckpt        = nil;
        ckpt->b_sconv_ckpt         = nil;
        ckpt->b_mtp_swa_k_ckpt     = nil;
        ckpt->b_mtp_swa_v_ckpt     = nil;
        ckpt->b_mtp_glob_k_ckpt    = nil;
        ckpt->b_mtp_glob_v_ckpt    = nil;
        ckpt->b_mtp_taps_last_ckpt = nil;
        delete ckpt;
    }
}

int strata_inkling2_checkpoint_save(StrataInkling2GpuEngine* eng, StrataInkling2Checkpoint* ckpt) {
    if (!eng || !ckpt) return -1;
    @autoreleasepool {
        if (!sik_drain_pending_fix(eng)) return -1;

        const StrataInkling2Config& cfg = eng->cfg;
        const uint32_t n_kv   = cfg.num_key_value_heads;
        const uint32_t hd     = cfg.head_dim;
        const uint32_t D      = cfg.hidden_size;
        const uint32_t pos    = eng->seq_pos;
        const bool     mtp_on = (eng->hdr.mtp.enabled != 0U);

        uint32_t mtp_loc = 0U, mtp_glob = 0U;
        if (mtp_on) {
            for (uint32_t l = 0U; l < eng->hdr.mtp.num_layers; ++l) {
                if (eng->hdr.mtp.layers[l].is_local_swa != 0U) ++mtp_loc;
                else                                            ++mtp_glob;
            }
        }

        // Lazily size global KV checkpoint buffers to current `pos` (supports both F16 & Q8_0!)
        const bool is_q8_kv = (eng->kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0);
        if (pos > ckpt->glob_stride_alloc || ckpt->kv_dtype != eng->kv_dtype) {
            const uint32_t need = std::max<uint32_t>(1024U, (pos + 1023U) & ~1023U);
            uint64_t layer_bytes = 0ULL, q8_data = 0ULL, q8_scale = 0ULL;
            if (is_q8_kv) {
                strata_inkling2_kv_q8_layer_layout(n_kv, need, hd, &q8_data, &q8_scale);
                layer_bytes = q8_data + q8_scale;
            } else {
                layer_bytes = static_cast<uint64_t>(n_kv) * need * hd * sizeof(uint16_t);
            }
            const size_t gbytes = static_cast<size_t>(cfg.num_global_visits) * layer_bytes;
            ckpt->b_glob_k_ckpt = sik_alloc_private_aligned(eng->device, gbytes);
            ckpt->b_glob_v_ckpt = sik_alloc_private_aligned(eng->device, gbytes);
            if (!ckpt->b_glob_k_ckpt || !ckpt->b_glob_v_ckpt) return -1;
            ckpt->glob_stride_alloc     = need;
            ckpt->kv_dtype              = eng->kv_dtype;
            ckpt->glob_ckpt_layer_bytes = layer_bytes;
            ckpt->glob_ckpt_q8_data     = q8_data;
        }
        if (mtp_on && mtp_glob > 0U && pos > ckpt->mtp_glob_stride_alloc) {
            const uint32_t need = (pos + 1023U) & ~1023U;
            const size_t mgbytes = static_cast<size_t>(mtp_glob) * n_kv * need * hd * sizeof(uint16_t);
            ckpt->b_mtp_glob_k_ckpt = sik_alloc_private_aligned(eng->device, mgbytes);
            ckpt->b_mtp_glob_v_ckpt = sik_alloc_private_aligned(eng->device, mgbytes);
            if (!ckpt->b_mtp_glob_k_ckpt || !ckpt->b_mtp_glob_v_ckpt) return -1;
            ckpt->mtp_glob_stride_alloc = need;
        }

        id<MTLCommandBuffer> cb = [eng->queue commandBuffer];

        if (pos > 0U) {
            const uint32_t swa_cnt   = std::min<uint32_t>(pos, kSwaCkptLen);
            const uint32_t swa_start = pos - swa_cnt;
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            SikSwaCkptParamsHost sp{cfg.num_local_visits, n_kv, hd, swa_start, swa_cnt, kSwaRingLen, kSwaCkptLen, 1U};
            [enc setComputePipelineState:eng->pso_swa_ring_copy];
            [enc setBuffer:eng->b_swa_k_cache offset:0 atIndex:0];
            [enc setBuffer:eng->b_swa_v_cache offset:0 atIndex:1];
            [enc setBuffer:ckpt->b_swa_k_ckpt offset:0 atIndex:2];
            [enc setBuffer:ckpt->b_swa_v_ckpt offset:0 atIndex:3];
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(swa_cnt, cfg.num_local_visits * n_kv, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            if (mtp_on && mtp_loc > 0U) {
                SikSwaCkptParamsHost msp{mtp_loc, n_kv, hd, swa_start, swa_cnt, kSwaRingLen, kSwaCkptLen, 1U};
                [enc setComputePipelineState:eng->pso_swa_ring_copy];
                [enc setBuffer:eng->b_mtp_swa_k_cache offset:0 atIndex:0];
                [enc setBuffer:eng->b_mtp_swa_v_cache offset:0 atIndex:1];
                [enc setBuffer:ckpt->b_mtp_swa_k_ckpt offset:0 atIndex:2];
                [enc setBuffer:ckpt->b_mtp_swa_v_ckpt offset:0 atIndex:3];
                [enc setBytes:&msp length:sizeof(msp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(swa_cnt, mtp_loc * n_kv, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            [enc endEncoding];
        }

        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:eng->b_sconv_state sourceOffset:0
                    toBuffer:ckpt->b_sconv_ckpt destinationOffset:0
                        size:static_cast<NSUInteger>(eng->sconv_total_bytes)];

        if (pos > 0U) {
            if (is_q8_kv) {
                const NSUInteger q_copy_sz = static_cast<NSUInteger>(pos) * hd;
                const NSUInteger s_copy_sz = static_cast<NSUInteger>(pos) * sizeof(uint16_t);
                for (uint32_t l = 0U; l < cfg.num_global_visits; ++l) {
                    const NSUInteger src_l = static_cast<NSUInteger>(l * eng->glob_kv_layer_stride);
                    const NSUInteger dst_l = static_cast<NSUInteger>(l * ckpt->glob_ckpt_layer_bytes);
                    for (uint32_t h = 0U; h < n_kv; ++h) {
                        const NSUInteger sq_off = src_l + static_cast<NSUInteger>(h) * eng->max_seq_alloc * hd;
                        const NSUInteger dq_off = dst_l + static_cast<NSUInteger>(h) * ckpt->glob_stride_alloc * hd;
                        const NSUInteger ss_off = src_l + static_cast<NSUInteger>(eng->glob_kv_q8_data_bytes)
                                                + static_cast<NSUInteger>(h) * eng->max_seq_alloc * sizeof(uint16_t);
                        const NSUInteger ds_off = dst_l + static_cast<NSUInteger>(ckpt->glob_ckpt_q8_data)
                                                + static_cast<NSUInteger>(h) * ckpt->glob_stride_alloc * sizeof(uint16_t);
                        [blit copyFromBuffer:eng->b_glob_k_cache sourceOffset:sq_off toBuffer:ckpt->b_glob_k_ckpt destinationOffset:dq_off size:q_copy_sz];
                        [blit copyFromBuffer:eng->b_glob_v_cache sourceOffset:sq_off toBuffer:ckpt->b_glob_v_ckpt destinationOffset:dq_off size:q_copy_sz];
                        [blit copyFromBuffer:eng->b_glob_k_cache sourceOffset:ss_off toBuffer:ckpt->b_glob_k_ckpt destinationOffset:ds_off size:s_copy_sz];
                        [blit copyFromBuffer:eng->b_glob_v_cache sourceOffset:ss_off toBuffer:ckpt->b_glob_v_ckpt destinationOffset:ds_off size:s_copy_sz];
                    }
                }
            } else {
                const NSUInteger copy_sz = static_cast<NSUInteger>(pos) * hd * sizeof(uint16_t);
                const NSUInteger src_h   = static_cast<NSUInteger>(eng->max_seq_alloc) * hd * sizeof(uint16_t);
                const NSUInteger dst_h   = static_cast<NSUInteger>(ckpt->glob_stride_alloc) * hd * sizeof(uint16_t);
                for (uint32_t lh = 0U; lh < cfg.num_global_visits * n_kv; ++lh) {
                    [blit copyFromBuffer:eng->b_glob_k_cache sourceOffset:lh * src_h
                                toBuffer:ckpt->b_glob_k_ckpt destinationOffset:lh * dst_h
                                    size:copy_sz];
                    [blit copyFromBuffer:eng->b_glob_v_cache sourceOffset:lh * src_h
                                toBuffer:ckpt->b_glob_v_ckpt destinationOffset:lh * dst_h
                                    size:copy_sz];
                }
            }
            if (mtp_on && mtp_glob > 0U) {
                const NSUInteger copy_sz = static_cast<NSUInteger>(pos) * hd * sizeof(uint16_t);
                const NSUInteger src_h   = static_cast<NSUInteger>(eng->max_seq_alloc) * hd * sizeof(uint16_t);
                const NSUInteger mdst_h  = static_cast<NSUInteger>(ckpt->mtp_glob_stride_alloc) * hd * sizeof(uint16_t);
                for (uint32_t lh = 0U; lh < mtp_glob * n_kv; ++lh) {
                    [blit copyFromBuffer:eng->b_mtp_glob_k_cache sourceOffset:lh * src_h
                                toBuffer:ckpt->b_mtp_glob_k_ckpt destinationOffset:lh * mdst_h
                                    size:copy_sz];
                    [blit copyFromBuffer:eng->b_mtp_glob_v_cache sourceOffset:lh * src_h
                                toBuffer:ckpt->b_mtp_glob_v_ckpt destinationOffset:lh * mdst_h
                                    size:copy_sz];
                }
            }
        }
        if (mtp_on && eng->b_mtp_taps_last && ckpt->b_mtp_taps_last_ckpt) {
            [blit copyFromBuffer:eng->b_mtp_taps_last sourceOffset:0
                        toBuffer:ckpt->b_mtp_taps_last_ckpt destinationOffset:0
                            size:static_cast<NSUInteger>(STRATA_INKLING2_MAX_MTP_TAPS) * D * sizeof(float)];
        }
        [blit endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] == MTLCommandBufferStatusError) return -1;

        ckpt->seq_pos     = eng->seq_pos;
        ckpt->mtp_ctx_pos = eng->mtp_ctx_pos;
        ckpt->sample_step = eng->sample_step;
        if (eng->bound_slot) {
            ckpt->ngram_hist = eng->bound_slot->ngram_hist;
            ckpt->pen_hist   = eng->bound_slot->pen_hist;
            ckpt->mtp_synced = eng->bound_slot->mtp_synced;
        } else {
            ckpt->pen_hist.clear();
            ckpt->mtp_synced = true;
        }
        return 0;
    }
}

int strata_inkling2_checkpoint_restore(StrataInkling2GpuEngine* eng, const StrataInkling2Checkpoint* ckpt) {
    if (!eng || !ckpt) return -1;
    if (ckpt->seq_pos > eng->max_seq_len) return -1;
    if (ckpt->seq_pos > 0U && eng->kv_dtype != ckpt->kv_dtype) return -1;

    @autoreleasepool {
        if (!sik_drain_pending_fix(eng)) return -1;

        const StrataInkling2Config& cfg = eng->cfg;
        const uint32_t n_kv   = cfg.num_key_value_heads;
        const uint32_t hd     = cfg.head_dim;
        const uint32_t D      = cfg.hidden_size;
        const uint32_t pos    = ckpt->seq_pos;
        const bool     mtp_on = (eng->hdr.mtp.enabled != 0U);

        uint32_t mtp_loc = 0U, mtp_glob = 0U;
        if (mtp_on) {
            for (uint32_t l = 0U; l < eng->hdr.mtp.num_layers; ++l) {
                if (eng->hdr.mtp.layers[l].is_local_swa != 0U) ++mtp_loc;
                else                                            ++mtp_glob;
            }
        }

        id<MTLCommandBuffer> cb = [eng->queue commandBuffer];

        if (pos > 0U) {
            const uint32_t swa_cnt   = std::min<uint32_t>(pos, kSwaCkptLen);
            const uint32_t swa_start = pos - swa_cnt;
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            SikSwaCkptParamsHost sp{cfg.num_local_visits, n_kv, hd, swa_start, swa_cnt, kSwaRingLen, kSwaCkptLen, 0U};
            [enc setComputePipelineState:eng->pso_swa_ring_copy];
            [enc setBuffer:eng->b_swa_k_cache offset:0 atIndex:0];
            [enc setBuffer:eng->b_swa_v_cache offset:0 atIndex:1];
            [enc setBuffer:ckpt->b_swa_k_ckpt offset:0 atIndex:2];
            [enc setBuffer:ckpt->b_swa_v_ckpt offset:0 atIndex:3];
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(swa_cnt, cfg.num_local_visits * n_kv, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            if (mtp_on && mtp_loc > 0U) {
                SikSwaCkptParamsHost msp{mtp_loc, n_kv, hd, swa_start, swa_cnt, kSwaRingLen, kSwaCkptLen, 0U};
                [enc setComputePipelineState:eng->pso_swa_ring_copy];
                [enc setBuffer:eng->b_mtp_swa_k_cache offset:0 atIndex:0];
                [enc setBuffer:eng->b_mtp_swa_v_cache offset:0 atIndex:1];
                [enc setBuffer:ckpt->b_mtp_swa_k_ckpt offset:0 atIndex:2];
                [enc setBuffer:ckpt->b_mtp_swa_v_ckpt offset:0 atIndex:3];
                [enc setBytes:&msp length:sizeof(msp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(swa_cnt, mtp_loc * n_kv, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            [enc endEncoding];
        }

        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:ckpt->b_sconv_ckpt sourceOffset:0
                    toBuffer:eng->b_sconv_state destinationOffset:0
                        size:static_cast<NSUInteger>(eng->sconv_total_bytes)];

        if (pos > 0U) {
            const bool is_q8_kv = (eng->kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0);
            if (is_q8_kv) {
                const NSUInteger q_copy_sz = static_cast<NSUInteger>(pos) * hd;
                const NSUInteger s_copy_sz = static_cast<NSUInteger>(pos) * sizeof(uint16_t);
                for (uint32_t l = 0U; l < cfg.num_global_visits; ++l) {
                    const NSUInteger src_l = static_cast<NSUInteger>(l * ckpt->glob_ckpt_layer_bytes);
                    const NSUInteger dst_l = static_cast<NSUInteger>(l * eng->glob_kv_layer_stride);
                    for (uint32_t h = 0U; h < n_kv; ++h) {
                        const NSUInteger sq_off = src_l + static_cast<NSUInteger>(h) * ckpt->glob_stride_alloc * hd;
                        const NSUInteger dq_off = dst_l + static_cast<NSUInteger>(h) * eng->max_seq_alloc * hd;
                        const NSUInteger ss_off = src_l + static_cast<NSUInteger>(ckpt->glob_ckpt_q8_data)
                                                + static_cast<NSUInteger>(h) * ckpt->glob_stride_alloc * sizeof(uint16_t);
                        const NSUInteger ds_off = dst_l + static_cast<NSUInteger>(eng->glob_kv_q8_data_bytes)
                                                + static_cast<NSUInteger>(h) * eng->max_seq_alloc * sizeof(uint16_t);
                        [blit copyFromBuffer:ckpt->b_glob_k_ckpt sourceOffset:sq_off toBuffer:eng->b_glob_k_cache destinationOffset:dq_off size:q_copy_sz];
                        [blit copyFromBuffer:ckpt->b_glob_v_ckpt sourceOffset:sq_off toBuffer:eng->b_glob_v_cache destinationOffset:dq_off size:q_copy_sz];
                        [blit copyFromBuffer:ckpt->b_glob_k_ckpt sourceOffset:ss_off toBuffer:eng->b_glob_k_cache destinationOffset:ds_off size:s_copy_sz];
                        [blit copyFromBuffer:ckpt->b_glob_v_ckpt sourceOffset:ss_off toBuffer:eng->b_glob_v_cache destinationOffset:ds_off size:s_copy_sz];
                    }
                }
            } else {
                const NSUInteger copy_sz = static_cast<NSUInteger>(pos) * hd * sizeof(uint16_t);
                const NSUInteger dst_h   = static_cast<NSUInteger>(eng->max_seq_alloc) * hd * sizeof(uint16_t);
                const NSUInteger src_h   = static_cast<NSUInteger>(ckpt->glob_stride_alloc) * hd * sizeof(uint16_t);
                for (uint32_t lh = 0U; lh < cfg.num_global_visits * n_kv; ++lh) {
                    [blit copyFromBuffer:ckpt->b_glob_k_ckpt sourceOffset:lh * src_h
                                toBuffer:eng->b_glob_k_cache destinationOffset:lh * dst_h
                                    size:copy_sz];
                    [blit copyFromBuffer:ckpt->b_glob_v_ckpt sourceOffset:lh * src_h
                                toBuffer:eng->b_glob_v_cache destinationOffset:lh * dst_h
                                    size:copy_sz];
                }
            }
            if (mtp_on && mtp_glob > 0U) {
                const NSUInteger copy_sz = static_cast<NSUInteger>(pos) * hd * sizeof(uint16_t);
                const NSUInteger dst_h   = static_cast<NSUInteger>(eng->max_seq_alloc) * hd * sizeof(uint16_t);
                const NSUInteger msrc_h  = static_cast<NSUInteger>(ckpt->mtp_glob_stride_alloc) * hd * sizeof(uint16_t);
                for (uint32_t lh = 0U; lh < mtp_glob * n_kv; ++lh) {
                    [blit copyFromBuffer:ckpt->b_mtp_glob_k_ckpt sourceOffset:lh * msrc_h
                                toBuffer:eng->b_mtp_glob_k_cache destinationOffset:lh * dst_h
                                    size:copy_sz];
                    [blit copyFromBuffer:ckpt->b_mtp_glob_v_ckpt sourceOffset:lh * msrc_h
                                toBuffer:eng->b_mtp_glob_v_cache destinationOffset:lh * dst_h
                                    size:copy_sz];
                }
            }
        }
        if (mtp_on && eng->b_mtp_taps_last && ckpt->b_mtp_taps_last_ckpt) {
            [blit copyFromBuffer:ckpt->b_mtp_taps_last_ckpt sourceOffset:0
                        toBuffer:eng->b_mtp_taps_last destinationOffset:0
                            size:static_cast<NSUInteger>(STRATA_INKLING2_MAX_MTP_TAPS) * D * sizeof(float)];
        }
        [blit endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] == MTLCommandBufferStatusError) return -1;

        eng->seq_pos             = ckpt->seq_pos;
        eng->mtp_ctx_pos         = ckpt->mtp_ctx_pos;
        eng->sample_step         = ckpt->sample_step;
        eng->frspec_widened      = false;
        eng->shortlist_miss_mask = 0U;
        eng->ema_cal_num         = 4.80;
        eng->ema_cal_den         = 5.00;
        eng->ema_acc_cal         = 0.96;
        sik_reset_spec_ctl(eng);
        if (eng->bound_slot) {
            eng->bound_slot->ngram_hist = ckpt->ngram_hist;
            sik_ngram_rewind(eng->bound_slot, static_cast<uint32_t>(ckpt->ngram_hist.size()));
            eng->bound_slot->pen_hist   = ckpt->pen_hist;
            eng->bound_slot->mtp_synced = ckpt->mtp_synced;
            if (eng->pen_cache_slot == eng->bound_slot) {
                sik_penalties_clear_active(eng);
                eng->pen_cache_slot   = nullptr;
                eng->pen_window_start = 0U;
                eng->pen_window_end   = 0U;
            }
            sik_sync_engine_to_slot(eng);
        }
        return 0;
    }
}

StrataInkling2ContextSlot* strata_inkling2_slot_create(StrataInkling2GpuEngine* eng) {
    if (!eng) return nullptr;
    @autoreleasepool {
        StrataInkling2ContextSlot* slot = new StrataInkling2ContextSlot();
        slot->eng = eng;
        if (!sik_alloc_slot_buffers(eng, slot)) {
            sik_free_slot_contents(eng, slot);
            delete slot;
            return nullptr;
        }
        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        sik_encode_zero_buffer(enc, eng->pso_zero_u32, slot->b_sconv_state, static_cast<size_t>(eng->sconv_total_bytes));
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return slot;
    }
}

void strata_inkling2_slot_destroy(StrataInkling2ContextSlot* slot) {
    if (!slot || slot->is_default) return;
    @autoreleasepool {
        StrataInkling2GpuEngine* eng = slot->eng;
        if (eng && eng->pen_cache_slot == slot) {
            sik_penalties_clear_active(eng);
            eng->pen_cache_slot   = nullptr;
            eng->pen_window_start = 0U;
            eng->pen_window_end   = 0U;
        }
        if (eng && eng->bound_slot == slot) {
            strata_inkling2_bind_slot(eng, nullptr);
        }
        sik_free_slot_contents(eng, slot);
        delete slot;
    }
}

void strata_inkling2_bind_slot(StrataInkling2GpuEngine* eng, StrataInkling2ContextSlot* slot) {
    if (!eng) return;
    @autoreleasepool {
        sik_drain_pending_fix(eng);
        sik_sync_engine_to_slot(eng);
        StrataInkling2ContextSlot* target = slot ? slot : eng->default_slot;
        if (target) {
            sik_sync_slot_to_engine(eng, target);
        }
        sik_reset_spec_ctl(eng);
    }
}

uint32_t strata_inkling2_slot_seq_pos(const StrataInkling2ContextSlot* slot) {
    if (!slot) return 0U;
    if (slot->eng && slot->eng->bound_slot == slot) return slot->eng->seq_pos;
    return slot->seq_pos;
}

bool strata_inkling2_slot_mtp_synced(const StrataInkling2ContextSlot* slot) {
    return slot ? slot->mtp_synced : false;
}

size_t strata_inkling2_slot_memory_bytes(const StrataInkling2GpuEngine* eng) {
    return eng ? eng->slot_memory_bytes : 0U;
}

} /* extern "C" */

/* ============================================================================
 * True GPU-Batched Continuous-Batching Multi-Stream Decode (`B = 1..64`)
 * ============================================================================ */

namespace {

bool sik_ensure_batch_workspaces(
    StrataInkling2GpuEngine* eng,
    uint32_t                 B,
    uint32_t                 total_pen_entries
) {
    const StrataInkling2Config& cfg = eng->cfg;
    const uint32_t n_q = cfg.num_attention_heads;
    const uint32_t hd  = cfg.head_dim;
    const uint32_t V   = cfg.vocab_size_unpadded;

    NSMutableArray<id<MTLBuffer>>* newly_added = nil;
    NSMutableArray<id<MTLBuffer>>* removed     = nil;
    auto track_add = [&](id<MTLBuffer> buf) {
        if (!buf) return;
        if (!newly_added) newly_added = [NSMutableArray array];
        [newly_added addObject:buf];
    };
    auto track_rm = [&](id<MTLBuffer> buf) {
        if (!buf) return;
        if (!removed) removed = [NSMutableArray array];
        [removed addObject:buf];
    };

    if (!eng->b_batch_slot_meta) {
        eng->b_batch_slot_meta = sik_alloc_shared_prefaulted(
            eng->device, static_cast<size_t>(STRATA_INKLING2_MAX_BATCH_SIZE) * sizeof(SikBatchSlotMetaHost));
        eng->b_batch_next_tokens = sik_alloc_shared_prefaulted(
            eng->device, static_cast<size_t>(STRATA_INKLING2_MAX_BATCH_SIZE) * sizeof(uint32_t));
        eng->b_batch_pen_items = sik_alloc_shared_prefaulted(
            eng->device, static_cast<size_t>(STRATA_INKLING2_MAX_BATCH_SIZE) * sizeof(SikBatchPenItemHost));
        eng->b_batch_sample_items = sik_alloc_shared_prefaulted(
            eng->device, static_cast<size_t>(STRATA_INKLING2_MAX_BATCH_SIZE) * sizeof(SikBatchSampleItemHost));
        if (!eng->b_batch_slot_meta || !eng->b_batch_next_tokens ||
            !eng->b_batch_pen_items || !eng->b_batch_sample_items) {
            return false;
        }
        track_add(eng->b_batch_slot_meta);
        track_add(eng->b_batch_next_tokens);
        track_add(eng->b_batch_pen_items);
        track_add(eng->b_batch_sample_items);
    }

    const uint32_t need_pen_cap = std::max<uint32_t>(256U, (total_pen_entries + 255U) & ~255U);
    if (!eng->b_batch_pen_toks || eng->batch_pen_cap < need_pen_cap) {
        id<MTLBuffer> new_toks = sik_alloc_shared_prefaulted(
            eng->device, static_cast<size_t>(need_pen_cap) * sizeof(uint32_t));
        id<MTLBuffer> new_cnts = sik_alloc_shared_prefaulted(
            eng->device, static_cast<size_t>(need_pen_cap) * sizeof(uint16_t));
        if (!new_toks || !new_cnts) return false;
        track_rm(eng->b_batch_pen_toks);
        track_rm(eng->b_batch_pen_cnts);
        eng->b_batch_pen_toks = new_toks;
        eng->b_batch_pen_cnts = new_cnts;
        eng->batch_pen_cap    = need_pen_cap;
        track_add(new_toks);
        track_add(new_cnts);
    }

    if (B > kMaxSpecBatch) {
        if (!eng->b_batch_part_m) {
            const size_t   part_base = static_cast<size_t>(STRATA_INKLING2_MAX_BATCH_SIZE) * n_q * kMaxDecodeSplits;
            const uint32_t D         = cfg.hidden_size;
            const uint32_t R         = cfg.loop_gate_rank;
            eng->b_batch_part_m         = sik_alloc_private_aligned(eng->device, part_base * sizeof(float));
            eng->b_batch_part_l         = sik_alloc_private_aligned(eng->device, part_base * sizeof(float));
            eng->b_batch_part_o         = sik_alloc_private_aligned(eng->device, part_base * hd * sizeof(float));
            eng->b_batch_sample_v       = sik_alloc_private_aligned(eng->device, 32ULL * 4096ULL * sizeof(float));
            eng->b_batch_sample_i       = sik_alloc_private_aligned(eng->device, 32ULL * 4096ULL * sizeof(uint32_t));
            eng->b_batch_msg_bank12     = sik_alloc_private_aligned(eng->device, 12ULL * 4ULL * STRATA_INKLING2_MAX_BATCH_SIZE * D * sizeof(float));
            eng->b_batch_msg_norm12_f16 = sik_alloc_private_aligned(eng->device, 12ULL * 4ULL * STRATA_INKLING2_MAX_BATCH_SIZE * D * sizeof(uint16_t));
            eng->b_batch_ctrl_h_msg12   = sik_alloc_private_aligned(eng->device, 12ULL * 4ULL * STRATA_INKLING2_MAX_BATCH_SIZE * R * sizeof(float));
            if (!eng->b_batch_part_m || !eng->b_batch_part_l || !eng->b_batch_part_o ||
                !eng->b_batch_sample_v || !eng->b_batch_sample_i ||
                !eng->b_batch_msg_bank12 || !eng->b_batch_msg_norm12_f16 || !eng->b_batch_ctrl_h_msg12) {
                return false;
            }
            track_add(eng->b_batch_part_m);
            track_add(eng->b_batch_part_l);
            track_add(eng->b_batch_part_o);
            track_add(eng->b_batch_sample_v);
            track_add(eng->b_batch_sample_i);
            track_add(eng->b_batch_msg_bank12);
            track_add(eng->b_batch_msg_norm12_f16);
            track_add(eng->b_batch_ctrl_h_msg12);
        }
        const uint32_t want_rows = (B <= 16U) ? 16U : 32U;
        if (!eng->b_batch_logits || eng->batch_logits_rows < want_rows) {
            id<MTLBuffer> new_logits = sik_alloc_private_aligned(
                eng->device, static_cast<size_t>(want_rows) * V * sizeof(float));
            if (!new_logits) return false;
            track_rm(eng->b_batch_logits);
            eng->b_batch_logits    = new_logits;
            eng->batch_logits_rows = want_rows;
            track_add(new_logits);
        }
    }

    if (eng->residency_set && (newly_added != nil || removed != nil)) {
        if (@available(macOS 15.0, *)) {
            for (id<MTLBuffer> b in removed)     [eng->residency_set removeAllocation:b];
            for (id<MTLBuffer> b in newly_added) [eng->residency_set addAllocation:b];
            [eng->residency_set commit];
        }
    }
    return true;
}

int sik_decode_batch_homogeneous(
    StrataInkling2GpuEngine* eng,
    StrataInkling2BatchItem* items,
    const uint32_t*          sub_idx,
    uint32_t                 B,
    bool                     use_alt_wo_sub
) {
    if (B == 0U || B > STRATA_INKLING2_MAX_BATCH_SIZE) return -1;

    const StrataInkling2Config& cfg = eng->cfg;
    const uint32_t D         = cfg.hidden_size;
    const uint32_t n_q       = cfg.num_attention_heads;
    const uint32_t n_kv      = cfg.num_key_value_heads;
    const uint32_t hd        = cfg.head_dim;
    const uint32_t inter     = cfg.intermediate_size;
    const uint32_t E         = cfg.num_experts;
    const uint32_t top_k     = cfg.num_experts_per_tok;
    const uint32_t V         = cfg.vocab_size_unpadded;
    const uint32_t core_beg  = cfg.loop_core_start;
    const uint32_t core_end  = cfg.loop_core_end;
    const uint32_t n_visits  = cfg.num_logical_layers;
    const size_t   cur_bytes = static_cast<size_t>(B) * D * sizeof(float);

    const bool     is_batch_nax   = (eng->has_nax && B > kMaxSpecBatch);
    const bool     use_int8_batch = false; // Always preserve 100% bit-exact FP16-act decode even with --int8-prefill!
    const bool     use_exact_b64  = true;
    const uint32_t M_padded       = is_batch_nax ? ((B <= 16U) ? 16U : ((B <= 32U) ? 32U : 64U)) : B;
    const bool     force_gemv     = !is_batch_nax;
    const uint32_t n_full         = B / kMaxSpecBatch;
    const uint32_t rem            = B % kMaxSpecBatch;
    const uint32_t m_rem          = n_full * kMaxSpecBatch;

    // 1. Record incoming tokens into each slot's own `pen_hist` & pack sparse per-stream penalty lists
    SikBatchPenItemHost pen_items_host[STRATA_INKLING2_MAX_BATCH_SIZE]{};
    std::vector<uint32_t> tmp_pen_toks;
    std::vector<uint16_t> tmp_pen_cnts;

    for (uint32_t b = 0U; b < B; ++b) {
        StrataInkling2BatchItem&   it = items[sub_idx[b]];
        StrataInkling2ContextSlot* sl = it.slot;
        sl->pen_hist.push_back(it.token_id);

        if (sik_penalties_active(it.sampling)) {
            if (eng->batch_pen_scratch.size() < V) eng->batch_pen_scratch.assign(V, 0U);
            uint16_t* const scratch = eng->batch_pen_scratch.data();
            const uint32_t eff_window = (it.sampling->penalty_window == 0U)
                ? 0U : std::max<uint32_t>(it.sampling->penalty_window, 8U);
            const uint32_t L = static_cast<uint32_t>(sl->pen_hist.size());
            const uint32_t start = (eff_window > 0U && L > eff_window) ? (L - eff_window) : 0U;
            const uint32_t off   = static_cast<uint32_t>(tmp_pen_toks.size());
            for (uint32_t i = start; i < L; ++i) {
                const uint32_t t = sl->pen_hist[i];
                if (t < V) {
                    if (scratch[t] == 0U) tmp_pen_toks.push_back(t);
                    if (scratch[t] < 65535U) ++scratch[t];
                }
            }
            const uint32_t len = static_cast<uint32_t>(tmp_pen_toks.size()) - off;
            tmp_pen_cnts.reserve(tmp_pen_toks.size());
            for (uint32_t k = 0U; k < len; ++k) {
                const uint32_t t = tmp_pen_toks[off + k];
                tmp_pen_cnts.push_back(scratch[t]);
                scratch[t] = 0U;
            }
            pen_items_host[b] = SikBatchPenItemHost{
                off, len,
                it.sampling->presence_penalty,
                it.sampling->frequency_penalty,
                it.sampling->repetition_penalty,
                0U
            };
        } else {
            pen_items_host[b] = SikBatchPenItemHost{0U, 0U, 0.0f, 0.0f, 1.0f, 0U};
        }
    }

    if (!sik_ensure_batch_workspaces(eng, B, static_cast<uint32_t>(tmp_pen_toks.size()))) {
        std::fprintf(stderr, "[strata_inkling2_gpu] Failed allocating continuous-batch workspaces (B=%u)\n", B);
        return -1;
    }

    // 2. Populate input token IDs, slot GPU pointer table, penalty items, sampling items, and attention split bounds
    uint32_t*               tok_ptr = reinterpret_cast<uint32_t*>([eng->b_token_ids contents]);
    SikBatchSlotMetaHost*   sm_ptr  = reinterpret_cast<SikBatchSlotMetaHost*>([eng->b_batch_slot_meta contents]);
    SikBatchPenItemHost*    pi_ptr  = reinterpret_cast<SikBatchPenItemHost*>([eng->b_batch_pen_items contents]);
    SikBatchSampleItemHost* si_ptr  = reinterpret_cast<SikBatchSampleItemHost*>([eng->b_batch_sample_items contents]);

    std::memcpy(pi_ptr, pen_items_host, static_cast<size_t>(B) * sizeof(SikBatchPenItemHost));
    if (!tmp_pen_toks.empty()) {
        std::memcpy([eng->b_batch_pen_toks contents], tmp_pen_toks.data(), tmp_pen_toks.size() * sizeof(uint32_t));
        std::memcpy([eng->b_batch_pen_cnts contents], tmp_pen_cnts.data(), tmp_pen_cnts.size() * sizeof(uint16_t));
    }

    std::vector<id<MTLResource>> slot_res;
    slot_res.reserve(static_cast<size_t>(B) * 5U);

    bool     has_swa_under256  = false;
    bool     has_swa_ge256     = false;
    uint32_t max_swa_splits    = 0U;
    bool     has_glob_under256 = false;
    bool     has_glob_ge256    = false;
    uint32_t max_glob_splits   = 0U;
    uint32_t max_glob_q8_splits = 0U;

    for (uint32_t b = 0U; b < B; ++b) {
        const StrataInkling2BatchItem& it = items[sub_idx[b]];
        StrataInkling2ContextSlot*     sl = it.slot;
        const uint32_t                 qp = sl->seq_pos;

        tok_ptr[b] = it.token_id;
        sm_ptr[b]  = SikBatchSlotMetaHost{
            [sl->b_swa_k_cache gpuAddress],
            [sl->b_swa_v_cache gpuAddress],
            [sl->b_glob_k_cache gpuAddress],
            [sl->b_glob_v_cache gpuAddress],
            [sl->b_sconv_state gpuAddress],
            qp,
            0U
        };
        slot_res.push_back(sl->b_swa_k_cache);
        slot_res.push_back(sl->b_swa_v_cache);
        slot_res.push_back(sl->b_glob_k_cache);
        slot_res.push_back(sl->b_glob_v_cache);
        slot_res.push_back(sl->b_sconv_state);

        if (qp < 256U) {
            has_swa_under256 = true;
        } else {
            has_swa_ge256 = true;
            const uint32_t B_c = sik_host_attn_chunk_size(qp, 1U);
            const uint32_t ws  = (qp + 1U > cfg.sliding_window_size) ? (qp + 1U - cfg.sliding_window_size) : 0U;
            const uint32_t sp  = (qp / B_c) - (ws / B_c) + 1U;
            if (sp > max_swa_splits) max_swa_splits = sp;
        }
        {
            const uint32_t B_cg = sik_host_attn_chunk_size(qp, 0U);
            const uint32_t spg  = (qp / B_cg) + 1U;
            if (spg > max_glob_q8_splits) max_glob_q8_splits = spg;
            if (qp < 256U) {
                has_glob_under256 = true;
            } else {
                has_glob_ge256 = true;
                if (spg > max_glob_splits) max_glob_splits = spg;
            }
        }

        const StrataInkling2SamplingParams* sp = it.sampling;
        const bool is_greedy = (!sp) || sp->greedy || (sp->temperature <= 0.0f);
        if (is_greedy) {
            si_ptr[b] = SikBatchSampleItemHost{1U, 1.0f, 1U, 1.0f, 0U, 0U, 0U, 0U};
        } else {
            sl->sample_step     = static_cast<uint64_t>(sl->seq_pos + 1U) << 8U;
            const uint64_t seed = sp->rng_seed ^ (sl->sample_step * 0xD1342543DE82EF95ULL);
            sl->sample_step += 1ULL;
            si_ptr[b] = SikBatchSampleItemHost{
                0U,
                1.0f / std::max<float>(sp->temperature, 1e-5f),
                (sp->top_k > 0U && sp->top_k <= 64U) ? sp->top_k : 20U,
                sp->top_p,
                static_cast<uint32_t>(seed & 0xFFFFFFFFULL),
                static_cast<uint32_t>(seed >> 32ULL),
                0U,
                0U
            };
        }
    }

    // 3. Encode Entire 48-Visit Forward Pass + Batched Head + Penalties + Sampling in ONE Command Buffer!
    eng->alt_wo_active = use_alt_wo_sub && (eng->b_alt_wo != nil) && (eng->alt_wo_layers > 0U);
    const bool prev_prof = eng->prof_active;
    eng->prof_active = false;

    id<MTLCommandBuffer>         cb  = [eng->queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    auto bind_slot_resources = [&](id<MTLComputeCommandEncoder> e) {
        [e useResources:slot_res.data()
                  count:slot_res.size()
                  usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
    };
    bind_slot_resources(enc);

    // 3a. Embedding (FP16 or Q4_A) + embed_norm -> `b_cur [B, D]` F32
    {
        SikEmbedParamsHost ep{B, D, cfg.vocab_size_padded, cfg.rms_norm_eps};
        if (eng->hdr.embed_tokens.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
            uint64_t ew = 0ULL, es = 0ULL, eb = 0ULL;
            sik_slice_sub_offsets(eng->hdr.embed_tokens, ew, es, eb);
            [enc setComputePipelineState:eng->pso_embed_rmsnorm_q4a];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ew) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(es) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eb) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_norm.offset) atIndex:3];
            [enc setBuffer:eng->b_token_ids offset:0 atIndex:4];
            [enc setBuffer:eng->b_cur offset:0 atIndex:5];
            [enc setBytes:&ep length:sizeof(ep) atIndex:6];
        } else {
            [enc setComputePipelineState:eng->pso_embed_rmsnorm];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.offset) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.embed_norm.offset) atIndex:1];
            [enc setBuffer:eng->b_token_ids offset:0 atIndex:2];
            [enc setBuffer:eng->b_cur offset:0 atIndex:3];
            [enc setBytes:&ep length:sizeof(ep) atIndex:4];
        }
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    uint32_t num_blocks         = 0U;
    uint32_t prelude_num_blocks = 0U;
    uint32_t bank_count         = 0U;
    const uint32_t core_len     = (core_end > core_beg) ? (core_end - core_beg) : 0U;
    bool use_batch12_loop       = (use_exact_b64 &&
                                   D == 2304U && core_len > 0U && core_len <= 12U && cfg.loop_num_passes > 1U);
    if (use_batch12_loop) {
        for (uint32_t li = 0U; li < core_len; ++li) {
            const StrataInkling2LayerDesc& cld = eng->hdr.layers[core_beg + li];
            for (uint32_t j = 0U; j < 4U; ++j) {
                if (cld.controllers[j].down_weight.dtype != STRATA_INKLING2_DTYPE_Q8_0) {
                    use_batch12_loop = false;
                    break;
                }
            }
            if (!use_batch12_loop) break;
        }
    }

    id<MTLBuffer> part_m_buf       = (B <= kMaxSpecBatch) ? eng->b_part_m         : eng->b_batch_part_m;
    id<MTLBuffer> part_l_buf       = (B <= kMaxSpecBatch) ? eng->b_part_l         : eng->b_batch_part_l;
    id<MTLBuffer> part_o_buf       = (B <= kMaxSpecBatch) ? eng->b_part_o         : eng->b_batch_part_o;
    id<MTLBuffer> msg_bank12_buf   = (B <= kMaxSpecBatch) ? eng->b_msg_bank12     : eng->b_batch_msg_bank12;
    id<MTLBuffer> msg_norm12_buf   = (B <= kMaxSpecBatch) ? eng->b_msg_norm12_f16 : eng->b_batch_msg_norm12_f16;
    id<MTLBuffer> ctrl_h_msg12_buf = (B <= kMaxSpecBatch) ? eng->b_ctrl_h_msg12   : eng->b_batch_ctrl_h_msg12;

    // 3b. 48 Logical Layer Visits (`il = 0..47`)
    for (uint32_t il = 0U; il < n_visits; ++il) {
        const uint32_t ip       = cfg.logical_to_phys[il];
        const uint32_t pass     = cfg.logical_pass_idx[il];
        const bool     is_local = (cfg.logical_is_local[il] != 0U);
        const uint32_t kv_slot  = cfg.logical_kv_slot[il];
        const StrataInkling2LayerDesc& ld = eng->hdr.layers[ip];

        // Loop Core Pass Boundary (`ip == core_start`)
        if (cfg.loop_num_passes > 1U && ip >= core_beg && ip < core_end && ip == core_beg) {
            if (pass == 0U) {
                prelude_num_blocks = num_blocks;
                sik_encode_copy_buffer(enc, eng->pso_copy_u32,
                                       eng->b_cur, 0, eng->b_prelude_cur, 0, cur_bytes);
            } else {
                const uint32_t src_start = prelude_num_blocks + 1U;
                bank_count = (num_blocks > src_start) ? std::min<uint32_t>(num_blocks - src_start, kMaxBankBlocks) : 0U;
                SikPassCopyParamsHost cp{
                    B, D, bank_count, src_start,
                    static_cast<uint32_t>(eng->block_slot_elems), 0U, 0U, 0U
                };
                [enc setComputePipelineState:eng->pso_pass_boundary_copy];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_prelude_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_cur offset:0 atIndex:2];
                [enc setBuffer:eng->b_bank offset:0 atIndex:3];
                [enc setBuffer:eng->b_bank_last offset:0 atIndex:4];
                [enc setBytes:&cp length:sizeof(cp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(((D / 4U) + 255U) / 256U, B, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                num_blocks = prelude_num_blocks;

                if (use_batch12_loop) {
                    SikAttnResParamsHost bp{
                        B, D, bank_count, static_cast<uint32_t>(eng->block_slot_elems),
                        core_len, cfg.rms_norm_eps, 0U, 0U
                    };
                    [enc setComputePipelineState:eng->pso_cross_pass_batch12_576];
                    [enc setBuffer:eng->b_bank offset:0 atIndex:0];
                    [enc setBuffer:eng->b_bank_last offset:0 atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_queries.offset) atIndex:2];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_kv_queries.offset) atIndex:3];
                    [enc setBuffer:msg_bank12_buf offset:0 atIndex:4];
                    [enc setBuffer:msg_norm12_buf offset:0 atIndex:5];
                    [enc setBytes:&bp length:sizeof(bp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(B, core_len, 1)
                        threadsPerThreadgroup:MTLSizeMake(576, 1, 1)];

                    SikLoopBatch12OffsetsHost bo{};
                    bo.M_valid    = B;
                    bo.D          = D;
                    bo.gate_rank  = cfg.loop_gate_rank;
                    bo.num_layers = core_len;
                    for (uint32_t li = 0U; li < core_len; ++li) {
                        const StrataInkling2LayerDesc& cld = eng->hdr.layers[core_beg + li];
                        for (uint32_t j = 0U; j < 4U; ++j) {
                            uint64_t dummy_b = 0ULL;
                            sik_slice_sub_offsets(cld.controllers[j].down_weight, bo.w_off[li][j], bo.s_off[li][j], dummy_b);
                        }
                    }
                    if (B <= kMaxSpecBatch) {
                        [enc setComputePipelineState:eng->pso_loop_ctrl_down_right_batch12_q8_m[B - 1U]];
                        [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                        [enc setBuffer:msg_norm12_buf offset:0 atIndex:1];
                        [enc setBuffer:ctrl_h_msg12_buf offset:0 atIndex:2];
                        [enc setBytes:&bo length:sizeof(bo) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(cfg.loop_gate_rank, 4, core_len)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } else {
                        if (n_full > 0U) {
                            const uint32_t m_off_0 = 0U;
                            [enc setComputePipelineState:eng->pso_loop_ctrl_down_right_batch12_q8_2d_m[kMaxSpecBatch - 1U]];
                            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                            [enc setBuffer:msg_norm12_buf offset:0 atIndex:1];
                            [enc setBuffer:ctrl_h_msg12_buf offset:0 atIndex:2];
                            [enc setBytes:&bo length:sizeof(bo) atIndex:3];
                            [enc setBytes:&m_off_0 length:sizeof(m_off_0) atIndex:4];
                            [enc dispatchThreadgroups:MTLSizeMake(n_full, cfg.loop_gate_rank, core_len * 4U)
                                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        }
                        if (rem > 0U) {
                            const uint32_t m_off_r = m_rem;
                            [enc setComputePipelineState:eng->pso_loop_ctrl_down_right_batch12_q8_2d_m[rem - 1U]];
                            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                            [enc setBuffer:msg_norm12_buf offset:0 atIndex:1];
                            [enc setBuffer:ctrl_h_msg12_buf offset:0 atIndex:2];
                            [enc setBytes:&bo length:sizeof(bo) atIndex:3];
                            [enc setBytes:&m_off_r length:sizeof(m_off_r) atIndex:4];
                            [enc dispatchThreadgroups:MTLSizeMake(1, cfg.loop_gate_rank, core_len * 4U)
                                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        }
                    }
                }
            }
        }

        // Cross-Pass Messages + Pre-Attention Block AttnRes + Loop Controllers 0..2 + Input RMSNorms
        const bool has_msgs       = (pass > 0U && ip >= core_beg && ip < core_end);
        const bool has_attn_res_q = (ip > 0U && ld.attn_res_query.elem_count > 0U);
        const bool fused_pre_attn = (has_msgs && has_attn_res_q && (B <= kMaxSpecBatch || use_batch12_loop));
        if (has_msgs && !fused_pre_attn) {
            const uint64_t row_off = static_cast<uint64_t>(2U * (ip - core_beg)) * D * sizeof(float);
            SikAttnResParamsHost ap{
                B, D, bank_count, static_cast<uint32_t>(eng->block_slot_elems),
                4U, cfg.rms_norm_eps, 0U, 0U
            };
            [enc setComputePipelineState:eng->pso_cross_pass_attn_res4];
            [enc setBuffer:eng->b_bank offset:0 atIndex:0];
            [enc setBuffer:eng->b_bank_last offset:0 atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_queries.offset + row_off) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_kv_queries.offset + row_off) atIndex:3];
            [enc setBuffer:eng->b_messages[0] offset:0 atIndex:4];
            [enc setBuffer:eng->b_messages[1] offset:0 atIndex:5];
            [enc setBuffer:eng->b_messages[2] offset:0 atIndex:6];
            [enc setBuffer:eng->b_messages[3] offset:0 atIndex:7];
            [enc setBytes:&ap length:sizeof(ap) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        id<MTLBuffer> in_q = eng->b_in_f16[0];
        id<MTLBuffer> in_k = eng->b_in_f16[1];
        id<MTLBuffer> in_v = eng->b_in_f16[2];
        id<MTLBuffer> xs_q = eng->b_x_blk_sum[0];
        id<MTLBuffer> xs_k = eng->b_x_blk_sum[1];
        id<MTLBuffer> xs_v = eng->b_x_blk_sum[2];

        if (has_attn_res_q && !has_msgs) {
            if (is_local) {
                sik_encode_attn_res_rmsnorm(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                            ld.attn_res_query.offset, ld.attn_norm.offset,
                                            eng->b_in_f16[0], nil, false, B, M_padded);
                in_k = in_q; in_v = in_q;
                xs_k = xs_q; xs_v = xs_q;
            } else {
                SikAttnResNormParamsHost ap{
                    B, M_padded, D, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    cfg.rms_norm_eps, 0U, 0U, 0U, 0U, 0U, 0U
                };
                const bool use_576 = (D == 2304U);
                [enc setComputePipelineState:(use_576 ? eng->pso_attn_res3_rmsnorm3_576 : eng->pso_attn_res3_rmsnorm3)];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_res_query.offset) atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.q_input_norm.offset) atIndex:3];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.k_input_norm.offset) atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.v_input_norm.offset) atIndex:5];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:6];
                [enc setBuffer:eng->b_in_f16[1] offset:0 atIndex:7];
                [enc setBuffer:eng->b_in_f16[2] offset:0 atIndex:8];
                [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:10];
                [enc setBuffer:eng->b_x_blk_sum[1] offset:0 atIndex:11];
                [enc setBuffer:eng->b_x_blk_sum[2] offset:0 atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake(M_padded, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(use_576 ? 576 : 256, 1, 1)];
            }
        } else {
            id<MTLBuffer> inp_src[3] = {eng->b_cur, eng->b_cur, eng->b_cur};
            const bool batch12_here = (fused_pre_attn && use_batch12_loop);
            if (batch12_here) {
                SikAttnResParamsHost lp{
                    B, D, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    is_local ? 1U : 3U, cfg.rms_norm_eps, 3U, 0U
                };
                id<MTLBuffer> i0 = is_local ? eng->b_x_f32 : eng->b_inputs[0];
                id<MTLBuffer> i1 = is_local ? eng->b_x_f32 : eng->b_inputs[1];
                id<MTLBuffer> i2 = is_local ? eng->b_x_f32 : eng->b_inputs[2];
                [enc setComputePipelineState:eng->pso_attn_res_prep_left_576];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_res_query.offset) atIndex:2];
                [enc setBuffer:i0 offset:0 atIndex:3];
                [enc setBuffer:i1 offset:0 atIndex:4];
                [enc setBuffer:i2 offset:0 atIndex:5];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:6];
                [enc setBytes:&lp length:sizeof(lp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(576, 1, 1)];
                inp_src[0] = i0; inp_src[1] = i1; inp_src[2] = i2;
            } else if (fused_pre_attn) {
                const uint64_t row_off = static_cast<uint64_t>(2U * (ip - core_beg)) * D * sizeof(float);
                SikLoopPreAttnParamsHost lp{
                    B, D, bank_count, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    is_local ? 1U : 3U, cfg.rms_norm_eps, 0U
                };
                id<MTLBuffer> i0 = is_local ? eng->b_x_f32 : eng->b_inputs[0];
                id<MTLBuffer> i1 = is_local ? eng->b_x_f32 : eng->b_inputs[1];
                id<MTLBuffer> i2 = is_local ? eng->b_x_f32 : eng->b_inputs[2];
                [enc setComputePipelineState:eng->pso_loop_pre_attn_prep3];
                [enc setBuffer:eng->b_bank offset:0 atIndex:0];
                [enc setBuffer:eng->b_bank_last offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_queries.offset + row_off) atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cross_pass_kv_queries.offset + row_off) atIndex:3];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:4];
                [enc setBuffer:eng->b_cur offset:0 atIndex:5];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_res_query.offset) atIndex:6];
                [enc setBuffer:eng->b_messages[0] offset:0 atIndex:7];
                [enc setBuffer:eng->b_messages[1] offset:0 atIndex:8];
                [enc setBuffer:eng->b_messages[2] offset:0 atIndex:9];
                [enc setBuffer:eng->b_messages[3] offset:0 atIndex:10];
                [enc setBuffer:i0 offset:0 atIndex:11];
                [enc setBuffer:i1 offset:0 atIndex:12];
                [enc setBuffer:i2 offset:0 atIndex:13];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:14];
                [enc setBytes:&lp length:sizeof(lp) atIndex:15];
                [enc dispatchThreadgroups:MTLSizeMake(B, 2, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                inp_src[0] = i0; inp_src[1] = i1; inp_src[2] = i2;
            } else if (has_attn_res_q) {
                if (is_local) {
                    sik_encode_attn_res(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                        ld.attn_res_query.offset, 1U, eng->b_x_f32, nil, nil, B);
                    inp_src[0] = inp_src[1] = inp_src[2] = eng->b_x_f32;
                } else {
                    sik_encode_attn_res(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                        ld.attn_res_query.offset, 3U,
                                        eng->b_inputs[0], eng->b_inputs[1], eng->b_inputs[2], B);
                    inp_src[0] = eng->b_inputs[0];
                    inp_src[1] = eng->b_inputs[1];
                    inp_src[2] = eng->b_inputs[2];
                }
            }
            const bool q8_pre_ctrl = (has_msgs && ld.controllers[0].down_weight.dtype == STRATA_INKLING2_DTYPE_Q8_0);
            if (has_msgs) {
                if (batch12_here) {
                    const uint32_t   li    = ip - core_beg;
                    const uint32_t   R     = cfg.loop_gate_rank;
                    const NSUInteger mslot = static_cast<NSUInteger>(B) * D * sizeof(float);
                    const NSUInteger hslot = static_cast<NSUInteger>(B) * R * sizeof(float);
                    if (B <= kMaxSpecBatch) {
                        sik_encode_loop_ctrl_batch(
                            cb, enc, eng, &ld.controllers[0], 3U,
                            inp_src[0], inp_src[1], inp_src[2],
                            msg_bank12_buf, msg_bank12_buf, msg_bank12_buf,
                            eng->b_inputs[0], eng->b_inputs[1], eng->b_inputs[2],
                            B, M_padded, force_gemv, /*skip_prep=*/true,
                            static_cast<NSUInteger>(li * 4U + 0U) * mslot,
                            static_cast<NSUInteger>(li * 4U + 1U) * mslot,
                            static_cast<NSUInteger>(li * 4U + 2U) * mslot,
                            ctrl_h_msg12_buf,
                            static_cast<NSUInteger>(li * 4U + 0U) * hslot
                        );
                    } else {
                        const uint32_t grid_sq = (D + 31U) / 32U;
                        SikLoopCtrl3OffsetsHost down_off{}, apply_off{};
                        down_off.M_valid   = apply_off.M_valid   = B;
                        down_off.M_padded  = apply_off.M_padded  = B;
                        down_off.D         = apply_off.D         = D;
                        down_off.gate_rank = apply_off.gate_rank = R;
                        down_off.num_ctrls = apply_off.num_ctrls = 3U;
                        for (uint32_t j = 0U; j < 3U; ++j) {
                            uint64_t dw = 0ULL, ds = 0ULL, db = 0ULL;
                            uint64_t gw = 0ULL, gs = 0ULL, gb = 0ULL;
                            uint64_t fw = 0ULL, fs = 0ULL, fb = 0ULL;
                            sik_slice_sub_offsets(ld.controllers[j].down_weight,   dw, ds, db);
                            sik_slice_sub_offsets(ld.controllers[j].gain_weight,   gw, gs, gb);
                            sik_slice_sub_offsets(ld.controllers[j].forget_weight, fw, fs, fb);
                            down_off.w_off[j]   = dw; down_off.s_off[j]   = ds;
                            apply_off.w_off[j]  = gw; apply_off.s_off[j]  = gs;
                            apply_off.f_off[j]  = fw; apply_off.fs_off[j] = fs;
                        }
                        const NSUInteger h_li_off = static_cast<NSUInteger>(li * 4U) * hslot;
                        if (n_full > 0U) {
                            const uint32_t m_off_0 = 0U;
                            [enc setComputePipelineState:eng->pso_loop_ctrl_down3_left_q8_2d_m[kMaxSpecBatch - 1U]];
                            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                            [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:1];
                            [enc setBuffer:ctrl_h_msg12_buf offset:h_li_off atIndex:2];
                            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:3];
                            [enc setBytes:&down_off length:sizeof(down_off) atIndex:4];
                            [enc setBytes:&m_off_0 length:sizeof(m_off_0) atIndex:5];
                            [enc dispatchThreadgroups:MTLSizeMake(n_full, R, 3)
                                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                        }
                        if (rem > 0U) {
                            const uint32_t m_off_r = m_rem;
                            [enc setComputePipelineState:eng->pso_loop_ctrl_down3_left_q8_2d_m[rem - 1U]];
                            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                            [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:1];
                            [enc setBuffer:ctrl_h_msg12_buf offset:h_li_off atIndex:2];
                            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:3];
                            [enc setBytes:&down_off length:sizeof(down_off) atIndex:4];
                            [enc setBytes:&m_off_r length:sizeof(m_off_r) atIndex:5];
                            [enc dispatchThreadgroups:MTLSizeMake(1, R, 3)
                                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                        }
                        if (n_full > 0U) {
                            [enc setComputePipelineState:eng->pso_loop_ctrl_apply3_q8_m[kMaxSpecBatch - 1U]];
                            [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                            [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:1];
                            [enc setBuffer:inp_src[0] offset:0 atIndex:2];
                            [enc setBuffer:inp_src[1] offset:0 atIndex:3];
                            [enc setBuffer:inp_src[2] offset:0 atIndex:4];
                            [enc setBuffer:msg_bank12_buf offset:static_cast<NSUInteger>(li * 4U + 0U) * mslot atIndex:5];
                            [enc setBuffer:msg_bank12_buf offset:static_cast<NSUInteger>(li * 4U + 1U) * mslot atIndex:6];
                            [enc setBuffer:msg_bank12_buf offset:static_cast<NSUInteger>(li * 4U + 2U) * mslot atIndex:7];
                            [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:8];
                            [enc setBuffer:eng->b_inputs[1] offset:0 atIndex:9];
                            [enc setBuffer:eng->b_inputs[2] offset:0 atIndex:10];
                            [enc setBytes:&apply_off length:sizeof(apply_off) atIndex:11];
                            [enc setBuffer:eng->b_rms_sq offset:0 atIndex:12];
                            [enc dispatchThreadgroups:MTLSizeMake(grid_sq, n_full, 3)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }
                        if (rem > 0U) {
                            SikLoopCtrlApplyParamsHost ap{rem, rem, D, R};
                            const NSUInteger d_o = static_cast<NSUInteger>(m_rem) * D * sizeof(float);
                            for (uint32_t j = 0U; j < 3U; ++j) {
                                const NSUInteger h_o  = static_cast<NSUInteger>(j * B + m_rem) * R * sizeof(float);
                                const NSUInteger sq_o = static_cast<NSUInteger>(j * B + m_rem) * grid_sq * sizeof(float);
                                const NSUInteger m_o  = static_cast<NSUInteger>(li * 4U + j) * mslot + d_o;
                                [enc setComputePipelineState:eng->pso_loop_ctrl_apply_q8_m[rem - 1U]];
                                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.w_off[j]) atIndex:0];
                                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.s_off[j]) atIndex:1];
                                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.f_off[j]) atIndex:2];
                                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(apply_off.fs_off[j]) atIndex:3];
                                [enc setBuffer:eng->b_ctrl_h_f32 offset:h_o atIndex:4];
                                [enc setBuffer:inp_src[j] offset:d_o atIndex:5];
                                [enc setBuffer:msg_bank12_buf offset:m_o atIndex:6];
                                [enc setBuffer:eng->b_inputs[j] offset:d_o atIndex:7];
                                [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                                [enc setBuffer:eng->b_rms_sq offset:sq_o atIndex:9];
                                [enc dispatchThreadgroups:MTLSizeMake(grid_sq, 1, 1)
                                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            }
                        }
                    }
                } else {
                    sik_encode_loop_ctrl_batch(
                        cb, enc, eng, &ld.controllers[0], 3U,
                        inp_src[0], inp_src[1], inp_src[2],
                        eng->b_messages[0], eng->b_messages[1], eng->b_messages[2],
                        eng->b_inputs[0], eng->b_inputs[1], eng->b_inputs[2],
                        B, M_padded, force_gemv, /*skip_prep=*/fused_pre_attn
                    );
                }
                inp_src[0] = eng->b_inputs[0];
                inp_src[1] = eng->b_inputs[1];
                inp_src[2] = eng->b_inputs[2];
            }
            if (is_local && !has_msgs) {
                sik_encode_rmsnorm(enc, eng, inp_src[0], 0, ld.attn_norm.offset, true,
                                   eng->b_in_f16[0], nil, false, B, M_padded);
                in_k = in_q; in_v = in_q;
                xs_k = xs_q; xs_v = xs_q;
            } else if ((batch12_here || (q8_pre_ctrl && !is_batch_nax && B <= kMaxSpecBatch))) {
                const uint64_t nw0 = is_local ? ld.attn_norm.offset : ld.q_input_norm.offset;
                const uint64_t nw1 = is_local ? ld.attn_norm.offset : ld.k_input_norm.offset;
                const uint64_t nw2 = is_local ? ld.attn_norm.offset : ld.v_input_norm.offset;
                SikRmsNormParamsHost np{B, M_padded, D, cfg.rms_norm_eps, (D + 31U) / 32U, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_rmsnorm3_fast];
                [enc setBuffer:inp_src[0] offset:0 atIndex:0];
                [enc setBuffer:inp_src[1] offset:0 atIndex:1];
                [enc setBuffer:inp_src[2] offset:0 atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw0) atIndex:3];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw1) atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw2) atIndex:5];
                [enc setBuffer:eng->b_rms_sq offset:0 atIndex:6];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:7];
                [enc setBuffer:eng->b_in_f16[1] offset:0 atIndex:8];
                [enc setBuffer:eng->b_in_f16[2] offset:0 atIndex:9];
                [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:10];
                [enc setBuffer:eng->b_x_blk_sum[1] offset:0 atIndex:11];
                [enc setBuffer:eng->b_x_blk_sum[2] offset:0 atIndex:12];
                [enc setBytes:&np length:sizeof(np) atIndex:13];
                [enc dispatchThreadgroups:MTLSizeMake(D / 64U, 3, B)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                const uint64_t nw0 = is_local ? ld.attn_norm.offset : ld.q_input_norm.offset;
                const uint64_t nw1 = is_local ? ld.attn_norm.offset : ld.k_input_norm.offset;
                const uint64_t nw2 = is_local ? ld.attn_norm.offset : ld.v_input_norm.offset;
                SikRmsNormParamsHost np{B, M_padded, D, cfg.rms_norm_eps, 1U, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_rmsnorm3];
                [enc setBuffer:inp_src[0] offset:0 atIndex:0];
                [enc setBuffer:inp_src[1] offset:0 atIndex:1];
                [enc setBuffer:inp_src[2] offset:0 atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw0) atIndex:3];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw1) atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(nw2) atIndex:5];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:6];
                [enc setBuffer:eng->b_in_f16[1] offset:0 atIndex:7];
                [enc setBuffer:eng->b_in_f16[2] offset:0 atIndex:8];
                [enc setBytes:&np length:sizeof(np) atIndex:9];
                [enc setBuffer:eng->b_x_blk_sum[0] offset:0 atIndex:10];
                [enc setBuffer:eng->b_x_blk_sum[1] offset:0 atIndex:11];
                [enc setBuffer:eng->b_x_blk_sum[2] offset:0 atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake(M_padded, 3, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        // Self-Attention Projections (shared weight reads across all B streams!)
        if (use_exact_b64 || (!is_batch_nax && B <= kMaxSpecBatch)) {
            const StrataInkling2TensorSlice slices[8] = {
                ld.wq, ld.gate_attn, ld.wr, ld.attn_sink_proj,
                ld.wk, ld.wv,
                is_local ? StrataInkling2TensorSlice{} : ld.k_sconv_proj,
                is_local ? StrataInkling2TensorSlice{} : ld.v_sconv_proj
            };
            SikAttnQgrsMixedParamsHost qp{};
            qp.M    = B;
            qp.cols = D;
            uint32_t tot_rows = 0U;
            for (uint32_t j = 0U; j < 8U; ++j) {
                qp.rows[j]  = slices[j].shape0;
                qp.dtype[j] = slices[j].dtype;
                if (slices[j].shape0 > 0U) {
                    sik_slice_sub_offsets(slices[j], qp.w_off[j], qp.s_off[j], qp.b_off[j]);
                    tot_rows += slices[j].shape0;
                }
            }
            if (B <= kMaxSpecBatch) {
                [enc setComputePipelineState:eng->pso_attn_qgrs_mixed_m[B - 1U]];
                [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                [enc setBuffer:in_q offset:0 atIndex:1];
                [enc setBuffer:in_k offset:0 atIndex:2];
                [enc setBuffer:in_v offset:0 atIndex:3];
                [enc setBuffer:eng->b_q_raw offset:0 atIndex:4];
                [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:5];
                [enc setBuffer:eng->b_r_raw offset:0 atIndex:6];
                [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:7];
                [enc setBuffer:eng->b_k_raw offset:0 atIndex:8];
                [enc setBuffer:eng->b_v_raw offset:0 atIndex:9];
                [enc setBuffer:eng->b_k_dyn offset:0 atIndex:10];
                [enc setBuffer:eng->b_v_dyn offset:0 atIndex:11];
                [enc setBytes:&qp length:sizeof(qp) atIndex:12];
                [enc setBuffer:xs_q offset:0 atIndex:13];
                [enc setBuffer:xs_k offset:0 atIndex:14];
                [enc setBuffer:xs_v offset:0 atIndex:15];
                [enc dispatchThreadgroups:MTLSizeMake((tot_rows + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                if (n_full > 0U) {
                    [enc setComputePipelineState:eng->pso_attn_qgrs_mixed_2d];
                    [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                    [enc setBuffer:in_q offset:0 atIndex:1];
                    [enc setBuffer:in_k offset:0 atIndex:2];
                    [enc setBuffer:in_v offset:0 atIndex:3];
                    [enc setBuffer:eng->b_q_raw offset:0 atIndex:4];
                    [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:5];
                    [enc setBuffer:eng->b_r_raw offset:0 atIndex:6];
                    [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:7];
                    [enc setBuffer:eng->b_k_raw offset:0 atIndex:8];
                    [enc setBuffer:eng->b_v_raw offset:0 atIndex:9];
                    [enc setBuffer:eng->b_k_dyn offset:0 atIndex:10];
                    [enc setBuffer:eng->b_v_dyn offset:0 atIndex:11];
                    [enc setBytes:&qp length:sizeof(qp) atIndex:12];
                    [enc setBuffer:xs_q offset:0 atIndex:13];
                    [enc setBuffer:xs_k offset:0 atIndex:14];
                    [enc setBuffer:xs_v offset:0 atIndex:15];
                    [enc dispatchThreadgroups:MTLSizeMake(n_full, (tot_rows + 7U) / 8U, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                if (rem > 0U) {
                    const NSUInteger x_o  = static_cast<NSUInteger>(m_rem) * D * sizeof(uint16_t);
                    const NSUInteger xs_o = static_cast<NSUInteger>(m_rem) * (D >> 6U) * sizeof(float);
                    qp.M = rem;
                    [enc setComputePipelineState:eng->pso_attn_qgrs_mixed_m[rem - 1U]];
                    [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                    [enc setBuffer:in_q offset:x_o atIndex:1];
                    [enc setBuffer:in_k offset:x_o atIndex:2];
                    [enc setBuffer:in_v offset:x_o atIndex:3];
                    [enc setBuffer:eng->b_q_raw offset:static_cast<NSUInteger>(m_rem) * qp.rows[0] * sizeof(float) atIndex:4];
                    [enc setBuffer:eng->b_gate_f32 offset:static_cast<NSUInteger>(m_rem) * qp.rows[1] * sizeof(float) atIndex:5];
                    [enc setBuffer:eng->b_r_raw offset:static_cast<NSUInteger>(m_rem) * qp.rows[2] * sizeof(float) atIndex:6];
                    [enc setBuffer:eng->b_sink_f32 offset:static_cast<NSUInteger>(m_rem) * qp.rows[3] * sizeof(float) atIndex:7];
                    [enc setBuffer:eng->b_k_raw offset:static_cast<NSUInteger>(m_rem) * qp.rows[4] * sizeof(float) atIndex:8];
                    [enc setBuffer:eng->b_v_raw offset:static_cast<NSUInteger>(m_rem) * qp.rows[5] * sizeof(float) atIndex:9];
                    [enc setBuffer:eng->b_k_dyn offset:static_cast<NSUInteger>(m_rem) * qp.rows[6] * sizeof(float) atIndex:10];
                    [enc setBuffer:eng->b_v_dyn offset:static_cast<NSUInteger>(m_rem) * qp.rows[7] * sizeof(float) atIndex:11];
                    [enc setBytes:&qp length:sizeof(qp) atIndex:12];
                    [enc setBuffer:xs_q offset:xs_o atIndex:13];
                    [enc setBuffer:xs_k offset:xs_o atIndex:14];
                    [enc setBuffer:xs_v offset:xs_o atIndex:15];
                    [enc dispatchThreadgroups:MTLSizeMake((tot_rows + 7U) / 8U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            }
        } else {
            [enc endEncoding];
            id<MTLComputeCommandEncoder> cenc_ap = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            const bool ap_i8 = use_int8_batch &&
                (ld.wq.dtype == STRATA_INKLING2_DTYPE_Q4_A ||
                 ld.wq.dtype == STRATA_INKLING2_DTYPE_Q8_0 ||
                 ld.wr.dtype == STRATA_INKLING2_DTYPE_Q8_0);
            if (ap_i8) {
                const bool same_qkv = (in_k == in_q && in_v == in_q);
                if (same_qkv) {
                    SikQuantI8ParamsHost qp_in{B, M_padded, D, 0U, 1U, 0U, 0U, 0U};
                    [cenc_ap setComputePipelineState:eng->pso_quant_f16_to_i8_row];
                    [cenc_ap setBuffer:in_q offset:0 atIndex:0];
                    [cenc_ap setBuffer:eng->b_in_i8[0] offset:0 atIndex:1];
                    [cenc_ap setBuffer:eng->b_in_xsc[0] offset:0 atIndex:2];
                    [cenc_ap setBuffer:eng->b_in_xsum_f16[0] offset:0 atIndex:3];
                    [cenc_ap setBytes:&qp_in length:sizeof(qp_in) atIndex:4];
                    [cenc_ap dispatchThreadgroups:MTLSizeMake((M_padded + 3U) / 4U, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    SikQuantI8ParamsHost qp_in{B, M_padded, D, 0U, 3U, 0U, 0U, 0U};
                    [cenc_ap setComputePipelineState:eng->pso_quant_f16_to_i8_row3];
                    [cenc_ap setBuffer:in_q offset:0 atIndex:0];
                    [cenc_ap setBuffer:in_k offset:0 atIndex:1];
                    [cenc_ap setBuffer:in_v offset:0 atIndex:2];
                    [cenc_ap setBuffer:eng->b_in_i8[0] offset:0 atIndex:3];
                    [cenc_ap setBuffer:eng->b_in_i8[1] offset:0 atIndex:4];
                    [cenc_ap setBuffer:eng->b_in_i8[2] offset:0 atIndex:5];
                    [cenc_ap setBuffer:eng->b_in_xsc[0] offset:0 atIndex:6];
                    [cenc_ap setBuffer:eng->b_in_xsc[1] offset:0 atIndex:7];
                    [cenc_ap setBuffer:eng->b_in_xsc[2] offset:0 atIndex:8];
                    [cenc_ap setBuffer:eng->b_in_xsum_f16[0] offset:0 atIndex:9];
                    [cenc_ap setBuffer:eng->b_in_xsum_f16[1] offset:0 atIndex:10];
                    [cenc_ap setBuffer:eng->b_in_xsum_f16[2] offset:0 atIndex:11];
                    [cenc_ap setBytes:&qp_in length:sizeof(qp_in) atIndex:12];
                    [cenc_ap dispatchThreadgroups:MTLSizeMake((M_padded + 3U) / 4U, 3, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [cenc_ap memoryBarrierWithScope:MTLBarrierScopeBuffers];
                id<MTLBuffer> i8_q   = eng->b_in_i8[0];
                id<MTLBuffer> i8_k   = same_qkv ? eng->b_in_i8[0] : eng->b_in_i8[1];
                id<MTLBuffer> i8_v   = same_qkv ? eng->b_in_i8[0] : eng->b_in_i8[2];
                id<MTLBuffer> sc_q   = eng->b_in_xsc[0];
                id<MTLBuffer> sc_k   = same_qkv ? eng->b_in_xsc[0] : eng->b_in_xsc[1];
                id<MTLBuffer> sc_v   = same_qkv ? eng->b_in_xsc[0] : eng->b_in_xsc[2];
                id<MTLBuffer> xs_q16 = eng->b_in_xsum_f16[0];
                id<MTLBuffer> xs_k16 = same_qkv ? eng->b_in_xsum_f16[0] : eng->b_in_xsum_f16[1];
                id<MTLBuffer> xs_v16 = same_qkv ? eng->b_in_xsum_f16[0] : eng->b_in_xsum_f16[2];

                sik_encode_dense_linear_i8(cenc_ap, eng, ld.wq, in_q, i8_q, sc_q, xs_q16, nil, 0, eng->b_q_raw, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear_i8(cenc_ap, eng, ld.gate_attn, in_q, i8_q, sc_q, xs_q16, nil, 0, eng->b_gate_f32, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear_i8(cenc_ap, eng, ld.wk, in_k, i8_k, sc_k, xs_k16, nil, 0, eng->b_k_raw, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear_i8(cenc_ap, eng, ld.wv, in_v, i8_v, sc_v, xs_v16, nil, 0, eng->b_v_raw, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear_i8(cenc_ap, eng, ld.wr, in_q, i8_q, sc_q, xs_q16, nil, 0, eng->b_r_raw, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.attn_sink_proj, in_q, 0, nil, 0, eng->b_sink_f32, 0, B, M_padded, kOutF32, force_gemv);
                if (!is_local) {
                    sik_encode_dense_linear(cenc_ap, eng, ld.k_sconv_proj, in_k, 0, nil, 0, eng->b_k_dyn, 0, B, M_padded, kOutF32, force_gemv);
                    sik_encode_dense_linear(cenc_ap, eng, ld.v_sconv_proj, in_v, 0, nil, 0, eng->b_v_dyn, 0, B, M_padded, kOutF32, force_gemv);
                }
            } else {
                sik_encode_dense_linear(cenc_ap, eng, ld.wq, in_q, 0, nil, 0, eng->b_q_raw, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.gate_attn, in_q, 0, nil, 0, eng->b_gate_f32, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.wk, in_k, 0, nil, 0, eng->b_k_raw, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.wv, in_v, 0, nil, 0, eng->b_v_raw, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.wr, in_q, 0, nil, 0, eng->b_r_raw, 0, B, M_padded, kOutF32, force_gemv);
                sik_encode_dense_linear(cenc_ap, eng, ld.attn_sink_proj, in_q, 0, nil, 0, eng->b_sink_f32, 0, B, M_padded, kOutF32, force_gemv);
                if (!is_local) {
                    sik_encode_dense_linear(cenc_ap, eng, ld.k_sconv_proj, in_k, 0, nil, 0, eng->b_k_dyn, 0, B, M_padded, kOutF32, force_gemv);
                    sik_encode_dense_linear(cenc_ap, eng, ld.v_sconv_proj, in_v, 0, nil, 0, eng->b_v_dyn, 0, B, M_padded, kOutF32, force_gemv);
                }
            }
            [cenc_ap endEncoding];
            enc = [cb computeCommandEncoder];
            bind_slot_resources(enc);
        }

        // Per-Stream Independent `SConvKV + QKVR-Prep + KV Store` (FP16 & Q8_0!)
        const uint64_t visit_sconv_base = static_cast<uint64_t>(il) * eng->sconv_visit_stride;
        const uint64_t k_sconv_off      = visit_sconv_base;
        const uint64_t v_sconv_off      = k_sconv_off + eng->sconv_k_bytes;
        const uint64_t attn_sconv_off   = v_sconv_off + eng->sconv_v_bytes;
        const uint64_t mlp_sconv_off    = attn_sconv_off + eng->sconv_attn_bytes;
        const uint32_t v_state_off_f4   = static_cast<uint32_t>(eng->sconv_k_bytes / 16ULL);
        const uint32_t kv_stride        = is_local ? kSwaRingLen : eng->max_seq_alloc;
        const uint64_t kv_lyr_off       = static_cast<uint64_t>(kv_slot) *
            (is_local ? eng->swa_kv_layer_stride : eng->glob_kv_layer_stride);
        const bool     use_q8_kv        = (!is_local && eng->kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0);
        const float    attn_scale       = 1.0f / std::sqrt(static_cast<float>(hd));

        {
            SikBatchAttnPrepParamsHost bpp{
                B, n_q, n_kv, hd, cfg.d_rel, kv_stride,
                is_local ? 1U : 0U,
                use_q8_kv ? STRATA_INKLING2_KV_DTYPE_Q8_0 : STRATA_INKLING2_KV_DTYPE_F16,
                static_cast<uint32_t>(k_sconv_off / 16ULL), v_state_off_f4, 0U, 0U,
                kv_lyr_off, eng->glob_kv_q8_data_bytes,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps,
                cfg.log_scaling_alpha, static_cast<float>(cfg.log_scaling_n_floor), 0.0f
            };
            [enc setComputePipelineState:eng->pso_sconv_kv_qkvr_prep_batched];
            [enc setBuffer:eng->b_q_raw offset:0 atIndex:0];
            [enc setBuffer:eng->b_k_raw offset:0 atIndex:1];
            [enc setBuffer:eng->b_v_raw offset:0 atIndex:2];
            [enc setBuffer:eng->b_r_raw offset:0 atIndex:3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.k_sconv_w.offset) atIndex:4];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.v_sconv_w.offset) atIndex:5];
            [enc setBuffer:eng->b_k_dyn offset:0 atIndex:6];
            [enc setBuffer:eng->b_v_dyn offset:0 atIndex:7];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.q_l2_clip_scale.offset) atIndex:8];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_l2_clip_scale.offset) atIndex:9];
            [enc setBuffer:eng->b_q_f32 offset:0 atIndex:10];
            [enc setBuffer:eng->b_r_f32 offset:0 atIndex:11];
            [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:12];
            [enc setBytes:&bpp length:sizeof(bpp) atIndex:13];
            [enc dispatchThreadgroups:MTLSizeMake(2U * n_q + 2U * n_kv, B, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            // Ensure indirect pointer-table KV writes are visible to batched attention reads
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // Per-Stream Independent Batched Attention (SWA + Global, FP16 + Q8_0, single-pass & split-KV!)
        {
            SikBatchAttnParamsHost bap{
                B, n_q, n_kv, hd, cfg.d_rel, cfg.rel_extent, cfg.sliding_window_size,
                is_local ? 1U : 0U, kv_stride, kMaxDecodeSplits,
                use_q8_kv ? 0U : 1U, 0U,
                kv_lyr_off, eng->glob_kv_q8_data_bytes,
                attn_scale, 0U
            };
            if (use_q8_kv) {
                [enc setComputePipelineState:eng->pso_attn_decode_q8_reg_batched];
                [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                [enc setBuffer:part_m_buf offset:0 atIndex:3];
                [enc setBuffer:part_l_buf offset:0 atIndex:4];
                [enc setBuffer:part_o_buf offset:0 atIndex:5];
                [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:6];
                [enc setBytes:&bap length:sizeof(bap) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(max_glob_q8_splits, n_kv, B)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_attn_decode_reduce_batched];
                [enc setBuffer:part_m_buf offset:0 atIndex:0];
                [enc setBuffer:part_l_buf offset:0 atIndex:1];
                [enc setBuffer:part_o_buf offset:0 atIndex:2];
                [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:3];
                [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:4];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:5];
                [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:6];
                [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:7];
                [enc setBytes:&bap length:sizeof(bap) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(n_q, B, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                const bool     has_under  = is_local ? has_swa_under256 : has_glob_under256;
                const bool     has_ge     = is_local ? has_swa_ge256    : has_glob_ge256;
                const uint32_t max_splits = is_local ? max_swa_splits   : max_glob_splits;
                if (has_under) {
                    [enc setComputePipelineState:eng->pso_attn_decode_single_gqa3_batched];
                    [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                    [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                    [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:3];
                    [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:4];
                    [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:5];
                    [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:6];
                    [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:7];
                    [enc setBytes:&bap length:sizeof(bap) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake(n_kv, B, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                if (has_ge) {
                    [enc setComputePipelineState:eng->pso_attn_decode_split_gqa3_batched];
                    [enc setBuffer:eng->b_q_f32 offset:0 atIndex:0];
                    [enc setBuffer:eng->b_r_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.rel_logits_proj.offset) atIndex:2];
                    [enc setBuffer:part_m_buf offset:0 atIndex:3];
                    [enc setBuffer:part_l_buf offset:0 atIndex:4];
                    [enc setBuffer:part_o_buf offset:0 atIndex:5];
                    [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:6];
                    [enc setBytes:&bap length:sizeof(bap) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(max_splits, n_kv, B)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:eng->pso_attn_decode_reduce_batched];
                    [enc setBuffer:part_m_buf offset:0 atIndex:0];
                    [enc setBuffer:part_l_buf offset:0 atIndex:1];
                    [enc setBuffer:part_o_buf offset:0 atIndex:2];
                    [enc setBuffer:eng->b_sink_f32 offset:0 atIndex:3];
                    [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:4];
                    [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:5];
                    [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:6];
                    [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:7];
                    [enc setBytes:&bap length:sizeof(bap) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake(n_q, B, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
            }
        }

        // Output Projection (`wo`, supporting `--no-refusal` `alt_wo`!) + Per-Stream `attn_sconv` + Block Boundary
        const bool is_block_boundary = ((ip % cfg.attn_res_block_size) == 0U);
        const bool use_alt_here      = (eng->alt_wo_active && eng->b_alt_wo != nil && eng->alt_wo_has[ip]);
        id<MTLBuffer> wo_buf         = use_alt_here ? eng->b_alt_wo : eng->b_uma;
        const StrataInkling2TensorSlice& wo_sl = use_alt_here ? eng->alt_wo_slice[ip] : ld.wo;
        const uint32_t   res_mode    = is_block_boundary ? 2U : 1U;
        const NSUInteger blk_off     = is_block_boundary
            ? static_cast<NSUInteger>(num_blocks * eng->block_slot_bytes) : 0U;
        id<MTLBuffer>    blk_buf     = is_block_boundary ? eng->b_blocks : eng->b_cur;

        if ((use_exact_b64 || (!is_batch_nax && B <= kMaxSpecBatch)) && wo_sl.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
            uint64_t ww = 0ULL, ws = 0ULL, wb = 0ULL;
            sik_slice_sub_offsets(wo_sl, ww, ws, wb);
            SikBatchSconvParamsHost bsp{
                B, wo_sl.shape0, res_mode, static_cast<uint32_t>(attn_sconv_off / 16ULL),
                top_k, wo_sl.shape1, 0U, 0U
            };
            if (B <= kMaxSpecBatch) {
                [enc setComputePipelineState:eng->pso_q4a_wo_sconv_batched_m[B - 1U]];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ww) atIndex:0];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ws) atIndex:1];
                [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(wb) atIndex:2];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:3];
                [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:4];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_sconv_w.offset) atIndex:5];
                [enc setBuffer:eng->b_cur offset:0 atIndex:6];
                [enc setBuffer:blk_buf offset:blk_off atIndex:7];
                [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:8];
                [enc setBytes:&bsp length:sizeof(bsp) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake((wo_sl.shape0 + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                if (n_full > 0U) {
                    [enc setComputePipelineState:eng->pso_q4a_wo_sconv_batched_2d];
                    [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ww) atIndex:0];
                    [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ws) atIndex:1];
                    [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(wb) atIndex:2];
                    [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:3];
                    [enc setBuffer:eng->b_wo_blk_sum offset:0 atIndex:4];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_sconv_w.offset) atIndex:5];
                    [enc setBuffer:eng->b_cur offset:0 atIndex:6];
                    [enc setBuffer:blk_buf offset:blk_off atIndex:7];
                    [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:8];
                    [enc setBytes:&bsp length:sizeof(bsp) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake(n_full, (wo_sl.shape0 + 7U) / 8U, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                if (rem > 0U) {
                    const NSUInteger xo   = static_cast<NSUInteger>(m_rem) * wo_sl.shape1 * sizeof(uint16_t);
                    const NSUInteger xso  = static_cast<NSUInteger>(m_rem) * (wo_sl.shape1 >> 6U) * sizeof(float);
                    const NSUInteger curo = static_cast<NSUInteger>(m_rem) * wo_sl.shape0 * sizeof(float);
                    const NSUInteger smo  = static_cast<NSUInteger>(m_rem) * sizeof(SikBatchSlotMetaHost);
                    bsp.B = rem;
                    [enc setComputePipelineState:eng->pso_q4a_wo_sconv_batched_m[rem - 1U]];
                    [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ww) atIndex:0];
                    [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(ws) atIndex:1];
                    [enc setBuffer:wo_buf offset:static_cast<NSUInteger>(wb) atIndex:2];
                    [enc setBuffer:eng->b_attn_gated_f16 offset:xo atIndex:3];
                    [enc setBuffer:eng->b_wo_blk_sum offset:xso atIndex:4];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_sconv_w.offset) atIndex:5];
                    [enc setBuffer:eng->b_cur offset:curo atIndex:6];
                    [enc setBuffer:blk_buf offset:blk_off + curo atIndex:7];
                    [enc setBuffer:eng->b_batch_slot_meta offset:smo atIndex:8];
                    [enc setBytes:&bsp length:sizeof(bsp) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake((wo_sl.shape0 + 7U) / 8U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            }
        } else {
            if (use_int8_batch && (wo_sl.dtype == STRATA_INKLING2_DTYPE_Q4_A || wo_sl.dtype == STRATA_INKLING2_DTYPE_Q8_0)) {
                SikQuantI8ParamsHost qp_wo{B, M_padded, wo_sl.shape1, 0U, 1U, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_quant_f16_to_i8_row];
                [enc setBuffer:eng->b_attn_gated_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_in_i8[0] offset:0 atIndex:1];
                [enc setBuffer:eng->b_in_xsc[0] offset:0 atIndex:2];
                [enc setBuffer:eng->b_in_xsum_f16[0] offset:0 atIndex:3];
                [enc setBytes:&qp_wo length:sizeof(qp_wo) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((M_padded + 3U) / 4U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                sik_encode_dense_linear_i8(enc, eng, wo_sl, eng->b_attn_gated_f16,
                                           eng->b_in_i8[0], eng->b_in_xsc[0], eng->b_in_xsum_f16[0],
                                           nil, 0, eng->b_wo_out_f32, 0, B, M_padded, kOutF32, force_gemv,
                                           0.0f, wo_buf);
            } else {
                sik_encode_dense_linear(enc, eng, wo_sl, eng->b_attn_gated_f16, 0,
                                        nil, 0, eng->b_wo_out_f32, 0, B, M_padded, kOutF32, force_gemv,
                                        0.0f, wo_buf);
            }
            SikBatchSconvParamsHost bsp{
                B, D, res_mode, static_cast<uint32_t>(attn_sconv_off / 16ULL),
                top_k, 0U, 0U, 0U
            };
            [enc setComputePipelineState:eng->pso_sconv_static_batched];
            [enc setBuffer:eng->b_wo_out_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.attn_sconv_w.offset) atIndex:1];
            [enc setBuffer:eng->b_cur offset:0 atIndex:2];
            [enc setBuffer:blk_buf offset:blk_off atIndex:3];
            [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:4];
            [enc setBytes:&bsp length:sizeof(bsp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((D + 255U) / 256U, B, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        if (is_block_boundary && num_blocks < kMaxAttnResBlocks) ++num_blocks;

        // Pre-MoE Block AttnRes + Optional Loop Ctrl 3 + `mlp_norm`
        const bool is_q8_mlp_ctrl = (has_msgs && ld.controllers[3].down_weight.dtype == STRATA_INKLING2_DTYPE_Q8_0);
        const bool batch12_mlp    = (has_msgs && use_batch12_loop);
        const bool fast_moe_rms   = (batch12_mlp || (!is_batch_nax && B <= kMaxSpecBatch && is_q8_mlp_ctrl));
        if (!has_msgs) {
            sik_encode_attn_res_rmsnorm(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                        ld.mlp_res_query.offset, ld.mlp_norm.offset,
                                        eng->b_in_f16[0], eng->b_x_f32, true, B, M_padded);
        } else {
            const bool fused_mlp_prep = (!is_batch_nax && B <= kMaxSpecBatch);
            if (batch12_mlp) {
                SikAttnResParamsHost ap{
                    B, D, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    1U, cfg.rms_norm_eps, 1U, 0U
                };
                [enc setComputePipelineState:eng->pso_attn_res_prep_left_576];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_res_query.offset) atIndex:2];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:3];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:4];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:5];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:6];
                [enc setBytes:&ap length:sizeof(ap) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(576, 1, 1)];
            } else if (fused_mlp_prep) {
                SikAttnResParamsHost ap{
                    B, D, num_blocks, static_cast<uint32_t>(eng->block_slot_elems),
                    1U, cfg.rms_norm_eps, 0U, 0U
                };
                [enc setComputePipelineState:eng->pso_attn_res_loop_prep1];
                [enc setBuffer:eng->b_blocks offset:0 atIndex:0];
                [enc setBuffer:eng->b_cur offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_res_query.offset) atIndex:2];
                [enc setBuffer:eng->b_messages[3] offset:0 atIndex:3];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:4];
                [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:5];
                [enc setBytes:&ap length:sizeof(ap) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                sik_encode_attn_res(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                    ld.mlp_res_query.offset, 1U, eng->b_inputs[0], nil, nil, B);
            }
            if (batch12_mlp) {
                const uint32_t   li    = ip - core_beg;
                const uint32_t   R     = cfg.loop_gate_rank;
                const NSUInteger mslot = static_cast<NSUInteger>(B) * D * sizeof(float);
                const NSUInteger hslot = static_cast<NSUInteger>(B) * R * sizeof(float);
                if (B <= kMaxSpecBatch) {
                    sik_encode_loop_ctrl_batch(
                        cb, enc, eng, &ld.controllers[3], 1U,
                        eng->b_inputs[0], nil, nil, msg_bank12_buf, nil, nil,
                        eng->b_inputs[0], nil, nil, B, M_padded, force_gemv, /*skip_prep=*/true,
                        static_cast<NSUInteger>(li * 4U + 3U) * mslot, 0U, 0U,
                        ctrl_h_msg12_buf, static_cast<NSUInteger>(li * 4U + 3U) * hslot
                    );
                } else {
                    const uint32_t   grid_sq    = (D + 31U) / 32U;
                    const NSUInteger m_base_off = static_cast<NSUInteger>(li * 4U + 3U) * mslot;
                    const NSUInteger h_base_off = static_cast<NSUInteger>(li * 4U + 3U) * hslot;
                    uint64_t dw = 0ULL, ds = 0ULL, db = 0ULL;
                    uint64_t gw = 0ULL, gs = 0ULL, gb = 0ULL;
                    uint64_t fw = 0ULL, fs = 0ULL, fb = 0ULL;
                    sik_slice_sub_offsets(ld.controllers[3].down_weight,   dw, ds, db);
                    sik_slice_sub_offsets(ld.controllers[3].gain_weight,   gw, gs, gb);
                    sik_slice_sub_offsets(ld.controllers[3].forget_weight, fw, fs, fb);

                    SikLoopCtrl3OffsetsHost down_off{};
                    down_off.w_off[0]  = dw;
                    down_off.s_off[0]  = ds;
                    down_off.M_valid   = B;
                    down_off.M_padded  = B;
                    down_off.D         = D;
                    down_off.gate_rank = R;
                    down_off.num_ctrls = 1U;

                    if (n_full > 0U) {
                        [enc setComputePipelineState:eng->pso_loop_ctrl_down1_left_q8_2d];
                        [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                        [enc setBuffer:eng->b_ctrl_cat_f16 offset:0 atIndex:1];
                        [enc setBuffer:ctrl_h_msg12_buf offset:h_base_off atIndex:2];
                        [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:3];
                        [enc setBytes:&down_off length:sizeof(down_off) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(n_full, R, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    if (rem > 0U) {
                        SikLoopCtrl3OffsetsHost down_rem = down_off;
                        down_rem.M_valid = down_rem.M_padded = rem;
                        const NSUInteger cat_o = static_cast<NSUInteger>(m_rem) * D * sizeof(uint16_t);
                        const NSUInteger h_o   = static_cast<NSUInteger>(m_rem) * R * sizeof(float);
                        [enc setComputePipelineState:eng->pso_loop_ctrl_down1_left_q8_m[rem - 1U]];
                        [enc setBuffer:eng->b_uma offset:0 atIndex:0];
                        [enc setBuffer:eng->b_ctrl_cat_f16 offset:cat_o atIndex:1];
                        [enc setBuffer:ctrl_h_msg12_buf offset:h_base_off + h_o atIndex:2];
                        [enc setBuffer:eng->b_ctrl_h_f32 offset:h_o atIndex:3];
                        [enc setBytes:&down_rem length:sizeof(down_rem) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    if (n_full > 0U) {
                        SikLoopCtrlApplyParamsHost ap{B, B, D, R};
                        [enc setComputePipelineState:eng->pso_loop_ctrl_apply_q8_m[kMaxSpecBatch - 1U]];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(gw) atIndex:0];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(gs) atIndex:1];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw) atIndex:2];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fs) atIndex:3];
                        [enc setBuffer:eng->b_ctrl_h_f32 offset:0 atIndex:4];
                        [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:5];
                        [enc setBuffer:msg_bank12_buf offset:m_base_off atIndex:6];
                        [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc setBuffer:eng->b_rms_sq offset:0 atIndex:9];
                        [enc dispatchThreadgroups:MTLSizeMake(grid_sq, n_full, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    if (rem > 0U) {
                        SikLoopCtrlApplyParamsHost ap{rem, rem, D, R};
                        const NSUInteger d_o  = static_cast<NSUInteger>(m_rem) * D * sizeof(float);
                        const NSUInteger h_o  = static_cast<NSUInteger>(m_rem) * R * sizeof(float);
                        const NSUInteger sq_o = static_cast<NSUInteger>(m_rem) * grid_sq * sizeof(float);
                        [enc setComputePipelineState:eng->pso_loop_ctrl_apply_q8_m[rem - 1U]];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(gw) atIndex:0];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(gs) atIndex:1];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fw) atIndex:2];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(fs) atIndex:3];
                        [enc setBuffer:eng->b_ctrl_h_f32 offset:h_o atIndex:4];
                        [enc setBuffer:eng->b_inputs[0] offset:d_o atIndex:5];
                        [enc setBuffer:msg_bank12_buf offset:m_base_off + d_o atIndex:6];
                        [enc setBuffer:eng->b_inputs[0] offset:d_o atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc setBuffer:eng->b_rms_sq offset:sq_o atIndex:9];
                        [enc dispatchThreadgroups:MTLSizeMake(grid_sq, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                }
            } else {
                sik_encode_loop_ctrl_batch(
                    cb, enc, eng, &ld.controllers[3], 1U,
                    eng->b_inputs[0], nil, nil, eng->b_messages[3], nil, nil,
                    eng->b_inputs[0], nil, nil, B, M_padded, force_gemv, /*skip_prep=*/fused_mlp_prep
                );
            }
            if (fast_moe_rms) {
                SikRmsNormParamsHost np{B, M_padded, D, cfg.rms_norm_eps, (D + 31U) / 32U, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_rmsnorm1_fast];
                [enc setBuffer:eng->b_inputs[0] offset:0 atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_norm.offset) atIndex:1];
                [enc setBuffer:eng->b_rms_sq offset:0 atIndex:2];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_f32 offset:0 atIndex:4];
                [enc setBytes:&np length:sizeof(np) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(D / 64U, B, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                sik_encode_rmsnorm(enc, eng, eng->b_inputs[0], 0, ld.mlp_norm.offset, true,
                                   eng->b_in_f16[0], eng->b_x_f32, true, B, M_padded);
            }
        }

        // MoE Router + TopK/Bin + Shared-Weight Grouped-4/8 NVFP4/MXFP8 Experts + Per-Stream `Combine+mlp_sconv`
        {
            const bool is_nvfp4 = (ld.expert_dtype == STRATA_INKLING2_DTYPE_NVFP4);
            const uint32_t small_tcap = (B == 1U) ? 1U : ((B == 2U) ? 2U : 4U);
            const uint32_t big_tcap   = 4U;
            SikMoeRouterParamsHost rp{B, D, E, top_k, cfg.route_scale, inter, small_tcap, 0U};
            SikMoeExecParamsHost   mp{
                B, top_k, D, inter, E, big_tcap,
                cfg.mlp_gate_softcap, cfg.mlp_up_softcap
            };

            uint32_t b_slot = 2U; // max_b = 4
            if (B > kMaxSpecBatch) {
                if (use_exact_b64) {
                    if (n_full > 0U) {
                        [enc setComputePipelineState:eng->pso_moe_router_f32_2d];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_weight.offset) atIndex:0];
                        [enc setBuffer:eng->b_x_f32 offset:0 atIndex:1];
                        [enc setBuffer:eng->b_router_logits offset:0 atIndex:2];
                        [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(n_full, E, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    if (rem > 0U) {
                        SikMoeRouterParamsHost rp_rem = rp;
                        rp_rem.M_valid = rem;
                        [enc setComputePipelineState:eng->pso_moe_router_f32_m[rem - 1U]];
                        [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_weight.offset) atIndex:0];
                        [enc setBuffer:eng->b_x_f32 offset:static_cast<NSUInteger>(m_rem) * D * sizeof(float) atIndex:1];
                        [enc setBuffer:eng->b_router_logits offset:static_cast<NSUInteger>(m_rem) * E * sizeof(float) atIndex:2];
                        [enc setBytes:&rp_rem length:sizeof(rp_rem) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(E, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                } else {
                    [enc setComputePipelineState:eng->pso_moe_router_f32_prefill];
                    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_weight.offset) atIndex:0];
                    [enc setBuffer:eng->b_x_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_router_logits offset:0 atIndex:2];
                    [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(E / 32U, (B + 15U) / 16U, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }

                [enc setComputePipelineState:eng->pso_moe_topk_sigmoid];
                [enc setBuffer:eng->b_router_logits offset:0 atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_bias.offset) atIndex:1];
                [enc setBuffer:eng->b_topk_idx offset:0 atIndex:2];
                [enc setBuffer:eng->b_topk_w offset:0 atIndex:3];
                [enc setBytes:&rp length:sizeof(rp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc setComputePipelineState:eng->pso_moe_bin_tokens_grouped];
                [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
                [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
                [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
                [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:3];
                [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:4];
                [enc setBytes:&mp length:sizeof(mp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                b_slot = 2U; // max_b = 4 (2x occupancy vs max_b = 8!)
            } else {
                [enc setComputePipelineState:eng->pso_moe_router_f32_m[B - 1U]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_weight.offset) atIndex:0];
                [enc setBuffer:eng->b_x_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_router_logits offset:0 atIndex:2];
                [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(E, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_moe_topk_bin_small];
                [enc setBuffer:eng->b_router_logits offset:0 atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.moe_gate_bias.offset) atIndex:1];
                [enc setBuffer:eng->b_topk_idx offset:0 atIndex:2];
                [enc setBuffer:eng->b_topk_w offset:0 atIndex:3];
                [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:7];
                [enc setBytes:&rp length:sizeof(rp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                b_slot = (B == 1U) ? 0U : ((B == 2U) ? 1U : 2U);
            }

            if (is_nvfp4) {
                [enc setComputePipelineState:eng->pso_moe_nvfp4_w13_b[b_slot]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.expert_scale_off) atIndex:2];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:7];
                [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                       indirectBufferOffset:0U
                                      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_moe_nvfp4_w2_b[b_slot]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.expert_scale_off) atIndex:2];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
                [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:6];
                [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:7];
                [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                       indirectBufferOffset:16U
                                      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc setComputePipelineState:eng->pso_moe_mxfp8_w13_b[b_slot]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.weight_off) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w13.scales_off) atIndex:1];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:2];
                [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:5];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:6];
                [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                       indirectBufferOffset:0U
                                      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_moe_mxfp8_w2_b[b_slot]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.weight_off) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.experts_w2.scales_off) atIndex:1];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
                [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                [enc setBuffer:eng->b_moe_num_tiles offset:0 atIndex:5];
                [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:6];
                [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                       indirectBufferOffset:16U
                                      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            SikBatchSconvParamsHost csp{
                B, D, 1U, static_cast<uint32_t>(mlp_sconv_off / 16ULL),
                top_k, 0U, 0U, 0U
            };
            [enc setComputePipelineState:eng->pso_moe_combine_sconv_batched];
            [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:0];
            [enc setBuffer:eng->b_topk_w offset:0 atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.mlp_sconv_w.offset) atIndex:2];
            [enc setBuffer:eng->b_cur offset:0 atIndex:3];
            [enc setBuffer:eng->b_batch_slot_meta offset:0 atIndex:4];
            [enc setBytes:&csp length:sizeof(csp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((D + 63U) / 64U, B, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
    }

    // 3c. Final Head: `attn_res(output_res_query) + rms_norm(final_norm)` -> `b_in_f16[0] [M_padded, D]`
    sik_encode_attn_res_rmsnorm(enc, eng, eng->b_blocks, num_blocks, eng->b_cur,
                                eng->hdr.output_res_query.offset, eng->hdr.final_norm.offset,
                                eng->b_in_f16[0], nil, false, B, M_padded, false, 0U);

    // 3d. Chunked High-BW Unembed + Per-Stream Penalties + Per-Stream Heterogeneous Sampling
    id<MTLBuffer>  logits_buf = (B <= kMaxSpecBatch) ? eng->b_logits         : eng->b_batch_logits;
    id<MTLBuffer>  stage_v    = (B <= kMaxSpecBatch) ? eng->b_sample_stage_v : eng->b_batch_sample_v;
    id<MTLBuffer>  stage_i    = (B <= kMaxSpecBatch) ? eng->b_sample_stage_i : eng->b_batch_sample_i;
    const uint32_t chunk_max  = (B <= kMaxSpecBatch) ? kMaxSpecBatch         : eng->batch_logits_rows;

    for (uint32_t c_off = 0U; c_off < B; c_off += chunk_max) {
        const uint32_t   B_c     = std::min<uint32_t>(chunk_max, B - c_off);
        const uint32_t   M_pad_c = (B_c <= kMaxSpecBatch) ? B_c : ((B_c <= 16U) ? 16U : 32U);
        const NSUInteger x_off   = static_cast<NSUInteger>(c_off) * D * sizeof(uint16_t);

        if (B_c <= kMaxSpecBatch && eng->hdr.unembed.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
            uint64_t uw = 0ULL, us = 0ULL, ub = 0ULL;
            sik_slice_sub_offsets(eng->hdr.unembed, uw, us, ub);
            const NSUInteger xs_off = static_cast<NSUInteger>(c_off) * (D >> 6U) * sizeof(float);
            SikGemvBatchParamsHost gp{B_c, eng->hdr.unembed.shape0, eng->hdr.unembed.shape1,
                                      kOutF32Softcap, cfg.final_logit_softcapping, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_q4a_gemv_xsum_m[B_c - 1U]];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ub) atIndex:2];
            [enc setBuffer:eng->b_in_f16[0] offset:x_off atIndex:3];
            [enc setBuffer:eng->b_x_blk_sum[0] offset:xs_off atIndex:4];
            [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:5];
            [enc setBuffer:logits_buf offset:0 atIndex:6];
            [enc setBytes:&gp length:sizeof(gp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((eng->hdr.unembed.shape0 + 7U) / 8U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (use_exact_b64 && B_c > kMaxSpecBatch && eng->hdr.unembed.dtype == STRATA_INKLING2_DTYPE_Q8_0) {
            uint64_t uw = 0ULL, us = 0ULL, ub = 0ULL;
            sik_slice_sub_offsets(eng->hdr.unembed, uw, us, ub);
            const uint32_t c_full = B_c / kMaxSpecBatch;
            const uint32_t c_rem  = B_c % kMaxSpecBatch;
            const uint32_t c_mrem = c_full * kMaxSpecBatch;
            if (c_full > 0U) {
                SikGemvBatchParamsHost gp{B_c, eng->hdr.unembed.shape0, eng->hdr.unembed.shape1,
                                          kOutF32Softcap, cfg.final_logit_softcapping, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_q8_gemv_batch_2d];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
                [enc setBuffer:eng->b_in_f16[0] offset:x_off atIndex:2];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                [enc setBuffer:logits_buf offset:0 atIndex:4];
                [enc setBytes:&gp length:sizeof(gp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(c_full, (eng->hdr.unembed.shape0 + 7U) / 8U, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            if (c_rem > 0U) {
                SikGemvBatchParamsHost gp{c_rem, eng->hdr.unembed.shape0, eng->hdr.unembed.shape1,
                                          kOutF32Softcap, cfg.final_logit_softcapping, 0U, 0U, 0U};
                const NSUInteger x_ro = x_off + static_cast<NSUInteger>(c_mrem) * D * sizeof(uint16_t);
                const NSUInteger y_ro = static_cast<NSUInteger>(c_mrem) * eng->hdr.unembed.shape0 * sizeof(float);
                [enc setComputePipelineState:eng->pso_q8_gemv_m[c_rem - 1U]];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(uw) atIndex:0];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(us) atIndex:1];
                [enc setBuffer:eng->b_in_f16[0] offset:x_ro atIndex:2];
                [enc setBuffer:eng->b_in_f16[0] offset:0 atIndex:3];
                [enc setBuffer:logits_buf offset:y_ro atIndex:4];
                [enc setBytes:&gp length:sizeof(gp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((eng->hdr.unembed.shape0 + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        } else {
            sik_encode_dense_linear(enc, eng, eng->hdr.unembed,
                                    eng->b_in_f16[0], x_off, nil, 0, logits_buf, 0,
                                    B_c, M_pad_c, kOutF32Softcap,
                                    /*force_gemv=*/(B_c <= kMaxSpecBatch),
                                    cfg.final_logit_softcapping);
        }

        uint32_t max_pen_len = 0U;
        for (uint32_t m = 0U; m < B_c; ++m) {
            if (pen_items_host[c_off + m].pen_len > max_pen_len) {
                max_pen_len = pen_items_host[c_off + m].pen_len;
            }
        }
        if (max_pen_len > 0U) {
            SikBatchPenParamsHost bpp{B_c, V, c_off, 0U};
            [enc setComputePipelineState:eng->pso_apply_logit_penalties_batched];
            [enc setBuffer:logits_buf offset:0 atIndex:0];
            [enc setBuffer:eng->b_batch_pen_toks offset:0 atIndex:1];
            [enc setBuffer:eng->b_batch_pen_cnts offset:0 atIndex:2];
            [enc setBuffer:eng->b_batch_pen_items offset:0 atIndex:3];
            [enc setBytes:&bpp length:sizeof(bpp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((max_pen_len + 255U) / 256U, B_c, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        SikBatchSampleParamsHost bsp{B_c, V, c_off, 0U};
        [enc setComputePipelineState:eng->pso_sample_batched_stage1];
        [enc setBuffer:logits_buf offset:0 atIndex:0];
        [enc setBuffer:stage_v offset:0 atIndex:1];
        [enc setBuffer:stage_i offset:0 atIndex:2];
        [enc setBuffer:eng->b_batch_sample_items offset:0 atIndex:3];
        [enc setBytes:&bsp length:sizeof(bsp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(64, B_c, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:eng->pso_sample_batched_stage2];
        [enc setBuffer:stage_v offset:0 atIndex:0];
        [enc setBuffer:stage_i offset:0 atIndex:1];
        [enc setBuffer:eng->b_batch_next_tokens offset:0 atIndex:2];
        [enc setBuffer:eng->b_batch_sample_items offset:0 atIndex:3];
        [enc setBytes:&bsp length:sizeof(bsp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(B_c, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    eng->prof_active = prev_prof;

    if ([cb status] == MTLCommandBufferStatusError) {
        NSError* err = [cb error];
        std::fprintf(stderr, "[strata_inkling2_gpu] Batched decode GPU error (B=%u): %s\n",
                     B, err ? [[err localizedDescription] UTF8String] : "unknown");
        return -1;
    }

    const uint32_t* out_toks = reinterpret_cast<const uint32_t*>([eng->b_batch_next_tokens contents]);
    for (uint32_t b = 0U; b < B; ++b) {
        StrataInkling2BatchItem& it = items[sub_idx[b]];
        it.next_token_out  = out_toks[b];
        it.slot->seq_pos  += 1U;
        it.slot->mtp_synced = false;
        sik_ngram_append(it.slot, &it.token_id, 1U);
    }

    eng->counters.decode_steps            += static_cast<uint64_t>(B);
    eng->counters.physical_dram_bytes_read += eng->hdr.active_bytes_per_tok;
    return 0;
}

} // namespace

extern "C" {

int strata_inkling2_gpu_decode_batch(
    StrataInkling2GpuEngine* eng,
    StrataInkling2BatchItem* items,
    uint32_t                 batch_size
) {
    if (!eng || !items || batch_size == 0U || batch_size > STRATA_INKLING2_MAX_BATCH_SIZE) return -1;

    @autoreleasepool {
        if (!sik_drain_pending_fix(eng)) return -1;
        sik_sync_engine_to_slot(eng);

        for (uint32_t i = 0U; i < batch_size; ++i) {
            StrataInkling2ContextSlot* sl = items[i].slot;
            if (!sl || sl->eng != eng) return -1;
            if (sl->seq_pos + 1U > eng->max_seq_len) return -1;
            for (uint32_t j = 0U; j < i; ++j) {
                if (items[j].slot == sl) return -1;
            }
        }

        // Fast-path `batch_size == 1`: preserves full 122 tok/s Bitplane Head + `mtp_synced`!
        if (batch_size == 1U) {
            StrataInkling2ContextSlot* const prev_slot = eng->bound_slot;
            const bool                       prev_alt  = eng->alt_wo_active;
            const bool                       want_alt  = items[0].use_alt_wo && (eng->b_alt_wo != nil) && (eng->alt_wo_layers > 0U);
            if (eng->bound_slot != items[0].slot) {
                sik_sync_slot_to_engine(eng, items[0].slot);
            }
            eng->alt_wo_active = want_alt;
            const int rc = strata_inkling2_gpu_decode_step(
                eng, items[0].token_id, items[0].sampling, &items[0].next_token_out, nullptr);
            eng->alt_wo_active = prev_alt;
            if (eng->bound_slot != prev_slot && prev_slot != nullptr) {
                sik_sync_engine_to_slot(eng);
                sik_sync_slot_to_engine(eng, prev_slot);
            }
            return rc;
        }

        // Partition mixed `--no-refusal` (`use_alt_wo`) batches into <= 2 homogeneous GPU sub-batches (0 weight mixing!)
        uint32_t idx_stock[STRATA_INKLING2_MAX_BATCH_SIZE];
        uint32_t idx_alt[STRATA_INKLING2_MAX_BATCH_SIZE];
        uint32_t n_stock = 0U, n_alt = 0U;
        const bool alt_avail = (eng->b_alt_wo != nil) && (eng->alt_wo_layers > 0U);
        for (uint32_t i = 0U; i < batch_size; ++i) {
            if (items[i].use_alt_wo && alt_avail) idx_alt[n_alt++]     = i;
            else                                  idx_stock[n_stock++] = i;
        }

        const bool prev_alt = eng->alt_wo_active;
        int rc = 0;
        if (n_stock > 0U) {
            rc = sik_decode_batch_homogeneous(eng, items, idx_stock, n_stock, false);
        }
        if (rc == 0 && n_alt > 0U) {
            rc = sik_decode_batch_homogeneous(eng, items, idx_alt, n_alt, true);
        }
        eng->alt_wo_active = prev_alt;
        if (eng->bound_slot) {
            sik_sync_slot_to_engine(eng, eng->bound_slot);
        }
        return rc;
    }
}

} /* extern "C" */
